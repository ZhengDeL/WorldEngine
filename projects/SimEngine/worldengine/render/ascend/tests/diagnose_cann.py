"""Small, explicit NPU diagnostic for the first forward validation case.

Produces measurements and intermediate summaries, never an acceptance manifest.
Run under the same external device/deadline watchdog as probe_cann.py.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from probe_cann import make_build_manifest
from numpy_oracle import project, composite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        parser.error('The deadline requires an explicit UTC offset')
    def guard():
        if time.time() >= deadline.timestamp():
            raise TimeoutError('Authorized diagnostic deadline reached')
    guard()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(result='running', scope='diagnostic measurements only; no acceptance',
                  started_utc=datetime.now(timezone.utc).isoformat(), native_calls=[])
    try:
        import torch
        import torch_npu
        import acl
        from worldengine.render.ascend.cann import (CANNRasterizer, runtime_identity,
                                                   pixel_coordinates, tiles_to_image)
        device = torch.device(args.device)
        if device.type != 'npu':
            raise ValueError('An explicitly selected NPU is required')
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32 = False
        native = importlib.import_module('gauss_splat.gsplat_c')
        manifest = make_build_manifest(native, runtime_identity(),
                        dict(max_gaussians=64, max_cameras=2, max_pixels=4096))
        manifest_path = args.output / 'build_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        renderer = CANNRasterizer.for_validation_only(manifest_path)
        report['runtime_versions'] = manifest['runtime_versions']
        report['compatibility_kernel_sha256'] = manifest['compatibility_kernel_sha256']
        report['adapter_sha256'] = manifest['adapter_sha256']
        def summarize(value):
            if torch.is_tensor(value):
                a = value.detach().cpu().numpy()
                flat = a.reshape(-1)
                return dict(shape=list(a.shape), dtype=str(a.dtype), device=str(value.device),
                    finite=bool(np.isfinite(a).all()), nonzero=int(np.count_nonzero(a)),
                    minimum=float(a.min()) if a.size else None,
                    maximum=float(a.max()) if a.size else None,
                    total=float(a.sum()) if a.size else 0.,
                    values=a.tolist() if a.size <= 256 else flat[:32].tolist())
            if isinstance(value, (list, tuple)):
                return [summarize(v) for v in value]
            return value
        captures = {}
        class NativeTrace:
            def __getattr__(self, name):
                original = getattr(native, name)
                if not callable(original):
                    return original
                def traced(*inputs):
                    guard()
                    entry = dict(method=name, inputs=summarize(inputs))
                    report['native_calls'].append(entry)
                    outputs = original(*inputs)
                    torch.npu.synchronize()
                    entry['outputs'] = summarize(outputs)
                    captures[name] = (inputs, outputs)
                    guard()
                    return outputs
                return traced
        renderer.native = NativeTrace()
        def tensor(value):
            return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.float32)
        means = np.array([[0., 0., 2.]], np.float32)
        quats = np.array([[1., 0., 0., 0.]], np.float32)
        scales = np.full((1, 3), .2, np.float32)
        opacity = np.array([.4], np.float32)
        colors = np.array([[[1., .2, .1]]], np.float32)
        views = np.eye(4, dtype=np.float32)[None]
        k = np.array([[[44.8, 0., 32.5], [0., 51.2, 32.5], [0., 0., 1.]]], np.float32)
        projected = project(means, quats, scales, views, k, 64, 64, near=.01, far=10)
        expected_rgb, expected_alpha = composite(projected, opacity, colors, 64, 64,
                                                   tile_size=16, antialiased=False, expected_depth=True)
        report['oracle_projection'] = [{key: np.asarray(v).tolist() for key, v in p.items()}
                                        for p in projected]
        inputs = {name: tensor(value) for name, value in dict(means=means, quats=quats,
                  scales=scales, opacities=opacity, colors=colors, viewmats=views, Ks=k).items()}
        with torch.no_grad():
            # Canonical tails use aligned vector copies; expose whether the
            # first real covariance depends on the number of padded points.
            for n in (1, 2, 3, 7, 8, 32):
                guard()
                q = tensor(np.tile(quats, (n, 1)))
                s = tensor(np.tile(scales, (n, 1)))
                cov = native.quat_scales_to_covars(q.T[None].contiguous(), s.T[None].contiguous())
                torch.npu.synchronize()
                report.setdefault('raw_covariance_tail_measurements', []).append(dict(n=n,
                    covariance=summarize(cov), first_covariance=cov[0, :, :, 0].cpu().tolist(),
                    expected_first_covariance=np.diag([.04, .04, .04]).tolist()))
            actual_rgb, actual_alpha, metadata = renderer(**inputs, width=64, height=64,
                tile_size=16, rasterize_mode='classic', render_mode='RGB+ED', near_plane=.01, far_plane=10)
            torch.npu.synchronize()
            rgb = actual_rgb.cpu().numpy()
            alpha = actual_alpha.cpu().numpy()
            report['renderer_output'] = dict(rgb=summarize(actual_rgb), alpha=summarize(actual_alpha),
                radii=summarize(metadata['radii']), pixel_32_32=rgb[0, 32, 32].tolist(),
                alpha_32_32=alpha[0, 32, 32].tolist(),
                expected_pixel_32_32=expected_rgb[0, 32, 32].tolist(),
                expected_alpha_32_32=expected_alpha[0, 32, 32].tolist(),
                rgb_max_absolute_difference=float(np.max(np.abs(rgb[..., :3]-expected_rgb[..., :3]))))
            # Independent known payload bypasses both covariance and projection.
            # One valid Gaussian is assigned to every tile; membership can only
            # add zero-alpha pixels, not suppress the center contribution.
            guard()
            vector_count, status = acl.get_device_capability(torch.npu.current_device(), 1)
            if status or vector_count < 1:
                raise RuntimeError('Cannot establish vector-core count')
            attributes = tensor(np.array([[32.5, 32.5, 1/(44.8**2*.2**2/4+.3), 0.,
                                          1/(51.2**2*.2**2/4+.3), .4, 1., .2, .1, 2.]], np.float32))
            coordinates = pixel_coordinates(64, 64, 16, device)
            counts = torch.ones(16, dtype=torch.int64)
            schedule = native.get_render_schedule(counts, vector_count).to(device)
            ids = torch.zeros(16, dtype=torch.int64, device=device)
            raw = native.calc_render_fwd_double_clip_gsids(attributes, coordinates, schedule, ids)
            torch.npu.synchronize()
            raw_rgb = tiles_to_image(raw[0], 64, 64, 16)
            report['manual_known_payload'] = dict(vector_count=vector_count,
                attributes=summarize(attributes), schedule=summarize(schedule), ids=summarize(ids),
                coordinate_center=coordinates[10, :, 0].cpu().tolist(), outputs=summarize(raw),
                pixel_32_32=raw_rgb[32, 32].cpu().tolist(), expected_rgb_32_32=[.4, .08, .04])
            # A tail/padding diagnostic with exactly the same nonzero Gaussian,
            # followed by zero-opacity points. This does not alter validation.
            for n in (8, 32, 64):
                guard()
                expanded = {name: value.clone() for name, value in inputs.items()}
                for name in ('means', 'quats', 'scales', 'opacities'):
                    expanded[name] = inputs[name].repeat(n, *([1]*(inputs[name].ndim-1)))
                expanded['colors'] = inputs['colors'].repeat(1, n, 1)
                expanded['opacities'][1:] = 0.
                # No tracer duplication for the padded runs; only their output.
                renderer.native = native
                out, a, meta = renderer(**expanded, width=64, height=64, tile_size=16,
                    rasterize_mode='classic', render_mode='RGB+ED', near_plane=.01, far_plane=10)
                torch.npu.synchronize()
                report.setdefault('padding_measurements', []).append(dict(n=n, rgb=summarize(out),
                    alpha=summarize(a), radii=summarize(meta['radii']),
                    pixel_32_32=out[0, 32, 32].cpu().tolist()))
        report['result'] = 'diagnostic_complete'
    except BaseException as error:
        report.update(result='error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        destination = args.output / 'diagnostic.json'
        # If nonfinite native values appear, keep their diagnostic spelling;
        # this file is evidence and is never consumed as a passing manifest.
        destination.write_text(json.dumps(report, indent=2, allow_nan=True) + '\n')
        print(json.dumps(dict(result=report['result'], report=str(destination))))


if __name__ == '__main__':
    main()
