"""Validate a complete captured forward against independent sampled pixels.

The full Gaussian assembly is rendered at its captured image resolution. All
output pixels must be finite, while numerical comparison covers the explicitly
recorded fixed-seed pixels only. This does not establish GPU parity, full-image
quality, closed-loop behavior, training or performance acceptance.
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
from worldengine.render.ascend.capture import load_capture
from worldengine.render.ascend.manifest import digest
from worldengine.render.ascend.real_validation import (EVIDENCE_TYPE, REAL_IDENTITY_FIELDS,
    create_lease, validate_lease)


def sample_coordinates(cameras, width, height, per_camera, seed):
    """Deterministic uniform pixels plus corner/center coverage; no output bias."""
    if type(per_camera) is not int or not 5 <= per_camera <= 4096:
        raise ValueError('Use 5..4096 reference pixels per camera')
    rng = np.random.default_rng(seed)
    rows = []
    for camera in range(cameras):
        coordinates = {(0,0), (0,width-1), (height-1,0), (height-1,width-1), (height//2,width//2)}
        target = min(per_camera, width*height)
        while len(coordinates) < target:
            remaining = target-len(coordinates)
            flat = rng.choice(width*height, size=remaining, replace=False)
            coordinates.update((int(index)//width, int(index)%width) for index in flat)
        rows.extend((camera,y,x) for y,x in sorted(coordinates))
    return rows


def _write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--small-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--samples-per-camera', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--max-candidates', type=int, default=2_000_000)
    parser.add_argument('--max-host-input-bytes', type=int, default=16*1024**3,
                        help='Reject larger uncompressed input archives before loading; no truncation')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if deadline.tzinfo is None:
        parser.error('The deadline requires an explicit UTC offset')
    if args.max_host_input_bytes < 1 or not 1 <= args.chunk_size <= 4096 or args.max_candidates < 1:
        parser.error('Use positive resource bounds and chunk-size 1..4096')
    def guard():
        if time.time() >= deadline.timestamp():
            raise TimeoutError('Authorized real-input validation deadline reached')
        if args.stop_file is not None and args.stop_file.exists():
            raise InterruptedError('Real-input validation stop file is present')
    guard()
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(result='running', evidence_type=EVIDENCE_TYPE,
        scope='complete captured forward; full-output finite check and sparse-pixel numerical reference only',
        started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_parity_acceptance=False, full_pixel_quality_acceptance=False,
        closed_loop_acceptance=False, performance_acceptance=False,
        all_forward_outputs_finite=False, sampled_reference_passed=False)
    identity = None
    report_path = args.output/'real_input_validation.json'
    try:
        from worldengine.render.ascend.capture import read_capture_manifest
        declared, _ = read_capture_manifest(args.capture)
        # NPZ is generated uncompressed by capture.py. Shape-based accounting
        # also catches an edited small archive claiming huge allocations.
        shape = declared['shape'];n,c = shape['gaussians'],shape['cameras']
        expected_bytes = 4*(n*(3+4+3+1)+c*n*3+c*(16+9))
        if expected_bytes > args.max_host_input_bytes:
            raise MemoryError('The complete capture exceeds the explicit host-input memory budget')
        loaded_manifest_sha256=digest(args.capture)
        inputs, captured = load_capture(args.capture)
        if digest(args.capture)!=loaded_manifest_sha256:
            raise ValueError('The capture manifest changed while loading the complete input')
        render = captured['render']
        if render['render_mode']!='RGB+ED':
            raise ValueError('Capture RGB+ED inputs to validate RGB, alpha and expected depth together')
        coordinates = sample_coordinates(c,render['width'],render['height'],args.samples_per_camera,args.seed)
        guard()
        import torch
        import torch_npu
        from worldengine.render.ascend.cann import CANNRasterizer, runtime_identity
        device = torch.device(args.device)
        if device.type != 'npu' or device.index is None:
            raise ValueError('An explicit isolated NPU is required')
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32=False
        native = importlib.import_module('gauss_splat.gsplat_c')
        lease = create_lease(args.small_manifest,args.capture,args.deadline_utc,native.__file__,runtime_identity())
        binding=lease['capture_binding']
        if (binding['manifest_sha256']!=loaded_manifest_sha256
                or binding['archive_sha256']!=captured['archive']['sha256']
                or binding['archive_bytes']!=captured['archive']['bytes']
                or binding['shape']!=captured['shape'] or binding['render']!=captured['render']
                or binding['provenance']!=captured['provenance']):
            raise ValueError('The lease does not identify the exact loaded capture')
        lease_path = args.output/'validation_lease.json'
        _write(lease_path,lease)
        identity, _ = validate_lease(lease_path,native.__file__,runtime_identity())
        report['identity'] = {name:identity[name] for name in REAL_IDENTITY_FIELDS}
        report['reference_pixel_count'] = len(coordinates)
        report['reference_pixels'] = [list(row) for row in coordinates]
        report['sampling'] = dict(seed=args.seed,requested_pixels_per_camera=args.samples_per_camera,
            selection='uniform fixed seed plus corners and center, independent of NPU outputs')
        report['host_input_array_bytes'] = sum(value.nbytes for value in inputs.values())
        renderer = CANNRasterizer.for_bounded_real_input_validation(lease_path)
        guard()
        tensors={name:torch.from_numpy(value).to(device=device) for name,value in inputs.items()}
        started=time.monotonic()
        with torch.no_grad():
            actual_rgb,actual_alpha,metadata=renderer(**tensors,**render)
            torch.npu.synchronize()
            if any(value.device!=device for value in (actual_rgb,actual_alpha,metadata['radii'])):
                raise AssertionError('The full renderer output left the explicitly selected NPU')
            if metadata['device_validation_accepted']:
                raise AssertionError('A validation lease was mislabeled as accepted')
            if not bool(torch.isfinite(actual_rgb).all()) or not bool(torch.isfinite(actual_alpha).all()):
                raise AssertionError('The complete forward contains nonfinite output pixels')
            if not bool(torch.isfinite(metadata['radii']).all()):
                raise AssertionError('The complete forward contains nonfinite projected radii')
            index=torch.tensor(coordinates,dtype=torch.int64,device=device)
            selected_rgb=actual_rgb[index[:,0],index[:,1],index[:,2]].cpu().numpy()
            selected_alpha=actual_alpha[index[:,0],index[:,1],index[:,2]].cpu().numpy()
        report['forward_validation_elapsed_seconds']=time.monotonic()-started
        report['timing_scope']='render, full-output finite checks and sparse output transfer; no performance acceptance'
        report['all_forward_outputs_finite']=True
        report['full_output_shapes']=dict(rgb=list(actual_rgb.shape),alpha=list(actual_alpha.shape),radii=list(metadata['radii'].shape))
        # Release the device payload/output before the independent CPU oracle.
        del actual_rgb,actual_alpha,metadata,tensors,index
        guard()
        from sampled_oracle import sampled_reference
        reference=sampled_reference(inputs,render,coordinates,chunk_size=args.chunk_size,
            max_candidates=args.max_candidates,guard=guard)
        expected_rgb=np.asarray(reference['rgb'])
        expected_alpha=np.asarray(reference['alpha'])
        expected_depth=np.asarray(reference['expected_depth'])
        np.testing.assert_allclose(selected_rgb[:,:3],expected_rgb,atol=3e-4,rtol=1e-3)
        np.testing.assert_allclose(selected_alpha,expected_alpha,atol=3e-4,rtol=1e-3)
        if render['render_mode']=='RGB+ED':
            np.testing.assert_allclose(selected_rgb[:,3:],expected_depth,atol=2e-3,rtol=1e-3)
        report['reference_statistics']=reference['statistics']
        report['measurements']=dict(rgb_max_absolute_difference=float(np.max(np.abs(selected_rgb[:,:3]-expected_rgb))),
            alpha_max_absolute_difference=float(np.max(np.abs(selected_alpha-expected_alpha))),
            expected_depth_max_absolute_difference=float(np.max(np.abs(selected_rgb[:,3:]-expected_depth))) if render['render_mode']=='RGB+ED' else None,
            rgb_atol=3e-4,rgb_rtol=1e-3,alpha_atol=3e-4,alpha_rtol=1e-3,expected_depth_atol=2e-3,expected_depth_rtol=1e-3)
        report['samples']=dict(actual_rgb=selected_rgb.tolist(),actual_alpha=selected_alpha.tolist(),
            reference_rgb=expected_rgb.tolist(),reference_alpha=expected_alpha.tolist(),reference_expected_depth=expected_depth.tolist())
        report['sampled_reference_passed']=True
        guard()
        final_identity,_=validate_lease(lease_path,native.__file__,runtime_identity())
        if final_identity!=identity:
            raise ValueError('Proof, input or validation implementation changed during validation')
        guard()
        report['result']='pass'
    except BaseException as error:
        report.update(result='error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['completed_utc']=datetime.now(timezone.utc).isoformat()
        _write(report_path,report)
        if report['result']=='pass':
            identity['validation']=dict(result='pass',evidence_path=str(report_path.resolve()),evidence_sha256=digest(report_path))
            _write(args.output/'validated_real_manifest.json',identity)
        print(json.dumps(dict(result=report['result'],report=str(report_path))))


if __name__=='__main__':
    main()
