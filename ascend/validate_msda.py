"""Unmodified MMCV CPU32 reference versus both AlgEngine NPU MSDA entries.

The first two cases reproduce the earlier probe's shapes, seed and RNG draw
order exactly.  Every comparison covers the complete output/gradient tensors.
This is a numerical gate, not a GPU parity or training-convergence claim.
"""
import gc
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
import torch_npu


FORWARD_ATOL, FORWARD_RTOL = 2e-5, 2e-5
GRAD_ATOL, GRAD_RTOL = 1e-3, 1e-4
OUTPUT = Path('/output/strict_matrix.json')
MAX_SAMPLES = 12


def case(name, shapes, q, *, seed=29, b=1, heads=8, channels=32,
         points=4, locations='random', weights='random', layout=False):
    return dict(name=name, spatial_shapes=shapes, queries=q, seed=seed,
                batch=b, heads=heads, channels=channels, points=points,
                locations=locations, weights=weights, noncontiguous=layout)


CASES = [
    case('original_small_seed29', [(5, 7), (3, 4)], 41),
    case('original_large_seed29', [(200, 200)], 300),
    case('large_seed7', [(200, 200)], 300, seed=7),
    case('large_seed113', [(200, 200)], 300, seed=113),
    case('large_seed2026', [(200, 200)], 300, seed=2026),
    case('four_levels_non_square', [(31, 47), (15, 23), (7, 11), (3, 5)],
         57, heads=4, channels=16, points=8),
    case('multiple_batch2', [(19, 31), (9, 15)], 43, b=2,
         heads=4, channels=16),
    case('camera_batch8', [(13, 17), (5, 7)], 31, b=8,
         heads=2, channels=8, points=8),
    case('boundaries_and_adjacent_fp32', [(7, 11), (1, 5)], 49,
         heads=2, channels=8, locations='boundaries'),
    case('height1', [(1, 37)], 37, heads=2, channels=8,
         locations='boundaries'),
    case('width1', [(23, 1)], 37, heads=2, channels=8,
         locations='boundaries'),
    case('one_pixel', [(1, 1)], 41, heads=2, channels=8,
         locations='boundaries'),
    case('noncontiguous_offset_inputs_and_grad', [(13, 17), (5, 7)], 41,
         b=2, heads=4, channels=16, layout=True),
    case('softmax_weights', [(23, 31), (11, 15)], 53,
         heads=4, channels=16, weights='softmax'),
    case('zero_weights', [(23, 31), (11, 15)], 53,
         heads=4, channels=16, weights='zero'),
    case('single_active_weight', [(23, 31), (11, 15)], 53,
         heads=4, channels=16, weights='onepoint'),
    case('high_magnitude_weights', [(23, 31), (11, 15)], 53,
         heads=4, channels=16, weights='high'),
    case('query40000_reduced_heads_channels_strict', [(200, 200)], 40000,
         heads=2, channels=8),
]


def offset_layout_copy(source, device='cpu', requires_grad=False):
    """A leaf with last-dimension stride 2 and nonzero storage offset."""
    n = source.numel()
    storage = torch.empty(n * 2 + 7, dtype=source.dtype, device=device)
    view = storage[7:].view(*source.shape[:-1], source.shape[-1] * 2)[..., ::2]
    view.copy_(source.detach().to(device))
    return view.detach().requires_grad_(requires_grad)


def transfer(source, layout=False, requires_grad=False):
    if layout:
        return offset_layout_copy(source, 'npu', requires_grad)
    return source.detach().to('npu').requires_grad_(requires_grad)


def tensor_layout(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                contiguous=tensor.is_contiguous(), dtype=str(tensor.dtype),
                device=str(tensor.device))


def axis_boundaries(size):
    # Half-pixel padding limits, image edges, pixel centers and a pixel knot.
    values = [-0.2, -0.5 / size, 0.0, 0.5 / size,
              min(1.5 / size, 1.0), 0.5, 1.0 - 0.5 / size,
              1.0, 1.0 + 0.5 / size, 1.2]
    rows = []
    for value in values:
        scalar = torch.tensor(value, dtype=torch.float32)
        rows.extend([torch.nextafter(scalar, torch.tensor(float('-inf'))),
                     scalar,
                     torch.nextafter(scalar, torch.tensor(float('inf')))])
    return torch.stack(rows)


def build_inputs(spec):
    torch.manual_seed(spec['seed'])
    shapes = torch.tensor(spec['spatial_shapes'], dtype=torch.long)
    sizes = shapes.prod(1)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
    b, q, h, c, p, levels = (spec['batch'], spec['queries'], spec['heads'],
                             spec['channels'], spec['points'], len(shapes))
    # Preserve the original probe's RNG draw order without any extra draws.
    value = torch.randn(b, int(sizes.sum()), h, c)
    locations = torch.rand(b, q, h, levels, p, 2) * 1.4 - 0.2
    weights = torch.rand(b, q, h, levels, p)
    if spec['locations'] == 'boundaries':
        for level, (height, width) in enumerate(spec['spatial_shapes']):
            xs, ys = axis_boundaries(width), axis_boundaries(height)
            count = b * q * h * p
            indices = torch.arange(count)
            # Different cycling periods provide edge/corner and interior pairs.
            plane = torch.stack([xs[indices % len(xs)],
                                 ys[(indices // len(xs) + indices * 7) % len(ys)]], -1)
            locations[:, :, :, level, :, :] = plane.view(b, q, h, p, 2)
    if spec['weights'] == 'softmax':
        weights = weights.flatten(-2).softmax(-1).view(b, q, h, levels, p)
    elif spec['weights'] == 'zero':
        weights.zero_()
    elif spec['weights'] == 'onepoint':
        weights.zero_()
        weights[..., 0, 0] = 1.0
    elif spec['weights'] == 'high':
        weights = (weights - 0.5) * 128.0
    if spec['noncontiguous']:
        value, locations, weights = [offset_layout_copy(x, requires_grad=True)
                                    for x in (value, locations, weights)]
        shapes, starts = [offset_layout_copy(x) for x in (shapes, starts)]
    else:
        value, locations, weights = [x.requires_grad_(True)
                                    for x in (value, locations, weights)]
    return (value, locations, weights), shapes, starts


def unravel(flat, shape):
    result = []
    for width in reversed(shape):
        result.append(flat % width)
        flat //= width
    return list(reversed(result))


def point_context(index, name, spec, inputs):
    if name not in ('sampling_locations', 'attention_weights'):
        return None
    b, q, head, level, point = index[:5]
    value, locations, weights = inputs
    loc = locations.detach()[b, q, head, level, point]
    height, width = spec['spatial_shapes'][level]
    # Explicitly use the CPU32 vectorized GridSampler arithmetic order.
    grid = loc * 2.0 - 1.0
    pixel = (grid + 1.0) * torch.tensor([width / 2.0, height / 2.0]) - 0.5
    x0, y0 = int(torch.floor(pixel[0])), int(torch.floor(pixel[1]))
    start = sum(h * w for h, w in spec['spatial_shapes'][:level])
    neighbours = []
    for name_, x, y in [('nw', x0, y0), ('ne', x0 + 1, y0),
                        ('sw', x0, y0 + 1), ('se', x0 + 1, y0 + 1)]:
        valid = 0 <= x < width and 0 <= y < height
        vector = value.detach()[b, start + y * width + x, head] if valid else torch.zeros(spec['channels'])
        neighbours.append(dict(corner=name_, x=x, y=y, valid=valid,
                               first_eight_channels=vector[:8].tolist(),
                               max_abs=float(vector.abs().max())))
    return dict(location=loc.tolist(), cpu32_grid=grid.tolist(),
                cpu32_pixel=pixel.tolist(), width=width, height=height,
                attention_weight=float(weights.detach()[b, q, head, level, point]),
                four_neighbours=neighbours)


def compare(actual, reference, atol, rtol, name, spec, inputs):
    actual = actual.detach().cpu()
    reference = reference.detach().cpu()
    if actual.shape != reference.shape:
        return dict(allclose=False, shape_error=True, actual_shape=list(actual.shape),
                    reference_shape=list(reference.shape))
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    delta = (actual - reference).abs()
    threshold = atol + rtol * reference.abs()
    bad = (~finite) | (delta > threshold)
    passed = bool(finite.all()) and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol,
                                 equal_nan=False))
    a64, r64 = actual.double(), reference.double()
    diff64 = (a64 - r64).abs()
    samples = []
    flat_bad = bad.reshape(-1).nonzero().flatten()
    if len(flat_bad):
        # Report the worst failures, rather than only the first flat indices.
        errors = diff64.reshape(-1)[flat_bad]
        scores = torch.nan_to_num(errors, nan=float('inf'), posinf=float('inf'))
        chosen = flat_bad[torch.topk(scores, min(MAX_SAMPLES, len(flat_bad))).indices]
        for flat in chosen.tolist():
            index = unravel(flat, list(reference.shape))
            key = tuple(index)
            sample = dict(index=index, actual=float(actual[key]),
                          reference=float(reference[key]),
                          absolute_difference=float(diff64[key]),
                          allowed_difference=float(threshold[key]))
            context = point_context(index, name, spec, inputs)
            if context is not None:
                sample['point_context'] = context
            samples.append(sample)
    return dict(allclose=passed, shape=list(actual.shape), dtype=str(actual.dtype),
                elements=actual.numel(), finite_elements=int(finite.sum()),
                failing_elements=int(bad.sum()), max_abs=float(diff64.max()),
                relative_l2=float((a64 - r64).norm() / r64.norm().clamp_min(1e-30)),
                reference_max_abs=float(r64.abs().max()),
                absolute_tolerance=atol, relative_tolerance=rtol,
                actual_tensor_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
                reference_tensor_sha256=hashlib.sha256(reference.contiguous().numpy().tobytes()).hexdigest(),
                failure_samples=samples)


def persist(report):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.json.tmp')
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value
    temporary.write_text(json.dumps(safe(report), indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(OUTPUT)


def run_case(spec, cpu_reference, mmcv_function, mx_function):
    started = time.monotonic()
    inputs, shapes, starts = build_inputs(spec)
    cpu_start = time.monotonic()
    expected = cpu_reference(inputs[0], shapes, inputs[1], inputs[2])
    assert expected.device.type == 'cpu', 'The independent reference must remain CPU32.'
    assert expected.dtype == torch.float32
    grad_output = torch.randn_like(expected)
    if spec['noncontiguous']:
        grad_output = offset_layout_copy(grad_output)
    expected.backward(grad_output)
    reference_grads = [x.grad.detach().clone() for x in inputs]
    row = dict(**spec, cpu_reference_seconds=time.monotonic() - cpu_start,
               cpu_input_layouts=[tensor_layout(x) for x in inputs],
               cpu_grad_output_layout=tensor_layout(grad_output), entries=[])
    for entry in ('mmcv', 'mx_driving'):
        npu_inputs = actual = npu_shapes = npu_starts = npu_grad = None
        entry_start = time.monotonic()
        result = dict(entry=entry)
        try:
            npu_inputs = [transfer(x, spec['noncontiguous'], True) for x in inputs]
            npu_shapes, npu_starts = [transfer(x, spec['noncontiguous'])
                                     for x in (shapes, starts)]
            npu_grad = transfer(grad_output, spec['noncontiguous'])
            result['npu_input_layouts'] = [tensor_layout(x) for x in npu_inputs]
            result['npu_metadata_layouts'] = [tensor_layout(x) for x in (npu_shapes, npu_starts)]
            result['npu_grad_output_layout'] = tensor_layout(npu_grad)
            if entry == 'mmcv':
                actual = mmcv_function.apply(npu_inputs[0], npu_shapes, npu_starts,
                                              npu_inputs[1], npu_inputs[2], 64)
            else:
                actual = mx_function(npu_inputs[0], npu_shapes, npu_starts,
                                     npu_inputs[1], npu_inputs[2])
            assert actual.device.type == 'npu', 'NPU entry unexpectedly returned a CPU tensor.'
            actual.backward(npu_grad)
            torch.npu.synchronize()
            result['forward'] = compare(actual, expected, FORWARD_ATOL, FORWARD_RTOL,
                                         'forward', spec, inputs)
            result['gradients'] = {}
            for name, x, ref_grad in zip(('value', 'sampling_locations', 'attention_weights'),
                                         npu_inputs, reference_grads):
                if x.grad is None:
                    result['gradients'][name] = dict(allclose=False, missing_gradient=True)
                else:
                    result['gradients'][name] = compare(x.grad, ref_grad, GRAD_ATOL, GRAD_RTOL,
                                                         name, spec, inputs)
            result['passed'] = result['forward']['allclose'] and all(
                item['allclose'] for item in result['gradients'].values())
        except Exception as error:
            result.update(passed=False, exception=repr(error),
                          traceback=traceback.format_exc()[-6000:])
        finally:
            result['seconds'] = time.monotonic() - entry_start
            row['entries'].append(result)
            del actual, npu_inputs, npu_shapes, npu_starts, npu_grad
            gc.collect()
            try:
                torch.npu.synchronize()
                torch.npu.empty_cache()
            except Exception as cleanup_error:
                result['cleanup_exception'] = repr(cleanup_error)
                result['passed'] = False
    row['passed'] = len(row['entries']) == 2 and all(x['passed'] for x in row['entries'])
    row['seconds'] = time.monotonic() - started
    return row


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    from torch_npu.contrib import transfer_to_npu  # Match the pinned training environment.
    import mx_driving
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
    from ascend.compat import enable_training_compat
    enable_training_compat()
    # Match this turn's reproduced baseline and candidate coordinate probe.
    # This requests deterministic operations; it does not establish bitwise
    # repeatability of an external fused kernel that does not honor the flag.
    torch.use_deterministic_algorithms(True)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = True
    torch.manual_seed(29)
    torch.npu.manual_seed_all(29)
    started = time.monotonic()
    report = dict(result='running', reference='unmodified_mmcv_cpu32_pytorch',
                  forward_atol=FORWARD_ATOL, forward_rtol=FORWARD_RTOL,
                  gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL,
                  torch_version=str(torch.__version__), torch_npu_version=torch_npu.__version__,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  npu_matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                  npu_conv_allow_hf32=torch.npu.conv.allow_hf32,
                  hccl_deterministic=os.environ.get('HCCL_DETERMINISTIC'),
                  close_matmul_k_shift=os.environ.get('CLOSE_MATMUL_K_SHIFT'),
                  threads=torch.get_num_threads(), expected_cases=len(CASES), rows=[],
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  compat_file=inspect.getsourcefile(enable_training_compat),
                  scope='All outputs and all three gradients compared; FP32 only. '
                        'Q40000 uses reduced heads/channels; full actual geometry requires model regression. '
                        'No CPU64 substitution, GPU parity, convergence or strict replay claim.')
    compat_path = Path(report['compat_file'])
    report['compat_sha256'] = hashlib.sha256(compat_path.read_bytes()).hexdigest()
    report['cpu_reference_source_sha256'] = hashlib.sha256(
        inspect.getsource(multi_scale_deformable_attn_pytorch).encode()).hexdigest()
    report['entrypoints'] = dict(
        mx_driving_module=mx_driving.multi_scale_deformable_attn.__module__,
        mx_driving_qualname=getattr(mx_driving.multi_scale_deformable_attn, '__qualname__', None),
        mmcv_forward_module=MultiScaleDeformableAttnFunction.forward.__module__,
        mmcv_forward_qualname=getattr(MultiScaleDeformableAttnFunction.forward, '__qualname__', None))
    persist(report)
    for spec in CASES:
        try:
            row = run_case(spec, multi_scale_deformable_attn_pytorch,
                           MultiScaleDeformableAttnFunction,
                           mx_driving.multi_scale_deformable_attn)
        except Exception as error:
            row = dict(**spec, passed=False, cpu_or_setup_exception=repr(error),
                       traceback=traceback.format_exc()[-6000:])
        report['rows'].append(row)
        persist(report)
        print(json.dumps(dict(case=spec['name'], passed=row['passed'],
                              entries=[dict(entry=x['entry'], passed=x['passed'],
                                            location_failures=x.get('gradients', {}).get('sampling_locations', {}).get('failing_elements'),
                                            exception=x.get('exception'))
                                       for x in row.get('entries', [])],
                              seconds=row.get('seconds'))), flush=True)
        del row
        gc.collect()
    report['seconds'] = time.monotonic() - started
    report['cpu_reference_total_seconds'] = sum(x.get('cpu_reference_seconds', 0.0)
                                                for x in report['rows'])
    report['passed_cases'] = sum(x['passed'] for x in report['rows'])
    report['passed_entries'] = sum(e['passed'] for x in report['rows'] for e in x.get('entries', []))
    report['result'] = ('pass_strict_cpu32_matrix_both_npu_entries'
                        if len(report['rows']) == len(CASES) and all(x['passed'] for x in report['rows'])
                        else 'fail_strict_cpu32_matrix')
    persist(report)
    print(json.dumps(dict(result=report['result'], cases=len(CASES),
                          passed_cases=report['passed_cases'], passed_entries=report['passed_entries'],
                          seconds=report['seconds'], cpu_seconds=report['cpu_reference_total_seconds'],
                          output=str(OUTPUT))), flush=True)
    return 0 if report['result'].startswith('pass_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
