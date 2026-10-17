"""Validate the original panoptic-map MSDA level-layout compatibility on NPU.

The affected original map decoder supplies four raw location/weight levels but
only one spatial level. Such raw inputs are not valid inputs to MMCV's PyTorch
reference. The independent CPU32 oracle therefore first interprets the raw
storage with the original CUDA kernel's global flat-prefix indexing, then calls
the unmodified MMCV reference with valid one-level views. CPU autograd writes
gradients back to the original four-level leaves, including exact-zero unused
tails. This is a layout/CPU32 numerical test, not measured GPU numerical parity.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import traceback


FORWARD_ATOL, FORWARD_RTOL = 2e-5, 2e-5
GRADIENT_ATOL, GRADIENT_RTOL = 1e-3, 1e-4
RAW_LEVELS = 4
ACTUAL_LEVELS = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/output/msda_layout.json'))
    return parser.parse_args()


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def persist(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(safe(report), indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def layout(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                contiguous=tensor.is_contiguous(), storage_offset=tensor.storage_offset(),
                dtype=str(tensor.dtype), device=str(tensor.device))


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def callable_description(function):
    result = dict(module=getattr(function, '__module__', None),
                  qualname=getattr(function, '__qualname__', getattr(function, '__name__', None)))
    try:
        filename = inspect.getsourcefile(function)
        if filename:
            path = Path(filename)
            result.update(source_file=str(path), source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (TypeError, OSError):
        pass
    return result


def compare(torch, actual, reference, atol, rtol):
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    if actual.shape != reference.shape:
        return dict(allclose=False, actual_shape=list(actual.shape),
                    reference_shape=list(reference.shape), shape_mismatch=True)
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    diff = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    bad = (~finite) | (diff > allowed)
    passed = bool(finite.all()) and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False))
    a64, r64 = actual.double(), reference.double()
    delta64 = (a64 - r64).abs()
    samples = []
    indices = bad.reshape(-1).nonzero().flatten()
    if len(indices):
        errors = torch.nan_to_num(delta64.reshape(-1)[indices], nan=float('inf'), posinf=float('inf'))
        selected = indices[torch.topk(errors, min(12, len(indices))).indices]
        for flat_index in selected.tolist():
            index, remainder = [], flat_index
            for size in reversed(actual.shape):
                index.append(remainder % size)
                remainder //= size
            index.reverse()
            key = tuple(index)
            samples.append(dict(index=index, actual=float(actual[key]), reference=float(reference[key]),
                                absolute_difference=float(delta64[key]), allowed_difference=float(allowed[key])))
    return dict(allclose=passed, shape=list(actual.shape), elements=actual.numel(),
                finite_elements=int(finite.sum()), failing_elements=int(bad.sum()),
                max_abs=float(delta64.max()),
                relative_l2=float((a64 - r64).norm() / r64.norm().clamp_min(1e-30)),
                absolute_tolerance=atol, relative_tolerance=rtol,
                actual_sha256=fingerprint(actual), reference_sha256=fingerprint(reference),
                failure_samples=samples)


def global_prefix(locations, weights, actual_levels=ACTUAL_LEVELS):
    batch, queries, heads, _, points, _ = locations.shape
    used = batch * queries * heads * actual_levels * points
    effective_locations = locations.reshape(-1, 2)[:used].reshape(
        batch, queries, heads, actual_levels, points, 2)
    effective_weights = weights.reshape(-1)[:used].reshape(
        batch, queries, heads, actual_levels, points)
    return effective_locations, effective_weights, used


def per_query_prefix(locations, weights, actual_levels=ACTUAL_LEVELS):
    """Diagnostic hypothesis only; this does not define the compatibility gate."""
    batch, queries, heads, _, points, _ = locations.shape
    per_query = heads * actual_levels * points
    effective_locations = locations.reshape(batch, queries, -1, 2)[:, :, :per_query].reshape(
        batch, queries, heads, actual_levels, points, 2)
    effective_weights = weights.reshape(batch, queries, -1)[:, :, :per_query].reshape(
        batch, queries, heads, actual_levels, points)
    return effective_locations, effective_weights


def build_inputs(torch, spec):
    torch.manual_seed(spec['seed'])
    batch, queries, heads, channels, points = (spec['batch'], spec['queries'], spec['heads'],
                                              spec['channels'], spec['points'])
    height, width = spec['spatial_shape']
    if spec['pattern'] == 'encoded':
        value = (torch.arange(heads, dtype=torch.float32) + 1).view(1, 1, heads, 1).expand(
            batch, height * width, heads, channels).clone()
        locations = torch.full((batch, queries, heads, RAW_LEVELS, points, 2), 0.5)
        b = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1, 1)
        q = torch.arange(queries, dtype=torch.float32).view(1, queries, 1, 1, 1)
        h = torch.arange(heads, dtype=torch.float32).view(1, 1, heads, 1, 1)
        level = torch.arange(RAW_LEVELS, dtype=torch.float32).view(1, 1, 1, RAW_LEVELS, 1)
        point = torch.arange(points, dtype=torch.float32).view(1, 1, 1, 1, points)
        weights = b * 10000 + q * 1000 + h * 100 + level * 10 + point + 1
    else:
        value = torch.randn(batch, height * width, heads, channels)
        locations = torch.rand(batch, queries, heads, RAW_LEVELS, points, 2) * 1.4 - 0.2
        weights = torch.rand(batch, queries, heads, RAW_LEVELS, points)
    inputs = [item.requires_grad_(True) for item in (value, locations, weights)]
    shapes = torch.tensor([[height, width]], dtype=torch.long)
    starts = torch.zeros(1, dtype=torch.long)
    return inputs, shapes, starts


def tail_check(torch, gradient, used, coordinates=False):
    flat = gradient.detach().reshape(-1, 2) if coordinates else gradient.detach().reshape(-1)
    tail = flat[used:]
    return dict(elements=tail.numel(), nonzero_elements=int(torch.count_nonzero(tail)),
                max_abs=float(tail.abs().max()) if tail.numel() else 0.0,
                exact_zero=bool(torch.count_nonzero(tail) == 0))


def run_case(torch, spec, cpu_reference, mmcv_function):
    started = time.monotonic()
    inputs, shapes, starts = build_inputs(torch, spec)
    effective_locations, effective_weights, used = global_prefix(inputs[1], inputs[2])
    # The CPU reference is unchanged and sees valid L1 shapes. The slicing and
    # reshape are autograd views of the original raw L4 leaves.
    expected = cpu_reference(inputs[0], shapes, effective_locations, effective_weights)
    assert expected.device.type == 'cpu' and expected.dtype == torch.float32
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    reference_gradients = [item.grad.detach().clone() for item in inputs]
    npu_inputs = [item.detach().to('npu').requires_grad_(True) for item in inputs]
    row = dict(**spec, raw_levels=RAW_LEVELS, actual_spatial_levels=ACTUAL_LEVELS,
               used_global_prefix_points=used,
               call=dict(api='mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.apply',
                         im2col_step=64, inputs=[layout(x) for x in npu_inputs],
                         spatial_shapes=shapes.tolist(), level_start_index=starts.tolist()),
               raw_input_sha256={name: fingerprint(item) for name, item in
                                 zip(('value', 'locations', 'weights'), inputs)},
               effective_cpu_layouts=[layout(effective_locations), layout(effective_weights)])
    actual = mmcv_function.apply(npu_inputs[0], shapes.to('npu'), starts.to('npu'),
                                npu_inputs[1], npu_inputs[2], 64)
    if actual.device.type != 'npu':
        raise RuntimeError('The patched MMCV path returned a non-NPU output')
    actual.backward(upstream.to('npu'))
    torch.npu.synchronize()
    row['actual_output_layout'] = layout(actual)
    row['forward'] = compare(torch, actual, expected, FORWARD_ATOL, FORWARD_RTOL)
    row['gradients'] = {}
    for name, raw, reference in zip(('value', 'sampling_locations', 'attention_weights'),
                                    npu_inputs, reference_gradients):
        if raw.grad is None:
            row['gradients'][name] = dict(allclose=False, missing_gradient=True)
        else:
            row['gradients'][name] = compare(torch, raw.grad, reference, GRADIENT_ATOL, GRADIENT_RTOL)
            row['gradients'][name]['raw_gradient_shape_preserved'] = tuple(raw.grad.shape) == tuple(raw.shape)
            row['gradients'][name]['actual_gradient_device'] = str(raw.grad.device)
            if name in ('sampling_locations', 'attention_weights') and raw.grad.shape == raw.shape:
                coordinates = name == 'sampling_locations'
                row['gradients'][name]['unused_tail'] = dict(
                    npu=tail_check(torch, raw.grad, used, coordinates),
                    cpu=tail_check(torch, reference, used, coordinates))
    row['passed'] = row['forward']['allclose'] and all(
        item['allclose'] and item.get('raw_gradient_shape_preserved', False) and
        all(tail['exact_zero'] for tail in item.get('unused_tail', {}).values())
        for item in row['gradients'].values())
    row['seconds'] = time.monotonic() - started
    return row


def sdk_diagnostic(torch, spec, cpu_reference, mx_driving):
    started = time.monotonic()
    row = dict(result='diagnostic_only', affects_fix_gate=False,
               api='mx_driving._C.multi_scale_deformable_attn',
               scope='Raw installed SDK forward is compared with two storage-indexing hypotheses. '
                     'A mismatch or SDK exception is recorded, not classified as a compatibility-fix gate failure.')
    try:
        inputs, shapes, starts = build_inputs(torch, spec)
        locations_global, weights_global, _ = global_prefix(inputs[1], inputs[2])
        locations_query, weights_query = per_query_prefix(inputs[1], inputs[2])
        with torch.no_grad():
            global_output = cpu_reference(inputs[0], shapes, locations_global, weights_global)
            query_output = cpu_reference(inputs[0], shapes, locations_query, weights_query)
            npu_inputs = [item.detach().to('npu').contiguous() for item in inputs]
            actual = mx_driving._C.multi_scale_deformable_attn(
                npu_inputs[0], shapes.to('npu', dtype=torch.int32).contiguous(),
                starts.to('npu', dtype=torch.int32).contiguous(), npu_inputs[1], npu_inputs[2])
            torch.npu.synchronize()
        row['input_layouts'] = [layout(item) for item in npu_inputs]
        row['actual_output_layout'] = layout(actual)
        row['vs_cuda_global_prefix_cpu32'] = compare(torch, actual, global_output, FORWARD_ATOL, FORWARD_RTOL)
        row['vs_per_query_prefix_cpu32'] = compare(torch, actual, query_output, FORWARD_ATOL, FORWARD_RTOL)
        row['hypotheses_differ'] = not torch.allclose(global_output, query_output,
                                                      atol=FORWARD_ATOL, rtol=FORWARD_RTOL)
        row['first_channel_by_query_and_head'] = dict(
            sdk=actual.detach().cpu().view(spec['batch'], spec['queries'], spec['heads'], spec['channels'])[..., 0].tolist(),
            cuda_global_prefix=global_output.view(spec['batch'], spec['queries'], spec['heads'], spec['channels'])[..., 0].tolist(),
            per_query_prefix=query_output.view(spec['batch'], spec['queries'], spec['heads'], spec['channels'])[..., 0].tolist())
    except Exception as error:
        row.update(exception=repr(error), traceback=traceback.format_exc()[-6000:])
    row['seconds'] = time.monotonic() - started
    return row


def main():
    args = parse_args()
    report = dict(result='running', cases=[],
                  reference='unmodified_mmcv_pytorch_cpu32_after_valid_global_flat_prefix_view',
                  storage_rule='raw locations.reshape(-1,2)[:B*Q*heads*actualL*P] and '
                               'raw weights.reshape(-1)[:B*Q*heads*actualL*P], reshaped to valid actualL',
                  forward_atol=FORWARD_ATOL, forward_rtol=FORWARD_RTOL,
                  gradient_atol=GRADIENT_ATOL, gradient_rtol=GRADIENT_RTOL,
                  scope='Known original panoptic-map raw L4 versus spatial L1 storage compatibility only. '
                        'The raw L4 tensors cannot be passed directly to the PyTorch CPU oracle. '
                        'All floating outputs/three full raw gradients are checked. Unused raw tails must be exactly zero. '
                        'No GPU device comparison, full-data quality, convergence or bitwise replay acceptance. '
                        'This script records dispatched framework APIs/devices; native CANN kernel traces are separate evidence.')
    persist(args.output, report)
    try:
        import gc
        import torch
        import torch_npu
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        from torch_npu.contrib import transfer_to_npu
        import mx_driving
        from mmcv.ops.multi_scale_deform_attn import (
            MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
        from ascend.compat import enable_training_compat
        enable_training_compat()
        torch.use_deterministic_algorithms(True)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = True
        report['environment'] = dict(torch_version=str(torch.__version__),
                                     torch_npu_version=torch_npu.__version__,
                                     mx_driving_version=getattr(mx_driving, '__version__', None),
                                     npu_device=str(torch.device('npu', 0)),
                                     npu_name=torch.npu.get_device_name(0),
                                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                     matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                                     conv_allow_hf32=torch.npu.conv.allow_hf32)
        report['entrypoints'] = dict(
            patched_mmcv_forward=callable_description(MultiScaleDeformableAttnFunction.forward),
            cpu_reference=callable_description(multi_scale_deformable_attn_pytorch),
            raw_sdk_forward=callable_description(mx_driving._C.multi_scale_deformable_attn))
        report['cpu_reference_source_sha256'] = hashlib.sha256(
            inspect.getsource(multi_scale_deformable_attn_pytorch).encode()).hexdigest()
        specifications = [
            dict(name='encoded_layout_B1', spatial_shape=[5, 7], batch=1, queries=3,
                 heads=8, channels=32, points=4, seed=29, pattern='encoded'),
            dict(name='large_raw_map_seed29', spatial_shape=[200, 200], batch=1, queries=300,
                 heads=8, channels=32, points=4, seed=29, pattern='random'),
            dict(name='small_raw_map_B2', spatial_shape=[5, 7], batch=2, queries=13,
                 heads=8, channels=32, points=4, seed=29, pattern='random'),
            dict(name='large_raw_map_seed7', spatial_shape=[200, 200], batch=1, queries=300,
                 heads=8, channels=32, points=4, seed=7, pattern='random'),
        ]
        started = time.monotonic()
        for spec in specifications:
            try:
                row = run_case(torch, spec, multi_scale_deformable_attn_pytorch,
                               MultiScaleDeformableAttnFunction)
            except Exception as error:
                row = dict(**spec, passed=False, exception=repr(error),
                           traceback=traceback.format_exc()[-7000:])
            report['cases'].append(row)
            persist(args.output, report)
            print(json.dumps(dict(case=spec['name'], passed=row['passed'],
                                  exception=row.get('exception'))), flush=True)
            gc.collect()
            try:
                torch.npu.synchronize()
                torch.npu.empty_cache()
            except Exception as error:
                row['cleanup_exception'] = repr(error)
                row['passed'] = False
        report['sdk_diagnostic'] = sdk_diagnostic(torch, specifications[0],
                                                 multi_scale_deformable_attn_pytorch, mx_driving)
        report['seconds'] = time.monotonic() - started
        report['passed_cases'] = sum(row['passed'] for row in report['cases'])
        report['result'] = ('pass_raw_map_global_prefix_cpu32_compatibility'
                            if len(report['cases']) == len(specifications) and
                            all(row['passed'] for row in report['cases'])
                            else 'fail_raw_map_global_prefix_cpu32_compatibility')
    except Exception as error:
        report.update(result='fail_setup', exception=repr(error),
                      traceback=traceback.format_exc()[-7000:])
    persist(args.output, report)
    print(json.dumps(dict(result=report['result'], passed_cases=report.get('passed_cases', 0),
                          output=str(args.output))), flush=True)
    return 0 if report['result'].startswith('pass_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
