"""Original MMCV 1.6.2 CUDA MSDA: full FP32 CPU-reference and repeat checks.

Default dry-run imports no framework and writes no output. With --execute,
complete outputs and three gradients remain on the selected GPU until all
repetitions of a case finish. Full CPU comparisons use unchanged thresholds.
An external process timeout is needed if a CUDA C++ call does not return.
No model, native operator, CPU reference or NPU implementation is replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import uuid
import warnings

EXPECTED_PYTHON = {
    'CPU_reference': '4ef9c84cb9016a3554b7fd62d35a6e8f8cb440099103679de01fc0491eaf5128',
    'CUDA_forward': '3608dc1cd24471538c289cfa89d3866f7c496b18c9f00e85c9c1143385a4c332',
    'CUDA_backward': '7efd29b77fbc5baf6abeb462b2169080c2f0891d17c278c0c09d00cbef0adbb8',
}
FORWARD_ATOL, FORWARD_RTOL = 2e-5, 2e-5
GRAD_ATOL, GRAD_RTOL = 1e-3, 1e-4
CPU_THREADS = 4
CASES = {
    'temporal40000': dict(name='temporal_B2_full_Q40000_H8_C32_P4', batch=2,
                          queries=40000, spatial_shapes=[[200, 200]],
                          heads=8, channels=32, points=4, seed=29),
    'query65536': dict(name='actual_Q65536_H8_C32_P4_seed29', batch=1,
                       queries=65536, spatial_shapes=[[200, 200]],
                       heads=8, channels=32, points=4, seed=29),
}
ROLES = ('forward', 'value_gradient', 'locations_gradient', 'weights_gradient')


class Cutoff(BaseException):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(function, reference, extension):
    source = Path(inspect.getsourcefile(function))
    return dict(extension=sha(extension.__file__),
                CPU_reference=hashlib.sha256(inspect.getsource(reference).encode()).hexdigest(),
                CUDA_forward=hashlib.sha256(inspect.getsource(function.forward).encode()).hexdigest(),
                CUDA_backward=hashlib.sha256(inspect.getsource(function.backward).encode()).hexdigest(),
                Python_module_file=str(source), Python_module_sha256=sha(source))


def precision(torch):
    return dict(default_dtype=str(torch.get_default_dtype()),
                matmul_TF32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_TF32=bool(torch.backends.cudnn.allow_tf32),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                autocast_enabled=bool(torch.is_autocast_enabled()),
                deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                CPU_threads=torch.get_num_threads())


def check_precision(row, threads):
    if (row['default_dtype'] != 'torch.float32' or row['matmul_TF32'] or
            row['cudnn_TF32'] or row['cudnn_benchmark'] or
            not row['cudnn_deterministic'] or row['autocast_enabled'] or
            not row['deterministic_algorithms'] or row['deterministic_warn_only'] or
            row['CPU_threads'] != threads):
        raise RuntimeError('Actual strict deterministic FP32/TF32-off policy differs')


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def persist(path, report):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), dtype=str(tensor.dtype),
                device=str(tensor.device), contiguous=tensor.is_contiguous(),
                requires_grad=tensor.requires_grad)


def tensor_sha(tensor):
    if tensor.device.type != 'cpu':
        raise RuntimeError('Tensor hashing belongs only to the post-repeat CPU phase')
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def compare(torch, actual, reference, role):
    if actual.device.type != 'cpu' or reference.device.type != 'cpu':
        raise RuntimeError('Complete comparisons must execute on CPU after repetitions')
    if actual.dtype != torch.float32 or reference.dtype != torch.float32 or actual.shape != reference.shape:
        raise RuntimeError('Complete original FP32 tensor identity differs')
    atol, rtol = ((FORWARD_ATOL, FORWARD_RTOL) if role == 'forward'
                  else (GRAD_ATOL, GRAD_RTOL))
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    delta32 = (actual - reference).abs()
    bad = (~finite) | (delta32 > atol + rtol * reference.abs())
    delta64 = actual.double() - reference.double()
    actual_hash, reference_hash = tensor_sha(actual), tensor_sha(reference)
    return dict(shape=list(actual.shape), elements=actual.numel(), dtype=str(actual.dtype),
                all_finite=bool(finite.all()), failing_elements=int(bad.sum()),
                max_absolute_difference=float(delta64.abs().max()),
                mean_absolute_difference=float(delta64.abs().mean()),
                relative_l2=float(delta64.norm() / reference.double().norm().clamp_min(1e-30)),
                bitwise_equal=actual_hash == reference_hash,
                original_gate_passed=bool(torch.allclose(actual, reference, atol=atol, rtol=rtol,
                                                        equal_nan=False)),
                absolute_tolerance=atol, relative_tolerance=rtol,
                actual_tensor_sha256=actual_hash, reference_tensor_sha256=reference_hash)


def inputs(torch, spec):
    # Same CPU FP32 RNG draw order and formula as ascend.validate_msda.build_inputs.
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    shapes = torch.tensor(spec['spatial_shapes'], dtype=torch.long)
    sizes = shapes.prod(1)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
    b, q, h, c, p, levels = tuple(spec[k] for k in ('batch', 'queries', 'heads', 'channels', 'points')) + (len(shapes),)
    raw = [torch.randn((b, int(sizes.sum()), h, c), generator=generator),
           torch.rand((b, q, h, levels, p, 2), generator=generator) * 1.4 - 0.2,
           torch.rand((b, q, h, levels, p), generator=generator)]
    go = torch.randn((b, q, h * c), generator=generator)
    return raw, shapes, starts, go


def run_case(torch, spec, function, reference, repetitions, device):
    started = time.monotonic()
    raw, shapes, starts, go = inputs(torch, spec)
    before = [tensor_sha(x) for x in raw + [shapes, starts, go]]
    cpu = [x.detach().clone().requires_grad_(True) for x in raw]
    expected = reference(cpu[0], shapes, cpu[1], cpu[2])
    if expected.device.type != 'cpu' or expected.dtype != torch.float32:
        raise RuntimeError('Original MMCV oracle must remain complete CPU FP32')
    expected.backward(go)
    if any(x.grad is None for x in cpu):
        raise RuntimeError('Original full CPU reference gradient missing')
    reference_outputs = [expected.detach()] + [x.grad.detach() for x in cpu]
    row = dict(**spec, CPU_reference_seconds=time.monotonic() - started,
               input_sha256_before=before, cpu_input_metadata=[tensor_metadata(x) for x in raw],
               grad_output_metadata=tensor_metadata(go), repetitions=repetitions)
    gpu = [x.detach().to(device).requires_grad_(True) for x in raw]
    gpu_shapes, gpu_starts, gpu_go = [x.to(device) for x in (shapes, starts, go)]
    row['GPU_input_metadata'] = [tensor_metadata(x) for x in gpu]
    row['GPU_metadata_tensor_metadata'] = [tensor_metadata(x) for x in (gpu_shapes, gpu_starts, gpu_go)]
    row['GPU_input_data_ptrs'] = [x.data_ptr() for x in gpu]
    retained = []
    loop_started = time.monotonic()
    # All executions use the same input/metadata/grad_output storages. No CPU
    # tensor read, .item(), validation sync or per-repeat reseed in this loop.
    for _ in range(repetitions):
        for x in gpu:
            x.grad = None
        actual = function.apply(gpu[0], gpu_shapes, gpu_starts, gpu[1], gpu[2], 64)
        actual.backward(gpu_go)
        if actual.device != device or actual.dtype != torch.float32 or any(x.grad is None for x in gpu):
            raise RuntimeError('Actual original CUDA output/gradient missing or wrong device/dtype')
        retained.append([actual.detach()] + [x.grad.detach() for x in gpu])
    torch.cuda.synchronize(device)
    row['GPU_repeat_loop_seconds'] = time.monotonic() - loop_started
    row['retained_complete_GPU_bytes'] = sum(x.numel() * x.element_size() for values in retained for x in values)
    actual_cpu = [[x.cpu() for x in values] for values in retained]
    row['GPU_inputs_unchanged_after_all_repetitions'] = [tensor_sha(x.cpu()) for x in gpu] == before[:3]
    row['GPU_metadata_and_grad_output_unchanged_after_all_repetitions'] = [tensor_sha(x.cpu()) for x in (gpu_shapes, gpu_starts, gpu_go)] == before[3:]
    row['CPU_inputs_unchanged_after_all_repetitions'] = [tensor_sha(x) for x in raw + [shapes, starts, go]] == before
    row['comparisons_to_original_CPU32'] = [dict(repetition=i, tensors={
        role: compare(torch, left, right, role) for role, left, right in zip(ROLES, values, reference_outputs)})
        for i, values in enumerate(actual_cpu)]
    row['comparisons_to_GPU_repetition0'] = [dict(repetition=i, reference_repetition=0, tensors={
        role: compare(torch, left, right, role) for role, left, right in zip(ROLES, values, actual_cpu[0])})
        for i, values in enumerate(actual_cpu[1:], 1)]
    row['all_CPU32_original_gates_passed'] = all(x['original_gate_passed'] for r in row['comparisons_to_original_CPU32'] for x in r['tensors'].values())
    row['all_repeat_original_gates_passed'] = all(x['original_gate_passed'] for r in row['comparisons_to_GPU_repetition0'] for x in r['tensors'].values())
    row['all_repeat_bitwise_equal'] = all(x['bitwise_equal'] for r in row['comparisons_to_GPU_repetition0'] for x in r['tensors'].values())
    row['comparison_tensor_count'] = 4 * (2 * repetitions - 1)
    if not all(row[k] for k in ('GPU_inputs_unchanged_after_all_repetitions', 'GPU_metadata_and_grad_output_unchanged_after_all_repetitions', 'CPU_inputs_unchanged_after_all_repetitions')):
        raise RuntimeError('Read-only MSDA inputs changed')
    row['seconds'] = time.monotonic() - started
    return row

def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True,
                   help='Fresh JSON report; its parent directory must exist.')
    p.add_argument('--device', default='cuda:0', help='One CUDA device, e.g. cuda:0.')
    p.add_argument('--maximum-seconds', type=int, default=300,
                   help='Relative process cutoff (1..86400 seconds); use an external timeout too.')
    p.add_argument('--case', dest='cases', action='append', choices=tuple(CASES),
                   help='Select a case; repeat to select both. Default: both full shapes.')
    p.add_argument('--repetitions', type=int, default=4, choices=range(4, 9),
                   help='Same-storage executions per case (default 4).')
    p.add_argument('--execute', action='store_true', help='Run the CUDA and complete CPU comparisons.')
    return p


def validate_args(p, a):
    names = a.cases if a.cases is not None else list(CASES)
    if len(names) != len(set(names)):
        p.error('Duplicate case selections are not allowed')
    if re.fullmatch(r'cuda(?::[0-9]+)?', a.device) is None:
        p.error('--device must be cuda or cuda:<nonnegative local index>')
    if not 1 <= a.maximum_seconds <= 86400:
        p.error('--maximum-seconds must be between 1 and 86400')
    return names


def small_summary(report, output=None):
    return dict(result=report['result'], actual_exit=report.get('actual_exit'),
                completed_cases=report.get('completed_cases', 0),
                full_tensor_comparison_count=report.get('full_tensor_comparison_count', 0),
                all_CPU32_original_gates_passed=report.get('all_CPU32_original_gates_passed'),
                all_repeat_original_gates_passed=report.get('all_repeat_original_gates_passed'),
                all_repeat_bitwise_equal=report.get('all_repeat_bitwise_equal'),
                error=report.get('error'), output=str(output) if output is not None else None)


def main():
    p = parser()
    a = p.parse_args()
    names = validate_args(p, a)
    report = dict(result='dry_run_no_framework_no_output_no_device',
                  validator_sha256=sha(__file__), cases=[CASES[n] for n in names],
                  requested_device=a.device, repetitions=a.repetitions,
                  CPU_threads=CPU_THREADS, maximum_seconds=a.maximum_seconds,
                  strict_deterministic_requested=True,
                  external_CUDA_kernel_deterministic_registration='unknown_not_established_by_global_flag',
                  forward_tolerance=dict(atol=FORWARD_ATOL, rtol=FORWARD_RTOL),
                  gradient_tolerance=dict(atol=GRAD_ATOL, rtol=GRAD_RTOL),
                  expected_original_Python_source=EXPECTED_PYTHON,
                  native_extension_binary_SHA_is_recorded_not_a_portability_gate=True,
                  native_extension_compilation_source_not_independently_attested=True,
                  scope='Installed original MMCV162 CUDA same-input full forward/three-gradient repetitions and unchanged CPU32 reference; no model, training, NPU or cross-platform equivalence claim',
                  no_CPU_tensor_reads_inside_GPU_repeat_loop=True, rows=[])
    if not a.execute:
        print(json.dumps(dict(**small_summary(report), selected_cases=names,
                              repetitions=a.repetitions, CPU_threads=CPU_THREADS,
                              forward_tolerance=report['forward_tolerance'],
                              gradient_tolerance=report['gradient_tolerance']), separators=(',', ':')))
        return 0
    output = a.output.expanduser().absolute()
    if (not sys.platform.startswith('linux') or not hasattr(signal, 'setitimer') or
            any(n == 'torch' or n == 'ascend' or n.startswith(('torch.', 'torch_npu', 'mx_driving', 'ascend.')) for n in sys.modules)):
        print(json.dumps(dict(result='incomplete_or_blocked_original_CUDA_MSDA_validation',
                              actual_exit=2, error='Fresh Linux process without framework preimports required')))
        return 2
    try:
        if (output.exists() or output.is_symlink() or not output.parent.is_dir() or
                output.parent.resolve(strict=True) != output.parent):
            raise RuntimeError('Fresh report under an existing canonical parent directory required')
        with output.open('xb') as stream:
            stream.write(b'{}\n')
    except (OSError, RuntimeError) as exc:
        print(json.dumps(dict(result='incomplete_or_blocked_original_CUDA_MSDA_validation',
                              actual_exit=2, error=str(exc))))
        return 2
    report.update(result='incomplete_or_blocked_original_CUDA_MSDA_validation',
                  started_UTC=datetime.now(timezone.utc).isoformat())
    end = time.monotonic() + a.maximum_seconds
    persist(output, report)
    def cutoff(signum, _frame):
        raise Cutoff('Relative MSDA process cutoff/signal ' + str(signum))
    signal.signal(signal.SIGALRM, cutoff)
    signal.signal(signal.SIGTERM, cutoff)
    signal.setitimer(signal.ITIMER_REAL, a.maximum_seconds)
    code, torch, function, reference, extension, before, policy = 2, None, None, None, None, None, None
    try:
        workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if workspace not in (':4096:8', ':16:8'):
            raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before Torch import')
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Fresh CUDA context required')
        mmcv_version = importlib.metadata.version('mmcv-full')
        if mmcv_version != '1.6.2':
            raise RuntimeError('Original mmcv-full==1.6.2 required')
        device = torch.device(a.device)
        local_index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or local_index >= torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device is unavailable')
        torch.cuda.set_device(local_index)
        device = torch.device('cuda', local_index)
        torch.set_num_threads(CPU_THREADS)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as function, multi_scale_deformable_attn_pytorch as reference
        import mmcv._ext as extension
        before = source_identity(function, reference, extension)
        if any(before[k] != value for k, value in EXPECTED_PYTHON.items()):
            raise RuntimeError('Original MMCV162 CPU32 reference or CUDA Python-wrapper source hash differs')
        policy = precision(torch)
        check_precision(policy, CPU_THREADS)
        free, total = torch.cuda.mem_get_info(device)
        if free < 4 * 1024**3:
            raise RuntimeError('At least 4GiB currently free on the selected GPU is required')
        torch.cuda.reset_peak_memory_stats(device)
        report.update(actual_precision_before=policy, source_before=before,
                      torch_version=str(torch.__version__), CUDA_build=torch.version.cuda,
                      mmcv_version=mmcv_version, numpy_version=importlib.metadata.version('numpy'),
                      device=dict(local_index=local_index, name=torch.cuda.get_device_name(local_index),
                                  capability=list(torch.cuda.get_device_capability(local_index)),
                                  free_bytes_before=free, total_bytes=total),
                      actual_environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH')})
        persist(output, report)
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter('always')
            for name in names:
                if time.monotonic() >= end:
                    raise Cutoff('Relative duration exhausted before next complete case')
                report['rows'].append(run_case(torch, CASES[name], function, reference, a.repetitions, device))
                persist(output, report)
            report['actual_warnings'] = [dict(category=x.category.__name__, message=str(x.message),
                                             filename=x.filename, lineno=x.lineno) for x in actual_warnings]
        report['actual_precision_after'] = precision(torch)
        check_precision(report['actual_precision_after'], CPU_THREADS)
        report['source_after'] = source_identity(function, reference, extension)
        if (report['source_after'] != before or report['actual_precision_after'] != policy or
                sha(__file__) != report['validator_sha256']):
            raise RuntimeError('Original source/precision/validator changed during execution')
        if any(n == 'ascend' or n.startswith(('ascend.', 'torch_npu', 'mx_driving')) for n in sys.modules):
            raise RuntimeError('Unexpected NPU implementation imported')
        report.update(completed_cases=len(report['rows']),
                      full_tensor_comparison_count=sum(r['comparison_tensor_count'] for r in report['rows']),
                      all_CPU32_original_gates_passed=all(r['all_CPU32_original_gates_passed'] for r in report['rows']),
                      all_repeat_original_gates_passed=all(r['all_repeat_original_gates_passed'] for r in report['rows']),
                      all_repeat_bitwise_equal=all(r['all_repeat_bitwise_equal'] for r in report['rows']),
                      original_source_unchanged=True, original_math_or_dropout_modified=False,
                      peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(device))
        inside = report['all_CPU32_original_gates_passed'] and report['all_repeat_original_gates_passed']
        report['result'] = ('completed_original_CUDA_MSDA_within_original_gates_not_model'
                            if inside else 'completed_original_CUDA_MSDA_outside_original_gates_not_attributed')
        code = 0 if inside else 1
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=repr(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if before is not None:
            try:
                report['source_after'] = source_identity(function, reference, extension)
                report['actual_precision_after'] = precision(torch)
                report['original_source_unchanged'] = report['source_after'] == before
                if (not report['original_source_unchanged'] or report['actual_precision_after'] != policy or
                        sha(__file__) != report['validator_sha256']):
                    report['result'] = 'incomplete_or_blocked_original_CUDA_MSDA_validation'
                    code = 2
            except BaseException as exc:
                report.update(source_after_error=repr(exc), result='incomplete_or_blocked_original_CUDA_MSDA_validation')
                code = 2
        report.update(actual_exit=code, finished_UTC=datetime.now(timezone.utc).isoformat())
        persist(output, report)
    print(json.dumps(small_summary(report, output), separators=(',', ':')), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
