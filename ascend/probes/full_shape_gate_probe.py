"""Extend the pinned public CPU32 gates to the original Q40000/H8/C32 geometry.

Private server probe: no source edits, no changed tolerances and no CPU64 oracle.
The two valid-layout cases each test MMCV and mx_driving. The two original raw
map-layout cases test MMCV and require exact-zero unused raw-gradient tails.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


EXPECTED_SOURCE_SHA256 = '858f9d01181d3d62c35d0635881b22c62daf1a5f5af8123f622055de05da422d'


def source_fingerprint(source):
    files = sorted(list((source / 'projects/AlgEngine').rglob('*.py')) +
                   list((source / 'ascend').glob('*.py')))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(source).as_posix().encode())
        digest.update(file.read_bytes().replace(b'\r\n', b'\n'))
    return dict(sha256=digest.hexdigest(), python_files=len(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/output/full_shape_gate.json'))
    args = parser.parse_args()
    from ascend import validate_msda as valid
    from ascend import validate_msda_layout as legacy
    report = dict(result='running', expected_source_sha256=EXPECTED_SOURCE_SHA256,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  forward_atol=valid.FORWARD_ATOL, forward_rtol=valid.FORWARD_RTOL,
                  gradient_atol=valid.GRAD_ATOL, gradient_rtol=valid.GRAD_RTOL,
                  expected_cases=4, expected_entry_comparisons=6, rows=[],
                  scope='Original BEV 200x200/Q40000/H8/C32/P4/B1 operator geometry, seeds 29 and 7. '
                        'Unmodified CPU32 reference and original tolerances. Both normal-layout entry points; '
                        'MMCV-only original raw-map L4/spatial L1 global-prefix compatibility. '
                        'Full outputs and all three full gradients, plus exact-zero unused legacy tails. '
                        'No measured GPU parity, model convergence, sustained training or bitwise replay claim.')
    legacy.persist(args.output, report)
    try:
        import torch
        import torch_npu
        source = Path(valid.__file__).resolve().parents[1]
        if Path(legacy.__file__).resolve().parents[1] != source:
            raise RuntimeError('Public validators resolve to different source trees')
        report['source'] = str(source)
        report['source_fingerprint'] = source_fingerprint(source)
        if report['source_fingerprint']['sha256'] != EXPECTED_SOURCE_SHA256:
            raise RuntimeError('Source fingerprint differs from the fixed v6 acceptance source')
        report['public_validator_sha256'] = {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (valid, legacy)}
        assert (valid.FORWARD_ATOL, valid.FORWARD_RTOL, valid.GRAD_ATOL, valid.GRAD_RTOL) == (2e-5, 2e-5, 1e-3, 1e-4)
        assert (legacy.FORWARD_ATOL, legacy.FORWARD_RTOL,
                legacy.GRADIENT_ATOL, legacy.GRADIENT_RTOL) == (2e-5, 2e-5, 1e-3, 1e-4)
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
        torch.manual_seed(29)
        torch.npu.manual_seed_all(29)
        report['environment'] = dict(
            torch_version=str(torch.__version__), torch_npu_version=torch_npu.__version__,
            device=str(torch.device('npu', 0)), device_name=torch.npu.get_device_name(0),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_allow_hf32=torch.npu.matmul.allow_hf32,
            conv_allow_hf32=torch.npu.conv.allow_hf32,
            HCCL_DETERMINISTIC=os.environ.get('HCCL_DETERMINISTIC'),
            CLOSE_MATMUL_K_SHIFT=os.environ.get('CLOSE_MATMUL_K_SHIFT'),
            cpu_threads=torch.get_num_threads())
        report['cpu_reference_source_sha256'] = hashlib.sha256(
            inspect.getsource(multi_scale_deformable_attn_pytorch).encode()).hexdigest()
        started = time.monotonic()
        for seed in (29, 7):
            for kind in ('normal', 'legacy'):
                name = kind + '_original_encoder_q40000_h8_c32_seed' + str(seed)
                try:
                    if kind == 'normal':
                        spec = valid.case(name, [(200, 200)], 40000,
                                          seed=seed, b=1, heads=8, channels=32, points=4)
                        row = valid.run_case(spec, multi_scale_deformable_attn_pytorch,
                                             MultiScaleDeformableAttnFunction,
                                             mx_driving.multi_scale_deformable_attn)
                    else:
                        spec = dict(name=name, spatial_shape=[200, 200], batch=1, queries=40000,
                                    heads=8, channels=32, points=4, seed=seed, pattern='random')
                        row = legacy.run_case(torch, spec, multi_scale_deformable_attn_pytorch,
                                              MultiScaleDeformableAttnFunction)
                    row['layout_kind'] = kind
                except Exception as error:
                    row = dict(name=name, seed=seed, layout_kind=kind, passed=False,
                               exception=repr(error), traceback=traceback.format_exc()[-7000:])
                report['rows'].append(row)
                print(json.dumps(dict(case=name, passed=row['passed'], seconds=row.get('seconds'),
                                      exception=row.get('exception'))), flush=True)
                del row
                gc.collect()
                try:
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
                except Exception as error:
                    report['rows'][-1].update(passed=False, cleanup_exception=repr(error))
                legacy.persist(args.output, report)
        report['seconds'] = time.monotonic() - started
        report['passed_cases'] = sum(row['passed'] for row in report['rows'])
        report['entry_comparisons'] = sum(len(row.get('entries', []))
                                           if row['layout_kind'] == 'normal' else 1
                                           for row in report['rows'])
        report['source_fingerprint_after'] = source_fingerprint(source)
        report['source_unchanged'] = report['source_fingerprint_after'] == report['source_fingerprint']
        report['result'] = ('pass_original_encoder_full_shape_cpu32_gates'
                            if len(report['rows']) == 4 and report['entry_comparisons'] == 6 and
                            report['passed_cases'] == 4 and report['source_unchanged']
                            else 'fail_original_encoder_full_shape_cpu32_gates')
    except Exception as error:
        report.update(result='fail_setup', exception=repr(error),
                      traceback=traceback.format_exc()[-7000:])
    legacy.persist(args.output, report)
    print(json.dumps(dict(result=report['result'], passed_cases=report.get('passed_cases', 0),
                          entry_comparisons=report.get('entry_comparisons', 0),
                          seconds=report.get('seconds'), output=str(args.output))), flush=True)
    return 0 if report['result'].startswith('pass_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
