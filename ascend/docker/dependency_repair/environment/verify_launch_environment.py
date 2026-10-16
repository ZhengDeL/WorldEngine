"""CPU-only check after the launcher's normal CANN set_env.sh sequence."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Fresh receipt required')
    report = {'result': 'running', 'failures': [], 'scope': 'CPU source-env/import check; no device or compiled-kernel acceptance'}
    try:
        if list(Path('/dev').glob('davinci[0-9]*')) or os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
            raise RuntimeError('CPU-only no-device container and backend autoload0 required')
        import worldengine_application_environment as startup
        report['startup'] = startup.STATUS
        if not startup.STATUS['active'] or sys.prefix != '/opt/conda/envs/bevformer':
            raise RuntimeError('Real model application interpreter is not active')
        if os.environ.get('HI_PYTHON') != '/opt/worldengine-environments/compiler_python311.sh':
            raise RuntimeError('CANN sourcing changed the real compiler interpreter selection')
        import torch
        import torch_npu
        import mmcv
        import mmcv._ext
        report['sys_path_after_framework_imports'] = list(sys.path)
        report['PYTHONPATH_after_framework_imports'] = os.environ.get('PYTHONPATH')
        report['HI_PYTHON'] = os.environ.get('HI_PYTHON')
        if any(startup.sdk_python_path(value) for value in sys.path):
            raise RuntimeError('SDK Python package path leaked after original framework imports')
        visible = []
        for name in ('te', 'ms-service-profiler', 'affinity-sched'):
            try:
                found = metadata.distribution(name);visible.append({'name': name, 'path': str(found._path)})
            except metadata.PackageNotFoundError:
                pass
        report['SDK_distributions_visible_in_app'] = visible
        if visible:
            raise RuntimeError('SDK tools unexpectedly visible in model application environment')
        report['NPU_initialized'] = torch_npu.npu.is_initialized()
        report['CUDA_initialized'] = torch.cuda.is_initialized()
        if report['NPU_initialized'] or report['CUDA_initialized']:
            raise RuntimeError('CPU verification initialized an accelerator')
    except BaseException as error:
        report['failures'].append(repr(error));report['traceback'] = traceback.format_exc()
    report['result'] = 'pass_actual_CPU_CANN_resourcing_environment' if not report['failures'] else 'failed_actual_CPU_CANN_resourcing_environment'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return bool(report['failures'])


if __name__ == '__main__':
    raise SystemExit(main())
