#!/usr/bin/env python3
"""Build the pinned inference-only CANN backend in a new isolated workspace."""
import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

COMMIT = '8c0445ebcca91514fce3a7050241f41ed132c340'
CONTRACT = 'mtgs-gsplat14-forward-v1'
COMPATIBILITY_FILES = {
    'kernels/calc_render_fwd_double_clip_gsids/op_kernel/calc_render_fwd_double_clip_gsids.cpp',
    'gauss_splat/csrc/pybind.cpp', 'gauss_splat/csrc/CalcRender.cpp',
}
PATCH_ROOT = Path(__file__).resolve().parents[2] / 'projects/SimEngine/worldengine/render/ascend/patches'


class Paused(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'expected regular file, not a symlink: {path}')
    return info


class Guard:
    def __init__(self, deadline, stop_file=None):
        self.deadline = deadline.timestamp()
        self.stop_file = stop_file
        self.signaled = None

    def check(self):
        if self.signaled or time.time() >= self.deadline:
            raise Paused('signal or absolute deadline reached')
        if self.stop_file is not None and os.path.lexists(self.stop_file):
            raise Paused('stop file present')


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def stop_child(child, unreaped=False):
    """Retained live Popen child owns this process group; never enumerate services."""
    def send(number):
        if not unreaped and child.poll() is not None:
            return
        try:
            if os.name == 'posix' and os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, number)
            else:
                child.send_signal(number)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    if unreaped:
        # Keep the direct leader allocated (possibly a zombie) until the whole
        # original process group receives SIGKILL. Its PID/PGID cannot be reused.
        until = time.monotonic() + 5
        while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None and time.monotonic() < until:
            time.sleep(0.1)
        send(signal.SIGKILL)
        child.wait(timeout=5)
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':send(signal.SIGKILL)
        else:child.kill()
        child.wait(timeout=5)


def run_owned(command, cwd, environment, log_path, guard, commands, phase):
    guard.check()
    row = {'phase': phase, 'command': [str(x) for x in command], 'cwd': str(cwd),
           'started_utc': utc(), 'log': str(log_path)}
    commands.append(row)
    with log_path.open('xb') as log:
        child = subprocess.Popen(row['command'], cwd=cwd, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        row['owned_pid'] = child.pid
        unreaped = os.name == 'posix' and hasattr(os, 'WNOWAIT') and hasattr(os, 'waitid')
        try:
            while (os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
                   if unreaped else child.poll() is None):
                guard.check();time.sleep(0.2)
            if unreaped:
                result = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                row['exit_code'] = result.si_status if result.si_code == os.CLD_EXITED else -result.si_status
            else:
                row['exit_code'] = child.wait()
            if row['exit_code'] != 0:
                raise RuntimeError(f'{phase} exited {row["exit_code"]}; see {log_path}')
        finally:
            if unreaped:
                stop_child(child, unreaped=True)
            elif child.poll() is None:
                stop_child(child)
            row['exit_code'] = child.returncode
            row['finished_utc'] = utc()


def git_output(source, arguments, guard):
    guard.check()
    remaining = max(0.01, min(20, guard.deadline - time.time()))
    try:
        value = subprocess.run(['git', '--no-optional-locks', '-C', str(source), *arguments], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=remaining, check=True)
    except subprocess.TimeoutExpired:
        guard.check()
        raise
    guard.check()
    return value.stdout


def source_inventory(source, guard, expected_commit=COMMIT):
    source = source.resolve(strict=True)
    if git_output(source, ['rev-parse', 'HEAD'], guard).decode().strip() != expected_commit:
        raise ValueError('source must be at the exact pinned CANN commit')
    if Path(git_output(source, ['rev-parse', '--show-toplevel'], guard).decode().strip()).resolve() != source:
        raise ValueError('--source must be the CANN repository root')
    if git_output(source, ['status', '--porcelain', '--untracked-files=no', '--ignore-submodules=none'], guard).strip():
        raise ValueError('source has tracked changes; preserve it and use a clean checkout')
    entries = git_output(source, ['ls-tree', '-rz', '--full-tree', 'HEAD'], guard)
    files = []
    for entry in entries.split(b'\0'):
        if not entry:continue
        header, raw_name = entry.split(b'\t', 1)
        mode, kind, blob = header.decode().split(' ')
        name = raw_name.decode('utf-8')
        path = PurePosixPath(name)
        if mode not in ('100644', '100755') or kind != 'blob' or path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('source contains unsupported tracked entries')
        actual = source.joinpath(*path.parts)
        regular(actual)
        if actual.resolve() != actual:
            raise ValueError('source tracked path contains a symlink component')
        data = actual.read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if git_blob != blob:
            raise ValueError(f'worktree bytes differ from the pinned Git blob: {name}')
        files.append({'path': name, 'size': len(data), 'sha256': sha(data), 'git_blob': blob, 'mode': mode})
        guard.check()
    if not files:raise ValueError('empty source tree')
    return files


def copy_tracked(source, destination, files, guard):
    destination.mkdir()
    for row in files:
        guard.check()
        path = source / row['path']
        regular(path)
        data = path.read_bytes()
        if len(data) != row['size'] or sha(data) != row['sha256']:
            raise RuntimeError('source changed while copying')
        target = destination / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        target.chmod(0o755 if row['mode'] == '100755' else 0o644)


def operational_patches(source, jobs):
    if not 1 <= jobs <= 16:raise ValueError('jobs must be between 1 and 16')
    rows = []
    def update(name, transform):
        path = source / name
        before = path.read_bytes()
        after = transform(before.decode('utf-8').replace('\r\n', '\n')).encode('utf-8')
        path.write_bytes(after)
        rows.append({'path': name, 'before_sha256': sha(before), 'after_sha256': sha(after)})
    def setup(text):
        marker = 'multiprocessing.cpu_count()'
        if text.count(marker) != 2:raise ValueError('unexpected original setup.py concurrency calls')
        return text.replace(marker, f'min({jobs}, multiprocessing.cpu_count())')
    update('setup.py', setup)
    def presets(text):
        value = json.loads(text)
        if len(value['configurePresets']) != 1:raise ValueError('unexpected CMake preset count')
        cache = value['configurePresets'][0]['cacheVariables']
        if cache['ASCEND_COMPUTE_UNIT']['value'] != 'ascend910b;ascend910_93' or cache['ENABLE_TEST']['value'] != 'True':
            raise ValueError('unexpected original architecture/test settings')
        cache['ASCEND_COMPUTE_UNIT']['value'] = 'ascend910_93'
        cache['ENABLE_TEST']['value'] = 'False'
        return json.dumps(value, indent=2) + '\n'
    update('CMakePresets.json', presets)
    def torch_npu_path(text):
        old = "import os; import site; print(site.getsitepackages()[0] + '/torch_npu')"
        new = "import os,importlib.util; print(os.path.dirname(importlib.util.find_spec('torch_npu').origin))"
        if text.count(old) != 1:raise ValueError('unexpected original Torch-NPU path query')
        return text.replace(old, new)
    update('gauss_splat/csrc/CMakeLists.txt', torch_npu_path)
    return rows


VERSION_PROBE = r'''
import importlib.metadata,importlib.util,json,os,pathlib,sys,torch
if sys.prefix==sys.base_prefix: raise RuntimeError('--python must belong to an existing isolated venv')
spec=importlib.util.find_spec('torch_npu')
if spec is None or spec.origin is None: raise RuntimeError('torch_npu is absent from selected venv search path')
root=os.path.dirname(spec.origin)
if not os.path.isdir(os.path.join(root,'include')): raise RuntimeError('actual Torch-NPU headers are absent')
versions={}
for name in ('pip','setuptools','wheel','torch','torch-npu','numpy'):
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
acl_spec=importlib.util.find_spec('acl')
acl_origin=str(pathlib.Path(acl_spec.origin).resolve()) if acl_spec is not None and acl_spec.origin else None
if sys.argv[3]=='1' and (acl_origin is None or not pathlib.Path(acl_origin).is_relative_to(pathlib.Path(sys.argv[2]).resolve())):
 raise RuntimeError('pyACL is missing or outside the declared CANN root; check validated --cann-python-path')
result={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
        'packages':versions,'torch_npu_path':root,'torch_cxx11_abi':torch.compiled_with_cxx11_abi(),
        'acl_spec_origin':acl_origin,
        'cann_environment':{k:os.environ.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_OPP_PATH','LD_LIBRARY_PATH')}}
with open(sys.argv[1],'x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
'''
IMPORT_PROBE = r'''
import hashlib,json,pathlib,sys,acl,torch_npu,gauss_splat
import gauss_splat.gsplat_c as native
prefix=pathlib.Path(sys.prefix).resolve();package=pathlib.Path(gauss_splat.__file__).resolve();extension=pathlib.Path(native.__file__).resolve()
cann_root=pathlib.Path(sys.argv[2]).resolve();acl_path=pathlib.Path(acl.__file__).resolve()
if not acl_path.is_relative_to(cann_root): raise RuntimeError('pyACL import is outside the declared CANN installation')
if not package.is_relative_to(prefix) or not extension.is_relative_to(prefix): raise RuntimeError('import did not resolve to the selected venv installation')
if native.mtgs_compatibility_contract()!='mtgs-gsplat14-forward-v1': raise RuntimeError('unexpected installed compatibility contract')
result={'package':str(package),'extension':str(extension),'extension_sha256':hashlib.sha256(extension.read_bytes()).hexdigest(),
        'acl_path':str(acl_path),'acl_sha256':hashlib.sha256(acl_path.read_bytes()).hexdigest(),
        'contract':native.mtgs_compatibility_contract(),'numerical_or_device_validation_passed':False}
with open(sys.argv[1],'x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
'''


def validated_cann_python_paths(cann_root, inherited, explicit=(), excluded=()):
    """Retain only existing absolute SDK-contained paths; no module is imported.

    The declared SDK root and its libascendcl marker establish path scope, not a
    vendor signature. Actual acl import origin is checked after installation.
    """
    if cann_root is None:
        if explicit:raise ValueError('--cann-python-path requires a declared CANN root')
        return [], {'result': 'not_configured', 'accepted_paths': []}
    root = Path(cann_root).resolve(strict=True)
    markers = [root / suffix for suffix in ('lib64/libascendcl.so', 'runtime/lib64/libascendcl.so',
               'aarch64-linux/lib64/libascendcl.so', 'x86_64-linux/lib64/libascendcl.so')]
    marker = next((path.resolve() for path in markers if path.is_file() and path.resolve().is_relative_to(root)), None)
    if marker is None:raise ValueError('declared CANN root has no supported libascendcl.so installation marker')
    accepted = []
    def candidate(value, required):
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            if required:raise ValueError('explicit CANN Python paths must be absolute without ..')
            return
        try:actual = path.resolve(strict=True)
        except FileNotFoundError:
            if required:raise ValueError('explicit CANN Python path is missing')
            return
        allowed = actual.is_dir() and actual != root and actual.is_relative_to(root)
        for other in excluded:
            resolved = Path(other).resolve()
            if actual == resolved or actual.is_relative_to(resolved) or resolved.is_relative_to(actual):allowed = False
        if not allowed:
            if required:raise ValueError('explicit CANN Python path escapes the SDK or overlaps source/output')
            return
        if str(actual) not in accepted:accepted.append(str(actual))
    for value in inherited.split(os.pathsep):
        if value:candidate(value, False)
    for value in explicit:candidate(value, True)
    return accepted, {'result': 'SDK_path_scope_validated', 'root': str(root), 'accepted_paths': accepted,
                      'libascendcl_path': str(marker), 'libascendcl_sha256': sha(marker.read_bytes()),
                      'vendor_signature_verified': False}


def child_environment(python, jobs, cann_python_paths=()):
    result = dict(os.environ)
    for name in list(result):
        if name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE') or name.startswith('PIP_'):
            result.pop(name, None)
    result.update(PATH=str(python.parent) + os.pathsep + result.get('PATH', ''), TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                  PYTHONNOUSERSITE='1', MAX_JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                  PIP_CONFIG_FILE=os.devnull, PIP_REQUIRE_VIRTUALENV='1', BUILD_WITHOUT_SHA='1')
    if cann_python_paths:result['PYTHONPATH'] = os.pathsep.join(cann_python_paths)
    return result


def build(args):
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:raise ValueError('deadline must have explicit UTC timezone')
    guard = Guard(deadline, args.stop_file)
    for number in (signal.SIGTERM, signal.SIGINT):signal.signal(number, lambda signum, _: setattr(guard, 'signaled', signum))
    guard.check()
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    python = args.python.absolute()  # Preserve the venv launcher symlink; do not resolve to base Python.
    if '..' in output.parts or output.parent.resolve() != output.parent or not output.parent.is_dir():raise ValueError('output parent must exist without symlink components or ..')
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('source and output must be disjoint')
    if not python.is_file() or python.is_relative_to(source) or python.is_relative_to(output):raise ValueError('choose an existing venv Python outside source/output')
    if args.install and args.dry_run:raise ValueError('--install cannot be combined with --dry-run')
    cann_root = getattr(args, 'cann_root', None) or os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    cann_paths, cann_scope = validated_cann_python_paths(cann_root, os.environ.get('PYTHONPATH', ''),
                               getattr(args, 'cann_python_path', ()), excluded=(source, output))
    if args.install and cann_scope['result'] == 'not_configured':raise ValueError('--install requires --cann-root or a sourced CANN installation root')
    files = source_inventory(source, guard)
    patcher = PATCH_ROOT / 'apply_compat_patch.py'
    replacement = PATCH_ROOT / 'calc_render_mtgs_compat.cpp'
    for path in (patcher, replacement):regular(path)
    patch_hashes = {path.name: sha(path.read_bytes()) for path in (patcher, replacement)}
    output.mkdir(mode=0o700)  # Exclusive new workspace, retained on any error.
    receipt = {'schema_version': 1, 'source_url': 'https://gitcode.com/cann/gauss-splat.git', 'canonical_commit': COMMIT,
               'source_path': str(source), 'output': str(output), 'python_requested': str(python), 'jobs': args.jobs,
               'deadline_utc': deadline.isoformat(), 'started_utc': utc(), 'result': 'building',
               'inference_only': True, 'device_validation_passed': False, 'commands': [], 'patch_tool_sha256': patch_hashes}
    receipt['cann_python_environment'] = cann_scope
    receipt['wheel_version_policy'] = 'BUILD_WITHOUT_SHA=1; exact canonical source commit is recorded in this receipt'
    try:
        write_new(output / 'source_manifest.json', files)
        receipt['source_manifest_sha256'] = sha((output / 'source_manifest.json').read_bytes())
        copied = output / 'source'
        control = output / 'control';control.mkdir()
        wheels = output / 'wheels';wheels.mkdir()
        copy_tracked(source, copied, files, guard)
        environment = child_environment(python, args.jobs, cann_paths)
        run_owned([sys.executable, str(patcher), '--source', str(copied), '--apply', '--receipt', str(control / 'compat_patch.json')],
                  control, environment, control / 'compat_patch.log', guard, receipt['commands'], 'apply_compatibility_patch')
        patch = json.loads((control / 'compat_patch.json').read_text())
        if (patch.get('canonical_commit'), patch.get('contract'), patch.get('applied'), patch.get('inference_only')) != (COMMIT, CONTRACT, True, True):
            raise ValueError('unexpected compatibility patch receipt')
        if set(patch.get('original_sha256', {})) != COMPATIBILITY_FILES or set(patch.get('patched_sha256', {})) != COMPATIBILITY_FILES:
            raise ValueError('compatibility patch did not cover the exact three required files')
        originals = {row['path']: row['sha256'] for row in files}
        for name, expected in patch['original_sha256'].items():
            if originals.get(name) != expected or sha((copied / name).read_bytes()) != patch['patched_sha256'][name]:raise ValueError('patch receipt does not match copied source')
        receipt['compatibility_patch'] = patch
        receipt['operational_patches'] = operational_patches(copied, args.jobs)
        receipt['patched_source_files'] = {row['path']: sha((copied / row['path']).read_bytes()) for row in files}
        allowed_changes = set(patch['patched_sha256']) | {row['path'] for row in receipt['operational_patches']}
        changed = {name for name, digest in receipt['patched_source_files'].items() if digest != originals[name]}
        if changed != allowed_changes:raise ValueError('unexpected patched source changes')
        wheel_command = [str(python), '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps', '.', '--wheel-dir', str(wheels)]
        receipt['planned_wheel_command'] = wheel_command
        if args.dry_run:
            receipt['result'] = 'dry_run_source_prepared_no_wheel_built'
        else:
            if sys.platform != 'linux':raise ValueError('actual CANN build is Linux-only')
            run_owned([str(python), '-c', VERSION_PROBE, str(control / 'versions.json'), cann_scope.get('root', ''), '1' if args.install else '0'], control, environment, control / 'versions.log', guard, receipt['commands'], 'selected_venv_versions')
            receipt['versions'] = json.loads((control / 'versions.json').read_text())
            for program in ('cmake', 'c++', 'bisheng'):
                try:
                    run_owned([program, '--version'], control, environment, control / (program.replace('+','p') + '_version.log'), guard, receipt['commands'], program + '_version')
                except FileNotFoundError:
                    receipt.setdefault('version_programs_unavailable', []).append(program)
            run_owned(wheel_command, copied, environment, control / 'wheel_build.log', guard, receipt['commands'], 'pip_wheel')
            artifacts = list(wheels.iterdir())
            if len(artifacts) != 1 or artifacts[0].suffix != '.whl' or not artifacts[0].name.startswith('gauss_splat-'):raise ValueError('expected exactly one gauss_splat wheel')
            wheel = artifacts[0];regular(wheel)
            receipt['wheel'] = {'path': str(wheel), 'size': wheel.stat().st_size, 'sha256': sha(wheel.read_bytes())}
            if args.install:
                if sha(wheel.read_bytes()) != receipt['wheel']['sha256']:raise ValueError('wheel changed before installation')
                run_owned([str(python), '-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(wheel)], control, environment, control / 'install.log', guard, receipt['commands'], 'install_exact_wheel')
                run_owned([str(python), '-c', IMPORT_PROBE, str(control / 'installed_import.json'), cann_scope['root']], control, environment, control / 'installed_import.log', guard, receipt['commands'], 'verify_installed_import')
                receipt['installed_import'] = json.loads((control / 'installed_import.json').read_text())
            receipt['result'] = 'pass_wheel_build_and_install_import_only' if args.install else 'pass_wheel_build_only'
        if source_inventory(source, guard) != files:raise RuntimeError('original source changed during the build')
        if {row['path']: sha((copied / row['path']).read_bytes()) for row in files} != receipt['patched_source_files']:
            raise RuntimeError('tracked build inputs changed after their patch receipt')
        if {p.name: sha(p.read_bytes()) for p in (patcher,replacement)} != patch_hashes:raise RuntimeError('compatibility patch inputs changed during the build')
        guard.check()
        receipt['exit_code'] = 0
    except BaseException as error:
        receipt.update(result='paused_incomplete' if isinstance(error, Paused) else 'blocked_or_rejected',
                       error=f'{type(error).__name__}: {error}', exit_code=2 if isinstance(error, Paused) else 1)
    receipt['finished_utc'] = utc()
    write_new(output / 'build_receipt.json', receipt)
    print(json.dumps({'result': receipt['result'], 'receipt': str(output / 'build_receipt.json'), 'device_validation_passed': False}), flush=True)
    return receipt['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Existing isolated venv Python; never base/global Python')
    parser.add_argument('--jobs', type=int, choices=range(1,17), default=8)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--cann-root', type=Path, help='SDK installation root; default ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME')
    parser.add_argument('--cann-python-path', type=Path, action='append', default=[], help='Optional explicit SDK Python directory, validated inside --cann-root')
    parser.add_argument('--install', action='store_true', help='Install exact wheel only into the selected venv, then verify import')
    parser.add_argument('--dry-run', action='store_true', help='Verify/copy/patch inputs and record commands without compiling or installing')
    return build(parser.parse_args())


if __name__ == '__main__':
    try:sys.exit(main())
    except Exception as error:
        print(json.dumps({'result':'startup_rejected','error':f'{type(error).__name__}: {error}'}),file=sys.stderr)
        sys.exit(1)
