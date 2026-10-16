"""Apply eight genuine wheel packages in an isolated image build; not on a host."""
import hashlib
from importlib import metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlparse,unquote
from urllib.request import url2pathname

ROOT=Path(__file__).resolve().parent
OVERLAY=Path('/opt/worldengine-dependency-fix')
PROTECTED={'torch':'2.7.1+cpu','torch-npu':'2.7.1.post8','mx-driving':'26.1.0.post2',
 'mmcv-full':'1.7.2','mmdet':'2.24.0','mmdet3d':'1.0.0rc4','numpy':'1.23.5',
 'opencv-python-headless':'4.11.0.86'}
MODULES={'torch':'torch','torch-npu':'torch_npu','mx-driving':'mx_driving',
 'mmcv-full':'mmcv','mmdet':'mmdet','mmdet3d':'mmdet3d','numpy':'numpy',
 'opencv-python-headless':'cv2'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def protected():
    records={}
    for name,version in PROTECTED.items():
        d=metadata.distribution(name)
        if d.version!=version:raise RuntimeError(('Unexpected protected version',name,d.version))
        files={};native=set();sources=set();roots=[];projects=set()
        def capture(path):
            path=Path(path)
            if not path.is_file():return
            resolved=path.resolve();files[str(resolved)]=sha(resolved)
            if '.so' in path.name:native.add(str(resolved))
            if path.suffix in ('.py','.pyi','.cpp','.cc','.c','.cu','.h','.hpp'):sources.add(str(resolved))
        for rel in d.files or []:
            if '.so' in Path(str(rel)).name or str(rel).endswith(('/METADATA','/RECORD')):
                capture(d.locate_file(rel))
        metadata_path=Path(getattr(d,'_path',''))
        if not metadata_path.is_dir():raise RuntimeError(('Missing actual package metadata directory',name))
        for path in sorted(metadata_path.rglob('*')):
            if path.is_file():capture(path)
        module=MODULES[name]
        spec=PathFinder.find_spec(module,sys.path)
        if spec is not None and spec.submodule_search_locations:
            roots.extend(Path(x).resolve() for x in spec.submodule_search_locations)
        if metadata_path.name.endswith('.egg-info') and (metadata_path.parent/module/'__init__.py').is_file():
            roots.append(metadata_path.parent/module)
            projects.add(metadata_path.parent)
        direct=d.read_text('direct_url.json')
        if direct:
            info=json.loads(direct);url=urlparse(info.get('url',''))
            if info.get('dir_info',{}).get('editable') and url.scheme=='file':
                project=Path(url2pathname(url.path)).resolve()
                if (project/module/'__init__.py').is_file():roots.append(project/module)
                projects.add(project)
        for project in sorted(projects):
                for filename in ('setup.py','setup.cfg','pyproject.toml'):
                    capture(project/filename)
                requirements=project/'requirements'
                if requirements.is_dir():
                    for path in sorted(requirements.rglob('*.txt')):capture(path)
        roots=sorted(set(roots))
        if not roots:raise RuntimeError(('Actual protected package source root unresolved without importing',name))
        for root in roots:
            if not (root/'__init__.py').is_file():raise RuntimeError(('Protected package lacks real initializer',name))
            for path in sorted(root.rglob('*')):
                if path.is_file() and ('.so' in path.name or path.suffix in ('.py','.pyi','.cpp','.cc','.c','.cu','.h','.hpp')):
                    capture(path)
        # Protect editable source resolution itself, not just wheel metadata.
        for directory in [Path(x) for x in sys.path if x and Path(x).is_dir()]:
            for pattern in ('*.egg-link','*.pth','__editable__*.py'):
                for path in sorted(directory.glob(pattern)):capture(path)
        if not files or not sources:raise RuntimeError(('No actual protected metadata/source bytes',name))
        if name not in ('mmdet','mmdet3d') and not native:raise RuntimeError(('No actual protected native files',name))
        records[name]={'version':d.version,'metadata_path':str(metadata_path.resolve()),
          'module_roots':[str(x) for x in roots],'native_files':sorted(native),'source_files':sorted(sources),'files':files}
    return records

def call(args):
    done=subprocess.run([sys.executable,'-m','pip']+args,check=True)
    return done.returncode

def main():
    if platform.machine()!='aarch64' or sys.version_info[:2]!=(3,10):raise RuntimeError('Linux ARM64 CPython3.10 only')
    if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')!='0':raise RuntimeError('Backend autoload must be disabled')
    if OVERLAY.exists():raise RuntimeError('Fresh repair layer required')
    manifest=json.loads((ROOT/'WHEELS.lock.json').read_text())
    for item in manifest['artifacts']:
        path=ROOT/'wheelhouse'/item['filename']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise RuntimeError(('Wheel lock mismatch',item['name']))
    before=protected()
    # The actual SDK requires BOTH distributions. Keep the real headless prefix
    # byte-identical, and install the real full wheel in the new separate prefix.
    # No distribution metadata is fabricated or changed; cv2 active origin is gated.
    call(['install','--no-index','--find-links',str(ROOT/'wheelhouse'),'--no-deps',
      '--require-hashes','--only-binary=:all:','--target',str(OVERLAY),'-r',str(ROOT/'runtime-repair.lock')])
    # Py3 argparse comes from stdlib. The genuine backport distribution satisfies
    # affinity-sched's declaration without putting its module ahead of stdlib.
    call(['install','--no-index','--find-links',str(ROOT/'wheelhouse'),'--no-deps',
      '--require-hashes','--only-binary=:all:','-r',str(ROOT/'argparse.lock')])
    after=protected()
    if before!=after:raise RuntimeError('Protected SDK/framework/native files changed')
    receipt={'result':'minimal_layer_applied_not_full_dependency_acceptance',
      'protected_before':before,'protected_after':after,'protected_identical':True,
      'wheel_lock_sha256':sha(ROOT/'WHEELS.lock.json'),
      'remaining_expected_issues':['six ms-service-profiler dependencies','te CPython3.11 wheel tag on CPython3.10'],
      'accelerator_modules_imported':[x for x in sys.modules if x.startswith(('torch','mx_driving'))]}
    if receipt['accelerator_modules_imported']:raise RuntimeError('Unexpected accelerator import')
    (ROOT/'apply_receipt.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())

if __name__=='__main__':main()
