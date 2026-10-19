#!/usr/bin/env python3
"""Prepare one unchanged official scene and its matching original MTGS asset.

Both complete publisher files must pass their fixed official SHA256 before any
pickle loading or tar extraction. No download or accelerator use is performed.
"""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import pickle
import signal
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
HF_REV = 'd0e88d858533ca407a3387a26577c083aa7b3617'
MS_REV = '092ac5f3f3f828698c207b0ad75ca258c9f07ad2'
DEFAULT_ID = '2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4'
PREFIX = 'navtest_failures/assets'
EXPECTED_LOCK = {
    'schema_version': 1, 'repository': 'OpenDriveLab/WorldEngine',
    'huggingface_revision': HF_REV, 'modelscope_revision': MS_REV,
    'files': {
        'scenario': {'path': 'data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl',
                     'size': 911923264, 'sha256': '27203d52454f2538e6a70e95136b85ee8df35822f7707dc0091b4943ca7ad41d'},
        'assets': {'path': 'data/sim_engine/assets/navtest_failures/assets/part003.tar.gz',
                   'size': 27886330535, 'sha256': '102d16b9f608ac8f2314aef2ba3e1907cee9db7e2a3d094b1833b4ff60de0d05'},
    },
    'default_asset_id': DEFAULT_ID, 'archive_prefix': PREFIX,
    'evidence': {
        'huggingface_tree': f'https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/{HF_REV}/data/sim_engine/assets/navtest_failures/assets?recursive=false&limit=1000',
        'modelscope_tree': f'https://www.modelscope.cn/api/v1/datasets/OpenDriveLab/WorldEngine/repo/tree?Revision={MS_REV}&Root=data/sim_engine/assets/navtest_failures/assets&Recursive=false&PageNumber=1&PageSize=1000',
        'scenario_tree': f'https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/{HF_REV}/data/sim_engine/scenarios/original/navtest_failures?recursive=false&limit=1000',
        'default_member': f'{PREFIX}/{DEFAULT_ID}/background/{DEFAULT_ID}.ckpt',
        'default_member_declared_size': 336430727,
    },
}


class Paused(RuntimeError):
    pass


class Guard:
    def __init__(self, deadline=None, stop_file=None):
        self.deadline = deadline.timestamp() if deadline else None
        self.stop_file = stop_file
        self.signal_number = None
        self.next_stop_check = 0.0

    def check(self):
        if self.signal_number:
            raise Paused(f'signal {self.signal_number}')
        if self.deadline is not None and time.time() >= self.deadline:
            raise Paused('absolute deadline reached')
        if self.stop_file is not None and time.monotonic() >= self.next_stop_check:
            self.next_stop_check = time.monotonic() + 0.25
            if os.path.lexists(self.stop_file):
                raise Paused('stop file present')


def utc():
    return datetime.now(timezone.utc).isoformat()


def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'expected a regular file, not a symlink: {path}')
    return info


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def write_json_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def hash_file(path, guard):
    with VerifiedInput(path) as source:
        return source.digest(guard)


class VerifiedInput:
    """Keep one regular-file descriptor from verification through consumption."""
    def __init__(self, path):
        self.path = Path(path).absolute()
        before = regular(self.path)
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        self.stream = os.fdopen(descriptor, 'rb')
        self.snapshot = identity(os.fstat(descriptor))
        self.verified = False
        if self.snapshot != identity(before):
            self.stream.close()
            raise RuntimeError('input changed while opening')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()

    def unchanged(self):
        if identity(os.fstat(self.stream.fileno())) != self.snapshot or identity(regular(self.path)) != self.snapshot:
            raise RuntimeError(f'input changed: {self.path}')

    def digest(self, guard):
        self.unchanged()
        self.stream.seek(0)
        sha = hashlib.sha256()
        while True:
            guard.check()
            block = self.stream.read(4 * 1024 * 1024)
            if not block:
                break
            sha.update(block)
        self.unchanged()
        self.stream.seek(0)
        return {'size': self.snapshot[2], 'sha256': sha.hexdigest()}

    def verify(self, expected, guard):
        if self.snapshot[2] != expected['size']:
            raise ValueError(f'official size mismatch: {self.path}')
        actual = self.digest(guard)
        if actual['sha256'] != expected['sha256']:
            raise ValueError(f'official whole-file SHA256 mismatch: {self.path}')
        self.verified = True
        return actual


def load_official_lock(path):
    regular(path)
    content = path.read_bytes()
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f'duplicate asset lock key: {key}')
            result[key] = item
        return result
    value = json.loads(content, object_pairs_hook=unique_object)
    # Alternate formatting is allowed; alternate artifact hashes or schema are not.
    if json.dumps(value, sort_keys=True, separators=(',', ':')) != json.dumps(EXPECTED_LOCK, sort_keys=True, separators=(',', ':')):
        raise ValueError('asset lock differs from the exact supported official schema/revisions/artifacts')
    return value, hashlib.sha256(content).hexdigest()


def normalized_asset_id(scene):
    """Match MTGSRenderEngine.reset / MTGSAssetManager.reset without guessing.

    RenderManager passes scene['id']; upstream strips the last four characters
    whenever asset_id[-4:].startswith('-'), without checking they are digits.
    """
    if not isinstance(scene, Mapping) or not isinstance(scene.get('metadata'), Mapping):
        raise ValueError('scenario must contain a metadata mapping')
    internal_id = scene.get('id')
    if not isinstance(internal_id, str) or not internal_id:
        raise ValueError('scenario has no valid internal id')
    asset_id = scene['metadata'].get('digitaltwin_asset_id', internal_id)
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError('scenario has no valid digitaltwin asset id')
    if asset_id[-4:].startswith('-'):
        asset_id = asset_id[:-4]
    if not asset_id or '/' in asset_id or '\\' in asset_id or asset_id in ('.', '..') or ':' in asset_id:
        raise ValueError('asset id is not a safe single path component')
    return asset_id


def select_scene(source, requested_scene_id, default_asset_id, guard):
    if not source.verified:
        raise RuntimeError('pickle loading requires completed whole-file verification')
    source.unchanged()
    guard.check()
    source.stream.seek(0)
    scenes = pickle.load(source.stream)  # Only the exact, whole-hash-verified official file.
    if source.stream.read(1):
        raise ValueError('unexpected trailing bytes after the scene dictionary')
    source.unchanged()
    guard.check()
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError('official scenario input is not a nonempty dictionary')
    if requested_scene_id is not None:
        if requested_scene_id not in scenes:
            raise ValueError('requested scene dictionary key is absent')
        key = requested_scene_id
        matches = [key]
    else:
        matches = []
        for candidate, scene in scenes.items():
            guard.check()
            if not isinstance(candidate, str):
                raise ValueError('scenario dictionary keys must be strings')
            if normalized_asset_id(scene) == default_asset_id:
                matches.append(candidate)
        if not matches:
            raise ValueError('no original scenario maps to the known first asset in part003')
        key = matches[0]  # Preserve the publisher dictionary's original order.
    selected = scenes[key]
    asset_id = normalized_asset_id(selected)
    return key, selected, asset_id, len(scenes), matches


def tar_path(member):
    name = member.name.rstrip('/')
    if not name or len(name) > 8192 or '\\' in name or ':' in name or any(ord(c) < 32 for c in name):
        raise ValueError(f'unsafe tar member name: {member.name!r}')
    pieces = name.split('/')
    if any(piece in ('', '.', '..') for piece in pieces) or name.startswith('/'):
        raise ValueError(f'unsafe tar path: {member.name!r}')
    if not (member.isdir() or member.isfile()) or member.issparse() or member.sparse is not None:
        raise ValueError(f'links, sparse and special tar entries are rejected: {name}')
    if member.size < 0 or member.size > 8 * 1024**3 or (member.isdir() and member.size != 0):
        raise ValueError(f'unsupported tar member size: {name}')
    if 'linkpath' in member.pax_headers or any(k.startswith('GNU.sparse') for k in member.pax_headers):
        raise ValueError(f'unsupported tar link/sparse metadata: {name}')
    return PurePosixPath(name)


def extract_selected_asset(source, asset_id, stage, guard):
    if not source.verified:
        raise RuntimeError('archive extraction requires completed whole-file verification')
    source.unchanged()
    source.stream.seek(0)
    selected_prefix = PurePosixPath(PREFIX) / asset_id
    checkpoint_member = selected_prefix / 'background' / (asset_id + '.ckpt')
    paths = {}
    folded = set()
    folded_files = set()
    canonical_names = {}
    extracted = []
    selected_bytes = 0
    archive_bytes = 0
    asset_root = stage / 'assets' / asset_id
    class GuardedRead:
        def read(self, size=-1):
            guard.check()
            result = source.stream.read(size)
            guard.check()
            return result
    with tarfile.open(fileobj=GuardedRead(), mode='r|gz') as archive:
        for member in archive:
            guard.check()
            path = tar_path(member)
            name = str(path)
            prefix = PurePosixPath(PREFIX)
            if path != prefix and prefix not in path.parents and path not in prefix.parents:
                raise ValueError(f'unexpected archive top-level layout: {name}')
            if name in paths or name.casefold() in folded:
                raise ValueError(f'duplicate or case-alias tar member: {name}')
            for component in (path, *path.parents):
                component_name = str(component)
                if component_name == '.':
                    continue
                previous = canonical_names.setdefault(component_name.casefold(), component_name)
                if previous != component_name:
                    raise ValueError(f'case-alias tar path component: {name}')
            for parent in path.parents:
                if str(parent).casefold() in folded_files:
                    raise ValueError(f'tar file/directory conflict: {name}')
            if member.isfile() and any(previous.startswith(name + '/') for previous in paths):
                raise ValueError(f'tar directory/file conflict: {name}')
            paths[name] = 'dir' if member.isdir() else 'file'
            folded.add(name.casefold())
            if member.isfile():
                folded_files.add(name.casefold())
            archive_bytes += member.size
            if archive_bytes > source.snapshot[2] * 8:
                raise ValueError('archive declared expansion exceeds the supported limit')
            try:
                relative = path.relative_to(selected_prefix)
            except ValueError:
                continue
            destination = asset_root.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            selected_bytes += member.size
            if selected_bytes > 16 * 1024**3:
                raise ValueError('selected asset exceeds the 16 GiB staging limit')
            destination.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            remaining = member.size
            with archive.extractfile(member) as incoming, destination.open('xb') as outgoing:
                while remaining:
                    guard.check()
                    chunk = incoming.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('truncated selected archive member')
                    outgoing.write(chunk)
                    sha.update(chunk)
                    remaining -= len(chunk)
                if incoming.read(1):
                    raise ValueError('selected archive member exceeded its declared size')
                outgoing.flush()
                os.fsync(outgoing.fileno())
            extracted.append({'archive_member': name, 'output_path': str(destination.relative_to(stage)).replace('\\', '/'),
                              'size': member.size, 'sha256': sha.hexdigest()})
    source.unchanged()
    if str(checkpoint_member) not in {item['archive_member'] for item in extracted}:
        raise ValueError('selected original scenario has no matching background checkpoint in part003')
    checkpoint = asset_root / 'background' / (asset_id + '.ckpt')
    if regular(checkpoint).st_size == 0:
        raise ValueError('selected checkpoint is empty')
    return checkpoint, extracted, len(paths), archive_bytes


def publish_directory_noreplace(stage, destination):
    if os.path.lexists(destination):
        raise FileExistsError(f'output already exists: {destination}')
    if os.name == 'nt':
        os.rename(stage, destination)  # Windows rename does not replace existing destinations.
    elif sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, 'renameat2', None)
        if renameat2 is None:
            raise RuntimeError('atomic no-replace directory publication requires Linux renameat2')
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise FileExistsError(f'output appeared during publication: {destination}')
            raise OSError(number, os.strerror(number), str(destination))
    else:
        raise RuntimeError('atomic no-replace directory publication supports Linux and Windows')


CPU_INSPECTION = r'''
import json,os,sys
from collections.abc import Mapping
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
if torch.cuda.is_initialized(): raise RuntimeError('CUDA was initialized')
asset=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if not isinstance(asset,Mapping): raise ValueError('checkpoint must contain a submodel mapping')
supported={'VanillaGaussianSplattingModel','SkyboxGaussianSplattingModel','RigidSubModel','MirroredRigidSubModel'}
rows=[]
for name,entry in asset.items():
 if not isinstance(name,str) or not isinstance(entry,Mapping): raise ValueError('unsupported submodel entry')
 config=entry.get('config');state=entry.get('state_dict')
 if not isinstance(config,Mapping) or not isinstance(state,Mapping): raise ValueError('missing config/state_dict mappings')
 kind=config.get('type');tensors=[]
 def visit(value,path):
  if isinstance(value,torch.Tensor):
   if value.device.type!='cpu': raise ValueError('non-CPU tensor in inspected checkpoint')
   finite=bool(torch.isfinite(value).all()) if value.is_floating_point() or value.is_complex() else True
   tensors.append({'path':path,'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),'finite':finite})
  elif isinstance(value,Mapping):
   for key,item in value.items(): visit(item,path+'.'+str(key))
  elif isinstance(value,(list,tuple)):
   for index,item in enumerate(value): visit(item,path+'.'+str(index))
 visit(state,'state_dict')
 rows.append({'name':name,'config_type':kind,'supported_by_original_MODEL_MAPPING':kind in supported,'tensors':tensors})
if 'background' not in asset: raise ValueError('checkpoint contains no background submodel')
if not rows or any(not row['tensors'] for row in rows): raise ValueError('submodel contains no inspected state tensors')
if any(not tensor['finite'] for row in rows for tensor in row['tensors']): raise ValueError('nonfinite state tensor')
if torch.cuda.is_initialized(): raise RuntimeError('CUDA initialized during inspection')
npu=getattr(torch,'npu',None)
if npu is not None and getattr(npu,'is_initialized',lambda:False)(): raise RuntimeError('NPU initialized during inspection')
result={'result':'pass_CPU_checkpoint_inventory_only','torch_version':torch.__version__,'submodels':rows,
        'cuda_initialized':False,'npu_initialized':False,'renderer_execution_accepted':False}
with open(sys.argv[2],'x',encoding='utf-8') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
'''


def inspect_checkpoint_cpu(checkpoint, stage, guard):
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', CUDA_VISIBLE_DEVICES='',
                       ASCEND_RT_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output = stage / 'checkpoint_inventory.json'
    with (stage / 'checkpoint_inventory.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-c', CPU_INSPECTION, str(checkpoint), str(output)],
                                 env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        try:
            while child.poll() is None:
                guard.check()
                time.sleep(0.25)
            if child.wait() != 0:
                raise RuntimeError('CPU checkpoint inspection did not complete; inspect checkpoint_inventory.log')
        finally:
            if child.poll() is None:
                try:
                    if os.name == 'posix':
                        os.killpg(child.pid, signal.SIGTERM)
                    else:
                        child.terminate()
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == 'posix':
                            os.killpg(child.pid, signal.SIGKILL)
                        else:
                            child.kill()
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)
    regular(output)
    result = json.loads(output.read_text(encoding='utf-8'))
    if result.get('result') != 'pass_CPU_checkpoint_inventory_only' or result.get('renderer_execution_accepted') is not False:
        raise ValueError('unexpected CPU checkpoint inventory result')
    return {'result': result['result'], 'receipt': output.name, **hash_file(output, guard)}


def checked_output(path):
    output = path.absolute()
    if '..' in output.parts or os.path.lexists(output):
        raise ValueError('output must be a new path without ..')
    parent = output.parent
    if not parent.is_dir() or parent.resolve() != parent:
        raise ValueError('output parent must already exist and contain no symlink components')
    return output


def prepare(scenario_source, archive_source, lock, lock_sha, output, scene_id, inspect, guard):
    """Sources must already have passed whole-file verification; no mock CLI mode."""
    if not scenario_source.verified or not archive_source.verified:
        raise RuntimeError('both official input files must be fully verified before preparation')
    output = checked_output(output)
    stage = output.with_name('.' + output.name + '.stage-' + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    receipt = {'schema_version': 1, 'started_utc': utc(), 'result': 'preparing',
               'output': str(output), 'stage': str(stage), 'asset_lock_sha256': lock_sha,
               'publisher': lock['repository'], 'huggingface_revision': lock['huggingface_revision'],
               'modelscope_revision': lock['modelscope_revision'],
               'sources': {name: dict(lock['files'][name], local_path=str(source.path))
                           for name, source in [('scenario', scenario_source), ('assets', archive_source)]},
               'renderer_execution_accepted': False, 'complete_SimEngine_acceptance': False}
    try:
        guard.check()
        key, scene, asset_id, total_scenes, matches = select_scene(scenario_source, scene_id, lock['default_asset_id'], guard)
        receipt['selected'] = {'dictionary_key': key, 'scene_internal_id': scene['id'], 'normalized_asset_id': asset_id,
                               'official_dictionary_items': total_scenes, 'candidate_dictionary_keys': matches,
                               'normalization_contract': 'RenderManager scene.id; metadata.get(digitaltwin_asset_id, scene.id); strip last four iff asset_id[-4:].startswith(-)'}
        scene_output = stage / 'scene.pkl'
        with scene_output.open('xb') as stream:
            pickle.dump({key: scene}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        del scene
        receipt['derived_scene'] = {'path': scene_output.name, 'original_value_preserved': True, **hash_file(scene_output, guard)}
        checkpoint, extracted, members, expanded = extract_selected_asset(archive_source, asset_id, stage, guard)
        receipt['asset_files'] = extracted
        receipt['archive_members_checked'] = members
        receipt['archive_declared_expanded_bytes'] = expanded
        receipt['asset_root'] = 'assets'
        receipt['checkpoint'] = str(checkpoint.relative_to(stage)).replace('\\', '/')
        receipt['checkpoint_inspection'] = inspect_checkpoint_cpu(checkpoint, stage, guard) if inspect else {'result': 'not_requested', 'submodel_contents_accepted': False}
        scenario_source.unchanged()
        archive_source.unchanged()
        guard.check()
        receipt.update(result='pass_official_single_scene_asset_preparation', finished_utc=utc())
        write_json_new(stage / 'prepare_receipt.json', receipt)
        publish_directory_noreplace(stage, output)
        return receipt
    except BaseException as error:
        receipt.update(result='paused_incomplete' if isinstance(error, Paused) else 'blocked_or_rejected',
                       error=f'{type(error).__name__}: {error}', finished_utc=utc())
        if stage.is_dir() and not (stage / 'prepare_error.json').exists():
            write_json_new(stage / 'prepare_error.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-pkl', type=Path, required=True)
    parser.add_argument('--asset-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; parent must already exist')
    parser.add_argument('--scene-id', help='Exact original pickle dictionary key; default selects the first mapped known part003 asset')
    parser.add_argument('--asset-lock', type=Path, default=HERE / 'official_scene_assets.lock.json')
    parser.add_argument('--inspect-checkpoint', action='store_true', help='Optional isolated CPU torch.load inventory; needs torch/project dependencies')
    parser.add_argument('--deadline-utc', help='Optional absolute UTC cutoff, e.g. 2026-10-20T10:50:00+00:00')
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc) if args.deadline_utc else None
    if deadline and (deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0):
        parser.error('--deadline-utc must include an explicit UTC timezone')
    output = checked_output(args.output)
    lock, lock_sha = load_official_lock(args.asset_lock)
    guard = Guard(deadline, args.stop_file)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, _frame: setattr(guard, 'signal_number', signum))
    guard.check()
    with VerifiedInput(args.scenario_pkl) as scenario, VerifiedInput(args.asset_archive) as archive:
        for role, source in [('scenario', scenario), ('assets', archive)]:
            print(json.dumps({'phase': 'whole_file_verification', 'role': role, 'utc': utc()}), flush=True)
            source.verify(lock['files'][role], guard)
        receipt = prepare(scenario, archive, lock, lock_sha, output, args.scene_id, args.inspect_checkpoint, guard)
    print(json.dumps({'result': receipt['result'], 'output': str(output), 'selected': receipt['selected'],
                      'renderer_execution_accepted': False}), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Paused as error:
        print(json.dumps({'result': 'paused_incomplete', 'reason': str(error)}), file=sys.stderr)
        sys.exit(2)
    except Exception as error:
        print(json.dumps({'result': 'blocked_or_rejected', 'reason': f'{type(error).__name__}: {error}'}), file=sys.stderr)
        sys.exit(1)
