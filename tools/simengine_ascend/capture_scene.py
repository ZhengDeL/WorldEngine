#!/usr/bin/env python3
"""Capture one original official MTGS initial frame without rendering it.

The original BaseEnv initializes the logged scene, policies, assets and cameras.
Only its renderer instance's rasterization callable is replaced, once, by an
explicit full-input CPU export. No fake image is returned and no simulation
completion is claimed. The ordinary renderer's shape gate is not weakened.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIMULATION = ROOT / 'projects/SimEngine'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_scene as preparation
from simengine_ascend.file_protocol import check_stop
from simengine_ascend.launch_closed_loop import publish
from simengine_ascend.run_simulation import (build_overrides, existing, fresh_output,
    run_owned, sha256, source_identity)

ARRAYS = ('means', 'quats', 'scales', 'opacities', 'colors', 'viewmats', 'Ks')


def read_json(path, maximum=16*1024**2):
    existing(path)
    if Path(path).stat().st_size > maximum:
        raise ValueError('JSON input exceeds its explicit size limit')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def relative_file(root, name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('Expected a safe relative POSIX file path')
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe prepared file path')
    path = root.joinpath(*value.parts)
    if path.resolve(strict=True) != path.absolute():
        raise ValueError('Prepared file path contains a symlink')
    return existing(path)


def checked_digest(root, item, path_key, guard):
    if (type(item.get('size')) is not int or item['size'] < 0
            or re.fullmatch('[0-9a-f]{64}', str(item.get('sha256', ''))) is None):
        raise ValueError('Prepared file lacks an exact size/SHA256')
    path = relative_file(root, item[path_key])
    actual = preparation.hash_file(path, SimpleNamespace(check=guard))
    if actual != dict(size=item['size'], sha256=item['sha256']):
        raise ValueError('Prepared file content differs: ' + str(path))
    return path


def verify_prepared(prepared, expected_receipt_sha256, guard):
    """Rehash the derived scene and every selected asset, including full CKPT.

    Publisher archive hashes are checked against the fixed preparation lock;
    this does not reread the 27.89GB publisher archive or 0.91GB source pickle.
    Their prior whole-file verification remains identified by the receipt.
    """
    guard()
    prepared = existing(prepared, directory=True)
    receipt_path = relative_file(prepared, 'prepare_receipt.json')
    if sha256(receipt_path, guard) != expected_receipt_sha256:
        raise ValueError('Official preparation receipt SHA256 differs')
    receipt = read_json(receipt_path)
    lock, lock_sha = preparation.load_official_lock(preparation.HERE / 'official_scene_assets.lock.json')
    if (receipt.get('schema_version') != 1 or receipt.get('result') != 'pass_official_single_scene_asset_preparation'
            or receipt.get('publisher') != lock['repository']
            or receipt.get('huggingface_revision') != lock['huggingface_revision']
            or receipt.get('modelscope_revision') != lock['modelscope_revision']
            or receipt.get('asset_lock_sha256') != lock_sha
            or receipt.get('renderer_execution_accepted') is not False
            or receipt.get('complete_SimEngine_acceptance') is not False):
        raise ValueError('Receipt is not the exact supported official scene preparation')
    for role, expected in lock['files'].items():
        source = receipt.get('sources', {}).get(role, {})
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError('Receipt publisher source differs: ' + role)
    selected = receipt['selected']
    if any(not isinstance(selected.get(k), str) or not selected[k]
           for k in ('dictionary_key', 'scene_internal_id', 'normalized_asset_id')):
        raise ValueError('Prepared scene/asset identifiers are missing')
    asset_id = selected['normalized_asset_id']
    if '/' in asset_id or '\\' in asset_id or ':' in asset_id or asset_id in ('.', '..'):
        raise ValueError('Unsafe selected asset ID')
    scene = receipt['derived_scene']
    if scene.get('path') != 'scene.pkl' or scene.get('original_value_preserved') is not True or receipt.get('asset_root') != 'assets':
        raise ValueError('Receipt does not preserve the original scene value/asset layout')
    scene_path = checked_digest(prepared, scene, 'path', guard)
    rows = receipt.get('asset_files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Receipt must identify every extracted asset file')
    declared, checkpoint = set(), None
    expected_checkpoint = 'assets/%s/background/%s.ckpt' % (asset_id, asset_id)
    if receipt.get('checkpoint') != expected_checkpoint:
        raise ValueError('Receipt checkpoint does not match the original MTGS lookup')
    for row in rows:
        guard()
        path = checked_digest(prepared, row, 'output_path', guard)
        name = row['output_path']
        prefix = 'assets/' + asset_id + '/'
        if not name.startswith(prefix) or name in declared:
            raise ValueError('Duplicate or foreign prepared asset path')
        expected_member = preparation.PREFIX + '/' + asset_id + '/' + name[len(prefix):]
        if row.get('archive_member') != expected_member:
            raise ValueError('Asset member does not identify the official extraction location')
        declared.add(name)
        if name == expected_checkpoint:
            checkpoint = dict(name=name, sha256=row['sha256'], bytes=row['size'], path=str(path))
    actual = set()
    for directory, directories, files in os.walk(prepared / 'assets', followlinks=False):
        guard()
        for name in directories + files:
            if (Path(directory) / name).is_symlink():
                raise ValueError('Symlink in prepared assets')
        for name in files:
            actual.add((Path(directory) / name).relative_to(prepared).as_posix())
    if declared != actual or checkpoint is None:
        raise ValueError('Extracted assets are incomplete or contain unrecorded files')
    if sha256(receipt_path, guard) != expected_receipt_sha256:
        raise ValueError('Preparation receipt changed during verification')
    return dict(prepared=str(prepared), prepare_receipt_sha256=expected_receipt_sha256,
        asset_lock_sha256=lock_sha, selected=selected, scene_path=str(scene_path),
        scene_sha256=scene['sha256'], asset_root=str(prepared / 'assets'), checkpoint=checkpoint,
        all_extracted_asset_files_rehashed=len(rows), publisher_source_records=lock['files'],
        publisher_containers_rehashed_by_capture_tool=False)


def capture_source_identity(guard):
    base = source_identity(guard)
    files = [Path(__file__), preparation.HERE / 'prepare_scene.py', preparation.HERE / 'official_scene_assets.lock.json']
    tools = {path.relative_to(ROOT).as_posix(): sha256(path, guard) for path in files}
    value = json.dumps(dict(simulation=base, capture_tools=tools), sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(value).hexdigest(), files=base['files']+len(files),
                simulation=base, capture_tools=tools)


class CaptureComplete(BaseException):
    """A single successful export, never an image/simulation completion."""


class CaptureOnce:
    def __init__(self, renderer, scene, proof, request, writer, guard):
        self.renderer, self.scene, self.proof = renderer, scene, proof
        self.request, self.writer, self.guard = request, writer, guard
        self.calls, self.path = 0, None

    def __call__(self, **kwargs):
        self.guard()
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError('The capture hook must execute exactly once')
        expected = set(ARRAYS) | {'width', 'height', 'tile_size', 'packed', 'near_plane', 'far_plane',
                                  'render_mode', 'sparse_grad', 'absgrad', 'rasterize_mode', 'radius_clip'}
        if set(kwargs) != expected and set(kwargs) != expected | {'eps2d'}:
            raise ValueError('Original MTGS rasterization call contract changed')
        if kwargs['packed'] or kwargs['sparse_grad'] or kwargs['render_mode'] != 'RGB+ED':
            raise ValueError('Capture requires the original unpacked RGB+ED forward')
        names = [name for name, _ in sorted(self.renderer.sensor_mapping.items(), key=lambda pair: pair[1])]
        cameras = self.scene['cameras']
        if names != list(cameras) or len(names) != len(kwargs['viewmats']):
            raise ValueError('The capture omits/reorders original scene cameras')
        if any(camera['width'] != kwargs['width'] or camera['height'] != kwargs['height'] for camera in cameras.values()):
            raise ValueError('The capture changed original camera resolution')
        for name, source in (('means', 'means'), ('quats', 'quats'), ('scales', 'scales'),
                             ('opacities', 'opacities'), ('colors', 'rgbs')):
            if kwargs[name] is not self.renderer.collected_gaussians[source]:
                raise ValueError('The capture replaced an original full Gaussian array')
        for name in ARRAYS:
            value = kwargs[name]
            if str(value.device) != self.request['device'] or str(value.dtype) != 'torch.float32':
                raise ValueError('Every captured array must remain FP32 on the explicitly selected device')
        n, c = len(kwargs['means']), len(names)
        if n < 1:
            raise ValueError('The official scene produced no Gaussians')
        needed = 4*(n*(11+3*c)+25*c)+1024**2
        free = shutil.disk_usage(self.request['output']).free
        if free-needed < self.request['reserve_free_bytes']:
            raise OSError('Insufficient disk space for the complete capture plus reserve')
        timestamp = float(self.renderer.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError('Original frame timestamp is nonfinite')
        render = {name: kwargs[name] for name in ('width', 'height', 'tile_size', 'near_plane', 'far_plane',
                                                  'render_mode', 'rasterize_mode', 'radius_clip')}
        render['eps2d'] = kwargs.get('eps2d', .3)  # Original rasterization default.
        provenance = dict(scene_id=self.proof['selected']['dictionary_key'], frame_id=format(timestamp, '.6f'),
            camera_names=names, checkpoints=[{key: self.proof['checkpoint'][key] for key in ('name', 'sha256')}],
            source_commit=self.request['source_commit'], source_commit_role='declared base revision; runtime source separately hashed',
            source_fingerprint_sha256=self.request['source']['sha256'], input_origin='official_mtgs_scene',
            official_asset_manifest_sha256=self.proof['prepare_receipt_sha256'],
            official_asset_lock_sha256=self.proof['asset_lock_sha256'], derived_scene_sha256=self.proof['scene_sha256'],
            original_scene_internal_id=self.proof['selected']['scene_internal_id'],
            original_asset_id=self.proof['selected']['normalized_asset_id'], timestamp_us=timestamp,
            reaction=self.request['reaction'], frame_scope='original BaseEnv.reset initial frame',
            complete_gaussian_set=True, original_camera_resolution=True,
            publisher_containers_rehashed_by_capture_tool=False)
        self.path = self.writer(Path(self.request['output']) / 'capture',
                                {name: kwargs[name] for name in ARRAYS}, render, provenance)
        self.guard()
        raise CaptureComplete()


def initialization_overrides(request, proof):
    args = SimpleNamespace(scene_pkl=Path(proof['scene_path']), asset_root=Path(proof['asset_root']),
        maps_root=Path(request['maps_root']), output=Path(request['output']) / 'initialization',
        device=request['device'], validated_manifest=request.get('small_manifest'), cpu_threads=request['cpu_threads'],
        pdm_metrics=False, dense_rewards=False, reaction=request['reaction'], mode='replay')
    values = build_overrides(args)
    if request['device'].startswith('cuda:'):
        values = [value for value in values if not value.startswith(('+renderer.rasterization_backend=', '+renderer.npu_compat_manifest='))]
    return values


def capture_child(request_path):
    request = read_json(request_path)
    def guard():
        check_stop(Path(request['stop_file']), request['deadline_utc'])
    guard()
    proof = verify_prepared(request['prepared'], request['prepare_receipt_sha256'], guard)
    if proof != request['prepared_proof'] or capture_source_identity(guard) != request['source']:
        raise ValueError('Prepared input or source changed before child initialization')
    import pickle
    with preparation.VerifiedInput(proof['scene_path']) as source:
        source.verify(dict(size=Path(proof['scene_path']).stat().st_size, sha256=proof['scene_sha256']), SimpleNamespace(check=guard))
        scenes = pickle.load(source.stream)
        if source.stream.read(1):
            raise ValueError('Trailing bytes in prepared scene pickle')
        source.unchanged()
    key = proof['selected']['dictionary_key']
    if (not isinstance(scenes, dict) or list(scenes) != [key]
            or scenes[key].get('id') != proof['selected']['scene_internal_id']
            or preparation.normalized_asset_id(scenes[key]) != proof['selected']['normalized_asset_id']):
        raise ValueError('Prepared pickle does not identify the exact selected official scene')
    sys.path.insert(0, str(SIMULATION))
    import torch
    if request['device'].startswith('npu:'):
        import torch_npu  # noqa: F401
    device = torch.device(request['device'])
    if device.type == 'npu':
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32 = False
    else:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(request['cpu_threads'])
    import worldengine
    if not Path(worldengine.__file__).resolve().is_relative_to(SIMULATION / 'worldengine'):
        raise ValueError('Foreign worldengine package shadows the pinned simulator')
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from worldengine.envs.base_env import BaseEnv
    from worldengine.engine.engine_utils import close_engine
    from worldengine.render.mtgs.mtgs import MTGSRenderEngine
    from worldengine.render.ascend.capture import write_capture, load_capture
    with initialize_config_dir(config_dir=str(SIMULATION / 'worldengine/configs'), version_base='1.2'):
        config = compose(config_name='default_runner', overrides=initialization_overrides(request, proof))
    Path(config.output_dir).mkdir(parents=True, exist_ok=False)
    with (Path(request['output']) / 'resolved_initialization.yaml').open('x', encoding='utf-8') as stream:
        stream.write(OmegaConf.to_yaml(config, resolve=True))
    env = BaseEnv(config, 'official_scene_capture', scenes)
    hook = None
    try:
        guard()
        env.lazy_init()
        renderer = env.engine.managers['render_manager'].renderer
        if type(renderer) is not MTGSRenderEngine:
            raise ValueError('Capture requires the original MTGSRenderEngine')
        original = renderer._rasterization  # Constructor already validated the small manifest.
        scene = env.engine.managers['scenario_manager'].current_scene
        hook = CaptureOnce(renderer, scene, proof, request, write_capture, guard)
        renderer._rasterization = hook
        try:
            env.reset(seed=0)
        except CaptureComplete:
            if hook.calls != 1 or hook.path is None:
                raise RuntimeError('Capture completed without its one exact export')
        else:
            raise RuntimeError('Original reset returned without an intercepted rasterization call')
        finally:
            renderer._rasterization = original
        arrays, manifest = load_capture(hook.path)
        del arrays
        guard()
        if verify_prepared(request['prepared'], request['prepare_receipt_sha256'], guard) != proof:
            raise ValueError('Prepared asset/scene changed during capture')
        if capture_source_identity(guard) != request['source']:
            raise ValueError('Simulator/capture source changed during capture')
        if request.get('small_manifest') and sha256(request['small_manifest'], guard) != request['small_manifest_sha256']:
            raise ValueError('Small renderer proof changed during capture')
        publish(Path(request['output']) / 'capture_child_receipt.json', dict(result='complete_official_initial_frame_capture_only',
            capture_manifest=str(hook.path), capture_manifest_sha256=sha256(hook.path, guard), shape=manifest['shape'],
            render=manifest['render'], prepared_proof=proof, source=request['source'],
            rasterization_executed=False, simulation_accepted=False, gpu_parity_accepted=False))
    finally:
        # Original cleanup is allowed to raise; it must not turn an error into success.
        close_engine()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--prepare-receipt-sha256', required=True)
    parser.add_argument('--maps-root', type=Path, required=True)
    parser.add_argument('--small-manifest', type=Path, help='Required passing small CANN proof for npu:N')
    parser.add_argument('--device', required=True, help='Explicit device npu:N or cuda:N; CPU is never a renderer')
    parser.add_argument('--source-commit', required=True, help='Declared 40-hex base revision; actual source is separately hashed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--reaction', choices=('NR', 'R'), default='NR')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--reserve-free-bytes', type=int, default=80*1024**3)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'(npu|cuda):\d+', args.device):
        parser.error('--device must explicitly name npu:N or cuda:N')
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit) or not re.fullmatch('[0-9a-f]{64}', args.prepare_receipt_sha256):
        parser.error('Exact lowercase source commit and preparation receipt SHA256 are required')
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
        parser.error('--deadline-utc must have an explicit UTC timezone')
    if args.cpu_threads < 1 or args.reserve_free_bytes < 0:
        parser.error('CPU threads must be positive and the disk reserve nonnegative')
    if args.device.startswith('npu:') and args.small_manifest is None:
        parser.error('An NPU capture requires its passing --small-manifest')
    if args.device.startswith('cuda:') and args.small_manifest is not None:
        parser.error('A CANN manifest cannot authorize a CUDA renderer')
    args.output = fresh_output(args.output)
    args.maps_root = existing(args.maps_root, directory=True)
    args.prepared_dir = existing(args.prepared_dir, directory=True)
    if args.small_manifest is not None:
        args.small_manifest = existing(args.small_manifest)
    args.deadline_utc = deadline.isoformat()
    args.stop_file = (args.stop_file or args.output / 'STOP').absolute()
    interrupted = []
    def guard():
        if interrupted:
            raise InterruptedError('Capture supervisor received a signal')
        check_stop(args.stop_file, args.deadline_utc)
    guard()
    proof = verify_prepared(args.prepared_dir, args.prepare_receipt_sha256, guard)
    request = dict(prepared=str(args.prepared_dir), prepare_receipt_sha256=args.prepare_receipt_sha256,
        prepared_proof=proof, maps_root=str(args.maps_root), device=args.device, source_commit=args.source_commit,
        output=str(args.output), deadline_utc=args.deadline_utc, stop_file=str(args.stop_file),
        reaction=args.reaction, cpu_threads=args.cpu_threads, reserve_free_bytes=args.reserve_free_bytes,
        source=capture_source_identity(guard), small_manifest=str(args.small_manifest) if args.small_manifest else None,
        small_manifest_sha256=sha256(args.small_manifest, guard) if args.small_manifest else None)
    if args.dry_run:
        print(json.dumps(dict(result='capture_prepared_not_executed', request=request, simulation_accepted=False), indent=2))
        return 0
    args.output.mkdir(mode=0o700)
    request_path = args.output / 'capture_request.json'
    publish(request_path, request)
    environment = dict(os.environ, WORLDENGINE_ROOT=str(ROOT), HYDRA_FULL_ERROR='1',
        OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(SIMULATION)] + ([environment['PYTHONPATH']] if environment.get('PYTHONPATH') else []))
    if args.device.startswith('npu:'):
        environment['CUDA_VISIBLE_DEVICES'] = ''
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in handlers:
        signal.signal(number, lambda *_: interrupted.append(True))
    report = dict(result='capture_running', simulation_accepted=False, rasterization_executed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), request=request)
    status = 1
    try:
        run_owned([sys.executable, str(Path(__file__).resolve()), '_capture', str(request_path)],
                  ROOT, environment, args.output / 'capture.log', guard)
        guard()
        child = read_json(args.output / 'capture_child_receipt.json')
        capture = Path(child['capture_manifest'])
        if (child.get('result') != 'complete_official_initial_frame_capture_only'
                or child.get('simulation_accepted') is not False or child.get('rasterization_executed') is not False
                or child.get('source') != request['source'] or child.get('prepared_proof') != proof
                or capture != args.output / 'capture/capture.json'
                or sha256(capture, guard) != child['capture_manifest_sha256']):
            raise ValueError('Child capture receipt is incomplete or inconsistent')
        if verify_prepared(args.prepared_dir, args.prepare_receipt_sha256, guard) != proof or capture_source_identity(guard) != request['source']:
            raise ValueError('Prepared input/source changed after child capture')
        report.update(result='complete_official_initial_frame_capture_only', child_receipt=child)
        status = 0
    except BaseException as error:
        report.update(result='capture_error_or_interrupted', error_type=type(error).__name__, error=str(error))
        status = 2 if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else 1
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        report.update(exit_code=status, completed_utc=datetime.now(timezone.utc).isoformat())
        publish(args.output / 'capture_report.json', report)
    print(json.dumps(dict(result=report['result'], report=str(args.output / 'capture_report.json'))))
    return status


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '_capture':
        capture_child(Path(sys.argv[2]))
    else:
        raise SystemExit(main())
