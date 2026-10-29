"""Run the original SINGLE_NODE simulation with an explicit NPU renderer.

The parent uses CPU-only supervision.  The child imports the original Hydra
runner, BaseEnv and run_runners; no simulation or scoring formula is replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time

from simengine_ascend.file_protocol import check_stop, deadline_epoch
from simengine_ascend.launch_closed_loop import publish, validate_runner_reports
from simengine_ascend.owned_process import cleanup_owned, peek_exit
from simengine_ascend.window_contract import window_contract, inspect_scene_window

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / 'projects/SimEngine'
ORIGINAL_RUNNER = SIMULATION / 'worldengine/runner/run_simulation.py'


def sha256(path, guard=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard()
            block = stream.read(8 * 1024 ** 2)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_identity(guard=lambda: None):
    digest = hashlib.sha256()
    files = sorted(
        list((SIMULATION / 'worldengine').rglob('*.py'))
        + list((SIMULATION / 'worldengine').rglob('*.yaml'))
        + list((ROOT / 'simengine_ascend').glob('*.py'))
    )
    for path in files:
        guard()
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return {'sha256': digest.hexdigest(), 'files': len(files)}


def existing(path, directory=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Input must not be a symlink: ' + str(path))
    resolved = path.resolve(strict=True)
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError('Invalid input path: ' + str(path))
    return resolved


def fresh_output(path):
    path = Path(path).absolute()
    if '..' in path.parts or os.path.lexists(path):
        raise ValueError('--output must be a new path without parent traversal')
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise ValueError('Output parent must exist and contain no symlink components')
    return path


def hydra_value(value):
    """A quoted Hydra scalar is one argv item, including paths with spaces."""
    return json.dumps(str(value), ensure_ascii=False)


def load_scene_ids(path):
    values = json.loads(Path(path).read_bytes())
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('Expected scene IDs must be a nonempty unique JSON string list')
    return values


def build_overrides(args):
    window = window_contract(getattr(args, 'num_history', 4), getattr(args, 'num_future', 8))
    values = {
        'num_history': window['num_history'],
        'num_future': window['num_future'],
        'distributed_mode': 'SINGLE_NODE',
        'hydra.job.chdir': False,
        'gpu': False,
        'number_of_gpus_allocated_per_simulation': 0,
        'number_of_cpus_allocated_per_simulation': args.cpu_threads,
        'debug_mode': False,
        'debug_scene_name': None,
        'debug_static_world': False,
        'data_file_path': args.scene_pkl,
        'asset_folder_path': args.asset_root,
        'nuplan_map_root': args.maps_root,
        'output_dir': args.output,
        'data_output_dir': args.output / 'openscene_format',
        'renderer.device': args.device,
        'renderer.render_depth': True,
        'renderer.rasterization_backend': 'cann',
        'renderer.npu_compat_manifest': args.validated_manifest,
        'exit_on_failure': True,
        'enable_resume': False,
        'with_render_manager': True,
        'with_data_manager': True,
        'with_metric_manager': args.pdm_metrics,
        'with_dense_reward_manager': args.dense_rewards,
        'visualize_BEV': False,
        'visualize_video': False,
        'NC_DAC_video_capture': False,
        'save_data': False,
        'ego_controller': 'log_play_controller',
        'ego_navigation': 'trajectory_navigation',
        'agent_controller': 'log_play_controller',
        'agent_policy': 'idm_policy' if args.reaction == 'R' else 'trajectory_policy',
        'agent_navigation': 'idm_navigation' if args.reaction == 'R' else 'trajectory_navigation',
        'use_planner_actions': args.mode == 'closed-loop',
        'ego_policy': 'env_input_policy' if args.mode == 'closed-loop' else 'trajectory_policy',
        'ego_client': 'navformer_client' if args.mode == 'closed-loop' else 'base_client',
    }
    if args.mode == 'closed-loop':
        values['planner_data_path'] = args.planner_data_path
        values['planner_client_folder'] = args.planner_client_folder
    additions = {'renderer.rasterization_backend', 'renderer.npu_compat_manifest'}
    overrides = []
    for name, value in values.items():
        if isinstance(value, bool):
            scalar = str(value).lower()
        elif value is None:
            scalar = 'null'
        elif isinstance(value, int):
            scalar = str(value)
        else:
            scalar = hydra_value(value)
        overrides.append(('+' if name in additions else '') + name + '=' + scalar)
    return overrides


def runner_environment(args):
    environment = dict(os.environ)
    # Keep the original CPU PDM branch: transfer_to_npu must not rewrite CUDA
    # availability in the simulation process. The NPU renderer is explicit.
    environment.update(
        WORLDENGINE_ROOT=str(ROOT),
        CUDA_VISIBLE_DEVICES='',
        HYDRA_FULL_ERROR='1',
        WORLDENGINE_CLOSED_LOOP_DEADLINE_UTC=args.deadline_utc,
        WORLDENGINE_CLOSED_LOOP_STOP_FILE=str(args.stop_file),
        OMP_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads),
    )
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT), str(SIMULATION)]
        + ([environment['PYTHONPATH']] if environment.get('PYTHONPATH') else [])
    )
    return environment


def run_owned(command, cwd, environment, log_path, guard):
    """Supervise one direct child group; never signal an unrelated process."""
    child = None
    with Path(log_path).open('xb', buffering=0) as log:
        try:
            guard()
            child = subprocess.Popen(
                command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while peek_exit(child) is None:
                guard()
                time.sleep(0.1)
            original_exit = peek_exit(child)
            if original_exit != 0:
                raise RuntimeError('Original simulation runner exited %s; inspect %s'
                                   % (original_exit, log_path))
            return original_exit
        finally:
            if child is not None:
                cleanup_owned(child, grace_seconds=3)


def validate_images(output, guard):
    """Decode actual saved JPEGs; this is not a reference-image comparison."""
    from PIL import Image
    folder = Path(output) / 'openscene_format/sensor_blobs'
    count = total = 0
    digest = hashlib.sha256()
    for path in sorted(folder.rglob('*.jpg')):
        guard()
        if path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe rendered image: ' + str(path))
        with Image.open(path) as picture:
            if picture.format != 'JPEG' or min(picture.size) < 1:
                raise ValueError('Invalid rendered JPEG: ' + str(path))
            picture.verify()
        with Image.open(path) as picture:
            picture.load()
        digest.update(path.relative_to(output).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path, guard)))
        count += 1
        total += path.stat().st_size
    if count == 0:
        raise ValueError('No real rendered JPEG was saved by the original DataManager')
    return {'decoded_jpegs': count, 'total_bytes': total, 'inventory_sha256': digest.hexdigest(),
            'full_camera_frame_coverage_accepted': False, 'reference_image_comparison': False}


def run_original(device, overrides):
    """Child bootstrap: register NPU, then execute the untouched Hydra runner."""
    sys.path.insert(0, str(SIMULATION))
    import torch
    import torch_npu  # noqa: F401; only the explicitly selected NPU child.
    torch.npu.set_device(device)
    if torch.cuda.is_available():
        raise RuntimeError('CUDA must be hidden so original optional PDM uses CPU')
    import worldengine
    if not Path(worldengine.__file__).resolve().is_relative_to(SIMULATION / 'worldengine'):
        raise RuntimeError('A foreign worldengine package shadows the selected simulator')
    sys.argv = [str(ORIGINAL_RUNNER)] + overrides
    try:
        runpy.run_path(str(ORIGINAL_RUNNER), run_name='__main__')
    finally:
        # The original runner does not explicitly close BaseEnv. Flush its
        # original DataManager once for the final scene, then close the engine.
        from worldengine.engine.engine_utils import get_engine, close_engine
        engine = get_engine()
        try:
            if engine is not None and 'data_manager' in engine.managers:
                engine.managers['data_manager'].save_data()
        finally:
            close_engine()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-pkl', type=Path, required=True)
    parser.add_argument('--scene-sha256', help='Optional expected digest from official scene preparation')
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--maps-root', type=Path, required=True)
    parser.add_argument('--validated-manifest', type=Path, required=True)
    parser.add_argument('--device', required=True, help='Explicit visible logical NPU, e.g. npu:0')
    parser.add_argument('--output', type=Path, required=True, help='New job output; parent must already exist')
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--expected-scene-ids', type=Path, required=True)
    parser.add_argument('--mode', choices=('replay', 'closed-loop'), required=True)
    parser.add_argument('--reaction', choices=('NR', 'R'), default='NR')
    parser.add_argument('--pdm-metrics', action='store_true', help='Enable the original CPU PDM MetricManager')
    parser.add_argument('--dense-rewards', action='store_true', help='Enable original CPU dense scoring; needs original vocabulary')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--num-history', type=int, choices=(4,), default=4,
                        help='Four real consecutive planner-history frames; unchanged original model contract')
    parser.add_argument('--num-future', type=int, choices=(8, 20), default=8,
                        help='Eight preserves the validated 12-frame window; twenty requests 24 real frames')
    parser.add_argument('--planner-data-path', type=Path)
    parser.add_argument('--planner-client-folder', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print the exact command without executing the simulator')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'npu:\d+', args.device):
        parser.error('--device must explicitly be npu:N')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    cutoff = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if cutoff.tzinfo is None or cutoff.utcoffset().total_seconds() != 0:
        parser.error('--deadline-utc must include an explicit UTC timezone')
    if deadline_epoch(args.deadline_utc.replace('Z', '+00:00')) <= time.time():
        parser.error('--deadline-utc must be in the future')
    args.deadline_utc = cutoff.isoformat()
    for name in ('scene_pkl', 'validated_manifest', 'expected_scene_ids'):
        setattr(args, name, existing(getattr(args, name)))
    for name in ('asset_root', 'maps_root'):
        setattr(args, name, existing(getattr(args, name), directory=True))
    args.output = fresh_output(args.output)
    stop_path = args.stop_file or os.environ.get('WORLDENGINE_CLOSED_LOOP_STOP_FILE')
    args.stop_file = Path(stop_path).absolute() if stop_path else args.output / 'STOP'
    if args.mode == 'closed-loop':
        if args.planner_data_path is None or args.planner_client_folder is None:
            parser.error('Closed-loop mode requires both shared planner directories')
        for name in ('planner_data_path', 'planner_client_folder'):
            setattr(args, name, existing(getattr(args, name), directory=True))
    elif args.planner_data_path is not None or args.planner_client_folder is not None:
        parser.error('Replay mode does not use external planner directories')
    if args.scene_sha256 and not re.fullmatch(r'[0-9a-f]{64}', args.scene_sha256):
        parser.error('--scene-sha256 must be a lowercase SHA-256 digest')
    window = window_contract(args.num_history, args.num_future)
    if window['frames'] == 24 and not args.scene_sha256:
        parser.error('The 24-frame mode requires --scene-sha256 from trusted official preparation')
    if window['frames'] == 24 and (args.dense_rewards or args.pdm_metrics):
        parser.error('The 24-frame candidate covers replay and closed-loop consumption only; PDM and dense-reward scoring require a separately reviewed original-horizon profile')
    scene_ids = load_scene_ids(args.expected_scene_ids)
    interrupted = []
    def guard():
        if interrupted:
            raise InterruptedError('Simulation wrapper received a stop signal')
        check_stop(args.stop_file, args.deadline_utc)
    guard()
    scene_sha = sha256(args.scene_pkl, guard)
    if args.scene_sha256 and scene_sha != args.scene_sha256:
        raise ValueError('Prepared scene SHA-256 differs from the expected input')
    scene_window = None
    if window['frames'] == 24:
        scene_window = inspect_scene_window(args.scene_pkl, args.scene_sha256, scene_ids,
                                            args.num_history, args.num_future, guard)
    vocabulary = ROOT / 'data/alg_engine/test_8192_kmeans.npy'
    if args.dense_rewards:
        existing(vocabulary)
    command = [sys.executable, '-m', 'simengine_ascend.run_simulation', '_run_original', args.device]
    command += build_overrides(args)
    report = {
        'result': 'command_prepared_not_executed', 'mode': args.mode, 'reaction': args.reaction,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': args.deadline_utc,
        'scene_pkl': str(args.scene_pkl), 'scene_sha256': scene_sha,
        'configured_window': window, 'scene_window_preflight': scene_window,
        'expected_scene_ids': scene_ids, 'asset_root': str(args.asset_root), 'maps_root': str(args.maps_root),
        'validated_manifest': str(args.validated_manifest), 'manifest_sha256': sha256(args.validated_manifest, guard),
        'source': source_identity(guard), 'command': command, 'cwd': str(ROOT),
        'device': args.device, 'pdm_metrics_CPU': args.pdm_metrics, 'dense_rewards_CPU': args.dense_rewards,
        'ray_worker_created': False, 'gpu_field_scope': 'Original worker scheduling only; renderer remains explicitly NPU',
        'simulation_quality_accepted': False, 'GPU_parity_accepted': False,
    }
    if args.dense_rewards:
        report['dense_reward_vocabulary_sha256'] = sha256(vocabulary, guard)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    args.output.mkdir(mode=0o700)
    report_path = args.output / 'simulation_launch_report.json'
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, lambda *_: interrupted.append(True))
    result = 1
    try:
        report['result'] = 'running_original_simulation'
        publish(report_path, report)
        run_owned(command, ROOT, runner_environment(args), args.output / 'simulation.log', guard)
        guard()
        report['runner_reports'] = validate_runner_reports(args.output, len(scene_ids), scene_ids)
        if not (args.output / 'simulation_completed.flag').is_file():
            raise ValueError('Original simulation did not publish its completion flag')
        report['rendered_images'] = validate_images(args.output, guard)
        if source_identity(guard) != report['source'] or sha256(args.scene_pkl, guard) != scene_sha:
            raise RuntimeError('Simulation source or scene input changed during execution')
        if sha256(args.validated_manifest, guard) != report['manifest_sha256']:
            raise RuntimeError('NPU renderer manifest changed during execution')
        report['result'] = 'completed_original_simulation_and_output_checks'
        result = 0
    except BaseException as error:
        report.update(result='simulation_error_or_interrupted', error_type=type(error).__name__, error=str(error))
        result = 2 if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else 1
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        report.update(actual_exit=result, completed_utc=datetime.now(timezone.utc).isoformat())
        publish(report_path, report)
    print(json.dumps({'result': report['result'], 'report': str(report_path)}, indent=2))
    return result


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_run_original':
        run_original(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(main())
