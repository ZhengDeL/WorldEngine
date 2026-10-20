#!/usr/bin/env python3
"""Audit metadata and all JPEGs from original configured NR/R replay windows.

Use a CPU environment with NumPy, Pillow and PyYAML. The job contains replay_NR
and replay_R outputs; the prepared directory defaults to job/prepared and can
be supplied separately. Explicit preparation/source SHA256 values bind the
inputs. Mount these inputs read-only and write the new JSON outside them.
No renderer, planner, subprocess or accelerator API is called. This checks the
configured replay window; it does not establish full-log, planner, image-quality
or performance acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import stat
import time

import numpy as np
from PIL import Image
import yaml

SCHEMA = 1


def need(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    need(path.resolve(strict=True) == path, 'Symlink component: '+str(path))
    need(stat.S_ISREG(path.lstat().st_mode), 'Expected regular file: '+str(path))
    return path


def within(root, relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
         'Expected a relative POSIX path')
    value = PurePosixPath(relative)
    need(not value.is_absolute() and all(p not in ('', '.', '..') for p in relative.split('/')),
         'Unsafe relative path: '+relative)
    return regular(Path(root).joinpath(*value.parts))


def digest(path, guard=lambda: None):
    path = regular(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            guard()
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    regular(path)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def load_pickle(path, expected_sha, maximum, guard):
    path = regular(path)
    need(path.stat().st_size <= maximum, 'Pickle exceeds explicit byte budget')
    actual = digest(path, guard)
    if expected_sha is not None:
        need(actual == expected_sha, 'Verified pickle content differs: '+str(path))
    with path.open('rb') as stream:
        value = pickle.load(stream)
        need(not stream.read(1), 'Trailing pickle bytes')
    guard()
    need(digest(path, guard) == actual, 'Pickle changed during audit')
    return value, dict(bytes=path.stat().st_size, sha256=actual)


def source_identity(root, guard):
    root = Path(root)
    world = root/'projects/SimEngine/worldengine'
    need(world.is_dir() and (root/'simengine_ascend').is_dir(), 'Missing pinned source tree')
    paths = list(world.rglob('*.py')) + list(world.rglob('*.yaml')) + list((root/'simengine_ascend').glob('*.py'))
    h = hashlib.sha256()
    # Match the Linux run's POSIX ordering even when fixtures run on Windows.
    for p in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        guard()
        regular(p)
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return dict(sha256=h.hexdigest(), files=len(paths))


def integer(value, name, minimum=0):
    need(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
         and value >= minimum, 'Unsupported/non-integer '+name)
    return int(value)


def microseconds(value, name):
    need(isinstance(value, (int, float, np.integer, np.floating))
         and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
         and value >= 0 and float(value).is_integer(), 'Unsupported microsecond timestamp '+name)
    if isinstance(value, (float, np.floating)):
        need(value <= 2**53, 'Floating timestamp cannot establish exact integer identity')
    return int(value)


def log_steps(path, scenario_id, final_step):
    text = regular(path).read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    pattern = re.compile(r'Step (\d+)/(\d+) for scenario \((\d+)/(\d+)\) ([^\r\n]+)')
    matches = [(int(a), int(b), int(c), int(d), e.strip()) for a, b, c, d, e in pattern.findall(text)]
    need(matches, 'No original BaseEnv Step records found')
    need([x[0] for x in matches] == list(range(final_step)),
         'Logged update count/order does not reach the configured truncation')
    need(all(x[1:] == (final_step, 1, 1, scenario_id) for x in matches),
         'Logged scenario or truncation denominator differs')
    need('Generated completion flag file:' in text, 'Original runner did not log completion-flag publication')
    return dict(logged_updates=len(matches), episode_steps_before_update=[x[0] for x in matches],
                truncation_episode_step=final_step,
                convention='Reset renders frame 0; each logged pre-update step renders the next frame',
                log_sha256=digest(path), final_lines=text.splitlines()[-18:])


def audit_phase(job, reaction, scene, key, prepared, source, guard, maximum_pickle,
                expected_window, expected_cameras, observed=None):
    folder = Path(job)/('replay_'+reaction)
    if observed is None:
        observed = {}
    observed.update(original_scene_keys=sorted(scene) if isinstance(scene, dict) else None,
                    original_scene_type=type(scene).__name__)
    report = read_json(within(folder, 'simulation_launch_report.json'))
    need(report.get('result') == 'completed_original_simulation_and_output_checks'
         and report.get('actual_exit') == 0 and report.get('mode') == 'replay'
         and report.get('reaction') == reaction, 'Replay wrapper did not establish the requested mode')
    need(report.get('source') == source and report.get('scene_sha256') == prepared['derived_scene']['sha256'],
         'Wrapper source or derived scene SHA differs')
    need(report.get('expected_scene_ids') == [key], 'Wrapper selected a different scene')
    config_path = within(folder, '.hydra/config.yaml')
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    observed['configuration_keys'] = sorted(cfg) if isinstance(cfg, dict) else None
    observed['configuration_type'] = type(cfg).__name__
    need(isinstance(cfg, dict), 'Hydra config is not a mapping')
    history = integer(cfg.get('num_history'), 'num_history', 1)
    future = integer(cfg.get('num_future'), 'num_future', 1)
    frames = history+future
    need(frames == expected_window, 'Resolved configuration differs from the explicitly audited window')
    final_step = frames-1
    need(integer(cfg.get('max_step'), 'max_step', 1) >= final_step,
         'max_step could terminate before the configured window')
    need(integer(cfg.get('decision_repeat'), 'decision_repeat', 1) == 1,
         'This audit only implements the original decision_repeat=1 convention')
    wanted = dict(distributed_mode='SINGLE_NODE', exit_on_failure=True, enable_resume=False,
                  with_render_manager=True, with_data_manager=True, use_planner_actions=False,
                  ego_policy='trajectory_policy', ego_controller='log_play_controller',
                  agent_policy='idm_policy' if reaction == 'R' else 'trajectory_policy',
                  agent_navigation='idm_navigation' if reaction == 'R' else 'trajectory_navigation')
    need(all(cfg.get(name) == val for name, val in wanted.items()), 'Actual original policies/mode differ')
    renderer = cfg.get('renderer')
    need(isinstance(renderer, dict) and isinstance(report.get('device'), str)
         and re.fullmatch(r'npu:\d+', report['device']) is not None
         and renderer.get('device') == report['device']
         and renderer.get('rasterization_backend') == 'cann' and renderer.get('render_depth') is True,
         'Resolved renderer is not the recorded explicit CANN RGB+ED path')
    need(cfg.get('with_metric_manager') is False and cfg.get('with_dense_reward_manager') is False,
         'Unexpected scoring mode; this audit does not establish scoring acceptance')
    steps = log_steps(within(folder, 'simulation.log'), key, final_step)
    flag = within(folder, 'simulation_completed.flag')
    original_rows = report.get('runner_reports', {}).get('scenarios')
    need(isinstance(original_rows, list) and len(original_rows) == 1 and original_rows[0].get('scenario_name') == key
         and original_rows[0].get('succeeded') is True and original_rows[0].get('planner_name') is None,
         'Original runner success/identity is incomplete')
    runner_hashes = report['runner_reports'].get('original_runner_report_sha256')
    need(isinstance(runner_hashes, dict) and len(runner_hashes) == 1, 'Missing/ambiguous original report identity')
    need({p.name for p in folder.glob('runner_report_*.json')} == {PurePosixPath(p).name for p in runner_hashes},
         'Original runner-report file inventory differs')
    for name, expected_sha in runner_hashes.items():
        p = within(folder, PurePosixPath(name).name)
        need(digest(p, guard) == expected_sha and read_json(p) == original_rows, 'Original runner report changed')
    scene_id = scene['id']
    metadata_root = folder/'openscene_format/meta_datas'
    expected_name = scene_id+'.pkl'
    need(sorted(p.name for p in metadata_root.iterdir()) == [expected_name], 'Unexpected/missing scene metadata files')
    rows, metadata_identity = load_pickle(within(metadata_root, expected_name), None, maximum_pickle, guard)
    observed.update(exported_metadata_type=type(rows).__name__,
        exported_metadata_count=len(rows) if isinstance(rows, (dict, list, tuple)) else None,
        exported_first_frame_keys=sorted(rows[0]) if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None)
    need(isinstance(rows, list) and len(rows) == frames, 'Metadata does not contain the exact configured frame count')
    cameras = scene.get('cameras')
    need(isinstance(cameras, dict) and len(cameras) == expected_cameras,
         'The original camera count differs from the explicitly audited count')
    names = list(cameras)
    original = scene.get('metadata', {}).get('openscene_data_infos_dict')
    need(isinstance(original, dict) and len(original) >= frames, 'Missing original frame calibration/metadata')
    original_values = list(original.values())
    log_length = integer(scene.get('log_length'), 'scene.log_length', 1)
    need(log_length >= frames, 'Original source log is shorter than the configured window')
    base = microseconds(scene.get('base_timestamp'), 'scene.base_timestamp')
    rate = scene.get('sample_rate')
    need(isinstance(rate, (float, int, np.integer, np.floating)) and not isinstance(rate, (bool, np.bool_))
         and math.isfinite(float(rate)) and rate > 0, 'Unsupported scene.sample_rate')
    suffix = scene_id.split('-')[-1] if len(scene_id.split('-')[-1]) == 3 else '000'
    scene_token = '-'.join(scene_id.split('-')[-2:]) if len(scene_id.split('-')[-1]) == 3 else scene_id.split('-')[-1]
    tokens = [item['token']+'-'+suffix for item in original_values[:frames]]
    need(len(set(tokens)) == frames, 'Original frame token identities are not unique')
    expected_paths, image_rows, frame_rows = set(), [], []
    for index, item in enumerate(rows):
        guard()
        need(isinstance(item, dict), 'Unexpected exported metadata row type')
        need(integer(item.get('frame_idx'), 'export.frame_idx') == index and item.get('token') == tokens[index],
             'Exported frame indices/tokens omit, reorder or replace original frames')
        timestamp = base+int(index*rate*.05*1e6)
        need(microseconds(item.get('timestamp'), 'export.timestamp') == timestamp, 'Export timestamp differs from original formula')
        need(item.get('sample_prev') == (tokens[index-1] if index else None)
             and item.get('sample_next') == (tokens[index+1] if index+1 < frames else None),
             'Exported previous/next links are broken')
        need(item.get('log_name') == scene_id and item.get('scene_name') == scene_id
             and item.get('scene_token') == scene_token, 'Exported scene identities differ')
        cams = item.get('cams')
        original_cams = original_values[index].get('cams')
        need(isinstance(cams, dict) and isinstance(original_cams, dict)
             and set(cams) == set(names) == set(original_cams), 'A frame omits/adds original cameras')
        for name in names:
            guard()
            source_path = original_cams[name].get('data_path')
            need(isinstance(source_path, str) and source_path, 'Unknown original camera data_path')
            filename = source_path.split('/')[-1].split('.')[0]
            relative = scene_token+'/'+name+'/'+filename+'-'+suffix+'.jpg'
            need(cams[name].get('data_path') == relative and relative not in expected_paths,
                 'Camera output path differs or was reused across frames')
            expected_paths.add(relative)
            p = within(folder/'openscene_format/sensor_blobs', relative)
            dims = (integer(cameras[name].get('width'), 'camera.width', 1),
                    integer(cameras[name].get('height'), 'camera.height', 1))
            with Image.open(p) as image:
                need(image.format == 'JPEG' and image.size == dims, 'JPEG format/dimensions differ from original camera')
                image.verify()
            with Image.open(p) as image:
                image.load()
                need(image.size == dims, 'Decoded JPEG dimensions differ')
            image_rows.append(dict(frame_idx=index, token=tokens[index], camera=name,
                path=p.relative_to(folder).as_posix(), bytes=p.stat().st_size, sha256=digest(p, guard),
                width=dims[0], height=dims[1], fully_decoded=True))
        frame_rows.append(dict(frame_idx=index, token=tokens[index], timestamp=timestamp,
            sample_prev=item['sample_prev'], sample_next=item['sample_next'], camera_names=names))
    actual_paths = set()
    for p in (folder/'openscene_format/sensor_blobs').rglob('*'):
        guard()
        need(not p.is_symlink(), 'Symlink in rendered image tree')
        if p.is_file():
            regular(p)
            need(p.suffix == '.jpg', 'Unexpected non-JPEG file in sensor_blobs')
            actual_paths.add(p.relative_to(folder/'openscene_format/sensor_blobs').as_posix())
    need(actual_paths == expected_paths, 'Extra/missing camera-frame JPEGs')
    h = hashlib.sha256()
    for item in sorted(image_rows, key=lambda item: item['path']):
        h.update(item['path'].encode())
        h.update(bytes.fromhex(item['sha256']))
    image_inventory = dict(decoded_jpegs=len(image_rows), total_bytes=sum(x['bytes'] for x in image_rows),
                           inventory_sha256=h.hexdigest())
    need(all(report['rendered_images'].get(k) == val for k, val in image_inventory.items()),
         'Independent full JPEG inventory differs from original wrapper inventory')
    return dict(result='pass_configured_replay_window_coverage', reaction=reaction,
        source=source, configured_window_frames=frames, original_log_length=log_length,
        original_frame_metadata_count=len(original_values),
        full_original_log_coverage=(frames == log_length == len(original_values)),
        time_span_seconds=(frame_rows[-1]['timestamp']-frame_rows[0]['timestamp'])/1e6,
        num_history=history, num_future=future, actual_original_policies=wanted,
        original_camera_dimensions={n:[int(cameras[n]['width']),int(cameras[n]['height'])] for n in names},
        hydra_config_sha256=digest(config_path, guard), configuration_keys=sorted(cfg),
        original_scene_keys=sorted(scene), exported_frame_keys=sorted(rows[0]),
        metadata_identity=metadata_identity, completion_flag_sha256=digest(flag, guard),
        log_step_check=steps, original_runner_rows=original_rows, frame_inventory=frame_rows,
        jpeg_inventory=image_inventory, jpeg_files=image_rows,
        acceptance=dict(configured_window_camera_frame_coverage=True, GPU_parity=False,
                        reference_image_quality=False, planner_feedback=False, scores=False, performance=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True, help='Parent of replay_NR and replay_R output directories')
    parser.add_argument('--source-root', type=Path, required=True, help='Exact simulator checkout used by the replay')
    parser.add_argument('--prepared-dir', type=Path, help='Official prepare_scene output; default: JOB/prepared')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON outside read-only job/source roots')
    parser.add_argument('--expected-prepare-receipt-sha256', required=True, help='Pinned prepare_receipt.json SHA256')
    parser.add_argument('--expected-source-sha256', required=True, help='Approved simulation runtime source SHA256')
    parser.add_argument('--expected-window-frames', type=int, default=12)
    parser.add_argument('--expected-cameras', type=int, default=8)
    parser.add_argument('--max-pickle-bytes', type=int, default=256*1024**2)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args(argv)
    cutoff = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    need(cutoff.tzinfo is not None and cutoff.utcoffset().total_seconds() == 0, 'Explicit UTC deadline required')
    need(args.expected_window_frames >= 2 and args.expected_cameras >= 1 and args.max_pickle_bytes > 0,
         'Invalid explicit audit bounds')
    need(all(re.fullmatch('[0-9a-f]{64}', x) for x in (args.expected_source_sha256, args.expected_prepare_receipt_sha256)),
         'Expected exact lowercase source/input SHA256')
    job, source_root = args.job.resolve(strict=True), args.source_root.resolve(strict=True)
    prepared_dir = (args.prepared_dir or job/'prepared').resolve(strict=True)
    output = args.output.absolute()
    need(output.parent.resolve(strict=True) == output.parent and not os.path.lexists(output), 'Output must be fresh in an existing nonsymlink parent')
    need(all(not output.is_relative_to(root) for root in (job, source_root, prepared_dir)),
         'Audit output must be outside input roots')
    report = dict(schema_version=SCHEMA, result='running', scope='original configured replay-window coverage only',
        started_utc=datetime.now(timezone.utc).isoformat(), phases={},
        auditor_sha256=digest(Path(__file__)), accelerator_execution=False,
        inputs=dict(job=str(job), source_root=str(source_root), prepared_dir=str(prepared_dir)),
        expected_source_sha256=args.expected_source_sha256,
        expected_prepare_receipt_sha256=args.expected_prepare_receipt_sha256)
    def guard():
        if time.time() >= cutoff.timestamp() or args.stop_file is not None and args.stop_file.exists():
            raise TimeoutError('Audit deadline/STOP reached')
    status = 1
    try:
        guard()
        source = source_identity(source_root, guard)
        need(source['sha256'] == args.expected_source_sha256, 'Selected simulator source changed')
        prepare_path = within(prepared_dir, 'prepare_receipt.json')
        need(digest(prepare_path, guard) == args.expected_prepare_receipt_sha256, 'Preparation receipt changed')
        prepared = read_json(prepare_path)
        need(prepared.get('result') == 'pass_official_single_scene_asset_preparation', 'Official preparation is not complete')
        scene_pickle, scene_identity = load_pickle(within(prepared_dir, prepared['derived_scene']['path']),
            prepared['derived_scene']['sha256'], args.max_pickle_bytes, guard)
        key = prepared['selected']['dictionary_key']
        need(isinstance(scene_pickle, dict) and list(scene_pickle) == [key], 'Derived scene selection differs')
        scene = scene_pickle[key]
        report['derived_scene_identity'] = scene_identity
        report['selected'] = prepared['selected']
        for reaction in ('NR', 'R'):
            observed = {}
            report['phases'][reaction] = dict(result='audit_incomplete', observed=observed)
            report['phases'][reaction] = audit_phase(job, reaction, scene, key, prepared, source,
                guard, args.max_pickle_bytes, args.expected_window_frames, args.expected_cameras, observed)
        guard()
        need(source_identity(source_root, guard) == source, 'Simulator source changed during audit')
        need(digest(prepare_path, guard) == args.expected_prepare_receipt_sha256, 'Preparation receipt changed during audit')
        report['result'] = 'pass_two_configured_replay_windows_coverage'
        status = 0
    except BaseException as error:
        report.update(result='incomplete_or_rejected', error_type=type(error).__name__, error=str(error))
        status = 2 if isinstance(error, (TimeoutError, KeyboardInterrupt)) else 1
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(result=report['result'], report=str(output), exit_code=status)))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
