"""Server-only genuine hist65 clips through the unchanged WorldEngine converter.

convert: extract 26 pinned raw PKLs to a new private directory and call original
parse_log; history: extract only those clips' four-frame camera references.
No rewards, token filters, source files or candidate dimensions are modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import runpy
import signal
import sys
import tarfile
import time

FOLDER = 'synthetic_reaction_pdm_v1.0-0'
LOCKS = {
    'metadata': (3320437526, 'a7a6d725d926d1200410e5e9b8c39368b5b519bba6535816bf528bcc965331da'),
    'history': (241575885, '25ab91a08a38e7a7270ed983b51e01fc6d6f02bbb955ef8f4abdfcccfc9fc48c'),
    'index': (1824152, 'a1d2eb52a8f93d6ba6e3037e0375a88706bcdc471d44117e1bba0a853e4c42ee'),
}
CONVERTER = 'projects/AlgEngine/scripts/data_converter/nuplan/e2e_nuplan_openscene_navsim_multiprocessing.py'
CONVERTER_SHA_LF = 'e924459ca95c67ca51cc36fb24e4b03e91738ff100a72e6ad3a7b3b13a1f74e5'
CAMS = {'CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2'}


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locked(path, key):
    size, expected = LOCKS[key]
    if path.is_symlink() or path.stat().st_size != size or sha(path) != expected:
        raise ValueError('Incomplete or changed publisher input: ' + key)
    return dict(path=str(path), bytes=size, sha256=expected)


def source_fp(source):
    digest = hashlib.sha256()
    files = sorted([*(source / 'projects/AlgEngine').rglob('*.py'), *(source / 'ascend').glob('*.py')],
        key=lambda p: p.relative_to(source).as_posix())
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe source camera path: ' + str(value))
    return path.as_posix()


def clip_checks(raw, infos):
    import numpy as np
    if not isinstance(raw, list) or len(raw) != 12 or len(infos) != 12:
        raise ValueError('Expected original twelve-frame clip')
    if [int(x['frame_idx']) for x in raw] != list(range(12)):
        raise ValueError('Original frame order/indices differ; no resampling allowed')
    stamps = np.array([x['timestamp'] for x in raw], dtype=np.int64)
    if not np.all(np.abs(np.diff(stamps) - 500000) <= 20000):
        raise ValueError('Original clip is not continuous at 2 Hz')
    if len({x['log_name'] for x in raw}) != 1 or len({x['token'] for x in raw}) != 12:
        raise ValueError('Mixed log or duplicate frame tokens')
    cameras = []
    for pos, (r, d) in enumerate(zip(raw, infos)):
        if any(d[k] != r[k] for k in ('token', 'log_name', 'frame_idx', 'timestamp')):
            raise ValueError('Converter changed original frame identity')
        for k in ('ego2global', 'lidar2ego', 'lidar2global'):
            m = np.asarray(r[k])
            if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-3):
                raise ValueError('Invalid original rigid calibration: ' + k)
            if not np.array_equal(m, d[k]):
                raise ValueError('Converter changed original calibration')
        if not np.allclose(r['ego2global'] @ r['lidar2ego'], r['lidar2global'], rtol=1e-5, atol=1e-4):
            raise ValueError('Original lidar/ego/global transforms are inconsistent')
        if set(r['cams']) != CAMS:
            raise ValueError('Expected eight original cameras')
        for name, cam in r['cams'].items():
            for k in ('sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic', 'distortion'):
                if not np.isfinite(np.asarray(cam[k])).all() or not np.array_equal(cam[k], d['cams'][name][k]):
                    raise ValueError('Camera calibration changed or nonfinite')
            rot = np.asarray(cam['sensor2lidar_rotation'])
            if rot.shape != (3, 3) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-3):
                raise ValueError('Invalid camera rotation')
            if pos < 4:
                cameras.append(dict(frame_idx=pos, camera=name, data_path=relative(cam['data_path'])))
    anchor = infos[3]
    if np.asarray(anchor['gt_fut_bbox_sdc_lidar']).shape != (1, 8, 9) or np.asarray(anchor['gt_pre_bbox_sdc_lidar']).shape != (1, 3, 9):
        raise ValueError('Unexpected original converter trajectory dimensions')
    if int(np.asarray(anchor['gt_fut_bbox_sdc_mask']).sum()) != 8 or int(np.asarray(anchor['gt_pre_bbox_sdc_mask']).sum()) != 3:
        raise ValueError('Anchor does not have complete genuine past/future labels')
    global_xyz = np.array([x['ego2global'][:3, 3] for x in raw[4:12]])
    m = raw[3]['lidar2global']
    lidar_xyz = (global_xyz - m[:3, 3]) @ m[:3, :3]
    if not np.allclose(anchor['gt_fut_bbox_sdc_global'][0, :, :3], global_xyz, rtol=0, atol=1e-8):
        raise ValueError('Future global labels disagree with real future ego states')
    if not np.allclose(anchor['gt_fut_bbox_sdc_lidar'][0, :, :3], lidar_xyz, rtol=1e-5, atol=1e-4):
        raise ValueError('Future lidar labels disagree with real future ego states')
    for key in ('gt_boxes', 'gt_velocity', 'gt_bboxes_global', 'gt_fut_bbox_lidar', 'gt_pre_bbox_lidar',
            'gt_fut_bbox_sdc_lidar', 'gt_fut_bbox_sdc_global', 'gt_pre_bbox_sdc_lidar'):
        if not np.isfinite(np.asarray(anchor[key])).all():
            raise ValueError('Nonfinite converted real annotation: ' + key)
    return dict(log_name=raw[3]['log_name'], original_log_name='-'.join(raw[3]['log_name'].split('-')[:2]),
        original_scenario_token=raw[3]['log_name'].split('-')[2], anchor_token=raw[3]['token'],
        frame_count=12, history_camera_references=cameras, future_mask_sum=8, past_mask_sum=3,
        agent_count=len(anchor['gt_boxes']), agent_future_shape=list(anchor['gt_fut_bbox_lidar'].shape))


def convert(args, report):
    import yaml
    root = args.output / 'synthetic_root'
    rawroot = root / 'meta_datas' / FOLDER
    rawroot.mkdir(parents=True)
    report['inputs'] = dict(metadata=locked(args.archive, 'metadata'), index=locked(args.hist_index, 'index'))
    index = json.loads(args.hist_index.read_text())
    keys = [key for key in index if re.search(r'hist_65(?:\.tar\.gz)?$', key)]
    if len(keys) != 1:
        raise ValueError('No unique hist65 archive key')
    scenes = set(index[keys[0]])
    splitpath = args.source / ('projects/AlgEngine/configs/navsim_splits/simscale_split/' + FOLDER + '.yaml')
    split = yaml.safe_load(splitpath.read_text())
    logmap = {'-'.join(x.rsplit('-', 2)[-2:]): x for x in split['log_names']}
    if len(scenes) != 26 or not scenes <= set(logmap):
        raise ValueError('Publisher hist65/split mapping differs')
    report.update(split_sha256=sha(splitpath), original_split_filter_unchanged=True, clips=[], inputs_sha_verified=True)
    script = args.source / CONVERTER
    if hashlib.sha256(script.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != CONVERTER_SHA_LF:
        raise ValueError('Original converter source differs')
    report['converter'] = dict(path=CONVERTER, sha256=sha(script), sha256_canonical_lf=CONVERTER_SHA_LF,
        filter_boxes=False, invocation='runpy original parse_log; default 2Hz/1.5s past/4s future')
    oldargv = sys.argv[:]
    sys.path.insert(0, str(script.parent))
    try:
        sys.argv = [str(script), '--data_root', str(root), '--split', FOLDER, '--num_workers', '1', '--output_folder', 'meta_datas_navformer']
        module = runpy.run_path(str(script), run_name='private_original_converter')
    finally:
        sys.argv = oldargv
    remaining = dict((logmap[x] + '.pkl', x) for x in scenes)
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            basename = PurePosixPath(member.name).name
            if basename not in remaining or not member.isfile():
                continue
            if member.size > 64 * 1024 * 1024:
                raise ValueError('Selected metadata member exceeds 64MiB')
            payload = archive.extractfile(member).read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError('Incomplete selected metadata member')
            rawpath = rawroot / basename
            if rawpath.exists():
                raise ValueError('Duplicate selected metadata member')
            rawpath.write_bytes(payload)
            raw = pickle.loads(payload)
            module['parse_log'](str(rawpath))
            converted = root / 'meta_datas_navformer' / FOLDER / basename
            with converted.open('rb') as stream:
                infos = pickle.load(stream)['infos']
            detail = clip_checks(raw, infos)
            detail.update(scene_id=remaining.pop(basename), raw_member=member.name,
                raw_bytes=len(payload), raw_sha256=hashlib.sha256(payload).hexdigest(),
                converted_relative_path=converted.relative_to(args.output).as_posix(), converted_sha256=sha(converted))
            if sha(rawpath) != detail['raw_sha256']:
                raise ValueError('Raw extracted source changed')
            report['clips'].append(detail)
            save(args.output / 'conversion_manifest.json', report)
            if not remaining:
                break
    if remaining:
        raise ValueError('Missing selected real scenes: ' + repr(remaining))
    report.update(result='pass_original_converter_26_real_clips', reward_alignment_checked=False,
        images_checked=False, dataset_training_accepted=False)


def history(args, report):
    import cv2
    import numpy as np
    base = json.loads((args.output / 'conversion_manifest.json').read_text())
    if base['result'] != 'pass_original_converter_26_real_clips' or base['source_python_sha256'] != args.source_fingerprint:
        raise ValueError('No accepted matching conversion manifest')
    report['inputs'] = dict(history=locked(args.archive, 'history'), conversion_manifest_sha256=sha(args.output / 'conversion_manifest.json'))
    wanted = {x['data_path'] for clip in base['clips'] for x in clip['history_camera_references']}
    report.update(expected_camera_paths=len(wanted), images=[], dataset_training_accepted=False, reward_alignment_checked=False)
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            hits = [path for path in wanted if member.name == path or member.name.endswith('/' + path)]
            if not hits:
                continue
            if len(hits) != 1 or member.size > 32 * 1024 * 1024:
                raise ValueError('Ambiguous or oversized selected image')
            path = hits[0]
            dest = args.output / 'synthetic_root' / 'sensor_blobs' / FOLDER / path
            if dest.exists():
                raise ValueError('Duplicate selected camera image')
            payload = archive.extractfile(member).read(member.size + 1)
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if len(payload) != member.size or decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
                raise ValueError('Original image incomplete or decode failed')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            report['images'].append(dict(data_path=path, raw_member=member.name, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), decoded_shape=list(decoded.shape)))
    found = {x['data_path'] for x in report['images']}
    report['missing_paths'] = sorted(wanted - found)
    if found != wanted:
        raise ValueError('History tar does not cover all original 4x8 camera references')
    report.update(result='pass_all_real_history_camera_references', actual_image_count=len(found), images_checked=True)



def audit_archive(path):
    records=hashlib.sha256();files=directories=expanded=0;seen=set()
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Archive contains links, special or unsafe members')
            if member.isfile():
                if member.name in seen:raise ValueError('Duplicate archive file member')
                seen.add(member.name);files+=1;expanded+=member.size
            else:directories+=1
            records.update((member.name+'\0'+str(member.size)+'\0'+str(member.type)+'\n').encode())
    return dict(files=files,directories=directories,expanded_file_bytes=expanded,member_inventory_sha256=records.hexdigest(),extractall_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['convert', 'history'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-runtime-sha256', '--source-fingerprint', dest='source_fingerprint', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--hist-index', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--execute', action='store_true', help='Execute CPU preparation; omitted means a read-only plan')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(result='dry_run_not_executed', mode=args.mode, source=str(args.source), expected_runtime_sha256=args.source_fingerprint, output=str(args.output), archive=str(args.archive), original_converter_sha256=CONVERTER_SHA_LF, no_training_acceptance=True), indent=2))
        return
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if deadline.tzinfo is None or not 1 <= args.max_seconds <= 3600:
        parser.error('Authorized deadline and bounded duration required')
    cutoff = min(deadline.timestamp() - 120, time.time() + args.max_seconds)
    if cutoff <= time.time() or 'meta_datas' in str(args.output) or (args.mode == 'convert' and args.output.exists()):
        parser.error('Use future cutoff and new output path without meta_datas substring')
    if args.mode == 'convert' and not args.hist_index:
        parser.error('convert needs original history index')
    if list(Path('/dev').glob('davinci*')):
        parser.error('CPU-only container must not mount NPU devices')
    args.output.mkdir(parents=True, exist_ok=args.mode == 'history')
    receipt = args.output / ('conversion_manifest.json' if args.mode == 'convert' else 'history_manifest.json')
    if receipt.exists() and args.mode == 'history':
        parser.error('History receipt already exists')
    report = dict(result='running', mode=args.mode, started_unix_seconds=time.time(),
        source_python_sha256=source_fp(args.source), raw_data_modified=False, rewards_modified=False,
        helper_sha256=sha(Path(__file__)), command=sys.argv[:])
    def expired(_signum, _frame):
        raise TimeoutError('Authorized CPU conversion deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(cutoff - time.time())))
    try:
        if report['source_python_sha256'] != args.source_fingerprint:
            raise ValueError('Pinned original source fingerprint differs')
        report['whole_archive_safety'] = audit_archive(args.archive)
        (convert if args.mode == 'convert' else history)(args, report)
        report['archive_sha256_after'] = sha(args.archive)
        expected = LOCKS['metadata' if args.mode == 'convert' else 'history'][1]
        if report['archive_sha256_after'] != expected:raise ValueError('Publisher archive changed during preparation')
    except BaseException as error:
        report.update(result='failed_retained', error=repr(error))
    finally:
        signal.alarm(0)
        report.update(source_python_sha256_after=source_fp(args.source), completed_unix_seconds=time.time())
        if report['source_python_sha256_after'] != args.source_fingerprint:
            report.update(result='failed_retained', source_changed=True)
        save(receipt, report)
    print(json.dumps(dict(result=report['result'], receipt=str(receipt), clips=len(report.get('clips', [])),
        images=len(report.get('images', [])), error=report.get('error')), indent=2), flush=True)
    raise SystemExit(0 if report['result'].startswith('pass_') else 1)


if __name__ == '__main__':
    main()
