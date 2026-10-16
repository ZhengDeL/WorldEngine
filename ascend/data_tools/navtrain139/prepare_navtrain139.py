"""Rebuild a genuine official-data NAVTRAIN139/NAVTEST22 validation subset.

CPU-only, cached original publisher inputs, no download or model execution.
The default invocation prints a plan. Run with --execute on the Linux server.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import pickle
import signal
import struct
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
INPUTS_SHA = 'a37413649ee37fee2971b44c9d983c8ab991aa5f7210edae1716464ec463886f'
REFERENCE_SHA = 'b50b11d4892c43d9c4cd04d8f687973a7a340ab714fb6cee87da91c7efa46956'
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'score')
CAMS = {'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0'}
FILES = ('nuplan_openscene_navtrain_log.pkl', 'pdm_8192_gt_cache_navtrain_log.pkl',
         'navtrain_log_anchors.yaml', 'worldengine_navtest_eval.pkl',
         'worldengine_navtest_eval_pdm.pkl', 'worldengine_navtest_eval.yaml')


def sha(path, canonical=False):
    digest = hashlib.sha256()
    if canonical:
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    else:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
                digest.update(block)
    return digest.hexdigest()


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False);stream.write('\n')


def text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def safe_parts(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Nonempty safe original relative path required')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Absolute or escaping source path refused')
    return path.parts


def payload_digest(value):
    """Exact nested values/order, including every original numeric array bit."""
    import numpy as np
    digest = hashlib.sha256()

    def walk(node):
        if isinstance(node, np.ndarray):
            digest.update(b'array\0' + node.dtype.str.encode() + repr(node.shape).encode())
            if node.dtype.hasobject:
                for item in node.flat:walk(item)
            else:digest.update(node.tobytes(order='C'))
        elif isinstance(node, np.generic):
            digest.update(b'numpy_scalar\0' + node.dtype.str.encode() + node.tobytes())
        elif isinstance(node, dict):
            digest.update(b'dict\0' + str(len(node)).encode())
            for key, item in node.items():walk(key);walk(item)
        elif isinstance(node, (list, tuple)):
            digest.update(type(node).__name__.encode() + str(len(node)).encode())
            for item in node:walk(item)
        elif isinstance(node, str):
            data = node.encode('utf-8');digest.update(b'str\0' + str(len(data)).encode() + b':' + data)
        elif isinstance(node, bytes):digest.update(b'bytes\0' + str(len(node)).encode() + b':' + node)
        elif isinstance(node, bool):digest.update(b'bool1' if node else b'bool0')
        elif isinstance(node, int):digest.update(b'int\0' + str(node).encode() + b';')
        elif isinstance(node, float):digest.update(b'float\0' + struct.pack('!d', node))
        elif node is None:digest.update(b'None\0')
        else:raise TypeError('Unsupported original metadata value; preserve failure: ' + type(node).__module__ + '.' + type(node).__name__)
    walk(value)
    return digest.hexdigest()


def dump_verified(path, value):
    before = payload_digest(value)
    with path.open('xb') as stream:pickle.dump(value, stream, protocol=4)
    with path.open('rb') as stream:restored = pickle.load(stream)
    if payload_digest(restored) != before:
        raise RuntimeError('Serialized original metadata/array values changed')
    return {'bytes': path.stat().st_size, 'sha256': sha(path), 'semantic_payload_sha256': before}


def read_infos(path):
    with path.open('rb') as stream:value = pickle.load(stream)
    infos = value['infos'] if isinstance(value, dict) else value
    if not isinstance(infos, list) or not infos:
        raise ValueError('Nonempty original annotation list required')
    return infos


def original_load_annotations(source, annotation, split):
    """Execute only the unchanged original metadata method, with real pickle IO.

    This is not the complete dataset constructor or pipeline. No SDK/device
    bridge, fake examples, model imports or replacement selection logic is used.
    """
    import yaml
    file = source / 'projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_nuplan.py'
    tree = ast.parse(file.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    methods = [node for cls in classes for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name == 'load_annotations']
    if len(methods) != 1:
        raise RuntimeError('Exact original metadata method could not be identified')

    def actual_pickle(path, file_format=None):
        if Path(path).resolve() != annotation.resolve() or file_format != 'pkl':
            raise RuntimeError('Original metadata IO requested an unexpected input')
        with annotation.open('rb') as stream:return pickle.load(stream)

    scope = {'yaml': yaml, 'mmcv': SimpleNamespace(load=actual_pickle),
             'logger': logging.getLogger('WorldEngine.real_subset_metadata')}
    module = ast.Module(body=[methods[0]], type_ignores=[])
    exec(compile(module, str(file), 'exec'), scope)
    state = SimpleNamespace(nav_filter_path=str(split))
    infos = scope['load_annotations'](state, str(annotation))
    return state, infos


def queue_rows(infos, tokens, allowed_log, portable):
    by_token = {text(row['token']): index for index, row in enumerate(infos)}
    if len(by_token) != len(infos):raise ValueError('Duplicate original frame token')
    queues = [];images = {};used = set()
    for token in tokens:
        if token not in by_token or by_token[token] < 3:
            raise ValueError('Selected original anchor has no three predecessors')
        final = by_token[token];indices = list(range(final - 3, final + 1))
        records = [infos[index] for index in indices]
        if (any(text(row['log_name']) != allowed_log for row in records)
                or len({text(row['log_token']) for row in records}) != 1):
            raise ValueError('Original four-frame queue crosses a log')
        timestamps = [float(row['timestamp']) for row in records]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError('Original queue timestamps must increase; do not fabricate missing frames')
        references = []
        for row in records:
            if set(row['cams']) != CAMS:raise ValueError('Eight original distinct camera channels required')
            used.add(text(row['token']));paths = set()
            for name, camera in row['cams'].items():
                original_parts = safe_parts(text(camera['data_path']))
                if original_parts.count(allowed_log) != 1 or original_parts.count(name) != 1:
                    # The legacy portable eval layout omits the camera directory;
                    # its original source key was retained explicitly beforehand.
                    if not portable:raise ValueError('Original annotation log/camera path identity ambiguous')
                basename = original_parts[-1]
                relative = allowed_log + '/' + basename if portable else text(camera['data_path'])
                key = (allowed_log, name, basename)
                if relative in images and images[relative] != key:raise ValueError('Two original camera sources alias one destination')
                images[relative] = key;paths.add(relative)
                references.append({'frame_token': text(row['token']), 'camera': name, 'data_path': relative})
            if len(paths) != 8:raise ValueError('Eight distinct original camera paths required')
        queues.append({'anchor': token, 'index': final, 'queue_indices': indices,
                       'queue_tokens': [text(row['token']) for row in records], 'camera_references': references})
    return queues, images, used


def selected_pdm(path, tokens):
    import numpy as np
    with path.open('rb') as stream:full = pickle.load(stream)
    if not isinstance(full, dict) or not set(tokens).issubset(full):
        raise ValueError('Missing genuine original selected PDM entry')
    selected = {token: full[token] for token in tokens}
    for entry in selected.values():
        for name in METRICS:
            array = entry.get(name)
            if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != (8192,) or not np.isfinite(array).all():
                raise ValueError('Original selected PDM must be finite FP32[8192]; no slicing/remapping allowed')
    return selected


def extract_images(archive_path, wanted, output, expected):
    """Safe whole-archive scan, exact log/channel/name joins, raw byte copies."""
    from PIL import Image
    by_source = {}
    for relative, key in wanted.items():
        safe_parts(relative)
        if key in by_source and by_source[key] != relative:raise ValueError('Duplicate original source image key')
        by_source[key] = relative
    copied = {};members = 0
    with tarfile.open(archive_path, 'r|gz') as stream:
        for member in stream:
            members += 1
            parts = safe_parts(member.name)
            if member.isdir():continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('Links and special archive members are forbidden, including unused entries')
            if not parts or not parts[-1].lower().endswith(('.jpg', '.jpeg')):continue
            key = tuple(parts[-3:]) if len(parts) >= 3 else ()
            if key not in by_source:continue
            relative = by_source[key];name = 'cameras/' + relative
            if relative in copied:raise ValueError('Duplicate exact original image in archive')
            reference = expected[name]
            if member.size != reference['bytes']:raise ValueError('Original publisher image size differs from measured reference')
            target = output / 'cameras' / relative;target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(output.resolve()):raise ValueError('Image output path escape')
            payload = stream.extractfile(member)
            if payload is None:raise ValueError('Original tar image stream absent')
            digest = hashlib.sha256();size = 0
            with target.open('xb') as destination:
                for block in iter(lambda: payload.read(1024 ** 2), b''):
                    size += len(block)
                    if size > member.size:raise ValueError('Oversized original tar image')
                    digest.update(block);destination.write(block)
            if size != member.size or digest.hexdigest() != reference['sha256']:
                raise ValueError('Original JPEG bytes/SHA differ; no re-encoding or replacement permitted')
            with Image.open(target) as image:
                image.load();shape = [image.height, image.width, 3]
                if image.format != 'JPEG' or image.mode != 'RGB' or shape != reference['decoded_shape']:
                    raise ValueError('Actual original JPEG decode/geometry differs')
            copied[relative] = {'path': name, 'bytes': size, 'sha256': digest.hexdigest(), 'decoded_shape': shape}
    if set(copied) != set(wanted):raise ValueError('Original archive lacks exact requested image sources')
    return list(copied.values()), members


def prepare(args, lock, reference):
    import yaml
    sources = dict(navtrain_annotations=args.navtrain_annotations, navtrain_pdm=args.navtrain_pdm,
                   navtest_annotations=args.navtest_annotations, navtest_pdm=args.navtest_pdm,
                   train_cameras=args.train_cameras, eval_cameras=args.eval_cameras)
    positions = {'navtrain_annotations': 0, 'navtrain_pdm': 1, 'navtest_annotations': 2,
                 'navtest_pdm': 3, 'train_cameras': 4, 'eval_cameras': 5}
    before = {}
    for name, path in sources.items():
        expected = lock['artifacts'][positions[name]];actual = identity(path)
        if actual != {'bytes': expected['bytes'], 'sha256': expected['publisher_LFS_SHA256']}:
            raise ValueError('Original complete publisher input bytes/SHA differ: ' + name)
        before[name] = actual
    split_paths = {name: args.source / row['path'] for name, row in lock['original_splits'].items()}
    for name, path in split_paths.items():
        if sha(path, True) != lock['original_splits'][name]['canonical_LF_SHA256']:
            raise ValueError('Original upstream split differs')
    source = args.source / lock['original_dataset_source']['path']
    if sha(source, True) != lock['original_dataset_source']['canonical_LF_SHA256']:
        raise ValueError('Original metadata load_annotations source differs')
    train_split = yaml.safe_load(split_paths['NAVTRAIN'].read_text());test_split = yaml.safe_load(split_paths['NAVTEST'].read_text())
    if (len(set(train_split['tokens'])) != 103288 or len(set(train_split['log_names'])) != 1192
            or len(set(test_split['tokens'])) != 12146 or reference['train_log'] not in train_split['log_names']):
        raise ValueError('Original full split cardinality or target log differs')
    args.output.mkdir(parents=True);metadata = {};semantic = {}
    state, infos = original_load_annotations(args.source, args.navtrain_annotations, split_paths['NAVTRAIN'])
    if len(infos) != 246165 or len(state.index_map) != 103288:
        raise ValueError('Original full training metadata coverage differs')
    indices = [index for index, info in enumerate(infos) if text(info['log_name']) == reference['train_log']]
    if len(indices) != 284 or indices != list(range(182722, 183006)):
        raise ValueError('Original complete target log adjacency differs')
    index_set = set(indices)
    selected = [text(infos[index]['token']) for index in state.index_map if index in index_set]
    if selected != reference['train_anchor_tokens'] or set(selected) & set(test_split['tokens']):
        raise ValueError('Original139 anchor order or full NAVTEST disjointness differs')
    train_infos = [infos[index] for index in indices]
    train_queues, train_images, train_frames = queue_rows(train_infos, selected, reference['train_log'], False)
    if len(train_images) != 1536 or len(train_frames) != 192:raise ValueError('Original complete four-frame training coverage differs')
    metadata[FILES[0]] = dump_verified(args.output / FILES[0], {'infos': train_infos})
    semantic['training_annotation'] = metadata[FILES[0]]['semantic_payload_sha256']
    train_values = selected_pdm(args.navtrain_pdm, selected)
    metadata[FILES[1]] = dump_verified(args.output / FILES[1], train_values)
    filtered = copy.deepcopy(train_split);filtered['log_names'] = [reference['train_log']];filtered['tokens'] = selected
    with (args.output / FILES[2]).open('x', newline='\n') as stream:stream.write(yaml.safe_dump(filtered, sort_keys=False))
    metadata[FILES[2]] = identity(args.output / FILES[2])
    del state, infos, train_values, train_infos;gc.collect()
    all_eval = read_infos(args.navtest_annotations)
    eval_infos = [copy.deepcopy(row) for row in all_eval if text(row['log_name']) == reference['eval_log']]
    if len(eval_infos) != 51:raise ValueError('Original full evaluation log differs')
    allowed = set(test_split['tokens']);selected_eval = reference['eval_anchor_tokens']
    if not set(selected_eval).issubset(allowed):raise ValueError('Reference evaluation anchors absent from original official NAVTEST')
    by_token = {text(row['token']): index for index, row in enumerate(eval_infos)}
    if [text(row['token']) for row in eval_infos if text(row['token']) in set(selected_eval)] != selected_eval:
        raise ValueError('Original22 evaluation anchor order differs')
    for token in selected_eval:
        index = by_token[token]
        if index < 3 or index + 8 >= len(eval_infos) or any(text(row['log_token']) != text(eval_infos[index]['log_token']) for row in eval_infos[index-3:index+9]):
            raise ValueError('Historical official evaluation queue/future eligibility differs')
    # Calculate exact original log/channel/name archive identities before the
    # documented portable data_path rewrite. Every other field is untouched.
    eval_queues, eval_images, eval_frames = queue_rows(eval_infos, selected_eval, reference['eval_log'], True)
    if len(eval_images) != 296 or len(eval_frames) != 37:raise ValueError('Original evaluation input coverage differs')
    for info in eval_infos:
        for name, camera in info['cams'].items():
            parts = safe_parts(text(camera['data_path']))
            if parts.count(reference['eval_log']) != 1 or parts.count(name) != 1:
                raise ValueError('Original evaluation log/channel path identity ambiguous')
            camera['data_path'] = reference['eval_log'] + '/' + parts[-1]
    eval_value = {'infos': eval_infos, 'metadata': {'source': 'official real subset', 'phase': 'eval'}}
    metadata[FILES[3]] = dump_verified(args.output / FILES[3], eval_value)
    eval_values = selected_pdm(args.navtest_pdm, selected_eval)
    metadata[FILES[4]] = dump_verified(args.output / FILES[4], eval_values)
    filtered_eval = copy.deepcopy(test_split);filtered_eval.pop('scenario_tokens', None)
    filtered_eval['tokens'] = selected_eval;filtered_eval['log_names'] = [reference['eval_log']]
    with (args.output / FILES[5]).open('x', newline='\n') as stream:stream.write(yaml.safe_dump(filtered_eval))
    metadata[FILES[5]] = identity(args.output / FILES[5])
    del all_eval, eval_infos, eval_values, eval_value;gc.collect()
    if set(selected) & set(selected_eval) or train_frames & eval_frames or set(train_images) & set(eval_images):
        raise ValueError('Training/evaluation data overlap')
    images, train_members = extract_images(args.train_cameras, train_images, args.output, reference['expected_images'])
    eval_copied, eval_members = extract_images(args.eval_cameras, eval_images, args.output, reference['expected_images']);images += eval_copied
    after = {name: identity(path) for name, path in sources.items()}
    if before != after or any(sha(path, True) != lock['original_splits'][name]['canonical_LF_SHA256'] for name, path in split_paths.items()) or sha(source, True) != lock['original_dataset_source']['canonical_LF_SHA256']:
        raise ValueError('Original input files or metadata code changed during preparation')
    legacy = {name: {'expected_sha256': row['sha256'], 'actual_sha256': metadata[name]['sha256'],
                     'byte_identical': metadata[name]['sha256'] == row['sha256']}
              for name, row in reference['original_runtime_six_derived_payloads'].items()}
    manifest = {'model': 'worldengine', 'dataset_scope': 'Deterministic genuine NAVTRAIN139 and independent NAVTEST22; not an official prepackaged mini or full benchmark',
                'trajectory_labels_changed': False, 'diffusiondrive_data_mode': False,
                'logs': [reference['train_log'], reference['eval_log']], 'annotation_frames': 335,
                'training_anchors': 139, 'heldout_samples': 22, 'camera_files': len(images), 'pdm_tokens': 161,
                'heldout_log': reference['eval_log'], 'train_eval_token_overlap': 0, 'train_eval_frame_overlap': 0,
                'split_by': 'Original disjoint NAVTRAIN/NAVTEST logs', 'portable_camera_paths': True, 'train_rare_tokens': [],
                'splits': {'train': {'annotations': FILES[0], 'filter': FILES[2], 'pdm_prefix': FILES[1][:-4], 'anchors': 139, 'frames': 284, 'logs': [reference['train_log']]},
                           'eval': {'annotations': FILES[3], 'filter': FILES[5], 'pdm_prefix': FILES[4][:-4], 'anchors': 22, 'frames': 51, 'logs': [reference['eval_log']]}},
                'file_hashes': {**{name: row['sha256'] for name, row in metadata.items()}, **{row['path']: row['sha256'] for row in images}},
                'preparation': {'helper_sha256': sha(Path(__file__)), 'input_lock_sha256': INPUTS_SHA, 'subset_reference_sha256': REFERENCE_SHA,
                                'original_input_before': before, 'original_input_after': after, 'metadata_semantic_roundtrip_verified': True,
                                'original_PDM_complete_entry_values_and_order_preserved': True, 'six_legacy_payload_byte_identity': legacy,
                                'new_output_must_be_independently_model_validated': True}}
    save(args.output / 'manifest.json', manifest)
    receipt = {'result': 'pass_actual_official_input_NAVTRAIN139_NAVTEST22_preparation_not_model_training',
               'source_original_metadata_AST_executed': True, 'complete_dataset_constructor_or_pipeline_executed': False,
               'source_original_metadata_method_file_LF_SHA256': sha(source, True),
               'train_anchor_queues': train_queues, 'eval_anchor_queues': eval_queues, 'copied_images': images,
               'metadata': metadata, 'legacy_payload_byte_identity': legacy, 'all_six_legacy_payloads_byte_identical': all(row['byte_identical'] for row in legacy.values()),
               'original_input_before': before, 'original_input_after': after, 'manifest_sha256': sha(args.output / 'manifest.json'),
               'tar_members_scanned': {'train': train_members, 'eval': eval_members},
               'no_manual_or_synthetic_labels': True, 'PDM_finite_FP32_arrays_checked': 966,
               'train_eval_token_overlap': 0, 'used_frame_overlap': 0, 'camera_reference_overlap': 0,
               'new_metadata_serialization_requires_own_acceptance': not all(row['byte_identical'] for row in legacy.values()),
               'Torch_imported': 'torch' in sys.modules, 'TorchNPU_imported': 'torch_npu' in sys.modules,
               'model_training_executed': False}
    save(args.output / 'preparation.json', receipt)
    return {'result': receipt['result'], 'manifest_sha256': receipt['manifest_sha256'], 'training_anchors': 139,
            'independent_eval_anchors': 22, 'copied_JPEGs': len(images),
            'legacy_metadata_byte_identity': receipt['all_six_legacy_payloads_byte_identical'],
            'PDM_original_candidate_values_and_order_preserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('source', 'navtrain-annotations', 'navtrain-pdm', 'navtest-annotations', 'navtest-pdm',
                 'train-cameras', 'eval-cameras', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', required=True);parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--execute', action='store_true');args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if cutoff.tzinfo is None or not 60 <= args.max_seconds <= 1800:
        parser.error('Explicit timezone and bounded60..1800 second operation required')
    if sha(ROOT / 'INPUTS.lock.json') != INPUTS_SHA or sha(ROOT / 'SUBSET_REFERENCE.json') != REFERENCE_SHA:
        raise ValueError('Published input/subset reference lock changed')
    lock = json.loads((ROOT / 'INPUTS.lock.json').read_text());reference = json.loads((ROOT / 'SUBSET_REFERENCE.json').read_text())
    plan = {'result': 'dry_run_no_input_load_no_download', 'official_input_artifacts': 6, 'official_input_bytes': sum(row['bytes'] for row in lock['artifacts']),
            'training_anchors': 139, 'independent_eval_anchors': 22, 'required_JPEGs': 1832,
            'scope': 'Cached original publisher inputs only; metadata method/queue/path preparation, no dataset pipeline or model'}
    if not args.execute:
        print(json.dumps(plan, indent=2));return 0
    if not sys.platform.startswith('linux') or any(name.startswith(('torch', 'mx_driving')) for name in sys.modules):
        parser.error('Fresh Linux CPU-only interpreter; binary data stays on the server')
    import resource
    if list(Path('/dev').glob('davinci*')) or any(Path(name).exists() for name in ('/dev/devmm_svm', '/dev/hisi_hdc')):
        parser.error('Use a CPU-only container without any accelerator devices')
    for name in ('source', 'navtrain_annotations', 'navtrain_pdm', 'navtest_annotations', 'navtest_pdm', 'train_cameras', 'eval_cameras'):
        path = getattr(args, name)
        if path.is_symlink():parser.error('Pass exact original resolved source/input paths')
        setattr(args, name, path.resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists() or args.output.is_relative_to(args.source):parser.error('Fresh separate output required')
    if any(args.output.is_relative_to(getattr(args, name).parent) for name in ('navtrain_annotations', 'navtrain_pdm', 'navtest_annotations', 'navtest_pdm', 'train_cameras', 'eval_cameras')):
        parser.error('Output must be separate from immutable official input directories')
    if not args.output.parent.is_dir():parser.error('Existing dedicated parent directory required')
    seconds = min(args.max_seconds, cutoff.timestamp() - time.time())
    if seconds < 60:parser.error('Insufficient preparation window')
    def stop(*_):raise TimeoutError('Whole CPU data-preparation cutoff')
    previous = signal.signal(signal.SIGALRM, stop);signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.time()
    try:
        report = prepare(args, lock, reference)
        if any(name.startswith(('torch', 'mx_driving')) for name in sys.modules):raise RuntimeError('Data preparation imported an accelerator framework')
        code = 0
    except BaseException as error:
        report = {'result': 'failed_or_incomplete_original_data_preparation_evidence_preserved', 'error': repr(error), 'traceback': traceback.format_exc()};code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0);signal.signal(signal.SIGALRM, previous)
    report.update(elapsed_seconds=time.time() - started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  finished_UTC=datetime.now(timezone.utc).isoformat(), helper_sha256=sha(Path(__file__)))
    if args.output.is_dir():save(args.output / 'result.json', report)
    print(json.dumps(report, indent=2));return code


if __name__ == '__main__':
    raise SystemExit(main())
