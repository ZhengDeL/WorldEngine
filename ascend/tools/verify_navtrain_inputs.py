"""Train-only, full NAVTRAIN reference/PDM/JPEG checks in a CPU-only container.

Uses the selected checkout's unmodified ascend.preflight.inspect_phase. It does
not create a dataset/model, execute training, score NAVTEST, or claim convergence.
The public CLI preserves the earlier full-data inspector's algorithms, adds
explicit UTC deadlines and enforces original 1920x1080 RGB JPEG dimensions.
Historical full-data results identify their original worker hash separately.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import warnings


class Paused(Exception):
    pass


class Rejected(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_UTC_deadline(value):
    try:
        deadline = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, ValueError) as error:
        raise ValueError('A timezone-aware UTC deadline is required') from error
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('The deadline must have an explicit UTC timezone')
    return deadline


class Guard:
    def __init__(self, deadline, stop_file=None):
        self.deadline = parse_UTC_deadline(deadline).timestamp()
        self.stop_file = stop_file
        self.event = threading.Event()
        self.reason = None

    def stop(self, reason):
        self.reason = reason
        self.event.set()

    def check(self):
        if time.time() >= self.deadline:
            self.stop('authorization_deadline')
        elif self.stop_file and self.stop_file.exists():
            self.stop('stop_file')
        if self.event.is_set():
            raise Paused(self.reason or 'stopped')


def memory():
    if sys.platform != 'linux':
        return dict(rss_bytes=None, peak_rss_bytes=None)
    import resource
    return dict(rss_bytes=int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE'),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def atomic(path, report):
    temporary = path.with_name(path.name + '.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise Rejected('Output metadata must not be symlinked')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest_file(path, guard):
    guard.check()
    if not path.is_file():
        raise Rejected('Required regular file absent: ' + str(path))
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            guard.check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise Rejected('Input changed while hashing: ' + str(path))
    return dict(bytes=size, sha256=digest.hexdigest())


def source_identity(source, guard):
    paths = sorted(list((source/'projects/AlgEngine').rglob('*.py')) + list((source/'ascend').glob('*.py')))
    if not paths:
        raise Rejected('Runtime source inventory is empty')
    digest = hashlib.sha256()
    for path in paths:
        guard.check()
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return dict(sha256=digest.hexdigest(), python_files=len(paths),
                scope='projects/AlgEngine/**/*.py + ascend/*.py; path then canonical-LF bytes, matching Linux training contract')


def validate_inspection(result, errors):
    if not isinstance(result, tuple) or len(result) != 3:
        raise Rejected('Original inspect_phase did not return (phase, present_tokens, logs); inputs may be missing')
    phase, present, logs = result
    if not isinstance(phase, dict) or not isinstance(present, set) or not isinstance(logs, set):
        raise Rejected('Unexpected original inspect_phase result types')
    required = ('requested_anchors', 'selected_anchors', 'missing_annotation_tokens', 'history_frames',
                'camera_files', 'missing_camera_files', 'camera_bytes', 'pdm_tokens',
                'missing_pdm_tokens', 'invalid_pdm_rows')
    if any(key not in phase for key in required):
        raise Rejected('Original inspect_phase omitted required train/PDM counters')
    if errors:
        raise Rejected('Original inspect_phase reported %d input errors' % len(errors))
    if phase['missing_annotation_tokens'] or phase['missing_camera_files'] or phase['missing_pdm_tokens'] or phase['invalid_pdm_rows']:
        raise Rejected('Original inspect_phase reported incomplete or nonfinite inputs')
    if phase['requested_anchors'] != phase['selected_anchors'] or len(present) != phase['selected_anchors']:
        raise Rejected('Selected / requested anchor counts differ')
    return phase, present, logs


def referenced_images(infos, requested, image_root, queue_length, guard):
    """Reconstruct the original index queue and independently join its camera set."""
    needed, selected = set(), 0
    for index, info in enumerate(infos):
        if index % 2048 == 0:
            guard.check()
        if info['token'] not in requested:
            continue
        selected += 1
        first = index - queue_length + 1
        if first < 0 or any(infos[j]['log_token'] != info['log_token'] for j in range(first, index+1)):
            raise Rejected('Selected anchor lacks its original same-log predecessor queue')
        times = [float(infos[j]['timestamp']) for j in range(first,index+1)]
        if not all(math.isfinite(value) for value in times) or any(right <= left for left,right in zip(times,times[1:])):
            raise Rejected('Original queue timestamps do not strictly increase')
        needed.update(range(first, index+1))
    references = {}
    reference_occurrences = 0
    expected_cameras = {'CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0'}
    for index in sorted(needed):
        guard.check()
        cameras = infos[index].get('cams', {})
        if set(cameras) != expected_cameras:
            raise Rejected('Original queue frame does not contain the exact eight camera channels')
        frame_paths = set()
        for camera, value in cameras.items():
            relative = Path(value['data_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise Rejected('Camera reference is not a safe relative image-root path')
            if len(relative.parts) != 3 or relative.parts[0] != str(infos[index]['log_name']) or relative.parts[1] != camera:
                raise Rejected('Original log/camera path identity differs')
            image = image_root / relative
            key = str(image)
            normalized = relative.as_posix()
            previous = references.get(key)
            if previous is not None and previous != normalized:
                raise Rejected('Ambiguous normalized image reference')
            references[key] = normalized
            frame_paths.add(normalized)
            reference_occurrences += 1
        if len(frame_paths) != 8:
            raise Rejected('Eight camera channels do not reference eight distinct files')
    return references, dict(selected_anchors=selected, queue_frames=len(needed),
                            camera_reference_occurrences=reference_occurrences)


def decode_jpeg(path, relative, guard):
    from PIL import Image, ImageFile
    guard.check()
    row = dict(relative=relative, verified=False, loaded=False, result='error')
    try:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise Rejected('Pillow truncated-image acceptance must remain disabled')
        before = path.stat()
        row['bytes'] = before.st_size
        if not path.is_file() or before.st_size <= 0:
            raise Rejected('Missing or empty regular JPEG')
        content = digest_file(path, guard)
        row['sha256'] = content['sha256']
        with Image.open(path) as image:
            guard.check()
            if image.format != 'JPEG':
                raise Rejected('Camera file is not a JPEG')
            if image.size != (1920, 1080) or image.mode != 'RGB':
                raise Rejected('Original NAVTRAIN cameras require 1920x1080 RGB JPEGs')
            image.verify()
            row['verified'] = True
        guard.check()
        with Image.open(path) as image:
            if image.format != 'JPEG':
                raise Rejected('Reopened camera file is not a JPEG')
            image.load()
            row.update(loaded=True, width=image.width, height=image.height, mode=image.mode)
        guard.check()
        after = path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise Rejected('Camera file changed during hash / verify / load')
        if digest_file(path, guard) != content:
            raise Rejected('Camera content changed during hash / verify / load')
        row['result'] = 'pass'
    except Paused:
        raise
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error))
    return row


def accelerator_state():
    nodes = sorted(str(path) for pattern in ('davinci*', 'nvidia*')
                   for path in Path('/dev').glob(pattern))
    nodes += [str(path) for path in (Path('/dev/devmm_svm'),Path('/dev/hisi_hdc')) if path.exists()]
    torch, npu = sys.modules.get('torch'), sys.modules.get('torch_npu')
    return dict(device_nodes=nodes,
                CUDA_initialized=bool(torch is not None and torch.cuda.is_initialized()),
                NPU_initialized=bool(npu is not None and npu.npu.is_initialized()))


def validate_CPU_only(state):
    if state['device_nodes'] or state['CUDA_initialized'] or state['NPU_initialized']:
        raise Rejected('A CPU-only container without accelerator device nodes is required')


def run(args):
    if sys.platform != 'linux':
        raise Rejected('Full-data inspector requires Linux /proc/resource; local helper fixtures are separate')
    source, data = args.source.resolve(strict=True), args.data_root.resolve(strict=True)
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise Rejected('Use a fresh QC output file')
    image_list = output.with_name(output.name + '.images.jsonl')
    image_part = image_list.with_name(image_list.name + '.part')
    if image_list.exists() or image_part.exists() or image_part.is_symlink():
        raise Rejected('Use a fresh image audit output')
    with output.open('x', encoding='utf8') as stream:
        stream.write('{}\n')
    guard = Guard(args.deadline_utc, args.stop_file)
    report = dict(schema_version=1, result='running_CPU_only', started_utc=utc(), errors=[],
                  scope='Original full NAVTRAIN100 train-only inputs and every required JPEG; no NAVTEST, model, training or convergence',
                  source=str(source), data_root=str(data), worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  no_accelerator_execution=True, convergence_accepted=False, full_training_memory_accepted=False,
                  decode_workers=args.decode_workers, decode_queue_bound=args.decode_workers*2,
                  decoded=0, decode_passed=0, decode_failed=0, failure_examples=[],
                  total_decoded_file_bytes=0, image_audit=str(image_list), image_audit_complete=False)
    retain, pool = [], None
    started = time.monotonic()
    original_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGTERM, lambda *_: guard.stop('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: guard.stop('SIGINT'))
    signal.signal(signal.SIGALRM, lambda *_: guard.check())
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    exit_code = 1
    try:
        guard.check()
        os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
        if os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] != '0':
            raise Rejected('TORCH_DEVICE_BACKEND_AUTOLOAD must be 0')
        report['accelerator_before'] = accelerator_state()
        validate_CPU_only(report['accelerator_before'])
        preflight_path = source/'ascend/preflight.py'
        report['source_fingerprint_before'] = source_identity(source, guard)
        report['preflight_file'] = digest_file(preflight_path, guard)
        spec = importlib.util.spec_from_file_location('worldengine_official_preflight_CPU', preflight_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.environ['WORLDENGINE_ROOT'] = str(source)
        config = args.config
        if not config.is_absolute():
            config = source/config if (source/config).is_file() else source/'projects/AlgEngine'/config
        config = config.resolve(strict=True)
        if not config.is_relative_to(source/'projects/AlgEngine'):
            raise Rejected('Config must belong to the selected AlgEngine checkout')
        cfg = module.Config.fromfile(str(config), import_custom_modules=False)
        ds = cfg.data.train
        if any(name in ds.type for name in ('Synthetic', 'SimScale')):
            raise Rejected('Synthetic configs are outside this NAVTRAIN-only inspection')
        if int(ds.get('queue_length', 4)) != 4 or int(ds.get('load_interval', 1)) != 1:
            raise Rejected('Original current+3-history, load_interval=1 config is required')
        split = Path(ds.nav_filter_path)
        if not split.is_absolute():
            split = source/'projects/AlgEngine'/split
        split = split.resolve(strict=True)
        expected_split = (source/'projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain.yaml').resolve(strict=True)
        if split != expected_split:
            raise Rejected('The original full 100% NAVTRAIN filter is required')
        annotations = data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl'
        pdm = data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl'
        vocab = data/'alg_engine/test_8192_kmeans.npy'
        image_root = data/'raw/openscene-v1.1/sensor_blobs/trainval'
        asset_lock_path = source/'ascend/assets.lock.json'
        asset_lock = json.loads(asset_lock_path.read_text())
        rows = {row['path']: row for row in asset_lock['assets']}
        report.update(config=str(config), config_file=digest_file(config, guard), dataset_type=ds.type,
                      filter=str(split), filter_file=digest_file(split, guard),
                      asset_lock_file=digest_file(asset_lock_path, guard), input_files_before={})
        for path in (annotations, pdm, vocab):
            relative = 'data/' + path.relative_to(data).as_posix()
            expected = rows[relative]
            actual = digest_file(path, guard)
            report['input_files_before'][relative] = actual
            if actual != dict(bytes=expected['bytes'], sha256=expected['sha256']):
                raise Rejected('Publisher asset checksum / size mismatch: ' + relative)
        split_lock_path = source/'ascend/data_tools/navtrain139/INPUTS.lock.json'
        report['split_lock_file'] = digest_file(split_lock_path, guard)
        split_lock = json.loads(split_lock_path.read_text())['original_splits']['NAVTRAIN']
        if hashlib.sha256(split.read_bytes().replace(b'\r\n',b'\n')).hexdigest() != split_lock['canonical_LF_SHA256']:
            raise Rejected('Original full NAVTRAIN filter differs from its published lock')
        report['expected_original_anchors'] = split_lock['tokens']
        atomic(output, dict(report, stage='original_inspect_phase', **memory()))
        inspected = module.inspect_phase(ds, annotations, split, image_root, pdm, report['errors'], retain,
                                         include_test_history=False)
        report['original_phase'] = inspected[0] if isinstance(inspected, tuple) else inspected
        phase, present, logs = validate_inspection(inspected, report['errors'])
        if phase['requested_anchors'] != split_lock['tokens']:
            raise Rejected('Full NAVTRAIN anchor count differs from its original lock')
        if not retain:
            raise Rejected('Original inspector did not retain the annotation object')
        infos = retain[0]['infos'] if isinstance(retain[0],dict) else retain[0]
        references, counts = referenced_images(infos, present, image_root, 4, guard)
        if counts['selected_anchors'] != phase['selected_anchors'] or counts['queue_frames'] != phase['history_frames'] or len(references) != phase['camera_files']:
            raise Rejected('Independent index/camera reference counts differ from the original inspector')
        report.update(unique_required_JPEGs=len(references), selected_logs=len(logs), reference_counts=counts,
                      expected_PDM_arrays=len(present)*len(module.PDM_KEYS), original_memory_after_inspection=memory())
        from PIL import Image, ImageFile
        import PIL
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        Image.MAX_IMAGE_PIXELS = args.max_image_pixels
        warnings.filterwarnings('error', category=Image.DecompressionBombWarning)
        report.update(Pillow_version=PIL.__version__, max_image_pixels=args.max_image_pixels,
                      allow_truncated_images=False, required_JPEG_geometry=[1920,1080],
                      required_JPEG_mode='RGB', stage='decoding_all_required_JPEGs')
        atomic(output, dict(report, **memory()))
        pool = ThreadPoolExecutor(max_workers=args.decode_workers)
        pending = deque()
        ordered = iter(sorted(references.items(), key=lambda item:item[1]))
        digest, modes, geometries = hashlib.sha256(), Counter(), Counter()
        list_digest = hashlib.sha256()
        last_progress = time.monotonic()
        with image_part.open('x', encoding='utf8') as ledger:
            while True:
                guard.check()
                while len(pending) < args.decode_workers*2:
                    try:
                        path, relative = next(ordered)
                    except StopIteration:
                        break
                    pending.append(pool.submit(decode_jpeg, Path(path), relative, guard))
                if not pending:
                    break
                row = pending.popleft().result()
                guard.check()
                encoded = json.dumps(row, sort_keys=True, allow_nan=False)+'\n'
                ledger.write(encoded)
                list_digest.update(encoded.encode())
                report['decoded'] += 1
                report['total_decoded_file_bytes'] += row.get('bytes',0)
                if row['result'] == 'pass':
                    report['decode_passed'] += 1
                    modes[row['mode']] += 1
                    geometries['%dx%d' % (row['width'],row['height'])] += 1
                else:
                    report['decode_failed'] += 1
                    if len(report['failure_examples']) < 20:
                        report['failure_examples'].append(row)
                digest.update((row['relative']+'\0'+str(row.get('bytes'))+'\0'+str(row.get('sha256'))+'\0'+row['result']+'\n').encode())
                if time.monotonic()-last_progress > 30:
                    ledger.flush()
                    atomic(output, dict(report, image_content_inventory_sha256=digest.hexdigest(), **memory()))
                    last_progress = time.monotonic()
            ledger.flush()
            os.fsync(ledger.fileno())
        pool.shutdown(wait=True)
        pool = None
        if report['decoded'] != len(references):
            raise Rejected('JPEG decode audit did not visit every unique reference')
        if report['total_decoded_file_bytes'] != phase['camera_bytes']:
            raise Rejected('Decoded file byte count differs from original preflight')
        os.replace(image_part, image_list)
        report.update(image_audit_complete=True, image_audit_sha256=list_digest.hexdigest(),
                      image_content_inventory_sha256=digest.hexdigest(), decoded_modes=dict(modes),
                      decoded_geometries=dict(geometries), input_files_after={})
        for path in (annotations,pdm,vocab):
            relative = 'data/'+path.relative_to(data).as_posix()
            report['input_files_after'][relative] = digest_file(path,guard)
        report['source_fingerprint_after'] = source_identity(source,guard)
        if report['input_files_after'] != report['input_files_before'] or report['source_fingerprint_after'] != report['source_fingerprint_before']:
            raise Rejected('Source or locked assets changed during inspection')
        if digest_file(config,guard) != report['config_file'] or digest_file(split,guard) != report['filter_file']:
            raise Rejected('Config or original filter changed during inspection')
        for path,field in ((preflight_path,'preflight_file'),(asset_lock_path,'asset_lock_file'),(split_lock_path,'split_lock_file')):
            if digest_file(path,guard) != report[field]:
                raise Rejected('Inspector or publisher lock changed during inspection')
        if report['decode_failed']:
            report['errors'].append('%d required JPEGs failed strict verify/reopen/load' % report['decode_failed'])
            raise Rejected('Required JPEG decode failures')
        report.update(result='pass_complete_original_NAVTRAIN_inputs_and_required_JPEG_decode_CPU_only', stage='complete')
        exit_code = 0
    except Paused as error:
        report.update(result='paused_incomplete_CPU_inspection', reason=str(error))
        exit_code = 2
    except Exception as error:
        report.update(result='blocked_or_rejected_CPU_inspection', error_type=type(error).__name__, reason=str(error))
        exit_code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_alarm)
        guard.stop('CPU_inspection_finished')
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        report.update(completed_utc=utc(), elapsed_seconds=time.monotonic()-started,
                      CPU_memory=memory(), accelerator_after=accelerator_state(), actual_exit=exit_code)
        if report['accelerator_after']['device_nodes'] or report['accelerator_after']['CUDA_initialized'] or report['accelerator_after']['NPU_initialized']:
            report.update(result='rejected_non_CPU_execution', no_accelerator_execution=False, actual_exit=1)
            exit_code = 1
        atomic(output, report)
        print(json.dumps(dict(result=report['result'], actual_exit=exit_code, output=str(output),
                              decoded=report['decoded'], decode_failed=report['decode_failed'])), flush=True)
    return exit_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True, help='Future timezone-aware UTC cutoff; no fixed date cap')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--decode-workers', type=int, default=4)
    parser.add_argument('--max-image-pixels', type=int, default=20000000)
    args = parser.parse_args(argv)
    try:
        deadline = parse_UTC_deadline(args.deadline_utc)
    except ValueError as error:
        parser.error(str(error))
    if deadline.timestamp() <= time.time():
        parser.error('The UTC deadline must be in the future')
    if not 1 <= args.decode_workers <= 4 or args.max_image_pixels < 1:
        parser.error('At most four decode workers and a positive pixel limit are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(json.dumps(dict(result='rejected_before_inspection', reason=repr(error))), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
