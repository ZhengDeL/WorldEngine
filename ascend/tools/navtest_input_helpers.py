"""Shared CPU input algorithms extracted from the public inspector.

No train/eval selection, CLI, model import or device initialization is included.
The caller selects and certifies NAVTEST independently.
"""
import hashlib, math, os, sys
from pathlib import Path

class Paused(Exception): pass
class Rejected(Exception): pass


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
    paths = sorted(list((source/'projects/AlgEngine').rglob('*.py')) + list((source/'ascend').glob('*.py')),
                   key=lambda path:path.relative_to(source).as_posix())
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
                raise Rejected('Original NAVTEST cameras require 1920x1080 RGB JPEGs')
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
