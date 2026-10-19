"""Archive the exact MTGS forward inputs for an independent, bounded replay.

This is an explicit diagnostic export. CPU copies here are never used by the
production renderer. No Gaussian subsampling or image resizing is performed.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

ARRAYS = ('means', 'quats', 'scales', 'opacities', 'colors', 'viewmats', 'Ks')
KIND = 'mtgs_render_inputs_full_gaussians'
RENDER_KEYS = frozenset(('width', 'height', 'tile_size', 'near_plane', 'far_plane',
    'render_mode', 'rasterize_mode', 'radius_clip', 'eps2d'))


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(value):
    if not value.flags.c_contiguous:
        raise ValueError('Capture arrays must be C-contiguous')
    result = hashlib.sha256()
    if not value.size:
        return result.hexdigest()
    data = memoryview(value).cast('B')
    for start in range(0, len(data), 8 * 1024 * 1024):
        result.update(data[start:start + 8 * 1024 * 1024])
    return result.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_render(render):
    if set(render) != RENDER_KEYS:
        raise ValueError('Capture render settings must contain the exact supported fields')
    for name in ('width', 'height', 'tile_size'):
        if type(render[name]) is not int or render[name] < 1:
            raise ValueError('Positive integer render dimensions are required')
    if render['tile_size'] not in (16, 32):
        raise ValueError('Only tile sizes 16 and 32 have a tested kernel layout')
    if render['render_mode'] not in ('RGB', 'RGB+ED') or render['rasterize_mode'] not in ('classic', 'antialiased'):
        raise ValueError('Unsupported rendering mode')
    for name in ('near_plane', 'far_plane', 'radius_clip', 'eps2d'):
        if isinstance(render[name], bool) or not isinstance(render[name], (int, float)) or not math.isfinite(render[name]):
            raise ValueError('Finite numeric render parameters are required')
    if not 0 < render['near_plane'] < render['far_plane'] or render['radius_clip'] < 0 or render['eps2d'] <= 0:
        raise ValueError('Invalid projection or clipping settings')
    return dict(render)


def validate_arrays(inputs, render):
    validate_render(render)
    if set(inputs) != set(ARRAYS):
        raise ValueError('The exact seven forward arrays are required')
    means, views = inputs['means'], inputs['viewmats']
    if means.ndim != 2 or views.ndim != 3:
        raise ValueError('Invalid Gaussian or camera rank')
    n, c = len(means), len(views)
    if c < 1:
        raise ValueError('At least one camera is required')
    shapes = ((n, 3), (n, 4), (n, 3), (n,), (c, n, 3), (c, 4, 4), (c, 3, 3))
    for name, shape in zip(ARRAYS, shapes):
        value = inputs[name]
        if value.dtype != np.dtype('float32') or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError('Unexpected FP32/C-order input shape: ' + name)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite captured input: ' + name)
    if (np.any(inputs['scales'] <= 0) or np.any(inputs['opacities'] < 0)
            or np.any(inputs['opacities'] > 1) or np.any(np.linalg.norm(inputs['quats'].astype(np.float64), axis=-1) == 0)):
        raise ValueError('Captured Gaussian parameters are not finite activated values')
    return n, c


def validate_provenance(value, cameras):
    required = ('scene_id', 'frame_id', 'camera_names', 'checkpoints', 'source_commit', 'source_fingerprint_sha256')
    if not isinstance(value, dict) or any(name not in value for name in required):
        raise ValueError('Scene, frame, checkpoints and exact source provenance are required')
    if any(not isinstance(value[name], str) or not value[name] for name in ('scene_id', 'frame_id')):
        raise ValueError('Scene and frame identifiers must be nonempty strings')
    names = value['camera_names']
    if (not isinstance(names, list) or len(names) != cameras or len(set(names)) != cameras
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError('Each captured camera requires a unique name')
    if not re.fullmatch('[0-9a-f]{40}', str(value['source_commit'])) or not _sha(value['source_fingerprint_sha256']):
        raise ValueError('Pin the source commit and actual runtime source fingerprint')
    checkpoints = value['checkpoints']
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError('Checkpoint SHA256 provenance is required')
    for checkpoint in checkpoints:
        name = checkpoint.get('name', '')
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or not _sha(checkpoint.get('sha256'))):
            raise ValueError('Checkpoint provenance needs relative names and SHA256 digests')
    # Force a plain JSON object and reject nonfinite/unsupported metadata.
    return json.loads(json.dumps(value, allow_nan=False))


def read_capture_manifest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Capture manifest is not a bounded regular JSON file')
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('kind') != KIND:
        raise ValueError('Unsupported captured-input contract')
    render = validate_render(manifest['render'])
    shape = manifest['shape']
    if (set(shape) != {'gaussians', 'cameras'} or type(shape['gaussians']) is not int
            or shape['gaussians'] < 0 or type(shape['cameras']) is not int or shape['cameras'] < 1):
        raise ValueError('Invalid captured logical shape')
    validate_provenance(manifest['provenance'], shape['cameras'])
    archive = manifest['archive']
    if (archive.get('name') != 'inputs.npz' or not _sha(archive.get('sha256'))
            or type(archive.get('bytes')) is not int or archive['bytes'] < 1):
        raise ValueError('Invalid captured archive identity')
    return manifest, path.parent / archive['name']


def write_capture(output_dir, inputs, render, provenance):
    """Export exact tensors at the rasterization call boundary into a new dir.

    ``inputs`` uses the seven keyword names in ``ARRAYS``. Tensors may be on
    NPU, CUDA or CPU; export is explicit and performs an owning CPU snapshot.
    The caller must identify every checkpoint used for this Gaussian assembly.
    """
    if set(inputs) != set(ARRAYS):
        raise ValueError('The exact seven forward arrays are required')
    arrays = {}
    for name in ARRAYS:
        value = inputs[name]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        if np.asarray(value).dtype != np.dtype('float32'):
            raise ValueError('Capture preserves FP32 inputs without implicit casting: ' + name)
        arrays[name] = np.array(value, copy=True, order='C')
    n, c = validate_arrays(arrays, render)
    provenance = validate_provenance(provenance, c)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / 'inputs.npz'
    np.savez(archive, **arrays)
    manifest = dict(schema_version=1, kind=KIND, shape=dict(gaussians=n, cameras=c),
        archive=dict(name=archive.name, bytes=archive.stat().st_size, sha256=file_digest(archive)),
        arrays={name:dict(shape=list(value.shape), dtype='float32', sha256=array_digest(value)) for name,value in arrays.items()},
        render=dict(render), provenance=provenance,
        scope='exact forward input capture; no rendering, quality or performance acceptance')
    path = output / 'capture.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False);stream.write('\n')
    return path


def load_capture(path):
    manifest, archive = read_capture_manifest(path)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != manifest['archive']['bytes']:
        raise ValueError('Captured archive size or file type differs')
    if file_digest(archive) != manifest['archive']['sha256']:
        raise ValueError('Captured archive whole-file SHA256 differs')
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if (len(entries) != len(ARRAYS) or {item.filename for item in entries} != {name+'.npy' for name in ARRAYS}
                or any(item.is_dir() or item.flag_bits & 1 or item.compress_type!=zipfile.ZIP_STORED for item in entries)):
            raise ValueError('Captured NPZ contains unsupported or duplicate members')
        n,c=manifest['shape']['gaussians'],manifest['shape']['cameras']
        shapes=((n,3),(n,4),(n,3),(n,),(c,n,3),(c,4,4),(c,3,3))
        for name,expected in zip(ARRAYS,shapes):
            info=zipped.getinfo(name+'.npy')
            expected_bytes=math.prod(expected)*4
            if not expected_bytes+10<=info.file_size<=expected_bytes+65536:
                raise ValueError('Captured NPY member exceeds its declared complete shape')
            with zipped.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported captured NPY header version')
                if shape!=expected or fortran or dtype!=np.dtype('float32'):
                    raise ValueError('Captured NPY header disagrees with its complete FP32 shape')
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {name:saved[name] for name in ARRAYS}
    n, c = validate_arrays(arrays, manifest['render'])
    if manifest['shape'] != dict(gaussians=n, cameras=c) or set(manifest['arrays']) != set(ARRAYS):
        raise ValueError('Captured logical shape or array names differ')
    for name, value in arrays.items():
        if manifest['arrays'][name] != dict(shape=list(value.shape),dtype='float32',sha256=array_digest(value)):
            raise ValueError('Captured array shape or content differs: ' + name)
    return arrays, manifest
