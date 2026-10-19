"""Bounded real-input validation leases and linked acceptance evidence.

A lease authorizes a diagnostic execution, not a passing device result. It is
an explicit API unavailable to the ordinary MTGS configuration. Production
expansion requires both the current small proof and a complete, linked large
forward report with an independent sparse-pixel reference.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import numpy as np

from .capture import file_digest, read_capture_manifest
from .manifest import IDENTITY_FIELDS, digest, validate_manifest

EVIDENCE_TYPE = 'real_full_input_sparse_pixel_reference'
REAL_SOURCES = ('capture.py', 'real_validation.py', 'tests/validate_real_input.py', 'tests/sampled_oracle.py')
REAL_IDENTITY_FIELDS = IDENTITY_FIELDS + ('evidence_type', 'real_sources_sha256',
    'small_proof', 'capture_binding', 'validated_render_settings')


def sources_identity():
    root = Path(__file__).parent
    return {name:digest(root/name) for name in REAL_SOURCES}


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Expected a bounded regular evidence JSON file')
    return json.loads(path.read_text())


def _small(path, extension_path, versions):
    value = _read(path)
    if value.get('evidence_type') is not None:
        raise ValueError('A small synthetic proof must be the direct base of real-input validation')
    return validate_manifest(path, extension_path, versions)


def capture_binding(path):
    path = Path(path).resolve(strict=True)
    manifest, archive = read_capture_manifest(path)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != manifest['archive']['bytes']:
        raise ValueError('The exact captured archive is unavailable')
    if file_digest(archive) != manifest['archive']['sha256']:
        raise ValueError('The captured archive changed')
    return dict(manifest_path=str(path), manifest_sha256=digest(path),
        archive_sha256=manifest['archive']['sha256'], archive_bytes=manifest['archive']['bytes'],
        shape=manifest['shape'], render=manifest['render'], provenance=manifest['provenance'])


def create_lease(small_path, capture_path, deadline_utc, extension_path, versions):
    deadline = datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        raise ValueError('The validation lease requires a future, explicit UTC deadline')
    small_path = Path(small_path).resolve(strict=True)
    _small(small_path, extension_path, versions)
    binding = capture_binding(capture_path)
    if binding['shape']['gaussians'] < 1:
        raise ValueError('A real-input expansion requires a nonempty Gaussian assembly')
    return dict(schema_version=1, kind='bounded_real_input_validation_only',
        issued_utc=datetime.now(timezone.utc).isoformat(), deadline_utc=deadline.isoformat(),
        small_proof=dict(path=str(small_path), sha256=digest(small_path)),
        capture_binding=binding, real_sources_sha256=sources_identity(),
        scope='diagnostic lease only; not a production compatibility manifest')


def validate_lease(path, extension_path, versions):
    lease = _read(path)
    if lease.get('schema_version') != 1 or lease.get('kind') != 'bounded_real_input_validation_only':
        raise ValueError('Unsupported real-input validation lease')
    deadline = datetime.fromisoformat(lease['deadline_utc'].replace('Z', '+00:00'))
    issued = datetime.fromisoformat(lease['issued_utc'].replace('Z', '+00:00'))
    if (deadline.tzinfo is None or issued.tzinfo is None or issued.timestamp() > time.time() + 5
            or deadline.timestamp() <= time.time() or issued.timestamp() >= deadline.timestamp()):
        raise ValueError('The real-input validation lease is not currently valid')
    proof = lease['small_proof']
    if digest(proof['path']) != proof['sha256']:
        raise ValueError('The small proof manifest changed')
    small = _small(proof['path'], extension_path, versions)
    binding = capture_binding(lease['capture_binding']['manifest_path'])
    if binding != lease['capture_binding'] or lease.get('real_sources_sha256') != sources_identity():
        raise ValueError('The real input or validation implementation changed')
    candidate = expanded_identity(small, proof, binding)
    return candidate, deadline.timestamp()


def expanded_identity(small, proof, binding):
    candidate = {name:small[name] for name in IDENTITY_FIELDS}
    shape, render = binding['shape'], binding['render']
    if shape['gaussians']<1:
        raise ValueError('A real-input expansion requires a nonempty complete Gaussian assembly')
    provenance=binding['provenance']
    if (provenance.get('input_origin')!='official_mtgs_scene'
            or re.fullmatch('[0-9a-f]{64}',str(provenance.get('official_asset_manifest_sha256',''))) is None):
        raise ValueError('Real-input evidence requires declared official-scene origin and its asset manifest SHA256')
    candidate.update(evidence_type=EVIDENCE_TYPE, real_sources_sha256=sources_identity(),
        small_proof=proof, capture_binding=binding, validated_render_settings=render,
        validated_limits=dict(max_gaussians=shape['gaussians'], max_cameras=shape['cameras'],
                              max_pixels=render['width']*render['height']))
    return candidate


def validate_real_manifest(path, extension_path, versions):
    manifest = _read(path)
    if manifest.get('evidence_type') != EVIDENCE_TYPE:
        raise ValueError('Unsupported real-input evidence type')
    proof = manifest['small_proof']
    if digest(proof['path']) != proof['sha256']:
        raise ValueError('The small proof manifest changed')
    small = _small(proof['path'], extension_path, versions)
    binding = capture_binding(manifest['capture_binding']['manifest_path'])
    expected = expanded_identity(small, proof, binding)
    if any(manifest.get(name) != expected[name] for name in REAL_IDENTITY_FIELDS):
        raise ValueError('The expanded manifest is not bound to this proof, input, implementation and shape')
    validation = manifest.get('validation', {})
    if validation.get('result') != 'pass' or digest(validation['evidence_path']) != validation.get('evidence_sha256'):
        raise ValueError('Passing real-input evidence is missing or changed')
    evidence = _read(validation['evidence_path'])
    if (evidence.get('result') != 'pass' or evidence.get('evidence_type') != EVIDENCE_TYPE
            or evidence.get('identity') != {name:manifest[name] for name in REAL_IDENTITY_FIELDS}
            or evidence.get('all_forward_outputs_finite') is not True
            or evidence.get('sampled_reference_passed') is not True
            or type(evidence.get('reference_pixel_count')) is not int or evidence['reference_pixel_count'] < 1):
        raise ValueError('The real-input report does not establish the required bounded forward check')
    render=manifest['validated_render_settings'];shape=binding['shape']
    if render['render_mode']!='RGB+ED':
        raise ValueError('A real-input expansion requires RGB, alpha and expected-depth reference checks')
    pixels=np.asarray(evidence.get('reference_pixels',[]))
    count=evidence['reference_pixel_count']
    if (pixels.shape!=(count,3) or pixels.dtype.kind not in ('i','u')
            or np.any(pixels<0) or np.any(pixels[:,0]>=shape['cameras'])
            or np.any(pixels[:,1]>=render['height']) or np.any(pixels[:,2]>=render['width'])
            or len(set(map(tuple,pixels.tolist())))!=count
            or len(set(pixels[:,0].tolist()))!=shape['cameras']):
        raise ValueError('The reference pixel set is invalid or omits captured cameras')
    expected_shapes=dict(rgb=[shape['cameras'],render['height'],render['width'],4],
        alpha=[shape['cameras'],render['height'],render['width'],1],radii=[shape['cameras'],shape['gaussians']])
    if evidence.get('full_output_shapes')!=expected_shapes:
        raise ValueError('The finite check did not cover the captured complete forward shape')
    statistics=evidence.get('reference_statistics',{})
    cameras=statistics.get('camera_statistics',[])
    if (statistics.get('gaussians')!=shape['gaussians'] or statistics.get('cameras')!=shape['cameras']
            or statistics.get('sample_pixels')!=count
            or statistics.get('all_gaussians_projected_per_camera') is not True
            or statistics.get('gaussian_subsampling') is not False
            or statistics.get('candidate_truncation') is not False
            or len(cameras)!=shape['cameras']
            or {row.get('camera') for row in cameras}!=set(range(shape['cameras']))
            or any(row.get('gaussians_projected')!=shape['gaussians'] for row in cameras)):
        raise ValueError('The independent reference did not account for every Gaussian and camera without truncation')
    samples=evidence.get('samples',{})
    required=(('actual_rgb',4),('actual_alpha',1),('reference_rgb',3),
              ('reference_alpha',1),('reference_expected_depth',1))
    arrays={name:np.asarray(samples.get(name,[]),dtype=np.float64) for name,_ in required}
    if any(arrays[name].shape!=(count,channels) or not np.isfinite(arrays[name]).all() for name,channels in required):
        raise ValueError('The numerical reference report is incomplete or nonfinite')
    if (not np.allclose(arrays['actual_rgb'][:,:3],arrays['reference_rgb'],atol=3e-4,rtol=1e-3)
            or not np.allclose(arrays['actual_alpha'],arrays['reference_alpha'],atol=3e-4,rtol=1e-3)
            or not np.allclose(arrays['actual_rgb'][:,3:],arrays['reference_expected_depth'],atol=2e-3,rtol=1e-3)):
        raise ValueError('The linked real-input samples exceed the fixed numerical tolerance')
    return manifest


def check_render_contract(expected, *, width, height, tile_size, near_plane, far_plane,
                          render_mode, rasterize_mode, radius_clip, eps2d):
    actual = dict(width=width, height=height, tile_size=tile_size, near_plane=near_plane,
        far_plane=far_plane, render_mode=render_mode, rasterize_mode=rasterize_mode,
        radius_clip=radius_clip, eps2d=eps2d)
    if actual != expected:
        raise ValueError('This real-input envelope was validated for different render settings or image dimensions')
