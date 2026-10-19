"""Validation of a specific compatibility build, without importing accelerator code."""
import hashlib
import json
from pathlib import Path

CONTRACT = 'mtgs-gsplat14-forward-v1'
CANONICAL_COMMIT = '8c0445ebcca91514fce3a7050241f41ed132c340'
REQUIRED_CASES = frozenset((
    'pixel_center', 'alpha_threshold', 'exclusive_transmittance', 'empty_tiles',
    'depth_expected', 'per_camera_rgb', 'near_far_radius', 'antialias_projection',
    'offcenter_projection', 'sh_degree_0_4',
    'rotated_gaussians_and_cameras',
    'zero_gaussians', 'tile32_nonsquare_partial_tiles',
    'covariance_nonmultiple_of_eight',
))
IDENTITY_FIELDS = ('contract','canonical_commit','runtime_versions','runtime_files',
                   'adapter_sha256','validation_sources_sha256','compatibility_kernel_sha256','validated_limits')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_manifest(path, extension_path, versions, validation_only=False):
    if path is None:
        raise ValueError('An explicit NPU compatibility manifest is required')
    manifest=json.loads(Path(path).read_text())
    if manifest.get('evidence_type') is not None:
        if validation_only:
            raise ValueError('The small synthetic probe requires its own direct build manifest')
        from .real_validation import validate_real_manifest
        return validate_real_manifest(path,extension_path,versions)
    if (manifest.get('contract') != CONTRACT
            or manifest.get('canonical_commit') != CANONICAL_COMMIT
            or manifest.get('runtime_versions') != versions):
        raise ValueError('CANN compatibility contract, revision or runtime versions differ')
    extension=Path(extension_path).resolve(strict=True)
    package=extension.parent
    opp=package/'packages/vendors/customize'
    expected_api=(opp/'op_api/lib/libcust_opapi.so').resolve(strict=True)
    roles=set()
    files=manifest.get('runtime_files', [])
    if not files:
        raise ValueError('Compatibility manifest has no runtime file identities')
    for item in files:
        actual=Path(item['path']).resolve(strict=True)
        if not actual.is_file() or digest(actual) != item['sha256']:
            raise ValueError('A CANN compatibility runtime file changed: '+str(actual))
        role=item['role']
        if (role=='python_extension' and actual != extension
                or role=='operator_api' and actual != expected_api
                or role=='operator_kernel' and not actual.is_relative_to(opp.resolve())):
            raise ValueError('Compatibility manifest points at a different loaded package')
        roles.add(role)
    if not {'python_extension','operator_api','operator_kernel'}.issubset(roles):
        raise ValueError('Pin the extension, operator API and compiled operator kernel files')
    here=Path(__file__).parent
    for name in ('cann.py','sh.py','manifest.py'):
        if manifest.get('adapter_sha256',{}).get(name) != digest(here/name):
            raise ValueError('The tested MTGS adapter source differs: '+name)
    for name in ('tests/probe_cann.py','tests/numpy_oracle.py'):
        if manifest.get('validation_sources_sha256',{}).get(name)!=digest(here/name):
            raise ValueError('The tested validation/oracle source differs: '+name)
    patch=here/'patches/calc_render_mtgs_compat.cpp'
    if manifest.get('compatibility_kernel_sha256') != digest(patch):
        raise ValueError('Compatibility kernel source differs from the manifest')
    if not validation_only:
        validation=manifest.get('validation',{})
        if (validation.get('result') != 'pass'
                or not REQUIRED_CASES.issubset(set(validation.get('passed_cases',[])))
                or not validation.get('evidence_sha256') or not validation.get('evidence_path')):
            raise ValueError('This CANN build has not passed the required device validation')
        evidence_path=Path(validation['evidence_path'])
        if digest(evidence_path)!=validation['evidence_sha256']:
            raise ValueError('Device validation evidence changed')
        evidence=json.loads(evidence_path.read_text())
        if (evidence.get('result')!='pass'
                or not REQUIRED_CASES.issubset(set(evidence.get('passed_cases',[])))
                or any(evidence.get('identity',{}).get(k)!=manifest.get(k) for k in IDENTITY_FIELDS)):
            raise ValueError('Device validation does not cover this exact build and tested limits')
    limits=manifest.get('validated_limits',{})
    if not all(isinstance(limits.get(k),int) and limits[k]>0
               for k in ('max_gaussians','max_cameras','max_pixels')):
        raise ValueError('An explicit tested shape envelope is required')
    return manifest
