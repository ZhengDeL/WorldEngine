"""Filesystem-only gate tests; fixture files are not accelerator binaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manifest_under_test',HERE/'manifest.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='MTGS_manifest_fixture_')
        self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name)
        self.extension=root/'package/gsplat_c.so'
        self.api=root/'package/packages/vendors/customize/op_api/lib/libcust_opapi.so'
        self.kernel=root/'package/packages/vendors/customize/op_impl/kernel.o'
        self.path=root/'build.json';self.evidence=root/'evidence.json'
        for i,path in enumerate((self.extension,self.api,self.kernel)):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('fixture_'+str(i)).encode())
        self.versions={'torch':'fixture_only','torch_npu':'fixture_only'}
        self.manifest=dict(contract=module.CONTRACT,canonical_commit=module.CANONICAL_COMMIT,
            runtime_versions=self.versions,
            runtime_files=[dict(role=role,path=str(p),sha256=module.digest(p)) for role,p in
                           [('python_extension',self.extension),('operator_api',self.api),('operator_kernel',self.kernel)]],
            adapter_sha256={name:module.digest(HERE/name) for name in ('cann.py','sh.py','manifest.py')},
            validation_sources_sha256={name:module.digest(HERE/name) for name in ('tests/probe_cann.py','tests/numpy_oracle.py')},
            compatibility_kernel_sha256=module.digest(HERE/'patches/calc_render_mtgs_compat.cpp'),
            validated_limits=dict(max_gaussians=32,max_cameras=2,max_pixels=1024),validation=dict(result='not_run'))

    def check(self,probe=False):
        self.path.write_text(json.dumps(self.manifest))
        return module.validate_manifest(self.path,self.extension,self.versions,validation_only=probe)

    def approve(self):
        evidence=dict(result='pass',passed_cases=sorted(module.REQUIRED_CASES),
                      identity={k:self.manifest[k] for k in module.IDENTITY_FIELDS})
        self.evidence.write_text(json.dumps(evidence))
        self.manifest['validation']=dict(result='pass',passed_cases=sorted(module.REQUIRED_CASES),
             evidence_path=str(self.evidence),evidence_sha256=module.digest(self.evidence))

    def test_probe_explicitly_allows_unvalidated_build(self):
        self.check(probe=True)

    def test_production_rejects_unvalidated_build(self):
        with self.assertRaises(ValueError):self.check()

    def test_production_requires_bound_evidence(self):
        self.approve();self.check()

    def test_changed_runtime_binary_rejected(self):
        self.kernel.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check(probe=True)

    def test_different_loaded_extension_rejected(self):
        other=self.extension.with_name('foreign.so');other.write_bytes(self.extension.read_bytes())
        self.manifest['runtime_files'][0]['path']=str(other)
        with self.assertRaises(ValueError):self.check(probe=True)

    def test_different_adapter_rejected(self):
        self.manifest['adapter_sha256']['cann.py']='0'*64
        with self.assertRaises(ValueError):self.check(probe=True)

    def test_different_validation_oracle_rejected(self):
        self.manifest['validation_sources_sha256']['tests/numpy_oracle.py']='0'*64
        with self.assertRaises(ValueError):self.check(probe=True)

    def test_missing_operator_kernel_identity_rejected(self):
        self.manifest['runtime_files']=self.manifest['runtime_files'][:2]
        with self.assertRaises(ValueError):self.check(probe=True)

    def test_claimed_pass_without_required_cases_rejected(self):
        self.approve();self.manifest['validation']['passed_cases']=[]
        with self.assertRaises(ValueError):self.check()

    def test_shape_limits_cannot_be_expanded_without_new_evidence(self):
        self.approve();self.manifest['validated_limits']['max_gaussians']=999999
        with self.assertRaises(ValueError):self.check()

    def test_changed_evidence_rejected(self):
        self.approve();self.evidence.write_text('{}')
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
