"""Mock evidence-chain tests; no accelerator or scene acceptance is claimed."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from worldengine.render.ascend import real_validation as gate
from worldengine.render.ascend.capture import write_capture
from worldengine.render.ascend.manifest import validate_manifest,digest
from test_capture import fixture_inputs,fixture_render,fixture_provenance
import test_manifest as _small_fixtures


class RealEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixture=_small_fixtures.ManifestTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.approve();fixture.check()
        self.fixture=fixture;self.root=fixture.path.parent
        self.small_path=fixture.path
        provenance=fixture_provenance()
        # Declarations here belong to a mocked chain, not a real dataset.
        provenance.update(input_origin='official_mtgs_scene',official_asset_manifest_sha256='5'*64)
        self.capture_path=write_capture(self.root/'capture',fixture_inputs(n=65),fixture_render(),provenance)
        mocked=patch.object(gate,'sources_identity',return_value={'fixture_only':'4'*64})
        mocked.start();self.addCleanup(mocked.stop)
        self.deadline=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat()
        self.lease=gate.create_lease(self.small_path,self.capture_path,self.deadline,fixture.extension,fixture.versions)
        self.lease_path=self.root/'lease.json'

    def check_lease(self):
        self.lease_path.write_text(json.dumps(self.lease))
        return gate.validate_lease(self.lease_path,self.fixture.extension,self.fixture.versions)

    def approve_real_fixture(self):
        identity,_=self.check_lease()
        self.real=identity
        pixels=[[camera,y,x] for camera in range(2) for y,x in [(0,0),(0,31),(31,0),(31,31),(16,16)]]
        count=len(pixels)
        self.report=dict(result='pass',evidence_type=gate.EVIDENCE_TYPE,
            identity={name:identity[name] for name in gate.REAL_IDENTITY_FIELDS},
            all_forward_outputs_finite=True,sampled_reference_passed=True,
            reference_pixel_count=count,reference_pixels=pixels,
            reference_statistics=dict(gaussians=65,cameras=2,sample_pixels=count,
                all_gaussians_projected_per_camera=True,gaussian_subsampling=False,candidate_truncation=False,
                camera_statistics=[dict(camera=i,gaussians_projected=65) for i in range(2)]),
            full_output_shapes=dict(rgb=[2,32,32,4],alpha=[2,32,32,1],radii=[2,65]),
            samples=dict(actual_rgb=[[0]*4 for _ in pixels],actual_alpha=[[0] for _ in pixels],
                reference_rgb=[[0]*3 for _ in pixels],reference_alpha=[[0] for _ in pixels],
                reference_expected_depth=[[0] for _ in pixels]))
        self.report_path=self.root/'real_fixture_evidence.json';self.real_path=self.root/'real_fixture_manifest.json'

    def check_real(self):
        self.report_path.write_text(json.dumps(self.report))
        self.real['validation']=dict(result='pass',evidence_path=str(self.report_path),evidence_sha256=digest(self.report_path))
        self.real_path.write_text(json.dumps(self.real))
        return validate_manifest(self.real_path,self.fixture.extension,self.fixture.versions)

    def test_explicit_lease_binds_full_capture_but_does_not_claim_pass(self):
        candidate,deadline=self.check_lease()
        self.assertGreater(deadline,datetime.now(timezone.utc).timestamp())
        self.assertEqual(candidate['validated_limits']['max_gaussians'],65)
        self.assertNotIn('validation',candidate)
        self.assertEqual(candidate['evidence_type'],gate.EVIDENCE_TYPE)

    def test_expired_lease_rejected(self):
        self.lease['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError,'not currently valid'):self.check_lease()

    def test_changed_small_proof_rejected(self):
        self.small_path.write_text(self.small_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'small proof manifest changed'):self.check_lease()

    def test_changed_capture_rejected(self):
        self.capture_path.write_text(self.capture_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'real input or validation implementation changed'):self.check_lease()

    def test_changed_validator_source_rejected(self):
        self.lease['real_sources_sha256']={'fixture_only':'5'*64}
        with self.assertRaisesRegex(ValueError,'validation implementation changed'):self.check_lease()

    def test_two_linked_proofs_required_for_production_expansion(self):
        self.approve_real_fixture();result=self.check_real()
        self.assertEqual(result['validated_limits']['max_gaussians'],65)
        self.assertEqual(result['capture_binding']['shape']['gaussians'],65)

    def test_limits_cannot_be_manually_expanded(self):
        self.approve_real_fixture();self.real['validated_limits']['max_gaussians']=66
        with self.assertRaisesRegex(ValueError,'expanded manifest'):self.check_real()

    def test_pass_flag_cannot_hide_numerical_error(self):
        self.approve_real_fixture();self.report['samples']['actual_rgb'][0][0]=.1
        with self.assertRaisesRegex(ValueError,'fixed numerical tolerance'):self.check_real()

    def test_nonfinite_pixel_evidence_rejected(self):
        self.approve_real_fixture();self.report['samples']['reference_alpha'][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.check_real()

    def test_incomplete_finite_check_shape_rejected(self):
        self.approve_real_fixture();self.report['full_output_shapes']['rgb']=[1,32,32,4]
        with self.assertRaisesRegex(ValueError,'complete forward shape'):self.check_real()

    def test_omitted_camera_reference_rejected(self):
        self.approve_real_fixture();self.report['reference_pixels']=[[0,0,0]]
        self.report['reference_pixel_count']=1
        with self.assertRaisesRegex(ValueError,'omits captured cameras'):self.check_real()

    def test_candidate_or_gaussian_truncation_cannot_be_accepted(self):
        self.approve_real_fixture();self.report['reference_statistics']['camera_statistics'][1]['gaussians_projected']=64
        with self.assertRaisesRegex(ValueError,'every Gaussian'):self.check_real()
        self.report['reference_statistics']['camera_statistics'][1]['gaussians_projected']=65
        self.report['reference_statistics']['candidate_truncation']=True
        with self.assertRaisesRegex(ValueError,'without truncation'):self.check_real()

    def test_lease_is_not_a_production_manifest(self):
        self.lease_path.write_text(json.dumps(self.lease))
        with self.assertRaises(ValueError):validate_manifest(self.lease_path,self.fixture.extension,self.fixture.versions)

    def test_render_shape_and_settings_are_exact(self):
        settings=fixture_render();gate.check_render_contract(settings,**settings)
        changed=dict(settings,width=64)
        with self.assertRaisesRegex(ValueError,'different render settings'):gate.check_render_contract(settings,**changed)


if __name__=='__main__':unittest.main()
