"""CPU-only checks for immutable full-input capture and replay contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from worldengine.render.ascend.capture import write_capture,load_capture,file_digest


def fixture_inputs(n=9,c=2):
    return dict(means=np.tile(np.array([[0,0,2]],np.float32),(n,1)),
        quats=np.tile(np.array([[1,0,0,0]],np.float32),(n,1)),
        scales=np.full((n,3),.1,np.float32),opacities=np.full(n,.5,np.float32),
        colors=np.full((c,n,3),.3,np.float32),viewmats=np.tile(np.eye(4,dtype=np.float32),(c,1,1)),
        Ks=np.tile(np.array([[20,0,16],[0,20,16],[0,0,1]],np.float32),(c,1,1)))


def fixture_render():
    return dict(width=32,height=32,tile_size=16,near_plane=.01,far_plane=10.,
        render_mode='RGB+ED',rasterize_mode='antialiased',radius_clip=0.,eps2d=.3)


def fixture_provenance(c=2):
    return dict(scene_id='synthetic_fixture_only',frame_id='0',camera_names=['cam'+str(i) for i in range(c)],
        checkpoints=[dict(name='fixture.ckpt',sha256='1'*64)],source_commit='2'*40,source_fingerprint_sha256='3'*64)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='MTGS_capture_fixture_')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def capture(self,inputs=None):
        return write_capture(self.root/'capture',fixture_inputs() if inputs is None else inputs,
                             fixture_render(),fixture_provenance())

    def test_exact_roundtrip_and_owned_snapshot(self):
        inputs=fixture_inputs();path=self.capture(inputs)
        inputs['means'][0,0]=999
        loaded,metadata=load_capture(path)
        self.assertEqual(loaded['means'][0,0],0)
        self.assertEqual(metadata['shape'],dict(gaussians=9,cameras=2))
        for name,value in fixture_inputs().items():
            np.testing.assert_array_equal(loaded[name],value)

    def test_empty_gaussian_capture_has_well_defined_array_hashes(self):
        path=self.capture(fixture_inputs(n=0));loaded,_=load_capture(path)
        self.assertEqual(loaded['colors'].shape,(2,0,3))

    def test_fresh_output_is_required(self):
        self.capture()
        with self.assertRaises(FileExistsError):self.capture()

    def test_fp64_inputs_are_not_silently_cast(self):
        inputs=fixture_inputs();inputs['means']=inputs['means'].astype(np.float64)
        with self.assertRaisesRegex(ValueError,'without implicit casting'):self.capture(inputs)
        self.assertFalse((self.root/'capture').exists())

    def test_nonfinite_and_inactive_parameters_rejected(self):
        for field,value in [('means',float('nan')),('scales',0.),('opacities',2.),('quats',0.)]:
            inputs=fixture_inputs();inputs[field][...]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.capture(inputs)

    def test_corrupt_archive_rejected_before_loading(self):
        path=self.capture();archive=path.parent/'inputs.npz'
        archive.write_bytes(archive.read_bytes()+b'corruption')
        with self.assertRaises(ValueError):load_capture(path)

    def test_edited_array_identity_rejected(self):
        path=self.capture();value=json.loads(path.read_text())
        value['arrays']['means']['sha256']='0'*64;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'array shape or content'):load_capture(path)

    def test_duplicate_zip_members_rejected_even_with_new_whole_hash(self):
        path=self.capture();archive=path.parent/'inputs.npz'
        with zipfile.ZipFile(archive,'a') as saved,warnings.catch_warnings():
            warnings.simplefilter('ignore');saved.writestr('means.npy',b'not an array')
        value=json.loads(path.read_text());value['archive'].update(bytes=archive.stat().st_size,sha256=file_digest(archive))
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'duplicate members'):load_capture(path)

    def test_checkpoint_provenance_cannot_escape_relative_asset_names(self):
        provenance=fixture_provenance();provenance['checkpoints'][0]['name']='../foreign.ckpt'
        with self.assertRaises(ValueError):
            write_capture(self.root/'bad',fixture_inputs(),fixture_render(),provenance)

    def test_changed_render_shape_rejected(self):
        path=self.capture();value=json.loads(path.read_text());value['shape']['gaussians']=8
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'complete FP32 shape'):load_capture(path)


if __name__=='__main__':unittest.main()
