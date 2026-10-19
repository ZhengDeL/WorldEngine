"""Torch CPU checks; device execution is covered separately by probe_cann.py."""
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy_oracle import sh_bases
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'A real Torch installation is required')
class TorchContractTests(unittest.TestCase):
    def test_positive_sh_nonunit_scale_invariance_and_independent_reference(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        rng=np.random.default_rng(20261020)
        directions=rng.normal(size=(64,3)).astype(np.float32)
        coefficients=rng.normal(size=(64,25,3)).astype(np.float32)
        scaled=directions*np.linspace(.125,8.,64,dtype=np.float32)[:,None]
        for degree in range(1,5):
            actual=spherical_harmonics(degree,torch.from_numpy(directions),torch.from_numpy(coefficients))
            changed=spherical_harmonics(degree,torch.from_numpy(scaled),torch.from_numpy(coefficients))
            expected=np.sum(sh_bases(degree,scaled)[...,None]*coefficients[:,:((degree+1)**2)],axis=-2)
            torch.testing.assert_close(actual,changed,atol=3e-6,rtol=3e-6)
            np.testing.assert_allclose(changed.numpy(),expected,atol=3e-6,rtol=3e-6)

    def test_zero_direction_preserves_gsplat_degree_dependent_behavior(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        directions=torch.zeros(2,3)
        coefficients=torch.ones(2,25,3)
        actual=spherical_harmonics(0,directions,coefficients)
        torch.testing.assert_close(actual,torch.full((2,3),.2820947917738781))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        for degree in range(1,5):
            self.assertTrue(bool((~torch.isfinite(spherical_harmonics(degree,directions,coefficients))).all()))

    def test_covariance_native_padding_preserves_real_columns_and_inputs(self):
        from worldengine.render.ascend.cann import _covariances_without_tail
        for n in (1,2,3,7,8,9,15,16,17,31,32):
            with self.subTest(gaussians=n):
                quats=torch.arange(n*4,dtype=torch.float32).reshape(n,4)+1
                scales=torch.arange(n*3,dtype=torch.float32).reshape(n,3)+1
                before_quats=quats.clone();before_scales=scales.clone()
                recorded={}
                class Native:
                    def quat_scales_to_covars(self,q,s):
                        recorded.update(q=q.clone(),s=s.clone())
                        aligned=q.shape[-1]
                        if aligned%8 or not q.is_contiguous() or not s.is_contiguous():
                            raise AssertionError('Unaligned native covariance input')
                        # Column identifiers expose accidental inclusion of any
                        # padded covariance in the logical result.
                        return torch.arange(aligned,dtype=torch.float32).expand(1,3,3,aligned).clone()
                covars=_covariances_without_tail(Native(),quats,scales)
                aligned=(n+7)//8*8
                self.assertEqual(recorded['q'].shape,(1,4,aligned))
                self.assertEqual(recorded['s'].shape,(1,3,aligned))
                torch.testing.assert_close(recorded['q'][0,:,:n].T,quats,rtol=0,atol=0)
                torch.testing.assert_close(recorded['s'][0,:,:n].T,scales,rtol=0,atol=0)
                if aligned>n:
                    torch.testing.assert_close(recorded['q'][0,:,n:].T,quats[:1].expand(aligned-n,-1),rtol=0,atol=0)
                    torch.testing.assert_close(recorded['s'][0,:,n:].T,scales[:1].expand(aligned-n,-1),rtol=0,atol=0)
                self.assertTrue(covars.is_contiguous())
                torch.testing.assert_close(covars,torch.arange(n,dtype=torch.float32).expand(1,3,3,n),rtol=0,atol=0)
                torch.testing.assert_close(quats,before_quats,rtol=0,atol=0)
                torch.testing.assert_close(scales,before_scales,rtol=0,atol=0)

    def test_empty_covariance_does_not_call_native(self):
        from worldengine.render.ascend.cann import _covariances_without_tail
        class Native:
            def quat_scales_to_covars(self,*args):
                raise AssertionError('The empty scene entered the native kernel')
        with self.assertRaisesRegex(ValueError,'empty scene'):
            _covariances_without_tail(Native(),torch.empty(0,4),torch.empty(0,3))

    def test_nonfinite_native_covariance_is_reported(self):
        from worldengine.render.ascend.cann import _covariances_without_tail
        class Native:
            def quat_scales_to_covars(self,q,s):
                result=torch.zeros(1,3,3,q.shape[-1])
                result[0,0,0,0]=float('nan')
                return result
        with self.assertRaisesRegex(RuntimeError,'covariance kernel returned nonfinite'):
            _covariances_without_tail(Native(),torch.tensor([[1.,0,0,0]]),torch.ones(1,3))

    def test_all_sh_bases_against_independent_legendre_recurrence(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        rng=np.random.default_rng(20261020)
        directions=rng.normal(size=(64,3)).astype(np.float32)
        directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
        rays=torch.from_numpy(np.broadcast_to(directions,(25,64,3)).copy())
        coefficients=torch.eye(25)[:,None,:,None].expand(25,64,25,3)
        for degree in range(5):
            bases=sh_bases(degree,directions)
            expected=np.zeros((25,64,3),np.float32)
            expected[:(degree+1)**2]=bases.T[...,None]
            actual=spherical_harmonics(degree,rays,coefficients)
            torch.testing.assert_close(actual,torch.from_numpy(expected),atol=3e-6,rtol=3e-6)

    def test_cross_device_tensors_rejected_before_native_call(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_device_guard import FakeNPU
        from worldengine.render.ascend.cann import CANNRasterizer
        def tensor(index):return SimpleNamespace(device=SimpleNamespace(type='npu',index=index),dtype=torch.float32)
        tensors={name:tensor(1) for name in ('means','quats','scales','opacities','colors','viewmats','Ks')}
        tensors['quats']=tensor(0)
        renderer=CANNRasterizer.__new__(CANNRasterizer)
        state=FakeNPU()
        with patch.object(torch,'npu',state,create=True):
            with self.assertRaisesRegex(ValueError,'same explicit NPU'):
                renderer(**tensors,width=16,height=16)
        self.assertEqual(state.entered,[1]);self.assertEqual(state.current,0)

    def test_sh_axis_values_through_degree_four(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        direction=torch.tensor([[0.,0.,1.]])
        values=[(1,2,.48860251190292),(2,6,.63078313050504),
                (3,12,.746352665180231),(4,20,.8462843753216345)]
        for degree,index,expected in values:
            coefficients=torch.zeros(1,25,3);coefficients[:,index]=1
            actual=spherical_harmonics(degree,direction,coefficients)
            torch.testing.assert_close(actual,torch.full((1,3),expected),atol=2e-6,rtol=2e-6)

    def test_sh_x_sign_and_unused_coefficients(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        coefficients=torch.zeros(1,25,3);coefficients[:,3]=1;coefficients[:,24]=100
        actual=spherical_harmonics(1,torch.tensor([[1.,0.,0.]]),coefficients)
        torch.testing.assert_close(actual,torch.full((1,3),-.48860251190292))

    def test_sh_invalid_degree_rejected(self):
        from worldengine.render.ascend.sh import spherical_harmonics
        with self.assertRaises(ValueError):spherical_harmonics(5,torch.zeros(1,3),torch.zeros(1,36,3))

    def test_pixel_centers_tile_roundtrip_and_crop(self):
        from worldengine.render.ascend.cann import pixel_coordinates,tiles_to_image
        coords=pixel_coordinates(17,19,16,'cpu')
        image=tiles_to_image(coords.permute(1,0,2),17,19,16)
        y,x=torch.meshgrid(torch.arange(19),torch.arange(17),indexing='ij')
        torch.testing.assert_close(image,torch.stack((x+.5,y+.5),dim=-1))

    def test_rectangular_tiles_and_stable_equal_depth(self):
        from worldengine.render.ascend.cann import tile_intersections
        ids,counts=tile_intersections(torch.tensor([[16.,8.],[16.,8.]]),torch.tensor([1,1]),
                    torch.tensor([2.,2.]),32,16,16)
        torch.testing.assert_close(ids,torch.tensor([0,1,0,1]))
        torch.testing.assert_close(counts,torch.tensor([2,2]))

    def test_depth_order_and_empty_radius(self):
        from worldengine.render.ascend.cann import tile_intersections
        ids,counts=tile_intersections(torch.tensor([[8.,8.],[8.,8.],[8.,8.]]),torch.tensor([1,1,0]),
                    torch.tensor([3.,2.,1.]),16,16,16)
        torch.testing.assert_close(ids,torch.tensor([1,0]))
        torch.testing.assert_close(counts,torch.tensor([2]))

    def test_intersection_allocation_bound(self):
        from worldengine.render.ascend.cann import tile_intersections
        with self.assertRaises(ValueError):
            tile_intersections(torch.tensor([[8.,8.]]),torch.tensor([1]),torch.tensor([2.]),16,16,16,maximum=0)


if __name__=='__main__':unittest.main()
