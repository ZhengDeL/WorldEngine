"""CPU analytic tests for the independent, bounded validation oracle."""
import unittest
import numpy as np
from numpy_oracle import composite,project,sh_bases


def projected(opacity=.5,conic=1,center=(.5,.5),depth=2):
    p=dict(xy=np.array([center]),depth=np.array([depth]),conics=np.array([[conic,0,conic]]),
           compensation=np.array([.25]),radius=np.array([5]))
    return [p],np.array([opacity]),np.array([[[1.,0.,0.]]])


class ContractMath(unittest.TestCase):
    def test_independent_SH_poles_and_first_order_signs(self):
        bases=sh_bases(4,[[0,0,1],[1,0,0],[0,1,0]])
        for ell in range(5):
            self.assertAlmostEqual(bases[0,ell*ell+ell],np.sqrt((2*ell+1)/(4*np.pi)))
        self.assertAlmostEqual(bases[1,3],-np.sqrt(3/(4*np.pi)))
        self.assertAlmostEqual(bases[2,1],-np.sqrt(3/(4*np.pi)))

    def test_independent_SH_addition_theorem(self):
        rng=np.random.default_rng(20261020)
        bases=sh_bases(4,rng.normal(size=(64,3)))
        for ell in range(5):
            np.testing.assert_allclose(np.sum(bases[:,ell*ell:(ell+1)**2]**2,axis=-1),
                                       (2*ell+1)/(4*np.pi),atol=1e-12,rtol=1e-12)
    def test_half_pixel_center(self):
        p,o,c=projected()
        rgb,a=composite(p,o,c,1,1)
        self.assertEqual(a.item(),.5)
        self.assertEqual(rgb[0,0,0,0],.5)

    def test_nonzero_pixel_offset(self):
        p,o,c=projected(center=(0.,0.))
        _,a=composite(p,o,c,1,1)
        self.assertAlmostEqual(a.item(),.5*np.exp(-.25))

    def test_small_alpha_is_retained(self):
        p,o,c=projected(opacity=.005)
        rgb,a=composite(p,o,c,1,1)
        self.assertAlmostEqual(a.item(),.005)
        self.assertGreater(rgb.sum(),0)

    def test_sub_threshold_alpha_skipped(self):
        p,o,c=projected(opacity=1/256)
        rgb,a=composite(p,o,c,1,1)
        self.assertEqual(a.item(),0)
        self.assertEqual(rgb.sum(),0)

    def test_exclusive_transmittance(self):
        p,o,c=projected(opacity=.999)
        p[0]={key:np.repeat(value,3,axis=0) for key,value in p[0].items()}
        p[0]['depth']=np.array([2.,3.,4.]);o=np.full(3,.999)
        c=np.eye(3)[None]
        rgb,a=composite(p,o,c,1,1,expected_depth=True)
        np.testing.assert_allclose(rgb[0,0,0],[.999,0,0,2],atol=1e-12)
        self.assertAlmostEqual(a.item(),.999)

    def test_alpha_compensation_and_expected_depth(self):
        p,o,c=projected()
        rgb,a=composite(p,o,c,1,1,antialiased=True,expected_depth=True)
        self.assertAlmostEqual(a.item(),.125)
        self.assertAlmostEqual(rgb[0,0,0,3],2)

    def test_camera_specific_rgb(self):
        p,o,c=projected();p*=2;c=np.array([[[1.,0,0]],[[0.,1,0]]])
        rgb,a=composite(p,o,c,1,1)
        np.testing.assert_allclose(rgb[:,0,0],[[.5,0,0],[0,.5,0]])

    def test_empty_radii(self):
        p,o,c=projected();p[0]['radius'][:]=0
        rgb,a=composite(p,o,c,1,1,expected_depth=True)
        self.assertEqual(rgb.sum()+a.sum(),0)

    def test_inclusive_near_far(self):
        means=np.array([[0.,0.,z] for z in [.009,.01,.02,9,10,11]])
        q=np.tile([1.,0,0,0],(len(means),1));s=np.full_like(means,.01)
        k=np.array([[[10.,0,8.],[0,10.,8.],[0,0,1.]]])
        p=project(means,q,s,np.eye(4)[None],k,16,16,near=.01,far=10)
        np.testing.assert_array_equal(p[0]['radius']>0,[False,True,True,True,True,False])

    def test_activated_scale_is_not_exponentiated(self):
        means=np.array([[0.,0.,2.]])
        k=np.array([[[10.,0,8.],[0,10.,8.],[0,0,1.]]])
        p=project(means,np.array([[1.,0,0,0]]),np.full((1,3),.2),np.eye(4)[None],k,16,16)
        # At the optical axis: covariance2 diagonal = (10/2*.2)^2 + .3.
        np.testing.assert_allclose(p[0]['conics'],[[1/1.3,0,1/1.3]],atol=1e-12)


if __name__=='__main__':unittest.main()
