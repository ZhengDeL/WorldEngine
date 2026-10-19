"""CPU checks for complete-set, selected-pixel validation semantics."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from numpy_oracle import composite, project
import sampled_oracle
from sampled_oracle import sampled_reference


def fixture(n=5, cameras=2, width=33, height=25):
    rng = np.random.default_rng(20261020)
    means = rng.normal(0, .4, (n, 3)).astype(np.float32)
    means[:, 2] = rng.uniform(2, 5, n)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    views = np.tile(np.eye(4, dtype=np.float32), (cameras, 1, 1))
    if cameras > 1:
        views[1, 0, 3] = .2
    ks = np.tile([[width*.7, 0, width*.5+.5], [0, height*.8, height*.5+.5], [0, 0, 1]],
                 (cameras, 1, 1)).astype(np.float32)
    inputs = dict(means=means, quats=quats, scales=rng.uniform(.1, .5, (n, 3)).astype(np.float32),
                  opacities=rng.uniform(.01, .8, n).astype(np.float32),
                  colors=rng.uniform(0, 1, (cameras, n, 3)).astype(np.float32), viewmats=views, Ks=ks)
    render = dict(width=width, height=height, tile_size=16, near_plane=.01, far_plane=100.,
                  render_mode='RGB+ED', rasterize_mode='classic', radius_clip=0., eps2d=.3)
    return inputs, render


def all_pixels(inputs, render):
    return [(c, y, x) for c in range(len(inputs['viewmats']))
            for y in range(render['height']) for x in range(render['width'])]


def full_reference(inputs, render):
    projections = project(inputs['means'], inputs['quats'], inputs['scales'], inputs['viewmats'], inputs['Ks'],
                          render['width'], render['height'], eps=render['eps2d'], near=render['near_plane'],
                          far=render['far_plane'], radius_clip=render['radius_clip'])
    rgb, alpha = composite(projections, inputs['opacities'], inputs['colors'], render['width'], render['height'],
                           tile_size=render['tile_size'], antialiased=render['rasterize_mode'] == 'antialiased',
                           expected_depth=True)
    return projections, rgb, alpha


class SampledOracleTests(unittest.TestCase):
    def assert_full_agreement(self, inputs, render, chunk=3):
        pixels = all_pixels(inputs, render)
        actual = sampled_reference(inputs, render, pixels, chunk_size=chunk)
        _, rgb, alpha = full_reference(inputs, render)
        np.testing.assert_allclose(actual['rgb'], rgb[..., :3].reshape(-1, 3), atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(actual['alpha'], alpha.reshape(-1, 1), atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(actual['expected_depth'], rgb[..., 3:].reshape(-1, 1), atol=2e-12, rtol=2e-12)
        json.dumps(actual['statistics'], allow_nan=False)
        return actual

    def test_all_small_pixels_classic_and_antialiased(self):
        for mode in ('classic', 'antialiased'):
            with self.subTest(mode=mode):
                inputs, render = fixture(n=17, width=37, height=23)
                render['rasterize_mode'] = mode
                actual = self.assert_full_agreement(inputs, render)
                self.assertEqual([s['gaussians_projected'] for s in actual['statistics']['camera_statistics']], [17, 17])
                self.assertFalse(actual['statistics']['candidate_truncation'])

    def test_tile32_nonsquare_and_nonmultiple_chunk_tail(self):
        inputs, render = fixture(n=7, width=71, height=37)
        render['tile_size'] = 32
        self.assert_full_agreement(inputs, render, chunk=2)

    def test_equal_depth_global_id_order_across_chunks(self):
        inputs, render = fixture(n=5, cameras=1)
        inputs['means'][:] = [0, 0, 2]
        inputs['quats'][:] = [1, 0, 0, 0]
        inputs['opacities'][:] = .2
        inputs['colors'][0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5]]
        # Integer-sized image places the mean at pixel (height//2,width//2).
        inputs['Ks'][:, 0, 2] = render['width']//2+.5
        inputs['Ks'][:, 1, 2] = render['height']//2+.5
        samples = [(0, render['height']//2, render['width']//2)]
        actual = sampled_reference(inputs, render, samples, chunk_size=2)
        opacity = float(inputs['opacities'][0])
        weights = opacity*(1-opacity)**np.arange(5)
        np.testing.assert_allclose(actual['rgb'][0], weights @ inputs['colors'][0], atol=1e-15)
        np.testing.assert_allclose(actual['expected_depth'], [[2]], atol=1e-14)
        self.assertEqual(actual['statistics']['camera_statistics'][0]['samples'][0]['complete_candidates'], 5)

    def test_rectangular_tile_membership_retains_outside_radius_pixel(self):
        inputs, render = fixture(n=1, cameras=1, width=64, height=64)
        inputs['means'][:] = [0, 0, 2]
        inputs['quats'][:] = [1, 0, 0, 0]
        inputs['Ks'][:] = [[40, 0, 32.5], [0, 40, 32.5], [0, 0, 1]]
        inputs['scales'][:] = np.sqrt(5.5**2-.3)*2/40
        inputs['opacities'][:] = 1
        inputs['colors'][:] = 1
        projections, full, _ = full_reference(inputs, render)
        self.assertEqual(projections[0]['radius'][0], 17)
        # Offset (12,13) lies outside radius17, inside a touched tile, and its
        # alpha exceeds1/255. A circular candidate shortcut would lose it.
        self.assertGreater(np.hypot(12, 13), projections[0]['radius'][0])
        actual = sampled_reference(inputs, render, [(0, 19, 20)], chunk_size=1)
        self.assertGreater(actual['alpha'][0, 0], 1/255)
        np.testing.assert_allclose(actual['rgb'][0], full[0, 19, 20, :3], atol=1e-15)

    def test_threshold_and_exclusive_transmittance(self):
        inputs, render = fixture(n=3, cameras=1, width=32, height=32)
        inputs['means'][:] = [0, 0, 2]
        inputs['quats'][:] = [1, 0, 0, 0]
        inputs['colors'][0] = np.eye(3)
        center = [(0, 16, 16)]
        inputs['opacities'][:] = [1/256, .005, 0]
        actual = sampled_reference(inputs, render, center, chunk_size=1)
        np.testing.assert_allclose(actual['rgb'], [[0, float(np.float32(.005)), 0]], atol=1e-15)
        inputs['opacities'][:] = .999
        actual = sampled_reference(inputs, render, center, chunk_size=1)
        np.testing.assert_allclose(actual['rgb'], [[.999, 0, 0]], atol=1e-15)
        np.testing.assert_allclose(actual['expected_depth'], [[2]], atol=1e-14)
        detail = actual['statistics']['camera_statistics'][0]['samples'][0]
        self.assertEqual(detail['complete_candidates'], 3)
        self.assertEqual(detail['contributing'], 1)
        self.assertTrue(detail['physical_transmittance_termination'])

    def test_empty_and_offscreen_return_zero(self):
        for n in (0, 3):
            inputs, render = fixture(n=n, cameras=2)
            inputs['means'][:] = [1000, 0, 2]
            actual = sampled_reference(inputs, render, [(0, 0, 0), (1, 2, 3)], chunk_size=2)
            self.assertEqual(actual['rgb'].sum()+actual['alpha'].sum()+actual['expected_depth'].sum(), 0)
            self.assertEqual([s['gaussians_projected'] for s in actual['statistics']['camera_statistics']], [n, n])

    def test_every_camera_projected_even_without_sample(self):
        inputs, render = fixture(n=7, cameras=2)
        actual = sampled_reference(inputs, render, [(0, 1, 1)], chunk_size=3)
        self.assertEqual([s['gaussians_projected'] for s in actual['statistics']['camera_statistics']], [7, 7])
        self.assertEqual(actual['statistics']['camera_statistics'][1]['sampled_tiles'], 0)

    def test_candidate_budget_includes_zero_opacity_and_shares_sample_tile(self):
        inputs, render = fixture(n=5, cameras=1, width=32, height=32)
        inputs['means'][:] = [0, 0, 2]
        inputs['quats'][:] = [1, 0, 0, 0]
        inputs['opacities'][:] = 0
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            sampled_reference(inputs, render, [(0, 16, 16)], chunk_size=2, max_candidates=4)
        actual = sampled_reference(inputs, render, [(0, 16, 16), (0, 17, 17)], chunk_size=2, max_candidates=5)
        self.assertEqual(actual['statistics']['peak_camera_tile_pairs'], 5)
        self.assertEqual(actual['statistics']['camera_statistics'][0]['sampled_tiles'], 1)
        self.assertEqual(actual['alpha'].sum(), 0)
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            sampled_reference(inputs, render, [(0, 16, 16), (0, 15, 15)], max_candidates=9)

    def test_near_far_clipping_and_radius_filter(self):
        inputs, render = fixture(n=7, cameras=1, width=32, height=32)
        inputs['means'][:, 0:2] = 0
        inputs['means'][:, 2] = [.009, .01, .02, 9, 10, 11, -1]
        inputs['quats'][:] = [1, 0, 0, 0]
        render['far_plane'] = 10.
        self.assert_full_agreement(inputs, render, chunk=2)
        render['radius_clip'] = 100000.
        self.assertEqual(sampled_reference(inputs, render, [(0, 16, 16)])['alpha'].sum(), 0)

    def test_large_resolution_does_not_allocate_full_image(self):
        inputs, render = fixture(n=4103, cameras=2, width=1920, height=1080)
        actual = sampled_reference(inputs, render, [(0, 540, 960), (1, 10, 10)], chunk_size=257)
        self.assertEqual(actual['rgb'].shape, (2, 3))
        self.assertTrue(np.isfinite(actual['rgb']).all())
        self.assertFalse(actual['statistics']['full_image_comparison'])
        self.assertEqual([s['gaussians_projected'] for s in actual['statistics']['camera_statistics']], [4103, 4103])

    def test_invalid_samples_and_parameters_fail_closed(self):
        inputs, render = fixture()
        for samples in ([], [(0, 1, 1), (0, 1, 1)], [(2, 0, 0)], [(0, -1, 0)], [(0, 0, 33)],
                        [(0., 0, 0)], [(False, 0, 0)], [(0, 0)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                sampled_reference(inputs, render, samples)
        for kwargs in (dict(chunk_size=0), dict(chunk_size=True), dict(max_candidates=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampled_reference(inputs, render, [(0, 0, 0)], **kwargs)

    def test_nonfinite_and_invalid_gaussians_fail_even_unsampled(self):
        for field, index, value in (('means', (0, 0), np.nan), ('colors', (1, 0, 0), np.inf),
                                     ('viewmats', (1, 0, 0), np.nan), ('quats', (0, slice(None)), 0),
                                     ('scales', (0, 0), 0), ('opacities', (0,), 2), ('Ks', (1, 0, 0), 0)):
            inputs, render = fixture()
            inputs[field][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sampled_reference(inputs, render, [(0, 0, 0)])
        inputs, render = fixture(n=0)
        inputs['viewmats'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            sampled_reference(inputs, render, [(0, 0, 0)])

    def test_guard_exception_propagates_during_bounded_work(self):
        inputs, render = fixture(n=101)
        calls = []
        class StopRequested(RuntimeError):
            pass
        def guard():
            calls.append(None)
            if len(calls) == 100:
                raise StopRequested('authorized deadline')
        with self.assertRaisesRegex(StopRequested, 'authorized deadline'):
            sampled_reference(inputs, render, [(0, 0, 0)], chunk_size=3, guard=guard)
        self.assertEqual(len(calls), 100)

    def test_guard_can_interrupt_projection_and_compositing(self):
        inputs, render = fixture(n=7, cameras=1)
        for function_name in ('_project_chunk', '_composite_pixel'):
            state = dict(entered=False)
            original = getattr(sampled_oracle, function_name)
            def mark(*args, **kwargs):
                state['entered'] = True
                return original(*args, **kwargs)
            def guard():
                if state['entered']:
                    raise TimeoutError('deadline inside ' + function_name)
            with self.subTest(stage=function_name), patch.object(sampled_oracle, function_name, side_effect=mark):
                with self.assertRaisesRegex(TimeoutError, 'deadline inside ' + function_name):
                    sampled_reference(inputs, render, [(0, 12, 16)], chunk_size=2, guard=guard)


if __name__ == '__main__':
    unittest.main()
