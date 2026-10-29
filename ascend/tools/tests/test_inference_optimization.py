"""CPU checks for arithmetic, gradient fallback and context restoration.

These tests do not establish NPU runtime or end-to-end model acceptance.
"""
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from ascend import msda
from ascend.inference_optimization import optimized_inference, validate_options
from ascend.msda_cuda_compatible import make_function as original_function
from ascend.msda_point_parallel import make_function
from ascend.tools.compare_inference import compare


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values


class InferenceOptimizationTests(unittest.TestCase):
    def test_original_level_point_order_and_legacy_prefix(self):
        generator = torch.Generator().manual_seed(729)
        geometries = [([(2, 3)], 1, False), ([(1, 1), (2, 3)], 2, False),
                      ([(3, 2)], 4, True)]
        for pairs, levels, legacy in geometries:
            value = torch.randn(2, sum(y*x for y, x in pairs), 8, 32, generator=generator)
            locations = torch.rand(2, 7, 8, levels, 4, 2, generator=generator) * 1.6 - .3
            weights = torch.randn(2, 7, 8, levels, 4, generator=generator)
            wanted = original_function(torch).forward(Context(), value, locations, weights, pairs, legacy, 2048)
            for chunk in (2048, 4096, 8192):
                with self.subTest(pairs=pairs, legacy=legacy, chunk=chunk):
                    actual = make_function(torch, chunk).forward(Context(), value, locations, weights, pairs, legacy, 2048)
                    self.assertTrue(torch.equal(actual.view(torch.int32), wanted.view(torch.int32)))

    def test_gradient_bearing_inputs_use_original_forward_and_backward(self):
        value = torch.randn(1, 4, 8, 32, requires_grad=True)
        locations = torch.full((1, 2, 8, 1, 4, 2), .35, requires_grad=True)
        weights = torch.ones(1, 2, 8, 1, 4, requires_grad=True)
        ctx = Context()
        original = msda._cuda_function()
        with optimized_inference('point-parallel', 8192, precision='fp32', backend='cuda-compatible') as evidence:
            facade = msda._cuda_function()
            actual = facade.forward(ctx, value, locations, weights, [(2, 2)], False, 2048)
            expected = original.forward(Context(), value, locations, weights, [(2, 2)], False, 2048)
            self.assertTrue(torch.equal(actual, expected))
            self.assertIs(facade.backward, original.backward)
            self.assertEqual(evidence['point_parallel_calls'], 0)
            self.assertEqual(evidence['original_msda_fallback_calls'], 1)
        self.assertIs(msda._cuda_function(), original)

    def test_exception_restores_original_backend(self):
        original = msda._cuda_function()
        with self.assertRaisesRegex(RuntimeError, 'controlled'):
            with optimized_inference('point-parallel', precision='fp32', backend='cuda-compatible'):
                raise RuntimeError('controlled')
        self.assertIs(msda._cuda_function(), original)

    def test_invalid_precision_and_training_chunk_do_not_change_state(self):
        original = msda._cuda_function()
        for values in [('point-parallel', 8192, 'bf16', 'cuda-compatible'),
                       ('point-parallel', 2048, 'fp32', 'reference'),
                       ('none', 8192, 'fp32', 'cuda-compatible')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_options(*values)
        self.assertIs(msda._cuda_function(), original)

    def test_export_comparison_rejects_changed_tensor_identity_and_empty_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'baseline'
            candidate = Path(temporary) / 'candidate'
            base.mkdir()
            candidate.mkdir()
            report = dict(image_id='image', source_sha256='source', source_config_sha256='config',
                          checkpoint_sha256='locked', torch_version='torch', torch_npu_version='npu',
                          world_size=16, device_provenance={'rank': 0},
                          result='pass_real_planning_inference', anchors=1,
                          precision=dict(before_model=dict(mode='fp32', public=dict(
                              matmul_allow_hf32=False, conv_allow_hf32=False))),
                          inference_optimization=dict(
                              requested='point-parallel-last-frame', point_parallel_calls=4,
                              last_frame_calls=1))
            row = dict(token='official-fixture-token', trajectory=np.zeros((40, 3), dtype=np.float32),
                       chosen_ind=0, score=.8)

            def write(folder, metadata, values):
                (folder / 'inference_validation.json').write_text(json.dumps(metadata))
                (folder / 'predictions.pkl').write_bytes(pickle.dumps(values))

            write(base, report, [row])
            write(candidate, report, [row])
            self.assertEqual(compare(base, candidate)['result'], 'exact')
            changed = dict(row, trajectory=row['trajectory'].copy())
            changed['trajectory'][0, 0] = np.nextafter(np.float32(0), np.float32(1))
            write(candidate, report, [changed])
            self.assertEqual(compare(base, candidate)['mismatched_tokens'], [row['token']])
            write(candidate, dict(report, checkpoint_sha256='different'), [row])
            self.assertIn('checkpoint_sha256', compare(base, candidate)['identity_mismatches'])
            write(candidate, dict(report, anchors=0), [])
            self.assertEqual(compare(base, candidate)['result'], 'requires_diagnosis')

    def test_fallback_execution_does_not_count_as_requested_optimization(self):
        with tempfile.TemporaryDirectory() as temporary:
            folders = [Path(temporary) / name for name in ('baseline', 'candidate')]
            report = dict(image_id='image', source_sha256='source', source_config_sha256='config',
                          checkpoint_sha256='locked', torch_version='torch', torch_npu_version='npu',
                          world_size=16, device_provenance={'rank': 0},
                          result='pass_real_planning_inference', anchors=1,
                          precision=dict(before_model=dict(mode='fp32', public=dict(
                              matmul_allow_hf32=False, conv_allow_hf32=False))),
                          inference_optimization=dict(requested='point-parallel-last-frame',
                                                     point_parallel_calls=4, last_frame_calls=0))
            for folder in folders:
                folder.mkdir()
                (folder / 'inference_validation.json').write_text(json.dumps(report))
                (folder / 'predictions.pkl').write_bytes(pickle.dumps([dict(token='fixture')]))
            actual = compare(*folders)
            self.assertEqual(actual['result'], 'exact')
            self.assertFalse(actual['requested_optimization_actually_used'])


if __name__ == '__main__':
    unittest.main()
