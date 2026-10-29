"""CPU regressions for report-only raw equality; original numeric gates stay fixed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, 'Actual Torch CPU is required')
class RawTensorEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'compare_resume.py'
        spec = importlib.util.spec_from_file_location('comparison_raw_bytes', path)
        cls.comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.comparison)

    def test_signed_zero_is_value_equal_and_raw_different(self):
        first = torch.tensor([0.0], dtype=torch.float32)
        second = torch.tensor([-0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-5))
        self.assertFalse(self.comparison.raw_tensor_equal(first, second))

    def test_equal_different_and_noncontiguous_values(self):
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        self.assertFalse(first.is_contiguous())
        self.assertTrue(self.comparison.raw_tensor_equal(first, first.clone()))
        second = first.clone()
        second[0, 0] += 1
        self.assertFalse(self.comparison.raw_tensor_equal(first, second))

    def test_shape_and_dtype_mismatch(self):
        first = torch.tensor([1.0], dtype=torch.float32)
        self.assertFalse(self.comparison.raw_tensor_equal(first, first.double()))
        self.assertFalse(self.comparison.raw_tensor_equal(first, first.reshape(1, 1)))

    def test_empty_tensors_keep_shape_and_dtype(self):
        first = torch.empty((0, 3), dtype=torch.float32)
        self.assertTrue(self.comparison.raw_tensor_equal(first, first.clone()))
        self.assertFalse(self.comparison.raw_tensor_equal(first, first.double()))

    def test_raw_counter_does_not_change_original_allclose_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('continuous', 'partial', 'resumed'):
                (root / name).mkdir()
            row = dict(iter=1, tokens=['fixture_token'], lr=[0.0002], loss=1.0)
            for name in ('continuous', 'resumed'):
                (root / name / 'runtime_rank0.jsonl').write_text(json.dumps(row) + '\n')
            (root / 'partial/runtime_rank0.jsonl').write_text('')
            for name, value in (('continuous', 0.0), ('resumed', -0.0)):
                saved = dict(state_dict={'weight': torch.tensor([value])},
                             optimizer={'state': {}, 'param_groups': []},
                             meta={'ascend_resume': dict(contract={'world_size': 1},
                                                        completed_updates=1, completed_epochs=0,
                                                        rng_by_rank=[{'numpy': {'fixture': []}}])})
                torch.save(saved, root / (name + '.pth'))
            output = root / 'comparison.json'
            argv = ['compare_resume.py', '--continuous', str(root / 'continuous.pth'),
                    '--resumed', str(root / 'resumed.pth'), '--continuous-run', str(root / 'continuous'),
                    '--partial-run', str(root / 'partial'), '--resumed-run', str(root / 'resumed'),
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.comparison.main()
            report = json.loads(output.read_text())
            self.assertEqual(report['result'], 'pass')
            self.assertEqual(report['model']['bitwise_equal_tensors'], 0)
            self.assertEqual(report['model']['max_absolute_difference'], 0.0)
            self.assertEqual(report['tensor_tolerance'], {'atol': 1e-6, 'rtol': 1e-5})


if __name__ == '__main__':
    unittest.main()
