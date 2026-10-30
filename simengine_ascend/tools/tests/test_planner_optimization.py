"""CPU bridge contracts; these fixtures do not execute a model or accelerator."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from simengine_ascend import planner_optimization as bridge


class PlannerOptimizationTest(unittest.TestCase):
    def args(self, **changes):
        values = dict(precision='fp32', msda_backend='cuda-compatible',
                      inference_optimization='point-parallel-last-frame', msda_query_chunk=8192)
        values.update(changes)
        return SimpleNamespace(**values)

    def evidence(self):
        return dict(requested='point-parallel-last-frame', query_chunk=8192,
                    point_parallel_calls=32, original_msda_fallback_calls=0,
                    original_frame_fallback_calls=0, fallback_reasons={}, last_frame_calls=2,
                    last_frame_source_contract='pinned_original_methods', preserved_NumPy_draws=12)

    def test_default_cuda_and_original_BF16_options_unchanged(self):
        self.assertEqual(bridge.options(SimpleNamespace(precision=None), 'cuda'), ('none', 2048))
        self.assertEqual(bridge.options(SimpleNamespace(precision='bf16'), 'npu'), ('none', 2048))

    def test_explicit_supported_NPU_FP32_options(self):
        self.assertEqual(bridge.options(self.args(), 'npu'), ('point-parallel-last-frame', 8192))
        for chunk in (2048, 4096, 8192):
            self.assertEqual(bridge.options(self.args(msda_query_chunk=chunk), 'npu')[1], chunk)

    def test_reject_CUDA_BF16_or_reference_optimization(self):
        for args, backend in ((self.args(), 'cuda'), (self.args(precision='bf16'), 'npu'),
                              (self.args(msda_backend='reference'), 'npu')):
            with self.assertRaises(ValueError): bridge.options(args, backend)

    def test_reject_unknown_mode_chunk_and_boolean(self):
        for changes in (dict(inference_optimization='automatic'), dict(msda_query_chunk=1),
                        dict(msda_query_chunk=True), dict(inference_optimization='none')):
            with self.assertRaises(ValueError): bridge.options(self.args(**changes), 'npu')

    def test_default_source_does_not_require_new_AlgEngine(self):
        self.assertIsNone(bridge.bind_source('/does/not/exist', {}, 'none'))

    def test_reject_wrong_external_source_or_file_count(self):
        for identity in ({'source_sha256': '0' * 64, 'python_files': 249},
                         {'source_sha256': bridge.ALG_SOURCE, 'python_files': 245}):
            with self.assertRaises(ValueError): bridge.bind_source('.', identity, 'point-parallel')

    def test_bind_reads_actual_helper_bytes_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'ascend/example.py'; path.parent.mkdir()
            path.write_bytes(b'fixture actual bytes')
            pins = {'ascend/example.py': hashlib.sha256(path.read_bytes()).hexdigest()}
            identity = dict(source_sha256=bridge.ALG_SOURCE, python_files=bridge.ALG_FILES)
            with patch.object(bridge, 'IMPLEMENTATION', pins):
                self.assertEqual(bridge.bind_source(root, identity, 'point-parallel')['helpers'], pins)
                path.write_bytes(b'replaced')
                with self.assertRaises(ValueError): bridge.bind_source(root, identity, 'point-parallel')

    def test_accept_actual_last_frame_counter_contract(self):
        bridge.validate_actual(self.evidence(), 'point-parallel-last-frame', 8192, 2)

    def test_reject_missing_execution_or_silent_original_fallback(self):
        for field, value in (('point_parallel_calls', 0), ('original_msda_fallback_calls', 1),
                             ('original_frame_fallback_calls', 1), ('fallback_reasons', {'unknown': 1})):
            evidence = self.evidence(); evidence[field] = value
            with self.assertRaises(RuntimeError): bridge.validate_actual(evidence, 'point-parallel-last-frame', 8192, 2)

    def test_reject_incomplete_frame_or_RNG_contract(self):
        for field, value in (('last_frame_calls', 1), ('preserved_NumPy_draws', 6),
                             ('last_frame_source_contract', 'changed_method')):
            evidence = self.evidence(); evidence[field] = value
            with self.assertRaises(RuntimeError): bridge.validate_actual(evidence, 'point-parallel-last-frame', 8192, 2)

    def test_reject_no_model_predictions(self):
        for count in (0, True, -1):
            with self.assertRaises(RuntimeError): bridge.validate_actual(self.evidence(), 'point-parallel-last-frame', 8192, count)

    def test_none_context_does_not_import_external_Ascend(self):
        runtime = SimpleNamespace(optimization_options=('none', 2048), provenance={})
        before = {name for name in sys.modules if name == 'ascend' or name.startswith('ascend.')}
        with bridge.activate(runtime): pass
        self.assertEqual(before, {name for name in sys.modules if name == 'ascend' or name.startswith('ascend.')})
        self.assertEqual(runtime.provenance, {})

    def test_external_context_finally_and_actual_evidence_preserved_on_error(self):
        marker = []
        evidence = self.evidence()
        @contextmanager
        def original(mode, chunk, **kwargs):
            marker.append(('enter', mode, chunk, kwargs))
            try: yield evidence
            finally: marker.append(('restored',))
        package = ModuleType('ascend'); package.__path__ = []
        module = ModuleType('ascend.inference_optimization'); module.optimized_inference = original
        runtime = SimpleNamespace(optimization_options=('point-parallel-last-frame', 8192), root=Path('/fixture'), provenance={})
        with patch.dict(sys.modules, {'ascend': package, 'ascend.inference_optimization': module}):
            with self.assertRaisesRegex(RuntimeError, 'model fixture'):
                with bridge.activate(runtime): raise RuntimeError('model fixture')
        self.assertEqual(marker[-1], ('restored',))
        self.assertEqual(runtime.provenance['inference_optimization'], evidence)
        evidence['point_parallel_calls'] = 999
        self.assertEqual(runtime.provenance['inference_optimization']['point_parallel_calls'], 32)


if __name__ == '__main__':
    unittest.main()
