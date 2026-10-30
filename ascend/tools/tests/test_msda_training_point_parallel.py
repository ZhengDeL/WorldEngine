"""Strict CPU integration checks for the explicit point/head training candidate."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]
FIXTURE = HERE / 'fixtures/msda_point_parallel'
sys.path.insert(0, str(SOURCE))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These names are confined to this independently executed qualification suite.
baseline = load('baseline', FIXTURE / 'baseline.py')
sys.modules['baseline'] = baseline
point = load('candidate', FIXTURE / 'candidate.py')
sys.modules['candidate'] = point
head = load('head_major_original', FIXTURE / 'head_major_original.py')
sys.modules['head_major_original'] = head
research = load('combined_candidate', FIXTURE / 'combined_candidate.py')
fixtures = load('point_head_original_tensor_fixtures', FIXTURE / 'test_candidate.py')
torch = fixtures.torch
from ascend import msda_training_point_parallel as production
from ascend import msda_value_gradient as original_head


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(tree, name):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, name
    return matches[0]


class TrainingPointHeadContracts(unittest.TestCase):
    def test_frozen_research_and_original_head_whole_sources(self):
        pins = {'baseline.py': 'feb35fc82b67dec1fbf8d24c0f5f60d6ff30356e8f419cf37e90d92481e854d9',
                'candidate.py': '26e89b99a2e67c6295ef5946275c7c76eb3cfd2f103411224123c561584bb2bd',
                'combined_candidate.py': '135f2d062ee7704ae9fb6a4765a1ba326723ecf97de8967e53d22d03bed73a8f',
                'head_major_original.py': '3ea227d93d4a01050c5f83b719cd8b650435d7e94b14f004f58006d42415e03b',
                'test_candidate.py': 'a7e3dceaa859defb9ccd304f889e087782a6e02130a8af72c8a961d58fb46c81'}
        for name, expected in pins.items():
            self.assertEqual(sha(FIXTURE / name), expected)
        self.assertEqual(sha(Path(original_head.__file__)), pins['head_major_original.py'])
        self.assertEqual(original_head.MODES, ('none', 'head-major'))
        self.assertIs(production.make_accumulator, original_head.make_accumulator)

    def test_complete_measured_math_factory_AST_is_unchanged(self):
        before = ast.parse((FIXTURE / 'combined_candidate.py').read_text())
        after = ast.parse(Path(production.__file__).read_text())
        for name in ('make_function', 'samples', 'serial_channels', 'forward', 'backward'):
            self.assertEqual(ast.dump(node(before, name), include_attributes=False),
                             ast.dump(node(after, name), include_attributes=False), name)
        coordinate = ast.parse((SOURCE / 'ascend/msda_cuda_compatible.py').read_text())
        old = ast.parse((FIXTURE / 'baseline.py').read_text())
        for name in ('canonical', 'integer_scale_parts', 'finite_padding_coordinate',
                     'compensated_scale_offset_FP32', 'pixel_coordinates'):
            self.assertEqual(ast.dump(node(coordinate, name), include_attributes=False),
                             ast.dump(node(old, name), include_attributes=False), name)

    def test_no_point_reduction_and_original_point_corner_channel_order(self):
        tree = ast.parse(Path(production.__file__).read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(name.endswith(('.sum', '.matmul', '.einsum', '.compile')) for name in calls))
        backward = node(tree, 'backward')
        point = next(n for n in ast.walk(backward) if isinstance(n, ast.For) and ast.unparse(n.target) == 'point')
        self.assertEqual(ast.unparse(point.body[0].iter), 'zip(coef, addresses, masks)')
        callback = point.body[0].body[2]
        self.assertEqual(ast.unparse(callback), "returned = grad_value_accumulator(flat, addr[..., point].reshape(-1).contiguous(), contribution.reshape(-1, c).contiguous(), (b, s, h, c))")
        self.assertEqual(ast.unparse(node(tree, 'serial_channels').body[1].iter), 'range(1, 32)')
        chunk = next(n for n in ast.walk(node(tree, 'samples')) if isinstance(n, ast.For) and ast.unparse(n.target) == 'begin')
        self.assertEqual(ast.unparse(chunk.iter), 'range(0, q, chunk)')

    def test_new_mode_requires_training_FP32_cuda_and_runtime(self):
        self.assertEqual(production.MODES, ('none', 'head-major', 'point-parallel-head-major'))
        production.validate_training_option(production.MODE, precision='fp32', backend='cuda-compatible')
        for mode in ('none', 'head-major'):
            production.validate_training_option(mode, precision='fp32', backend='cuda-compatible')
        for change in [dict(precision='bf16'), dict(precision='fp32-hf32-conv'), dict(backend='reference'),
                       dict(task='inference'), dict(runtime=False)]:
            options = dict(precision='fp32', backend='cuda-compatible', task='train', runtime=True)
            options.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                production.validate_training_option(production.MODE, **options)
        with self.assertRaises(ValueError):
            production.validate_training_option('auto', precision='fp32', backend='cuda-compatible')

    def test_launcher_refuses_missing_conditions_before_device_or_input_access(self):
        common = [sys.executable, '-B', str(SOURCE / 'ascend/run_container.py'), '--devices', '0',
                  '--checkpoint', '/generated_missing', '--vocab', '/generated_missing',
                  '--output', '/generated_missing', '--msda-value-grad-layout', production.MODE]
        for arguments, message in [
            (['--task', 'inference', '--precision', 'fp32', '--msda-backend', 'cuda-compatible'], 'requires training'),
            (['--task', 'train', '--precision', 'bf16', '--msda-backend', 'cuda-compatible'], 'requires FP32'),
            (['--task', 'train', '--precision', 'fp32', '--msda-backend', 'reference'], 'requires FP32'),
            (['--task', 'train', '--precision', 'fp32', '--msda-backend', 'cuda-compatible'], 'complete runtime')]:
            with self.subTest(arguments=arguments):
                result = subprocess.run(common + arguments, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

    def test_train_and_host_CHOICES_preserve_default_and_only_explicit_child_flag(self):
        for name in ('train.py', 'run_container.py'):
            tree = ast.parse((SOURCE / 'ascend' / name).read_text())
            options = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == '--msda-value-grad-layout']
            self.assertEqual(len(options), 1)
            kwargs = {n.arg: ast.literal_eval(n.value) for n in options[0].keywords if n.arg in ('choices', 'default')}
            self.assertEqual(kwargs['default'], 'none')
            self.assertEqual(kwargs['choices'], production.MODES)
        text = (SOURCE / 'ascend/run_container.py').read_text()
        self.assertIn("'--msda-value-grad-layout',args.msda_value_grad_layout", text)

    def test_resume_refuses_old_source_mode_or_helper_before_tensor_restore(self):
        tree = ast.parse((SOURCE / 'ascend/runtime.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResumeMixin')
        saved = {'optimizer': {}, 'meta': {'ascend_resume': {'contract': {}}}}
        namespace = {'torch': SimpleNamespace(load=lambda *a, **kw: saved)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual unchanged ResumeMixin>', 'exec'), namespace)
        current = dict(source_sha256='generated_new_source', precision=dict(msda=dict(
            training_value_gradient=dict(mode=production.MODE, helper_source_sha256=sha(Path(production.__file__))))))
        for change in ('source', 'mode', 'helper'):
            old = copy.deepcopy(current)
            if change == 'source':
                old['source_sha256'] = '216499c4268afe5e6ed32564765616b636bde1723b38c47b924ebae8ca9a0e5c'
            elif change == 'mode':
                old['precision']['msda']['training_value_gradient']['mode'] = 'head-major'
            else:
                old['precision']['msda']['training_value_gradient']['helper_source_sha256'] = '0' * 64
            saved['meta']['ascend_resume']['contract'] = old
            runner = namespace['ResumeMixin']()
            runner.resume_contract = current
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'Resume contract changed'):
                runner.resume('/generated_checkpoint_not_read')


@unittest.skipUnless(torch is not None, 'Real Torch CPU is mandatory on the target; zero skips there')
class ProductionRawEquivalence(fixtures.TensorEquivalence):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.b = production.make_function(torch)
        cls.research = research.make_function(torch)

    def compare(self, name, data, pairs=((3, 4), (2, 2)), chunk=4, legacy=False, needed=(True, True, True)):
        before = {}
        for repeat in (1, 2):
            left, right = super().compare(name, data, pairs, chunk, legacy, needed)
            measured = self.run_one(self.research, data, pairs, chunk, legacy, needed)
            for label, actual, original in zip(('forward', 'Gv', 'Gloc', 'Ga'), right, measured):
                if actual is None:
                    self.assertIsNone(original)
                    continue
                actual_raw = actual.detach().contiguous().reshape(-1).view(torch.uint8)
                research_raw = original.detach().contiguous().reshape(-1).view(torch.uint8)
                self.assertTrue(torch.equal(actual_raw, research_raw), name + '/' + label)
                if repeat == 2:
                    self.assertTrue(torch.equal(actual_raw, before[label]), name + '/repeat/' + label)
                before[label] = actual_raw.clone()
        return left, right

    def test_B8_eight_points_two_levels_full_three_gradients(self):
        self.compare('B8_H8_P8', self.inputs(b=8, q=9, h=8, p=8), chunk=4)

    def test_factory_none_and_head_major_behavior_and_policy_remain_unchanged(self):
        old_backend = os.environ.get('WORLDENGINE_MSDA_BACKEND')
        try:
            data = self.inputs(q=5)
            for mode in ('none', 'head-major'):
                old = load('old_msda_policy_' + mode, FIXTURE / 'msda_policy_before.py')
                new = load('new_msda_policy_' + mode, SOURCE / 'ascend/msda.py')
                old.configure_backend('cuda-compatible', value_gradient_layout=mode)
                new.configure_backend('cuda-compatible', value_gradient_layout=mode)
                self.assertEqual(old.backend_policy(), new.backend_policy())
                expected = self.run_one(old._cuda_function(), data, ((3, 4), (2, 2)), 4, False, (True, True, True))
                actual = self.run_one(new._cuda_function(), data, ((3, 4), (2, 2)), 4, False, (True, True, True))
                for a, b in zip(expected, actual):
                    self.assertTrue(torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))
        finally:
            if old_backend is None:
                os.environ.pop('WORLDENGINE_MSDA_BACKEND', None)
            else:
                os.environ['WORLDENGINE_MSDA_BACKEND'] = old_backend

    def test_explicit_mode_freezes_factory_and_locks_all_actual_helper_sources(self):
        old_backend = os.environ.get('WORLDENGINE_MSDA_BACKEND')
        old_setting = os.environ.get('WORLDENGINE_MSDA_TRAINING_OPTIMIZATION')
        try:
            os.environ['WORLDENGINE_MSDA_TRAINING_OPTIMIZATION'] = production.MODE
            module = load('fresh_production_point_head_policy', SOURCE / 'ascend/msda.py')
            module.configure_backend('cuda-compatible')
            self.assertEqual(module.backend_policy()['training_value_gradient']['mode'], 'none')
            module.configure_backend('cuda-compatible', value_gradient_layout=production.MODE)
            factory = module._cuda_function()
            self.assertIs(factory, module._cuda_function())
            policy = module.backend_policy()['training_value_gradient']
            self.assertEqual(policy['mode'], production.MODE)
            self.assertEqual(policy['query_chunk'], 2048)
            self.assertEqual(policy['helper_source_sha256'], sha(Path(production.__file__)))
            self.assertEqual(policy['head_helper_source_sha256'], '3ea227d93d4a01050c5f83b719cd8b650435d7e94b14f004f58006d42415e03b')
            self.assertEqual(policy['coordinate_source_sha256'], sha(SOURCE / 'ascend/msda_cuda_compatible.py'))
            self.assertEqual(policy['measured_combined_source_sha256'], '135f2d062ee7704ae9fb6a4765a1ba326723ecf97de8967e53d22d03bed73a8f')
            self.assertFalse(policy['global_Tensor_method_replacement'])
            for other in ('none', 'head-major'):
                with self.assertRaises(RuntimeError):
                    module.configure_backend('cuda-compatible', value_gradient_layout=other)
        finally:
            if old_backend is None:
                os.environ.pop('WORLDENGINE_MSDA_BACKEND', None)
            else:
                os.environ['WORLDENGINE_MSDA_BACKEND'] = old_backend
            if old_setting is None:
                os.environ.pop('WORLDENGINE_MSDA_TRAINING_OPTIMIZATION', None)
            else:
                os.environ['WORLDENGINE_MSDA_TRAINING_OPTIMIZATION'] = old_setting

    def test_address_contribution_stream_and_contiguity_match_original_head(self):
        data = self.inputs(b=8, q=5, h=8, p=8)
        original = load('original_head_callback_training', SOURCE / 'ascend/msda_cuda_compatible.py')
        events = [[], []]
        delegate = original_head.make_accumulator(torch)
        def accumulator(which):
            def checked(target, index, source, layout):
                self.assertTrue(all(x.is_contiguous() for x in (target, index, source)))
                self.assertEqual(layout, (8, 16, 8, 32))
                events[which].append((index.clone(), source.clone()))
                return delegate(target, index, source, layout)
            return checked
        left_fn = original.make_function(torch, grad_value_accumulator=accumulator(0))
        previous = production.make_accumulator
        try:
            production.make_accumulator = lambda api: accumulator(1)
            right_fn = production.make_function(torch)
        finally:
            production.make_accumulator = previous
        self.run_one(left_fn, data, ((3, 4), (2, 2)), 4, False, (True, True, True))
        self.run_one(right_fn, data, ((3, 4), (2, 2)), 4, False, (True, True, True))
        self.assertEqual(len(events[0]), 2 * 2 * 8 * 4)
        self.assertEqual(len(events[0]), len(events[1]))
        for (index_a, source_a), (index_b, source_b) in zip(*events):
            self.assertTrue(torch.equal(index_a, index_b))
            self.assertTrue(torch.equal(source_a.contiguous().view(torch.uint8), source_b.contiguous().view(torch.uint8)))


if __name__ == '__main__':
    unittest.main()
