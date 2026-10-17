"""CPU regression tests for validation gates and launcher control flow.

Run from a checkout with CPU-capable PyTorch and Bash installed. This creates
only tiny, trusted temporary checkpoints; it does not run a model or Docker.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Process-local settings, before importing Torch; no global configuration writes.
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
SOURCE = None


def selected_functions(path, names, namespace):
    """Compile exact source functions without importing a device module."""
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if sorted(n.name for n in nodes) != sorted(names):
        raise AssertionError('Missing or duplicate function: ' + str(path))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def strict_json(path):
    def reject(value):
        raise AssertionError('Invalid JSON constant: ' + value)
    return json.loads(path.read_bytes(), parse_constant=reject)


class ValidationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.files = [SOURCE / 'ascend' / name for name in
                     ('compare_resume.py', 'validate_msda.py',
                      'validate_msda_layout.py', 'train.py', 'run_container.py')]
        cls.before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in cls.files}
        cls.assert_cpu_only()

    @classmethod
    def assert_cpu_only(cls):
        if cls.torch.cuda.is_initialized():
            raise AssertionError('CUDA must remain uninitialized')
        if any(n == 'torch_npu' or n.startswith('torch_npu.') for n in sys.modules):
            raise AssertionError('No NPU module may be imported')

    @classmethod
    def tearDownClass(cls):
        cls.assert_cpu_only()
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in cls.files}
        if after != cls.before:
            raise AssertionError('Tested sources changed during CPU regressions')

    def tensor(self, values, dtype=None):
        return self.torch.tensor(values, dtype=dtype or self.torch.float32,
                                 device='cpu')

    def comparator(self):
        stats = dict(tensors=0, bitwise_equal_tensors=0,
                     max_absolute_difference=0., failures=[])
        ns = selected_functions(SOURCE / 'ascend/compare_resume.py', ['compare'],
                                dict(torch=self.torch, math=math, stats=stats))
        return stats, ns

    def test_checkpoint_tensor_type_dtype_finiteness(self):
        t, torch = self.tensor, self.torch
        cases = [
            ('equal', t([1.]), t([1.]), True),
            ('within_original_tolerance', t([1.]), t([1.+1e-6]), True),
            ('outside', t([1.]), t([2.]), False),
            ('float_dtype', t([1.]), t([1.], torch.float64), False),
            ('integer_dtype', t([1], torch.int32), t([1], torch.int64), False),
            ('integer_float', t([1], torch.int32), t([1.]), False),
            ('float_integer', t([1.]), t([1], torch.int32), False),
            ('same_inf', t([float('inf')]), t([float('inf')]), False),
            ('same_nan', t([float('nan')]), t([float('nan')]), False),
            ('shape', t([1., 2.]), t([1.]), False),
            ('tensor_list', t([1.]), [1.], False),
            ('list_tensor', [1.], t([1.]), False),
            ('integer_equal', t([1], torch.int64), t([1], torch.int64), True),
        ]
        for name, left, right, accepted in cases:
            with self.subTest(case=name):
                stats, ns = self.comparator()
                ns['compare'](left, right, 'model/test')
                self.assertEqual(not stats['failures'], accepted)

    def test_model_failure_snapshot_is_independent(self):
        stats, ns = self.comparator()
        ns['compare'](self.tensor([1.]), self.tensor([1.]), 'model/one')
        tree = ast.parse((SOURCE / 'ascend/compare_resume.py').read_bytes())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'model_stats'
                                  for t in n.targets))
        module = ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[]))
        exec(compile(module, 'actual_model_stats_assignment', 'exec'), ns)
        ns['compare'](self.tensor([1.]), self.tensor([2.]), 'optimizer/one')
        self.assertEqual(ns['model_stats']['failures'], [])
        self.assertEqual(stats['failures'], ['optimizer/one'])
        self.assertIsNot(ns['model_stats']['failures'], stats['failures'])

    def test_operator_comparators_reject_nonfinite(self):
        for filename in ('validate_msda.py', 'validate_msda_layout.py'):
            names = (['compare', 'unravel', 'point_context']
                     if filename == 'validate_msda.py' else ['compare', 'fingerprint'])
            ns = selected_functions(SOURCE / 'ascend' / filename, names,
                                    dict(torch=self.torch, hashlib=hashlib, MAX_SAMPLES=12))
            for name, left, right, accepted in (
                ('finite', [1.], [1.], True),
                ('within', [1.], [1.+1e-6], True),
                ('outside', [1.], [2.], False),
                ('inf', [float('inf')], [float('inf')], False),
                ('nan', [float('nan')], [float('nan')], False),
                ('shape', [1., 2.], [1.], False),
            ):
                with self.subTest(tool=filename, case=name):
                    a, b = self.tensor(left), self.tensor(right)
                    result = (ns['compare'](a, b, 2e-5, 2e-5, 'forward', {}, [])
                              if filename == 'validate_msda.py'
                              else ns['compare'](self.torch, a, b, 2e-5, 2e-5))
                    self.assertEqual(result['allclose'], accepted)
                    if name in ('inf', 'nan'):
                        self.assertEqual(result['finite_elements'], 0)
                        self.assertEqual(result['failing_elements'], 1)

    def saved_checkpoint(self):
        t = self.tensor
        rng = dict(python=(3, (1, 2), None),
                   numpy=dict(generator='MT19937', keys=[1, 2], position=0,
                              has_gauss=0, cached_gaussian=0.),
                   CPU=t([1, 2], self.torch.uint8), NPU=t([1, 2], self.torch.uint8))
        return dict(state_dict={'weight': t([1.])},
                    optimizer=dict(state={0: dict(step=t(2.), exp_avg=t([1.]),
                                                 exp_avg_sq=t([1.]))},
                                   param_groups=[dict(params=[0])]),
                    meta=dict(ascend_resume=dict(contract={'world_size': 1},
                              completed_updates=2, completed_epochs=0,
                              rng_by_rank=[rng])))

    def test_real_cpu_checkpoint_and_jsonl_cli(self):
        cases = ('finite', 'loss_nan', 'loss_inf', 'lr_inf', 'saved_inf',
                 'dtype', 'missing_step', 'empty_rng', 'wrong_world',
                 'empty_tail', 'unordered', 'horizon_zero', 'horizon_large')
        with tempfile.TemporaryDirectory(prefix='validation_guard_') as directory:
            for case in cases:
                with self.subTest(case=case):
                    root = Path(directory) / case
                    continuous, resumed = root / 'continuous', root / 'resume'
                    continuous.mkdir(parents=True)
                    resumed.mkdir()
                    left, right = self.saved_checkpoint(), self.saved_checkpoint()
                    records = [dict(iter=i, tokens=['own_CPU_fixture'],
                                    lr={'lr': [0.0001]}, loss=1.) for i in (1, 2)]
                    tail = [copy.deepcopy(records[-1])]
                    extra = []
                    if case == 'loss_nan': records[-1]['loss'] = float('nan')
                    if case == 'loss_inf': tail[-1]['loss'] = float('inf')
                    if case == 'lr_inf':
                        records[-1]['lr'] = tail[-1]['lr'] = {'lr': [float('inf')]}
                    if case == 'saved_inf':
                        for checkpoint in (left, right):
                            checkpoint['meta']['ascend_resume']['rng_by_rank'][0]['numpy']['cached_gaussian'] = float('inf')
                    if case == 'dtype': left['state_dict']['weight'] = self.tensor([1], self.torch.int32)
                    if case == 'missing_step': records = records[1:]
                    if case == 'empty_rng':
                        for checkpoint in (left, right):
                            checkpoint['meta']['ascend_resume']['rng_by_rank'] = []
                    if case == 'wrong_world':
                        for checkpoint in (left, right):
                            checkpoint['meta']['ascend_resume']['contract']['world_size'] = 2
                    if case == 'empty_tail': tail = []
                    if case == 'unordered': records.reverse()
                    if case.startswith('horizon_'):
                        extra = ['--comparison-updates', '0' if case == 'horizon_zero' else '3']
                    self.torch.save(left, continuous / 'iter_2.pth')
                    self.torch.save(right, resumed / 'iter_2.pth')
                    for folder, rows in ((continuous, records), (resumed, tail)):
                        (folder / 'runtime_rank0.jsonl').write_text(
                            ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
                    output = root / 'comparison.json'
                    argv = [str(SOURCE / 'ascend/compare_resume.py'),
                            '--continuous', str(continuous / 'iter_2.pth'),
                            '--resumed', str(resumed / 'iter_2.pth'),
                            '--continuous-run', str(continuous), '--partial-run', str(continuous),
                            '--resumed-run', str(resumed), '--output', str(output)] + extra
                    previous, code = sys.argv[:], 0
                    try:
                        sys.argv = argv
                        with contextlib.redirect_stdout(io.StringIO()):
                            try:
                                runpy.run_path(str(SOURCE / 'ascend/compare_resume.py'), run_name='__main__')
                            except SystemExit as error:
                                code = error.code or 0
                    finally:
                        sys.argv = previous
                    result = strict_json(output)
                    self.assertEqual(code, 0 if case == 'finite' else 1)
                    self.assertEqual(result['result'], 'pass' if case == 'finite' else 'fail')
                    self.assertEqual(result['tensor_tolerance'], {'atol': 1e-6, 'rtol': 1e-5})
                    self.assertEqual(result['loss_absolute_tolerance'], 1e-4)

    def test_stop_options(self):
        cases = [({}, True), ({'steps': 2, 'stop_after': 1}, True),
                 ({'steps': 2, 'stop_after': 2}, True),
                 ({'steps': 2, 'duration_seconds': .1}, True),
                 ({'epochs': 2, 'stop_after_epochs': 1}, True),
                 ({'stop_after_epochs': 1}, True)]
        cases += [(values, False) for values in (
            {'stop_after': 1}, {'steps': 2, 'stop_after': 0},
            {'steps': 2, 'stop_after': -1}, {'steps': 2, 'stop_after': 3},
            {'duration_seconds': 1}, {'steps': 2, 'duration_seconds': 0},
            {'steps': 2, 'duration_seconds': -1},
            {'steps': 2, 'duration_seconds': float('nan')},
            {'steps': 2, 'duration_seconds': float('inf')},
            {'steps': 2, 'stop_after_epochs': 1}, {'stop_after_epochs': 0},
            {'stop_after_epochs': -1}, {'epochs': 2, 'stop_after_epochs': 3})]
        for filename in ('train.py', 'run_container.py'):
            fn = selected_functions(SOURCE / 'ascend' / filename,
                                    ['_validate_stop_options'], {})['_validate_stop_options']
            for values, accepted in cases:
                with self.subTest(tool=filename, values=values):
                    args = SimpleNamespace(steps=None, epochs=None, stop_after=None,
                                           stop_after_epochs=None, duration_seconds=None)
                    args.__dict__.update(values)
                    code = 0
                    with contextlib.redirect_stderr(io.StringIO()):
                        try: fn(args, argparse.ArgumentParser(add_help=False))
                        except SystemExit as error: code = error.code
                    self.assertEqual(code == 0, accepted)

    def test_actual_launcher_main_and_early_stop_bash(self):
        if sys.platform != 'linux' or shutil.which('bash') is None:
            self.skipTest('Actual generated-shell regression requires Linux and Bash')
        path = SOURCE / 'ascend/run_container.py'
        native_path = type(Path())
        class DeviceLookupPath(native_path):
            def is_char_device(self):
                return True if self.parent == DeviceLookupPath('/dev') else super().is_char_device()
        ns = dict(__name__='launcher_CPU_definitions', __file__=str(path))
        exec(compile(path.read_bytes(), str(path), 'exec'), ns)
        with tempfile.TemporaryDirectory(prefix='launcher_guard_') as directory:
            base, calls = Path(directory), []
            checkpoint, vocab, mini = base / 'initializer.pth', base / 'vocab.npy', base / 'mini'
            checkpoint.write_bytes(b'own opaque CPU fixture; never loaded')
            vocab.write_bytes(b'own opaque CPU fixture; never loaded')
            mini.mkdir()
            image = ns['IMAGE_ID']
            def inspect(argv, **kwargs):
                self.assertEqual(argv[:4], ['docker', 'image', 'inspect', '--format'])
                return image + '\n'
            def capture(argv, **kwargs):
                self.assertEqual(argv[:2], ['docker', 'run'])
                calls.append(argv)
                return SimpleNamespace(returncode=0)
            ns.update(Path=DeviceLookupPath, ROOT=DeviceLookupPath(base / 'source'),
                      subprocess=SimpleNamespace(check_output=inspect, run=capture))
            output = base / 'output'
            argv = [str(path), '--devices', '0', '--output', str(output),
                    '--checkpoint', str(checkpoint), '--vocab', str(vocab),
                    '--mini-prepared', str(mini), '--steps', '2', '--stop-after', '1',
                    '--inference-after', '--image', image, '--expected-image-id', image]
            previous = sys.argv[:]
            try:
                sys.argv = argv
                with self.assertRaises(SystemExit) as caught:
                    ns['main']()
                self.assertEqual(caught.exception.code, 0)
            finally: sys.argv = previous
            self.assertEqual(len(calls), 1)
            command = calls[0]
            self.assertEqual(command[-3:-1], ['bash', '-lc'])
            shell = command[-1].replace('/output', str(output))
            prelude = 'CPU_FIXTURE_OUTPUT=' + shlex.quote(str(output)) + '''
source() { return 0; }
torchrun() {
  local checkpoint='' previous='' arg
  if [[ "$*" == *"ascend/train.py"* ]]; then
    printf 'opaque own fixture\\n' > "$CPU_FIXTURE_OUTPUT/iter_1.pth"
    ln -s iter_1.pth "$CPU_FIXTURE_OUTPUT/latest.pth"
    printf 'fixture config\\n' > "$CPU_FIXTURE_OUTPUT/imitation_rank0.py"
    return 0
  fi
  for arg in "$@"; do
    if [[ "$previous" == --checkpoint ]]; then checkpoint="$arg"; fi
    previous="$arg"
  done
  printf '%s\\n' "$checkpoint" > "$CPU_FIXTURE_OUTPUT/selected_checkpoint.txt"
  [[ -f "$checkpoint" ]] || return 7
}
'''
            completed = subprocess.run(['bash', '-c', prelude + shell],
                                       stdin=subprocess.DEVNULL, capture_output=True,
                                       text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(int((output / 'exit_code').read_text()), 0)
            self.assertEqual(int((output / 'train_exit_code').read_text()), 0)
            self.assertTrue((output / 'latest.pth').is_symlink())
            self.assertEqual((output / 'latest.pth').resolve(), output / 'iter_1.pth')
            self.assertFalse((output / 'iter_2.pth').exists())
            self.assertEqual(Path((output / 'selected_checkpoint.txt').read_text().strip()).name,
                             'latest.pth')
            calls.clear()
            invalid_output = base / 'invalid_output'
            invalid = argv[:]
            invalid[invalid.index('--output') + 1] = str(invalid_output)
            invalid += ['--task', 'inference', '--config', str(checkpoint)]
            previous = sys.argv[:]
            try:
                sys.argv = invalid
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught: ns['main']()
                self.assertEqual(caught.exception.code, 2)
            finally: sys.argv = previous
            self.assertEqual(calls, [])
            self.assertFalse(invalid_output.exists())


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE,
                        help='Checkout root containing ascend/ (default: this tools/validation checkout)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if args.source is None:
        parents = Path(__file__).resolve().parents
        if len(parents) <= 2:
            parser.error('--source is required when the test is outside tools/validation/')
        args.source = parents[2]
    SOURCE = args.source.resolve(strict=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidationGuards)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
