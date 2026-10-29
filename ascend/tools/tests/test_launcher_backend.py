"""Real shell frontend arguments parsed by the original Python CLI; CPU only."""
import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
BASH = (os.environ.get('WORLDENGINE_TEST_BASH') or shutil.which('bash')
        or ('C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else None))
if BASH and not Path(BASH).is_file():
    BASH = None
spec = importlib.util.spec_from_file_location('original_container_cli', ROOT / 'ascend/run_container.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Parsed(BaseException):
    def __init__(self, args):
        self.parsed_args = args


@unittest.skipUnless(BASH, 'A real Bash executable is required for frontend CPU checks')
class BackendArguments(unittest.TestCase):
    def frontend(self, task, extra=(), backend=None):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Path(tmp) / 'record_arguments.sh'
            recorder.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n')
            recorder.chmod(0o700)
            recorder_path = str(recorder)
            if os.name == 'nt':
                recorder_path = subprocess.check_output(
                    [BASH, '-c', 'cygpath -u "$1"', 'fixture', str(recorder)], text=True).strip()
            env = dict(os.environ)
            for key in ('DEVICE_IDS', 'DATA_ROOT', 'MINI_PREPARED', 'MSDA_BACKEND',
                        'RARE_SPLIT', 'SYNTHETIC_ROOT', 'MAPS', 'METRIC_CACHE'):
                env.pop(key, None)
            env.update(IMAGE='fixture/image', IMAGE_ID='sha256:fixture',
                       VOCAB='/fixtures/vocab.npy', RUN_OUTPUT='/fixtures/fresh_output',
                       CPU_MEMORY='64g', DATA_ROOT='/fixtures/full',
                       INIT_CHECKPOINT='/fixtures/initial.pth', PYTHON_BIN=recorder_path,
                       MSYS2_ARG_CONV_EXCL='*')
            if backend is not None:
                env['MSDA_BACKEND'] = backend
            script = ROOT / ('projects/AlgEngine/scripts/e2e_dist_%s_npu.sh' %
                             ('train' if task == 'train' else 'eval'))
            if task == 'train':
                args = ['configs/worldengine/e2e_vadv2_100pct.py', '2', '--']
            else:
                args = ['configs/worldengine/e2e_vadv2_100pct.py', '/fixtures/initial.pth', '2', '--']
            return subprocess.run([BASH, script.as_posix(), *args, *extra], cwd=ROOT,
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def parse_original_cli(self, completed):
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        argv = [item.decode() for item in completed.stdout.split(b'\0') if item]
        self.assertTrue(argv[0].endswith('/ascend/run_container.py'), argv)
        original_parse = argparse.ArgumentParser.parse_args

        def capture(parser, *unused, **kwargs):
            raise Parsed(original_parse(parser, argv[1:]))

        # Run the actual main's parser, then stop before any Docker/filesystem action.
        with patch.object(argparse.ArgumentParser, 'parse_args', capture), \
             patch.object(launcher.subprocess, 'check_output') as inspect_image, \
             patch.object(launcher.subprocess, 'run') as run_container:
            try:
                launcher.main()
            except Parsed as result:
                parsed = result.parsed_args
            else:
                self.fail('Original CLI parser was not called')
            inspect_image.assert_not_called()
            run_container.assert_not_called()
        return parsed

    def test_default_reference_for_train_and_inference(self):
        for task in ('train', 'inference'):
            with self.subTest(task=task):
                args = self.parse_original_cli(self.frontend(task))
                self.assertEqual((args.task, args.msda_backend, args.precision),
                                 (task, 'reference', 'fp32'))
                self.assertEqual(args.devices, '0,1')

    def test_environment_selects_backend_for_both_frontends(self):
        for task in ('train', 'inference'):
            with self.subTest(task=task):
                args = self.parse_original_cli(self.frontend(task, backend='cuda-compatible'))
                self.assertEqual(args.msda_backend, 'cuda-compatible')

    def test_explicit_backend_and_equals_override_environment(self):
        for task in ('train', 'inference'):
            for extra, env, expected in (
                    (('--msda-backend', 'cuda-compatible'), 'reference', 'cuda-compatible'),
                    (('--msda-backend=reference',), 'cuda-compatible', 'reference')):
                with self.subTest(task=task, extra=extra):
                    args = self.parse_original_cli(self.frontend(task, extra, env))
                    self.assertEqual(args.msda_backend, expected)

    def test_valid_explicit_backend_overrides_invalid_environment(self):
        for task in ('train', 'inference'):
            for extra, expected in ((('--msda-backend', 'cuda-compatible'), 'cuda-compatible'),
                                    (('--msda-backend=reference',), 'reference')):
                with self.subTest(task=task, extra=extra):
                    args = self.parse_original_cli(self.frontend(task, extra, 'invalid'))
                    self.assertEqual(args.msda_backend, expected)

    def test_invalid_environment_and_abbreviations_stop_in_shell(self):
        for task in ('train', 'inference'):
            for extra, backend in (((), 'native'), (('--msda-back', 'reference'), None)):
                with self.subTest(task=task, extra=extra, backend=backend):
                    result = self.frontend(task, extra, backend)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, b'')

    def test_invalid_explicit_backend_is_rejected_by_original_cli(self):
        for task in ('train', 'inference'):
            for backend in (None, 'invalid'):
                with self.subTest(task=task, backend=backend), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.parse_original_cli(self.frontend(task, ('--msda-backend', 'native'), backend))
                    self.assertEqual(caught.exception.code, 2)

    def test_inference_rejects_every_training_only_option_before_python(self):
        options = {
            '--profile': 'imitation', '--train-filter': '/fixtures/navtrain.yaml', '--lr': '0.0002',
            '--steps': '8', '--epochs': '8', '--resume': '/fixtures/resume.pth', '--stop-after': '4',
            '--stop-after-epochs': '2', '--duration-seconds': '20', '--checkpoint-interval': '4',
            '--check-interval': '1', '--validate': None, '--inference-after': None,
            '--runtime': None, '--upstream-schedule': None,
        }
        for option, value in options.items():
            forms = ((option,),) if value is None else ((option, value), (option + '=' + value,))
            for extra in forms:
                with self.subTest(extra=extra):
                    result = self.frontend('inference', extra)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, b'')
                    self.assertIn(b'Training-only option', result.stderr)

    def test_inference_accepts_only_applied_host_and_execution_options(self):
        extra = ('--pids-limit', '4096', '--shm-size=16g', '--deterministic',
                 '--debug-anomaly', '--detach', '--msda-backend=cuda-compatible')
        args = self.parse_original_cli(self.frontend('inference', extra))
        self.assertEqual((args.pids_limit, args.shm_size, args.msda_backend),
                         (4096, '16g', 'cuda-compatible'))
        self.assertTrue(args.deterministic and args.debug_anomaly and args.detach)

    def test_training_options_remain_available(self):
        args = self.parse_original_cli(self.frontend('train', (
            '--profile', 'imitation', '--lr', '0.0002', '--steps', '8',
            '--resume', '/fixtures/resume.pth', '--stop-after', '4',
            '--duration-seconds', '20', '--checkpoint-interval', '4', '--check-interval', '1',
            '--validate', '--inference-after', '--deterministic')))
        self.assertEqual((args.task, args.steps, args.stop_after, args.checkpoint_interval),
                         ('train', 8, 4, 4))
        self.assertEqual(str(args.resume).replace('\\', '/'), '/fixtures/resume.pth')

    def test_inference_still_rejects_protected_and_abbreviated_options(self):
        for extra in (('--checkpoint', '/other.pth'), ('--devices=1',),
                      ('--prec', 'bf16'), ('--msda-back', 'reference')):
            with self.subTest(extra=extra):
                result = self.frontend('inference', extra)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b'')


if __name__ == '__main__':
    unittest.main()
