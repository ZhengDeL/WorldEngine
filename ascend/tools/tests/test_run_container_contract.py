"""CLI regressions; Docker and character-device discovery are mocked."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('container_cli_under_test', Path(__file__).resolve().parents[2] / 'run_container.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ContainerCLIContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('initial.pth', 'vocab.npy', 'config.py', 'resume.pth'):
            (self.root / name).touch()
        (self.root / 'mini').mkdir()

    def invoke(self, task, options=(), timeout=None):
        output = self.root / ('output_' + str(len(list(self.root.glob('output_*')))))
        argv = ['run_container.py', '--task', task, '--devices', '0',
                '--output', str(output), '--checkpoint', str(self.root / 'initial.pth'),
                '--vocab', str(self.root / 'vocab.npy'), '--mini-prepared', str(self.root / 'mini')]
        argv += ['--config', str(self.root / 'config.py')] if task == 'inference' else ['--steps', '2']
        argv += list(options)
        environment = {} if timeout is None else {'HCCL_CONNECT_TIMEOUT': timeout}
        stderr = io.StringIO()
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, 'argv', argv), \
             patch.object(cli.subprocess, 'check_output', return_value=cli.IMAGE_ID + '\n') as inspect, \
             patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as docker, \
             patch.object(Path, 'is_char_device', return_value=True), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                cli.main()
        return result.exception.code, inspect, docker, output, stderr.getvalue()

    def test_inference_resume_rejected_before_docker_or_output(self):
        code, inspect, docker, output, error = self.invoke('inference', ['--resume', str(self.root / 'resume.pth')])
        self.assertEqual(code, 2)
        self.assertIn('--resume requires --task train', error)
        inspect.assert_not_called()
        docker.assert_not_called()
        self.assertFalse(output.exists())

    def test_inference_training_options_rejected_in_both_cli_forms(self):
        values = {'profile': 'imitation', 'train-filter': str(self.root / 'config.py'),
                  'rare-split': str(self.root / 'config.py'), 'synthetic-root': str(self.root / 'mini'),
                  'steps': '1', 'batch-size': '1', 'workers': '0', 'lr': '0.001', 'epochs': '1',
                  'stop-after': '1', 'stop-after-epochs': '1', 'duration-seconds': '1',
                  'checkpoint-interval': '1', 'check-interval': '1'}
        for name, value in values.items():
            for options in (['--' + name, value], ['--' + name + '=' + value]):
                with self.subTest(options=options):
                    code, inspect, docker, output, error = self.invoke('inference', options)
                    self.assertEqual(code, 2)
                    self.assertIn('--' + name + ' requires --task train', error)
                    inspect.assert_not_called()
                    docker.assert_not_called()
                    self.assertFalse(output.exists())
        for name in ('validate', 'inference-after', 'runtime', 'upstream-schedule'):
            with self.subTest(name=name):
                code, inspect, docker, output, error = self.invoke('inference', ['--' + name])
                self.assertEqual(code, 2)
                self.assertIn('--' + name + ' requires --task train', error)
                inspect.assert_not_called()
                docker.assert_not_called()
                self.assertFalse(output.exists())

    def test_training_resume_and_default_profile_preserved(self):
        code, inspect, docker, output, error = self.invoke('train', ['--resume', str(self.root / 'resume.pth')])
        self.assertEqual((code, error), (0, ''))
        launch = json.loads((output / 'launch.json').read_text())
        self.assertEqual(launch['profile'], 'imitation')
        self.assertIn('--resume /weights/resume.pth', docker.call_args.args[0][-1])

    def test_inference_uses_explicit_checkpoint_and_backend(self):
        code, inspect, docker, output, error = self.invoke('inference', ['--msda-backend', 'cuda-compatible', '--deterministic'])
        self.assertEqual((code, error), (0, ''))
        command = docker.call_args.args[0]
        self.assertIn(str((self.root / 'initial.pth').resolve()) + ':/weights/initial.pth:ro', command)
        self.assertIn('ascend/inference.py', command[-1])
        self.assertIn('--checkpoint /weights/initial.pth', command[-1])
        self.assertIn('--msda-backend cuda-compatible --deterministic', command[-1])
        self.assertNotIn('--resume', command[-1])

    def test_timeout_override_is_forwarded_and_recorded(self):
        for task in ('train', 'inference'):
            with self.subTest(task=task):
                code, inspect, docker, output, error = self.invoke(task, timeout='1800')
                self.assertEqual((code, error), (0, ''))
                command = docker.call_args.args[0]
                self.assertEqual([x for x in command if x.startswith('HCCL_CONNECT_TIMEOUT=')], ['HCCL_CONNECT_TIMEOUT=1800'])
                self.assertEqual(json.loads((output / 'launch.json').read_text())['hccl_connect_timeout'], '1800')

    def test_timeout_default_is_unchanged(self):
        code, inspect, docker, output, error = self.invoke('train')
        self.assertEqual(code, 0)
        self.assertIn('HCCL_CONNECT_TIMEOUT=120', docker.call_args.args[0])

    def test_invalid_timeout_rejected_before_docker(self):
        for value in ('', '0', '-1', '1.5', 'NaN', '\u0661\u0662'):
            with self.subTest(value=value):
                code, inspect, docker, output, error = self.invoke('train', timeout=value)
                self.assertEqual(code, 2)
                self.assertIn('positive integer in seconds', error)
                inspect.assert_not_called()
                docker.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
