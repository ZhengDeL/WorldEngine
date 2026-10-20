"""CPU fixtures for evidence integrity; no model, NPU, map or rendering acceptance."""
import ast
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from simengine_ascend.consumption_audit import ConsumptionAudit, ENVIRONMENT
from simengine_ascend.file_protocol import atomic_numpy, atomic_pickle
from simengine_ascend.closed_loop_runtime import ClosedLoopRuntime, Float32InferenceEvidence


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'projects/SimEngine/worldengine'


def original_methods(relative, name, methods, namespace):
    """Execute actual method ASTs without importing the simulator's heavy stack."""
    path = ENGINE / relative
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    original = next(row for row in parsed.body if isinstance(row, ast.ClassDef) and row.name == name)
    selected = [row for row in original.body if isinstance(row, ast.FunctionDef) and row.name in methods]
    if len(selected) != len(methods):
        raise AssertionError('Original fixture method selection is incomplete')
    module = ast.Module(body=selected, type_ignores=[])
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return {method: scope[method] for method in methods}


class Action:
    def __init__(self, offset=0):
        self.waypoints = np.arange(18, dtype=np.float64).reshape(9, 2) + offset
        self.velocities = np.arange(18, dtype=np.float64).reshape(9, 2) / 10
        self.headings = np.arange(9, dtype=np.float64) / 30
        self.angular_velocities = np.arange(9, dtype=np.float64) / 40


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'WE_output'
        self.output.mkdir()
        self.frames = self.root / 'frames'
        self.frames.mkdir()
        self.plans = self.root / 'plans'
        self.plans.mkdir()
        self.config = dict(data_output_dir=str(self.output / 'openscene_format'), use_planner_actions=True,
                           ego_policy='env_input_policy', ego_controller='log_play_controller',
                           num_history=4, decision_repeat=1)
        self.agent = SimpleNamespace(current_position=np.array([0., 0.]), current_velocity=np.array([0., 0.]),
                                     current_heading=0., current_angular_velocity=0., navigation=None, _traj_step=0)
        self.agent.rear_vehicle = SimpleNamespace(current_position=np.array([0., 0.]))
        self.audit = ConsumptionAudit(self.output / 'consumption_audit', self.agent, self.config)
        self.engine = SimpleNamespace(episode_step=3, external_actions=None,
                                     current_scene={'id': 'official-scene-abc'},
                                     agent_manager=SimpleNamespace(ego_agent=self.agent))
        self.engine.managers = {'agent_manager': self.engine.agent_manager}
        policy_methods = original_methods('components/agents/policy/env_input_policy.py', 'EnvInputPolicy',
                                         ['act'], {'Trajectory': Action})
        Policy = type('EnvInputPolicy', (), policy_methods)
        self.agent.policy = Policy()
        self.agent.policy.engine = self.engine
        self.agent.policy.is_current_step_valid = True
        controller_methods = original_methods('components/agents/controller/log_play_controller.py', 'LogPlayController',
                                             ['step'], {})
        Controller = type('LogPlayController', (), controller_methods)
        self.agent.controller = Controller()
        self.agent.controller.agent = self.agent
        self.agent.set_position = lambda value: setattr(self.agent, 'current_position', value)
        self.agent.set_velocity = lambda value, in_local_frame: setattr(self.agent, 'current_velocity', value * 1)
        self.agent.set_heading_theta = lambda value: setattr(self.agent, 'current_heading', value)
        self.agent.set_angular_velocity = lambda value: setattr(self.agent, 'current_angular_velocity', value)
        self.agent.set_acceleration = lambda value: None
        ego_step = original_methods('components/agents/ego_agent.py', 'EgoAgent', ['step'], {})['step']
        def step(count):
            self.engine.episode_step += 1
            for _ in range(count):
                ego_step(self.agent)
        self.engine.step = step
        self.engine.before_step = lambda actions: (setattr(self.engine, 'external_actions', actions) or {})
        self.engine.after_step = lambda: {}
        env_methods = original_methods('envs/base_env.py', 'BaseEnv', ['step', '_step_original'],
                                      {'nullcontext': nullcontext, 'merge_dicts': lambda first, second, **kw: dict(first, **second)})
        Env = type('BaseEnv', (), env_methods)
        self.env = Env()
        self.env.config = SimpleNamespace(use_planner_actions=True, decision_repeat=1)
        self.env.engine = self.engine
        self.agent.client = SimpleNamespace(consumption_audit=self.audit)
        def step_return(actions, engine_info):
            self.publish_frame(self.engine.episode_step)
            return 'unchanged-original-return'
        self.env._get_step_return = step_return
        for index in range(4):
            self.publish_frame(index)

    def tearDown(self):
        self.temporary.cleanup()

    def publish_frame(self, index):
        camera_names = ('CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2')
        row = dict(frame_idx=index, token='token-%d' % index, timestamp=1000000 + index * 500000,
                   scene_token='abc', log_token='abc', scene_name='official-scene-abc', ego2global=np.eye(4),
                   cams={camera: {'data_path': 'abc/%s/%d.jpg' % (camera, index)} for camera in camera_names})
        for camera, value in row['cams'].items():
            path = Path(self.config['data_output_dir']) / 'sensor_blobs' / value['data_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture-bytes-%s-%d' % (camera, index)).encode())
        path = self.frames / ('abc_%d.pkl' % index)
        atomic_pickle(path, row)
        self.audit.frame_published(index, path, row)

    def bind(self, index=4):
        path = self.plans / ('abc_%d.npy' % index)
        atomic_numpy(path, np.arange(120, dtype=np.float32).reshape(40, 3), replace=False)
        value, read = self.audit.read_plan(path, .1)
        self.assertEqual(read['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        np.testing.assert_array_equal(value, np.arange(120, dtype=np.float32).reshape(40, 3))
        action = Action(index)
        self.audit.bind_plan(index - 1, 'abc', read, action)
        return action

    def receipts(self):
        return sorted(self.audit.directory.glob('consumed_*.json'))

    def test_actual_original_policy_ego_controller_and_env_eight_request_chain(self):
        previous = None
        policy_method, controller_method = self.agent.policy.act.__func__, self.agent.controller.step.__func__
        for index in range(4, 12):
            action = self.bind(index)
            self.assertEqual(self.env.step(action), 'unchanged-original-return')
            path = self.audit.directory / ('consumed_%06d.json' % index)
            row = json.loads(path.read_text())
            self.assertEqual((row['request_step'], row['control_index'], row['successor_step']), (index - 1, index, index))
            self.assertEqual([frame['frame_idx'] for frame in row['input_frames']], list(range(index - 4, index)))
            self.assertEqual(row['previous_receipt_sha256'], previous)
            self.assertTrue(row['controller_application']['exact_state_update_verified'])
            self.assertEqual(row['actual_calls'], {'policy': 1, 'controller': 1})
            np.testing.assert_array_equal(self.agent.current_position, action.waypoints[1])
            previous = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(len(self.receipts()), 8)
        self.assertEqual(len(self.audit.frames), 12)
        self.assertIs(self.agent.policy.act.__func__, policy_method)
        self.assertIs(self.agent.controller.step.__func__, controller_method)

    def test_other_action_identity_is_rejected(self):
        self.bind()
        with self.assertRaises(ValueError):
            self.env.step(Action(4))
        self.assertEqual(self.receipts(), [])

    def test_policy_return_substitution_is_rejected_and_methods_restored(self):
        action = self.bind()
        replacement = lambda: Action(4)
        self.agent.policy.act = replacement
        with self.assertRaises(ValueError):
            self.env.step(action)
        self.assertIs(self.agent.policy.act, replacement)
        self.assertNotIn('step', self.agent.controller.__dict__)
        self.assertEqual(self.receipts(), [])

    def test_controller_noop_does_not_count_as_application(self):
        action = self.bind()
        self.agent.controller.step = lambda: None
        with self.assertRaisesRegex(ValueError, 'did not apply'):
            self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_controller_exception_does_not_publish_success(self):
        action = self.bind()
        def fail():
            raise RuntimeError('original controller failed')
        self.agent.controller.step = fail
        with self.assertRaisesRegex(RuntimeError, 'original controller failed'):
            self.env.step(action)
        self.assertIs(self.agent.controller.step, fail)
        self.assertEqual(self.receipts(), [])

    def test_missing_successor_metadata_is_rejected(self):
        action = self.bind()
        self.env._get_step_return = lambda *args, **kw: None
        with self.assertRaisesRegex(ValueError, 'No real successor'):
            self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_changed_plan_bytes_are_rejected_after_application(self):
        action = self.bind()
        path = self.plans / 'abc_4.npy'
        atomic_numpy(path, np.ones((40, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'changed before'):
            self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_changed_source_is_rejected(self):
        action = self.bind()
        with mock.patch('simengine_ascend.consumption_audit._sources', return_value={}):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_changed_camera_image_is_rejected(self):
        action = self.bind()
        image = self.audit.frames[1]['cameras']['CAM_B0']
        Path(image['path']).write_bytes(b'changed-image')
        with self.assertRaisesRegex(ValueError, 'camera image changed'):
            self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_duplicate_receipt_is_never_replaced(self):
        action = self.bind()
        path = self.audit.directory / 'consumed_000004.json'
        path.write_text('existing-owner')
        with self.assertRaises(FileExistsError):
            self.env.step(action)
        self.assertEqual(path.read_text(), 'existing-owner')

    def test_explicit_stop_and_missing_plan_timeout(self):
        with self.assertRaises(TimeoutError):
            self.audit.read_plan(self.plans / 'absent.npy', .001)
        stop = self.root / 'STOP'
        stop.touch()
        self.config['planner_stop_file'] = str(stop)
        with self.assertRaises(InterruptedError):
            self.audit.read_plan(self.plans / 'absent.npy', 1)

    def test_python39_path_stat_interface_reads_the_exact_plan_bytes(self):
        class LegacyPath(type(Path())):
            # Python 3.9 Path.stat has no follow_symlinks keyword. lstat is
            # its portable API for the required no-follow identity check.
            def stat(self):
                return os.stat(self)
            def lstat(self):
                return os.lstat(self)
        path = self.plans/'legacy.npy'
        atomic_numpy(path, np.zeros((40, 3), dtype=np.float32))
        with mock.patch('simengine_ascend.consumption_audit.Path', LegacyPath):
            value, row = self.audit.read_plan(path, .1)
        self.assertEqual(value.shape, (40, 3))
        self.assertEqual(row['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_bad_shape_nonfinite_and_trailing_bytes_are_rejected(self):
        for name, array, trailing in [('shape', np.zeros((8, 3)), b''),
                                      ('nan', np.full((40, 3), np.nan), b''),
                                      ('trailing', np.zeros((40, 3)), b'x')]:
            with self.subTest(name=name):
                path = self.plans / (name + '.npy')
                atomic_numpy(path, array)
                if trailing:
                    with path.open('ab') as stream:
                        stream.write(trailing)
                with self.assertRaises(ValueError):
                    self.audit.read_plan(path, .1)

    def test_default_disabled_fresh_and_contract_guards(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ConsumptionAudit.from_environment(self.agent, self.config))
        with self.assertRaises(FileExistsError):
            ConsumptionAudit(self.output / 'consumption_audit', self.agent, self.config)
        with self.assertRaises(ValueError):
            ConsumptionAudit(self.root / 'foreign', self.agent, self.config)
        with self.assertRaises(ValueError):
            ConsumptionAudit(self.output / '..' / 'consumption_audit', self.agent, self.config)

    def test_missing_or_duplicate_input_frames_and_skipped_control_rejected(self):
        with self.assertRaises(ValueError):
            self.publish_frame(3)
        with self.assertRaises(ValueError):
            self.bind(5)

    def test_changed_controller_trajectory_is_rejected(self):
        action = self.bind()
        action.waypoints[1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'unchanged audited trajectory'):
            self.env.step(action)
        self.assertEqual(self.receipts(), [])

    def test_failed_audit_cannot_resume_in_place(self):
        action = self.bind()
        self.agent.controller.step = lambda: None
        with self.assertRaises(ValueError):
            self.env.step(action)
        with self.assertRaisesRegex(RuntimeError, 'cannot be resumed'):
            self.audit.read_plan(self.plans / 'abc_4.npy', .1)


class Float32ObserverTest(unittest.TestCase):
    """Observer-contract mocks; these do not constitute real NPU activation evidence."""
    def setUp(self):
        class Tensor:
            def __init__(self, dtype='torch.float32', device='npu:0'):
                self.dtype, self.device, self.shape = dtype, device, (1, 4)
            def is_floating_point(self):
                return True
        class Module:
            def __init__(self):
                self.handles, self.training = [], False
            def register_forward_hook(self, function):
                handle = SimpleNamespace(active=True)
                handle.remove = lambda: setattr(handle, 'active', False)
                handle.function = function
                self.handles.append(handle)
                return handle
            def emit(self, value):
                for handle in self.handles:
                    if handle.active:
                        self_result = handle.function(self, (), value)
                        if self_result is not None:
                            raise AssertionError('Observer replaced a native Tensor result')
                return value
        self.Conv2d, self.Linear = type('Conv2d', (Module,), {}), type('Linear', (Module,), {})
        self.conv, self.linear = self.Conv2d(), self.Linear()
        self.Tensor = Tensor
        self.parameter = Tensor()
        self.model = SimpleNamespace(parameters=lambda: iter([self.parameter]),
                                     named_modules=lambda: iter([('conv', self.conv), ('linear', self.linear)]))
        self.autocast = False
        self.torch = SimpleNamespace(nn=SimpleNamespace(Conv2d=self.Conv2d, Linear=self.Linear),
                                     is_tensor=lambda value: isinstance(value, Tensor),
                                     is_autocast_enabled=lambda device: self.autocast,
                                     get_autocast_dtype=lambda device: 'torch.bfloat16')

    def all_removed(self):
        return all(not item.active for module in (self.conv, self.linear) for item in module.handles)

    def test_two_actual_observations_preserve_original_objects_and_close_hooks(self):
        observer = Float32InferenceEvidence(self.model, self.torch, 0)
        values = [self.Tensor(), self.Tensor()]
        self.assertIs(self.conv.emit(values[0]), values[0])
        self.assertIs(self.linear.emit(values[1]), values[1])
        report = observer.finish()
        self.assertEqual(set(report['first_observed_fp32_activations']), {'Conv2d', 'Linear'})
        self.assertEqual(report['master_parameters']['dtypes'], {'torch.float32': 1})
        self.assertTrue(self.all_removed())

    def test_bf16_or_cpu_master_parameters_are_rejected(self):
        for dtype, device in [('torch.bfloat16', 'npu:0'), ('torch.float32', 'cpu')]:
            self.parameter.dtype, self.parameter.device = dtype, device
            with self.assertRaises(RuntimeError):
                Float32InferenceEvidence(self.model, self.torch, 0)

    def test_wrong_activation_dtype_or_device_rejected_with_all_hooks_removed(self):
        for dtype, device in [('torch.bfloat16', 'npu:0'), ('torch.float32', 'cpu')]:
            with self.subTest(dtype=dtype, device=device):
                observer = Float32InferenceEvidence(self.model, self.torch, 0)
                with self.assertRaises(RuntimeError):
                    self.conv.emit(self.Tensor(dtype, device))
                self.assertTrue(self.all_removed())
                observer.close()

    def test_ambient_autocast_and_training_are_rejected(self):
        observer = Float32InferenceEvidence(self.model, self.torch, 0)
        self.autocast = True
        with self.assertRaises(RuntimeError):
            self.conv.emit(self.Tensor())
        self.assertTrue(self.all_removed())
        self.autocast, self.conv.training = False, True
        observer = Float32InferenceEvidence(self.model, self.torch, 0)
        with self.assertRaises(RuntimeError):
            self.conv.emit(self.Tensor())
        self.assertTrue(self.all_removed())

    def test_missing_linear_or_changed_parameters_cannot_finish(self):
        observer = Float32InferenceEvidence(self.model, self.torch, 0)
        self.conv.emit(self.Tensor())
        with self.assertRaises(RuntimeError):
            observer.finish()
        self.assertTrue(self.all_removed())
        observer = Float32InferenceEvidence(self.model, self.torch, 0)
        self.conv.emit(self.Tensor())
        self.linear.emit(self.Tensor())
        self.parameter.dtype = 'torch.bfloat16'
        with self.assertRaises(RuntimeError):
            observer.finish()

    def test_runtime_context_removes_pending_hooks_on_unrelated_model_error(self):
        runtime = ClosedLoopRuntime(SimpleNamespace(device='cuda:0', precision=None), ROOT)
        runtime.fp32_evidence = Float32InferenceEvidence(self.model, self.torch, 0)
        with self.assertRaisesRegex(RuntimeError, 'original forward error'):
            with runtime.fp32_context():
                raise RuntimeError('original forward error')
        self.assertTrue(self.all_removed())


if __name__ == '__main__':
    unittest.main()
