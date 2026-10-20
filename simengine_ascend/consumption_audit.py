"""Opt-in CPU evidence for the original planner -> policy -> controller path.

This records a configured control window, not driving quality or GPU parity.
The original trajectory conversion, policy return and controller are unchanged.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import time

import numpy as np

from .file_protocol import atomic_write, check_stop, validate_trajectory


ENVIRONMENT = 'WORLDENGINE_CLOSED_LOOP_AUDIT_DIR'
_ATTRIBUTES = ('waypoints', 'velocities', 'headings', 'angular_velocities')
_CAMERAS = {'CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2'}
_MISSING = object()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _publish(path, record):
    payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    atomic_write(path, lambda stream: stream.write(payload), replace=False)
    return hashlib.sha256(payload).hexdigest()


def _image(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Camera path must remain inside this job sensor_blobs directory')
    path = root / relative
    for candidate in (path,) + tuple(path.parents):
        if candidate.is_symlink():
            raise ValueError('Camera evidence must not traverse symlinks')
        if candidate == root:
            break
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('Real successor camera file is absent or outside this job')
    return dict(data_path=relative.as_posix(), path=str(path), bytes=path.stat().st_size, sha256=_sha(path))


def _finite(value):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('Consumption evidence requires finite numeric values')
    return array.tolist()


def _trajectory(value):
    result = {name: _finite(getattr(value, name)) for name in _ATTRIBUTES}
    if (np.asarray(result['waypoints']).shape != (9, 2)
            or np.asarray(result['velocities']).shape != (9, 2)
            or np.asarray(result['headings']).shape != (9,)
            or np.asarray(result['angular_velocities']).shape != (9,)):
        raise ValueError('Original 40-pose/0.1s conversion must yield nine controller states')
    return result


def _state(agent):
    return dict(position=_finite(agent.current_position), heading=_finite(agent.current_heading),
                velocity=_finite(agent.current_velocity), angular_velocity=_finite(agent.current_angular_velocity),
                rear_position=_finite(agent.rear_vehicle.current_position))


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = [
        'simengine_ascend/consumption_audit.py', 'simengine_ascend/file_protocol.py',
        'projects/SimEngine/worldengine/components/agents/client/navformer_client.py',
        'projects/SimEngine/worldengine/components/agents/planner/outside_planner.py',
        'projects/SimEngine/worldengine/components/agents/policy/env_input_policy.py',
        'projects/SimEngine/worldengine/components/agents/ego_agent.py',
        'projects/SimEngine/worldengine/components/agents/controller/log_play_controller.py',
        'projects/SimEngine/worldengine/engine/base_engine.py',
        'projects/SimEngine/worldengine/envs/base_env.py',
        'projects/SimEngine/worldengine/manager/data_manager.py',
        'projects/SimEngine/worldengine/manager/render_manager.py',
    ]
    return {name: _sha(root / name) for name in names}


class ConsumptionAudit:
    @classmethod
    def from_environment(cls, agent, config):
        directory = os.environ.get(ENVIRONMENT)
        return cls(directory, agent, config) if directory else None

    def __init__(self, directory, agent, config):
        path = Path(directory)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Consumption audit directory must be a regular absolute path')
        for parent in (path,) + tuple(path.parents):
            if parent.is_symlink():
                raise ValueError('Consumption audit directory cannot contain a symlink')
        expected = Path(config['data_output_dir']).resolve().parent / 'consumption_audit'
        if path.resolve() != expected:
            raise ValueError('Consumption audit must be inside this job output/consumption_audit')
        if (not config['use_planner_actions'] or config['ego_policy'] != 'env_input_policy'
                or config['ego_controller'] != 'log_play_controller'
                or int(config['num_history']) != 4 or int(config['decision_repeat']) != 1):
            raise ValueError('Consumption audit supports the original four-history/log-play closed-loop contract')
        path.mkdir()  # Explicitly fresh; neither reuse nor overwrite a prior job.
        self.directory, self.agent, self.config = path, agent, config
        self.frames, self.pending, self.previous = {}, None, None
        self.next_step, self.failed = 3, False
        self.source_files = _sources()
        self.session = dict(schema_version=1, result='instrumentation_initialized_not_acceptance',
                            process_id=os.getpid(), started_utc=_now(), source_files=self.source_files,
                            policy='env_input_policy', controller='log_play_controller',
                            num_history=4, decision_repeat=1)
        self.session_sha = _publish(path / 'session.json', self.session)

    def guard(self):
        if self.failed:
            raise RuntimeError('A failed consumption audit cannot be resumed in place')
        check_stop(self.config.get('planner_stop_file'), self.config.get('planner_deadline_utc'))

    def read_plan(self, path, timeout_seconds):
        """Hash the exact bounded byte string from which the used array is parsed."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Trajectory wait timeout must be finite and positive')
        path = Path(path)
        end = time.monotonic() + timeout_seconds
        while True:
            self.guard()
            if path.is_symlink():
                raise ValueError('Trajectory input must not be a symlink')
            try:
                descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                with os.fdopen(descriptor, 'rb') as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024 * 1024:
                        raise ValueError('Trajectory input must be a bounded regular file')
                    payload = stream.read(1024 * 1024 + 1)
                    after = os.fstat(stream.fileno())
                current = path.lstat()
                identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
                if (identity(before) != identity(after) or identity(after) != identity(current)
                        or len(payload) != after.st_size or not stat.S_ISREG(current.st_mode)):
                    raise ValueError('Trajectory file changed during the audited read')
                buffer = io.BytesIO(payload)
                version = np.lib.format.read_magic(buffer)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(buffer)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(buffer)
                else:
                    raise ValueError('Unsupported audited NPY format version')
                if shape != (40, 3) or dtype.kind != 'f' or buffer.tell() + 120 * dtype.itemsize != len(payload):
                    raise ValueError('Audited trajectory header must describe finite floating[40,3] bytes')
                buffer.seek(0)
                value = validate_trajectory(np.load(buffer, allow_pickle=False))
                if buffer.read(1):
                    raise ValueError('Unexpected trailing trajectory bytes')
                self.guard()
                return value, dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                                   shape=list(value.shape), dtype=str(value.dtype), finite=True, read_utc=_now())
            except FileNotFoundError:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Timed out waiting for trajectory: ' + str(path))
                time.sleep(min(.1, remaining))

    def frame_published(self, step, path, record):
        self.guard()
        step = int(step)
        if step in self.frames or int(record['frame_idx']) != step:
            raise ValueError('Duplicate or mismatched real frame in consumption evidence')
        if step != len(self.frames):
            raise ValueError('Real frame publication must be consecutive from zero')
        cameras = record['cams']
        if set(cameras) != _CAMERAS:
            raise ValueError('Consumption evidence requires all eight original cameras')
        image_root = Path(self.config['data_output_dir']).resolve() / 'sensor_blobs'
        image_rows = {name: _image(image_root, value['data_path']) for name, value in sorted(cameras.items())}
        if len({row['path'] for row in image_rows.values()}) != 8:
            raise ValueError('Each camera must publish its own real image file')
        row = dict(frame_idx=step, path=str(path), sha256=_sha(path), token=str(record['token']),
                   timestamp=int(record['timestamp']), scene_token=str(record['scene_token']),
                   scene_name=str(record['scene_name']), log_token=str(record['log_token']),
                   cameras=image_rows,
                   ego2global=_finite(record['ego2global']))
        if step and row['timestamp'] <= self.frames[step - 1]['timestamp']:
            raise ValueError('Published frame timestamps must increase')
        _publish(self.directory / ('frame_%06d.json' % step), row)
        self.frames[step] = row

    def bind_plan(self, step, prefix, read, trajectory):
        self.guard()
        step = int(step)
        if self.pending is not None or step != self.next_step:
            raise ValueError('A pending control request cannot be replaced')
        inputs = [self.frames[index] for index in range(step - 3, step + 1)]
        if any(row['log_token'] != prefix for row in inputs):
            raise ValueError('Planner history does not belong to the requested scene')
        if Path(read['path']).name != '%s_%d.npy' % (prefix, step + 1):
            raise ValueError('Planner filename does not match the next control index')
        self.pending = dict(step=step, prefix=prefix, plan=read, action=trajectory,
                            controller_trajectory=_trajectory(trajectory), input_frames=inputs)

    @contextmanager
    def controlled_step(self, actions, engine):
        self.guard()
        step = int(engine.episode_step)
        if step < 3:
            if self.pending is not None:
                raise ValueError('Unexpected model plan during history replay')
            yield
            return
        pending = self.pending
        if pending is None or pending['step'] != step or pending['action'] is not actions:
            raise ValueError('BaseEnv action is not the actual audited client trajectory')
        agent = self.agent
        if engine.agent_manager.ego_agent is not agent:
            raise ValueError('Consumption audit attached to a different ego agent')
        policy, controller = agent.policy, agent.controller
        calls = {'policy': 0, 'controller': 0}
        before = _state(agent)
        old_policy, old_controller = policy.__dict__.get('act', _MISSING), controller.__dict__.get('step', _MISSING)
        original_policy, original_controller = policy.act, controller.step
        application = {}
        def policy_act(*args, **kwargs):
            value = original_policy(*args, **kwargs)
            if value is not actions or engine.external_actions is not actions:
                raise ValueError('Original EnvInputPolicy did not return the audited external action')
            calls['policy'] += 1
            return value
        def controller_step(*args, **kwargs):
            if (agent.trajectory is not actions or engine.external_actions is not actions
                    or calls['policy'] != calls['controller'] + 1
                    or _trajectory(agent.trajectory) != pending['controller_trajectory']):
                raise ValueError('Original controller did not receive the unchanged audited trajectory')
            result = original_controller(*args, **kwargs)
            # LogPlayController assigns sample 1 (0.5s) of this exact trajectory.
            # Check its actual state update, rather than counting entry alone.
            desired = dict(position=pending['controller_trajectory']['waypoints'][1],
                           velocity=pending['controller_trajectory']['velocities'][1],
                           heading=pending['controller_trajectory']['headings'][1],
                           angular_velocity=pending['controller_trajectory']['angular_velocities'][1])
            actual = _state(agent)
            if any(actual[name] != target for name, target in desired.items()):
                raise ValueError('Original log-play controller did not apply the requested sample')
            application.update(requested=desired, actual=actual, exact_state_update_verified=True)
            calls['controller'] += 1  # Only a successfully returned original controller counts.
            return result
        policy.act, controller.step = policy_act, controller_step
        succeeded = False
        try:
            yield
            self.guard()
            if (calls != {'policy': 1, 'controller': 1} or int(engine.episode_step) != step + 1
                    or engine.external_actions is not actions or agent.trajectory is not actions):
                raise ValueError('Original policy/controller/episode transition did not complete exactly once')
            successor = self.frames.get(step + 1)
            if successor is None or successor['log_token'] != pending['prefix']:
                raise ValueError('No real successor metadata was published after control')
            if (_sha(pending['plan']['path']) != pending['plan']['sha256']
                    or any(_sha(row['path']) != row['sha256'] for row in pending['input_frames'] + [successor])):
                raise ValueError('Audited plan or real frame changed before control completion')
            for frame in pending['input_frames'] + [successor]:
                for image in frame['cameras'].values():
                    self.guard()
                    if _image(Path(self.config['data_output_dir']).resolve() / 'sensor_blobs', image['data_path']) != image:
                        raise ValueError('Audited camera image changed before control completion')
            if _sources() != self.source_files:
                raise ValueError('Closed-loop consumption source changed during the step')
            if policy.__dict__.get('act') is not policy_act or controller.__dict__.get('step') is not controller_step:
                raise RuntimeError('Original method changed during consumption audit')
            row = dict(schema_version=1, result='consumed_by_original_policy_and_controller',
                       request_step=step, control_index=step + 1, successor_step=step + 1,
                       scene_id=str(engine.current_scene['id']), prefix=pending['prefix'], plan=pending['plan'],
                       input_frames=pending['input_frames'], successor_frame=successor,
                       controller_trajectory=pending['controller_trajectory'], ego_state_before=before,
                       ego_state_after=_state(agent), controller_application=application, actual_calls=calls,
                       actual_policy=type(policy).__module__ + '.' + type(policy).__qualname__,
                       actual_controller=type(controller).__module__ + '.' + type(controller).__qualname__,
                       policy_return_identity_verified=True, controller_input_identity_verified=True,
                       session_sha256=self.session_sha, previous_receipt_sha256=self.previous, completed_utc=_now(),
                       quality_accepted=False)
            self.previous = _publish(self.directory / ('consumed_%06d.json' % (step + 1)), row)
            self.next_step += 1
            succeeded = True
        finally:
            if not succeeded:
                self.failed = True
            for instance, name, original, observed in ((policy, 'act', old_policy, policy_act),
                                                       (controller, 'step', old_controller, controller_step)):
                if instance.__dict__.get(name) is observed:
                    if original is _MISSING:
                        delattr(instance, name)
                    else:
                        setattr(instance, name, original)
                elif succeeded:
                    raise RuntimeError('Original method changed during consumption audit')
            # Failed steps retain no successful receipt and cannot be retried in place.
            self.pending = None
