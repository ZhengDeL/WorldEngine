"""CPU audit of one official real 24-frame FP32 NR/R planner/control window.

Run with NumPy/Pillow/PyYAML, no accelerator mounts, read-only inputs and a fresh
external JSON output. This experiment profile requires the original 100% VADv2
or HydraMDP checkpoint, original 8192-entry vocabulary, separately adapted
AlgEngine source 0488bfb7... (public Git commit 55efdef5...) and the pinned image.
The expected declared commit is explicit so original experiment records can be
checked without rewriting their historical metadata.
Two ordered host chips must be supplied explicitly; the logical mapping remains
planner npu:0 / renderer npu:1. Other model, image or source profiles need separate
review; neither the hardware expectation nor another profile is inferred.

Passing proves the explicitly configured real-frame/control window and
recorded representative precision activations. Full-log driving quality, scores,
GPU equivalence, training convergence and performance are outside this audit.
No model, renderer, subprocess or accelerator is run by the auditor itself.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import stat
import sys
import time

import numpy as np
from PIL import Image
import yaml


CAMERAS = {'CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2'}
MODELS = {
    'vadv2': ('projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py',
              'data/alg_engine/ckpts/vadv2/e2e_vadv2_100pct_ep8.pth',
              'fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a'),
    'hydramdp': ('projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py',
                 'data/alg_engine/ckpts/hydramdp/e2e_hydramdp_100pct_ep8.pth',
                 '298c403d6d4ba5ac1f4a301ad5e257f72c7c641e4daef138ff88fa30b2534f04')}
ALG_SOURCE = '0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922'
ALG_COMMIT = '55efdef54139c63ab9afce9178ebb3bd130717ef'
IMAGE = 'sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48'
VOCAB_SHA = 'cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35'
CONSUMPTION_SOURCES = {
    'simengine_ascend/consumption_audit.py', 'simengine_ascend/file_protocol.py',
    'projects/SimEngine/worldengine/components/agents/client/navformer_client.py',
    'projects/SimEngine/worldengine/components/agents/planner/outside_planner.py',
    'projects/SimEngine/worldengine/components/agents/policy/env_input_policy.py',
    'projects/SimEngine/worldengine/components/agents/ego_agent.py',
    'projects/SimEngine/worldengine/components/agents/controller/log_play_controller.py',
    'projects/SimEngine/worldengine/engine/base_engine.py', 'projects/SimEngine/worldengine/envs/base_env.py',
    'projects/SimEngine/worldengine/manager/data_manager.py', 'projects/SimEngine/worldengine/manager/render_manager.py'}


from simengine_ascend.window_contract import window_contract

def need(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    need(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), 'Not a regular nonsymlink file: '+str(path))
    return path


def sha(path, guard=lambda: None):
    h = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            guard(); h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(regular(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def within(root, relative):
    value = PurePosixPath(relative)
    need(isinstance(relative, str) and relative and not value.is_absolute()
         and '\\' not in relative and ':' not in relative and all(p not in ('', '.', '..') for p in relative.split('/')),
         'Unsafe relative evidence path')
    return regular(Path(root).joinpath(*value.parts))


def load_pickle(path, guard, expected=None):
    path = regular(path)
    need(path.stat().st_size <= 256 * 1024**2, 'Pickle exceeds the audit byte budget')
    identity = sha(path, guard)
    need(expected is None or identity == expected, 'Approved pickle SHA differs')
    with path.open('rb') as stream:
        value = pickle.load(stream)
        need(not stream.read(1), 'Trailing pickle bytes')
    guard(); need(sha(path, guard) == identity, 'Pickle changed while auditing')
    return value


def same(first, second):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray) and first.dtype == second.dtype
                and np.array_equal(first, second, equal_nan=first.dtype.kind in 'fc'))
    if isinstance(first, dict):
        return isinstance(second, dict) and set(first) == set(second) and all(same(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(same(a, b) for a, b in zip(first, second))
    if isinstance(first, (float, np.floating)) and isinstance(second, (float, np.floating)) and math.isnan(first) and math.isnan(second):
        return True
    return bool(first == second)


def finite(value, shape, message):
    result = np.asarray(value)
    need(result.shape == shape and result.dtype.kind in 'fiu' and np.isfinite(result).all(), message)
    return result


def host_mapping(values):
    need(isinstance(values, (list, tuple)) and len(values) == 2
         and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 15 for value in values)
         and values[0] != values[1], 'Expected two distinct ordered host chips in 0..15')
    return {'WORLDENGINE_HOST_DEVICE_IDS': ','.join(str(value) for value in values), 'ASCEND_RT_VISIBLE_DEVICES': '0,1'}


def validate_host_mapping(values, runtime, plan, launcher):
    expected = host_mapping(values)
    need(runtime['host_device_mapping'] == expected, 'Recorded physical/logical mapping differs from the root-supplied host devices')
    for phase, device in [('planner', 'npu:0'), ('simulation', 'npu:1')]:
        need(plan[phase].get('env', {}).get('WORLDENGINE_HOST_DEVICE_IDS') == expected['WORLDENGINE_HOST_DEVICE_IDS'],
             'Saved launch-plan host mapping differs from the root-supplied host devices')
        phase_env = plan[phase].get('env', {})
        need('ASCEND_RT_VISIBLE_DEVICES' not in phase_env or phase_env['ASCEND_RT_VISIBLE_DEVICES'] == '0,1',
             'Saved launch-plan phase overrides the required logical device order 0,1')
        command = launcher['children'][phase]['command']
        need(command.count('--device') == 1 and command[command.index('--device')+1] == device,
             'Actual owned planner/renderer command uses a different logical device')
    return expected


def npy(path, guard):
    path = regular(path)
    need(0 < path.stat().st_size <= 1024**2, 'Unbounded NPY file')
    raw = path.read_bytes(); buffer = io.BytesIO(raw)
    version = np.lib.format.read_magic(buffer)
    need(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
    header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = header(buffer)
    need(shape == (40, 3) and dtype.kind == 'f' and buffer.tell() + 120 * dtype.itemsize == len(raw), 'NPY header is not a complete floating[40,3] trajectory')
    buffer.seek(0); array = np.load(buffer, allow_pickle=False)
    finite(array, (40, 3), 'Nonfinite model trajectory'); guard()
    need(sha(path, guard) == hashlib.sha256(raw).hexdigest(), 'Trajectory changed while auditing')
    return array


def validate_precision(runtime, precision):
    need(runtime.get('backend') == 'npu' and runtime.get('device') == 'npu:0' and runtime.get('precision') == precision,
         'Actual planner device/precision differs')
    before, after = runtime['precision_before'], runtime['precision_after']
    for key in ('mode', 'public', 'msda', 'autocast_policy', 'ambient_autocast'):
        need(before.get(key) == after.get(key), 'Planner precision state changed: '+key)
    need(before['mode'] == precision and before['public'] == {'matmul_allow_hf32': False, 'conv_allow_hf32': False}, 'HF32/FP32 public policy differs')
    if precision == 'fp32':
        report = runtime.get('fp32')
        need(isinstance(report, dict) and report.get('mode') == 'fp32' and report.get('observation_only') is True
             and report.get('ambient_autocast_enabled') is False, 'Missing actual FP32 observer evidence')
        need(set(report['master_parameters']['dtypes']) == {'torch.float32'}
             and set(report['master_parameters']['devices']) == {'npu:0'}, 'Actual FP32 master parameters/device differ')
        activations = report['first_observed_fp32_activations']
        need(set(activations) == {'Conv2d', 'Linear'} and report.get('expected_activation_kinds') == ['Conv2d', 'Linear'], 'FP32 activation evidence is incomplete')
        for row in activations.values():
            need(row['dtype'] == 'torch.float32' and row['device'] == 'npu:0' and row['autocast_enabled'] is False
                 and row['training'] is False and isinstance(row['shape'], list) and row['shape']
                 and all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in row['shape']), 'Actual FP32 activation contract differs')
    else:
        report = runtime.get('bf16')
        need(isinstance(report, dict) and report.get('BF16_weight_cache_disabled') is True
             and report.get('inside_before_original_test') == {'enabled': True, 'dtype': 'torch.bfloat16'}
             and report['ambient_after_original_test']['enabled'] is False, 'Missing actual BF16 autocast/restoration evidence')
        actual = report['actual_model']
        need(actual['mode'] == 'bf16' and set(actual['master_parameter_dtypes']) == {'torch.float32'}, 'BF16 master parameters differ')
        activations = actual['first_observed_bf16_activations']
        need(set(activations) == {'Conv2d', 'Linear'}, 'BF16 activation evidence is incomplete')
        for row in activations.values():
            need(row['dtype'] == 'torch.bfloat16' and row['autocast'] == {'enabled': True, 'dtype': 'torch.bfloat16'}, 'No actual BF16 activation')
        geometry = actual['geometry_FP32']
        need(geometry.get('original_formulas_unchanged') is True and geometry.get('reference_grid_created_in_FP32') is True
             and geometry.get('first_actual'), 'No actual original FP32 camera/reference geometry')
        for row in geometry['first_actual'].values():
            need(row['autocast_inside']['enabled'] is False and set(row['output_floating_dtypes']) == {'torch.float32'}, 'Geometry FP32 island differs')
    return dict(precision=precision, actual_Conv2d_and_Linear_outputs_observed=True,
                all_layers_all_frames_dtype_claimed=False)


def audit_consumption(job, recorded_root, source_root, scene, exported, vocab, indices, guard, num_history=4, num_future=8):
    window = window_contract(num_history, num_future); frames = window['frames']; history = window['num_history']
    folder = job / 'WE_output/consumption_audit'
    expected = {'session.json'} | {'frame_%06d.json' % i for i in range(frames)} | {'consumed_%06d.json' % i for i in range(history, frames)}
    need({p.name for p in folder.iterdir()} == expected, 'Missing/extra consumption evidence; exit 0 alone is insufficient')
    session_path = within(folder, 'session.json'); session = read_json(session_path); session_sha = sha(session_path, guard)
    need(session['schema_version'] == 1 and session['num_history'] == history and session['decision_repeat'] == 1
         and session['policy'] == 'env_input_policy' and session['controller'] == 'log_play_controller', 'Consumption session contract differs')
    need(set(session['source_files']) == CONSUMPTION_SOURCES, 'Incomplete consumption source binding')
    for relative, expected_sha in session['source_files'].items():
        need(sha(within(source_root, relative), guard) == expected_sha, 'Consumption source SHA differs: '+relative)
    prefix = scene['id'].split('-')[-1]
    original_frames, frame_rows = [], []
    for step in range(frames):
        guard(); row = read_json(within(folder, 'frame_%06d.json' % step))
        path = within(job, 'frames/%s_%d.pkl' % (prefix, step))
        need(row['path'] == str(recorded_root / 'frames' / path.name) and row['sha256'] == sha(path, guard), 'Published planner-frame path/SHA differs')
        actual = load_pickle(path, guard, row['sha256']); original_frames.append(actual)
        need(row['frame_idx'] == actual['frame_idx'] == exported[step]['frame_idx'] == step
             and row['token'] == actual['token'] == exported[step]['token']
             and row['timestamp'] == actual['timestamp'] == exported[step]['timestamp']
             and row['scene_name'] == actual['scene_name'] == scene['id']
             and row['log_token'] == actual['log_token'] == prefix, 'Planner-frame metadata is not the real successor frame')
        need(np.array_equal(np.asarray(row['ego2global']), actual['ego2global'])
             and np.array_equal(actual['ego2global'], exported[step]['ego2global']), 'Planner/exported ego transforms differ')
        need(set(row['cameras']) == set(actual['cams']) == CAMERAS, 'Missing planner-frame cameras')
        for camera, image in row['cameras'].items():
            relative = image['data_path']; image_path = within(job/'WE_output/openscene_format/sensor_blobs', relative)
            need(relative == actual['cams'][camera]['data_path'] == exported[step]['cams'][camera]['data_path']
                 and image['path'] == str(recorded_root/'WE_output/openscene_format/sensor_blobs'/relative)
                 and image['bytes'] == image_path.stat().st_size and image['sha256'] == sha(image_path, guard), 'Camera content differs from the actual consumed history')
        frame_rows.append(row)
    previous, rows = None, []
    for control in range(history, frames):
        guard(); path = within(folder, 'consumed_%06d.json' % control); row = read_json(path)
        need(row['result'] == 'consumed_by_original_policy_and_controller' and row['schema_version'] == 1
             and (row['request_step'], row['control_index'], row['successor_step']) == (control-1, control, control)
             and row['scene_id'] == scene['id'] and row['prefix'] == prefix
             and row['session_sha256'] == session_sha and row['previous_receipt_sha256'] == previous, 'Broken exact control/receipt chain')
        need(row['policy_return_identity_verified'] is True and row['controller_input_identity_verified'] is True
             and row['actual_calls'] == {'policy': 1, 'controller': 1}
             and row['actual_policy'] == 'worldengine.components.agents.policy.env_input_policy.EnvInputPolicy'
             and row['actual_controller'] == 'worldengine.components.agents.controller.log_play_controller.LogPlayController', 'Original policy/controller application was not established')
        need(row['input_frames'] == frame_rows[control-history:control] and row['successor_frame'] == frame_rows[control], 'Control history/successor dependency differs')
        plan_path = within(job, 'plan_traj/%s_%d.npy' % (prefix, control)); plan = npy(plan_path, guard)
        need(row['plan']['path'] == str(recorded_root/'plan_traj'/plan_path.name)
             and row['plan']['sha256'] == sha(plan_path, guard) and row['plan']['bytes'] == plan_path.stat().st_size
             and row['plan']['shape'] == [40, 3] and row['plan']['dtype'] == str(plan.dtype) and row['plan']['finite'] is True,
             'Consumed NPY bytes/shape/dtype differ')
        need(control in indices and np.allclose(plan, vocab[indices[control]], atol=1e-6, rtol=1e-6), 'Consumed plan is not its selected official vocabulary entry')
        merged = load_pickle(within(job, 'merged_ann_files/merged_%d.pkl' % control), guard)
        need(isinstance(merged, dict) and list(merged) == ['infos'] and same(merged['infos'], original_frames[control-history:control]), 'Original model input did not use exactly four real consecutive frames')
        trajectory = row['controller_trajectory']
        for name, shape in [('waypoints', (9, 2)), ('velocities', (9, 2)), ('headings', (9,)), ('angular_velocities', (9,))]:
            finite(trajectory[name], shape, 'Nonfinite/malformed actual controller trajectory')
        wanted = dict(position=trajectory['waypoints'][1], velocity=trajectory['velocities'][1],
                      heading=trajectory['headings'][1], angular_velocity=trajectory['angular_velocities'][1])
        application = row['controller_application']
        for state in (row['ego_state_before'], row['ego_state_after'], application['actual']):
            for name, shape in [('position', (2,)), ('velocity', (2,)), ('rear_position', (2,)),
                                ('heading', ()), ('angular_velocity', ())]:
                finite(state[name], shape, 'Malformed actual ego state')
        need(application['exact_state_update_verified'] is True and application['requested'] == wanted
             and row['ego_state_after'] == application['actual']
             and all(application['actual'][name] == target for name, target in wanted.items()), 'Original log-play state update was not verified')
        previous = sha(path, guard)
        rows.append(dict(control_index=control, plan_sha256=row['plan']['sha256'], receipt_sha256=previous,
                         input_frame_indices=list(range(control-history, control)), successor_frame=control))
    return dict(consumed_control_indices=list(range(history, frames)), controls=rows,
                session_sha256=session_sha, real_frames=frames, camera_references=frames*len(CAMERAS),
                planner_consumption_established=True, quality_accepted=False)


def audit_metadata(job, scene, wrapper, guard, num_history=4, num_future=8):
    window = window_contract(num_history, num_future); frames = window['frames']
    folder = job/'WE_output'; scene_id = scene['id']; prefix = scene_id.split('-')[-1]
    need({path.name for path in (folder/'openscene_format/meta_datas').iterdir()} == {scene_id+'.pkl'}, 'Extra/missing exported scene metadata')
    rows = load_pickle(within(folder, 'openscene_format/meta_datas/'+scene_id+'.pkl'), guard)
    need(isinstance(rows, list) and len(rows) == frames, 'Exactly the configured real exported frame count is required')
    original = list(scene['metadata']['openscene_data_infos_dict'].values())
    need(len(original) >= frames and int(scene['log_length']) >= frames and set(scene['cameras']) == CAMERAS, 'Original scene/camera contract differs')
    tokens = [row['token']+'-000' for row in original[:frames]]
    expected, images, inventory = set(), [], hashlib.sha256()
    for index, row in enumerate(rows):
        guard()
        need(row['frame_idx'] == index and row['token'] == tokens[index]
             and row['timestamp'] == int(scene['base_timestamp'])+int(index*scene['sample_rate']*.05*1e6)
             and row['sample_prev'] == (tokens[index-1] if index else None)
             and row['sample_next'] == (tokens[index+1] if index+1 < frames else None)
             and row['scene_name'] == row['log_name'] == scene_id and row['scene_token'] == prefix
             and set(row['cams']) == CAMERAS, 'Exported frame coverage/identity/links differ')
        finite(row['ego2global'], (4, 4), 'Nonfinite exported ego transform')
        for camera in sorted(CAMERAS):
            source = original[index]['cams'][camera]['data_path'].split('/')[-1].split('.')[0]
            relative = prefix+'/'+camera+'/'+source+'-000.jpg'
            need(row['cams'][camera]['data_path'] == relative and relative not in expected, 'Missing/reused original camera output path')
            expected.add(relative); path = within(folder/'openscene_format/sensor_blobs', relative)
            dimensions = (int(scene['cameras'][camera]['width']), int(scene['cameras'][camera]['height']))
            with Image.open(path) as image:
                need(image.format == 'JPEG' and image.size == dimensions, 'JPEG format or original dimensions differ'); image.verify()
            with Image.open(path) as image:
                image.load(); need(image.size == dimensions, 'Full JPEG decode dimensions differ')
            images.append(dict(path='openscene_format/sensor_blobs/'+relative, bytes=path.stat().st_size, sha256=sha(path, guard)))
    root = folder/'openscene_format/sensor_blobs'
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    need(actual == expected, 'Extra/missing real JPEG files')
    for row in sorted(images, key=lambda row: row['path']):
        inventory.update(row['path'].encode()); inventory.update(bytes.fromhex(row['sha256']))
    result = dict(decoded_jpegs=frames*len(CAMERAS), total_bytes=sum(row['bytes'] for row in images), inventory_sha256=inventory.hexdigest())
    need(all(wrapper['rendered_images'].get(name) == value for name, value in result.items()), 'Full image inventory differs from wrapper evidence')
    return rows, dict(configured_frames=frames, original_frames=int(scene['log_length']),
                      simulation_time_span_seconds=(rows[-1]['timestamp']-rows[0]['timestamp'])/1e6, **result)


def validate_simulation_contract(wrapper, cfg, source, prepared, key, recorded_root, reaction, num_history=4, num_future=8):
    window = window_contract(num_history, num_future)
    need(reaction in ('NR', 'R'), 'An explicit NR/R reaction contract is required')
    need(wrapper['result'] == 'completed_original_simulation_and_output_checks' and wrapper['actual_exit'] == 0
         and wrapper['mode'] == 'closed-loop' and wrapper['reaction'] == reaction and wrapper['source'] == source
         and wrapper['scene_sha256'] == prepared['derived_scene']['sha256'] and wrapper['expected_scene_ids'] == [key],
         'Original simulation/source/scene did not complete the requested closed-loop reaction/window')
    wanted = dict(distributed_mode='SINGLE_NODE', exit_on_failure=True, enable_resume=False, num_history=window['num_history'],
                  num_future=window['num_future'], decision_repeat=1, use_planner_actions=True, ego_policy='env_input_policy',
                  ego_controller='log_play_controller', ego_client='navformer_client', ego_navigation='trajectory_navigation',
                  agent_policy='idm_policy' if reaction == 'R' else 'trajectory_policy',
                  agent_navigation='idm_navigation' if reaction == 'R' else 'trajectory_navigation',
                  with_render_manager=True, with_data_manager=True, with_metric_manager=False, with_dense_reward_manager=False)
    need(all(cfg.get(name) == value for name, value in wanted.items()), 'Actual original Hydra closed-loop reaction contract differs')
    need(cfg['data_file_path'] == wrapper['scene_pkl'] and cfg['asset_folder_path'] == wrapper['asset_root']
         and cfg['output_dir'] == str(recorded_root/'WE_output')
         and cfg['data_output_dir'] == str(recorded_root/'WE_output/openscene_format')
         and cfg['planner_data_path'] == str(recorded_root/'plan_traj')
         and cfg['planner_client_folder'] == str(recorded_root/'frames'), 'Resolved original scene/output/protocol paths differ')
    need(cfg['renderer']['rasterization_backend'] == 'cann' and cfg['renderer']['render_depth'] is True
         and cfg['renderer']['device'] == wrapper['device'] == 'npu:1', 'Actual renderer device/CANN RGB+ED contract differs')


def validate_planner_window_command(command, frames):
    window_contract(4, frames-4)
    if frames == 24:
        need(command.count('--cfg-options') == 1, 'The 24-frame planner needs one explicit original configuration override list')
        options = command[command.index('--cfg-options')+1:]
        options = options[:next((i for i, value in enumerate(options) if value.startswith('--')), len(options))]
        limits = [value for value in options if value.startswith('sim.max_frame_per_scene=')]
        need(limits == ['sim.max_frame_per_scene=24'], 'Original planner monitor must explicitly cover all 24 frames')
    return dict(configured_window_frames=frames, explicit_planner_window_required=frames == 24,
                real_consumption_acceptance_established_by_separate_receipts=True)


def validate_current_planner_contract(runtime, command, recorded_algengine_root='/alg'):
    """Accept only this new FP32 / current245 / CUDA-compatible route."""
    need(runtime['AlgEngine_runtime'] == dict(source_sha256=ALG_SOURCE, python_files=245),
         'Current external AlgEngine must be exact 0488 / 245 files')
    need(runtime.get('precision') == 'fp32', 'This separate Sim24 route is FP32 only')
    for key in ('precision_before', 'precision_after'):
        need(runtime[key].get('msda', {}).get('backend') == 'cuda-compatible',
             'Current Sim24 route requires actual CUDA-compatible MSDA before and after')
    for option, value in [('--precision', 'fp32'), ('--algengine-root', recorded_algengine_root),
                          ('--expected-algengine-source-sha256', ALG_SOURCE),
                          ('--msda-backend', 'cuda-compatible')]:
        need(command.count(option) == 1 and command.index(option)+1 < len(command)
             and command[command.index(option)+1] == value,
             'Actual current planner argv differs: '+option)
    return dict(source_sha256=ALG_SOURCE, Python_files=245, precision='fp32',
                msda_backend='cuda-compatible', old_4099_results_reused=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'recorded-job-root', 'source-root', 'algengine-root', 'prepared-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--expected-algengine-source-sha256', choices=(ALG_SOURCE,), required=True)
    parser.add_argument('--expected-prepare-receipt-sha256', required=True)
    parser.add_argument('--expected-algengine-commit', default=ALG_COMMIT,
                        help='Expected literal commit recorded by the planner; this does not replace source verification')
    parser.add_argument('--recorded-algengine-root', default='/alg',
                        help='Absolute AlgEngine path recorded by the original planner, which may differ from this audit mount')
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--precision', choices=('fp32',), required=True)
    parser.add_argument('--host-devices', type=int, nargs=2, required=True,
                        help='Explicit ordered host chips: planner then renderer, two distinct indices in 0..15')
    parser.add_argument('--reaction', choices=('NR', 'R'), default='NR',
                        help='Required resolved reaction contract; default NR preserves the original NR audit profile')
    parser.add_argument('--expected-window-frames', type=int, choices=(24,), default=24,
                        help='Exactly four history plus twenty future frames from original metadata')
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args(argv)
    need(re.fullmatch('[0-9a-f]{64}', args.expected_source_sha256), 'Expected source must be an explicit SHA256')
    need(re.fullmatch('[0-9a-f]{40}', args.expected_algengine_commit), 'Expected declared AlgEngine commit must be a full Git SHA')
    recorded_algengine_root = PurePosixPath(args.recorded_algengine_root)
    need(recorded_algengine_root.is_absolute() and '..' not in recorded_algengine_root.parts, 'Recorded AlgEngine root must be absolute without traversal')
    need(Path(window_contract.__code__.co_filename).resolve() ==
         (args.source_root/'simengine_ascend/window_contract.py').resolve(strict=True)
         and sha(args.source_root/'simengine_ascend/window_contract.py') == '0290f9ec5edb5e995908332ce9ffafd6fea7ca598a5ffab059331406714a981b',
         'A foreign window helper shadows the exact installed Sim24 source')
    window = window_contract(4, args.expected_window_frames-4); frames = window['frames']; history = window['num_history']
    need(frames == 12 or args.precision == 'fp32', 'The new 24-frame candidate is FP32 only')
    expected_host_mapping = host_mapping(args.host_devices)
    need(not list(Path('/dev').glob('davinci[0-9]*')) and not list(Path('/dev').glob('nvidia[0-9]*')),
         'Run this audit in an isolated CPU-only container without accelerator device mounts')
    cutoff = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    need(cutoff.tzinfo is not None and cutoff.utcoffset().total_seconds() == 0, 'Explicit UTC deadline required')
    need(args.recorded_job_root.is_absolute() and '..' not in args.recorded_job_root.parts, 'Explicit absolute original job path required')
    output = args.output.absolute()
    need(not os.path.lexists(output) and output.parent.resolve(strict=True) == output.parent, 'Fresh external output required')
    for root in (args.job, args.source_root, args.algengine_root, args.prepared_dir):
        need(not output.is_relative_to(root.resolve()), 'Audit output cannot modify input trees')
    report = dict(schema_version=1, result='running', auditor_sha256=sha(__file__), accelerator_execution=False,
                  scope='single official scene / configured %d-frame / %d real controls' % (frames, frames-history), model=args.model, precision=args.precision,
                  reaction=args.reaction)
    report['expected_host_device_mapping'] = expected_host_mapping
    def guard():
        if time.time() >= cutoff.timestamp() or args.stop_file is not None and args.stop_file.exists():
            raise TimeoutError('CPU audit deadline/STOP reached')
    status = 1
    try:
        guard(); sys.path.insert(0, str(args.source_root))
        from simengine_ascend.tools.audit_replay import source_identity, log_steps
        from simengine_ascend.closed_loop_runtime import algengine_identity, integration_identity
        need(Path(source_identity.__code__.co_filename).resolve().is_relative_to(args.source_root.resolve())
             and Path(algengine_identity.__code__.co_filename).resolve().is_relative_to(args.source_root.resolve()), 'A foreign audit/runtime helper shadows the selected source')
        source = source_identity(args.source_root, guard)
        need(source['sha256'] == args.expected_source_sha256, 'Approved simulation source differs')
        need(algengine_identity(args.algengine_root)['source_sha256'] == ALG_SOURCE, 'Approved external AlgEngine source differs')
        prepare_path = within(args.prepared_dir, 'prepare_receipt.json')
        need(sha(prepare_path, guard) == args.expected_prepare_receipt_sha256, 'Approved official prepare receipt differs')
        prepared = read_json(prepare_path)
        need(prepared['result'] == 'pass_official_single_scene_asset_preparation' and prepared['publisher'] == 'OpenDriveLab/WorldEngine', 'Official preparation is incomplete')
        scene_dict = load_pickle(within(args.prepared_dir, prepared['derived_scene']['path']), guard, prepared['derived_scene']['sha256'])
        key = prepared['selected']['dictionary_key']; need(list(scene_dict) == [key], 'Wrong official scene selection')
        scene = scene_dict[key]
        for asset in prepared['asset_files']:
            need(sha(within(args.prepared_dir, asset['output_path']), guard) == asset['sha256'], 'Prepared official asset changed')
        job = args.job.resolve(strict=True); wrapper = read_json(within(job, 'WE_output/simulation_launch_report.json'))
        cfg = yaml.safe_load(within(job, 'WE_output/.hydra/config.yaml').read_text())
        validate_simulation_contract(wrapper, cfg, source, prepared, key, args.recorded_job_root, args.reaction, history, window['num_future'])
        launcher = read_json(within(job, 'launcher_report.json'))
        need(launcher['result'] == 'both_owned_commands_completed' and launcher['actual_exit'] == 0
             and set(launcher['children']) == {'planner', 'simulation'}
             and all(value['actual_exit'] == 0 for value in launcher['children'].values()), 'Owned planner/simulation phases did not complete')
        from simengine_ascend.launch_closed_loop import validate_runner_reports
        original_runner = validate_runner_reports(job/'WE_output', 1, [key])
        need(original_runner['scenarios'] == wrapper['runner_reports']['scenarios'], 'Original scenario reports differ from the wrapper evidence')
        report['completion_flag_sha256'] = sha(within(job, 'WE_output/simulation_completed.flag'), guard)
        report['log_steps'] = log_steps(within(job, 'WE_output/simulation.log'), key, frames-1)
        exported, report['camera_frame_coverage'] = audit_metadata(job, scene, wrapper, guard, history, window['num_future'])
        planner = read_json(within(job, 'plan_traj/closed_loop_planner_report.json'))
        config, checkpoint, checkpoint_sha = MODELS[args.model]
        need(planner['result'] == 'completed_configured_closed_loop_planner_run' and planner['checkpoint_strict_state_dict'] is True
             and planner['config_sha256'] == sha(within(args.algengine_root, config), guard)
             and planner['checkpoint_sha256'] == checkpoint_sha == sha(within(args.algengine_root, checkpoint), guard), 'Formal model/config/checkpoint provenance differs')
        runtime = planner['runtime']
        need(runtime['AlgEngine_runtime']['source_sha256'] == ALG_SOURCE and runtime['AlgEngine_declared_commit'] == args.expected_algengine_commit
             and runtime['image_id'] == IMAGE and runtime['integration_files'] == integration_identity(args.source_root), 'Planner source/image/integration provenance differs')
        saved_plan_path = within(job, 'launch_plan.json'); saved_plan = read_json(saved_plan_path)
        validate_host_mapping(args.host_devices, runtime, saved_plan, launcher)
        need(isinstance(runtime.get('torch'), str) and isinstance(runtime.get('torch_npu'), str), 'Missing actual runtime versions')
        need(isinstance(launcher.get('plan_sha256'), str) and re.fullmatch('[0-9a-f]{64}', launcher['plan_sha256']), 'Missing original launch-plan SHA')
        report['launch_plan_identity'] = dict(original_plan_declared_sha256=launcher['plan_sha256'],
                                               saved_launch_plan_sha256=sha(saved_plan_path, guard))
        need(runtime['AlgEngine_root'] == str(recorded_algengine_root) and runtime.get('module_origins'), 'Missing actual adapted module origins')
        for name, origin in runtime['module_origins'].items():
            boundary = recorded_algengine_root / ('ascend' if name.split('.')[0] == 'ascend' else 'projects/AlgEngine/mmdet3d_plugin')
            paths = origin.get('search_locations', []) if isinstance(origin, dict) else [origin]
            need(paths and all(isinstance(value, str) and PurePosixPath(value).is_relative_to(PurePosixPath(boundary)) for value in paths), 'Foreign/CUDA planning module origin')
        report['precision_evidence'] = validate_precision(runtime, args.precision)
        command = launcher['children']['planner']['command']
        need('--max-predictions' not in command and command[command.index('--precision')+1] == args.precision, 'Planner was shortened or precision command differs')
        report['planner_window_command'] = validate_planner_window_command(command, frames)
        report['current_AlgEngine_route'] = validate_current_planner_contract(runtime, command, str(recorded_algengine_root))
        vocab_path = within(args.algengine_root, 'data/alg_engine/test_8192_kmeans.npy')
        need(sha(vocab_path, guard) == VOCAB_SHA, 'Official trajectory vocabulary differs')
        vocab = np.load(vocab_path, allow_pickle=False); finite(vocab, (8192, 40, 3), 'Original trajectory vocabulary shape/values differ')
        prefix = scene['id'].split('-')[-1]; indices = {}
        with within(job, 'plan_traj/plan_idx.csv').open(newline='') as stream:
            reader = csv.DictReader(stream); need(reader.fieldnames == ['prefix', 'step', 'plan_idx'], 'Unexpected plan CSV columns')
            for row in reader:
                step, index = int(row['step']), int(row['plan_idx'])
                need(row['prefix'] == prefix and history <= step <= frames and step not in indices and 0 <= index < 8192, 'Extra/duplicate/invalid published model selection')
                indices[step] = index
                plan = npy(within(job, 'plan_traj/%s_%d.npy' % (prefix, step)), guard)
                need(np.allclose(plan, vocab[index], atol=1e-6, rtol=1e-6), 'Published model plan does not match selected official vocabulary')
        need(set(range(history, frames)).issubset(indices) and planner['predictions'] == len(indices), 'All configured real control plans were not produced')
        need({path.name for path in (job/'plan_traj').glob(prefix+'_*.npy')}
             == {'%s_%d.npy' % (prefix, step) for step in indices}, 'Unexpected/unindexed published trajectory file')
        last = planner['last_prediction']; last_step = max(indices)
        need(last['prefix'] == prefix and last['step'] == last_step and last['plan_idx'] == indices[last_step]
             and last['path'] == str(args.recorded_job_root/'plan_traj'/('%s_%d.npy' % (prefix, last_step)))
             and last['sha256'] == sha(job/'plan_traj'/('%s_%d.npy' % (prefix, last_step)), guard)
             and last['selection_space'] == 'original_8192_trajectory_vocabulary', 'Final model prediction provenance differs')
        report['consumption'] = audit_consumption(job, args.recorded_job_root, args.source_root, scene, exported, vocab, indices, guard, history, window['num_future'])
        need(source_identity(args.source_root, guard) == source and sha(prepare_path, guard) == args.expected_prepare_receipt_sha256, 'Pinned inputs changed during audit')
        report.update(result='pass_configured_real_closed_loop_consumption_window', source=source,
                      prepare_receipt_sha256=args.expected_prepare_receipt_sha256, formal_checkpoint_sha256=checkpoint_sha,
                      planner_provenance={name: runtime[name] for name in ('device', 'precision', 'AlgEngine_declared_commit', 'AlgEngine_runtime', 'image_id', 'torch', 'torch_npu', 'host_device_mapping')},
                      original_runner_reports=original_runner, published_plan_indices=sorted(indices),
                      acceptance=dict(configured_window=True, actual_planner_policy_controller_consumption=True,
                                      GPU_parity=False, full_original_log=(frames == int(scene['log_length']) == len(scene['metadata']['openscene_data_infos_dict'])), quality=False, scores=False, performance=False))
        status = 0
    except BaseException as error:
        report.update(result='incomplete_or_rejected', error_type=type(error).__name__, error=str(error))
        status = 2 if isinstance(error, (TimeoutError, KeyboardInterrupt)) else 1
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(result=report['result'], report=str(output), exit_code=status)))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
