"""CPU artifact audit of the explicit Sim212/Alg245 FP32 default8 scored profile.

Use read-only original inputs, no accelerator devices and a fresh external output.
This tool checks the original PDM/dense artifacts and real eight-control chain;
it does not execute scoring formulas, initialize Torch, or certify GPU parity,
all-scene quality, synthetic training exports or renderer backward support.
The older unscored 12-frame and full24 auditors keep their original contracts.
"""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np
import yaml

from simengine_ascend.tools.audit_full24 import (
    ALG_SOURCE, ALG_COMMIT, IMAGE, MODELS, VOCAB_SHA,
    need, regular, sha, read_json, within, load_pickle, finite,
    host_mapping, validate_host_mapping, validate_precision,
    validate_planner_window_command, validate_current_planner_contract,
    audit_metadata, audit_consumption, npy, window_contract)

SIM_SOURCE = 'daabd5c34ed290038ad4d0da13c32e81100b9692a0b63ff9b657f72edd9c7d66'
PREPARE_SHA = 'ac246caf243f2e1a679c4159dbd92ecb1cfe84f4197fc023295909d7943414fc'
SCENE_SHA = '8b2d308365eb88060b30c35db1e3ba13622ddf827e3c43ba91200174ca40d27e'
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score')
DENSE = METRICS[:2] + ('lane_keeping',) + METRICS[2:]


def validate_simulation_contract(wrapper, cfg, source, prepared, key, recorded_root,
                                 reaction, num_history=4, num_future=8):
    """Reuse the full control contract while explicitly requiring both scorers."""
    need(num_history == 4 and num_future == 8 and source == dict(sha256=SIM_SOURCE, files=212),
         'Scored profile requires exact Sim212 and the original default8 window')
    need(wrapper.get('pdm_metrics_CPU') is True and wrapper.get('dense_rewards_CPU') is True,
         'Both original CPU scorers must be enabled in the actual wrapper')
    need(cfg.get('with_metric_manager') is True and cfg.get('with_dense_reward_manager') is True,
         'Both original scorers must be enabled in the resolved Hydra configuration')
    # The shared validator has a deliberately unscored default contract. Verify
    # the two actual True fields above, then reuse all its remaining conditions
    # against a local copy; never mutate original reports or the old auditor.
    from simengine_ascend.tools.audit_full24 import validate_simulation_contract as unscored
    control_cfg = dict(cfg, with_metric_manager=False, with_dense_reward_manager=False)
    unscored(wrapper, control_cfg, source, prepared, key, recorded_root, reaction, 4, 8)


def audit_scoring(job, recorded_root, scene, reaction, guard=lambda: None):
    need(scene['log_length'] == 24 and len(scene['metadata']['openscene_data_infos_dict']) == 24,
         'Keep the complete original real24 scene; no shortening or frame padding')
    for obj in scene['object_track'].values():
        for name in ('position', 'velocity', 'heading'):
            x = np.asarray(obj['state'][name])
            need(len(x) >= 20 and np.isfinite(x[:20]).all(), 'Original future actor context is incomplete')
    folder = job/'WE_output/openscene_format'
    # Reaction is taken from the actual resolved wrapper and checked by the
    # control audit; the selected CSV must exist exactly once for that run.
    averages = list(folder.glob('all_scenes_pdm_averages_*.csv'))
    need(reaction in ('NR', 'R') and len(averages) == 1 and averages[0].name == 'all_scenes_pdm_averages_'+reaction+'.csv',
         'Exactly the selected original reaction PDM CSV is required')
    pdm = regular(averages[0]); pdm_before = sha(pdm, guard)
    with pdm.open(newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == ['token', 'step'] + list(METRICS) + ['first_violation_step'], 'PDM CSV columns differ')
        rows = list(reader)
    actual = [r for r in rows if r['token'] != 'overall_average']
    mean = [r for r in rows if r['token'] == 'overall_average']
    need(len(rows) == 2 and len(actual) == len(mean) == 1 and actual[0]['token'] == scene['id']
         and float(actual[0]['step']) == float(mean[0]['step']) == 11, 'One original endpoint PDM result is required')
    metrics = {k: float(actual[0][k]) for k in METRICS}
    need(all(np.isfinite(v) and 0 <= v <= 1 for v in metrics.values()), 'PDM metrics must be finite and in [0,1]')
    need(all(float(mean[0][k]) == metrics[k] for k in METRICS), 'Single-scene PDM average differs')
    first_violation = actual[0]['first_violation_step']
    need(mean[0]['first_violation_step'] == first_violation, 'Single-scene first NC/DAC violation differs')
    violation = None if first_violation == '' else float(first_violation)
    need(violation is None or np.isfinite(violation) and violation.is_integer() and 0 <= violation <= 11,
         'Invalid original first NC/DAC violation step')
    need(sha(pdm, guard) == pdm_before, 'PDM CSV changed while auditing')
    pkls = folder/'pdms_pkl'
    expected = {scene['id']+'_step_'+str(i)+'_scores.pkl' for i in range(3, 12)}
    need({x.name for x in pkls.iterdir()} == expected, 'Exactly nine original dense positions are required')
    dense_rows = []
    for step in range(3, 12):
        guard(); path = within(pkls, scene['id']+'_step_'+str(step)+'_scores.pkl'); identity = sha(path, guard)
        values = load_pickle(path, guard, identity)
        need(isinstance(values, dict) and set(values) == set(DENSE) | {'IL_plan_idx', 'target_traj'}, 'Dense metric keys differ')
        arrays = {}
        for key in DENSE:
            x = np.asarray(values[key]); dtype = 'bool' if key in METRICS[:2] else 'float64'
            need(x.shape == (8192,) and str(x.dtype) == dtype and np.isfinite(x).all(),
                 'Dense metric requires a complete finite original 8192-entry array: '+key)
            arrays[key] = dict(shape=list(x.shape), dtype=str(x.dtype),
                               raw_sha256=hashlib.sha256(x.tobytes(order='C')).hexdigest())
        target = np.asarray(values['target_traj'])
        need(target.shape == (8, 2) and target.dtype.kind == 'f' and np.isfinite(target).all(), 'Eight real finite ego future targets required')
        index = values['IL_plan_idx']
        need(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)) and 0 <= int(index) < 8192,
             'Invalid original IL trajectory candidate index')
        need(sha(path, guard) == identity, 'Dense payload changed while auditing')
        dense_rows.append(dict(step=step, pickle_sha256=identity, metrics=arrays,
                               target_traj_sha256=hashlib.sha256(target.tobytes(order='C')).hexdigest()))
    index_path = within(folder, 'all_scenes_pdm_pkl_paths_'+reaction+'.csv'); before = sha(index_path, guard)
    with index_path.open(newline='') as stream:
        reader = csv.DictReader(stream); need(reader.fieldnames == ['token', 'step', 'pkl_path'], 'Dense CSV columns differ'); indices = list(reader)
    need(len(indices) == 9 and {int(r['step']) for r in indices} == set(range(3, 12)), 'Dense index must cover all nine positions once')
    for row in indices:
        need(row['token'] == scene['token'] and row['pkl_path'] == str(recorded_root/'WE_output/openscene_format/pdms_pkl'/
             (scene['id']+'_step_'+str(int(row['step']))+'_scores.pkl')), 'Dense index points outside the original run')
    need(sha(index_path, guard) == before, 'Dense index changed while auditing')
    return dict(original_PDM_metrics=metrics, PDM_times100=metrics['score']*100,
                original_PDM_CSV_precision_decimals=5, PDM_CSV_sha256=pdm_before,
                original_first_NC_DAC_violation_step=violation,
                TTC_event_time_claimed=False, dense_index_CSV_sha256=before,
                original_real_context_frames=24, used_real_context_frames=20,
                rendered_frames=12, consumed_controls=list(range(4, 12)),
                dense_positions=list(range(3, 12)), proposals_per_position=8192,
                metrics_per_proposal=8, all_dense_arrays_finite=True, dense_rows=dense_rows,
                formulas_executed_by_this_audit=False, GPU_score_parity=False, all_scene_quality=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'recorded-job-root', 'source-root', 'algengine-root', 'prepared-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-source-sha256', choices=(SIM_SOURCE,), required=True)
    parser.add_argument('--expected-algengine-source-sha256', choices=(ALG_SOURCE,), required=True)
    parser.add_argument('--expected-prepare-receipt-sha256', required=True)
    parser.add_argument('--expected-algengine-commit', choices=(ALG_COMMIT,), default=ALG_COMMIT,
                        help='Expected literal commit recorded by the planner; this does not replace source verification')
    parser.add_argument('--recorded-algengine-root', default='/alg',
                        help='Absolute AlgEngine path recorded by the original planner, which may differ from this audit mount')
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--precision', choices=('fp32',), required=True)
    parser.add_argument('--host-devices', type=int, nargs=2, required=True,
                        help='Explicit ordered host chips: planner then renderer, two distinct indices in 0..15')
    parser.add_argument('--reaction', choices=('NR', 'R'), default='NR',
                        help='Required resolved reaction contract; default NR preserves the original NR audit profile')
    parser.add_argument('--expected-window-frames', type=int, choices=(12,), default=12,
                        help='Exactly four genuine history frames plus eight applied controls')
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
        need(source == dict(sha256=SIM_SOURCE, files=212), 'This scored profile requires the exact Sim212 runtime')
        need(algengine_identity(args.algengine_root)['source_sha256'] == ALG_SOURCE, 'Approved external AlgEngine source differs')
        prepare_path = within(args.prepared_dir, 'prepare_receipt.json')
        need(sha(prepare_path, guard) == args.expected_prepare_receipt_sha256 == PREPARE_SHA, 'Approved official prepare receipt differs')
        prepared = read_json(prepare_path)
        need(prepared['result'] == 'pass_official_single_scene_asset_preparation' and prepared['publisher'] == 'OpenDriveLab/WorldEngine', 'Official preparation is incomplete')
        scene_dict = load_pickle(within(args.prepared_dir, prepared['derived_scene']['path']), guard, prepared['derived_scene']['sha256'])
        key = prepared['selected']['dictionary_key']; need(list(scene_dict) == [key], 'Wrong official scene selection')
        scene = scene_dict[key]
        need(prepared['derived_scene']['sha256'] == SCENE_SHA, 'This scored profile requires the original prepared real24 scene')
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
        report['scoring'] = audit_scoring(job, args.recorded_job_root, scene, args.reaction, guard)
        report['consumption'] = audit_consumption(job, args.recorded_job_root, args.source_root, scene, exported, vocab, indices, guard, history, window['num_future'])
        need(source_identity(args.source_root, guard) == source and sha(prepare_path, guard) == args.expected_prepare_receipt_sha256, 'Pinned inputs changed during audit')
        report.update(result='pass_original_default8_scored_control_window', profile='Sim212_Alg245_FP32_default8_PDM_dense', source=source,
                      prepare_receipt_sha256=args.expected_prepare_receipt_sha256, formal_checkpoint_sha256=checkpoint_sha,
                      planner_provenance={name: runtime[name] for name in ('device', 'precision', 'AlgEngine_declared_commit', 'AlgEngine_runtime', 'image_id', 'torch', 'torch_npu', 'host_device_mapping')},
                      original_runner_reports=original_runner, published_plan_indices=sorted(indices),
                      acceptance=dict(configured_window=True, actual_planner_policy_controller_consumption=True,
                                      GPU_parity=False, full_original_log=(frames == int(scene['log_length']) == len(scene['metadata']['openscene_data_infos_dict'])), quality=False, original_PDM_and_dense_artifacts=True, GPU_score_parity=False, performance=False))
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
