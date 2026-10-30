"""CPU audit and portable original-prefix regression for Sim213/CBE249 FP32.

Use read-only original inputs, no accelerator devices and a fresh external output.
This private explicit-none profile checks original artifacts without activating optimization;
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
from simengine_ascend.tools.audit_scored_default8 import audit_scoring

from simengine_ascend.tools.audit_full24 import (
    ALG_SOURCE, ALG_COMMIT, IMAGE, MODELS, VOCAB_SHA, CAMERAS,
    need, regular, sha, read_json, within, load_pickle, finite,
    host_mapping, validate_host_mapping, validate_precision,
    validate_planner_window_command, validate_current_planner_contract,
    audit_metadata, audit_consumption, npy, window_contract)

SIM_SOURCE = '9d09e7bb239fcb0238255e689423598115c0115841b47912b10ffbef80b1e541'
ALG_SOURCE = 'cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610'
PREPARE_SHA = 'ac246caf243f2e1a679c4159dbd92ecb1cfe84f4197fc023295909d7943414fc'
SCENE_SHA = '8b2d308365eb88060b30c35db1e3ba13622ddf827e3c43ba91200174ca40d27e'
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score')
DENSE = METRICS[:2] + ('lane_keeping',) + METRICS[2:]


def validate_simulation_contract(wrapper, cfg, source, prepared, key, recorded_root,
                                 reaction, num_history=4, num_future=8):
    """Reuse the full control contract while explicitly requiring both scorers."""
    need(num_history == 4 and num_future == 8 and source == dict(sha256=SIM_SOURCE, files=213),
         'Scored profile requires exact Sim213 and the original default8 window')
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



def validate_current_planner_contract(runtime, command, recorded_root, predictions):
    """Explicit unoptimized A profile; no optimized execution is claimed."""
    need(runtime['AlgEngine_runtime'] == dict(source_sha256=ALG_SOURCE, python_files=249), 'Exact CBE249 source required')
    need(runtime['precision'] == 'fp32', 'The baseline profile is FP32 only')
    need(type(predictions) is int and predictions == 8, 'Exactly eight necessary actual model predictions required')
    for name in ('precision_before', 'precision_after'):
        need(runtime[name].get('msda', {}).get('backend') == 'cuda-compatible', 'Actual compatible MSDA required before/after')
    for option, value in [('--precision','fp32'), ('--algengine-root',recorded_root), ('--expected-algengine-source-sha256',ALG_SOURCE),
                          ('--msda-backend','cuda-compatible'), ('--inference-optimization','none'), ('--msda-query-chunk','2048')]:
        need(command.count(option) == 1 and command.index(option)+1 < len(command) and command[command.index(option)+1] == value,
             'Actual baseline command differs: '+option)
    need('inference_optimization_source' not in runtime and 'inference_optimization' not in runtime,
         'The original default-none route must not report an activated optimization')
    return dict(source_sha256=ALG_SOURCE, python_files=249, predictions=predictions, requested='none', query_chunk=2048,
                optimization_activated=False, preserved_NumPy_draws=None,
                preserved_counter_scope='Not instrumented in the current default-none path; no total RNG-call count is inferred')


def audit_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'recorded-job-root', 'source-root', 'algengine-root', 'prepared-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-inventory', type=Path, required=True)
    parser.add_argument('--expected-reference-sha256', required=True)
    parser.add_argument('--expected-source-sha256', choices=(SIM_SOURCE,), required=True)
    parser.add_argument('--expected-algengine-source-sha256', choices=(ALG_SOURCE,), required=True)
    parser.add_argument('--expected-prepare-receipt-sha256', required=True)
    parser.add_argument('--expected-algengine-commit', required=True,
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
    need(re.fullmatch('[0-9a-f]{40}',args.expected_algengine_commit) is not None, 'Explicit full published AlgEngine Git commit is required')
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
        need(source == dict(sha256=SIM_SOURCE, files=213), 'This scored profile requires the exact Sim213 runtime')
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
        report['current_AlgEngine_route'] = validate_current_planner_contract(runtime, command, str(recorded_algengine_root), planner['predictions'])
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
        reference = read_reference(args.reference_inventory, args.expected_reference_sha256, args.model, args.reaction, guard)
        current = prefix_inventory(job, scene, exported, report['scoring'], args.model, args.reaction, guard)
        report['reference_regression'] = compare_reference(reference, current)
        need(sha(args.reference_inventory, guard) == args.expected_reference_sha256,
             'Portable reference changed during the complete audit')
        report['reference_inventory_sha256'] = args.expected_reference_sha256
        report.update(result='pass_baseline_none_default8_scored_window_and_original_prefix_raw', profile='Sim213_CBE249_baseline_none_FP32_default8_PDM_dense', source=source,
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





REFERENCE_SOURCE = 'daabd5c34ed290038ad4d0da13c32e81100b9692a0b63ff9b657f72edd9c7d66'
REFERENCE_ALG_SOURCE = '0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922'
REFERENCE_AUDITOR_SHA = '22bf0ee7439b5b5b2b09a6ffe822e65a6cbeeb6f7e25bd893ceb4484be56a2f2'


def prefix_inventory(job, scene, exported, scoring, model, reaction, guard):
    """Hash complete persisted outputs; never read model tensors or run scoring."""
    prefix = scene['id'].split('-')[-1]
    frames, images, controls = [], [], []
    for step, row in enumerate(exported):
        path = within(job/'frames', prefix+'_'+str(step)+'.pkl')
        frames.append(dict(step=step, name=path.name, bytes=path.stat().st_size, sha256=sha(path, guard)))
        for camera in sorted(CAMERAS):
            relative = row['cams'][camera]['data_path']
            path = within(job/'WE_output/openscene_format/sensor_blobs', relative)
            images.append(dict(step=step, camera=camera, relative=relative,
                               bytes=path.stat().st_size, sha256=sha(path, guard)))
    for step in range(4, 12):
        plan = within(job/'plan_traj', prefix+'_'+str(step)+'.npy')
        merged = within(job/'merged_ann_files', 'merged_'+str(step)+'.pkl')
        controls.append(dict(step=step, plan_name=plan.name, plan_bytes=plan.stat().st_size,
                             plan_sha256=sha(plan, guard), input_name=merged.name,
                             input_bytes=merged.stat().st_size, input_sha256=sha(merged, guard)))
    return dict(model=model, reaction=reaction, original_scene_sha256=SCENE_SHA,
                frames=frames, images=images, controls=controls,
                dense_rows=scoring['dense_rows'], PDM_CSV_sha256=scoring['PDM_CSV_sha256'],
                PDM_metrics=scoring['original_PDM_metrics'], PDM_CSV_precision_decimals=5,
                unconsumed_terminal_plans_excluded=True,
                unrounded_internal_PDM_values_compared=False)


def validate_reference_report(report, model, reaction):
    need(report.get('result') == 'pass_original_default8_scored_control_window'
         and report.get('auditor_sha256') == REFERENCE_AUDITOR_SHA,
         'A completed pinned public212 scored audit is required')
    need(report.get('model') == model and report.get('reaction') == reaction and report.get('precision') == 'fp32'
         and report.get('source') == dict(sha256=REFERENCE_SOURCE, files=212),
         'Reference model/reaction/source must be the original212 scored profile')
    need(report['prepare_receipt_sha256'] == PREPARE_SHA
         and report['formal_checkpoint_sha256'] == MODELS[model][2], 'Reference official input/weight differs')
    provenance = report['planner_provenance']
    need(provenance['AlgEngine_runtime'] == dict(source_sha256=REFERENCE_ALG_SOURCE, python_files=245)
         and provenance['AlgEngine_declared_commit'] == '55efdef54139c63ab9afce9178ebb3bd130717ef'
         and provenance['image_id'] == IMAGE, 'Reference Alg245/image differs')
    need(report['camera_frame_coverage']['decoded_jpegs'] == 96
         and report['consumption']['consumed_control_indices'] == list(range(4,12))
         and report['scoring']['all_dense_arrays_finite'] is True
         and report['acceptance']['original_PDM_and_dense_artifacts'] is True,
         'Reference complete scored eight-control window is absent')


def read_reference(path, expected, model, reaction, guard=lambda: None):
    need(re.fullmatch('[0-9a-f]{64}', expected) is not None and sha(path, guard) == expected,
         'Portable reference whole SHA differs')
    value = read_json(path)
    need(value.get('schema_version') == 1 and value.get('kind') == 'public212_default8_persisted_prefix_reference'
         and value.get('reference_runtime') == dict(sha256=REFERENCE_SOURCE, files=212)
         and value.get('reference_auditor_sha256') == REFERENCE_AUDITOR_SHA
         and value.get('reference_creator_sha256') == '289a874477e13c31fc0e5218557122524ad74484dd753dc831fdf1cc0f8b5929',
         'Only the independently prepared public212 reference is supported')
    data = value['inventory']
    need(data['model'] == model and data['reaction'] == reaction and data['original_scene_sha256'] == SCENE_SHA,
         'Reference model/reaction/scene differs')
    need(len(data['images']) == 96 and [x['step'] for x in data['frames']] == list(range(12))
         and [x['step'] for x in data['controls']] == list(range(4,12))
         and [x['step'] for x in data['dense_rows']] == list(range(3,12)), 'Reference coverage is incomplete')
    need(sha(path, guard) == expected, 'Portable reference changed while reading')
    return value


def compare_reference(reference, current):
    expected = reference['inventory']
    keys = ('model','reaction','original_scene_sha256','frames','images','controls','dense_rows',
            'PDM_CSV_sha256','PDM_metrics','PDM_CSV_precision_decimals')
    differences = [key for key in keys if expected.get(key) != current.get(key)]
    need(not differences, 'Original complete persisted prefix differs: '+','.join(differences))
    return dict(result='pass_complete_original212_prefix_persisted_raw_comparison',
                reference_audit_sha256=reference['reference_audit_sha256'],
                reference_runtime=reference['reference_runtime'], decoded_JPEGs=96,
                frame_metadata_files=12, consumed_plan_files=8, actual_input_files=8,
                dense_positions=9, dense_arrays_per_position=8, all_full_dense_pickle_bytes_equal=True,
                PDM_persisted_CSV_bytes_equal=True, original_PDM_CSV_precision_decimals=5,
                unrounded_internal_PDM_values_compared=False, GPU_equivalence=False,
                all_scene_quality=False, formal_end_to_end_performance=False)


def prepare_reference(argv):
    parser = argparse.ArgumentParser(description='Prepare a portable byte reference from a completed public212 scored job; CPU only.')
    for name in ('job','recorded-job-root','validated-audit','source-root','prepared-dir','vocabulary','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-audit-sha256', required=True)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--reaction', choices=('NR','R'), required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args(argv)
    need(not list(Path('/dev').glob('davinci[0-9]*')) and not list(Path('/dev').glob('nvidia[0-9]*')),
         'Reference preparation requires a CPU-only container')
    cutoff = datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    need(cutoff.tzinfo is not None and cutoff.utcoffset().total_seconds() == 0, 'Explicit future UTC deadline required')
    def guard():
        if time.time() >= cutoff.timestamp() or args.stop_file is not None and args.stop_file.exists():
            raise TimeoutError('Reference preparation deadline/STOP')
    output = args.output.absolute()
    need(not os.path.lexists(output) and output.parent.resolve(strict=True) == output.parent,
         'Fresh external portable reference output required')
    for root in (args.job,args.source_root,args.prepared_dir):
        need(not output.is_relative_to(root.resolve(strict=True)), 'Output cannot modify an input tree')
    need(args.recorded_job_root.is_absolute() and '..' not in args.recorded_job_root.parts,
         'Original recorded job path must be absolute')
    guard(); need(sha(args.validated_audit,guard) == args.expected_audit_sha256, 'Trusted reference-audit SHA differs')
    report = read_json(args.validated_audit); validate_reference_report(report,args.model,args.reaction)
    prepare_path = within(args.prepared_dir,'prepare_receipt.json')
    need(sha(prepare_path,guard) == PREPARE_SHA, 'Original preparation receipt differs')
    prepared = read_json(prepare_path)
    need(prepared['derived_scene']['sha256'] == SCENE_SHA, 'Original real24 scene differs')
    scenes = load_pickle(within(args.prepared_dir,prepared['derived_scene']['path']),guard,SCENE_SHA)
    key = prepared['selected']['dictionary_key']; need(list(scenes) == [key], 'Wrong official scene')
    scene = scenes[key]
    from simengine_ascend.tools.audit_replay import source_identity
    identity = source_identity(args.source_root,guard)
    need(identity == dict(sha256=SIM_SOURCE,files=213), 'Use the exact current213 audit helper tree')
    job = args.job.resolve(strict=True); wrapper = read_json(within(job,'WE_output/simulation_launch_report.json'))
    need(wrapper['source'] == dict(sha256=REFERENCE_SOURCE,files=212)
         and wrapper['reaction'] == args.reaction and wrapper['mode'] == 'closed-loop'
         and wrapper['pdm_metrics_CPU'] is True and wrapper['dense_rewards_CPU'] is True,
         'The original212 scored simulation wrapper differs')
    exported, coverage = audit_metadata(job,scene,wrapper,guard,4,8)
    need(coverage == report['camera_frame_coverage'], 'Reference complete camera inventory differs')
    need(sha(args.vocabulary,guard) == VOCAB_SHA, 'Original8192 vocabulary differs')
    vocab = np.load(args.vocabulary,allow_pickle=False); finite(vocab,(8192,40,3),'Original vocabulary values differ')
    prefix = scene['id'].split('-')[-1]; indices = {}
    with within(job,'plan_traj/plan_idx.csv').open(newline='') as stream:
        reader=csv.DictReader(stream); need(reader.fieldnames == ['prefix','step','plan_idx'],'Original plan CSV columns differ')
        for row in reader:
            step,index=int(row['step']),int(row['plan_idx'])
            need(row['prefix']==prefix and 4<=step<=12 and step not in indices and 0<=index<8192,'Original plan selection differs')
            indices[step]=index
            need(np.allclose(npy(within(job,'plan_traj/'+prefix+'_'+str(step)+'.npy'),guard),vocab[index],atol=1e-6,rtol=1e-6),'Original selected vocabulary plan differs')
    need(sorted(indices) == report['published_plan_indices'], 'Reference published selections changed')
    consumption = audit_consumption(job,args.recorded_job_root,args.source_root,scene,exported,vocab,indices,guard,4,8)
    need(consumption == report['consumption'], 'Reference policy/controller receipt chain differs')
    scoring = audit_scoring(job,args.recorded_job_root,scene,args.reaction,guard)
    need(scoring == report['scoring'], 'Reference complete PDM/dense artifacts differ')
    value = dict(schema_version=1,kind='public212_default8_persisted_prefix_reference',
                 reference_runtime=dict(sha256=REFERENCE_SOURCE,files=212),
                 reference_audit_sha256=args.expected_audit_sha256,reference_auditor_sha256=REFERENCE_AUDITOR_SHA,
                 reference_creator_sha256=sha(__file__),
                 inventory=prefix_inventory(job,scene,exported,scoring,args.model,args.reaction,guard),
                 scope='One original scored scene prefix; complete persisted bytes only, no GPU or all-scene quality claim')
    need(sha(args.validated_audit,guard) == args.expected_audit_sha256
         and source_identity(args.source_root,guard) == identity
         and sha(prepare_path,guard) == PREPARE_SHA, 'Reference inputs/source changed')
    with output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(result='pass_public212_portable_reference_preparation',output=str(output),sha256=sha(output))))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ('prepare-reference','audit'):
        raise SystemExit('Use prepare-reference or audit; both modes require CPU-only read-only inputs.')
    return prepare_reference(argv[1:]) if argv[0] == 'prepare-reference' else audit_main(argv[1:])


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != 'audit':
        raise SystemExit('This explicit-none profile only supports audit; use the pinned public 289a tool to prepare references.')
    raise SystemExit(audit_main(sys.argv[2:]))
