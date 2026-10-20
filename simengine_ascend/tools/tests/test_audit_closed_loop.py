"""Synthetic CPU integrity fixtures, not real driving/model/device acceptance."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

SIM = Path(os.environ.get('SIM_ROOT', str(Path(__file__).resolve().parents[3]))).resolve(strict=True)
if not (SIM/'simengine_ascend/test_consumption_audit.py').is_file():
    raise RuntimeError('Set SIM_ROOT to the exact reviewed simulator source (CPU container: /sim)')
sys.path.insert(0, str(SIM))
from simengine_ascend.tools import audit_closed_loop as audit
import simengine_ascend.test_consumption_audit as original_fixture
from simengine_ascend.file_protocol import atomic_pickle


class ConsumptionTest(unittest.TestCase):
    def test_full_eight_receipts_and_independent_missing_tampered_control_rejections(self):
        fixture = original_fixture.AuditTest(methodName='test_actual_original_policy_ego_controller_and_env_eight_request_chain')
        fixture.setUp()
        try:
            type(fixture.agent.policy).__module__ = 'worldengine.components.agents.policy.env_input_policy'
            type(fixture.agent.controller).__module__ = 'worldengine.components.agents.controller.log_play_controller'
            (fixture.root/'merged_ann_files').mkdir()
            for control in range(4, 12):
                action = fixture.bind(control); fixture.env.step(action)
                records = []
                for index in range(control-4, control):
                    records.append(pickle.loads((fixture.frames/('abc_%d.pkl' % index)).read_bytes()))
                atomic_pickle(fixture.root/'merged_ann_files'/('merged_%d.pkl' % control), {'infos': records})
            # Test fixture keeps its plans under `plans`; the production contract
            # uses `plan_traj`. Move only this fresh fixture's directory and update
            # its recorded absolute plan paths/receipt hash chain consistently.
            (fixture.root/'plans').rename(fixture.root/'plan_traj')
            previous = None
            for control in range(4, 12):
                path = fixture.audit.directory/('consumed_%06d.json' % control)
                row = json.loads(path.read_bytes())
                row['plan']['path'] = str(fixture.root/'plan_traj'/('abc_%d.npy' % control))
                row['previous_receipt_sha256'] = previous
                raw = json.dumps(row, sort_keys=True, indent=2).encode()+b'\n'; path.write_bytes(raw)
                previous = hashlib.sha256(raw).hexdigest()
            exported = [pickle.loads((fixture.frames/('abc_%d.pkl' % i)).read_bytes()) for i in range(12)]
            vocab = np.zeros((8192, 40, 3), dtype=np.float32)
            vocab[0] = np.arange(120, dtype=np.float32).reshape(40, 3)
            def run(rows=exported):
                return audit.audit_consumption(fixture.root, fixture.root, SIM, {'id': 'official-scene-abc'}, rows,
                                               vocab, {i: 0 for i in range(4, 12)}, lambda: None)
            result = run()
            self.assertEqual(result['consumed_control_indices'], list(range(4, 12)))
            self.assertEqual(result['real_frames'], 12)
            missing = fixture.audit.directory/'consumed_000009.json'; saved = missing.read_bytes(); missing.unlink()
            with self.subTest(case='missing real consumption'):
                with self.assertRaises(ValueError): run()
            missing.write_bytes(saved)
            for field, value in [('actual_calls', {'policy': 1, 'controller': 0}),
                                 ('control_index', 10), ('policy_return_identity_verified', False),
                                 ('previous_receipt_sha256', '0'*64)]:
                with self.subTest(case=field):
                    path = fixture.audit.directory/'consumed_000011.json'; saved = path.read_bytes()
                    row = json.loads(saved); row[field] = value; path.write_text(json.dumps(row))
                    with self.assertRaises(ValueError): run()
                    path.write_bytes(saved)
            image = Path(fixture.audit.frames[1]['cameras']['CAM_B0']['path']); saved = image.read_bytes()
            image.write_bytes(b'corrupted-camera')
            with self.subTest(case='camera content'):
                with self.assertRaises(ValueError): run()
            image.write_bytes(saved)
            path = fixture.root/'merged_ann_files/merged_8.pkl'; saved = path.read_bytes()
            merged = pickle.loads(saved); merged['infos'] = merged['infos'][1:]+merged['infos'][:1]
            path.write_bytes(pickle.dumps(merged))
            with self.subTest(case='shuffled actual input history'):
                with self.assertRaises(ValueError): run()
            path.write_bytes(saved)
            bad_export = copy.deepcopy(exported); bad_export[6]['frame_idx'] = 5
            with self.subTest(case='incorrect real successor metadata'):
                with self.assertRaises(ValueError): run(bad_export)
        finally:
            fixture.tearDown()


class MetadataTest(unittest.TestCase):
    def test_real_jpeg_decode_exact_matrix_and_negative_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary); folder = job/'WE_output'
            metadata = folder/'openscene_format/meta_datas'; metadata.mkdir(parents=True)
            scene_id, prefix = 'fixture-official-0123456789abcdef', '0123456789abcdef'
            rows, source_rows, inventory_items = [], {}, []
            cameras = {name: {'width': 32, 'height': 16} for name in audit.CAMERAS}
            for index in range(12):
                source_cameras, out_cameras = {}, {}
                for ordinal, camera in enumerate(sorted(audit.CAMERAS)):
                    original = 'source/'+camera+'/image_%d.jpg' % index
                    relative = prefix+'/'+camera+'/image_%d-000.jpg' % index
                    path = folder/'openscene_format/sensor_blobs'/relative; path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (32, 16), (index*10, ordinal*20, 40)).save(path, format='JPEG')
                    source_cameras[camera], out_cameras[camera] = {'data_path': original}, {'data_path': relative}
                    inventory_items.append(('openscene_format/sensor_blobs/'+relative, path))
                source_rows[index] = {'token': 'original-%d' % index, 'cams': source_cameras}
                rows.append(dict(frame_idx=index, token='original-%d-000' % index, timestamp=1000000+index*500000,
                                 scene_name=scene_id, log_name=scene_id, scene_token=prefix, ego2global=np.eye(4),
                                 sample_prev='original-%d-000' % (index-1) if index else None,
                                 sample_next='original-%d-000' % (index+1) if index < 11 else None, cams=out_cameras))
            path = metadata/(scene_id+'.pkl'); atomic_pickle(path, rows)
            h = hashlib.sha256()
            for relative, image in sorted(inventory_items):
                h.update(relative.encode()); h.update(hashlib.sha256(image.read_bytes()).digest())
            wrapper = {'rendered_images': dict(decoded_jpegs=96, total_bytes=sum(p.stat().st_size for _, p in inventory_items), inventory_sha256=h.hexdigest())}
            scene = dict(id=scene_id, cameras=cameras, log_length=24, metadata={'openscene_data_infos_dict': source_rows}, base_timestamp=1000000, sample_rate=10)
            result, evidence = audit.audit_metadata(job, scene, wrapper, lambda: None)
            self.assertEqual(evidence['decoded_jpegs'], 96)
            self.assertEqual(evidence['simulation_time_span_seconds'], 5.5)
            self.assertEqual(len(result), 12)
            saved = path.read_bytes(); wrong = copy.deepcopy(rows); wrong[5]['frame_idx'] = 4; atomic_pickle(path, wrong)
            with self.subTest(case='duplicate metadata index'):
                with self.assertRaises(ValueError): audit.audit_metadata(job, scene, wrapper, lambda: None)
            path.write_bytes(saved)
            wrong = copy.deepcopy(rows); wrong.pop(); atomic_pickle(path, wrong)
            with self.subTest(case='short metadata'):
                with self.assertRaises(ValueError): audit.audit_metadata(job, scene, wrapper, lambda: None)
            path.write_bytes(saved)
            image = inventory_items[0][1]; image_saved = image.read_bytes(); image.write_bytes(b'not-jpeg')
            with self.subTest(case='bad JPEG bytes'):
                with self.assertRaises((ValueError, OSError)): audit.audit_metadata(job, scene, wrapper, lambda: None)
            image.write_bytes(image_saved)
            Image.new('RGB', (31, 16)).save(image, format='JPEG')
            with self.subTest(case='wrong original camera dimensions'):
                with self.assertRaises(ValueError): audit.audit_metadata(job, scene, wrapper, lambda: None)


class PrecisionTest(unittest.TestCase):
    def fp32(self):
        state = dict(mode='fp32', public={'matmul_allow_hf32': False, 'conv_allow_hf32': False})
        activation = dict(dtype='torch.float32', device='npu:0', autocast_enabled=False, training=False, shape=[1, 2])
        return dict(backend='npu', device='npu:0', precision='fp32', precision_before=state, precision_after=copy.deepcopy(state),
                    fp32=dict(mode='fp32', observation_only=True, ambient_autocast_enabled=False,
                              master_parameters={'dtypes': {'torch.float32': 2}, 'devices': {'npu:0': 2}},
                              expected_activation_kinds=['Conv2d', 'Linear'],
                              first_observed_fp32_activations={'Conv2d': dict(activation), 'Linear': dict(activation)}))

    def test_fp32_actual_fields_required(self):
        runtime = self.fp32(); self.assertTrue(audit.validate_precision(runtime, 'fp32')['actual_Conv2d_and_Linear_outputs_observed'])
        for name, mutation in [('missing observer', lambda row: row.pop('fp32')),
                               ('BF16 output mislabeled FP32', lambda row: row['fp32']['first_observed_fp32_activations']['Linear'].update(dtype='torch.bfloat16')),
                               ('CPU output', lambda row: row['fp32']['first_observed_fp32_activations']['Conv2d'].update(device='cpu')),
                               ('autocast active', lambda row: row['fp32']['first_observed_fp32_activations']['Conv2d'].update(autocast_enabled=True))]:
            with self.subTest(case=name):
                row = self.fp32(); mutation(row)
                with self.assertRaises((ValueError, KeyError)): audit.validate_precision(row, 'fp32')

    def test_bf16_real_activation_geometry_and_restoration_fields_required(self):
        state = dict(mode='bf16', public={'matmul_allow_hf32': False, 'conv_allow_hf32': False})
        runtime = dict(backend='npu', device='npu:0', precision='bf16', precision_before=state, precision_after=copy.deepcopy(state),
            bf16=dict(BF16_weight_cache_disabled=True, inside_before_original_test={'enabled': True, 'dtype': 'torch.bfloat16'},
                ambient_after_original_test={'enabled': False, 'dtype': 'torch.bfloat16'}, actual_model=dict(mode='bf16',
                    master_parameter_dtypes={'torch.float32': 2}, first_observed_bf16_activations={
                        kind: {'dtype': 'torch.bfloat16', 'autocast': {'enabled': True, 'dtype': 'torch.bfloat16'}} for kind in ('Conv2d', 'Linear')},
                    geometry_FP32=dict(original_formulas_unchanged=True, reference_grid_created_in_FP32=True,
                        first_actual={'camera': {'autocast_inside': {'enabled': False}, 'output_floating_dtypes': {'torch.float32': 1}}}))))
        self.assertTrue(audit.validate_precision(runtime, 'bf16')['actual_Conv2d_and_Linear_outputs_observed'])
        runtime['bf16']['actual_model']['first_observed_bf16_activations']['Linear']['dtype'] = 'torch.float32'
        with self.assertRaises(ValueError): audit.validate_precision(runtime, 'bf16')


class CommandTest(unittest.TestCase):
    def test_real_CLI_stop_and_wrong_approved_source_publish_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('job', 'alg', 'prepared'):
                (root/name).mkdir()
            deadline = (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
            command = [sys.executable, '-m', 'simengine_ascend.tools.audit_closed_loop', '--job', str(root/'job'),
                       '--recorded-job-root', str(root/'job'), '--source-root', str(SIM),
                       '--algengine-root', str(root/'alg'), '--prepared-dir', str(root/'prepared'),
                       '--expected-source-sha256', '0'*64, '--expected-prepare-receipt-sha256', '0'*64,
                       '--model', 'vadv2', '--precision', 'fp32', '--reaction', 'NR',
                       '--host-devices', '4', '5', '--deadline-utc', deadline]
            environment = dict(os.environ)
            environment['PYTHONPATH'] = os.pathsep.join([str(SIM), str(SIM/'projects/SimEngine')])
            for stop in (False, True):
                with self.subTest(stop=stop):
                    output = root/('stop.json' if stop else 'wrong_source.json')
                    arguments = command+['--output', str(output)]
                    if stop:
                        marker = root/'STOP'; marker.touch(); arguments += ['--stop-file', str(marker)]
                    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30, cwd=SIM, env=environment)
                    self.assertEqual(result.returncode, 2 if stop else 1, result.stdout+result.stderr)
                    report = json.loads(output.read_bytes())
                    self.assertEqual(report['result'], 'incomplete_or_rejected')
                    self.assertFalse(report['accelerator_execution'])
                    self.assertIn('deadline/STOP' if stop else 'Approved simulation source differs', report['error'])


class ReactionContractTest(unittest.TestCase):
    def fixture(self, reaction):
        recorded = Path.cwd().resolve()/'synthetic-recorded-job'
        source = {'sha256': 'a'*64, 'files': 211}
        prepared = {'derived_scene': {'sha256': 'b'*64}}
        key = 'official-fixture-0123456789abcdef'
        wrapper = dict(result='completed_original_simulation_and_output_checks', actual_exit=0,
                       mode='closed-loop', reaction=reaction, source=source, scene_sha256='b'*64,
                       expected_scene_ids=[key], scene_pkl='/prepared/scene.pkl', asset_root='/prepared/assets', device='npu:1')
        cfg = dict(distributed_mode='SINGLE_NODE', exit_on_failure=True, enable_resume=False, num_history=4,
                   num_future=8, decision_repeat=1, use_planner_actions=True, ego_policy='env_input_policy',
                   ego_controller='log_play_controller', ego_client='navformer_client', ego_navigation='trajectory_navigation',
                   agent_policy='idm_policy' if reaction == 'R' else 'trajectory_policy',
                   agent_navigation='idm_navigation' if reaction == 'R' else 'trajectory_navigation',
                   with_render_manager=True, with_data_manager=True, with_metric_manager=False, with_dense_reward_manager=False,
                   data_file_path=wrapper['scene_pkl'], asset_folder_path=wrapper['asset_root'],
                   output_dir=str(recorded/'WE_output'), data_output_dir=str(recorded/'WE_output/openscene_format'),
                   planner_data_path=str(recorded/'plan_traj'), planner_client_folder=str(recorded/'frames'),
                   renderer=dict(rasterization_backend='cann', render_depth=True, device='npu:1'))
        return wrapper, cfg, source, prepared, key, recorded

    def test_original_NR_and_R_contracts_both_require_their_resolved_policies(self):
        for reaction in ('NR', 'R'):
            with self.subTest(reaction=reaction):
                values = self.fixture(reaction)
                audit.validate_simulation_contract(*values, reaction)

    def test_reaction_label_policy_navigation_or_scene_mismatch_are_rejected(self):
        for requested, recorded in [('NR', 'R'), ('R', 'NR')]:
            with self.subTest(requested=requested, recorded=recorded):
                with self.assertRaises(ValueError):
                    audit.validate_simulation_contract(*self.fixture(recorded), requested)
        for name, value in [('agent_policy', 'trajectory_policy'), ('agent_navigation', 'trajectory_navigation'),
                            ('ego_policy', 'trajectory_policy'), ('data_file_path', '/foreign/scene.pkl'),
                            ('planner_data_path', '/foreign/plans')]:
            with self.subTest(field=name):
                wrapper, cfg, source, prepared, key, recorded = self.fixture('R')
                cfg[name] = value
                with self.assertRaises(ValueError):
                    audit.validate_simulation_contract(wrapper, cfg, source, prepared, key, recorded, 'R')
        wrapper, cfg, source, prepared, key, recorded = self.fixture('NR')
        wrapper['expected_scene_ids'] = ['another-official-scene']
        with self.assertRaises(ValueError):
            audit.validate_simulation_contract(wrapper, cfg, source, prepared, key, recorded, 'NR')


class HostMappingTest(unittest.TestCase):
    def artifacts(self, values):
        mapping = audit.host_mapping(values)
        runtime = {'host_device_mapping': mapping}
        plan = {name: {'env': {'WORLDENGINE_HOST_DEVICE_IDS': mapping['WORLDENGINE_HOST_DEVICE_IDS']}} for name in ('planner', 'simulation')}
        launcher = {'children': {'planner': {'command': ['python', 'planner', '--device', 'npu:0']},
                                 'simulation': {'command': ['python', 'simulation', '--device', 'npu:1']}}}
        return runtime, plan, launcher

    def test_all_four_actual_ordered_host_pairs_are_explicitly_bound(self):
        for pair in ([4, 5], [6, 7], [8, 9], [10, 11]):
            with self.subTest(pair=pair):
                self.assertEqual(audit.validate_host_mapping(pair, *self.artifacts(pair)), audit.host_mapping(pair))

    def test_duplicate_out_of_range_noninteger_or_wrong_order_is_rejected(self):
        for pair in ([4, 4], [-1, 4], [0, 16], [0], [0, '1'], [True, 2]):
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError): audit.host_mapping(pair)
        for expected in ([6, 7], [5, 4]):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError): audit.validate_host_mapping(expected, *self.artifacts([4, 5]))
        runtime, plan, launcher = self.artifacts([4, 5])
        plan['simulation']['env']['WORLDENGINE_HOST_DEVICE_IDS'] = '14,15'
        with self.assertRaises(ValueError): audit.validate_host_mapping([4, 5], runtime, plan, launcher)
        runtime, plan, launcher = self.artifacts([4, 5])
        launcher['children']['planner']['command'][-1] = 'npu:1'
        with self.assertRaises(ValueError): audit.validate_host_mapping([4, 5], runtime, plan, launcher)

    def test_explicit_phase_visible_mask_must_preserve_logical_order(self):
        for phase in ('planner', 'simulation'):
            runtime, plan, launcher = self.artifacts([4, 5])
            plan[phase]['env']['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
            self.assertEqual(audit.validate_host_mapping([4, 5], runtime, plan, launcher), audit.host_mapping([4, 5]))
            for invalid in ('1,0', '', '0', '4,5', [0, 1], None):
                with self.subTest(phase=phase, invalid=invalid):
                    plan[phase]['env']['ASCEND_RT_VISIBLE_DEVICES'] = invalid
                    with self.assertRaises(ValueError): audit.validate_host_mapping([4, 5], runtime, plan, launcher)


if __name__ == '__main__':
    unittest.main()
