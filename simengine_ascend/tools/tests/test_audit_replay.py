"""CPU fixtures for replay coverage; generated pixels are not device evidence."""
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from simengine_ascend.tools import audit_replay as audit


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    job = root/'job';job.mkdir()
    source = root/'source'
    (source/'projects/SimEngine/worldengine').mkdir(parents=True)
    (source/'simengine_ascend').mkdir()
    (source/'projects/SimEngine/worldengine/source.py').write_text('value=1\n')
    (source/'simengine_ascend/source.py').write_text('value=2\n')
    fingerprint = audit.source_identity(source, lambda: None)
    key = 'official-fixture-6326d00e52115da4'
    names = ['CAM_B0','CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2']
    scene = dict(id=key, log_length=24, base_timestamp=1632929928299917,
                 sample_rate=10, cameras={n:dict(width=4,height=3) for n in names},
                 metadata=dict(openscene_data_infos_dict={}))
    for i in range(24):
        scene['metadata']['openscene_data_infos_dict'][str(i)] = dict(token='original%03d'%i,
            cams={n:dict(data_path='original/'+n+'/image%03d.jpg'%i) for n in names})
    prepared = job/'prepared';prepared.mkdir()
    (prepared/'scene.pkl').write_bytes(pickle.dumps({key:scene}))
    receipt = dict(result='pass_official_single_scene_asset_preparation',
        selected=dict(dictionary_key=key,scene_internal_id=key,normalized_asset_id=key),
        derived_scene=dict(path='scene.pkl',sha256=audit.digest(prepared/'scene.pkl')))
    save_json(prepared/'prepare_receipt.json',receipt)
    for reaction in ('NR','R'):
        folder = job/('replay_'+reaction);folder.mkdir()
        cfg = dict(num_history=4,num_future=8,max_step=1000000,decision_repeat=1,
            distributed_mode='SINGLE_NODE',exit_on_failure=True,enable_resume=False,
            with_render_manager=True,with_data_manager=True,use_planner_actions=False,
            ego_policy='trajectory_policy',ego_controller='log_play_controller',
            agent_policy='idm_policy' if reaction=='R' else 'trajectory_policy',
            agent_navigation='idm_navigation' if reaction=='R' else 'trajectory_navigation',
            renderer=dict(device='npu:1',rasterization_backend='cann',render_depth=True),
            with_metric_manager=False,with_dense_reward_manager=False)
        (folder/'.hydra').mkdir();(folder/'.hydra/config.yaml').write_text(yaml.safe_dump(cfg))
        (folder/'simulation.log').write_text(''.join('INFO Step %d/11 for scenario (1/1) %s\n'%(i,key) for i in range(11))
            +'INFO Generated completion flag file: /job/'+folder.name+'/simulation_completed.flag\n')
        (folder/'simulation_completed.flag').touch()
        tokens=['original%03d-000'%i for i in range(12)]
        rows=[]
        for i in range(12):
            cams={}
            for n in names:
                relative='6326d00e52115da4/'+n+'/image%03d-000.jpg'%i
                p=folder/'openscene_format/sensor_blobs'/relative;p.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(4,3),(i*10,0,40)).save(p,format='JPEG')
                cams[n]=dict(data_path=relative)
            rows.append(dict(frame_idx=i,token=tokens[i],timestamp=scene['base_timestamp']+i*500000,
                sample_prev=tokens[i-1] if i else None,sample_next=tokens[i+1] if i<11 else None,
                log_name=key,scene_name=key,scene_token='6326d00e52115da4',cams=cams))
        meta=folder/'openscene_format/meta_datas'/(key+'.pkl');meta.parent.mkdir();meta.write_bytes(pickle.dumps(rows))
        originals=[dict(scenario_name=key,log_name='main_thread',planner_name=None,succeeded=True,
                        error_message=None,duration_s=1.2)]
        report=folder/'runner_report_fixture.json';save_json(report,originals)
        h=hashlib.sha256();total=0
        for p in sorted((folder/'openscene_format/sensor_blobs').rglob('*.jpg'),key=lambda p:p.relative_to(folder).as_posix()):
            h.update(p.relative_to(folder).as_posix().encode());h.update(bytes.fromhex(audit.digest(p)));total+=p.stat().st_size
        wrapper=dict(result='completed_original_simulation_and_output_checks',actual_exit=0,mode='replay',reaction=reaction,device='npu:1',
            source=fingerprint,scene_sha256=receipt['derived_scene']['sha256'],expected_scene_ids=[key],
            runner_reports=dict(scenarios=originals,original_runner_report_sha256={
                '/job/'+folder.name+'/'+report.name:audit.digest(report)}),
            rendered_images=dict(decoded_jpegs=96,total_bytes=total,inventory_sha256=h.hexdigest()))
        save_json(folder/'simulation_launch_report.json',wrapper)
    return job,source,receipt,key


class CoverageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.job,self.source,self.receipt,self.key=fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self,**changes):
        output=self.root/('audit%02d.json'%len(list(self.root.glob('audit*.json'))))
        argv=['--job',str(self.job),'--source-root',str(self.source),'--output',str(output),
            '--expected-prepare-receipt-sha256',audit.digest(self.job/'prepared/prepare_receipt.json'),
            '--expected-source-sha256',audit.source_identity(self.source,lambda:None)['sha256'],
            '--deadline-utc',(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat()]
        for name,value in changes.items():
            flag='--'+name.replace('_','-')
            if flag in argv:argv[argv.index(flag)+1]=str(value)
            else:argv.extend([flag,str(value)])
        with contextlib.redirect_stdout(io.StringIO()):
            status=audit.main(argv)
        return status,json.loads(output.read_bytes())

    def mutate_rows(self,operation,reaction='NR'):
        p=self.job/('replay_'+reaction)/'openscene_format/meta_datas'/(self.key+'.pkl')
        rows=pickle.loads(p.read_bytes());operation(rows);p.write_bytes(pickle.dumps(rows))

    def test_original_window_96_images_each_full_decode_without_full_log_claim(self):
        status,result=self.run_audit();self.assertEqual(status,0)
        self.assertEqual(result['result'],'pass_two_configured_replay_windows_coverage')
        for phase in result['phases'].values():
            self.assertEqual(phase['configured_window_frames'],12)
            self.assertEqual(phase['original_log_length'],24)
            self.assertEqual(phase['time_span_seconds'],5.5)
            self.assertFalse(phase['full_original_log_coverage'])
            self.assertEqual(phase['jpeg_inventory']['decoded_jpegs'],96)
            self.assertEqual(len(phase['jpeg_files']),96)
            self.assertEqual(phase['log_step_check']['logged_updates'],11)
        external=self.root/'prepared_external';external.mkdir()
        for p in (self.job/'prepared').iterdir():(external/p.name).write_bytes(p.read_bytes())
        with patch.object(audit,'load_pickle',wraps=audit.load_pickle) as load:
            status,result=self.run_audit(prepared_dir=external)
            self.assertEqual(load.call_args_list[0].args[0].parent,external)
        self.assertEqual(status,0)
        self.assertEqual(result['inputs']['prepared_dir'],str(external.resolve()))

    def test_missing_frame_and_broken_token_links_rejected(self):
        for field,value in [('frame_idx',99),('token','wrong'),('sample_next','wrong'),('timestamp',1)]:
            with self.subTest(field=field):
                p=self.job/'replay_NR/openscene_format/meta_datas'/(self.key+'.pkl');original=p.read_bytes()
                self.mutate_rows(lambda rows:rows[2].update({field:value}))
                self.assertEqual(self.run_audit()[0],1);p.write_bytes(original)
        self.mutate_rows(lambda rows:rows.pop())
        self.assertEqual(self.run_audit()[0],1)

    def test_missing_camera_and_traversal_are_rejected(self):
        p=self.job/'replay_NR/openscene_format/meta_datas'/(self.key+'.pkl');original=p.read_bytes()
        self.mutate_rows(lambda rows:rows[1]['cams'].pop('CAM_B0'))
        self.assertEqual(self.run_audit()[0],1);p.write_bytes(original)
        self.mutate_rows(lambda rows:rows[1]['cams']['CAM_B0'].update(data_path='../foreign.jpg'))
        self.assertEqual(self.run_audit()[0],1)

    def test_wrong_dimensions_corrupt_and_extra_JPEGs_rejected(self):
        p=next((self.job/'replay_NR/openscene_format/sensor_blobs').rglob('*.jpg'));original=p.read_bytes()
        Image.new('RGB',(2,2)).save(p,format='JPEG');self.assertEqual(self.run_audit()[0],1)
        p.write_bytes(b'not a complete JPEG');self.assertEqual(self.run_audit()[0],1);p.write_bytes(original)
        Image.new('RGB',(4,3)).save(p.parent/'extra.jpg',format='JPEG')
        self.assertEqual(self.run_audit()[0],1)

    def test_runner_success_cannot_hide_early_max_step_or_missing_update(self):
        p=self.job/'replay_NR/.hydra/config.yaml';cfg=yaml.safe_load(p.read_text());cfg['max_step']=10
        p.write_text(yaml.safe_dump(cfg));self.assertEqual(self.run_audit()[0],1)
        cfg['max_step']=1000000;p.write_text(yaml.safe_dump(cfg))
        log=self.job/'replay_NR/simulation.log';log.write_text(log.read_text().replace('INFO Step 10/11 for scenario (1/1) '+self.key+'\n',''))
        self.assertEqual(self.run_audit()[0],1)

    def test_unknown_config_field_type_retains_structure_and_does_not_guess(self):
        p=self.job/'replay_NR/.hydra/config.yaml';cfg=yaml.safe_load(p.read_text());cfg['num_history']='${unknown}'
        p.write_text(yaml.safe_dump(cfg));status,result=self.run_audit();self.assertEqual(status,1)
        self.assertIn('num_history',result['phases']['NR']['observed']['configuration_keys'])
        self.assertIn('id',result['phases']['NR']['observed']['original_scene_keys'])

    def test_changed_input_is_rejected_before_pickle_load(self):
        with patch.object(audit.pickle,'load',side_effect=AssertionError('must not load')) as load:
            status,_=self.run_audit(expected_prepare_receipt_sha256='0'*64)
        self.assertEqual(status,1);load.assert_not_called()

    def test_deadline_is_nonzero_and_preserves_audit_error(self):
        status,result=self.run_audit(deadline_utc='2020-01-01T00:00:00+00:00')
        self.assertEqual(status,2);self.assertEqual(result['error_type'],'TimeoutError')

    def test_mode_policy_mismatch_and_changed_runner_report_rejected(self):
        p=self.job/'replay_R/.hydra/config.yaml';cfg=yaml.safe_load(p.read_text());cfg['agent_policy']='trajectory_policy'
        p.write_text(yaml.safe_dump(cfg));self.assertEqual(self.run_audit()[0],1)
        cfg['agent_policy']='idm_policy';p.write_text(yaml.safe_dump(cfg))
        p=self.job/'replay_R/runner_report_fixture.json';p.write_text('[]')
        self.assertEqual(self.run_audit()[0],1)

    def test_integral_floating_microseconds_supported_but_fractional_rejected(self):
        self.assertEqual(audit.microseconds(1632929928299917.,'fixture'),1632929928299917)
        with self.assertRaises(ValueError):audit.microseconds(1.5,'fixture')

    def test_missing_original_report_binding_is_not_accepted(self):
        p=self.job/'replay_NR/simulation_launch_report.json';value=json.loads(p.read_bytes())
        value['runner_reports']['original_runner_report_sha256']={};save_json(p,value)
        self.assertEqual(self.run_audit()[0],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
