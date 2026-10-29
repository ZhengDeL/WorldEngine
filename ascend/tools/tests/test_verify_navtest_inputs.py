"""Synthetic/local CPU fixtures only; no official data, Docker or NPU execution."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image,ImageFile

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parent
spec=importlib.util.spec_from_file_location('public_eval_QC_fixture',TOOLS/'verify_navtest_inputs.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)


class NAVTESTInputQC(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.guard=type('FixtureGuard',(),{'check':lambda self:None})()
        old=ImageFile.LOAD_TRUNCATED_IMAGES;ImageFile.LOAD_TRUNCATED_IMAGES=False
        self.addCleanup(setattr,ImageFile,'LOAD_TRUNCATED_IMAGES',old)

    def jpeg(self,name='camera.jpg',size=(1920,1080),mode='RGB'):
        path=self.root/name;Image.new(mode,size).save(path,format='JPEG');return path

    def row(self):
        path=self.jpeg();return path,W.decode(path,'log/CAM_F0/a.jpg',self.guard)

    def save_previous(self,rows,contract=None,change=None):
        ledger=self.root/'old_images.jsonl'
        ledger.write_bytes(b''.join((json.dumps(row,sort_keys=True)+'\n').encode() for row in rows))
        contract=contract or {'source_sha256':W.RUNTIME_SHA,'anchors':12146,'queue_length':4,'cameras':8}
        value=dict(result='paused_incomplete_NAVTEST_CPU_QC',input_contract=contract,
            worker_sha256=W.raw_sha(W.__file__),helpers_sha256=W.HELPERS_SHA,no_accelerator_execution=True,
            image_audit_sha256=W.raw_sha(ledger),decoded=len(rows))
        if change:change(value)
        report=self.root/'old_summary.json';report.write_text(json.dumps(value))
        return report,W.raw_sha(report),ledger,W.raw_sha(ledger),contract

    def test_current_runtime_fingerprint_and_file_count(self):
        identity=W.H.source_identity(TOOLS.parents[1],self.guard)
        self.assertEqual(identity['sha256'],'cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610')
        self.assertEqual(identity['python_files'],249)
        W.validate_runtime_identity(identity)

    def test_runtime_binding_rejects_legacy_or_changed_source(self):
        for fingerprint,count in [
            ('76c51f77d6d0c17f33652ea307a654bee5057c76ccd6f122b34e1fab66a1d3c5',245),
            ('0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922',245),
            ('7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb',247),
            ('0'*64,249)]:
            with self.subTest(source=fingerprint),self.assertRaises(W.Rejected):
                W.validate_runtime_identity(dict(sha256=fingerprint,python_files=count))

    def test_runtime_binding_rejects_wrong_file_inventory(self):
        for count in [245,247,248,250,True,249.0]:
            with self.subTest(count=count),self.assertRaises(W.Rejected):
                W.validate_runtime_identity(dict(sha256=W.RUNTIME_SHA,python_files=count))

    def test_emitted_runtime_contract_uses_shared_file_count(self):
        tree=ast.parse((TOOLS/'verify_navtest_inputs.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        contract=next(n.value for n in ast.walk(main) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='contract' for t in n.targets))
        count=next(k.value for k in contract.keywords if k.arg=='python_files')
        self.assertIsInstance(count,ast.Name);self.assertEqual(count.id,'RUNTIME_PYTHON_FILES')
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=='validate_runtime_identity' for n in ast.walk(main)),1)

    def test_cleanup_disarms_timer_before_pool_shutdown_can_observe_stop(self):
        tree=ast.parse(Path(W.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        block=next(n for n in run.body if isinstance(n,ast.Try) and n.finalbody).finalbody
        function=ast.FunctionDef(name='cleanup_fixture',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='exit_code')],kwonlyargs=[],kw_defaults=[],defaults=[ast.Constant(value=2)]),body=block,decorator_list=[])
        program=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        events=[];timer={'active':True};stopped={'value':False};saved=[]
        def stop(reason):stopped['value']=True;events.append('stop')
        def shutdown(**kwargs):
            events.append('shutdown')
            if timer['active'] and stopped['value']:raise W.Paused('timer fired during cleanup')
        fake_signal=type('Signal',(),{'ITIMER_REAL':1,'setitimer':staticmethod(lambda *a:timer.update(active=False)),
            'signal':staticmethod(lambda *a:None)})
        clean=dict(device_nodes=[],CUDA_initialized=False,NPU_initialized=False)
        h=type('Helpers',(),{'accelerator_state':staticmethod(lambda:clean),'validate_CPU_only':staticmethod(lambda state:None)})
        environment=dict(signal=fake_signal,guard=type('Guard',(),{'stop':staticmethod(stop)})(),
            pool=type('Pool',(),{'shutdown':staticmethod(shutdown)})(),old_handlers={},report={},utc=lambda:'fixture',
            time=type('Clock',(),{'monotonic':staticmethod(lambda:1)}),began=0,exit_code=2,
            ledger_hash=hashlib.sha256(),H=h,memory=lambda:{},Rejected=W.Rejected,
            atomic=lambda path,report:saved.append(dict(report)),report_path='unused')
        exec(compile(program,'actual_run_cleanup_fixture','exec'),environment)
        environment['cleanup_fixture']()
        self.assertEqual(events,['stop','shutdown']);self.assertEqual(saved[0]['actual_exit'],2)

    def test_valid_JPEG_full_load_dimensions_and_whole_SHA(self):
        path,row=self.row()
        self.assertEqual(row['result'],'pass');self.assertTrue(row['verified'] and row['loaded'])
        self.assertEqual((row['width'],row['height'],row['mode']),(1920,1080,'RGB'))
        self.assertEqual(row['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_truncated_wrong_geometry_grayscale_and_disguised_PNG_reject(self):
        path=self.jpeg();path.write_bytes(path.read_bytes()[:-64])
        self.assertEqual(W.decode(path,'camera.jpg',self.guard)['result'],'error')
        for name,geometry,mode in [('small.jpg',(960,540),'RGB'),('gray.jpg',(1920,1080),'L')]:
            row=W.decode(self.jpeg(name,geometry,mode),name,self.guard)
            self.assertEqual(row['result'],'error');self.assertIn('NAVTEST',row['error'])
        path=self.root/'not_JPEG.jpg';Image.new('RGB',(1920,1080)).save(path,format='PNG')
        self.assertEqual(W.decode(path,'not_JPEG.jpg',self.guard)['result'],'error')

    def test_content_mutation_after_decode_even_preserved_stat_is_rejected(self):
        path=self.jpeg();original=W.H.decode_jpeg
        def mutate(path,relative,guard):
            row=original(path,relative,guard);stamp=path.stat();raw=bytearray(path.read_bytes());raw[-3]^=1
            path.write_bytes(raw);os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns));return row
        with patch.object(W.H,'decode_jpeg',mutate):row=W.decode(path,'camera.jpg',self.guard)
        self.assertEqual(row['result'],'error');self.assertIn('content changed',row['error'])

    def test_exact_NAVTEST12146_and136_original_logs_contract(self):
        split=dict(tokens=['token_%05d'%i for i in range(12146)],log_names=['log_%03d'%i for i in range(136)])
        phase=dict(requested_anchors=12146,selected_anchors=12146,pdm_tokens=12146)
        logs={'hex_%03d'%i for i in range(136)}
        infos=[dict(log_token='hex_%03d'%i,log_name='log_%03d'%i) for i in range(136)]
        mapping=W.inspect_contract(phase,set(split['tokens']),logs,split,infos)
        self.assertEqual(mapping['hex_000'],'log_000')
        for field in ('tokens','log_names'):
            bad=copy.deepcopy(split);bad[field][-1]=bad[field][0]
            with self.subTest(field=field),self.assertRaises(W.Rejected):
                W.inspect_contract(phase,set(split['tokens']),logs,bad,infos)
        with self.assertRaises(W.Rejected):W.inspect_contract(dict(phase,pdm_tokens=12145),set(split['tokens']),logs,split,infos)

        with self.assertRaises(W.Rejected):W.inspect_contract(phase,set(split['tokens']),logs,split,infos+[dict(log_token='hex_000',log_name='foreign')])

    def test_current_plus_three_history_exact_eight_cameras(self):
        cameras={'CAM_F0','CAM_B0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2'}
        infos=[dict(token='t%d'%i,log_token='a',log_name='log',timestamp=i+1,
            cams={c:dict(data_path='log/'+c+'/%d.jpg'%i) for c in cameras}) for i in range(4)]
        refs,counts=W.H.referenced_images(infos,{'t3'},self.root,4,self.guard)
        self.assertEqual((len(refs),counts['queue_frames']),(32,4))
        for kind in ('camera','time','log'):
            bad=copy.deepcopy(infos)
            if kind=='camera':bad[2]['cams'].pop('CAM_B0')
            elif kind=='time':bad[2]['timestamp']=bad[1]['timestamp']
            else:bad[0]['log_token']='foreign'
            with self.subTest(kind=kind),self.assertRaises(W.Rejected):W.H.referenced_images(bad,{'t3'},self.root,4,self.guard)

    def test_archive_lock_requires_all32_fixed_revision_entries(self):
        lock=TOOLS.parent/'data_tools/navtest12146/ARCHIVES.lock.json'
        self.assertEqual(W.raw_sha(lock),W.ARCHIVE_LOCK_SHA)
        row=json.loads(lock.read_text());self.assertEqual(len(W.validate_archive_lock(row)),32)
        for change in (dict(archive_count=31),dict(huggingface_revision='master'),dict(files=row['files'][:-1]),
                       dict(member_prefix='foreign')):
            with self.subTest(change=list(change)),self.assertRaises(W.Rejected):W.validate_archive_lock(dict(row,**change))

    def test_explicit_resume_reuses_verified_bytes_and_does_not_edit_old_artifacts(self):
        path,row=self.row();report,rsha,ledger,lsha,contract=self.save_previous([row])
        before=(report.read_bytes(),ledger.read_bytes())
        previous=W.load_previous(report,rsha,ledger,lsha,contract,{str(path):row['relative']},self.guard)
        self.assertEqual(set(previous),{row['relative']})
        reused=W.reused_row(path,row['relative'],previous[row['relative']],self.guard)
        self.assertTrue(reused['verification_reused_from_previous_ledger'])
        self.assertEqual(before,(report.read_bytes(),ledger.read_bytes()))

    def test_resume_wrong_contract_source_SHA_and_missing_locks_reject(self):
        path,row=self.row();report,rsha,ledger,lsha,contract=self.save_previous([row])
        with self.assertRaises(W.Rejected):W.load_previous(report,'0'*64,ledger,lsha,contract,{str(path):row['relative']},self.guard)
        with self.assertRaises(W.Rejected):W.load_previous(report,rsha,ledger,lsha,dict(contract,source_sha256='0'*64),{str(path):row['relative']},self.guard)
        with self.assertRaises(W.Rejected):W.load_previous(report,rsha,ledger,'0'*64,contract,{str(path):row['relative']},self.guard)

    def test_resume_truncated_duplicate_or_forged_pass_rows_reject(self):
        path,row=self.row()
        for rows in ([row,row],[dict(row,verified=False)],[dict(row,relative='../foreign.jpg')]):
            report,rsha,ledger,lsha,contract=self.save_previous(rows)
            with self.subTest(rows=rows),self.assertRaises(W.Rejected):W.load_previous(report,rsha,ledger,lsha,contract,{str(path):row['relative']},self.guard)
        report,rsha,ledger,lsha,contract=self.save_previous([row]);ledger.write_bytes(ledger.read_bytes()[:-1]);lsha=W.raw_sha(ledger)
        previous=json.loads(report.read_text());previous['image_audit_sha256']=lsha;report.write_text(json.dumps(previous));rsha=W.raw_sha(report)
        with self.assertRaises(W.Rejected):W.load_previous(report,rsha,ledger,lsha,contract,{str(path):row['relative']},self.guard)

    def test_reused_image_changed_bytes_with_same_size_mtime_rejects(self):
        path,row=self.row();stamp=path.stat();raw=bytearray(path.read_bytes());raw[-3]^=1;path.write_bytes(raw)
        os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        with self.assertRaises(W.Rejected):W.reused_row(path,row['relative'],row,self.guard)
    def test_final_dataset_rehash_rejects_change_after_archive_stage(self):
        path,row=self.row();references={str(path):row['relative']};records={row['relative']:dict(bytes=row['bytes'],sha256=row['sha256'])}
        self.assertEqual(W.verify_current_JPEG_bytes(references,records,self.guard)['files'],1)
        stamp=path.stat();raw=bytearray(path.read_bytes());raw[-3]^=1;path.write_bytes(raw)
        os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        with self.assertRaisesRegex(W.Rejected,'after decode/archive'):W.verify_current_JPEG_bytes(references,records,self.guard)

    def test_device_nodes_and_initialized_backends_reject_without_loading_torch(self):
        clean=dict(device_nodes=[],CUDA_initialized=False,NPU_initialized=False);W.H.validate_CPU_only(clean)
        for item in (dict(device_nodes=['/dev/nvidia1']),dict(device_nodes=['/dev/davinci15']),dict(CUDA_initialized=True),dict(NPU_initialized=True)):
            with self.subTest(item=item),self.assertRaises(W.Rejected):W.H.validate_CPU_only(dict(clean,**item))
        with patch.object(W.H.Path,'glob',side_effect=lambda pat:[Path('/dev/nvidia1')] if pat=='nvidia*' else []):
            self.assertIn(str(Path('/dev/nvidia1')),W.H.accelerator_state()['device_nodes'])

    def test_deadline_STOP_and_signal_guard_prevent_reads(self):
        with patch.object(W.time,'time',return_value=100):
            guard=W.Guard('2026-10-20T10:50:00Z');guard.check();guard.stop('SIGTERM')
            with self.assertRaises(W.Paused):guard.check()
            stop=self.root/'STOP';stop.touch();guard=W.Guard('2026-10-20T10:50:00Z',stop)
            with self.assertRaises(W.Paused):W.decode(self.root/'absent.jpg','absent.jpg',guard)
        with patch.object(W.time,'time',return_value=2000000000),self.assertRaises(W.Rejected):W.Guard('2026-10-20T10:50:00Z')
        for value in ('2026-10-20T18:50:00+08:00','2026-10-20T10:50:00','not-a-date'):
            with self.subTest(value=value),self.assertRaises(W.Rejected):W.Guard(value)

    def test_public_CLI_requires_archive_provenance_or_explicit_assembled_mode(self):
        base=['--source',str(self.root),'--data-root',str(self.root),'--output-dir',str(self.root/'new'),
              '--deadline-utc','2030-01-01T00:00:00Z','--stop-file',str(self.root/'STOP')]
        with self.assertRaises(SystemExit):W.parse_args(base)
        self.assertTrue(W.parse_args(base+['--assembled-only']).assembled_only)
        for extra in (['--assembled-only','--archive-root',str(self.root)],['--assembled-only','--decode-workers','5']):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):W.parse_args(base+extra)
        with self.assertRaises(SystemExit):W.parse_args(base[:-2]+['--assembled-only'])

    def test_future_deadline_has_no_historical_date_cap(self):
        with patch.object(W.time,'time',return_value=2000000000):
            guard=W.Guard('2040-01-01T00:00:00Z');guard.check()
            with self.assertRaises(W.Rejected):W.Guard('2000-01-01T00:00:00Z')

    def test_unsafe_symlink_input_and_duplicate_nonfinite_JSON_reject(self):
        target=self.jpeg();link=self.root/'alias.jpg';link.symlink_to(target)
        with self.assertRaises(ValueError):W.canonical_regular(link)
        for raw in ('{"same":1,"same":2}','{"value":NaN}'):
            with self.subTest(raw=raw),self.assertRaises(W.Rejected):W.unique_json(raw)


if __name__=='__main__':unittest.main()
