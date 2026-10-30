"""Independent, eval-only full NAVTEST CPU input and JPEG audit.

Exactly 12,146 official anchors, current + three history frames, eight cameras.
No model, NPU inference, training, metric-cache generation or quality claim.
Fresh output only; an explicitly SHA-locked previous ledger may be reused after
rehashing its previously verified JPEG bytes. Old reports/ledgers are never edited.
"""
import argparse
import ast
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import types
import warnings

HELPERS_SHA = '23689b27ffc5ed20453e5a2181e1c37c3fcab41654976f05d33227c21246f0a8'
RUNTIME_SHA = 'a0d8820b69d5bce3d61a8a6cc6b8135c1c3a4ffbd8623e3cd10b14784d989a81'
RUNTIME_PYTHON_FILES = 250
PREFLIGHT_SHA = 'f39bd2cbe8d6351b25bacd4c5b28632cd5b26e76052ea0d3e3f67f98c11cb85e'
ASSET_LOCK_SHA = 'd9803fc42ddbda274f703dcda6d831db2658c1d403dc616704ad6da600bf8062'
SPLIT_SHA = '61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd'
ARCHIVE_LOCK_SHA = 'c62488d01a9fe43c254085aa009a06f1e001e5011f4081a56069d209b7e6ed54'
SUCCESS = 'pass_complete_original_NAVTEST_inputs_and_required_JPEG_decode_CPU_only'
ASSEMBLED_SUCCESS = 'pass_NAVTEST_metadata_and_JPEG_decode_without_publisher_pixel_provenance_CPU_only'
ASSETS = {
 'alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl': (848516627,'59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b'),
 'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl': (2791314758,'58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316'),
 'alg_engine/test_8192_kmeans.npy': (3932288,'cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35')}
CONFIGS = ('worldengine/e2e_vadv2_100pct.py','hydramdp/e2e_hydramdp_100pct.py')


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_regular(path):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True)!=path:
        raise ValueError('A canonical regular input is required: '+str(path))
    return path


helper_path=canonical_regular(Path(__file__).with_name('navtest_input_helpers.py'))
if raw_sha(helper_path)!=HELPERS_SHA:
    raise RuntimeError('Pinned shared CPU algorithms differ')
H=types.ModuleType('NAVTEST_shared_CPU_algorithms')
exec(compile(helper_path.read_bytes(),str(helper_path),'exec'),H.__dict__)
Paused,Rejected=H.Paused,H.Rejected
ARCHIVE_HELPERS_SHA='7c5ccd3ba2025e266f46eec05feae8cdb6c00a1810d85f358c6445dd9875bed8'
archive_helper_path=canonical_regular(Path(__file__).with_name('navtest_archive_provenance.py'))
if raw_sha(archive_helper_path)!=ARCHIVE_HELPERS_SHA:raise RuntimeError('Pinned read-only archive algorithms differ')
A=types.ModuleType('NAVTEST_archive_provenance')
exec(compile(archive_helper_path.read_bytes(),str(archive_helper_path),'exec'),A.__dict__)


def need(value,message):
    if not value:raise Rejected(message)



def validate_runtime_identity(identity):
    need(isinstance(identity,dict) and identity.get('sha256')==RUNTIME_SHA
         and type(identity.get('python_files')) is int
         and identity['python_files']==RUNTIME_PYTHON_FILES,
         'Pinned runtime source or Python file inventory differs')

def utc():return datetime.now(timezone.utc).isoformat()


def memory():
    if os.name!='posix' or not Path('/proc/self/statm').is_file():return dict(rss_bytes=None,peak_rss_bytes=None)
    import resource
    return dict(rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def unique_json(raw):
    def unique(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(Rejected('Nonfinite JSON constant')))


def locked_json(path,expected,guard):
    guard.check()
    path=canonical_regular(path)
    need(re.fullmatch('[0-9a-f]{64}',expected) is not None,'Explicit SHA256 required')
    before=H.digest_file(path,guard)
    need(before['sha256']==expected,'Locked JSON SHA differs: '+str(path))
    value=unique_json(path.read_bytes())
    need(H.digest_file(path,guard)==before,'Locked JSON changed while parsing')
    return value


class Guard:
    def __init__(self,deadline,stop_file=None):
        try:parsed=datetime.fromisoformat(deadline.replace('Z','+00:00'))
        except (AttributeError,ValueError) as error:raise Rejected('A future explicit UTC deadline is required') from error
        need(parsed.tzinfo is not None and parsed.utcoffset().total_seconds()==0,'Explicit UTC deadline required')
        self.deadline=parsed.timestamp();need(self.deadline>time.time(),'A future UTC deadline is required')
        self.stop_file=stop_file;self.event=threading.Event();self.reason=None
    def stop(self,reason):self.reason=reason;self.event.set()
    def check(self):
        if time.time()>=self.deadline:self.stop('authorization_deadline')
        if self.stop_file and self.stop_file.exists():self.stop('STOP')
        if self.event.is_set():raise Paused(self.reason or 'stopped')


def atomic(path,value):
    temporary=path.with_name(path.name+'.tmp')
    need(not path.is_symlink() and not temporary.is_symlink(),'Output metadata cannot be symlinked')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def validate_archive_lock(row):
    need(set(row)=={'dataset','huggingface_revision','modelscope_revision','member_prefix','archive_count','total_bytes','files'}
         and row['dataset']=='OpenDriveLab/OpenScene'
         and row['huggingface_revision']=='a76f840b65e972bc45e56c2adced897498e9a026'
         and row['modelscope_revision']=='9439af9e240a950dd0b6862ce3bc34233c5d8690'
         and row['member_prefix']=='openscene-v1.1/sensor_blobs/test'
         and row['archive_count']==32 and row['total_bytes']==127882665618,
         'Approved fixed-revision official32 archive lock required')
    files=row['files']
    need(isinstance(files,list) and len(files)==32 and
         {r.get('file') for r in files}=={'openscene_sensor_test_camera_%d.tgz'%i for i in range(32)}
         and sum(r.get('bytes',0) for r in files)==127882665618,
         'Exactly32 publisher archives and complete byte total required')
    for entry in files:
        need(set(entry)=={'file','url','bytes','sha256'} and type(entry['bytes']) is int and entry['bytes']>0
             and re.fullmatch('[0-9a-f]{64}',entry['sha256']) is not None
             and entry['url']=='https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/'
                 +row['modelscope_revision']+'/openscene-v1.1/openscene_sensor_test_camera/'+entry['file'],
             'Canonical fixed-revision archive entry differs')
    return files


def inspect_contract(phase,present,logs,split,infos):
    need(len(split['tokens'])==len(set(split['tokens']))==12146 and len(split['log_names'])==len(set(split['log_names']))==136,
         'Official NAVTEST split counts or uniqueness differ')
    need(present==set(split['tokens']) and len(logs)==136
         and phase['requested_anchors']==phase['selected_anchors']==phase['pdm_tokens']==12146,
         'Original NAVTEST annotation/PDM/token/log coverage differs')
    # The original inspector returns hex log_token IDs; the official YAML lists
    # long log_names. Establish their bijection from the locked annotations.
    mapping={}
    for info in infos:
        token=info['log_token']
        if token not in logs:continue
        name=info['log_name']
        need(token not in mapping or mapping[token]==name,'One log_token maps to multiple log_names')
        mapping[token]=name
    need(set(mapping)==logs and set(mapping.values())==set(split['log_names'])
         and len(set(mapping.values()))==136,'Official log_token/log_name mapping differs')
    return mapping


def decode(path,relative,guard):
    guard.check()
    canonical_regular(path)
    row=H.decode_jpeg(path,relative,guard)
    if row['result']=='pass':
        after=H.digest_file(path,guard)
        if after['sha256']!=row['sha256'] or after['bytes']!=row['bytes']:
            row.update(result='error',error_type='Rejected',error='JPEG content changed across hash/verify/load')
    elif 'error' in row:
        row['error']=row['error'].replace('NAVTRAIN','NAVTEST')
    return row


def load_previous(report_path,report_sha,ledger_path,ledger_sha,contract,references,guard):
    previous=locked_json(report_path,report_sha,guard)
    need(previous.get('result') in (SUCCESS,ASSEMBLED_SUCCESS,'paused_incomplete_NAVTEST_CPU_QC','blocked_or_rejected_NAVTEST_CPU_QC')
         and previous.get('input_contract')==contract and previous.get('worker_sha256')==raw_sha(__file__)
         and previous.get('helpers_sha256')==HELPERS_SHA and previous.get('no_accelerator_execution') is True
         and previous.get('image_audit_sha256')==ledger_sha,'Previous audit identity/contract differs')
    ledger_path=canonical_regular(ledger_path)
    before=H.digest_file(ledger_path,guard)
    need(before['sha256']==ledger_sha and before['bytes']<=512*1024**2,'Previous ledger identity/size differs')
    expected=set(references.values());seen=set();accepted={};count=0
    with ledger_path.open('rb') as stream:
        while True:
            guard.check();raw=stream.readline(1024*1024+1)
            if not raw:break
            need(raw.endswith(b'\n') and len(raw)<=1024*1024,'Truncated or oversized previous ledger row')
            row=unique_json(raw);relative=row.get('relative')
            need(relative in expected and relative not in seen and row.get('result') in ('pass','error'),
                 'Unexpected/duplicate previous JPEG reference')
            seen.add(relative);count+=1
            if row['result']=='pass':
                need(row.get('verified') is True and row.get('loaded') is True
                     and (row.get('width'),row.get('height'),row.get('mode'))==(1920,1080,'RGB')
                     and type(row.get('bytes')) is int and row['bytes']>0
                     and re.fullmatch('[0-9a-f]{64}',row.get('sha256','')) is not None,
                     'Previous pass row lacks actual strict JPEG evidence')
                accepted[relative]=row
    need(count==previous.get('decoded') and H.digest_file(ledger_path,guard)==before,'Previous ledger changed or row count differs')
    return accepted


def reused_row(path,relative,previous,guard):
    guard.check()
    canonical_regular(path);current=H.digest_file(path,guard)
    need(current==dict(bytes=previous['bytes'],sha256=previous['sha256']),'Previously decoded JPEG content changed')
    return dict(previous,relative=relative,verification_reused_from_previous_ledger=True)


def verify_current_JPEG_bytes(references,decoded_records,guard):
    need(set(references.values())==set(decoded_records),'Final JPEG reference/decoded-record coverage differs')
    count=total=0;digest=hashlib.sha256()
    for path,relative in sorted(references.items(),key=lambda item:item[1]):
        guard.check();current=H.digest_file(canonical_regular(path),guard)
        need(current==decoded_records[relative],'Dataset JPEG changed after decode/archive inspection: '+relative)
        digest.update((json.dumps(dict(relative=relative,**current),sort_keys=True)+'\n').encode())
        count+=1;total+=current['bytes']
    return dict(result='pass_all_current_required_JPEG_SHA256_rechecked_after_archive_stage',
                files=count,bytes=total,ordered_reference_size_content_sha256=digest.hexdigest(),
                scope='Each JPEG is rehashed in this final phase; subsequent external host changes are not prevented')


def run(args):
    need(sys.platform=='linux','Full NAVTEST audit requires Linux; local fixtures are separate')
    guard=Guard(args.deadline_utc,args.stop_file);guard.check()
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before starting')
    H.validate_CPU_only(H.accelerator_state())
    source=args.source.resolve(strict=True);data=args.data_root.resolve(strict=True)
    output=args.output_dir.absolute()
    need(output.parent.resolve(strict=True)==output.parent and not os.path.lexists(output),'A fresh canonical output directory is required')
    need(not output.is_relative_to(source) and not output.is_relative_to(data),'Output cannot modify the source/data trees')
    output.mkdir(exist_ok=False);report_path=output/'summary.json';ledger_path=output/'images.jsonl'
    report=dict(schema_version=1,result='running_NAVTEST_CPU_QC',started_utc=utc(),errors=[],
        scope='Exactly original NAVTEST12146 eval-only inputs and required current+3history eight-camera JPEGs; no model/scoring/training',
        worker_sha256=raw_sha(__file__),helpers_sha256=HELPERS_SHA,no_accelerator_execution=True,
        convergence_accepted=False,quality_accepted=False,GPU_parity_accepted=False,
        decoded=0,decode_passed=0,decode_failed=0,reused_decodes=0,failure_examples=[],
        image_audit=str(ledger_path),image_audit_complete=False,decode_workers=args.decode_workers)
    pool=None;retain=[];exit_code=1;ledger_hash=hashlib.sha256();began=time.monotonic()
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.signal(signal.SIGTERM,lambda *_:guard.stop('SIGTERM'));signal.signal(signal.SIGINT,lambda *_:guard.stop('SIGINT'))
    signal.signal(signal.SIGALRM,lambda *_:guard.check());signal.setitimer(signal.ITIMER_REAL,1,1)
    try:
        report['accelerator_before']=H.accelerator_state();H.validate_CPU_only(report['accelerator_before'])
        report['source_before']=H.source_identity(source,guard)
        validate_runtime_identity(report['source_before'])
        preflight=canonical_regular(source/'ascend/preflight.py')
        need(H.digest_file(preflight,guard)['sha256']==PREFLIGHT_SHA,'Pinned original inspect_phase differs')
        asset_lock=canonical_regular(source/'ascend/assets.lock.json')
        need(H.digest_file(asset_lock,guard)['sha256']==ASSET_LOCK_SHA,'Publisher asset lock differs')
        locked_rows={r['path']:r for r in unique_json(asset_lock.read_bytes())['assets']}
        report['input_files_before']={}
        for relative,(size,expected) in ASSETS.items():
            guard.check();path=canonical_regular(data/relative);observed=H.digest_file(path,guard)
            need(observed==dict(bytes=size,sha256=expected),'Publisher input size/SHA differs: '+relative)
            need(locked_rows['data/'+relative]['bytes']==size and locked_rows['data/'+relative]['sha256']==expected,'Source asset lock disagrees')
            report['input_files_before'][relative]=observed
        archive_entries=None
        if not args.assembled_only:
            need(args.archive_lock_sha256==ARCHIVE_LOCK_SHA,'Use the published exact official32 archive lock SHA')
            archive_lock=locked_json(args.archive_lock,args.archive_lock_sha256,guard)
            archive_entries=validate_archive_lock(archive_lock)
            report['archive_lock_before']=H.digest_file(canonical_regular(args.archive_lock),guard)
        report['image_pixel_provenance_accepted']=False
        spec=importlib.util.spec_from_file_location('original_NAVTEST_preflight',preflight);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        os.environ['WORLDENGINE_ROOT']=str(source)
        configs=[];specs=[]
        for relative in CONFIGS:
            path=canonical_regular(source/'projects/AlgEngine/configs'/relative)
            cfg=module.Config.fromfile(str(path),import_custom_modules=False);ds=cfg.data.test
            split=Path(ds.nav_filter_path)
            if not split.is_absolute():split=source/'projects/AlgEngine'/split
            split=canonical_regular(split)
            need(split==source/'projects/AlgEngine/configs/navsim_splits/navtest_split/navtest.yaml'
                 and hashlib.sha256(split.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SPLIT_SHA,'Original full NAVTEST filter is required')
            signature=dict(type=ds.type,queue_length=int(ds.get('queue_length',4)),history_frame_num=int(ds.get('history_frame_num',3)),
                load_interval=int(ds.get('load_interval',1)),test_mode=ds.get('test_mode'),ann_basename=Path(ds.ann_file).name)
            need(signature==dict(type='NavSimOpenSceneE2E',queue_length=4,history_frame_num=3,load_interval=1,test_mode=True,
                 ann_basename='nuplan_openscene_navtest.pkl'),'Original eval four-frame dataset contract differs')
            configs.append(dict(path=relative,sha256=H.digest_file(path,guard)['sha256'],dataset=signature));specs.append(ds)
        split_data=module.yaml.safe_load(split.read_text())
        need(len(split_data['tokens'])==len(set(split_data['tokens']))==12146,'Exactly12146 official tokens required')
        image_root=data/'raw/openscene-v1.1/sensor_blobs/test'
        vocabulary=module.np.load(data/'alg_engine/test_8192_kmeans.npy',allow_pickle=False)
        need(vocabulary.shape==(8192,40,3) and module.np.isfinite(vocabulary).all(),'Official vocabulary shape/finite contract differs')
        del vocabulary
        inspected=module.inspect_phase(specs[0],data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl',split,image_root,
             data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl',report['errors'],retain,include_test_history=True)
        phase,present,logs=H.validate_inspection(inspected,report['errors'])
        infos=retain[0]['infos'] if isinstance(retain[0],dict) else retain[0]
        log_mapping=inspect_contract(phase,present,logs,split_data,infos)
        references,counts=H.referenced_images(infos,present,image_root,4,guard)
        need(counts['selected_anchors']==12146 and counts['queue_frames']==phase['history_frames']
             and len(references)==phase['camera_files']==142976
             and counts['queue_frames']==17872 and counts['camera_reference_occurrences']==142976,
             'Independent complete NAVTEST17872-frame/142976-JPEG reference counts differ')
        for path in references:guard.check();canonical_regular(path)
        reference_hash=hashlib.sha256()
        for relative in sorted(references.values()):reference_hash.update((relative+'\n').encode())
        contract=dict(source_sha256=RUNTIME_SHA,python_files=RUNTIME_PYTHON_FILES,inputs=report['input_files_before'],configs=configs,
            split_canonical_LF_SHA256=SPLIT_SHA,anchors=12146,logs=136,queue_length=4,cameras=8,
            log_token_to_official_name_sha256=hashlib.sha256(json.dumps(log_mapping,sort_keys=True).encode()).hexdigest(),
            reference_relative_paths_sha256=reference_hash.hexdigest(),unique_JPEGs=len(references),queue_frames=counts['queue_frames'])
        report['CPU_memory_after_original_eval_inspection']=memory()
        report['CPU_inspector_RSS_times16_not_model_capacity']=(memory()['rss_bytes'] or 0)*16
        report.update(input_contract=contract,original_eval_phase=phase,reference_counts=counts,
            required_JPEGs=len(references),configs=configs,stage='decoding_all_required_NAVTEST_JPEGs')
        prior={}
        if args.resume_report:
            prior=load_previous(args.resume_report,args.resume_report_sha256,args.resume_ledger,args.resume_ledger_sha256,contract,references,guard)
            report['resume']=dict(report=str(args.resume_report),report_sha256=args.resume_report_sha256,
                ledger=str(args.resume_ledger),ledger_sha256=args.resume_ledger_sha256,old_artifacts_modified=False)
        from PIL import Image,ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES=False;Image.MAX_IMAGE_PIXELS=20000000
        warnings.filterwarnings('error',category=Image.DecompressionBombWarning)
        pool=ThreadPoolExecutor(max_workers=args.decode_workers);pending=deque();iterator=iter(sorted(references.items(),key=lambda x:x[1]))
        modes=Counter();geometries=Counter();last=time.monotonic();total=0;decoded_records={}
        atomic(report_path,report)
        with ledger_path.open('xb') as ledger:
            while True:
                guard.check()
                while len(pending)<args.decode_workers*2:
                    try:path,relative=next(iterator)
                    except StopIteration:break
                    fn=reused_row if relative in prior else decode
                    argv=(Path(path),relative,prior[relative],guard) if relative in prior else (Path(path),relative,guard)
                    pending.append(pool.submit(fn,*argv))
                if not pending:break
                row=pending.popleft().result();guard.check()
                raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();ledger.write(raw);ledger_hash.update(raw)
                report['decoded']+=1;total+=row.get('bytes',0)
                if row['result']=='pass':
                    decoded_records[row['relative']]=dict(bytes=row['bytes'],sha256=row['sha256'])
                    report['decode_passed']+=1;report['reused_decodes']+=int(row.get('verification_reused_from_previous_ledger',False))
                    modes[row['mode']]+=1;geometries['%dx%d'%(row['width'],row['height'])]+=1
                else:
                    report['decode_failed']+=1
                    if len(report['failure_examples'])<20:report['failure_examples'].append(row)
                if time.monotonic()-last>30:
                    ledger.flush();report.update(image_audit_sha256=ledger_hash.hexdigest(),total_decoded_file_bytes=total);report['CPU_memory']=memory();atomic(report_path,report);last=time.monotonic()
            ledger.flush();os.fsync(ledger.fileno())
        pool.shutdown(wait=True);pool=None
        report.update(image_audit_sha256=ledger_hash.hexdigest(),total_decoded_file_bytes=total,
            decoded_modes=dict(modes),decoded_geometries=dict(geometries))
        need(report['decoded']==len(references) and report['decode_failed']==0 and report['decode_passed']==len(references)
             and total==phase['camera_bytes'],'Full NAVTEST JPEG coverage/decoding did not pass')
        if archive_entries is not None:
            report['stage']='streaming_all_official32_archives_and_binding_required_JPEG_bytes'
            atomic(report_path,report)
            report['archive_provenance']=A.audit_all(args.archive_root.resolve(strict=True),archive_entries,decoded_records,guard,
                                                   output/'archive_provenance',args.decode_workers)
            report['archive_lock_after']=H.digest_file(canonical_regular(args.archive_lock),guard)
            need(report['archive_lock_after']==report['archive_lock_before'],'Archive lock changed during audit')
        report['stage']='final_required_dataset_JPEG_SHA256_recheck'
        atomic(report_path,report)
        report['final_JPEG_rehash']=verify_current_JPEG_bytes(references,decoded_records,guard)
        report['input_files_after']={r:H.digest_file(data/r,guard) for r in ASSETS}
        report['source_after']=H.source_identity(source,guard)
        need(report['input_files_after']==report['input_files_before'] and report['source_after']==report['source_before'],
             'Publisher inputs or runtime source changed during audit')
        need(hashlib.sha256(split.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==SPLIT_SHA,'Split changed during audit')
        need(H.digest_file(asset_lock,guard)['sha256']==ASSET_LOCK_SHA,'Publisher lock changed during audit')
        if archive_entries is not None:locked_json(args.archive_lock,args.archive_lock_sha256,guard)
        need(raw_sha(__file__)==report['worker_sha256'] and raw_sha(helper_path)==HELPERS_SHA and raw_sha(archive_helper_path)==ARCHIVE_HELPERS_SHA,'Audit scripts changed while running')
        report.update(result=ASSEMBLED_SUCCESS if args.assembled_only else SUCCESS,stage='complete',image_audit_complete=True,
                      publisher_archive_provenance_accepted=not args.assembled_only,
                      image_pixel_provenance_accepted=not args.assembled_only,shared_by_six_evaluation_cases=True)
        exit_code=0
    except Paused as error:
        report.update(result='paused_incomplete_NAVTEST_CPU_QC',reason=str(error));exit_code=2
    except Exception as error:
        report.update(result='blocked_or_rejected_NAVTEST_CPU_QC',error_type=type(error).__name__,reason=str(error));exit_code=1
    finally:
        # Disarm before setting the stop event: a pending timer must not raise
        # Paused from guard.check during thread shutdown and lose the summary.
        signal.setitimer(signal.ITIMER_REAL,0)
        guard.stop('worker_finished')
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        report.update(completed_utc=utc(),elapsed_seconds=time.monotonic()-began,actual_exit=exit_code,
            image_audit_sha256=ledger_hash.hexdigest(),accelerator_after=H.accelerator_state(),CPU_memory=memory())
        try:H.validate_CPU_only(report['accelerator_after'])
        except Rejected as error:report.update(result='rejected_non_CPU_execution',reason=str(error),actual_exit=1);exit_code=1
        atomic(report_path,report)
    print(json.dumps(dict(result=report['result'],summary=str(report_path),actual_exit=exit_code)))
    return exit_code


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','data-root','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--archive-root',type=Path);p.add_argument('--archive-lock',type=Path)
    p.add_argument('--archive-lock-sha256');p.add_argument('--assembled-only',action='store_true')
    p.add_argument('--deadline-utc',required=True);p.add_argument('--stop-file',type=Path,required=True)
    p.add_argument('--decode-workers',type=int,default=4)
    for name in ('resume-report','resume-ledger'):
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-sha256')
    a=p.parse_args(argv)
    archives=(a.archive_root,a.archive_lock,a.archive_lock_sha256)
    if (a.assembled_only and any(x is not None for x in archives)) or (not a.assembled_only and not all(x is not None for x in archives)):
        p.error('Use all three archive options for publisher provenance, or explicit --assembled-only without archive options')
    if not 1<=a.decode_workers<=4:p.error('Use1..4 decode workers')
    if not a.stop_file.is_absolute() or '..' in a.stop_file.parts:p.error('Use an explicit absolute STOP path without parent traversal')
    resume=[a.resume_report,a.resume_report_sha256,a.resume_ledger,a.resume_ledger_sha256]
    if any(v is not None for v in resume) and not all(v is not None for v in resume):p.error('Resume requires both files and explicit SHA256 locks')
    return a


if __name__=='__main__':
    raise SystemExit(run(parse_args()))
