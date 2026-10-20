"""Private original-MMDetection inference observer with durable per-rank resume.

Restricted to frozen0488 NAVFormer vocabulary planning, process_perception=False,
workers=0, four-frame original inputs and the original unshuffled world16 sampler.
This is additional execution instrumentation, not part of the frozen source.
Partial journals are never complete NAVTEST predictions or an acceptance result.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
import uuid

from rank_journal import (RankJournal,canonical_json,common_start,need,rank_indices,
                          rng_pack,rng_unpack,sha,write_file)

MULTI_SHA='1c17038cc42177a369516d2ca6520d5716c061b83c854e54b577d6d8301c82a7'
COLLECT_SHA='b1bd40c200c4629cee65b59cc38eaa47aa459ff9a269b948052b8755e57bbe98'
SOURCE_SHA='0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922'
IMAGE='sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48'


class JournalPause(BaseException):pass


class Guard:
    def __init__(self,deadline,stop):self.deadline=deadline;self.stop=stop;self.signalled=False
    def check(self):
        if self.signalled or datetime.now(timezone.utc)>=self.deadline or (self.stop and self.stop.exists()):
            raise JournalPause('Authorized deadline, STOP file, or own process signal')


def parse_deadline(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    need(parsed.tzinfo is not None and parsed.utcoffset().total_seconds()==0,'Explicit UTC deadline required')
    need(parsed>datetime.now(timezone.utc),'Deadline has expired')
    return parsed


def fingerprint(root):
    paths=list((root/'projects/AlgEngine').rglob('*.py'))+list((root/'ascend').glob('*.py'))
    digest=hashlib.sha256()
    for path in sorted(paths,key=lambda p:p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


class SuffixSampler:
    def __init__(self,indices,start):self.indices=indices;self.start=start
    def __iter__(self):return iter(self.indices[self.start:])
    def __len__(self):return len(self.indices)-self.start


class RestoringLoader:
    """Preserve original DataLoader seed consumption, then restore saved boundary."""
    def __init__(self,loader,journal,start,get_rng,restore_rng,guard):
        self.loader=loader;self.journal=journal;self.start=start
        self.get_rng=get_rng;self.restore_rng=restore_rng;self.guard=guard;self.used=False
    def __getattr__(self,name):return getattr(self.loader,name)
    def __len__(self):return len(self.loader)
    def __iter__(self):
        need(not self.used,'Original test loader unexpectedly iterated twice');self.used=True;self.guard.check()
        iterator=iter(self.loader)  # consumes the original Torch DataLoader base_seed
        if (self.journal.folder/'initial_rng').exists():
            state=self.journal.read(self.start-1)['rng_after'] if self.start else self.journal.initial_rng()
            self.restore_rng(state)
        else:
            need(self.start==0,'Initial RNG missing for resumed positive boundary')
            self.journal.initial_rng(self.get_rng())  # fresh: AFTER original iter construction
        return iterator


def observed_test(original,model,loader,tmpdir,gpu_collect,*,torch,contract,journal_root,mode,
                  guard,get_rng,restore_rng,stop_after_steps=0):
    """Use unchanged original loop/collector, adding a None-returning forward hook."""
    guard.check();need(not gpu_collect and tmpdir is not None,'Original explicit CPU collector required')
    rank=torch.distributed.get_rank();world=torch.distributed.get_world_size()
    need(world==contract['world_size']==16,'Unchanged original world16 required')
    module=model.module
    need(type(module).__name__=='NAVFormer' and module.process_perception is False,'Only independent original NAVFormer planning anchors supported')
    need(type(module.planning_head).__name__ in ('TrajScoringHead','TrajScoringHeadRL'),'Only original vocabulary-scoring heads supported')
    dataset=loader.dataset
    need(type(dataset).__name__=='NavSimOpenSceneE2E' and loader.num_workers==0
         and loader.batch_sampler.batch_size==1 and not loader.batch_sampler.drop_last,'Original singleton workers0 dataset required')
    need(getattr(dataset,'queue_length',None)==4 and getattr(dataset,'history_frame_num',3)==3
         and getattr(dataset,'process_perception',False) is False
         and getattr(dataset,'diffusiondrive_data_mode',False) is False,'Original complete planning four-frame queues required')
    tokens=[dataset.data_infos[index]['token'] for index in dataset.index_map]
    need(len(tokens)==contract['anchors'],'Dataset anchor count differs from external contract')
    original_sampler=loader.batch_sampler.sampler
    need(getattr(original_sampler,'shuffle',None) is False,'Original unshuffled distributed sampler required')
    indices=list(original_sampler)
    need(indices==rank_indices(len(tokens),world,rank),'Original padded stride sampler differs')
    journal=RankJournal(journal_root,contract,tokens,world,rank,mode)
    report=dict(rank=rank,world_size=world,committed_before=journal.committed,result='partial_not_full_inference',replayed_exact_steps=0,new_committed_steps=0)
    attempts=journal.folder/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/(uuid.uuid4().hex+'.json')
    handle=None;original_collect=original.__globals__['collect_results_cpu']
    step=0
    try:
        before_collective=get_rng();inventories=[None]*world
        torch.distributed.all_gather_object(inventories,dict(rank=rank,contract_sha256=journal.contract_sha,committed=journal.committed))
        restore_rng(before_collective)  # additional object collectives must not perturb original RNG
        need([item['rank'] for item in inventories]==list(range(world))
             and all(item['contract_sha256']==journal.contract_sha for item in inventories),'All original ranks and identical immutable contracts required')
        start=common_start([item['committed'] for item in inventories],world);step=start
        report.update(common_resume_step=start,all_rank_prefixes=[item['committed'] for item in inventories])
        loader.batch_sampler.sampler=SuffixSampler(indices,start)
        proxy=RestoringLoader(loader,journal,start,get_rng,restore_rng,guard)
        def observe(_model,_args,result):
            nonlocal step
            need(isinstance(result,list) and len(result)==1,'Original batch must return one result dictionary')
            outcome=journal.append(step,result[0],get_rng());step+=1
            key='replayed_exact_steps' if outcome=='replayed_exactly_no_overwrite' else 'new_committed_steps'
            report[key]+=1
            print(json.dumps(dict(journal_rank=rank,committed=journal.committed,step=step,outcome=outcome)),flush=True)
            guard.check()
            if stop_after_steps and step-start>=stop_after_steps:raise JournalPause('Controlled committed-step stop; not a full result')
            return None
        def collect(result_part,size,tmpdir=None):
            guard.check();need(size==len(tokens) and step==len(indices),'Partial original loop cannot enter final collector')
            prefix=[journal.read(i)['result'] for i in range(start)]
            need(len(prefix)+len(result_part)==len(indices),'Every original padded per-rank result required')
            # No replacement collector: the actual installed MMDet function still
            # writes rank parts, barriers, interleaves and truncates padding.
            return original_collect(prefix+result_part,size,tmpdir)
        handle=model.register_forward_hook(observe)
        original.__globals__['collect_results_cpu']=collect
        result=original(model,proxy,tmpdir,gpu_collect)
        need(step==len(indices) and journal.committed==len(indices),'Full original rank sequence incomplete')
        report['result']='pass_original_padded_rank_sequence_and_exact_replay'
        return result,report
    finally:
        original.__globals__['collect_results_cpu']=original_collect
        loader.batch_sampler.sampler=original_sampler
        if handle is not None:handle.remove()
        report.update(committed_after=journal.committed,forward_steps_this_attempt=step-report.get('common_resume_step',0),
                      journal_header_sha256=sha(journal.folder/'header.json'))
        try:write_file(attempt,canonical_json(report))
        finally:journal.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','config','checkpoint','output','journal-root','locked-inputs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--attempt-id',required=True,help='New unique UUID hex chosen by host for this output attempt')
    p.add_argument('--journal-mode',choices=('fresh','resume'),required=True)
    p.add_argument('--precision',choices=('fp32','bf16'),required=True)
    p.add_argument('--msda-backend',choices=('reference','cuda-compatible'),default='cuda-compatible')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--expected-config-sha256',required=True);p.add_argument('--expected-checkpoint-sha256',required=True)
    p.add_argument('--expected-source-sha256',default=SOURCE_SHA);p.add_argument('--expected-image-id',default=IMAGE)
    p.add_argument('--expected-locked-inputs-sha256',required=True)
    p.add_argument('--expected-anchors',type=int,default=12146)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--stop-file',type=Path)
    p.add_argument('--stop-after-steps',type=int,default=0,help='Diagnostic partial stop per rank, never full acceptance')
    args=p.parse_args();need(args.stop_after_steps>=0 and args.expected_anchors>0,'Invalid positive scope/step limit')
    guard=Guard(parse_deadline(args.deadline_utc),args.stop_file)
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:setattr(guard,'signalled',True))
    root=args.source_root.resolve(strict=True);config=args.config.resolve(strict=True);checkpoint=args.checkpoint.resolve(strict=True)
    output=args.output.resolve();journal_root=args.journal_root.resolve();locked_path=args.locked_inputs.resolve(strict=True)
    need(len(args.attempt_id)==32 and all(c in '0123456789abcdef' for c in args.attempt_id),'New lowercase UUID hex required')
    need(output!=journal_root and journal_root not in output.parents and output not in journal_root.parents,'Journal and attempt output must be separate')
    need(fingerprint(root)==args.expected_source_sha256==SOURCE_SHA,'Frozen original0488 model/runtime source required')
    need(os.environ.get('WORLDENGINE_IMAGE_ID')==args.expected_image_id==IMAGE,'Frozen original image required')
    need(int(os.environ.get('WORLD_SIZE','0'))==16,'torchrun unchanged world16 required')
    need(sha(config)==args.expected_config_sha256 and sha(checkpoint)==args.expected_checkpoint_sha256,'Config/checkpoint SHA differs')
    need(sha(locked_path)==args.expected_locked_inputs_sha256,'External locked input manifest changed')
    locked=json.loads(locked_path.read_text());need(set(locked)=={'annotation','pdm','vocabulary','filter','input_qc'},'Five explicit externally locked input files required')
    for item in locked.values():
        path=Path(item['path']);need(path.is_absolute() and path.resolve(strict=True)==path
            and path.stat().st_size==item['bytes'] and len(item['sha256'])==64,'Locked input path/size/SHA invalid')
    instrumentation=dict(wrapper_sha256=sha(Path(__file__)),journal_sha256=sha(Path(__file__).with_name('rank_journal.py')),
                         original_multi_gpu_test_sha256=MULTI_SHA,original_collect_results_cpu_sha256=COLLECT_SHA)
    contract=dict(schema_version=1,world_size=16,anchors=args.expected_anchors,source_sha256=SOURCE_SHA,image_id=IMAGE,
                  config_sha256=args.expected_config_sha256,checkpoint_sha256=args.expected_checkpoint_sha256,
                  precision=args.precision,msda_backend=args.msda_backend,deterministic=args.deterministic,
                  locked_inputs=locked,locked_inputs_sha256=sha(locked_path),instrumentation=instrumentation)
    os.environ['WORLDENGINE_ROOT']=str(root)
    sys.path[:0]=[str(root),str(root/'projects/AlgEngine')]
    import torch
    import mmdet.apis
    from mmcv import Config
    original=mmdet.apis.multi_gpu_test
    need(hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==MULTI_SHA,'Installed original multi_gpu_test changed')
    collect_original=original.__globals__['collect_results_cpu']
    need(hashlib.sha256(inspect.getsource(collect_original).encode()).hexdigest()==COLLECT_SHA,'Installed original CPU collector changed')
    cfg=Config.fromfile(str(config),import_custom_modules=False)
    expected_paths=dict(annotation=cfg.data.test.ann_file,vocabulary=cfg.model.planning_head.vocab_path,filter=cfg.data.test.nav_filter_path)
    need(all(Path(value).resolve(strict=True)==Path(locked[key]['path']) for key,value in expected_paths.items()),'Actual configuration input paths differ from external lock')
    need(cfg.model.get('process_perception',False) is False and cfg.model.planning_head.type in ('TrajScoringHead','TrajScoringHeadRL'),'Unsupported persistent/perception/generative profile')
    # The host mounts an already existing parent; each launch selects its own
    # new child. Only rank0 creates it. Other ranks wait for this exact UUID
    # claim instead of racing a shared output.exists() check.
    claim=dict(attempt_id=args.attempt_id,contract_sha256=hashlib.sha256(canonical_json(contract)).hexdigest())
    if int(os.environ.get('RANK','0'))==0:
        output.mkdir(parents=True,exist_ok=False);write_file(output/'journal_attempt_start.json',canonical_json(claim))
    else:
        while not (output/'journal_attempt_start.json').exists():guard.check();time.sleep(.1)
        need(json.loads((output/'journal_attempt_start.json').read_text())==claim,'Attempt output claim belongs to another launch')
    original_reports=[]
    def wrapped(model,loader,tmpdir=None,gpu_collect=False):
        guard.check()
        from ascend.runtime import rng_state,restore_rng
        def get_rng():return rng_pack(rng_state())
        def restore(value):restore_rng(rng_unpack(value,torch))
        before_collective=get_rng()
        model_path=Path(inspect.getfile(type(model.module))).resolve(strict=True)
        need(model_path==root/'projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py',
             'A foreign plugin shadows the frozen original NAVFormer')
        # Hash large annotation/PDM files once on rank0, then broadcast the
        # checked immutable file inventory; all ranks already checked paths/size.
        rank=torch.distributed.get_rank();verified=[None]
        if rank==0:
            for item in locked.values():guard.check();need(sha(item['path'])==item['sha256'],'Locked data/QC SHA differs')
            verified[0]=hashlib.sha256(canonical_json(locked)).hexdigest()
        torch.distributed.broadcast_object_list(verified,src=0)
        need(verified[0]==hashlib.sha256(canonical_json(locked)).hexdigest(),'Rank0 locked-input verification differs')
        restore(before_collective)
        need(Path(str(loader.dataset.pdm_path)+'.pkl').resolve(strict=True)==Path(locked['pdm']['path']),
             'Actual selected dataset PDM differs from external lock')
        result,report=observed_test(original,model,loader,tmpdir,gpu_collect,torch=torch,contract=contract,
            journal_root=journal_root,mode=args.journal_mode,guard=guard,get_rng=get_rng,restore_rng=restore,
            stop_after_steps=args.stop_after_steps)
        original_reports.append(report);return result
    mmdet.apis.multi_gpu_test=wrapped
    argv=[str(root/'ascend/inference.py'),'--config',str(config),'--checkpoint',str(checkpoint),'--output',str(output),
          '--precision',args.precision,'--msda-backend',args.msda_backend]
    if args.deterministic:argv.append('--deterministic')
    sys.argv=argv
    try:
        runpy.run_path(str(root/'ascend/inference.py'),run_name='__main__')
        guard.check();need(len(original_reports)==1,'Exactly one original planning test required')
        gathered=[None]*16;torch.distributed.all_gather_object(gathered,original_reports[0])
        if torch.distributed.get_rank()==0:
            need([item['rank'] for item in gathered]==list(range(16)) and all(
                item['result']=='pass_original_padded_rank_sequence_and_exact_replay'
                and item['committed_after']==(args.expected_anchors+15)//16 for item in gathered),
                'Every original rank must finish its complete padded sequence')
            path=output/'inference_validation.json';original_raw=path.read_bytes();original_report=json.loads(original_raw)
            need(original_report['result']=='pass_real_planning_inference' and original_report['anchors']==args.expected_anchors
                 and original_report['world_size']==16,'Original full inference validation required')
            need(fingerprint(root)==SOURCE_SHA and sha(config)==args.expected_config_sha256
                 and sha(checkpoint)==args.expected_checkpoint_sha256,'Original source/config/checkpoint changed during attempt')
            for item in locked.values():guard.check();need(sha(item['path'])==item['sha256'],'Locked data/QC changed during attempt')
            write_file(output/'original_inference_validation.json',original_raw)
            original_report['execution_instrumentation']=dict(**instrumentation,contract_sha256=hashlib.sha256(canonical_json(contract)).hexdigest(),
                journal_mode=args.journal_mode,rank_attempts=gathered,original_report_sha256=hashlib.sha256(original_raw).hexdigest(),
                acceptance='Full original configured anchor inference with per-rank durable outputs/RNG and exact boundary replay; no official-score or GPU-quality claim.')
            replacement=output/('inference_validation.'+uuid.uuid4().hex+'.tmp');write_file(replacement,canonical_json(original_report));os.replace(replacement,path)
            write_file(output/'journal_execution.json',canonical_json(dict(result='pass_full_original_configured_inference_with_journal',contract=contract,rank_attempts=gathered)))
    except JournalPause as exc:
        rank=int(os.environ.get('RANK','0'))
        output.mkdir(parents=True,exist_ok=True)
        write_file(output/('journal_partial_rank%d.json'%rank),canonical_json(dict(result='partial_committed_journal_not_complete_inference',reason=str(exc),contract=contract)))
        if str(exc).startswith('Controlled committed-step stop'):
            # All ranks have the same diagnostic step limit. Publish every
            # partial receipt before torchrun observes the first worker exit.
            torch.distributed.barrier()
        raise SystemExit(3)
    finally:mmdet.apis.multi_gpu_test=original


if __name__=='__main__':main()
