"""Pinned planning observer with explicit original sampler worlds 4, 12 and 16.

The inference loop and journal format stay in their original modules. Only the
world-size guard differs from the pinned legacy observer's observed_test.
"""
import json
import uuid
from journal_inference import (COLLECT_SHA, Guard, IMAGE, JournalPause, MULTI_SHA,
    SOURCE_SHA, SuffixSampler, RestoringLoader, fingerprint, parse_deadline)
from rank_journal import (RankJournal, canonical_json, common_start, need,
    rank_indices, sha, write_file)


def observed_test(original,model,loader,tmpdir,gpu_collect,*,torch,contract,journal_root,mode,
                  guard,get_rng,restore_rng,stop_after_steps=0):
    """Use unchanged original loop/collector, adding a None-returning forward hook."""
    guard.check();need(not gpu_collect and tmpdir is not None,'Original explicit CPU collector required')
    rank=torch.distributed.get_rank();world=torch.distributed.get_world_size()
    need(world == contract['world_size'] and world in (4, 12, 16), 'Explicit supported planning world4/12/16 required')
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
            # The wrapper supplies the explicitly locked collector. Saved
            # prefixes remain actual journal rows and retain padded rank order.
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
