"""Synthetic CPU evidence only; exercises installed original loop/collector text."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
FIXTURES=Path(os.environ['WORLDENGINE_JOURNAL_FIXTURES']).resolve(strict=True) if os.environ.get('WORLDENGINE_JOURNAL_FIXTURES') else None

from rank_journal import (RankJournal,common_start,merge_original_order,rank_indices,
                          rng_pack,rng_unpack,sha)
from journal_inference import (JournalPause,RestoringLoader,SuffixSampler,observed_test,
                               MULTI_SHA,COLLECT_SHA,parse_deadline)

HERE=Path(__file__).resolve().parent


class ByteTensor:
    dtype='torch.uint8'
    def __init__(self,value):self.value=list(value);self.shape=(len(self.value),)
    def cpu(self):return self
    def tolist(self):return list(self.value)


class RNG:
    def __init__(self):self.cpu=0;self.npu=0;random.seed(867);np.random.seed(531)
    def state(self):
        return rng_pack(dict(python=random.getstate(),numpy=np.random.get_state(),
            cpu=ByteTensor(self.cpu.to_bytes(8,'little')),npu=ByteTensor(self.npu.to_bytes(8,'little'))))
    def restore(self,state):
        random.setstate(state['python']);n=state['numpy']
        np.random.set_state((n['generator'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        self.cpu=int.from_bytes(state['cpu'],'little');self.npu=int.from_bytes(state['npu'],'little')
    def draw(self):
        self.cpu+=3;self.npu+=7
        return random.random()+np.random.random()+self.cpu/1000+self.npu/10000


def row(token,value=1.):
    item=dict(token=token,trajectory=np.full((40,3),value,dtype=np.float32),chosen_ind=3)
    item.update({key:float(value) for key in ('ade_4s','fde_4s','no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score')})
    return item


class JournalTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.rng=RNG();self.tokens=['t%d'%i for i in range(22)];self.contract=dict(world_size=16,source='sourceA',image='imageA')
    def tearDown(self):self.temp.cleanup()
    def journal(self,mode='fresh',rank=0,contract=None):return RankJournal(self.root,contract or self.contract,self.tokens,16,rank,mode)
    def test_padding_exact_original_global_order(self):
        parts=[[self.tokens[i] for i in rank_indices(22,16,r)] for r in range(16)]
        self.assertEqual(merge_original_order(parts,22,16),self.tokens)
        self.assertEqual([x for p in zip(*parts) for x in p][22:],self.tokens[:10])
        full=[rank_indices(12146,16,r) for r in range(16)]
        self.assertTrue(all(len(x)==760 for x in full));self.assertEqual([v for batch in zip(*full) for v in batch][12146:],list(range(14)))
    def test_real_numpy_and_four_rng_roundtrip_exact_replay(self):
        state=self.rng.state();future=self.rng.draw();self.rng.restore(state);self.assertEqual(future,self.rng.draw())
        with self.journal() as j:
            j.initial_rng(state);j.append(0,row('t0'),state)
            self.assertEqual(j.append(0,row('t0'),state),'replayed_exactly_no_overwrite')
            restored=rng_unpack(state,types.SimpleNamespace(uint8='uint8',tensor=lambda x,**_:ByteTensor(x)))
            self.assertEqual(restored['cpu'].tolist(),list(state['cpu']))
        with self.journal('resume') as j:self.assertEqual(j.initial_rng(),state)
    def test_exact_replay_mismatch_preserves_committed_bytes(self):
        with self.journal() as j:
            state=self.rng.state();j.initial_rng(state);j.append(0,row('t0'),state);before=sha(j.steps/'000000'/'payload.pkl')
            with self.assertRaisesRegex(RuntimeError,'differs exactly'):j.append(0,row('t0',2),state)
            changed=dict(state,cpu=b'changed')
            with self.assertRaisesRegex(RuntimeError,'differs exactly'):j.append(0,row('t0'),changed)
            self.assertEqual(sha(j.steps/'000000'/'payload.pkl'),before)
    def test_initial_interruption_before_commit_and_after_commit(self):
        with self.journal() as j:
            orphan=j.staging/'unfinished';orphan.mkdir();(orphan/'state.pkl').write_bytes(b'interrupted')
        with self.journal('resume') as j:
            state=self.rng.state();self.assertEqual(j.initial_rng(state),state);self.assertTrue((j.staging/'unfinished').exists())
        with self.journal('resume') as j:self.assertEqual(j.initial_rng(),state)
    def test_uncommitted_stage_is_not_valid_prefix(self):
        with self.journal() as j:
            j.initial_rng(self.rng.state());j.append(0,row('t0'),self.rng.state());orphan=j.staging/'orphan';orphan.mkdir();(orphan/'payload.pkl').write_bytes(b'junk')
        with self.journal('resume') as j:self.assertEqual(j.committed,1);self.assertTrue(orphan.exists())
    def test_contract_rank_token_and_hole_rejected(self):
        with self.journal() as j:j.initial_rng(self.rng.state());j.append(0,row('t0'),self.rng.state())
        with self.assertRaisesRegex(RuntimeError,'contract changed'):self.journal('resume',contract=dict(self.contract,source='sourceB'))
        header=self.root/'rank_00'/'header.json';raw=header.read_bytes();value=json.loads(raw);value['rank']=1;header.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError,'contract changed'):self.journal('resume')
        header.write_bytes(raw)
        (self.root/'rank_00'/'steps'/'000000').rename(self.root/'rank_00'/'steps'/'000001')
        with self.assertRaisesRegex(RuntimeError,'hole'):self.journal('resume')
    def test_corrupt_step_rejected(self):
        with self.journal() as j:j.initial_rng(self.rng.state());j.append(0,row('t0'),self.rng.state())
        with (self.root/'rank_00'/'steps'/'000000'/'payload.pkl').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(RuntimeError,'hash chain'):self.journal('resume')
    def test_two_writers_and_missing_rank(self):
        with self.journal():
            with self.assertRaisesRegex(RuntimeError,'Another writer'):self.journal('resume')
        with self.assertRaisesRegex(RuntimeError,'Missing canonical'):self.journal('resume',rank=1)
    def test_wrong_token_nonfinite_result_and_empty_rng_never_commit(self):
        with self.journal() as j:
            j.initial_rng(self.rng.state())
            bad=row('wrong')
            with self.assertRaisesRegex(RuntimeError,'token differs'):j.append(0,bad,self.rng.state())
            bad=row('t0');bad['trajectory'][0,0]=np.nan
            with self.assertRaisesRegex(RuntimeError,'finite'):j.append(0,bad,self.rng.state())
            state=dict(self.rng.state(),npu=b'')
            with self.assertRaisesRegex(RuntimeError,'Nonempty'):j.append(0,row('t0'),state)
            self.assertEqual(j.committed,0);self.assertFalse(list(j.steps.iterdir()))
    def test_common_prefix_boundary_not_independent_rank_skips(self):
        self.assertEqual(common_start([4,3,2,4]*4,16),1);self.assertEqual(common_start([760]*16,16),759)
        for invalid in ([1]*15,[True]*16,[-1]*16):
            with self.assertRaises(RuntimeError):common_start(invalid,16)


class FakeSampler:
    shuffle=False
    def __iter__(self):return iter([0,16])


class FakeLoader:
    def __init__(self,rng):
        dataset_type=type('NavSimOpenSceneE2E',(),{'__len__':lambda s:22})
        self.dataset=dataset_type();self.dataset.queue_length=4;self.dataset.data_infos=[dict(token='t%d'%i) for i in range(22)];self.dataset.index_map=list(range(22))
        self.num_workers=0;self.batch_sampler=types.SimpleNamespace(batch_size=1,drop_last=False,sampler=FakeSampler());self.rng=rng
    def __len__(self):return len(list(self.batch_sampler.sampler))
    def __iter__(self):
        self.rng.cpu+=1  # real PyTorch iter(DataLoader) consumes a CPU base_seed
        def items():
            for i in self.batch_sampler.sampler:
                np.random.random()  # deterministic original preprocessing may consume RNG
                yield dict(index=i,history=tuple(range(i-3,i+1)))
        return items()


class Model:
    def __init__(self,rng):
        self.module=type('NAVFormer',(),{})();self.module.process_perception=False;self.module.planning_head=type('TrajScoringHead',(),{})();self.rng=rng;self.hooks=[];self.training=True
    def eval(self):self.training=False
    def register_forward_hook(self,hook):
        self.hooks.append(hook);return types.SimpleNamespace(remove=lambda:self.hooks.remove(hook))
    def __call__(self,index,history,**kwargs):
        assert history==tuple(range(index-3,index+1))
        output=[row('t%d'%index,self.rng.draw()+index)]
        for hook in list(self.hooks):assert hook(self,(),output) is None
        return output


class OriginalLoopTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()
    def make(self,rng,counts):
        def collective(values,item):
            rng.draw()  # test preservation even if added collective consumes RNG
            for rank in range(16):values[rank]=dict(item,rank=rank,committed=counts[rank])
        torch=types.SimpleNamespace(no_grad=contextlib.nullcontext,distributed=types.SimpleNamespace(get_rank=lambda:0,get_world_size=lambda:16,all_gather_object=collective))
        collect=lambda result,size,tmpdir=None:result
        namespace=dict(torch=torch,get_dist_info=lambda:(0,16),time=types.SimpleNamespace(sleep=lambda _:None),
            mmcv=types.SimpleNamespace(ProgressBar=lambda _:types.SimpleNamespace(update=lambda:None)),collect_results_cpu=collect)
        source=(FIXTURES/'installed_multi_gpu_test.py.txt').read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),MULTI_SHA);exec(source,namespace)
        return namespace['multi_gpu_test'],torch
    @unittest.skipUnless(FIXTURES is not None,'Extract the SHA-locked installed MMDetection fixtures first')
    def test_real_original_loop_fresh_stop_resume_exact_outputs_and_rng(self):
        rng=RNG();initial=rng.state();original,torch=self.make(rng,[0]*16)
        baseline=original(Model(rng),FakeLoader(rng),'cpu_collect');baseline_state=rng.state()
        rng.restore(initial)
        guard=types.SimpleNamespace(check=lambda:None);contract=dict(world_size=16,anchors=22)
        def observed(mode,counts,stop=0):
            function,torch=self.make(rng,counts)
            return observed_test(function,Model(rng),FakeLoader(rng),'cpu_collect',False,torch=torch,contract=contract,
                journal_root=self.root,mode=mode,guard=guard,get_rng=rng.state,restore_rng=rng.restore,stop_after_steps=stop)
        with self.assertRaises(JournalPause):observed('fresh',[0]*16,1)
        rng.restore(initial)  # a new process/model seed, before DataLoader iteration
        actual,report=observed('resume',[1]*16)
        self.assertEqual(pickle.dumps(actual,protocol=4),pickle.dumps(baseline,protocol=4));self.assertEqual(rng.state(),baseline_state)
        self.assertEqual(report['replayed_exact_steps'],1);self.assertEqual(report['new_committed_steps'],1)
        # A fully completed journal also replays one boundary and original
        # collector prepends the earlier saved row without dropping padding.
        rng.restore(initial);again,report=observed('resume',[2]*16)
        self.assertEqual([pickle.dumps(x,protocol=4) for x in again],[pickle.dumps(x,protocol=4) for x in baseline]);self.assertEqual(report['common_resume_step'],1)
    @unittest.skipUnless(FIXTURES is not None,'Extract the SHA-locked installed MMDetection fixtures first')
    def test_original_cpu_collector_interleave_truncate(self):
        rank=[0];namespace=dict(get_dist_info=lambda:(rank[0],16),osp=os.path,shutil=shutil,
            dist=types.SimpleNamespace(barrier=lambda:None),mmcv=types.SimpleNamespace(
                mkdir_or_exist=lambda p:Path(p).mkdir(parents=True,exist_ok=True),
                dump=lambda value,p:Path(p).write_bytes(pickle.dumps(value)),load=lambda p:pickle.loads(Path(p).read_bytes())))
        source=(FIXTURES/'installed_collect_results_cpu.py.txt').read_text();self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),COLLECT_SHA);exec(source,namespace)
        collect=namespace['collect_results_cpu'];parts=[rank_indices(22,16,r) for r in range(16)]
        for r in list(range(1,16))+[0]:rank[0]=r;output=collect(parts[r],22,str(self.root/'collect'))
        self.assertEqual(output,list(range(22)));self.assertFalse((self.root/'collect').exists())
    def test_stop_or_deadline_is_partial_and_future_utc_required(self):
        from journal_inference import Guard
        from datetime import datetime,timezone,timedelta
        stop=self.root/'STOP';stop.write_text('stop');guard=Guard(datetime.now(timezone.utc)+timedelta(minutes=1),stop)
        with self.assertRaises(JournalPause):guard.check()
        with self.assertRaises(RuntimeError):parse_deadline('2026-10-20T12:00:00')


if __name__=='__main__':unittest.main(verbosity=2)
