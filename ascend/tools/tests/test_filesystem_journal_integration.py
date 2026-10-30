"""CPU-only actual installed-loop/world4/12/16 journal and filesystem integration."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import contextlib,hashlib,os,pickle,sys,tempfile,types,unittest
TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
sys.path.insert(0,str(Path(__file__).parent))
FIXTURES=Path(os.environ['WORLDENGINE_JOURNAL_FIXTURES']) if os.environ.get('WORLDENGINE_JOURNAL_FIXTURES') else None
from test_rank_journal import RNG,Model,FakeLoader,row
from filesystem_journal_observer import observed_test,JournalPause,MULTI_SHA
from rank_journal import rank_indices,semantic_digest,sha
from file_collection import create_collection,publish_part,collect_results_file

class Integration(unittest.TestCase):
 def case(self,world):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);tokens=['t%d'%i for i in range(22)];rng=RNG();initial=rng.state();calls=[]
   deadline=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat();rank0=rank_indices(22,world,0)
   source=(FIXTURES/'installed_multi_gpu_test.py.txt').read_text();self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),MULTI_SHA)
   def make(counts,collector):
    def collective(values,item):
     rng.draw()
     for rank in range(world):values[rank]=dict(item,rank=rank,committed=counts[rank])
    torch=types.SimpleNamespace(no_grad=contextlib.nullcontext,distributed=types.SimpleNamespace(get_rank=lambda:0,get_world_size=lambda:world,all_gather_object=collective))
    ns=dict(torch=torch,get_dist_info=lambda:(0,world),time=types.SimpleNamespace(sleep=lambda _:None),mmcv=types.SimpleNamespace(ProgressBar=lambda _:types.SimpleNamespace(update=lambda:None)),collect_results_cpu=collector);exec(source,ns)
    return ns['multi_gpu_test'],torch
   def loader():
    x=FakeLoader(rng)
    x.batch_sampler.sampler=type('Sampler',(),dict(shuffle=False,__iter__=lambda _:iter(rank0)))()
    return x
   baseline,_=make([0]*world,lambda part,size,tmpdir=None:part)
   baseline_rows=baseline(Model(rng),loader(),'cpu_collect');baseline_rng=rng.state()
   assert len(baseline_rows)==len(rank0)
   expected=[[row(tokens[i]) for i in rank_indices(22,world,r)] for r in range(world)];expected[0]=baseline_rows
   expected=[x for group in zip(*expected) for x in group][:22]
   contract=dict(world_size=world,anchors=22,source='pinned_cpu_test_source',filesystem_collector=sha(TOOLS/'file_collection.py'))
   def observed(stage,mode,counts,stop=0):
    collection=root/('collect_'+stage);digest=create_collection(collection,run_id=stage,world_size=world,size=22,source_sha256='a'*64,sample_field='token',expected_ids=tokens)
    for rank in range(1,world):publish_part([row(tokens[i]) for i in rank_indices(22,world,rank)],22,rank=rank,world_size=world,root=collection,run_id=stage,contract_sha256=digest,deadline_utc=deadline)
    def collect(part,size,tmpdir=None):
     calls.append(dict(stage=stage,rows=len(part)))
     return collect_results_file(part,size,rank=0,world_size=world,root=collection,run_id=stage,contract_sha256=digest,deadline_utc=deadline)
    original,torch=make(counts,collect)
    return observed_test(original,Model(rng),loader(),'cpu_collect',False,torch=torch,contract=contract,journal_root=root/'journal',mode=mode,guard=types.SimpleNamespace(check=lambda:None),get_rng=rng.state,restore_rng=rng.restore,stop_after_steps=stop)
   rng.restore(initial)
   with self.assertRaises(JournalPause):observed('stop1','fresh',[0]*world,1)
   before={p.relative_to(root/'journal').as_posix():sha(p) for p in (root/'journal').rglob('*') if p.is_file() and 'attempts' not in p.parts and p.name!='writer.lock'}
   self.assertFalse((root/'collect_stop1/part_000000.pkl').exists())
   rng.restore(initial);resumed,report=observed('resumed','resume',[1]*world)
   self.assertEqual([semantic_digest(x) for x in resumed],[semantic_digest(x) for x in expected]);self.assertEqual(rng.state(),baseline_rng)
   self.assertEqual(report['committed_after'],len(rank0));self.assertEqual(report['replayed_exact_steps'],1)
   for path,value in before.items():self.assertEqual(sha(root/'journal'/path),value)
   rng.restore(initial);replay,report=observed('replay','resume',[len(rank0)]*world)
   self.assertEqual([semantic_digest(x) for x in replay],[semantic_digest(x) for x in expected]);self.assertEqual(rng.state(),baseline_rng)
   self.assertEqual(report['common_resume_step'],len(rank0)-1);self.assertEqual(report['replayed_exact_steps'],1);self.assertEqual(report['new_committed_steps'],0)
   self.assertEqual(calls,[dict(stage='resumed',rows=len(rank0)),dict(stage='replay',rows=len(rank0))])
   reports=root/'reports';digest=create_collection(reports,run_id='reports',world_size=world,size=world,source_sha256='a'*64)
   for rank in range(1,world):publish_part([dict(rank=rank)],world,rank=rank,world_size=world,root=reports,run_id='reports',contract_sha256=digest,deadline_utc=deadline)
   self.assertEqual(collect_results_file([dict(rank=0)],world,rank=0,world_size=world,root=reports,run_id='reports',contract_sha256=digest,deadline_utc=deadline),[dict(rank=r) for r in range(world)])
 @unittest.skipUnless(FIXTURES is not None,'Extract the SHA-locked installed MMDetection fixtures first')
 def test_world4(self):self.case(4)
 @unittest.skipUnless(FIXTURES is not None,'Extract the SHA-locked installed MMDetection fixtures first')
 def test_world12(self):self.case(12)
 @unittest.skipUnless(FIXTURES is not None,'Extract the SHA-locked installed MMDetection fixtures first')
 def test_world16(self):self.case(16)
 def test_only_observer_world_gate_changed(self):
  import ast
  original=ast.parse((TOOLS/'journal_inference.py').read_text());candidate=ast.parse((TOOLS/'filesystem_journal_observer.py').read_text())
  def body(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed_test')
  a,b=body(original),body(candidate)
  self.assertEqual(len(a.body),len(b.body));changed=[i for i,(x,y) in enumerate(zip(a.body,b.body)) if ast.dump(x)!=ast.dump(y)]
  self.assertEqual(len(changed),1);self.assertIsInstance(a.body[changed[0]],ast.Expr);self.assertIn('Unchanged original world16',ast.unparse(a.body[changed[0]]));self.assertIn('(4, 12, 16)',ast.unparse(b.body[changed[0]]))

class CooperativeWaitGuard(unittest.TestCase):
 def test_signal_guard_interrupts_wait_without_STOP(self):
  import time
  from file_collection import wait_and_merge
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'collection';digest=create_collection(root,run_id='signal',world_size=4,size=22,source_sha256='a'*64)
   def stopped():raise RuntimeError('owned signal requested')
   started=time.monotonic()
   with self.assertRaisesRegex(RuntimeError,'owned signal requested'):wait_and_merge(22,world_size=4,root=root,run_id='signal',contract_sha256=digest,deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat(),guard_callback=stopped)
   self.assertLess(time.monotonic()-started,2);self.assertFalse((root/'merge_complete.json').exists())
 def test_signal_guard_interrupts_publish_before_payload(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'collection';digest=create_collection(root,run_id='signal',world_size=4,size=4,source_sha256='a'*64)
   def stopped():raise RuntimeError('owned signal requested')
   with self.assertRaisesRegex(RuntimeError,'owned signal requested'):collect_results_file([dict(rank=0)],4,rank=0,world_size=4,root=root,run_id='signal',contract_sha256=digest,deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat(),guard_callback=stopped)
   self.assertFalse((root/'part_000000.pkl').exists())

class CurrentFilesystemSourceContract(unittest.TestCase):
 def test_current_filesystem_source_and_optimized_gate_agree(self):
  import filesystem_journal_inference as entry
  import optimized_journal_inference as optimized
  expected='a0d8820b69d5bce3d61a8a6cc6b8135c1c3a4ffbd8623e3cd10b14784d989a81'
  self.assertEqual(entry.fingerprint(TOOLS.parents[1]),expected)
  self.assertEqual(entry.OPTIMIZED_SOURCE_SHA,expected)
  self.assertEqual(entry.OPTIMIZED_SOURCE_SHA,optimized.OPTIMIZED_SOURCE_SHA)
  for mode,chunk in [('none',2048),('point-parallel',8192),('point-parallel-last-frame',8192)]:
   self.assertEqual(entry.optimization_contract(expected,expected,mode,chunk,'cuda-compatible')['query_chunk'],chunk)
 def test_old_optimized_journal_source_cannot_be_adopted(self):
  import filesystem_journal_inference as entry
  old='7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb'
  with self.assertRaisesRegex(RuntimeError,'Unsupported pinned runtime source'):
   entry.optimization_contract(old,entry.OPTIMIZED_SOURCE_SHA,'point-parallel-last-frame',8192,'cuda-compatible')
  with self.assertRaisesRegex(RuntimeError,'fingerprint differs'):
   entry.optimization_contract(entry.OPTIMIZED_SOURCE_SHA,old,'point-parallel-last-frame',8192,'cuda-compatible')

if __name__=='__main__':unittest.main(verbosity=2)
