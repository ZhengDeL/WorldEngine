"""CPU replay contracts including toy CPU models and AdamW; no NPU/CUDA jobs."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
FIXTURES=HERE/'fixtures/epoch_resume'
helper_path=SOURCE/'ascend/epoch_cursor.py'
spec=importlib.util.spec_from_file_location('candidate_epoch_cursor',helper_path)
cursor=importlib.util.module_from_spec(spec);spec.loader.exec_module(cursor)
MMCV_SHA='a245963572e675811c8caa8b7e110160567289fa6c1d7519aa04e913b4fc7f1b'
OLD_SHA='bf48ddb35cca4aecc6d15007fee0d107c8463af55ba679441146ccb395693369'
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
try:
    import torch as real_torch
except ImportError:
    real_torch=None


class Streams:
    names=('python','numpy','cpu','npu')
    def __init__(self,seed=123):self.generators={name:random.Random(seed+i) for i,name in enumerate(self.names)}
    def draw(self,name):return self.generators[name].random()
    def state(self):return {name:gen.getstate() for name,gen in self.generators.items()}
    def restore(self,state):
        for name in self.names:self.generators[name].setstate(state[name])


class Flags:
    ndim=1
    dtype=SimpleNamespace(hasobject=False,str='|u1')
    def __init__(self,n):self.shape=(n,)
    def __len__(self):return self.shape[0]
    def tobytes(self,order='C'):return bytes(self.shape[0])


class Dataset:
    def __init__(self,rng,n=16):self.rng=rng;self.flag=Flags(n);self.calls=[]
    def __len__(self):return len(self.flag)
    def __getitem__(self,index):
        self.calls.append(index)
        return dict(token=index,augmentation=[self.rng.draw(k) for k in self.rng.names])


class DistributedGroupSampler:
    def __init__(self,n):
        self.n=n;self.seed=55;self.num_replicas=1;self.samples_per_gpu=2;self.num_samples=n;self.total_size=n
        self.epoch=0;self.rank=0;self.skip_iter_at_epoch=False
    def __iter__(self):
        indices=list(range(self.n));random.Random(self.seed+self.epoch).shuffle(indices);return iter(indices)
    def __len__(self):return self.n
    def set_epoch(self,epoch):self.epoch=epoch


class BatchSampler:
    def __init__(self,sampler,batch=2):self.sampler=sampler;self.batch_size=batch;self.drop_last=False
    def __len__(self):return (len(self.sampler)+self.batch_size-1)//self.batch_size
    def __iter__(self):
        values=list(iter(self.sampler))
        for offset in range(0,len(values),self.batch_size):yield values[offset:offset+self.batch_size]


class Loader:
    num_workers=0;_auto_collation=True;persistent_workers=False
    def __init__(self,rng,n=16):self.dataset=Dataset(rng,n);self.sampler=DistributedGroupSampler(n);self.batch_sampler=BatchSampler(self.sampler)
    def __len__(self):return len(self.batch_sampler)
    def __iter__(self):
        self.dataset.rng.draw('cpu')  # Real DataLoader base-seed draw.
        sampler=iter(self.batch_sampler)
        dataset=self.dataset
        class Iterator:
            def __iter__(self):return self
            def __next__(self):
                values=[dataset[index] for index in next(sampler)]
                return dict(img_metas=[dict(sample_idx=str(x['token'])) for x in values],values=values)
        return Iterator()


class Loss(float):
    def detach(self):return float(self)


class Model:
    def __init__(self,env):self.env=env;self.module=self;self.values=dict(weight=.25,buffer=.5)
    def train(self):pass
    def state_dict(self):return copy.deepcopy(self.values)
    def load_state_dict(self,state,strict=True):self.values=copy.deepcopy(state)
    def parameters(self):return [SimpleNamespace(grad=1.)]


class Optimizer:
    def __init__(self):self.param_groups=[dict(lr=.0002)];self.values=dict(step=0,exp_avg=0.,exp_avg_sq=0.)
    def state_dict(self):return dict(state=copy.deepcopy(self.values),param_groups=copy.deepcopy(self.param_groups))
    def load_state_dict(self,state):self.values=copy.deepcopy(state['state']);self.param_groups=copy.deepcopy(state['param_groups'])


def environment(output,interval=3,pause_at=None):
    env=SimpleNamespace(rng=Streams(),output=output,interval=interval,pause_at=pause_at,rows=[],after_run=False)
    class Base:
        def __init__(self,**kwargs):
            self.env=env;self.model=Model(env);self.optimizer=Optimizer();self.meta={};self._iter=0;self._epoch=0;self._inner_iter=0
            self._max_epochs=8;self._max_iters=64;self._stop_iters=-1;self.work_dir=str(output)
            self.logger=SimpleNamespace(info=lambda *args:None)
        @property
        def iter(self):return self._iter
        @property
        def epoch(self):return self._epoch
        @property
        def inner_iter(self):return self._inner_iter
        @property
        def max_epochs(self):return self._max_epochs
        def get_hook_info(self):return 'CPU fixture hooks'
        def call_hook(self,name):
            if name=='before_run':
                self.hook.rank=0;self.hook.model=self.model;self.hook.run_started=time.monotonic()
            elif name=='before_train_epoch':
                self.data_loader.sampler.set_epoch(self.epoch)
                self.env.rng.draw('numpy')  # A real hook may consume RNG.
            elif name=='before_train_iter':self.hook.before_train_iter(self)
            elif name=='after_train_iter':
                if env.pause_at==self._iter+1:(output/'PAUSE').write_text('fixture cooperative pause')
                self.hook.after_train_iter(self)
            elif name=='after_train_epoch':self.hook.after_train_epoch(self)
            elif name=='after_run':env.after_run=True
        def run_iter(self,data_batch,train_mode=True,**kwargs):
            grad=sum(sum(v['augmentation']) for v in data_batch['values'])+env.rng.draw('npu')
            o=self.optimizer.values;o['step']+=1;o['exp_avg']=.9*o['exp_avg']+.1*grad;o['exp_avg_sq']=.999*o['exp_avg_sq']+.001*grad*grad
            # Preserve an eight-epoch global horizon and500-step warmup.
            lr=.0002*(1/3+(2/3)*min(self.iter,499)/500)
            self.optimizer.param_groups[0]['lr']=lr
            self.model.values['weight']-=lr*o['exp_avg'];self.model.values['buffer']+=grad*.001
            self.outputs=dict(loss=Loss(grad),num_samples=len(data_batch['values']))
            self.hook.current_raw_losses=dict(loss=grad)
            env.rows.append(dict(iter=self.iter+1,epoch=self.epoch,inner=self.inner_iter,input=copy.deepcopy(data_batch),lr=lr,
                model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),rng=env.rng.state()))
    mmcv_source=(FIXTURES/'mmcv_epoch_based_runner.py').read_bytes()
    assert hashlib.sha256(mmcv_source).hexdigest()==MMCV_SHA
    parsed=ast.parse(mmcv_source);original=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='EpochBasedRunner')
    methods=[copy.deepcopy(n) for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('train','run')]
    ns=dict(time=SimpleNamespace(sleep=lambda _:None),warnings=SimpleNamespace(warn=lambda *a:None),
        mmcv=SimpleNamespace(is_list_of=lambda a,b:all(isinstance(x,b) for x in a)),get_host_info=lambda:'CPUfixture')
    tree=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[]))
    exec(compile(tree,'<exact installed MMCV train/run AST>','exec'),ns)
    Base.train=ns['train'];Base.run=ns['run']
    class Flag:
        def __init__(self,value):self.value=value
        def item(self):return self.value
    fake_npu=SimpleNamespace(synchronize=lambda:None,memory_allocated=lambda:0,memory_reserved=lambda:0)
    fake_torch=SimpleNamespace(npu=fake_npu,tensor=lambda value,device=None:Flag(value),is_tensor=lambda _:False,
        load=lambda path,**kwargs:pickle.loads(Path(path).read_bytes()))
    def gather(states,current):
        states[0]=copy.deepcopy(current);env.rng.draw('cpu')  # Checkpoint collective side effect neutralized by restore.
    fake_dist=SimpleNamespace(get_rank=lambda:0,get_world_size=lambda:1,all_gather_object=gather,barrier=lambda:None,
        all_reduce=lambda *args,**kwargs:None,ReduceOp=SimpleNamespace(MAX='MAX'))
    def save(model,path,optimizer,meta):
        Path(path).write_bytes(pickle.dumps(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),meta=meta)))
    class ProcPath:
        def read_text(self):return '1000 500'
    def paths(value):return ProcPath() if str(value)=='/proc/self/statm' else Path(value)
    registry=SimpleNamespace(register_module=lambda:lambda cls:cls)
    source=(SOURCE/'ascend/runtime.py').read_bytes();parsed=ast.parse(source)
    selected=[]
    for node in parsed.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ('ResumeMixin','AscendEpochRunner','epoch_sampler_contract','epoch_finalization_only','AscendRuntimeHook'):
            node=copy.deepcopy(node)
            if node.name=='AscendRuntimeHook':
                node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in (
                    '__init__','before_train_iter','checkpoint','after_train_iter','after_train_epoch')]
            selected.append(node)
    runtime=dict(EpochBasedRunner=Base,IterBasedRunner=type('IterBasedRunner',(),{}),Hook=object,RUNNERS=registry,HOOKS=registry,
        EpochCursorLoader=cursor.EpochCursorLoader,EpochPaused=cursor.EpochPaused,validate_cursor=cursor.validate_cursor,
        torch=fake_torch,dist=fake_dist,np=SimpleNamespace(asarray=lambda value:value),hashlib=hashlib,json=json,
        os=SimpleNamespace(path=os.path,sysconf=lambda _:4096),Path=paths,time=time,
        resource=SimpleNamespace(getrusage=lambda _:SimpleNamespace(ru_maxrss=1000),RUSAGE_SELF=0),
        rng_state=env.rng.state,restore_rng=env.rng.restore,save_checkpoint=save,finite=lambda _:True)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'<candidate runtime actual methods>','exec'),runtime)
    runner=runtime['AscendEpochRunner'](resume_contract=dict(runner=dict(type='EpochBasedRunner',max_epochs=8),precision=dict(mode='fp32'),world_size=1,batch=2,workers=0))
    runner.hook=runtime['AscendRuntimeHook'](output,'CPUfixture',checkpoint_interval=interval,check_interval=1)
    env.runner=runner;env.loader=Loader(env.rng);env.runtime=runtime
    return env


class EpochReplay(unittest.TestCase):
    def test_exact_mid_epoch_original_MMCV_loop_and_four_RNG_without_extra_getitem(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);full=root/'full';first=root/'first';resumed=root/'resumed'
            for p in (full,first,resumed):p.mkdir()
            continuous=environment(full);continuous.runner.run([continuous.loader],[('train',1)])
            partial=environment(first,pause_at=3);partial.runner.run([partial.loader],[('train',1)])
            self.assertTrue(partial.after_run);self.assertEqual((partial.runner.iter,partial.runner.epoch),(3,0))
            cp=pickle.loads((first/'iter_3.pth').read_bytes());state=cp['meta']['ascend_resume']
            self.assertEqual(state['epoch_cursor']['completed_batches'],3);self.assertTrue(state['epoch_cursor']['iterator_created'])
            self.assertEqual((cp['meta']['iter'],cp['meta']['epoch']),(3,0));self.assertIn('optimizer',cp)
            restored=environment(resumed);restored.runner.resume(first/'iter_3.pth')
            restored.runner.run([restored.loader],[('train',1)])
            self.assertEqual(partial.rows+restored.rows,continuous.rows)
            self.assertEqual(restored.rng.state(),continuous.rng.state())
            self.assertEqual(len(partial.loader.dataset.calls),6)
            self.assertEqual(len(restored.loader.dataset.calls),len(continuous.loader.dataset.calls)-6)
            self.assertEqual((restored.runner.iter,restored.runner.epoch,restored.runner._max_iters),(64,8,64))

    def test_epoch_boundary_retains_before_epoch_hook_and_base_seed_draw(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ('full','resumed'):(root/name).mkdir()
            full=environment(root/'full');full.runner.run([full.loader],[('train',1)])
            checkpoint=root/'full/epoch_1.pth';cp=pickle.loads(checkpoint.read_bytes());state=cp['meta']['ascend_resume']
            self.assertEqual((state['completed_updates'],state['completed_epochs']),(8,1))
            self.assertEqual(state['epoch_cursor']['completed_batches'],0);self.assertFalse(state['epoch_cursor']['iterator_created'])
            restored=environment(root/'resumed');restored.runner.resume(checkpoint);restored.runner.run([restored.loader],[('train',1)])
            self.assertEqual(restored.rows,full.rows[8:]);self.assertEqual(restored.rng.state(),full.rng.state())

    def test_pause_on_last_batch_does_not_skip_original_epoch_hooks(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ('full','first','resumed'):(root/name).mkdir()
            full=environment(root/'full');full.runner.run([full.loader],[('train',1)])
            partial=environment(root/'first',pause_at=8);partial.runner.run([partial.loader],[('train',1)])
            restored=environment(root/'resumed');restored.runner.resume(root/'first/iter_8.pth');restored.runner.run([restored.loader],[('train',1)])
            self.assertEqual(partial.rows+restored.rows,full.rows)
            self.assertEqual(restored.rng.state(),full.rng.state())

    def test_checkpoint_interval_is_update_based_and_original_horizon_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);env=environment(root,interval=3,pause_at=5);env.runner.run([env.loader],[('train',1)])
            self.assertTrue((root/'iter_3.pth').is_file());self.assertTrue((root/'iter_5.pth').is_file())
            self.assertFalse((root/'iter_4.pth').exists())
            cp=pickle.loads((root/'iter_5.pth').read_bytes())
            self.assertEqual(cp['meta']['ascend_resume']['contract']['runner']['max_epochs'],8)
            self.assertEqual(env.runner.epoch,0)
            if os.name=='posix':
                self.assertTrue((root/'latest.pth').is_symlink())
                self.assertEqual((root/'latest.pth').resolve(),root/'iter_5.pth')

    def test_invalid_cursor_or_changed_sampler_rejected_before_dataset_getitem(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);first=root/'first';first.mkdir();env=environment(first,pause_at=3);env.runner.run([env.loader],[('train',1)])
            saved=pickle.loads((first/'iter_3.pth').read_bytes())
            for index,change in enumerate((lambda c:c.update(completed_batches=9),lambda c:c.update(loader_batches=7),
                    lambda c:c.update(iterator_created=False),lambda c:c['sampler'].update(seed=999))):
                bad=copy.deepcopy(saved);change(bad['meta']['ascend_resume']['epoch_cursor'])
                path=root/('bad%d.pth'%index);path.write_bytes(pickle.dumps(bad));out=root/('r%d'%index);out.mkdir()
                restored=environment(out);restored.runner.resume(path)
                with self.assertRaises(RuntimeError):restored.runner.run([restored.loader],[('train',1)])
                self.assertEqual(restored.loader.dataset.calls,[])

    def test_batch_indices_only_and_original_loader_restored(self):
        rng=Streams();loader=Loader(rng);before=loader.batch_sampler
        proxy=cursor.EpochCursorLoader(loader,3,rng.state(),rng.restore)
        iterator=iter(proxy);self.assertIs(loader.batch_sampler,before);self.assertEqual(loader.dataset.calls,[])
        list(iterator);self.assertEqual(len(loader.dataset.calls),10);self.assertEqual(len(proxy),8)
        with self.assertRaises(RuntimeError):iter(proxy)

    def test_workers_and_nonbatch_modes_rejected(self):
        for name,value in [('num_workers',1),('_auto_collation',False),('persistent_workers',True)]:
            loader=Loader(Streams());setattr(loader,name,value)
            with self.assertRaises(RuntimeError):cursor.EpochCursorLoader(loader,1,{},lambda _:None)

    def test_iterator_creation_failure_restores_original_loader_without_getitem(self):
        class Broken(Loader):
            def __iter__(self):raise RuntimeError('iterator setup failed')
        loader=Broken(Streams());original=loader.batch_sampler;restored=[]
        proxy=cursor.EpochCursorLoader(loader,1,{},lambda state:restored.append(state))
        with self.assertRaises(RuntimeError):iter(proxy)
        self.assertIs(loader.batch_sampler,original);self.assertEqual(loader.dataset.calls,[])
        self.assertEqual(restored,[])

    def test_sampler_rank_world_batch_and_flag_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            env=environment(Path(t));contract=env.runtime['epoch_sampler_contract']
            for name,value in [('rank',2),('num_replicas',2),('samples_per_gpu',1),('skip_iter_at_epoch',True)]:
                loader=Loader(Streams());setattr(loader.sampler,name,value)
                with self.assertRaises(RuntimeError):contract(loader)

    def test_cursor_schema_and_counter_negative_cases(self):
        valid=dict(schema_version=1,epoch=2,completed_batches=3,loader_batches=8,sampler=dict(seed=1),iterator_created=True)
        self.assertEqual(cursor.validate_cursor(valid,completed_updates=19,completed_epochs=2,loader_batches=8),3)
        for mutate in (lambda d:d.update(epoch=True),lambda d:d.update(completed_batches=-1),lambda d:d.update(schema_version=2),
                       lambda d:d.update(extra=1),lambda d:d.update(iterator_created=False)):
            wrong=copy.deepcopy(valid);mutate(wrong)
            with self.assertRaises(RuntimeError):cursor.validate_cursor(wrong,completed_updates=19,completed_epochs=2,loader_batches=8)

    def test_iter_runtime_AST_is_unchanged(self):
        old=(FIXTURES/'runtime_before.py').read_bytes();self.assertEqual(hashlib.sha256(old).hexdigest(),OLD_SHA)
        a=ast.parse(old);b=ast.parse((SOURCE/'ascend/runtime.py').read_bytes())
        olddefs={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};newdefs={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name in ('AscendIterRunner','ResumeMixin','rng_state','restore_rng','finite','optimizer_counters'):
            self.assertEqual(ast.dump(olddefs[name]),ast.dump(newdefs[name]))
        oldhook=next(n for n in olddefs['AscendRuntimeHook'].body if isinstance(n,ast.FunctionDef) and n.name=='after_train_iter')
        newhook=next(n for n in newdefs['AscendRuntimeHook'].body if isinstance(n,ast.FunctionDef) and n.name=='after_train_iter')
        # Existing IterBased branch is byte-for-byte AST identical; only a new
        # sibling elif implements the epoch checkpoint/pause behavior.
        oldbranch=next(n for n in oldhook.body if isinstance(n,ast.If) and ast.unparse(n.test)=='isinstance(runner, IterBasedRunner)')
        newbranch=next(n for n in newhook.body if isinstance(n,ast.If) and ast.unparse(n.test)=='isinstance(runner, IterBasedRunner)')
        self.assertEqual(ast.dump(ast.Module(body=oldbranch.body,type_ignores=[])),ast.dump(ast.Module(body=newbranch.body,type_ignores=[])))

    @unittest.skipIf(real_torch is None,'Requires target actual PyTorch CPU DataLoader; target must have zero skips')
    def test_actual_torch_dataloader_base_seed_and_no_replayed_getitem(self):
        class ActualDataset(real_torch.utils.data.Dataset):
            def __init__(self):self.calls=[]
            def __len__(self):return 16
            def __getitem__(self,index):self.calls.append(index);return index,random.random()
        random.seed(123);real_torch.manual_seed(321)
        original=ActualDataset();loader=real_torch.utils.data.DataLoader(original,batch_size=2,num_workers=0)
        iterator=iter(loader)
        for _ in range(3):next(iterator)
        saved=(random.getstate(),real_torch.random.get_rng_state().clone())
        expected=[(a.clone(),b.clone()) for a,b in iterator];final=(random.getstate(),real_torch.random.get_rng_state().clone())
        random.seed(999);real_torch.manual_seed(888);target=ActualDataset()
        def restore(state):random.setstate(state[0]);real_torch.random.set_rng_state(state[1])
        proxy=cursor.EpochCursorLoader(real_torch.utils.data.DataLoader(target,batch_size=2,num_workers=0),3,saved,restore)
        actual=list(proxy)
        self.assertEqual(len(target.calls),10);self.assertEqual(target.calls,list(range(6,16)))
        self.assertTrue(all(real_torch.equal(a,x) and real_torch.equal(b,y) for (a,b),(x,y) in zip(actual,expected)))
        self.assertEqual(random.getstate(),final[0]);self.assertTrue(real_torch.equal(real_torch.random.get_rng_state(),final[1]))


def actual_after_run(env,changed=False,expected_pg=False):
    """Execute the candidate's complete after_run body, not a mirrored guard."""
    class Watch:
        def __init__(self,value):self.value=value
        def detach(self):return self
        def contiguous(self):return self
        def __ne__(self,other):return SimpleNamespace(any=lambda:self.value!=other.value)
    runner=env.runner;hook=runner.hook;marker=runner._ascend_epoch_finalization
    hook.initial_iter=marker['completed_updates'];hook.initial_epoch=marker['completed_epochs']
    hook.expected_pg=expected_pg;hook.before=Watch(1.);hook.watch=Watch(2. if changed else 1.)
    if expected_pg:runner.resume_contract['expected_pg']=True
    hook.raw_losses=[];hook.raw_loss_count=0;hook.initialization_matches=[];hook.optimizer_initial_steps=[runner.iter]
    hook.device_provenance={};hook.watched_parameter_name='fixture.weight';hook.pg_observed_nonzero=False
    runner.model.parameters=lambda:[SimpleNamespace(numel=lambda:1,requires_grad=True)]
    precision=dict(mode='fp32',public={},torch_deterministic=False,environment={},msda={},
        raw_options=dict(ALLOW_MATMUL_HF32=None,ALLOW_CONV_HF32=None))
    runner.resume_contract['precision']=copy.deepcopy(precision)
    env.runtime['precision_state']=lambda *args,**kwargs:copy.deepcopy(precision)
    env.runtime['torch'].empty_like=lambda value:Watch(value.value)
    env.runtime['torch'].allclose=lambda left,right,**kwargs:left.value==right.value
    env.runtime['torch'].npu.max_memory_allocated=lambda:0
    env.runtime['torch'].npu.max_memory_reserved=lambda:0
    env.runtime['dist'].all_gather=lambda output,value:output.__setitem__(0,Watch(value.value))
    tree=ast.parse((SOURCE/'ascend/runtime.py').read_bytes())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='AscendRuntimeHook')
    method=next(copy.deepcopy(node) for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='after_run')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<actual after_run full body>','exec'),env.runtime)
    env.runtime['after_run'](hook,runner)
    return json.loads((env.output/'runtime_rank0.json').read_text())


class FinalEpochFinalization(unittest.TestCase):
    def prepared(self,root):
        first=root/'first';resumed=root/'resumed';first.mkdir();resumed.mkdir()
        partial=environment(first,pause_at=64);partial.runner.run([partial.loader],[('train',1)])
        saved=pickle.loads((first/'iter_64.pth').read_bytes())
        self.assertEqual((saved['meta']['iter'],saved['meta']['epoch']),(64,7))
        self.assertEqual(saved['meta']['ascend_resume']['epoch_cursor']['completed_batches'],8)
        restored=environment(resumed);restored.runner.resume(first/'iter_64.pth')
        restored.runner.run([restored.loader],[('train',1)])
        self.assertEqual(restored.rows,[]);self.assertEqual(restored.loader.dataset.calls,[])
        self.assertEqual((restored.runner.iter,restored.runner.epoch),(64,8))
        return partial,restored

    def test_final_epoch_last_batch_no_updates_real_after_run_only_finalization(self):
        with tempfile.TemporaryDirectory() as t:
            partial,restored=self.prepared(Path(t))
            self.assertEqual(restored.runner.model.state_dict(),partial.runner.model.state_dict())
            self.assertEqual(restored.runner.optimizer.state_dict(),partial.runner.optimizer.state_dict())
            self.assertEqual(restored.rng.state(),partial.rng.state())
            report=actual_after_run(restored)
            self.assertEqual(report['result'],'pass_epoch_finalization_only_no_optimizer_updates')
            self.assertEqual(report['updates_this_run'],0);self.assertEqual(report['train_seconds'],0.)
            self.assertFalse(report['parameter_updated']);self.assertTrue(report['watched_parameter_sync'])
            self.assertTrue(report['epoch_finalization_only']);self.assertTrue(report['original_epoch_horizon_complete'])
            self.assertEqual(report['final_iter'],64);self.assertEqual(report['final_epoch'],8)
            self.assertTrue((restored.output/'epoch_8.pth').is_file())

    def test_wrong_saved_boundary_restore_epoch_update_or_precision_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _,restored=self.prepared(Path(t));runner=restored.runner;hook=runner.hook
            marker=runner._ascend_epoch_finalization;hook.initial_iter=64;hook.initial_epoch=7;hook.expected_pg=False
            helper=restored.runtime['epoch_finalization_only'];self.assertTrue(helper(runner,hook))
            for field,value in [('completed_updates',63),('completed_epochs',6),('completed_batches',7),('loader_batches',0)]:
                before=marker[field];marker[field]=value
                with self.subTest(field=field),self.assertRaises(RuntimeError):helper(runner,hook)
                marker[field]=before
            for name,value in [('steps',1),('initial_iter',63),('initial_epoch',6),('expected_pg',True)]:
                before=getattr(hook,name);setattr(hook,name,value)
                with self.subTest(name=name),self.assertRaises(RuntimeError):helper(runner,hook)
                setattr(hook,name,before)
            runner.resume_contract['precision']['mode']='bf16'
            with self.assertRaises(RuntimeError):helper(runner,hook)
            runner.resume_contract['precision']['mode']='fp32';runner._ascend_restoration['result']='unknown'
            with self.assertRaises(RuntimeError):helper(runner,hook)

    def test_zero_update_cannot_change_parameter_or_accept_unmarked_fresh_run(self):
        with tempfile.TemporaryDirectory() as t:
            _,restored=self.prepared(Path(t))
            with self.assertRaisesRegex(RuntimeError,'Parameter changed during zero-update'):actual_after_run(restored,changed=True)
            actual_after_run(restored)
            del restored.runner._ascend_epoch_finalization
            with self.assertRaisesRegex(RuntimeError,'did not update'):restored.runtime['after_run'](restored.runner.hook,restored.runner)


@unittest.skipIf(real_torch is None, 'Actual PyTorch/MMCV CPU regression requires the target stack; target must have zero skips')
class InstalledMMCVWarmupResume(unittest.TestCase):
    """Real LR hooks and AdamW; a CPU sampler fixture, not distributed acceptance."""

    @classmethod
    def setUpClass(cls):
        import logging
        import numpy as np
        from mmcv.runner import EpochBasedRunner, Hook
        from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook
        from mmcv.runner.hooks.sampler_seed import DistSamplerSeedHook
        cls.np, cls.Hook = np, Hook
        cls.Base, cls.LRHook, cls.SamplerHook = EpochBasedRunner, CosineAnnealingLrUpdaterHook, DistSamplerSeedHook
        cls.logger = logging.getLogger('installed-mmcv-warmup-cpu')
        selected = [copy.deepcopy(node) for node in ast.parse((SOURCE/'ascend/runtime.py').read_bytes()).body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name in ('AscendEpochRunner', 'epoch_sampler_contract', 'epoch_finalization_only')]
        if len(selected) != 3:
            raise RuntimeError('Exact production epoch runner, sampler contract and finalization gate required')
        namespace = dict(json=json, hashlib=hashlib, np=np, EpochBasedRunner=EpochBasedRunner,
            ResumeMixin=type('UnusedResumeMixin', (), {}),
            RUNNERS=SimpleNamespace(register_module=lambda: lambda value: value),
            EpochCursorLoader=cursor.EpochCursorLoader, EpochPaused=cursor.EpochPaused,
            validate_cursor=cursor.validate_cursor, restore_rng=cls.restore_rng,
            dist=SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 16))
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<production epoch methods, installed MMCV>', 'exec'), namespace)
        cls.Runner, cls.sampler_contract = namespace['AscendEpochRunner'], staticmethod(namespace['epoch_sampler_contract'])

    @classmethod
    def rng(cls):
        return (random.getstate(), cls.np.random.get_state(), real_torch.random.get_rng_state().clone())

    @classmethod
    def restore_rng(cls, state):
        random.setstate(state[0]); cls.np.random.set_state(state[1]); real_torch.random.set_rng_state(state[2])

    def setUp(self):
        self.initial_rng = self.rng()
        self.addCleanup(self.restore_rng, self.initial_rng)

    def assert_raw_state(self, actual, expected):
        import struct
        self.assertIs(type(actual), type(expected))
        if real_torch.is_tensor(actual):
            self.assertEqual(actual.device.type, 'cpu'); self.assertEqual(expected.device.type, 'cpu')
            self.assertEqual(actual.dtype, expected.dtype); self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(real_torch.equal(actual.contiguous().reshape(-1).view(real_torch.uint8),
                                            expected.contiguous().reshape(-1).view(real_torch.uint8)))
        elif isinstance(actual, self.np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype); self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.tobytes(order='C'), expected.tobytes(order='C'))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual: self.assert_raw_state(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected): self.assert_raw_state(value, reference)
        elif isinstance(actual, float):
            self.assertEqual(struct.pack('>d', actual), struct.pack('>d', expected))
        else:
            self.assertEqual(actual, expected)

    def run_cpu(self, *, saved=None, batches=1, pause=None, original_loop=False):
        np, torch, test = self.np, real_torch, self
        random.seed(31); np.random.seed(32); torch.manual_seed(33)

        class ActualDataset(torch.utils.data.Dataset):
            def __init__(self): self.flag=np.zeros(16*batches, dtype=np.uint8)
            def __len__(self): return len(self.flag)
            def __getitem__(self, index):
                return torch.tensor([float(index)+random.random()+float(np.random.random())+float(torch.rand(()))])

        class DistributedGroupSampler(torch.utils.data.Sampler):
            # Controlled CPU fixture. The separate NPU acceptance uses the actual sampler.
            def __init__(self):
                self.seed=20261012; self.num_replicas=16; self.rank=0; self.samples_per_gpu=1
                self.num_samples=batches; self.total_size=16*batches; self.epoch=0; self.skip_iter_at_epoch=False
            def set_epoch(self, epoch): self.epoch=epoch
            def __len__(self): return batches
            def __iter__(self): return iter(range(self.epoch % 16, self.epoch % 16+batches))

        class ActualModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float32))
            def train_step(self, batch, optimizer):
                optimizer.zero_grad()
                loss=(self.weight*torch.tensor([.125,-.25])*(1+batch.sum()/100)).sum()
                loss.backward(); optimizer.step()
                return dict(loss=loss.detach(), num_samples=1)

        class Capture(self.Hook):
            snapshot=None
            def after_train_iter(self, runner):
                if runner.iter+1 != pause: return
                self.snapshot=dict(model=copy.deepcopy(runner.model.state_dict()), optimizer=copy.deepcopy(runner.optimizer.state_dict()),
                    completed_updates=runner.iter+1, completed_epochs=runner.epoch,
                    rng_by_rank=[test.rng() for _ in range(16)],
                    epoch_cursor=dict(schema_version=1,epoch=runner.epoch,completed_batches=runner.inner_iter+1,
                        loader_batches=len(runner.data_loader),iterator_created=True,sampler=test.sampler_contract(runner.data_loader)))
                raise cursor.EpochPaused(runner.iter+1)

        model=ActualModel(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01,foreach=False)
        with tempfile.TemporaryDirectory() as work:
            args=dict(model=model,optimizer=optimizer,work_dir=work,logger=self.logger,max_epochs=8)
            runner=self.Base(**args) if original_loop else self.Runner(resume_contract=dict(runner=dict(max_epochs=8)),**args)
            runner.register_hook(self.LRHook(min_lr_ratio=.001,warmup='linear',warmup_iters=500,warmup_ratio=1/3),priority='VERY_HIGH')
            runner.register_hook(self.SamplerHook(),priority='NORMAL')
            capture=Capture()
            if pause is not None: runner.register_hook(capture,priority='LOWEST')
            if saved is not None:
                model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
                runner._iter,runner._epoch=saved['completed_updates'],saved['completed_epochs']
                runner._ascend_resume=copy.deepcopy(saved)
            runner.call_hook('before_run')
            loader=torch.utils.data.DataLoader(ActualDataset(),batch_size=1,sampler=DistributedGroupSampler(),num_workers=0)
            # The installed loop's artificial epoch-start sleep has no numerical role.
            with patch('mmcv.runner.epoch_based_runner.time.sleep',lambda _:None):
                while runner.epoch<8:
                    try: runner.train(loader)
                    except cursor.EpochPaused: break
            result=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                        rng=self.rng(),updates=runner.iter,epochs=runner.epoch)
        return runner,result,capture.snapshot

    def test_last_batch_zero_update_preserves_actual_warmup_LR_and_full_CPU_state(self):
        _,expected,_=self.run_cpu(original_loop=True)
        _,_,saved=self.run_cpu(pause=8)
        self.assertEqual((saved['completed_updates'],saved['completed_epochs']),(8,7))
        runner,actual,_=self.run_cpu(saved=saved)
        self.assert_raw_state(actual,expected)
        self.assertEqual(actual['updates'],saved['completed_updates'])
        self.assertFalse(hasattr(runner,'_ascend_empty_suffix_lrs'))

    def test_cross_epoch_two_updates_match_one_plus_one(self):
        _,expected,_=self.run_cpu(pause=2)
        _,_,saved=self.run_cpu(pause=1)
        _,actual,_=self.run_cpu(saved=saved,pause=2)
        self.assert_raw_state(actual,expected)

    def test_nonempty_suffix_matches_continuous_actual_CPU_state(self):
        _,expected,_=self.run_cpu(batches=2,pause=2)
        _,_,saved=self.run_cpu(batches=2,pause=1)
        _,actual,_=self.run_cpu(batches=2,saved=saved,pause=2)
        self.assert_raw_state(actual,expected)

    def test_no_resume_matches_unmodified_installed_MMCV_loop(self):
        _,expected,_=self.run_cpu(original_loop=True)
        _,actual,_=self.run_cpu()
        self.assert_raw_state(actual,expected)

    def reject_changed_groups(self, mutate):
        torch=real_torch
        class GroupModel(torch.nn.Linear):
            def train_step(self, data, optimizer):
                raise AssertionError('Parameter-group hook tests must not run an optimizer update')
        model=GroupModel(1,1)
        optimizer=torch.optim.AdamW([dict(params=[model.weight]),dict(params=[model.bias])],lr=.0002,foreach=False)
        with tempfile.TemporaryDirectory() as work:
            runner=self.Runner(resume_contract=dict(runner=dict(max_epochs=8)),model=model,optimizer=optimizer,
                               work_dir=work,logger=self.logger,max_epochs=8)
            class ChangeGroups(self.Hook):
                def before_train_epoch(self, r): mutate(r.optimizer.param_groups)
            runner.register_hook(self.LRHook(min_lr_ratio=.001,warmup='linear',warmup_iters=500,warmup_ratio=1/3),priority='VERY_HIGH')
            runner.register_hook(ChangeGroups(),priority='LOW'); runner.call_hook('before_run')
            runner._ascend_empty_suffix_lrs=[(group,group['lr']) for group in optimizer.param_groups]
            with self.assertRaisesRegex(RuntimeError,'parameter groups changed'): runner.call_hook('before_train_epoch')

    def test_reordered_parameter_groups_rejected(self):
        self.reject_changed_groups(lambda groups:groups.reverse())

    def test_replaced_parameter_group_rejected(self):
        self.reject_changed_groups(lambda groups:groups.__setitem__(0,dict(groups[0])))

    def test_added_parameter_group_rejected(self):
        self.reject_changed_groups(lambda groups:groups.append(dict(groups[0])))


if __name__=='__main__':
    # Windows fixtures use a plain alias only when OS symlink permission is
    # unavailable. Linux target exercises the original actual symlink behavior.
    if os.name=='nt':
        with patch.object(Path,'symlink_to',lambda self,target,**kwargs:self.write_text(str(target))):unittest.main()
    else:unittest.main()
