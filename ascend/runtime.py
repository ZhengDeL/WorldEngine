"""Native optimizer observation and rank-aware, deterministic training resume.

Exact RNG/data replay is supported with workers=0 and unchanged topology/data.
Checkpoints are saved after completed optimizer updates, including every rank's
Python/NumPy/CPU/NPU RNG state. Old weight-only checkpoints cannot resume here.
"""
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, RUNNERS, Hook, IterBasedRunner, EpochBasedRunner
from mmcv.runner.checkpoint import save_checkpoint


def rng_state():
    state=np.random.get_state()
    numpy=dict(generator=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4])
    return dict(python=random.getstate(), numpy=numpy,
                cpu=torch.random.get_rng_state(), npu=torch.npu.get_rng_state().cpu())


def restore_rng(state):
    random.setstate(state['python'])
    numpy=state['numpy']
    if isinstance(numpy,dict):
        numpy=(numpy['generator'],np.asarray(numpy['keys'],dtype=np.uint32),numpy['position'],numpy['has_gauss'],numpy['cached_gaussian'])
    np.random.set_state(numpy)
    torch.random.set_rng_state(state['cpu'].cpu())
    torch.npu.set_rng_state(state['npu'].cpu())


def finite(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)): return all(finite(x) for x in value)
    return True


class ResumeMixin:
    def resume(self, checkpoint, resume_optimizer=True, map_location='cpu'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=saved.get('meta',{}).get('ascend_resume')
        if state is None or 'optimizer' not in saved:
            raise RuntimeError('Resume requires a complete Ascend runtime checkpoint')
        if state['contract'] != self.resume_contract:
            raise RuntimeError('Resume contract changed: data, topology, optimizer or schedule differs')
        if len(state['rng_by_rank']) != dist.get_world_size():
            raise RuntimeError('Resume world size mismatch')
        # This file is produced by our runtime. Read it once with the explicit
        # setting required for complete legacy training state, then restore the
        # full model strictly; never call MMCV's default weights-only reload.
        self.model.module.load_state_dict(saved['state_dict'],strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        counts=dict(model_tensors=0,optimizer_tensors=0)
        def exact(current,expected,kind):
            if torch.is_tensor(expected):
                if not torch.equal(current.detach().cpu(),expected.detach().cpu()):
                    raise RuntimeError('Incomplete resume tensor copy: '+kind)
                counts[kind+'_tensors']+=1
            elif isinstance(expected,dict):
                if set(current)!=set(expected):raise RuntimeError('Resume state keys differ')
                for key in expected:exact(current[key],expected[key],kind)
            elif isinstance(expected,(list,tuple)):
                if len(current)!=len(expected):raise RuntimeError('Resume state length differs')
                for x,y in zip(current,expected):exact(x,y,kind)
            elif current!=expected:raise RuntimeError('Resume scalar/hyperparameter differs')
        exact(self.model.module.state_dict(),saved['state_dict'],'model')
        exact(self.optimizer.state_dict(),saved['optimizer'],'optimizer')
        self._ascend_restoration=dict(result='pass_exact_saved_state_restore',**counts)
        # TextLoggerHook serializes runner.meta to JSON in before_run. Keep the
        # tensor RNG payload in the private resume state, outside logging meta.
        self.meta={k:v for k,v in saved['meta'].items() if k!='ascend_resume'}
        self._iter=state['completed_updates']
        self._epoch=state['completed_epochs']
        self._ascend_resume=state
        self.logger.info('Restored complete Ascend state: %d updates, %d epochs',self._iter,self._epoch)


@RUNNERS.register_module()
class AscendIterRunner(ResumeMixin, IterBasedRunner):
    def __init__(self,resume_contract,**kwargs):
        self.resume_contract=json.loads(json.dumps(resume_contract))
        super().__init__(**kwargs)

    def train(self,data_loader,**kwargs):
        state=getattr(self,'_ascend_resume',None)
        if state is not None:
            # IterLoader's iterator has already been constructed. Rebuild its
            # saved sampler epoch, consume the saved offset, then restore RNG.
            loader=data_loader._dataloader
            if loader.num_workers != 0: raise RuntimeError('Exact replay requires workers=0')
            completed=state['completed_updates']
            epoch=(completed-1)//len(loader) if completed else 0
            offset=(completed-1)%len(loader)+1 if completed else 0
            loader.sampler.set_epoch(epoch)
            data_loader._epoch=epoch
            data_loader.iter_loader=iter(loader)
            for _ in range(offset): next(data_loader.iter_loader)
            restore_rng(state['rng_by_rank'][dist.get_rank()])
            del self._ascend_resume
        return super().train(data_loader,**kwargs)


@RUNNERS.register_module()
class AscendEpochRunner(ResumeMixin, EpochBasedRunner):
    def __init__(self,resume_contract,**kwargs):
        self.resume_contract=json.loads(json.dumps(resume_contract))
        super().__init__(**kwargs)

    def train(self,data_loader,**kwargs):
        state=getattr(self,'_ascend_resume',None)
        if state is not None:
            restore_rng(state['rng_by_rank'][dist.get_rank()])
            del self._ascend_resume
        return super().train(data_loader,**kwargs)


@HOOKS.register_module()
class AscendRuntimeHook(Hook):
    def __init__(self,out_dir,profile,checkpoint_interval=100,stop_after=None,
                 stop_after_epochs=None,duration_seconds=None,check_interval=10,initialization_checkpoint=None):
        self.out_dir=Path(out_dir)
        self.profile=profile
        self.interval=checkpoint_interval
        self.stop_after=stop_after
        self.stop_epochs=stop_after_epochs
        self.duration=duration_seconds
        self.check_interval=check_interval
        self.initialization_checkpoint=initialization_checkpoint
        self.records=[]
        self.steps=0
        self.start=None

    def before_run(self,runner):
        self.initial_iter=runner.iter
        self.initial_epoch=runner.epoch
        self.rank=dist.get_rank()
        self.model=runner.model.module
        self.initialization_matches=[]
        if self.initialization_checkpoint:
            saved=torch.load(self.initialization_checkpoint,map_location='cpu',weights_only=False)
            initial=saved.get('state_dict',saved)
            state=self.model.state_dict()
            for name,value in initial.items():
                name=name.removeprefix('module.')
                if name in state and state[name].shape==value.shape:
                    if not torch.equal(value.cpu(),state[name].detach().cpu()):
                        raise RuntimeError('Initialization tensor mismatch: '+name)
                    self.initialization_matches.append(name)
            if len(self.initialization_matches)<100:raise RuntimeError('Insufficient official checkpoint matches')
            del saved,initial,state
        self.optimizer_initial_steps=sorted({float(v['step']) for v in runner.optimizer.state.values() if 'step' in v})
        if runner.iter and self.optimizer_initial_steps != [float(runner.iter)]:
            raise RuntimeError('Resumed optimizer steps differ from completed runner updates')
        self.watch=next((p for n,p in self.model.named_parameters()
                         if p.requires_grad and 'planning_head' in n and n.endswith('weight')))
        self.before=self.watch.detach().clone()
        head=self.model.planning_head
        original=head.loss
        self.raw_losses=[]
        def checked_loss(*args,**kwargs):
            result=original(*args,**kwargs)
            if not finite(result): raise RuntimeError('Nonfinite raw planning loss before nan_to_num')
            if (self.profile=='hydramdp_rl') != ('loss.RL.PG' in result):
                raise RuntimeError('Wrong policy-gradient participation')
            self.raw_losses.append({k:float(v.detach()) for k,v in result.items()})
            return result
        head.loss=checked_loss
        torch.npu.reset_peak_memory_stats()
        self.run_started=time.monotonic()

    def before_train_iter(self,runner):
        torch.npu.synchronize()
        self.iter_started=time.monotonic()
        if self.start is None:self.start=self.iter_started
        self.tokens=[]
        batch=getattr(runner,'data_batch',{})
        metas=batch.get('img_metas') if isinstance(batch,dict) else None
        if metas is not None:
            def collect(v):
                if hasattr(v,'data') and not torch.is_tensor(v):collect(v.data)
                elif isinstance(v,dict):
                    if 'sample_idx' in v:self.tokens.append(str(v['sample_idx']))
                    else:
                        for x in v.values():collect(x)
                elif isinstance(v,(list,tuple)):
                    for x in v:collect(x)
            collect(metas)
        self.lr=[g['lr'] for g in runner.optimizer.param_groups]

    def checkpoint(self,runner,completed,epoch,kind):
        torch.npu.synchronize()
        states=[None]*dist.get_world_size()
        dist.all_gather_object(states,rng_state())
        meta=dict(runner.meta or {})
        meta.update(iter=completed,epoch=epoch,ascend_resume=dict(
            version=1,contract=runner.resume_contract,completed_updates=completed,
            completed_epochs=epoch,rng_by_rank=states))
        path=self.out_dir/f'{kind}_{completed if kind=="iter" else epoch}.pth'
        if self.rank==0:
            temporary=path.with_suffix('.pth.tmp')
            save_checkpoint(runner.model,str(temporary),optimizer=runner.optimizer,meta=meta)
            temporary.replace(path)
            link=self.out_dir/'latest.pth'
            if link.is_symlink():link.unlink()
            if not link.exists():link.symlink_to(path.name)
        dist.barrier()
        # Collective object serialization can consume CPU RNG in some stacks.
        # Resume the exact state used in the checkpoint on the continuing run.
        restore_rng(states[self.rank])
        self.last_checkpoint=str(path)

    def after_train_iter(self,runner):
        torch.npu.synchronize()
        elapsed=time.monotonic()-self.iter_started
        self.steps+=1
        if not finite(runner.outputs['loss']):raise RuntimeError('Nonfinite total loss')
        if self.steps==1 or self.steps%self.check_interval==0:
            gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients or not all(finite(g) for g in gradients):
                raise RuntimeError('Missing or nonfinite gradients')
        rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        row=dict(iter=runner.iter+1,epoch=runner.epoch,seconds=elapsed,
                 elapsed_seconds=time.monotonic()-self.start,loss=float(runner.outputs['loss'].detach()),
                 lr=self.lr,tokens=self.tokens,samples=runner.outputs['num_samples'],rss_bytes=rss,
                 rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 npu_allocated=torch.npu.memory_allocated(),npu_reserved=torch.npu.memory_reserved())
        self.records.append(row)
        with (self.out_dir/f'runtime_rank{self.rank}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if isinstance(runner,IterBasedRunner):
            completed=runner.iter+1
            stop=bool(self.stop_after and completed>=self.stop_after)
            if self.duration:
                flag=torch.tensor(int(time.monotonic()-self.start>=self.duration),device='npu')
                dist.all_reduce(flag,op=dist.ReduceOp.MIN)
                stop=stop or bool(flag.item())
            if completed%self.interval==0 or completed==runner.max_iters or stop:
                self.checkpoint(runner,completed,0,'iter')
            if stop:runner._max_iters=completed

    def after_train_epoch(self,runner):
        if isinstance(runner,EpochBasedRunner):
            epoch=runner.epoch+1
            self.checkpoint(runner,runner.iter,epoch,'epoch')
            if self.stop_epochs and epoch>=self.stop_epochs:runner._max_epochs=epoch

    def after_run(self,runner):
        changed=bool((self.before!=self.watch.detach()).any())
        if not changed:raise RuntimeError('Watched trainable planning parameter did not update')
        if self.profile=='hydramdp_rl' and not any(abs(x['loss.RL.PG'])>0 for x in self.raw_losses):
            raise RuntimeError('No nonzero real policy-gradient loss')
        gathered=[torch.empty_like(self.watch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered,self.watch.detach().contiguous())
        if not all(torch.allclose(x,self.watch,atol=1e-6,rtol=1e-5) for x in gathered):
            raise RuntimeError('DDP watched parameter mismatch')
        report=dict(result='pass',rank=self.rank,world_size=dist.get_world_size(),profile=self.profile,
                    initial_iter=self.initial_iter,final_iter=runner.iter,updates_this_run=self.steps,
                    initial_epoch=self.initial_epoch,final_epoch=runner.epoch,
                    train_seconds=time.monotonic()-self.start,total_seconds=time.monotonic()-self.run_started,
                    raw_loss_records=self.raw_losses,checkpoint=getattr(self,'last_checkpoint',None),
                    parameter_updated=changed,watched_parameter_sync=True,
                    peak_allocated_bytes=torch.npu.max_memory_allocated(),peak_reserved_bytes=torch.npu.max_memory_reserved(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    resume_contract=runner.resume_contract)
        report.update(initialization_matches=self.initialization_matches,optimizer_initial_steps=self.optimizer_initial_steps,
                      restoration=getattr(runner,'_ascend_restoration',None),
                      model_parameters=sum(p.numel() for p in self.model.parameters()),
                      trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        (self.out_dir/f'runtime_rank{self.rank}.json').write_text(json.dumps(report,indent=2))
