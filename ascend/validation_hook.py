"""Record actual AlgEngine losses, gradients, updates and per-rank NPU memory."""
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, OptimizerHook


def finite(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


@HOOKS.register_module()
class AscendValidationHook(OptimizerHook):
    def __init__(self, profile, expected_steps, expected_pg, dataset_scope, initialization_checkpoint, grad_clip=None):
        super().__init__(grad_clip=grad_clip)
        self.profile, self.expected_steps, self.expected_pg = profile, expected_steps, expected_pg
        self.dataset_scope, self.initialization_checkpoint = dataset_scope, initialization_checkpoint
        self.loss_records, self.gradient_counts, self.raw_loss_records = [], [], []
        self.input_records = []
        self.raw_outputs = 0

    def before_run(self, runner):
        model = runner.model.module
        head = model.planning_head
        initial = torch.load(self.initialization_checkpoint, map_location='cpu', weights_only=False)
        initial = initial.get('state_dict', initial)
        state = model.state_dict()
        self.initialization_matches = []
        for prefix in ('img_backbone.', 'pts_bbox_head.transformer.encoder.'):
            names = [n for n in initial if n.startswith(prefix) and n in state and initial[n].is_floating_point() and initial[n].numel()>100]
            if not names or not torch.equal(initial[names[0]], state[names[0]].detach().cpu()):
                raise RuntimeError('Checkpoint initialization mismatch: '+prefix)
            self.initialization_matches.append(names[0])
        self.parameter_count = sum(p.numel() for p in model.parameters())
        self.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.profile == 'imitation':
            self.watched_name = 'planning_head.heads.imi.4.weight'
            self.watched = head.heads['imi'][-1].weight
        else:
            name, self.watched = next((n,p) for n,p in head.named_parameters() if n.startswith('heads_lora.lora_noc') and '.B.weight' in n and p.requires_grad)
            self.watched_name = 'planning_head.'+name
        self.before = self.watched.detach().clone()
        original_forward, original_loss = head.forward, head.loss
        def checked_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if not finite(result):
                raise RuntimeError('Nonfinite raw head output')
            self.raw_outputs += 1
            return result
        def checked_loss(*args, **kwargs):
            gt = kwargs.get('gt_pdm_score')
            if gt is not None:
                row = {'score_shape':list(gt['score'].shape), 'score_nonzero':int((gt['score']!=0).sum()),
                       'score_min':float(gt['score'].min()), 'score_max':float(gt['score'].max())}
                if 'fail_mask' in gt:
                    row['fail_mask_before_loss'] = gt['fail_mask'].detach().cpu().tolist()
                self.input_records.append(row)
            result = original_loss(*args, **kwargs)
            if not finite(result):
                raise RuntimeError('Nonfinite raw loss before upstream nan_to_num')
            if self.expected_pg != ('loss.RL.PG' in result):
                raise RuntimeError('Unexpected PG loss participation')
            self.raw_loss_records.append({k:float(v.detach()) for k,v in result.items()})
            return result
        head.forward, head.loss = checked_forward, checked_loss
        torch.npu.reset_peak_memory_stats()

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        if not finite(loss):
            raise RuntimeError('Nonfinite total loss')
        runner.optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in runner.model.parameters() if p.grad is not None]
        if not gradients or not all(finite(g) for g in gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        if self.grad_clip:
            norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), foreach=False, **self.grad_clip)
            if not finite(norm):
                raise RuntimeError('Nonfinite gradient norm')
        runner.optimizer.step()
        self.loss_records.append(float(loss.detach()))
        self.gradient_counts.append(len(gradients))

    def after_run(self, runner):
        rank, world = dist.get_rank(), dist.get_world_size()
        if len(self.loss_records) != self.expected_steps or len(self.raw_loss_records)!=self.expected_steps:
            raise RuntimeError('Incomplete bounded training')
        changed = bool((self.before!=self.watched.detach()).any())
        if not changed:
            raise RuntimeError('Watched parameter did not update')
        if self.expected_pg and not any(abs(r['loss.RL.PG'])>0 for r in self.raw_loss_records):
            raise RuntimeError('PG was enabled but zero for all real samples')
        gathered = [torch.empty_like(self.watched) for _ in range(world)]
        dist.all_gather(gathered, self.watched.detach().contiguous())
        synchronized = all(torch.allclose(self.watched.detach(), x, atol=1e-6,rtol=1e-5) for x in gathered)
        if not synchronized:
            raise RuntimeError('Watched parameter synchronization mismatch')
        report = {'result':'pass_real_data_bounded_training','profile':self.profile,'rank':rank,'world_size':world,
                  'steps':self.expected_steps,'dataset_scope':self.dataset_scope,'raw_loss_records':self.raw_loss_records,
                  'raw_forward_checks':self.raw_outputs,'gradient_tensors':self.gradient_counts,'input_metric_checks':self.input_records,
                  'watched_parameter':self.watched_name,'parameter_updated':changed,'watched_parameter_sync':synchronized,
                  'model_parameters':self.parameter_count,'trainable_parameters':self.trainable_parameters,
                  'peak_allocated_bytes':torch.npu.max_memory_allocated(),'peak_reserved_bytes':torch.npu.max_memory_reserved(),
                  'initialization_matches':self.initialization_matches,'expected_pg':self.expected_pg}
        if rank==0:
            ckpt = Path(runner.work_dir)/f'iter_{self.expected_steps}.pth'
            saved = torch.load(ckpt,map_location='cpu',weights_only=False)
            if not finite(saved):
                raise RuntimeError('Nonfinite saved model/optimizer')
            steps = sorted({float(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s})
            if steps != [float(self.expected_steps)]:
                raise RuntimeError('Saved optimizer update count mismatch')
            h=hashlib.sha256()
            with ckpt.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            report.update(checkpoint_sha256=h.hexdigest(),checkpoint_bytes=ckpt.stat().st_size,
                          saved_model_tensors=len(saved['state_dict']),optimizer_steps=steps)
        (Path(runner.work_dir)/f'ascend_validation_rank{rank}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
