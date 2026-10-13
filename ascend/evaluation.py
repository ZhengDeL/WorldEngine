"""Per-epoch native evaluation with preserved predictions and strict CPU scoring."""
import datetime,json,pickle,time
from pathlib import Path
import numpy as np
import yaml
import torch.distributed as dist
from mmdet.core import DistEvalHook
from ascend.scoring import score_predictions
from ascend.perception import validate_perception_outputs

class AscendDistEvalHook(DistEvalHook):
    def __init__(self,*args,official_metric_cache=None,**kwargs):
        self.official_metric_cache=official_metric_cache
        # Complete runtime epoch checkpoints are selected separately. The
        # standard hook's model-only best checkpoint cannot resume all RNGs.
        kwargs.pop('save_best',None)
        super().__init__(*args,**kwargs)

    def before_run(self,runner):
        super().before_run(runner)
        # Rank 0 CPU scoring may take longer than an NPU collective timeout.
        # Keep every rank waiting on a bounded independent CPU process group.
        self.cpu_group=dist.new_group(backend='gloo',timeout=datetime.timedelta(hours=4))
        self.best=(runner.meta or {}).get('ascend_eval_best')

    def _do_evaluate(self,runner):
        if not self._should_evaluate(runner):return
        self.evaluation_started=time.monotonic()
        status=[None]
        try:
            super()._do_evaluate(runner)
            if runner.rank==0:status[0]=dict(result='pass')
        except Exception as error:
            if runner.rank!=0:raise
            status[0]=dict(result='fail',error=repr(error))
        dist.broadcast_object_list(status,src=0,group=self.cpu_group)
        if status[0]['result']!='pass':raise RuntimeError('Epoch quality evaluation failed: '+status[0]['error'])

    def evaluate(self,runner,results):
        started=time.monotonic();dataset=self.dataloader.dataset
        expected={dataset.data_infos[i]['token'] for i in dataset.index_map}
        tokens=[row['token'] for row in results]
        if len(tokens)!=len(expected) or set(tokens)!=expected:raise RuntimeError('Epoch evaluation did not cover configured frames exactly once')
        output=Path(runner.work_dir)/'validation'/f'epoch_{runner.epoch+1}'
        output.mkdir(parents=True,exist_ok=True)
        pickle.dump(results,(output/'predictions.pkl').open('wb'))
        kwargs=dict(self.eval_kwargs);kwargs['jsonfile_prefix']=str(output/'metrics')
        metrics={k:float(v) for k,v in dataset.evaluate(results,logger=runner.logger,**kwargs).items()}
        planning=bool(results and 'trajectory' in results[0])
        perception_outputs=validate_perception_outputs(results) if not planning else None
        official=None
        if planning and self.official_metric_cache:
            official=score_predictions(results,self.official_metric_cache,output/'official')
            metrics.update({k:v for k,v in official.items() if k in metrics})
            metrics['pdm_score']=official['score']
        undefined={key for key,value in metrics.items() if np.isnan(value)}
        # The unchanged NuPlan protocol deliberately marks these class-specific
        # TP errors as inapplicable. Preserve them as null in our JSON report;
        # unexpected NaN/Inf and missing aggregate metrics still fail the gate.
        allowed={f'NuPlan/{name}_{error}' for name,errors in {
            'traffic_cone': ('attr_err','vel_err','orient_err'),
            'barrier': ('attr_err','vel_err'), 'czone_sign': ('attr_err','vel_err'),
        }.items() for error in errors} if not planning else set()
        if perception_outputs:
            allowed.update(name+'_iou' for name in perception_outputs['empty_union_categories'])
        if undefined-allowed or any(np.isinf(value) for value in metrics.values()):
            raise RuntimeError('Unexpected nonfinite epoch metric; generated trajectories require official metric cache')
        logged_metrics={key:value for key,value in metrics.items() if key not in undefined}
        key='pdm_score' if planning else 'pts_bbox_NuScenes/NDS'
        if key not in metrics:
            key=next((k for k in metrics if k.endswith('/NDS')),None)
        selected=bool(key and (self.best is None or metrics[key]>self.best['value']))
        if selected:self.best=dict(metric=key,value=metrics[key],epoch=runner.epoch+1,checkpoint=f'epoch_{runner.epoch+1}.pth')
        runner.meta=runner.meta or {};runner.meta['ascend_eval_best']=self.best
        rare_metrics=None;rare_tokens=0
        if planning:
            root=Path(__file__).resolve().parents[1]
            rare=set(yaml.safe_load((root/'projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml').read_text())['tokens'])
            selected_rows=[row for row in results if row['token'] in rare];rare_tokens=len(selected_rows)
            if rare_tokens:
                rare_metrics={key:float(np.mean([row[key] for row in selected_rows])) for key in ('ade_4s','fde_4s')}
                if official is not None:
                    import pandas as pd
                    table=pd.read_csv(output/'official/official_scores.csv')
                    subset=table[table['token'].isin(rare)]
                    if len(subset)!=rare_tokens:raise RuntimeError('Incomplete official rare-case scores')
                    rare_metrics.update(subset.drop(columns=['token','valid']).mean().to_dict())
        report=dict(result='pass_epoch_evaluation',epoch=runner.epoch+1,unique_tokens=len(expected),
                    metrics={key:None if key in undefined else value for key,value in metrics.items()},
                    inapplicable_protocol_metrics=sorted(undefined),
                    perception_outputs=perception_outputs,
                    official_scoring=official is not None,official_metrics=official,selected_best=selected,best=self.best,
                    rare_tokens=rare_tokens,rare_metrics=rare_metrics,
                    scoring_seconds=time.monotonic()-started,total_evaluation_seconds=time.monotonic()-self.evaluation_started,
                    acceptance='Epoch evaluation only. Full convergence requires original data/schedule and independently defined final quality criteria.')
        runner._ascend_epoch_eval=report
        (output/'evaluation.json').write_text(json.dumps(report,indent=2))
        with (Path(runner.work_dir)/'epoch_metrics.jsonl').open('a') as file:file.write(json.dumps(report)+'\n')
        runner.log_buffer.output.update(logged_metrics);runner.log_buffer.ready=True
        return None
