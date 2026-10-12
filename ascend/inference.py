"""Run the original planning test path and validate actual exported trajectories."""
import argparse
import json
import os
import pickle
import runpy
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALG = ROOT/'projects/AlgEngine'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path[:0]=[str(ALG),str(ROOT)]
    sys.path.insert(0,str(ROOT/'ascend/third_party/navsim'))
    os.environ['WORLDENGINE_ROOT']=str(ROOT)
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK','0')))
    cfg=Config.fromfile(str(args.config),import_custom_modules=False)
    cfg.pop('fp16',None)
    cfg.dist_params.backend='hccl'
    cfg.data.workers_per_gpu=0
    cfg.evaluation=dict(interval=1)
    cfg.custom_imports=dict(imports=['mmdet3d_plugin'],allow_failed_imports=False)
    args.output.mkdir(parents=True,exist_ok=True)
    # Check the official CPU export dependency before spending time on NPU inference.
    from navsim.common.dataclasses import Trajectory
    config=args.output/f'inference_rank{os.environ.get("LOCAL_RANK","0")}.py'
    cfg.dump(str(config))
    torch.npu.reset_peak_memory_stats()
    started=time.monotonic()
    sys.argv=['scripts/test.py',str(config),str(args.checkpoint),'--launcher','pytorch',
              '--out',str(args.output/'predictions.pkl'),'--tmpdir',str(args.output/'collect'), '--eval','bbox',
              '--eval-options','jsonfile_prefix='+str(args.output/'metrics')]
    os.chdir(ALG)
    runpy.run_path(str(ALG/'scripts/test.py'),run_name='__main__')
    torch.npu.synchronize()
    if torch.distributed.get_rank()==0:
        with (args.output/'predictions.pkl').open('rb') as stream:
            results=pickle.load(stream)
        with open(cfg.data.test.nav_filter_path) as stream:
            expected=set(yaml.safe_load(stream)['tokens'])
        tokens=[r['token'] for r in results]
        if len(tokens)!=len(expected) or set(tokens)!=expected:
            raise RuntimeError('Inference did not cover every configured anchor exactly once')
        trajectories=[np.asarray(r['trajectory']) for r in results]
        if any(t.shape!=(40,3) or not np.isfinite(t).all() for t in trajectories):
            raise RuntimeError('Invalid actual planning trajectory')
        vocabulary=np.load(cfg.model.planning_head.vocab_path)
        indices=[int(r['chosen_ind']) for r in results]
        if vocabulary.shape!=(8192,40,3) or any(i<0 or i>=8192 for i in indices):
            raise RuntimeError('Invalid trajectory vocabulary or selected index')
        selected_matches=all(np.allclose(t,vocabulary[i],atol=1e-6,rtol=1e-6) for t,i in zip(trajectories,indices))
        if not selected_matches:
            raise RuntimeError('Exported trajectory differs from the selected vocabulary entry')
        keys=['ade_4s','fde_4s','no_at_fault_collisions','drivable_area_compliance',
              'ego_progress','time_to_collision_within_bound','comfort','score']
        metrics={k:float(np.mean([r[k] for r in results])) for k in keys}
        if not all(np.isfinite(list(metrics.values()))):
            raise RuntimeError('Nonfinite planning metric')
        report=dict(result='pass_real_planning_inference',anchors=len(tokens),unique_tokens=len(set(tokens)),
                    trajectory_shape=[40,3],all_trajectories_finite=True,metrics=metrics,
                    selection_indices_in_range=True,selected_trajectories_match_vocabulary=selected_matches,
                    elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
                    peak_reserved_bytes=torch.npu.max_memory_reserved(),
                    acceptance='Functional inference and embedded/cached PDM only. Reused mini anchors are not held-out accuracy.')
        (args.output/'inference_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
