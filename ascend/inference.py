"""Run the original planning test path and validate actual exported trajectories."""
import argparse
import hashlib
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
from ascend.compat import enable_training_compat
from ascend.precision import precision_state, device_provenance

ROOT = Path(__file__).resolve().parents[1]
ALG = ROOT/'projects/AlgEngine'




def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def source_fingerprint():
    value=hashlib.sha256()
    for file in sorted(list(ALG.rglob('*.py'))+list((ROOT/'ascend').glob('*.py'))):
        value.update(str(file.relative_to(ROOT)).encode())
        value.update(file.read_bytes().replace(b'\r\n',b'\n'))
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=('fp32','fp32-hf32-conv'),default='fp32',
                   help='FP32 inference with Matmul/Conv HF32 disabled; explicit legacy Conv HF32 mode is fp32-hf32-conv')
    p.add_argument('--deterministic',action='store_true',help='Request original Torch/CANN deterministic algorithms with required initial launcher environment')
    args=p.parse_args()
    # Keep actual loaded files and exported artifacts stable across os.chdir(ALG).
    args.config=args.config.resolve(strict=True)
    args.checkpoint=args.checkpoint.resolve(strict=True)
    args.output=args.output.resolve()
    sys.path[:0]=[str(ALG),str(ROOT)]
    sys.path.insert(0,str(ROOT/'ascend/third_party/navsim'))
    if args.deterministic and (os.environ.get('CLOSE_MATMUL_K_SHIFT')!='1' or
                               os.environ.get('HCCL_DETERMINISTIC')!='true'):
        p.error('--deterministic requires CLOSE_MATMUL_K_SHIFT=1 and HCCL_DETERMINISTIC=true before the Torch process starts')
    os.environ['WORLDENGINE_ROOT']=str(ROOT)
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.npu.matmul.allow_hf32=False
    torch.npu.conv.allow_hf32=args.precision=='fp32-hf32-conv'
    enable_training_compat()
    if args.deterministic:torch.use_deterministic_algorithms(True)
    actual_precision=precision_state(args.precision)
    if actual_precision['public']!=dict(matmul_allow_hf32=False,conv_allow_hf32=args.precision=='fp32-hf32-conv'):
        raise RuntimeError('Public NPU inference precision settings differ from --precision')
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
    final_precision=precision_state(args.precision,stage='after_original_inference')
    if final_precision['public']!=actual_precision['public']:
        raise RuntimeError('Public NPU precision changed during original inference')
    if torch.distributed.get_rank()==0:
        identity=dict(image_id=os.environ.get('WORLDENGINE_IMAGE_ID'),source_sha256=source_fingerprint(),
                      source_config_sha256=digest(args.config),checkpoint_sha256=digest(args.checkpoint),
                      torch_version=str(torch.__version__),torch_npu_version=torch_npu.__version__,
                      torch_npu_git_version=getattr(getattr(torch_npu,'version',None),'git_version',None),
                      precision=dict(before_model=actual_precision,after_original_inference=final_precision),
                      precision_observation_rank=0,world_size=torch.distributed.get_world_size(),
                      device_provenance=device_provenance())
        with (args.output/'predictions.pkl').open('rb') as stream:
            results=pickle.load(stream)
        with open(cfg.data.test.nav_filter_path) as stream:
            expected=set(yaml.safe_load(stream)['tokens'])
        perception=cfg.model.get('planning_head') is None
        if perception:
            annotations=pickle.load(open(cfg.data.test.ann_file,'rb'))
            infos=annotations['infos'] if isinstance(annotations,dict) else annotations
            indices=set()
            history=cfg.data.test.get('history_frame_num',3)
            for i,info in enumerate(infos):
                if info['token'] in expected:indices.update(j for j in range(i-history,i+1) if j>=0)
            expected={infos[i]['token'] for i in indices}
        tokens=[r['token'] for r in results]
        if len(tokens)!=len(expected) or set(tokens)!=expected:
            raise RuntimeError('Inference did not cover every configured anchor exactly once')
        if perception:
            from ascend.perception import validate_perception_outputs
            checked=validate_perception_outputs(results)
            report=dict(result='pass_real_perception_inference',frames=len(tokens),unique_tokens=len(set(tokens)),
                        **checked,original_detection_mapping_evaluator_executed=True,
                        elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
                        peak_reserved_bytes=torch.npu.max_memory_reserved(),
                        acceptance='Functional original detection/mapping inference on mini frames. Full detection/mapping quality and GPU comparison are separate acceptance items.')
            report.update(**identity)
            (args.output/'inference_validation.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2),flush=True)
            return
        trajectories=[np.asarray(r['trajectory']) for r in results]
        if any(t.shape!=(40,3) or not np.isfinite(t).all() for t in trajectories):
            raise RuntimeError('Invalid actual planning trajectory')
        generated=cfg.model.planning_head.type=='DiffusionPlanningHead'
        selected_matches=None
        if not generated:
            vocabulary=np.load(cfg.model.planning_head.vocab_path)
            indices=[int(r['chosen_ind']) for r in results]
            if vocabulary.shape!=(8192,40,3) or any(i<0 or i>=8192 for i in indices):
                raise RuntimeError('Invalid trajectory vocabulary or selected index')
            selected_matches=all(np.allclose(t,vocabulary[i],atol=1e-6,rtol=1e-6) for t,i in zip(trajectories,indices))
            if not selected_matches:raise RuntimeError('Exported trajectory differs from the selected vocabulary entry')
        keys=['ade_4s','fde_4s','no_at_fault_collisions','drivable_area_compliance',
              'ego_progress','time_to_collision_within_bound','comfort','score']
        metric_keys=keys[:2] if generated else keys
        metrics={k:float(np.mean([r[k] for r in results])) for k in metric_keys}
        if not all(np.isfinite(list(metrics.values()))):
            raise RuntimeError('Nonfinite planning metric')
        report=dict(result='pass_real_planning_inference',anchors=len(tokens),unique_tokens=len(set(tokens)),
                    trajectory_shape=[40,3],all_trajectories_finite=True,metrics=metrics,
                    selection_indices_in_range=None if generated else True,selected_trajectories_match_vocabulary=selected_matches,
                    generated_trajectories=generated,requires_official_rescoring=generated,
                    elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
                    peak_reserved_bytes=torch.npu.max_memory_reserved(),
                    acceptance='Functional inference and embedded/cached PDM only. Reused mini anchors are not held-out accuracy.')
        manifest_path=Path(cfg.data.test.nav_filter_path).parent/'manifest.json'
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text())
            independent=bool(manifest.get('heldout_samples')==len(tokens) and manifest.get('train_eval_token_overlap')==0)
            report.update(independent_log_split=independent,dataset_scope=manifest.get('dataset_scope'))
            if independent:
                report['acceptance']='Independent small-data inference; exported cached metrics still require separate official NAVSIM rescoring. No GPU comparison or full-benchmark claim.'
        report.update(**identity)
        (args.output/'inference_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
