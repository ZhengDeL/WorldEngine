"""Launch portable, isolated AlgEngine training on selected host NPU devices."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMAGE = 'local/worldengine:26.1.0-a3-clean-20261013'
IMAGE_ID = 'sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f'


def _validate_task_options(args, parser):
    """Reject training settings that inference would otherwise discard."""
    if args.task == 'inference':
        training_only = (
            'profile', 'train_filter', 'rare_split', 'synthetic_root', 'validate',
            'steps', 'batch_size', 'workers', 'lr', 'inference_after', 'runtime',
            'resume', 'epochs', 'stop_after', 'stop_after_epochs',
            'duration_seconds', 'checkpoint_interval', 'check_interval',
            'upstream_schedule',
        )
        for name in training_only:
            value = getattr(args, name)
            if value is not None and value is not False:
                parser.error('--' + name.replace('_', '-') + ' requires --task train')
    if args.profile is None:
        args.profile = 'imitation'


def _hccl_connect_timeout(parser):
    value = os.environ.get('HCCL_CONNECT_TIMEOUT', '120').strip()
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        parser.error('HCCL_CONNECT_TIMEOUT must be a positive integer in seconds')
    return str(int(value))


def _validate_stop_options(args, parser):
    """Reject stop settings that the selected runner cannot honor."""
    import math
    if args.stop_after is not None:
        if args.steps is None:
            parser.error('--stop-after requires --steps')
        if args.stop_after < 1 or args.stop_after > args.steps:
            parser.error('--stop-after must be between 1 and --steps')
    if args.duration_seconds is not None:
        if args.steps is None:
            parser.error('--duration-seconds requires --steps')
        if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
            parser.error('--duration-seconds must be finite and positive')
    if args.stop_after_epochs is not None:
        if args.steps is not None:
            parser.error('--stop-after-epochs requires an epoch schedule without --steps')
        if args.stop_after_epochs < 1:
            parser.error('--stop-after-epochs must be positive')
        if args.epochs is not None and args.stop_after_epochs > args.epochs:
            parser.error('--stop-after-epochs must not exceed --epochs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',required=True,help='Comma-separated host logical NPU IDs; they are chips, not server counts')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['imitation','vadv2_ft','hydramdp_rl'],
                   help='Training profile; defaults to imitation for --task train')
    p.add_argument('--task',choices=['train','inference'],default='train')
    p.add_argument('--config',type=Path,help='Train: any published AlgEngine config; inference: the corresponding dumped config')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--mini-prepared',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--train-filter',type=Path,help='Explicit NAVTRAIN anchor filter for --data-root; preserved by content hash for resume')
    p.add_argument('--rare-split',type=Path)
    p.add_argument('--synthetic-root',type=Path)
    p.add_argument('--maps',type=Path)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--official-metric-cache',type=Path)
    p.add_argument('--steps',type=int,help='Omit for original epoch training; mini requires a bounded run')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers',type=int)
    p.add_argument('--lr',type=float)
    p.add_argument('--memory',help='CPU RAM limit; defaults to 64g for mini, required explicitly for full PKL assets')
    p.add_argument('--cpus',type=int,default=16)
    p.add_argument('--pids-limit',type=int,help='Container tasks, including host threads; default max(2048, 512 per NPU process)')
    p.add_argument('--shm-size',default='8g')
    p.add_argument('--image',default=IMAGE)
    p.add_argument('--expected-image-id',default=IMAGE_ID)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--inference-after',action='store_true',help='After bounded training, run the original test path on configured test anchors')
    p.add_argument('--runtime',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--stop-after-epochs',type=int)
    p.add_argument('--duration-seconds',type=float)
    p.add_argument('--checkpoint-interval',type=int)
    p.add_argument('--check-interval',type=int)
    p.add_argument('--upstream-schedule',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--precision',choices=('fp32','fp32-hf32-conv','bf16'),default='fp32',
                   help='fp32 disables Matmul/Conv HF32; fp32-hf32-conv reproduces legacy Conv HF32; bf16 uses native NPU autocast with FP32 MSDA and FP32 master training state, without a scaler')
    p.add_argument('--debug-anomaly',action='store_true')
    p.add_argument('--msda-backend',choices=('reference','cuda-compatible'),default='reference',
                   help='FP32 MSDA implementation; preserve the selection for training, resume and inference')
    from ascend.inference_optimization import MODES, CHUNKS, validate_options
    p.add_argument('--inference-optimization',choices=MODES,default='none')
    p.add_argument('--msda-query-chunk',type=int,choices=CHUNKS,default=2048)
    p.add_argument('--msda-value-grad-layout',choices=('none','head-major','point-parallel-head-major'),default='none')
    args=p.parse_args()
    from ascend.msda_training_point_parallel import validate_training_option
    try:
        validate_training_option(args.msda_value_grad_layout, precision=args.precision,
            backend=args.msda_backend, task=args.task, runtime=bool(args.runtime or args.resume or args.stop_after
                or args.duration_seconds or args.epochs or args.stop_after_epochs))
    except ValueError as exc:
        p.error(str(exc))
    _validate_task_options(args, p)
    if args.task!='inference' and (args.inference_optimization!='none' or args.msda_query_chunk!=2048):
        p.error('Inference optimization options require --task inference')
    try:
        validate_options(args.inference_optimization,args.msda_query_chunk,args.precision,args.msda_backend)
    except ValueError as exc:
        p.error(str(exc))
    _validate_stop_options(args, p)
    hccl_timeout = _hccl_connect_timeout(p)
    if args.inference_after and args.task != 'train':
        p.error('--inference-after requires --task train')
    if args.train_filter and (args.task!='train' or not args.data_root or args.mini_prepared):
        p.error('--train-filter requires train with --data-root and without --mini-prepared')
    devices=args.devices.split(',')
    if not devices or len(set(devices))!=len(devices) or not all(v.isdigit() for v in devices):
        p.error('Distinct numeric host device IDs required')
    if args.pids_limit is not None and args.pids_limit<1:
        p.error('--pids-limit must be positive')
    pids_limit=args.pids_limit if args.pids_limit is not None else max(2048,512*len(devices))
    if bool(args.mini_prepared)==bool(args.data_root):
        p.error('Select exactly one of --mini-prepared and --data-root')
    if args.task=='train' and args.mini_prepared and not (args.steps or args.epochs):
        p.error('Mini data is functional validation only; provide --steps')
    if args.task=='inference' and not args.config:
        p.error('--task inference requires --config')
    if args.data_root and not args.memory:
        p.error('Full PKL caches load per rank: size CPU RAM and specify --memory explicitly')
    if args.inference_after and not args.steps:
        p.error('--inference-after needs a bounded --steps checkpoint')
    image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image],text=True).strip()
    if image_id!=args.expected_image_id:
        raise RuntimeError('Image ID differs from the selected lock')
    output=args.output.resolve()
    if output.exists():
        raise RuntimeError('Output already exists; select a new directory')
    output.mkdir(parents=True)
    name='worldengine-npu-'+output.name
    cmd=['docker','run','--name',name,'--network','host','--memory',args.memory or '64g','--cpus',str(args.cpus),'--pids-limit',str(pids_limit),'--shm-size',args.shm_size,
         '-v',f'{ROOT}:/workspace/WorldEngine:ro','-v',f'{output}:/output:rw',
         '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
         '-v','/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro','-w','/workspace/WorldEngine']
    if args.detach:cmd+=['-d']
    cmd+=['-e','WORLDENGINE_IMAGE_ID='+image_id,'-e','WORLDENGINE_HOST_DEVICE_IDS='+','.join(devices)]
    cmd+=['-e','ASCEND_WORK_PATH=/output/ascend_work','-e','ASCEND_CACHE_PATH=/output/ascend_cache']
    if args.deterministic:
        cmd+=['-e','HCCL_DETERMINISTIC=true','-e','CLOSE_MATMUL_K_SHIFT=1']
    if args.debug_anomaly:
        cmd+=['-e','WORLDENGINE_DEBUG_AUTOGRAD=1','-e','ASCEND_SLOG_PRINT_TO_STDOUT=1','-e','ASCEND_GLOBAL_LOG_LEVEL=3']
    for node in ['davinci'+d for d in devices]+['davinci_manager','devmm_svm','hisi_hdc']:
        if not (Path('/dev')/node).is_char_device():raise RuntimeError('Missing device node '+node)
        cmd+=['--device','/dev/'+node]
    for key,value in {'ASCEND_RT_VISIBLE_DEVICES':','.join(str(i) for i in range(len(devices))),
                      'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','HCCL_CONNECT_TIMEOUT':hccl_timeout,
                      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():
        cmd+=['-e',key+'='+value]
    train=['ascend/train.py','--profile',args.profile,'--output','/output','--checkpoint','/weights/initial.pth',
           '--vocab','/weights/vocab.npy','--precision',args.precision,'--msda-backend',args.msda_backend,
           '--msda-value-grad-layout',args.msda_value_grad_layout]
    for key in ('batch_size','workers','lr','epochs','stop_after','stop_after_epochs','duration_seconds','checkpoint_interval','check_interval'):
        value=getattr(args,key)
        if value is not None:train+=['--'+key.replace('_','-'),str(value)]
    if args.runtime:train+=['--runtime']
    if args.upstream_schedule:train+=['--upstream-schedule']
    if args.deterministic:train+=['--deterministic']
    if args.debug_anomaly:train+=['--debug-anomaly']
    if args.validate:train+=['--validate']
    if args.config and args.task=='train':
        config=args.config.resolve(strict=True)
        cmd+=['-v',f'{config}:/config/train.py:ro']
        # Preserve relative _base_ references by passing the original file in the source mount.
        try:relative=config.relative_to(ROOT)
        except ValueError:raise RuntimeError('Training config must be inside the selected source checkout')
        train+=['--config','/workspace/WorldEngine/'+relative.as_posix()]
    if args.resume:
        path=args.resume.resolve(strict=True)
        cmd+=['-v',f'{path}:/weights/resume.pth:ro']
        train+=['--resume','/weights/resume.pth']
    for path,target in [(args.checkpoint,'/weights/initial.pth'),(args.vocab,'/weights/vocab.npy')]:
        path=path.resolve(strict=True)
        cmd+=['-v',f'{path}:{target}:ro']
    if args.mini_prepared:
        mini=args.mini_prepared.resolve(strict=True)
        # Prepared camera symlinks may reference the original asset parent.
        cmd+=['-v',f'{mini}:/data/mini:ro','-v',f'{mini.parent}:{mini.parent}:ro']
        camera=mini/'cameras'
        if camera.is_symlink():
            target=camera.resolve(strict=True)
            cmd+=['-v',f'{target}:{target}:ro']
        train+=['--mini-prepared','/data/mini']
    else:
        data=args.data_root.resolve(strict=True)
        cmd+=['-v',f'{data}:/data/full:ro']
        train+=['--data-root','/data/full']
    if args.rare_split:
        path=args.rare_split.resolve(strict=True)
        cmd+=['-v',f'{path}:/data/rare.yaml:ro']
        train+=['--rare-split','/data/rare.yaml']
    if args.train_filter:
        path=args.train_filter.resolve(strict=True)
        if not path.is_file():raise RuntimeError('--train-filter must be a file')
        # The original dataset selects the NAVTRAIN PDM cache by basename.
        cmd+=['-v',f'{path}:/data/navtrain_selected_filter.yaml:ro']
        train+=['--train-filter','/data/navtrain_selected_filter.yaml']
    if args.synthetic_root:
        root=args.synthetic_root.resolve(strict=True)
        cmd+=['-v',f'{root}:/data/synthetic:ro']
        train+=['--synthetic-root','/data/synthetic']
    if args.maps:
        root=args.maps.resolve(strict=True)
        cmd+=['-v',f'{root}:/data/maps:ro','-e','NUPLAN_MAPS_ROOT=/data/maps']
    if args.official_metric_cache:
        root=args.official_metric_cache.resolve(strict=True)
        cmd+=['-v',f'{root}:/data/metric_cache:ro']
        train+=['--official-metric-cache','/data/metric_cache']
    if args.steps:train+=['--steps',str(args.steps)]
    if args.task=='inference':
        config=args.config.resolve(strict=True)
        cmd+=['-v',f'{config}:/config/inference.py:ro']
        train=['ascend/inference.py','--config','/config/inference.py','--checkpoint','/weights/initial.pth','--output','/output','--precision',args.precision,'--msda-backend',args.msda_backend]
        if args.inference_optimization!='none':
            train+=['--inference-optimization',args.inference_optimization,'--msda-query-chunk',str(args.msda_query_chunk)]
        if args.deterministic:train+=['--deterministic']
    import shlex
    inner='set +u\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\nexport PATH=/opt/conda/envs/bevformer/bin:/opt/conda/bin:$PATH\n'
    inner+='mkdir -p /output/ascend_work /output/ascend_cache\n'
    inner+='export NAVSIM_DEVKIT_ROOT=/workspace/WorldEngine/ascend/third_party/navsim\n'
    inner+='export PYTHONPATH=/workspace/WorldEngine:/workspace/WorldEngine/projects/AlgEngine:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH\n'
    inner+='torchrun --standalone --nproc_per_node='+str(len(devices))+' '+shlex.join(train)+' > /output/console.log 2>&1\n'
    inner+='rc=$?\nprintf "%s\\n" "$rc" > /output/train_exit_code\n'
    if args.inference_after:
        profile=args.config.stem if args.config else args.profile
        inference=['ascend/inference.py','--config',f'/output/{profile}_rank0.py',
                   '--checkpoint','/output/latest.pth','--output','/output/inference','--precision',args.precision,'--msda-backend',args.msda_backend]
        if args.deterministic:inference+=['--deterministic']
        inner+='if [ "$rc" -eq 0 ]; then\n  torchrun --standalone --nproc_per_node=1 '+shlex.join(inference)+' > /output/inference_console.log 2>&1\n  rc=$?\nfi\n'
    inner+='printf "%s\\n" "$rc" > /output/exit_code\nexit "$rc"'
    cmd += [args.image,'bash','-lc',inner]
    (output/'launch.json').write_text(json.dumps({'command':cmd,'image_id':image_id,'host_devices':devices,'nproc':len(devices),'profile':args.profile,'pids_limit':pids_limit,'precision':args.precision,'deterministic':args.deterministic,'msda_backend':args.msda_backend,'msda_value_grad_layout':args.msda_value_grad_layout,'hccl_connect_timeout':hccl_timeout,'inference_optimization':args.inference_optimization,'msda_query_chunk':args.msda_query_chunk},indent=2))
    result=subprocess.run(cmd,check=False)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
