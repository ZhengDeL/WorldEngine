"""Launch portable, isolated AlgEngine training on selected host NPU devices."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'local/worldengine:26.1.0-a3-clean-20261013'
IMAGE_ID = 'sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',required=True,help='Comma-separated host logical NPU IDs; they are chips, not server counts')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['imitation','vadv2_ft','hydramdp_rl'],default='imitation')
    p.add_argument('--task',choices=['train','inference'],default='train')
    p.add_argument('--config',type=Path,help='Train: any published AlgEngine config; inference: the corresponding dumped config')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--mini-prepared',type=Path)
    p.add_argument('--data-root',type=Path)
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
    p.add_argument('--debug-anomaly',action='store_true')
    args=p.parse_args()
    devices=args.devices.split(',')
    if not devices or len(set(devices))!=len(devices) or not all(v.isdigit() for v in devices):
        p.error('Distinct numeric host device IDs required')
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
    cmd=['docker','run','--name',name,'--network','host','--memory',args.memory or '64g','--cpus',str(args.cpus),'--pids-limit','2048','--shm-size',args.shm_size,
         '-v',f'{ROOT}:/workspace/WorldEngine:ro','-v',f'{output}:/output:rw',
         '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
         '-v','/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro','-w','/workspace/WorldEngine']
    if args.detach:cmd+=['-d']
    cmd+=['-e','WORLDENGINE_IMAGE_ID='+image_id]
    cmd+=['-e','ASCEND_WORK_PATH=/output/ascend_work','-e','ASCEND_CACHE_PATH=/output/ascend_cache']
    if args.deterministic:
        cmd+=['-e','HCCL_DETERMINISTIC=true']
    if args.debug_anomaly:
        cmd+=['-e','WORLDENGINE_DEBUG_AUTOGRAD=1','-e','ASCEND_SLOG_PRINT_TO_STDOUT=1','-e','ASCEND_GLOBAL_LOG_LEVEL=3']
    for node in ['davinci'+d for d in devices]+['davinci_manager','devmm_svm','hisi_hdc']:
        if not (Path('/dev')/node).is_char_device():raise RuntimeError('Missing device node '+node)
        cmd+=['--device','/dev/'+node]
    for key,value in {'ASCEND_RT_VISIBLE_DEVICES':','.join(str(i) for i in range(len(devices))),
                      'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','HCCL_CONNECT_TIMEOUT':'120',
                      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():
        cmd+=['-e',key+'='+value]
    train=['ascend/train.py','--profile',args.profile,'--output','/output','--checkpoint','/weights/initial.pth',
           '--vocab','/weights/vocab.npy']
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
        train=['ascend/inference.py','--config','/config/inference.py','--checkpoint','/weights/initial.pth','--output','/output']
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
                   '--checkpoint',f'/output/iter_{args.steps}.pth','--output','/output/inference']
        inner+='if [ "$rc" -eq 0 ]; then\n  torchrun --standalone --nproc_per_node=1 '+shlex.join(inference)+' > /output/inference_console.log 2>&1\n  rc=$?\nfi\n'
    inner+='printf "%s\\n" "$rc" > /output/exit_code\nexit "$rc"'
    cmd += [args.image,'bash','-lc',inner]
    (output/'launch.json').write_text(json.dumps({'command':cmd,'image_id':image_id,'host_devices':devices,'nproc':len(devices),'profile':args.profile},indent=2))
    result=subprocess.run(cmd,check=False)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
