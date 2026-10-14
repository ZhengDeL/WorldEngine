"""Bounded synchronous candidate for the two original AlgEngine base planners.

This stdlib-only adapter runs the unchanged ascend/run_container.py. Its only
Docker argv change is two initial environment options. The original launch.json
is preserved; effective_launch.json separately records the actual argv change.
Default mode performs the original read-only image/device/path checks and writes
fresh launch plans, but intercepts Docker run. Only --execute actually runs it.
Neither a generated plan nor a Docker exit code establishes model acceptance.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

SOURCE_SHA = 'cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88'
SOURCE_FILES = 243
LAUNCHER_SHA = '54b316fc70c3b7b4afc8ef2b300b35bc5b8e80fb23490d287afb4636a5576f3f'
IMAGE_ID = 'sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45'
CONFIGS = {
    'projects/AlgEngine/configs/navformer/e2e_vadv2.py': 'b1ed3b5d0ae465506442f74100803d111c793f31112705d74db68c46c5c30480',
    'projects/AlgEngine/configs/navformer/e2e_hydramdp.py': 'a9555793a3df5efa66b7764bf3a362cba69f27b956e6f10492c5e95bedc25d60',
}
INJECTED = ['-e', 'TASK_QUEUE_ENABLE=0', '-e', 'ASCEND_LAUNCH_BLOCKING=1']
ALGORITHM_ENV = {'TASK_QUEUE_ENABLE': '0', 'ASCEND_LAUNCH_BLOCKING': '1',
                 'HCCL_DETERMINISTIC': 'true', 'CLOSE_MATMUL_K_SHIFT': '1'}

def sha(path, canonical_lf=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if canonical_lf else raw).hexdigest()

def fingerprint(source):
    files = sorted([*(source/'projects/AlgEngine').rglob('*.py'), *(source/'ascend').glob('*.py')],
                   key=lambda p: p.relative_to(source).as_posix())
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(source).as_posix().encode())
        h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return {'sha256': h.hexdigest(), 'python_files': len(files)}

def one_option(arguments, name, required=True):
    positions = [i for i, x in enumerate(arguments) if x == name or x.startswith(name+'=')]
    if len(positions) != 1:
        if not positions and not required: return None
        raise ValueError('Exactly one '+name+' is required')
    i = positions[0]
    if arguments[i].startswith(name+'='): return arguments[i].split('=', 1)[1]
    if i+1 == len(arguments) or arguments[i+1].startswith('--'): raise ValueError('Missing value for '+name)
    return arguments[i+1]

def validate_arguments(source, arguments):
    # The original parser still validates all remaining options and input paths.
    tree=ast.parse((source/'ascend/run_container.py').read_text(encoding='utf8'))
    declared={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)}
    if any(x.startswith('--') and x.split('=',1)[0] not in declared for x in arguments):
        raise ValueError('Use complete original option names; abbreviations are not accepted by this adapter')
    if any(x.split('=',1)[0] in ['--detach','--inference-after','--validate','--debug-anomaly','--epochs','--duration-seconds','--stop-after-epochs'] for x in arguments):
        raise ValueError('This candidate covers foreground bounded training only')
    if one_option(arguments, '--task', False) not in [None, 'train'] or one_option(arguments, '--profile', False) not in [None, 'imitation']:
        raise ValueError('Only original imitation base-planner training is covered')
    for name, value in [('--devices', ','.join(map(str,range(16)))), ('--batch-size','1'), ('--workers','0'),
                        ('--steps','8'), ('--checkpoint-interval','4'), ('--precision','fp32'), ('--expected-image-id', IMAGE_ID)]:
        if one_option(arguments,name) != value: raise ValueError(name+' must explicitly be '+value)
    for flag in ['--runtime','--deterministic']:
        if arguments.count(flag) != 1: raise ValueError('Exactly one '+flag+' is required')
    if one_option(arguments,'--pids-limit',False) not in [None,'8192']:
        raise ValueError('Original sixteen-rank PID limit must remain8192')
    if one_option(arguments,'--stop-after',False) not in [None,'8']:
        raise ValueError('Bounded eight-update horizon must be preserved')
    image = one_option(arguments,'--image')
    config = Path(one_option(arguments,'--config')).resolve(strict=True)
    relative = config.relative_to(source).as_posix()
    if relative not in CONFIGS or sha(config,True) != CONFIGS[relative]:
        raise ValueError('Only the two unchanged original config hashes are covered')
    output = Path(one_option(arguments,'--output')).resolve()
    if output.exists(): raise ValueError('Output must be fresh; dry-run also creates its own plans')
    checkpoint = Path(one_option(arguments,'--checkpoint')).resolve(strict=True)
    vocab = Path(one_option(arguments,'--vocab')).resolve(strict=True)
    resume = one_option(arguments,'--resume',False)
    if resume and Path(resume).name != 'iter_4.pth': raise ValueError('Only own original iter_4.pth replay is covered')
    for key, expected in ALGORITHM_ENV.items():
        if os.environ.get(key) not in [None,expected]: raise ValueError('Conflicting host algorithm environment: '+key)
    return {'image':image,'config':relative,'config_sha256':CONFIGS[relative],'output':output,
            'checkpoint_sha256':sha(checkpoint),'vocab_sha256':sha(vocab),'resume':resume,
            'resume_state_compatibility_checked_by_original_trainer':bool(resume)}

def write_receipt(path, data, initial=False):
    text = json.dumps(data, indent=2, ensure_ascii=False)+'\n'
    if initial:
        with path.open('x',encoding='utf8',newline='\n') as f: f.write(text)
    else:
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('x',encoding='utf8',newline='\n') as f: f.write(text)
        temporary.replace(path)

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv: raise ValueError('Use -- before the unchanged original launcher arguments')
    split = argv.index('--'); wrapper, arguments = argv[:split],argv[split+1:]
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    opts=p.parse_args(wrapper)
    source=opts.source.resolve(strict=True)
    expected={'sha256':SOURCE_SHA,'python_files':SOURCE_FILES}
    if fingerprint(source)!=expected or sha(source/'ascend/run_container.py',True)!=LAUNCHER_SHA:
        raise ValueError('Fixed original runtime source/launcher fingerprint differs')
    validated=validate_arguments(source,arguments); output=validated['output']
    original_run=subprocess.run; original_argv=list(sys.argv); captured=[]
    def intercept(command,*args,**kwargs):
        if not isinstance(command,(list,tuple)) or list(command[:2])!=['docker','run']:
            return original_run(command,*args,**kwargs)
        command=list(command)
        if captured: raise RuntimeError('Exactly one original Docker run is allowed')
        if command[-4]!=validated['image'] or command[-3:-1]!=['bash','-lc'] or command[-1].count('torchrun --standalone --nproc_per_node=16 ascend/train.py ')!=1 or '/private/' in command[-1]:
            raise RuntimeError('Original unwrapped training tail differs')
        for key in ['TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING']:
            if any(x==key or x.startswith(key+'=') for x in command[:-4]) or any(x.startswith('--env='+key) or x.startswith('-e'+key) for x in command[:-4]):
                raise RuntimeError('Original Docker argv already declares '+key)
        # This image inspect is real. The original launcher's own ID inspect also
        # travels through the original subprocess path unchanged.
        detail=json.loads(subprocess.check_output(['docker','image','inspect',validated['image']],text=True))
        if len(detail)!=1 or detail[0]['Id']!=IMAGE_ID: raise RuntimeError('Actual fixed candidate image differs')
        image_env={x.split('=',1)[0]:x.split('=',1)[1] for x in detail[0].get('Config',{}).get('Env',[]) if '=' in x}
        if any(image_env.get(k) not in [None,v] for k,v in ALGORITHM_ENV.items()):
            raise RuntimeError('Conflicting algorithm environment inside fixed image')
        launch=output/'launch.json'; original_bytes=launch.read_bytes(); manifest=json.loads(original_bytes)
        if manifest.get('command')!=command or manifest.get('image_id')!=IMAGE_ID or manifest.get('profile')!='imitation' or manifest.get('host_devices')!=list(map(str,range(16))) or manifest.get('nproc')!=16 or manifest.get('pids_limit')!=8192 or manifest.get('precision')!='fp32' or manifest.get('deterministic') is not True:
            raise RuntimeError('Actual original launch manifest differs from bounded protocol')
        expected_mounts=[str(source)+':/workspace/WorldEngine:ro',str(output)+':/output:rw']
        if any(command.count(x)!=1 for x in expected_mounts) or command.count('--name')!=1 or command[command.index('--name')+1]!='worldengine-npu-'+output.name:
            raise RuntimeError('Exact original source/output/name differs')
        device_nodes=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--device']
        if device_nodes!=['/dev/davinci'+str(i) for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
            raise RuntimeError('Actual original sixteen-chip device mapping differs')
        effective=command[:2]+INJECTED+command[2:]
        receipt={'result':'dry_run_plan_only_not_training','created_at_utc':datetime.now(timezone.utc).isoformat(),
                 'source_fingerprint_before':expected,'image_id':IMAGE_ID,'config':validated['config'],
                 'config_sha256':validated['config_sha256'],'checkpoint_sha256':validated['checkpoint_sha256'],
                 'vocab_sha256':validated['vocab_sha256'],'resume':validated['resume'],
                 'original_launcher_arguments':arguments,'original_docker_argv':command,'effective_docker_argv':effective,
                 'only_added_Docker_options':INJECTED,'original_training_tail_unchanged':effective[6:]==command[2:],
                 'original_launch_sha256':hashlib.sha256(original_bytes).hexdigest(),
                 'adapter_sha256':sha(__file__),'actual_docker_run_executed':False,
                 'NPU_or_model_acceptance':False,'scope':'Bounded original2config16rank/B1/workers0/FP32/8update/checkpoint4 candidate only; no quality, convergence, GPU or universal determinism guarantee.'}
        path=output/'effective_launch.json';write_receipt(path,receipt,True);captured.append(receipt)
        if opts.execute:
            receipt['result']='executing_original_Docker_with_two_ENV_options';receipt['actual_docker_run_executed']=True;write_receipt(path,receipt)
            result=original_run(effective,*args,**kwargs)
            receipt.update(result='actual_Docker_exit_recorded_not_model_acceptance',actual_docker_exit=result.returncode)
        else:result=subprocess.CompletedProcess(effective,0)
        receipt['source_fingerprint_after']=fingerprint(source)
        receipt['original_launch_bytes_unchanged']=launch.read_bytes()==original_bytes
        write_receipt(path,receipt)
        if receipt['source_fingerprint_after']!=expected or not receipt['original_launch_bytes_unchanged']:
            raise RuntimeError('Original source or launch manifest changed')
        return result
    try:
        subprocess.run=intercept;sys.argv=[str(source/'ascend/run_container.py')]+arguments
        try:runpy.run_path(str(source/'ascend/run_container.py'),run_name='__main__')
        except SystemExit as error:
            if error.code not in [None,0]:raise
        if len(captured)!=1:raise RuntimeError('Original launcher did not reach the one reviewed Docker run')
    finally:
        subprocess.run=original_run;sys.argv=original_argv
    print(json.dumps({'result':captured[0]['result'],'receipt':str(output/'effective_launch.json'),'actual_docker_run_executed':opts.execute}))
    return 0

if __name__=='__main__':raise SystemExit(main())
