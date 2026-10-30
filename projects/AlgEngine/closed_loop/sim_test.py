"""Original closed-loop planner with explicit CUDA/NPU runtime selection."""
import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import time

SIM_ROOT=Path(__file__).resolve().parents[3]
ALG=SIM_ROOT/'projects/AlgEngine'
sys.path[:0]=[str(SIM_ROOT),str(ALG)]
from closed_loop.monitor import AsyncFileMonitor
from simengine_ascend.file_protocol import atomic_pickle,atomic_numpy,atomic_write,check_stop,validate_trajectory
from simengine_ascend.closed_loop_runtime import ClosedLoopRuntime,file_sha


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('checkpoint')
    parser.add_argument('--log-dir',default='');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--deterministic',action='store_true');parser.add_argument('--fuse-conv-bn',action='store_true')
    parser.add_argument('--device',default='cuda:0',help='Explicit visible device, e.g. cuda:0 or npu:0')
    parser.add_argument('--precision',choices=('fp32','bf16'),default=None,
                        help='Required for NPU; omitted CUDA preserves the original config fp16 setting')
    parser.add_argument('--algengine-root',type=Path)
    parser.add_argument('--expected-algengine-source-sha256')
    parser.add_argument('--algengine-commit');parser.add_argument('--expected-image-id')
    parser.add_argument('--msda-backend',choices=('reference','cuda-compatible'),default='reference')
    parser.add_argument('--inference-optimization',choices=('none','point-parallel','point-parallel-last-frame'),default='none')
    parser.add_argument('--msda-query-chunk',type=int,default=2048)
    parser.add_argument('--wait-timeout',type=float,default=120)
    parser.add_argument('--deadline-utc',default=os.environ.get('WORLDENGINE_CLOSED_LOOP_DEADLINE_UTC'))
    parser.add_argument('--stop-file',default=os.environ.get('WORLDENGINE_CLOSED_LOOP_STOP_FILE'))
    parser.add_argument('--max-predictions',type=int)
    parser.add_argument('--cfg-options',nargs='+',default=[])
    args=parser.parse_args()
    if not math.isfinite(args.wait_timeout) or args.wait_timeout<=0:parser.error('--wait-timeout must be finite and positive')
    if args.max_predictions is not None and args.max_predictions<1:parser.error('--max-predictions must be positive')
    if args.device.startswith('npu:') and not args.deadline_utc:parser.error('NPU execution requires --deadline-utc')
    return args


def merge_ann_files(cfg,ann_files_list,merged_index):
    merged={'infos':[]}
    for name in ann_files_list:
        with open(name,'rb') as stream:record=pickle.load(stream)
        if not isinstance(record,dict):raise ValueError('Published frame is not a metadata dictionary: '+str(name))
        merged['infos'].append(record)
    if len(merged['infos'])!=cfg.queue_length:raise ValueError('Incomplete original temporal frame queue')
    path=Path(cfg.sim.merged_ann_save_dir)/('merged_%s.pkl'%merged_index)
    atomic_pickle(path,merged)
    return str(path)


def clean_related_files(cfg,logger,owned_files):
    if cfg.sim.get('clean_record_data',False):
        raise ValueError('Planner does not own renderer records; recursive record deletion is unsupported')
    if cfg.sim.get('clean_temp_files',False):
        for path in owned_files:
            if path.is_symlink():raise ValueError('Refusing cleanup of a replaced protocol symlink')
            try:path.unlink()
            except FileNotFoundError:pass
        logger.info('Removed %d files created by this planner only',len(owned_files))


def host_array(value,torch):
    if torch.is_tensor(value):return value.detach().float().cpu().numpy()
    import numpy as np
    return np.asarray(value)


def model_inference(model,data_loader,torch):
    for data in data_loader:
        with torch.no_grad():result=model(return_loss=False,rescale=True,**data)
        if not isinstance(result,list) or len(result)!=1:raise ValueError('Expected one closed-loop planning result')
        return result[0]
    raise ValueError('Closed-loop dataset produced no inference sample')


def planner_window(history,frames,producer_report,required=False):
    """Bind the control horizon to this run's renderer window, not source-log length."""
    if type(history) is not int or type(frames) is not int or not 0<history<frames:
        raise ValueError('Explicit positive history and rendered-frame counts required')
    expected=dict(num_history=history,num_future=frames-history,frames=frames,
                  final_frame=frames-1,consumed_control_indices=list(range(history,frames)))
    value=dict(expected,source='planner_configuration_without_producer_report')
    path=Path(producer_report)
    if path.exists():
        if path.is_symlink() or not path.is_file():raise ValueError('Unsafe simulator launch report')
        raw=path.read_bytes()
        if len(raw)>1024*1024:raise ValueError('Unexpected simulator report size')
        producer=json.loads(raw)
        window=producer.get('configured_window') if isinstance(producer,dict) else None
        if (not isinstance(window,dict) or set(window)!=set(expected)
                or any(type(window.get(key)) is not int for key in ('num_history','num_future','frames','final_frame'))
                or not isinstance(window.get('consumed_control_indices'),list)
                or any(type(index) is not int for index in window['consumed_control_indices'])
                or window!=expected):
            raise ValueError('Planner history/frame limit differs from the actual configured rendering window')
        value.update(source='simulation_launch_report.configured_window',
                     producer_report_snapshot_sha256=hashlib.sha256(raw).hexdigest())
    elif required:
        raise ValueError('Controlled NPU loop requires the simulator launch window report')
    return value


def control_queue(monitor,current,scene_step,history,completion):
    """Keep the last required plan; wait for the next scene before an unused one."""
    step=scene_step+history
    if (type(scene_step) is not int or type(history) is not int or scene_step<0
            or history!=monitor.max_history or step!=monitor.next_idx
            or len(current)!=history):
        raise ValueError('Planner queue/control counters differ')
    if step<monitor.max_frame_per_scene:return current,scene_step,True
    if step!=monitor.max_frame_per_scene:
        raise ValueError('Planner control index exceeds the rendered-frame horizon')
    if monitor._reinitialize_queue():
        queue=list(monitor.file_history)
        if len(queue)!=history or monitor.next_idx!=history:
            raise ValueError('Next scene did not supply its complete initial history')
        return queue,0,True
    if Path(completion).exists():return current,scene_step,False
    raise ValueError('Rendered horizon ended without a next scene or completion')


async def run_inference_loop(model,cfg,logger,runtime,args,report):
    import numpy as np
    from mmdet3d.datasets import build_dataset
    from mmdet3d_plugin.datasets.builder import build_dataloader
    from closed_loop.post_processor import ScorePostProcessor
    from simengine_ascend.closed_loop_runtime import assert_module_origins
    completion=Path(cfg.data_root.rstrip('/')).parent/'simulation_completed.flag'
    report['completion_file']=str(completion)
    monitor=AsyncFileMonitor(cfg.sim.monitored_folder,max_history=cfg.queue_length,
                             max_frame_per_scene=cfg.sim.get('max_frame_per_scene',12),
                             max_wait_time=args.wait_timeout,logger=logger,stop_file_path=str(completion),
                             cancellation_file=args.stop_file,deadline_utc=args.deadline_utc)
    save=Path(cfg.sim.plan_save_path);save.mkdir(parents=True,exist_ok=True)
    csv_path=save/'plan_idx.csv';owned=set();rows=[]
    if csv_path.exists():
        with csv_path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['prefix','step','plan_idx']:raise ValueError('Unexpected existing plan index CSV')
            rows=list(reader)
        if len({(row['prefix'],row['step']) for row in rows})!=len(rows):raise ValueError('Duplicate existing plan index rows')
    current=list(monitor.file_history);scene_step=0
    controlled=bool(os.environ.get('WORLDENGINE_CLOSED_LOOP_DEADLINE_UTC'))
    report['control_horizon']=planner_window(cfg.queue_length,monitor.max_frame_per_scene,
        completion.with_name('simulation_launch_report.json'),
        required=runtime.backend=='npu' and controlled and args.max_predictions is None)
    maximum=args.max_predictions or int(cfg.sim.maximum_step)
    if maximum<1:raise ValueError('Maximum prediction count must be positive')
    try:
        if len(current)!=cfg.queue_length:raise ValueError('Simulation completed without a full initial frame queue')
        while report['predictions']<maximum:
            check_stop(args.stop_file,args.deadline_utc)
            current,scene_step,active=control_queue(monitor,current,scene_step,cfg.queue_length,completion)
            if not active:return
            ann=merge_ann_files(cfg,current,scene_step+cfg.queue_length);owned.add(Path(ann))
            cfg.data.test.ann_file=ann
            cfg.data.test.pipeline[0].img_root=cfg.data_root+'sensor_blobs/'
            dataset=build_dataset(cfg.data.test)
            loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=0 if runtime.backend=='npu' else 1,
                                    dist=False,shuffle=False,nonshuffler_sampler=cfg.data.nonshuffler_sampler)
            if runtime.backend=='npu':assert_module_origins(runtime.root,require_plugin=True)
            result=model_inference(model,loader,runtime.torch)
            processor=ScorePostProcessor(cfg.sim.post_process_path,current[-1])
            plan,index=processor.process(result)
            plan=validate_trajectory(host_array(plan,runtime.torch))
            if runtime.torch.is_tensor(index):index=index.item()
            scalar=np.asarray(index)
            if scalar.shape!=() or scalar.dtype.kind not in 'iu':
                raise ValueError('Selected planning index must be an integer scalar')
            index=int(scalar)
            head=model.module.planning_head
            generated=bool(getattr(head,'requires_online_pdm_scoring',False))
            if generated:
                if not 0<=index<int(head.num_anchors):raise ValueError('Invalid generated-trajectory anchor index')
            elif not 0<=index<8192 or processor.traj.shape!=(8192,40,3) or not np.allclose(plan,processor.traj[index],atol=1e-6,rtol=1e-6):
                raise ValueError('Published trajectory differs from its selected original vocabulary entry')
            step=cfg.queue_length+scene_step;path=save/('%s_%s.npy'%(monitor.prefix,step))
            if any(row['prefix']==monitor.prefix and str(row['step'])==str(step) for row in rows):
                raise ValueError('Refusing duplicate scene/step publication')
            if path.exists():raise ValueError('Refusing to replace an existing planning result: '+str(path))
            atomic_numpy(path,plan,replace=False);owned.add(path)
            if result.get('value') is not None:
                value_path=save/('value_%s_%s.npy'%(monitor.prefix,step))
                atomic_numpy(value_path,host_array(result['value'],runtime.torch),replace=False);owned.add(value_path)
            rows.append(dict(prefix=monitor.prefix,step=step,plan_idx=index))
            text=io.StringIO();writer=csv.DictWriter(text,fieldnames=['prefix','step','plan_idx']);writer.writeheader();writer.writerows(rows)
            atomic_write(csv_path,lambda stream:stream.write(text.getvalue().encode()));owned.add(csv_path)
            report['predictions']+=1
            report['last_prediction']=dict(prefix=monitor.prefix,step=step,plan_idx=index,path=str(path),sha256=file_sha(path),
                                           selection_space='generated_trajectory_anchor' if generated else 'original_8192_trajectory_vocabulary')
            logger.info('Published finite selected trajectory %s',path.name)
            if report['predictions']>=maximum or completion.exists():break
            scene_step+=1
            if monitor._reinitialize_queue():
                current=list(monitor.file_history);scene_step=0;continue
            until=time.monotonic()+args.wait_timeout
            while True:
                check_stop(args.stop_file,args.deadline_utc)
                if completion.exists():return
                current=monitor.get_current_queue()
                if current:break
                if time.monotonic()>=until:raise TimeoutError('Timed out waiting for next frame for '+str(monitor.prefix))
                await asyncio.sleep(.1)
    finally:
        monitor.stop()
        clean_related_files(cfg,logger,owned)


def main():
    args=parse_args();check_stop(args.stop_file,args.deadline_utc)
    runtime=ClosedLoopRuntime(args,SIM_ROOT).prepare()
    from mmcv import Config,ConfigDict,DictAction
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet.apis import set_random_seed
    import mmdet3d.models as models
    cfg=Config.fromfile(args.config,import_custom_modules=False)
    if 'sim' not in cfg:
        cfg.sim=ConfigDict(max_wait_time=120,maximum_step=100000000,
            post_process_path=str(runtime.root/'data/alg_engine/test_8192_kmeans.npy'),
            plan_save_path='',merged_ann_save_dir='',monitored_folder='')
    cfg.data.test.type='NavSimOpenSceneE2EClosedLoop'
    if args.cfg_options:
        action=DictAction(option_strings=[],dest='cfg_options');namespace=argparse.Namespace()
        action(argparse.ArgumentParser(),namespace,args.cfg_options)
        cfg.merge_from_dict(namespace.cfg_options)
    if cfg.data.test.type!='NavSimOpenSceneE2EClosedLoop':
        raise ValueError('Closed-loop inference requires the original closed-loop dataset')
    for key in ('plan_save_path','merged_ann_save_dir','monitored_folder'):
        if not cfg.sim.get(key):raise ValueError('Missing explicit simulation path sim.'+key)
        Path(cfg.sim[key]).mkdir(parents=True,exist_ok=True)
    if cfg.sim.get('clean_record_data',False):raise ValueError('Recursive renderer record cleanup is unsupported')
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if runtime.backend=='npu':
        from simengine_ascend.closed_loop_runtime import assert_module_origins
        assert_module_origins(runtime.root,require_plugin=True)
        cfg.pop('fp16',None)
    elif args.precision=='fp32':cfg.pop('fp16',None)
    if cfg.get('cudnn_benchmark',False) and runtime.backend=='cuda':runtime.torch.backends.cudnn.benchmark=True
    cfg.model.pretrained=None;cfg.model.train_cfg=None;cfg.data.test.test_mode=True
    if cfg.model.get('img_backbone'):cfg.model.img_backbone.norm_cfg=dict(type='BN',requires_grad=True)
    set_random_seed(args.seed,deterministic=args.deterministic)
    logger=logging.getLogger('worldengine.closed_loop');logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True,exist_ok=True)
        logger.addHandler(logging.FileHandler(Path(args.log_dir)/'closed_loop_planner.log'))
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    report=dict(result='running',predictions=0,device=args.device,config=str(Path(args.config).resolve()),
                config_sha256=file_sha(args.config),checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),real_closed_loop_quality_accepted=False)
    report_path=Path(cfg.sim.plan_save_path)/'closed_loop_planner_report.json'
    def publish():atomic_write(report_path,lambda stream:stream.write(json.dumps(report,indent=2,allow_nan=False).encode()))
    publish()
    try:
        with runtime.precision_context():
            model=models.build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
            if runtime.backend=='npu':
                saved=runtime.torch.load(checkpoint,map_location='cpu',weights_only=False)
                weights=saved.get('state_dict',saved)
                revised={key.removeprefix('module.'):value for key,value in weights.items()}
                if len(revised)!=len(weights):raise ValueError('Checkpoint prefix rewrite collides')
                model.load_state_dict(revised,strict=True)
                report['checkpoint_strict_state_dict']=True
                del saved,weights,revised
            else:
                if cfg.get('fp16') is not None:wrap_fp16_model(model)
                load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
                report['checkpoint_strict_state_dict']=True
            if args.fuse_conv_bn:model=fuse_conv_bn(model)
            model=runtime.wrap(model)
            with runtime.optimization_context():
                asyncio.run(run_inference_loop(model,cfg,logger,runtime,args,report))
            runtime.validate_optimization(report['predictions'])
        if report['predictions']<1:raise ValueError('No real model prediction was produced')
        report.update(result='completed_configured_closed_loop_planner_run',runtime=runtime.finish())
    except BaseException as error:
        report.update(result='planner_error_or_interrupted',error_type=type(error).__name__,error=str(error),runtime=runtime.provenance)
        raise
    finally:
        report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        publish()


if __name__=='__main__':main()
