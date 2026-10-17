"""Export complete original MMCV1.6.2 CUDA MSDA goldens for the NPU checker.

Requires the separately published validate_original_cuda_msda.py validator at
its recorded SHA256. Same two normal-L1 inputs, unchanged original CPU32 math,
and four same-storage CUDA executions. The complete tensors stay on the host.
Preview does not import Torch or create output. Export completion is distinct
from numerical agreement with CPU32. No NPU or model training is involved.
"""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,shutil,signal,sys,time,traceback,uuid,warnings
from datetime import datetime,timezone
from pathlib import Path
BASE_SHA="af7309e616f9a02648f9a5b35f8960ea97f51e108b1915bca7843448bd733fea"
PRIOR_SHA=None
RESERVE=8*1024**3
CASE_NAMES=('temporal40000','query65536')
ROLES=('forward','value_gradient','locations_gradient','weights_gradient')
INPUT_ROLES=('value','sampling_locations','attention_weights','spatial_shapes','level_start_index','grad_output')


def need(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def persist(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb')as stream:
        stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def payload_estimate(spec,repetitions):
    b,q,h,c,p=(spec[k]for k in('batch','queries','heads','channels','points'))
    levels=len(spec['spatial_shapes']);s=sum(y*x for y,x in spec['spatial_shapes'])
    value=b*s*h*c*4;locations=b*q*h*levels*p*2*4;weights=b*q*h*levels*p*4;go=b*q*h*c*4
    roles=value+locations+weights+go
    integer_metadata=(2*levels+levels)*8
    return dict(original_CPU_input_bytes=roles+integer_metadata,
                one_complete_output_and_three_gradient_bytes=roles,
                repetitions=repetitions,original_CPU32_reference_bytes=roles,
                raw_tensor_payload_bytes=(repetitions+2)*roles+integer_metadata)

def input_rows(torch,base,raw,shapes,starts,go):
    values=raw+[shapes,starts,go]
    need(len(values)==6,'All original inputs/metadata/GO required')
    for index,value in enumerate(values):
        need(value.device.type=='cpu'and value.is_contiguous(),'Original CPU input must remain contiguous')
        need(value.dtype==(torch.long if index in(3,4)else torch.float32),'Original FP32/int64 input dtype differs')
        if value.is_floating_point():need(bool(torch.isfinite(value).all()),'Original CPU input contains nonfinite values')
    return {role:dict(**base.tensor_metadata(value),tensor_raw_SHA256=base.tensor_sha(value))
            for role,value in zip(INPUT_ROLES,values)}

def save_tensors(torch,base,path,values,case_name,kind,repetition=None):
    need(not path.exists()and not path.is_symlink(),'Fresh exclusive tensor artifact required')
    need(all(value.device.type=='cpu'for value in values.values()),'Only post-loop complete CPU tensors may be saved')
    tensor_bytes=sum(value.numel()*value.element_size()for value in values.values())
    need(shutil.disk_usage(path.parent).free>=RESERVE+tensor_bytes+1024**2,
         '8GiB disk reserve plus next complete file required')
    metadata={}
    for role,value in values.items():
        need(value.is_contiguous(),'Saved tensor storage must be explicit contiguous CPU')
        if value.is_floating_point():need(bool(torch.isfinite(value).all()),'Nonfinite exported complete tensor')
        metadata[role]=dict(**base.tensor_metadata(value),tensor_raw_SHA256=base.tensor_sha(value))
    payload=dict(schema_version=1,case=case_name,kind=kind,repetition=repetition,
                 original_base_worker_SHA256=BASE_SHA,original_prior_report_SHA256=PRIOR_SHA,
                 tensors=values)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
    with temporary.open('xb')as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    need(shutil.disk_usage(path.parent).free>=RESERVE,'8GiB disk reserve crossed while writing own export')
    os.link(temporary,path,follow_symlinks=False);temporary.unlink()
    return dict(file=path.name,bytes=path.stat().st_size,file_SHA256=sha(path),raw_tensor_bytes=tensor_bytes,
                tensors=metadata,format='torch.save; all stored tensors CPU; original integer metadata kept int64')

def export_case(torch,base,spec,function,reference,repetitions,output,device):
    start=time.monotonic()
    raw,shapes,starts,go=base.inputs(torch,spec)
    before=[base.tensor_sha(x)for x in raw+[shapes,starts,go]]
    metadata=input_rows(torch,base,raw,shapes,starts,go)
    cpu=[x.detach().clone().requires_grad_(True)for x in raw]
    expected=reference(cpu[0],shapes,cpu[1],cpu[2]);expected.backward(go)
    need(expected.device.type=='cpu'and expected.dtype==torch.float32 and all(x.grad is not None for x in cpu),
         'Complete original CPU32 oracle output/three gradients required')
    reference_outputs=[expected.detach()]+[x.grad.detach()for x in cpu]
    gpu=[x.detach().to(device).requires_grad_(True)for x in raw]
    gpu_shapes,gpu_starts,gpu_go=[x.to(device)for x in(shapes,starts,go)]
    GPU_metadata={role:base.tensor_metadata(value)for role,value in zip(INPUT_ROLES,gpu+[gpu_shapes,gpu_starts,gpu_go])}
    pointers=[x.data_ptr()for x in gpu+[gpu_shapes,gpu_starts,gpu_go]]
    retained=[];loop_start=time.monotonic()
    # Same source loop as frozen base: original function, same buffers, no CPU
    # transfer, hashing, scalar validation sync, serialization or RNG reset.
    for _ in range(repetitions):
        for value in gpu:value.grad=None
        actual=function.apply(gpu[0],gpu_shapes,gpu_starts,gpu[1],gpu[2],64)
        actual.backward(gpu_go)
        need(actual.device==device and actual.dtype==torch.float32 and all(x.grad is not None for x in gpu),
             'Complete original CUDA output/three gradients required')
        retained.append([actual.detach()]+[x.grad.detach()for x in gpu])
    torch.cuda.synchronize(device);loop_seconds=time.monotonic()-loop_start
    need([x.data_ptr()for x in gpu+[gpu_shapes,gpu_starts,gpu_go]]==pointers,'Actual GPU input storages changed')
    actual_cpu=[[value.cpu().contiguous()for value in row]for row in retained]
    need([base.tensor_sha(x.cpu())for x in gpu+[gpu_shapes,gpu_starts,gpu_go]]==before,
         'Actual GPU input/GO/intmetadata values changed')
    need([base.tensor_sha(x)for x in raw+[shapes,starts,go]]==before,'Original CPU input values changed')
    case_dir=output/spec['name'];case_dir.mkdir()
    files=[]
    input_artifact=save_tensors(torch,base,case_dir/'inputs.pt',
            {role:value.detach().contiguous()for role,value in zip(INPUT_ROLES,raw+[shapes,starts,go])},
            spec['name'],'original_CPU_FP32_inputs_GO_and_int64_metadata')
    files.append(input_artifact)
    files.append(save_tensors(torch,base,case_dir/'original_CPU32_reference.pt',
            {role:value.contiguous()for role,value in zip(ROLES,reference_outputs)},spec['name'],'unchanged_original_CPU32_oracle'))
    comparisons=[];repeats=[]
    for index,values in enumerate(actual_cpu):
        files.append(save_tensors(torch,base,case_dir/('cuda_repeat_%03d.pt'%index),
                dict(zip(ROLES,values)),spec['name'],'original_CUDA_forward_and_three_gradients',index))
        comparisons.append(dict(repetition=index,tensors={role:base.compare(torch,x,y,role)
                               for role,x,y in zip(ROLES,values,reference_outputs)}))
        if index:repeats.append(dict(repetition=index,reference_repetition=0,tensors={role:base.compare(torch,x,y,role)
                               for role,x,y in zip(ROLES,values,actual_cpu[0])}))
    row=dict(case_key=next(key for key,value in base.CASES.items()if value==spec),spec=spec,
             same_prior_complete_input_SHA256=before,CPU_input_metadata=metadata,
             GPU_input_metadata=GPU_metadata,repetitions=repetitions,original_GPU_calls=repetitions,
             original_CPU32_calls=1,no_CPU_reads_or_reseed_inside_GPU_repeat_loop=True,
             same_GPU_input_storages_for_all_repetitions=True,GPU_loop_seconds=loop_seconds,
             original_CPU32_comparisons=comparisons,CUDA_repetition0_comparisons=repeats,
             all_original_CPU32_gates_passed=all(v['original_gate_passed']for r in comparisons for v in r['tensors'].values()),
             all_CUDA_repetitions_raw_tensor_bits_equal=all(v['bitwise_equal']for r in repeats for v in r['tensors'].values()),
             original_inputs_unchanged=True,files=files,total_saved_file_bytes=sum(r['bytes']for r in files),
             raw_tensor_payload_bytes=sum(r['raw_tensor_bytes']for r in files),elapsed_seconds=time.monotonic()-start)
    persist(case_dir/'case_manifest.json',row)
    row['case_manifest']=dict(file='case_manifest.json',bytes=(case_dir/'case_manifest.json').stat().st_size,
                              file_SHA256=sha(case_dir/'case_manifest.json'))
    return row

def load_base(path):
    need(path.is_file() and not path.is_symlink() and sha(path)==BASE_SHA,'Exact published original CUDA validator SHA required')
    spec=importlib.util.spec_from_file_location('published_original_cuda_msda_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    global PRIOR_SHA
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--validator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh golden directory under existing canonical parent')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--maximum-seconds',type=int,default=300)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    need(args.device.startswith('cuda:') and args.device[5:].isdigit(),'Explicit logical cuda:N device required')
    need(1<=args.maximum_seconds<=3600,'maximum-seconds must be in 1..3600')
    report=dict(result='preview_no_framework_no_output_no_device',worker_SHA256=sha(__file__),base_worker_SHA256=BASE_SHA,
        case_keys=list(CASE_NAMES),repetitions=4,rows=[],export_completion_is_numerical_acceptance=False,
        intended_use='Copy complete original tensors and authenticate manifest SHA; cross-version RNG seeds do not establish equal inputs.')
    if not args.execute:print(json.dumps(report));return 0
    need(sys.platform=='linux' and 'torch' not in sys.modules,'Fresh isolated Linux CUDA process required')
    need(args.output.is_absolute() and args.output.resolve()==args.output and not args.output.exists()
         and not args.output.is_symlink() and args.output.parent.is_dir() and args.output.parent.resolve()==args.output.parent,
         'Fresh output directory under canonical existing parent required')
    base=load_base(args.validator)
    total=sum(payload_estimate(base.CASES[name],4)['raw_tensor_payload_bytes'] for name in CASE_NAMES)
    need(shutil.disk_usage(args.output.parent).free>=RESERVE+total+64*1024**2,'8GiB reserve plus complete export bytes required')
    args.output.mkdir();manifest_path=args.output/'manifest.json'
    report.update(result='incomplete_original_CUDA_MSDA_goldens_export',started_UTC=datetime.now(timezone.utc).isoformat(),
                  estimated_raw_tensor_payload_bytes=total)
    persist(manifest_path,report)
    def cutoff(*_):raise TimeoutError('Own CUDA golden exporter maximum-seconds reached')
    signal.signal(signal.SIGALRM,cutoff);signal.signal(signal.SIGTERM,cutoff)
    signal.setitimer(signal.ITIMER_REAL,args.maximum_seconds)
    torch=None;function=None;reference=None;extension=None;before=None;policy=None;code=2
    try:
        workspace=os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        need(workspace in (':4096:8',':16:8'),'Legal CUBLAS workspace policy required before Torch import')
        import torch
        need(not torch.cuda.is_initialized(),'Fresh CUDA context required')
        need(importlib.metadata.version('mmcv-full')=='1.6.2','Original MMCV1.6.2 required')
        index=int(args.device[5:]);need(torch.cuda.is_available() and index<torch.cuda.device_count(),'Selected logical CUDA device unavailable')
        torch.cuda.set_device(index);device=torch.device('cuda',index)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=False)
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as function,multi_scale_deformable_attn_pytorch as reference
        import mmcv._ext as extension
        before=base.source_identity(function,reference,extension)
        need(all(before[k]==v for k,v in base.EXPECTED_PYTHON.items()),'Original MMCV CUDA Python wrapper or CPU32 source differs')
        policy=base.precision(torch);base.check_precision(policy,4)
        free,total_GPU=torch.cuda.mem_get_info(device);need(free>=4*1024**3,'4GiB selected GPU workspace required')
        # Immutable pre-execution source report, explicitly not a prior numerical result.
        # Legacy schema-1 payload field original_prior_report_SHA256 binds this file.
        proof=dict(schema_version=1,result='actual_pre_execution_original_CUDA_source_identity_not_numerical_measurement',
            validator_SHA256=BASE_SHA,exporter_SHA256=report['worker_SHA256'],source=before,precision=policy,
            torch_version=str(torch.__version__),CUDA_build=torch.version.cuda,mmcv_version='1.6.2',
            cases={name:base.CASES[name] for name in CASE_NAMES})
        persist(args.output/'source_proof.json',proof);PRIOR_SHA=sha(args.output/'source_proof.json')
        report.update(prior_report_SHA256=PRIOR_SHA,prior_report_kind='immutable_pre_execution_source_proof_not_prior_numerical_measurement',
            prior_report_file='source_proof.json',source_before=before,precision_before=policy,
            torch_version=str(torch.__version__),CUDA_build=torch.version.cuda,mmcv_version='1.6.2',
            numpy_version=importlib.metadata.version('numpy'),actual_device=dict(local_index=index,
                name=torch.cuda.get_device_name(index),capability=list(torch.cuda.get_device_capability(index)),
                free_GPU_bytes=free,total_GPU_bytes=total_GPU))
        persist(manifest_path,report)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            for name in CASE_NAMES:
                report['rows'].append(export_case(torch,base,base.CASES[name],function,reference,4,args.output,device))
                persist(manifest_path,report)
            report['actual_warnings']=[dict(category=w.category.__name__,message=str(w.message)) for w in recorded]
        report.update(source_after=base.source_identity(function,reference,extension),precision_after=base.precision(torch))
        base.check_precision(report['precision_after'],4)
        need(before==report['source_after'] and policy==report['precision_after'] and sha(args.validator)==BASE_SHA
             and sha(__file__)==report['worker_SHA256'] and sha(args.output/'source_proof.json')==PRIOR_SHA,
             'Source/precision/validator/exporter/source-proof changed')
        for row in report['rows']:
            for metadata in row['files']+[row['case_manifest']]:
                path=args.output/row['spec']['name']/metadata['file']
                need(path.stat().st_size==metadata['bytes'] and sha(path)==metadata['file_SHA256'],'Complete exported artifact changed')
        need(not any(n=='ascend' or n.startswith(('ascend.','torch_npu','mx_driving')) for n in sys.modules),'Unexpected NPU source imported')
        report.update(result='complete_original_CUDA_MSDA_full_tensor_goldens_export_not_numerical_acceptance',
            completed_cases=2,saved_tensor_files=12,total_saved_file_bytes=sum(r['total_saved_file_bytes'] for r in report['rows']),
            source_and_precision_unchanged=True,all_files_final_wholeSHA_verified=True,
            all_original_CPU32_gates_passed=all(r['all_original_CPU32_gates_passed'] for r in report['rows']),
            all_CUDA_repetitions_raw_tensor_bits_equal=all(r['all_CUDA_repetitions_raw_tensor_bits_equal'] for r in report['rows']),
            original_CPU32_gate_tolerances=dict(forward=dict(atol=2e-5,rtol=2e-5),gradients=dict(atol=1e-3,rtol=1e-4)))
        code=0
    except BaseException as error:report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if before is not None:
            try:
                report.update(source_after=base.source_identity(function,reference,extension),precision_after=base.precision(torch))
                unchanged=(before==report['source_after'] and policy==report['precision_after'] and sha(args.validator)==BASE_SHA
                           and sha(__file__)==report['worker_SHA256'])
                report['source_and_precision_unchanged']=unchanged
                need(unchanged,'Final source/precision integrity differs')
            except BaseException as error:report.update(result='incomplete_final_integrity',final_integrity_error=repr(error));code=2
        report.update(actual_exit=code,finished_UTC=datetime.now(timezone.utc).isoformat());persist(manifest_path,report)
    print(json.dumps({k:report.get(k) for k in ('result','actual_exit','completed_cases','saved_tensor_files','total_saved_file_bytes','all_original_CPU32_gates_passed','error')}))
    return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,OSError) as error:
        print(json.dumps(dict(result='blocked_before_execution',actual_exit=2,error=str(error))),file=sys.stderr)
        raise SystemExit(2)
