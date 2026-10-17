"""Compare actual registered NPU MSDA entries with complete saved CUDA goldens.

Preview is the default: no Torch import, device initialization or output write.
The supplied manifest SHA authenticates an existing schema-1 golden export;
tensor payloads are loaded with weights_only=True. Use only a manifest whose
origin you trust. No RNG inputs or CPU reference tensors are regenerated.

Exit 0: every complete CUDA and CPU32 comparison is within its original gate.
Exit 1: the full matrix completed, with at least one comparison outside a gate.
Exit 2: blocked, incomplete, invalid input, or source/precision integrity failure.
CUDA and CPU32 conclusions are also reported separately. Bitwise inequality
alone is not a tolerance failure. This tool does not validate model training.
"""
import argparse,gc,hashlib,importlib,inspect,json,math,os,re,signal,sys,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

FORWARD_ATOL=2e-5
FORWARD_RTOL=2e-5
GRAD_ATOL=1e-3
GRAD_RTOL=1e-4
INPUT_ROLES=('value','sampling_locations','attention_weights','spatial_shapes','level_start_index','grad_output')
OUTPUT_ROLES=('forward','value_gradient','locations_gradient','weights_gradient')
INPUTS=INPUT_ROLES
ROLES=OUTPUT_ROLES


def need(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()

def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

def persist(path, report):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def tensor_metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), dtype=str(tensor.dtype),
                device=str(tensor.device), contiguous=tensor.is_contiguous(),
                requires_grad=tensor.requires_grad)

def tensor_sha(tensor):
    if tensor.device.type != 'cpu':
        raise RuntimeError('Tensor hashing belongs only to the post-repeat CPU phase')
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

def compare(torch, actual, reference, role):
    if actual.device.type != 'cpu' or reference.device.type != 'cpu':
        raise RuntimeError('Complete comparisons must execute on CPU after repetitions')
    if actual.dtype != torch.float32 or reference.dtype != torch.float32 or actual.shape != reference.shape:
        raise RuntimeError('Complete original FP32 tensor identity differs')
    atol, rtol = ((FORWARD_ATOL, FORWARD_RTOL) if role == 'forward'
                  else (GRAD_ATOL, GRAD_RTOL))
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    delta32 = (actual - reference).abs()
    bad = (~finite) | (delta32 > atol + rtol * reference.abs())
    delta64 = actual.double() - reference.double()
    actual_hash, reference_hash = tensor_sha(actual), tensor_sha(reference)
    return dict(shape=list(actual.shape), elements=actual.numel(), dtype=str(actual.dtype),
                all_finite=bool(finite.all()), failing_elements=int(bad.sum()),
                max_absolute_difference=float(delta64.abs().max()),
                mean_absolute_difference=float(delta64.abs().mean()),
                relative_l2=float(delta64.norm() / reference.double().norm().clamp_min(1e-30)),
                bitwise_equal=actual_hash == reference_hash,
                original_gate_passed=bool(torch.allclose(actual, reference, atol=atol, rtol=rtol,
                                                        equal_nan=False)),
                absolute_tolerance=atol, relative_tolerance=rtol,
                actual_tensor_sha256=actual_hash, reference_tensor_sha256=reference_hash)

def precision(torch,torch_npu):
    def option(key):
        try:
            v=torch_npu._C._npu_getOption(key);return v.decode()if isinstance(v,bytes)else v
        except Exception:return None
    return dict(default_dtype=str(torch.get_default_dtype()),autocast=bool(torch.is_autocast_enabled('npu')),
        matmul_HF32=bool(torch.npu.matmul.allow_hf32),conv_HF32=bool(torch.npu.conv.allow_hf32),
        deterministic=bool(torch.are_deterministic_algorithms_enabled()),warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        CPU_threads=torch.get_num_threads(),raw_options={k:option(k)for k in('ALLOW_MATMUL_HF32','ALLOW_CONV_HF32','ALLOW_INTERNAL_FORMAT')},
        environment={k:os.environ.get(k)for k in('ASCEND_RT_VISIBLE_DEVICES','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','CLOSE_MATMUL_K_SHIFT','HCCL_DETERMINISTIC','LCCL_DETERMINISTIC','OPENBLAS_CORETYPE')})

def execute_entry(torch,function,entry,inputs,public_msda,device_name,backend):
    device={k:v.to(device_name)for k,v in inputs.items()}
    for k in INPUTS[:3]:device[k]=device[k].detach().requires_grad_(True)
    args=[device['value'],device['spatial_shapes'],device['level_start_index'],device['sampling_locations'],device['attention_weights']]
    if entry=='mmcv':args.append(64)
    with torch.autocast('npu',enabled=False):out=function.apply(*args)
    need(out.device.type=='npu'and out.dtype==torch.float32 and out.shape==inputs['grad_output'].shape,'Registered full-shape FP32 forward differs')
    flags=list(out.grad_fn.needs_input_grad);need(flags==[True,False,False,True,True]+([False]if entry=='mmcv'else[]),'Actual registered argument-gradient positions differ')
    selected=getattr(out.grad_fn,'_worldengine_msda_backend',None)
    need(selected==backend and getattr(out.grad_fn,'_worldengine_reference_msda',None)is True,'Real registered entry did not execute selected public CUDA-compatible backend')
    if backend=='cuda-compatible':need(public_msda._CUDA_FUNCTION is not None,'Public numerical core factory was not reached')
    before={k:tensor_metadata(t)for k,t in device.items()}
    out.backward(device['grad_output']);torch.npu.synchronize()
    need(all(device[k].grad is not None and device[k].grad.shape==device[k].shape and device[k].grad.device==device[k].device and device[k].grad.dtype==torch.float32 for k in INPUTS[:3]),'Full registered-entry original leaf gradients missing')
    outputs={'forward':out.detach().cpu(),**{role:device[k].grad.detach().cpu()for k,role in zip(INPUTS[:3],ROLES[1:])}}
    need(all(bool(torch.isfinite(v).all())for v in outputs.values()),'Registered full output/gradients nonfinite')
    need(all(tensor_sha(device[k].detach().cpu())==tensor_sha(inputs[k])for k in INPUTS),'Original fixed input mutated')
    result=dict(entry=entry,actual_public_registered_apply=True,actual_forward_ctx_backend=selected,
        actual_factory_needs_input_grad=flags,actual_input_metadata=before,
        actual_output_NPU_metadata={'forward':tensor_metadata(out),**{role:tensor_metadata(device[k].grad)for k,role in zip(INPUTS[:3],ROLES[1:])}},
        output_tensor_SHA256={k:tensor_sha(v)for k,v in outputs.items()},all_original_inputs_unchanged=True)
    del device,out;gc.collect();torch.npu.empty_cache();return outputs,result

def valid_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def literal_component(value):
    need(isinstance(value,str) and value not in ('','.','..') and len(value)<=200
         and re.fullmatch('[A-Za-z0-9_.-]+',value) is not None,'Unsafe literal artifact component')
    return value

def canonical_directory(path):
    need(path.is_absolute() and path.is_dir() and not path.is_symlink()
         and path.resolve(strict=True)==path,'Canonical directory required: '+str(path))
    return path

def file_gate(path,metadata):
    need(isinstance(metadata,dict) and type(metadata.get('bytes')) is int and metadata['bytes']>0
         and valid_sha(metadata.get('file_SHA256')),'Invalid complete artifact metadata')
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==metadata['bytes']
         and sha(path)==metadata['file_SHA256'],'Complete artifact bytes/SHA differ: '+str(path))
    return dict(bytes=metadata['bytes'],sha256=metadata['file_SHA256'])

def manifest_gate(root,digest):
    canonical_directory(root)
    need(valid_sha(digest),'Manifest requires its complete lowercase SHA256')
    path=root/'manifest.json'
    need(path.is_file() and not path.is_symlink() and sha(path)==digest,'Golden manifest wholeSHA differs')
    m=json.loads(path.read_bytes())
    need(m.get('result')=='complete_original_CUDA_MSDA_full_tensor_goldens_export_not_numerical_acceptance'
         and m.get('actual_exit')==0 and m.get('source_and_precision_unchanged') is True
         and m.get('all_files_final_wholeSHA_verified') is True,'Golden export is not complete and immutable')
    need(valid_sha(m.get('base_worker_SHA256')) and valid_sha(m.get('prior_report_SHA256')),
         'Original CUDA generator/prior provenance SHA required')
    need(m.get('completed_cases')==2 and m.get('saved_tensor_files')==12 and m.get('repetitions')==4,
         'This verifier covers exactly the two complete original shapes/four repeats')
    original=m.get('source_before',{})
    need(m.get('mmcv_version')=='1.6.2' and original==m.get('source_after')
         and original.get('CPU_reference')=='4ef9c84cb9016a3554b7fd62d35a6e8f8cb440099103679de01fc0491eaf5128'
         and original.get('CUDA_forward')=='3608dc1cd24471538c289cfa89d3866f7c496b18c9f00e85c9c1143385a4c332'
         and original.get('CUDA_backward')=='7efd29b77fbc5baf6abeb462b2169080c2f0891d17c278c0c09d00cbef0adbb8',
         'Original MMCV1.6.2 Python forward/backward and CPU32 source hashes required')
    need(m.get('original_CPU32_gate_tolerances')==dict(forward=dict(atol=FORWARD_ATOL,rtol=FORWARD_RTOL),
         gradients=dict(atol=GRAD_ATOL,rtol=GRAD_RTOL)),'Original golden tolerances differ')
    need(m.get('precision_before')==m.get('precision_after') and m['precision_before']['default_dtype']=='torch.float32'
         and m['precision_before']['matmul_TF32'] is False and m['precision_before']['cudnn_TF32'] is False
         and m['precision_before']['autocast_enabled'] is False and m['precision_before']['deterministic_algorithms'] is True
         and m['precision_before']['deterministic_warn_only'] is False and m['precision_before']['CPU_threads']==4,
         'Original CUDA strict FP32/TF32/autocast/thread policy differs')
    rows=m.get('rows');need(isinstance(rows,list) and len(rows)==2,'Two complete golden cases required')
    need([r.get('case_key') for r in rows]==['temporal40000','query65536'],'Original full case order differs')
    names=set()
    for row in rows:
        spec=row['spec'];literal_component(spec['name']);need(spec['name'] not in names,'Duplicate case directory');names.add(spec['name'])
        batch,queries=(2,40000) if row['case_key']=='temporal40000' else (1,65536)
        need(spec['batch']==batch and spec['queries']==queries and spec['spatial_shapes']==[[200,200]]
             and spec['heads']==8 and spec['channels']==32 and spec['points']==4,'Full original geometry differs')
        expected=['inputs.pt','original_CPU32_reference.pt']+['cuda_repeat_%03d.pt'%i for i in range(4)]
        need([f['file'] for f in row['files']]==expected,'Six literal complete files per case required')
        need(row['case_manifest']['file']=='case_manifest.json','Literal case manifest required')
        need(len(row['same_prior_complete_input_SHA256'])==6 and all(valid_sha(x) for x in row['same_prior_complete_input_SHA256']),
             'Six original complete input hashes required')
        need(row.get('original_inputs_unchanged') is True,'Original CUDA inputs were not verified unchanged')
        for metadata in row['files']+[row['case_manifest']]:literal_component(metadata['file'])
    return m

def inventory(root,manifest):
    result={'manifest.json':dict(bytes=(root/'manifest.json').stat().st_size,sha256=sha(root/'manifest.json'))}
    if 'prior_report_file' in manifest:
        need(manifest['prior_report_file']=='source_proof.json'
             and manifest.get('prior_report_kind')=='immutable_pre_execution_source_proof_not_prior_numerical_measurement',
             'Unsupported portable source-proof identity')
        path=root/'source_proof.json'
        need(path.is_file() and not path.is_symlink() and sha(path)==manifest['prior_report_SHA256'],
             'Portable source proof wholeSHA differs')
        proof=json.loads(path.read_bytes())
        need(proof.get('schema_version')==1
             and proof.get('result')=='actual_pre_execution_original_CUDA_source_identity_not_numerical_measurement'
             and proof.get('validator_SHA256')==manifest['base_worker_SHA256']
             and valid_sha(proof.get('exporter_SHA256'))
             and proof.get('source')==manifest['source_before']
             and proof.get('precision')==manifest['precision_before']
             and proof.get('torch_version')==manifest['torch_version']
             and proof.get('CUDA_build')==manifest['CUDA_build']
             and proof.get('mmcv_version')==manifest['mmcv_version']
             and proof.get('cases')=={row['case_key']:row['spec'] for row in manifest['rows']},
             'Portable source proof does not match export source/precision/cases')
        result['source_proof.json']=dict(bytes=path.stat().st_size,sha256=sha(path))
    for row in manifest['rows']:
        name=literal_component(row['spec']['name']);directory=canonical_directory(root/name)
        for metadata in row['files']+[row['case_manifest']]:
            result[name+'/'+metadata['file']]=file_gate(directory/metadata['file'],metadata)
        case=json.loads((directory/'case_manifest.json').read_bytes())
        need(case=={k:v for k,v in row.items() if k!='case_manifest'},'Case manifest does not match outer immutable row')
    return result

def load_tensor_file(torch,root,manifest,row,file_name,kind):
    metadata=next(f for f in row['files'] if f['file']==file_name)
    path=root/row['spec']['name']/file_name;file_gate(path,metadata)
    # Restricted Torch loader: no general pickle globals or arbitrary constructors.
    payload=torch.load(path,map_location='cpu',weights_only=True)
    need(isinstance(payload,dict) and payload.get('schema_version')==1 and payload.get('case')==row['spec']['name']
         and payload.get('kind')==kind and payload.get('original_base_worker_SHA256')==manifest['base_worker_SHA256']
         and payload.get('original_prior_report_SHA256')==manifest['prior_report_SHA256'],'Complete tensor payload provenance differs')
    repetition=int(file_name[12:15]) if file_name.startswith('cuda_repeat_') else None
    need(payload.get('repetition')==repetition,'Saved CUDA repeat identity differs')
    tensors=payload['tensors'];roles=INPUT_ROLES if file_name=='inputs.pt' else OUTPUT_ROLES
    need(isinstance(tensors,dict) and set(tensors)==set(roles)==set(metadata['tensors']),'Exact complete tensor role set required')
    for role in roles:
        tensor=tensors[role];meta=metadata['tensors'][role]
        need(isinstance(tensor,torch.Tensor) and tensor_metadata(tensor)=={k:v for k,v in meta.items() if k!='tensor_raw_SHA256'}
             and tensor_sha(tensor)==meta['tensor_raw_SHA256'],'Complete tensor metadata/rawSHA differs: '+role)
        need(tensor.device.type=='cpu' and tensor.is_contiguous() and tensor.storage_offset()==0 and not tensor.requires_grad,
             'Original contiguous CPU payload required')
        dtype=torch.int64 if role in ('spatial_shapes','level_start_index') else torch.float32
        need(tensor.dtype==dtype and bool(torch.isfinite(tensor).all()),'Finite original FP32/int64 tensor required')
    if file_name=='inputs.pt':
        need([tensor_sha(tensors[k]) for k in INPUT_ROLES]==row['same_prior_complete_input_SHA256'],'Six original input rawSHA differ')
        b,q,h,c,p=(row['spec'][k] for k in ('batch','queries','heads','channels','points'))
        shapes={'value':[b,40000,h,c],'sampling_locations':[b,q,h,1,p,2],'attention_weights':[b,q,h,1,p],
                'spatial_shapes':[1,2],'level_start_index':[1],'grad_output':[b,q,h*c]}
        need(all(list(tensors[k].shape)==shapes[k] for k in INPUT_ROLES),'Full input tensor geometry differs')
        need(tensors['spatial_shapes'].tolist()==[[200,200]] and tensors['level_start_index'].tolist()==[0],
             'Original level metadata differs')
    file_gate(path,metadata);return tensors

def fingerprint(root):
    files=sorted(list((root/'projects/AlgEngine').rglob('*.py'))+list((root/'ascend').glob('*.py')))
    need(files and all(not p.is_symlink() for p in files),'Source tree must contain ordinary Python files')
    digest=hashlib.sha256()
    for path in files:digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return dict(sha256=digest.hexdigest(),python_files=len(files))

def source_inventory(root):
    paths=[root/'ascend'/name for name in ('msda.py','msda_cuda_compatible.py','compat.py')]
    need(all(p.is_file() and not p.is_symlink() for p in paths),'Public backend/dispatch/compatibility source files required')
    return {p.relative_to(root).as_posix():sha(p) for p in paths}

def policy_gate(policy):
    need(policy['default_dtype']=='torch.float32' and policy['autocast'] is False and policy['CPU_threads']==4
         and policy['matmul_HF32'] is False and policy['conv_HF32'] is False and policy['deterministic'] is True
         and policy['warn_only'] is False,'Actual strict FP32/HF32/autocast/thread readback differs')

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--goldens',type=Path,required=True)
    parser.add_argument('--manifest',required=True,help='Expected whole SHA256 of goldens/manifest.json')
    parser.add_argument('--source',type=Path,default=None,help='WorldEngine repository root; inferred only when this tool is inside the repository')
    parser.add_argument('--output',type=Path,required=True,help='Fresh report JSON; parent directory must exist')
    parser.add_argument('--backend',choices=['reference','cuda-compatible'],default='cuda-compatible')
    parser.add_argument('--device',default='npu:0',help='Visible logical NPU device, for example npu:0')
    parser.add_argument('--entries',nargs='+',choices=['mx','mmcv'],default=['mx','mmcv'])
    parser.add_argument('--maximum-seconds',type=int,default=300)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.source is None:
        parents=Path(__file__).resolve().parents
        if len(parents)>2 and (parents[2]/'ascend/msda.py').is_file():args.source=parents[2]
    need(args.source is not None,'Explicit --source is required when the tool is outside the repository')
    need(valid_sha(args.manifest),'Manifest requires its complete lowercase SHA256')
    need(re.fullmatch('npu:[0-9]+',args.device) is not None,'Explicit logical npu:N device required')
    need(1<=args.maximum_seconds<=3600,'maximum-seconds must be in 1..3600')
    need(args.entries==[k for k in ('mx','mmcv') if k in args.entries],'Ordered unique MX/MMCV entries required')
    return args

def main(argv=None):
    args=parse_args(argv)
    report=dict(schema_version=1,result='preview_no_framework_no_output_no_device',worker_SHA256=sha(__file__),
                backend=args.backend,entries=args.entries,device=args.device,golden_manifest_SHA256=args.manifest,
                expected_full_tensor_comparisons=40*len(args.entries),rows=[],
                scope='Two complete original normal-L1 cases; fixed saved CUDA and CPU32 targets separately. No model, quality, performance or NPU root-cause acceptance.')
    if not args.execute:print(json.dumps(report));return 0
    need(sys.platform=='linux' and 'torch' not in sys.modules,'Fresh isolated Linux NPU process required')
    canonical_directory(args.source);canonical_directory(args.goldens)
    need(args.output.is_absolute() and args.output.resolve()==args.output and not args.output.exists()
         and not args.output.is_symlink() and args.output.parent.is_dir(),'Fresh canonical output report required')
    need(not args.output.is_relative_to(args.goldens) and not args.output.is_relative_to(args.source),
         'Keep report output outside read-only source and golden directories')
    manifest=manifest_gate(args.goldens,args.manifest)
    before=inventory(args.goldens,manifest);sources=source_inventory(args.source);fp=fingerprint(args.source)
    with args.output.open('xb') as stream:stream.write(b'{}\n')
    report.update(result='incomplete_registered_NPU_MSDA_fixed_goldens',started_UTC=datetime.now(timezone.utc).isoformat(),
                  inputs_before=before,source_before=sources,source_fingerprint_before=fp,
                  original_CUDA_source=manifest.get('source_before'),original_CUDA_precision=manifest.get('precision_before'),
                  CPU_oracle_regenerated=False,RNG_inputs_generated=False,model_acceptance=False)
    persist(args.output,report)
    def cutoff(*_):raise TimeoutError('Own NPU MSDA verifier maximum-seconds reached')
    signal.signal(signal.SIGALRM,cutoff);signal.signal(signal.SIGTERM,cutoff)
    signal.setitimer(signal.ITIMER_REAL,args.maximum_seconds)
    torch=None;public_msda=None;real_cuda=None;precision_before=None;backend_before=None;code=2
    try:
        import torch,torch_npu
        real_cuda=torch.cuda.is_initialized
        need(real_cuda.__module__=='torch.cuda' and not real_cuda(),'Fresh real CUDA state must be uninitialized')
        index=int(args.device.split(':')[1]);need(index<torch.npu.device_count(),'Logical NPU index is not visible')
        torch.npu.set_device(index)
        sys.path.insert(0,str(args.source));sys.path.insert(0,str(args.source/'projects/AlgEngine'))
        from ascend import msda as public_msda
        need(Path(public_msda.__file__).resolve()==args.source/'ascend/msda.py','Public dispatch imported from another source')
        public_msda.configure_backend(args.backend)
        from ascend.compat import enable_training_compat
        need(Path(sys.modules['ascend.compat'].__file__).resolve()==args.source/'ascend/compat.py',
             'Compatibility registration imported from another source')
        if args.backend=='cuda-compatible':
            core=importlib.import_module('ascend.msda_cuda_compatible')
            need(core is not None and Path(core.__file__).resolve()==args.source/'ascend/msda_cuda_compatible.py',
                 'Public numerical core imported from another source')
        enable_training_compat()
        mx=importlib.import_module('mx_driving.ops.multi_scale_deformable_attn').MultiScaleDeformableAttnFunction
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as mmcv
        need(getattr(mx,'_worldengine_reference_precision',None) is True
             and getattr(mmcv,'_worldengine_npu_bridge',None) is True,'Actual MX/MMCV registration not installed')
        need(not real_cuda(),'Registration initialized real CUDA unexpectedly')
        torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        precision_before=precision(torch,torch_npu);policy_gate(precision_before);backend_before=public_msda.backend_policy()
        need(backend_before['backend']==args.backend,'Actual selected MSDA backend differs')
        report.update(precision_before=precision_before,backend_before=backend_before,
            actual_versions=dict(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
                                 mmcv=importlib.import_module('mmcv').__version__),device_name=torch.npu.get_device_name(index),
            real_CUDA_initialized_before=bool(real_cuda()),comparison_function_SHA256=hashlib.sha256(inspect.getsource(compare).encode()).hexdigest(),
            actual_registered_functions={entry:dict(module=cls.__module__,qualified_name=cls.__qualname__,
                forward_function_SHA256=hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest(),
                backward_function_SHA256=hashlib.sha256(inspect.getsource(cls.backward).encode()).hexdigest())
                for entry,cls in (('mx',mx),('mmcv',mmcv))})
        classes=dict(mx=mx,mmcv=mmcv)
        for entry in args.entries:
            for row in manifest['rows']:
                item=dict(entry=entry,case=row['case_key'],spec=row['spec'],comparisons=[]);report['rows'].append(item);persist(args.output,report)
                inputs=load_tensor_file(torch,args.goldens,manifest,row,'inputs.pt','original_CPU_FP32_inputs_GO_and_int64_metadata')
                values,execution=execute_entry(torch,classes[entry],entry,inputs,public_msda,args.device,args.backend);item['execution']=execution
                targets=[('CPU32','original_CPU32_reference.pt','unchanged_original_CPU32_oracle')]+[
                    ('CUDA_repeat_%03d'%i,'cuda_repeat_%03d.pt'%i,'original_CUDA_forward_and_three_gradients') for i in range(4)]
                for target,name,kind in targets:
                    reference=load_tensor_file(torch,args.goldens,manifest,row,name,kind)
                    for role in OUTPUT_ROLES:item['comparisons'].append(dict(target=target,role=role,**compare(torch,values[role],reference[role],role)))
                    del reference;gc.collect();persist(args.output,report)
                item['result']='complete_registered_entry_four_role_comparison'
                del inputs,values;gc.collect();torch.npu.empty_cache();persist(args.output,report)
        comparisons=[c for row in report['rows'] for c in row['comparisons']]
        need(len(report['rows'])==2*len(args.entries) and len(comparisons)==40*len(args.entries),'Incomplete comparison matrix')
        passed=all(c['original_gate_passed'] for c in comparisons)
        report.update(result='complete_registered_NPU_MSDA_fixed_goldens_'+('within_original_gates' if passed else 'outside_original_gates'),
            completed_full_tensor_comparisons=len(comparisons),
            per_entry={entry:{target:dict(comparisons=len(selected),within=sum(c['original_gate_passed'] for c in selected),
                outside=sum(not c['original_gate_passed'] for c in selected),all_original_gates_passed=all(c['original_gate_passed'] for c in selected))
                for target,selected in ((target,[c for row in report['rows'] if row['entry']==entry for c in row['comparisons']
                    if (c['target']=='CPU32')==(target=='CPU32')]) for target in ('CPU32','CUDA'))} for entry in args.entries},
            all_original_gates_passed=passed)
        code=0 if passed else 1
    except BaseException as error:report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            report.update(inputs_after=inventory(args.goldens,manifest),source_after=source_inventory(args.source),source_fingerprint_after=fingerprint(args.source))
            integrity=(report['inputs_after']==before and report['source_after']==sources and report['source_fingerprint_after']==fp
                       and sha(__file__)==report['worker_SHA256'])
            if precision_before is not None:
                report['precision_after']=precision(torch,torch_npu);policy_gate(report['precision_after'])
                report['backend_after']=public_msda.backend_policy();report['real_CUDA_initialized_after']=bool(real_cuda())
                integrity=integrity and precision_before==report['precision_after'] and backend_before==report['backend_after'] and not real_cuda()
            report['all_input_source_precision_and_backend_unchanged']=bool(integrity)
            need(integrity,'Source/input/worker/precision/backend integrity changed')
        except BaseException as error:report.update(final_integrity_error=repr(error),result='incomplete_final_integrity');code=2
        report.update(actual_exit=code,finished_UTC=datetime.now(timezone.utc).isoformat());persist(args.output,report)
    print(json.dumps({k:report.get(k) for k in ('result','actual_exit','completed_full_tensor_comparisons','per_entry','error','final_integrity_error')}))
    return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,OSError) as error:
        print(json.dumps(dict(result='blocked_before_execution',actual_exit=2,error=str(error))),file=sys.stderr)
        raise SystemExit(2)
