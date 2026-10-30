"""Reuse a separately pinned AlgEngine adaptation; never shadow it with CUDA code."""
from contextlib import contextmanager, nullcontext
import hashlib
import os
from pathlib import Path
import re
import sys
from .planner_optimization import options


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def algengine_identity(root):
    root=Path(root).resolve(strict=True)
    files=sorted(list((root/'projects/AlgEngine').rglob('*.py'))+list((root/'ascend').glob('*.py')))
    if not files or not (root/'ascend/compat.py').is_file():
        raise ValueError('NPU execution requires the separately adapted AlgEngine checkout')
    digest=hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return dict(source_sha256=digest.hexdigest(),python_files=len(files))


def assert_module_origins(root,require_plugin=False):
    root=Path(root).resolve()
    expected={'ascend':root/'ascend','mmdet3d_plugin':root/'projects/AlgEngine/mmdet3d_plugin'}
    loaded={}
    for name,module in list(sys.modules.items()):
        package=name.split('.')[0]
        if package not in expected:continue
        boundary=expected[package].resolve(strict=True)
        if not boundary.is_relative_to(root):
            raise RuntimeError('Adaptation package escapes the selected checkout: '+name)
        spec=getattr(module,'__spec__',None)
        filename=getattr(module,'__file__',None)
        module_paths=getattr(module,'__path__',None)
        spec_paths=getattr(spec,'submodule_search_locations',None)
        locations=None
        if module_paths is not None or spec_paths is not None:
            if module_paths is None or spec_paths is None or getattr(spec,'name',None)!=name:
                raise RuntimeError('Adaptation package has inconsistent search-path metadata: '+name)
            try:
                locations=[Path(value).resolve(strict=True) for value in module_paths]
                declared=[Path(value).resolve(strict=True) for value in spec_paths]
            except (OSError,TypeError,ValueError) as error:
                raise RuntimeError('Adaptation package has invalid search paths: '+name) from error
            if (not locations or locations!=declared or len(set(locations))!=len(locations)
                    or any(not path.is_dir() or not path.is_relative_to(boundary) for path in locations)):
                raise RuntimeError('A CUDA/foreign package search path shadows the selected adaptation: '+name)
        if filename is None:
            # PEP 420 namespace packages have no __file__.  Audit both import
            # search-path views instead of treating their missing file as an
            # exemption from the selected checkout's source boundary.
            namespace_dir=expected[package].joinpath(*name.split('.')[1:]).resolve(strict=True)
            if (spec is None or getattr(spec,'origin',None) is not None
                    or getattr(spec,'has_location',True) or locations!=[namespace_dir]):
                raise RuntimeError('Adaptation module has no auditable namespace origin: '+name)
            loaded[name]=dict(kind='namespace_package',search_locations=[str(path) for path in locations])
            continue
        path=Path(filename).resolve(strict=True)
        if not path.is_relative_to(boundary):
            raise RuntimeError('A CUDA/foreign module shadows the selected adaptation: '+name+' '+str(path))
        if locations is not None and locations!=[path.parent]:
            raise RuntimeError('Adaptation package search path differs from its file origin: '+name)
        loaded[name]=str(path)
    if require_plugin and 'mmdet3d_plugin' not in loaded:
        raise RuntimeError('The selected adapted planning plugin was not imported')
    return loaded


def integration_identity(root):
    """Identify this integration separately from its external AlgEngine dependency."""
    root=Path(root)
    paths=list((root/'simengine_ascend').glob('*.py'))
    paths+=list((root/'projects/AlgEngine/closed_loop').glob('*.py'))
    paths+=[root/'projects/SimEngine/worldengine/components/agents/client/navformer_client.py']
    return {path.relative_to(root).as_posix():file_sha(path) for path in sorted(paths)}


class Float32InferenceEvidence:
    """Observe native outputs; never cast, copy or replace a computed Tensor."""
    def __init__(self, model, torch, index):
        self.model, self.torch, self.index = model, torch, index
        self.handles, self.expected, self.observed = {}, set(), {}
        self.before = self.parameters()
        if (set(self.before['dtypes']) != {'torch.float32'}
                or set(self.before['devices']) != {'npu:' + str(index)}):
            raise RuntimeError('FP32 inference requires actual FP32 parameters on the selected NPU')
        try:
            self.install()
        except BaseException:
            self.close()
            raise

    def install(self):
        model, torch = self.model, self.torch
        for name, module in model.named_modules():
            kind = 'Conv2d' if isinstance(module, torch.nn.Conv2d) else 'Linear' if isinstance(module, torch.nn.Linear) else None
            if kind is None:
                continue
            self.expected.add(kind)
            self.handles.setdefault(kind, [])
            def observe(owner, args, output, name=name, kind=kind):
                try:
                    if not torch.is_tensor(output):
                        raise RuntimeError('FP32 activation observer requires an original Tensor output')
                    row = dict(module=name, class_name=type(owner).__module__ + '.' + type(owner).__qualname__,
                               dtype=str(output.dtype), device=str(output.device), shape=list(output.shape),
                               autocast_enabled=bool(torch.is_autocast_enabled('npu')),
                               ambient_autocast_dtype=str(torch.get_autocast_dtype('npu')), training=bool(owner.training))
                    if (row['dtype'] != 'torch.float32' or row['device'] != 'npu:' + str(self.index)
                            or row['autocast_enabled'] or row['training']):
                        raise RuntimeError('Actual FP32 inference activation/device/autocast contract differs')
                    self.observed[kind] = row
                    for handle in self.handles[kind]:
                        handle.remove()
                    self.handles[kind] = []
                except BaseException:
                    self.close()
                    raise
            self.handles[kind].append(module.register_forward_hook(observe))
        if not self.expected:
            self.close()
            raise RuntimeError('No Conv2d/Linear exists to observe actual FP32 inference')

    def parameters(self):
        dtypes, devices = {}, {}
        for value in self.model.parameters():
            if value.is_floating_point():
                dtype, device = str(value.dtype), str(value.device)
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                devices[device] = devices.get(device, 0) + 1
        return dict(dtypes=dtypes, devices=devices)

    def close(self):
        for handles in self.handles.values():
            for handle in handles:
                handle.remove()
            handles.clear()

    def finish(self):
        self.close()
        if self.parameters() != self.before or set(self.observed) != self.expected:
            raise RuntimeError('Actual FP32 inference evidence is incomplete or master parameters changed')
        if self.torch.is_autocast_enabled('npu'):
            raise RuntimeError('FP32 inference escaped into ambient autocast')
        return dict(mode='fp32', master_parameters=self.before, expected_activation_kinds=sorted(self.expected),
                    first_observed_fp32_activations=self.observed, ambient_autocast_enabled=False,
                    observation_only=True, activation_values_copied_to_CPU=False)


class ClosedLoopRuntime:
    def __init__(self,args,sim_root):
        self.args=args;self.sim_root=Path(sim_root).resolve()
        match=re.fullmatch(r'(cuda|npu):(\d+)',args.device)
        if not match:raise ValueError('--device must be cuda:N or npu:N')
        self.backend,self.index=match.group(1),int(match.group(2))
        self.optimization_options=options(args,self.backend)
        self.root=self.sim_root
        self.evidence={}
        self.fp32_evidence=None
        self.provenance=dict(device=args.device,precision=args.precision or 'original_config',backend=self.backend,
                             integration_files=integration_identity(self.sim_root),
                             host_device_mapping={name:os.environ.get(name) for name in
                                                  ('WORLDENGINE_HOST_DEVICE_IDS','ASCEND_RT_VISIBLE_DEVICES')})
        if self.backend=='npu':
            if args.precision not in ('fp32','bf16'):
                raise ValueError('NPU execution requires explicit --precision fp32 or bf16')
            if not args.algengine_root or not args.expected_algengine_source_sha256 or not args.algengine_commit or not args.expected_image_id:
                raise ValueError('NPU requires explicit AlgEngine root, source SHA256, commit and expected image ID')
            if not re.fullmatch('[0-9a-f]{40}',args.algengine_commit) or not re.fullmatch('[0-9a-f]{64}',args.expected_algengine_source_sha256):
                raise ValueError('Malformed AlgEngine commit/source identity')
            if os.environ.get('WORLDENGINE_IMAGE_ID')!=args.expected_image_id:
                raise ValueError('WORLDENGINE_IMAGE_ID must match the image verified by the outer launcher')
            self.root=Path(args.algengine_root).resolve(strict=True)
            identity=algengine_identity(self.root)
            if identity['source_sha256']!=args.expected_algengine_source_sha256:
                raise ValueError('External AlgEngine runtime source fingerprint differs')
            from .planner_optimization import bind_source
            optimization_binding=bind_source(self.root,identity,self.optimization_options[0])
            if optimization_binding is not None:
                self.provenance['inference_optimization_source']=optimization_binding
            # Refuse already-imported local CUDA modules before changing paths.
            assert_module_origins(self.root)
            sys.path[:0]=[str(self.root/'projects/AlgEngine'),str(self.root),str(self.root/'ascend/third_party/navsim')]
            self.provenance.update(AlgEngine_root=str(self.root),AlgEngine_declared_commit=args.algengine_commit,
                                   AlgEngine_runtime=identity,image_id=args.expected_image_id,
                                   image_identity_source='Operator-supplied image identity; container inspection is an external prerequisite')
        os.environ['WORLDENGINE_ROOT']=str(self.root)

    def prepare(self):
        import torch
        self.torch=torch
        if self.backend=='npu':
            import torch_npu
            from torch_npu.contrib import transfer_to_npu  # noqa: F401; explicit NPU process only
            from ascend.msda import configure_backend
            from ascend.compat import enable_training_compat
            from ascend.precision import precision_state
            configure_backend(self.args.msda_backend)
            torch.npu.set_device(self.index)
            torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
            enable_training_compat()
            self.provenance.update(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
                                   precision_before=precision_state(self.args.precision))
        else:
            if self.args.precision=='bf16':
                raise ValueError('GPU BF16 closed-loop execution is not implemented by this NPU adapter')
            torch.cuda.set_device(self.index)
        return self

    def optimization_context(self):
        from .planner_optimization import activate
        return activate(self)

    def validate_optimization(self,predictions):
        mode,chunk=self.optimization_options
        if mode!='none':
            from .planner_optimization import validate_actual
            validate_actual(self.provenance.get('inference_optimization',{}),mode,chunk,predictions)

    def precision_context(self):
        if self.backend!='npu':return nullcontext()
        if self.args.precision=='fp32':return self.fp32_context()
        from ascend.autocast import inference_precision
        return inference_precision(self.args.precision,self.evidence)

    @contextmanager
    def fp32_context(self):
        try:
            yield
        finally:
            if self.fp32_evidence is not None:self.fp32_evidence.close()

    def wrap(self,model):
        if self.backend=='npu':
            from mmcv.device.npu import NPUDataParallel
            model=model.to('npu:'+str(self.index))
            if self.args.precision=='fp32':self.fp32_evidence=Float32InferenceEvidence(model,self.torch,self.index)
            model=NPUDataParallel(model,device_ids=[self.index])
        else:
            from mmcv.parallel import MMDataParallel
            model=MMDataParallel(model.cuda(self.index),device_ids=[self.index])
        model.eval()
        return model

    def finish(self):
        if integration_identity(self.sim_root)!=self.provenance['integration_files']:
            raise RuntimeError('Closed-loop integration source changed during execution')
        if self.backend=='npu':
            from ascend.precision import precision_state
            self.torch.npu.synchronize()
            after=algengine_identity(self.root)
            if after!=self.provenance['AlgEngine_runtime']:
                raise RuntimeError('External AlgEngine source changed while the planner was executing')
            final=precision_state(self.args.precision,stage='after_closed_loop')
            for field in ('mode','public','msda','autocast_policy','ambient_autocast'):
                if self.provenance['precision_before'].get(field)!=final.get(field):
                    raise RuntimeError('Planner precision policy changed during execution: '+field)
            self.provenance.update(module_origins=assert_module_origins(self.root,require_plugin=True),
                                   precision_after=final,
                                   bf16=self.evidence if self.args.precision=='bf16' else None,
                                   fp32=self.fp32_evidence.finish() if self.fp32_evidence is not None else None)
        return self.provenance
