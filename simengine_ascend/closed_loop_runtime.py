"""Reuse a separately pinned AlgEngine adaptation; never shadow it with CUDA code."""
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import re
import sys


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
        filename=getattr(module,'__file__',None)
        if filename is None:
            raise RuntimeError('Adaptation module has no auditable file origin: '+name)
        path=Path(filename).resolve()
        if not path.is_relative_to(expected[package]):
            raise RuntimeError('A CUDA/foreign module shadows the selected adaptation: '+name+' '+str(path))
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


class ClosedLoopRuntime:
    def __init__(self,args,sim_root):
        self.args=args;self.sim_root=Path(sim_root).resolve()
        match=re.fullmatch(r'(cuda|npu):(\d+)',args.device)
        if not match:raise ValueError('--device must be cuda:N or npu:N')
        self.backend,self.index=match.group(1),int(match.group(2))
        self.root=self.sim_root
        self.evidence={}
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

    def precision_context(self):
        if self.backend!='npu':return nullcontext()
        from ascend.autocast import inference_precision
        return inference_precision(self.args.precision,self.evidence)

    def wrap(self,model):
        if self.backend=='npu':
            from mmcv.device.npu import NPUDataParallel
            model=NPUDataParallel(model.to('npu:'+str(self.index)),device_ids=[self.index])
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
                                   bf16=self.evidence if self.args.precision=='bf16' else None)
        return self.provenance
