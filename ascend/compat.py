"""Scoped compatibility for the pinned DrivingSDK environment."""
import torch
import torch.nn.functional as F
from ascend.msda import reference_forward, reference_backward


def enable_training_compat():
    import mx_driving
    import mx_driving.fused
    # DrivingSDK 26.1.0.post2's fused MaxPool2d autograd function only supplies
    # forward. Its ResNet integration fails when the backbone is unfrozen.
    # Use PyTorch's NPU-dispatched max_pool2d, which supplies its backward.
    def training_max_pool2d(x,kernel_size,stride,padding):
        return F.max_pool2d(x,kernel_size,stride,padding)
    mx_driving.fused.npu_max_pool2d=training_max_pool2d
    import importlib,json,os
    from pathlib import Path
    from torch.autograd.function import once_differentiable
    module=importlib.import_module('mx_driving.ops.multi_scale_deformable_attn')
    cls=module.MultiScaleDeformableAttnFunction
    if not getattr(cls,'_worldengine_reference_precision',False):
        original_mx_forward=cls.forward
        def npu_reference_forward(ctx,value,shapes,starts,locations,weights,**kwargs):
            if value.device.type=='npu' and value.dtype==torch.float32:
                return reference_forward(ctx,value,shapes,starts,locations,weights)
            return original_mx_forward(ctx,value,shapes,starts,locations,weights,**kwargs)
        cls.forward=staticmethod(npu_reference_forward)
        cls._worldengine_reference_precision=True
    if not getattr(cls,'_worldengine_contiguous_backward',False):
        original=cls.backward
        @once_differentiable
        def contiguous_backward(ctx,grad_output):
            if os.environ.get('WORLDENGINE_DEBUG_AUTOGRAD')=='1':
                row=dict(shape=list(grad_output.shape),stride=list(grad_output.stride()),
                         contiguous=grad_output.is_contiguous(),storage_offset=grad_output.storage_offset())
                with (Path('/output')/'msda_backward_layout.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            grad=grad_output.contiguous()
            if grad.storage_offset():grad=grad.clone()
            if getattr(ctx,'_worldengine_reference_msda',False):
                return reference_backward(ctx,grad)
            return original(ctx,grad)
        cls.backward=staticmethod(contiguous_backward)
        cls._worldengine_contiguous_backward=True
    # The panoptic map decoder uses MMCV's separate MSDA autograd class. Its
    # CUDA-style extension path is not one of AlgEngine's explicit NPU branches.
    mmcv_module=importlib.import_module('mmcv.ops.multi_scale_deform_attn')
    mmcv_cls=mmcv_module.MultiScaleDeformableAttnFunction
    if not getattr(mmcv_cls,'_worldengine_npu_bridge',False):
        original_mmcv_forward,original_mmcv_backward=mmcv_cls.forward,mmcv_cls.backward
        def npu_forward(ctx,value,shapes,starts,locations,weights,im2col_step):
            if value.device.type!='npu':return original_mmcv_forward(ctx,value,shapes,starts,locations,weights,im2col_step)
            if value.dtype==torch.float32:
                return reference_forward(ctx,value,shapes,starts,locations,weights,
                                         allow_legacy_map_layout=True)
            ctx._worldengine_npu=True
            shapes,starts=shapes.int().contiguous(),starts.int().contiguous()
            locations,weights=locations.type_as(value).contiguous(),weights.type_as(value).contiguous()
            value=value.contiguous()
            ctx.save_for_backward(value,shapes,starts,locations,weights)
            return mx_driving._C.multi_scale_deformable_attn(value,shapes,starts,locations,weights)
        @once_differentiable
        def npu_backward(ctx,grad_output):
            if getattr(ctx,'_worldengine_reference_msda',False):
                return (*reference_backward(ctx,grad_output),None)
            if not getattr(ctx,'_worldengine_npu',False):return original_mmcv_backward(ctx,grad_output)
            value,shapes,starts,locations,weights=ctx.saved_tensors
            grad=grad_output.contiguous()
            gv,gl,gw=mx_driving._C.multi_scale_deformable_attn_backward(value,shapes,starts,locations,weights,grad)
            if os.environ.get('WORLDENGINE_DEBUG_AUTOGRAD')=='1':
                with (Path('/output')/'mmcv_msda_bridge.jsonl').open('a') as file:
                    file.write(json.dumps(dict(value_shape=list(value.shape),grad_shape=list(grad.shape),shape_dtype=str(shapes.dtype),start_dtype=str(starts.dtype)))+'\n')
            return gv,None,None,gl,gw,None
        mmcv_cls.forward=staticmethod(npu_forward);mmcv_cls.backward=staticmethod(npu_backward)
        mmcv_cls._worldengine_npu_bridge=True
