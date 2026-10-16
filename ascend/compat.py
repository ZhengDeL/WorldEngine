"""Scoped compatibility for the pinned DrivingSDK environment."""
import torch
import torch.nn.functional as F
from ascend.msda import reference_forward, reference_backward


_BF16_MSDA_STATS=dict(forward_calls=0,backward_calls=0,input_dtype_combinations={},first_forward=None,first_backward=None)

def bf16_msda_statistics():
    import copy
    return copy.deepcopy(_BF16_MSDA_STATS)

def _reference_precision_forward(ctx,value,shapes,starts,locations,weights,allow_legacy_map_layout=False):
    original_dtypes=(value.dtype,locations.dtype,weights.dtype)
    promoted=any(dtype==torch.bfloat16 for dtype in original_dtypes)
    if promoted:
        if any(dtype not in (torch.float32,torch.bfloat16)for dtype in original_dtypes):
            raise ValueError('Native BF16 MSDA island supports only FP32/BF16 floating inputs')
        legacy=(allow_legacy_map_layout and shapes.shape[0]==1 and locations.ndim==6 and
                locations.shape[3]==4 and locations.shape[2]==8 and locations.shape[4]==4 and value.shape[3]==32)
        if legacy and not all(x.is_contiguous()for x in (value,locations,weights)):
            raise ValueError('Original map flat-prefix layout requires contiguous raw floating inputs before BF16 promotion')
        ctx._worldengine_msda_input_dtypes=original_dtypes
        value,locations,weights=value.float(),locations.float(),weights.float()
        _BF16_MSDA_STATS['forward_calls']+=1
        key=','.join(map(str,original_dtypes));counts=_BF16_MSDA_STATS['input_dtype_combinations']
        counts[key]=counts.get(key,0)+1
    if not promoted:
        return reference_forward(ctx,value,shapes,starts,locations,weights,allow_legacy_map_layout=allow_legacy_map_layout)
    with torch.autocast(device_type='npu',enabled=False):
        output=reference_forward(ctx,value,shapes,starts,locations,weights,allow_legacy_map_layout=allow_legacy_map_layout)
        if _BF16_MSDA_STATS['first_forward'] is None:
            _BF16_MSDA_STATS['first_forward']=dict(original_input_dtypes=list(map(str,original_dtypes)),
                compute_input_dtypes=[str(x.dtype)for x in(value,locations,weights)],
                compute_input_shapes=[list(x.shape)for x in(value,locations,weights)],output_dtype=str(output.dtype),
                autocast_enabled_inside=bool(torch.is_autocast_enabled('npu')))
        return output

def _reference_precision_backward(ctx,grad_output):
    dtypes=getattr(ctx,'_worldengine_msda_input_dtypes',None)
    if not dtypes:return reference_backward(ctx,grad_output)
    with torch.autocast(device_type='npu',enabled=False):
        gradients=reference_backward(ctx,grad_output.float())
    if dtypes:
        result=list(gradients)
        for index,dtype in zip((0,3,4),dtypes):
            if result[index] is not None:result[index]=result[index].to(dtype=dtype)
        _BF16_MSDA_STATS['backward_calls']+=1
        if _BF16_MSDA_STATS['first_backward'] is None:
            _BF16_MSDA_STATS['first_backward']=dict(original_input_dtypes=list(map(str,dtypes)),
                FP32_computed_gradient_dtypes=[str(gradients[i].dtype)if gradients[i] is not None else None for i in(0,3,4)],
                returned_gradient_dtypes=[str(result[i].dtype)if result[i] is not None else None for i in(0,3,4)],
                needs_input_grad=list(ctx.needs_input_grad))
        return tuple(result)
    return gradients


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
            if value.device.type=='npu' and value.dtype in (torch.float32,torch.bfloat16):
                return _reference_precision_forward(ctx,value,shapes,starts,locations,weights)
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
                return _reference_precision_backward(ctx,grad)
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
            if value.dtype in (torch.float32,torch.bfloat16):
                return _reference_precision_forward(ctx,value,shapes,starts,locations,weights,
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
                return (*_reference_precision_backward(ctx,grad_output),None)
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
