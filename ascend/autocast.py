"""Native NPU BF16 forward/loss autocast with FP32 master training state.

The original optimizer hook owns backward, clipping and step outside this
context. No model.half(), legacy MMCV fp16 hook or gradient scaler is used.
MSDA is an explicit FP32 island in ascend.compat. Mode support still requires
real operator, model, resume and inference validation on the selected image.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import importlib
import inspect
import json
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import HOOKS, Hook

_ACTIVE_TRAINING_HOOKS=[]


@contextmanager
def training_precision_cleanup(mode):
    """Release metadata hooks even when an original optimizer/backward fails."""
    try:yield
    except BaseException:
        if mode=='bf16':
            for hook in list(_ACTIVE_TRAINING_HOOKS):hook.abort()
        raise
    else:
        if mode=='bf16':
            for hook in list(_ACTIVE_TRAINING_HOOKS):hook.close()


def autocast_state():
    return dict(enabled=bool(torch.is_autocast_enabled('npu')),
                dtype=str(torch.get_autocast_dtype('npu')))


def precision_context(mode):
    # History no_grad and current train forwards share this scope. Cached
    # low-precision leaf casts created under no_grad can sever current weight
    # gradients, so native BF16 explicitly disables weight-cast caching.
    return torch.autocast(device_type='npu',dtype=torch.bfloat16,cache_enabled=False) if mode=='bf16' else nullcontext()


def floating_dtypes(values):
    result={}
    for value in values:
        if torch.is_tensor(value) and value.is_floating_point():
            name=str(value.dtype);result[name]=result.get(name,0)+1
    return result


def optimizer_tensors(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from optimizer_tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from optimizer_tensors(item)


def parameter_training_coverage(model,optimizer):
    """Host metadata only; unused parameters remain explicit, not invented grads.

    Reading .grad presence, dtype, shape and optimizer dictionary membership
    adds no tensor value copy, reduction, collective or backward hook. The
    original model uses find_unused_parameters, so missing gradients are
    reported by name and must be judged against its actual FP32 counterpart.
    """
    parameters=list(model.named_parameters())
    names={id(parameter):name for name,parameter in parameters}
    groups={key:[] for key in names}
    group_rows=[]
    for index,group in enumerate(optimizer.param_groups):
        group_rows.append(dict(index=index,parameter_count=len(group['params'])))
        for parameter in group['params']:
            if id(parameter) not in names:
                raise RuntimeError('Optimizer parameter has no original model name')
            groups[id(parameter)].append(index)
    if any(id(parameter) not in names for parameter in optimizer.state):
        raise RuntimeError('Optimizer state has no original model parameter')
    rows=[]
    for name,parameter in parameters:
        gradient=parameter.grad
        state=optimizer.state.get(parameter,{})
        rows.append(dict(name=name,shape=list(parameter.shape),requires_grad=bool(parameter.requires_grad),
                         parameter_dtype=str(parameter.dtype),optimizer_groups=groups[id(parameter)],
                         gradient_present=gradient is not None,
                         gradient_dtype=str(gradient.dtype)if gradient is not None else None,
                         optimizer_state_present=bool(state),optimizer_state_keys=sorted(str(key)for key in state),
                         optimizer_state_floating_dtypes=floating_dtypes(optimizer_tensors(state))))
    trainable=[row for row in rows if row['requires_grad']]
    return dict(model_parameter_count=len(rows),trainable_parameter_count=len(trainable),
                frozen_parameter_count=len(rows)-len(trainable),
                optimizer_group_parameter_count=sum(row['parameter_count']for row in group_rows),
                gradient_present_count=sum(row['gradient_present']for row in rows),
                trainable_gradient_missing_names=[row['name']for row in trainable if not row['gradient_present']],
                optimizer_state_present_count=sum(row['optimizer_state_present']for row in rows),
                trainable_optimizer_state_missing_names=[row['name']for row in trainable if not row['optimizer_state_present']],
                optimizer_groups=group_rows,parameters=rows,
                tensor_values_copied_or_reduced=False,
                missing_gradient_is_not_automatically_failure=True,
                acceptance='Compare full named coverage with the same original FP32 task; nonempty FP32 dtypes alone are insufficient.')


def loss_fp32(value):
    """Preserve the original loss formula/graph and non-floating metadata."""
    if torch.is_tensor(value):return value.float() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:loss_fp32(v)for k,v in value.items()}
    if isinstance(value,list):return [loss_fp32(v)for v in value]
    if isinstance(value,tuple):return tuple(loss_fp32(v)for v in value)
    return value


def loss_tensor_metadata(value,path='input'):
    if torch.is_tensor(value):
        return {path:dict(dtype=str(value.dtype),shape=list(value.shape))}
    rows={}
    if isinstance(value,dict):
        for key,item in value.items():rows.update(loss_tensor_metadata(item,path+'.'+str(key)))
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):rows.update(loss_tensor_metadata(item,path+'['+str(index)+']'))
    return rows


class BFloat16AddRelu:
    """BF16-only public dispatch; retain the SDK's supported dtype paths.

    The installed SDK's small-tensor branch already uses relu(x + y). Its
    large fused operator does not accept BF16. The BF16 branch uses that same
    Torch formula with normal dtype promotion and autograd, out of place.
    ResNet immediately rebinds the output and does not reuse the old alias.
    """
    def __init__(self):
        operation=importlib.import_module('mx_driving.ops.npu_add_relu')
        fused=importlib.import_module('mx_driving.fused')
        package=importlib.import_module('mx_driving')
        self.original=operation.npu_add_relu
        self.owners=(operation,fused,package)
        self.calls=0;self.original_calls=0;self.first_actual=None;self.closed=False
        if not callable(self.original) or any(getattr(owner,'npu_add_relu',None)is not self.original for owner in self.owners):
            raise RuntimeError('Unexpected installed SDK AddRelu public alias')
        def add_relu(x,y):
            if not any(torch.is_tensor(value)and value.dtype==torch.bfloat16 for value in(x,y)):
                self.original_calls+=1
                return self.original(x,y)
            if not torch.is_tensor(x)or not torch.is_tensor(y):
                raise TypeError('Original AddRelu requires two Tensor inputs')
            with torch.autocast(device_type='npu',enabled=False):
                result=torch.nn.functional.relu(x+y)
                sample=dict(input_dtypes=[str(x.dtype),str(y.dtype)],output_dtype=str(result.dtype),
                            input_shapes=[list(x.shape),list(y.shape)],output_shape=list(result.shape),
                            autocast_inside=autocast_state(),formula='torch.nn.functional.relu(x + y)',
                            dtype_promotion_preserved=True,input_modified_in_place=False,
                            original_custom_Function_forward_replaced=False)
            self.calls+=1
            if self.first_actual is None:self.first_actual=sample
            return result
        self.wrapper=add_relu
        for owner in self.owners:owner.npu_add_relu=add_relu

    def close(self):
        if self.closed:return
        if any(getattr(owner,'npu_add_relu',None)is not self.wrapper for owner in self.owners):
            raise RuntimeError('SDK AddRelu public alias changed during BF16 execution')
        for owner in self.owners:owner.npu_add_relu=self.original
        self.closed=True

    def report(self):
        return dict(BF16_calls=self.calls,supported_dtype_original_SDK_calls=self.original_calls,
                    first_actual=self.first_actual,public_aliases=['mx_driving.ops.npu_add_relu.npu_add_relu',
                        'mx_driving.fused.npu_add_relu','mx_driving.npu_add_relu'],
                    original_FP32_FP16_callable_preserved=True,
                    BF16_formula_uses_original_SDK_small_tensor_Torch_branch=True,
                    BF16_out_of_place_output=True)


class BFloat16Evidence:
    """Read dtype/shape metadata only; never copy activation values to CPU."""
    def __init__(self,model,inference=False):
        self.model=model;self.activations={};self.first_observed={};self.handles={'Conv2d':[],'Linear':[]};self.expected=set()
        self.train_step_ordinal=None
        self.loss_methods=[];self.loss_calls={};self.first_loss={}
        self.geometry_methods=[];self.geometry_calls={};self.first_geometry={}
        self.geometry_encoder_found=False
        self.bev_query_methods=[];self.bev_query_calls={};self.first_bev_query={}
        self.export_methods=[];self.export_calls={};self.first_export={}
        self.normalization_methods=[];self.normalization_calls={};self.first_normalization={}
        self.normalization_training_calls={};self.first_training_normalization={}
        self.add_relu=None
        self.parameters_before=floating_dtypes(model.parameters())
        if not self.parameters_before or set(self.parameters_before)!={'torch.float32'}:
            raise RuntimeError('BF16 autocast requires existing FP32 master parameters')
        if any(getattr(m,'fp16_enabled',False) for m in model.modules()):
            raise RuntimeError('Disable legacy MMCV fp16 model casting before native BF16 autocast')
        try:self.install_model_hooks(inference)
        except BaseException:
            self.close();raise

    def install_model_hooks(self,inference):
        for name,module in self.model.named_modules():
            if isinstance(module,_BatchNorm):
                original_normalization=module.forward
                self.normalization_methods.append((module,original_normalization));self.normalization_calls[name]=0
                self.normalization_training_calls[name]=0
                def normalization(input,*args,_original=original_normalization,_name=name,_module=module,**kwargs):
                    if not torch.is_tensor(input)or not input.is_floating_point():
                        raise TypeError('Original BatchNorm requires its floating Tensor input')
                    dtype=input.dtype;training=_module.training
                    process_group=getattr(_module,'process_group',None)
                    is_sync=isinstance(_module,torch.nn.SyncBatchNorm)
                    world_size=None
                    if is_sync and torch.distributed.is_available()and torch.distributed.is_initialized():
                        world_size=torch.distributed.get_world_size(process_group)
                    with torch.autocast(device_type='npu',enabled=False):
                        converted=input.float()
                        # Delegate the same module exactly once. SyncBatchNorm
                        # retains its original process group and collectives;
                        # running statistics and FP32 affine parameters remain
                        # owned and updated by that original implementation.
                        result=_original(converted,*args,**kwargs)
                        if not torch.is_tensor(result)or result.dtype!=torch.float32:
                            raise RuntimeError('Original normalization must return its FP32 island result')
                        output=result.to(dtype=dtype)
                        sample=dict(module=_name,class_name=type(_module).__module__+'.'+type(_module).__qualname__,
                                    original_callable=_original.__module__+'.'+_original.__qualname__,
                                    input_dtype=str(dtype),input_shape=list(input.shape),
                                    island_input_dtype=str(converted.dtype),island_output_dtype=str(result.dtype),
                                    output_dtype=str(output.dtype),output_shape=list(output.shape),
                                    autocast_inside=autocast_state(),original_training_flag=training,
                                    training_flag_unchanged=_module.training==training,
                                    SyncBatchNorm=is_sync,SyncBatchNorm_world_size=world_size,
                                    process_group_object_unchanged=getattr(_module,'process_group',None)is process_group,
                                    original_module_forward_called_once=True,original_collectives_preserved=True,
                                    affine_parameter_dtypes=floating_dtypes(_module.parameters(recurse=False)),
                                    running_buffer_floating_dtypes=floating_dtypes(_module.buffers(recurse=False)),
                                    train_step_ordinal=self.train_step_ordinal)
                    if not sample['training_flag_unchanged']or not sample['process_group_object_unchanged']:
                        raise RuntimeError('Original normalization training state or process group changed')
                    self.normalization_calls[_name]+=1;self.first_normalization.setdefault(_name,sample)
                    if training:
                        self.normalization_training_calls[_name]+=1
                        self.first_training_normalization.setdefault(_name,sample)
                    return output
                module.forward=normalization
            if any((base.__module__,base.__name__)==('mmdet3d_plugin.uniad.modules.transformer','PerceptionTransformer')for base in type(module).__mro__):
                original_bev=module.get_bev_features;bev_signature=inspect.signature(original_bev)
                if original_bev.__module__!='mmdet3d_plugin.uniad.modules.transformer'or'bev_queries'not in bev_signature.parameters:
                    raise RuntimeError('Unexpected original BEV query/geometry constructor')
                self.bev_query_methods.append((module,original_bev));self.bev_query_calls[name]=0
                def bev_features(*args,_original=original_bev,_signature=bev_signature,_name=name,**kwargs):
                    bound=_signature.bind(*args,**kwargs);query=bound.arguments['bev_queries']
                    if not torch.is_tensor(query):raise RuntimeError('Original BEV query Tensor required')
                    bound.arguments['bev_queries']=query.float()
                    inside=autocast_state()
                    # The original function creates shift/can_bus with
                    # bev_queries.new_tensor. Preserve their FP32 construction
                    # while every original eligible layer retains BF16 autocast.
                    result=_original(*bound.args,**bound.kwargs)
                    self.bev_query_calls[_name]+=1
                    self.first_bev_query.setdefault(_name,dict(original_dtype=str(query.dtype),
                        geometry_constructor_query_dtype=str(bound.arguments['bev_queries'].dtype),
                        original_autocast_state=inside,after_original=autocast_state(),
                        original_formula_called_once=True,feature_inputs_not_cast=True,
                        train_step_ordinal=self.train_step_ordinal))
                    return result
                module.get_bev_features=bev_features
            # These two original encoder classes explicitly require FP32 camera
            # projection. Legacy force_fp32 skips its body when fp16_enabled is
            # False, so it does not protect native BF16 autocast. Reference-grid
            # creation must also be FP32 before projection, not cast afterwards.
            encoders={('mmdet3d_plugin.navformer.modules.encoder','NAVFormerEncoder'),
                      ('mmdet3d_plugin.uniad.modules.encoder','BEVFormerEncoder')}
            if any((base.__module__,base.__name__)in encoders for base in type(module).__mro__):
                self.geometry_encoder_found=True
                for method in ('get_reference_points','point_sampling','point_sampling_distortion'):
                    original_geometry=getattr(module,method,None)
                    if not callable(original_geometry):continue
                    if original_geometry.__module__ not in {item[0]for item in encoders}:
                        raise RuntimeError('Unexpected override of the original FP32 geometry method')
                    key=name+'.'+method;signature=inspect.signature(original_geometry)
                    self.geometry_methods.append((module,method,original_geometry));self.geometry_calls[key]=0
                    def geometry(*args,_original=original_geometry,_key=key,_method=method,_signature=signature,**kwargs):
                        before=loss_tensor_metadata((args,kwargs))
                        with torch.autocast(device_type='npu',enabled=False):
                            converted_args=loss_fp32(args);converted_kwargs=loss_fp32(kwargs)
                            if _method=='get_reference_points':
                                bound=_signature.bind(*converted_args,**converted_kwargs)
                                bound.arguments['dtype']=torch.float32
                                converted_args,converted_kwargs=bound.args,bound.kwargs
                            inside=autocast_state()
                            result=_original(*converted_args,**converted_kwargs)
                            sample=dict(method=_method,original_callable=_original.__module__+'.'+_original.__qualname__,
                                autocast_inside=inside,input_tensors_before=before,
                                input_tensors_inside=loss_tensor_metadata((converted_args,converted_kwargs)),
                                output_tensors=loss_tensor_metadata(result),
                                output_floating_dtypes=floating_dtypes(optimizer_tensors(result)),
                                reference_grid_dtype_forced_FP32=_method=='get_reference_points',
                                original_formula_called_once=True,train_step_ordinal=self.train_step_ordinal)
                        self.geometry_calls[_key]+=1;self.first_geometry.setdefault(_key,sample)
                        return result
                    setattr(module,method,geometry)
            if (name=='planning_head' or name.endswith('.planning_head')) and callable(getattr(module,'loss',None)):
                original_loss=module.loss
                self.loss_methods.append((module,original_loss))
                self.loss_calls[name]=0
                def planning_loss(*args,_original=original_loss,_name=name,**kwargs):
                    before=floating_dtypes(optimizer_tensors((args,kwargs)))
                    with torch.autocast(device_type='npu',enabled=False):
                        converted_args=loss_fp32(args);converted_kwargs=loss_fp32(kwargs)
                        inside=autocast_state()
                        result=_original(*converted_args,**converted_kwargs)
                        sample=dict(module=_name,input_dtypes_before=before,
                                    input_dtypes_inside=floating_dtypes(optimizer_tensors((converted_args,converted_kwargs))),
                                    actual_input_tensors_before=loss_tensor_metadata((args,kwargs)),
                                    actual_input_tensors_inside=loss_tensor_metadata((converted_args,converted_kwargs)),
                                    output_dtypes=floating_dtypes(optimizer_tensors(result)),
                                    autocast_inside=inside,train_step_ordinal=self.train_step_ordinal,
                                    original_loss_formula_used=True,loss_replaced_with_logits=False)
                    self.loss_calls[_name]+=1;self.first_loss.setdefault(_name,sample)
                    return result
                module.loss=planning_loss
            if inference and (name=='planning_head'or name.endswith('.planning_head')):
                original_head_forward=module.forward
                self.export_methods.append((module,original_head_forward));self.export_calls[name]=0
                def export_forward(*args,_original=original_head_forward,_name=name,**kwargs):
                    result=_original(*args,**kwargs)
                    trajectory=result.get('trajectory')if isinstance(result,dict)else None
                    if not torch.is_tensor(trajectory):raise RuntimeError('Original planning inference must return its trajectory Tensor')
                    original_dtype=str(trajectory.dtype);converted=trajectory.dtype==torch.bfloat16
                    if converted:
                        # A serialization boundary only: all represented BF16
                        # values are exactly representable in NumPy FP32. Do not
                        # recompute poses, change selections or round to FP16.
                        result=dict(result);result['trajectory']=trajectory.float()
                    sample=dict(original_dtype=original_dtype,export_dtype=str(result['trajectory'].dtype),
                                shape=list(trajectory.shape),BF16_values_losslessly_promoted_to_FP32=converted,
                                trajectory_values_recomputed=False,CPU_copy_added=False)
                    self.export_calls[_name]+=1;self.first_export.setdefault(_name,sample)
                    return result
                module.forward=export_forward
            kind='Conv2d'if isinstance(module,torch.nn.Conv2d)else'Linear'if isinstance(module,torch.nn.Linear)else None
            if kind:
                self.expected.add(kind)
                def observed(owner,args,output,name=name,kind=kind):
                    if torch.is_tensor(output):
                        sample=dict(module=name,class_name=type(owner).__module__+'.'+type(owner).__qualname__,
                                    dtype=str(output.dtype),shape=list(output.shape),autocast=autocast_state(),
                                    train_step_ordinal=self.train_step_ordinal)
                        self.first_observed.setdefault(kind,sample)
                        # An earlier legitimate FP32 island is evidence of its
                        # own execution, not proof that later autocast was unused.
                        if sample['dtype']=='torch.bfloat16' and sample['autocast']=={'enabled':True,'dtype':'torch.bfloat16'}:
                            self.activations[kind]=sample
                            for handle in self.handles[kind]:handle.remove()
                            self.handles[kind]=[]
                self.handles[kind].append(module.register_forward_hook(observed))
        if not self.expected:raise RuntimeError('No Linear/Conv2d activation available to verify actual BF16 execution')
        self.add_relu=BFloat16AddRelu()

    def close(self):
        for handles in self.handles.values():
            for handle in handles:handle.remove()
            handles.clear()
        for module,original in self.loss_methods:module.loss=original
        for module,method,original in self.geometry_methods:setattr(module,method,original)
        for module,original in self.bev_query_methods:module.get_bev_features=original
        for module,original in self.export_methods:module.forward=original
        for module,original in self.normalization_methods:module.forward=original
        if self.add_relu is not None:self.add_relu.close()

    def finish(self,optimizer=None):
        self.close()
        parameters=floating_dtypes(self.model.parameters())
        if parameters!=self.parameters_before or set(self.activations)!=self.expected or any(x['dtype']!='torch.bfloat16'or x['autocast']!={'enabled':True,'dtype':'torch.bfloat16'}for x in self.activations.values()):
            raise RuntimeError('BF16 mode did not retain FP32 parameters and execute an actual BF16 activation')
        states=floating_dtypes(optimizer_tensors(optimizer.state))if optimizer is not None else None
        gradients=floating_dtypes(p.grad for p in self.model.parameters()if p.grad is not None)
        if optimizer is not None and (not states or set(states)!={'torch.float32'} or not gradients or set(gradients)!={'torch.float32'}):
            raise RuntimeError('BF16 mode requires FP32 gradients and Adam/master optimizer state')
        if optimizer is not None and any(not self.loss_calls[name] or self.first_loss[name]['autocast_inside']['enabled'] or set(self.first_loss[name]['input_dtypes_inside'])!={'torch.float32'} or set(self.first_loss[name]['output_dtypes'])!={'torch.float32'}for name in self.loss_calls):
            raise RuntimeError('Actual original planning loss must execute its FP32 island')
        active_geometry={name:self.first_geometry[name]for name,count in self.geometry_calls.items()if count}
        if self.geometry_encoder_found and (not any(row['method'].startswith('point_sampling')for row in active_geometry.values())or
                not any(row['method']=='get_reference_points'for row in active_geometry.values())or
                any(row['autocast_inside']['enabled']or set(row['output_floating_dtypes'])!={'torch.float32'}for row in active_geometry.values())):
            raise RuntimeError('Original reference-grid and camera geometry must execute their FP32 islands')
        if self.bev_query_methods and (not any(self.bev_query_calls.values())or any(row['geometry_constructor_query_dtype']!='torch.float32'or row['after_original']!=row['original_autocast_state']for row in self.first_bev_query.values())):
            raise RuntimeError('Original BEV shift geometry must be constructed in FP32 without changing layer autocast')
        if self.normalization_methods and (not any(self.normalization_calls.values())or any(
                row['autocast_inside']['enabled']or row['island_input_dtype']!='torch.float32'or
                row['island_output_dtype']!='torch.float32'or row['input_dtype']!=row['output_dtype']or
                not row['training_flag_unchanged']or not row['process_group_object_unchanged']or
                set(row['affine_parameter_dtypes'])-{'torch.float32'}or
                set(row['running_buffer_floating_dtypes'])-{'torch.float32'}for row in self.first_normalization.values())):
            raise RuntimeError('Original BatchNorm/SyncBatchNorm must retain its FP32 computation/state and original output dtype')
        from ascend.compat import bf16_msda_statistics
        return dict(mode='bf16',implementation='native_NPU_autocast_forward_and_loss_only',
                    master_parameter_dtypes=parameters,gradient_dtypes=gradients,optimizer_state_dtypes=states,
                    parameter_training_coverage=parameter_training_coverage(self.model,optimizer)if optimizer is not None else None,
                    first_observed_activations=self.first_observed,
                    first_observed_bf16_activations=self.activations,MSDA=bf16_msda_statistics(),
                    planning_loss_FP32=dict(calls=self.loss_calls,first_actual=self.first_loss,
                                           original_formula_unchanged=True,loss_replaced_with_logits=False),
                    geometry_FP32=dict(calls=self.geometry_calls,first_actual=self.first_geometry,
                                       original_formulas_unchanged=True,reference_grid_created_in_FP32=True,
                                       BEV_query_calls=self.bev_query_calls,first_BEV_query=self.first_bev_query),
                    planning_inference_export=dict(calls=self.export_calls,first_actual=self.first_export,
                                                   BF16_to_FP32_for_NumPy_only=True,trajectory_formula_unchanged=True),
                    AddRelu_BF16=self.add_relu.report(),
                    normalization_FP32=dict(calls=self.normalization_calls,first_actual=self.first_normalization,
                                            training_calls=self.normalization_training_calls,
                                            first_actual_training=self.first_training_normalization,
                                            original_modules_statistics_and_collectives_preserved=True,
                                            output_dtype_restored_to_original_input=True),
                    loss_scaler_used=False,legacy_fp16_model_cast_used=False,
                    activation_values_copied_to_CPU=False)


@HOOKS.register_module()
class AscendBFloat16Hook(Hook):
    """LOW priority: forward context closes before original optimizer hook."""
    def __init__(self,out_dir=None):
        # Original custom_train_detector injects out_dir into every hook cfg.
        # Accept its existing work directory without changing loss or AMP.
        super().__init__()
        self.out_dir=out_dir

    def close(self):
        self.model.train_step=self.original
        self.model.forward=self.original_forward
        if self.original_val_step is not None:self.model.val_step=self.original_val_step
        self.evidence.close()
        if self in _ACTIVE_TRAINING_HOOKS:_ACTIVE_TRAINING_HOOKS.remove(self)

    def abort(self):
        # Cleanup must not replace an exception from the original model.
        try:self.close()
        except Exception:pass

    def before_run(self,runner):
        if any('fp16'in type(h).__name__.lower()for h in runner.hooks):
            raise RuntimeError('Native BF16 cannot be combined with MMCV fp16 optimizer hooks')
        self.model=runner.model;self.original=self.model.train_step
        self.original_forward=self.model.forward
        self.original_val_step=getattr(self.model,'val_step',None)
        self.evidence=BFloat16Evidence(self.model.module)
        self.calls=0;self.forward_calls=0;self.val_calls=0;self.first_inside=None;self.last_inside=None;self.last_loss_dtype=None
        self.first_inside_cache_enabled=None;self.last_inside_cache_enabled=None
        def forward(*args,**kwargs):
            try:
                with precision_context('bf16'):result=self.original_forward(*args,**kwargs)
            except BaseException:
                self.abort();raise
            self.forward_calls+=1
            return result
        self.model.forward=forward
        if self.original_val_step is not None:
            def val_step(*args,**kwargs):
                try:
                    with precision_context('bf16'):result=self.original_val_step(*args,**kwargs)
                except BaseException:
                    self.abort();raise
                self.val_calls+=1
                return result
            self.model.val_step=val_step
        def train_step(*args,**kwargs):
            try:
                before=autocast_state();before_cache_enabled=bool(torch.is_autocast_cache_enabled())
                if before['enabled']:raise RuntimeError('BF16 forward-only context unexpectedly nested in ambient autocast')
                self.evidence.train_step_ordinal=self.calls+1
                with precision_context('bf16'):
                    inside=autocast_state()
                    if inside!={'enabled':True,'dtype':'torch.bfloat16'}:raise RuntimeError('Actual NPU BF16 autocast is unavailable')
                    inside_cache_enabled=bool(torch.is_autocast_cache_enabled())
                    if inside_cache_enabled:raise RuntimeError('BF16 history/current weight-cast cache must be disabled')
                    result=self.original(*args,**kwargs)
                    if self.first_inside is None:self.first_inside=inside
                    self.last_inside=inside
                    if self.first_inside_cache_enabled is None:self.first_inside_cache_enabled=inside_cache_enabled
                    self.last_inside_cache_enabled=inside_cache_enabled
                    loss=result.get('loss')if isinstance(result,dict)else None
                    self.last_loss_dtype=str(loss.dtype)if torch.is_tensor(loss)else None
                if autocast_state()!=before or bool(torch.is_autocast_cache_enabled())!=before_cache_enabled:raise RuntimeError('BF16 forward/cache context was not restored before original backward')
            except BaseException:
                self.abort();raise
            self.calls+=1
            return result
        self.model.train_step=train_step
        _ACTIVE_TRAINING_HOOKS.append(self)

    def before_train_iter(self,runner):
        if autocast_state()['enabled']:raise RuntimeError('Original training iteration must begin outside autocast')

    def after_train_iter(self,runner):
        if autocast_state()['enabled']:raise RuntimeError('Original optimizer/backward escaped into BF16 autocast context')

    def after_run(self,runner):
        self.close()
        report=self.evidence.finish(runner.optimizer)
        report.update(train_step_calls=self.calls,first_inside=self.first_inside,last_inside=self.last_inside,
                      last_loss_dtype=self.last_loss_dtype,ambient_after_training=autocast_state(),
                      wrapped_forward_calls=self.forward_calls,validation_step_calls=self.val_calls,
                      first_inside_cache_enabled=self.first_inside_cache_enabled,
                      last_inside_cache_enabled=self.last_inside_cache_enabled,
                      BF16_weight_cache_disabled=True,
                      backward_clip_optimizer_original=True)
        if self.calls<1:raise RuntimeError('No actual BF16 training forward/loss executed')
        runner._ascend_bf16_evidence=report
        rank=torch.distributed.get_rank()
        (Path(runner.work_dir)/f'bf16_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')


@contextmanager
def inference_precision(mode,receipt):
    if mode!='bf16':
        yield
        return
    import mmdet3d.models as models
    original=models.build_model;built=[];before_cache_enabled=bool(torch.is_autocast_cache_enabled())
    def build_model(*args,**kwargs):
        model=original(*args,**kwargs);built.append(BFloat16Evidence(model,inference=True));return model
    models.build_model=build_model
    try:
        with precision_context(mode):
            receipt['inside_before_original_test']=autocast_state()
            receipt['inside_before_original_test_cache_enabled']=bool(torch.is_autocast_cache_enabled())
            if receipt['inside_before_original_test_cache_enabled']:raise RuntimeError('BF16 inference weight-cast cache must be disabled')
            yield
        if len(built)!=1:raise RuntimeError('Expected one original inference model builder call')
        if bool(torch.is_autocast_cache_enabled())!=before_cache_enabled:raise RuntimeError('BF16 inference cache context was not restored')
        receipt.update(actual_model=built[0].finish(),ambient_after_original_test=autocast_state(),
                       ambient_after_original_test_cache_enabled=bool(torch.is_autocast_cache_enabled()),BF16_weight_cache_disabled=True)
    finally:
        models.build_model=original
        for evidence in reversed(built):evidence.close()
