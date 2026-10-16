"""Read numerical settings and device provenance without changing either."""
import os
import torch
import torch_npu


def precision_state(mode,stage='after_public_precision_setters_before_model_construction'):
    def option(name):
        try:
            value=torch_npu._C._npu_getOption(name)
            return value.decode() if isinstance(value,bytes) else value
        except Exception:
            return None
    result=dict(mode=mode,torch_deterministic=torch.are_deterministic_algorithms_enabled(),
                public=dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                            conv_allow_hf32=torch.npu.conv.allow_hf32),
                raw_options={name:option(name) for name in
                             ('ALLOW_MATMUL_HF32','ALLOW_CONV_HF32','ALLOW_INTERNAL_FORMAT','jitCompile')},
                environment={name:os.environ.get(name) for name in
                             ('CLOSE_MATMUL_K_SHIFT','HCCL_DETERMINISTIC','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING')},
                observation_stage=stage)
    if mode=='bf16':
        result['autocast_policy']=dict(enabled=True,device_type='npu',dtype='torch.bfloat16',
            master_parameter_dtype='torch.float32',optimizer_state_dtype='torch.float32',
            forward_and_loss_only=True,loss_scaler=False,weight_cache_enabled=False,MSDA_compute_dtype='torch.float32',planning_loss_compute_dtype='torch.float32',
            camera_geometry_compute_dtype='torch.float32',reference_grid_creation_dtype='torch.float32',
            BEV_shift_can_bus_creation_dtype='torch.float32',BF16_AddRelu='native_Torch_relu_of_sum_with_dtype_promotion',
            BatchNorm_SyncBatchNorm_compute_dtype='torch.float32',normalization_output_dtype='original_input_dtype',
            BF16_inference_trajectory_export='lossless_FP32_promotion_for_NumPy')
        result['ambient_autocast']=dict(enabled=bool(torch.is_autocast_enabled('npu')),
                                       dtype=str(torch.get_autocast_dtype('npu')))
    return result


def device_provenance():
    # Physical chip selection is provenance, not an algorithm resume invariant.
    return dict(WORLDENGINE_HOST_DEVICE_IDS=os.environ.get('WORLDENGINE_HOST_DEVICE_IDS'),
                ASCEND_RT_VISIBLE_DEVICES=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                rank=int(os.environ.get('RANK','0')),local_rank=int(os.environ.get('LOCAL_RANK','0')))
