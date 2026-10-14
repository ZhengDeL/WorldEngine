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
    return dict(mode=mode,torch_deterministic=torch.are_deterministic_algorithms_enabled(),
                public=dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                            conv_allow_hf32=torch.npu.conv.allow_hf32),
                raw_options={name:option(name) for name in
                             ('ALLOW_MATMUL_HF32','ALLOW_CONV_HF32','ALLOW_INTERNAL_FORMAT','jitCompile')},
                environment={name:os.environ.get(name) for name in
                             ('CLOSE_MATMUL_K_SHIFT','HCCL_DETERMINISTIC','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING')},
                observation_stage=stage)


def device_provenance():
    # Physical chip selection is provenance, not an algorithm resume invariant.
    return dict(WORLDENGINE_HOST_DEVICE_IDS=os.environ.get('WORLDENGINE_HOST_DEVICE_IDS'),
                ASCEND_RT_VISIBLE_DEVICES=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                rank=int(os.environ.get('RANK','0')),local_rank=int(os.environ.get('LOCAL_RANK','0')))
