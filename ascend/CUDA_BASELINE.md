# Original CUDA baseline — October 17, 2026

[中文](CUDA_BASELINE_zh.md) · [Portable operator tool](../tools/validation/README_cuda_msda.md)

On eight H20 GPUs, the original CUDA model paths completed initialization from both official 100% checkpoints, fresh eight-update training, independent four-update training,
and fresh own4→8 resume. Every rank's saved model/optimizer restore and RNG
restore passed. Both complete three-arm comparisons remain **outside the
unchanged numerical replay thresholds**. This result establishes neither the
cause nor acceptance of NPU numerical differences. [Full training evidence](results/cuda-training-original-20261017.json)
contains the original failure paths, thresholds, counts and provenance.

## Training and state restoration

The source is upstream AlgEngine at `3eded7e8` (263 original files); models are VADv2 and
HydraMDP with their original official 100% weights and backbone-freeze policies.
The input contains 139 legal anchors, 284 annotation frames and 1,536 JPEGs from one genuine
NAVTRAIN log, preserving the public prepared manifest and labels. It is not
full NAVTRAIN. See [official input origins](evidence/official-data-origins-20261016.json).
The stack is PyTorch 2.0.1+cu118/MMCV 1.6.2. Pip check reports the two documented
MMDet3D dependency declarations for Numba/NetworkX; the actual versions are listed
in the evidence.

All six jobs used eight GPUs, batch size 2 per GPU (global batch 16), `seed=20261012`, `max_iters=8` and the original
500-update warmup. The independent partial stopped at 4 without changing that
horizon; a new process resumed its own full checkpoint through 8. Original
AdamW default foreach policy, gradient clipping at norm 35, sampler, dropout and model math
were retained. Strict deterministic flags/TF32-off were read back, with no
fallback to warn-only. All 48 rank processes and 256 update records were
verified. All 16 resumed ranks restored the exact saved common model/optimizer,
their rank RNG and epoch 0 / batch offset 4 cursor; named coverage and finite checks
include every parameter and preserve legitimate unused parameters.

| Model | Model max abs difference | Model+optimizer max | Model tensors outside tolerance | Optimizer tensors outside tolerance | Max loss difference |
|---|---:|---:|---:|---:|---:|
| VADv2 | 0.0002568736672401428 | 0.06632401049137115 | 196/889 | 276/777 | 0.0017986297607421875 |
| HydraMDP | 0.00034481287002563477 | 0.26890525221824646 | 232/909 | 304/837 | 0.0031709671020507812 |

Model/optimizer floating Tensor tolerances are `atol=1e-6, rtol=1e-5`; loss
requires strictly `<1e-4`. Tokens, learning rates, final complete all-rank
Python/NumPy/CPU/CUDA RNG and named coverage match exactly for both profiles.
The combined maximum is not an optimizer-only maximum. Original `torch.equal`
counts are exact numeric equality counts, not serialized-PTH or signed-zero
raw-byte identity. Common checkpoint Tensor values are rank0 model/optimizer;
RNG and runtime coverage checks cover all ranks.

A separate same-run prefix control also completed: fresh processes resumed
`iter_4` saved by the same continuous 0→8 run, then compared only updates 5–8.
The saved model/optimizer, all eight ranks' RNG and cursor restored exactly;
tokens, learning rates, final RNG and named coverage also match. Both controls
remain **outside the same original numerical thresholds**:

| Model | Model max abs difference | Model+optimizer max | Model tensors outside tolerance | Optimizer tensors outside tolerance | Max loss difference |
|---|---:|---:|---:|---:|---:|
| VADv2 | 0.000011362135410308838 | 0.027376020327210426 | 24/889 | 188/777 | 0.00010085105895996094 |
| HydraMDP | 0.00003832578659057617 | 0.48403623700141907 | 82/909 | 274/837 | 0.0007677078247070312 |

This control shares updates 1–4 and is distinct from the independent partial4
experiment above. It adds two jobs, 16 rank processes and 64 update records;
together the two protocols cover eight jobs, 64 rank processes and 320 records,
without counting the reused continuous runs twice. Exact checkpoint restoration
does not establish numerically repeatable subsequent training.

Eight updates are below warmup and do not establish convergence, full-data
quality or sustained performance. No newly trained checkpoint was evaluated
or officially scored in this training campaign.

## Complete MSDA operator measurements

The [portable actual result](results/cuda-msda-original-20261017.json) reproduces
the six full-input hashes and all four CPU32 reference Tensor hashes of the
original fixed-input experiment. Two normal-L1 shapes, `seed=29` and four repeats
produce 56 full-Tensor comparisons of forward and all three gradients:

| Case | B/Q/S/H/C/L/P | CPU32 forward failing elements per repeat | CPU32 location-gradient failing elements per repeat | Max location-gradient difference | Native repeat value-gradient max |
|---|---|---:|---:|---:|---:|
| temporal40000 | 2/40000/40000/8/32/1/4 | 14 | 10129 | 2411.2860107421875 | 7.152557373046875e-7 |
| query65536 | 1/65536/40000/8/32/1/4 | 9 | 8204 | 1824.4653930664062 | 9.5367431640625e-7 |

Both use `spatial_shapes=[[200,200]]`. The original CPU32 forward tolerance is
`2e-5/2e-5` (atol/rtol), and each gradient tolerance is `1e-3/1e-4`. CPU32
forward/location negatives are retained; no CUDA or CPU64 target replaces them.
Native repetitions are within the original tolerance, but only value gradient
is not byte-identical. Forward/location/weight bytes are stable. Global strict
mode does not certify an external custom kernel's deterministic registration.

These tests do **not** measure the historical raw-L4/spatial-L1 layout described
in [MSDA](MSDA.md). That older interpretation cites MMCV 1.7.2; this normal-L1
CUDA run is MMCV 1.6.2. The separate [official 100% / 22-anchor FP32 GPU/NPU inference result](results/gpu-npu-official100-eval22-20261016.json)
remains valid within its own bounded scope. Neither result proves full operator
GPU/NPU equivalence. No new NPU run was performed in this GPU-only session.

The [original CUDA source](https://github.com/open-mmlab/mmcv/blob/v1.6.2/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh#L61-L120) uses atomic accumulation for value gradients; a
small gradient difference can propagate through trainable projections and
updates, but no real first-divergent model Tensor was captured here. The
synthetic operator's maximum cannot quantitatively explain the model loss
difference or all prior NPU drift. No operator/formula/tolerance was changed.
