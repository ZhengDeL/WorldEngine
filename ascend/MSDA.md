# FP32 MSDA implementation and validation

[中文说明](MSDA_zh.md) · [measured results](evidence/msda-precision-20261013.json)

This page records the default `reference` route and its historical CPU32 target. The optional [CUDA-compatible backend](MSDA_CUDA.md) has a separate target: public MX/MMCV entries pass all 64 complete CUDA tensor comparisons; of 16 CPU32 comparisons, eight pass and eight remain outside tolerance. See the [operator record](results/npu-cuda-msda-20261017.json). These results preserve the historical CPU32 gates; full-model results are separate.

The tested FP32 MSDA location-gradient path passes the original CPU32 tolerance. The original full-module report remains historical evidence for source fingerprint `96a7bef…`; its hour-long throughput and memory numbers are not measurements of this implementation.

The [October 17 original CUDA baseline](CUDA_BASELINE.md) adds full forward/three-gradient measurements on two normal L1 shapes and original eight-GPU training controls. The CUDA extension itself contains elements outside the unchanged CPU32 gates on these inputs. This does not invalidate the recorded NPU-to-CPU32 results or establish matched-input CUDA/NPU gradient agreement.

The recorded runtime fingerprint is `858f9d01181d3d62c35d0635881b22c62daf1a5f5af8123f622055de05da422d` (242 Python files: recursive AlgEngine Python plus top-level Ascend Python; CRLF normalized to LF). The normal and layout validators are included. Supplemental probe files have their own SHA256 in the result. The locked runtime image is `local/worldengine:26.1.0-a3-clean-20261013`, ID `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`.

## Implementation

`ascend/msda.py` uses NPU-native `grid_sample` forward and first-order backward, with the unchanged MMCV PyTorch reference's `2 * locations - 1`, bilinear sampling, zero padding and `align_corners=False`. Floating inputs, outputs and gradients remain on NPU. Only small integer shape/start metadata moves to CPU. This statement concerns MSDA: the full model also contains an existing CPU fallback for matrix inversion.

Both actual FP32 entry points are covered: MMCV and `mx_driving.multi_scale_deformable_attn`. Forward/backward disable ambient autocast. Backward recomputes sampling intermediates; 2048-query chunks bound individual temporary tensors, but backward still builds a graph for all chunks. Recompute and additional dispatches introduce a possible speed cost; use the new model measurements rather than inferring performance from the implementation. Non-FP32 and second-order gradients do not inherit this acceptance.

The coordinate-only candidate reproduced the old seed but left location-gradient elements outside tolerance in broader cases, so it was not adopted. CPU64 diagnosis did not replace the CPU32 oracle. The acceptance rule remains `abs(actual-reference) <= atol + rtol * abs(reference)`: forward **2e-5 / 2e-5**, all three gradients **1e-3 / 1e-4**. Every tensor element is compared.

The original `200×200/Q300/H8/C32/seed29` location-gradient case recorded **0/19200** elements outside tolerance, versus the previous 35. Maximum absolute error is **0.001220703125**, accepted by the unchanged combined absolute/relative rule. This is CPU32 agreement, not bitwise equality or measured GPU fused-kernel parity.

## Original map layout

The original map head projects four location/weight levels while supplying one BEV level. In [MMCV 1.7.2 CUDA host](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu) and [kernel indexing](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh), the actual spatial level count determines raw-buffer strides. We infer a **global flat-prefix** interpretation across batch/query/head from these sources. The historical layout experiment did not measure complete CUDA and NPU outputs and gradients from the same serialized inputs. The October 17 normal-L1 CUDA tests do not establish this mismatched-level layout result.

The installed SDK forward experiment instead matched a per-query prefix and differed from that CUDA-source interpretation. Therefore preserving the earlier SDK result would preserve a different storage interpretation. The new MMCV compatibility path retains original checkpoint/projection sizes and the inferred CUDA flat prefix. It only accepts actual L1/projected L4, H8/P4/C32 and contiguous floating inputs; other mismatched level counts are rejected. It is selected by this layout structure, not by a module name. Unused raw location/weight tails receive exactly zero gradients. The SDK experiment is a forward-only diagnostic, not a GPU comparison or a complete SDK backward characterization.

## Executed checks and limits

| Check | Measured result |
|---|---|
| Normal layouts: 18 cases, both entries | 36/36 complete forward + value/location/weight gradient comparisons passed |
| Original raw-map layout: four cases | 4/4 passed, including complete raw gradients and exact-zero unused tails |
| Full encoder shape: Q40000/H8/C32, seeds 29 and 7 | Four normal/legacy cases, 6/6 entry comparisons passed |
| Frozen subsets, no-gradient inputs, ambient autocast | 12/12 checks passed |
| Device profiler | Native NPU grid-sampler forward and backward kernel records collected |
| Real original configurations | 36/36, three updates each, exact original initializer for each configuration |
| Eight-rank perception training on this source | 160 updates per rank; all ranks exceeded one hour, exit 0, no OOM |
| Actual eight-rank update-100 checkpoint | Complete state restored on every rank and training continued to update 103 |
| Independent perception after that resume | Original detection/mapping path completed 37 frames and checked 370 finite prediction fields |
| Controlled VADv2/Hydra numerical replay | Both original gates passed; all compared model/optimizer tensors, losses, tokens, LR and final RNG matched exactly in these two single-rank controls |
| Complete checkpoint CPU audit | 45/45 distinct checkpoint paths passed model/optimizer finiteness, source/counter/RNG checks |
| Representative planning inference and official scoring | VADv2, Hydra and DiffusionDrive each passed 22 unique independent anchors and official CPU score coverage |

The normal matrix also covers multiple seeds, four feature levels, nonsquare shapes, B2/B8, boundaries/adjacent FP32 values, size-one axes, noncontiguous inputs/gradients and varied attention weights. Original model regressions retain 200×200 BEV, 900 tracking queries and eight cameras. Complete acceptance of all eight genuine synthetic configurations, full NAVTRAIN convergence, full-image training RAM/I/O/time, matched-input full-tensor CUDA/NPU MSDA comparisons and paired training/quality validation remain separate. The bounded [official-weight FP32 model comparison](results/gpu-npu-official100-eval22-20261016.json) and [original CUDA operator/training baseline](CUDA_BASELINE.md) have separate scopes. [Subset preparation tools](data_tools/README.md) report bounded results for two synthetic configurations.

## Source858 training and continuation

The eight-rank run used B1/workers0, the original perception initializer, model geometry, unfrozen backbone, losses, optimizer and warmup. Its unchanged LR horizon was 10000 updates, with a measured one-hour stop. Every rank completed **160 optimizer updates**; measured training time ranged from **3616.849 to 3617.775 seconds**. The 1280 global anchor exposures include repeated samples and sampler padding from **42 unique real training anchors**. They do not represent 1280 different examples or complete NAVTRAIN epochs.

End-to-end observed throughput was **0.35381 global anchor exposures/s**, using 1280 exposures divided by the slowest rank's measured training time. This includes data loading, training observation and checkpoint work. Raw losses were observed each update and present gradients checked every ten updates. Peak NPU allocated/reserved memory was **43.63/48.62 GiB per logical chip**. Observed charged CPU cgroup peak was **66,488,492,032 bytes (61.92 GiB)** under a **128 GiB limit**, with memory fail count zero. Charged host memory includes cgroup-accounted cache; it is not a complete-data RAM requirement or a process RSS sum. All actual containers exited zero without OOM. One hour does not establish 24-hour stability or exclude a longer-term leak.

The historical `96a7bef…` implementation observed **0.94808 exposures/s** on its earlier mini run. The recorded source858 rate is lower. These runs are not a paired benchmark: the MSDA/map storage interpretation, implementation and observation/check frequency differ. The measurements expose a speed cost to investigate; they do not isolate its entire cause, establish a GPU/NPU performance ratio, or provide a full NAVTRAIN training-time estimate.

The complete checkpoint saved by this run at update 100 was restored on all eight ranks, including **1393 model and 3594 optimizer tensors per rank**, with the same source/data/topology/schedule contract and saved rank RNG. Training reached update **103** with three additional updates on each rank and exit zero. The restored model then completed **37 independent current/history frames**, with **370 finite prediction fields** checked through the original detection/mapping evaluator. These are full-state restoration, continuation and functional inference results. They do not establish strict numerical replay of every subsequent update or trained perception quality.

Separate VADv2/Hydra controls each completed eight continuous updates and resumed their own update-4 checkpoint to update 8, with a deterministic request, B1/workers0 and the same eight-update horizon. Both original numerical gates passed: tensor `atol=1e-6/rtol=1e-5` and loss absolute-difference `<1e-4`. In these specific controls, all **889 VADv2 model / 777 optimizer tensors**, and all **909 Hydra model / 837 optimizer tensors**, were **bitwise equal**. The original comparator reports combined totals of 1668/1748 after also counting two rank-RNG byte tensors; its original fields and values are retained. Maximum tensor and loss differences were zero; batch tokens, learning rates and final saved RNG matched. This is bounded single-rank replay evidence, not a bitwise determinism guarantee for every configuration, seed, batch, distributed topology or training duration. Earlier historical-source numerical failures remain historical evidence.

Their initial CPU comparison attempts stopped during NPU checkpoint deserialization before any comparison JSON was produced; those attempts produced **no numerical gate verdict**. The retry imported the installed official `torch_npu` serialization hooks before running the original unchanged target. Its existing `map_location='cpu'` activated the official CPU remapping flag, `RE_MAP_CPU` false → true. All three CPU jobs exited zero with no NPU initialization and unchanged target source hashes; no custom loader, checkpoint rewrite or threshold change was used. The subsequent complete checkpoint audit passed **45/45 distinct paths**, checking every saved model/optimizer tensor's finiteness, source fingerprint, legal conditional optimizer counters and rank-specific RNG coverage. Equal SHA files in different runs were audited separately; symlinks were excluded.

Independent inference and unmodified official NAVSIM CPU rescoring passed for all three representative checkpoints. Every planner exported **22 unique finite 40×3 trajectories**, verified the independent whole-log split, and received scores for all **22 valid tokens**. All inference and final CPU-scoring jobs exited zero.

| Planner | Checkpoint workload | Official mean PDMS |
|---|---|---:|
| VADv2 | Eight bounded real-mini training updates | 0.7851926239693333 |
| Hydra | Eight bounded real-mini training updates | 0.6084522121144075 |
| DiffusionDrive | Three bounded real-mini training updates | 0.18729393472412512 |

These are actual small-data scores after only three/eight updates. They do **not** establish convergence or qualified driving quality. The low DiffusionDrive score is retained. Different training/update/batch workloads and no GPU control prevent a migration-quality or GPU/NPU equivalence conclusion from these numbers.

An initial CPU-scoring invocation replaced the image's `PYTHONPATH` and hid its `nuplan` dependency. The retry restored inherited directories from the same pinned image while keeping the same prediction files and unchanged scoring code; original and retry code SHA256 match. The incomplete invocation remains in the versioned record. No production Python source or official scorer changed.

## Source858 eight-rank numerical replay

The October 13, 2026 VADv2/Hydra controls completed four training/restoration jobs and 32 rank runtime checks. Saved-state loading matched, but both final comparisons contained tensors outside the original tolerance: VADv2 had 12 model and 21 optimizer tensors, Hydra had 2 and 2. Batch tokens, LR and final RNG matched; loss differences remained within 1e-4. The [versioned eight-rank record](evidence/eight-rank-resume-20261013-evening.json) preserves thresholds and paths. The cause of full-model divergence remains unconfirmed; fixed-input MSDA agreement does not resolve it.

At the October 13 22:44 snapshot, perception training was still active. The later completed [source858/post6 eight-hour result](results/eight-hour-perception-stability-20261014.json) is separate from current-source/post8 long-run acceptance.

## Repeat the strict gates on the server

Use an idle logical device and a new output directory. The following mounts source read-only and uses the pinned image; it downloads no data. The full-shape probe invokes the same public validator functions with original H8/C32/Q40000 shapes.

```bash
IMAGE=local/worldengine:26.1.0-a3-clean-20261013
EXPECTED=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$EXPECTED"
NPU_DEVICE=0
EVIDENCE_ROOT=/path/to/msda-validation
OUT="$EVIDENCE_ROOT/rechecks/msda_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
docker run --rm --network host --memory 24g --cpus 8 --shm-size 4g \
  --device "/dev/davinci$NPU_DEVICE" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e HCCL_DETERMINISTIC=true -e CLOSE_MATMUL_K_SHIFT=1 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/workspace/WorldEngine:ro" -v "$OUT:/output:rw" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -w /workspace/WorldEngine "$IMAGE" bash -lc '
    set -e
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PATH=/opt/conda/envs/bevformer/bin:/opt/conda/bin:$PATH
    export PYTHONPATH=/workspace/WorldEngine:/workspace/WorldEngine/projects/AlgEngine:$PYTHONPATH
    python ascend/validate_msda.py --output /output/strict_matrix.json
    python ascend/validate_msda_layout.py --output /output/msda_layout.json
    python ascend/probes/full_shape_gate_probe.py --output /output/full_shape_gate.json
  '
```

The normal and map-layout validator entry points set matmul HF32 false and convolution HF32 true while MSDA sampling remains FP32. These standalone settings do not record the current training FP32 preset, which disables both public HF32 flags; model runs need their own before/after precision reports.

Check exit zero and every complete case in the three reports. Use the [full-module commands](FULL_ALGENGINE.md) for real model training and complete resume. Source, original config, data, topology, optimizer/schedule and image are part of the resume contract: old-source training checkpoints are not silently accepted as complete resumes on this source.

## Repeat CPU replay comparison without changing the comparator

In the locked image, import `torch_npu` before the original CPU comparator so its installed serialization hooks can honor CPU remapping for NPU-saved tensor metadata. The following VAD example mounts completed evidence read-only and no NPU devices. Set `SRC` to this exact source checkout; use a new writable output directory. The continuous run serves as its own update-4 prefix control. For Hydra, replace the three VAD directory names with their corresponding Hydra names.

```bash
IMAGE=local/worldengine:26.1.0-a3-clean-20261013
EXPECTED=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$EXPECTED"
SRC="$PWD"
EVIDENCE_ROOT=/path/to/msda-validation
CASE="$EVIDENCE_ROOT/msda_precision_20261013/controlled_replays_v6/navformer_vadv2_v6"
OUT="$EVIDENCE_ROOT/rechecks/cpu_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
docker run --rm --network none --memory 32g --cpus 8 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SRC:/workspace/WorldEngine:ro" -v "$CASE:/evidence:ro" -v "$OUT:/output:rw" \
  -w /workspace/WorldEngine "$IMAGE" /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
assert not torch.npu.is_initialized()
' \
  --continuous /evidence/navformer_vadv2_continuous_v6/iter_8.pth \
  --resumed /evidence/navformer_vadv2_resumed_v6/iter_8.pth \
  --continuous-run /evidence/navformer_vadv2_continuous_v6 \
  --partial-run /evidence/navformer_vadv2_continuous_v6 \
  --resumed-run /evidence/navformer_vadv2_resumed_v6 \
  --output /output/compare_resume.json
```

Verify Docker exit zero and `result="pass"` in the actual JSON. This command keeps the public comparator, original tolerances and checkpoint payloads unchanged.

## NAVTRAIN camera archives

The eight current/history `.tgz` candidates total **449,046,313,693 bytes**, or **449.05 decimal GB compressed**. They must be extracted because the data loader reads individual camera-image files. The [official NAVSIM v1.1 download script](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_navtrain.sh) explicitly extracts each archive. Already-extracted copied directories need no further extraction.

This total is neither the expanded footprint nor the 5 TB SimEngine asset publication. Expanded size and complete WorldEngine annotation/history/eight-camera coverage remain unmeasured. Allow space for expanded images, annotation/PDM PKLs, NAVTEST, metadata/maps, checkpoints and temporary files. Sequential extraction can avoid retaining all compressed copies. The recorded MSDA tests used real-image subsets; full NAVTRAIN camera coverage was not validated.
