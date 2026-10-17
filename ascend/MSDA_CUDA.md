# CUDA-compatible MSDA on Ascend

`--msda-backend cuda-compatible` selects an opt-in FP32 implementation of multi-scale deformable attention (MSDA) for Ascend. It follows the measured coordinate arithmetic and first-order reduction order of the original MMCV 1.6.2 CUDA implementation. It uses native PyTorch NPU operations; floating-point inputs, outputs and gradients remain on the NPU. Small integer spatial metadata is read on the host.

The default remains `--msda-backend reference`: the historical MMCV CPU32 mathematical route implemented with NPU `grid_sample`. The two backends have separate numerical targets. See the [reference implementation](MSDA.md), [original CUDA baseline](CUDA_BASELINE.md) and [installation and usage guide](../docs/algengine_ascend.md).

## Supported scope

| Property | CUDA-compatible backend |
| --- | --- |
| Compute dtype | FP32 in both FP32 and BF16 model modes; BF16 model autocast does not change MSDA's compute dtype |
| Channels per attention head | Exactly 32 |
| Spatial axes | Positive integer height and width, each from 1 through 4096 |
| Differentiation | First-order gradients for value, sampling locations and attention weights |
| Query processing | Every query is processed in chunks of at most 2048; chunking does not truncate the query range |
| Ordinary layouts | Valid matching spatial/projected levels; the value-gradient buffer has explicit contiguous storage, including when the input value is not contiguous |
| Original map layout | Only the supported contiguous raw L4 / spatial L1 case: eight heads and four points, with the original global flat-prefix interpretation |
| Floating inputs | Finite value, weight and upstream-gradient tensors; coordinate compensation has the finite/intermediate-underflow limits stated in the implementation policy |

In the original map layout, the backend interprets the entire contiguous location/weight flat prefix **before** splitting queries into chunks. It does not select level zero separately within each head or query. Gradients map back to the original raw tensors, and their unused contiguous tails remain zero.

The coordinate path uses FP32 compensated multiply/add arithmetic without an FP64 or floating-point CPU fallback. Far-padding and nonfinite sampling locations map to an inactive padding coordinate. This does not establish general support for nonfinite values, weights or upstream gradients, a general correctly rounded IEEE FMA operation, or higher-order differentiation. Value-gradient accumulation can also differ in order from CUDA atomics; tolerance agreement is distinct from bitwise equality.

## Numerical evidence and acceptance

Every element must be finite and satisfy `abs(actual - reference) <= atol + rtol * abs(reference)`:

| Complete tensor | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Forward output | `2e-5` | `2e-5` |
| Each of the three gradients | `1e-3` | `1e-4` |

The [operator validation record](results/npu-cuda-msda-20261017.json) identifies the public-entry source fingerprint `3d46c07f…` (245 Python files) and public numerical core `feb35fc8…`. These operator inputs are numerical test tensors, not driving-data samples. The fixed CUDA inputs and saved reference tensors were transferred with complete file and tensor hashes; NPU inputs were not regenerated from a seed. A seed alone does not establish identical inputs across Torch versions.

| Artifact or test | Identity and measured scope | Status |
| --- | --- | --- |
| Public registered entries | Actual DrivingSDK MX and MMCV functions; two full normal-L1 geometries, report `b7054570…` | All 64 CUDA comparisons pass, 32 per entry. Of 16 CPU32 comparisons, eight pass and eight remain outside tolerance. All 80 comparisons completed with unchanged inputs/source/precision/backend and normal release |
| Public MMCV original map layout | Actual registered MMCV six-argument entry; four fixed raw L4 / spatial L1 cases, report `a377773f…` | All 16 CUDA comparisons pass. Of 16 CPU32 comparisons, 14 pass and two remain outside tolerance. All 32 comparisons completed; complete raw gradients are preserved and unused tails are zero. MX raw-L4 remains unsupported |
| Original CUDA target | H20, Torch 2.0.1+cu118, MMCV 1.6.2; full outputs and three gradients from fixed inputs | Measured target; some boundary results differ from the saved CPU32 reference |

The supplementary checks below identify their own numerical source and extend the tested mathematical/layout cases. The public-entry results above cover the stated normal-L1 and MMCV raw L4 / spatial L1 cases; they do not establish acceptance for other legacy layouts, BF16 models or training.

| Supplementary test | Numerical source and measured scope | Result |
| --- | --- | --- |
| Private NPU candidate v8 | Numerical source `264676e4…`; B2/Q40000 and B1/Q65536, both 200×200, H8/C32/P4; four saved CUDA repetitions per case | All 32 complete CUDA tensor comparisons satisfy the original tolerances; saved CPU32 negative comparisons remain separate |
| Private legacy-layout candidate v6 | Numerical source `6f8e19f9…`; four raw L4 / spatial L1 cases | All 16 complete CUDA tensor comparisons satisfy the original tolerances; unused raw gradient tails are zero |
| Private v8 layout checks | Six cases, including noncontiguous value layouts; new exactly dyadic inputs and the original Torch 2.7.1 CPU mathematical function | All 24 complete tensor comparisons pass; these are separate from the saved CUDA/CPU32 tensors |
| Private guarded-coordinate checks | 22 axis pairs / 2112 coordinate components; padding and exactly dyadic mathematical cases | Coordinate checks pass; 28 conditional complete-tensor mathematical comparisons pass against their stated Torch 2.7.1 CPU reference |
| Private v8 fixed dyadic cases | Report `58e60d57…`; ten saved Torch 2.0.1 cases, including all eight gradient-need combinations and two additional multilevel geometries | All 56 present-role comparisons pass: 28 against saved CUDA201 and 28 against saved CPU201; absent gradients are verified separately |

The original CUDA and CPU32 mathematical implementations can choose different boundary arithmetic. Passing the CUDA target does not turn the retained CPU32 negative comparisons into passes. The public-entry report therefore retains a complete-outside verdict and worker exit 1; its guarded controller exited normally with no owned work remaining. Full-model training results are recorded separately. These operator results do not establish full-model replay, cross-platform training equivalence, convergence or performance.

The public [CUDA golden exporter](../tools/validation/README_export_cuda_msda_goldens.md) and [NPU checker](../tools/validation/README_npu_cuda_msda.md) were also executed end to end: a fresh original CUDA export, complete 16-file transfer and both actual public NPU entries. All 64 CUDA tensor comparisons passed; the separate CPU32 target retained eight within tolerance and eight outside. The checker verified all 16 input files before and after execution. Source, backend and original tolerances remained unchanged.

## FP32 same-prefix resume control (October 18, 2026)

[Full result](results/cuda-compatible-fp32-strict16-resume2-20261018.json). Public source `3d46c07f4ff0` (245 Python files), `cuda-compatible` MSDA, 16 ranks with batch 1 per rank (global batch 16), workers 0, and a two-update horizon. A continuous run saved updates 1 and 2; a fresh process resumed that same run’s update-1 checkpoint and completed update 2. The first update is shared by construction, so the numerical control tests continuation from that checkpoint rather than independent initialization.

| Model | Model tensors: torch.equal / total | All compared tensors: torch.equal / total | Max model / all absolute difference | Max loss difference | Original gates |
|---|---:|---:|---:|---:|---|
| VADv2 | 889 / 889 | 1698 / 1698 | 0 / 0 | 0 | within tolerance |
| HydraMDP | 909 / 909 | 1778 / 1778 | 0 / 0 | 0 | within tolerance |

The original comparator retains tensor `atol=1e-6`, `rtol=1e-5` and loss difference strictly below `1e-4`. Its equality count uses `torch.equal`; it does not establish tensor-byte or serialized checkpoint-file identity. Common model and Adam state are saved by rank 0; the all-tensor count additionally includes the final CPU/NPU RNG tensors for all 16 ranks. It is not a dump of every rank’s full final model. Each completed model has 16 exact full-state restores, 64 classified four-state RNG restoration events and 48 downloaded training records. All original participating gradients and Adam states were checked by parameter name, shape, optimizer group position and FP32 finite values.

The data configuration contains 139 training anchors. Actual unique current-anchor counts are VADv2: 32, HydraMDP: 32; the union across completed models is 32. Counts use the last token of each history queue. This control adds no inference or NAVSIM scoring result. Two updates remain within the original 500-update warmup and do not establish convergence, quality, independent-initialization reproducibility or an eight-update fix. Historical completed eight-update numerical negatives remain separate evidence.

Full-state resume requires the checkpoint’s exact source, precision, topology and training-horizon contract. The measured device runs retain the source identity recorded above. Later validation or CLI fixes can change a checkout’s source fingerprint even when model formulas remain unchanged; that newer checkout is not interchangeable for full-state resume. Read-only CPU artifact comparisons may use a separately identified validation-tool version.

A separately identified, stricter CPU validator rechecked the same saved artifacts for shape/dtype, finite tensors and records, complete update sequences and full contract-sized RNG coverage. Its receipts and unchanged-checkpoint hashes are recorded separately from the original comparator results.

The recorded comparison used a metadata observer (`8148ff59`) around the original public training entry to inspect RNG restoration and named gradient/Adam coverage; the numerical functions and checkpoint contract were preserved, but observation can affect execution timing. The public launcher commands omit that observer and its full diagnostic records, so the same `torch.equal` outcome has not been verified for an unobserved run; compare each new run's saved artifacts independently using the [CPU checkpoint comparison](../docs/algengine_ascend.md#5-save-and-resume-complete-training-state).


### Run the same-prefix training/resume protocol

Select the matching official100 config and initializer for one model. Follow the measured training/resume protocol with its exact source/image/data versions; complete each pair within the same checkout. These commands create a new continuous prefix rather than loading a checkpoint from another source version.

```bash
set -euo pipefail
: "${WORLDENGINE_ROOT:?Set the public checkout}"
: "${PREPARED139:?Set the validated NAVTRAIN139 prepared directory}"
: "${VOCAB:?Set test_8192_kmeans.npy}"
: "${MAPS:?Set the NuPlan maps directory}"
: "${IMAGE:?Set the dependency image}"
: "${EXPECTED_IMAGE_ID:?Set the verified image ID}"
: "${CONFIG:?Set the VADv2 or HydraMDP official100 config}"
: "${INITIALIZER:?Set the matching official100 checkpoint}"
: "${OUTPUT_ROOT:?Set a fresh absolute output directory}"
: "${MEMORY:?Set the validated container memory limit}"
: "${CPUS:?Set the validated container CPU limit}"
: "${PIDS_LIMIT:?Set the validated container task limit}"
: "${SHM_SIZE:?Set the validated shared-memory limit}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$WORLDENGINE_ROOT"
continuous="$OUTPUT_ROOT/continuous0to2"
resumed="$OUTPUT_ROOT/sameprefix_own1to2"
launcher=(python3 ascend/run_container.py
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --mini-prepared "$PREPARED139" --vocab "$VOCAB" --maps "$MAPS"
  --image "$IMAGE" --expected-image-id "$EXPECTED_IMAGE_ID"
  --memory "$MEMORY" --cpus "$CPUS" --pids-limit "$PIDS_LIMIT" --shm-size "$SHM_SIZE"
  --precision fp32 --msda-backend cuda-compatible --deterministic)
train=(--task train --config "$CONFIG" --checkpoint "$INITIALIZER"
  --runtime --upstream-schedule --batch-size 1 --workers 0
  --steps 2 --checkpoint-interval 1 --check-interval 1)
"${launcher[@]}" "${train[@]}" --output "$continuous"
test -s "$continuous/iter_1.pth"
test -s "$continuous/iter_2.pth"
"${launcher[@]}" "${train[@]}" --resume "$continuous/iter_1.pth" --output "$resumed"
test -s "$resumed/iter_2.pth"
```

## Source and checkpoint version contract

The validation, CLI and data-path revision has source fingerprint `40997952fdc9` (245 Python files); the device control above was measured on `3d46c07f4ff0` (245 files). The MSDA core is unchanged, while the six reviewed source changes make these different full-state resume contracts. Use the exact source, precision, topology and horizon recorded by a checkpoint for full resume; official planning weights can still initialize a fresh run. Read-only CPU comparison uses the separately identified `ba855451…` validator. Its six-method public CPU regression suite is [reproducible](../tools/validation/test_validation_guards.py).

The measured `3d46` source is available at commit [4946f15](https://github.com/ZhengDeL/WorldEngine/tree/4946f15dd2073b7c0974bc2335091246a4ec9023). Use a separate checkout of that commit to reproduce or resume those recorded runs. The updated `4099` source can initialize new training; its additional one-update VADv2 check is recorded in the [source regression report](results/source-review-regression-20261018.json). The recorded 16-rank container limits were `MEMORY=256g`, `CPUS=128`, `PIDS_LIMIT=8192`, `SHM_SIZE=8g`; these are test limits, not measured minimum requirements.

## Training, own-checkpoint resume and inference

On 17 October 2026, the public `cuda-compatible` MSDA entry completed training, own-checkpoint resume, original DDP4 inference and independent official CPU scoring for VADv2 and HydraMDP in both FP32 and BF16. Four disjoint groups each used four ranks and batch size 1 per rank, giving global batch size 4 per group. Each configuration started from its official 100% planning checkpoint, ran update 1, and resumed its own checkpoint to update 2. Original backbone-freeze settings, geometry, losses, gradient clipping and AdamW were retained. The measured runs explicitly set `--msda-backend cuda-compatible`; FP32 remains the default precision. BF16 used disabled autocast weight caching with actual BF16 Conv/Linear activations and FP32 master weights, participating gradients and Adam state.

The configured training subset contains 139 anchors deterministically derived from the original official data. Each configuration actually consumed 8 distinct current anchors in its two updates; all four configurations consumed the same set, so their union is also 8. History-queue frames are not counted as additional current anchors.

| Model | Precision | Ranks × batch per rank | Updates and own resume | Held-out anchors | Official PDMS |
|---|---|---|---|---|---|
| VADv2 | FP32 | 4 × 1 | 1 → own checkpoint → 2 | 22 | 0.871938963172 |
| VADv2 | BF16 | 4 × 1 | 1 → own checkpoint → 2 | 22 | 0.908986383163 |
| HydraMDP | FP32 | 4 × 1 | 1 → own checkpoint → 2 | 22 | 0.953343369212 |
| HydraMDP | BF16 | 4 × 1 | 1 → own checkpoint → 2 | 22 | 0.953676351654 |

All four branches use the same 22 held-out anchors; these are not 88 independent scenes. Every inference branch exported 22 unique finite 40×3 trajectories with the original vocabulary check. The original NAVSIM scorer independently evaluated genuine official cache payloads: all 22 tokens were valid in each branch, and all seven metrics plus their per-token CSVs are retained. [Complete result and hashes](results/cuda-compatible-navtrain139-short2-20261017.json) provides full-precision metrics, source/input/checkpoint/receipt hashes and the measured scope.

All eight original training containers completed with exit0, and 32 rank-stage receipts passed. All 16 own-resume rank receipts restored the original full model and Adam state, with classified Python/NumPy/CPU/NPU RNG restoration exact. The original participating parameter tensor counts with gradients and Adam state were 259 for VADv2 and 279 for HydraMDP; the full named parameter audits covered 730 and 750 tensors respectively. This does not claim every trainable parameter participates in every update. The four inference containers and four CPU scoring containers also completed with exit0 and normal release; original sources and inputs remained unchanged.

These two updates are inside the original 500-update warmup. The scores are bounded functional measurements; score differences do not establish quality improvement or FP32/BF16 quality equivalence. This run does not replace the original 16-rank/global16 strict replay, full-data training, long-run convergence or the untested genuine synthetic-producer configurations. Existing strict numerical replay limits remain open.

The following sequential commands reproduce each measured four-rank configuration. The actual campaign ran the four disjoint groups concurrently. Set absolute directories and the accepted image ID after following the dependency and official-data preparation instructions. Each invocation uses the original public numerical entry; retain its runtime, checkpoint, inference and official scoring records.

```bash
set -euo pipefail
# Use absolute paths. OUTPUT_ROOT must be a new campaign directory.
: "${WORLDENGINE_ROOT:?Set the public repository root}"
: "${PREPARED139:?Set the official NAVTRAIN139/NAVTEST22 prepared directory}"
: "${VAD100_CKPT:?Set the official100 VADv2 checkpoint}"
: "${HYDRA100_CKPT:?Set the official100 HydraMDP checkpoint}"
: "${VOCAB:?Set test_8192_kmeans.npy}"
: "${MAPS:?Set the NuPlan maps directory}"
: "${OFFICIAL_CACHE22:?Set the genuine official metric-cache directory}"
: "${IMAGE:?Set the CPU-accepted dependency image tag or ID}"
: "${EXPECTED_IMAGE_ID:?Set its verified docker image ID}"
: "${OUTPUT_ROOT:?Set a new absolute output directory}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$WORLDENGINE_ROOT"

run_case() {
  local model=$1 precision=$2 devices=$3 cfg initializer stem fresh resumed inference score
  case "$model" in
    vad)
      cfg="$WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py"
      initializer="$VAD100_CKPT" ;;
    hydra)
      cfg="$WORLDENGINE_ROOT/projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py"
      initializer="$HYDRA100_CKPT" ;;
    *) return 2 ;;
  esac
  stem=${cfg##*/}; stem=${stem%.py}
  fresh="$OUTPUT_ROOT/${model}_${precision}_fresh1"
  resumed="$OUTPUT_ROOT/${model}_${precision}_own1to2"
  inference="$OUTPUT_ROOT/${model}_${precision}_eval22"
  score="$OUTPUT_ROOT/${model}_${precision}_official_score22"
  local -a launcher train
  launcher=(python3 ascend/run_container.py
    --devices "$devices" --mini-prepared "$PREPARED139"
    --vocab "$VOCAB" --maps "$MAPS"
    --image "$IMAGE" --expected-image-id "$EXPECTED_IMAGE_ID"
    --precision "$precision" --msda-backend cuda-compatible --deterministic
    --memory 64g --cpus 32 --pids-limit 8192 --shm-size 8g)
  train=(--task train --config "$cfg" --checkpoint "$initializer"
    --runtime --upstream-schedule --batch-size 1 --workers 0
    --steps 2 --checkpoint-interval 1 --check-interval 1)

  # Horizon stays at two updates in both processes; only the first stops early.
  "${launcher[@]}" "${train[@]}" --stop-after 1 --output "$fresh"
  test -s "$fresh/iter_1.pth"
  "${launcher[@]}" "${train[@]}" --resume "$fresh/iter_1.pth" --output "$resumed"
  test -s "$resumed/iter_2.pth"

  # Original public DDP4 inference; the public exporter requires exactly22 tokens.
  "${launcher[@]}" --task inference \
    --config "$resumed/${stem}_rank0.py" --checkpoint "$resumed/iter_2.pth" \
    --output "$inference"

  # Independent original NAVSIM scoring, CPU-only, no accelerator devices.
  docker run --rm --runtime runc --network none \
    --memory 64g --cpus 8 --pids-limit 2048 \
    -e CUDA_VISIBLE_DEVICES= -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "$WORLDENGINE_ROOT:/source:ro" \
    -v "$OUTPUT_ROOT:$OUTPUT_ROOT:rw" \
    -v "$OFFICIAL_CACHE22:$OFFICIAL_CACHE22:ro" -w /source \
    "$EXPECTED_IMAGE_ID" /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
      --predictions "$inference/predictions.pkl" \
      --metric-cache "$OFFICIAL_CACHE22" --output "$score"
}

# Sequential reproduction of each tested four-rank configuration.
run_case vad fp32 0,1,2,3
run_case hydra fp32 4,5,6,7
run_case vad bf16 8,9,10,11
run_case hydra bf16 12,13,14,15
```

Training, full-state resume and inference must select the same `--msda-backend cuda-compatible`. Backend policy, source, image, precision, world size, data and schedule form the resume contract. Old-source or other-backend checkpoints require a new initialization rather than a full-state continuation. Numerical replay against a separately executed continuous control is a separate check; the original tensor `atol=1e-6, rtol=1e-5` and loss-difference `<1e-4` gates remain unchanged.

## Reference route and rollback

Omit `--msda-backend`, or pass `--msda-backend reference`, to use the historical reference route. Start a new output and use a checkpoint whose full-state contract matches that route, or initialize a new optimizer from compatible weights. Preserve the original CUDA-target, CPU32-target and full-model comparison reports as separate evidence.
