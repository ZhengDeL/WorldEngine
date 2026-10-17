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
