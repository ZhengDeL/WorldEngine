# AlgEngine Ascend 910C Validation Report

Assessment updated: October 17, 2026. Earlier results retain their recorded source and image.

[Chinese validation report](VALIDATION_zh.md) · [Installation and operation guide](../docs/algengine_ascend.md) · [Versioned evidence](results/)

## Scope and validation status

WorldEngine contains AlgEngine for model training and evaluation, and SimEngine
for simulation, rendering and rollout generation. This report covers AlgEngine.
SimEngine deployment on NPU is outside the assessed scope.

The CUDA-compatible NPU backend completed VADv2 and HydraMDP training, own-checkpoint restoration, independent inference and official CPU scoring in FP32 and BF16. The measured configuration uses four ranks per group and two updates, with eight actual training anchors from the configured NAVTRAIN139 subset and the same 22 held-out evaluation anchors.

Earlier 16-rank, eight-update reference-backend comparisons completed outside their original replay tolerances. Those records remain valid historical limits. The new short functional chain and separate private-candidate two-update positive control do not close longer strict replay, full NAVTRAIN convergence, full NAVTEST quality or untested genuine synthetic-input configurations.

## CUDA-compatible model validation — October 17, 2026

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

A metadata lookup error in the orchestration controller was preserved after all eight model jobs had finished. An independent read-only CPU reconciliation supplied the missing Hydra model-state count and rechecked the existing eight checkpoints, 112 rank evidence files, eight launch records and five original control records by whole-file SHA before and after. It did not rerun training or modify the production source. The result keeps the original controller error and the successful reconciliation as separate evidence. Model-process time and CPU-reconciliation time are labeled separately.

These two updates are inside the original 500-update warmup. The scores are bounded functional measurements; score differences do not establish quality improvement or FP32/BF16 quality equivalence. This run does not replace the original 16-rank/global16 strict replay, full-data training, long-run convergence or the untested genuine synthetic-producer configurations. Existing strict numerical replay limits remain open.

## CUDA-compatible MSDA — October 17, 2026

The optional `--msda-backend cuda-compatible` uses FP32 compensated sampling
coordinates and first-order gradients on NPU. Both actual public registered
entries, MX and MMCV, passed all 64 complete comparisons against saved original
CUDA outputs and gradients on the two full shapes, B2/Q40000 and B1/Q65536.
The original tolerances were retained. The 16 separate CPU32 comparisons contain
eight within tolerance and eight outside tolerance; these complete negatives
are retained alongside the CUDA results.

The measured public source fingerprint is `3d46c07f…` / 245 Python files,
with the `3522f3ac…` image. The default `reference` backend retains the
historical CPU32 mathematical target. Train, resume and inference record and
check the selected backend. See the [implementation and usage guide](MSDA_CUDA.md)
and [complete operator evidence](results/npu-cuda-msda-20261017.json) for exact
source identities, supplementary layout checks and supported scope.

A separate 16-rank VAD FP32 two-update control used the private v8 numerical
overlay on source `e14dd6b2…`. Independent continuous training and own-checkpoint
restoration produced exactly equal values in all 889 model tensors and all
1,698 complete-checkpoint tensors, with zero loss difference. All 144 rank
evidence files were checked. This is a short-run positive control, not a
public-backend model result or an eight-update replay acceptance. The original
comparator uses `torch.equal`; this does not assert byte identity of serialized
checkpoints. [Bounded replay evidence](results/npu-private-msda-vad-fp32-replay2-20261017.json)

This operator result and the separate two-update control do not close the
historical full-model replay comparisons below. The [original CUDA training baseline](CUDA_BASELINE.md) also records
completed eight-GPU VADv2/HydraMDP training and checkpoint-restoration controls,
whose complete replay comparisons remain outside their original tolerances.

The previously interrupted Hydra BF16 FT37 evaluation was independently rerun
to completion: all 22 trajectories and all 22 official NAVSIM scores are valid,
with PDMS **0.9338557270555728**. It uses the original reference backend and the
same historical eight-update checkpoint; it is separate from the new MSDA
backend. [Follow-up evidence](results/hydra-ft37-bf16-eval22-20261017.json)

## Original CUDA/NPU FP32 inference comparison — October 16, 2026

The original CUDA and Ascend FP32 paths completed inference on the same official
100% VADv2/HydraMDP weights and the same 22 held-out anchors. Each model's 22
selected indices and all 880 exported `(x, y, heading)` poses matched bitwise,
including dtype. Independent original NAVSIM rescoring also produced identical
per-token CSV bytes: all seven metrics for each of the 22 tokens matched.

| Model | Selected indices / trajectories matched | Maximum position / wrapped-heading difference | GPU / NPU official score |
| --- | --- | --- | --- |
| VADv2 | 22/22; all 40 poses per token | 0 m / 0 rad | 0.8696212394149101 / 0.8696212394149101 |
| HydraMDP | 22/22; all 40 poses per token | 0 m / 0 rad | 0.9493509518718398 / 0.9493509518718398 |

This was one FP32 inference run per model on one H20 GPU, with TF32 disabled;
it was not GPU training. The GPU stack used Python 3.9.9, PyTorch 2.0.1+cu118,
MMCV-full 1.6.2 and original AlgEngine source `3eded7…`. The preserved NPU
reference used Python 3.10, PyTorch 2.7.1/Torch-NPU 2.7.1.post8, source
`4ae8ed9f…` and image `3522f3ac…`. The GPU environment retained two explicit
MMDet3D dependency-declaration conflicts: Numba 0.56.4 against `==0.53.0`, and
NetworkX 2.5 against `<2.3`. It was not a dependency-clean environment.

Initial CPU scoring on the GPU host returned zero scores. On its Xeon Gold 6430,
NumPy 1.23.4/OpenBLAS 0.3.20 selected Cooperlake and produced an invalid LQR
pseudoinverse. Selecting `OPENBLAS_CORETYPE=Haswell` **only in fresh CPU scoring
processes** restored the same-matrix residual from 2293.842 to 4.607e-15 and
the complete 22-token scores above. The original packages, loaded-library hashes,
model predictions and scoring formula were unchanged; the initial zero-score
records remain preserved. The workaround is consistent with the
[upstream NumPy report](https://github.com/numpy/numpy/issues/24903), and is
validated here only for this CPU/software combination.

The [complete comparison record](results/gpu-npu-official100-eval22-20261016.json)
separates trajectory identity, corrected CPU scoring and the original scoring
failure. [CPU rescoring instructions](../docs/algengine_ascend.md#x86-cpu-scoring-of-gpu-exports)
use the unchanged public scorer. These are two model comparisons on the same 22
anchors, not 44 independent scenarios. Equal JPEG bytes do not establish
bitwise-equal preprocessed model tensors. Different software stacks prevent
isolating a pure hardware cause; no quality or platform-equivalence threshold
was applied. Full-data quality, GPU training, repeated runs, throughput and the
unresolved 16-rank base-training replay issue remain outside this result.

## FP32 short-run validation — October 16, 2026

Both original 100% planning configurations completed the NAVTRAIN139/NAVTEST22
chain on source `4ae8ed9f…`/244 Python files and image `3522f3ac…`: one-update
smoke check, continuous updates 0–8, independent partial updates 0–4 under the
same eight-update horizon, own checkpoint restoration 4–8, and official
initializer/restored-checkpoint inference and CPU scoring on 22 held-out
tokens. The original backbone-freeze configuration, 8192-candidate labels,
16 logical NPUs/eight physical cards, B1/global batch16 and workers0 were retained.

All eight training jobs, 128 rank runtime reports, 32 exact state-restoration
reports, four inference jobs and four official CPU scoring jobs completed.
Source, official initializers, data and metric-cache hashes remained unchanged;
all owned jobs exited without OOM. Both unchanged-gate replay comparisons
completed outside tolerance:

| Original model | Model / optimizer leaves outside tolerance | Maximum model / combined tensor difference | Maximum loss difference |
| --- | ---: | ---: | ---: |
| VADv2 100% | 177 / 270 | 0.0002471804618835449 / 0.2677420377731323 | 0.002409696578979492 |
| HydraMDP 100% | 252 / 314 | 0.00036209821701049805 / 0.22671931982040405 | 0.037529945373535156 |

VAD compared 889 model/1698 combined tensors; Hydra compared 909/1778. The
combined counts include optimizer state and 32 rank RNG tensors. Original
tensor `atol=1e-6, rtol=1e-5` and loss tolerance `1e-4` were retained. Tokens,
learning rates and final rank RNG states matched in both pairs.

| Model | Official initializer, FP32 inference | Restored update-8 checkpoint, FP32 inference |
| --- | ---: | ---: |
| VADv2 | 0.8696212394149101 | 0.9033869555660513 |
| HydraMDP | 0.9493509518718398 | 0.9548414947194644 |

These are22-token score measurements, without a quality acceptance threshold.
Eight updates remain within 500 warmup updates; this is functional short-run
evidence, not convergence or numerical replay acceptance. The original
eight-epoch workflow remains documented but is not completed by this result.
Evidence: [FP32 short-run record](results/fp32-navtrain139-short8-20261016.json).

## BF16 v7 short-run validation — October 16, 2026

The cache-disabled v7 implementation completed the same official100
NAVTRAIN139/NAVTEST22 short-run chain for both models: a one-update cold start, continuous updates 0–8,
independent partial updates 0–4, own restoration 4–8, and official-initializer/restored
checkpoint inference and scoring on 22 held-out tokens. All eight training
jobs, 128 rank runtime reports, 32 exact restoration reports, four inference jobs
and four official CPU scoring jobs completed without OOM; input/source/cache
hashes remained unchanged and owned jobs exited.

The runtime source is `e14dd6b2…`/244 Python files with image `3522f3ac…`.
It disables BF16 autocast weight caching and records complete named parameter,
freeze, shape, optimizer-group, gradient-presence and Adam-state metadata.
FP32 master parameters, gradients and Adam moments remain FP32. The original
backbone-freeze configuration, 8192 candidates, 16 logical NPUs/eight physical
cards, B1/global 16 and workers0 are unchanged. Two implementation files differ
from the FP32 reference source; the other 242 canonical-LF files match, including
14 individually pinned raw EOL conversions.

| Model | FP32 / old BF16 v6 / current BF16 v7 IDs with Adam state | Previously missing entries restored on every rank |
| --- | ---: | ---: |
| VADv2 | 259 / 121 / 259 | 138 |
| HydraMDP | 279 / 141 / 279 | 138 |

Cold start 1 and continuous 8 each verified all 16 ranks. Read-only inspection of each
actual saved rank0 checkpoint found all 138 Adam steps equal to 8, establishing
eight optimizer visits with non-None gradients. The restored entries comprise
120 BEV-encoder, 4 can-bus MLP and 14 image-neck/FPN parameters. This verifies the
bounded coverage repair. It does not establish nonzero gradients, update
magnitudes, use of every trainable parameter or numerical/quality equivalence.

| Original model | Model / optimizer leaves outside tolerance | Maximum model / combined tensor difference | Maximum loss difference |
| --- | ---: | ---: | ---: |
| VADv2 100% | 252 / 285 | 0.0002559646964073181 / 0.023486733436584473 | 0.050566673278808594 |
| HydraMDP 100% | 274 / 320 | 0.0004060305655002594 / 0.31602728366851807 | 0.04635047912597656 |

Both complete original replay comparisons remain outside tolerance. VAD
compared 889 model/1698 combined tensors; Hydra compared 909/1778. Combined counts
include 32 rank RNG tensors. Original tensor `atol=1e-6, rtol=1e-5` and loss
tolerance `1e-4` were retained; tokens,LR and final RNG matched. Coverage repair
does not resolve the separate strict-replay discrepancy.

| Model | Official initializer, BF16 inference | Restored update-8 checkpoint, BF16 inference |
| --- | ---: | ---: |
| VADv2 | 0.8704142103684525 | 0.9057495010259622 |
| HydraMDP | 0.9502729247085159 | 0.9562358040305806 |

These 22-token measurements have no quality acceptance threshold. Eight updates
remain inside 500 warmup updates; they are not eight epochs or convergence.
BF16 remains opt-in and experimental. The separate 37/22 real-log fine-tuning
run is not included in this baseline result. Evidence: [complete BF16 v7
short-run record](results/bf16-v7-navtrain139-short8-20261016.json) and
[all-rank/saved-Adam coverage verification](results/bf16-v7-gradient-coverage-20261016.json).

## Real-log fine-tuning: BF16 v7

The original 50% rare-log configurations used the separate NAVTEST-derived
37-train/22-evaluation subset, official 50% initializers and the `e14dd6b2…`
source with the same repaired `3522f3ac…` image. Both ran on 16 logical NPUs
with batch 1 per rank and workers 0. Original LoRA and frozen BatchNorm were
preserved: VAD uses reward/LoRA with `rl_finetuning=False`; Hydra enables its
original policy-gradient loss. These tests add no synthetic-data coverage.

Each completed continuous updates 0–8 and a fresh-process restore of that
run's own update-4 checkpoint through update 8. There was no separate partial
training run or initializer-baseline inference. All four training jobs, 64
rank reports and 32 exact-restoration reports passed. Both original strict
comparisons were bitwise equal at the unchanged tensor/loss tolerances.

| Configuration | Model / complete tensor comparison | Maximum tensor / loss difference | Independent evaluation |
| --- | --- | --- | --- |
| VADv2 rare-log LoRA/reward, BF16 v7 | 989/989; 1321/1321 bitwise equal | 0 / 0 | Complete 22-anchor inference and original NAVSIM scoring; PDMS 0.8631407605887709 |
| HydraMDP rare-log LoRA+PG, BF16 v7 | 989/989; 1261/1261 bitwise equal | 0 / 0 | [October 17 follow-up](results/hydra-ft37-bf16-eval22-20261017.json): complete 22-anchor inference and official scoring, PDMS 0.9338557270555728; the original interrupted attempt is retained |

The complete tensor counts include per-rank RNG tensors. The original Hydra progress
counter was not a saved prediction count; no complete 22-token prediction or
score is accepted from that interrupted attempt. Its separate OOM state was not retained. The watchdog
reported no stop actions and had not reached its cutoff, so the interruption
is not attributed to a watchdog kill. All nine registered containers were
stopped, and source/data/weights/cache hashes were unchanged.
The [terminal record](results/real-log-ft37-bf16-v7-short8-20261016.json)
therefore contains one complete evaluation chain and two complete replay
comparisons. It does not resolve the base-planner comparisons above, establish
quality improvement or extend the eight-update schedule to convergence.

## FP32 / BF16 trajectory measurements

Four CPU comparisons measured the same 22 held-out anchors, 40 poses per anchor:
880 pose comparisons per model/stage and 3520 in total. The official-weight
stages use identical checkpoint bytes and isolate inference precision. The
adapted stages compare each precision's own eight-update/restored checkpoint;
they include training-run differences and are not isolated inference effects.
FP32 used source `4ae8ed9f…`, BF16 v7 used `e14dd6b2…`, both with image
`3522f3ac…`; the exact two-file/14-EOL source difference was checked.

| Model / weights | Same selected candidate, out of 22 | Position distance mean / p95 / max, metres | Heading difference max, radians | Official PDMS FP32 / BF16 |
| --- | ---: | ---: | ---: | ---: |
| VADv2 /same official initializer | 20 | 0.0175644 /0.186362 /0.509340 | 0.0265967 | 0.869621 /0.870414 |
| VADv2 /each own eight-update checkpoint | 16 | 0.0889670 /0.480473 /1.382480 | 0.0805511 | 0.903387 /0.905750 |
| HydraMDP /same official initializer | 17 | 0.1438604 /0.589519 /6.578134 | 0.2058898 | 0.949351 /0.950273 |
| HydraMDP /each own eight-update checkpoint | 15 | 0.1173387 /0.655239 /1.039315 | 0.0785244 | 0.954841 /0.956236 |

Values are rounded here; the [complete measurement record](results/fp32-bf16-v7-matched8-planning-20261016.json)
retains full precision, all 40-pose differences and all seven official metrics.
Position is Euclidean distance in metres; heading differences are wrapped to
the circular angular distance. The original official scorer uses eight poses
selected from each 40-pose trajectory, while this table's trajectory statistics
cover all 40. All four CPU jobs completed with unchanged inputs and owned
resources released. No quality-equivalence threshold was specified. Similar
aggregate scores do not establish trajectory equivalence, improvement or
generalization; these are NPU FP32/BF16 measurements, not GPU comparisons.

The [public CPU comparison tool](../tools/validation/README.md) recomputes these measurements from original prediction exports and official score CSVs; it also accepts GPU/NPU pairs.

## BF16 v6 short-run record — superseded as training acceptance

Both original 100% planning configurations ran on 16 logical NPUs/eight
physical cards, batch 1 per rank, global batch 16 and workers 0. The inputs
were the deterministically derived NAVTRAIN139/NAVTEST22 subset and official
100% planning initializers. The original backbone-freeze configuration and
8192-candidate labels/order were retained; this was not fully unfrozen training.

The measured runtime fingerprint was
`4ae8ed9f3d34c88c7122b8d08df9b4ab9f5817bab794c0d429da3f3125e5fa5e`
(244 Python files), and the public dependency-repaired image was
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`.
BF16 autocast used FP32 model/optimizer master parameters and the recorded
FP32 geometry, loss and normalization paths. TASK_QUEUE_ENABLE and
ASCEND_LAUNCH_BLOCKING were unset. HF32 was disabled, Torch deterministic
algorithms were enabled, CLOSE_MATMUL_K_SHIFT=1 and HCCL_DETERMINISTIC=true.

For each model, the completed chain comprised a one-update cold-start check,
continuous updates 0–8, an independent interruption at update 4 under the same
eight-update schedule, and restoration of that checkpoint to update 8. All
eight training containers exited 0 without OOM; 128 rank runtime reports and
32 exact restoration reports passed. Four inference jobs and four official
CPU scoring jobs each covered the same 22 held-out tokens. Source, input,
initializer and metric-cache hashes remained unchanged, and owned jobs exited.
The recorded gradient-dtype checks covered only gradients that were present.
Subsequent read-only checkpoint inspection confirmed an optimizer-state
coverage defect relative to matched FP32 checkpoints:

| Model | FP32 parameter IDs with Adam state | BF16 v6 parameter IDs with Adam state | FP32-only state IDs |
| --- | ---: | ---: | ---: |
| VADv2 | 259 | 121 | 138 |
| HydraMDP | 279 | 141 | 138 |

The groups, parameter-ID order, hyperparameters, source, config, inputs, seed
and eight-update horizon matched; only the precision contract differed. Both
models had the same 138 missing IDs. This is cumulative optimizer-state
coverage evidence, not a live-gradient comparison or a parameter-name mapping
inferred from `state_dict` order. Frozen or unused parameters can legitimately
lack state; the test does not require state for every optimizer-group ID.

A subsequent real CPU Torch 2.7.1 probe reproduced a relevant autocast-cache
mechanism in both Linear and Conv2d toy models: history execution under
`no_grad` cached shared weights without gradients; the current head still
updated, while the shared layer had no gradient or Adam state. Disabling or
clearing the cache restored coverage in those toy cases. This establishes the
CPU mechanism, not its whole-model cause or an explanation of earlier FP32
replay differences. The v6 runs below remain execution records, not accepted
BF16 training correctness. They are superseded as BF16 training acceptance.
The subsequent cache-disabled v7 coverage repair and complete short-run results
are recorded above; they do not change these historical v6 outcomes.

| Original configuration | Replay result | Maximum model / combined tensor difference | Maximum loss difference | Model / optimizer leaves outside tolerance |
| --- | --- | ---: | ---: | ---: |
| `worldengine/e2e_vadv2_100pct.py` | Completed, outside tolerance | 0.000337112694978714 / 0.012727420777082443 | 0.04748106002807617 | 119 / 140 |
| `hydramdp/e2e_hydramdp_100pct.py` | Completed, outside tolerance | 0.00038817524909973145 / 0.09518872201442719 | 0.03777027130126953 | 136 / 153 |

The unchanged comparator used tensor `atol=1e-6, rtol=1e-5` and loss tolerance
`1e-4`. VAD compared 889 model tensors and 1284 combined tensors; Hydra compared
909 and 1364. The combined count includes optimizer state and 32 rank RNG
tensors. Tokens, learning rates and final rank RNG states matched in both
comparisons. Successful checkpoint restoration does not imply replay within
tolerance.

| Model | Official initializer, BF16 inference | Restored update-8 checkpoint, BF16 inference |
| --- | ---: | ---: |
| VADv2 | 0.8704142103684525 | 0.9036690039512171 |
| HydraMDP | 0.9502729247085159 | 0.9552234684337766 |

These are measurements on 22 tokens, without a quality acceptance threshold.
The eight updates remain inside the original 500-update warmup. They are not
eight epochs, convergence evidence or evidence of a quality improvement.
The matching FP32 short-run chain is recorded above; no valid FP32/BF16
equivalence is accepted from the defective v6 run. The separate original-CUDA
FP32/official100 comparison is recorded above; it does not validate these
historical BF16 outputs or establish general GPU/NPU equivalence.

Evidence: [complete BF16 short-run record](results/bf16-navtrain139-short8-20261016.json),
[actual CPU cache-mechanism probe](results/bf16-autocast-history-cache-cpu-20261016.json),
[matched checkpoint coverage](results/bf16-v6-optimizer-coverage-20261016.json),
[earlier one-update checks](results/bf16-official100-navtrain139-cold-20261016.json),
[public environment checks](docker/dependency_repair/BUILD_VALIDATION.json),
[data reproduction](data_tools/navtrain139/VALIDATION.json) and
[official input origins](evidence/official-data-origins-20261016.json).

## Historical public recipe and CLI reproduction — October 14, 2026

The published parent/child Docker recipes and AlgEngine entry points completed an actual reproduction from public commit `ec153c1`, runtime fingerprint `cc91deb8` (243 Python files). The SDK base was a verified cached official image; both recipes were rebuilt with `--no-cache`, using host networking for the parent build. The new training image was `sha256:1541450982f8c2938f3c387b929ad0ba8a90024d73712b3eabe49631812d12ce`. CPU data preparation used the separately identified `ba17931d` image. The published runtime and recipes were unchanged. `pip check` returned 1 with dependency warnings, including the MMCV protobuf constraint; the executed AlgEngine commands passed, but package metadata consistency is not claimed.

The public downloader verified official NAVTEST annotations/PDM/vocabulary by whole-file hashes. Existing genuine mini JPEGs were copied into the exact original camera-directory layout after frame-token/camera-key, calibration and image-hash checks. The unchanged preparation/split tools produced 59 anchors, 94 frames and 752 JPEGs: 37 training candidate anchors and 22 held-out anchors in disjoint driving logs. The selected rare pool contained one anchor with 21 valid same-log normal candidates. This differs from the historical 42/22 subset.

Both original 50% rare-log LoRA fine-tune configurations completed continuous updates 0–8 and fresh-process restoration of their own update 4 checkpoint to update 8 on 16 logical NPUs/eight physical cards, FP32, batch 1 per rank and workers 0. All 64 rank runtime reports and 32 exact restoration reports passed. TASK_QUEUE_ENABLE and ASCEND_LAUNCH_BLOCKING were unset; CLOSE_MATMUL_K_SHIFT=1, HCCL_DETERMINISTIC=true and both HF32 flags were false before/after training. VAD retained its configured zero PG term; Hydra's configured PG term was nonzero.

| Configuration | Model bitwise equality | Model/optimizer/RNG bitwise equality | Maximum tensor/loss difference | Official NAVSIM score / 22 valid tokens |
|---|---:|---:|---:|---:|
| VADv2 rare-log LoRA FT |989/989|1321/1321|0/0|0.8850917719220494|
| HydraMDP rare-log LoRA FT |989/989|1261/1261|0/0|0.9340064182046998|

The original comparator retained tensor atol 1e-6 / rtol 1e-5 and loss tolerance 1e-4. Both models completed 22 held-out trajectory predictions and separate official NAVSIM rescoring. All 14 stages exited 0 without OOM, source hashes remained unchanged, and owned jobs exited. These are one pair per real-log fine-tune configuration; they do not resolve the intermittent replay issue in base training with the original backbone-freeze configuration or establish quality improvement, full NAVTRAIN convergence, genuine synthetic-producer coverage, GPU parity or sustained operation of that image.


[Versioned reproduction evidence](results/actual-public-reproduction-16chip-20261014.json).

## Tested runtime and version boundaries

### Earlier base-planner control environment

| Item | Recorded configuration |
| --- | --- |
| Runtime Python fingerprint | cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88 |
| Fingerprint scope | 243 files: recursive AlgEngine Python and top-level Ascend Python, LF-normalized |
| Image ID | sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45 |
| PyTorch / Torch-NPU | 2.7.1+cpu / 2.7.1.post8 |
| Torch-NPU git version | a783f11e456f27d7451a1ea745f1556823357c95 |
| Recorded host stack | Ascend 910C, reported chip model Ascend910_9362; CANN 9.1, driver 26.1.1 |
| Topology | One server, eight physical dual-chip cards, sixteen logical chips/ranks |
| Batch and loader workers | B1 per rank, global batch 16, workers 0 |
| Precision | FP32; public Matmul HF32 and Conv HF32 both disabled |
| Determinism request | Torch deterministic algorithms enabled; CLOSE_MATMUL_K_SHIFT=1 and HCCL_DETERMINISTIC=true |
| Synchronous candidate settings | TASK_QUEUE_ENABLE=0, ASCEND_LAUNCH_BLOCKING=1, set before Torch startup |
| Data | Genuine NAVTEST-derived mini: 42 training anchors in two logs, 22 independent evaluation anchors in a third log |
| Trajectory vocabulary | Original 8192 candidates, each 40x3; labels and original model geometry retained |
| Checkpoint control | Continuous updates 1–8; own update 4 checkpoint restored to update 8 |

The runtime fingerprint identifies the measured Python subset. It is not a hash
of every documentation, shell or data-preparation file in a later checkout.
Public installation and launcher additions have separate local checks; a newly
rebuilt image requires its own identity and actual run evidence.

| Version group | Evidence assigned to that version |
| --- | --- |
| Historical fused path, 96a7bef… / a182… / post6 | Original 44-config catalogue and 36 real-config short runs, native mini epochs and evaluators; large MSDA location-gradient gate remained outside tolerance |
| Native MSDA path, 858f9d… / a182… / post6 | Expanded CPU32 MSDA gates, 36 real-config regression, single-rank controlled replay, independent planning scoring, original dataset construction and eight-hour perception stability |
| Earlier post8 runtime, cc91deb… / 3fef77b… / post8 | Default and synchronous sixteen-rank VAD/Hydra controls and full-rank state diagnosis |
| Earlier post8 runtime, cc91deb… / ba17931… / post8 | Genuine SimScale subset: original dataset access and one/two-rank controls for two configurations |
| Public recipe rebuild, cc91deb… / 15414509… / post8 | Two original LoRA fine-tuning configurations: 16-rank train/own-checkpoint replay, independent inference and official scoring; CPU preparation used ba17931 separately |
| BF16 runtime, 4ae8ed9… / 3522f3ac… / post8 | Two original 100% planning configurations: one-update checks, eight-update train/restore, independent 22-token inference/scoring; both final replay comparisons outside original tolerance |

The genuine SimScale image ID is
sha256:ba17931d5af64c6070417bc2059887096abf42d76b44484a9bd45262e3918707.
Its results are separate from the 3fef planning controls. Historical results are
retained under their own source and image; they are not measurements of the
current sixteen-rank environment.

## Numerical correctness

### Initial synchronous sixteen-rank controls

Each model completed one independent continuous/resumed pair. Both training
containers exited 0, all 32 rank reports passed, and all 64 before/after precision
snapshots matched the declared FP32 settings. Each resumed run used its own
continuous run's update 4 checkpoint.

The original comparison rule was unchanged: tensor atol 1e-6 / rtol 1e-5, and
loss absolute-difference tolerance 1e-4.

| Model | Training/resume completion | Final numerical result | Model maximum absolute difference | Combined maximum difference | Maximum loss difference | Bitwise-equal model / combined tensors |
| --- | --- | --- | ---: | ---: | ---: | --- |
| VAD | Continuous 8; own checkpoint 4→8 | Within original tolerance; all bitwise equal | 0 | 0 | 0 | 889/889; 1698/1698 |
| Hydra | Continuous 8; own checkpoint 4→8 | Within original tolerance; not all bitwise equal | 2.9802322387695312e-8 | 2.9802322387695312e-8 | 0 | 868/909; 1468/1778 |

Combined counts include model, optimizer and two RNG tensors per rank. Batch
tokens, learning rates and final rank RNG states matched. No model or optimizer
tensor exceeded tolerance. These initial results are one positive pair per model. The independent repeats
below supersede the earlier absence of repeat evidence; sustained-operation
acceptance remains open.
[VAD evidence](results/sync-vad-16chip-20261014.json),
[Hydra evidence](results/sync-hydra-16chip-20261014.json).

### Repeated public synchronous controls

The published synchronous adapter was executed for four further base-planner
pairs on the recorded cc91/243 source, post8/3fef image and historical 42/22
real-data split. All eight training jobs completed updates 0→8 or own checkpoint
4→8; 128 rank reports passed, and all 64 restored rank states matched the saved
checkpoint. The four original CPU comparisons completed with unchanged gates.

| Pair | Training/resume | Model maximum difference | Combined maximum difference | Model/optimizer tensors outside tolerance | Numerical result |
| --- | --- | ---: | ---: | ---: | --- |
| VAD / 1 | Completed 0→8 / 4→8 | 1.2110918760299683e-5 | 1.2110918760299683e-5 | 5 / 3 | Outside original tolerance |
| Hydra / 1 | Completed 0→8 / 4→8 | 9.164214134216309e-7 | 6.0823484091088176e-5 | 0 / 20 | Outside original tolerance |
| VAD / 2 | Completed 0→8 / 4→8 | 0 | 0 | 0 / 0 | Within original tolerance; all bitwise equal |
| Hydra / 2 | Completed 0→8 / 4→8 | 1.4901161193847656e-8 | 1.4901161193847656e-8 | 0 / 0 | Within original tolerance |

Batch tokens, learning rates and final rank RNG states matched in all four
pairs. These results do not support treating synchronous execution as a general
repair. [Repeated public-adapter evidence](results/public-sync-repeated-16chip-20261014.json).

### CPU input and optimizer diagnosis

A separate default-execution Hydra pair compared complete CPU inputs at all
16 ranks and updates 5–8. All 64 value-tree and exact metadata joins matched;
the original final comparison still contained 14 model and 21 optimizer paths
outside tolerance. Model and combined maximum differences were
1.3105571269989014e-5 and 7.020923658274114e-5. CPU hashing changes launch timing;
it does not establish equality of NPU intermediates or backend state.
[Full CPU-input evidence](results/default-cpu-full-input-hydra-16chip-20261014.json).

A private `grad_clip.foreach=False` candidate was tested separately. Hydra passed
one pair bitwise; VAD retained three optimizer tensors outside tolerance. All
observed gradient norms were below the configured limit 35, so these results do
not demonstrate a rounding error in an applied clipping coefficient. The
candidate was not promoted to the public training code.

A further diagnostic used the private 72c288c8 source variant and the 3fef
image. It captured complete real Adam inputs and outputs for three
selected VAD parameter tensors at updates 5 and 8. Four captured input states
were each replayed four times: 48 single-parameter updates, 192 CPU FP32 output
checks and 144 repeat-output checks passed the original tensor tolerance.
Maximum CPU difference was 2.9802322387695312e-8; all outputs matched their
corresponding captured NPU output bitwise. In this observed run, the update-8
gradient already differed before Adam; its maximum difference was
4.547473508864641e-13. This narrows that observed difference to an upstream input
and does not identify the cause of the earlier unobserved, above-tolerance
comparisons. NPU-to-CPU observation changes execution timing.
[Real Adam diagnostic](results/real-adam-vad-diagnostic-20261014.json).

### Default sixteen-rank controls

The same runtime, image and tolerance were tested with TASK_QUEUE_ENABLE and
ASCEND_LAUNCH_BLOCKING unset. All eight training/restoration jobs reached their
planned endpoints, exited 0 and passed exact saved-state loading and runtime
checks. Every pair executed the final CPU comparison at update 8.

| Model / repetition | Training/resume completion | Final numerical comparison | Model maximum absolute difference | Combined maximum difference | Maximum loss difference | Model / optimizer tensors above tolerance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| VAD /1 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 4.231929779052734e-6 | 5.522248102352023e-6 | 1.9073486328125e-6 | 6/7 |
| Hydra /1 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 6.511807441711426e-6 | 1.5364399587269872e-5 | 1.9073486328125e-6 | 7/14 |
| VAD /2 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 2.5041401386260986e-5 | 4.742613236885518e-5 | 1.9073486328125e-6 | 12/25 |
| Hydra /2 | Continuous 8; own checkpoint 4→8 | Within original tolerance; bitwise equal | 0 | 0 | 0 | 0/0 |

All four short controls completed. The three out-of-tolerance results concern
the final model/optimizer comparison, rather than incomplete training. Tokens,
LR and final RNG matched. Small maximum differences do not replace the
elementwise absolute/relative rule. Completion of eight updates does not mean
completion of NAVTRAIN training.
[Default-execution evidence](results/strict-replay-post8-20261014.json).

The cause of default-execution divergence remains unresolved. A separate
full-rank state diagnosis completed both models' controls: recorded registered
model/optimizer values and layouts before update 5 matched, while later
differences remained. The observer can change execution timing and does not
cover all unregistered caches or backend state.
[State-diagnosis evidence](results/full-rank-state-diagnostic-20261014.json).

### Incomplete and initialization-blocked tests

| Test | Recorded progress | Numerical conclusion |
| --- | --- | --- |
| Separate limited VAD control | Continuous run completed 8; own-checkpoint resume completed 4→6, exited 137 without OOM | Incomplete; target 8 was not reached and the final comparator was not executed |
| Cold start on another host | Driver device initialization blocked with EL0020 /507899; zero updates | Training not started; no final numerical comparison |

These tests are distinct from the completed planning controls above. The second
host subsequently passed initialization and the public-rebuilt fine-tuning
workflow; the earlier initialization-blocked record remains historical.

### MSDA and other compatibility mechanisms

On source 858f9d… / post6, the native NPU MSDA path passed the original CPU32 oracle
at unchanged tolerances: forward atol/rtol 2e-5/2e-5; value, location and attention-weight
gradients 1e-3/1e-4. Tested coverage includes eighteen ordinary cases through two
APIs, four legacy-layout cases and complete Q40000 geometry. The original
200x200/Q300/H8/C32/seed29 location-gradient case has 0/19200 out-of-tolerance
elements. Native grid-sampler device execution was observed.
[MSDA report](MSDA.md), [strict matrix](evidence/msda-strict-matrix-20261013.json),
[legacy layout](evidence/msda-map-layout-20261013.json),
[full shape](evidence/msda-full-shape-20261013.json).

| Mechanism | Implemented behavior and boundary |
| --- | --- |
| NPU attention compatibility | NPU-native grid sampling and first-order backward retain the MMCV reference coordinate convention, zero padding and reduction semantics. Legacy projected-L4/actual-L1 input uses the restricted CUDA-source flat-prefix interpretation; actual GPU parity is untested. |
| ResNet MaxPool | Uses the differentiable NPU-dispatched PyTorch path instead of the SDK forward-only fusion. |
| Complete checkpoints | Preserve model, optimizer, per-rank RNG, sampling/LR progress and source/image/data/precision contracts. Exact-state continuation requires workers 0; exact loading and subsequent numerical replay are separate checks. |
| CPU checkpoint comparison | Uses the installed official Torch-NPU serialization registration before loading legacy NPU storage on CPU, without NPU initialization; original numerical thresholds remain unchanged. |
| Original evaluation | Detection/mapping evaluators and pinned NAVSIM scoring are retained. Planning scoring uses the original 40-to-8 pose conversion and independent scenario caches. |
| Matrix inversion | An existing CPU fallback keeps the full model executable; its performance cost has not been optimized or accepted. |

MSDA floating computation remains on NPU; small integer shape metadata can be
read on CPU. This operator statement does not imply every full-model operation
runs on NPU. MSDA agreement has not identified the cause of default full-model
divergence. [Versioned native-path evidence](evidence/msda-precision-20261013.json).

## Configuration and data-path coverage

The original catalogue contains 44 configurations: 42 planning configurations
with original B2, and two perception configurations with original B1.

| Coverage | Completed validation | Version and limit |
| --- | --- | --- |
| Catalogue | 44 configurations resolved | Original catalogue; configuration resolution is not training acceptance |
| Real-data configurations | 36/36 completed three optimizer updates each, preserving original geometry and publisher-verified initializers | Historical 96a and native 858/post6 matrices; B1/workers 0/real mini and short schedule, not all 44 current-post8 full training |
| Base planning and real-log FT | Original training, LoRA/reward FT, Hydra policy-gradient, inference and checkpoint paths exercised | Config-specific small-data tests; VAD rare-log reward/LoRA has rl_finetuning=False, while Hydra's base enables policy gradients |
| Configurations requiring synthetic inputs | Historical narrow tests of two SimScale entries: original CPU dataset access and one/two-rank continuous 8/own checkpoint 4→8 controls | cc91/post8 on ba179; one published synthetic scene, narrowed normal pool and explicitly derived WE rewards. The six WorldEngine rollout/replay/BWM entries remain unverified with genuine exports |
| Independent planning scoring | Three representative planners processed 22 held-out anchors each and completed original official CPU scoring | Native 858/post6; small common-log evaluation, no complete rare benchmark or GPU comparison |

The eight entries comprise two SimScale configurations and six WorldEngine
rollout/replay/BWM configurations. The latter still require their genuine
producer exports. The two historical SimScale tests used one published scene
and the declared WE four-second, 8192-candidate derived-reward protocol; they
do not establish complete published-protocol or full-data sampling acceptance.
Their one/two-rank comparisons were bitwise equal at the original gate, with
both real-normal and synthetic branches observed. The separate NAVTEST-derived
37-train/22-eval rare-log LoRA/PG tests use real logs and do not add synthetic
configuration coverage. SimEngine rendering was not executed.
[Catalogue](evidence/original-configs-20261013.json),
[real-config/native-path results](evidence/msda-precision-20261013.json),
[CPU data access](data_tools/GETITEM_CPU_VERIFIED.json),
[single-rank controls](data_tools/NPU_SINGLE_RANK_VERIFIED.json),
[two-rank controls](data_tools/NPU_TWO_RANK_VERIFIED.json).

The portable SimScale recipe uses approximately 47.50 GB of pinned inputs,
excluding maps, source, the container image and generated outputs.
It computes 8192 labels with original WE four-second logged-traffic mathematics
and an explicit history/observation input adapter. This does not establish
published SimScale-v2 reward or full-engine lifecycle equivalence. The published
16384-candidate reward has incompatible fields/candidate alignment and cannot be
made equivalent by truncation or renaming. Finetune explicitly overrides the
unavailable e2e_vadv2_ep8.pth basename with the published 100pct checkpoint.
[Input/derivation contract and recipe](data_tools/README.md),
[CPU reproduction evidence](data_tools/CPU_REPRODUCTION_VERIFIED.json).

### Historical quality and numerical observations

| Version / workload | Preserved result | Interpretation |
| --- | --- | --- |
| Historical 96a fused MSDA | 35/19200 large-shape location-gradient elements outside tolerance; maximum 0.0309448 | Superseded by the native-path gate; the original negative result remains recorded |
| Historical post6 short FT,22 held-out anchors | VAD PDMS 0.910051108772398→0.8762690973035009; Hydra 0.9340064182046998→0.9340064182046998 | No demonstrated quality benefit from that short FT |
| Historical 96a perception mini epochs | Detection mAP 0; NDS 0.01950/0.02179 | Evaluator executed; trained perception quality and convergence unaccepted |
| Historical 96a VAD reward/LoRA epochs | PDMS 0.8455829502665201→0.8850917719220494; rare evaluation tokens 0 | Small common-log result, without rare-case quality acceptance |
| Native 858/post6 eight-rank replay | Both models completed continuous 8/own checkpoint 4→8; VAD 12 model/21 optimizer and Hydra 2/2 tensors outside tolerance | Exact restoration passed; distributed numerical replay did not meet the original gate |
| Historical 96a DiffusionDrive, three updates | 22 finite 40x3 generated trajectories; official PDMS 0.377858654551, ADE/FDE 10.96562/19.69085 | Functional generated-trajectory scoring, not trained planner quality |

Sources: [early bounded results](validation_results_20261013.json),
[historical full-module evidence](evidence/full-algengine-20261013.json),
[native eight-rank replay](evidence/eight-rank-resume-20261013-evening.json),
[representative planning results](evidence/msda-planning-20261013.json).

## Full NAVTRAIN metadata and CPU capacity

Both official NAVTRAIN PKLs were fully transferred and publisher-hash verified;
their total serialized size is 39,862,469,866 bytes.

| Inspection | Measured result | What it establishes |
| --- | --- | --- |
| Raw annotation inventory | 246,165 unique frames,1,192 logs, 1,969,320 valid camera references | Required fields/path references; zero JPEG files read |
| Official split/PDM coverage | 103,288 split tokens and 103,288 PDM entries; no missing split tokens | Complete metadata/PDM token coverage |
| Six required PDM metrics | 619,728 finite float32[8192] arrays, values 0–1 | All six required numerical arrays checked unchanged; seventh driving_direction field received schema inventory only |
| Sequential raw-pickle QC | Peak RSS 34,747,150,336 bytes | CPU QC memory; not original dataset-constructor or training RSS |
| One resident original dataset | RSS 52,208,001,024 bytes; PSS 52,136,001,536 bytes | Original constructor/Compose residency |
| Eight resident original datasets | Summed RSS 416,754,966,528 bytes; PSS 413,451,488,256 bytes | Measured multi-process dataset residency |
| Sixteen resident original datasets | Summed RSS 834,131,996,672 bytes; PSS 826,988,978,176 bytes | Measured sixteen-process dataset residency |

Constructor tests used source 858/post6, the original 50% filter with 50,000
anchors, all 246,165 annotation frames and 103,288 PDM entries per dataset. An
explicit import-time SDK device-name bridge enabled CPU-only construction; no
images, model, NPU operations or loader-worker training were executed. These
measurements do not include full-image/model/optimizer/worker/evaluation memory.
The 103,288-anchor full training filter also needs its own runtime measurement.
[Complete metadata QC](results/full-navtrain-metadata-qc-20261014.json),
[original constructor capacity evidence](results/full-navtrain-dataset-construction-20261014.json).

The targeted current/history training camera archives total 449,046,313,693
compressed bytes. Existing disk-based pipelines require extracted, readable
JPEGs with complete four-frame/eight-camera coverage. Their extraction footprint
and full training I/O have not been measured. These archives are separate from
the roughly 5 TB SimEngine scene publication. PDM vocabulary labels and official
NAVSIM scenario caches are also separate inputs.

## Operational stability

The completed eight-hour perception experiment belongs to source 858f9d…
(242 files), image a182f52… and Torch-NPU 2.7.1.post6. It used eight physical cards,
one logical chip per card, B1/global 8 and workers 0 on the 42-anchor mini.

- Each rank completed 1023 updates; minimum measured training time was
  28,827.677454 seconds.
- Restoring the same run's final checkpoint completed 1023→1026, with 1393 model
  and 3594 optimizer tensors per rank checked exactly after loading.
- Independent perception evaluation covered 37 held-out current/history frames;
  two complete checkpoints passed CPU audit.
- All four training/resume/inference/audit stages exited 0 without OOM.
  All 16 rank streams were checked: 8208 contiguous update records and 837,216
  finite raw-loss scalars.
- Gradient checks ran at the first update and every tenth update. The CPU RNG
  audit checked field/type/count structure rather than equality. Strict
  continuous-versus-resumed numerical replay was not performed for this run.

Its measured peak allocated/reserved NPU memory was 46,908,357,632 /
52,208,599,040 bytes per chip; observed charged CPU cgroup peak was
74,654,064,640 bytes. These are mini-run measurements. This legacy run lacked
the current per-rank precision before/after instrumentation and does not accept
cc91/post8 sustained training,24-hour stability or a general memory-leak absence.
[Eight-hour evidence and full-stream supplement](results/eight-hour-perception-stability-20261014.json).

## Equipment and topology planning

| Workload | Measured device layout | Capacity boundary |
| --- | --- | --- |
| Small-data functional training | One logical chip on one 910C server | Measured functional cases; not a complete-data minimum guarantee |
| Current planning global batch 16 | One server, eight dual-chip cards / sixteen logical chips at B1 | Matches the original eight GPUs × B2 sample count; numerical/grouping equivalence with GPU is untested |
| Genuine SimScale two-rank subset | Two physical cards, one chip per card, B1/global 2 | Does not validate both chips of a single card as a two-rank layout |
| Eight-hour perception | Eight physical cards, one chip per card, B1/global 8 | Historical 858/post6 mini stability only |

The measured logical chips have 64 GiB HBM. A complete sixteen-rank training plan
must budget above the 834.13 GB measured dataset RSS for image/pipeline buffers,
model/optimizer host state, validation, workers and reserve. Full training RAM,
HBM headroom and storage/I/O must be measured together; serialized PKL size
cannot determine the device count or RAM requirement. Four physical cards with
eight chips at B2, larger per-chip batches and multi-node execution are not
accepted by these measurements.

Performance optimization is deferred. Historical speeds are workload-specific
observations and do not establish paired GPU/NPU throughput or cost advantages.

## Outstanding acceptance

| Area | Remaining requirement |
| --- | --- |
| Default distributed numerics | Identify the cause of completed default comparisons outside tolerance and repeat unchanged-gate controls |
| Synchronous candidate | Four independent pairs completed; two remain outside tolerance. Resolve the divergence and validate sustained operation of the same source/image |
| Complete training | Original 50%/100% NAVTRAIN horizons, convergence and target batch/worker settings |
| Independent quality | Complete NAVTEST common/rare evaluation, adequate rare coverage and agreed quality thresholds |
| Data and capacity | Full current/history JPEG coverage, image decode/pipeline quality, full training/evaluation RAM, HBM and I/O |
| Synthetic scope | Genuine producer exports and original training/evaluation coverage for the other six configurations |
| Precision and platform extensions | Resolve the completed BF16 replay comparisons outside tolerance; extend the fixed-subset FP32/BF16 and official100 GPU/NPU inference measurements to broader data and repeated runs; validate multi-node training and second-order gradients |
| Public environment reproduction | The October 16 public repaired image passed application, selected SDK-tool and original-CANN-source CPU checks, including two unfiltered pip checks with exit 0. All optional SDK tools and sustained operation of the final environment remain outside that acceptance |
| Sustained operation | 24-hour/target-duration testing with the same final source, image and execution settings |

FP32 is the default. BF16 remains opt-in and experimental while strict replay
is unresolved. Performance tuning is separate from these correctness checks.
Installation, data preparation, training, resume and scoring procedures are
maintained in the [AlgEngine operation guide](../docs/algengine_ascend.md).
The [synchronous protocol](execution/README.md) and
[portable SimScale tools](data_tools/README.md) define their narrower reproducible
inputs. Local tool/command checks do not constitute a new NPU acceptance run.
