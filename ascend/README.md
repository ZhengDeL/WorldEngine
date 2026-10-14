# AlgEngine on Ascend 910C / A3

Start with the [NPU usage guide](../docs/algengine_ascend.md) for installation, data preparation, training, resume, evaluation and fine-tuning. This document retains the versioned measured results.

## Current acceptance snapshot: October 14, 2026

AlgEngine can run the measured training, fine-tuning, inference and checkpoint
restoration workflows on the NPU. Complete migration acceptance remains open.
MSDA's strict location-gradient issue is resolved at the original tolerances;
the remaining sixteen-rank numerical failure has not been attributed to MSDA.

The current runtime under evaluation is `cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88`
(243 runtime Python files). On its post8 candidate image, the four fresh,
untraced sixteen-rank VAD/Hydra continuous-eight/own-checkpoint-four/resume-eight
pairs used the original unset queue/blocking environment and completed their
original runtime and saved-state restoration gates. The
unchanged final numerical comparator passed **one pair and failed three**.
The tensor tolerances remain `atol=1e-6, rtol=1e-5`; loss tolerance remains `1e-4`.
See the [untraced strict replay results](results/strict-replay-post8-20261014.json).

| Post8 sixteen-rank pair | Numerical result | Model maximum absolute difference | Complete comparison maximum difference | Model / optimizer tensors above tolerance |
| --- | --- | ---: | ---: | ---: |
| VAD, repetition 1 | Fail | 4.231929779e-6 | 5.522248102e-6 | 6 / 7 |
| Hydra, repetition 1 | Fail | 6.511807442e-6 | 1.536439959e-5 | 7 / 14 |
| VAD, repetition 2 | Fail | 2.504140139e-5 | 4.742613237e-5 | 12 / 25 |
| Hydra, repetition 2 | Pass, bitwise equal | 0 | 0 | 0 / 0 |

Complete comparisons include rank RNG tensors. A passing pair does not establish
a general determinism repair, and loading a checkpoint exactly does not ensure
that all subsequent computations reproduce a continuous run. Root cause remains
unconfirmed. The additional state diagnosis completed with failures below.
The first synchronous-execution control ended as a partial run; the later VAD
and Hydra candidates below each passed once. These are different runs. Partial progress,
timely device release or a single positive candidate does not establish a general
determinism repair.

| Acceptance area | Verified result | Work still required |
| --- | --- | --- |
| Genuine data | Both full NAVTRAIN PKLs, totaling 39,862,469,866 bytes, are present and hash-verified. Raw metadata QC covers 246,165 unique frames in 1,192 logs and all 103,288 official split tokens. All 619,728 required PDM arrays are finite `float32[8192]`, range 0–1. | Complete current/history camera coverage and independent full NAVTEST/cache preparation. The QC read **zero image files**; 1,969,320 valid camera references are not JPEG availability/decode acceptance. |
| Eight synthetic configurations | Portable preparation, original four-branch getitem, and one/two-rank own-checkpoint controls passed for **two configurations** using one genuine scene and WE-derived 8192 rewards. All tested tensors were bitwise equal. | The other **six rollout/replay/BWM configurations** lack complete genuine exports. The narrowed normal pool and four-second WE logged-traffic protocol do not establish published SimScale-v2 reward or full-data sampling equivalence. |
| Full training and independent quality | Small-data training, original evaluators and earlier independent official CPU scoring work. | Full NAVTRAIN horizons/convergence, independent full NAVTEST common/rare quality and GPU output comparison are unaccepted. |
| CPU/device capacity | Original full-PKL dataset/Compose construction passed with 1, 8 and 16 independent resident datasets under the original 50% filter: 246,165 frames, 50,000 anchors and 103,288 PDM entries per dataset. Sixteen-dataset summed RSS reached 834,131,996,672 bytes. | Actual full-image/model/optimizer/worker training memory and I/O capacity. The constructor used source858/post6 and an explicit SDK import-time metadata bridge; it is not an unshimmed CPU-only deployment or a training-memory result. |
| Stability and restoration | The source858/post6 eight-rank perception run exceeded eight hours, completed 1,023 updates per rank, restored its own final checkpoint for three more updates, completed 37-frame perception inference/evaluation and audited both checkpoints on CPU. | Target-source/post8 sustained training and the unresolved sixteen-rank strict numerical gate. No 24-hour, full-data or universal replay acceptance. |

The full metadata QC's 34,747,150,336-byte sequential process RSS is separate
from original dataset construction. One constructor peaked around 52.208 GB;
eight resident datasets reached 416,754,966,528 summed RSS bytes. The sixteen
dataset result does not include decoded full-training images or NPU model work.
The seventh `driving_direction_compliance` cache field received schema inventory
only; it must not be counted as a seventh numerically accepted metric.

The eight-hour result belongs to source `858f9d…` (242 files), image `a182…`
and `torch_npu 2.7.1.post6`. Minimum recorded rank training time was
28,827.677454 seconds. All four training/resume/inference/audit containers exited
0 without OOM; all sixteen original rank streams were checked on the server:
8,208 contiguous update records and 837,216 finite raw-loss scalars. The original
gradient guard checks the first update and every tenth update, not all gradients
at every step. CPU checkpoint RNG checks establish field/type/count structure,
not RNG equality or strict numerical replay. That old run has no new per-rank
precision before/after snapshots and cannot be reassigned to cc91/post8.

Portable data preparation has an independent three-CLI SHA manifest. Its recipe
requires 47.50 GB of pinned public inputs, excluding maps/source/image/output;
it does not require the incompatible 19.3 GB published 16384-reward file, all
449 GB of NAVTRAIN cameras or the 5 TB scene publication. Finetune uses the
published 100pct checkpoint as an **explicit override**; the original
`e2e_vadv2_ep8.pth` basename remains unavailable. Reproduction evidence is
`data_tools/CPU_REPRODUCTION_VERIFIED.json`, `GETITEM_CPU_VERIFIED.json`,
`NPU_SINGLE_RANK_VERIFIED.json` and `NPU_TWO_RANK_VERIFIED.json`; the full-data QC
and old eight-hour supplements are [complete NAVTRAIN metadata QC](results/full-navtrain-metadata-qc-20261014.json)
and [old-source eight-hour stability](results/eight-hour-perception-stability-20261014.json).

The original catalogue contains 44 configurations: **42 planning configurations
use batch 2 per GPU**, and two perception configurations use batch 1. For the
planning baselines, 16 NPU ranks at B1 match the original 8-GPU global sample
count of 16; they do not establish numerical or grouping equivalence. The
eight-rank old stability run used eight physical cards, one chip per card.
The genuine two-rank controls used two physical cards, one chip per card;
neither proves a one-card dual-chip topology. A four-card/eight-chip B2 capacity
configuration and multi-node extension still need their own acceptance.

Performance optimization remains deferred. The checks above concern correctness,
function, capacity and bounded stability; historical speeds are not paired
GPU/NPU benchmarks.


### Additional short-window controls: results still bounded

A follow-up post8 sixteen-rank state-observation run completed all four
VAD/Hydra training/restoration jobs and the CPU checkpoint identity audits.
Both original numerical comparisons failed. Each model completed 144 state
joins with zero blocked entries and 16 tensor-differing joins, all at final
update 8; see [the full-state diagnosis](results/full-rank-state-diagnostic-20261014.json). Recorded registered model/optimizer state and layout before update 5
matched, but later execution diverged. This does not establish equality of
unregistered caches or every backend state; root cause remains unconfirmed.
Source remained cc91/243, with no remaining owned jobs or cleanup errors. The
observer reads NPU state onto CPU and can change timing; this diagnosis is
separate from untraced production replay and is not evidence of a repair.

The separate synchronous-execution candidate
`TASK_QUEUE_ENABLE=0, ASCEND_LAUNCH_BLOCKING=1` failed initialization on the
second node with `EL0020 / Device0 occupied by multiple containers / 507899`.
It performed zero updates and produced no numerical comparison. No reset,
sharing-mode change or shared-workload stop was used. Healthy npu-smi output
and absent host PID/FD evidence alone did not establish usable driver state.

The independent untraced single-VAD synchronous control on the first node
ended as a partial execution. Its continuous run completed eight updates,
exited 0 and passed all sixteen original runtime gates. Resuming its own
checkpoint at update 4 completed only two further updates to 6. The original
helper's earlier 120-second cutoff stopped the owned resume job at 01:38:10 UTC;
its exit was 137 without OOM. It did not reach update 8 and the original strict
comparator did not run. This is neither a numerical pass nor a numerical fail,
and provides no Hydra or two-repeat acceptance.

That first short device window ended. All owned jobs were released with no
remaining owned work or cleanup errors. The independent watchdog completed
with an empty stops list; the earlier stop came from the original helper. The
last session closed at 01:39:59 UTC, before the 01:40:10 authorization cutoff.
The running v1 was not restarted or replaced to extend the window. The variant
with unified 60-second guards was not run in that first window.

In the subsequent window ending at 02:00 UTC / 10:00 CST, one fresh untraced
VAD pair completed continuous updates 1–8 and resumed its own update-4 checkpoint
through update 8. With initial `TASK_QUEUE_ENABLE=0` and
`ASCEND_LAUNCH_BLOCKING=1`, both training jobs exited 0 and all 32 rank runtime
reports passed. All 64 before/after precision snapshots recorded FP32 with
both public HF32 flags false and the expected deterministic environment.
The unchanged numerical comparator, using the official Torch-NPU CPU loader,
found all 889 model tensors and all
1,698 complete comparison tensors (including rank RNG) bitwise equal, with
maximum difference and loss difference both 0. Source remained cc91/243,
with no remaining owned jobs or cleanup errors. This is one VAD candidate
positive; see the [synchronous VAD result](results/sync-vad-16chip-20261014.json).

The separate Hydra pair also completed both training jobs and all 32 rank
runtime gates, including 64 matching before/after precision snapshots, with
the same synchronous environment and unchanged source/image. Its original
comparison passed at the original tolerances: model maximum and complete
comparison maximum were both `2.9802322387695312e-8`, and loss difference was 0.
868 of 909 model tensors and 1,468 of 1,778 complete comparison tensors were
bitwise equal; **Hydra was not fully bitwise equal**. Batch tokens, LR and final
rank RNG states matched. See the [synchronous Hydra result](results/sync-hydra-16chip-20261014.json).

These are one VAD and one Hydra candidate pair, one repetition each. They do
not establish four-pair repeatability, a root cause, universal determinism or
full-data acceptance. Both completed with no remaining owned jobs or cleanup
errors; the Hydra watchdog completed with an empty stops list. The complete
CPU-input diagnostic controller was prepared locally and never executed.
Production defaults and numerical tolerances remain unchanged.

For the bounded candidate, see [synchronous reproduction commands](execution/README.md). The public adapter has local command-construction evidence; the NPU results above came from the earlier controlled launch flow.


The current FP32 sampler and its strict-gradient/model regressions are documented
in [English](MSDA.md), [中文](MSDA_zh.md), and the
[versioned MSDA results](evidence/msda-precision-20261013.json). Measurements in
the original full-module result retain their historical source fingerprint.

**2026-10-13 evening acceptance update:** the historical `2bed8f4d7c78` /
`858f9d…` FP32 source completed all four eight-rank training/resume jobs and
all 32 runtime checks, including exact saved-state restoration. However, the
unchanged final numerical gate **failed for both VADv2 and HydraMDP**: respectively
12 model + 21 optimizer tensors and 2 model + 2 optimizer tensors exceeded the
original tolerance. Batch tokens, LR and final RNG states matched. See the
[eight-rank evidence and conditions](evidence/eight-rank-resume-20261013-evening.json).
At the October13 22:44 snapshot the planned perception run was still running.
The separately completed old-source eight-hour run is reported in the current
acceptance section above; the earlier interrupted run is not an eight-hour pass. Performance optimization is deferred
by the user, outside the current five correctness acceptance items; full-data
CPU RAM, I/O and device-capacity checks remain necessary.

[中文部署、训练、续训与验收操作文档](README_zh.md) ·
[English full-module guide](FULL_ALGENGINE.md) ·
[Bounded measured results](evidence/full-algengine-20261013.json)

This branch implements and exercises AlgEngine training, VADv2 reward/LoRA fine-tuning, HydraMDP policy-gradient fine-tuning, planning inference and complete checkpoint continuation. SimEngine remains the GPU subsystem and is outside this acceptance.

The expanded candidate accepts all 44 published AlgEngine configurations through
`--config`, including DiffusionDrive and the two UniAD perception pretraining
configurations. See [the full-module guide](FULL_ALGENGINE.md) for the additional
data requirements, actual tested revisions and pending acceptance. Whole
NAVTRAIN convergence and the eight genuine synthetic/SimScale configurations
are **not accepted**. The expanded source and each exercised revision are
distinguished in the evidence. See [the current MSDA supplement](MSDA.md) for
current regression and numerical results. The full-module guide preserves
historical measurements and provides full-data operation/acceptance templates.

Upstream: `3eded7e8ac64e5bed817860765089b5914ab698f`. The required official NAVSIM **v1.1** submodule is pinned to `3e8291bfa89ff247231e0227778840cd0a036896`.

## Earlier three-profile acceptance: 4fbd616 (2026-10-13)

This section records the earlier published revision. Current full-module results
are in [FULL_ALGENGINE.md](FULL_ALGENGINE.md) and the
[bounded evidence](evidence/full-algengine-20261013.json).

The full model geometry is retained: FP32 tensors, 200x200 BEV, 900 queries, four temporal frames, eight cameras and the 8192x40x3 trajectory vocabulary. The real subset is split by entire logs: **42 training anchors in two logs, 22 independent evaluation anchors in a third log**. Fine-tuning selects one genuine official failure anchor and one normal anchor. This is useful for functional acceptance, not a representative rare-case training benchmark.

These are the three original planning configurations named by the launcher:
`e2e_vadv2_50pct`, `e2e_vadv2_50pct_rlft_rare_log`, and
`e2e_hydramdp_50pct_rlft_rare_log`. Original freezing settings remain: imitation
freezes the image backbone; fine-tuning freezes perception and trains the
configured heads/LoRA. The measured trainable parameter counts were 33,716,740,
3,561,547 and 2,240,582 respectively. Full-network geometry does not imply that
all perception tasks or every parameter were trained.

| Item | Observed result | Boundary |
|---|---|---|
| Clean image | No-cache build from the exact imported official ARM64 base; archive save/load preserved its image ID | Same host; separate hardware/OS/driver installation untested |
| Published bases | VADv2 and HydraMDP epoch-8 checkpoints downloaded and publisher SHA256 checked; matched initialization tensors checked | No full-data convergence claim |
| Native epoch | 8 ranks, batch 2/rank, 8 complete mini epochs, 24 AdamW updates/rank | Original optimizer and epoch semantics; small data and workers=0 |
| Fine-tuning | VADv2 reward/LoRA and HydraMDP actual nonzero PG, 30 updates each, real images and PDM targets | Two training anchors repeat; VADv2 configuration has PG disabled |
| Resume | Model, optimizer, LR horizon, every rank's RNG, sampler progress and topology/data/source/image contract restored | Exact tensor restoration passes; numerical continuation has limits below |
| Independent scoring | Four model variants, each 22/22 valid outputs, unmodified official CPU NAVSIM scorer | No GPU parity or complete official benchmark |
| Portable data | 1,120 referenced original JPEGs, labels and split filters, 242.3 MB compressed; NPU run passed without old camera directory mounts | Labels retained; small-data scope |

Eight ranks completed 460 updates each, with at least 3609.89 seconds of measured training per rank. Observed steady end-to-end throughput was 2.042 anchor exposures/s; median compute time 5.610 s/update. Peak allocator usage was 27.04/38.34 GiB allocated/reserved per logical device. This is repeated small-data FP32 training with monitoring and frequent sampler epoch rollover, not a full-data or GPU performance benchmark.

Independent official small-data PDMS was **0.910051 / 0.876269** for VADv2 base/fine-tuned and **0.934006 / 0.934006** for HydraMDP base/fine-tuned. VADv2 degraded on this small log; HydraMDP selected the same trajectories in this evaluation. Functional fine-tuning is demonstrated, quality improvement is not accepted.

### Numerical continuation boundary

All resumed cases strictly checked every saved model and optimizer tensor immediately after loading. VADv2 LoRA's continuation passed `atol=1e-6`, `rtol=1e-5` and loss tolerance `1e-4`. Base-model continuation and HydraMDP PG did **not** pass the same complete numerical gate. Data tokens, LR and final RNG states match. HydraMDP's maximum parameter difference was `2.34818435e-06`; controlled eight-rank base continuation had parameter/loss differences `0.000323150307` / `0.000865221024`.

An isolated fixed-input MSDA repeat had identical forwards but up to `4.76837158203125e-7` variation in the value gradient, including with deterministic algorithms requested. The official DrivingSDK kernel contains floating-point atomic accumulation. This establishes an operator repeatability limitation; it does not attribute every model difference to that operator. `--deterministic` requests determinism and cannot guarantee it for every custom operator. Failed gates are retained.

## Source and runtime

```bash
git clone --branch ascend-910c-algengine --recurse-submodules https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git submodule status
```

The newly built and tested image is local:

```bash
IMAGE=local/worldengine:26.1.0-a3-clean-20261013
IMAGE_ID=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
docker image inspect --format '{{.Id}}' "$IMAGE"
```

Image archive: **23.444539904 GB**, SHA256 `5c730fd509d9dea6ac1599094c7e1e60ab0fad78996031fcb46e7571988c5cac`. Import with `docker load -i worldengine_clean_20261013.tar` and verify the image ID. No public registry upload is implied. The launcher now defaults to the clean image and its exact image ID; prior image information remains in the lock for historical runs.

The executed recipe is `ascend/docker/Dockerfile`, with `--no-cache` and an imported official base whose config ID is `sha256:1a63c030ea4036467cfc486bef006acf6f1cfbb1cc3b0dc362d07f86b84349bd`, equivalent to the pinned official ARM64 manifest in `runtime.lock.json`. A registry-connected host can build from the Dockerfile's pinned official manifest. Rebuilt image IDs are locked separately; channel resolution is not a bit-identical image claim.

CANN 9.1.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, DrivingSDK 26.1.0.post2, MMCV 1.7.2, MMDet 2.24.0, MMDet3D 1.0.0rc4 and NumPy 1.23.5 were exercised. Torch's `+cpu` version string is expected for this NPU stack. In that historical post6/source858 runtime, matmul HF32 was disabled and convolution HF32 allowed. The current cc91 FP32 mode explicitly disables both and records actual per-rank before/after values. FP32 still does not establish GPU numerical equivalence; checkpoints from a different source/precision/backend contract must not bypass the resume guard.

Observed package versions are retained in `docker/versions-20261013.json`; the
separate geospatial Conda prefix's resolved ARM64 packages and package checksums
are in `docker/geo-linux-aarch64-20261013.explicit.txt`. Its Python patch version
can differ from the main training prefix. These are build snapshots; the exact
saved image is the verified offline reproduction artifact.

The tested host was Atlas 800I A3 inference series, Ascend910_9362, with 64 GiB HBM per logical chip.

## Small real data and official weights

`fetch_assets.py` downloads only the two released planners, approximately 886 MB, with byte-range resume and final publisher hashes. The 5 TB publication is not downloaded.

```bash
python3 ascend/fetch_assets.py --output /datasets/worldengine/weights
```

Use the delivered `mini_real_data.tar.gz` or create a log-disjoint split from an already prepared official subset:

```bash
python ascend/split_mini.py --prepared /datasets/worldengine/prepared \
  --output /datasets/worldengine/log_split \
  --rare-split projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml
```

The manifest's phase-specific annotation/filter/PDM files and real `cameras/` must exist. The portable packet uses relative camera paths. Camera path changes alter annotation hashes: start a new run with a relocated/repacked dataset rather than changing data under a saved resume contract.

## Run, save and resume

```bash
MINI=/datasets/worldengine/mini_real_data
VAD=/datasets/worldengine/weights/e2e_vadv2_50pct_ep8.pth
HYDRA=/datasets/worldengine/weights/e2e_hydramdp_50pct_ep8.pth
VOCAB=/datasets/worldengine/test_8192_kmeans.npy
RUNS=/experiments/worldengine

# Keep the schedule horizon at 30; stop at 10 without shortening cosine LR.
python3 ascend/run_container.py --devices 0 --profile vadv2_ft --runtime \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --checkpoint "$VAD" --vocab "$VOCAB" \
  --output "$RUNS/vad_partial" --steps 30 --stop-after 10 \
  --batch-size 2 --workers 0 --deterministic --checkpoint-interval 10 --detach

# Wait for exit_code=0 and a complete iter_10.pth, then use a NEW output.
python3 ascend/run_container.py --devices 0 --profile vadv2_ft --runtime \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --checkpoint "$VAD" --vocab "$VOCAB" \
  --output "$RUNS/vad_resumed" --steps 30 --resume "$RUNS/vad_partial/iter_10.pth" \
  --batch-size 2 --workers 0 --deterministic --checkpoint-interval 10 --detach
```

For HydraMDP select `--profile hydramdp_rl` and its matching `--checkpoint "$HYDRA"`. For eight-rank native epochs select `--devices 0,2,4,6,8,10,12,14 --profile imitation --runtime --epochs 8 --batch-size 2 --workers 0 --memory 96g --cpus 32`; optionally stop with `--stop-after-epochs 2`, then resume `epoch_2.pth` with the **same** epoch horizon in a new output directory.

Resume supports FP32/native AdamW, workers=0, unchanged source/image/data/topology/batch/seed/optimizer/LR schedule. Every rank RNG is embedded in the checkpoint. Full tensor restoration is checked immediately; use `runtime_rank*.json` and `runtime_rank*.jsonl` for actual updates, LR, tokens, RSS and allocator memory. `--checkpoint` initializes weights; **`--resume` restores training state**. Old weight-only checkpoints are not complete resume checkpoints. Keep the exact source revision used by a saved run.

For a numerical continuation control, finish a continuous 30-step run that saves
`iter_10.pth`, then resume that exact file in a new directory to step 30. Compare
both final checkpoints on CPU; use the continuous run itself as the saved prefix
so independent initialization differences do not enter the comparison:

```bash
REPLAY_OUT="$RUNS/resume_checks/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPLAY_OUT"
docker run --rm --network none --memory 24g --cpus 8 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/source:ro" -v "$RUNS:/runs:ro" \
  -v "$REPLAY_OUT:/checks:rw" -w /source "$IMAGE" \
  /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
' \
  --continuous /runs/vad_continuous/iter_30.pth \
  --resumed /runs/vad_controlled_resumed/iter_30.pth \
  --continuous-run /runs/vad_continuous --partial-run /runs/vad_continuous \
  --resumed-run /runs/vad_controlled_resumed --output /checks/comparison.json
```

This compares model/optimizer tensors, loss, tokens, LR and final rank RNG. A
nonzero exit preserves a failed numerical gate. Importing the installed official
`torch_npu` hooks first is required for these NPU-saved checkpoints; no NPU
devices are mounted. The earlier `96a7bef…` VADv2 LoRA case passed, while its
base/Hydra cases failed. Current `858f9d…` single-rank eight-update controls
passed; its eight-rank controls failed, as recorded in the
[versioned results](evidence/eight-rank-resume-20261013-evening.json).

## Inference and official scoring

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$RUNS/vad_resumed/vadv2_ft_rank0.py" \
  --mini-prepared "$MINI" --checkpoint "$RUNS/vad_resumed/iter_30.pth" --vocab "$VOCAB" \
  --output "$RUNS/vad_eval" --detach
```

Inference verifies complete token coverage, finite 40x3 trajectories and exact selection from the official vocabulary. Its embedded metrics are cached targets. For independent official rescoring, use real scene logs and maps with pinned NAVSIM v1.1:

```bash
export PYTHONPATH="$PWD/ascend/third_party/navsim:$PYTHONPATH"
python ascend/official_score.py --prepared "$MINI" \
  --logs /datasets/worldengine/navsim_logs --maps /datasets/worldengine/maps \
  --cache /experiments/worldengine/metric_cache \
  --predictions "$RUNS/vad_eval/predictions.pkl" \
  --output /experiments/worldengine/vad_official_score.json
```

The scorer uses upstream WorldEngine's 40-to-8 pose conversion (`[4::5]`), unmodified official cache generation, simulator and scorer. Invalid/missing tokens fail the run. GPU/NPU trajectory parity remains untested because GPU hardware was unavailable.

## Sustained run and equipment

```bash
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 --profile imitation --runtime \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --checkpoint "$VAD" --vocab "$VOCAB" \
  --output "$RUNS/hour_8p" --steps 100000 --upstream-schedule --duration-seconds 3600 \
  --batch-size 2 --workers 0 --memory 96g --cpus 32 --checkpoint-interval 100 --check-interval 20 --detach
```

This pressure-test horizon is 100000 iterations with a measured one-hour stop, original optimizer LR/warmup parameters and an iter-based cosine schedule. It is distinct from the original eight-epoch dataset schedule. Checks include raw head losses every iteration, gradients on the first and every twentieth step, checkpoints, watched DDP parameter synchronization, CPU RSS and NPU allocator memory. Throughput includes the actual validation/IO/checkpoint overhead.

One 64 GiB logical NPU runs the measured single-rank tasks. The historical planning B2/eight-rank run used one chip on each of eight physical cards. The newer sixteen-rank B1 run matches the original planning global sample count of16 on one server. In the default post8 execution environment, its strict numerical gate passed only1/4 pairs; the later synchronous candidate passed one separate pair per model. Candidate repeatability remains unaccepted. Four complete dual-chip cards with eight ranks at B2 and cross-server layouts need separate acceptance.

Complete available NAVTEST caches (3.640 GB serialized, 19470 annotation frames and 12146 PDM tokens) used **4.467 GB / 4.16 GiB peak RSS** in a single CPU process. That older NAVTEST check did not measure NAVTRAIN. Original full-NAVTRAIN dataset construction is now measured separately at1/8/16 resident datasets, reaching834.13 GB summed RSS for16; see the current acceptance table. The39.86 GB serialized PKL size is not the runtime RSS. Full-image/model/worker training still needs its own capacity budget.

For full data use `--data-root` with original `raw/openscene-v1.1/sensor_blobs/{trainval,test}` and AlgEngine merged-info/PDM files, `--runtime --epochs 8 --workers 0`, plus an explicit CPU memory budget. Do not install upstream CUDA wheels over this pinned NPU stack. Full-data training/quality, GPU comparisons, separate-machine driver reproduction, BF16/whole-model AMP, additional AlgEngine configurations, multi-node scaling and 24-hour stability require separate acceptance.

### Run the official CPU scorer in the tested image

The scoring Python command above needs the pinned dependencies. A fresh host can
use the delivered image without installing those dependencies on the host:

```bash
SCORING=/datasets/worldengine/scoring  # contains navsim_logs/ and maps/
mkdir -p "$RUNS/cpu_score"
docker run --rm --memory 24g --cpus 8 --entrypoint bash \
  -v "$PWD:/source:ro" -v "$MINI:/mini:ro" \
  -v "$SCORING:/scoring:ro" -v "$RUNS/vad_eval:/predictions:ro" \
  -v "$RUNS/cpu_score:/output:rw" "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:$PYTHONPATH
    cd /source
    python ascend/official_score.py --prepared /mini \
      --logs /scoring/navsim_logs --maps /scoring/maps --cache /output/cache \
      --predictions /predictions/predictions.pkl --output /output/vad_score.json
  '
```

This regenerates official metric caches from genuine logs and maps. The
reproduction packet also includes 22 previously generated official cache
payloads and the three raw logs; their metadata CSV paths must be regenerated
after relocation. Reusing those payloads with NAVSIM's official
`run_pdm_score_from_submission` entry does not require downloading all maps
again. Maps are needed when regenerating cache payloads.

The packet includes source, its pinned NAVSIM files, image, planners, vocabulary,
mini data and official cache. Verify its `SHA256SUMS` before import. These
artifacts provide a small-data reproduction route; they do not certify another
host's driver installation or complete numerical replay.

The earlier offline packet's private cache helper produced four successful
relocation checks. For a repository checkout, use the public
`ascend/tools/score_cached.py --predictions <predictions.pkl> --metric-cache
<cache-root> --output <new-output>` in a CPU container, with no NPU device mounts.
It calls the existing `ascend.scoring.score_predictions` implementation, which
regenerates the metadata index and invokes the unchanged official scorer.
The new CLI wrapper has local checks; it does not inherit a new device or
scoring acceptance result from the earlier private helper.
