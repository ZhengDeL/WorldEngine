# Full AlgEngine migration candidate

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


**2026-10-13 FP32 MSDA update:** use the [MSDA repair supplement](MSDA.md)
and its [versioned results](evidence/msda-precision-20261013.json) for the current
native NPU sampler, strict gradients and new model/resource regressions. The
`96a7bef…` fingerprint and measurements below describe the earlier fused-path
snapshot. They remain historical evidence and are not reassigned to the new source.

**2026-10-13 evening update:** the historical `2bed8f4d7c78` / `858f9d…` FP32
source completed four eight-rank continuous/resumed jobs, with all 32 runtime
checks and exact saved-state restoration passing. The unchanged final numerical
gate nevertheless **failed for both VADv2 and HydraMDP**: 12 model / 21 optimizer
and 2 model / 2 optimizer tensors exceeded tolerance, respectively. Batch tokens,
LR and final RNG states matched. The [evening evidence](evidence/eight-rank-resume-20261013-evening.json)
records original configuration/initializer hashes, tensor paths and reporting
limitations. At the October13 22:44 snapshot the planned perception run was still running.
The separately completed old-source eight-hour result is reported above; it
does not retroactively pass the earlier interrupted job. Performance optimization is
deferred by the user, outside the current five correctness acceptance items;
full-data RAM, I/O and device-capacity checks remain necessary.

[中文操作文档](README_zh.md) provides deployment, data preparation, training,
fine-tuning, inference, complete resume and acceptance commands. See also the historical
[bounded machine-readable evidence](evidence/full-algengine-20261013.json).

This is a candidate for the complete **published AlgEngine scope**, based on
upstream `3eded7e8ac64e5bed817860765089b5914ab698f`. SimEngine rendering and scene
generation remain GPU tasks. No 5 TB download is required for the existing
functional tests. Downloads and data processing are performed on the Linux
server; no dataset is downloaded to the Windows workspace.

## What has actually been exercised

All 44 configurations resolved on the locked clean image. There are 36 real-data
configurations and eight configurations requiring genuine synthetic/SimScale
exports. The historical `96a7bef…` candidate executed **36/36 real configurations**, three
real-mini optimizer updates each, using each config's original initializer.
Both perception configurations started from the **official BEVFormer epoch-48
initializer**, including unfrozen backbone, tracking and mapping losses. Mini
filters, B1/workers0 and a short LR horizon are explicit overrides; the original
data ratios and convergence have not been accepted.

That historical candidate's normalized AlgEngine/Ascend Python fingerprint is
`96a7bef17fd27a681b5f90c367507a3a9f72767e5698fbe514d077d20ec0f997`.
An earlier regression attempt was interrupted at initialization by an external
vLLM workload (`EL0020 / 507899`). After the devices were released the final
regression above passed. This candidate is not a guarantee that every precision
mode, data distribution, topology or continuous-training workload is accepted.

The final per-epoch planning workflow completed independent 22-token inference
and unmodified official NAVSIM CPU scoring, resumed all 989 model / 300 optimizer
tensors exactly, and completed the next epoch's scoring. A controlled resume
from the continuous run's own checkpoint passed the existing numerical gate:
maximum model difference 7.45e-9, optimizer/model difference 2.98e-8, loss
difference zero, matching data tokens, LR and final RNG. This is the VAD reward
LoRA mini case, not a fix for the earlier base/Hydra numerical failures.

Two-rank perception completed a 21-update epoch, saved a complete checkpoint,
restored all 1393 model / 3594 optimizer tensors exactly, and reached update 42
with another complete 37-frame detection/mapping evaluation. The original
protocol's inapplicable TP metrics and actual empty map unions are reported as
null with coverage counts; other nonfinite predictions or metrics fail.
Detection NDS was 0.01950 / 0.02179, mAP zero: functional evaluation passes,
perception quality and convergence are not accepted.

Eight-rank DiffusionDrive completed three updates from the original 100%
perception initializer. All 22 independent generated trajectories were finite
40x3 arrays and officially rescored, with PDMS 0.377858654551. ADE/FDE were
10.96562 / 19.69085; this is a short randomly initialized planning-head test,
not a trained planner quality result. Forty final-source complete checkpoints
passed independent CPU serialization/counter/finiteness checks.

The new eight-rank unfrozen perception hour completed **428 updates per rank**,
with the shortest measured duration **3610.270 s**. All ranks exited normally.
Observed end-to-end throughput was **0.94808 anchor exposures/s**, or 0.95762
after excluding the first 20 updates. This repeats 42 real anchors with sampler
padding; it is not a full-data or GPU performance result. Peak allocated/reserved
memory was **43.68/47.73 GiB per chip**. CPU cgroup high water observed near the
end was 67,895,468,032 bytes (63.23 GiB, including charged cache); the container
limit was 128 GiB. Rank 0 RSS rose from 3.329 GB to 4.370 GB, with an observed
process peak of 6.118 GB. This bounded run does not exclude a longer-term leak.
All five complete hour checkpoints passed independent CPU tensor/counter/RNG
checks. Their legitimate Adam counter values differ by conditional parameter;
at global update 100 they were [91,99,100].

That actual complete eight-rank checkpoint was then resumed with the unchanged
10000-update horizon, data, source, image, batch and topology, stopping at update
103. All ranks strictly restored 1393 model / 3594 optimizer tensors and the
saved conditional counters [91,99,100], completed three more updates and exited
normally. This confirms multi-rank restoration/continuation; it does not promise
strict numerical replay of every perception training step.

Both the epoch-resumed model and the hour's final checkpoint completed separate
CLI inference on all 37 independent current/history frames, checking 370 actual
box/score/dense-map/soft-output fields. A supplementary read-only post-run DDP
probe also compared every parameter after three native updates: all **1233
parameter tensors / 79,777,383 elements** were bitwise identical across eight
ranks. This does not assert equality of rank-local nonparameter buffers.

## Additional compatibility fixes

`compat.py` replaces DrivingSDK's forward-only fused ResNet MaxPool with
PyTorch's NPU-dispatched implementation. On three representative shapes its
forward exactly matched both CPU and the fused forward; the largest observed
gradient difference against CPU was `9.5367431640625e-7`.

The historical `96a7bef...` map decoder bridge used MMCV's independent MSDA
autograd function. Its CUDA-style
extension path caused `MultiScaleDeformableAttentionGrad_fp32` Vector Core
exceptions. The candidate connects this NPU path to the official DrivingSDK
forward/backward API and preserves `int32` spatial shapes/start indices.
Perception and distributed training passed after this bridge. The small-shape
CPU32 reference passed all three gradients. At 200x200 / 300 queries, forwards,
value gradients and attention-weight gradients passed, while 35/19200 location
gradient entries failed atol=1e-3 / rtol=1e-4 (maximum 0.0309448, relative L2
2.73677e-6). The CPU64 diagnostic also finds FP32 coordinate arithmetic error
in the CPU32 baseline (119 failed entries versus 92 for NPU under that gate).
This does not turn the original strict CPU32 gate into a pass; no threshold was
relaxed and no GPU/model quality equivalence is inferred.
The current `858f9d...` FP32 native NPU implementation passes this unchanged
CPU32 gate and the expanded complete-gradient matrix. See [the current MSDA
results](MSDA.md); the failures above remain historical evidence.
The standard CPU/CUDA branch is retained. These process-scoped fixes do not
modify installed packages or the saved image.

The launcher places CANN compile/work/cache files in writable output directories
while mounting source read-only. `--config` preserves original model, head,
freezing, loss, optimizer and schedule settings. Bounded mini tests explicitly
override the filter, batch, worker count and short LR horizon; those overrides
do not validate the original data proportions or convergence.

## Hardware and capacity

The exercised host is Atlas 800I A3 inference series, Ascend910_9362, with eight
physical cards / 16 logical chips and 64 GiB HBM per logical chip.

A single logical chip with 64 GiB executed the full-geometry perception model at
B1: the observed allocator peak was about **43.61 GiB allocated / 46.00 GiB
reserved**. The original planning B2 eight-rank run used up to 27.04/38.34 GiB
allocated/reserved per chip. At least one server can run the measured small-data cases. Original planning
8GPU/B2 gives global batch16; measured16NPU/B1 matches this sample count.
Eight NPU chips at B2 on four complete dual-chip cards still need their own
capacity and numerical acceptance. Full-PKL constructor RAM is now measured
separately; full-image/model training RAM, I/O, runtime and convergence remain
unaccepted. Do not infer full training RAM from PKL
file size or extrapolate these peaks to arbitrary batches.

The completed1/8/16 full-PKL constructor measurements above are prerequisites
for sizing, not training-memory budgets. Verify full camera/model/worker capacity
before starting full-data training. `preflight.py` retains both train and evaluation inputs and
records actual RSS. Its rank projection is only data-process RSS multiplied by
ranks; model/optimizer memory, validation, workers and runtime headroom must
also be measured. Exact resume currently requires `--workers 0`.

## Required real data

Use a dedicated root such as `/srv/worldengine/data`, or the agreed
NAS path through that root. The inspected local filesystem had about 595.86 GB
free before the expanded tests; a large dataset's compressed archives plus
extraction will need a larger agreed storage target. No mount or shared service
is changed automatically.

```text
data/
  alg_engine/
    merged_infos_navformer/
      nuplan_openscene_navtrain.pkl
      nuplan_openscene_navtest.pkl
    pdms_cache/
      pdm_8192_gt_cache_navtrain.pkl
      pdm_8192_gt_cache_navtest.pkl
    test_8192_kmeans.npy
    ckpts/                         # required original warm-starts / planners
  raw/
    openscene-v1.1/
      sensor_blobs/trainval/       # every required current/history camera
      sensor_blobs/test/
      meta_datas/trainval/         # official scenario/metric-cache generation
      meta_datas/test/
    nuplan/dataset/maps/           # metadata JSON plus all referenced .gpkg
```

The two full NAVTRAIN PKLs total **39,862,469,866 bytes**; the two NAVTEST PKLs
total **3,639,831,385 bytes**. Original VADv2/HydraMDP selection training requires
the six genuine 8192-entry PDM metric arrays per selected anchor. DiffusionDrive
and perception training do not require those selection caches, but independent
trajectory-quality acceptance requires official NAVSIM scenario caches.

The pinned NAVSIM v1.1 `download_navtrain.sh` names eight current/history camera
archives, observed at **449,046,313,693 compressed bytes** in total. This is a
targeted candidate rather than the complete 5 TB scene publication. Verify it
against the selected annotations, four-frame history and all eight cameras;
archive presence alone does not establish full coverage. Raw LiDAR files are
not read by the published camera-only pipelines.

For full official CPU scoring, prepare genuine metadata and maps, then generate
the official NAVSIM metric cache. The 8192-candidate PDM training cache and the
official scenario metric cache are separate inputs.

## Server-only domestic downloads

The official publisher's ModelScope repositories were reachable from the server:

- `https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine`
- `https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene`
- `https://www.modelscope.cn/datasets/OpenDriveLab/SimScale`

`https://hf-mirror.com` successfully delivered the two required additional
initializers on the server. Both matched the pinned publisher SHA256 and byte
count. The authoritative WorldEngine data revision is
`d0e88d858533ca407a3387a26577c083aa7b3617`.

```bash
# Run on the Linux server; this downloads only selected official weights.
python3 ascend/fetch_assets.py --group warmstarts --source hf-mirror \
  --output /srv/worldengine/weights
# If the selected source fails, select the publisher's domestic repository.
python3 ascend/fetch_assets.py --group warmstarts --source modelscope \
  --output /srv/worldengine/weights
```

`assets.lock.json` contains original file sizes and SHA256. Do not download the
whole ModelScope dataset by default. For large cache files arranged by the user,
keep their original names and verify the lock after transfer. Use explicit
server output/cache paths for every download.

When network access requires the provided server proxy, limit it to the download
process; replace the example address with the agreed proxy:

```bash
PROXY_URL=http://proxy-host:1080
http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" \
  python3 ascend/fetch_assets.py --group warmstarts --source hf-mirror \
    --output /datasets/worldengine/weights
```

## Clean-image and real-mini reproduction commands

The previously executed no-cache build, save/load and relocated-mini tests used
the exact clean image described in [README.md](README.md#source-and-runtime).
Its saved archive has 23,444,539,904 bytes and SHA256
`5c730fd509d9dea6ac1599094c7e1e60ab0fad78996031fcb46e7571988c5cac`.
There is no public image-registry upload claim. On the Linux server:

```bash
IMAGE=local/worldengine:26.1.0-a3-clean-20261013
IMAGE_ID=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
sha256sum worldengine_clean_20261013.tar
docker load -i worldengine_clean_20261013.tar
docker image inspect --format '{{.Id}}' "$IMAGE"

# An online recipe rebuild is a new image that needs its own verification/lock.
docker build --no-cache --platform linux/arm64 \
  -f ascend/docker/Dockerfile -t local/worldengine:rebuild ascend/docker
```

The saved image preserves its exercised ID. Re-resolving package channels in a
new recipe build is not a bit-identical image guarantee. Record the rebuilt
image ID and versions, run the functional gates, then specify both `--image`
and `--expected-image-id`; the original image's acceptance does not automatically
apply to that new image or another host's driver installation.

After extracting the delivered real-mini packet and arranging maps/weights:

```bash
MINI=/datasets/worldengine/mini_real_data
WEIGHTS=/datasets/worldengine/weights
VOCAB=/datasets/worldengine/test_8192_kmeans.npy
MAPS=/datasets/worldengine/data/raw/nuplan/dataset/maps
RUNS=/experiments/worldengine
npu-smi info

# Unfrozen backbone + tracking + mapping, original BEVFormer initialization.
python3 ascend/run_container.py --devices 0 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 3 --batch-size 1 --workers 0 \
  --output "$RUNS/perception_short" --detach

# Eight-rank perception: measured-duration stop with an unchanged 10000 horizon.
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 10000 --upstream-schedule \
  --duration-seconds 3600 --batch-size 1 --workers 0 --memory 128g --cpus 64 \
  --checkpoint-interval 100 --check-interval 20 --output "$RUNS/perception_hour" --detach

# Genuine DiffusionDrive generation, not vocabulary selection.
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_100pct.py \
  --mini-prepared "$MINI" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth" --vocab "$VOCAB" \
  --steps 3 --batch-size 1 --workers 0 --inference-after \
  --memory 128g --cpus 64 --output "$RUNS/dd_short" --detach
```

Run these sequentially after the previous job has released the selected chips.
Each output must be a new directory. The 128g limit is the observed mini
configuration, not a full NAVTRAIN memory allowance. The iter-based pressure
horizon is distinct from the original epoch schedule. Reward/LoRA, Hydra PG and
controlled numerical-resume examples are in [README.md](README.md#run-save-and-resume).

For native two-rank perception mini epochs, use the same perception config and
initializer with `--devices 0,2 --epochs 2 --stop-after-epochs 1 --validate`,
B1/workers0, maps and a new output. After `epoch_1.pth` and exit 0, remove the
stop argument, retain the two-epoch horizon and all other settings, and add
`--resume <partial-output>/epoch_1.pth` in another new output. Exact restoration
supports conditional Adam counts; it does not assert strict numerical replay
of every subsequent perception step.

For independent perception inference select `--task inference`, the dumped
`track_map_nuplan_r50_navtrain_50pct_rank0.py` from the training output, its
complete checkpoint, the unchanged mini and maps. For independent planning
scoring, pass the generated `predictions.pkl` to the CPU-only official scorer
command in [README.md](README.md#run-the-official-cpu-scorer-in-the-tested-image).
The small-data scoring container limit there does not size full-data scoring.

Completion requires the specific container's exit code and OOM status, actual
updates, raw loss/gradient checks and complete evaluation coverage. Read
`launch.json`, `runtime_rank*.json/jsonl`, `validation/epoch_*/evaluation.json`,
official score files and `inference_validation.json`. For example:

```bash
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  worldengine-npu-perception_hour
cat "$RUNS/perception_hour/exit_code"
tail -n 50 "$RUNS/perception_hour/console.log"
npu-smi info
```

Keep the delivered source tar's original config bytes for existing checkpoints.
A Git checkout with changed config line endings can have the same normalized
Python fingerprint but still fail the raw `original_config` resume contract.
Use a fresh run for changed data/source rather than bypassing that guard.

## Input gate and training

`preflight.py` checks selected annotation tokens, same-log temporal queues, eight
camera file references, genuine PDM token coverage/vector shapes/finiteness,
train/eval overlap, maps, official cache coverage and actual CPU RSS. The launcher
also enables strict selected-token PDM coverage; missing training targets cannot
silently become zero rewards. The gate does not certify synthetic content merely
from directory names and does not claim image-content hashes from file sizes.

Run the following CPU gate in the locked image, mounting the agreed data/maps/
metric-cache locations read-only and its report directory writable:

```bash
python /workspace/WorldEngine/ascend/preflight.py \
  --config configs/worldengine/e2e_vadv2_50pct.py \
  --data-root /data/full --maps /data/maps --metric-cache /data/cache \
  --require-official-score --hash-assets --ranks 8 --output /output/preflight.json
```

After complete inputs and RAM sizing pass, the original 50% selection training
can use the native eight-epoch schedule:

```bash
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --data-root /srv/worldengine/data \
  --checkpoint /srv/worldengine/weights/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth \
  --vocab /srv/worldengine/data/alg_engine/test_8192_kmeans.npy \
  --maps /srv/worldengine/data/raw/nuplan/dataset/maps \
  --official-metric-cache /srv/worldengine/metric_cache_navtest \
  --validate --workers 0 --memory "$MEASURED_CPU_MEMORY_LIMIT" --cpus 64 \
  --output /srv/worldengine/runs/navtrain_vad50 --detach
```

The CPU memory placeholder must be filled from measured complete-cache and
representative training RSS, not guessed. Do not start this command while the
devices are occupied. The original planning configurations use eight epochs;
perception uses 20 or 40; DiffusionDrive uses 100. The 50% NAVTRAIN filter has
50,000 anchors; full NAVTRAIN has 103,288 in the pinned YAML. No loss-only
mini-data test proves convergence on either split.

`--validate` preserves every epoch's predictions and metrics in separate
directories, verifies complete coverage, optionally rescores all trajectories
with unmodified official NAVSIM on CPU, records common/available rare metrics
and keeps complete epoch checkpoints. CPU scoring uses a separate bounded Gloo
group while other ranks wait; it does not change the scorer or simulation.
Best-metric metadata points to the corresponding complete epoch checkpoint.

For interrupted native epochs keep the original configuration, horizon, data,
batch, world size, workers, seed, source and image unchanged. Add
`--stop-after-epochs 2` to an eight-epoch run; resume its complete `epoch_2.pth`
with `--resume` in a new output directory and the same horizon. Check the explicit
saved-state restoration result and subsequent metrics. On the historical source858 FP32
snapshot, the single-rank VADv2/Hydra controlled numerical comparisons passed.
Eight-rank complete-state restoration/continuation also passed, but the new
eight-rank continuous-eight-update versus same-run-iter-four-resume-to-eight
comparison **failed the original final tensor gate for both models**. See the
[actual eight-rank results](evidence/eight-rank-resume-20261013-evening.json).
The cause has not been isolated in this run; the earlier fused SDK atomic
accumulation observation cannot establish the cause of this new-source failure.

## Genuine synthetic and SimScale inputs

Eight configurations need genuine exported camera sequences, converted
metadata and matching PDM rewards. Pass `--synthetic-root` to an existing export
root containing `sensor_blobs`, `pdms_pkl` and `meta_datas` (WorldEngine rollouts)
or `meta_datas_navformer` (SimScale). Preserve each config's `folder_name` and
filters. WorldEngine rollouts also consume `plan_idx.csv`; SimScale consumes
per-folder reward PKLs and split YAMLs.

Two SimScale configurations have now completed portable CPU production,
original dataset getitem and one/two-rank NPU own-checkpoint controls for one
genuine scene. The other six rollout/replay/BWM configurations still lack
complete genuine exports. No real mini log is substituted for a synthetic
rollout. The original synthetic-root/filter checks remain active.

The inspected19.3 GB published reward has16384 candidates and different metric
fields; the original loader's conditional removal produces16383. Exact full
trajectory-row matches from the8192 vocabulary into the16384 vocabulary are
0/8192. Truncation, a renamed PKL or an assumed index mapping is not a valid
adapter. The portable recipe freshly computes8192 labels with original WE
four-second logged-traffic scoring and explicitly does not establish
SimScale-v2 reward equivalence or native nuPlanDB/full-engine equivalence.

Hydra BWM still needs its `/path/to/synthetic/rollouts/...` placeholders mapped
to actual genuine exports; synthetic-root does not supply those files.
The SimScale fine-tune basename `e2e_vadv2_ep8.pth` remains unavailable.
The tested recipe explicitly overrides it with published
`e2e_vadv2_100pct_ep8.pth`,434,885,463 bytes, SHA256
`fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a`.
It is not silently renamed or presented as the missing initializer.
These remaining export/configuration conditions are distinct from observed NPU
operator failures.

## Remaining acceptance

The strict large-shape MSDA gate is passed at the original tolerances.
Full-PKL NAVTRAIN dataset construction at1/8/16 residents and all six required
PDM arrays across the original103,288-token split are now checked. Remaining
work is full-image/model/worker capacity, representative complete-data training,
the other six genuine synthetic exports, original-horizon convergence and
independent complete NAVTEST common/rare quality.

Old-source858/post6 eight-hour operational stability and final-checkpoint
continuation passed. This does not accept target-sourcecc91/post8 sustained
training or general replay correctness. The untraced default post8 sixteen-rank
comparison passed1/4 pairs and failed3/4; retain those failures. State-observation
controls also failed. The first synchronous attempt was partial, followed by
two fresh completed pairs in the extended window: VAD was bitwise equal and
Hydra passed the original tolerances with maximum difference
2.9802322387695312e-8. Each model has one positive repetition; candidate
repeatability and sustained training remain unaccepted.

Performance optimization remains deferred. GPU comparison, mixed precision,
second-order gradients and cross-server scaling require separate evidence.
Sixteen-chip functionality and exact state
restoration have been executed. The synchronous candidate has two bounded
sixteen-chip numerical positives, while default execution and repeated
acceptance remain unresolved. Published benchmark scores are reference values, not measured NPU
results. Historical source-specific measurements remain in [MSDA.md](MSDA.md)
and its versioned results.
