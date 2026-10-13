# Full AlgEngine migration candidate

[中文操作文档](README_zh.md) provides deployment, data preparation, training,
fine-tuning, inference, complete resume and acceptance commands for this same
Python fingerprint and measured result set. See also the
[bounded machine-readable evidence](evidence/full-algengine-20261013.json).

This is a candidate for the complete **published AlgEngine scope**, based on
upstream `3eded7e8ac64e5bed817860765089b5914ab698f`. SimEngine rendering and scene
generation remain GPU tasks. No 5 TB download is required for the existing
functional tests. Downloads and data processing are performed on the Linux
server; no dataset is downloaded to the Windows workspace.

## What has actually been exercised

All 44 configurations resolved on the locked clean image. There are 36 real-data
configurations and eight configurations requiring genuine synthetic/SimScale
exports. The final candidate executed **36/36 real configurations**, three
real-mini optimizer updates each, using each config's original initializer.
Both perception configurations started from the **official BEVFormer epoch-48
initializer**, including unfrozen backbone, tracking and mapping losses. Mini
filters, B1/workers0 and a short LR horizon are explicit overrides; the original
data ratios and convergence have not been accepted.

The final candidate's normalized AlgEngine/Ascend Python fingerprint is
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

The map decoder uses MMCV's independent MSDA autograd function. Its CUDA-style
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
The standard CPU/CUDA branch is retained. These process-scoped fixes do not
modify installed packages or the saved image.

The launcher places CANN compile/work/cache files in writable output directories
while mounting source read-only. `--config` preserves original model, head,
freezing, loss, optimizer and schedule settings. Bounded mini tests explicitly
override the filter, batch, worker count and short LR horizon; those overrides
do not validate the original data proportions or convergence.

## Hardware and capacity

The exercised host is Atlas 800I A3 inference series, Ascend910_9362, with eight
physical cards / 16 logical chips and 64 GiB HBM per logical chip. Experimental
backpropagation does not establish vendor training support for this SKU.

A single logical chip with 64 GiB executed the full-geometry perception model at
B1: the observed allocator peak was about **43.61 GiB allocated / 46.00 GiB
reserved**. The original planning B2 eight-rank run used up to 27.04/38.34 GiB
allocated/reserved per chip. At least one server can therefore run the measured
small-data cases; the usual original eight-rank global batch can use eight
logical chips in one server. Full NAVTRAIN RAM, I/O throughput, runtime and
convergence have not been measured. Do not infer full training RAM from PKL
file size or extrapolate these peaks to arbitrary batches.

Start full-data deserialization in one CPU container before allocating eight
training ranks. `preflight.py` retains both train and evaluation inputs and
records actual RSS. Its rank projection is only data-process RSS multiplied by
ranks; model/optimizer memory, validation, workers and runtime headroom must
also be measured. Exact resume currently requires `--workers 0`.

## Required real data

Use a dedicated root such as `/var/ai-model/lzd/worldengine/data`, or the agreed
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
  --output /var/ai-model/lzd/worldengine/weights
# If the selected source fails, select the publisher's domestic repository.
python3 ascend/fetch_assets.py --group warmstarts --source modelscope \
  --output /var/ai-model/lzd/worldengine/weights
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
  --data-root /var/ai-model/lzd/worldengine/data \
  --checkpoint /var/ai-model/lzd/worldengine/weights/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth \
  --vocab /var/ai-model/lzd/worldengine/data/alg_engine/test_8192_kmeans.npy \
  --maps /var/ai-model/lzd/worldengine/data/raw/nuplan/dataset/maps \
  --official-metric-cache /var/ai-model/lzd/worldengine/metric_cache_navtest \
  --validate --workers 0 --memory "$MEASURED_CPU_MEMORY_LIMIT" --cpus 64 \
  --output /var/ai-model/lzd/worldengine/runs/navtrain_vad50 --detach
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
saved-state restoration result and subsequent metrics. Strict numerical replay
still has the documented MSDA atomic-accumulation limitations.

## Genuine synthetic and SimScale inputs

Eight configurations need genuine exported camera sequences, converted
metadata and matching PDM rewards. Pass `--synthetic-root` to an existing export
root containing `sensor_blobs`, `pdms_pkl` and `meta_datas` (WorldEngine rollouts)
or `meta_datas_navformer` (SimScale). Preserve each config's `folder_name` and
filters. WorldEngine rollouts also consume `plan_idx.csv`; SimScale consumes
per-folder reward PKLs and split YAMLs.

These configurations remain unexecuted without genuine exports. The inspected
public SimScale reward filenames advertise 16384 trajectories; the WorldEngine
configs use the 8192 vocabulary. Actual vocabulary order, reward dimensions and
conversion must be verified before reuse. A matching filename or a renamed PKL
does not establish compatibility. No substitute real mini log is presented as
a genuine synthetic rollout. A synthetic config's `--synthetic-root` gate fails
early instead of silently training a different dataset.

The original Hydra BWM config contains `/path/to/synthetic/rollouts/...`
placeholders. These must be replaced by the actual exported folder names in a
reviewable configuration; `--synthetic-root` does not resolve that placeholder
mapping automatically. The SimScale fine-tune config names `e2e_vadv2_ep8.pth`;
its exact original initialization asset/alias must also be established rather
than silently substituting the tested 50% planner. These are data/configuration
prerequisites, not observed NPU operator failures.

## Remaining acceptance

The strict large-shape location-gradient gate, full NAVTRAIN deserialization/RAM,
representative complete-data training,
all genuine synthetic consumers, original-horizon convergence, final official
NAVTEST common/rare quality, sustained target-config training, mixed precision,
cross-server execution, independent GPU/NPU model comparison and formal vendor
training support remain separately recorded. The published base scores are
reference values for the exact benchmark; they are not measured NPU results.
