# AlgEngine on Ascend 910C: installation and usage

This is the NPU counterpart of the [original AlgEngine usage guide](algengine_usage.md). It follows the same workflow: prepare data, train, evaluate, select rare cases, and fine-tune. Run the commands below on the Linux NPU server, from the repository root. The original GPU shell scripts remain available for GPU users; the NPU entry point is `ascend/run_container.py`.

For the CUDA-compatible route, pass `--msda-backend cuda-compatible` consistently to training, full-state resume and inference. Omitting it keeps the default `reference`. See the [MSDA backend guide](../ascend/MSDA_CUDA.md) for commands and scope.

WorldEngine contains **AlgEngine**, for model training and evaluation, and **SimEngine**, for rendering, simulation and rollout generation. This guide covers AlgEngine. Existing genuine SimEngine exports can be used as training inputs; deploying its renderer on NPU is outside this guide.

## Quick reference

| Task | NPU entry point |
| --- | --- |
| Build the runtime | [Public parent recipe](../ascend/docker/README.md), then [dependency environment](../ascend/docker/dependency_repair/README.md) |
| Download selected official assets | `ascend/tools/download_algengine_assets.py` |
| Prepare the fixed official-data subset | [NAVTRAIN139 / NAVTEST22](../ascend/data_tools/navtrain139/README.md); the alternative NAVTEST-only mini remains in section 3 |
| Train a published configuration | `ascend/run_container.py --config ...` |
| Original-style shell frontends | `projects/AlgEngine/scripts/e2e_dist_train_npu.sh`, `e2e_dist_eval_npu.sh` |
| Resume complete training state | The same training arguments plus `--resume ...` and a new output directory |
| Run inference | `ascend/run_container.py --task inference --config <dumped_config>` |
| Official CPU scoring | `ascend/tools/score_cached.py` with genuine metric-cache payloads; `ascend/official_score.py` can first generate them from logs/maps |
| Select rare cases | Original `projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py` |
| Prepare genuine SimScale inputs | [Data tools](../ascend/data_tools/README.md) |
| Reproduce the measured 16-chip synchronous candidate | [Eight-update protocol](../ascend/execution/README.md) |

## 1. Checkout and environment

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git rev-parse HEAD
git submodule status
```

Build the [public parent image](../ascend/docker/README.md), then complete the [dependency environment and its three CPU checks](../ascend/docker/dependency_repair/README.md) before training. The host needs a working Ascend driver, Docker, Python 3 and readable NPU character devices. The image supplies PyTorch/Torch-NPU, OpenMMLab, DrivingSDK and the geospatial dependencies. Do not replace them with the upstream CUDA wheels.

Use the image tag and **actual image ID** obtained by that installation procedure. The examples explicitly pass both values rather than relying on the launcher's historical defaults. A saved checkpoint must resume with the same image and source contract.

Set these host paths once in the current shell. `DATA` is the directory containing `alg_engine/` and `raw/`, not `data/alg_engine/` itself. Put `WORK` outside the source checkout; all launcher outputs must be new directories.

```bash
set -euo pipefail
SOURCE="$PWD"
DATA=/datasets/worldengine/data
WORK=/experiments/worldengine_npu
RUNS="$WORK/runs"
MAPS="$DATA/raw/nuplan/dataset/maps"
LOGS="$DATA/raw/navsim_logs/test"
: "${IMAGE:?Set IMAGE using the container installation guide}"
: "${IMAGE_ID:?Set the verified Docker image ID}"
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$IMAGE_ID"
mkdir -p "$DATA" "$RUNS"
npu-smi info
```

`--devices` takes host **logical chip IDs**. On the measured 910C server, eight dual-chip physical cards expose 16 logical chips, each with 64 GiB HBM. Check the actual host mapping and availability before choosing IDs. The launcher exposes only the selected devices; its per-process device numbering starts at zero inside the container.

## 2. Data and official weights

All downloads and extraction below run on the server. Select only AlgEngine inputs. The [original data organization](data_organization.md) remains the reference for dataset names; downloading the entire roughly 5 TB publication is unnecessary for this workflow.

Download official NAVTEST annotations/PDM/vocabulary and the released initialization weights:

```bash
python3 ascend/tools/download_algengine_assets.py \
  --group navtest --data-root "$DATA" --source hf-mirror
python3 ascend/tools/download_algengine_assets.py \
  --group weights --data-root "$DATA" --source hf-mirror
WEIGHTS="$DATA/alg_engine/ckpts"
VOCAB="$DATA/alg_engine/test_8192_kmeans.npy"
VAD_INIT="$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth"
VAD_BASE="$WEIGHTS/vadv2/e2e_vadv2_50pct_ep8.pth"
HYDRA_BASE="$WEIGHTS/hydramdp/e2e_hydramdp_50pct_ep8.pth"
```

The content lock also includes the official 100% planning checkpoints. Both complete files have been [verified against their published sizes and SHA-256 hashes](../ascend/evidence/planning-checkpoints-20261016.json). Pass the selected file as `--checkpoint` when starting a new fine-tuning task; the configuration and filter still determine the training-data scope.

```bash
VAD_BASE_100="$WEIGHTS/vadv2/e2e_vadv2_100pct_ep8.pth"
HYDRA_BASE_100="$WEIGHTS/hydramdp/e2e_hydramdp_100pct_ep8.pth"
```

The Hugging Face and hf-mirror routes request the pinned publication revision. The ModelScope route requests its `master` revision and verifies the same locked file sizes and SHA256 values; it does not claim to request the Hugging Face commit on ModelScope. The receipt records the requested revision and content-reference revision separately. Select the route with `--source modelscope` or `--source huggingface`. The downloader excludes camera packages, maps, raw logs and SimEngine assets. Existing valid files are reused; retain the checksum checks when changing routes.

The NAVTEST annotation and PDM files total about **3.64 GB**, excluding camera images, vocabulary, weights and maps. A tiny prepared split can use only the genuinely referenced camera JPEGs, but those files must already be available from the corresponding OpenScene data or a supplied subset. The repository does not promise a separate public download containing exactly the selected JPEGs.

### Obtain camera images, scene logs and maps

The original [NAVSIM test downloader](../ascend/third_party/navsim/download/download_test.sh) names the OpenScene metadata and 32 camera shards. Select the shard containing the logs you need; `SHARD=0` below is a download example and does **not** guarantee the three usable logs required by mini preparation. Do not run the original whole download script for a small check: it also downloads all camera shards and LiDAR. The OpenScene revision below is the pinned revision recorded in the [portable input locks](../ascend/data_tools/INPUT_LOCKS.json); these two new test-package transfers have not been executed as part of this documentation update.

```bash
DOWNLOADS="$WORK/downloads"
EXTRACT="$DATA/raw/openscene_download_test"
SHARD=0   # Select a required shard; the original test camera shards are numbered 0..31.
OS_REV=a76f840b65e972bc45e56c2adced897498e9a026
OS_URL="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/$OS_REV/openscene-v1.1"
mkdir -p "$DOWNLOADS"
test ! -e "$EXTRACT"
mkdir -p "$EXTRACT"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/openscene_metadata_test.tgz.part" \
  "$OS_URL/openscene_metadata_test.tgz"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" \
  "$OS_URL/openscene_sensor_test_camera/openscene_sensor_test_camera_${SHARD}.tgz"
tar -tzf "$DOWNLOADS/openscene_metadata_test.tgz.part" > "$DOWNLOADS/metadata_members.txt"
tar -tzf "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" > "$DOWNLOADS/camera_members.txt"
sed -n '1,20p' "$DOWNLOADS/metadata_members.txt"
sed -n '1,20p' "$DOWNLOADS/camera_members.txt"
tar -xzf "$DOWNLOADS/openscene_metadata_test.tgz.part" -C "$EXTRACT"
tar -xzf "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" -C "$EXTRACT"
test -d "$EXTRACT/openscene-v1.1/meta_datas"
test -d "$EXTRACT/openscene-v1.1/sensor_blobs"
mkdir -p "$DATA/raw/openscene-v1.1/sensor_blobs" "$DATA/raw/navsim_logs"
test ! -e "$DATA/raw/openscene-v1.1/sensor_blobs/test"
test ! -e "$DATA/raw/navsim_logs/test"
# Relative links remain inside DATA when it is mounted as /data/full.
ln -s ../../openscene_download_test/openscene-v1.1/sensor_blobs \
  "$DATA/raw/openscene-v1.1/sensor_blobs/test"
ln -s ../openscene_download_test/openscene-v1.1/meta_datas "$DATA/raw/navsim_logs/test"
sha256sum "$DOWNLOADS"/*.tgz.part > "$DOWNLOADS/transfer_SHA256SUMS"
```

Inspect the archive members and resolved paths. If another shard is needed, download/extract only that shard into the existing OpenScene extraction tree; do not replace existing validated JPEGs with different data. `transfer_SHA256SUMS` records your downloaded bytes; it is not a preverified publisher checksum. The preparation tool checks and decodes the selected JPEGs and reports insufficient coverage.

Maps come from the URL in the original [map downloader](../ascend/third_party/navsim/download/download_maps.sh). If complete maps are already installed, set `MAPS` to that directory instead of downloading again:

```bash
MAP_STAGE="$DATA/raw/nuplan/dataset"
mkdir -p "$MAP_STAGE"
test ! -e "$MAP_STAGE/nuplan-maps-v1.0"
test ! -e "$MAP_STAGE/maps"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/nuplan-maps-v1.1.zip.part" \
  https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip -l "$DOWNLOADS/nuplan-maps-v1.1.zip.part"
unzip "$DOWNLOADS/nuplan-maps-v1.1.zip.part" -d "$MAP_STAGE"
test -f "$MAP_STAGE/nuplan-maps-v1.0/nuplan-maps-v1.0.json"
ln -s nuplan-maps-v1.0 "$MAP_STAGE/maps"
MAPS="$MAP_STAGE/maps"
LOGS="$DATA/raw/navsim_logs/test"
```

The archive version in the URL and the extracted `nuplan-maps-v1.0` directory follow the original downloader. Keep the map JSON and regional `.gpkg` files together. For full NAVTRAIN images, the original [NAVTRAIN downloader](../ascend/third_party/navsim/download/download_navtrain.sh) identifies the training camera packages; copy/download only the required AlgEngine camera data and retain its `trainval` layout. Do not download SimEngine scene assets for model training.

For full NAVTRAIN, also fetch its two metadata/cache files:

```bash
python3 ascend/tools/download_algengine_assets.py \
  --group navtrain --data-root "$DATA" --source hf-mirror
```

They total **39,862,469,866 bytes**. Arrange the images, maps and logs separately. Compressed camera archives must be extracted so every annotation `cams[*].data_path` resolves under the chosen camera root; a `.tar.gz` in the directory is not a readable image dataset.

```text
DATA/
├── alg_engine/
│   ├── ckpts/
│   │   ├── vadv2/e2e_vadv2_50pct_ep8.pth
│   │   ├── hydramdp/e2e_hydramdp_50pct_ep8.pth
│   │   └── track_map_nuplan_r50_navtrain_{50,100}pct_bs1x8.pth
│   ├── merged_infos_navformer/nuplan_openscene_{navtrain,navtest}.pkl
│   ├── pdms_cache/pdm_8192_gt_cache_{navtrain,navtest}.pkl
│   └── test_8192_kmeans.npy
└── raw/
    ├── openscene-v1.1/sensor_blobs/{trainval,test}/
    ├── nuplan/dataset/maps/nuplan-maps-v1.0.json
    │                         # plus the corresponding regional .gpkg maps
    └── navsim_logs/test/      # genuine logs for independent CPU rescoring
```

## Recommended workflow: NAVTRAIN139 / NAVTEST22

Use this route for a fixed real-data validation run. The [public preparation tool](../ascend/data_tools/navtrain139/README.md) derives 139 training anchors from an original NAVTRAIN log and 22 evaluation anchors from a separate NAVTEST log. It preserves the original labels and all 8192 PDM candidates. This is a documented derived subset, not an official prepackaged mini or the complete benchmark.

### Prepare the environment, inputs and weights

Build the public post8 parent, then the [application/compiler dependency environment](../ascend/docker/dependency_repair/README.md), and run all three CPU acceptance commands there. Set `IMAGE` and `IMAGE_ID` to that final image. The measured repair passed both unfiltered `pip check` invocations and the API check after sourcing CANN. These CPU checks do not establish model acceptance.

Download the annotation/PDM groups and official weights with the commands in section 2, including `--group navtrain`. The data tool's [input lock](../ascend/data_tools/navtrain139/INPUTS.lock.json) gives the six exact URLs, revisions, sizes and hashes. Reuse verified server files. Its two required camera archives are **mini camera 12** and **test camera 30**; a different shard cannot substitute for them. The six original inputs total 47,913,227,335 bytes when none are cached. An [independent official API audit](../ascend/evidence/official-data-origins-20261016.json) matched all six files' pinned revisions, paths, sizes and LFS hashes on 2026-10-16; that metadata query did not download the file payloads. The full camera corpus and SimEngine data are not required.

Run the complete [CPU preparation command](../ascend/data_tools/navtrain139/README.md#run-on-the-linux-server), setting its `OUTPUT_PARENT="$WORK/navtrain139"`. Create that parent first and keep `prepared/` nonexistent. The tool reads the original archives directly, so these two archives do not need a separate extraction. It writes a launcher-ready split and copies only 1832 referenced original JPEGs. Do **not** run `split_mini.py` on this output. Its CPU replay, input hashes, full metadata comparison and image checks are in [VALIDATION.json](../ascend/data_tools/navtrain139/VALIDATION.json).

Set `VOCAB`, `VAD_BASE_100` and `HYDRA_BASE_100` from section 2. These are official 100%-trained planning initializers. The `--mini-prepared` option below selects the 139/22 validation data; the weight filename does not turn this into full NAVTRAIN training. Maps must be present at `MAPS`.

### Short validation: eight updates

Use this route when the device window is limited. Run VADv2/HydraMDP and FP32/BF16 as four separate combinations with independent outputs. The continuous branch performs eight updates. A separate interrupted branch keeps `--steps 8`, stops with `--stop-after 4`, and resumes its own update-4 checkpoint to update 8. Setting `--steps 4` would change the learning-rate contract. This uses IterBasedRunner and does not establish eight epochs or convergence.

```bash
PREPARED="$WORK/navtrain139/prepared"
test -f "$PREPARED/manifest.json"
test -f "$PREPARED/preparation.json"
test -f "$MAPS/nuplan-maps-v1.0.json"
MODEL=vad                 # vad or hydra
PRECISION=fp32            # repeat separately with bf16
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
case "$MODEL" in
  vad) CONFIG=worldengine/e2e_vadv2_100pct.py; INITIALIZER="$VAD_BASE_100" ;;
  hydra) CONFIG=hydramdp/e2e_hydramdp_100pct.py; INITIALIZER="$HYDRA_BASE_100" ;;
  *) printf 'Select MODEL=vad or MODEL=hydra\n' >&2; exit 2 ;;
esac
STEM=$(basename "$CONFIG" .py)
PREFIX="$RUNS/${MODEL}_navtrain139_${PRECISION}_updates8"
SHORT_COMMON=(--devices "$DEVICES" --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$SOURCE/projects/AlgEngine/configs/$CONFIG"
  --mini-prepared "$PREPARED" --checkpoint "$INITIALIZER" --vocab "$VOCAB"
  --maps "$MAPS" --runtime --steps 8 --batch-size 1 --workers 0
  --precision "$PRECISION" --deterministic --upstream-schedule
  --checkpoint-interval 4 --check-interval 1 --memory 256g --cpus 128
  --pids-limit 8192)

python3 ascend/run_container.py "${SHORT_COMMON[@]}" --output "${PREFIX}_continuous"
python3 ascend/run_container.py "${SHORT_COMMON[@]}" --stop-after 4 \
  --output "${PREFIX}_partial"
python3 ascend/run_container.py "${SHORT_COMMON[@]}" \
  --resume "${PREFIX}_partial/iter_4.pth" --output "${PREFIX}_resumed"
END_CHECKPOINT=iter_8.pth
END_UPDATES=8
```

After selecting this route, skip the epoch commands below and continue with the shared CPU comparison, independent 22-token inference and official scoring. Keep the `PREFIX`, `END_CHECKPOINT` and `END_UPDATES` set above.

### Native epoch workflow: eight epochs

Select this alternative for native epoch-runner validation. Its checkpoints cannot be exchanged with the eight-update route. Each command uses a new output; continuous, interrupted and resumed runs keep the same eight-epoch horizon.

```bash
PREPARED="$WORK/navtrain139/prepared"
test -f "$PREPARED/manifest.json"
test -f "$PREPARED/preparation.json"
test -f "$MAPS/nuplan-maps-v1.0.json"
MODEL=vad                 # vad or hydra
PRECISION=fp32            # bf16 is an experimental, separate run
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
case "$MODEL" in
  vad) CONFIG=worldengine/e2e_vadv2_100pct.py; INITIALIZER="$VAD_BASE_100" ;;
  hydra) CONFIG=hydramdp/e2e_hydramdp_100pct.py; INITIALIZER="$HYDRA_BASE_100" ;;
  *) printf 'Select MODEL=vad or MODEL=hydra\n' >&2; exit 2 ;;
esac
STEM=$(basename "$CONFIG" .py)
PREFIX="$RUNS/${MODEL}_navtrain139_${PRECISION}_epochs8"
COMMON=(--devices "$DEVICES" --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$SOURCE/projects/AlgEngine/configs/$CONFIG"
  --mini-prepared "$PREPARED" --checkpoint "$INITIALIZER" --vocab "$VOCAB"
  --maps "$MAPS" --runtime --epochs 8 --batch-size 1 --workers 0
  --precision "$PRECISION" --deterministic --upstream-schedule
  --checkpoint-interval 4 --check-interval 1 --memory 256g --cpus 128
  --pids-limit 8192)

python3 ascend/run_container.py "${COMMON[@]}" --output "${PREFIX}_continuous"
python3 ascend/run_container.py "${COMMON[@]}" --stop-after-epochs 4 \
  --output "${PREFIX}_partial"
python3 ascend/run_container.py "${COMMON[@]}" \
  --resume "${PREFIX}_partial/epoch_4.pth" --output "${PREFIX}_resumed"
END_CHECKPOINT=epoch_8.pth
END_UPDATES=72
```

At 16 ranks and B1, 139 anchors give nine updates per epoch: the continuous and resumed endpoints have 72 updates; the interrupted run stops at 36. This exercises the original epoch runner on a small subset, not convergence over the original 103,288-anchor split. The 256g limit is a subset training allocation and is not suitable evidence for full-data training capacity. This device list describes 16 logical chips on one host; check the physical mapping before using it.

`fp32` is the default and disables both public Matmul/Conv HF32 flags. `bf16` uses native NPU forward autocast with the history/current shared-weight cache disabled; master parameters, gradients, optimizer state, MSDA and planning loss remain FP32, without a loss scaler. Geometry and original BatchNorm/SyncBatchNorm use FP32 islands while retaining original modules, running statistics and collectives. On source `e14dd6b2…`/244 Python files and image `3522…`, both official100 planners completed [eight BF16 updates, own restoration and 22-token inference/scoring on 16 ranks](../ascend/results/bf16-v7-navtrain139-short8-20261016.json). [All-rank and saved-Adam evidence](../ascend/results/bf16-v7-gradient-coverage-20261016.json) verifies restoration of the prior 138 coverage omissions per model. Both complete numerical replays still exceed unchanged original tolerances. Eight updates are within 500 warmup updates, not convergence or quality/numerical equivalence. Set `PRECISION=bf16` to test this opt-in experimental mode with separate outputs and own checkpoints. An FP32 checkpoint cannot resume under a BF16 contract.

The v6 execution record is superseded as training acceptance: [matched checkpoints confirmed 138 optimizer-state IDs missing from each BF16 model relative to FP32](../ascend/results/bf16-v6-optimizer-coverage-20261016.json). A [real CPU cache probe](../ascend/results/bf16-autocast-history-cache-cpu-20261016.json) reproduced missing shared-layer gradients after history `no_grad`, despite a finite loss and an updating head. The CPU mechanism and old checkpoint-coverage defect are established; the v7 coverage repair has separate actual NPU/saved-state verification. This does not explain earlier FP32 replay differences or establish a universal numerical repair.

The [recorded FP32/BF16 comparison](../ascend/results/fp32-bf16-v7-matched8-planning-20261016.json) covers four model/stage combinations on the same 22 held-out anchors, each with all 40 poses. Same-official-weight inference is separate from each precision's independently trained eight-update checkpoint. These are measurements without a quality-equivalence threshold, not GPU comparisons.

Compare the two complete endpoints on CPU using the unchanged public comparator:

```bash
COMPARISON="${PREFIX}_comparison"
mkdir -p "$COMPARISON"
COMPARE_EXIT=0
docker run --rm --runtime runc --network none --memory 24g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$RUNS:$RUNS:ro" \
  -v "$COMPARISON:/checks:rw" -w /source "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
' --continuous "${PREFIX}_continuous/$END_CHECKPOINT" \
  --resumed "${PREFIX}_resumed/$END_CHECKPOINT" \
  --continuous-run "${PREFIX}_continuous" --partial-run "${PREFIX}_partial" \
  --resumed-run "${PREFIX}_resumed" --output /checks/comparison.json \
  || COMPARE_EXIT=$?
python3 - "$COMPARE_EXIT" "$COMPARISON/comparison.json" "$END_UPDATES" <<'PY'
import json, sys
from pathlib import Path
code = int(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text())
expected = {0: "pass", 1: "fail"}
assert code in expected and report["result"] == expected[code]
assert report["tensor_tolerance"] == {"atol": 1e-6, "rtol": 1e-5}
assert report["loss_absolute_tolerance"] == 1e-4
assert report["world_size"] == 16 and report["completed_updates"] == int(sys.argv[3])
print("Original numerical comparison:", report["result"])
PY
```

The comparison checks model/optimizer tensors, rank RNG, tokens, LR and losses. Its original tensor tolerance is `atol=1e-6, rtol=1e-5`, and maximum loss difference must be below `1e-4`. The command captures completed comparisons outside the original tolerance (original JSON `result=fail`, exit1, with a complete matching report) so independent inference/scoring can still run; missing or inconsistent reports stop the workflow. Retain an above-tolerance comparison as its numerical result, even when every restoration and training command exited successfully. Do not bypass the source, data, image or precision contract.

### Independent inference and official CPU scoring

Use the selected route's resumed configuration and endpoint checkpoint. `END_CHECKPOINT` is `iter_8.pth` or `epoch_8.pth`. The same command can evaluate the continuous endpoint by selecting that run's paths and a different output.

```bash
INFERENCE="${PREFIX}_resumed_eval"
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "${PREFIX}_resumed/${STEM}_rank0.py" \
  --checkpoint "${PREFIX}_resumed/$END_CHECKPOINT" --mini-prepared "$PREPARED" \
  --vocab "$VOCAB" --maps "$MAPS" --precision "$PRECISION" --deterministic \
  --memory 64g --cpus 16 --pids-limit 2048 --output "$INFERENCE"

CACHE=/datasets/worldengine/official_metric_cache
SCORE="${PREFIX}_resumed_score"
mkdir -p "$SCORE"
docker run --rm --runtime runc --network none --memory 24g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$INFERENCE:/predictions:ro" \
  -v "$CACHE:$CACHE:ro" -v "$SCORE:/output:rw" -w /source "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
  --predictions /predictions/predictions.pkl --metric-cache "$CACHE" \
  --output /output/score
cat "$SCORE/score/official_score.json"
```

The scorer requires original official NAVSIM metric-cache payloads for all 22 prediction tokens. Training PDM labels are not this cache. The helper selects exactly those prediction tokens and leaves payload bytes and scoring mathematics unchanged. Keep symlink targets reachable inside the container; mount additional target roots at the same absolute paths when needed. If the cache is unavailable, section 6 describes generation from genuine logs and maps.

Check `inference_validation.json`, the 22-token score and the original comparison JSON independently. The separate [official100 GPU/NPU FP32 comparison](../ascend/results/gpu-npu-official100-eval22-20261016.json) covers these same 22 evaluation anchors; it does not establish full NAVTRAIN convergence, complete NAVTEST quality, GPU training or general 16-rank numerical reproducibility. Current model results are recorded in the [validation report](../ascend/FULL_ALGENGINE.md); environment/data CPU results must not be presented as a model pass.

Recompute all exported poses and the seven official-score differences with the [public CPU comparison tool](../tools/validation/README.md). It accepts either FP32/BF16 or GPU/NPU exports and does not apply a quality-equivalence threshold.

## 3. Prepare a small real dataset and run eight updates

This is the alternative NAVTEST-only smoke-data path. For genuine NAVTRAIN training anchors and a fixed independent evaluation split, use the recommended NAVTRAIN139/NAVTEST22 workflow above.

This route prepares data from the real NAVTEST annotation/PDM and available camera files. It preserves the original labels and 8192-entry reward vectors. The resulting train/evaluation split is separated by whole driving logs; it is an adaptation check, not the official full NAVTRAIN/NAVTEST benchmark.

The preparation runs in the image on CPU, with **no NPU device mounts**. Keep `WORK` mounted at the same absolute host path so the split's camera links remain usable by later containers.

```bash
RARE="$SOURCE/projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml"
MINI="$WORK/mini_real_data"
MINI_RARE="$WORK/prepared/worldengine_navtest_smoke_rare.yaml"
test -d "$DATA/raw/openscene-v1.1/sensor_blobs/test"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$DATA:$DATA:ro" -v "$WORK:$WORK:rw" \
  -w /source "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python ascend/tools/prepare_real_mini.py \
      --annotations "$1/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl" \
      --pdm "$1/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl" \
      --cameras "$1/raw/openscene-v1.1/sensor_blobs/test" \
      --rare-split /source/projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml \
      --camera-mode copy \
      --output "$2/prepared"
    python ascend/split_mini.py --prepared "$2/prepared" \
      --output "$2/mini_real_data" \
      --rare-split "$2/prepared/worldengine_navtest_smoke_rare.yaml"
  ' bash "$DATA" "$WORK"
cat "$MINI/manifest.json"
```

Use a fresh `WORK/prepared` and `MINI`; the tools reject existing outputs. Preparation needs readable current/history JPEGs from more than one log, including a genuine rare training anchor and a retrievable non-rare anchor in that same log. The complete `RARE` file defines rare cases during preparation. The generated `MINI_RARE` lists only selected rare anchors with complete original queues; use it for splitting and mini fine-tuning so other rare tokens retained only as history are not treated as training anchors. Missing images or an insufficient log-disjoint split must be corrected rather than replaced with fabricated samples. The 24g limit is a small NAVTEST preparation budget, not a full NAVTRAIN budget.

The new preparation tool defaults to three logs and at most 22 anchors per log, so it selects at most 66 anchors before splitting. Inspect its actual train/evaluation counts and hashes in `manifest.json`; these are not the fixed 42/22 counts of an earlier measured packet. Start fresh training with this generated data and use its own checkpoints. Historical numerical/quality measurements cannot be reassigned to the new mini merely because the filenames match.

Run one logical chip, the original VAD configuration and the official perception initializer:

```bash
python3 ascend/run_container.py --devices 0 \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/vad_mini8"
```

This is a bounded eight-update run with the upstream optimizer LR/warmup parameters and an iteration horizon of eight. It does not complete the configuration's original eight epochs. `fp32` disables both public Matmul and Conv HF32 flags. A successful run writes `exit_code`, `runtime_rank0.json`, `runtime_rank0.jsonl`, `iter_4.pth`, `iter_8.pth` and `e2e_vadv2_50pct_rank0.py`.

For Hydra, use `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct.py` with the same perception initializer, and a distinct output directory. For the perception architecture, use `projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py` and `ckpts/bevformerv2-r50-t1-base_epoch_48.pth`; maps must be supplied with `--maps "$MAPS"`.

The original-style shell frontend accepts `<config> <num_chips> [resume_checkpoint]`. For the same small-data check in a fresh output:

```bash
export IMAGE IMAGE_ID VOCAB
export INIT_CHECKPOINT="$VAD_INIT" MINI_PREPARED="$MINI"
export RUN_OUTPUT="$RUNS/vad_mini8_shell" CPU_MEMORY=64g
bash projects/AlgEngine/scripts/e2e_dist_train_npu.sh \
  configs/worldengine/e2e_vadv2_50pct.py 1 -- \
  --steps 8 --checkpoint-interval 4 --check-interval 1 --deterministic
```

The frontend runs on the same single host and uses the NPU launcher. Sixteen chips are not interpreted as two eight-GPU nodes. Set `DEVICE_IDS` if the desired host chip IDs differ from the first `num_chips` IDs. For full data, unset `MINI_PREPARED` and export `DATA_ROOT="$DATA"`; `DATA_ROOT` and `MINI_PREPARED` are mutually exclusive.

## 4. Train on the original 50% or 100% NAVTRAIN split

First validate full annotations, current/history images, PDM entries, vocabulary, maps and the independent NAVTEST metric cache. `CACHE` must contain official NAVSIM cache payloads covering the configured NAVTEST tokens and their `metadata/*.csv` index. The label PDM cache under `alg_engine/pdms_cache/` is a different input and cannot substitute for this cache.

```bash
CACHE=/datasets/worldengine/metric_cache_navtest_v1
PREFLIGHT="$WORK/preflight_vad50"
mkdir -p "$PREFLIGHT"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 128g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" -v "$MAPS:/maps:ro" \
  -v "$CACHE:/metric_cache:ro" -v "$PREFLIGHT:/output:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export WORLDENGINE_ROOT=/source
    python ascend/preflight.py \
      --config /source/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
      --data-root /data --maps /maps --metric-cache /metric_cache \
      --require-official-score --hash-assets --ranks 16 --output /output/preflight.json
  '
cat "$PREFLIGHT/preflight.json"
```

If a complete official cache is unavailable, generate it from genuine NAVSIM logs and maps using the pinned [NAVSIM cache instructions](../ascend/third_party/navsim/docs/cache.md). The original [metric-caching script](../ascend/third_party/navsim/navsim/planning/script/run_metric_caching.py) runs on CPU and supports `train_test_split=navtest worker=sequential`; its paths must point to the real data. Cache generation does not require the SimEngine renderer.

For a new cache directory, the CPU command is:

```bash
mkdir -p "$CACHE"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 64g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$LOGS:/logs:ro" -v "$MAPS:/maps:ro" \
  -v "$CACHE:/output:rw" -w /source "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:${PYTHONPATH:-}
    export NAVSIM_EXP_ROOT=/output OPENSCENE_DATA_ROOT=/unused NUPLAN_MAPS_ROOT=/maps
    python ascend/third_party/navsim/navsim/planning/script/run_metric_caching.py \
      train_test_split=navtest worker=sequential navsim_log_path=/logs cache.cache_path=/output
  '
```

This uses the original official split and cache generator. Its 64g starting limit has not been measured for complete cache generation in this delivery. Check the generated payload count, logs and preflight token coverage; the upstream cache script can report failed entries without a nonzero process exit.

The preflight checks camera existence/size, history, token coverage and PDM values; it does not decode every JPEG. Read its errors and memory report before model training. Sixteen resident original NAVTRAIN dataset objects were measured at approximately **834 GB summed RSS**, before full images, model/optimizer, workers and evaluation. Choose `TRAIN_MEMORY` from available RAM and a measured training budget; do not reuse a mini limit.

The original planning configurations use eight GPUs with two samples per GPU, for global batch 16. The following NPU configuration uses **16 logical chips with one sample per chip**, preserving that sample count. The grouping changes; GPU numerical equivalence is unverified. A four-physical-card/eight-chip B2 capacity configuration has not been measured.

```bash
: "${TRAIN_MEMORY:?Set an adequate full-training Docker memory limit}"
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --data-root "$DATA" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --batch-size 1 --workers 0 --precision fp32 --deterministic \
  --maps "$MAPS" --official-metric-cache "$CACHE" --validate \
  --memory "$TRAIN_MEMORY" --cpus 64 --pids-limit 8192 \
  --output "$RUNS/vad50_full"
```

Omitting `--steps` and `--epochs` keeps the original eight-epoch horizon, optimizer and LR schedule. The migration's `--validate` runs evaluation each epoch. For 100%, change the config to `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py`, initialize with `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth`, and select a fresh output. The 50% filter has 50,000 anchors; the full official split has 103,288. Full-image training and convergence have not yet completed in this migration.

Do not apply the separate synchronous eight-update wrapper to a full epoch run. Its [documented protocol](../ascend/execution/README.md) is deliberately limited to the two measured planning configs and eight updates. Four independent repeated pairs completed; two were within the original tolerance and two were outside it. These settings are not an accepted general repair.

## 5. Save and resume complete training state

`--checkpoint` initializes model weights. `--resume` restores model, optimizer, per-rank RNG and sampling/LR progress. A weight-only published checkpoint is an initializer, not a complete migration resume checkpoint.

For the eight-update example, resume the **same continuous run's** update-4 checkpoint to the original endpoint, using a new output:

```bash
python3 ascend/run_container.py --devices 0 \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --resume "$RUNS/vad_mini8/iter_4.pth" \
  --output "$RUNS/vad_mini8_resumed"
```

For a controlled epoch interruption, add `--stop-after-epochs 2` to the full command and set its first output to `--output "$RUNS/vad50_partial"`, keeping the original eight-epoch horizon. Once it exits and `epoch_2.pth` is present, repeat the same command without the stopping option and add `--resume "$RUNS/vad50_partial/epoch_2.pth"` plus a fresh output. Do not shorten the LR horizon to two epochs.

Exact-state continuation requires `--workers 0` and the same source/config bytes, image, precision/environment, world size, data/filter/vocabulary, batch, seed and schedule. A contract mismatch is an error to investigate. Keep the training checkout and original data; do not bypass checks to load an incompatible checkpoint. The launcher requires explicit `--resume`; it does not auto-resume merely because `latest.pth` exists.

Compare the two update-8 endpoints on CPU. This imports the installed official Torch-NPU serialization hooks without mounting NPU devices:

```bash
CHECK="$WORK/vad_mini8_comparison"
mkdir -p "$CHECK"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$RUNS:/runs:ro" -v "$CHECK:/checks:rw" \
  -w /source "$IMAGE" /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
' \
  --continuous /runs/vad_mini8/iter_8.pth --resumed /runs/vad_mini8_resumed/iter_8.pth \
  --continuous-run /runs/vad_mini8 --partial-run /runs/vad_mini8 \
  --resumed-run /runs/vad_mini8_resumed --output /checks/comparison.json
cat "$CHECK/comparison.json"
```

The comparator checks tensors, losses, tokens, LR and final RNG at its unchanged thresholds. A restore report passing does not establish that all later computation reproduces the continuous run. Completed default 16-chip strict controls include above-tolerance results; consult the [current results](../ascend/results/strict-replay-post8-20261014.json) before adopting that topology for reproducibility-sensitive runs.

## 6. Evaluation and official scoring

Use the **dumped training configuration**, because it contains the actual mini/full data paths, vocabulary and head settings. The launcher does not rewrite an original GPU configuration's test data paths during inference.

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$RUNS/vad_mini8/e2e_vadv2_50pct_rank0.py" \
  --mini-prepared "$MINI" --checkpoint "$RUNS/vad_mini8/iter_8.pth" \
  --vocab "$VOCAB" --precision fp32 --deterministic \
  --output "$RUNS/vad_mini8_eval"
```

Outputs include `predictions.pkl`, `metrics.csv` and `inference_validation.json`. Planning validation checks every configured token exactly once, finite `40x3` trajectories and, for selection heads, correspondence with the 8192 vocabulary. Perception configs execute the original detection/mapping evaluator. Inspect `console.log` for its metrics as well as the validation JSON.

The corresponding original-style evaluation frontend is:

```bash
export RUN_OUTPUT="$RUNS/vad_mini8_eval_shell" CPU_MEMORY=64g
bash projects/AlgEngine/scripts/e2e_dist_eval_npu.sh \
  "$RUNS/vad_mini8/e2e_vadv2_50pct_rank0.py" "$RUNS/vad_mini8/iter_8.pth" 1 -- \
  --deterministic
```

It uses the `IMAGE`, `IMAGE_ID`, `VOCAB` and `MINI_PREPARED` exported for the earlier shell example. For full data, switch to `DATA_ROOT` and the matching full dumped config; retain a separate evaluation output.

For an independent small-data score, use real scene logs and maps on CPU. Paths below are host paths, mounted at the same names:

```bash
SCORE="$WORK/vad_mini8_official"
mkdir -p "$SCORE"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" \
  -v "$LOGS:/logs:ro" -v "$MAPS:/maps:ro" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:${PYTHONPATH:-}
    python ascend/official_score.py --prepared "$1" \
      --logs /logs --maps /maps --cache "$2/cache" \
      --predictions "$3/predictions.pkl" --output "$2/official_score.json"
  ' bash "$MINI" "$SCORE" "$RUNS/vad_mini8_eval"
```

This uses pinned NAVSIM v1.1, the original 40-to-8 pose conversion `[4::5]`, cache generation, simulator and scorer. Cached model-selection labels alone are not an independent score. The CPU scorer needs no NPU. Its 24g example is for this small split.

If matching official cache payloads are already available, score directly without logs/maps or regenerating payloads. This helper selects exactly the prediction tokens from the supplied cache, rejects missing or duplicate matches, and calls the existing `ascend.scoring.score_predictions` with a relocated index. Extra cache tokens are excluded; original payload bytes and scoring mathematics remain unchanged:

```bash
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
  --predictions "$RUNS/vad_mini8_eval/predictions.pkl" \
  --metric-cache "$SCORE/cache" --output "$WORK/vad_mini8_from_cache"
```

`--output` must not already exist. Every prediction token must have a real official cache payload. Preserve the scorer result and per-token CSV rather than treating an image import or container exit as a score.

For full NAVTEST, run inference with the dumped full-training config and `--data-root "$DATA" --memory "$EVAL_MEMORY"`, rather than `--mini-prepared`. Set `EVAL_MEMORY` from the evaluation budget. The original evaluator exports `metrics_navsim_submission.pkl` when the configured test filter is the original `navtest.yaml`. Independently rescore this submission with the original CPU wrapper and your complete official cache:

```bash
FULL_EVAL="$RUNS/vad50_full_eval"
: "${EVAL_MEMORY:?Set the full NAVTEST evaluation memory limit}"
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$RUNS/vad50_full/e2e_vadv2_50pct_rank0.py" \
  --data-root "$DATA" --memory "$EVAL_MEMORY" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --maps "$MAPS" --precision fp32 --output "$FULL_EVAL"

docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 64g --cpus 16 \
  -v "$SOURCE:/source:ro" -v "$FULL_EVAL:/predictions:ro" \
  -v "$CACHE:$CACHE:ro" -v "$WORK:$WORK:rw" -w /source/projects/AlgEngine \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export NAVSIM_DEVKIT_ROOT=/source/ascend/third_party/navsim
    bash scripts/e2e_navsim_official_rescore.sh \
      /predictions/metrics_navsim_submission.pkl "$1" "$2/vad50_full_score"
  ' bash "$CACHE" "$WORK"
```

The full-score CPU example is a starting resource limit, not a measured full benchmark budget. Keep cache payloads reachable at the paths in their metadata, or relocate/rebuild the index. Generated trajectory models such as DiffusionDrive require official rescoring; the inference CSV alone must not be treated as final PDMS. The recorded GPU/NPU comparison is limited to two official100 FP32 planners on 22 anchors, not a complete benchmark or GPU training.

<a id="x86-cpu-scoring-of-gpu-exports"></a>

### x86 CPU scoring of GPU exports

The public `ascend/scoring.py` is byte-identical to the scorer used for the
[recorded original-CUDA comparison](../ascend/results/gpu-npu-official100-eval22-20261016.json).
It accepts the original `predictions.pkl` list and matching official NAVSIM
metric-cache payloads. It samples `trajectory[4::5]` for the original eight-pose
score; the [CPU comparison tool](../tools/validation/README.md) measures all 40
exported poses separately. Training PDM labels and embedded prediction metrics
are not a substitute for the official cache or rescoring.

On the measured Xeon Gold 6430, NumPy 1.23.4/OpenBLAS 0.3.20 default dispatch
produced incorrect LQR results and zero scores. The unchanged packages and
scoring formula recovered with `OPENBLAS_CORETYPE=Haswell` in fresh CPU
processes. This condition is verified for that CPU/software combination;
it is not a global NPU environment setting or a general recommendation for
other BLAS builds. See the [upstream issue](https://github.com/numpy/numpy/issues/24903).

From this repository root, use the existing CUDA Python environment with the
original NAVSIM/NuPlan dependencies and the checked-out NAVSIM submodule. Set
paths to your own completed 22-token export, its official cache and a new output:

```bash
GPU_PREDICTIONS=/path/to/your_GPU_run/predictions.pkl
OFFICIAL_CACHE22=/path/to/official22/metric_cache
GPU_SCORE_OUT=/path/to/new_GPU_export_CPU_score
CUDA_VISIBLE_DEVICES='' OPENBLAS_CORETYPE=Haswell \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python - "$GPU_PREDICTIONS" "$OFFICIAL_CACHE22" "$GPU_SCORE_OUT" <<'PY'
import pickle
import sys
from pathlib import Path
from ascend.scoring import score_predictions

predictions, cache, output = map(Path, sys.argv[1:])
if output.exists() or output.is_symlink():
    raise SystemExit("Use a fresh CPU-score output directory")
with predictions.open("rb") as stream:
    records = pickle.load(stream)
if len(records) != 22 or len({row["token"] for row in records}) != 22:
    raise SystemExit("This example requires the complete 22-token export")
payloads = list(cache.rglob("metric_cache.pkl"))
if len(payloads) != 22 or {path.parent.name for path in payloads} != {row["token"] for row in records}:
    raise SystemExit("Provide exactly the 22 matching official cache payloads")
print(score_predictions(records, cache, output))
PY
```

Keep `official_score.json` and `official_scores.csv`, including any anomalous
attempts. The measured two models matched the NPU reference in all 40×3
trajectory values and every per-token official metric after the CPU environment
correction. The source/framework stacks differ, and matching JPEG bytes do not
prove bitwise-identical preprocessing tensors. This is bounded inference/output
evidence, without a quality-equivalence threshold or a pure-hardware diagnosis.

## 7. Evaluate NAVTRAIN and select rare cases

Rare-case selection needs actual per-scenario results from the training split. The original GPU chunking scripts are not NPU launchers. For NPU, create a separate inference config from the dumped full-run config and point its test phase at the corresponding NAVTRAIN inputs. This leaves the training config and checkpoint unchanged.

```bash
NAVTRAIN_CFG="$WORK/navtrain_inference.py"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 8g --cpus 2 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python - "$1" "$2" <<'"'"'PY'"'"'
import sys
from mmcv import Config
cfg = Config.fromfile(sys.argv[1], import_custom_modules=False)
for phase in ("val", "test"):
    ds = cfg.data[phase]
    ds.ann_file = "/data/full/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
    ds.nav_filter_path = "/workspace/WorldEngine/projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain_50pct.yaml"
    ds.train_pdm_path = "/data/full/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain"
    ds.test_pdm_path = ds.train_pdm_path
    ds.pipeline[0].img_root = "/data/full/raw/openscene-v1.1/sensor_blobs/trainval"
cfg.dump(sys.argv[2])
PY
  ' bash "$RUNS/vad50_full/e2e_vadv2_50pct_rank0.py" "$NAVTRAIN_CFG"

python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$NAVTRAIN_CFG" --data-root "$DATA" --memory "$EVAL_MEMORY" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --maps "$MAPS" --output "$RUNS/vad50_navtrain_eval"
```

The commands in this section and the full-data fine-tuning example below use the 50% split. For whole-split rare-case selection, use the corresponding 100% training dump and planner checkpoint, set the inference filter to `navtrain.yaml`, and also change the sampler's `--base-split` to that file. Merge `navtrain_collision.yaml`, `navtrain_off_road.yaml` and `navtrain_ep_1pct.yaml`; their prefix comes from the base-split filename, not from the model name. Use a fresh output for this separate result set.

The repository publishes 50% rare-log fine-tuning configs, not separate `100pct_rlft_rare_log.py` configs. A derived whole-split fine-tuning run must explicitly override the existing rare-log config with `--train-filter "$SOURCE/projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain.yaml"` and provide the newly generated whole-split `--rare-split`. Record the dataset override and initializer separately, and retain a new output; this is not a published or validated 100% fine-tuning recipe. Reproducing a 100%-trained base experiment requires its matching planner checkpoint. A compatible 50%-trained initializer can also be used, but must be recorded as that initializer rather than a 100%-trained base. Changing only the inference filter does not change the fine-tuning dataset.

Large evaluations may be divided into YAML token chunks using the same pattern; retain the original tokens, evaluate each chunk in a fresh output, and merge per-token results without duplicate tokens. The command above loads the full annotation/PDM; changing a filter does not eliminate that resident memory.

Remove the summary row from the exported CSV, then call the original rare-case script. Merge its three token sets into one explicit `--rare-split` input for NPU fine-tuning:

```bash
RARE_OUT="$WORK/vad50_rare"
mkdir -p "$RARE_OUT"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 8g --cpus 2 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python - "$1" "$2" <<'"'"'PY'"'"'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[df.token != "average"]
assert len(df) and not df.token.duplicated().any()
df.to_csv(sys.argv[2] + "/navtrain.csv", index=False)
PY
    python projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py \
      --pdm-result "$2/navtrain.csv" \
      --base-split projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain_50pct.yaml \
      --output-dir "$2" --ep-percentile 1
    python - "$2" <<'"'"'PY'"'"'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
files = [root / ("navtrain_50pct_" + name + ".yaml")
         for name in ("collision", "off_road", "ep_1pct")]
splits = [yaml.safe_load(path.read_text()) for path in files]
merged = dict(splits[0])
merged["tokens"] = sorted(set().union(*(set(row["tokens"]) for row in splits)))
assert merged["tokens"], "No rare tokens selected; inspect the evaluation results"
(root / "navtrain_rare_union.yaml").write_text(yaml.safe_dump(merged))
PY
  ' bash "$RUNS/vad50_navtrain_eval/metrics.csv" "$RARE_OUT"
```

These files select collision, off-road and bottom-1%-progress cases. Very small or entirely unsuccessful inputs may have no positive safe progress values; the original percentile selection then fails. Use a representative result set and inspect the source metrics instead of altering the selection logic to force output.

## 8. Fine-tuning and original configuration variants

The [October 16 BF16 v7 real-log FT record](../ascend/results/real-log-ft37-bf16-v7-short8-20261016.json) is separate from NAVTRAIN139: both original 37/22 configurations completed eight updates, own update-4 restoration and bitwise replay comparison. That record includes complete VAD evaluation and Hydra inference interrupted at console progress 14/22 before official scoring. A [fresh Hydra completion on October 17](../ascend/results/hydra-ft37-bf16-eval22-20261017.json) used the same own-update-8 checkpoint for all 22 anchors and the original NAVSIM CPU scorer, measuring PDMS 0.9338557270555728; the interrupted run remains recorded. These functional results do not add producer-synthetic coverage or establish convergence.

For rare **real-log** VAD fine-tuning, initialize from the matching planner checkpoint and supply the generated YAML. This does not require a SimEngine rollout:

```bash
python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
  --data-root "$DATA" --rare-split "$RARE_OUT/navtrain_rare_union.yaml" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --runtime --batch-size 1 --workers 0 --precision fp32 \
  --maps "$MAPS" --official-metric-cache "$CACHE" --validate \
  --memory "$TRAIN_MEMORY" --cpus 64 --pids-limit 8192 \
  --output "$RUNS/vad50_rare_ft"
```

The published `VAD_BASE` can also initialize this config. This original VAD config enables LoRA and reward-based fine-tuning, with `rl_finetuning=False`; it must not be described as having an enabled policy-gradient branch. The Hydra config `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct_rlft_rare_log.py` enables the policy-gradient path through its base config. Use `HYDRA_BASE` or a matching trained Hydra checkpoint, Hydra-derived rare cases, and a fresh output. Fine-tuning starts a new optimizer/schedule; use `--checkpoint`, not `--resume`, to initialize it from a different run.

For a small-data check, use the two profile commands below. They select the corresponding original fine-tuning configs through `ascend/train.py`'s profile mapping. Do not retain a VAD `--config` when switching to Hydra: an explicit config takes precedence over the profile. These iteration runs omit `--validate`; epoch validation cannot be combined with `--steps`.

```bash
python3 ascend/run_container.py --devices 0 --profile vadv2_ft \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --rare-split "$MINI_RARE" \
  --checkpoint "$VAD_BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/vad_rare_mini8"

python3 ascend/run_container.py --devices 0 --profile hydramdp_rl \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --rare-split "$MINI_RARE" \
  --checkpoint "$HYDRA_BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/hydra_rare_mini8"
```

The mini must include genuine rare anchors and retrievable non-rare candidates from those same logs, each with its original four-frame queue and camera coverage. A normal anchor in another training log does not meet the original fine-tuning dataset's same-log retrieval rule. Verify the returned tokens and both branches in the original dataset check. These eight-update commands are functional checks, not full fine-tuning or convergence results.

There are 44 published configurations. The original [configuration guide](config_guide.md) explains their relationships. Pass the actual repository config using `--config`; two perception configs use original batch 1, while 42 planning configs use original batch 2. The NPU wrapper preserves the model/head/dataset branch and exposes schedule/batch overrides explicitly.

Eight configs require genuine rollout/replay/BWM/SimScale inputs, supplied through `--synthetic-root`. Do not use a normal real-data mini as synthetic data. The [portable SimScale recipe](../ascend/data_tools/README.md) prepares the two measured configs from roughly 47.50 GB of locked inputs plus maps/image/output. The other six exports are still required from their actual producers. SimScale's published 16384-candidate reward is not the 8192 training label; follow the recipe rather than truncating or renaming it. Closed-loop simulation and new rollout rendering follow the original [SimEngine guide](simengine_usage.md) on the appropriate GPU setup.

## 9. Logs, outputs and troubleshooting

Each run mounts the source/data read-only and its output writable. A normal runtime output contains:

```text
run/
├── launch.json                  # exact image, device selection and command
├── console.log
├── train_exit_code, exit_code
├── <config_stem>_rank0.py       # actual configuration for later inference
├── runtime_rank<N>.json         # per-rank final checks and precision/state details
├── runtime_rank<N>.jsonl        # update/loss/tokens/LR/resource stream
├── iter_<N>.pth or epoch_<N>.pth
├── latest.pth
└── validation/epoch_<N>/        # with --validate on epoch training
```

Commands without `--detach` wait for completion. Add `--detach` only when the device reservation and monitoring are arranged; the launcher returning a container ID then does not mean training has finished.

```bash
cat "$RUNS/vad_mini8/exit_code"
tail -n 60 "$RUNS/vad_mini8/console.log"
cat "$RUNS/vad_mini8/runtime_rank0.json"
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  worldengine-npu-vad_mini8
```

| Symptom | Check and corrective action |
| --- | --- |
| Image ID differs | Inspect the selected tag; use the recorded ID for a new run. Resume requires the original image, not a changed tag. |
| Output already exists | Select a new output; preserve prior logs/checkpoints. |
| Missing assets/history/8192 PDM | Correct the real directory layout and selected camera coverage; rerun preparation/preflight. |
| Full PKL run requires `--memory` | Budget per-rank dataset residency plus training/evaluation/workers; use an explicit limit. |
| `can't start new thread` / EAGAIN | Inspect container PID/thread count. Default PID limit is `max(2048, 512*ranks)`; the 16-rank examples use 8192. |
| `EL0020`, device occupied, `507899` | Stop starting jobs on that chip; check exact device/container ownership with the machine administrator. Do not infer availability only from a blank process list. |
| Resume contract mismatch | Compare source, image, data/filter, precision/environment, world size and schedule. Restore the matching inputs instead of bypassing the guard. |
| NaN/Inf | Retain the run and investigate the recorded raw loss, gradients and training state. |
| Completed comparison outside the original tolerance | Retain the original numerical report and checkpoints. Keep the original thresholds; successful restoration does not establish numerical replay. |
| Official scoring cannot find cache payloads | Check token coverage and paths in `metadata/*.csv`; regenerate the relocated index/cache. |

The [validation report](../ascend/FULL_ALGENGINE.md) records the measured conditions, versions and results for small-data training, restoration and scoring. The public rebuild and two 16-rank LoRA fine-tuning workflows have actual small-data evidence. Four independent synchronous base-planner controls completed, with two comparisons outside the original tolerance. Full NAVTRAIN convergence, complete independent NAVTEST results, the other six genuine synthetic configurations and sustained operation in the target environment remain unvalidated.

The measured public-source reproduction used 37 training candidate anchors and 22 log-disjoint evaluation anchors. Both LoRA fine-tuning comparisons were bitwise equal on the newly built image. [Versioned evidence](../ascend/results/actual-public-reproduction-16chip-20261014.json) records its configurations, inputs and boundaries. Preserve the original camera directory layout when passing the camera root for the original NAVTEST annotation.
