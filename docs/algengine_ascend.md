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
| Audit complete NAVTRAIN inputs on CPU | `ascend/tools/verify_navtrain_inputs.py`; see [full-input checks](#cpu-audit-of-complete-navtrain-inputs) |
| Reproduce full-input short training | [Four-rank FP32/BF16, eight updates and own 4→8 resume](#full-input-short-training) |
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
git clone --branch ascend-910c-algengine-fp32-opt-20261029 --recurse-submodules \
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

### CPU audit of complete NAVTRAIN inputs

[`verify_navtrain_inputs.py`](../ascend/tools/verify_navtrain_inputs.py) checks only the original 100% NAVTRAIN split. It reuses the checkout's original `inspect_phase` for all selected anchors, current plus three history frames, eight camera channels and finite 8192-entry PDM arrays. It verifies the complete annotation/PDM/vocabulary files against the publisher lock, then hashes, verifies, reopens and fully decodes every unique referenced JPEG. The public CLI also requires original 1920×1080 RGB JPEGs. It neither loads a model nor checks NAVTEST, maps or metric-cache coverage; keep the broader preflight below for evaluation inputs.

Use a new output directory outside the source/data roots and a CPU-only container. This command deliberately exposes no accelerator devices and uses the verified image ID from section 1. The installed environment needs MMCV/NumPy/PyYAML for `preflight.py` and Pillow for decoding.

```bash
: "${DEADLINE_UTC:?Set a future UTC cutoff, including Z or +00:00}"
QC_DIR="$WORK/navtrain100_inputs"
mkdir "$QC_DIR"

docker run --rm --runtime runc --network none --memory 128g --cpus 8 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" -v "$QC_DIR:/output:rw" -w /source \
  "$IMAGE_ID" /opt/conda/envs/bevformer/bin/python ascend/tools/verify_navtrain_inputs.py \
    --source /source --data-root /data \
    --config projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py \
    --decode-workers 4 --output /output/navtrain100.json \
    --deadline-utc "$DEADLINE_UTC" --stop-file /output/STOP
```

Inspect the process exit code and `navtrain100.json`. Acceptance requires `actual_exit=0`, `result=pass_complete_original_NAVTRAIN_inputs_and_required_JPEG_decode_CPU_only`, an empty error list, matching requested/selected counts, zero missing/invalid annotation/image/PDM entries, and `decoded=decode_passed=unique_required_JPEGs` with `decode_failed=0`. The completed `navtrain100.json.images.jsonl` records every relative path, content SHA256 and decode result; the JSON also binds its digest, source/config/filter/asset identities and before/after memory/device states. A `.part` ledger or an interrupted/rejected report is not complete coverage.

The deadline is mandatory, timezone-aware UTC and must be in the future; it has no project-specific calendar cap. Create `$QC_DIR/STOP` to request a cooperative stop. Four decode workers keep at most eight futures queued. The 128g container limit is an input-inspection setting. The earlier inspector peaked at approximately 51.99 GB RSS, which is a reference measurement rather than a training RAM budget.

The [October 20 full-data result](../ascend/results/full-navtrain-inputs-20261020.json) covers 103,288 anchors and 1,219,960 required JPEGs. That actual run used the original private worker SHA256 `453d8e893ccef3720b4323e12e6c862f0fdee5020b85384c3ef37e1b3c31ab0b`; its raw report SHA256 is `882d5a98a5476b7ae135cd9b33a3c37f6c65fa7d3ef3d5a7c36d6d7bfdf72284`. The public CLI retains the inspection algorithms and has passed 14 local synthetic CPU fixtures; a complete-data run of this CLI has not been repeated. Its deadline/CPU guards and explicit image-dimension gate must not be presented as new full-data or NPU training acceptance. Run `python -m unittest discover -s ascend/tools -p test_verify_navtrain_inputs.py -v` for those fixtures.

### Full-input short training

The [October 20 measurement](../ascend/results/full-navtrain-short8-20261020.json) used the complete 103,288-anchor NAVTRAIN loader and the [audited 1,219,960 JPEGs](../ascend/results/full-navtrain-inputs-20261020.json). VADv2 and HydraMDP each completed FP32 and BF16 imitation training on four ranks, with eight optimizer updates and exact restoration of their own update-4 model/optimizer state. All 32 rank reports and 128 per-rank update records passed. Each case consumed the same 32 distinct current anchors, with 128 history/current input frames; this does not establish a complete epoch or convergence. HydraMDP policy-gradient fine-tuning was not enabled in these `e2e_*_100pct` cases.

The host has eight dual-chip physical NPU cards and 16 chips. These jobs used host chips 0–3 on physical cards 0–1. Source `0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922` / 245 Python files and image `sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48` identify the measured runtime. Configured split YAML paths are resolved against AlgEngine before runtime hashing; checkpoints from an earlier source contract require a new training initialization rather than full-state restoration across that source change.

Set `MODEL` and `PRECISION` for one case below; repeat with fresh outputs for the other combinations. `DATA` is the complete dataset root, not a prepared mini. The 512g / 32-CPU limits reproduce the four-rank allocation and are not minimum resource requirements. The original optimizer and LR options are retained with an explicit eight-update horizon; the 500-update warmup is therefore unfinished.

```bash
MODEL=vadv2                         # or hydramdp
PRECISION=fp32                      # or bf16; keep unchanged on resume
DEVICES=0,1,2,3                     # verify actual host ownership/mapping
case "$MODEL" in
  vadv2) CONFIG=projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py
         BASE="$VAD_BASE_100" ;;
  hydramdp) CONFIG=projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py
            BASE="$HYDRA_BASE_100" ;;
  *) exit 2 ;;
esac
FULL_RUN="$RUNS/${MODEL}_${PRECISION}_fullinput"

python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$CONFIG" --data-root "$DATA" --checkpoint "$BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --stop-after 4 --upstream-schedule \
  --checkpoint-interval 4 --check-interval 1 --batch-size 1 --workers 0 \
  --precision "$PRECISION" --msda-backend cuda-compatible \
  --memory 512g --cpus 32 --output "${FULL_RUN}_first4"

python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$CONFIG" --data-root "$DATA" --checkpoint "$BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --upstream-schedule \
  --checkpoint-interval 4 --check-interval 1 --batch-size 1 --workers 0 \
  --precision "$PRECISION" --msda-backend cuda-compatible \
  --resume "${FULL_RUN}_first4/latest.pth" \
  --memory 512g --cpus 32 --output "${FULL_RUN}_resume8"
```

Require both commands to exit 0, every rank to report 0→4 / 4→8 with four updates, finite raw losses and an updated/synchronized monitored parameter, and every resume rank to report `pass_exact_saved_state_restore`. With `--check-interval 1`, all gradients present at each update pass the finite-value gate. Synchronization checks one monitored parameter with `atol=1e-6, rtol=1e-5`; it does not compare the complete model bitwise. Independent CPU inspection found all eight saved model and Adam payloads finite. Restored model/Adam Tensor counts were 889/777 for VADv2 and 909/837 for HydraMDP. Exact saved-state restoration is separate from continuous-eight-versus-four-plus-four numerical replay equivalence.

BF16 recorded actual Conv2d/Linear BF16 activations, FP32 masters and present gradients/Adam states, and FP32 MSDA/geometry/planning-loss/normalization regions. Training-end metadata covered 730/750 named parameters and 259/279 gradient-bearing parameters for VADv2/HydraMDP; 310 trainable parameters per model had no gradient record. Saved optimizer group/state-ID/schema coverage matched FP32. This is membership/dtype evidence, not a comparison of complete gradient values.

The separate [single-chip FP32 measurement](../ascend/results/single-chip-capacity-20261020.json) ran each model for four updates with batch 1, workers 0, a 192g limit and 16 CPUs. Single-process peak RSS was 50.69/50.79 GiB; NPU allocator peaks were 13.05/13.07 GiB allocated and 19.81/19.81 GiB reserved. To reproduce that scope, use only the first command with one available host chip, `PRECISION=fp32`, `--memory 192g --cpus 16` and fresh outputs. Single-chip BF16, resume, complete epochs, minimum RAM and throughput were not measured by those cases. Neither single-process RSS nor allocator reservations represent total host RAM or total device HBM consumption.

The [complete NAVTEST input audit](../ascend/results/full-navtest-inputs-20261020.json) independently passed 12,146 anchors, 136 logs and all 142,976 required JPEGs, including three history frames and eight cameras. This input audit can be shared by the six evaluation cases; complete model inference, official metric-cache validation and official scoring require their own evidence. Co-running job times are observations rather than a performance benchmark.

### Training and evaluation preflight

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

Omitting `--steps` and `--epochs` keeps the original eight-epoch horizon, optimizer and LR schedule. The migration's `--validate` runs evaluation each epoch. For 100%, change the config to `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py`, keep the original initializer `track_map_nuplan_r50_navtrain_50pct_bs1x8.pth`, and select a fresh output. The 50% filter has 50,000 anchors; the full official split has 103,288. Full-input eight-update training has completed as recorded above; complete-epoch training and convergence remain unaccepted.

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

## 10. Optional FP32 planning inference optimizations

Inference keeps the original path by default. `--inference-optimization` selects a temporary inference-only implementation; it does not change training, checkpoints, model parameters, the test pipeline or scoring. The options require `--precision fp32 --msda-backend cuda-compatible` with Matmul/Conv HF32 disabled.

| Option | Behavior and scope |
| --- | --- |
| `none` | Original implementation and query chunk 2048. |
| `point-parallel` | Batches independent MSDA sampling points. Retains compensated FP32 coordinates and ascending level/point accumulation. `--msda-query-chunk` accepts 2048, 4096 or 8192; larger chunks require more transient memory. |
| `point-parallel-last-frame` | Adds planning-only NAVFormer frame elimination for a batch of one and exactly three history frames plus the current frame. The final original call still processes all four frames. Three discarded outer prefix calls are removed, reducing BEV frame evaluations from 11 to 4. The six NumPy draws from those discarded eval GridMask calls are preserved. |

The last-frame path checks the original model methods and GridMask source before use. Training mode, enabled gradients, perception, different queues or changed source use the original frame path and record the reason. Gradient-bearing MSDA inputs use the original forward/backward implementation. The original functions are restored when inference exits, including on exceptions. These fallbacks preserve execution; inspect the recorded counters to confirm that the requested optimization was actually used.

Use the same dumped inference configuration, official weights, data, image, precision, deterministic settings and logical chips for both runs. The paths below follow the mounts in section 6; the configuration must already reference those paths. Use fresh output directories.

```bash
# DEVICES selects logical chips: 0,...,15 is 8 dual-chip physical cards.
COMMON=(--task inference --devices "$DEVICES"
  --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$INFERENCE_CONFIG" --data-root "$DATA"
  --checkpoint "$CHECKPOINT" --vocab "$VOCAB" --maps "$MAPS"
  --memory "$EVAL_MEMORY" --cpus 64 --pids-limit 8192
  --precision fp32 --msda-backend cuda-compatible --deterministic)
python3 ascend/run_container.py "${COMMON[@]}" \
  --output "$RUNS/fp32_original"
python3 ascend/run_container.py "${COMMON[@]}" \
  --inference-optimization point-parallel-last-frame --msda-query-chunk 8192 \
  --output "$RUNS/fp32_optimized"
```

Inside the same installed runtime, compare the two result directories:

```bash
python ascend/tools/compare_inference.py \
  --baseline /results/fp32_original --candidate /results/fp32_optimized \
  --output /results/fp32_comparison.json
```

The comparator checks source/config/checkpoint/runtime/precision/topology identity, ordered tokens and every typed result field with raw tensor/array equality. Exit 0 also requires actual use of the requested optimization. `inference_validation.json` records the selected option, query chunk, actual calls and fallback reasons observed on rank 0. Independent NAVSIM scoring remains required as described in section 6.

For performance acceptance, repeat unprofiled A/B runs with the same device mapping and data, and report full job time separately from warmed model windows. The original `elapsed_seconds` starts immediately before the original test script. It includes model/checkpoint loading, inference and result collection, and excludes earlier Python/Torch imports and configuration/precision setup. Container walltime includes those earlier phases as well; neither measure is isolated kernel latency. Do not compare an audited/profiled run against an uninstrumented run. A short subset confirms functionality and numerical consistency only; it does not establish complete NAVTEST quality or training convergence. To return to the original execution path, omit the optimization flags or set `--inference-optimization none` and leave the chunk at 2048.

[Validation record](validation/algengine_fp32_optimization_20261029.json): both models completed the public `ascend/inference.py` baseline/optimized paths on 22 genuine official NAVTEST anchors with 16 ranks. All typed output fields were raw exact and the requested optimization counters were nonzero. The record separates auxiliary operator/139-anchor audits and retains the full-benchmark, GPU and training limits.

The check was repeated on merged source `f24d5f5`, which includes published migration baseline `01445caa`: all four NPU runs and both public CPU comparisons exited 0; both models retained raw-exact outputs. The same record appends this source identity and 51 passing CPU regressions without replacing the original `0bcaff` measurements.

### Recovering optimized inference

Use `ascend/tools/optimized_journal_inference.py` for the explicitly pinned optimized runtime `7a8c50dc…`. Optimization remains disabled by default; select `point-parallel-last-frame` and chunk 8192 explicitly. The original `journal_inference.py` is retained unchanged at SHA `89e24b8d…` and pins the original migration source `76c51f77…`; it does not accept optimization flags. The older October 20 archive uses a different source and tool hash. Never reuse its journal with this entry.

Prepare the image, 16 chip mounts, full historical annotation and five-file immutable input manifest using the [journal input and mount contract](algengine_journal.md#inputs-and-mounts). The manifest covers annotation, PDM, vocabulary, filter and approved input QC. Run inside the verified image with the same source, checkpoint, data, FP32 settings and topology throughout. Every attempt needs a new output child and UUID; the journal directory stays the same.

```bash
COMMON=(--source-root /workspace/WorldEngine --config /config/inference.py
  --checkpoint /weights/initial.pth --locked-inputs /proof/locked_inputs.json
  --journal-root /journal --precision fp32 --msda-backend cuda-compatible
  --inference-optimization point-parallel-last-frame --msda-query-chunk 8192
  --deterministic --expected-anchors 22
  --expected-source-sha256 7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb
  --expected-config-sha256 "$CONFIG_SHA" --expected-checkpoint-sha256 "$CHECKPOINT_SHA"
  --expected-locked-inputs-sha256 "$LOCKED_INPUTS_SHA" --deadline-utc "$DEADLINE_UTC")
torchrun --standalone --nproc_per_node=16 ascend/tools/optimized_journal_inference.py \
  "${COMMON[@]}" --journal-mode fresh --attempt-id "$STOP_UUID" \
  --output /attempts/stop1 --stop-after-steps 1
# Expected controlled pause: real torchrun return 1, all 16 partial receipts,
# one committed position per rank, and no complete predictions.pkl.
torchrun --standalone --nproc_per_node=16 ascend/tools/optimized_journal_inference.py \
  "${COMMON[@]}" --journal-mode resume --attempt-id "$RESUME_UUID" \
  --output /attempts/resumed
```

`CONFIG_SHA`, `CHECKPOINT_SHA` and `LOCKED_INPUTS_SHA` are SHA256 hashes of the exact mounted files. `DEADLINE_UTC` must be a future UTC timestamp. Set `HCCL_DETERMINISTIC=true` and `CLOSE_MATMUL_K_SHIFT=1` before starting Torch. A return of 1 alone does not prove a controlled pause: verify every partial receipt and its committed prefix before resuming. Resume must return 0 and cover every configured anchor.

The [same validation record](validation/algengine_fp32_optimization_20261029.json) appends both models' actual 22-anchor/16-rank continuous, stop1, resume and completed-journal positive-position replay checks. Every exported typed field matched the uninstrumented baseline. Initial and per-step four-source RNG matched between the two journal routes; the uninstrumented baseline has no per-step RNG trace. Saved prefix bytes were unchanged. This validates that bounded recovery scope; full-dataset recovery, training convergence, GPU equivalence, scoring and performance remain separate gates. Source, tool, mode, chunk, precision, world size and input changes require a fresh compatible contract rather than adopting old progress.

### Inference recovery with file collection

`ascend/tools/filesystem_journal_inference.py` writes results and rank reports to separate shared directories. Each rank atomically publishes a complete part and SHA receipt; rank 0 merges in original sampler order and truncates padding. The final collections do not use an HCCL barrier. Initial model synchronization and journal-prefix inventory still use HCCL. This entry requires the pinned `7a8c50dc…` runtime, FP32, original vocabulary-planning heads and an explicit world size of 4, 12 or 16.

Use the [input manifest and mount layout](algengine_journal.md) for the configuration, complete history annotation, PDM, vocabulary and passed input QC. The [current NAVTEST input-check commands](navtest_input_audit.md) bind runtime 7a8 / 247 files. Retain the fixed image and five-file manifest; keep journals from different entries and tool versions separate.

The measured mapping mounts only host `/dev/davinci4` through `/dev/davinci7` plus the three control devices: `WORLDENGINE_HOST_DEVICE_IDS=4,5,6,7`, container `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, local_rank 0–3. Reuse the installation's source, driver and dependency mounts and set the working directory to `/workspace/WorldEngine`. Mount source-external output, journal and collection directories as `/attempts:rw`, `/journal:rw` and `/collections:rw`; keep inputs, configuration, weights and `/proof` read-only. Run preparation in a device-free `--runtime runc` CPU container with Torch backend autoload disabled; mount the four assigned chips only for the NPU run.

Generate and export a fresh `ATTEMPT_ID` for every attempt (`python -c 'import uuid; print(uuid.uuid4().hex)'`). Run the code below in the CPU container with `/config/inference.py`, `/weights/initial.pth` and `/proof/locked_inputs.json`, using the same input paths as inference. Set `WORLDENGINE_IMAGE_ID` to the verified fixed image ID. The code derives tokens in original annotation order and keeps the full history annotation. Collection directories and `plan.json` must be fresh; preserve `run_context.json` for attempts sharing a journal.

```bash
/opt/conda/envs/bevformer/bin/python - <<'PY'
# Run in the CPU-only image with the same input paths and /collections mounted RW.
import hashlib, json, os, pickle, sys
from pathlib import Path
import yaml
sys.path.insert(0, "/workspace/WorldEngine/ascend/tools")
from file_collection import create_collection

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

locked_path = Path("/proof/locked_inputs.json")
locked = json.loads(locked_path.read_text())
for item in locked.values():
    assert Path(item["path"]).stat().st_size == item["bytes"]
    assert digest(item["path"]) == item["sha256"]
assert locked["annotation"]["sha256"] == "59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b"
with open(locked["annotation"]["path"], "rb") as stream:
    infos = pickle.load(stream)  # Only after verifying the official bytes above.
infos = infos["infos"] if isinstance(infos, dict) else infos
selected = set(yaml.safe_load(Path(locked["filter"]["path"]).read_text())["tokens"])
ids = [row["token"] for row in infos if row["token"] in selected]
assert len(ids) == len(set(ids)) == len(selected)
world = 4
context = dict(source_sha256="7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb",
    config_sha256=digest("/config/inference.py"), checkpoint_sha256=digest("/weights/initial.pth"),
    locked_inputs_sha256=digest(locked_path), image_id=os.environ["WORLDENGINE_IMAGE_ID"],
    world_size=world, anchors=len(ids), precision="fp32", backend="cuda-compatible",
    optimization="point-parallel-last-frame", query_chunk=8192,
    intended_host_devices=[4, 5, 6, 7], intended_container_devices=[0, 1, 2, 3])
raw = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
context_path = Path("/collections/run_context.json")
if context_path.exists():
    assert context_path.read_bytes() == raw
else:
    with context_path.open("xb") as stream:
        stream.write(raw)
run_hash = hashlib.sha256(raw).hexdigest()
attempt = os.environ["ATTEMPT_ID"]  # A fresh UUID hex for every attempt.
root = Path("/collections") / attempt
root.mkdir(exist_ok=False)
plan = dict(run_id=attempt, run_contract_sha256=run_hash)
for name, size in [("results", len(ids)), ("reports", world)]:
    folder = root / name
    checksum = create_collection(folder, run_id=attempt, world_size=world, size=size,
        source_sha256=run_hash, sample_field="token" if name == "results" else None,
        expected_ids=ids if name == "results" else None)
    plan[name] = dict(root=str(folder), contract_sha256=checksum)
path = root / "plan.json"
path.write_text(json.dumps(plan, indent=2))
print("PLAN_SHA=" + digest(path), "ANCHORS=" + str(len(ids)))
PY
```

Pass the printed `PLAN_SHA` and `ANCHORS`, and the actual configuration/checkpoint/manifest hashes `CONFIG_SHA`, `CHECKPOINT_SHA`, `LOCKED_INPUTS_SHA`, to the NPU container. Set `DEADLINE_UTC` to a future UTC time. Full NAVTEST uses its original complete configuration and 12,146 anchors; the 22-sample gate uses its corresponding official-token filter.

```bash
# Inside the NPU image; input/proof paths match the CPU preparation above.
export WORLDENGINE_IMAGE_ID=sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
export WORLDENGINE_HOST_DEVICE_IDS=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_DETERMINISTIC=true CLOSE_MATMUL_K_SHIFT=1
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200

/opt/conda/envs/bevformer/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 ascend/tools/filesystem_journal_inference.py \
  --source-root /workspace/WorldEngine --config /config/inference.py \
  --checkpoint /weights/initial.pth --locked-inputs /proof/locked_inputs.json \
  --expected-source-sha256 7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb \
  --expected-config-sha256 "$CONFIG_SHA" --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-locked-inputs-sha256 "$LOCKED_INPUTS_SHA" \
  --precision fp32 --msda-backend cuda-compatible --deterministic \
  --inference-optimization point-parallel-last-frame --msda-query-chunk 8192 \
  --expected-world-size 4 --expected-anchors "$ANCHORS" \
  --journal-root /journal --journal-mode "${JOURNAL_MODE:-fresh}" \
  --attempt-id "$ATTEMPT_ID" --output "/attempts/$ATTEMPT_ID" \
  --collection-plan "/collections/$ATTEMPT_ID/plan.json" \
  --expected-collection-plan-sha256 "$PLAN_SHA" \
  --deadline-utc "$DEADLINE_UTC" --stop-file /attempts/STOP
```

After an interruption, retain `/journal` and earlier outputs. Prepare a new attempt UUID, collection directories and plan with the same source, inputs, configuration, weights, mode, chunk and device contract, then set `JOURNAL_MODE=resume`. Add `--stop-after-steps 1` for a controlled check and verify every rank's partial receipt and saved position before recovery. The operator controls the STOP file. Complete acceptance requires actual inference exit 0, the configured anchor count and complete rank reports. Run independent official scoring as described in section 6.

The [measured record](validation/algengine_fp32_optimization_20261029.json) separates the public entry's two-model, 22-anchor/world4 tests from earlier private world16/world4 auxiliary runs. VAD continuous, resumed and completed-journal replay outputs, step RNG and saved prefix bytes were exact; Hydra's continuous outputs matched. World12 has CPU contract coverage only. These small-set results do not accept full NAVTEST or full-data recovery of both models.
