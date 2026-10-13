# AlgEngine on Ascend 910C / A3

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
distinguished in the evidence. See the full-module guide for current regression
results and remaining numerical, data and convergence boundaries.

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

CANN 9.1.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, DrivingSDK 26.1.0.post2, MMCV 1.7.2, MMDet 2.24.0, MMDet3D 1.0.0rc4 and NumPy 1.23.5 were exercised. Torch's `+cpu` version string is expected for this NPU stack. Matmul HF32 was disabled and convolution HF32 allowed; FP32 denotes tensor/training dtype, not a GPU numerical-equivalence claim.

Observed package versions are retained in `docker/versions-20261013.json`; the
separate geospatial Conda prefix's resolved ARM64 packages and package checksums
are in `docker/geo-linux-aarch64-20261013.explicit.txt`. Its Python patch version
can differ from the main training prefix. These are build snapshots; the exact
saved image is the verified offline reproduction artifact.

The tested host was Atlas 800I A3 inference series, Ascend910_9362, with 64 GiB HBM per logical chip. Vendor training support for this SKU is a separate boundary: [official ModelZoo hardware table](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md). Experimental backpropagation does not extend that support table.

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
mkdir -p "$RUNS/resume_checks"
docker run --rm --memory 24g --cpus 8 --entrypoint bash \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/source:ro" -v "$RUNS:/runs:ro" \
  -v "$RUNS/resume_checks:/checks:rw" "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python /source/ascend/compare_resume.py \
      --continuous /runs/vad_continuous/iter_30.pth \
      --resumed /runs/vad_controlled_resumed/iter_30.pth \
      --continuous-run /runs/vad_continuous --partial-run /runs/vad_continuous \
      --resumed-run /runs/vad_controlled_resumed --output /checks/comparison.json
  '
```

This compares model/optimizer tensors, loss, tokens, LR and final rank RNG. A
nonzero exit preserves a failed numerical gate. The measured VADv2 LoRA case
passed; the measured base and HydraMDP cases did not.

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

One 64 GiB logical NPU runs the measured single-rank tasks. One server with eight suitable logical NPUs runs global batch 16; eight NPUs do not mean eight servers. The exercised mapping used one chip from each of eight physical cards. Other placements and all sixteen chips are untested.

Complete available NAVTEST caches (3.640 GB serialized, 19470 annotation frames and 12146 PDM tokens) used **4.467 GB / 4.16 GiB peak RSS** in a single CPU process. This does not measure NAVTRAIN: its two serialized training PKLs total 39.86 GB per rank, approximately 318.9 GB across eight ranks before object expansion and runtime overhead. Begin full-data RAM sizing on a sufficiently large host (for example an available 1 TiB budget) and measure actual RSS; no full-data capacity guarantee is made.

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

The packet's `score_from_cache.py` and `README.md` provide a CPU-only container
command that uses a freshly extracted cache and exported `predictions.pkl`,
without a map or NPU mount. Four variants passed relocation with all official
mean metrics identical. The helper regenerates metadata paths and calls the
unchanged official scorer; use a fresh cache extraction for each evaluation.
