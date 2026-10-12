# AlgEngine on Ascend 910C / A3

This branch adapts **AlgEngine** training, reward/LoRA fine-tuning, policy-gradient fine-tuning and planning inference. **SimEngine** remains a GPU-side subsystem. Its rendering, simulation and closed-loop services have not been deployed or validated by this branch.

Upstream is pinned to `3eded7e8ac64e5bed817860765089b5914ab698f`. NAVSIM **v1.1**, required by upstream, is pinned as a submodule to `3e8291bfa89ff247231e0227778840cd0a036896`. Keep the submodule pin; NAVSIM main is a different evaluation version.

## What was actually exercised

All full-model tests retain 200x200 BEV, 900 queries, four temporal frames, eight cameras, the full 8192x40x3 vocabulary, FP32 and the original configuration's freeze policy. The dataset is a functional subset of three official NAVTEST logs: 140 annotation frames and 64 anchors. It has **zero held-out samples**. No trajectory or PDM label was invented.

| Path | Functional evidence | Boundary |
|---|---|---|
| VADv2 imitation | One NPU, batch 1, 10 AdamW updates; raw losses/gradients/updates/checkpoint checked | Original backbone freeze; planning imitation loss, not all UniAD perception tasks |
| VADv2 reward/LoRA | One NPU, batch 2, 5 updates; one genuine official failure token plus one normal anchor | Original `rl_finetuning=False`: no policy-gradient loss |
| HydraMDP policy gradient | One NPU, batch 2, 5 updates; nonzero actual `loss.RL.PG`, real PDM labels and LoRA updates | Initialized from the functional VADv2 checkpoint, not the released HydraMDP epoch-8 baseline |
| Planning inference | Original `scripts/test.py`, 64 real anchors, finite trajectories and cached PDM CSV | Reuses functional anchors; not held-out accuracy or official independent NAVSIM rescoring |
| HCCL | Four ranks x 10 updates previously exercised with the same operator/device patch | Watched parameter synchronization, not every tensor or multi-node scaling |
| Original global batch | Eight NPUs x batch 2 x 5 updates, HCCL, full-size VADv2 model, finite gradients and synchronized watched parameters | Global batch 16 matches the original eight-GPU recipe; mini data and short lr schedule remain functional validation |

The tested single-NPU imitation run peaked at **12.96 GiB allocated / 18.45 GiB reserved**. VADv2 and HydraMDP fine-tuning each used approximately **6.8 GiB allocated / 12.34 GiB reserved**. These are PyTorch allocator measurements, not total board HBM or full-dataset CPU RAM requirements.

At batch 2, the eight-rank imitation run used a maximum of **27.04 GiB allocated / 38.34 GiB reserved across ranks**, on one logical NPU from each of eight physical cards. One server completed this run. Planning inference used **3.47 GiB allocated / 8.12 GiB reserved** and processed 64 anchors in **184.51 seconds**, including dataset/model setup and metrics. This is an offline FP32 functional path, not an optimized online latency or GPU comparison.

**Hardware support boundary:** physical execution was on an Atlas 800I A3 inference-series system with Ascend910_9362 devices. The published ModelZoo training table distinguishes A3 training-series support from inference-series support. Successful bounded backpropagation does not extend the vendor's supported hardware list. Use a supported training-series system or obtain support confirmation for the specific hardware when planning supported training delivery.

References: [ModelZoo hardware support](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md), [DrivingSDK hardware guidance](https://github.com/Ascend/DrivingSDK/blob/master/docs/zh/installation/installation.md).

## Runtime and source

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git submodule status
```

`runtime.lock.json` records the tested image ID and versions: CANN 9.1.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, DrivingSDK 26.1.0.post2, MMCV 1.7.2, MMDet 2.24.0, MMDet3D 1.0.0rc4 and NumPy 1.23.5. Do not install the upstream CUDA wheels into this image. The `+cpu` PyTorch version string is expected in this Ascend stack; actual tensors and kernels execute on NPU.

The tested image is local, not a public registry image. Transfer it with `docker save` / `docker load` when preparing another host, and verify:

```bash
docker image inspect --format '{{.Id}}' \
  local/migration-worldengine:26.1.0-a3-20261011
# sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3
```

Alternatively, `docker/Dockerfile` describes the tested dependency layers over the pinned official DrivingSDK image:

```bash
docker build -t worldengine-ascend:local ascend/docker
```

This combined cold-build recipe has not been re-executed as a new image acceptance test. A rebuilt image has its own ID: explicitly select `--image` and `--expected-image-id`, then repeat operator/model validation. The lock for the already tested image remains the reproducible runtime baseline.

The launcher requires Linux, Docker, the Ascend driver/firmware directories and selected `/dev/davinci*` nodes. Run `npu-smi info` and select idle devices. It creates a new output directory, leaves exited containers and logs for review, and rejects an existing output directory. It does not install or restart host drivers/services.

## Functional runs with already prepared mini assets

The prepared directory must contain:

```text
prepared/
  worldengine_navtest_smoke.pkl
  worldengine_navtest_smoke_pdm.pkl
  worldengine_navtest_smoke.yaml
  manifest.json
  cameras/                       # actual camera files or valid asset-parent symlinks
```

Its manifest describes the official subset, unchanged trajectory labels and `heldout_samples=0`. These data/checkpoint files are not included in Git. Camera symlinks must resolve inside the mounted asset parent. Set the following paths to existing assets:

```bash
MINI=/datasets/worldengine/prepared
PERCEPTION=/datasets/worldengine/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth
VOCAB=/datasets/worldengine/test_8192_kmeans.npy
RUNS=/experiments/worldengine

# --devices contains HOST LOGICAL NPU IDs, not physical-card or server counts.
python3 ascend/run_container.py --devices 0 --profile imitation \
  --mini-prepared "$MINI" --checkpoint "$PERCEPTION" --vocab "$VOCAB" \
  --output "$RUNS/imitation_1p" --steps 10 --batch-size 1 --inference-after --detach

# Run these after the imitation checkpoint exists and the selected NPU is idle.
python3 ascend/run_container.py --devices 0 --profile vadv2_ft \
  --mini-prepared "$MINI" --checkpoint "$RUNS/imitation_1p/iter_10.pth" --vocab "$VOCAB" \
  --output "$RUNS/vadv2_ft_1p" --steps 5 --batch-size 2 --detach

python3 ascend/run_container.py --devices 0 --profile hydramdp_rl \
  --mini-prepared "$MINI" --checkpoint "$RUNS/imitation_1p/iter_10.pth" --vocab "$VOCAB" \
  --output "$RUNS/hydramdp_rl_1p" --steps 5 --batch-size 2 --detach
```

The real mini fine-tuning selection intersects the upstream `navtest_failures_filtered.yaml` with the supplied subset; `normal_ratio=1`, `normal_only=False`. Its two anchors are repeated to exercise the graph. This is not a meaningful rare-case training sample size.

Independent planning inference from an existing dumped configuration:

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --config "$RUNS/imitation_1p/imitation_rank0.py" \
  --mini-prepared "$MINI" --checkpoint "$RUNS/imitation_1p/iter_10.pth" --vocab "$VOCAB" \
  --output "$RUNS/inference_1p" --detach
```

Both one-rank training and inference use `torchrun --launcher pytorch` internally. Upstream `test.py` raises `NotImplementedError` with the non-distributed launcher; `nproc_per_node=1` is the supported entry even on a single NPU.

Check the result, not just Docker launch success:

```bash
docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' worldengine-npu-imitation_1p
cat "$RUNS/imitation_1p/train_exit_code"
cat "$RUNS/imitation_1p/exit_code"
cat "$RUNS/imitation_1p/ascend_validation_rank0.json"
cat "$RUNS/imitation_1p/inference/inference_validation.json"
```

An initial inference attempt in the tested environment computed all 64 predictions but failed during CPU-side export because NAVSIM was absent. The pinned submodule/PYTHONPATH and early NAVSIM import fix that dependency. `test.py` now saves `--out` and honors an explicit `jsonfile_prefix`, allowing checkpoints to be mounted read-only.

## Full assets and equipment planning

HF revision is locked in `assets.lock.json`; download is never automatic. The complete HF publication is approximately 4.864 TB; its AlgEngine section is approximately **54.2 GB**, including all published checkpoints. The huge SimEngine assets are unnecessary for offline AlgEngine log training. Full raw OpenScene images, maps and official NAVSIM metric cache are separate inputs; 54.2 GB is not the complete training data budget.

```text
DATA/
  raw/openscene-v1.1/sensor_blobs/{trainval,test}/
  alg_engine/merged_infos_navformer/nuplan_openscene_{navtrain,navtest}.pkl
  alg_engine/pdms_cache/pdm_8192_gt_cache_{navtrain,navtest}.pkl
  alg_engine/test_8192_kmeans.npy
  alg_engine/ckpts/...
```

The published navtrain metadata and PDM PKLs alone total approximately **39.86 GB on disk**. Each training rank loads these through Python; deserialized objects, validation data and dataloader workers add memory. Eight ranks therefore already correspond to about **318.9 GB of serialized training inputs**, before in-memory expansion. Full-dataset CPU RSS has not been measured. Size RAM from actual deserialization; the mini run's 64 GB limit cannot establish full training sufficiency. Full launches require an explicit `--memory`.

After full assets have been provisioned, this is the full-epoch entry. It is a **procedure to validate**, not a claim that the full 8 epochs have been completed:

```bash
DATA=/datasets/worldengine
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 --profile imitation \
  --data-root "$DATA" \
  --checkpoint "$DATA/alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth" \
  --vocab "$DATA/alg_engine/test_8192_kmeans.npy" \
  --output /experiments/worldengine/full_imitation \
  --memory 1t --cpus 32 --shm-size 16g --workers 0 --detach
```

Here `1t` is a RAM sizing starting point for a host that actually has more than 1 TiB available; full RSS and dataloader tuning remain unverified. Without `--steps`, the wrapper retains the upstream epoch schedule, learning rate and default per-rank batch. For VADv2 imitation that is **8 epochs, batch 2 per rank, lr=2e-4**. The bounded tests deliberately use lr=5e-5 and a short iter schedule. Changing the rank count changes the global batch; convergence equivalence requires a separate schedule/accuracy comparison.

For formal VADv2 or HydraMDP fine-tuning, choose `--profile vadv2_ft` or `--profile hydramdp_rl` and supply the corresponding **released base planner checkpoint**, not the 10-step functional checkpoint. The wrapper preserves the selected profile's loss switches. Complete the full epoch, independent NAVTEST scoring and rare/normal regression comparison before accepting accuracy.

One 64 GiB logical NPU is sufficient for the measured single-rank functionality. More NPUs are throughput capacity, not a prerequisite imposed by a 5 TB dataset. Four-rank and eight-rank HCCL are exercised. Multi-node scaling, all sixteen logical devices, BF16/whole-model FP16, UniAD all-task training, DiffusionDrive, resume-training state and GPU-versus-NPU accuracy/throughput comparisons remain outside the demonstrated acceptance.

## Changes and operating limits

The adaptation imports torch_npu/transfer_to_npu, selects HCCL and MMCV NPU parallel wrappers, and explicitly dispatches four deformable-attention paths to `mx_driving.multi_scale_deformable_attn`. A Matplotlib style compatibility fallback keeps legacy dataset imports usable. No SimEngine source file is changed.

The validation hook checks raw head outputs and losses **before upstream `nan_to_num`**, actual gradients and optimizer updates, selected parameter synchronization and saved optimizer steps. Synchronization is sampled; the hook does not compare every model tensor. Full-epoch runs use the upstream optimizer hook rather than the bounded diagnostic hook.

The portable launcher here is single-host. Multi-node training needs an explicit node/rank rendezvous launcher and HCCL network validation. The FP32 inference smoke is an offline test, not an optimized real-time driving service. No faster-than-GPU claim or official WorldEngine NPU support claim is made by these measurements.
