# SimEngine assessment for Ascend 910C

[中文](simengine_ascend_assessment_zh.md)

Assessed on 2026-10-19 against WorldEngine commit `3eded7e8ac64e5bed817860765089b5914ab698f`, with SimEngine tree `8eedf247a89ce5b022209e4cf0f3988cc2a10a2f`. This branch currently contains a source and dependency assessment. It does not provide an implemented SimEngine NPU runtime.

**Unmodified SimEngine cannot run its complete workflow on an NPU-only accelerator setup. CANN provides Gaussian rendering operators for Atlas A2/A3, giving a concrete migration candidate, but compatibility with this project's gsplat 1.4.0 interface and rendering semantics remains unverified.** No SimEngine NPU rendering, closed-loop or paired performance experiment was executed for this assessment. Minimum device counts, memory requirements and speed differences are unmeasured.

## Scope and evidence

The review covers scenario loading, vehicle/traffic behavior, MTGS rendering, the AlgEngine closed-loop client, NR/R testing, reward-producing rollouts, Ray scheduling and synthetic-data export. It distinguishes pinned WorldEngine code, capabilities stated or implemented by candidate maintainers, and work that still requires experiments.

MTGS is the default renderer. The source does not import CARLA or launch an Unreal service; CARLA references are legacy comments. CARLA hardware requirements therefore do not determine this project's migration feasibility. Its Gaussian rasterization path depends on CUDA extensions. See the [renderer configuration](../projects/SimEngine/worldengine/configs/common/renderer/mtgs.yaml#L1) and [import](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L14).

Existing AlgEngine NPU evidence does not establish SimEngine compatibility. The original simulation scripts launch a separate entry, `projects/AlgEngine/closed_loop/sim_test.py`, which still calls `model.cuda()` and uses `MMDataParallel`. That client needs its own adaptation and acceptance. [Model construction](../projects/AlgEngine/closed_loop/sim_test.py#L274).

| Function | Pinned implementation | Migration work and acceptance |
| --- | --- | --- |
| Scenarios, dynamics, replay, IDM and maps | Python, NumPy, SciPy and nuPlan; [runner](../projects/SimEngine/worldengine/runner/run_simulation.py#L16), [environment](../projects/SimEngine/worldengine/envs/base_env.py#L327) | Retain CPU execution; validate ARM64 dependencies, termination, NR/R behavior and determinism |
| Static background, sky and dynamic objects | Vanilla, Rigid and MirroredRigid Gaussian models; [mapping](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L31) | Tensor transforms are candidates for porting; replace CUDA SH/rasterization with verified operators |
| Spherical harmonics | Direct `gsplat.cuda._wrapper.spherical_harmonics` calls; [base model](../projects/SimEngine/worldengine/render/mtgs/gaussian_model/vanilla_gaussian_splatting.py#L123) | CANN has SH operators; verify shapes, degree, directions, clipping and each camera |
| Multi-camera RGB and optional depth | [Actual MTGS call](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L300) | Implement an adapter preserving antialiasing, alpha, depth semantics and resolution |
| Geometry and poses | Several device-side FP64 paths; [calibration](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L128), [poses](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L395) | Probe dtype/operator support. Keeping CPU FP64 geometry with FP32 NPU render inputs is an initial option; silent precision reduction does not establish equivalence |
| Closed-loop planner | Separate processes exchange frame PKLs and trajectory NPYs; [client](../projects/SimEngine/worldengine/components/agents/client/navformer_client.py#L81) | Adapt model entry, precision/MSDA, atomic exchange and timeouts; offline inference is a separate workflow |
| Dense rewards | 8192-candidate vocabulary, CUDA TorchSimulator or CPU PDMSimulator; [branch](../projects/SimEngine/worldengine/manager/dense_reward_manager.py#L275) | Existing CPU path is an initial option. NPU LQR/FP64 acceleration needs separate validation; preserve rewards and candidate order |
| Ray parallelism | CUDA discovery, `num_gpus`, `ray.get_gpu_ids()`; [discovery](../projects/SimEngine/worldengine/utils/multithreading/worker_ray.py#L49), [partitioning](../projects/SimEngine/worldengine/runner/builders/env_builder.py#L23) | Use NPU resources, visibility and consistent worker IDs. Ray 2.48 already has an NPU manager; GPU resource requests do not become NPU requests automatically |
| Testing, rollout and export | [Testing](../projects/SimEngine/scripts/run_ray_distributed_testing.sh#L1), [rollout](../projects/SimEngine/scripts/run_ray_distributed_rollout.sh#L1), [export](../projects/SimEngine/scripts/export_simulation_data.py#L1) | Replace fixed eight-way CUDA launch/cleanup with controlled NPU processes and artifact checks. File existence alone does not validate images, metadata or rewards |

Disabling rendering does not provide the complete visual workflow: the environment requires RenderManager when DataManager is enabled. CPU-only vehicle state tests cannot replace rendered observations. [Constraint](../projects/SimEngine/worldengine/envs/base_env.py#L336).

## Candidate backends and version conditions

The preferred candidate is CANN [gauss-splat](https://gitcode.com/cann/gauss-splat). Its README declares Atlas A2/A3 support and SH, projection, filtering, sorting and forward/backward rendering operators, with CANN 8.5+ and matching PyTorch 2.7.1+. The README's `torch_npu v7.3.0` identifies a plugin release line; it cannot be numerically compared with a wheel version such as `2.7.1.post8`. **The appropriate tag, build ABI and runtime compatibility with CANN 9.1 and the installed Torch-NPU wheel have not been established.**

An earlier official [3DGS recipe](https://gitcode.com/cann/cann-recipes-spatial-intelligence/blob/master/algorithms/gaussian_splatting/README.md) uses `meta_gauss_render`, documenting A2, CANN 8.2.RC1, PyTorch 2.1.0 and Torch-NPU 2.1.0.post12. [SLARM's authors](https://github.com/kevinchiu19/SLARM#installation) additionally describe PyTorch 2.1/910C use. These are feasibility evidence, not WorldEngine compatibility results.

[Pinned Ray 2.48 source](https://github.com/ray-project/ray/blob/ray-2.48.0/python/ray/_private/accelerators/npu.py) already defines `NPU` resources and `ASCEND_RT_VISIBLE_DEVICES`. Basic NPU scheduling does not require upgrading to a newer Ray plugin, but WorldEngine's discovery, split counts and worker paths still need adaptation.

Original WorldEngine documents Python 3.9, PyTorch 2.0.1/CUDA 11.8 and gsplat 1.4.0; reconstruction training lives in the external MTGS project. Use a separate NPU environment/image and preserve the GPU baseline. Do not replace packages inside an existing AlgEngine environment. [Original installation](installation.md#environment-1-simengine-simengine). Newer PyTorch also requires checking scene checkpoint serialization and changed `torch.load` defaults before attributing failures to rendering.

The following canonical CANN `master` text files were sampled on 2026-10-19. SHA256 values identify the fetched bytes, not a Git commit or a promise that `master` remains unchanged. Select and record a compatible tag/commit before building.

| Source | Sampled SHA256 |
| --- | --- |
| [README.md](https://raw.gitcode.com/cann/gauss-splat/raw/master/README.md) | `8c640465d0c978b8d22b5b736cff4944226b037074c28e6d7778cef2a5f17483` |
| [ops/rendering.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/rendering.py) | `bb4b4b2ea3115c28ce68fce8327dede71effad7c94aaae33b5fdd331bfd59294` |
| [ops/projection_three_dims_gaussian_fused.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/projection_three_dims_gaussian_fused.py) | `a8020eb1bd1c7eebc68d6802f20b45e514ee71af7c0cdfa0806426434a84a281` |
| [ops/calc_render.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/calc_render.py) | `55538ed671be7cefcee77ba3089956cb7a0a217ffa43e53cd3e12c5a55a431cc` |
| [build.sh](https://raw.gitcode.com/cann/gauss-splat/raw/master/build.sh) | `126616ed2406a29a8df77b3c9dc71a713a3554b9dec92e9c7d68b91f40d6c700` |

## Differences from the gsplat call

This table compares the original [MTGS invocation](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L338) with the sampled CANN implementation. Lower-level operators may support an adapter; its public Rasterizer is not a drop-in `gsplat.rendering.rasterization` replacement.

| Contract | Original WorldEngine | Sampled CANN implementation and required work |
| --- | --- | --- |
| Gaussian parameters | World-space means, normalized quaternions, activated `exp(scales)` and `sigmoid(opacities)` | Public entry takes log-scales/logit-opacities and activates them. Passing original values would activate twice; convert explicitly or use the lower projection API |
| Colors | Precomputed per-camera `C×N×3` RGB, including object/Fourier/mirror processing | Public entry requires `sh0/shN` coefficients and forces SH evaluation. RGB cannot be relabeled as SH; the validation branch accepting `sh_degree=None` is not an implemented RGB execution path |
| SH conventions | Per object/camera; degree zero uses sigmoid, higher degrees use SH plus 0.5 clipped to `[0,1]` | Candidate provides camera-dependent SH and degrees 0–4. Its public degree-zero path still uses SH and clips only the lower bound; preserve original branches |
| Camera matrices | World-to-camera matrices and `C×3×3` intrinsics | Public entry expects camera-to-world then inverts it; preserve direction and avoid an unintended second inverse |
| Clipping | near 0.01, far `1e10`, radius clip 0 | Public implementation fixes near at 0.2; lower projection exposes near/far and must receive the original values |
| Antialiasing | Default antialiased covariance/opacity compensation | Public entry lacks this option and leaves projection compensation disabled. Lower-level compensation exists but needs output comparisons |
| Tiles / packing | Tile 16, `packed=False` | Candidate defaults to tile 32 and compacted projections. Tile 16, sorting and Gaussian index mappings require validation |
| RGB, alpha and background | RGB/optional depth channels plus separate alpha used for background composition | Candidate returns RGB and depth, without alpha. Extra low-level cumulative state may permit alpha export, subject to checking its definition, clipping and termination |
| `RGB+ED` | Normalized expected depth, with original empty-pixel handling | Candidate low-level documentation defines weighted depth sums. Public mode strings are validated without a corresponding output-mode branch. D cannot be substituted for ED; normalization and empty pixels need testing |
| Dynamic scenes | Background, sky and multiple moving actors, including mirrored/time-varying SH | Preserve MTGS transforms and submit the combined Gaussians/per-camera colors. A static-cloud test is insufficient |
| Gradients | SimEngine `render()` uses `@torch.no_grad()` while passing `absgrad=True` | Begin with forward rendering. It does not validate external MTGS training, backward or reconstruction convergence |
| Image export | FP32 to uint8, RGB-to-BGR and OpenCV distortion remapping | Retain operation order, metadata, resolution and camera count when comparing quality/performance |

The pinned upstream [v1.4.0 CUDA backend](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/cuda/_backend.py) and [rendering.py](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/rendering.py) document the original operator/build path.

## Deployment candidates

**NPU compute without NVIDIA GPUs:** keep scenario/map/file operations and any required FP64 geometry on CPU; run Gaussian rendering and the planner on NPU. This requires the rendering adapter, closed-loop model entry and resource scheduling changes. A CPU host remains part of the system. This arrangement is untested.

**GPU rendering with an NPU planner:** preserve original MTGS/gsplat CUDA rendering and run the planner in a separate NPU process over the existing frame/trajectory file protocol. The architecture permits investigating this split, but client execution, shared paths, synchronization, termination and performance remain untested. It still requires a GPU. Cross-host deployment additionally requires shared-storage and transfer measurements.

## Minimum real-scene inputs

Select one original published scene and retain its identifiers, calibration, timestamps and required actors; the complete WorldEngine corpus is unnecessary for a pilot.

| Input | Pilot scope |
| --- | --- |
| Scenario description | One original entry from `all_scenarios.pkl`, retaining history/future, calibration, map and trajectory metadata; record source SHA and selected ID |
| MTGS asset | Matching `<asset-root>/<id>/background/<id>.ckpt` for `digitaltwin_asset_id`, containing all background/sky/rigid/mirrored models used by the scene; [loader](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L471) |
| Maps | Actual regional nuPlan maps and version JSON for closed-loop metrics/rewards |
| Planner | Matching official VADv2 or HydraMDP checkpoint/config; not needed for a rendering-only operator probe |
| Vocabulary | Original `test_8192_kmeans.npy`, preserving 8192 candidates and order |
| Output capacity | Original-resolution cameras, per-frame metadata, plans, rewards and reports; measure selected-scene disk use |

A metadata query on 2026-10-19 at WorldEngine dataset revision `d0e88d858533ca407a3387a26577c083aa7b3617` found `navtest_failures/all_scenarios.pkl` at **911,923,264 B**, and three published asset archives at **30,904,422,680 B, 29,680,462,746 B and 27,886,330,535 B**. No payloads or archive indexes were downloaded. Which archive contains a selected scene is unknown; none is an accepted small benchmark package. [Scenario listing](https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/d0e88d858533ca407a3387a26577c083aa7b3617/data/sim_engine/scenarios/original/navtest_failures?recursive=false&limit=1000), [asset listing](https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/d0e88d858533ca407a3387a26577c083aa7b3617/data/sim_engine/assets/navtest_failures/assets?recursive=false&limit=1000).

AlgEngine NAVTRAIN camera archives do not replace Gaussian scene checkpoints. Existing prebuilt scenarios/assets avoid repeating reconstruction for a rendering pilot. Creating new assets requires the external [MTGS](https://github.com/OpenDriveLab/MTGS) dataset, reconstruction training and export, with separate acceptance.

## Executable research preflight

These are proposed research steps, not an existing NPU SimEngine launcher. Run device operations in an isolated selected environment and retain command exits/version outputs.

Record the original source and select the candidate. `CANN_GAUSS_COMMIT` must be an actual commit whose CANN/plugin pairing has been checked; do not use `master`:

```bash
git rev-parse HEAD
git status --short
npu-smi info
python -m pip freeze
python -m pip check

WORKDIR=/path/to/isolated-simengine-research
: "${CANN_GAUSS_COMMIT:?Select a compatible fixed gauss-splat commit}"
git clone https://gitcode.com/cann/gauss-splat.git "$WORKDIR/gauss-splat"
git -C "$WORKDIR/gauss-splat" checkout --detach "$CANN_GAUSS_COMMIT"
git -C "$WORKDIR/gauss-splat" rev-parse HEAD
```

In an isolated environment with matching PyTorch/Torch-NPU, source CANN and prepare the maintainer-listed build dependencies. README offers `build.sh --python=<version>` or `pip install . --no-build-isolation`. A compiler, CMake, ACL/CANN headers and operator development environment are required; an upstream CUDA gsplat wheel is insufficient.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$WORKDIR/gauss-splat"
python -m pip install numpy==1.23 decorator sympy scipy attrs cloudpickle \
  psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf
python -m pip install . --no-build-isolation
python -m pip check
cd "$WORKDIR"
python -c 'import acl, torch, torch_npu, gauss_splat.gsplat_c; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())'
```

After a successful build/import, use one idle logical NPU for the following proposed forward probe: 256 synthetic Gaussians, one camera and a 64×64 image, without model or scene downloads. **This probe has not been executed; passing it would validate only this candidate call, not MTGS compatibility.**

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import torch_npu
import gauss_splat

torch.npu.set_device(0)
device = 'npu:0'
n = 256
means = torch.zeros(n, 3, device=device)
means[:, 2] = 2.0
quats = torch.zeros(n, 4, device=device)
quats[:, 0] = 1.0
splats = dict(means=means, quats=quats,
              scales=torch.full((n, 3), -3.0, device=device),
              opacities=torch.zeros(n, device=device),
              sh0=torch.zeros(n, 1, 3, device=device),
              shN=torch.zeros(n, 15, 3, device=device))
pose = torch.eye(4, device=device)[None]
intrinsics = torch.tensor([[64., 0., 32.], [0., 64., 32.],
                           [0., 0., 1.]], device=device)[None]
with torch.no_grad():
    rgb, depth, info = gauss_splat.Rasterizer().rasterization(
        cam=(pose, intrinsics, 'RGB'), size=(64, 64), tile_size=32,
        active_sh_degree=0, splats=splats, camera_model='pinhole')
torch.npu.synchronize()
assert rgb.shape == (1, 64, 64, 3)
assert depth.shape == (1, 64, 64, 1)
assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
print({'rgb': list(rgb.shape), 'depth': list(depth.shape),
       'rgb_max': float(rgb.max()), 'meta_keys': sorted(info)})
PY
```

Only then implement a separate rendering adapter and compare original gsplat 1.4.0 inputs/full outputs for SH, projection, sorting, RGB, alpha and ED. Cover one/multiple cameras, clipping boundaries, occlusion, empty pixels, antialiasing, mirrors and moving actors. After a real single scene passes, connect both planners and verify NR/R, reward rollouts, recovery and exported data. A completion flag or process exit alone is insufficient: inspect every `RunnerReport.succeeded`, actual frames, finite outputs, trajectories and reward coverage.

## Devices and performance acceptance

Begin operator probes with **one logical NPU**. For the first closed loop, separate rendering and planning onto one logical chip each to isolate memory/execution issues. Sharing a chip, minimum HBM/host RAM and sixteen-chip scaling all need measurements. The original GPU guide's 8 GB minimum/24 GB recommended VRAM and 32/64 GB host RAM are GPU guidance, not established NPU requirements.

Pass output-quality gates before measuring three levels under matched conditions:

| Level | Inputs/configuration to fix | Measurements |
| --- | --- | --- |
| Operators | Same Gaussians, cameras, dtype, clipping, sorting and antialiasing | Synchronized warm latency, peak HBM and full-output error; separate compilation/cold start |
| Eight-camera rendering | Same real asset, actors, timestamps, original resolution and postprocessing | p50/p95 per complete eight-camera frame, frames/s, camera images/s, memory and image quality |
| Full closed loop | Same planner weights, scenes, NR/R, rewards and export scope | Total wall time, completed/errored scenes, plans, closed-loop metrics, reward completeness and IO/communication time |

Retain raw logs and source/input/version hashes. Distinguish one GPU, one NPU chip, a dual-chip physical NPU card and multi-process runs. Sixteen chips versus eight GPUs does not imply a speed ratio. No paired WorldEngine measurements currently support a speed, cost or energy advantage figure.

## Existing upstream boundaries

- `DeformableSubModel` maps to `None` in the original [type table](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L36); arbitrary external MTGS models are not automatically supported assets.
- The [render result](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L380) contains an empty LiDAR mapping with a TODO. Published LiDAR inputs do not establish runtime LiDAR rendering.
- BWM integration remains incomplete in the original [roadmap](../README.md#roadmap). Unreleased BWM generation and external MTGS reconstruction training must not be counted as already implemented SimEngine migration scope.

Advance from real single-scene rendering and closed-loop evidence. Keep each unvalidated function explicitly marked as research until its acceptance gates are complete.
