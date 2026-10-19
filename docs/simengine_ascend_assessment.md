# SimEngine on Ascend 910C

[中文说明](simengine_ascend_assessment_zh.md)

Updated: October 20, 2026.

This branch implements an NPU forward-rendering adapter for MTGS. Twenty small synthetic numerical cases passed on one 910C chip. The repository provides backend building, official scene preparation, full-input capture and validation, replay supervision, and planner integration. **The measured result covers small FP32 rendering inputs. Original-resolution official scenes, complete planner feedback, and GPU comparisons remain unvalidated.**

The upstream WorldEngine baseline is `3eded7e8ac64e5bed817860765089b5914ab698f`; the original rendering contract comes from `gsplat==1.4.0`. The adapter uses CANN gauss-splat commit `8c0445ebcca91514fce3a7050241f41ed132c340` with an explicit inference-only compatibility patch. Its public high-level Rasterizer is not used as a drop-in replacement.

## Measured scope

The server has **8 physical 910C cards, two chips per card, for 16 visible chips**. This probe used one chip, reported as `Ascend910_9362` and mapped to `npu:0` inside the container. Upstream eight-way launch scripts target 8 GPUs. Sixteen-chip SimEngine scaling has not been measured, and those device counts do not imply a throughput ratio.

| Item | Recorded evidence |
| --- | --- |
| Environment | Python 3.10, PyTorch `2.7.1+cpu`, Torch-NPU `2.7.1.post8`, CANN Toolkit 9.1.0; each build/probe records its own runtime fingerprints |
| Public backend build | `build_backend.py` built a wheel in a fresh output and isolated venv, installed it and checked actual native/pyACL imports; this CPU stage passed |
| Numerical cases | 20 passed: projection, covariance tails, pixel centers, clipping, occlusion, alpha, expected depth, empty images, camera-dependent colors, antialiasing, rotation and SH degrees 0–4 |
| Input envelope | Up to 256 Gaussians, 8 cameras, 4096 pixels per camera; mainly 64×64, plus a 71×37 non-square case; tiles 16/32 |
| Maximum RGB absolute difference | `1.2237711302e-6` |
| Maximum alpha absolute difference | `1.5866665948e-6` |
| Maximum expected-depth absolute difference | `2.5749206546e-5` |
| CPU checks | 136 Linux CPU tests passed with no skips: 79 renderer/gating, 21 entry/process, 26 build/preparation, 10 scene capture; synthetic fixtures |

The independent reference is NumPy. RGB/alpha tolerances are `atol=3e-4, rtol=1e-3`; expected depth uses `atol=2e-3, rtol=1e-3`; SH uses `atol=3e-5, rtol=3e-5`. No GPU was available for this run. There is no CUDA gsplat parity result or measured speed, cost, or energy advantage.

The [numerical result and tested source fingerprints](validation/simengine_npu_forward_20261020.json) identify the current adapter, the publicly built wheel, the actual loaded native/kernel files, 20 passed cases and 18 image-error records. The [CPU check record](validation/simengine_cpu_checks_20261020.json) contains the four test commands and tested source inventory. A newly built wheel or changed adapter must rerun the probe and generate its own manifest. The public summary does not replace a locally generated runtime manifest or authorize larger inputs.

| Component | Implementation provided | Acceptance boundary |
| --- | --- | --- |
| NPU renderer | CANN projection/forward kernel, Torch-NPU stable sorting, RGB/alpha/expected depth, SH, MTGS integration | Small FP32 forward passed; complete real-asset inputs remain pending |
| Geometry and actors | Background, sky, rigid, mirrored and temporal transforms; required FP64 poses remain on CPU | Integrated code paths; real dynamic scenes remain pending |
| Official input preparation | Whole-file hashes, exact scene-to-asset mapping, safe extraction, unchanged one-item derivation | Tool fixtures passed; real assets still require complete download/hash/preparation |
| Single-node replay | Original Hydra runner, BaseEnv, DataManager, report checks and deadlines | CPU command/process tests provided; official-scene NPU replay remains pending |
| VADv2/HydraMDP planner feedback | Separate AlgEngine process, frame/trajectory protocol, explicit FP32/BF16 planner entry | Protocol tests provided; no real closed-loop result in this run; rendering remains FP32 |
| R/NR, rewards and export | Original policies, CPU PDM/dense scoring branches and output checks | Require real scenes/maps/planners and the corresponding output acceptance |

## Environment and backend build

Use a separate Linux/ARM64 environment with compatible CANN, PyTorch, Torch-NPU, native build tools and original SimEngine dependencies. This run created isolated venvs from an existing image; it does not establish deployment in an arbitrary clean container. The upstream pyproject still declares CUDA/PyTorch 2.0.1 and must not overwrite the NPU stack.

The original environment also imports nuPlan, Hydra, OpenCV, SciPy, Pillow and Ray 2.48.0. `SINGLE_NODE` creates no Ray worker, but the original code still imports Ray. Dependency imports and device kernels are separate checks.

The [tool guide](../tools/simengine_ascend/README.md#build-the-pinned-cann-backend) documents all prerequisites and artifacts. Variables below refer to actual deployment values; `DEADLINE_UTC` is an explicit UTC job cutoff.

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.1.0
source "$CANN_ROOT/set_env.sh"
cd "$SIM_ROOT"
"$CONTROL_PYTHON" tools/simengine_ascend/build_backend.py \
  --source "$CANN_SOURCE" --output "$NEW_BUILD" \
  --python "$SIM_PYTHON" --jobs 8 --install \
  --cann-root "$CANN_ROOT" \
  --cann-python-path "$CANN_ROOT/python/site-packages" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

`CANN_SOURCE` must be a clean Git checkout of the pinned commit. `SIM_PYTHON` must belong to an existing isolated venv, and the output must be new. The builder copies only tracked files and applies the compatibility patch to the copy. Operational changes bound parallelism to 1–16, target only `ascend910_93`, disable upstream test executables and resolve the actual Torch-NPU header path in a venv.

The example uses this run's installed SDK path. On another server, replace it with the actual installation and keep `set_env.sh`, `--cann-root` and the pyACL directory under the same SDK. A stale `ascend-toolkit` symlink pointing to a different version can cause the installation preflight to reject the pyACL path.

The build uses `pip wheel --no-build-isolation --no-deps`. Installation is opt-in: `--install` force-installs the exact wheel from outside the source directory and checks the imported module locations. Inherited global pip destinations are removed; only verified SDK-scoped CANN Python directories are retained. `build_receipt.json` records source, patch, wheel and installed module hashes, commands and versions. A successful build/import accepts only that stage.

The [public build record](validation/simengine_public_backend_build_20261020.json) identifies this run's successful fresh build and its wheel/extension SHA256 values. The subsequent device probe loaded the same extension SHA256 and passed the small-input checks with the current adapter. Build and numerical results are recorded separately so their binary identity and stage-specific outcomes can be checked.

## Repeat the small NPU probe

Expose one authorized chip and record its host/process mapping. `npu:0` means visible device zero, not necessarily host chip zero. Use an external container/job cutoff for a device or runtime hang.

```bash
cd "$SIM_ROOT"
export PYTHONPATH="$SIM_ROOT/projects/SimEngine:${PYTHONPATH:-}"
"$SIM_PYTHON" projects/SimEngine/worldengine/render/ascend/tests/probe_cann.py \
  --device npu:0 --output "$NEW_PROBE" \
  --deadline-utc "$DEADLINE_UTC"
```

Check the exit code, `device_validation.json` with `result=pass`, the required cases, and `validated_manifest.json`. The manifest binds loaded runtime files, adapter/oracle sources, compatibility kernel and input limits. Probe timings include validation operations and are not a performance benchmark. Do not edit the manifest to enlarge its Gaussian, camera or pixel limits.

## Prepare an official scene

The minimal candidate requires official `all_scenarios.pkl` and `navtest_failures/assets/part003.tar.gz`: **28,798,253,799 bytes (28.80 GB decimal)**. A one-scene check does not require the full 5 TB corpus. Fixed HF/ModelScope sources, lengths and complete hashes are in the [asset lock](../tools/simengine_ascend/official_scene_assets.lock.json) and [download instructions](../tools/simengine_ascend/README.md#inputs).

The complete scenario-index SHA was checked on the server, and the original pickle contains 288 scenes. One entry maps to the first confirmed asset in part003:

```text
2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4
```

The scene uses the Boston map. A real tar-header probe identified a 336,430,727-byte background checkpoint. Neither a package header nor an index proves complete archive integrity or the checkpoint's internal submodels.

```bash
"$SIM_PYTHON" tools/simengine_ascend/prepare_scene.py \
  --scenario-pkl "$INPUTS/all_scenarios.pkl" \
  --asset-archive "$INPUTS/part003.tar.gz" --output "$PREPARED" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

Both complete original files are verified before pickle loading or extraction. The tool matches an actual asset and extracts its complete directory. `scene.pkl` preserves the original selected value without removing frames, changing cameras or replacing images. `prepare_receipt.json` records original, derived and extracted-file hashes. Optional `--inspect-checkpoint` performs a separate CPU inventory with PyTorch/project dependencies; it does not render.

## Capture and validate complete real inputs

After preparation, use original BaseEnv.reset/MTGS processing to assemble the initial real frame. Capture preserves all Gaussians, cameras, original resolution and RGB+ED settings, then exports before rasterization. A successful capture is not a successful image or simulation.

```bash
PREPARE_RECEIPT_SHA=$("$SIM_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PREPARED/prepare_receipt.json")
SOURCE_COMMIT=$(git -C "$SIM_ROOT" rev-parse HEAD)
"$SIM_PYTHON" tools/simengine_ascend/capture_scene.py \
  --prepared-dir "$PREPARED" --prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --maps-root "$MAPS_ROOT" --small-manifest "$NEW_PROBE/validated_manifest.json" \
  --device npu:0 --source-commit "$SOURCE_COMMIT" --reaction NR \
  --output "$NEW_CAPTURE" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"

"$SIM_PYTHON" projects/SimEngine/worldengine/render/ascend/tests/validate_real_input.py \
  --capture "$NEW_CAPTURE/capture/capture.json" \
  --small-manifest "$NEW_PROBE/validated_manifest.json" \
  --device npu:0 --output "$NEW_REAL_CHECK" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

The commit declares the base revision; capture also hashes the actual runtime source. It rechecks the derived scene and every extracted asset. The original 27.89 GB archive's complete verification remains linked through the preparation receipt rather than being repeated by capture.

The validator executes the complete forward pass, checks every output for finite values, and compares fixed-seed sampled pixels against an independent CPU reference. The default is 128 sampled pixels in total per camera, including corners, center and the remaining random positions. Passing creates `validated_real_manifest.json`. That evidence is tied to its input and render parameters; it does not establish full-pixel parity, an entire scenario, or GPU equivalence. This run has no real-input device result yet.

## Replay and planner feedback

The [English runtime guide](../simengine_ascend/README.md) and [Chinese guide](../simengine_ascend/README_zh.md) provide dependency pins, expected-ID generation and dual-process commands. Read the original dictionary key and derived scene SHA from the preparation receipt and create the exact unique expected-ID list.

```bash
cd "$SIM_ROOT"
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:0 \
  --output "$NEW_REPLAY" --expected-scene-ids "$SCENE_IDS_JSON" \
  --mode replay --reaction NR --deadline-utc "$DEADLINE_UTC"
```

A real scene needs an applicable manifest; the wrapper does not enlarge the validated envelope. `replay` follows the logged ego trajectory. `closed-loop` connects a separate VADv2 or HydraMDP planner. The two processes use their own environments and controlled frame/trajectory directories. The NPU planner reuses a separately pinned AlgEngine adaptation and checks actual source, image identity and complete checkpoint. Planner FP32/BF16 commands are in the runtime guide; the renderer accepts FP32 only.

`R` selects original IDM policies; `NR` replays other actors' logs. PDM metrics and dense rewards are opt-in and need real maps and the original trajectory vocabulary. The wrapper checks original RunnerReports, exact scene IDs, completion flags and actual JPEG decoding. Complete camera/frame coverage, vehicle feedback and applicable scores still require their own acceptance.

## Device planning and remaining acceptance

The single-chip evidence supports proceeding to real inputs. It does not determine minimum HBM, host RAM or device count for full scenes. An initial planner integration can allocate one chip to rendering and one to planning to isolate resources; sharing a chip has not been measured.

Remaining work covers original-size frames, complete R/NR scenes, both planners, applicable metrics/rewards, export coverage and interruption recovery. Multi-chip/Ray execution, sustained runs and performance need separate acceptance. Any comparison with upstream 8 GPU execution must fix assets, cameras, resolution, model, dtype, score scope and output content, while reporting physical cards and chip/process counts separately.

External MTGS reconstruction training, arbitrary deformable models, unfinished BWM generation and complete LiDAR rendering are outside this forward result. Upstream maps DeformableSubModel to None and retains a LiDAR-output TODO. Camera-rendering results cannot establish those capabilities.
