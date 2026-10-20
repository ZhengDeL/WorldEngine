# SimEngine on Ascend 910C

[中文说明](simengine_ascend_assessment_zh.md)

Updated: October 20, 2026.

This branch implements an NPU forward-rendering adapter for MTGS. In addition to 20 small numerical cases, one 910C chip completed an official original-resolution initial frame: all 1,550,733 Gaussians and eight 1920×1080 cameras, finite complete outputs, and 1,024 sampled pixels checked against independent NumPy. The original configured 12-frame NR/R replay windows also completed, each producing 96 decoded JPEGs. **Exact 12-frame/eight-camera coverage is independently verified. Complete original-log coverage, model-in-the-loop feedback and GPU parity remain unvalidated.**

The upstream WorldEngine baseline is `3eded7e8ac64e5bed817860765089b5914ab698f`; the original rendering contract comes from `gsplat==1.4.0`. The adapter uses CANN gauss-splat commit `8c0445ebcca91514fce3a7050241f41ed132c340` with an explicit inference-only compatibility patch. Its public high-level Rasterizer is not used as a drop-in replacement.

## Measured scope

The server has **8 physical 910C cards, two chips per card, for 16 visible chips**. The small probe and real-scene stages each used one chip, reported as `Ascend910_9362` and mapped to `npu:0` inside the container. Upstream eight-way launch scripts target 8 GPUs. Sixteen-chip SimEngine scaling has not been measured, and those device counts do not imply a throughput ratio.

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

The independent reference is NumPy. RGB/alpha tolerances are `atol=3e-4, rtol=1e-3`; expected depth uses `atol=2e-3, rtol=1e-3`; SH uses `atol=3e-5, rtol=3e-5`. CUDA gsplat parity was not measured. There is no measured speed, cost or energy advantage.

The [numerical result and tested source fingerprints](validation/simengine_npu_forward_20261020.json) identify the current adapter, the publicly built wheel, the actual loaded native/kernel files, 20 passed cases and 18 image-error records. The [CPU check record](validation/simengine_cpu_checks_20261020.json) contains the four test commands and tested source inventory. A newly built wheel or changed adapter must rerun the probe and generate its own manifest. The public summary does not replace a locally generated runtime manifest or authorize larger inputs.

## Official real-scene result

After whole-file verification of the official assets, original BaseEnv.reset/MTGS processing assembled the complete initial-frame input. The NPU forward kept every Gaussian, camera and original pixel. The independent NumPy reference projected all Gaussians for every camera and retained all candidates in sampled tiles, without candidate truncation.

| Item | Recorded result and scope |
| --- | --- |
| Checkpoint | One background, one sky and 17 mirrored rigid submodels; 1,425,076 raw Gaussians |
| Assembled initial input | 1,550,733 Gaussians after mirroring/transforms, eight 1920×1080 cameras, FP32, RGB+ED, tile 16 |
| Complete forward | All RGB+ED, alpha and radii outputs finite |
| Independent reference | Fixed seed `20261020`; 128 pixels per camera, 1,024 total, including corners, center and random positions |
| Maximum RGB/alpha absolute difference | `6.8172804954e-6` / `5.3911735484e-6` |
| Maximum expected-depth absolute difference | `9.7257744097e-3 m`; reference depth at that pixel approximately `491.54 m` |
| NR/R replay | One selected scene completed each original configured window, with 96 decoded JPEGs per run; PDM/dense scoring disabled |

The tolerances remain unchanged from the small probe. Depth uses `atol=2e-3, rtol=1e-3`, allowing approximately `0.49354 m` at the largest-error pixel. The real input did not receive a looser threshold. Numerical acceptance covers complete-output finiteness and sampled reference pixels, rather than a full-image pixelwise comparison. Original replay logs also record the integer ArgSort operation using an AiCPU path; performance is not accepted by this result.

The original runner's `num_history=4, num_future=8` gives a configured 12-frame window. An independent CPU audit verified frame indices 0–11, original tokens, timestamps, previous/next links, all eight original cameras and exactly 96 JPEG paths, dimensions, decoded content and hashes per replay. The exported timestamp span is 5.5 seconds of simulation time, not wall time or a performance measurement. The selected original scene has `log_length=24`; complete original-log coverage is not established. VADv2/HydraMDP FP32/BF16 planner feedback needs its own actual reports and is not included here.

The [portable result](validation/simengine_real_scene_20261020.json) retains input, runtime-source and original proof/report SHA256 values and links the [small numerical record](validation/simengine_npu_forward_20261020.json). Internal server paths are omitted. This result is not a runtime manifest.

| Component | Implementation provided | Acceptance boundary |
| --- | --- | --- |
| NPU renderer | CANN projection/forward kernel, Torch-NPU stable sorting, RGB/alpha/expected depth, SH, MTGS integration | Small cases and original-resolution official initial frame passed; real-input acceptance covers full-output finiteness and sampled reference pixels |
| Geometry and actors | Background, sky, rigid, mirrored and temporal transforms; required FP64 poses remain on CPU | Real background, sky and 17 mirrored rigid submodels exercised; other asset types need their own tests |
| Official input preparation | Whole-file hashes, exact scene-to-asset mapping, safe extraction, unchanged one-item derivation | Both complete official inputs and selected assets verified; 19 checkpoint submodels inventoried on CPU |
| Single-node replay | Original Hydra runner, BaseEnv, DataManager, report checks and deadlines | Official NR/R windows each produced 96 decoded JPEGs; an independent CPU audit verified exact 12-frame/eight-camera coverage |
| VADv2/HydraMDP planner feedback | Separate AlgEngine process, frame/trajectory protocol, explicit FP32/BF16 planner entry | Protocol tests provided; no real closed-loop result in this run; rendering remains FP32 |
| R/NR, rewards and export | Original policies, CPU PDM/dense scoring branches and output checks | NR/R configured windows executed; PDM and dense scoring were disabled in this run |

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

The scene uses the Boston map. Both official files passed whole-file SHA256 checks. The selected 336,430,727-byte checkpoint contains one background, one sky and 17 MirroredRigidSubModel instances. Its 1,425,076 raw Gaussians become 1,550,733 in the original MTGS initial-frame assembly after mirroring/transforms. No Gaussian sampling or truncation was used.

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

The validator executes the complete forward pass, checks every output for finite values, and compares fixed-seed sampled pixels against an independent CPU reference. The default is 128 sampled pixels in total per camera, including corners, center and the remaining random positions. Passing creates `validated_real_manifest.json`. That evidence is tied to its input and render parameters; it does not establish full-pixel parity, an entire scenario, or GPU equivalence. The real input passed these checks; the [portable result](validation/simengine_real_scene_20261020.json) records errors and original evidence hashes.

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

`R` selects original IDM policies; `NR` replays other actors' logs. PDM metrics and dense rewards are opt-in and need real maps and the original trajectory vocabulary. The wrapper checks original RunnerReports, exact scene IDs, completion flags and actual JPEG decoding. The recorded NR/R windows passed a separate exact camera/frame inventory audit; vehicle feedback and applicable scores still require their own acceptance.

## Audit a replay window

This public CLI was executed in the Linux CPU environment: all 11 tool fixtures passed, followed by successful exact inventories of both real NR/R outputs. Its report reproduces every NR/R phase field of the earlier private audit. The snapshot added only three audit tool files; all 223 original files, the 208-file simulation runtime and the 211-file capture runtime remained unchanged. The [result JSON](validation/simengine_real_scene_20261020.json) records the public report/tool SHA256 values and retains the older proof as history.

The public CPU tool checks the original Hydra configuration, RunnerReports, completion flags, consecutive frame metadata and every JPEG. It does not execute NPU or planner operations and requires Python 3.9 or later, NumPy, Pillow and PyYAML. `REPLAY_JOB` must contain both actual `replay_NR` and `replay_R` outputs. Run the preceding command once for each reaction with those corresponding output names; `PREPARED` may be stored separately. Keep each output's `.hydra/config.yaml`, `simulation.log`, completion flag, RunnerReports, `simulation_launch_report.json`, and `openscene_format/meta_datas` and `sensor_blobs` directories intact.

```bash
cd "$SIM_ROOT"
SIMULATION_SOURCE_SHA=$("$SIM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_identity"]["simulation_runtime"]["sha256"])' "$SIM_ROOT/docs/validation/simengine_real_scene_20261020.json")
"$SIM_PYTHON" -m simengine_ascend.tools.audit_replay \
  --job "$REPLAY_JOB" --source-root "$SIM_ROOT" --prepared-dir "$PREPARED" \
  --output "$NEW_AUDIT_JSON" \
  --expected-prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --expected-source-sha256 "$SIMULATION_SOURCE_SHA" \
  --expected-window-frames 12 --expected-cameras 8 \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

Pin both expected SHA256 values before auditing, using the approved preparation receipt and actual simulation-source record. This run's values are in `original_proof_sha256` and `source_identity.simulation_runtime` in the [result JSON](validation/simengine_real_scene_20261020.json). `NEW_AUDIT_JSON` must be absent in an existing parent outside all three read-only input roots: job, prepared and source.

Check exit code 0, `result=pass_two_configured_replay_windows_coverage`, each NR/R frame count, camera count and JPEG inventory, and `full_original_log_coverage`. This run covers frames 0–11, eight cameras, 96 JPEGs and 5.5 seconds of simulation time within an original 24-frame scene. It does not establish complete original-log coverage. The audit and NPU replay entrypoint produce separate reports; neither exit code substitutes for the other check.

## Device planning and remaining acceptance

One chip completed this original-resolution initial frame and the NR/R configured windows. This does not determine minimum HBM, host RAM or device count for arbitrary complete scenes. Planner feedback uses one chip for rendering and one for the planner as a validation configuration; sharing one chip between both has not been measured.

Remaining work covers complete original-log time spans, both planners with FP32/BF16 feedback, applicable metrics/rewards and interruption recovery. Multi-chip/Ray execution, sustained runs and performance need separate acceptance. Any comparison with upstream 8 GPU execution must fix assets, cameras, resolution, model, dtype, score scope and output content, while reporting physical cards and chip/process counts separately.

External MTGS reconstruction training, arbitrary deformable models, unfinished BWM generation and complete LiDAR rendering are outside this forward result. Upstream maps DeformableSubModel to None and retains a LiDAR-output TODO. Camera-rendering results cannot establish those capabilities.
