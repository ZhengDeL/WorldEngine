# NPU replay and closed-loop planner integration

[中文说明](README_zh.md)

Start with the [environment, build and validation guide](../docs/simengine_ascend_assessment.md).
Device evidence now includes 20 small FP32 cases and one official MTGS scene on
one 910C chip: all 1,550,733 Gaussians, eight 1920×1080 cameras, full-output finite
checks and 1,024 sampled pixels against NumPy. The original configured 12-frame
NR/R replay windows completed and each saved 96 decodable JPEGs. Independent
CPU inventory verified frames 0–11 and all eight cameras, spanning 5.5 seconds of
simulation time. These renderer/replay results are retained in the
[bounded real-scene record](../docs/validation/simengine_real_scene_20261020.json).
The subsequent VADv2/HydraMDP × FP32/BF16 × R/NR matrix passed all eight configured
closed-loop control windows. Each run rendered 12 frames, decoded 96 JPEGs and
applied eight model plans through the original policy and controller. See the
[closed-loop evidence](../docs/validation/simengine_closed_loop_20261020.json).
The selected source scene has 24 frames; this result covers the configured
12-frame window, not the complete source log or driving quality.
Small-probe manifests alone do not authorize larger real inputs. The server's
8 physical NPU cards contain 16 chips; this integration has no sixteen-chip
scaling result or GPU comparison.

This package connects SimEngine's frame exchange to an explicitly selected
AlgEngine planner device. The original simulation, camera geometry, selected
trajectory and vehicle-coordinate conversion remain in their respective
components. The planner runs in a separate process; importing the simulator
client does not import `torch_npu` or select an external AlgEngine checkout.

## FP32 tile processing

Tile grids with at most `2**24` entries use exact FP32 integer keys for the
stable tile sort. Larger grids retain the original integer keys. Gaussian
depth order and equal-depth Gaussian-ID order are preserved. The existing
Gaussian-ID expansion also supplies each intersection's start offset through
indexing, avoiding a second `repeat_interleave` over the same counts.

Updating the adapter changes its source hash. Regenerate the small-device
validation and complete captured-input proof before using a new manifest;
previous manifests are rejected for the changed adapter. Keep FP32, the same
input and device, warmup and explicit device synchronization when measuring
render time. Report that timer separately from planner and closed-loop time.
This change retains the forward-only rendering contract.

## Dependencies and source identity

The NPU planner requires the separately adapted AlgEngine repository. This
SimEngine branch retains its original CUDA plugin and does not vendor a second
copy of the adaptation. Use an immutable AlgEngine checkout or a read-only
container mount.

The integration was prepared against:

| Item | Pin |
| --- | --- |
| AlgEngine repository | `https://github.com/ZhengDeL/WorldEngine` |
| Declared commit | `972e1d9592860cef6520e0b39d7e4e9ab55a510a` |
| Algorithm implementation commit | `f0461d7639e38841d218d304f0ab07afebf61eba` |
| Algorithm runtime SHA-256 | `40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b` |
| Runtime fingerprint scope | 245 Python files: `projects/AlgEngine/**/*.py` and `ascend/*.py` |

The runtime fingerprint uses sorted relative POSIX paths followed by each
file's bytes with CRLF normalized to LF. It is checked before initialization
and after inference. The integration files have separate per-file hashes in
the planner report. The declared Git commit documents the intended dependency;
the content fingerprint enforces the actual imported algorithm source.

Use the AlgEngine NPU image and installation instructions from that pinned
repository. Before starting the container, record its actual immutable image ID
with `docker inspect --format '{{.Image}}' CONTAINER`. Pass that value both as
`WORLDENGINE_IMAGE_ID` in the planner environment and as `--expected-image-id`.
The planner checks agreement and records the value. It cannot inspect the host
container engine itself; the host-side image inspection is a deployment step.

NPU-specific dependencies are `torch_npu`, the pinned `ascend` adapter, and
MMCV's `mmcv.device.npu.NPUDataParallel`. They are imported only for an explicit
`--device npu:N`. The chosen logical device is relative to the container's
visible NPU mapping. Record `WORLDENGINE_HOST_DEVICE_IDS` and
`ASCEND_RT_VISIBLE_DEVICES` when the host mapping is configured.

## Planner command

Set these deployment values to existing absolute paths: `SIM_ROOT`,
`ALG_ROOT`, `ALG_PYTHON`, `CFG`, `CHECKPOINT`, `RUN_ROOT`, `IMAGE_ID`, and
`DEADLINE_UTC`. `CFG` must be a compatible planning configuration and
`CHECKPOINT` its complete checkpoint, not a LoRA-only delta. The checkpoint is
loaded with a strict state-dictionary check. `DEADLINE_UTC` must be an explicit
future time with timezone, for example the approved job cutoff in ISO 8601.

Use matching official full 100pct configurations and complete checkpoints for the two published planners:

| Planner | `CFG`, relative to `ALG_ROOT` | `CHECKPOINT`, relative to `ALG_ROOT` |
| --- | --- | --- |
| VADv2 | `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py` | `data/alg_engine/ckpts/vadv2/e2e_vadv2_100pct_ep8.pth` |
| HydraMDP | `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py` | `data/alg_engine/ckpts/hydramdp/e2e_hydramdp_100pct_ep8.pth` |

Both use the original `data/alg_engine/test_8192_kmeans.npy` vocabulary. The closed-loop dataset consumes genuine rendered JPEGs and four consecutive published metadata records; its upstream inference PDM placeholders do not constitute metric scores. Training annotations, full training PDM and the earlier 139-sample subset are not substituted for these inputs. The command allows 600 seconds for initial/following frame queues because a real full-resolution startup can exceed the old 120-second default. The absolute job cutoff still applies.

Run the planner with its adapted AlgEngine directory as the working directory:

```bash
cd "$ALG_ROOT/projects/AlgEngine"
WORLDENGINE_IMAGE_ID="$IMAGE_ID" "$ALG_PYTHON" \
  "$SIM_ROOT/projects/AlgEngine/closed_loop/sim_test.py" "$CFG" "$CHECKPOINT" \
  --device npu:0 --precision fp32 \
  --algengine-root "$ALG_ROOT" \
  --algengine-commit 972e1d9592860cef6520e0b39d7e4e9ab55a510a \
  --expected-algengine-source-sha256 40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b \
  --expected-image-id "$IMAGE_ID" \
  --msda-backend reference --wait-timeout 600 \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$RUN_ROOT/STOP" \
  --cfg-options \
    sim.monitored_folder="$RUN_ROOT/frames" \
    sim.plan_save_path="$RUN_ROOT/plan_traj" \
    sim.merged_ann_save_dir="$RUN_ROOT/merged_ann_files" \
    sim.clean_temp_files=False sim.clean_record_data=False \
    data_root="$RUN_ROOT/WE_output/openscene_format/" \
    data.test.data_root="$RUN_ROOT/WE_output/openscene_format/" \
    data.test.queue_length=4 \
    model.planning_head.vocab_path="$ALG_ROOT/data/alg_engine/test_8192_kmeans.npy"
```

`--precision bf16` uses the external adapter's native NPU autocast, FP32
geometry islands and actual activation checks. It does not cast the model's
parameters to BF16. The default MSDA backend is the adapted reference path;
`--msda-backend cuda-compatible` is a separate explicitly selected experiment.
The reference MSDA path is used by the eight accepted target-device control
windows. The CUDA-compatible MSDA option is not part of this closed-loop result.
The local filesystem fixtures described below remain CPU tests.

The original CUDA command remains available by omitting the NPU arguments.
With no `--precision`, CUDA keeps the configuration's original FP16 setting;
`--precision fp32` disables that setting. GPU BF16 is rejected explicitly.

For a bounded input replay, add `--max-predictions 1` and supply four valid
frame metadata files plus their referenced images. This checks planner input
and output processing. A replay does not execute vehicle feedback or validate
the renderer.

## Simulation and controlled startup

### Single-node NPU replay

`python -m simengine_ascend.run_simulation` supervises the original Hydra
simulation runner in a separate process. It fixes `distributed_mode=SINGLE_NODE`
and does not create a Ray worker. The original code still imports the Ray
Python package when resolving process IDs, and eagerly imports NuPlan and the
PDM managers even when scoring is disabled. Install the original SimEngine
dependencies in addition to the validated CANN renderer and Pillow. A successful
kernel probe does not establish that these simulator dependencies are installed.

Prepare the official scene and asset with
[`tools/simengine_ascend/prepare_scene.py`](../tools/simengine_ascend/prepare_scene.py).
Use its `scene.pkl`, `assets` directory and `prepare_receipt.json`. Define
`PREPARED`, `MAPS_ROOT`, `NPU_MANIFEST`, `JOB_PARENT`, `SCENE_IDS_JSON`,
`SIM_PYTHON`, `SIM_ROOT` and `DEADLINE_UTC` as existing deployment values.
`NPU_MANIFEST` must be a validated renderer manifest for the exact runtime and
shape envelope; the simulation wrapper does not expand that envelope.

Create the expected-ID list from the preparation receipt, using the dictionary
key used by the original ScenarioManager. Read the expected scene hash from the
same receipt; neither value is invented by the launcher:

```bash
export PREPARED SCENE_IDS_JSON
"$SIM_PYTHON" -c 'import json,os,pathlib; r=json.loads((pathlib.Path(os.environ["PREPARED"])/"prepare_receipt.json").read_text()); p=pathlib.Path(os.environ["SCENE_IDS_JSON"]); p.write_text(json.dumps([r["selected"]["dictionary_key"]])+"\n")'
SCENE_SHA256=$("$SIM_PYTHON" -c 'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["PREPARED"])/"prepare_receipt.json").read_text())["derived_scene"]["sha256"])')
cd "$SIM_ROOT"
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:0 \
  --output "$JOB_PARENT/replay_NR" --deadline-utc "$DEADLINE_UTC" \
  --expected-scene-ids "$SCENE_IDS_JSON" --mode replay --reaction NR
```

`JOB_PARENT` must exist and `replay_NR` must be new. Add `--dry-run` to print
the exact command, hashes and paths without importing Torch, loading the pickle,
creating the output directory or running the simulator. A dry run only checks
command preparation. The actual child registers Torch-NPU, verifies the
`worldengine` module origin and executes the original runner, BaseEnv and
`run_runners`. It flushes the original DataManager for the final scene and closes
the engine on exit.

Replay uses the original `trajectory_policy` and `log_play_controller` for
the ego vehicle; it does not request a model plan. `NR` replays other agents'
logged trajectories. `R` selects the original `idm_policy` and `idm_navigation`,
as in `scripts/run_testing.sh`. The wrapper sets `debug_static_world=false`;
at this source revision the field is declared in YAML but has no Python reader.
The actual R/NR behavior is selected by the policy and navigation settings.

### NPU simulation with a planner

Use the same command with `--mode closed-loop` and add
`--planner-data-path /absolute/shared/plan_traj` and
`--planner-client-folder /absolute/shared/frames`. Those directories must already
exist. This mode follows the original testing script's `env_input_policy`,
`navformer_client`, `trajectory_navigation` and `log_play_controller` settings.
For the validated two-chip configuration, set the simulator to `--device npu:1`
and keep the separate AlgEngine planner on `--device npu:0`, using the same
directories. The default wait for a missing plan is bounded by the client
protocol and the shared stop/deadline environment.

The `simulation` phase in a controlled launch plan can therefore use:

```json
{
  "argv": ["ABSOLUTE_SIM_PYTHON", "-m", "simengine_ascend.run_simulation", "--scene-pkl", "ABSOLUTE_PREPARED_SCENE", "--asset-root", "ABSOLUTE_PREPARED_ASSETS", "--maps-root", "ABSOLUTE_MAPS", "--validated-manifest", "ABSOLUTE_VALIDATED_MANIFEST", "--device", "npu:1", "--output", "{WE_output}", "--deadline-utc", "{deadline_utc}", "--stop-file", "{stop_file}", "--expected-scene-ids", "ABSOLUTE_EXPECTED_IDS_JSON", "--mode", "closed-loop", "--reaction", "NR", "--planner-data-path", "{plan_traj}", "--planner-client-folder", "{frames}"],
  "cwd": "{repo_root}"
}
```

Replace the uppercase deployment entries before use. The outer launcher leaves
`WE_output` absent so this wrapper can create a fresh simulation output. Point
the planner's `data_root` at `{WE_output}/openscene_format/`.

`gpu=false` changes the original worker scheduling fields. The renderer remains
explicitly on the chosen NPU. Vehicle dynamics, controllers and optional PDM
scoring retain their original CPU implementation. The simulation child hides
CUDA and does not import `transfer_to_npu`, so the original dense manager's CUDA
availability check selects its CPU branch. Add `--pdm-metrics` to enable the
original PDM MetricManager, or `--dense-rewards` for original dense scoring. Dense
scoring also requires the original vocabulary at
`SIM_ROOT/data/alg_engine/test_8192_kmeans.npy`, sufficient scene history/future
data and CPU RAM; it creates the original eight scoring processes. These are
optional runs with additional dependencies and cost, and are not enabled by
default.

`simulation_launch_report.json` records the exact child command, source/input
hashes, selected mode, device and original runner report checks. Every saved
JPEG is verified and decoded on CPU. It also requires the original completion
flag and the exact successful scene IDs. These checks do not certify complete
camera/frame coverage, GPU image parity or planning quality. Missing dependencies,
out-of-envelope rendering, scene errors, corrupt images and timeout remain
nonzero exits. Logs are kept in `simulation.log`.

Both standalone and nested execution use an explicit UTC cutoff and an owned
child process group. `STOP` or SIGTERM stops that child; the wrapper allows three
seconds for termination before killing remaining group members. On Linux it
retains an exited leader with `WNOWAIT` until group cleanup, so cleanup also
handles descendants left behind after the leader exits. Optional
detached processes still require the dedicated container/job cutoff described
below. This entrypoint has CPU command/image/process fixtures and completed
the recorded official NR/R replay windows. Each original RunnerReport contained
one successful selected scene and each run saved 96 decoded JPEGs. These
12-frame configured windows do not cover the original scene's `log_length=24`.
An independent CPU audit matched all 12 frame identities, timestamps, previous/next
links and all 96 expected camera JPEGs, including dimensions, full decoding and
hashes, against the original inputs. Its 5.5-second interval is simulation time,
not runtime or a performance result. This audit is separate from the wrapper
checks and does not establish planner feedback or GPU parity.

```bash
python -m simengine_ascend.test_run_simulation
```

The simulator must use this branch's `NAVFormerClient`, publish metadata to
`frames`, read plans from `plan_traj`, and write its OpenScene-format images to
`WE_output/openscene_format/sensor_blobs`. Its output directory is `WE_output`.
This preserves the original protocol: metadata indices `0,1,2,3` produce the
first `<scene>_4.npy`; subsequent requests advance one frame at a time. The
history length and camera pipeline are taken from the original model config.

Prepare a JSON plan containing exactly `planner` and `simulation`. Each phase
contains a nonempty `argv` array, an explicit `cwd`, and optionally an `env`
object. Use the reviewed simulation command for the selected renderer,
scenario assets and device. Commands are executed directly without a shell;
each array element is one argument. A minimal structure is:

```json
{
  "planner": {
    "argv": ["ABSOLUTE_ALG_PYTHON", "ABSOLUTE_SIM_TEST", "ABSOLUTE_CONFIG", "ABSOLUTE_CHECKPOINT", "REPLACE_WITH_THE_REVIEWED_PLANNER_ARGUMENTS"],
    "cwd": "ABSOLUTE_ALGENGINE_DIRECTORY",
    "env": {"WORLDENGINE_IMAGE_ID": "IMMUTABLE_IMAGE_ID"}
  },
  "simulation": {
    "argv": ["ABSOLUTE_SIM_PYTHON", "worldengine/runner/run_simulation.py", "REPLACE_WITH_THE_REVIEWED_SIMULATION_ARGUMENTS"],
    "cwd": "{repo_root}/projects/SimEngine",
    "env": {"WORLDENGINE_ROOT": "{repo_root}"}
  }
}
```

The uppercase entries are deployment placeholders and must be replaced before
execution. The launcher only expands its own literal placeholders:
`{repo_root}`, `{run_root}`, `{deadline_utc}`, `{stop_file}`, `{frames}`,
`{plan_traj}`, `{merged_ann_files}`, and `{WE_output}`. For example,
`sim.monitored_folder={frames}` is one planner argument. Set the simulator's
`planner_client_folder={frames}`, `planner_data_path={plan_traj}` and
`output_dir={WE_output}` to the same paths.

From the SimEngine repository root:

```bash
python -m simengine_ascend.launch_closed_loop \
  --plan /absolute/path/closed_loop_plan.json \
  --run-root /absolute/path/new_run_directory \
  --expected-scenarios "$EXPECTED_SCENARIOS" \
  --expected-scene-ids /absolute/path/selected_scene_ids.json \
  --deadline-utc "$DEADLINE_UTC"
```

Set `EXPECTED_SCENARIOS` to the exact number of selected scenes. Supply a JSON
list of their exact original scenario IDs in `selected_scene_ids.json`; its
unique entries must match this count. The launcher
requires a fresh run directory, creates the three exchange
directories, starts the planner and simulator, and records their exact commands
and exit codes. It injects the shared stop-file and cutoff environment into
both processes. Touch `RUN_ROOT/STOP`, send SIGTERM to the launcher, or wait for
the cutoff to stop its owned commands. A nonzero phase exit stops the peer.
On Linux, cleanup retains each direct-child leader until its original process
group has been stopped and reaped. It also stops group members left behind
after that leader exits. Other platforms limit cleanup to the direct child.
For Ray workers or container processes that detach into separate sessions,
enforce the same cutoff on the dedicated container or job boundary as well.
The launcher does not issue global Ray, NPU or unrelated process commands.

Use a private run root on a filesystem supporting atomic same-directory rename
and hard links. Keep its parent paths under the job owner's control. Standalone
planner use requires the same ownership rule; do not reuse another job's
directories. Published trajectories use no-overwrite creation. Temporary-file
cleanup deletes exact files created by this planner and does not delete renderer
records or glob shared frame directories.

## Audit a closed-loop control window

Enable consumption evidence in the **simulation phase** of the reviewed launch
plan before running a new job:

```json
"env": {"WORLDENGINE_CLOSED_LOOP_AUDIT_DIR": "{WE_output}/consumption_audit"}
```

Preserve the phase's other environment entries. The directory must be new and
inside this job's `WE_output`; replay and GPU defaults do not enable this observer.
The client records the exact NPY bytes parsed, the returned trajectory object,
the original policy/controller execution and actual log-play state update. A
successful consumption receipt is published only after the real successor frame
and its eight camera files exist. Interrupted jobs retain their completed records.

The public CPU auditor checks one official 12-frame window with four history
frames. It requires all eight applied control indices `4..11`, four consecutive
real model inputs per request, finite `40x3` plans matching the original vocabulary,
an unbroken consumption-receipt chain, and the exact 96-camera-image matrix with
complete JPEG decoding at the original dimensions. An additional published plan
12 may exist; it is not counted as an applied control.

Run in an isolated CPU container with **no NPU/GPU device mounts**, NumPy, Pillow
and PyYAML. Mount the job, selected source, external AlgEngine and prepared assets
read-only. Supply the original processes' absolute job path as
`RECORDED_RUN_ROOT`; `RUN_ROOT` may be a different read-only CPU mount. The output
must be a new JSON outside every input root, in an existing writable parent:

```bash
python -m simengine_ascend.tools.audit_closed_loop \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALGENGINE_ROOT" \
  --prepared-dir "$PREPARED" \
  --expected-source-sha256 "$APPROVED_SIM_RUNTIME_SHA" \
  --expected-prepare-receipt-sha256 "$APPROVED_PREPARE_RECEIPT_SHA" \
  --model vadv2 --precision fp32 --reaction NR \
  --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_CLOSED_LOOP_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

The two host chip indices are explicit, ordered, distinct values in `0..15`:
planner first, renderer second. Their recorded plan/runtime mapping must agree
with these independently supplied values; the logical devices remain `npu:0`
and `npu:1`. Each phase's `env` overrides inherited container environment values:
set `WORLDENGINE_HOST_DEVICE_IDS` in **both** phases to the actual comma-separated
pair. Changing only the container environment does not replace a value already
fixed in the plan. Use `--model hydramdp` for HydraMDP, `--precision bf16` for BF16 and
`--reaction R` for the original IDM reaction policy/navigation. Both the wrapper
reaction label and resolved Hydra policies, scene and exchange paths must match.

This audit profile fixes the official 100% VADv2/HydraMDP checkpoints and original
8192-entry vocabulary, external AlgEngine source
`40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b`
at commit `972e1d9592860cef6520e0b39d7e4e9ab55a510a`, and recorded image ID
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`.
It verifies these reported identities and input files; the job owner must also
verify the actual container image and device mounts. Other model/source/image
profiles need a separate reviewed acceptance contract. Source SHA values use
relative POSIX ordering and must identify the **snapshot actually executed**.
The tools are outside the simulator's top-level runtime fingerprint; keep their
own file SHA256 values in the delivery manifest and audit reports as well.
When auditing an older snapshot that predates this public tool, start a fresh
Python process with `python /absolute/path/to/audit_closed_loop.py` and the same
arguments, while `--source-root` still selects the actual executed snapshot.
This loads the CPU helpers from that selected snapshot without replacing its
recorded files. The module form above assumes the tool and selected helpers are
in the same checkout; a shadowed helper remains a rejected source mismatch.

FP32 requires representative native Conv2d/Linear outputs on the selected NPU
with FP32 parameters and autocast disabled. BF16 requires its actual BF16
activations, FP32 master parameters and geometry islands, and restored autocast
state. These observations do not constitute an all-layers/all-frames dtype trace.

The audit accepts the configured control window. Full source-log coverage,
driving quality, official scores, GPU equivalence, training convergence and
performance remain separate checks. Actual R and NR runs have each passed for
both models and planner precisions; the CPU fixtures separately check rejected
configurations and damaged evidence. Run those fixtures with:

```bash
python -m unittest simengine_ascend.tools.tests.test_audit_closed_loop -v
```

## Validated closed-loop windows

| Planner | Planner precision | NR | R |
| --- | --- | --- | --- |
| VADv2 | FP32 | Pass | Pass |
| VADv2 | BF16 | Pass | Pass |
| HydraMDP | FP32 | Pass | Pass |
| HydraMDP | BF16 | Pass | Pass |

Each entry is an actual NPU run followed by the public CPU auditor. The auditor
verified controls `4..11`, their four-frame input queues, exact NPY content and
vocabulary selection, policy/controller object identity, the applied sample-1
vehicle state and the next rendered frame. It also verified the full 12 × 8
camera/JPEG matrix. Both model checkpoints are official full 100pct weights.
FP32 observations recorded actual FP32 Conv2d/Linear outputs with autocast off;
BF16 observations recorded BF16 outputs with FP32 master weights and geometry/MSDA
islands. These are representative activation observations, not an all-layer trace.

Each case used two 910C chips on one physical card: planner `npu:0`, FP32 renderer
`npu:1`. This is a tested configuration, not a minimum-capacity or GPU-equivalence
claim. Each window spans 5.5 seconds of simulation time, not 5.5 seconds of wall
time. PDM metrics and dense rewards were disabled. Full-log execution, official
quality scores, dense image/GPU parity and performance remain unmeasured.

The accepted simulator source is `90ad1a295420d87ad63391733f8f28124161b43d939964c55831b7a8e259a7d8`
(211 runtime files); the public auditor SHA is
`83c79ef344e995d3d440580e8db85fe8d792888eeafee0fe452b285d636758fe`.
Per-case source, plan, checkpoint, image inventory and eight consumption-receipt
hashes are in the [result JSON](../docs/validation/simengine_closed_loop_20261020.json).
Earlier four-case diagnostics from source `1ef363…` are retained as history and
are excluded from acceptance because their host/plan device mappings differed.
The eight cases here are SimEngine control windows; they do not establish the
eight AlgEngine synthetic training configurations.

## Reports and validation scope

The public CPU replay auditor passed 11 Linux CPU fixtures and actually verified both recorded NR/R windows. It checks the exact frame/image inventory independently of the wrapper. Its [command and input contract](../docs/simengine_ascend_assessment.md#audit-a-replay-window) cover both NR/R sibling outputs, pinned preparation/source SHA256 values and a fresh report outside the inputs.

`plan_traj/closed_loop_planner_report.json` records strict checkpoint loading,
configuration/checkpoint hashes, actual prediction count, last trajectory
hash and selection index, dependency origins, source hashes, precision and
device mapping. For VADv2/HydraMDP, the output must match its original 8192-entry
vocabulary selection. Generated-trajectory heads use their original anchor
index range instead. Invalid trajectories, corrupt published files, missing
frames beyond the configured timeout, source changes and shadowed CUDA plugins
raise errors and stop the run.

`launcher_report.json` records both child process exits. The CLI also requires
the original `WE_output/runner_report_*.json` reports to contain exactly the
requested exact scenario IDs and number of unique successful scenarios, without an error message or
invalid duration. This catches original runner paths that exit zero after a
scene error. A zero launcher exit means these checks and both command exits
passed. Acceptance of a configured control window additionally requires the
camera-coverage and actual-consumption checks above. Official metrics evaluate
driving quality separately. A one-prediction replay cannot establish feedback.

Run the local CPU protocol and process tests with:

```bash
python -m simengine_ascend.test_closed_loop_protocol
```

The suite covers atomic publication and rollback, invalid/missing trajectory
handling, consecutive frame queues and scene switching, CUDA import isolation,
NPU identity/origin rejection, owned-process cleanup, cutoff enforcement and a
STOP-file write error. It uses synthetic filesystem fixtures and short-lived
Python child processes. It does not load model checkpoints, execute NPU kernels,
render a real scene or certify closed-loop quality.
