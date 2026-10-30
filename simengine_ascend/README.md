# NPU replay and closed-loop planner integration

[中文说明](README_zh.md)

## Explicit FP32 planner optimization

The current runtime adds an optional bridge to the separately pinned AlgEngine
inference context. The default `--inference-optimization none` retains the
original planner. The optimized scored profile selects
`--inference-optimization point-parallel-last-frame --msda-query-chunk 8192`
with `--device npu:0 --precision fp32 --msda-backend cuda-compatible`.
It requires AlgEngine runtime
`cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610`
(249 Python files), including the three helper hashes enforced by
`planner_optimization.py`. Pass its separately verified full Git commit as
`--algengine-commit`. This profile fixes `ALG_OPT_COMMIT` to
`a783047c92621ca3d2f658fbe054f200783e8e25`; check out that external commit.
A declared commit alone does not verify the source.

For this profile, use the eight-control simulation configuration below:
planner `sim.max_frame_per_scene=12`, simulator
`--num-history 4 --num-future 8 --pdm-metrics --dense-rewards`,
and the complete unchanged real 24-frame prepared scene. Add the two
optimization options to the **planner phase argv** before `--cfg-options` in
the controlled launch plan. Keep the renderer FP32 and its existing validated
CANN capture proof. No rendering operator or scoring formula is replaced.
The new SimEngine runtime fingerprint is
`9d09e7bb239fcb0238255e689423598115c0115841b47912b10ffbef80b1e541`
(213 runtime files). At commit `7e9389e9a4f6206007bb48c6b82a80e3eac42b6a`, four actual VADv2/HydraMDP × NR/R NPU jobs and the public CPU auditor passed; the complete persisted scored prefix matches the original 212/245 outputs byte for byte.

PDM × 100 remains 79.057/0 for VADv2 NR/R and 100/27.233 for HydraMDP NR/R, including the original low R scores. In those four runs, each VAD case generated nine plans and recorded 54 compensated historical GridMask calls; each Hydra case generated eight and recorded 48. All applied eight controls with no optimization fallback. Four later same-host/same-pair observations completed, but the sequence was interrupted and generated 9/8/9/8 plans. Their actual wall times are retained in the validation JSON without a formal speed ratio.

### Terminal planning boundary

Controlled NPU loops bind the history length and frame limit to the simulator's actual
`simulation_launch_report.json` window. For this scored configuration, twelve
rendered frames require plans 4 through 11. The original 24-frame source scene
continues to supply genuine scoring context; it does not authorize plan 12.
At the boundary, the existing bounded monitor waits for the next scene or the
completion flag before another model call. No arbitrary delay is added.

The repair passed VADv2/HydraMDP × NR/R on this one official scene. Each run
generated and consumed exactly eight plans, retained all 96 JPEGs, twelve
metadata files, eight plans and actual inputs, nine complete dense positions,
and the persisted five-decimal PDM CSV byte for byte, and left no `merged_12`
or plan 12. The three additional paths ran the published SimEngine commit
`933f6fb780f644c4bae82abfa93f3f488eaa2596`; the original VADv2 NR evidence retains
its earlier declaration and the same repaired entry hash. VADv2 R still scores
0 and HydraMDP R 27.233: functional/raw acceptance does not establish driving
quality. GPU execution, reconstruction training, and a new uninterrupted
equal-work end-to-end performance comparison remain unverified. The planner entry is outside the 213-file runtime
fingerprint; its separate integration SHA is
`f61e5d033887ab6051fabbba4e93defd8081b68af0c0e2fc8193ce3dcb955c5a`.
Use the matching complete checkout, not the runtime fingerprint alone.

Keep the launch and public audit commands below. After the audit succeeds,
check the repaired eight-control boundary explicitly:

```bash
python - "$NEW_AUDIT_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report['result'] == 'pass_optimized_default8_scored_window_and_original_prefix_raw'
assert report['published_plan_indices'] == list(range(4, 12))
PY
python -m unittest simengine_ascend.tools.tests.test_planner_horizon -v
```

The complete 24-frame/twenty-control results and original eight-control scores
later in this guide belong to SimEngine commit
`27b13d7a03da257c3d2a66a95b640bd522ab0c61` / runtime 212, with AlgEngine 245.
Use that immutable SimEngine checkout for those older audit commands. They
are not acceptance of the new planner bridge or its end-to-end performance.

The new public CPU auditor additionally compares every persisted prefix byte
with a completed original 212/245 scored job. First create a portable reference
from that real job and its successful public `audit_scored_default8` report.
`REFERENCE_AUDIT_SHA` is the independently recorded whole report SHA-256;
`REFERENCE_JSON` must be a fresh file outside all input trees. Use a CPU-only
container, read-only inputs, and the new checkout's Python path for both modes:

```bash
python -m simengine_ascend.tools.audit_optimized_default8 prepare-reference \
  --job "$ORIGINAL_RUN" --recorded-job-root "$ORIGINAL_RECORDED_RUN" \
  --validated-audit "$ORIGINAL_SCORED_AUDIT" \
  --expected-audit-sha256 "$REFERENCE_AUDIT_SHA" \
  --source-root "$SIM_ROOT" --prepared-dir "$PREPARED" \
  --vocabulary "$VOCABULARY" --model vadv2 --reaction NR \
  --output "$REFERENCE_JSON" --deadline-utc "$DEADLINE_UTC"
```

Independently record the resulting whole `REFERENCE_SHA`. Then audit the new
complete job, using its actual host chip pair and literal recorded AlgEngine
commit. `NEW_AUDIT_JSON` must also be fresh and outside the inputs:

```bash
python -m simengine_ascend.tools.audit_optimized_default8 audit \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALG_ROOT" \
  --recorded-algengine-root "$RECORDED_ALGENGINE_ROOT" \
  --prepared-dir "$PREPARED" \
  --reference-inventory "$REFERENCE_JSON" --expected-reference-sha256 "$REFERENCE_SHA" \
  --expected-source-sha256 9d09e7bb239fcb0238255e689423598115c0115841b47912b10ffbef80b1e541 \
  --expected-algengine-source-sha256 cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610 \
  --expected-algengine-commit "$ALG_OPT_COMMIT" \
  --expected-prepare-receipt-sha256 "$APPROVED_PREPARE_RECEIPT_SHA" \
  --model vadv2 --precision fp32 --reaction NR \
  --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

Use matching `--model hydramdp` / `--reaction R` in both modes for the other
paths. The audit checks 96 JPEGs, twelve metadata files, eight consumed plans
and actual four-frame model inputs, nine complete dense pickle files and all
eight arrays per position, and the persisted five-decimal PDM CSV. It also
requires actual optimization calls, no fallback and the actual compensation
counter for skipped historical GridMask calls (six NumPy draws per generated
prediction). This counter is not the total RNG calls in the original path.
The prediction count is recorded separately from
the eight consumed controls; a terminal unconsumed plan is excluded from the
prefix-byte comparison. Internal unrounded PDM values, GPU equivalence,
all-scene quality and renderer backward are outside this check.

CPU fixtures for the bridge and new audit:

```bash
python -m unittest simengine_ascend.tools.tests.test_planner_optimization \
  simengine_ascend.tools.tests.test_audit_optimized_default8 -v
```

Start with the [environment, build and validation guide](../docs/simengine_ascend_assessment.md).
This FP32 branch covers one official MTGS scene's complete 24-frame source log
for VADv2/HydraMDP × NR/R. Each of the four cases used four genuine history
frames and twenty applied model controls, then independently decoded all
192 original-size camera JPEGs. See the [FP32 full-scene and renderer timing
record](../docs/validation/simengine_fp32_20261029.json) for source, input and
per-case audit hashes. The original eight-control PDM/dense profile and its
measured low-score cases are documented below.

On one fixed full-resolution eight-camera capture, sequential same-chip ABBA
measurements reduced median renderer forward time from 38.9381 to 17.7366 seconds.
The complete uncompressed RGB+ED, alpha and radii were bitwise identical. This
is a renderer optimization comparison on one 910C chip, not GPU equivalence or
a closed-loop end-to-end timing result. Single-scene control acceptance does
not establish all-scene driving quality or reconstruction-training support.

Earlier [renderer/replay](../docs/validation/simengine_real_scene_20261020.json)
and [eight 12-frame FP32/BF16 closed-loop cases](../docs/validation/simengine_closed_loop_20261020.json)
retain their original source, dependencies and acceptance scope. A small-device
manifest alone does not authorize a larger input. Each current closed-loop case
uses two chips of one dual-chip physical NPU card; sixteen-chip scaling is not
part of this result.

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
| AlgEngine Git commit for the FP32 24-frame profile | `55efdef54139c63ab9afce9178ebb3bd130717ef` |
| Algorithm runtime SHA-256 | `0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922` |
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
future time with timezone. Use an ISO 8601 `+00:00` UTC offset for Python 3.9/3.10 compatibility. Set a technical cap appropriate to the scene, not a stale cutoff from an earlier experiment.

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
  --algengine-commit 55efdef54139c63ab9afce9178ebb3bd130717ef \
  --expected-algengine-source-sha256 0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922 \
  --expected-image-id "$IMAGE_ID" \
  --msda-backend cuda-compatible --wait-timeout 600 \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$RUN_ROOT/STOP" \
  --cfg-options \
    sim.max_frame_per_scene=24 \
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
parameters to BF16. The earlier eight 12-frame cases used the reference MSDA
backend. The current 24-frame profile is FP32 only and explicitly selects
`cuda-compatible`; the two profiles do not share acceptance results.
The local filesystem fixtures described below remain CPU tests.

The original CUDA command remains available by omitting the NPU arguments.
With no `--precision`, CUDA keeps the configuration's original FP16 setting;
`--precision fp32` disables that setting. GPU BF16 is rejected explicitly.

For a bounded input replay, add `--max-predictions 1` and supply four valid
frame metadata files plus their referenced images. This checks planner input
and output processing. A replay does not execute vehicle feedback or validate
the renderer.

## Complete selected 24-frame FP32 scene

Check out the external AlgEngine dependency separately at the full Git commit
in the table above. Keep both code trees immutable during execution. Set the
planner command above to `sim.max_frame_per_scene=24` in its single
`--cfg-options` list. Use the matching full official checkpoint and vocabulary.
Create new `frames`, `plan_traj` and `merged_ann_files` directories under
`RUN_ROOT`; `WE_output` must remain absent until the simulator starts.

For the simulation command below, add `--num-history 4 --num-future 20`, keep
`--scene-sha256`, set `--mode closed-loop --device npu:1`, and point
`--planner-data-path` and `--planner-client-folder` at the same `plan_traj` and
`frames` directories used by the planner. Start both commands with the existing
controlled launcher described below. Select `--reaction NR` or `R` explicitly.
The source scene must contain 24 original metadata frames and all eight
cameras. Short scenes, missing cameras, duplicate frame tokens and extra pickle
bytes are rejected; no frames are repeated or synthesized to fill the window.
Unlike the default 12-frame dry run, `--dry-run --num-future 20` also opens the
whole-SHA-verified prepared pickle to check real metadata availability. It
still executes no model or simulator and creates no run output.

This dedicated 24-frame acceptance profile explicitly rejects both scoring
flags. Dense scoring over twenty controls would need 32 real source frames,
which this scene does not contain. PDM has a different context contract; its
acceptance is established separately in the eight-control scored profile below.
The original twenty controls remain accepted without scoring. A small-device manifest alone is insufficient; regenerate and pass
the full captured-input renderer proof for the exact adapter before launch.

After both commands complete, run the dedicated full-window auditor in a
CPU-only container, with inputs mounted read-only and no accelerator devices:

```bash
python -m simengine_ascend.tools.audit_full24 \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALGENGINE_ROOT" \
  --prepared-dir "$PREPARED" \
  --expected-source-sha256 "$APPROVED_SIM_RUNTIME_SHA" \
  --expected-algengine-source-sha256 0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922 \
  --expected-algengine-commit 55efdef54139c63ab9afce9178ebb3bd130717ef \
  --recorded-algengine-root "$RECORDED_ALGENGINE_ROOT" \
  --expected-prepare-receipt-sha256 "$APPROVED_PREPARE_RECEIPT_SHA" \
  --model vadv2 --precision fp32 --reaction NR --expected-window-frames 24 \
  --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_CLOSED_LOOP_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

`RECORDED_ALGENGINE_ROOT` is the original planner's absolute code path; it may
differ from the CPU audit mount. The expected declared commit must equal the
literal value in the original report. The October 29 experiments recorded the
older declaration `972e1d9592860cef6520e0b39d7e4e9ab55a510a`; their executed
245-file source was independently matched file-for-file to clean public commit
`55efdef54139c63ab9afce9178ebb3bd130717ef`. Re-audit those original records with
their recorded declaration, without rewriting the reports. New launches should
use the verified public commit above. The content fingerprint is checked in
both cases; a declared Git SHA alone does not prove the executed source.

CPU checks for the window contract:

```bash
python -m unittest simengine_ascend.test_run_simulation -v
python -m unittest discover -s tools/simengine_ascend/tests -p test_window_contract.py -v
```

## Original PDM and dense scoring on an eight-control prefix

The FP32 scored profile uses the same complete official 24-frame scene and
AlgEngine commit `55efdef54139c63ab9afce9178ebb3bd130717ef`. It renders frames
`0..11`, applies eight controls `4..11`, and keeps the original, unrendered
frames `12..19` as real future context for CPU dense scoring. Do not shorten
the prepared pickle to twelve frames or repeat frames. This is a single
NAVTEST failure scene, not a NAVTRAIN rare-scene synthetic training set.

On 2026-10-30, VADv2/HydraMDP × NR/R completed the original PDM and dense paths.
Each case produced 96 fully decoded JPEGs, one endpoint PDM row at step 11,
and nine dense positions `3..11`, each containing 8192 candidates × eight
finite metric arrays. The original PDM CSV stores five decimal places:

| Planner | Reaction | PDM score × 100 | Recorded quality observation |
| --- | --- | ---: | --- |
| VADv2 | NR | 79.057 | Progress 0.49736; NC/DAC/TTC equal 1 |
| VADv2 | R | 0 | NC gate 0; first NC/DAC violation at step 11; DAC 1 |
| HydraMDP | NR | 100 | All seven endpoint fields equal 1 |
| HydraMDP | R | 27.233 | TTC 0, progress 0.25360; NC/DAC equal 1 |

These are measured NPU-planner/CPU-scoring outputs, not GPU comparisons or a
dataset-wide driving-quality acceptance. The low R scores are retained. The
`first_violation_step` field records NC/DAC events; it does not establish a TTC
event time. All four cases' 96 JPEGs, eight consumed plans and eight actual
merged planner inputs match the corresponding prefix of the earlier unscored
24-frame runs byte for byte. This checks that enabling scoring preserved that
prefix; it does not prove GPU equivalence. Per-case receipts and boundaries
are in the [versioned result](../docs/validation/simengine_fp32_20261029.json).

Use the existing controlled two-process launch plan. Keep the planner's FP32,
CUDA-compatible MSDA and pinned source/image checks, but set
`sim.max_frame_per_scene=12` in its single `--cfg-options` list. Also set
`sim.post_process_path` to the original vocabulary and retain both cleanup
flags as `False`. The simulator phase uses:

```bash
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:1 \
  --output "$RUN_ROOT/WE_output" --deadline-utc "$DEADLINE_UTC" \
  --stop-file "$RUN_ROOT/STOP" --expected-scene-ids "$SCENE_IDS_JSON" \
  --mode closed-loop --reaction NR --num-history 4 --num-future 8 \
  --planner-data-path "$RUN_ROOT/plan_traj" \
  --planner-client-folder "$RUN_ROOT/frames" --pdm-metrics --dense-rewards
```

Place this argv in the existing launcher plan with its `{WE_output}`,
`{deadline_utc}`, `{stop_file}`, `{plan_traj}` and `{frames}` substitutions.
Use a new `RUN_ROOT` and matching NR/R in both the plan and audit. Each case
uses a planner/renderer chip pair. The dense manager retains its original
eight CPU workers; allocate CPU RAM and workers separately from the NPU count.
The measured cases used a 48 GiB CPU memory limit and 16 CPU cores per complete
job, plus a separate 16 GiB/two-core audit. These are the tested limits, not a
measured minimum or a full-training memory requirement.

Both code trees, prepared assets, maps, official complete checkpoints and
vocabulary must be readable in each process. The dense manager resolves its
vocabulary relative to `SIM_ROOT/data/alg_engine/test_8192_kmeans.npy`; the
planner also needs `ALG_ROOT/data/alg_engine/test_8192_kmeans.npy`. Mount the
same official data tree read-only at both locations. Keep the complete CANN
proof closure: the full real-input manifest, its referenced small-device
manifest and device report, capture JSON and NPZ archive, real-input reference
report, and all 46 native runtime files. Preserve the absolute paths recorded
in those proofs with read-only aliases, or regenerate the proofs for the new
layout. Copying only `validated_real_manifest.json` is insufficient.

The simulator environment needs the original NuPlan/Ray/geometry dependencies
and maps. Preserve the validated Torch/NumPy/PROJ versions and import paths;
do not replace an entire site-packages tree. Set `PROJ_DATA`/`PROJ_LIB` to the
matching PROJ data directory, and bound BLAS threads in the CPU workers. The
original managers run their scoring formulas on CPU in this profile; enabling
these flags does not migrate scoring to NPU.

After both original child commands have actually completed, run the separate
fixed-profile audit in a CPU-only container with read-only inputs and no
accelerator devices:

```bash
python -m simengine_ascend.tools.audit_scored_default8 \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALG_ROOT" \
  --recorded-algengine-root "$RECORDED_ALG_ROOT" --prepared-dir "$PREPARED" \
  --expected-source-sha256 daabd5c34ed290038ad4d0da13c32e81100b9692a0b63ff9b657f72edd9c7d66 \
  --expected-algengine-source-sha256 0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922 \
  --expected-algengine-commit 55efdef54139c63ab9afce9178ebb3bd130717ef \
  --expected-prepare-receipt-sha256 ac246caf243f2e1a679c4159dbd92ecb1cfe84f4197fc023295909d7943414fc \
  --model vadv2 --precision fp32 --reaction NR --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_SCORED_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

`RECORDED_RUN_ROOT` and `RECORDED_ALG_ROOT` are the literal paths saved by the
original processes, not necessarily the CPU audit's mount locations. This
auditor checks the full consumption chain, original policy/controller state,
complete images, pinned source/checkpoint/precision, PDM CSV, and all dense
arrays. It checks that eight-step target fields are complete and finite; it
does not recompute targets or scoring formulas. A score of zero remains a
valid measured quality result. The older `audit_closed_loop` and `audit_full24`
commands retain their unscored profiles and must not be used for this case.

```bash
python -m unittest simengine_ascend.tools.tests.test_audit_scored_default8 -v
```

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
