# Recoverable planning inference on Ascend

> Historical record: this page preserves the October 20, 2026 acceptance scope and locked identities at [archived commit `55efdef5`](https://github.com/ZhengDeL/WorldEngine/tree/55efdef54139c63ab9afce9178ebb3bd130717ef). For the current checkout, use [optimized inference recovery](algengine_ascend.md#recovering-optimized-inference) or the [filesystem entry](algengine_ascend.md#inference-recovery-with-file-collection). Refer here only for the input manifest and mount layout; execution scripts and hashes belong to their corresponding versions.

Schema2 adds a durable per-rank journal to the existing AlgEngine planning
inference path. VADv2 and HydraMDP passed the recorded **FP32, 22-official-anchor,
16-rank** interruption and positive-position recovery checks on Ascend.
The [machine-readable result](../ascend/results/fp32-navtest22-journal-recovery-20261020.json)
records the source identities, criteria and evidence hashes.

Full 12,146-anchor official VADv2 FP32 inference was started separately on
2026-10-20 at 06:54:29 UTC and remains outside completed acceptance in this
record. The measured journal result does not establish full NAVTEST scoring,
training checkpoint recovery or convergence, GPU equivalence or performance.
FP32 is the current validation priority; this guide schedules no other precision.

The following hashes identify the two files locked for that recorded run; they do not assert byte identity with the current checkout:

| Tool | SHA256 |
| --- | --- |
| `ascend/tools/journal_inference.py` | `ad9d6429929305cb9054aab9df78b3ea9425ce7caa9af2de695ac42a350afe46` |
| `ascend/tools/rank_journal.py` | `f92e2fe35f428be30851c272921c450582e5b2c5846297b415385e13f8dfe4c4` |

They reside below `ascend/tools/`, which is outside the original top-level
runtime fingerprint scope. The model/runtime fingerprint remains
`0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922`.
Every journal also binds both additional tool hashes. An unmodified model
fingerprint does not imply an uninstrumented execution.

## Recorded FP32 validation: October 20, 2026

Both original 100% planning configurations used their official published
checkpoints, batch 1, workers 0, one node and 16 logical NPU chips on eight
dual-chip physical cards. No model arithmetic, sampler, padding or collector
formula was changed by the journal. Instrumentation remains explicit and is
bound separately from the original 245-file model/runtime fingerprint.

The 22 anchors are the ordered first 22 entries of the official full NAVTEST
filter. The full annotation and its history records were retained. This is a
deterministic official-data selection, not a separately published standard
mini dataset. Its filter SHA256 is
`446bfb797dbd747ae79673036b14e0659459276275e04c04401624d0d87e2097`.
Input evidence covers the original NAVTEST annotation, PDM cache, 8192-entry
vocabulary and full JPEG ledger; it does not change the processed count of 22.

| Check | VADv2 FP32 | HydraMDP FP32 |
| --- | --- | --- |
| Original baseline, uninterrupted journal, stop1 and resumed journal | Passed | Passed |
| Final 22 ordered result dictionaries against original baseline | Every field/type, ndarray dtype/shape/raw bytes and scalar bits exact | Same |
| Two journal routes: initial and per-step Python/NumPy/Torch CPU/NPU RNG | Exact on all 16 ranks, two positions per rank | Same |
| Previously committed stop1 prefix | Unchanged | Unchanged |
| Fifth attempt from the completed interrupted journal | Common position 1; skip 1 saved position, forward 1, replay 1, new commits 0 per rank | Same |
| Original four-route evidence inventory after the fifth attempt | All 283 recorded file hashes unchanged | Same |

The original sampler has 32 positions for this diagnostic, including ten
padding positions. Final exports contain 22 rows with finite 40×3 trajectories
and the original selected-vocabulary checks. Array and scalar dtypes are
compared exactly to the baseline. This is not a claim that every operation in
the software stack uses one dtype.

The uninstrumented baseline has **no per-step RNG trace**. The four-source RNG
comparison is between the uninterrupted and resumed journals. All actual NPU
and CPU audit containers in these accepted checks stopped with exit 0 and no
OOM, except the deliberately interrupted NPU containers, whose real torchrun
return is 1 and whose controlled host pause return is 3. The adopted VAD outer
supervisors were not reaped by the new sequence; their observed outer return
remains null, rather than being inferred from the container exit.

These are inference-recovery checks. Earlier training-restoration results,
full NAVTRAIN convergence, complete NAVTEST, scoring, GPU parity and throughput
have separate evidence and acceptance criteria. A single full journal run does
not by itself establish a paired full-dataset baseline/resume comparison.

## Known issue in schema1

The earlier schema1 core (`aaee2c7483a2a1edd496a7c4136785fbb8d592b535adb4bdc645747b62306879`)
was rejected during an actual resume before model forward, with
`Original result serialization changed`. NumPy array/scalar dtype sharing can
change during a pickle round-trip while every typed numerical value remains
bit-identical. Comparing reserialized pickle bytes therefore rejected valid
stored values. The original schema1 files and diagnostic evidence are retained;
that resume was not accepted as a numerical or recovery pass.

Schema2 retains the whole stored-payload SHA and additionally verifies a
versioned digest of all field types, dtype descriptors, shapes and raw value bits.
Dictionary order and object-sharing identity are excluded from the value digest.
Unsupported objects and object dtypes are rejected. Existing payloads are never
rewritten on replay. Schema1 journals are explicitly rejected: there is no
implicit migration or reuse of their progress under the new instrumentation.
Use a fresh schema2 journal and retain the old files for diagnosis.

## Supported execution contract

The wrapper requires one node, the same 16 ranks throughout, workers0, batch1,
the original unshuffled DistributedSampler, and NAVFormer vocabulary planning
with `process_perception=False` and TrajScoringHead/TrajScoringHeadRL. Each anchor
still uses its original current frame and three history frames from the full
annotation. Other temporal models, perception inference, generated trajectories,
different rank counts and different precision contracts are outside this tool.

Full NAVTEST has 12,146 anchors. The original sampler gives each rank 760
positions, including 14 repeated padding positions globally. The original
MMDetection CPU collector interleaves 12,160 positions and truncates the padding.
The wrapper preserves that order. Do not create independent token shards or
remove history records from the annotation to resume a run.

Required image ID:
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`.
The host supervisor must inspect the real Docker image ID before launching;
`WORLDENGINE_IMAGE_ID` alone is an environment label. The installed
`multi_gpu_test` and `collect_results_cpu` function sources must match,
respectively, `1c17038cc42177a369516d2ca6520d5716c061b83c854e54b577d6d8301c82a7`
and `b1bd40c200c4629cee65b59cc38eaa47aa459ff9a269b948052b8755e57bbe98`.

The deployment owner must supply this fixed Linux ARM64 image, for example
as an offline Docker archive together with a trusted archive SHA256. This
repository provides no public pull address or image archive. The commands below
check a supplied archive and its loaded image; delivery and save/load validation
of such an archive are not claimed by this result.

```bash
: "${FIXED_IMAGE_ARCHIVE:?Set the operator-supplied archive path}"
: "${FIXED_IMAGE_ARCHIVE_SHA256:?Set the trusted archive SHA256}"
printf '%s  %s\n' "$FIXED_IMAGE_ARCHIVE_SHA256" "$FIXED_IMAGE_ARCHIVE" | sha256sum -c - &&
  docker load --input "$FIXED_IMAGE_ARCHIVE" &&
  test "$(docker image inspect --format '{{.Id}}' sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48)" = sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48 &&
  test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48)" = linux/arm64
```

A rebuilt recipe image needs a separate validation. Setting the environment
label does not make a different image satisfy this wrapper's fixed-image gate.

## Inputs and mounts

Select the official checkpoint for the desired model, or an explicitly accepted
training checkpoint. Record its whole-file SHA256. Use its corresponding dumped
inference configuration, with actual annotation, PDM, vocabulary, camera and map
paths. Do not substitute a checkpoint solely because its filename matches.
The wrapper requires whole config/checkpoint SHA256 on the CLI and locks them
for every subsequent attempt.

An already accepted input-QC report is also required. The parent supervisor must
validate the report's scope, source, JPEG ledger and complete anchor coverage;
the wrapper verifies its file identity and cannot determine whether an arbitrary
JSON is an acceptable QC result. A 22-anchor diagnostic needs its matching filter
and input evidence. A locked full-set QC report may cover that subset's input
files; it does not turn 22 processed anchors into 12,146 processed anchors.

Use stable absolute container paths across all attempts. A typical mount layout
is shown below; adapt the host paths to the actual approved files.

| Host input/output | Container path | Access |
| --- | --- | --- |
| Frozen WorldEngine checkout, including these tools | `/workspace/WorldEngine` | read only |
| Full AlgEngine data root | `/data/full` | read only |
| Selected checkpoint | `/weights/initial.pth` | read only |
| Original vocabulary | `/weights/vocab.npy` | read only |
| Dumped inference config | `/config/inference.py` | read only |
| Locked input manifest and approved QC report | `/proof` | read only |
| Maps, if selected by the config | `/maps` | read only |
| Persistent journal for this one case | `/journal` | read/write |
| Fresh directory for this attempt | `/output` | read/write |

The config must resolve to these same mounted files. Additional absolute paths
in a dumped config must be mounted at those exact paths. Check this before model
startup. Maps must also agree with the config or `NUPLAN_MAPS_ROOT`. The wrapper checks the actual constructed dataset's PDM path, rather
than assuming which default path a config will select.

Expose the 16 approved host NPU chip nodes plus the required Ascend control nodes
using the same device contract as the validated environment. Mount the driver
and firmware read only. Set `ASCEND_RT_VISIBLE_DEVICES=0,1,...,15` to the container's
logical chip indices; do not substitute host physical-card numbering. Record the
ordered host-to-container mapping in the supervisor's launch receipt. Eight
dual-chip physical cards provide these 16 logical devices.

`WORLDENGINE_HOST_DEVICE_IDS` records the real host chip IDs in mount order.
The command below assumes host chips 0 through 15 are mounted in that order.
For another approved mapping, set this variable to that actual ordered list;
the container's logical indices remain a separate setting. Do not derive the
host mapping solely from `ASCEND_RT_VISIBLE_DEVICES`.

Create `/proof/locked_inputs.json` with exactly five keys: `annotation`, `pdm`,
`vocabulary`, `filter`, `input_qc`. Each value is
`{"path":"/absolute/container/file","bytes":123,"sha256":"64 lowercase hex characters"}`.
Populate sizes and whole SHA256 from the actual accepted files, not this
illustrative value. The annotation, vocabulary and filter paths must agree with
the selected config; the actual dataset PDM path must agree with `pdm`.

Use the original NAVTEST YAML for full inference. For the diagnostic, explicitly
select the official-data 22-anchor filter and retain its SHA and data provenance.
The input manifest itself needs a whole SHA256. Rank0 checks all five input files
before and after inference.

## Container command

Run this command inside an approved, isolated container supervised by an owner
that enforces the absolute deadline and only stops its recorded container ID.
The wrapper does not create Docker containers or clean up unrelated processes.
A timeout applied only to the Docker client does not guarantee that the
container stops. Host supervisor integration remains a deployment requirement.

```bash
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/opt/conda/envs/bevformer/bin:/opt/conda/bin:$PATH
export NAVSIM_DEVKIT_ROOT=/workspace/WorldEngine/ascend/third_party/navsim
export PYTHONPATH=/workspace/WorldEngine:/workspace/WorldEngine/projects/AlgEngine:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
export WORLDENGINE_IMAGE_ID=sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
export WORLDENGINE_HOST_DEVICE_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_WORK_PATH=/output/ascend_work ASCEND_CACHE_PATH=/output/ascend_cache

# Set these from the approved input manifest/launch plan. The deadline must be
# a future authorized UTC timestamp, and ATTEMPT_ID a new 32-character UUID hex.
: "${CONFIG_SHA:?}" "${CHECKPOINT_SHA:?}" "${INPUT_LOCK_SHA:?}"
: "${ATTEMPT_ID:?}" "${DEADLINE_UTC:?}"
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/tools/journal_inference.py \
  --source-root /workspace/WorldEngine \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --locked-inputs /proof/locked_inputs.json \
  --expected-locked-inputs-sha256 "$INPUT_LOCK_SHA" \
  --precision fp32 --msda-backend cuda-compatible \
  --expected-anchors 12146 --journal-root /journal --journal-mode fresh \
  --output /output/result --attempt-id "$ATTEMPT_ID" \
  --deadline-utc "$DEADLINE_UTC" --stop-file /journal/STOP
```

The measured result in this guide is FP32. Other precisions require their own
explicit validation and are deferred in the current work order. Do not change
precision, model, source or image within an existing journal.

The example preserves the original default `deterministic=False`. An explicit
`--deterministic` run additionally requires `HCCL_DETERMINISTIC=true` and
`CLOSE_MATMUL_K_SHIFT=1` before Torch starts. That is a separate contract requiring
its own validation; do not mix the two settings within one journal.

`/output` is the host-mounted attempt directory; `/output/result` must not exist.
Only rank0 creates that child and publishes its UUID/contract claim. Other ranks
wait for the same claim. Generate a new UUID on every attempt, for example with
`python -c 'import uuid; print(uuid.uuid4().hex)'`.

## Stop and resume

A successful step commit contains the original result, post-forward Python,
NumPy, Torch CPU and NPU RNG, and a receipt hash chain. Writes are staged, fsynced
and atomically renamed as a nonempty directory. An interrupted staging directory
is retained but excluded from the committed prefix. Journal pickle files are
trusted output from this job; do not accept arbitrary downloaded pickle journals.

Touching `/journal/STOP` requests a cooperative partial stop. A native forward
may still be executing, so the independent host deadline/watchdog remains
necessary. Confirm the previous owned process group/container has stopped before
resuming. Preserve the journal and old attempt output. Remove only the STOP file
created for this case once continuation is authorized.

To resume, mount the same persistent journal at `/journal`, mount a different
fresh attempt directory at `/output`, choose a new `ATTEMPT_ID`, and change only
`--journal-mode fresh` to `--journal-mode resume`. Preserve all source, image,
input, precision, MSDA, deterministic and world-size identities.

All ranks agree on `R=max(0,min(committed_prefixes)-1)`. They restore RNG at the
same boundary and replay it. All result values, including ndarray/scalar raw bits,
and normalized four-source RNG must match exactly; a mismatch preserves the
previous stored payload bytes and stops.
DataLoader seed consumption is accounted for before restoring that boundary.
Earlier saved results are prepended before the original collector runs.

Worker exit3 means partial progress, not full inference. Torchrun generally
returns exit1 when a worker exits3; preserve both statuses. An arbitrary exit1
does not prove a controlled stop. The fixed-step diagnostic writes all16 partial
receipts before a barrier and worker exit. Deadline, signal and STOP paths avoid
that barrier and may retain valid journals without all16 partial reports.

## Acceptance and output files

The two recorded FP32 profiles passed the following procedure, followed by
the fifth-attempt check below. Reproduce it with the same official 22-anchor
selection, 16 ranks and locked checkpoint/source/image. New inputs or precision
contracts need new evidence:

Prepare the 22-token filter by taking the original ordered first 22 tokens
from the full official YAML, and change only the test filter in the dumped
configuration. Retain all annotation/history/image/PDM/vocabulary records. The
original baseline has no `--expected-anchors` option: inspect its validation
report and require exactly 22 unique anchors.

Inside the supervised baseline container, with the same setup and mounts as
above, use the existing original entry:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/inference.py \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --output /output/result --precision fp32 --msda-backend cuda-compatible
```

Use a new attempt output for every row below. Reuse the journal only where the
table says so, and apply the wrapper command above with `--expected-anchors 22`.

| Execution | Journal | Wrapper arguments |
| --- | --- | --- |
| Original baseline | None | Original entry shown above |
| Uninterrupted journal | New journal A | `--journal-mode fresh --stop-after-steps 0` |
| Controlled pause | New journal B | `--journal-mode fresh --stop-after-steps 1` |
| Resume after the pause | Same journal B | `--journal-mode resume --stop-after-steps 0` |
| Positive-position check | Same now-complete journal B | `--journal-mode resume --stop-after-steps 0`, fresh output and UUID again |

The measured host supervisor and CPU measurement auditors are identified by
SHA in the result JSON; they are not shipped as new public executables in this
runtime patch. These are in-container runtime commands, not a substitute for
owned-container supervision or the independent CPU comparisons below. Use a
reviewed validator that implements all criteria; the synthetic CPU unit suite
alone does not certify a new device run.

1. Complete the original uninstrumented `ascend/inference.py` run as baseline.
2. Complete a separate fresh uninterrupted journal run with
   `--expected-anchors 22`, using its own persistent journal and output.
3. Start another fresh journal with `--expected-anchors 22 --stop-after-steps 1`.
4. Resume the interrupted journal into a new output without `--stop-after-steps`.
5. Check all16 rank receipts, exact boundary-result/RNG replay, both padded
   positions per rank, and all22 ordered final result dictionaries against the
   baseline, for both completed journal routes. Compare both journals' complete
   initial and per-step RNG/result values, and prove the paused prefix receipts
   were not overwritten. The uninstrumented baseline has no per-step RNG journal;
   do not claim it does. Compare ndarray shape/dtype/raw bytes and every scalar,
   metric and chosen vocabulary index. Whole pickle SHA alone is insufficient
   because pickle memoization can differ after restoring individual rows.

### Positive resume position

The four runs above are necessary but do not prove that completed sampler
positions were skipped. With 22 anchors and 16 ranks, each rank has two padded
positions. Stopping after one commit selects `R=0` on resume and executes both
positions again.

After the four-run comparison passes, retain the now-complete interrupted
journal and run a fifth attempt with the same 22-anchor config and precision.
Use another fresh `/output` mount and UUID; preserve the source, checkpoint,
input hashes, mapping and `/journal` mount. For FP32, inside the same approved
environment:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/tools/journal_inference.py \
  --source-root /workspace/WorldEngine \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --locked-inputs /proof/locked_inputs.json \
  --expected-locked-inputs-sha256 "$INPUT_LOCK_SHA" \
  --precision fp32 --msda-backend cuda-compatible \
  --expected-anchors 22 --journal-root /journal --journal-mode resume \
  --output /output/result --attempt-id "$ATTEMPT_ID" \
  --deadline-utc "$DEADLINE_UTC" --stop-file /journal/STOP
```

All 16 reports in `result/journal_execution.json` must show
`committed_before=committed_after=2`, `common_resume_step=1`,
`forward_steps_this_attempt=1`, `replayed_exact_steps=1`,
`new_committed_steps=0` and `all_rank_prefixes=[2]*16`. This proves that position
0 was supplied from the saved prefix and only position 1 was replayed. Compare
all 22 ordered outputs against the original baseline with the same exact
field/type/dtype/raw-byte checks. Freeze the original four-run audit inventory
before this attempt and verify every listed file SHA is unchanged afterwards;
new per-attempt receipts may be added, existing evidence must not be rewritten.
Also verify a distinct actual container ID, exit0, no OOM, stopped state and
successful exact-owned watchdog cleanup. Apply this check separately to each
model/precision profile. Both recorded FP32 models passed this measured NPU
check; that result does not transfer to an untested profile.

This diagnostic does not establish full NAVTEST coverage, official scores,
convergence, GPU agreement, performance or long-duration stability.

| Artifact | Meaning |
| --- | --- |
| `result/journal_partial_rank*.json` | Partial condition and immutable contract; no full result |
| `/journal/rank_*/steps/*/{payload.pkl,receipt.json}` | Committed rank results/RNG and hash chain |
| `/journal/rank_*/attempts/*.json` | Per-attempt committed/replayed counters and outcome |
| `result/original_inference_validation.json` | Exact original validator report before enrichment |
| `result/inference_validation.json` | Original report plus explicit instrumentation identity |
| `result/journal_execution.json` | Written only after complete original configured inference and all-rank validation |

Final acceptance still requires the original complete configured token coverage,
finite 40×3 trajectories, vocabulary checks and the complete original collector.
Official NAVSIM scoring remains a separate step that requires the complete
official metric cache. A journal or a completed partial attempt does not replace
those checks.

## CPU tests and upstream fixtures

The repository does not distribute MMDetection function-source fixture files.
Generate them locally from the exact installed image files using the standard
library extractor below. It statically parses the explicit `mmdet/apis/test.py`
file; it does not import or execute Torch, torch_npu, MMDetection or DrivingSDK.
The extractor therefore needs no NPU devices, driver/HAL libraries or CANN
initialization. Actual model inference still requires the full Ascend stack.

Do not replace this operation with `from mmdet.apis import multi_gpu_test` in a
CPU-only container. That import can reach DrivingSDK's device-name lookup and
initialize the NPU runtime. Supplying driver/HAL libraries alone does not make
that dynamic import a valid CPU-only fixture generator.

The frozen image's installed MMDetection version was statically measured as
**2.24.0**. Its approved whole-file identities are:

| Installed file | SHA256 |
| --- | --- |
| `mmdet/apis/test.py` | `33cb4ea46fc0a55d2571be7a47cb8486792c0775c4ab2695dbc508fbd6f62bdd` |
| `mmdet/version.py` | `2c12507eab845d59ca0b7d6c404189ddb12c52fb2c9e26547f898925ecffad09` |

These values were read from the actual frozen image, not inferred from API
similarity. They are separate from the already locked function-source hashes.
The original live-image API inspection supplied those function hashes; static
extraction is not a second dynamic API acceptance test.

Run the following commands from the checkout root inside the frozen image.

```bash
cd /workspace/WorldEngine
export WORLDENGINE_IMAGE_ID=sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
export PINNED_MMDET_TEST_FILE_SHA=33cb4ea46fc0a55d2571be7a47cb8486792c0775c4ab2695dbc508fbd6f62bdd
export PINNED_MMDET_VERSION_FILE_SHA=2c12507eab845d59ca0b7d6c404189ddb12c52fb2c9e26547f898925ecffad09
python -S ascend/tools/extract_journal_test_fixtures.py \
  --source-file /workspace/bevformer/mmdetection/mmdet/apis/test.py \
  --expected-source-file-sha256 "$PINNED_MMDET_TEST_FILE_SHA" \
  --version-file /workspace/bevformer/mmdetection/mmdet/version.py \
  --expected-version-file-sha256 "$PINNED_MMDET_VERSION_FILE_SHA" \
  --license-file /workspace/bevformer/mmdetection/LICENSE \
  --output /fixtures/new_journal_fixture_set
# Add --notice-file /actual/installed/NOTICE if the distribution has one.

# The NumPy CPU tests run normally, without accelerator device mounts.
export WORLDENGINE_JOURNAL_FIXTURES=/fixtures/new_journal_fixture_set
# Optional approved real-result fixture; never use arbitrary downloaded pickle.
# export JOURNAL_ACTUAL_SAMPLE_ROOT=/fixtures/approved_original_rank0_step0
python -m unittest discover -s ascend/tools/tests -p 'test_*journal*.py' -v
```

The helper validates the whole installed source/version files and both exact
function-source SHA256 values before writing a fresh fixture directory. It uses
AST to locate original top-level functions and the standard-library source-block
reader to preserve the same LF function representation as live inspect.getsource.
It statically reads a literal `__version__`, retains the actual LICENSE and
optional NOTICE byte-for-byte, and records original source/file hashes and
provenance. The project and reference license are published by
[OpenMMLab](https://github.com/open-mmlab/mmdetection), under
[Apache-2.0](https://github.com/open-mmlab/mmdetection/blob/v2.24.0/LICENSE).
The reference tag matches the measured installed version; whole-file and function
hashes remain the compatibility gates.

If `WORLDENGINE_JOURNAL_FIXTURES` is unset, the two installed-interface integration
tests report explicit skips. Five additional real-result regression tests require
`JOURNAL_ACTUAL_SAMPLE_ROOT`, containing the approved original `payload.pkl`,
`receipt.json` and `header.json`; this sample is not distributed with runtime
source. The payload SHA is fixed at
`7eeb89723c34deae06c04c63261404d4c80d91894bff6f4cadf8b1a2eeb51cde`.
These five tests explicitly skip when that input is absent and reject a supplied
wrong input. Remaining NumPy/stdlib tests can run without Torch.
An invalid supplied fixture path or mismatched function SHA is an error. A test
run containing these skips is not the complete CPU fixture suite. With both
approved fixture sets, the public test set contains 27 tests: 13 loop/journal,
10 typed-value/real-result and four static-extractor tests. CPU synthetic
fixtures and static source extraction do not establish NPU equivalence.

The static extractor (`a56545393e86388b9bf84c28bb2a7c42ef0a1f2458544b86d47f781a6f207001`) and the then-current 17 CPU tests
passed in the frozen Linux image, without device nodes or driver mounts. The
operator-held execution receipt `static_public_journal_CPU_v3_actual.json` has
SHA256 `62734dd4701cbe31965f3cd6e0faece56de20f836ce0ee6312ac9225e94730bd` and records extraction, all17 tests with no skips,
container exit0 and no OOM. This CPU result does not accept interrupted NPU
inference, full NAVTEST or official scores. The raw receipt stays with the local
validation evidence because it includes deployment-specific paths.

That static-extraction run used the earlier schema1 journal core. It verifies the
unchanged extractor/fixture path, not the schema2 repair.

The schema2 core `f92e2fe35f428be30851c272921c450582e5b2c5846297b415385e13f8dfe4c4`
separately passed 23 Linux CPU tests, with no skips: the 13 original loop/journal
tests and 10 new typed-value regressions, including reopening and replaying the
preserved real-result sample in another process. The receipt
`journal_v2_core_CPU_actual.json` has SHA256
`884f6015b3da289c5cf6dfa0c69ffef70541dfbf9c1d8ac80646414776c0ba2b`.
This is CPU regression evidence. The later FP32 NPU measurements are recorded
above; complete NAVTEST inference and official scoring remain unaccepted here.

The complete public schema2 candidate was then tested together in the fixed
Linux image: static installed-file extraction followed by all 27 public CPU
tests passed with no skips. The container used `runc`, without accelerator
devices or driver mounts, exited0 and was not OOM-killed. The receipt
`static_public_journal_CPU_v4_actual.json` has SHA256
`1cd9abe0356b6ea8530d1aedbcc3861af4fba982548604a9e324dc7946cd6c32`.
This accepts the public CPU reproduction path only. An earlier device attempt
stopped at the resource gate before inference-container creation; it is retained
separately. The later accepted FP32 four-route and positive-position checks
above used fresh attempts. CPU tests alone are not those NPU results.
