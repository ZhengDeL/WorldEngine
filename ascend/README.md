# AlgEngine on Ascend 910C / A3

[中文](README_zh.md) · [Installation and usage](../docs/algengine_ascend.md)

MSDA defaults to `reference`; use `--msda-backend cuda-compatible` for the optional [CUDA-compatible route](MSDA_CUDA.md).

The `ascend-910c-algengine` branch provides NPU entry points for WorldEngine
AlgEngine training, fine-tuning, checkpoint restoration, inference and
evaluation. SimEngine retains its upstream GPU simulation and rendering path.

Current NPU validation prioritizes FP32. Historical other-precision results
remain versioned below; they are not extended by the new inference-journal
result.

## Getting started

Run the public workflow on a Linux ARM64 Ascend server:

1. Build the [public post8 parent](docker/README.md), then the
   [application/compiler dependency environment](docker/dependency_repair/README.md).
   Run its three CPU checks and retain the actual image ID.
2. Prepare the [NAVTRAIN139 / NAVTEST22 validation subset](data_tools/navtrain139/README.md)
   from six hash-locked original publisher files. Reuse server caches where available.
3. Download the [official planning initializers and vocabulary](../docs/algengine_ascend.md#2-data-and-official-weights).
4. Follow the [training, resume, inference and scoring workflow](../docs/algengine_ascend.md#recommended-workflow-navtrain139--navtest22).
   FP32 is the default. BF16 is opt-in and experimental; its measured coverage
   repair and remaining replay differences are documented below.

The 139/22 subset is deterministically derived from official NAVTRAIN/NAVTEST
data; it is not a separately published standard mini. It retains the original
labels, 8192 candidate order and camera bytes. Preparation requires six original
files totaling 47.91 GB if none are cached; the prepared subset uses 1832 JPEGs.
The complete 5 TB publication is not needed for this workflow.

## Documentation

| Document | Contents |
| --- | --- |
| [Usage guide](../docs/algengine_ascend.md) / [中文](../docs/algengine_ascend_zh.md) | Full commands, original configs, real-data preparation, training, resume, inference, scoring and fine-tuning |
| [Per-file code review](FILE_AUDIT.md) / [中文](FILE_AUDIT_zh.md) | Reviewed files, corrective changes, CPU regressions and verification boundaries |
| [Dependency environment](docker/dependency_repair/README.md) | Public Docker recipe, application/tool interpreter separation, wheel locks and CPU acceptance |
| [Parent-image recipe](docker/README.md) | Official SDK base, post8 parent build and image transfer |
| [NAVTRAIN139 / NAVTEST22 preparation](data_tools/navtrain139/README.md) | Exact original inputs, deterministic extraction and measured CPU reproduction |
| [Full NAVTEST CPU input audit](../docs/navtest_input_audit.md) / [中文](../docs/navtest_input_audit_zh.md) | Public CLI, official 32-archive lock, full JPEG decoding and [input/ledger evidence](../docs/navtest_public_input_results.md) ([JSON](results/full-navtest-public-inputs-20261020.json)) |
| [Validation report](FULL_ALGENGINE.md) / [中文](VALIDATION_zh.md) | Versioned model results, numerical controls, resources and remaining acceptance |
| [FP32 MSDA](MSDA.md) | Implementation, original forward/gradient tolerances and reproduction |
| [Recoverable planning inference](../docs/algengine_journal.md) / [中文](../docs/algengine_journal_zh.md) | Per-rank journal, FP32 recovery results on 22 official anchors, five-run reproduction and boundaries |
| [Synchronous replay controls](execution/README.md) | Restricted historical eight-update VAD/Hydra controls |
| [SimScale data tools](data_tools/README.md) | Genuine small-data conversion and two synthetic-config workflows |

## Verification scope

| Area | Recorded result | Limit |
| --- | --- | --- |
| Dependency environment | Public combined image passed application, selected SDK-tool and original-CANN-source CPU checks; both unfiltered `pip check` commands exited 0 | CPU checks do not establish cold-cache NPU model or all optional SDK-tool acceptance |
| NAVTRAIN139 / NAVTEST22 | Public CLI rebuilt the fixed subset from six original inputs; all 1832 JPEGs passed SHA/decode checks; all six derived metadata files matched the prior subset by complete values, order and bytes | One training log and one evaluation log; full NAVTRAIN convergence and full NAVTEST quality remain unverified |
| Full NAVTRAIN inputs, October 20 | CPU coverage, per-file content hashes and decoding passed for [103,288 anchors and all 1,219,960 required JPEGs](results/full-navtrain-inputs-20261020.json); no missing annotations, history images or PDM rows | Input audit. The inspector's 51.99 GB peak RSS is not a training RAM budget; model results are recorded separately below |
| Full-input VADv2 / HydraMDP, FP32 / BF16, October 20 | [All four cases completed eight updates and exact own 4→8 restoration](results/full-navtrain-short8-20261020.json), using four ranks on two physical cards. All 32 rank reports passed; all eight saved model/Adam payloads were finite | The complete loader was available, but each case consumed the same 32 current anchors. No complete epoch, convergence, full-checkpoint GPU comparison or strict replay-equivalence acceptance |
| Single-chip FP32 capacity, October 20 | [Both original 100% imitation planners completed four updates](results/single-chip-capacity-20261020.json) with batch 1, complete NAVTRAIN inputs and clean owned-container termination | Tested allocation: 192g RAM / 16 CPUs. Single-process peak RSS was 50.69 / 50.79 GiB; this is not a minimum RAM/device requirement, BF16 capacity or throughput benchmark |
| FP32 inference journal, October 20 | [VADv2 and HydraMDP passed four-route and fifth positive-position recovery checks on 22 official NAVTEST anchors with 16 ranks](results/fp32-navtest22-journal-recovery-20261020.json). All final fields and two-journal four-source RNG were exact; each fifth attempt skipped one saved position and replayed one per rank | 22 anchors / 32 padded positions / ten padding positions. Baseline has no per-step RNG trace. Full 12,146-anchor official VAD FP32 inference is running separately and is not accepted; no training, GPU or performance conclusion |
| Full NAVTEST inputs, October 20 | [12,146 anchors, 136 logs and all 142,976 required JPEGs](results/full-navtest-inputs-20261020.json) passed independent CPU input, content-hash and full-decode checks | Shared input evidence for six evaluation cases; their complete model inference and official NAVSIM scoring remain separate acceptance items |
| CUDA-compatible MSDA | [Normal MX/MMCV entries and MMCV raw-L4 layout](results/npu-cuda-msda-20261017.json) meet the original CUDA tolerances; separate CPU32 negatives are retained. Fresh CUDA export-to-NPU reproduction also completed | FP32/C32, measured shapes and first-order gradients |
| CUDA-compatible VADv2 / HydraMDP, FP32, October 18 | [Sixteen-rank same-prefix two-update control](results/cuda-compatible-fp32-strict16-resume2-20261018.json): model / all tensor pairs 889 / 1698 and 909 / 1778, all `torch.equal`, maximum tensor and loss differences 0 | Same continuous prefix, 32 actual current anchors; no independent-initialization, longer-run or quality claim |
| CUDA-compatible VADv2 / HydraMDP, FP32 / BF16 | [Two updates, exact own-state restoration, DDP4 inference and official scoring](results/cuda-compatible-navtrain139-short2-20261017.json), four ranks per group | Eight actual training anchors; the same 22 evaluation anchors across all groups. No convergence, quality-equivalence or 16-rank strict-replay claim |
| FP32 reference — October 16 | Both original 100% planning configurations completed [eight-update training, own-checkpoint restoration and 22-token inference/scoring](results/fp32-navtrain139-short8-20261016.json); historical fine-tuning is versioned separately | Both recorded strict replay comparisons completed outside original tolerance. Restoration and score measurements do not establish convergence or quality improvement |
| BF16 v7 reference — October 16 | Both official 100% planners completed [eight updates, own restoration and 22-token inference/scoring](results/bf16-v7-navtrain139-short8-20261016.json); [all 16 ranks and saved checkpoints verify 138 gradient/Adam entries restored per model](results/bf16-v7-gradient-coverage-20261016.json) | Both complete strict comparisons remain outside original tolerance. Opt-in/experimental; no convergence or quality-equivalence acceptance. Defective v6 records are retained separately |
| FP32/BF16 output comparison | [Four complete 880-pose comparisons](results/fp32-bf16-v7-matched8-planning-20261016.json) on the same 22 held-out anchors, using same official weights and each own eight-update checkpoint | Measurements without a quality-equivalence threshold; adapted checkpoints include training differences. Not GPU/NPU parity |
| Rare-log FT, BF16 v7 | [Both 37/22 real-log configurations completed 8 updates and own 4→8 restoration](results/real-log-ft37-bf16-v7-short8-20261016.json); VAD/Hydra replay tensors were elementwise equal (`torch.equal`), and losses matched | Both checkpoints completed independent 22-anchor inference and official NAVSIM scoring; [Hydra follow-up](results/hydra-ft37-bf16-eval22-20261017.json): PDMS 0.933856. Historical interrupted output is retained. This is not NAVTRAIN139, synthetic coverage or a quality-equivalence result |
| Synthetic inputs | Historical two SimScale entries have one-scene, explicitly derived-WE-reward CPU and one/two-rank results | Six WorldEngine rollout/replay/BWM exports and complete published-protocol acceptance remain open; real-log 37/22 FT does not count as synthetic coverage |

The repaired image measured on 2026-10-16 is
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`.
Its [build record](docker/dependency_repair/BUILD_VALIDATION.json) and the
[data reproduction record](data_tools/navtrain139/VALIDATION.json) identify the
actual inputs and scope. A rebuilt image or newly prepared dataset requires its
own validation; matching filenames do not transfer checkpoint compatibility.

## Small real data and official weights

Use the [fixed official-data workflow](../docs/algengine_ascend.md#recommended-workflow-navtrain139--navtest22)
for NAVTRAIN training anchors and independent NAVTEST evaluation. The guide also
retains an [alternative NAVTEST-only smoke split](../docs/algengine_ascend.md#3-prepare-a-small-real-dataset-and-run-eight-updates),
the original full-data path and rare-log fine-tuning. Official published weights
initialize a new run; they are not migration resume checkpoints.

## Run, save and resume

For inference progress, use the [schema2 journal guide](../docs/algengine_journal.md).
Its per-rank result/RNG journal is distinct from model/optimizer training
checkpoints. The measured FP32 scope is the two original planners, 22 official
anchors and 16 ranks; complete inference and official scoring retain their own
coverage and cache gates.

The [full-input short-training workflow](../docs/algengine_ascend.md#full-input-short-training) reproduces the October 20 four-rank parameters with fresh outputs. Both `e2e_*_100pct` cases are imitation training; this measurement does not enable HydraMDP policy-gradient fine-tuning. The server has eight dual-chip physical NPU cards and 16 chips; this matrix used four chips on two cards. Source `0488bfb7…` / 245 Python files and the image ID above identify the measured runtime.

The [CUDA-compatible guide](MSDA_CUDA.md#training-own-checkpoint-resume-and-inference) provides the measured four-rank/two-update chain for both models and precisions, including independent official scoring.

The [usage guide](../docs/algengine_ascend.md#recommended-workflow-navtrain139--navtest22)
provides FP32/BF16 selection and two separate horizons: a short eight-update
chain with its own update-4 interruption, and native eight-epoch training with
an epoch-4 interruption. Both continue to own-checkpoint restoration,
independent inference and official CPU scoring.
Resume requires the same source, image, precision, data/filter, world size,
batch and schedule contract, with workers 0 and a fresh output directory.
The [original comparator](../docs/algengine_ascend.md#5-save-and-resume-complete-training-state)
retains its tensor and loss tolerances.

The [October 18 FP32 control](results/cuda-compatible-fp32-strict16-resume2-20261018.json) additionally verified VADv2 and HydraMDP with 16 ranks × batch 1: continuous updates 0→2, then a fresh process resuming that same run’s update-1 checkpoint to update 2. All compared tensors satisfy `torch.equal`; maximum tensor and loss differences are zero. This same-prefix control uses 32 distinct current anchors from the configured 139 and adds no inference or scoring result. See the [source and checkpoint boundary](MSDA_CUDA.md#source-and-checkpoint-version-contract).

## Historical controls and unresolved work

Earlier 16-rank base-planner synchronous controls on the `cc91…`/243-file
source and post8/3fef image completed four repeated pairs: two were within the
original tolerance and two were outside it. Default-execution controls also include completed comparisons outside
the original tolerance. These settings are not an accepted general numerical repair.
The 2026-10-14 public-image LoRA results are
[recorded separately](results/actual-public-reproduction-16chip-20261014.json);
the 16 dependency findings of that earlier image are not the repaired image's
CPU result.

The eight-hour perception run belongs to the earlier `858f9d…`/post6
environment. It does not establish sustained training in the current image.
A separate [official100 GPU/NPU FP32 comparison](results/gpu-npu-official100-eval22-20261016.json)
matched both models' selected trajectories and all seven per-token official-score
metrics on the same 22 anchors.
The [original CUDA baseline](CUDA_BASELINE.md) records full MSDA gradients and
eight-GPU short training/resume controls, including completed comparisons outside
the unchanged numerical gates. It does not resolve NPU training reproducibility.
Complete-epoch NAVTRAIN training and convergence, full independent NAVTEST,
paired GPU/NPU training validation, broader cross-precision quality validation, repeated numerical reproducibility and multi-node scaling
remain separate acceptance items. See the [validation report](FULL_ALGENGINE.md).

Report issues with the source commit/fingerprint, image ID, config, precision,
logical-chip mapping, dataset split, command, `launch.json`, rank runtime files
and comparison reports. Remove credentials and private data paths.
