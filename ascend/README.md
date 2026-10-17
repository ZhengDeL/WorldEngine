# AlgEngine on Ascend 910C / A3

[中文](README_zh.md) · [Installation and usage](../docs/algengine_ascend.md)

The `ascend-910c-algengine` branch provides NPU entry points for WorldEngine
AlgEngine training, fine-tuning, checkpoint restoration, inference and
evaluation. SimEngine retains its upstream GPU simulation and rendering path.

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
| [Dependency environment](docker/dependency_repair/README.md) | Public Docker recipe, application/tool interpreter separation, wheel locks and CPU acceptance |
| [Parent-image recipe](docker/README.md) | Official SDK base, post8 parent build and image transfer |
| [NAVTRAIN139 / NAVTEST22 preparation](data_tools/navtrain139/README.md) | Exact original inputs, deterministic extraction and measured CPU reproduction |
| [Validation report](FULL_ALGENGINE.md) / [中文](VALIDATION_zh.md) | Versioned model results, numerical controls, resources and remaining acceptance |
| [FP32 MSDA](MSDA.md) | Implementation, original forward/gradient tolerances and reproduction |
| [Synchronous replay controls](execution/README.md) | Restricted historical eight-update VAD/Hydra controls |
| [SimScale data tools](data_tools/README.md) | Genuine small-data conversion and two synthetic-config workflows |

## Verification scope

| Area | Recorded result | Limit |
| --- | --- | --- |
| Dependency environment | Public combined image passed application, selected SDK-tool and original-CANN-source CPU checks; both unfiltered `pip check` commands exited 0 | CPU checks do not establish cold-cache NPU model or all optional SDK-tool acceptance |
| NAVTRAIN139 / NAVTEST22 | Public CLI rebuilt the fixed subset from six original inputs; all 1832 JPEGs passed SHA/decode checks; all six derived metadata files matched the prior subset by complete values, order and bytes | One training log and one evaluation log; full NAVTRAIN convergence and full NAVTEST quality remain unverified |
| FP32 training and fine-tuning | Both original 100% planning configurations completed [eight-update training, own-checkpoint restoration and 22-token inference/scoring](results/fp32-navtrain139-short8-20261016.json); historical fine-tuning is versioned separately | Both current strict replay comparisons completed outside original tolerance. Restoration and score measurements do not establish convergence or quality improvement |
| BF16 v7 | Both official 100% planners completed [eight updates, own restoration and 22-token inference/scoring](results/bf16-v7-navtrain139-short8-20261016.json); [all 16 ranks and saved checkpoints verify 138 gradient/Adam entries restored per model](results/bf16-v7-gradient-coverage-20261016.json) | Both complete strict comparisons remain outside original tolerance. Opt-in/experimental; no convergence or quality-equivalence acceptance. Defective v6 records are retained separately |
| FP32/BF16 output comparison | [Four complete 880-pose comparisons](results/fp32-bf16-v7-matched8-planning-20261016.json) on the same 22 held-out anchors, using same official weights and each own eight-update checkpoint | Measurements without a quality-equivalence threshold; adapted checkpoints include training differences. Not GPU/NPU parity |
| Rare-log FT, BF16 v7 | [Both 37/22 real-log configurations completed 8 updates and own 4→8 restoration](results/real-log-ft37-bf16-v7-short8-20261016.json); VAD/Hydra replay tensors and losses were bitwise equal | VAD completed independent 22-anchor scoring; Hydra inference stopped at progress 14/22 after a bounded timeout and has no FT score. This is not NAVTRAIN139 or synthetic data |
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

The [usage guide](../docs/algengine_ascend.md#recommended-workflow-navtrain139--navtest22)
provides FP32/BF16 selection and two separate horizons: a short eight-update
chain with its own update-4 interruption, and native eight-epoch training with
an epoch-4 interruption. Both continue to own-checkpoint restoration,
independent inference and official CPU scoring.
Resume requires the same source, image, precision, data/filter, world size,
batch and schedule contract, with workers 0 and a fresh output directory.
The [original comparator](../docs/algengine_ascend.md#5-save-and-resume-complete-training-state)
retains its tensor and loss tolerances.

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
Complete NAVTRAIN image training/convergence, full independent NAVTEST,
paired GPU/NPU training validation, broader cross-precision quality validation, repeated numerical reproducibility and multi-node scaling
remain separate acceptance items. See the [validation report](FULL_ALGENGINE.md).

Report issues with the source commit/fingerprint, image ID, config, precision,
logical-chip mapping, dataset split, command, `launch.json`, rank runtime files
and comparison reports. Remove credentials and private data paths.
