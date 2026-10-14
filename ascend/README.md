# AlgEngine on Ascend 910C / A3

[中文](README_zh.md) · [English usage guide](../docs/algengine_ascend.md) · [中文使用指南](../docs/algengine_ascend_zh.md)

The `ascend-910c-algengine` branch adds Ascend NPU execution for WorldEngine's
AlgEngine training, fine-tuning and evaluation workflows. SimEngine retains its
upstream GPU workflow. Validation covers the specific configurations, datasets
and environments recorded below; complete AlgEngine migration acceptance remains open.

## Documentation

| Document | Contents |
| --- | --- |
| [Installation and usage](../docs/algengine_ascend.md) | Checkout, assets, real-data preparation, training, resume, evaluation, rare-case selection and fine-tuning commands |
| [Container environment](docker/README.md) | Public ARM64 build recipe, version checks, image identity and offline transfer |
| [Validation report](FULL_ALGENGINE.md) / [中文验证报告](VALIDATION_zh.md) | Test configurations, measured results, resource requirements and unresolved issues |
| [FP32 MSDA implementation](MSDA.md) | Strict forward/gradient checks, implementation constraints and reproduction commands |
| [Synchronous replay protocol](execution/README.md) | Restricted eight-update VAD/Hydra controls and original numerical comparison |
| [SimScale data preparation](data_tools/README.md) | Three preparation CLIs, public input locks and genuine small-data reproduction |

## Validation scope

| Workflow | Measured coverage | Remaining validation |
| --- | --- | --- |
| Base-planner training and resume | Four repeated 16-rank synchronous pairs completed; two passed the original tensor gate | Resolve intermittent numerical divergence; sustained training and full NAVTRAIN convergence |
| Fine-tuning | Two real-log LoRA FT configs passed 16-rank replay, inference and scoring on a public-rebuilt image; two genuine SimScale configs passed narrower one/two-rank controls | Six other genuine synthetic exports, full-data sampling and published SimScale-v2 reward equivalence |
| Inference and evaluation | Original planning/perception evaluation and independent official NAVSIM CPU rescoring on small datasets | Full independent NAVTEST quality and GPU/NPU output comparison |
| MSDA | FP32 forward and gradient gates passed at the original tolerances for the recorded shapes/layouts | Other precision modes and higher-order derivatives |
| Data and resources | Full NAVTRAIN metadata/PDM QC and 1/8/16 resident original dataset measurements | Full camera coverage and model/optimizer/worker training memory and I/O |
| Operational stability | An eight-hour perception run, own-checkpoint continuation, inference and CPU audit passed on the earlier `858f9d…` / post6 environment | Sustained training on the final `cc91…` / post8 synchronous profile |

## Public-source reproduction

The parent and post8 child Docker recipes were rebuilt without application-layer
cache from a verified official SDK base. The new image completed two original
rare-log LoRA fine-tuning workflows on a second Ascend server: 16 logical chips,
FP32, B1 per rank, workers 0, eight updates and own checkpoint 4→8. All 64 rank
reports and 32 restorations passed. Both original numerical comparisons were
bitwise equal, and each model completed inference and official CPU scoring on
22 held-out anchors. The public preparation tools selected 37 training
candidate anchors in separate logs. The image's dependency metadata still
reports 16 `pip check` findings.
[Reproduction evidence](results/actual-public-reproduction-16chip-20261014.json).

## Base-planner replay status

Four independent controls executed the published synchronous adapter on the
recorded cc91/243 source and post8/3fef image, using the historical 42/22 mini.
All training and restoration runs completed; two final comparisons were within
the original tolerance and two were outside it. Default-execution controls
also retain above-tolerance differences. The numerical cause remains unresolved;
synchronous settings are not an accepted general repair.
[Repeated controls](results/public-sync-repeated-16chip-20261014.json) and
[full validation report](FULL_ALGENGINE.md).

## Source and runtime

Use the [container installation guide](docker/README.md) to build or import the
environment and record its actual image ID. A rebuilt image requires its own
validation; historical measurements do not transfer to a different image.

## Small real data and official weights

The [asset instructions](../docs/algengine_ascend.md#2-data-and-official-weights)
and [real-mini preparation workflow](../docs/algengine_ascend.md#3-prepare-a-small-real-dataset-and-run-eight-updates)
select AlgEngine inputs without downloading the entire scene publication. Real
camera files, original history queues and genuine PDM labels are required.

## Run, save and resume

Follow the full commands for [training](../docs/algengine_ascend.md#4-train-on-the-original-50-or-100-navtrain-split),
[complete-state resume and CPU comparison](../docs/algengine_ascend.md#5-save-and-resume-complete-training-state),
and [fine-tuning](../docs/algengine_ascend.md#8-fine-tuning-and-original-configuration-variants).
Published weights initialize a new run; complete resume uses that run's own
checkpoint, compatible inputs and a fresh output directory.

## Inference and official scoring

Use the dumped training configuration and the [evaluation commands](../docs/algengine_ascend.md#6-evaluation-and-official-scoring).
Official CPU rescoring is separate from embedded PDM labels. The
[rare-case workflow](../docs/algengine_ascend.md#7-evaluate-navtrain-and-select-rare-cases)
uses actual per-token results from the selected training split.

## Sustained run and equipment

Device topology, full-PKL resident memory and the versioned eight-hour experiment
are documented in the [validation report](FULL_ALGENGINE.md). Full training
capacity and duration must be measured with the target images, model, workers
and environment; metadata file size is not a training-memory estimate.

## Run the official CPU scorer in the tested image

The [CPU scoring examples](../docs/algengine_ascend.md#6-evaluation-and-official-scoring)
cover genuine logs/maps and existing official cache payloads. They use the
original NAVSIM scoring path and explicit container mounts without NPU devices.
