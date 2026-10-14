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
| Base-planner training and resume | VADv2 and HydraMDP each passed one 16-rank synchronous continuous/resumed comparison | Independent repeats, final-environment sustained training and full NAVTRAIN convergence |
| Fine-tuning | Real-log fine-tuning and two genuine SimScale configurations; the latter passed one/two-rank controls on one scene with WE-derived 8192 rewards | Six other genuine synthetic exports, full-data sampling and published SimScale-v2 reward equivalence |
| Inference and evaluation | Original planning/perception evaluation and independent official NAVSIM CPU rescoring on small datasets | Full independent NAVTEST quality and GPU/NPU output comparison |
| MSDA | FP32 forward and gradient gates passed at the original tolerances for the recorded shapes/layouts | Other precision modes and higher-order derivatives |
| Data and resources | Full NAVTRAIN metadata/PDM QC and 1/8/16 resident original dataset measurements | Full camera coverage and model/optimizer/worker training memory and I/O |
| Operational stability | An eight-hour perception run, own-checkpoint continuation, inference and CPU audit passed on the earlier `858f9d…` / post6 environment | Sustained training on the final `cc91…` / post8 synchronous profile |

## Synchronous replay results

The measured runtime fingerprint is
`cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88`
(243 runtime Python files), using the recorded image with `torch_npu 2.7.1.post8`.
The controls used eight physical cards / sixteen logical chips, FP32 with Matmul
and Conv HF32 disabled, batch 1 per rank, zero workers, and a genuine NAVTEST-derived
mini containing 42 training and 22 held-out anchors. Initial settings were
`TASK_QUEUE_ENABLE=0`, `ASCEND_LAUNCH_BLOCKING=1`, `HCCL_DETERMINISTIC=true` and
`CLOSE_MATMUL_K_SHIFT=1`.

Each model trained for eight updates, then restored its own update-4 checkpoint
and continued to update 8. Tensor tolerances remained `atol=1e-6, rtol=1e-5`;
the loss tolerance remained `1e-4`.

| Configuration | Original numerical gate | Maximum tensor difference | Maximum loss difference |
| --- | --- | ---: | ---: |
| [VADv2 base planner](../projects/AlgEngine/configs/navformer/e2e_vadv2.py) | [Pass; all compared tensors bitwise equal](results/sync-vad-16chip-20261014.json) | 0 | 0 |
| [HydraMDP base planner](../projects/AlgEngine/configs/navformer/e2e_hydramdp.py) | [Pass within tolerance; not all tensors bitwise equal](results/sync-hydra-16chip-20261014.json) | 2.9802322387695312e-8 | 0 |

Each result is one complete pair. With queue/blocking settings unset, four other
completed controls produced one comparison within tolerance and three outside it;
the cause of divergence remains unresolved. The synchronous results do not establish
a repair of default execution or universal reproducibility. Detailed controls and
sources are retained in the [validation report](FULL_ALGENGINE.md).

The public synchronous launch adapter has local command-construction validation.
Its own NPU execution remains unverified; the measured pairs used a separate
controlled launch procedure with the same original trainer and initial settings.

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
