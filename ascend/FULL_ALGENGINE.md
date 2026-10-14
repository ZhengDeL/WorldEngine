# AlgEngine Ascend 910C Validation Report

Assessment date: October 14, 2026.

[Chinese validation report](VALIDATION_zh.md) · [Installation and operation guide](../docs/algengine_ascend.md) · [Versioned evidence](results/)

## Scope and validation status

WorldEngine contains AlgEngine for model training and evaluation, and SimEngine
for simulation, rendering and rollout generation. This report covers AlgEngine.
SimEngine deployment on NPU is outside the assessed scope.

Training, fine-tuning, inference, official CPU scoring and complete-checkpoint
restoration have run on Ascend 910C within the configurations recorded below.
The current sixteen-rank synchronous controls completed one VAD pair and one
Hydra pair within the unchanged numerical tolerance. Default execution completed
four pairs, with three final tensor comparisons outside that tolerance.
Complete AlgEngine acceptance remains open: the evidence does not establish
full NAVTRAIN convergence, complete independent quality, all genuine synthetic
inputs, GPU equivalence or sustained operation of the final environment.

## Tested runtime and version boundaries

### Current planning controls

| Item | Recorded configuration |
| --- | --- |
| Runtime Python fingerprint | cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88 |
| Fingerprint scope | 243 files: recursive AlgEngine Python and top-level Ascend Python, LF-normalized |
| Image ID | sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45 |
| PyTorch / Torch-NPU | 2.7.1+cpu / 2.7.1.post8 |
| Torch-NPU git version | a783f11e456f27d7451a1ea745f1556823357c95 |
| Recorded host stack | Ascend 910C, reported chip model Ascend910_9362; CANN 9.1, driver 26.1.1 |
| Topology | One server, eight physical dual-chip cards, sixteen logical chips/ranks |
| Batch and loader workers | B1 per rank, global batch 16, workers 0 |
| Precision | FP32; public Matmul HF32 and Conv HF32 both disabled |
| Determinism request | Torch deterministic algorithms enabled; CLOSE_MATMUL_K_SHIFT=1 and HCCL_DETERMINISTIC=true |
| Synchronous candidate settings | TASK_QUEUE_ENABLE=0, ASCEND_LAUNCH_BLOCKING=1, set before Torch startup |
| Data | Genuine NAVTEST-derived mini: 42 training anchors in two logs, 22 independent evaluation anchors in a third log |
| Trajectory vocabulary | Original 8192 candidates, each 40x3; labels and original model geometry retained |
| Checkpoint control | Continuous updates 1–8; own update 4 checkpoint restored to update 8 |

The runtime fingerprint identifies the measured Python subset. It is not a hash
of every documentation, shell or data-preparation file in a later checkout.
Public installation and launcher additions have separate local checks; a newly
rebuilt image requires its own identity and actual run evidence.

| Version group | Evidence assigned to that version |
| --- | --- |
| Historical fused path, 96a7bef… / a182… / post6 | Original 44-config catalogue and 36 real-config short runs, native mini epochs and evaluators; large MSDA location-gradient gate remained outside tolerance |
| Native MSDA path, 858f9d… / a182… / post6 | Expanded CPU32 MSDA gates, 36 real-config regression, single-rank controlled replay, independent planning scoring, original dataset construction and eight-hour perception stability |
| Current runtime, cc91deb… / 3fef77b… / post8 | Default and synchronous sixteen-rank VAD/Hydra controls and full-rank state diagnosis |
| Current runtime, cc91deb… / ba17931… / post8 | Genuine SimScale subset: original dataset access and one/two-rank controls for two configurations |

The genuine SimScale image ID is
sha256:ba17931d5af64c6070417bc2059887096abf42d76b44484a9bd45262e3918707.
Its results are separate from the 3fef planning controls. Historical results are
retained under their own source and image; they are not measurements of the
current sixteen-rank environment.

## Numerical correctness

### Synchronous sixteen-rank controls

Each model completed one independent continuous/resumed pair. Both training
containers exited 0, all 32 rank reports passed, and all 64 before/after precision
snapshots matched the declared FP32 settings. Each resumed run used its own
continuous run's update 4 checkpoint.

The original comparison rule was unchanged: tensor atol 1e-6 / rtol 1e-5, and
loss absolute-difference tolerance 1e-4.

| Model | Training/resume completion | Final numerical result | Model maximum absolute difference | Combined maximum difference | Maximum loss difference | Bitwise-equal model / combined tensors |
| --- | --- | --- | ---: | ---: | ---: | --- |
| VAD | Continuous 8; own checkpoint 4→8 | Within original tolerance; all bitwise equal | 0 | 0 | 0 | 889/889; 1698/1698 |
| Hydra | Continuous 8; own checkpoint 4→8 | Within original tolerance; not all bitwise equal | 2.9802322387695312e-8 | 2.9802322387695312e-8 | 0 | 868/909; 1468/1778 |

Combined counts include model, optimizer and two RNG tensors per rank. Batch
tokens, learning rates and final rank RNG states matched. No model or optimizer
tensor exceeded tolerance. These are one positive pair per model, without
independent repeated or long-run acceptance of this candidate.
[VAD evidence](results/sync-vad-16chip-20261014.json),
[Hydra evidence](results/sync-hydra-16chip-20261014.json).

### Default sixteen-rank controls

The same runtime, image and tolerance were tested with TASK_QUEUE_ENABLE and
ASCEND_LAUNCH_BLOCKING unset. All eight training/restoration jobs reached their
planned endpoints, exited 0 and passed exact saved-state loading and runtime
checks. Every pair executed the final CPU comparison at update 8.

| Model / repetition | Training/resume completion | Final numerical comparison | Model maximum absolute difference | Combined maximum difference | Maximum loss difference | Model / optimizer tensors above tolerance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| VAD /1 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 4.231929779052734e-6 | 5.522248102352023e-6 | 1.9073486328125e-6 | 6/7 |
| Hydra /1 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 6.511807441711426e-6 | 1.5364399587269872e-5 | 1.9073486328125e-6 | 7/14 |
| VAD /2 | Continuous 8; own checkpoint 4→8 | Outside original tolerance | 2.5041401386260986e-5 | 4.742613236885518e-5 | 1.9073486328125e-6 | 12/25 |
| Hydra /2 | Continuous 8; own checkpoint 4→8 | Within original tolerance; bitwise equal | 0 | 0 | 0 | 0/0 |

All four short controls completed. The three out-of-tolerance results concern
the final model/optimizer comparison, rather than incomplete training. Tokens,
LR and final RNG matched. Small maximum differences do not replace the
elementwise absolute/relative rule. Completion of eight updates does not mean
completion of NAVTRAIN training.
[Default-execution evidence](results/strict-replay-post8-20261014.json).

The cause of default-execution divergence remains unresolved. A separate
full-rank state diagnosis completed both models' controls: recorded registered
model/optimizer values and layouts before update 5 matched, while later
differences remained. The observer can change execution timing and does not
cover all unregistered caches or backend state.
[State-diagnosis evidence](results/full-rank-state-diagnostic-20261014.json).

### Incomplete and initialization-blocked tests

| Test | Recorded progress | Numerical conclusion |
| --- | --- | --- |
| Separate limited VAD control | Continuous run completed 8; own-checkpoint resume completed 4→6, exited 137 without OOM | Incomplete; target 8 was not reached and the final comparator was not executed |
| Cold start on another host | Driver device initialization blocked with EL0020 /507899; zero updates | Training not started; no final numerical comparison |

These tests are distinct from the six completed planning pairs above.

### MSDA and other compatibility mechanisms

On source 858f9d… / post6, the native NPU MSDA path passed the original CPU32 oracle
at unchanged tolerances: forward atol/rtol 2e-5/2e-5; value, location and attention-weight
gradients 1e-3/1e-4. Tested coverage includes eighteen ordinary cases through two
APIs, four legacy-layout cases and complete Q40000 geometry. The original
200x200/Q300/H8/C32/seed29 location-gradient case has 0/19200 out-of-tolerance
elements. Native grid-sampler device execution was observed.
[MSDA report](MSDA.md), [strict matrix](evidence/msda-strict-matrix-20261013.json),
[legacy layout](evidence/msda-map-layout-20261013.json),
[full shape](evidence/msda-full-shape-20261013.json).

| Mechanism | Implemented behavior and boundary |
| --- | --- |
| NPU attention compatibility | NPU-native grid sampling and first-order backward retain the MMCV reference coordinate convention, zero padding and reduction semantics. Legacy projected-L4/actual-L1 input uses the restricted CUDA-source flat-prefix interpretation; actual GPU parity is untested. |
| ResNet MaxPool | Uses the differentiable NPU-dispatched PyTorch path instead of the SDK forward-only fusion. |
| Complete checkpoints | Preserve model, optimizer, per-rank RNG, sampling/LR progress and source/image/data/precision contracts. Exact-state continuation requires workers 0; exact loading and subsequent numerical replay are separate checks. |
| CPU checkpoint comparison | Uses the installed official Torch-NPU serialization registration before loading legacy NPU storage on CPU, without NPU initialization; original numerical thresholds remain unchanged. |
| Original evaluation | Detection/mapping evaluators and pinned NAVSIM scoring are retained. Planning scoring uses the original 40-to-8 pose conversion and independent scenario caches. |
| Matrix inversion | An existing CPU fallback keeps the full model executable; its performance cost has not been optimized or accepted. |

MSDA floating computation remains on NPU; small integer shape metadata can be
read on CPU. This operator statement does not imply every full-model operation
runs on NPU. MSDA agreement has not identified the cause of default full-model
divergence. [Versioned native-path evidence](evidence/msda-precision-20261013.json).

## Configuration and data-path coverage

The original catalogue contains 44 configurations: 42 planning configurations
with original B2, and two perception configurations with original B1.

| Coverage | Completed validation | Version and limit |
| --- | --- | --- |
| Catalogue | 44 configurations resolved | Original catalogue; configuration resolution is not training acceptance |
| Real-data configurations | 36/36 completed three optimizer updates each, preserving original geometry and publisher-verified initializers | Historical 96a and native 858/post6 matrices; B1/workers 0/real mini and short schedule, not all 44 current-post8 full training |
| Base planning and real-log FT | Original training, LoRA/reward FT, Hydra policy-gradient, inference and checkpoint paths exercised | Config-specific small-data tests; VAD rare-log reward/LoRA has rl_finetuning=False, while Hydra's base enables policy gradients |
| Genuine synthetic configurations | Two of eight: SimScale finetune and cotrain, original CPU dataset access, one- and two-rank continuous 8/own checkpoint 4→8 controls | cc91/post8 on the separate ba179 image; genuine one-scene subset and narrowed normal pool |
| Independent planning scoring | Three representative planners processed 22 held-out anchors each and completed original official CPU scoring | Native 858/post6; small common-log evaluation, no complete rare benchmark or GPU comparison |

The six other rollout/replay/BWM configurations still require their genuine
producer exports. The two tested SimScale paths do not cover their distributions
or full-data sampling. Their one/two-rank comparisons were bitwise equal at the
original gate, with both real-normal and synthetic branches observed.
[Catalogue](evidence/original-configs-20261013.json),
[real-config/native-path results](evidence/msda-precision-20261013.json),
[CPU data access](data_tools/GETITEM_CPU_VERIFIED.json),
[single-rank controls](data_tools/NPU_SINGLE_RANK_VERIFIED.json),
[two-rank controls](data_tools/NPU_TWO_RANK_VERIFIED.json).

The portable SimScale recipe uses approximately 47.50 GB of pinned inputs,
excluding maps, source, the container image and generated outputs.
It computes 8192 labels with original WE four-second logged-traffic mathematics
and an explicit history/observation input adapter. This does not establish
published SimScale-v2 reward or full-engine lifecycle equivalence. The published
16384-candidate reward has incompatible fields/candidate alignment and cannot be
made equivalent by truncation or renaming. Finetune explicitly overrides the
unavailable e2e_vadv2_ep8.pth basename with the published 100pct checkpoint.
[Input/derivation contract and recipe](data_tools/README.md),
[CPU reproduction evidence](data_tools/CPU_REPRODUCTION_VERIFIED.json).

### Historical quality and numerical observations

| Version / workload | Preserved result | Interpretation |
| --- | --- | --- |
| Historical 96a fused MSDA | 35/19200 large-shape location-gradient elements outside tolerance; maximum 0.0309448 | Superseded by the native-path gate; the original negative result remains recorded |
| Historical post6 short FT,22 held-out anchors | VAD PDMS 0.910051108772398→0.8762690973035009; Hydra 0.9340064182046998→0.9340064182046998 | No demonstrated quality benefit from that short FT |
| Historical 96a perception mini epochs | Detection mAP 0; NDS 0.01950/0.02179 | Evaluator executed; trained perception quality and convergence unaccepted |
| Historical 96a VAD reward/LoRA epochs | PDMS 0.8455829502665201→0.8850917719220494; rare evaluation tokens 0 | Small common-log result, without rare-case quality acceptance |
| Native 858/post6 eight-rank replay | Both models completed continuous 8/own checkpoint 4→8; VAD 12 model/21 optimizer and Hydra 2/2 tensors outside tolerance | Exact restoration passed; distributed numerical replay did not meet the original gate |
| Historical 96a DiffusionDrive, three updates | 22 finite 40x3 generated trajectories; official PDMS 0.377858654551, ADE/FDE 10.96562/19.69085 | Functional generated-trajectory scoring, not trained planner quality |

Sources: [early bounded results](validation_results_20261013.json),
[historical full-module evidence](evidence/full-algengine-20261013.json),
[native eight-rank replay](evidence/eight-rank-resume-20261013-evening.json),
[representative planning results](evidence/msda-planning-20261013.json).

## Full NAVTRAIN metadata and CPU capacity

Both official NAVTRAIN PKLs were fully transferred and publisher-hash verified;
their total serialized size is 39,862,469,866 bytes.

| Inspection | Measured result | What it establishes |
| --- | --- | --- |
| Raw annotation inventory | 246,165 unique frames,1,192 logs, 1,969,320 valid camera references | Required fields/path references; zero JPEG files read |
| Official split/PDM coverage | 103,288 split tokens and 103,288 PDM entries; no missing split tokens | Complete metadata/PDM token coverage |
| Six required PDM metrics | 619,728 finite float32[8192] arrays, values 0–1 | All six required numerical arrays checked unchanged; seventh driving_direction field received schema inventory only |
| Sequential raw-pickle QC | Peak RSS 34,747,150,336 bytes | CPU QC memory; not original dataset-constructor or training RSS |
| One resident original dataset | RSS 52,208,001,024 bytes; PSS 52,136,001,536 bytes | Original constructor/Compose residency |
| Eight resident original datasets | Summed RSS 416,754,966,528 bytes; PSS 413,451,488,256 bytes | Measured multi-process dataset residency |
| Sixteen resident original datasets | Summed RSS 834,131,996,672 bytes; PSS 826,988,978,176 bytes | Measured sixteen-process dataset residency |

Constructor tests used source 858/post6, the original 50% filter with 50,000
anchors, all 246,165 annotation frames and 103,288 PDM entries per dataset. An
explicit import-time SDK device-name bridge enabled CPU-only construction; no
images, model, NPU operations or loader-worker training were executed. These
measurements do not include full-image/model/optimizer/worker/evaluation memory.
The 103,288-anchor full training filter also needs its own runtime measurement.
[Complete metadata QC](results/full-navtrain-metadata-qc-20261014.json),
[original constructor capacity evidence](results/full-navtrain-dataset-construction-20261014.json).

The targeted current/history training camera archives total 449,046,313,693
compressed bytes. Existing disk-based pipelines require extracted, readable
JPEGs with complete four-frame/eight-camera coverage. Their extraction footprint
and full training I/O have not been measured. These archives are separate from
the roughly 5 TB SimEngine scene publication. PDM vocabulary labels and official
NAVSIM scenario caches are also separate inputs.

## Operational stability

The completed eight-hour perception experiment belongs to source 858f9d…
(242 files), image a182f52… and Torch-NPU 2.7.1.post6. It used eight physical cards,
one logical chip per card, B1/global 8 and workers 0 on the 42-anchor mini.

- Each rank completed 1023 updates; minimum measured training time was
  28,827.677454 seconds.
- Restoring the same run's final checkpoint completed 1023→1026, with 1393 model
  and 3594 optimizer tensors per rank checked exactly after loading.
- Independent perception evaluation covered 37 held-out current/history frames;
  two complete checkpoints passed CPU audit.
- All four training/resume/inference/audit stages exited 0 without OOM.
  All 16 rank streams were checked: 8208 contiguous update records and 837,216
  finite raw-loss scalars.
- Gradient checks ran at the first update and every tenth update. The CPU RNG
  audit checked field/type/count structure rather than equality. Strict
  continuous-versus-resumed numerical replay was not performed for this run.

Its measured peak allocated/reserved NPU memory was 46,908,357,632 /
52,208,599,040 bytes per chip; observed charged CPU cgroup peak was
74,654,064,640 bytes. These are mini-run measurements. This legacy run lacked
the current per-rank precision before/after instrumentation and does not accept
cc91/post8 sustained training,24-hour stability or a general memory-leak absence.
[Eight-hour evidence and full-stream supplement](results/eight-hour-perception-stability-20261014.json).

## Equipment and topology planning

| Workload | Measured device layout | Capacity boundary |
| --- | --- | --- |
| Small-data functional training | One logical chip on one 910C server | Measured functional cases; not a complete-data minimum guarantee |
| Current planning global batch 16 | One server, eight dual-chip cards / sixteen logical chips at B1 | Matches the original eight GPUs × B2 sample count; numerical/grouping equivalence with GPU is untested |
| Genuine SimScale two-rank subset | Two physical cards, one chip per card, B1/global 2 | Does not validate both chips of a single card as a two-rank layout |
| Eight-hour perception | Eight physical cards, one chip per card, B1/global 8 | Historical 858/post6 mini stability only |

The measured logical chips have 64 GiB HBM. A complete sixteen-rank training plan
must budget above the 834.13 GB measured dataset RSS for image/pipeline buffers,
model/optimizer host state, validation, workers and reserve. Full training RAM,
HBM headroom and storage/I/O must be measured together; serialized PKL size
cannot determine the device count or RAM requirement. Four physical cards with
eight chips at B2, larger per-chip batches and multi-node execution are not
accepted by these measurements.

Performance optimization is deferred. Historical speeds are workload-specific
observations and do not establish paired GPU/NPU throughput or cost advantages.

## Outstanding acceptance

| Area | Remaining requirement |
| --- | --- |
| Default distributed numerics | Identify the cause of completed default comparisons outside tolerance and repeat unchanged-gate controls |
| Synchronous candidate | Multiple independent controlled pairs for both models and sustained operation of the same cc91/post8 environment |
| Complete training | Original 50%/100% NAVTRAIN horizons, convergence and target batch/worker settings |
| Independent quality | Complete NAVTEST common/rare evaluation, adequate rare coverage and agreed quality thresholds |
| Data and capacity | Full current/history JPEG coverage, image decode/pipeline quality, full training/evaluation RAM, HBM and I/O |
| Synthetic scope | Genuine producer exports and original training/evaluation coverage for the other six configurations |
| Platform extensions | GPU trajectory/model comparison, multi-node training, AMP/BF16 and second-order gradients |
| Clean-server reproduction | Public image-build and entry-point commands have local checks; a complete cold build followed by NPU training, own-checkpoint replay and scoring on a clean server remains unverified |
| Sustained operation | 24-hour/target-duration testing with the same final source, image and execution settings |

AMP/BF16 and performance tuning are extensions to the assessed FP32 route;
they are not prerequisites for executing the recorded FP32 workflows.
Installation, data preparation, training, resume and scoring procedures are
maintained in the [AlgEngine operation guide](../docs/algengine_ascend.md).
The [synchronous protocol](execution/README.md) and
[portable SimScale tools](data_tools/README.md) define their narrower reproducible
inputs. Local tool/command checks do not constitute a new NPU acceptance run.
