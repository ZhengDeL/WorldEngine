<div align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/WE_title.png" width="800px">

# Towards the Era of Post-Training for Autonomous Driving
> *The missing infrastructure for Physical AI post-training in AD. Open-source. Production-validated.*

[![Paper](https://img.shields.io/badge/arXiv-2606.19836-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2606.19836)
[![YouTube](https://img.shields.io/badge/YouTube-Video-FF0000.svg?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=P1zEyfqa1uY)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Dataset-ffc107.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-orange.svg?style=for-the-badge)](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)
<br>
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.9-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

## AlgEngine on Ascend 910C / A3

This branch provides FP32 NPU training, finetuning, complete-state resume and planning inference for VADv2 and HydraMDP. See the [installation and usage guide](docs/algengine_ascend.md) / [中文](docs/algengine_ascend_zh.md). SimEngine NPU work is maintained on the separate [`ascend-910c-simengine-fp32-opt` branch](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-simengine-fp32-opt).

Current commands pin runtime `a0d8820b69d5bce3d61a8a6cc6b8135c1c3a4ffbd8623e3cd10b14784d989a81` / **250 Python files**. Inference optimization defaults to `none`; select `point-parallel-last-frame` explicitly. Training layout defaults to `none`; `head-major` and the additional `point-parallel-head-major` require FP32, `cuda-compatible` MSDA and the complete runtime checkpoint contract. Keep the source, layout and checkpoint contract unchanged when resuming.

### Verified scope

| Workload | Measured result | Boundary |
| --- | --- | --- |
| VADv2, official weight, complete NAVTEST | A0 runtime: 12,146 samples, continuous NPU inference in 1:47:18 (saved GPU job: 1:03:24); independent official PDM-Score ×100 =82.16216548. Selected trajectories and all 85,022 official metric values agree with the saved original GPU output. | Includes startup and result collection; independent scoring is excluded from inference time. |
| HydraMDP, official weight, complete NAVTEST | A0 runtime: 12,146 samples, continuous NPU inference in 1:44:19 (saved GPU job: 1:03:29); independent official PDM-Score ×100 =91.45952568. Complete typed outputs and the official score CSV are unchanged from the previous NPU result. | Four selected candidates/trajectories and two official score rows remain outside the original GPU tolerance; the mean score does not establish per-sample equality. |
| Two-model imitation short training and independent 4+4 resume | Recorded `eade6ede…` runtime, official-derived 139-sample set: complete model/Adam, all 16-rank RNG, input/LR/loss and cursor checks agree. | Eight updates do not complete a full dataset epoch or establish convergence. |
| VADv2 LoRA / HydraMDP policy-gradient finetuning | Same recorded runtime: 40 positions/38 unique anchors for LoRA and 2 unique anchors for PG; eight updates and independent 4+4 controls agree. | Distributed padding is not additional independent training data. |
| Final-batch policy-gradient resume | A0 runtime: three real NPU stages 8/8/0 updates; independent complete-state comparison agrees. The zero-update stage only finalizes the saved epoch and reports that PG did not execute. | Normal PG training still requires an observed nonzero policy-gradient loss. |

The NPU figures are actual complete-job times, not estimates or model-only latency. GPU times are saved original CUDA jobs on eight H20-3e GPUs; NPU jobs use eight dual-chip cards / sixteen ranks. Different job conditions prevent a hardware speed-ratio claim.

The short NPU training observations show rank 0 update 2–8 median time reduced from 74.92 to 56.11 s for VADv2 and 75.06 to 56.43 s for HydraMDP, with unchanged saved state. LoRA changes from 45.76 to 28.62 s and PG from 44.90 to 28.39 s. These compare the NPU layouts on their recorded small inputs; the timer excludes DataLoader fetch and subsequent checkpoint/state checks. They are not full-training steady-state throughput or GPU/NPU training speed measurements.

The [versioned validation record](docs/validation/algengine_fp32_optimization_20261029.json) keeps each actual runtime, tool, data and score identity, including the earlier complete `216499c4…` inference and the original 22-anchor controls. The current 224 CPU checks are separate from device acceptance. The final-batch fix does not allow an old-source checkpoint to be adopted under a new source label.

Full NAVTRAIN eight-epoch training and convergence, GPU/NPU training equivalence, six remaining genuine synthetic-export configurations and the HydraMDP per-sample GPU gap remain unaccepted. The persistent-buffer research candidate is not included in this branch. BF16 retains its separate historical scope; current qualification is FP32 with HF32 disabled.

Historical evidence: [October 21 FP32 results](ascend/results/fp32-validation-20261021.json), [independent 128-sample training controls](ascend/results/fp32-both-models-strict16-independent-resume8-20261021.json), [CPU input audit](docs/navtest_input_audit.md) / [中文](docs/navtest_input_audit_zh.md), and [filesystem inference recovery](docs/algengine_ascend.md#inference-recovery-with-file-collection).

<div id="top" align="center">
<p align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/README_overall.jpg" width="800px" >
</p>
</div>


## Table of Contents
- [Highlights](#highlights)
- [News](#news)
- [Benchmark](#benchmark)
  - [Qualitative Results — Closed-Loop Simulation on nuPlan](#qualitative-results--closed-loop-simulation-on-nuplan)
  - [On-Road Deployment — Night Urban Driving](#on-road-deployment--night-urban-driving)
- [System Architecture](#system-architecture)
- [Roadmap](#roadmap)
- [Getting Started](#getting-started)
  - [Documentation Overview](#documentation-overview)
  - [Installation](#installation)
  - [Quick Test](#quick-test)
  - [Deep Dive by Module](#deep-dive-by-module)
    - [SimEngine - Photorealistic Closed-Loop Simulation](#simengine---photorealistic-closed-loop-simulation)
    - [AlgEngine - End-to-End Model Training \& Fine-Tuning](#algengine---end-to-end-model-training--fine-tuning)
    - [Scene Reconstruction - 3D Gaussian Splatting-based method, MTGS](#scene-reconstruction---3d-gaussian-splatting-based-method-mtgs)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Related Resources](#related-resources)

## Highlights

- **WorldEngine** is a post-training framework for Physical AI that systematically addresses the long-tail safety-critical data scarcity problem in autonomous driving.
- **Data-driven long-tail discovery**: Failure-prone scenarios are automatically identified from real-world driving logs by the pre-trained agent itself — no manual design, no synthetic perturbations.
- **Photorealistic interactive simulation** via 3D Gaussian Splatting (3DGS): Each discovered scenario is reconstructed into a fully controllable, real-time-renderable simulation environment with independent dynamic agent manipulation.
- **Behavior-driven scenario generation**: Leverages Behavior World Model (BWM) to generalize and synthesize diverse traffic variations from existing long-tail scenarios, expanding sparse safety-critical events into a dense, learnable distribution.
- **RL-based post-training** on synthesized safety-critical rollouts substantially outperforms scaling pre-training data alone — competitive with a ~10× increase in pre-training data.
- **Production-scale validation**: Deployed on a mass-produced ADAS platform trained on 80,000+ hours of real-world driving logs, reducing simulated collision rate by up to **45.5%** and achieving zero disengagements in a 200 km on-road test.


## News
- **[2026/09/15]** Stable checkpoints released.
- **[2026/06/19]** Paper released on arXiv. See [World Engine: Towards the Era of Post-Training for Autonomous Driving](https://arxiv.org/abs/2606.19836).
- **[2026/04/09]** Official dataset released. See [OpenDriveLab/WorldEngine](https://huggingface.co/datasets/OpenDriveLab/WorldEngine) or [OpenDriveLab/WorldEngine (ModelScope)](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)
- **[2026/04/10]** Official code repository established.


## Benchmark

We compare different post-training paradigms on the nuPlan dataset, evaluating on both open-loop and closed-loop metrics across common and rare driving scenarios.

> **Metric notes:**
> - **Open-loop PDMS** is aligned with [NAVSIM v1.1](https://github.com/autonomousvision/navsim) PDM Score. *Common* denotes the standard `navtest` split; *Rare* denotes the `navtest_failures` subset — failure-prone rare-case scenarios extracted from `navtest`.
> - **Closed-loop Success Rate (SR)** is computed as NC × DAC (no-at-fault-collision score × drivable-area compliance score), reported as a percentage.
> - **Closed-loop Ego Progress (EP)** measures the route progress made by the ego vehicle during **SimEngine closed-loop testing**, reflecting whether the agent makes meaningful forward progress rather than merely avoiding collision or off-road failure.
> - **Closed-loop PDMS*** is the PDM Score obtained via **SimEngine closed-loop testing**, where the planner interacts with reactive agents in simulation under real-time rendering.
>
> **Training notes:**
> - **Rare logs** are failure-prone scenarios automatically extracted from `navtrain` by the pre-trained agent itself (see [Rare Case Extraction](docs/algengine_usage.md#rare-case-extraction)). 
> - **Common logs** are the standard cases in `navtrain`.

#### VADv2 (Base Planner)

| Method | Open-loop PDMS ↑ (common) | Open-loop PDMS ↑ (rare) | Closed-loop SR ↑ (rare) | Closed-loop EP ↑ (rare) | Closed-loop PDMS* ↑ (rare) |
|:-------|:-------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:--------------------------:|
| Base model | 85.64 | 47.14 | 73.66 | 46.71 | 60.98 |
| Supervised fine-tuning on rare logs | 87.50 | 52.55 | 74.51 | 47.59 | 61.87 |
| Post-training on common logs | 87.69 | 49.36 | 69.63 | 51.02 | 60.21 |
| Post-training on rare logs | 88.51 | 59.20 | 73.35 | 51.86 | 62.78 |
| Post-training on rare synthetic replays | 82.61 | 62.69 | 87.20 | 32.49 | 63.22 |
| Post-training on rare rollouts w/o Behaviour WM | 88.53 | 61.88 | 77.96 | 56.74 | 67.33 |
| **Post-training with WorldEngine** | **88.95** | **59.83** | **88.89** | **47.66** | **70.12** |

**Key findings:**
- Post-training on **rare logs** substantially improves rare open-loop PDMS over supervised fine-tuning (**59.20 vs. 52.55**), but does not improve rare closed-loop SR, indicating that fixed rare logs alone are insufficient for robust interactive behaviour.
- Post-training on **common logs** provides limited long-tail benefit and degrades rare closed-loop performance, reducing SR from **73.66%** to **69.63%** and PDMS$^\ast$ from **60.98** to **60.21**, confirming the importance of long-tail event discovery.
- The full WorldEngine pipeline achieves the best overall rare closed-loop performance, with the highest SR (**88.89%**) and PDMS$^\ast$ (**70.12**). It improves rare closed-loop SR by **+15.23** percentage points and PDMS$^\ast$ by **+9.14** over the base model, while maintaining strong common open-loop performance.

#### HydraMDP

The base model and WorldEngine rows match **Table S1** of the paper. Additional ablations follow the selected HydraMDP experiment records: pure IL fine-tuning for the supervised row, and reward shaping enabled, RL fine-tuning enabled, PG = 0.01, entropy = 0 for the RL rows. Closed-loop values use the **reactive** evaluation results.

| Method | Open-loop PDMS ↑ (common) | Open-loop PDMS ↑ (rare) | Closed-loop SR ↑ (rare) | Closed-loop EP ↑ (rare) | Closed-loop PDMS* ↑ (rare) |
|:-------|:-------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:--------------------------:|
| Base model | 93.86 | 69.91 | 75.15 | 62.98 | 68.69 |
| Supervised fine-tuning on rare logs | 88.49 | 57.43 | 70.77 | 52.81 | 62.74 |
| Post-training on common logs | 93.90 | 69.64 | 76.61 | 64.80 | 68.70 |
| Post-training on rare synthetic replays | 93.87 | 69.91 | 74.26 | 62.01 | 67.68 |
| Post-training on rare rollouts w/o Behaviour WM | 93.78 | 71.59 | 78.57 | 67.60 | 72.06 |
| **Post-training with WorldEngine** | **93.89** | **72.28** | **81.63** | **67.95** | **74.49** |

WorldEngine improves HydraMDP's rare closed-loop SR by **+6.48** percentage points, EP by **+4.97**, and PDMS$^\ast$ by **+5.80**, matching the gains reported in Table S1.

### Qualitative Results — Closed-Loop Simulation on nuPlan

Each pair shows the **Base model** vs **WorldEngine post-trained model** on the same rare-case scenario. Left: front-camera rendering; Right: BEV trajectory visualization.

<div align="center">
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_1.png" width="400px"></td>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_2.png" width="400px"></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_3.png" width="400px"></td>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_4.png" width="400px"></td>
</tr>
</table>
</div>

### On-Road Deployment — Night Urban Driving

Zero disengagements in 200 km on-road testing on a mass-produced ADAS platform.

<div align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_01.gif" width="270px">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_02.gif" width="270px">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_03.gif" width="270px">
</div>


## System Architecture

WorldEngine consists of two tightly coupled subsystems:


| Module | Function | Core Technology |
|--------|----------|----------------|
| **[SimEngine](docs/simengine_usage.md)** | Closed-loop simulation with ego & agents | Hydra, Ray, rendering |
| **[AlgEngine](docs/algengine_usage.md)** | End-to-end model training & evaluation | MMDetection3D, UniAD/VADv2/HydraMDP |


## Roadmap

- [x] Core platform integration (SimEngine + AlgEngine)
- [x] Multi-GPU distributed simulation and training
- [x] Rare case extraction and fine-tuning pipeline
- [x] Comprehensive documentation and usage guides
- [x] Hugging Face / ModelScope dataset
- [x] Open-source release (code, data, early pre-trained models)
- [x] arXiv preprint
- [x] Stable pre-trained models
- [ ] Behavior World Model integration


## Getting Started

For Ascend AlgEngine, begin with the [NPU usage guide](docs/algengine_ascend.md) or [中文指南](docs/algengine_ascend_zh.md). It includes image construction, selective downloads, real-mini preparation and the original-style `e2e_dist_train_npu.sh` / `e2e_dist_eval_npu.sh` commands. The CUDA installation and GPU-count examples below describe the upstream workflow.

### Documentation Overview

WorldEngine provides comprehensive guides for each stage of your workflow:

| Guide | Purpose | Key Topics |
|-------|---------|------------|
| **[Installation](docs/installation.md)** | Set up both conda environments | Two-environment setup (simengine + algengine), dependencies, troubleshooting |
| **[Data Organization](docs/data_organization.md)** | Prepare datasets and checkpoints | Data structure, Hugging Face/ModelScope downloads, symlinks |
| **[Quick Start](docs/quick_start.md)** | Run your first experiment in 5 min | Quick test tutorial, understanding results, complete pipeline |
| **[SimEngine Usage](docs/simengine_usage.md)** | Master closed-loop simulation | Rollout scripts, distributed testing, configuration, metrics |
| **[AlgEngine Usage](docs/algengine_usage.md)** | Train and fine-tune models | Training from scratch, evaluation, rare case extraction, RL fine-tuning |

### Installation

WorldEngine requires **two separate conda environments** due to different Python requirements.

**Full installation guide:** [docs/installation.md](docs/installation.md)

### Quick Test

Verify your installation with a pre-trained model:

```bash
# Set up environment variable
export WORLDENGINE_ROOT=$(pwd)

# Option 1: Single GPU test 
bash scripts/closed_loop_test.sh

# Option 2: Multi-GPU test (Default 8 GPUs)
bash scripts/multigpu_closed_loop_test.sh
```

**What this does:**
- Loads a pre-trained VADv2 model (50% training data, epoch 8)
- Runs closed-loop simulation on 288 rare-case test scenarios
- Evaluates with navsim v1 PDMS (collision avoidance, progress, comfort, etc.)
- Saves results to `experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/`

**Detailed quick start tutorial:** [docs/quick_start.md](docs/quick_start.md)

### Deep Dive by Module

After the quick test, explore each subsystem in detail:

#### SimEngine - Photorealistic Closed-Loop Simulation

Learn how to run simulations, generate rollouts, and test models:

- **Rollout scripts** for data generation (no model required)
- **Testing scripts** for model evaluation (single/multi-GPU)
- **Ray distributed simulation** for large-scale testing
- **Reactive vs non-reactive** agent modes
- **Configuration guide** for all Hydra parameters

**[SimEngine Usage Guide](docs/simengine_usage.md)**

#### AlgEngine - End-to-End Model Training & Fine-Tuning

Learn how to train models, extract rare cases, and fine-tune:

- **Training from scratch**
- **Open-loop evaluation** on test sets
- **Rare case extraction** from evaluation failures
- **RL-based fine-tuning** on long-tail scenarios
- **Multi-GPU training** with distributed data parallel

**[AlgEngine Usage Guide](docs/algengine_usage.md)**

#### Scene Reconstruction - 3D Gaussian Splatting-based method, MTGS

WorldEngine's simulation environments are powered by 3D Gaussian Splatting (MTGS):

- **Multi-traversal reconstruction** from nuPlan data
- **Photorealistic rendering** for closed-loop simulation
- **Asset generation** for SimEngine scenes

**[MTGS Repository](https://github.com/OpenDriveLab/MTGS)**


## Citation

If any parts of our work help your research, please consider citing us and giving a star to our repository:
```bibtex
@misc{li2026worldengine,
  title={World Engine: Towards the Era of Post-Training for Autonomous Driving}, 
  author={Tianyu Li and Li Chen and Caojun Wang and Haochen Liu and Kashyap Chitta and Zhenjie Yang and Yuhang Lu and Naisheng Ye and Yihang Qiu and Yufei Wang and Luoxi Zou and Jiaxin Peng and Jin Pan and Zhaoyu Su and Andrei Bursuc and Shengbo Eben Li and Andreas Geiger and Peng Su and Hongyang Li},
  year={2026},
  eprint={2606.19836},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.19836}, 
}
```


## Contributing

We welcome contributions from the community! Whether you want to:

- **Report bugs** - Open an [Issue](https://github.com/OpenDriveLab/WorldEngine/issues)
- **Improve documentation** - Submit a [Pull Request](https://github.com/OpenDriveLab/WorldEngine/pulls)
- **Contribute code** - Fork, develop, and submit a PR

Please read our contributing guidelines before submitting PRs.

**For questions:**
1. Check the [documentation](docs/) first
2. Search existing [Issues](https://github.com/OpenDriveLab/WorldEngine/issues)


## License

All content in this repository is under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

The released data is based on [nuPlan](https://www.nuscenes.org/nuplan) and is under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


<!-- ## 👥 Team & Acknowledgements

WorldEngine is developed by **Shanghai Innovation Institute (SII)** and **OpenDriveLab** at The University of Hong Kong.

**Core Contributors:**
- Scene Reconstruction Team
- Simulation Platform Team
- Algorithm Development Team

We would like to thank all contributors and the open-source community for their support. -->


## Acknowledgements

WorldEngine builds on top of the following works. We sincerely thank the authors for their contributions:

- **MTGS** — Multi-Traversal Gaussian Splatting, the scene reconstruction backbone powering our photorealistic simulation environments.
  ```bibtex
  @article{li2025mtgs,
    title={MTGS: Multi-Traversal Gaussian Splatting},
    author={Li, Tianyu and Qiu, Yihang and Wu, Zhenhua and Lindstr{\"o}m, Carl and Su, Peng and Nie{\ss}ner, Matthias and Li, Hongyang},
    journal={arXiv preprint arXiv:2503.12552},
    year={2025}
  }
  ```

- **Nexus (Decoupled Diffusion)** — Decoupled Diffusion Sparks Adaptive Scene Generation, enabling our augmented scenario generation pipeline.
  ```bibtex
  @inproceedings{zhou2025nexus,
    title={Decoupled Diffusion Sparks Adaptive Scene Generation},
    author={Zhou, Yunsong and Ye, Naisheng and Ljungbergh, William and Li, Tianyu and Yang, Jiazhi and Yang, Zetong and Zhu, Hongzi and Petersson, Christoffer and Li, Hongyang},
    booktitle={ICCV},
    year={2025}
  }
  ```

- **Optimization-Guided Diffusion** — Optimization-Guided Diffusion for Interactive Scene Generation, underpinning behaviour-driven scenario synthesis.
  ```bibtex
  @inproceedings{li2025optimization,
    title={Optimization-Guided Diffusion for Interactive Scene Generation},
    author={Li, Shihao and Ye, Naisheng and Li, Tianyu and Chitta, Kashyap and An, Tuo and Su, Peng and Wang, Boyang and Liu, Haiou and Lv, Chen and Li, Hongyang},
    booktitle={ECCV},
    year={2026}
  }
  ```

- **AlgEngine (Reinforced Refinement)** — Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving, the RL-based fine-tuning backbone of AlgEngine.
  ```bibtex
  @ARTICLE{liu2026r2se,
    author={Liu, Haochen and Li, Tianyu and Yang, Haohan and Chen, Li and Wang, Caojun and Guo, Ke and Tian, Haochen and Li, Hongchen and Li, Hongyang and Lv, Chen},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    title={Reinforced Refinement With Self-Aware Expansion for End-to-End Autonomous Driving},
    year={2026},
    volume={48},
    number={5},
    pages={5774-5792},
    doi={10.1109/TPAMI.2026.3653866}
  }
  ```

- **SimScale** — Learning to Drive via Real-World Simulation at Scale, providing insights into large-scale simulation-based training.
  ```bibtex
  @article{tian2025simscale,
    title={SimScale: Learning to Drive via Real-World Simulation at Scale},
    author={Haochen Tian and Tianyu Li and Haochen Liu and Jiazhi Yang and Yihang Qiu and Guang Li and Junli Wang and Yinfeng Gao and Zhang Zhang and Liang Wang and Hangjun Ye and Tieniu Tan and Long Chen and Hongyang Li},
    journal={arXiv preprint arXiv:2511.23369},
    year={2025}
  }
  ```


## Related Resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

<div align="center">

| Project | Description |
|:-------:|:------------|
| [![MTGS](https://img.shields.io/badge/MTGS-Multi--Traversal_GS-blue?style=flat-square&logo=github)](https://github.com/OpenDriveLab/MTGS) | Multi-traversal Gaussian Splatting for scene reconstruction |
| [![SimScale](https://img.shields.io/badge/SimScale-Driving_Simulation-AD9BC2?style=flat-square&logo=github)](https://github.com/OpenDriveLab/SimScale) | Large scale driving simulation |
| [![nerfstudio](https://img.shields.io/badge/nerfstudio-NeRF_Framework-green?style=flat-square&logo=github)](https://github.com/nerfstudio-project/nerfstudio) | Collaboration-friendly NeRF toolkit |
| [![MMDetection3D](https://img.shields.io/badge/MMDetection3D-3D_Detection-orange?style=flat-square&logo=github)](https://github.com/open-mmlab/mmdetection3d) | 3D detection framework |
| [![UniAD](https://img.shields.io/badge/UniAD-End--to--End_AD-red?style=flat-square&logo=github)](https://github.com/OpenDriveLab/UniAD) | End-to-end autonomous driving framework |
| [![VADv2](https://img.shields.io/badge/VADv2-End--to--End_AD-crimson?style=flat-square&logo=github)](https://github.com/priest-yang/VADv2) | Vectorized autonomous driving framework |
| [![NAVSIM](https://img.shields.io/badge/NAVSIM-AD_Benchmark-teal?style=flat-square&logo=github)](https://github.com/autonomousvision/navsim) | Non-reactive autonomous vehicle simulation benchmark |
| [![nuPlan](https://img.shields.io/badge/nuPlan-Dataset-purple?style=flat-square&logo=github)](https://www.nuscenes.org/nuplan) | Large-scale autonomous driving dataset |
| [![MetaDrive](https://img.shields.io/badge/MetaDrive-Driving_Simulation-ff69b4?style=flat-square&logo=github)](https://github.com/metadriverse/metadrive) | Compositional driving simulation platform |
| [![Ray](https://img.shields.io/badge/Ray-Distributed_Computing-yellow?style=flat-square&logo=ray)](https://github.com/ray-project/ray) | Distributed execution framework |
| [![Hydra](https://img.shields.io/badge/Hydra-Config_Management-lightblue?style=flat-square&logo=python)](https://github.com/facebookresearch/hydra) | Configuration management framework |

</div>

---

<div align="center">

**If you find WorldEngine useful, please consider giving us a star!**

**Quick Links:** [Documentation](docs/) | [Installation](docs/installation.md) | [Quick Start](docs/quick_start.md) | [Issues](https://github.com/OpenDriveLab/WorldEngine/issues) | [Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions)

**Contact:** For research collaboration or questions, visit our [Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions)

</div>
