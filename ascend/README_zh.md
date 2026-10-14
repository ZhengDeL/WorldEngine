# WorldEngine AlgEngine · Ascend 910C

[English](README.md) | [安装与使用](../docs/algengine_ascend_zh.md) | [验证报告](VALIDATION_zh.md)

`ascend-910c-algengine` 分支提供 AlgEngine 的 Ascend NPU 运行入口，覆盖数据准备、模型训练、断点续训、推理、评价和微调。适配范围为 AlgEngine；SimEngine 的仿真与渲染保留 GPU 路径，其真实导出数据可作为 AlgEngine 的训练输入。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装与使用](../docs/algengine_ascend_zh.md) | 数据准备、原配置训练、续训、评价、稀有场景筛选和微调命令 |
| [运行环境](docker/README_zh.md) | 基础镜像、post8 构建、版本检查和离线转移 |
| [验证报告](VALIDATION_zh.md) | 实测配置、数值结果、资源需求、已知限制及证据索引 |
| [16 芯片同步重放](execution/README_zh.md) | VAD/Hydra 八次更新的连续训练与自身断点对照 |
| [真实 SimScale 数据准备](data_tools/README_zh.md) | 真实场景转换、8192 候选奖励及两个配置的小样本复现 |
| [MSDA 实现与验证](MSDA_zh.md) | FP32 采样实现、前向与梯度检查、复现方法 |

## 功能与验证范围

| 功能 | 已验证范围 | 适用边界 |
| --- | --- | --- |
| 原配置训练与微调 | VADv2、HydraMDP、DiffusionDrive 及两类 UniAD 感知配置的小数据训练；36 个真实数据配置各完成三次更新 | 配置覆盖结果有独立版本记录，未验证全部配置的完整训练与收敛 |
| 完整检查点恢复 | 模型、优化器、调度与各 rank RNG 恢复；原生 epoch 和短程迭代均有实测 | 精确载入与后续数值一致性分别判定 |
| 16 芯片严格重放 | 公开同步入口四组基座训练对照均完成，两组通过原容差；新镜像两类 LoRA 微调各一组全部逐位相同 | 基座训练仍有间歇性超容差项，微调通过不能替代全参数训练验收 |
| 推理与评价 | 原感知评价器、规划轨迹输出及官方 NAVSIM CPU 评分 | 小数据结果未覆盖完整独立 NAVTEST common/rare 基准或 GPU 对照 |
| 真实合成数据配置 | 两个 SimScale 配置的 CPU 数据链及单、双 rank 训练与续训 | 共八个相关配置，其余六个真实导出尚未齐备 |

各项结果的源码、镜像、数据范围和原始记录见[验证报告](VALIDATION_zh.md)。

## 公开源码实机复现

公开父、子镜像配方已从核验过的官方 SDK 基座无应用层缓存重建，并在第二台服务器完成 14 项真实小集流程。前四项 CPU 数据准备使用既有 post8 镜像，后十项训练与评价使用新镜像。两类原 rare-log LoRA 微调均完成 16 芯片训练、自身断点恢复、原严格比较、22 个独立锚点推理和官方评分；两组比较全部逐位相同。实际准备的训练候选为 37 个，评价为 22 个，按完整驾驶日志划分。[实测证据](results/actual-public-reproduction-16chip-20261014.json)。

新镜像的 `pip check` 仍报告 16 项依赖声明问题，包括 protobuf 约束。这些问题没有阻止已测流程完成，但完整依赖一致性及其他配置的运行仍需核实。

## 快速开始

在 Linux ARM64 昇腾服务器克隆迁移分支：

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
```

1. 按[运行环境说明](docker/README_zh.md)构建或导入镜像，记录实际镜像 ID。
2. 按[数据准备说明](../docs/algengine_ascend_zh.md#2-数据下载权重和目录)准备官方标注、权重、所选相机图像和地图。
3. 执行[真实小数据验证](../docs/algengine_ascend_zh.md#3-准备真实小数据完成单芯片八次更新)，核对运行记录后，再按目标配置开展训练。

数据、权重下载与解压均在服务器执行。小数据验证可选择性准备 AlgEngine 输入，无需下载全部约 5 TB 资产。

## 运行环境与设备

| 项目 | 实测环境 |
| --- | --- |
| 宿主平台 | Linux ARM64，Atlas 800I A3 / Ascend910_9362 |
| 设备布局 | 8 张双芯物理卡，16 个逻辑芯片，每芯片 64 GiB HBM |
| 目标运行栈 | PyTorch 2.7.1、Torch-NPU 2.7.1.post8；完整镜像与依赖记录见[环境说明](docker/README_zh.md) |
| 已验证精度 | FP32；整模型 AMP/BF16 尚未验证 |
| NPU 入口 | `ascend/run_container.py`；`e2e_dist_train_npu.sh`、`e2e_dist_eval_npu.sh` |

`--devices` 使用宿主逻辑芯片编号。单芯片功能验证与多芯片训练的资源需求不同；完整 NAVTRAIN 数据对象常驻内存已测量，完整训练的内存、I/O 和机器数量仍需验证。

## 已知限制

公开同步入口的四组独立基座训练对照已完成，其中两组超出原容差，严格数值问题仍待解决。已有八小时感知训练属于较早的 source858/post6 环境；目标 cc91/post8 环境的持续训练尚未完成。

完整 NAVTRAIN 图像训练与收敛、完整独立 NAVTEST、GPU 输出对照、其余六个真实合成配置及多机扩展尚未验证。公开配方重建及两类微调的小集完整流程已有实测，不能替代全部配置和完整训练验收。具体测量与限制见[验证报告](VALIDATION_zh.md#已知限制与待验证范围)。

## 问题报告

提交运行问题时，请提供源码提交与指纹、镜像 ID、配置、精度、逻辑芯片布局、数据 split、复现命令，以及 `launch.json`、`runtime_rank*.json/jsonl` 和相关比较报告。请移除凭据和私有数据路径。
