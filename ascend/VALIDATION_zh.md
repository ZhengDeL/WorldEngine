# AlgEngine Ascend 910C 验证报告

[English](FULL_ALGENGINE.md) | [安装与使用](../docs/algengine_ascend_zh.md) | [文档首页](README_zh.md)

验证记录日期：2026 年 10 月 14 日。

## 验证范围

本报告记录 AlgEngine 在 Ascend 910C 上的训练、微调、检查点恢复、推理、评价和资源测量。SimEngine 的仿真与渲染不在适配范围内；真实 SimEngine/SimScale 导出作为部分 AlgEngine 配置的输入。

已测 FP32 同步配置下，VAD 和 Hydra 的 16 芯片短程训练与续训对照均通过原数值容差，每个模型各一对。完整 NAVTRAIN 训练、收敛和完整独立 NAVTEST 评价尚未完成。各项结果仅适用于其记录的源码、镜像、数据和设备布局。

## 运行环境与版本

| 项目 | 16 芯片同步对照 |
| --- | --- |
| 上游 WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` |
| NAVSIM | v1.1，`3e8291bfa89ff247231e0227778840cd0a036896` |
| 运行源码指纹 | `cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88`，243 个 Python 文件 |
| 镜像 ID | `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45` |
| Torch-NPU | `2.7.1.post8` |
| 设备 | Atlas 800I A3 / Ascend910_9362，8 张物理卡、16 个逻辑芯片，每芯片 64 GiB HBM |
| 精度与执行 | FP32，Matmul/Conv HF32 均关闭；Torch 确定性开启 |
| 初始环境 | `TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=1`、`HCCL_DETERMINISTIC=true`、`CLOSE_MATMUL_K_SHIFT=1` |
| 批量与取数 | 每 rank batch=1，global batch=16，workers=0 |
| 数据 | NAVTEST 派生子集，42 个训练锚点、22 个独立评价锚点，按完整驾驶日志隔离 |

源码指纹按路径排序，覆盖 `projects/AlgEngine/**/*.py` 及 `ascend/*.py`，CRLF 规范为 LF。嵌套数据工具使用独立 SHA 清单。镜像名称为本地标签；重新构建的镜像 ID 与既有实测 ID 可能不同。

报告中的历史 source858 指纹为 `858f9d01181d3d62c35d0635881b22c62daf1a5f5af8123f622055de05da422d`（242 文件），镜像为 `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`，Torch-NPU 为 post6。更早 source96a 的配置与评分结果保留在[历史完整模块记录](evidence/full-algengine-20261013.json)。各版本结果分别引用。

## 16 芯片数值重放

### 实验协议

每对实验先连续训练到 iter8，保存该次训练的 iter4 完整检查点，再从自身 iter4 恢复到 iter8。比较最终模型、优化器、各 rank RNG，以及 loss、batch token 和学习率。两次独立初始化不作为断点重放的基准。

Tensor 容差为 `atol=1e-6, rtol=1e-5`，loss 绝对差门槛为 `1e-4`；原比较器及门槛均未修改。运行完成度与最终数值结果分别记录。“完成”指预定的八次更新对照完成，不代表完整 NAVTRAIN 训练完成。

### 显式同步执行

| 模型（各一对） | 训练与续训 | 最终数值结果 | 模型最大绝对差 | 联合 Tensor 最大差 | 最大 loss 差 |
| --- | --- | --- | ---: | ---: | ---: |
| VAD | 连续至8；自身4→8 | 通过，全部逐位相同 | 0 | 0 | 0 |
| Hydra | 连续至8；自身4→8 | 通过原容差，非全部逐位相同 | 2.9802322387695312e-8 | 2.9802322387695312e-8 | 0 |

联合比较包含模型、优化器及每 rank 的两个 RNG Tensor。VAD 的 889/889 个模型 Tensor、1,698/1,698 个联合 Tensor 逐位相同。Hydra 分别为 868/909 和 1,468/1,778，其他差异处于原容差内。两组的 batch token、学习率和最终各 rank RNG 相同，模型与优化器均无超容差项。四个训练/续训任务均正常退出，无 OOM。

证据：[VAD](results/sync-vad-16chip-20261014.json)、[Hydra](results/sync-hydra-16chip-20261014.json)。复现协议见[同步重放说明](execution/README_zh.md)。公开辅助器已完成本地命令构造检查，但该辅助器本身尚未在 NPU 执行；上述实测来自此前受控启动流程。

每个模型仅完成一对。同步方式的独立重复、持续训练及对其他配置的覆盖尚未验证，默认方式的数值分叉根因仍未确认。

### 默认执行方式

这四对实验使用相同 cc91/243 源码、post8 镜像及原容差，`TASK_QUEUE_ENABLE` 和 `ASCEND_LAUNCH_BLOCKING` 均未设置。八个训练/续训任务均完成预定终点并正常退出，精确检查点载入通过；最终比较结果如下。

| 对照 | 训练与续训 | 最终数值比较 | 模型最大绝对差 | 联合 Tensor 最大差 | 超容差模型/优化器 Tensor 数 |
| --- | --- | --- | ---: | ---: | ---: |
| VAD 第1次 | 连续至8；自身4→8 | 超出原容差 | 4.231929779e-6 | 5.522248102e-6 | 6 / 7 |
| Hydra 第1次 | 连续至8；自身4→8 | 超出原容差 | 6.511807442e-6 | 1.536439959e-5 | 7 / 14 |
| VAD 第2次 | 连续至8；自身4→8 | 超出原容差 | 2.504140139e-5 | 4.742613237e-5 | 12 / 25 |
| Hydra 第2次 | 连续至8；自身4→8 | 通过，逐位相同 | 0 | 0 | 0 / 0 |

证据：[默认执行严格对照](results/strict-replay-post8-20261014.json)。训练正常完成与最终 Tensor 处于容差内是两个独立判定；三组超容差记录仍是待解决问题。

追加的[全 rank 状态诊断](results/full-rank-state-diagnostic-20261014.json)完成每个模型 144 项状态关联，未阻塞；记录的 update5 前注册模型/优化器状态及布局相同，最终 update8 仍有差异。该诊断读取 NPU 状态到 CPU，会影响执行时序，未证明全部后端状态或未注册缓存相同，也未确定根因。

### 未完成的对照

早先限时 VAD 同步任务完成连续训练至 iter8，续训仅从 iter4 到 iter6，未达到预定终点，也未执行最终比较器。该项状态为“未完成，无最终数值结论”，与后续完整的 VAD/Hydra 同步实验分别记录。另一节点的冷启动在设备初始化阶段受阻，更新数为0，也无数值结论。

## 功能覆盖

| 验证项 | 已有结果 | 证据与边界 |
| --- | --- | --- |
| 公开配置解析 | 44 个配置完成解析 | [原配置契约](evidence/original-configs-20261013.json)；解析不代表完整训练通过 |
| 原模型训练与微调 | 36 个真实数据配置各完成三次优化器更新，使用原指定初始化权重 | [source858 回归](evidence/msda-precision-20261013.json)；真实小数据、B1/workers0，未覆盖原数据比例或收敛 |
| 模型结构 | 相关原配置保留 200×200 BEV、900 tracking queries、8 相机和原损失结构 | 按各模型原几何与配置核对；规划采用4帧时序，感知采用3帧队列 |
| 原生 epoch 与恢复 | 历史2 rank 感知训练每 rank 21 次更新，恢复后累计42次，两个 epoch 执行原检测/地图评价 | [历史完整模块记录](evidence/full-algengine-20261013.json)；mAP 为0，未证明模型质量 |
| 独立规划评分 | VAD、Hydra、DiffusionDrive 的22个独立锚点完成轨迹生成与原 NAVSIM CPU 评分 | [source858 记录](evidence/msda-precision-20261013.json)；子集 rare token 为0，未覆盖完整 rare 基准 |
| 真实合成数据 | 两个 SimScale 配置完成原数据分支、单 rank 及双 rank 连续/恢复对照，均逐位相同 | [单 rank](data_tools/NPU_SINGLE_RANK_VERIFIED.json)、[双 rank](data_tools/NPU_TWO_RANK_VERIFIED.json)；cc91/post8，使用各自记录的 `ba17931d…` 镜像 |

SimScale 验证仅使用一个真实场景、原 WorldEngine 四秒记录交通协议及8192候选奖励，每配置一个 normal 锚点。其余六个 rollout/replay/BWM 配置的真实导出尚未齐备；该窄样本未验证公开 SimScale-v2 的16384奖励或全量采样等价性。输入来源、容量和缺失初始化文件的显式覆盖见[数据工具说明](data_tools/README_zh.md)。

### MSDA 与兼容性

FP32 MSDA 使用原生 NPU `grid_sample` 前向与一阶后向，按原 MMCV CPU32 参考检查所有输出和三种输入梯度。source858 的18组普通布局×两个入口、四组原地图布局及完整 Q40000 几何均通过原容差；原 seed29 用例的位置梯度超差元素由35/19200降为0/19200。完整规则与实现见[MSDA 说明](MSDA_zh.md)。

MSDA 浮点张量保持在 NPU，完整模型仍保留矩阵求逆的 CPU fallback。迁移还提供 NPU 分布式封装、反向传播可用的 MaxPool 路径、完整训练状态恢复和官方 CPU 评分入口。AMP/BF16、二阶梯度及整模型 GPU 数值等价性尚未验证。

### 历史质量结果

较早 post6 小数据实验中，VAD 基座/微调 PDMS 为0.910051/0.876269，Hydra 为0.934006/0.934006，见[早期小数据结果](validation_results_20261013.json)。source96a 的感知 mAP 为0；该版本另一次 VAD reward/LoRA 两个 epoch 的 PDMS 为0.845583、0.885092，见[历史完整模块结果](evidence/full-algengine-20261013.json)。不同实验分别记录，这些结果未证明完整基准上的微调收益。

source858 的历史八 rank 短程重放也已完成，但 VAD/Hydra 最终 Tensor 超出原容差，模型/优化器超容差数分别为12/21、2/2，见[八 rank 记录](evidence/eight-rank-resume-20261013-evening.json)。早期 fused MSDA 和后续完整模型重放的数值问题具有独立版本与证据。

## 稳定性与数据容量

### 八小时感知训练

source858/post6 八 rank 感知训练使用8张物理卡、每卡一个芯片，B1/workers0。最短 rank 训练28,827.677454秒，每 rank 完成1,023次更新；从自身最终断点恢复三次至1,026，并完成37帧独立感知评价、370个有限预测字段及两个检查点的 CPU 审计。四个相关容器均正常退出，无 OOM。

该运行使用42个真实训练锚点重复采样。原始流包含8,208条训练/续训更新和837,216个有限 loss 标量；梯度有限值检查在首步及每十步执行。该结果不覆盖 cc91/post8 同步环境、完整数据训练、24小时运行或普遍严格数值重放。证据：[八小时稳定性](results/eight-hour-perception-stability-20261014.json)。

### 完整 NAVTRAIN 元数据

两个 NAVTRAIN 标注/PDM PKL 合计39,862,469,866字节，哈希已校验。完整元数据 QC 覆盖103,288个原 split token、1,192个日志、246,165个唯一标注帧，以及619,728个六项 PDM 数组；数组均为有限 `float32[8192]`，范围0–1。

此次 QC 读图数为0。1,969,320条有效相机路径引用只说明路径字段结构，未证明 JPEG 存在或可解码。第七项 `driving_direction_compliance` 仅做结构清点。顺序 QC 峰值 RSS 为34,747,150,336字节，不等于原 dataset 构造或训练内存。证据：[完整元数据 QC](results/full-navtrain-metadata-qc-20261014.json)。

### 原 dataset 构造内存

source858/post6 下，原 dataset/Compose 使用完整 PKL 及原50%过滤，每个对象含246,165帧、50,000锚点和103,288个 PDM 项。独立进程同时驻留的实测结果如下。

| 独立进程数 | 合计峰值 RSS（bytes） | 合计峰值 PSS（bytes） |
| ---: | ---: | ---: |
| 1 | 52,208,001,024 | 52,136,001,536 |
| 8 | 416,754,966,528 | 413,451,488,256 |
| 16 | 834,131,996,672 | 826,988,978,176 |

CPU 实验使用 SDK 导入期设备元数据桥，随后恢复原查询函数，未初始化 NPU。测量只包含原数据对象与 pipeline 常驻，不含图像批次、模型、优化器、DataLoader worker 或评价内存，也不表示无桥 CPU 部署通过。RSS/PSS 与 cgroup 计费内存分别记录。证据：[dataset 构造实测](results/full-navtrain-dataset-construction-20261014.json)。

### 设备与存储规划

| 使用场景 | 已有资源证据 | 待测量范围 |
| --- | --- | --- |
| 小数据单模型验证 | 一个64 GiB逻辑芯片可运行已测 B1 感知模型，历史 allocated 峰值约43.61 GiB | 更大 batch、不同芯片容量、混合精度 |
| 原规划 global batch=16 | 原8 GPU×B2；本次16 NPU×B1，同为16个样本 | 数值等价性、4张双芯卡/8芯片 B2容量、多机扩展 |
| 完整 NAVTRAIN | 原数据对象1/8/16驻留及元数据 QC 有实测 | 完整训练内存、I/O、收敛耗时及机器数量 |

DDP 每 rank 持有模型副本和数据对象，增加 NPU 数量不会自动降低单芯片模型内存或解决 CPU 数据常驻需求。完整训练所需服务器数量尚无实测结论。

AlgEngine 小样本方案所需固定公开输入合计47.50 GB，地图、源码、镜像和输出另计。完整 NAVTRAIN 相机包约449.05 GB压缩，训练读取解压后的所选相机文件；存储还需计入解压数据、独立评价输入和检查点。该包大小不能直接作为训练磁盘预算。数据准备命令见[使用指南](../docs/algengine_ascend_zh.md)。

## 已知限制与待验证范围

| 项目 | 当前边界 | 完成验证所需条件 |
| --- | --- | --- |
| 完整训练与收敛 | 未执行完整 NAVTRAIN 图像训练及原 horizon 收敛 | 完整相机/历史引用、地图与训练资源，按原配置完成训练 |
| 独立基准质量 | 未完成完整 NAVTEST common/rare；无 GPU 输出对照 | 独立完整数据与官方 cache；GPU 对照需匹配设备和模型 |
| 合成配置覆盖 | 两个 SimScale 配置窄样本通过；其余六个未验证 | 真实 rollout/replay/BWM 图像、转换标注及匹配奖励 |
| 数值重放与持续运行 | 同步 VAD/Hydra 各一对通过；默认方式仍有超容差项 | 相同目标环境下的独立重复、分叉定位与持续训练 |
| 完整训练容量 | 原数据常驻已测，训练内存与工期未测 | 包含图像、模型、优化器、worker、评价与 checkpoint 的整体测量 |
| 其他执行方式 | 整模型 AMP/BF16、二阶梯度、多机未验证 | 对应精度、算子和分布式协议的独立验证 |
| 公开环境复现 | 新公开镜像配方与辅助入口完成本地检查，未完成整套干净 NPU 环境复跑 | 从公开源码、镜像构建和官方输入执行完整流程 |

性能优化与 GPU/NPU 成对性能比较不在本报告已验证范围内。历史小数据速率未作为完整训练工期或性能比使用。

## 证据与复现入口

实测记录位于 `ascend/results/`、`ascend/evidence/` 和 `ascend/data_tools/*VERIFIED.json`，保留各自源码、镜像、输入锁、结果与限制。已有实验记录的整理与发布不会增加实机验证覆盖。

安装和完整操作命令以[使用指南](../docs/algengine_ascend_zh.md)为准；专项复现分别见[同步重放](execution/README_zh.md)、[MSDA 验证](MSDA_zh.md)和[SimScale 数据准备](data_tools/README_zh.md)。
