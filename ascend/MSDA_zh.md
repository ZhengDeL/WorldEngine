# MSDA 严格位置梯度修复与验收

[English](MSDA.md) · [MSDA 实测结果](evidence/msda-precision-20261013.json) · [八 rank 重放结果](evidence/eight-rank-resume-20261013-evening.json)

FP32 路径的位置梯度门禁已通过，原容差未改。本文说明 NPU 原生采样实现及对应实测。原报告中 `96a7bef…` 指纹、一小时 428 更新和内存/吞吐属于历史版本，当前实现的测试另行记录。

截至 2026 年 10 月 13 日 22:44，VADv2、HydraMDP 的八 rank 断点恢复均成功执行，**后续严格数值重放两项均失败**，结果见下文及[对应 JSON](evidence/eight-rank-resume-20261013-evening.json)。MSDA 算子门禁和此前单 rank 重放通过的结论仍有效。计划 8 小时的感知长训仍在运行，尚无稳定性终验结果。

实测代码指纹为 `858f9d01181d3d62c35d0635881b22c62daf1a5f5af8123f622055de05da422d`，覆盖 AlgEngine 递归 Python 和 ascend 顶层 Python 共 242 个文件，CRLF 统一为 LF；两个普通/布局验证脚本包含在内。补充 probe 脚本另外记录 SHA256。沿用固定干净镜像 `local/worldengine:26.1.0-a3-clean-20261013`，ID 为 `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`。

## 修复方式与原因

MSDA 改用 NPU 原生 `grid_sample` 前向和一阶后向。按未修改 MMCV PyTorch 参考执行 `2 * locations - 1`、双线性采样、零填充、`align_corners=False`。坐标运算与归约顺序差异会放大大图上的 FP32 位置梯度误差，因此仅修正坐标的早期候选虽然越过旧 seed，仍在边界、多 seed 等扩展测试中失败；该候选没有作为最终修复。

浮点 value、位置、注意力权重、输出和梯度始终在 NPU 上；只有很小的整数 shape/start 元数据读取到 CPU。Profiler 已记录实际 NPU 采样前后向核。这一结论针对 MSDA；完整模型仍有原有矩阵求逆的 CPU fallback，不能宣称整模型没有 CPU 计算。

MMCV 和 mx_driving 的实际 FP32 入口使用同一实现。算子内部关闭外部 autocast，后向重算采样中间结果。query 按 2048 分块，限制单个临时张量大小；后向仍会构建所有分块的计算图，总峰值需实测。重算及额外调度可能增加耗时，整模型结果见下文。非 FP32、整模型 AMP 和二阶梯度不在本次测试范围。

## 严格数值结果

判定规则仍是 `绝对误差 <= atol + rtol × |参考值|`：前向 **atol=2e-5、rtol=2e-5**；value、位置、权重三种梯度均为 **atol=1e-3、rtol=1e-4**。所有元素均检查，CPU64 仅用于诊断，不能替换 CPU32 参考。

原失败用例 `200×200、Q300、H8、C32、seed29` 的位置梯度超差元素由 **35/19200** 减至 **0/19200**，最大绝对误差为 **0.001220703125**，通过原绝对与相对联合容差。结果未达到逐位相同，GPU fused 算子对照也尚未实测。

| 已执行验证 | 实际结果 |
|---|---|
| 18 组普通布局 × 两个入口 | 36/36 组前向及完整三种梯度通过 |
| 原地图特殊布局四组 | 4/4 通过，原始梯度形状完整，未访问尾部严格为零 |
| 完整 encoder：Q40000/H8/C32，两组 seed | 普通与特殊布局四个用例、6/6 入口比较通过 |
| 条件梯度、冻结输入、无梯度输入及外部 autocast | 12/12 通过 |
| 设备 profiler | 已取得 NPU 原生采样前向和后向核记录 |
| 36 个真实数据原配置 | 各完成三次更新，全部通过，分别使用原指定 initializer |
| 当前源码八 rank 感知持续训练 | 每 rank 160 次更新，全部超过一小时，真实退出码 0，无 OOM |
| 该次训练的第 100 步完整断点 | 八 rank 全状态恢复后，各继续三步至第 103 步 |
| 恢复后的独立感知推理 | 原检测/地图链路完成 37 帧，检查 370 个有限预测字段 |
| VADv2/Hydra 单 rank 受控严格重放 | 两项原门禁通过；模型/optimizer、loss、token、LR、最终 RNG 精确相同 |
| VADv2/Hydra 八 rank 受控严格重放 | 四个训练/恢复任务实际退出 0，全状态载入检查通过；两项最终严格比较退出 1，模型/optimizer 有张量超容差 |
| 完整 checkpoint CPU 审计 | 45/45 个不同路径通过模型/optimizer 有限性、源码、counter 和 RNG 检查 |
| 代表性规划推理与官方重评分 | VADv2、Hydra、DiffusionDrive 各 22 个唯一独立锚点与官方 CPU 评分覆盖均通过 |

普通矩阵覆盖大图多 seed、四 level、非方形、B2/B8、边界及相邻 FP32 数、尺寸为 1、非连续输入/梯度、storage offset，以及 softmax、零、单点和高幅度权重。真实模型回归保持 200×200 BEV、900 tracking queries、八相机。

## 已完成的一小时训练、续训和推理实测

已完成的一小时测试使用八 rank、B1/workers0，保留原感知 initializer、模型形状、解冻 backbone、loss、优化器和 warmup，LR horizon 固定为 10000 次更新。每 rank 完成 **160 次优化器更新**，训练时间为 **3616.849～3617.775 秒**。全局 1280 次锚点曝光来自 **42 个真实训练锚点**的重复与 sampler padding，未按完整 NAVTRAIN 的正式 epoch 训练。

1280 次曝光除以最慢 rank 的训练时间，得到端到端速率 **0.35381 锚点曝光/秒**，包含取数、训练、监测和 checkpoint 开销。raw loss 逐次检查，梯度在首步及之后每十步检查。每逻辑芯片峰值 allocated/reserved 为 **43.63/48.62 GiB**。CPU cgroup 计费峰值为 **66,488,492,032 字节（61.92 GiB）**，容器上限 **128 GiB**，memory fail count 为 0。该计费值包含缓存，不能用作进程 RSS 之和或完整数据内存预算。所有容器正常退出，没有 OOM；测试只覆盖一小时，24 小时稳定性和长期内存增长尚未验证。

历史 `96a7bef…` 版本的小数据速率为 **0.94808 曝光/秒**，高于当前测得值。两次实验的 MSDA/地图存储解释、实现和监测频率都有变化，未做成对受控比较，差额不能归因于某个单独改动。这些记录也不能用于计算 GPU/NPU 性能比或完整 NAVTRAIN 工期。性能优化按当前安排另行开展。

从该次训练保存的第 100 步断点恢复时，每 rank 完整复制 **1393 个模型 Tensor、3594 个 optimizer Tensor**，保持源码、数据、拓扑和优化器/LR 契约，恢复各 rank RNG。八 rank 均继续三步至 **第 103 步**并正常退出。恢复后的模型通过原检测/地图评价链路处理 **37 个独立 current/history 帧**，检查 **370 个有限预测字段**。这些结果验证了全状态恢复、继续训练和推理功能，后续每一步的严格数值重放及感知质量还需单独比较。

此前 VADv2/Hydra 的单 rank 受控续训各连续运行八步，再从同次运行的第 4 步断点恢复到第 8 步，使用 B1/workers0、deterministic 请求和相同八步 LR horizon。两项严格比较通过，Tensor **atol=1e-6/rtol=1e-5**、loss 绝对差 **<1e-4**未改。VADv2 的 **889 个模型/777 个 optimizer Tensor**、Hydra 的 **909/837 个 Tensor**均逐位一致。原比较脚本计入两个 rank-RNG byte Tensor 后报告总计 1668/1748，保留这些原字段和值。最大 Tensor 和 loss 差均为 0，batch token、LR、最终保存 RNG 相同。结果适用于这两次单 rank、短步数、同前缀测试；其他配置、seed、batch、拓扑和训练时长需另测。历史源码的数值失败记录仍保留。

### 夜间八 rank 严格重放结果

2026 年 10 月 13 日夜间以同一 `858f9d…` FP32 源码测试 VADv2、HydraMDP。每个模型连续训练至第 8 步，再从该次连续训练的第 4 步完整断点恢复至第 8 步。四个训练/恢复任务实际退出 0，32 份 rank 运行门禁通过；载入时逐张量检查全部模型/optimizer 状态，VADv2 每 rank 为 889/777 个，Hydra 为 909/837 个。最终 CPU 比较均实际退出 1，原门禁判定为失败。

| 八 rank 比较 | 模型最大绝对差 | 模型/optimizer/RNG 联合最大绝对差 | 超容差模型/optimizer 张量数 | 最大 loss 差 |
|---|---:|---:|---:|---:|
| VADv2 | 4.546716809272766e-6 | 2.5303452275693417e-5 | 12/21 | 1.9073486328125e-6 |
| HydraMDP | 2.862885594367981e-6 | 2.862885594367981e-6 | 2/2 | 9.5367431640625e-7 |

容差保持 Tensor `atol=1e-6/rtol=1e-5`、loss 绝对差 `<1e-4`。batch token、LR 和最终 rank RNG 均相同，loss 比较通过，模型/optimizer 张量比较失败。原比较器的浅拷贝共享失败列表，表中按路径前缀分别统计模型和 optimizer，未将累计 33/4 条全部计为模型失败。联合 Tensor 总数 1682/1762 包含八 rank 的 16 个 CPU/NPU RNG byte Tensor。

这是成功载入后的实际数值失败，与下文首次 CPU 反序列化中断不同。`HCCL_DETERMINISTIC=true` 和 `torch.use_deterministic_algorithms(True)` 已设置，根因尚未确认。MSDA 的固定输入严格梯度结果仍通过，多卡整模型超差尚未归因到该算子，也不能用历史 fused atomic accumulation 观察解释当前失败。原日志、阈值及失败路径见[八 rank 重放结果](evidence/eight-rank-resume-20261013-evening.json)。计划 8 小时长训在 22:44 快照时仍运行中。

单 rank 比较和 CPU 审计首次执行时，在 NPU checkpoint 反序列化阶段中断，没有生成比较 JSON，因此当次没有数值门禁结论。复跑先导入已安装官方 `torch_npu` 序列化注册，再执行未改动的原目标脚本。原 `map_location='cpu'` 触发官方 `RE_MAP_CPU` 从 false 变为 true。三项 CPU 任务均实际退出 0，未初始化 NPU，目标源码 SHA 前后相同；loader、checkpoint 和门禁未改。随后 CPU 审计 **45/45 个不同路径通过**，检查全部模型/optimizer Tensor 有限性、源码指纹、合法条件更新 counter 和各 rank RNG。不同运行即使 SHA 相同也分别检查，排除 symlink。

VADv2、Hydra 和 DiffusionDrive 的三个代表性断点均完成独立推理，并由未改动的官方 NAVSIM CPU 代码重评分。每个模型输出 **22 条唯一、有限的 40×3 轨迹**，训练与评价按整 log 隔离，评分覆盖全部 **22/22 有效 token**。三个推理任务和最终 CPU 评分任务均实际退出 0。

| 模型 | 本次断点训练范围 | 官方平均 PDMS |
|---|---|---:|
| VADv2 | 真实小数据八步更新 | 0.7851926239693333 |
| Hydra | 真实小数据八步更新 | 0.6084522121144075 |
| DiffusionDrive | 真实小数据三步更新 | 0.18729393472412512 |

上述评分来自三步/八步训练后的真实小数据，尚未达到收敛或驾驶质量验收条件。DiffusionDrive 的低评分保留原值。各断点的训练步数、batch 等工作负载不同，也没有 GPU 对照，这组结果仅记录各自测试，不能用于判定 GPU/NPU 等价。

首次私有 CPU 评分辅助脚本覆盖镜像原 `PYTHONPATH`，导致 `nuplan` 依赖不可见。复跑恢复固定镜像的继承依赖目录，使用相同预测文件和评分代码，原/重试代码 SHA256 相同。首次失败报告保留，生产 Python 和官方评分器均未改动。

## 原地图头的特殊布局

完整模型回归发现，原地图头投影四个 level，空间输入却只有单层 BEV。[MMCV 1.7.2 CUDA host](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu) 与 [kernel 索引](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh) 使用实际空间 level 数确定连续原始缓冲区步长，据此推断原 CUDA 采用跨 batch/query/head 的全局扁平前缀。该解释来自源码，尚无 GPU 实机对照。

已安装 SDK 的固定输入前向实验符合逐 query 前缀，与上述 CUDA 源码解释不同。新实现保留原参数、checkpoint 投影尺寸和推断的 CUDA 存储解释，仅在 MMCV 实际 L1/投影 L4、H8/P4/C32、连续浮点输入条件下启用；其他 level 矛盾会被拒绝。实现按布局结构判断，不按模块名称匹配。梯度返回原 L4 形状，未访问尾部精确为零。旧 SDK 诊断仅检查前向，未验证其完整后向或 GPU/NPU 对照。

## 如何复跑

在服务器进入当前仓库，使用空闲逻辑 NPU 和新输出目录，按 [英文说明中的完整 Docker 命令](MSDA.md#repeat-the-strict-gates-on-the-server)执行。三个脚本分别生成普通、地图特殊布局和完整 Q40000 结果：

```bash
python ascend/validate_msda.py --output /output/strict_matrix.json
python ascend/validate_msda_layout.py --output /output/msda_layout.json
python ascend/probes/full_shape_gate_probe.py --output /output/full_shape_gate.json
```

命令须在固定 NPU 镜像和环境中运行，并核对整体退出码 0 及报告中的每项结果。真实训练、原生 epoch、完整断点续训和独立评价见 [中文操作文档](README_zh.md)。完整续训会检查源码、原配置、数据、拓扑、优化器/LR 和镜像；旧源码断点不能直接作为新源码完整续训的输入。

完整 NAVTRAIN 收敛、八个真实合成配置、完整数据 CPU 内存/I/O/训练工期、GPU 对照和厂商正式训练支持仍待验收。八 rank 严格数值重放失败也需修复并按原门禁复验。MSDA 修复仅完成了该算子的验收，整套 AlgEngine 正式验收尚未完成。

## CPU 严格重放对照如何复跑

在固定镜像中，先 `import torch_npu` 注册官方反序列化入口，再用 `runpy` 调用原 `ascend/compare_resume.py`；由原脚本的 CPU `map_location` 激活官方 CPU 重映射，不修改 loader、checkpoint 或容差。完整可执行命令见 [英文操作示例](MSDA.md#repeat-cpu-replay-comparison-without-changing-the-comparator)，挂载源码和已完成的控制证据为只读，输出使用新目录，不挂载 NPU 设备。VAD 示例以同次连续运行自身的第 4 步作为前缀控制；Hydra 替换对应目录名即可。须同时检查 Docker 真退出码 0 和比较 JSON 的 `result="pass"`。

## 449.05 GB 图像包是否需要解压

`.tgz` 包需要解压。八个 current/history 包的压缩总量为 **449,046,313,693 字节**。当前加载器按路径读取单张相机图片，[NAVSIM v1.1 官方下载脚本](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_navtrain.sh)也逐包解压。若拷贝的是已展开的图片目录，可直接核验引用文件，无需重复解压或下载。

展开后的占用以及 WorldEngine 标注、历史帧和八相机引用的覆盖情况仍需实测。449.05 GB 仅指这八个压缩包，整套 SimEngine/WorldEngine 发布资产约 5 TB。容量预算还需包含训练标注/PDM、独立 NAVTEST、地图/metadata、checkpoint 和临时文件；逐包处理可减少压缩副本的保留空间。本次修复没有下载这批大图像包。
