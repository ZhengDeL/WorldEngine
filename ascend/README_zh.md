# WorldEngine AlgEngine · Ascend 910C

[English](README.md) · [安装与使用](../docs/algengine_ascend_zh.md) · [验证报告](VALIDATION_zh.md)

MSDA 默认使用 `reference`；可通过 `--msda-backend cuda-compatible` 选择[CUDA-compatible 路径](MSDA_CUDA_zh.md)。

`ascend-910c-algengine` 分支提供 AlgEngine 的 NPU 训练、微调、断点恢复、推理和评价入口。SimEngine 保留原 GPU 仿真与渲染流程。

## 开始使用

在 Linux ARM64 昇腾服务器按以下顺序操作：

1. 构建[公开 post8 父镜像](docker/README_zh.md)，再构建[应用与编译工具依赖环境](docker/dependency_repair/README_zh.md)，执行三项 CPU 检查并记录实际镜像 ID。
2. 使用六个完整哈希锁定的官方原文件，准备[NAVTRAIN139 / NAVTEST22验证子集](data_tools/navtrain139/README_zh.md)。已有服务器缓存经校验后直接复用。
3. 下载[官方规划基座与词表](../docs/algengine_ascend_zh.md#2-数据下载权重和目录)。
4. 按[训练、续训、推理与评分步骤](../docs/algengine_ascend_zh.md#推荐流程navtrain139--navtest22)执行。默认精度为 FP32；BF16已有短周期实测，完整收敛与质量范围仍需验收。

139/22是从官方 NAVTRAIN/NAVTEST 确定性派生的验证子集，不是单独发布的标准 mini。原标签、8192候选顺序和图像字节保持；无缓存时需要六个原文件，共47.91 GB，最终只使用1832张 JPEG。此流程无需下载完整5 TB资产。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装与使用](../docs/algengine_ascend_zh.md) / [English](../docs/algengine_ascend.md) | 完整命令、原配置、数据准备、训练、续训、推理、评分和微调 |
| [依赖环境](docker/dependency_repair/README_zh.md) | 公开 Docker 配方、应用/工具解释器分离、wheel 锁和 CPU 验收 |
| [父镜像配方](docker/README_zh.md) | 官方 SDK 基座、post8 父镜像构建及离线转移 |
| [NAVTRAIN139 / NAVTEST22](data_tools/navtrain139/README_zh.md) | 六个官方原始输入、固定子集筛选与 CPU 重建实测 |
| [验证报告](VALIDATION_zh.md) / [English](FULL_ALGENGINE.md) | 分版本模型结果、数值对照、资源需求与待验收项 |
| [FP32 MSDA](MSDA_zh.md) | 实现、原前向/梯度门槛与复现方法 |
| [同步重放对照](execution/README_zh.md) | 历史八次更新 VAD/Hydra 专用对照 |
| [SimScale 数据工具](data_tools/README_zh.md) | 真实小样本转换和两个合成配置的数据流程 |

## 验证范围

| 项目 | 已有记录 | 尚未覆盖 |
| --- | --- | --- |
| 依赖环境 | 公开组合镜像的应用、所选 SDK 工具、原 CANN 重加载三项 CPU 检查通过；两个未过滤的 `pip check` 均退出0 | CPU通过不代表冷缓存 NPU 业务或全部可选 SDK 工具已验收 |
| NAVTRAIN139 / NAVTEST22 | 公开 CLI 从六个官方原文件重建；1832张 JPEG 完整 SHA/解码通过；六份派生元信息完整值、顺序及字节均与原子集一致 | 一个训练日志、一个评价日志；完整 NAVTRAIN 收敛和完整 NAVTEST 质量待验收 |
| CUDA-compatible MSDA | [MX/MMCV常规入口及MMCV raw-L4布局](results/npu-cuda-msda-20261017.json)满足原CUDA容差；独立CPU32超差结果保留；新CUDA金样导出到NPU的工具复现已完成 | FP32/C32、已测形状与一阶梯度 |
| CUDA-compatible VADv2／HydraMDP，FP32／BF16 | [2次更新、精确自身状态恢复、DDP4推理及官方评分](results/cuda-compatible-navtrain139-short2-20261017.json)，每组4rank | 实际8个训练锚点；四组复用同22评价锚点。不代表收敛、质量等价或16rank严格重放通过 |
| FP32 reference：10月16日 | 两原100%规划配置均完成[8次更新、自身断点恢复及22token推理/评分](results/fp32-navtrain139-short8-20261016.json)；历史微调分版本记录 | 所记录两组严格对照均完整完成且超原容差；恢复和分数测量不代表收敛或质量提升 |
| BF16 v7 reference：10月16日 | 两类官方100%规划配置完成[8次更新、自身恢复及22token推理/评分](results/bf16-v7-navtrain139-short8-20261016.json)；[全部16rank及实际检查点验证每模型138个梯度/Adam缺项恢复](results/bf16-v7-gradient-coverage-20261016.json) | 两组完整严格对照仍超原容差；继续为显式试验模式，不代表收敛或质量等价。存在覆盖缺陷的v6另存历史 |
| FP32／BF16输出对照 | 同一22个独立锚点完成[四组各880pose比较](results/fp32-bf16-v7-matched8-planning-20261016.json)，分别为相同官方权重与各自8次更新检查点 | 量测未设质量等价阈值；适配检查点包含训练运行差异，不是GPU／NPU对照 |
| 真实日志微调，BF16 v7 | [两个37/22配置均完成8次更新及自身4→8恢复](results/real-log-ft37-bf16-v7-short8-20261016.json)，VAD／Hydra续训张量及loss逐位一致 | 两个检查点均已完成独立22锚点推理及官方NAVSIM评分；[Hydra补充验证](results/hydra-ft37-bf16-eval22-20261017.json)的PDMS为0.933856，历史中断记录保留。此结果不代表NAVTRAIN139、合成配置覆盖或质量等价验收 |
| 合成输入 | 历史两个SimScale入口已有单场景、明确WE派生奖励协议的CPU及单/双rank结果 | 其余六个WorldEngine rollout/replay/BWM真实导出和完整发布协议未验收；37/22真实日志FT不计入合成覆盖 |

2026-10-16实测修复镜像为 `sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`。[构建记录](docker/dependency_repair/BUILD_VALIDATION.json)与[数据重建记录](data_tools/navtrain139/VALIDATION.json)列出实际输入和范围。重新构建的镜像、重新准备的数据都需要自身验证；文件同名不能替代断点兼容检查。

## 数据、权重与运行入口

[CUDA-compatible指南](MSDA_CUDA_zh.md#训练自身检查点恢复与推理)给出两模型、两精度已测的4rank／2次更新命令，包含独立官方评分。

[主指南](../docs/algengine_ascend_zh.md#推荐流程navtrain139--navtest22)给出两条独立调度路径：八次更新的短周期闭环在自身第4次更新处中断，原生八个 epoch 路径在第4个 epoch 处中断；各自继续断点恢复、独立推理和官方 CPU 评分。两条路径均可选择 FP32/BF16。恢复要求源码、镜像、精度、数据/filter、world size、batch和调度合同相同，workers为0，输出使用新目录。官方发布权重用于初始化新训练，不能作为迁移后的完整恢复断点。

原[NAVTEST-only smoke流程](../docs/algengine_ascend_zh.md#3-准备真实小数据完成单芯片八次更新)、全量数据入口和稀有日志微调命令仍保留。完整训练的 RAM、图像I/O和设备数量需按目标模型和数据实测，元信息文件大小不等于训练内存。

## 历史对照与未完成项

早期 `cc91…`/243文件源码、post8/3fef镜像的16 rank同步基座对照共四组，两组在原容差内、两组超限；默认执行环境的已完成对照也有超出原数值容差的结果。同步设置尚不能作为通用数值修复。2026-10-14公开镜像的 LoRA 实测见[独立记录](results/actual-public-reproduction-16chip-20261014.json)；当时镜像的16项依赖提示不代表本次修复镜像的 CPU 结果。

八小时感知训练属于较早的 `858f9d…`/post6环境，不能归到当前镜像。[官方100权重GPU／NPU FP32对照](results/gpu-npu-official100-eval22-20261016.json)中，两模型在同22锚点的选择轨迹和七项逐token官方指标一致。[原CUDA基线](CUDA_BASELINE_zh.md)补充完整MSDA梯度及八GPU短周期训练、恢复对照；完整比较中仍有数值超出原门槛，未解决NPU训练严格复现问题。完整NAVTRAIN图像训练与收敛、完整独立NAVTEST、成对GPU／NPU训练验证、更广范围的跨精度质量验证、重复数值复现和多机扩展仍需单独验收，详见[验证报告](VALIDATION_zh.md)。

提交问题时请提供源码提交/指纹、镜像 ID、配置、精度、逻辑芯片映射、数据 split、命令、`launch.json`、各 rank runtime文件及比较报告，并移除凭据与私有数据路径。
