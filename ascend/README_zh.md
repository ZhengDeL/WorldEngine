# WorldEngine AlgEngine · Ascend 910C

[English](README.md) · [安装与使用](../docs/algengine_ascend_zh.md) · [验证报告](VALIDATION_zh.md)

MSDA 默认使用 `reference`；可通过 `--msda-backend cuda-compatible` 选择[CUDA-compatible 路径](MSDA_CUDA_zh.md)。

`_npu.sh` 前端支持在 `--` 后传入 `--msda-backend reference|cuda-compatible`，也可设置环境变量 `MSDA_BACKEND`；显式参数优先。默认仍为 `reference`，训练、续训与推理须保持同一后端。本次已在 CPU 核对前端参数与原 Python CLI 的兼容性，新增入口尚无单独设备实测验收。

`ascend-910c-algengine` 分支提供 AlgEngine 的 NPU 训练、微调、断点恢复、推理和评价入口。SimEngine 保留原 GPU 仿真与渲染流程。

当前 NPU 验证优先完成 FP32。下文保留历史其他精度记录，新推理 journal 结果不扩展那些记录的验收范围。

## FP32 实测：2026 年 10 月 21 日

以下官方权重完整 NAVTEST 对照使用已留证的 `0488bfb7…`／245 文件 NPU 运行版本和原 CUDA 基线：GPU 为 8 rank，NPU 为 8 张物理卡上的 16 个逻辑 NPU 芯片／rank。这些结果不能作为当前公开 `76c51f77…` 源码的设备验收。表中 PDMS 使用 0–100 标度，原评分文件使用 0–1。

[实测结果（JSON）](results/fp32-validation-20261021.json) · [VADv2 全量逐 token 账本](results/fp32-full-navtest-vad-20261021.csv) · [HydraMDP 全量逐 token 账本](results/fp32-full-navtest-hydra-20261021.csv)

| 测量 | 实际结果 | 验收边界 |
| --- | --- | --- |
| VADv2，官方权重，完整 NAVTEST | 两端均完成 12,146 个锚点的推理及官方评分；选择轨迹和七项逐 token 指标的差异均为 0，两端 PDMS 均为 82.16216548 | 仅证明该源码、权重及数据下的输出和评分一致，不代表模型内部 Tensor 逐位相同 |
| HydraMDP，官方权重，完整 NAVTEST | 两端推理及官方评分均已完成；4 个样本的选择候选 ID 不同，4 条轨迹和 2 个逐 token 评分行超出原容差，轨迹最大绝对差为 5.454713821411133；GPU／NPU PDMS 分别为 91.45968923／91.45952568 | 逐 token 数值对照超出原容差：原比较器退出 2，控制器退出 3。七项聚合指标均在原 `atol=1e-6`、`rtol=1e-5` 门槛内，不能覆盖逐 token 超差；根因仍在定位 |
| MSDA，两组真实输入 tuple | 原前向及 value／location／attention 梯度对照均已完成，满足原容差 | 仅覆盖实测的两组 tuple 和一阶梯度，不代表全部模型 Tensor 或逐位等价 |
| 原生 NPU Adam，保存梯度对照 | 每模型累计完成 8 次受控优化器更新：两组保存梯度序列 × 两次原生重复 × 两步，均满足原算术容差 | 使用保存的 GPU 裁剪后梯度，不重放模型前向／后向、裁剪、Dataset 或 DDP，不代表完整训练或严格续训等价 |
| 当前公开源码 `76c51f77…`，单芯片短链 | 两模型均完成 2 次 FP32 更新及独立 22-anchor 推理；原脚本、容器、watchdog 和外层实际 wait 均为 0，完整保存的模型／Adam 状态为有限值，checkpoint 含单 rank 四源 RNG | World 1、batch 1、`cuda-compatible`，使用按原 NAVTEST 日志互斥派生的 42 个训练／22 个评价锚点；未验收独立续训、官方重评分、GPU 对齐、完整 epoch 或收敛，历史 `0488bfb7…` 结果单列 |

完整 NAVTRAIN 八个 epoch 的训练与收敛尚未验收。原生 GPU 重复训练和完整输入 pilot 已完成的数值超差记录继续保留，上述有限范围算子结果不解决这些差异。六组真实合成导出也仍未完成。

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
| [逐文件审查](FILE_AUDIT_zh.md) / [English](FILE_AUDIT.md) | 文件清单、修复内容、CPU 回归及验证边界 |
| [依赖环境](docker/dependency_repair/README_zh.md) | 公开 Docker 配方、应用/工具解释器分离、wheel 锁和 CPU 验收 |
| [父镜像配方](docker/README_zh.md) | 官方 SDK 基座、post8 父镜像构建及离线转移 |
| [NAVTRAIN139 / NAVTEST22](data_tools/navtrain139/README_zh.md) | 六个官方原始输入、固定子集筛选与 CPU 重建实测 |
| [完整 NAVTEST CPU 输入检查](../docs/navtest_input_audit_zh.md) / [English](../docs/navtest_input_audit.md) | 公开 CLI、官方 32 包锁定清单、完整 JPEG 解码与[输入及账本实测记录](../docs/navtest_public_input_results_zh.md)（[JSON](results/full-navtest-public-inputs-20261020.json)） |
| [验证报告](VALIDATION_zh.md) / [English](FULL_ALGENGINE.md) | 分版本模型结果、数值对照、资源需求与待验收项 |
| [FP32 MSDA](MSDA_zh.md) | 实现、原前向/梯度门槛与复现方法 |
| [规划推理断点续跑](../docs/algengine_journal_zh.md) / [English](../docs/algengine_journal.md) | 逐 rank journal、FP32 官方 22-anchor 恢复结果、五次运行复现与边界 |
| [同步重放对照](execution/README_zh.md) | 历史八次更新 VAD/Hydra 专用对照 |
| [SimScale 数据工具](data_tools/README_zh.md) | 真实小样本转换和两个合成配置的数据流程 |

## 验证范围

| 项目 | 已有记录 | 尚未覆盖 |
| --- | --- | --- |
| 依赖环境 | 公开组合镜像的应用、所选 SDK 工具、原 CANN 重加载三项 CPU 检查通过；两个未过滤的 `pip check` 均退出0 | CPU通过不代表冷缓存 NPU 业务或全部可选 SDK 工具已验收 |
| NAVTRAIN139 / NAVTEST22 | 公开 CLI 从六个官方原文件重建；1832张 JPEG 完整 SHA/解码通过；六份派生元信息完整值、顺序及字节均与原子集一致 | 一个训练日志、一个评价日志；该子集不能证明完整 NAVTRAIN 收敛或完整 NAVTEST 质量，官方权重全量对照另见上表 |
| 完整 NAVTRAIN 输入：10 月 20 日 | [103,288 个锚点和 1,219,960 张所需 JPEG](results/full-navtrain-inputs-20261020.json)完成 CPU 覆盖、逐文件内容哈希和解码检查；标注、历史图像及 PDM 无缺项 | 输入检查；检查器峰值 RSS 51.99 GB 不能作为训练内存预算，模型结果单列如下 |
| 完整输入 VADv2／HydraMDP，FP32／BF16：10 月 20 日 | [四组均完成 8 次更新及精确自身 4→8 恢复](results/full-navtrain-short8-20261020.json)，使用 2 张物理卡的 4 个芯片／rank；32 份 rank 报告通过，8 份 checkpoint 保存的模型与 Adam Tensor 均为有限值 | 完整 loader 可用，各组实际使用同一组 32 个当前锚点；不代表完整 epoch、收敛、新检查点 GPU 对照或严格重放等价验收 |
| 单芯片 FP32 容量：10 月 20 日 | [两类原 100% 模仿训练配置均完成 4 次更新](results/single-chip-capacity-20261020.json)，batch=1、完整 NAVTRAIN 输入，自有容器正常结束 | 实测分配 192g 内存／16 CPU；单进程峰值 RSS 分别为 50.69／50.79 GiB，不是最低内存或设备数，也不代表 BF16 容量与吞吐基准 |
| FP32 推理 journal：10 月 20 日 | [VADv2／HydraMDP 均通过官方 22-anchor、16 rank 的四路线和第五次正位置恢复检查](results/fp32-navtest22-journal-recovery-20261020.json)。最终全部字段、两条 journal 的四源 RNG 一致；第五次每 rank 跳过 1 个已保存位置并重播 1 步 | 22 个 anchor、32 个含补齐位置，其中补齐 10 个；原基线没有逐步 RNG。在 10 月 20 日快照中全量推理尚待完成，最新完整对照见上方 10 月 21 日表；该 journal 测试不推导训练或性能结论 |
| 完整 NAVTEST 输入：10 月 20 日 | [12,146 个锚点、136 个日志及 142,976 张所需 JPEG](results/full-navtest-inputs-20261020.json)完成独立 CPU 输入、内容哈希和完整解码检查 | 输入证据供六组评价复用；六组完整模型推理及官方 NAVSIM 评分另行验收 |
| CUDA-compatible MSDA | [MX/MMCV常规入口及MMCV raw-L4布局](results/npu-cuda-msda-20261017.json)满足原CUDA容差；独立CPU32超差结果保留；新CUDA金样导出到NPU的工具复现已完成 | FP32/C32、已测形状与一阶梯度 |
| CUDA-compatible VADv2／HydraMDP，FP32，10 月 18 日 | [16 rank 同前缀两步对照](results/cuda-compatible-fp32-strict16-resume2-20261018.json)：模型／全部张量对 889／1698、909／1778，均满足 `torch.equal`，最大张量差和 loss 差为 0 | 共享连续前缀，实际 32 个当前锚点；不代表独立初始化、更长训练或质量验收 |
| CUDA-compatible VADv2／HydraMDP，FP32／BF16 | [2次更新、精确自身状态恢复、DDP4推理及官方评分](results/cuda-compatible-navtrain139-short2-20261017.json)，每组4rank | 实际8个训练锚点；四组复用同22评价锚点。不代表收敛、质量等价或16rank严格重放通过 |
| FP32 reference：10月16日 | 两原100%规划配置均完成[8次更新、自身断点恢复及22token推理/评分](results/fp32-navtrain139-short8-20261016.json)；历史微调分版本记录 | 所记录两组严格对照均完整完成且超原容差；恢复和分数测量不代表收敛或质量提升 |
| BF16 v7 reference：10月16日 | 两类官方100%规划配置完成[8次更新、自身恢复及22token推理/评分](results/bf16-v7-navtrain139-short8-20261016.json)；[全部16rank及实际检查点验证每模型138个梯度/Adam缺项恢复](results/bf16-v7-gradient-coverage-20261016.json) | 两组完整严格对照仍超原容差；继续为显式试验模式，不代表收敛或质量等价。存在覆盖缺陷的v6另存历史 |
| FP32／BF16输出对照 | 同一22个独立锚点完成[四组各880pose比较](results/fp32-bf16-v7-matched8-planning-20261016.json)，分别为相同官方权重与各自8次更新检查点 | 量测未设质量等价阈值；适配检查点包含训练运行差异，不是GPU／NPU对照 |
| 真实日志微调，BF16 v7 | [两个37/22配置均完成8次更新及自身4→8恢复](results/real-log-ft37-bf16-v7-short8-20261016.json)，VAD／Hydra续训张量及loss逐位一致 | 两个检查点均已完成独立22锚点推理及官方NAVSIM评分；[Hydra补充验证](results/hydra-ft37-bf16-eval22-20261017.json)的PDMS为0.933856，历史中断记录保留。此结果不代表NAVTRAIN139、合成配置覆盖或质量等价验收 |
| 合成输入 | 历史两个SimScale入口已有单场景、明确WE派生奖励协议的CPU及单/双rank结果 | 其余六个WorldEngine rollout/replay/BWM真实导出和完整发布协议未验收；37/22真实日志FT不计入合成覆盖 |

2026-10-16实测修复镜像为 `sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`。[构建记录](docker/dependency_repair/BUILD_VALIDATION.json)与[数据重建记录](data_tools/navtrain139/VALIDATION.json)列出实际输入和范围。重新构建的镜像、重新准备的数据都需要自身验证；文件同名不能替代断点兼容检查。

## 数据、权重与运行入口

推理进度恢复使用 [schema2 journal 指南](../docs/algengine_journal_zh.md)。它记录逐 rank 的结果与 RNG，与模型／优化器训练 checkpoint 分别验收。10 月 20 日的 FP32 恢复实测限定为两个原规划配置、官方 22 个 anchor 和 16 rank；完整推理及官方评分仍需满足各自的覆盖与缓存门槛。

[完整输入短周期训练](../docs/algengine_ascend_zh.md#完整输入短周期训练)给出 10 月 20 日实测的 4 rank 参数，复现时使用新输出。两个 `e2e_*_100pct` 都是模仿训练配置，本轮没有启用 HydraMDP 策略梯度微调。服务器为 8 张双芯片物理卡、16 个芯片，本次矩阵使用其中 2 张物理卡的 4 个芯片。实测运行源码为 `0488bfb7…`／245 个 Python 文件，镜像 ID 沿用上文记录。

[CUDA-compatible指南](MSDA_CUDA_zh.md#训练自身检查点恢复与推理)给出两模型、两精度已测的4rank／2次更新命令，包含独立官方评分。

[主指南](../docs/algengine_ascend_zh.md#推荐流程navtrain139--navtest22)给出两条独立调度路径：八次更新的短周期闭环在自身第4次更新处中断，原生八个 epoch 路径在第4个 epoch 处中断；各自继续断点恢复、独立推理和官方 CPU 评分。两条路径均可选择 FP32/BF16。恢复要求源码、镜像、精度、数据/filter、world size、batch和调度合同相同，workers为0，输出使用新目录。官方发布权重用于初始化新训练，不能作为迁移后的完整恢复断点。

原[NAVTEST-only smoke流程](../docs/algengine_ascend_zh.md#3-准备真实小数据完成单芯片八次更新)、全量数据入口和稀有日志微调命令仍保留。完整训练的 RAM、图像I/O和设备数量需按目标模型和数据实测，元信息文件大小不等于训练内存。

[10 月 18 日 FP32 对照](results/cuda-compatible-fp32-strict16-resume2-20261018.json)另验证了 VADv2、HydraMDP 的 16 rank × batch 1：连续运行 0→2，再由新进程从同次运行的第 1 次更新检查点恢复至第 2 次更新。全部比较张量满足 `torch.equal`，最大张量差和 loss 差为 0。该同前缀对照实际使用配置 139 个锚点中的 32 个去重当前锚点，未新增推理或评分。版本条件见[源版本与检查点合同](MSDA_CUDA_zh.md#源版本与检查点合同)。

## 历史对照与未完成项

早期 `cc91…`/243文件源码、post8/3fef镜像的16 rank同步基座对照共四组，两组在原容差内、两组超限；默认执行环境的已完成对照也有超出原数值容差的结果。同步设置尚不能作为通用数值修复。2026-10-14公开镜像的 LoRA 实测见[独立记录](results/actual-public-reproduction-16chip-20261014.json)；当时镜像的16项依赖提示不代表本次修复镜像的 CPU 结果。

八小时感知训练属于较早的 `858f9d…`/post6环境，不能归到当前镜像。[官方100权重GPU／NPU FP32对照](results/gpu-npu-official100-eval22-20261016.json)中，两模型在同22锚点的选择轨迹和七项逐token官方指标一致。[原CUDA基线](CUDA_BASELINE_zh.md)补充完整MSDA梯度及八GPU短周期训练、恢复对照；完整比较中仍有数值超出原门槛，不构成历史八步重放差异已解决的证据。完整 epoch 训练与 NAVTRAIN 收敛、新训练权重的完整 NAVTEST、成对 GPU／NPU 训练验证、更广范围的跨精度质量验证、重复数值复现和多机扩展仍需单独验收，详见[验证报告](VALIDATION_zh.md)。

提交问题时请提供源码提交/指纹、镜像 ID、配置、精度、逻辑芯片映射、数据 split、命令、`launch.json`、各 rank runtime文件及比较报告，并移除凭据与私有数据路径。
