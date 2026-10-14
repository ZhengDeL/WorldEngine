# WorldEngine AlgEngine：Ascend 910C 部署、训练与验收操作文档

首次使用请从[按原仓流程编写的NPU使用指南](../docs/algengine_ascend_zh.md)开始，包括镜像、数据准备、训练、续训、评价和微调。本文保留各版本实测与验收记录。

## 2026 年 10 月 14 日验收现状

AlgEngine在910C上已有训练、微调、推理和完整检查点恢复实测。最新完成的16芯片同步执行对照中，**VAD和Hydra均通过原严格数值门槛，每个模型各完成一对**。这些结果适用于下述显式同步配置，整套AlgEngine正式验收尚未完成。

运行源码指纹为 `cc91deb86ab5f4a8d08e5009f8ac396d06a7ebbf03d4ed83e38a04a50d2b1d88`（243个运行Python文件），镜像ID为 `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45`，使用 `torch_npu 2.7.1.post8`。8张物理卡、16个逻辑芯片，FP32，每rank batch=1、workers=0；在Torch启动前显式设置 `TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=1`。数据为NAVTEST派生小样本，42个训练锚点、22个独立评价锚点，未覆盖完整NAVTRAIN。

每对先连续训练8次更新，再从该次训练自身iter4完整恢复到8，比较两次最终检查点。原Tensor容差 `atol=1e-6, rtol=1e-5` 及loss门槛 `1e-4` 均未改，模型与优化器Tensor均无超容差项。

| 同步执行16 rank对照（各一对） | 严格数值结果 | 模型最大绝对差 | 联合Tensor对照最大差 | 最大loss差 |
| --- | --- | ---: | ---: | ---: |
| VAD | 通过，全部逐位相同 | 0 | 0 | 0 |
| Hydra | 通过原容差，非全部逐位相同 | 2.9802322387695312e-8 | 2.9802322387695312e-8 | 0 |

联合对照包含模型、优化器及rank RNG。VAD为1,698/1,698个Tensor逐位相同；Hydra为1,468/1,778个逐位相同，其余差异仍在原容差内。两组的batch token、LR和最终各rank RNG相同。原始结果见[同步VAD实测](results/sync-vad-16chip-20261014.json)及[同步Hydra实测](results/sync-hydra-16chip-20261014.json)，操作步骤见[同步复跑文档](execution/README_zh.md)。公开辅助器仅通过本地命令检查，尚未直接在NPU执行；上述实测来自此前受控启动流程。

**验收范围：每个模型仅有一对同步执行正例，尚未完成多次独立重复或该环境的长时间训练。默认Queue/Blocking均未设置时仍为一对通过、三对失败，其严格数值问题未闭环，见下方默认执行对照。**同步结果不能据此推广到默认方式、其他配置或完整NAVTRAIN收敛。MSDA严格位置梯度已按原容差通过，默认方式的数值分叉根因尚未确认。

| 五项工作 | 已有实测 | 尚需完成 |
| --- | --- | --- |
| 真实数据 | 完整NAVTRAIN标注/PDM两个PKL共39,862,469,866字节已取得并校验。元数据检查覆盖246,165个唯一帧、1,192个日志及原103,288个split token；六项PDM共619,728个数组均为有限 `float32[8192]`，范围0–1。 | 完整当前/历史相机及独立完整NAVTEST/cache。此次QC读图数为**0**，1,969,320条有效相机路径引用不等于JPEG存在或解码通过。 |
| 8个合成配置 | 两个配置已用一个真实场景及原WE派生8192奖励完成公开准备工具CPU链、原getitem四分支、单rank及双rank连续/恢复训练，原严格比较均逐位相同。 | 其余六个rollout/replay/BWM配置的真实导出尚未齐备。每配置一个normal锚点的窄检索池、原WE四秒记录交通协议，不等价于SimScale v2公开奖励或全数据采样。 |
| 正式训练与独立质量 | 小数据训练、原评价器及此前独立官方CPU评分链已执行。 | 完整NAVTRAIN原horizon及收敛、完整独立NAVTEST common/rare质量、GPU输出对照均未验收。 |
| 容量与资源 | 1、8、16个独立原dataset/Compose完整PKL构造与同时驻留均通过；保留原50%过滤，每份246,165帧、50,000锚点及103,288 PDM项。16份合计RSS为834,131,996,672字节。 | 完整图像、模型/优化器及worker训练内存、实际I/O。构造实测使用旧858源码/post6和明确的SDK导入期元数据桥，不能写成无桥CPU部署通过，也不是训练内存。 |
| 续训与稳定性 | cc91/post8同步执行的VAD、Hydra各一对16 rank严格重放通过。旧858/post6八rank感知训练超过8小时，每rank1,023次更新；从自身最终断点恢复3次，完成37帧独立感知评价及两个检查点CPU审计。 | cc91/post8同步环境的独立重复与持续训练，以及默认方式的16 rank严格数值问题；未验收24小时、完整数据或普遍数值重放。 |

完整元数据QC的顺序处理峰值RSS为34,747,150,336字节，与原dataset构造不同。原单份构造峰值约52.208 GB，8份合计RSS为416,754,966,528字节。16份构造不含完整训练图像解码或NPU模型计算。第七项 `driving_direction_compliance` 仅做字段结构清点，不能写成七项数值指标都通过。

八小时结果属于 `858f9d…`（242文件）、`a182…`镜像及 `torch_npu 2.7.1.post6`，最短rank训练28,827.677454秒。四个训练/恢复/推理/CPU审计容器均退出0、无OOM；服务器补检16份完整rank JSONL，共8,208条连续更新记录、837,216个原始loss标量，全部有限。梯度有限值门禁在首步及每10步执行，没有每步全梯度转储。CPU RNG审计检查字段、类型及rank数量，不证明RNG相等或严格数值重放。该批旧运行没有新增逐rank精度前后回读，不能把八小时通过挂到cc91/post8。

真实小样本准备工具有独立的三CLI SHA清单，公开输入合计47.50 GB，maps、源码、镜像及新输出另计。不要求作为历史兼容性调查用的19.3 GB不兼容16384奖励、不要求下载全部449 GB训练相机或5 TB资产。Finetune明确覆盖为官方100pct checkpoint，原 `e2e_vadv2_ep8.pth` basename仍未找到，不能改名冒充。复现结果见 `data_tools/CPU_REPRODUCTION_VERIFIED.json`、`GETITEM_CPU_VERIFIED.json`、`NPU_SINGLE_RANK_VERIFIED.json` 和 `NPU_TWO_RANK_VERIFIED.json`；完整元数据与旧八小时补充文件分别为 [完整NAVTRAIN元数据QC](results/full-navtrain-metadata-qc-20261014.json)及[旧源码八小时稳定性](results/eight-hour-perception-stability-20261014.json)。

原44配置中，42个规划配置每GPU batch=2，两项感知配置batch=1。规划原8GPU全局batch=16，本次16 NPU、B1同样为16，只代表样本数量一致，不代表分组或数值等价。旧八rank长训实际使用8张物理卡各一个芯片；真实双rank对照使用两张物理卡各一个芯片，未证明一张物理卡双芯布局。4张物理卡/8芯片B2的容量及多机扩展仍须独立验收。

性能专项优化按用户要求后续单独开展。当前记录正确性、功能、基础容量和有限时长稳定性，不把历史速率当成GPU/NPU成对性能对照。

### 默认执行方式的16 rank严格对照（问题未闭环）

以下四对对照发生在同步候选之前，使用相同cc91/243源码、post8镜像及原容差，`TASK_QUEUE_ENABLE` 和 `ASCEND_LAUNCH_BLOCKING` 均未设置。八个训练/恢复任务全部正常完成，并通过运行与精确断点载入检查；最终连续训练/断点续训Tensor对照为**一对通过、三对失败**。

| 默认执行16 rank对照 | 严格数值结果 | 模型最大绝对差 | 联合Tensor对照最大差 | 超容差模型 / 优化器Tensor数 |
| --- | --- | ---: | ---: | ---: |
| VAD第1次重复 | 失败 | 4.231929779e-6 | 5.522248102e-6 | 6 / 7 |
| Hydra第1次重复 | 失败 | 6.511807442e-6 | 1.536439959e-5 | 7 / 14 |
| VAD第2次重复 | 失败 | 2.504140139e-5 | 4.742613237e-5 | 12 / 25 |
| Hydra第2次重复 | 通过，逐位相同 | 0 | 0 | 0 / 0 |

这里的失败指最终Tensor差异超出原容差，训练完成和精确载入检查点均已通过。较小的绝对差不能替代逐元素容差检查，失败结果仍保留，见[默认方式原严格重放结果](results/strict-replay-post8-20261014.json)。同步候选随后各通过一次，但尚未证明默认方式的问题已修复，也未确定数值分叉的根因。


### 追加短时控制的实际状态

post8 16 rank全状态观察任务已完成VAD/Hydra四个训练/恢复任务及CPU检查点身份审计，两组原严格数值比较均失败。完整状态JSON关联已完成：每个模型144项关联，blocked=0，各有16项Tensor差异，全部位于最终update8。已记录的update5前注册model/optimizer状态与布局相同，后续执行仍出现分叉。这不能证明未注册缓存或全部后端状态相同，根因尚未确定。四个任务正常完成，源码仍为cc91/243，结束时无自有任务运行、无清理错误；见[完整状态诊断](results/full-rank-state-diagnostic-20261014.json)。这项诊断把NPU状态拷到CPU，可能改变时序，应与无观察器复跑分开记录，不能据此宣称修复。

另一节点上的同步执行候选 `TASK_QUEUE_ENABLE=0, ASCEND_LAUNCH_BLOCKING=1` 冷启动遇到 `EL0020 / Device0 occupied by multiple containers / 507899`。实际更新数为0，没有数值对照结果。没有reset、开启共享模式或停止共享任务。芯片健康、宿主PID与设备FD查不到活动，不足以证明驱动能初始化。

第一节点的独立无观察器单VAD同步对照已终止：连续训练完成8次更新，actual exit0，16份原运行门禁通过；从自身iter4恢复后只完成2次更新到iter6。原helper的120秒截止于01:38:10 UTC触发自有任务停止，恢复容器exit137、无OOM。没有完成到iter8，也没有运行原严格比较器，因此既不是数值PASS，也不是数值FAIL。该结果仅为部分运行，不覆盖Hydra或两次重复。

第一轮短时设备窗口已结束。自有任务全部释放，无剩余自有运行或清理错误，独立watchdog完成且stops列表为空；较早停止来自原helper。最后会话于01:39:59 UTC关闭，早于01:40:10授权截止。没有为延长时间替换或重启运行中的v1；统一60秒门禁的v2没有在该窗口执行。

随后截至02:00 UTC / 10:00 CST的追加窗口中，一对全新无观察器VAD对照已完整执行：连续训练1–8次更新，从自身iter4恢复到8。仅在初始环境设置 `TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=1`，两个训练容器均退出0，32份rank运行门禁全部通过；64次精度前后回读均为FP32、两个公共HF32标志关闭，确定性环境符合要求。

使用官方Torch-NPU CPU加载器执行未修改的数值比较器，889个模型Tensor及1,698个完整Tensor（含rank RNG）全部逐位相同，最大差和loss差均为0；源码仍为cc91/243，无剩余自有任务或清理错误。这是一对VAD候选的实测正例，不是四对重复或普遍确定性验收，见[同步VAD实测](results/sync-vad-16chip-20261014.json)。

另一对Hydra对照也已完成两个训练任务、32份rank运行门禁及64次匹配的精度前后回读，使用相同同步环境，源码和镜像不变。原严格比较在原容差内通过：模型及完整Tensor的最大差均为 `2.9802322387695312e-8`，loss差为0。909个模型Tensor中868个逐位相同，1,778个完整Tensor中1,468个逐位相同，**Hydra并非全部逐位相同**；batch token、LR及最终各rank RNG相同，见[同步Hydra实测](results/sync-hydra-16chip-20261014.json)。

这两项是VAD、Hydra各一对、各一次重复的候选正例，没有完成四对重复，也不证明根因、普遍确定性或完整数据验收。两对结束后均无自有任务运行及清理错误；Hydra watchdog正常完成，stops为空。完整CPU输入诊断控制器仅本机准备，未执行。生产默认环境和原数值门槛未改。

候选复跑步骤见[同步操作文档](execution/README_zh.md)。公开辅助器已通过本地命令构造检查，尚未直接在NPU执行；上述NPU结果来自此前受控启动流程。


2026 年 10 月 13 日的 FP32 MSDA 修复已通过严格位置梯度检查，原容差未改。实现、模型回归和资源实测见 [MSDA 修复说明](MSDA_zh.md)及[对应结果](evidence/msda-precision-20261013.json)。下文 `96a7bef…` 指纹、一小时吞吐/内存及旧门禁失败属于此前 fused 实现，当前源码的结果另行记录。

历史快照截至 2026 年 10 月 13 日 22:44， `2bed8f4d7c78` / `858f9d…` FP32 版本已完成 VADv2、HydraMDP 的八 rank 断点重放。各模型连续训练 8 步，再从同一次训练的第 4 步完整检查点恢复至第 8 步。四个任务均实际退出 0，32 份 rank 运行门禁通过；载入断点时，每 rank 分别逐张量核对 889/777、909/837 个模型/优化器张量，完全一致。

**两组最终严格数值对照均失败。**原 `atol=1e-6`、`rtol=1e-5`、loss 差 `<1e-4` 未改。batch token、LR 和最终 RNG 均一致，部分模型和优化器张量仍超出容差。

| 八 rank 对照 | 模型最大绝对差 | 模型/优化器/RNG 联合最大绝对差 | 超容差模型 / 优化器张量数 | 最大 loss 差 |
|---|---:|---:|---:|---:|
| VADv2 | 4.546716809e-6 | 2.530345228e-5 | 12 / 21 | 1.907348633e-6 |
| HydraMDP | 2.862885594e-6 | 2.862885594e-6 | 2 / 2 | 9.536743164e-7 |

测试条件、失败张量路径及原比较器的计数/列表显示限制见[八 rank 夜间实测](evidence/eight-rank-resume-20261013-evening.json)。数据来自 NAVTEST，按驾驶日志划分为 42 个训练锚点和 22 个独立评估锚点，1126 个文件哈希已核对。该测试未验证正式 NAVTRAIN 收敛。当时感知长训在22:44仍运行；后来独立完成的旧源码八小时任务见本文新验收段。此前主动停止的任务不记为八小时通过。性能专项优化已延期；当前继续检查正确性、功能、容量和稳定性，完整数据的 CPU RAM、I/O 与设备需求还需实测。

本文与 [英文操作文档](FULL_ALGENGINE.md)、[三个规划入口的复跑说明](README.md) 使用同一份代码和实测证据。已发布迁移代码支持下述运行方式；目前真实小数据功能部分通过，完整训练与质量尚未验收。全部数据、精度模式和框架配置仍需分别验证。

## 1. 先明确模块、版本与验收对象

WorldEngine 包含两个模块。SimEngine 负责仿真与场景生成，继续使用 GPU，本次没有部署。AlgEngine 负责感知预训练、端到端轨迹规划、训练和微调，是本次 NPU 适配对象。使用 SimEngine/SimScale 导出数据的 AlgEngine 配置也需要验证，前提是取得真实导出图像、标注和奖励。

| 固定项 | 值 |
|---|---|
| 上游 WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` |
| 迁移分支 | [ascend-910c-algengine](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine) |
| 历史实测代码提交 | `f804a8754e6c08d07e5302599dda6b068a41586c`；对应下列历史 Python 指纹 |
| 历史 AlgEngine/Ascend Python 指纹 | `96a7bef17fd27a681b5f90c367507a3a9f72767e5698fbe514d077d20ec0f997`，239 个 Python 文件，路径排序且 CRLF 规范为 LF |
| 历史 MSDA 修复及夜间测试版本 | `2bed8f4d7c78`；Python 指纹 `858f9d01181d3d62c35d0635881b22c62daf1a5f5af8123f622055de05da422d`，242 个文件，详见 [MSDA 说明](MSDA_zh.md) |
| 官方 NAVSIM | v1.1，`3e8291bfa89ff247231e0227778840cd0a036896` |
| 实测镜像 | `local/worldengine:26.1.0-a3-clean-20261013` |
| 实测镜像 ID | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` |
| 历史机器可读验收结果 | [full-algengine-20261013.json](evidence/full-algengine-20261013.json) |
| 原始 44 个配置的契约 | [original-configs-20261013.json](evidence/original-configs-20261013.json) |

实测设备为 Atlas 800I A3 推理系列、Ascend910_9362，8 张物理卡、16 个逻辑芯片，每个逻辑芯片 64 GiB HBM。实验已完成反向传播。

## 2. 已经跑到什么程度

测试保留原模型结构，包括 200×200 BEV、900 个 tracking queries 和 8 相机。规划训练使用 4 帧时序；感知训练原配置使用 3 帧队列，评价补充历史后处理 37 帧。训练使用真实 JPEG、原标注与 PDM，42 个训练锚点来自两段日志，22 个独立评价锚点来自第三段日志，日志互不重叠。该小数据集用于功能验证，未覆盖完整 NAVTRAIN 分布。

下表记录历史 `96a7bef…` 版本的测试。当前 MSDA 修复后的结果见 [MSDA 说明](MSDA_zh.md)，旧八rank重放失败见上方历史快照及[对应 JSON](evidence/eight-rank-resume-20261013-evening.json)。

| 验证项 | 实际结果 | 适用范围与未验收项目 |
|---|---|---|
| 44 个公开配置 | 全部在固定环境解析；36 个真实数据配置各完成 3 次原模型优化器更新，使用各自原始初始化权重 | 覆盖 VADv2、HydraMDP、DiffusionDrive、两种 UniAD 感知配置；B1/workers0、小数据过滤及短 LR 是明确覆盖项，原数据比例与收敛未验收 |
| 8 个合成数据配置 | 缺真正导出数据，未执行训练 | 没有用真实 mini 冒充合成数据通过 |
| 原生感知 epoch 与续训 | 2 rank，epoch 1 每 rank 21 次更新；完整恢复后到 42；两个 epoch 都完成原始 detection/mapping 评价 | 恢复 1393 个 model、3594 个 optimizer Tensor，载入后逐项相同；mAP 为 0，质量不通过 |
| 规划 epoch 与独立评分 | 22/22 输出由原 NAVSIM CPU 评分；VAD reward/LoRA 两个 epoch 的 PDMS 为 0.845583、0.885092 | 小数据上升不等于完整基准提升；本次独立集 rare token 数为 0 |
| 控制变量数值续训 | 从同一连续训练的断点恢复，VAD reward/LoRA 的 model 最大差 7.45e-9、model/optimizer 最大差 2.98e-8、loss 差 0 | 通过既有 atol=1e-6、rtol=1e-5、loss=1e-4 门禁；历史 base/Hydra 的失败仍保留 |
| DiffusionDrive 8 rank | 每 rank 3 次更新；22/22 有限的 40×3 轨迹，官方 PDMS 0.377859，ADE/FDE 10.96562/19.69085 | 规划 head 仅短训练，不能作为成熟模型质量 |
| 感知 8 rank 持续训练 | 每 rank 428 次更新，最短 3610.270 秒，全部正常退出且没有 OOM | 重复 42 个训练锚点的一小时实测，不是完整数据吞吐、24 小时或内存泄漏排除 |
| 一小时资源 | 每芯片峰值 allocated/reserved 43.68/47.73 GiB；观察到 CPU cgroup 高水位 63.23 GiB，容器上限 128 GiB | cgroup 含被计费缓存；当时未测完整NAVTRAIN。后续原构造器容量另列，仍未测完整训练内存 |
| 多卡参数同步 | 8 rank、3 次原生更新后，1233 个参数 Tensor、79,777,383 个元素逐位相同 | 只声明参数一致，不声明 rank 本地非参数 buffer 一致 |
| 实际 8 rank 断点续训 | 从一小时训练的 iter_100 完整断点恢复，100→103；所有 rank 恢复 model/optimizer 和 8 份 RNG | Adam 条件参数合法计数为 [91,99,100]，不是每个参数都等于全局更新数；没有宣称全部步骤严格数值重放 |
| checkpoint 独立检查 | 40 个功能断点及 5 个一小时断点通过 CPU 序列化、有限性、计数/RNG 检查 | 不替代后续完整数据收敛验收 |
| 独立 CLI 感知推理 | epoch 续训模型和一小时最终模型均处理 37 帧，370 个实际预测字段有限，原评价器执行 | 空 map union 和原协议不适用项记 null 并保留覆盖数，不用 0 冒充有效指标 |

一小时端到端吞吐为 **0.94808 锚点曝光/秒**，包括真实取数、训练、监测和断点开销；前 20 次更新之后为 0.95762。428×8=3424 次曝光包含重复样本和 sampler padding。已记录的 update p50/p95 为 6.21183/7.00874 秒，其计时不包含取 batch，不能与端到端速率混为同一指标。没有 GPU 设备，当前没有 GPU/NPU 性能或模型轨迹对照。

## 3. 干净环境与镜像复现

以下命令在 **Linux 服务器**执行，示例目录可按实际交付路径替换。数据、权重下载与解压均留在服务器，不需要在 Windows 下载 5 TB。

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git rev-parse HEAD
git submodule status
npu-smi info
docker version

IMAGE=local/worldengine:26.1.0-a3-clean-20261013
IMAGE_ID=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
```

复现已测试镜像，优先使用交付的 `worldengine_clean_20261013.tar`。镜像没有宣称上传公开镜像仓库；离线包应由交付位置拷贝至服务器。其大小为 **23,444,539,904 bytes**，SHA256 为 `5c730fd509d9dea6ac1599094c7e1e60ab0fad78996031fcb46e7571988c5cac`。

```bash
sha256sum worldengine_clean_20261013.tar
docker load -i worldengine_clean_20261013.tar
docker image inspect --format '{{.Id}}' "$IMAGE"
# 必须与 IMAGE_ID 一致；launcher 也会校验。
```

有镜像仓库和 Python/Conda 下载访问条件时，可以按已公开 recipe 从固定官方 ARM64 基镜像重新构建：

```bash
docker build --no-cache --platform linux/arm64 \
  -f ascend/docker/Dockerfile -t local/worldengine:rebuild ascend/docker
docker image inspect --format '{{.Id}}' local/worldengine:rebuild
```

recipe 固定基镜像 manifest 和主要依赖，但重新解析软件包渠道不保证新 image ID 与离线包逐位相同。新镜像应记录实际 ID、检查包版本并重新跑功能门禁；使用 `--image` 和 `--expected-image-id` 指定它，不能直接把旧镜像的验收结论改写成新镜像已验收。

实测栈：驱动 26.1.1、CANN 9.1.0、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post6、mx_driving 26.1.0.post2、MMCV full 1.7.2、MMDet 2.24.0、MMDet3D 1.0.0rc4、NumPy 1.23.5。`+cpu` 是此 NPU 发行栈的版本标识，不表示训练跑在 CPU。训练 Python 为 3.10.21，单独 geospatial prefix 为 3.10.22；版本快照见 [docker/versions-20261013.json](docker/versions-20261013.json)。这份旧post6/source858快照使用FP32 Tensor、matmul HF32禁用、convolution HF32允许。当前cc91的FP32入口显式禁用两项HF32，并保存每rank实际前后回读。不同source/精度/backend合同的旧断点不能绕过恢复门禁；没有GPU数值等价或AMP/BF16验收结论。

启动器只读挂载源码、数据、地图和初始权重，将日志、checkpoint、CANN 编译文件及缓存写入独立 output。宿主共享包、共享服务和 NAS 挂载由运维管理，启动器不改动这些内容。每次运行使用不存在的新 output，保留原失败记录和对照结果。

## 4. 准备哪些数据、权重与存储

小数据功能验证使用已交付的 `mini_real_data.tar.gz`、原权重、8192 词表、地图及 22 个官方 scenario cache。压缩包约 242.3 MB，包含标注引用的 1120 张原始 JPEG，来自 NAVTEST 按驾驶日志提取的子集，与官方 OpenScene mini 数据集不同。移动数据时保留标注中的相对相机路径；重新打包若改变标注哈希，应开始新训练。

完整数据目录约定如下：

```text
/datasets/worldengine/data/
  alg_engine/
    merged_infos_navformer/
      nuplan_openscene_navtrain.pkl
      nuplan_openscene_navtest.pkl
    pdms_cache/
      pdm_8192_gt_cache_navtrain.pkl
      pdm_8192_gt_cache_navtest.pkl
    test_8192_kmeans.npy
    ckpts/
  raw/
    openscene-v1.1/
      sensor_blobs/trainval/       # 所选当前帧和历史帧的 8 相机文件
      sensor_blobs/test/
      meta_datas/trainval/         # 官方 scenario cache 生成所需真实元数据
      meta_datas/test/
    nuplan/dataset/maps/           # nuplan-maps-v1.0.json 及引用的 .gpkg
```

| 输入 | 已核实大小或用途 |
|---|---|
| NAVTRAIN 标注 + PDM 两个 PKL | 39,862,469,866 bytes，序列化文件大小，不等于内存需求 |
| NAVTEST 标注 + PDM 两个 PKL | 3,639,831,385 bytes |
| 8192×40×3 轨迹词表 | 3,932,288 bytes；SHA256 见 lock |
| 五个本次用到的官方权重 | 合计 1,769,585,126 bytes |
| 定向 NAVTRAIN 图像包 | 官方NAVSIM v1.1脚本列出8个current/history包，HEAD观察合计449,046,313,693 bytes压缩；完整图像引用与解码覆盖尚待核验 |
| NAVTEST 图像、真实 metadata、地图 | 必须补齐独立评价和历史引用；图像包文件名存在不代表标注引用完整 |
| 合成/SimScale 消费配置 | 还要真正导出图像、转换后的 metadata、plan_idx、匹配的 rewards；见第 9 节 |

固定文件大小与 SHA256 见 [assets.lock.json](assets.lock.json)。VADv2/HydraMDP 选择式训练需要所选 token 的六组真实 8192 维 PDM 指标。感知和 DiffusionDrive 训练不需要这些选择式 PDM targets。独立规划评分另需官方 scenario metric cache，两类 cache 用途不同，不能互换。

相机 pipeline 不读取原始 LiDAR，所需文件可按标注、时序和 8 相机引用定向拷贝，无需立即下载整个 5 TB 发布包。磁盘预算应包含压缩包、解压图像、NAVTEST、标注、cache、权重和 checkpoint，449 GB 的压缩包空间不足以覆盖这些文件。

### 国内来源与仅在服务器下载

已验证服务器可访问发布者的 [WorldEngine 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)、[OpenScene 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene)、[SimScale 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/SimScale)。hf-mirror 已实际下载两个新增官方初始化权重并通过原发布 SHA256；魔搭还完成词表 byte-range 前缀比对，这不等于魔搭整个大数据集已下载校验。

```bash
WEIGHTS=/datasets/worldengine/weights
python3 ascend/fetch_assets.py --group all --source hf-mirror --output "$WEIGHTS"
# 指定源不可用时，只下载同一组已锁定权重：
python3 ascend/fetch_assets.py --group all --source modelscope --output "$WEIGHTS"
```

`--group planners` 只取两个已发布 epoch-8 规划模型；`warmstarts` 取 BEVFormer 和 50%/100% 感知初始化；`all` 取五个文件。工具支持 Range 续传和最终大小/SHA 校验，输出 `official_planners.json`。不会自动下载图片、完整 PKL 或 SimEngine 资产。镜像构建/下载网络确实受限时，可用已提供的服务器代理，限定到当前进程，例如：

```bash
PROXY_URL=http://proxy-host:1080  # 替换为运维提供的实际代理
http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" \
  python3 ascend/fetch_assets.py --group warmstarts --source hf-mirror --output "$WEIGHTS"
```

这不更改系统或 Git 全局代理。hf-mirror 固定 WorldEngine 数据 revision `d0e88d858533ca407a3387a26577c083aa7b3617`；魔搭 master 仍用同一组原发布 SHA 限定内容。

## 5. 先做 CPU 输入门禁与内存检查

完整NAVTRAIN标注/PDM已取得并完成原数据构造与元数据检查。以下命令用于核验图像、地图及独立评价输入；实际训练内存仍需测量。1/8/16进程构造结果见本文开头，不能直接作为完整训练预算。

```bash
DATA=/datasets/worldengine/data
MAPS="$DATA/raw/nuplan/dataset/maps"
CACHE=/datasets/worldengine/metric_cache_navtest
RUNS=/experiments/worldengine
mkdir -p "$RUNS/preflight"
: "${PREFLIGHT_MEMORY_LIMIT:?先设置经过容量核对的单进程检查内存上限}"

docker run --rm --memory "$PREFLIGHT_MEMORY_LIMIT" --cpus 8 --entrypoint bash \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v "$PWD:/workspace/WorldEngine:ro" -v "$DATA:/data/full:ro" \
  -v "$MAPS:/data/maps:ro" -v "$CACHE:/data/cache:ro" \
  -v "$RUNS/preflight:/output:rw" "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    cd /workspace/WorldEngine
    python ascend/preflight.py --config configs/worldengine/e2e_vadv2_50pct.py \
      --data-root /data/full --maps /data/maps --metric-cache /data/cache \
      --require-official-score --hash-assets --ranks 8 --output /output/preflight.json
  '
```

检查 `result=pass_complete_inputs` 和退出码 0；缺文件、token、history、8 相机、PDM 维度/有限性、train/eval 日志重叠或官方 cache 覆盖时，报告具体原因并返回失败。`--hash-assets` 用于未改变的完整发布 PKL/词表，不用于重新打包的 mini。

`rss_bytes` 和 `peak_rss_bytes` 是实测值；`cpu_data_resident_rank_projection_bytes` 按检查进程 RSS×rank 数估算，尚未测量训练内存。训练预算还需计入模型/optimizer、评价、进程/runtime、worker 和余量。图像检查仅核对引用文件及大小，没有逐图像内容哈希或完整解码检查；地图也需在各地区评价中继续验证。合成数据另需构建 dataset，条件见第 9 节。

## 6. 小数据训练、微调与一小时复跑

下面使用交付的真实 mini 和固定 image。设备编号指宿主逻辑芯片，例子 0,2,...,14 使用每张物理卡的一个芯片；先确认 `npu-smi info` 无冲突任务。

```bash
MINI=/datasets/worldengine/mini_real_data
VOCAB=/datasets/worldengine/test_8192_kmeans.npy
VAD="$WEIGHTS/e2e_vadv2_50pct_ep8.pth"
HYDRA="$WEIGHTS/e2e_hydramdp_50pct_ep8.pth"

# 单芯片 imitation 功能检查，原感知初始化：
python3 ascend/run_container.py --devices 0 \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth" --vocab "$VOCAB" \
  --steps 3 --batch-size 1 --workers 0 --output "$RUNS/vad_short" --detach

# 已发布规划基座的 reward/LoRA 微调：
python3 ascend/run_container.py --devices 0 --profile vadv2_ft --runtime \
  --mini-prepared "$MINI" --checkpoint "$VAD" --vocab "$VOCAB" \
  --steps 30 --batch-size 2 --workers 0 --deterministic \
  --checkpoint-interval 10 --output "$RUNS/vad_ft" --detach

# Hydra 原配置包含实际 policy gradient；使用它自己的基座：
python3 ascend/run_container.py --devices 0 --profile hydramdp_rl --runtime \
  --mini-prepared "$MINI" --checkpoint "$HYDRA" --vocab "$VOCAB" \
  --steps 30 --batch-size 2 --workers 0 --checkpoint-interval 10 \
  --output "$RUNS/hydra_ft" --detach
```

微调 mini 按原 rare/common 语义构造训练选取；VAD reward/LoRA 原配置 PG 禁用，不能宣称两者均启用 PG。旧小数据独立评价中，VAD 基座/微调 PDMS 0.910051/0.876269，Hydra 0.934006/0.934006：功能运行成功，质量收益未验收，失败或退化结果都保留。

感知完整 backbone+tracking+map 路径，使用原 BEVFormer 初始化：

```bash
python3 ascend/run_container.py --devices 0 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 3 --batch-size 1 --workers 0 \
  --output "$RUNS/perception_short" --detach

# 感知 8 rank 一小时压力检查：10000 是 LR 总 horizon，3600 是实测时长停止条件。
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 10000 --upstream-schedule \
  --duration-seconds 3600 --batch-size 1 --workers 0 --memory 128g --cpus 64 \
  --checkpoint-interval 100 --check-interval 20 --output "$RUNS/perception_hour" --detach
```

128g 是本次 mini 一小时的实测容器配置，不是全数据预算。保留上游 optimizer LR/warmup 时，iter-based 压力 horizon 仍区别于原始 epoch 训练。单芯片短感知训练实测峰值约 43.61 GiB allocated；B1 适用，DDP 增加芯片不会自动减少每芯片模型内存。

原始 epoch 流程用 `--epochs` 而非 `--steps`。例如用 2 rank 保留 2-epoch horizon、在第一轮后停下：

```bash
python3 ascend/run_container.py --devices 0,2 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --epochs 2 --stop-after-epochs 1 --validate \
  --batch-size 1 --workers 0 --memory 64g --cpus 32 \
  --output "$RUNS/perception_partial" --detach
```

等待运行结束后再续训或复用设备。`--detach` 只表示启动容器；完整成功需同时检查退出码、真实更新、梯度/参数与推理评价，见第 10 节。

## 7. 完整状态断点续训

`--checkpoint` 是初始权重；`--resume` 才恢复 model、optimizer、每 rank RNG、采样进度、LR 与训练契约。旧的仅权重 checkpoint 不能替代完整状态。当前精确状态续训要求 workers=0，源代码、image、数据/过滤、world size、batch、seed、optimizer 和 LR 总 horizon 不变，output 必须换新目录。

```bash
# 接第 6 节的感知原生 epoch；保留 epochs=2 和其余参数，去掉 stop-after-epochs：
python3 ascend/run_container.py --devices 0,2 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --epochs 2 --validate \
  --batch-size 1 --workers 0 --memory 64g --cpus 32 \
  --resume "$RUNS/perception_partial/epoch_1.pth" \
  --output "$RUNS/perception_resumed" --detach
```

iter 训练同理：保持 `--steps 10000`，从一小时断点 `iter_100.pth` 恢复，测试可加 `--stop-after 103`，不保留 `--duration-seconds`。训练参数与原 run 相同。实际 8 rank 已完成该恢复及 3 次后续更新。

断点载入后的全部 model/optimizer Tensor 应与保存值精确一致。条件分支参数的 Adam 计数可以不同，但必须合法且不大于全局更新数。载入检查通过后，还需单独比较后续训练的数值结果；本轮八 rank 正是在该比较中失败。

严格数值对照以同一次连续 run 为基准：训练中途保存 checkpoint，再从这个 checkpoint 续训到同一终点，用 [compare_resume.py](compare_resume.py) 比较 model/optimizer、loss、tokens、LR 和 final RNG。CPU 容器命令见 [英文复跑说明](README.md#run-save-and-resume)。两次独立初始化可能在保存前就出现差异，不能据此判断恢复实现。

迁移工具还核对原配置文件的原始字节哈希。恢复旧断点时，使用交付的精确源码 tar 或实际训练源码目录。跨系统 Git checkout 可能改变配置换行；即使规范化 Python 指纹相同，resume 检查也可能拒绝载入。保留该检查，并查明源码或数据差异。新 clone 可用于开始新训练。

## 8. 推理和官方独立评分

推理使用训练输出的 `*_rank0.py`，其中包含已确定的实际数据路径与 head 类型。感知输出真实 boxes/scores/maps；DiffusionDrive 检查它生成的 40×3 轨迹；选择式规划还检查词表选取一致。不要把模型内部 PDM cache targets 当成独立质量结论。

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --config "$RUNS/perception_resumed/track_map_nuplan_r50_navtrain_50pct_rank0.py" \
  --mini-prepared "$MINI" --maps "$MAPS" \
  --checkpoint "$RUNS/perception_resumed/epoch_2.pth" --vocab "$VOCAB" \
  --output "$RUNS/perception_eval" --detach

# DiffusionDrive 的实际生成轨迹与 ADE/FDE：
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_100pct.py \
  --mini-prepared "$MINI" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth" --vocab "$VOCAB" \
  --steps 3 --batch-size 1 --workers 0 --inference-after \
  --memory 128g --cpus 64 --output "$RUNS/dd_short" --detach
```

为规划结果生成官方 CPU 评分，可使用固定 image、真实 logs/maps 与未改动的 NAVSIM v1.1：

```bash
SCORING=/datasets/worldengine/scoring  # 内含 navsim_logs/ 和 maps/
mkdir -p "$RUNS/dd_cpu_score"
docker run --rm --memory 24g --cpus 8 --entrypoint bash \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/source:ro" -v "$MINI:/mini:ro" -v "$SCORING:/scoring:ro" \
  -v "$RUNS/dd_short/inference:/predictions:ro" -v "$RUNS/dd_cpu_score:/output:rw" \
  "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:$PYTHONPATH
    cd /source
    python ascend/official_score.py --prepared /mini \
      --logs /scoring/navsim_logs --maps /scoring/maps --cache /output/cache \
      --predictions /predictions/predictions.pkl --output /output/official_score.json
  '
```

该评分按上游 `[4::5]` 从 40 个 pose 取 8 个，使用官方 cache generation、simulator 与 scorer。上例 24g 是小数据 CPU 评分模板，不能当作全数据评分内存预算。已经生成官方 metric cache 时，原生 epoch 的 `--official-metric-cache` 可以只读使用 payload，并在 output 重建搬迁后的 metadata 路径；每个 epoch 保留 predictions、逐 token 分数和整体/可用 rare 指标。

感知评价的协议不适用 TP 指标和真实 zero-union 项记 null，并保留 category 覆盖；其他非有限预测/指标返回失败。规划缺 token、非有限轨迹或官方 cache 不完整也返回失败。GPU 当前不可用，CPU 官方打分能验证 NPU 轨迹的业务指标，但不能代替 GPU/NPU 相同模型输出对照。

## 9. 真正合成数据与完整 NAVTRAIN 如何继续

当前两项SimScale配置已在一个真实场景、原WE四秒8192派生协议及每配置一个normal锚点上完成CPU、单rank和双rank实测。其余六个rollout/replay/BWM配置的真实输入仍未齐备；不把2/8写成全部配置已通过。需要 `--synthetic-root`，WorldEngine rollouts 包含 `sensor_blobs/`、`pdms_pkl/`、`meta_datas/`、`plan_idx.csv`；SimScale 使用 `meta_datas_navformer/`、per-folder reward PKL 与 split YAML。保留原配置的 folder_name 和过滤条件；缺根目录会提前失败。

Hydra BWM 原配置中的 `/path/to/synthetic/rollouts/...` 是占位路径，需要映射到真实导出目录。SimScale fine-tune原 `e2e_vadv2_ep8.pth` 尚未找到。本次明确使用官方 `e2e_vadv2_100pct_ep8.pth` override，按固定revision/434,885,463字节/fe86c0bf… SHA核验，没有改名冒充原基座。公开19.3 GB奖励实际为16384候选，字段与原六指标不同，原loader丢一项后为16383；8192词表完整轨迹行与16384词表精确匹配为0/8192，不能直接裁剪或假定映射。本次另按原WE四秒评分重新计算8192奖励，不宣称等价于SimScale v2。详细配置见 [原配置契约](evidence/original-configs-20261013.json) 和 [结果 JSON](evidence/full-algengine-20261013.json)。

完整 NAVTRAIN 收敛不依赖下载 SimEngine 的全部 5 TB，但确实依赖完整所选图像、标注、PDM、独立 NAVTEST 与官方评分 cache。数据到位后的顺序是：

1. 拷贝/下载到服务器，核对 lock、history、8 相机与独立日志划分；CPU preflight 不通过先修输入。
2. 复核已完成的1/8/16原dataset完整PKL构造与驻留容量，再实测完整图像/模型训练，记录rank RSS、cgroup、HBM及I/O。依据实际训练值设置上限，不能直接使用构造RSS或QC峰值。
3. 按各原配置完成全部 horizon。规划通常 8 epochs，感知 20/40，DiffusionDrive 100；每轮保留 checkpoint 与独立指标，验证中断恢复。
4. 核对 common/rare 覆盖与独立完整 NAVTEST 指标，不以训练 loss 下降代替质量。质量阈值与最终目标配置确定后才作正式通过判断。
5. 在目标 batch、worker、芯片布局与完整数据上持续训练，确认要求的运行时长、吞吐、内存和失败恢复；小数据一小时不能自动覆盖这些条件。

例如原始 VAD 50% NAVTRAIN 训练，保留原 8-epoch horizon、原 batch/optimizer/LR；以下必须等完整输入与容量检查通过后执行：

```bash
: "${MEASURED_CPU_MEMORY_LIMIT:?先设置依据完整数据实测的训练内存上限}"
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 --runtime \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --data-root "$DATA" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth" \
  --vocab "$DATA/alg_engine/test_8192_kmeans.npy" --maps "$MAPS" \
  --official-metric-cache "$CACHE" --validate --workers 0 \
  --memory "$MEASURED_CPU_MEMORY_LIMIT" --cpus 64 \
  --output "$RUNS/navtrain_vad50" --detach
```

要演练第 2 个 epoch 后中断，第一次增加 `--stop-after-epochs 2`；续训保持原 horizon、validate 与全部契约不变，用 `--resume .../epoch_2.pth` 和新 output。此处不使用 `--mini-prepared`、`--steps` 或三步短 LR。原始 50% filter 为 50,000 anchors，完整 NAVTRAIN YAML 为 103,288；小数据反复训练不能作为这些集合的收敛。

## 10. 怎样确认跑成功、需要几台设备

容器名为 `worldengine-npu-<output目录名>`。检查具体 run，不删除失败输出：

```bash
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  worldengine-npu-perception_hour
cat "$RUNS/perception_hour/exit_code"
tail -n 50 "$RUNS/perception_hour/console.log"
ls "$RUNS/perception_hour"/runtime_rank*.json
npu-smi info
```

`launch.json` 记录镜像、设备与命令；`runtime_rank*.json/jsonl` 记录更新数、原始 loss、梯度检查、tokens/LR、RSS/HBM 和恢复结果。核对这些记录后，再检查 `validation/epoch_*/evaluation.json`、官方评分目录与 `inference_validation.json`。Docker Up、单条 loss 或退出码 0 只说明其中一个环节的状态。NaN/Inf 会被记录为失败，原始 loss 检查不会用模型中的 nan_to_num 掩盖问题。

| 目的 | 当前有证据的设备/资源 | 尚待测量 |
|---|---|---|
| 小数据单模型功能、B1 感知 | 1 个 64 GiB 逻辑芯片即可运行完整结构；峰值约 43.61 GiB allocated | 更大 batch、AMP、其他芯片容量 |
| 历史8 rank感知稳定性 | 1台服务器，8张物理卡各取一个芯片，B1/global8；旧源一小时及八小时结果分别保留 | 新source/post8持续训练、4张卡双芯B2及多机扩展 |
| 原规划global16样本量 | 原8GPU/B2=16；本次16NPU/B1=16，功能/精确载入门禁通过；同步候选每模型各一对通过原数值门槛 | 默认post8仅1/4通过；同步候选重复验收、4张物理卡/8芯片B2容量仍待完成 |
| 完整NAVTRAIN | 完整PKL原构造1/8/16驻留已实测，16份RSS约834.13 GB；完整原split/PDM六指标QC通过 | 完整图像/模型/worker训练内存、I/O、耗时/收敛及所需机器数 |

各项1/2/8/16 rank小数据测试均为单机任务；8 rank旧长训每张物理卡用一个芯片，真实双rank对照为两张卡各一个芯片。16 rank任务使用单机8张双芯卡，默认执行仍有数值失败；同步候选已有两个单次正例，重复可靠性待验。完整数据所需机器数还取决于内存和交付时间。DDP 每 rank 保存模型副本并持有自己的数据对象，两个训练 PKL 的 39.86 GB 文件大小不能直接用作峰值 RAM。如果瓶颈在 CPU 内存或 I/O，增加 NPU 仍需同步解决数据加载问题。

## 11. 运行优势、代价与尚未通过项

迁移入口支持原配置、HCCL 多卡、完整状态恢复、原评价器和独立官方 CPU 打分。在没有 GPU 对照机的条件下，这些检查可以确认 NPU 功能和官方指标。真实小数据验证可以先行开展，镜像、数据和源码哈希用于核对版本；CPU 打分与 NPU 训练分开执行，保留上游评分语义。

旧 OpenMMLab/CUDA 依赖需要固定到已验证的 NPU 软件栈。解冻 backbone 时，fused MaxPool 缺少 backward，须使用 PyTorch NPU 路径。当前 FP32 MSDA 使用原生 NPU grid_sample 前后向、query 分块和后向重算，完整模型还保留矩阵求逆的 CPU fallback。新实现带监测的一小时速率为 0.35381 锚点曝光/秒。性能专项优化已延期，完整数据容量仍需实测。CANN 编译文件和 cache 写入可写 output；源码为只读挂载。CPU 大 PKL、地图和 I/O 的容量应单独计算。

`858f9d…` FP32 路径已按原 CPU32 参考和原容差，通过普通、特殊布局及完整 Q40000 几何的前向与三种梯度检查，seed29 超差元素由 35/19200 减至 0。两项单 rank VADv2/Hydra 受控续训也通过严格比较。历史 `96a7bef…` fused 路径的 35/19200 元素失败、最大差 0.0309448、relative L2 2.73677e-6，以及 atomic accumulation 观察和旧多卡续训失败，均保留为旧源码记录。

较早八rank全状态载入通过，后续严格数值重放失败：VADv2有12个模型、21个优化器张量超容差，HydraMDP为2个模型、2个优化器张量。根因尚未隔离，历史atomic accumulation观察不能直接解释新版本的失败。详见[八rank夜间实测](evidence/eight-rank-resume-20261013-evening.json)及[MSDA修复实测](MSDA_zh.md)。此处保留原配置结果，当前16rank默认与同步候选分别记录在本文开头。

正式验收还需完整NAVTRAIN图像训练及收敛、独立完整NAVTEST质量、其余六个真实合成导出配置、实际训练CPU/I/O/工期和目标源码持续稳定性。完整PKL构造及六PDM元数据QC已完成，不再写成尚未下载或尚未测量。旧858/post6八小时已通过，不能替代cc91/post8长训；默认post8 16rank数值为1/4通过，同步候选VAD/Hydra各一次通过，仍须完成重复回归并定位默认执行的分叉。性能专项优化后续单独开展。

GPU模型对照需要GPU设备。AMP/BF16、二阶梯度、多机及新源码多卡重复数值验收仍未完成；16芯片功能运行、精确恢复和两个同步单次正例均已实测。代码、操作方法和实测证据可供复跑，完整数据正式验收仍未完成。
