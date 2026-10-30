# NPU 场景回放与闭环规划器接入

[English](README.md)

## 显式启用 FP32 规划推理优化

当前运行代码新增外部 AlgEngine 推理优化入口，默认
`--inference-optimization none` 保留原规划路径。启用时，在受控启动计划的
**planner 阶段 argv** 中、`--cfg-options` 之前增加
`--inference-optimization point-parallel-last-frame --msda-query-chunk 8192`，
同时保留 `--device npu:0 --precision fp32 --msda-backend cuda-compatible`。
外部 AlgEngine 必须为源码指纹
`cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610`
的 249 文件版本，三个实际 helper 的完整 SHA 由 `planner_optimization.py` 固定。
外部仓库检出已核验公开提交 `a783047c92621ca3d2f658fbe054f200783e8e25`，
将同一值赋给 `ALG_OPT_COMMIT` 并传入 `--algengine-commit`；仅声明提交号不能代替源码检查。

本次优化配置限定下述八次控制评分窗口：规划器 `sim.max_frame_per_scene=12`，
仿真器 `--num-history 4 --num-future 8 --pdm-metrics --dense-rewards`。
完整 24 帧真实准备数据及原渲染 CANN 捕获证明保持不变，不裁掉评分所需未来帧。
渲染仍为 FP32，评分沿用原公式。新 SimEngine 指纹为
`9d09e7bb239fcb0238255e689423598115c0115841b47912b10ffbef80b1e541`
（213 个运行文件）。在提交 `7e9389e9a4f6206007bb48c6b82a80e3eac42b6a` 上，VADv2/HydraMDP × NR/R 四组实际 NPU 作业及公开 CPU 审计均通过，完整评分前缀输出与原 212/245 版本逐字节一致。

VADv2 NR/R 的 PDM×100 为 79.057/0，HydraMDP NR/R 为 100/27.233，原低分保持不变。上述四组中，VAD 两组实际各生成 9 份计划并补偿 54 次历史 GridMask 随机调用；Hydra 两组各生成 8 份并补偿 48 次；四组均只消费 8 份控制计划，优化回退数为零。随后四次同机同芯片端到端观察均完成，但序列曾中断，生成计划数为 9/8/9/8，故只在验证 JSON 保留实际耗时，不给正式性能倍率。

### 末次规划边界

受控 NPU 闭环将历史长度与帧数绑定到仿真器实际的 `simulation_launch_report.json`
窗口。本评分配置只渲染 12 帧，需要计划 4 至 11；原始 24 帧仍提供真实评分上下文，
不能据此生成未消费的计划 12。到达边界后，沿用原有有界监控等待下一场景或完成标志，
在此之前不再调用模型，没有增加任意等待时间。

末步修复已在同一官方场景的 VADv2/HydraMDP × NR/R 四条路径完成实际 NPU 与
独立 CPU 审计。每条均精确生成并消费 8 份计划，不再产生 `merged_12` 或计划 12；
96 张 JPEG、12 份元数据、8 份计划及实际输入、9 个完整 dense 位置和保留 5 位
小数的落盘 PDM CSV 均与原前缀逐字节一致，未比较 PDM 内部未舍入值。
新增三路实际执行公开 SimEngine 提交 `933f6fb780f644c4bae82abfa93f3f488eaa2596`；
最初 VADv2 NR 的原始源码声明保留，实际使用同一修复入口 SHA。
VADv2 R 的 0 分和 HydraMDP R 的 27.233 分保持不变，功能和原字节一致性不代表
驾驶质量通过。GPU 执行、重建训练及新的连续等工作量端到端性能对照仍未验收。
规划入口不在 213 文件指纹范围内，其独立集成 SHA 为
`f61e5d033887ab6051fabbba4e93defd8081b68af0c0e2fc8193ce3dcb955c5a`，
复现时必须使用匹配的完整源码，不能只核对运行指纹。

下文启动和公开审计命令保持不变。审计成功后，显式核对八次控制边界：

```bash
python - "$NEW_AUDIT_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report['result'] == 'pass_optimized_default8_scored_window_and_original_prefix_raw'
assert report['published_plan_indices'] == list(range(4, 12))
PY
python -m unittest simengine_ascend.tools.tests.test_planner_horizon -v
```

后文完整 24 帧／二十次控制及原八次控制评分结果，均属于 SimEngine 提交
`27b13d7a03da257c3d2a66a95b640bd522ab0c61` 的 212 文件版本及外部 AlgEngine 245。
复现原结果应单独检出该固定 SimEngine 提交，并使用其原审计命令。
这些历史结果不代替新规划优化或端到端耗时验收。

新公开 CPU 审计同时核对原控制链和完整输出字节。先从已完成的原 212/245 评分作业及
其公开 `audit_scored_default8` 成功报告生成便携参照。`REFERENCE_AUDIT_SHA` 为独立保存的
原报告整体 SHA-256；`REFERENCE_JSON` 必须是输入目录之外的新文件。两种模式都使用
新源码目录的 Python 路径，在不挂 GPU/NPU 的 CPU 容器中执行，输入只读：

```bash
python -m simengine_ascend.tools.audit_optimized_default8 prepare-reference \
  --job "$ORIGINAL_RUN" --recorded-job-root "$ORIGINAL_RECORDED_RUN" \
  --validated-audit "$ORIGINAL_SCORED_AUDIT" \
  --expected-audit-sha256 "$REFERENCE_AUDIT_SHA" \
  --source-root "$SIM_ROOT" --prepared-dir "$PREPARED" \
  --vocabulary "$VOCABULARY" --model vadv2 --reaction NR \
  --output "$REFERENCE_JSON" --deadline-utc "$DEADLINE_UTC"
```

独立记录参照文件整体 `REFERENCE_SHA`，再检查新作业。设备编号填写实际宿主机两枚芯片；
`ALG_OPT_COMMIT` 与新规划报告中的真实声明一致。审计输出同样使用输入目录之外的新文件：

```bash
python -m simengine_ascend.tools.audit_optimized_default8 audit \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALG_ROOT" \
  --recorded-algengine-root "$RECORDED_ALGENGINE_ROOT" \
  --prepared-dir "$PREPARED" \
  --reference-inventory "$REFERENCE_JSON" --expected-reference-sha256 "$REFERENCE_SHA" \
  --expected-source-sha256 9d09e7bb239fcb0238255e689423598115c0115841b47912b10ffbef80b1e541 \
  --expected-algengine-source-sha256 cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610 \
  --expected-algengine-commit "$ALG_OPT_COMMIT" \
  --expected-prepare-receipt-sha256 "$APPROVED_PREPARE_RECEIPT_SHA" \
  --model vadv2 --precision fp32 --reaction NR \
  --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

其他路径同时替换两条命令的 `--model hydramdp` 或 `--reaction R`。检查范围包括
96 张 JPEG、12 份逐帧元数据、8 份实际消费轨迹和四帧输入、9 份完整 dense pickle 及
每位置八个完整数组，以及五位小数保存的 PDM CSV。还必须存在真实优化调用、零回退和
补偿跳过历史 GridMask 调用的实际计数（每次预测六次 NumPy 随机抽样），不代表原路径
的全部随机调用次数。真实生成预测数单列，可能另有一个未消费末尾
计划，不将其计作第九次控制。未保存的内部原始精度 PDM 值、GPU 等价、全场景质量及
渲染训练反向不在本项验收范围内。

CPU 夹具命令：

```bash
python -m unittest simengine_ascend.tools.tests.test_planner_optimization \
  simengine_ascend.tools.tests.test_audit_optimized_default8 -v
```

环境、后端构建、官方数据与验收顺序见[部署与验证指南](../docs/simengine_ascend_assessment_zh.md)。
本 FP32 分支已完成同一官方 MTGS 场景的完整 24 帧验证，覆盖 VADv2/HydraMDP × NR/R
四种组合。每组均使用原始四帧历史、实际执行二十次模型控制，并独立解码 192 张原尺寸相机图像。
源码、输入和逐项审计哈希见 [FP32 完整单场景记录](../docs/validation/simengine_fp32_20261029.json)。

同一份八相机原尺寸捕获输入，在同一芯片上按 ABBA 顺序测试，渲染前向中位耗时由
38.9381 秒降至 17.7366 秒，完整未压缩 RGB+ED、alpha 和 radii 逐位一致。
这是单枚 910C 芯片上的渲染优化对比，不是 GPU 对照或闭环端到端性能结论。
单场景控制验收不代表全部场景的驾驶质量，也不代表重建训练已适配。

早期[真实场景回放记录](../docs/validation/simengine_real_scene_20261020.json)及
[八项 12 帧 FP32/BF16 闭环记录](../docs/validation/simengine_closed_loop_20261020.json)
仍保留各自原始源码、依赖和验收范围。小规模 manifest 不能授权更大的真实输入。
当前每项闭环使用同一张物理卡的两枚 NPU 芯片；本结果不包含十六芯片扩展验证。

此目录将 SimEngine 的逐帧文件协议接入可显式选择设备的 AlgEngine 规划器。
仿真器与规划器仍使用各自环境、各自进程。仿真客户端只加载 CPU 文件协议，
不会因导入客户端而加载 `torch_npu` 或切换 AlgEngine 源码。

## FP32 tile 处理

tile 网格数量不超过 `2**24` 时，稳定排序使用可精确表示整数的 FP32 键；
更大的网格保留原整数排序。深度顺序及相同深度下的 Gaussian ID 顺序保持不变。
每个交点的起始偏移直接通过已有 Gaussian ID 索引取得，省去对相同计数的第二次
`repeat_interleave`。

修改适配代码后，源码哈希会变化，旧渲染 manifest 会被拒绝。部署前须重新完成
小规模设备验证和完整捕获输入验证，生成与新代码匹配的 manifest。测量渲染耗时
时应固定 FP32、输入、设备和预热次数，并显式同步设备；渲染、规划及闭环总耗时
分别记录。本次修改仍限于前向渲染。

## 依赖和版本

NPU 规划器复用单独检出的 AlgEngine 迁移仓。本分支保留原 CUDA 插件，
运行 NPU 时必须提供外部迁移仓目录、声明的 Git 提交、实际源码 SHA-256 和镜像 ID。
代码会检查已导入插件的实际文件路径，防止误用本分支的 CUDA 实现。

| 项目 | 固定值 |
| --- | --- |
| AlgEngine 仓库 | `https://github.com/ZhengDeL/WorldEngine` |
| FP32 24 帧配置的 AlgEngine Git 提交 | `55efdef54139c63ab9afce9178ebb3bd130717ef` |
| 算法源码 SHA-256 | `0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922` |
| 校验范围 | `projects/AlgEngine/**/*.py`、`ascend/*.py`，共 245 个 Python 文件 |

运行前后都会重新校验外部源码，同时单独记录本次闭环接入文件的 SHA-256。
Git 提交用于说明依赖版本，实际文件内容由源码 SHA-256 校验。建议将依赖源码只读挂载。

环境安装沿用固定版本 AlgEngine 仓库的 NPU 文档。部署时在宿主机执行
`docker inspect --format '{{.Image}}' 容器名`，记录实际镜像 ID，并将同一个值传入
规划器的 `WORLDENGINE_IMAGE_ID` 环境变量和 `--expected-image-id` 参数。
规划器检查两者一致并写入报告；宿主机镜像核验由部署流程完成。

规划进程要求 `torch_npu`、固定版本的 `ascend` 适配代码和 MMCV 的
`NPUDataParallel`。`npu:0` 表示进程可见范围内的逻辑设备 0。
部署时应同时记录宿主机芯片映射；报告会保存 `WORLDENGINE_HOST_DEVICE_IDS` 和
`ASCEND_RT_VISIBLE_DEVICES`。

## 原场景完整 24 帧 FP32 配置

外部 AlgEngine 单独检出为上表的完整 Git 提交，运行期间将两套源码只读挂载。
规划器命令见英文版：显式选择 `--precision fp32 --msda-backend cuda-compatible`，
并在同一组 `--cfg-options` 中设置 `sim.max_frame_per_scene=24`、连续四帧输入及真实共享目录。
两个模型分别使用官方 100% 完整权重和原始 8192 项轨迹词表。

为每次作业新建 `RUN_ROOT/frames`、`plan_traj`、`merged_ann_files`，暂不创建 `WE_output`。
仿真命令增加 `--num-history 4 --num-future 20`，保留准备回执中的 `--scene-sha256`，
设置 `--mode closed-loop --device npu:1`；共享规划、帧目录必须与规划器完全一致。
规划器使用逻辑 `npu:0`，两进程由下述受控启动器一起启动，显式选择 `--reaction NR` 或 `R`。
原场景必须已有 24 条真实帧元数据和全部八路相机。短场景、相机缺失、重复 token 或 pickle
尾部多余内容会被拒绝，不会补帧或重复输入。
24 帧 `--dry-run` 会读取经过完整 SHA 核验的准备 pickle，检查真实元数据；
仍不执行模型或仿真器，也不创建作业输出。默认 12 帧 dry-run 不读取 pickle。

本 24 帧专用验收配置明确禁止两个评分开关。二十次控制的 dense 评分需要 32 条真实源帧，
当前场景只有 24 帧；PDM 的上下文要求不同，另由下述八次控制配置实际验收。
二十次无评分闭环控制的结论保持原样。渲染源码修改后须重新完成小规模
设备验证与完整真实捕获输入验证，不能沿用旧 manifest。截止时间使用未来的 ISO 8601
`+00:00` UTC 格式，兼容 Python 3.9/3.10；不要复用历史作业的绝对截止时间。

完整窗口独立审计使用 `python -m simengine_ascend.tools.audit_full24`，完整参数见英文版。
在不挂载任何 GPU/NPU 的 CPU 容器中，将原作业、源码、外部 AlgEngine、准备资产只读挂载；
审计输出使用输入目录之外的新文件。指定原始运行路径、两个宿主机芯片编号、源码和准备回执
SHA-256、模型、NR/R 以及 `--expected-window-frames 24`。审计逐项核对 `4..23` 控制索引、
每次四帧真实输入、原词表选中轨迹、原策略及控制器消费、后继状态与全部 24×8 张相机图像。

`--expected-algengine-commit` 必须与原报告声明值一致；新作业应使用上表已核验的公开提交。
10 月 29 日原始实验仍记录旧声明 `972e1d9592860cef6520e0b39d7e4e9ab55a510a`，
实际执行的 245 个文件已逐文件匹配公开提交 `55efdef54139c63ab9afce9178ebb3bd130717ef`。
复核旧作业时显式传入其旧声明，不修改历史报告；两者仍须通过相同的实际源码指纹校验。
`--recorded-algengine-root` 填原规划器路径，可以与 CPU 审计挂载路径不同。

CPU 检查命令：

```bash
python -m unittest simengine_ascend.test_run_simulation -v
python -m unittest discover -s tools/simengine_ascend/tests -p test_window_contract.py -v
```

## 原始 PDM 与 dense 评分：八次控制前缀

本配置仍使用同一份完整 24 帧官方场景和上表的 AlgEngine 提交。渲染帧为 `0..11`，
实际控制为 `4..11`，未渲染的原始 `12..19` 帧保留为 CPU dense 评分的真实未来上下文。
不要把准备 pickle 裁成 12 帧，也不要重复帧。该场景属于 NAVTEST failure，不能替代
NAVTRAIN rare 场景或六组原始合成训练配置。

2026 年 10 月 30 日，VADv2/HydraMDP × NR/R 四条路径均完成原始 PDM 和 dense 评分。
每条路径产出 96 张可完整解码的 JPEG、step 11 的一条终点评分，以及 `3..11` 九个位置的
密集奖励；每个位置均包含 8192 个候选的八项有限指标数组。原始 CSV 仅保留五位小数：

| 模型 | 其他车辆策略 | PDM × 100 | 实际质量观察 |
| --- | --- | ---: | --- |
| VADv2 | NR | 79.057 | progress 0.49736，NC/DAC/TTC 均为 1 |
| VADv2 | R | 0 | NC 门控为 0，首个 NC/DAC 违规记录在 step 11，DAC 为 1 |
| HydraMDP | NR | 100 | 终点评分七个字段均为 1 |
| HydraMDP | R | 27.233 | TTC 为 0，progress 0.25360，NC/DAC 均为 1 |

这些是本次 NPU 规划器与原 CPU 评分器的实测输出，低分照实保留。它们不代表 GPU 对照
或全部场景驾驶质量已验收。`first_violation_step` 只记录 NC/DAC 事件，不能据此推断 TTC
事件时刻。四条路径与先前不启评分的 24 帧运行相比，同一前缀的 96 张 JPEG、八份实际
消费的计划和八份真实模型输入均逐文件字节一致，说明启用评分未改变已核前缀；这项检查
仍不等于 GPU 等价。逐项回执和边界见[版本化结果](../docs/validation/simengine_fp32_20261029.json)。

使用原有双进程受控启动计划。规划器保留 FP32、`cuda-compatible` MSDA、完整权重及
源码/镜像校验，在同一组 `--cfg-options` 中设置 `sim.max_frame_per_scene=12`，
`sim.post_process_path` 指向原始词表，并保留两个清理选项为 `False`。仿真进程设置：

```bash
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:1 \
  --output "$RUN_ROOT/WE_output" --deadline-utc "$DEADLINE_UTC" \
  --stop-file "$RUN_ROOT/STOP" --expected-scene-ids "$SCENE_IDS_JSON" \
  --mode closed-loop --reaction NR --num-history 4 --num-future 8 \
  --planner-data-path "$RUN_ROOT/plan_traj" \
  --planner-client-folder "$RUN_ROOT/frames" --pdm-metrics --dense-rewards
```

将这组 argv 填入下述启动器计划，使用 `{WE_output}`、`{deadline_utc}`、`{stop_file}`、
`{plan_traj}` 和 `{frames}` 替换值。每次使用新目录，NR/R 与审计参数保持一致。
每条路径使用同一张物理卡的两个芯片，dense 保留原有八个 CPU 评分进程。本次每个完整
作业设置 48 GiB CPU 内存上限和 16 个 CPU 核，独立审计为 16 GiB/两核；这些是实测配置，
尚未测定最低资源需求，也不是全量训练的内存要求。

两个进程都需要可读取的源码、准备资产、地图、官方完整权重和原始词表。dense 从
`SIM_ROOT/data/alg_engine/test_8192_kmeans.npy` 读取词表，规划器还需要外部
`ALG_ROOT/data/alg_engine/test_8192_kmeans.npy`；将同一份官方数据只读挂载到两处。
CANN 完整输入 manifest 的依赖必须一并保留，包括其引用的小规模 manifest、设备报告、
capture JSON、NPZ、真实输入参考报告和 46 个 native runtime 文件。通过只读路径别名
保持回执中的绝对路径，或在新布局重新生成证明。只复制顶层 manifest 不能完成校验。

仿真环境需具备原 NuPlan/Ray/地图与几何依赖，保留已验证的 Torch、NumPy、PROJ 版本
及导入路径；不要整目录覆盖 site-packages。`PROJ_DATA`/`PROJ_LIB` 指向匹配的 PROJ 数据，
并限制评分进程的 BLAS 线程。此配置的原评分公式仍在 CPU 上执行，两个开关不表示评分
已迁到 NPU。

两条原始子进程都实际结束后，在无 GPU/NPU 挂载的 CPU 容器中只读挂载所有输入，运行
新的固定配置审计。完整参数与英文版一致：

```bash
python -m simengine_ascend.tools.audit_scored_default8 \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALG_ROOT" \
  --recorded-algengine-root "$RECORDED_ALG_ROOT" --prepared-dir "$PREPARED" \
  --expected-source-sha256 daabd5c34ed290038ad4d0da13c32e81100b9692a0b63ff9b657f72edd9c7d66 \
  --expected-algengine-source-sha256 0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922 \
  --expected-algengine-commit 55efdef54139c63ab9afce9178ebb3bd130717ef \
  --expected-prepare-receipt-sha256 ac246caf243f2e1a679c4159dbd92ecb1cfe84f4197fc023295909d7943414fc \
  --model vadv2 --precision fp32 --reaction NR --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_SCORED_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

`RECORDED_RUN_ROOT` 和 `RECORDED_ALG_ROOT` 填原进程报告中的路径，可与审计容器的
挂载路径不同。审计核对完整消费链、原策略/控制器状态、全部图像、源码/权重/精度、
PDM CSV 和 dense 数组。八步目标字段检查形状、有限性和 SHA，不重算目标或评分公式。
零分保留为实际质量结果。旧 `audit_closed_loop` 和 `audit_full24` 的无评分配置保持原样，
不能用于此项验收。

```bash
python -m unittest simengine_ascend.tools.tests.test_audit_scored_default8 -v
```

## 运行方式

### 单节点 NPU 场景回放

新增入口为 `python -m simengine_ascend.run_simulation`。父进程只做文件核验和进程监督，
子进程执行原 Hydra runner、BaseEnv 和 `run_runners`，不替换仿真或评分公式。
固定使用 `SINGLE_NODE`，不会创建 Ray worker；原代码获取进程 ID 时仍会导入 Ray，
BaseEnv 也会提前导入 NuPlan、MetricManager 和 DenseRewardManager，因此这些原项目依赖
仍需安装。还需已验证的 CANN 渲染环境和 Pillow。

先使用 [`prepare_scene.py`](../tools/simengine_ascend/prepare_scene.py) 准备官方单场景与资产。
英文说明给出了从 `prepare_receipt.json` 读取原场景字典 key、场景 SHA-256 并生成
预期 ID JSON 的命令。运行格式如下，所有变量均需替换为实际路径或值：

```bash
cd "$SIM_ROOT"
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:0 \
  --output "$JOB_PARENT/replay_NR" --deadline-utc "$DEADLINE_UTC" \
  --expected-scene-ids "$SCENE_IDS_JSON" --mode replay --reaction NR
```

输出父目录必须存在，`replay_NR` 必须是新目录。`NPU_MANIFEST` 必须是当前运行库和
输入规模对应的有效渲染验收文件；入口不会自行放大其 Gaussian、相机或像素上限。
加 `--dry-run` 可查看准确命令和哈希，不加载 Torch、不反序列化场景、不创建作业目录，
也不运行仿真。正式子进程会检查 `worldengine` 模块来源，在退出时调用原 DataManager
保存最后一段场景数据，并关闭引擎。

`replay` 使用原轨迹策略和日志回放控制器，不调用规划模型。`NR` 按日志回放其他车辆，
`R` 按原 `run_testing.sh` 使用 IDM 策略与导航。
入口显式设 `debug_static_world=false`；当前源码中该字段仅在 YAML 声明，未发现 Python
读取位置。实际 R/NR 行为由策略和导航配置决定。

### 连接 AlgEngine 规划器

将命令改为 `--mode closed-loop`，并补充
`--planner-data-path /共享目录/plan_traj` 和
`--planner-client-folder /共享目录/frames`。两个共享目录需已存在。
该模式沿用原测试脚本的 `env_input_policy`、`navformer_client`、
`trajectory_navigation` 和 `log_play_controller`。
按已验证的双芯片配置，将仿真器设为 `--device npu:1`，另一进程按下文命令
在 `--device npu:0` 启动 AlgEngine，两端使用相同文件目录。

受控双进程计划的 `simulation.argv` 可以直接调用此模块，
`--output` 使用 `{WE_output}`，共享目录使用 `{plan_traj}`、`{frames}`，
同时传入 `{deadline_utc}` 和 `{stop_file}`。外层启动器保留 `WE_output` 为未创建路径，
由本入口创建；规划器的 `data_root` 指向 `{WE_output}/openscene_format/`。
英文说明提供了完整 JSON 参数结构。

配置中的 `gpu=false` 只控制原任务调度，渲染器仍在显式指定的 NPU 上执行。
车辆动力学和控制器保留原 CPU 实现。可选 `--pdm-metrics` 开启原 PDM 评分，
`--dense-rewards` 开启原密集评分。仿真子进程隐藏 CUDA，也不导入 `transfer_to_npu`，
因此密集评分按原 CUDA 可用性判断走 CPU 分支。
密集评分还要求 `SIM_ROOT/data/alg_engine/test_8192_kmeans.npy`、足够的场景时序和 CPU
内存，并会创建原有的 8 个评分进程；默认不开启。

`simulation_launch_report.json` 保存子进程准确命令、源码与输入哈希、模式、设备及
原场景报告核验结果。入口要求实际场景 ID 与预期完全一致、场景成功并生成原完成标记，
还会逐个校验和解码保存的 JPEG。它不据此声明全部相机与帧覆盖、GPU 图像对齐或规划质量
通过。依赖缺失、超过渲染验收范围、场景异常、图像损坏和超时均返回非零状态。

截止时间必须是带 UTC 时区的绝对时间。`STOP` 文件或 SIGTERM 会停止本次直接创建的
子进程组，等待 3 秒后才对仍存活的自有组执行强制停止；独立分离的进程仍需专用容器或
作业边界保护。日志保留在 `simulation.log`。

该入口已通过 CPU 参数、JPEG 和进程夹具，并完成上述官方 NR/R 回放。
每次原始 RunnerReport 均记录一个选定场景成功结束，96 张输出 JPEG 均可解码。
这里的 12 帧来自原 runner 配置；原始场景的 log_length 为 24，不能按完整日志验收。
另一次独立 CPU 审计已核对 12 帧身份、时间戳、前后帧关联及 96 张相机 JPEG 的
路径、尺寸、完整解码和哈希。5.5 秒是仿真时间跨度，不是运行耗时。该审计独立于入口
原有检查，回放结果仍不代表规划反馈或 GPU 对齐。检查命令为
`python -m simengine_ascend.test_run_simulation`。

英文说明提供了完整规划器命令和双进程启动格式。运行前需落实以下路径：

- 本分支目录，以及外部 AlgEngine 固定版本目录。
- 两个环境各自的 Python 解释器。
- 模型配置、与配置匹配的完整 checkpoint、原始轨迹词表。
- 场景文件、渲染资产，以及本次作业独占的新输出目录。
- 设备映射、实际镜像 ID，以及带时区的绝对截止时间。

正式规划模型应使用匹配的官方完整配置与权重。以下路径均相对于外部 AlgEngine 根目录：

| 模型 | 配置 | 完整 checkpoint |
| --- | --- | --- |
| VADv2 | `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py` | `data/alg_engine/ckpts/vadv2/e2e_vadv2_100pct_ep8.pth` |
| HydraMDP | `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py` | `data/alg_engine/ckpts/hydramdp/e2e_hydramdp_100pct_ep8.pth` |

两者均使用原始 `data/alg_engine/test_8192_kmeans.npy` 轨迹词表。闭环专用数据集读取真实渲染 JPEG 和连续四帧元数据，不用训练标注或此前 139 样本小集替代输入；上游闭环推理使用的零值 PDM 占位数组不代表评分结果。英文命令显式设置 `--wait-timeout 600`，用于等待真实原尺寸场景的初始及后续帧，绝对截止时间仍生效。

checkpoint 使用严格参数检查。单独的 LoRA 增量权重需要先按对应模型的正式流程
合并为完整权重，不能直接作为完整 checkpoint 传入。

规划器使用 `--device npu:0 --precision fp32` 或
`--device npu:0 --precision bf16`。BF16 复用原迁移仓的 NPU autocast、FP32 几何计算
和实际激活检查，不将模型参数整体转成 BF16。早期 12 帧结果使用 `reference`；
本分支当前 24 帧流程只验收 FP32，显式使用 `cuda-compatible`。
原 GPU 调用保持默认 `cuda:0`；未指定精度时沿用配置中的 FP16 设置。

正式双进程启动使用 `python -m simengine_ascend.launch_closed_loop`，要求提供
JSON 命令计划、新建作业目录、`--deadline-utc`、`--expected-scenarios` 预期场景数，
以及 `--expected-scene-ids` 指定的 JSON 场景 ID 列表。ID 列表需与实际选定输入完全一致。
计划分别指定仿真器、规划器的
参数数组、工作目录及环境。启动器直接执行数组，不经过 shell。

两端必须共用以下目录和设置：

| 目录 | 规划器配置 | 仿真器配置 |
| --- | --- | --- |
| `frames` | `sim.monitored_folder` | `planner_client_folder` |
| `plan_traj` | `sim.plan_save_path` | `planner_data_path` |
| `merged_ann_files` | `sim.merged_ann_save_dir` | 无 |
| `WE_output` | `data_root=WE_output/openscene_format/` | `output_dir` |

元数据帧 `0、1、2、3` 构成首次输入，规划器发布 `<scene>_4.npy`，随后逐帧推进。
轨迹必须是有限的浮点数组，形状为 `40×3`。VADv2/HydraMDP 还会校验轨迹与原始
8192 词表选中项一致；生成式规划头使用其真实 anchor 索引范围。
相机、时序、坐标变换和车辆执行公式沿用原组件。

目录必须归本次作业独占，底层文件系统需支持原子重命名和硬链接。
可在作业目录创建 `STOP` 文件，或给启动器发送 SIGTERM。到达截止时间、任一子进程
异常退出时，启动器会停止本次创建的另一个进程。Linux 下先用 `WNOWAIT` 保留直接子进程的
leader PID，清理其原进程组后再回收；leader 已退出时仍会清理留在原组中的子孙进程。
其他平台仅清理直接子进程。Ray 或容器中另行分离的进程仍需由专用作业或容器边界执行相同截止时间。
启动器不会全局停止 Ray、重置 NPU 或清理其他任务。

文件发布采用临时文件写完并同步后原子发布。轨迹禁止覆盖已有结果。
已发布文件损坏会直接报错；等待缺失输入也有明确超时。
清理只处理当前规划器创建的具体文件，保留渲染记录和帧输入。

## 复核闭环控制窗口

运行新作业前，在已审查计划的 **simulation 阶段** 增加以下环境项，并保留该阶段原有的其他环境设置：

```json
"env": {"WORLDENGINE_CLOSED_LOOP_AUDIT_DIR": "{WE_output}/consumption_audit"}
```

该目录必须位于本次作业的 `WE_output` 内且尚不存在。回放和 GPU 默认路径不启用此观察器。
客户端记录实际解析的 NPY 字节、返回轨迹对象、原策略和控制器的实际执行，以及日志回放控制器
对车辆状态的更新。真实后继帧及其 8 张相机文件发布后，才原子写入对应的成功消费记录。
中断作业会保留已经完成的记录。

公开 CPU 工具复核一个 12 帧、4 帧历史的官方场景窗口：要求控制索引 `4..11` 共 8 次均被执行，
每次使用连续四帧真实输入，轨迹为有限的 `40×3` 数组且匹配原词表选中项，消费记录哈希链完整。
还会检查准确的 96 张相机图像清单，并按原始尺寸完整解码 JPEG。
末尾可能额外生成 plan12，它不计为第 9 次车辆控制。

在不挂载任何 NPU/GPU 设备的独立 CPU 容器内运行，依赖 NumPy、Pillow 和 PyYAML。
作业、对应源码、外部 AlgEngine 和已准备的官方资产均只读挂载。
`RECORDED_RUN_ROOT` 填原进程运行时使用的绝对作业路径；`RUN_ROOT` 可以是此次 CPU 容器内的另一个只读挂载路径。
输出必须是所有输入目录之外的新 JSON 文件，且父目录已存在并可写：

```bash
python -m simengine_ascend.tools.audit_closed_loop \
  --job "$RUN_ROOT" --recorded-job-root "$RECORDED_RUN_ROOT" \
  --source-root "$SIM_ROOT" --algengine-root "$ALGENGINE_ROOT" \
  --prepared-dir "$PREPARED" \
  --expected-source-sha256 "$APPROVED_SIM_RUNTIME_SHA" \
  --expected-prepare-receipt-sha256 "$APPROVED_PREPARE_RECEIPT_SHA" \
  --model vadv2 --precision fp32 --reaction NR \
  --host-devices "$PLANNER_HOST_CHIP" "$RENDERER_HOST_CHIP" \
  --output "$NEW_CLOSED_LOOP_AUDIT_JSON" --deadline-utc "$DEADLINE_UTC"
```

两个宿主机芯片索引必须明确提供，均在 `0..15` 内且不能重复，顺序为规划器、渲染器。
工具将计划与运行报告中的设备映射和这些独立提供的预期值比较；容器内逻辑设备仍为 `npu:0`、`npu:1`。
每个阶段的 `env` 会覆盖继承的容器环境。因此必须在两个阶段都把 `WORLDENGINE_HOST_DEVICE_IDS`
设为实际芯片对（逗号分隔）；只修改容器环境不会替换计划里已有的固定值。
HydraMDP 改用 `--model hydramdp`，BF16 改用 `--precision bf16`。
`--reaction R` 对应原 IDM 策略和导航；工具会同时核对入口的 R/NR 标签、解析后的 Hydra 策略、场景与交换目录。

当前验收配置固定使用官方 100% VADv2/HydraMDP 完整权重、原始 8192 轨迹词表、外部 AlgEngine
源码 `40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b`，
提交 `972e1d9592860cef6520e0b39d7e4e9ab55a510a`，以及报告中的镜像 ID
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`。
工具核对报告身份与实际输入文件；作业执行者还需核实真正使用的容器镜像和设备挂载。
其他模型、源码或镜像需要另行审查验收配置。源码 SHA 按相对 POSIX 路径排序计算，必须对应实际执行的快照。
此工具位于顶层运行指纹范围之外，交付清单和报告还需保留工具本身的 SHA-256。
复核尚未包含此公开工具的旧快照时，在新的 Python 进程中改用
`python /工具绝对路径/audit_closed_loop.py` 并传入相同参数，`--source-root` 仍选择实际执行的旧快照。
这样 CPU 辅助模块来自指定快照，无需改写原留证目录。上面的模块调用形式要求工具与所选辅助模块
来自同一工作副本；辅助模块被其他目录遮蔽仍会被拒绝。

FP32 要求记录选定 NPU 上真实 Conv2d/Linear 输出、FP32 参数及关闭的 autocast；
BF16 要求真实 BF16 激活、FP32 主参数与几何计算，以及恢复后的 autocast 状态。
这些是代表性激活观测，不是逐层、逐帧的数据类型跟踪。

该工具验收指定控制窗口。完整原日志、驾驶质量、官方评分、GPU 对照、训练收敛和性能另行验证。
R 策略夹具只验证配置门禁；R 的 NPU 结论需有对应真实作业及成功审计报告，不能从 CPU 夹具推导。
CPU 检查命令：

```bash
python -m unittest simengine_ascend.tools.tests.test_audit_closed_loop -v
```

## 已验证的闭环窗口

| 规划模型 | 规划精度 | NR | R |
| --- | --- | --- | --- |
| VADv2 | FP32 | 通过 | 通过 |
| VADv2 | BF16 | 通过 | 通过 |
| HydraMDP | FP32 | 通过 | 通过 |
| HydraMDP | BF16 | 通过 | 通过 |

表中每项均包含一次真实 NPU 运行和随后独立执行的公开 CPU 审计。审计核对控制索引
`4..11`、每次连续四帧输入、实际 NPY 字节及原轨迹词表选中项、原策略和控制器的对象身份、
sample-1 车辆状态更新与后继渲染帧，并完整核对 12 帧 × 8 相机的 JPEG 清单和解码结果。
两个模型均使用官方 100% 完整权重。FP32 记录了关闭 autocast 时的实际 Conv2d/Linear
FP32 输出；BF16 记录了实际 BF16 输出及 FP32 主权重、几何和 MSDA 计算区。
这些观察不代表所有层、所有帧的 dtype 跟踪。

每项实测使用同一张物理卡的两枚 910C 芯片，规划器为 `npu:0`，FP32 渲染器为 `npu:1`。
这是本次验证配置，不能据此断言最低设备需求或与 GPU 等价。每个窗口的 5.5 秒是仿真时间，
不是实际运行耗时。PDM 指标和稠密奖励均未开启；原日志完整时长、官方质量评分、整图/GPU
对照及性能仍待专项验证。

验收对应 SimEngine 源码 `90ad1a295420d87ad63391733f8f28124161b43d939964c55831b7a8e259a7d8`
（211 个运行文件），公开 CPU 审计工具 SHA 为
`83c79ef344e995d3d440580e8db85fe8d792888eeafee0fe452b285d636758fe`。
[结果 JSON](../docs/validation/simengine_closed_loop_20261020.json)记录每项的计划、权重、
图像清单和八条消费记录哈希。早期 `1ef363…` 源码的四项诊断因宿主机与计划内设备映射不一致，
保留为历史记录，未计入本次验收。这里的八项是 SimEngine 闭环控制组合，
不能作为 AlgEngine 八个合成数据训练配置已覆盖的证明。

## 结果与验收

公开 CPU 审计工具已通过 11 项 Linux CPU 夹具，并实际核验了本轮 NR/R 两个窗口；帧与图像清单检查独立于回放入口。[复核命令和输入约束](../docs/simengine_ascend_assessment_zh.md#复核回放窗口)说明 NR/R 输出的目录结构、固定的准备/源码 SHA-256，以及只读输入目录之外的新报告文件。

`plan_traj/closed_loop_planner_report.json` 记录实际预测次数、严格加载结果、
输入配置和权重哈希、最后轨迹的索引和哈希、源码来源、精度状态及设备映射。
`launcher_report.json` 记录双方实际退出状态。

启动器还会严格检查原始 `WE_output/runner_report_*.json`：场景身份唯一且 ID 与预期相同、实际数量
与预期一致，每条 `succeeded=true`、无错误消息且耗时有效。
原 runner 某些场景异常后仍可能返回 0，这项检查用于避免误判。
启动器返回 0 表示双方正常退出且场景报告检查通过。配置控制窗口还需通过上述相机覆盖和
实际消费审计；官方指标用于另行评价驾驶质量。
`--max-predictions 1` 只适用于限定输入的规划器回放，不能代替这些检查。

本地检查命令为 `python -m simengine_ascend.test_closed_loop_protocol`。
夹具验证原子写入、损坏与缺失数据、连续帧与场景切换、GPU 导入隔离、NPU 来源门禁、
截止时间和自有进程清理，并覆盖 STOP 文件写入异常。夹具使用临时合成文件和短时 Python
进程，不加载正式模型，不执行 NPU 算子，也不渲染真实场景。
