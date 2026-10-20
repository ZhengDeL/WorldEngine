# NPU 场景回放与闭环规划器接入

[English](README.md)

环境、后端构建、官方数据与验收顺序见[部署与验证指南](../docs/simengine_ascend_assessment_zh.md)。
设备结果已覆盖 20 项小规模 FP32 检查，以及一段官方 MTGS 场景的原尺寸首帧。
单枚 910C 芯片完成了全部 1,550,733 个 Gaussian、8 个 1920×1080 相机的前向有限值检查，
并通过 1,024 个采样像素的独立 NumPy 对照。原配置的 12 帧 NR/R 回放窗口均已结束，
每次生成并解码 96 张 JPEG。独立 CPU 清点已核对 0–11 帧、8 个相机及对应文件，
仿真时间跨度为 5.5 秒。原始完整日志时长和规划模型反馈仍待验收。
详见[真实场景结果](../docs/validation/simengine_real_scene_20261020.json)。
仅小规模 manifest 仍不能授权更大的真实输入。服务器有 8 张 NPU 物理卡、16 个芯片，
本分支尚无 16 芯片扩展或 GPU 对照结果。

此目录将 SimEngine 的逐帧文件协议接入可显式选择设备的 AlgEngine 规划器。
仿真器与规划器仍使用各自环境、各自进程。仿真客户端只加载 CPU 文件协议，
不会因导入客户端而加载 `torch_npu` 或切换 AlgEngine 源码。

## 依赖和版本

NPU 规划器复用单独检出的 AlgEngine 迁移仓。本分支保留原 CUDA 插件，
运行 NPU 时必须提供外部迁移仓目录、声明的 Git 提交、实际源码 SHA-256 和镜像 ID。
代码会检查已导入插件的实际文件路径，防止误用本分支的 CUDA 实现。

| 项目 | 固定值 |
| --- | --- |
| AlgEngine 仓库 | `https://github.com/ZhengDeL/WorldEngine` |
| 依赖提交 | `972e1d9592860cef6520e0b39d7e4e9ab55a510a` |
| 算法实现提交 | `f0461d7639e38841d218d304f0ab07afebf61eba` |
| 算法源码 SHA-256 | `40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b` |
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
另一进程按下文规划器命令启动 AlgEngine，两端使用相同文件目录。

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
和实际激活检查，不将模型参数整体转成 BF16。MSDA 默认使用 `reference` 路径。
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

## 结果与验收

公开 CPU 审计工具已通过 11 项 Linux CPU 夹具，并实际核验了本轮 NR/R 两个窗口；帧与图像清单检查独立于回放入口。[复核命令和输入约束](../docs/simengine_ascend_assessment_zh.md#复核回放窗口)说明 NR/R 输出的目录结构、固定的准备/源码 SHA-256，以及只读输入目录之外的新报告文件。

`plan_traj/closed_loop_planner_report.json` 记录实际预测次数、严格加载结果、
输入配置和权重哈希、最后轨迹的索引和哈希、源码来源、精度状态及设备映射。
`launcher_report.json` 记录双方实际退出状态。

启动器还会严格检查原始 `WE_output/runner_report_*.json`：场景身份唯一且 ID 与预期相同、实际数量
与预期一致，每条 `succeeded=true`、无错误消息且耗时有效。
原 runner 某些场景异常后仍可能返回 0，这项检查用于避免误判。
启动器返回 0 表示双方正常退出且场景报告检查通过。完整闭环验收仍需检查
真实相机渲染、车辆反馈和对应官方指标。
`--max-predictions 1` 只适用于限定输入的规划器回放，不能代替这些检查。

本地检查命令为 `python -m simengine_ascend.test_closed_loop_protocol`。
夹具验证原子写入、损坏与缺失数据、连续帧与场景切换、GPU 导入隔离、NPU 来源门禁、
截止时间和自有进程清理，并覆盖 STOP 文件写入异常。夹具使用临时合成文件和短时 Python
进程，不加载正式模型，不执行 NPU 算子，也不渲染真实场景。
