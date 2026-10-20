# SimEngine 昇腾 910C 适配与验证

[English](simengine_ascend_assessment.md)

更新日期：2026 年 10 月 20 日。

本分支已实现 MTGS 的 NPU 前向渲染兼容层。一枚 910C 芯片除通过 20 组小规模数值检查外，还完成了官方真实场景的原尺寸首帧验证：全部 1,550,733 个 Gaussian、8 个 1920×1080 相机，完整前向输出均为有限值，1,024 个采样像素通过独立 NumPy 对照。原配置的 12 帧 NR/R 回放窗口也已结束，每次生成并解码 96 张 JPEG。**12 帧、8 相机的准确覆盖已通过独立 CPU 审计；当前结果不包含原始日志完整时长、规划模型反馈或 GPU 对齐。**

WorldEngine 上游基线为 `3eded7e8ac64e5bed817860765089b5914ab698f`，原渲染接口来自 `gsplat==1.4.0`。本分支使用固定 CANN gauss-splat 提交 `8c0445ebcca91514fce3a7050241f41ed132c340` 及显式前向补丁，保持原 MTGS 的投影、颜色合成和深度语义。

## 实测结果与边界

服务器有 **8 张 910C 物理卡，每张双芯，共 16 个可见芯片**。小规模检查和真实场景阶段每次只使用一枚，设备名为 `Ascend910_9362`，容器内映射为 `npu:0`。原仓八路脚本面向 8 张 GPU；本分支尚无 16 芯片 SimEngine 扩展结果，不能由设备数量推出吞吐倍率。

| 项目 | 本轮证据 |
| --- | --- |
| 软件环境 | Python 3.10、PyTorch `2.7.1+cpu`、Torch-NPU `2.7.1.post8`、CANN Toolkit 9.1.0；每次构建与探针另记运行库指纹 |
| 公开后端构建 | 新建隔离 venv 与输出目录，使用 `build_backend.py` 完成 wheel 构建、安装及实际 native/pyACL 导入；CPU 阶段通过 |
| NPU 数值检查 | 20 项通过，覆盖投影、协方差非整块尾部、像素中心、裁剪、遮挡、alpha、期望深度、空图、逐相机颜色、抗锯齿、旋转和 0–4 阶 SH |
| 输入范围 | 最多 256 个 Gaussian、8 个相机、每相机 4096 像素；主要为 64×64，另有 71×37 非方形图像；tile 16/32 |
| RGB 最大绝对差 | `1.2237711302e-6` |
| alpha 最大绝对差 | `1.5866665948e-6` |
| 期望深度最大绝对差 | `2.5749206546e-5` |
| CPU 检查 | Linux CPU 环境中 136 项通过、无跳过：渲染/门禁 79、入口/进程 21、构建/准备 26、场景捕获 10；使用合成 fixture |

RGB/alpha 门限为 `atol=3e-4, rtol=1e-3`，期望深度为 `atol=2e-3, rtol=1e-3`，SH 为 `atol=3e-5, rtol=3e-5`。比较对象是独立 NumPy 参考实现。本轮未执行 CUDA gsplat 对照，也没有可发布的速度、成本或能耗百分比。

[数值结果与测试源码指纹](validation/simengine_npu_forward_20261020.json)保存了当前源码、公开工具构建的 wheel、实际加载的 native/kernel 身份，以及输入范围、20 项通过用例和 18 组图像误差记录。[CPU 检查记录](validation/simengine_cpu_checks_20261020.json)给出四组测试命令和实际源码清单。新环境生成的 wheel 或改动后的适配代码都必须重跑探针，生成自己的 manifest；公开摘要不能代替本机运行时 manifest，也不能授权更大的输入。

## 官方真实场景验证

固定官方资产完成整文件校验后，原 BaseEnv.reset/MTGS 生成了完整首帧输入。NPU 前向未减少 Gaussian、相机或分辨率；独立 NumPy 参考对每个相机投影全部 Gaussian，并保留采样像素所属 tile 的全部候选，没有候选截断。

| 项目 | 结果与范围 |
| --- | --- |
| checkpoint | 1 个背景、1 个天空、17 个镜像刚体；原始 1,425,076 个 Gaussian |
| MTGS 首帧输入 | 镜像及变换后 1,550,733 个 Gaussian，8 个 1920×1080 相机，FP32、RGB+ED、tile 16 |
| 完整前向 | RGB+ED、alpha 和 radii 的全部输出均为有限值 |
| 独立参考 | 固定种子 `20261020`，每相机 128 个像素，共 1,024 个；包含边角、中心及随机位置 |
| RGB/alpha 最大绝对差 | `6.8172804954e-6` / `5.3911735484e-6` |
| 期望深度最大绝对差 | `9.7257744097e-3 m`；该像素参考深度约 `491.54 m` |
| NR/R 回放 | 原配置窗口各结束 1 个选定场景，均生成并解码 96 张 JPEG；未启用 PDM/稠密评分 |

容差与小规模检查相同，深度仍使用 `atol=2e-3, rtol=1e-3`；最大误差像素对应的允许绝对差为约 `0.49354 m`。没有因真实输入调整门限。数值结论限于完整输出有限值和采样像素参考，不等同于整幅图逐像素对照。原始回放日志还记录了整数 ArgSort 使用 AiCPU 路径，当前不包含性能验收。

原 runner 的 `num_history=4, num_future=8` 对应 12 帧配置窗口。独立 CPU 审计已核对 0–11 帧的原始 token、时间戳、前后帧关联、8 个原始相机及恰好 96 张 JPEG 的路径、尺寸、完整解码和哈希。导出时间戳跨度为 5.5 秒，指仿真时间，不是运行耗时或性能结果。选定原始场景 `log_length=24`，本轮不能按原始日志完整时长验收。VADv2/HydraMDP 的 FP32/BF16 规划闭环须另有实际报告，目前不计入上述结果。

[脱敏结果记录](validation/simengine_real_scene_20261020.json)保留输入、运行源码、原始 proof 与报告的 SHA-256。它与[小规模数值记录](validation/simengine_npu_forward_20261020.json)关联，省略内部服务器路径，不能用作运行时 manifest。

| 功能 | 本分支实现 | 当前验收范围 |
| --- | --- | --- |
| NPU 渲染 | CANN 投影/前向 kernel、Torch-NPU 稳定排序、RGB/alpha/期望深度、SH 及 MTGS 接入 | 小规模检查和官方原尺寸首帧通过；真实输入为完整前向有限值及采样像素参考 |
| 几何与动态对象 | 保留背景、天空、刚体、镜像与时序变换；必要的 FP64 位姿在 CPU 计算 | 已使用真实背景、天空和 17 个镜像刚体子模型；其他资产类型仍需对应验证 |
| 官方输入准备 | 完整 SHA、真实 scene→asset 映射、安全提取、原值单场景派生 | 两个完整官方文件及目标资产已核验，19 个 checkpoint 子模型完成 CPU 清点 |
| 单节点回放 | 原 Hydra runner、BaseEnv、DataManager、报告检查与截止控制 | 官方 NR/R 配置窗口各生成 96 张可解码 JPEG；12 帧、8 相机的准确覆盖已独立核验 |
| VADv2/HydraMDP 闭环 | 独立 AlgEngine 规划进程、帧/轨迹协议、FP32/BF16 显式入口 | 已提供协议测试；本轮无真实闭环结果，渲染仍为 FP32 |
| R/NR、奖励及导出 | 原策略、CPU PDM/稠密评分分支及输出检查入口 | NR/R 配置窗口已运行；本轮未开启 PDM 或稠密评分 |

## 环境与后端构建

使用独立 Linux/ARM64 环境，先准备配套 CANN、PyTorch、Torch-NPU、编译器及原 SimEngine 依赖。本轮从既有镜像创建隔离 venv；这不构成任意干净容器的部署结果。上游 pyproject 仍声明 CUDA/PyTorch 2.0.1，不能直接用其锁文件覆盖 NPU 栈。

原环境还会导入 nuPlan、Hydra、OpenCV、SciPy、Pillow 和 Ray 2.48.0。`SINGLE_NODE` 不启动 Ray worker，但原代码仍导入 Ray。依赖能否导入和 kernel 能否运行需要分别检查。

[工具指南](../tools/simengine_ascend/README.md#build-the-pinned-cann-backend)列出了完整前提与产物。下列变量均为实际部署值；`DEADLINE_UTC` 是本次作业的明确 UTC 截止时间。

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.1.0
source "$CANN_ROOT/set_env.sh"
cd "$SIM_ROOT"
"$CONTROL_PYTHON" tools/simengine_ascend/build_backend.py \
  --source "$CANN_SOURCE" --output "$NEW_BUILD" \
  --python "$SIM_PYTHON" --jobs 8 --install \
  --cann-root "$CANN_ROOT" \
  --cann-python-path "$CANN_ROOT/python/site-packages" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

`CANN_SOURCE` 必须是固定提交的干净 Git 检出，`SIM_PYTHON` 属于已有隔离 venv，输出目录必须全新。builder 只复制 tracked 文件，在副本应用兼容补丁，限制构建并行为 1–16，只编译 `ascend910_93`，关闭原库测试程序，并修正 venv 中 Torch-NPU 头文件定位。

示例路径来自本轮安装。其他机器应替换为实际 SDK 路径，并确保 `set_env.sh`、`--cann-root` 与 pyACL 目录属于同一套 SDK。旧的 `ascend-toolkit` 链接若指向另一版本，会在安装预检时因 pyACL 路径不匹配被拒绝。

构建使用 `pip wheel --no-build-isolation --no-deps`。默认不安装；`--install` 从源码目录之外用确切 wheel 路径强制安装，并检查实际导入位置。工具阻止继承的 pip 全局目标配置，只保留经 SDK 根验证的 CANN Python 路径。`build_receipt.json` 记录源码、补丁、wheel、安装模块的 SHA、命令和版本。构建或导入成功只代表相应阶段完成。

[公开工具构建实测](validation/simengine_public_backend_build_20261020.json)记录了本轮新建环境的成功结果和 wheel/extension SHA。随后设备探针使用相同 extension SHA，通过当前源码的小规模检查。构建记录与数值记录分别保留，便于核对二进制身份与各阶段结果。

## 小规模 NPU 检查

先隔离一枚授权使用的芯片，记录宿主机与进程的映射。`npu:0` 指可见设备编号，不代表宿主机芯片 0。设备/runtime 卡住时仍需容器或作业级截止控制。

```bash
cd "$SIM_ROOT"
export PYTHONPATH="$SIM_ROOT/projects/SimEngine:${PYTHONPATH:-}"
"$SIM_PYTHON" projects/SimEngine/worldengine/render/ascend/tests/probe_cann.py \
  --device npu:0 --output "$NEW_PROBE" \
  --deadline-utc "$DEADLINE_UTC"
```

同时核对退出码、`device_validation.json` 中的 `result=pass`、所需用例和 `validated_manifest.json`。manifest 绑定运行库、适配源码、参考实现、兼容 kernel 与输入上限。探针计时含验证操作，不作为性能验收。不能手工放大 manifest 的 Gaussian、相机或像素上限。

## 官方单场景输入

最小候选为官方 `all_scenarios.pkl` 与 `navtest_failures/assets/part003.tar.gz`，共 **28,798,253,799 B（28.80 GB）**。单场景验证无需全部 5 TB。固定 HF/魔搭来源、尺寸和完整 SHA 见[官方资产锁](../tools/simengine_ascend/official_scene_assets.lock.json)及[下载说明](../tools/simengine_ascend/README.md#inputs)。

本轮对场景索引完成整文件 SHA 核验，实际读出 288 个原始场景。其中一个场景映射到 part003 首个已确认资产：

```text
2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4
```

该场景对应 Boston 地图。两个官方文件均已完成整文件 SHA-256 校验，目标 checkpoint 大小为 336,430,727 B，CPU 清点发现 1 个背景、1 个天空及 17 个 MirroredRigidSubModel。checkpoint 原始共有 1,425,076 个 Gaussian，原 MTGS 镜像与变换后首帧拼装为 1,550,733 个；未抽样或截断 Gaussian。

```bash
"$SIM_PYTHON" tools/simengine_ascend/prepare_scene.py \
  --scenario-pkl "$INPUTS/all_scenarios.pkl" \
  --asset-archive "$INPUTS/part003.tar.gz" --output "$PREPARED" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

工具验证两个完整官方文件后，才读取 pickle、匹配真实资产并提取目标完整目录。`scene.pkl` 保留原字典值，不删帧、改相机或替换图像。`prepare_receipt.json` 记录源、派生文件及每个资产的 SHA。可加 `--inspect-checkpoint` 做独立 CPU 清点；需要 PyTorch 和 checkpoint 所用项目依赖，不执行渲染。

## 真实输入捕获与验证

准备完成后，使用原 BaseEnv.reset 和 MTGS 拼装首帧完整输入。捕获保留全部 Gaussian、相机、原分辨率和 RGB+ED 设置，在光栅化调用前导出；捕获成功本身不代表图像或场景通过。

```bash
PREPARE_RECEIPT_SHA=$("$SIM_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PREPARED/prepare_receipt.json")
SOURCE_COMMIT=$(git -C "$SIM_ROOT" rev-parse HEAD)
"$SIM_PYTHON" tools/simengine_ascend/capture_scene.py \
  --prepared-dir "$PREPARED" --prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --maps-root "$MAPS_ROOT" --small-manifest "$NEW_PROBE/validated_manifest.json" \
  --device npu:0 --source-commit "$SOURCE_COMMIT" --reaction NR \
  --output "$NEW_CAPTURE" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"

"$SIM_PYTHON" projects/SimEngine/worldengine/render/ascend/tests/validate_real_input.py \
  --capture "$NEW_CAPTURE/capture/capture.json" \
  --small-manifest "$NEW_PROBE/validated_manifest.json" \
  --device npu:0 --output "$NEW_REAL_CHECK" \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

提交号说明基线，捕获工具还会记录实际运行源码指纹。捕获重新核验派生场景和每个已提取资产；原 27.89 GB 压缩包的整文件核验由准备 receipt 关联，此阶段不重读原包。

验证器执行完整前向、检查全部输出有限值，并对固定种子抽取的像素做独立 CPU 对照，默认每相机共 128 个采样像素，包含边角、中心和其余随机位置。通过后生成 `validated_real_manifest.json`。它限定输入和渲染参数，不能扩写为全部像素、整段场景或 GPU 等价验收。本轮真实输入已通过这些检查，误差和原始证据 SHA 见[公开结果](validation/simengine_real_scene_20261020.json)。

## 场景回放与规划闭环

完整依赖、ID 列表生成和双进程命令见[中文运行指南](../simengine_ascend/README_zh.md)与[英文运行指南](../simengine_ascend/README.md)。先从准备 receipt 读取字典 key、派生 SHA，生成唯一预期场景 ID 列表。

```bash
cd "$SIM_ROOT"
"$SIM_PYTHON" -m simengine_ascend.run_simulation \
  --scene-pkl "$PREPARED/scene.pkl" --scene-sha256 "$SCENE_SHA256" \
  --asset-root "$PREPARED/assets" --maps-root "$MAPS_ROOT" \
  --validated-manifest "$NPU_MANIFEST" --device npu:0 \
  --output "$NEW_REPLAY" --expected-scene-ids "$SCENE_IDS_JSON" \
  --mode replay --reaction NR --deadline-utc "$DEADLINE_UTC"
```

真实场景需要适用的 manifest，入口不会扩大已验证范围。`replay` 回放原轨迹；`closed-loop` 另接 VADv2 或 HydraMDP 规划进程。两者使用各自环境和受控帧/轨迹目录。NPU 规划器复用固定的独立 AlgEngine 迁移仓，并核对实际源码、镜像及完整 checkpoint。FP32/BF16 入口详见闭环指南；当前 SimEngine 渲染只接受 FP32。

`R` 选择原 IDM 策略，`NR` 回放其他车辆日志。PDM 指标、稠密奖励需显式开启，并提供真实地图和原轨迹词表。入口核验原 RunnerReport、准确场景 ID、完成标志和实际 JPEG 解码；本轮 NR/R 窗口的相机与帧清单已通过独立审计，车辆反馈和适用评分仍需单独验收。

## 复核回放窗口

本公开 CLI 已在 Linux CPU 环境中实际执行：11 项工具夹具全部通过，随后完成真实 NR/R 两组输出的准确清点。两阶段字段与此前私有审计完全一致。运行快照只新增 3 个审计工具文件，原 223 个文件均保持不变，仿真运行时的 208 文件清单和捕获运行时的 211 文件清单也未变化。[结果 JSON](validation/simengine_real_scene_20261020.json)记录了公开报告与工具的 SHA-256，并保留旧 proof 作为历史。

公开 CPU 工具逐项核对原 Hydra 配置、RunnerReport、完成标记、连续帧元数据和全部 JPEG。它不运行 NPU 或规划器；依赖 Python 3.9 或以上版本、NumPy、Pillow 和 PyYAML。`REPLAY_JOB` 下需同时有 `replay_NR`、`replay_R` 两个实际输出目录；分别运行上一节命令时，将输出目录和 `--reaction` 对应设置为 NR/R。`PREPARED` 可以独立放置。每次输出需完整保留 `.hydra/config.yaml`、`simulation.log`、完成标记、RunnerReport、`simulation_launch_report.json`，以及 `openscene_format/meta_datas` 和 `sensor_blobs`。

```bash
cd "$SIM_ROOT"
SIMULATION_SOURCE_SHA=$("$SIM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_identity"]["simulation_runtime"]["sha256"])' "$SIM_ROOT/docs/validation/simengine_real_scene_20261020.json")
"$SIM_PYTHON" -m simengine_ascend.tools.audit_replay \
  --job "$REPLAY_JOB" --source-root "$SIM_ROOT" --prepared-dir "$PREPARED" \
  --output "$NEW_AUDIT_JSON" \
  --expected-prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --expected-source-sha256 "$SIMULATION_SOURCE_SHA" \
  --expected-window-frames 12 --expected-cameras 8 \
  --deadline-utc "$DEADLINE_UTC" --stop-file "$STOP_FILE"
```

两个预期 SHA 应在审计前固定，分别来自已批准的准备 receipt 和实际回放源码记录。本轮值见[结果 JSON](validation/simengine_real_scene_20261020.json)中的 `original_proof_sha256` 与 `source_identity.simulation_runtime`。`NEW_AUDIT_JSON` 必须不存在，其父目录已存在，并位于 job、prepared、source 三个只读输入目录之外。

同时检查退出码为 0、报告 `result=pass_two_configured_replay_windows_coverage`、NR/R 各自的准确帧数、相机数、JPEG 清单以及 `full_original_log_coverage`。本轮覆盖是 0–11 帧、8 相机、96 张 JPEG、5.5 秒仿真时间，选定原场景共 24 帧；不能将配置窗口结果计为原始完整日志。公开复核工具与 NPU 回放入口各自生成报告，不能用其中一个的退出码代替另一项检查。

## 设备配置与后续验收

单芯片已完成本次原尺寸首帧和 NR/R 配置窗口。该结果尚不能确定任意完整场景的最低 HBM、CPU 内存或最少设备数。规划闭环采用渲染与规划器各一枚芯片的验证配置；两者共用一枚芯片的容量尚未实测。

后续重点是原始日志完整时长、两种规划模型的 FP32/BF16 反馈、适用指标/奖励及中断恢复。多芯片/Ray、长时间运行和性能另做验收。与原仓 8 GPU 比较时，固定同一资产、相机、分辨率、模型、dtype、指标范围及输出内容，分别报告物理卡数和芯片/进程数。

外部 MTGS 重建训练、任意 deformable 模型、未实现的 BWM 生成功能和完整 LiDAR 渲染不在本轮前向结果中。原 DeformableSubModel 映射为 None，LiDAR 输出仍有上游 TODO，不能由相机渲染结果推定这些功能已完成。
