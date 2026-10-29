# AlgEngine 规划推理断点续跑

> 历史记录：本文保存 2026 年 10 月 20 日的验收范围和当次锁定值，对应[归档提交 `55efdef5`](https://github.com/ZhengDeL/WorldEngine/tree/55efdef54139c63ab9afce9178ebb3bd130717ef)。当前 checkout 的操作见[优化推理恢复](algengine_ascend_zh.md#优化推理的中断恢复)及[文件收集入口](algengine_ascend_zh.md#使用文件收集完成推理与恢复)。本文仅供参考输入格式和挂载布局，执行脚本及 SHA 以相应版本为准。

schema2 为原有 AlgEngine 规划推理增加逐 rank 的持久化记录。VADv2 和 HydraMDP 已在昇腾上完成 **FP32、22 个官方真实 anchor、16 rank** 的中断恢复和正位置恢复检查。[结果 JSON](../ascend/results/fp32-navtest22-journal-recovery-20261020.json)列出源码、输入、通过标准和原始证据 SHA。

完整 12,146 个 NAVTEST anchor 的官方 VADv2 FP32 推理已于 2026 年 10 月 20 日 06:54:29 UTC 单独启动，本记录尚未验收该完整运行。已通过的小集检查不代表完整评分、训练断点恢复、收敛、GPU 一致性或性能验收。当前优先完成 FP32，本文不安排其他精度作业。

下表为该次实测所锁定的两个运行文件 SHA，不表示当前 checkout 与它们逐字节相同。

| 文件 | SHA256 |
| --- | --- |
| `ascend/tools/journal_inference.py` | `ad9d6429929305cb9054aab9df78b3ea9425ce7caa9af2de695ac42a350afe46` |
| `ascend/tools/rank_journal.py` | `f92e2fe35f428be30851c272921c450582e5b2c5846297b415385e13f8dfe4c4` |

文件放在 `ascend/tools/`，不进入原有顶层运行源码指纹的统计范围。模型与原运行代码指纹仍为 `0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922`。每份 journal 会另外锁定这两个工具的 SHA。模型指纹不变，不代表执行过程中没有插桩。

## 2026 年 10 月 20 日 FP32 实测

两类原 100% 规划配置均使用官方发布权重，单节点、16 个逻辑 NPU 芯片，对应 8 张双芯物理卡；每 rank 的 batch 为 1，workers 为 0。journal 保留原模型计算、采样补齐和结果收集规则。插桩身份单独记录，不能因原模型 245 个文件的指纹未变，就把执行称为未插桩。

本次从完整官方 NAVTEST filter 按原顺序选前 22 个 anchor，保留完整标注及历史帧记录。这是有固定筛选依据的官方数据子集，不是另行发布的标准 mini。filter SHA256 为 `446bfb797dbd747ae79673036b14e0659459276275e04c04401624d0d87e2097`。输入证据还锁定原 NAVTEST 标注、PDM、8192 候选词表和完整图像账本；实际处理范围仍是 22 个 anchor。

| 检查 | VADv2 FP32 | HydraMDP FP32 |
| --- | --- | --- |
| 原基线、不中断 journal、提交一步后暂停、恢复 | 通过 | 通过 |
| 最终 22 条有序结果对原基线 | 全部字段和类型、数组 dtype/shape/原始字节、标量数值位一致 | 同左 |
| 两条 journal 的初始及逐步 Python/NumPy/Torch CPU/NPU RNG | 全部 16 rank、每 rank 两个位置一致 | 同左 |
| 暂停前已提交前缀 | 未改写 | 未改写 |
| 从已完整的中断 journal 做第五次运行 | 公共恢复位置 1；每 rank 跳过 1 个已保存位置，前向 1 步、重播 1 步、新提交 0 步 | 同左 |
| 第五次运行后原四路线证据 | 283 项文件 SHA 全部保持 | 同左 |

该小集经原采样器补齐后共有 32 个位置，其中 10 个是补齐位置；最终导出 22 条结果。轨迹保持有限的 40×3 形状，并通过原词表选择检查。数组与标量 dtype 按原基线严格比较，这不表示整个软件栈的每个算子都使用同一种 dtype。

未插桩基线没有逐步 RNG 记录。四源 RNG 的严格比较发生在不中断 journal 与恢复 journal 之间。本次完整运行及 CPU 审计的实际容器均已停止、退出 0 且无 OOM；受控暂停的 NPU 容器按真实 torchrun 退出码 1 留证，由主机控制器结合完整暂停证据返回 3。VAD 四条路线由新 sequence 采用已有证据，未重新 reap 原外层进程，外层实际返回值保留 null，不能由容器退出 0 推定。

这些检查验证推理进度恢复。训练恢复、完整 NAVTRAIN 收敛、完整 NAVTEST、官方评分、GPU 对照和吞吐各有自己的证据范围。单次完整 journal 推理本身，也不等于已完成全量基线与恢复路线的成对比较。

## schema1 已知问题与版本边界

早期 schema1 core（SHA `aaee2c7483a2a1edd496a7c4136785fbb8d592b535adb4bdc645747b62306879`）在一次真实恢复中，尚未执行前向计算便被 `Original result serialization changed` 拒绝。NumPy array 与 scalar 的 dtype 对象共享关系可能在 pickle round-trip 后改变；所有字段类型与数值位相同，重新序列化的字节仍可能不同。该门禁错误地拒绝了有效存储值。原 schema1 文件与诊断证据全部保留，这次恢复不计为数值或恢复验收通过。

schema2 保留存储 payload 的整文件 SHA，另用有版本的值摘要检查全部字段类型、dtype 描述、shape 和数值原始位；字典排列与对象共享关系不属于数值相等条件。不支持的对象和 object dtype 会被拒绝，重播不覆写已有 payload。schema1 journal 会被明确拒绝，不做隐式迁移或在新插桩身份下复用旧进度。请建立全新的 schema2 journal，原文件保留供排查。

## 支持范围

当前合同限定为单节点、固定 16 rank、每 rank batch1、workers0、原有不打乱的 DistributedSampler，以及 `process_perception=False` 的 NAVFormer 词表规划推理。规划头仅支持 TrajScoringHead/TrajScoringHeadRL。每个 anchor 仍从完整标注中读取当前帧和前三帧。

感知推理、生成式轨迹、其他带跨样本状态的模型，以及改变 rank 数或精度后的恢复，都不在当前支持范围内。

完整 NAVTEST 有 12,146 个 anchor。原采样器给每个 rank 分配 760 个位置，全局包含 14 个重复补齐位置。原 MMDetection CPU 收集器交错合并 12,160 个位置，再截去补齐部分。本工具保留这套顺序，不另按 YAML 切分 token，也不裁掉标注中的历史帧。

要求使用以下镜像 ID：

```text
sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
```

主机监督程序应先通过 Docker 检查实际镜像 ID，不能只相信 `WORLDENGINE_IMAGE_ID` 环境变量。镜像内实际安装的 `multi_gpu_test` 与 `collect_results_cpu` 源码 SHA 还须分别等于：

```text
1c17038cc42177a369516d2ca6520d5716c061b83c854e54b577d6d8301c82a7
b1bd40c200c4629cee65b59cc38eaa47aa459ff9a269b948052b8755e57bbe98
```

部署方需要提供这一固定 Linux ARM64 镜像，例如离线 Docker 归档及独立确认的归档 SHA256。本仓库没有提供公开拉取地址或镜像归档。以下命令用于检查部署方提供的归档及加载后的镜像；本结果不声称已经交付或验收该归档的 save/load 流程。

```bash
: "${FIXED_IMAGE_ARCHIVE:?Set the operator-supplied archive path}"
: "${FIXED_IMAGE_ARCHIVE_SHA256:?Set the trusted archive SHA256}"
printf '%s  %s\n' "$FIXED_IMAGE_ARCHIVE_SHA256" "$FIXED_IMAGE_ARCHIVE" | sha256sum -c - &&
  docker load --input "$FIXED_IMAGE_ARCHIVE" &&
  test "$(docker image inspect --format '{{.Id}}' sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48)" = sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48 &&
  test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48)" = linux/arm64
```

按配方重新构建的镜像需要单独验证。修改环境变量标签不能让其他镜像通过 wrapper 的固定镜像门禁。

## 输入与挂载

使用对应模型的官方 checkpoint，或已独立验收的训练 checkpoint，并记录整个文件的 SHA256。配置应使用相应的已导出推理配置，确认标注、PDM、词表、图像和地图路径实际可读。文件名相同不能作为替换 checkpoint 的依据。

还需提供已通过检查的输入 QC 报告。主机监督程序负责核对报告范围、源码、JPEG 账本与完整 anchor 覆盖；本工具只锁定报告文件本身，不能把任意 JSON 自动判为合格数据证据。22 个 anchor 的诊断应使用对应的 filter；锁定的完整 QC 报告可证明小集所需文件已检查，但不能把实际处理的 22 个 anchor 写成 12,146 个。

各次恢复必须保留相同的容器绝对路径。可采用以下挂载布局，主机路径按实际文件填写。

| 主机文件或目录 | 容器路径 | 权限 |
| --- | --- | --- |
| 固定 WorldEngine 源码及本工具 | `/workspace/WorldEngine` | 只读 |
| 完整 AlgEngine 数据根目录 | `/data/full` | 只读 |
| 本次选择的 checkpoint | `/weights/initial.pth` | 只读 |
| 原始词表 | `/weights/vocab.npy` | 只读 |
| 已导出的推理配置 | `/config/inference.py` | 只读 |
| 输入锁清单与已验收 QC 报告 | `/proof` | 只读 |
| 配置需要的地图 | `/maps` | 只读 |
| 当前单个 case 的持久化 journal | `/journal` | 读写 |
| 本次 attempt 的全新输出目录 | `/output` | 读写 |

配置中的路径必须与实际挂载一致。若已导出的配置还引用其他绝对路径，也应在对应位置挂载原文件。地图路径应与配置或 `NUPLAN_MAPS_ROOT` 一致。本工具会在原数据集建立后检查其实际 PDM 路径，不自行推测配置会选择哪个默认缓存。

按已验证环境的设备合同映射 16 个获准使用的 NPU 芯片节点及 Ascend 控制节点，驱动与固件目录只读挂载。容器中的 `ASCEND_RT_VISIBLE_DEVICES=0,1,...,15` 使用容器逻辑芯片编号。主机监督程序应记录有序的主机芯片到容器编号映射。这里的 16 个逻辑设备对应 8 张双芯物理卡。

`WORLDENGINE_HOST_DEVICE_IDS` 记录按挂载顺序排列的真实主机芯片 ID。下例按顺序挂载主机芯片 0 至 15；使用其他已批准映射时，应填写实际有序列表。主机 ID 与容器逻辑编号是两项设置，不能只根据 `ASCEND_RT_VISIBLE_DEVICES` 推定主机映射。

`/proof/locked_inputs.json` 必须恰好包含五个键：`annotation`、`pdm`、`vocabulary`、`filter`、`input_qc`。每个键的值格式如下：

```json
{"path":"/absolute/container/file","bytes":123,"sha256":"64位小写十六进制SHA256"}
```

示例中的大小和 SHA 必须替换为实际已验收文件的值。标注、词表和 filter 路径应与配置一致，`pdm` 应与原数据集实际选择的 PDM 文件一致。完整 NAVTEST 使用官方原始 YAML；小数据诊断显式使用官方真实数据中的 22 个 anchor，并保留其筛选依据与 SHA。

输入锁清单本身也需要整个文件的 SHA256。rank0 会在推理前后重新计算五个输入文件的 SHA。

## 容器内运行命令

以下命令应在受监督的独立容器中执行。主机监督程序须执行绝对截止时间，并只停止其记录的自有容器 ID。本工具不负责创建 Docker 容器，也不清理其他进程。只给 Docker 客户端加 `timeout`，不能保证后台容器随之停止；部署时仍须接入容器监督程序。

```bash
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/opt/conda/envs/bevformer/bin:/opt/conda/bin:$PATH
export NAVSIM_DEVKIT_ROOT=/workspace/WorldEngine/ascend/third_party/navsim
export PYTHONPATH=/workspace/WorldEngine:/workspace/WorldEngine/projects/AlgEngine:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
export WORLDENGINE_IMAGE_ID=sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
export WORLDENGINE_HOST_DEVICE_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_WORK_PATH=/output/ascend_work ASCEND_CACHE_PATH=/output/ascend_cache

# 从已批准的输入和运行计划设置以下变量。
# DEADLINE_UTC 必须是仍有效的获准使用截止时间，显式带 UTC 时区。
# ATTEMPT_ID 每次新建，为 uuid.uuid4().hex 的32位小写值。
: "${CONFIG_SHA:?}" "${CHECKPOINT_SHA:?}" "${INPUT_LOCK_SHA:?}"
: "${ATTEMPT_ID:?}" "${DEADLINE_UTC:?}"
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/tools/journal_inference.py \
  --source-root /workspace/WorldEngine \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --locked-inputs /proof/locked_inputs.json \
  --expected-locked-inputs-sha256 "$INPUT_LOCK_SHA" \
  --precision fp32 --msda-backend cuda-compatible \
  --expected-anchors 12146 --journal-root /journal --journal-mode fresh \
  --output /output/result --attempt-id "$ATTEMPT_ID" \
  --deadline-utc "$DEADLINE_UTC" --stop-file /journal/STOP
```

本文结果限定为 FP32。其他精度需要单独验证，按当前工作顺序暂不安排；已有 journal 不能中途切换精度、模型、源码或镜像。

示例保留原有默认的 `deterministic=False`。显式使用 `--deterministic` 时，还须在 Torch 启动前设置 `HCCL_DETERMINISTIC=true` 和 `CLOSE_MATMUL_K_SHIFT=1`。这是另一套需要单独验证的合同，同一 journal 不能混用两种设置。

`/output` 是本次挂载的输出目录，其下的 `/output/result` 必须尚不存在。只有 rank0 创建该子目录并写入 UUID 与合同声明，其他 rank 等待并核对同一声明。每次运行生成新 UUID，例如：

```bash
python -c 'import uuid; print(uuid.uuid4().hex)'
```

## 停止与恢复

每个已提交步骤包含原 forward 结果、forward 结束后的 Python、NumPy、Torch CPU、NPU RNG 状态，以及收据哈希链。工具先写临时目录、执行 fsync，再原子重命名为完整步骤目录。中断留下的临时目录会保留，但不计入已完成进度。journal 中的 pickle 只接受本任务产生的可信文件。

创建 `/journal/STOP` 可请求合作式停止。正在执行的原生 forward 可能尚未返回，因此仍需主机截止时间与独立监督程序兜底。恢复前，先确认上一批自有进程组或容器已经结束，保留 journal 和旧 attempt 输出；在获准继续后，仅移除本 case 创建的 STOP 文件。

恢复时，将同一个持久化目录继续挂载到 `/journal`，将另一个全新 attempt 目录挂载到 `/output`，生成新的 `ATTEMPT_ID`，并把参数改为 `--journal-mode resume`。源码、镜像、输入、精度、MSDA、确定性设置与 world size 均须保持原合同。

所有 rank 共同选择 `R=max(0,min(各rank已提交前缀)-1)`，恢复同一个边界的 RNG 并重播。已有结果的全部类型值（含数组与标量原始位）以及规范化四源 RNG 必须严格一致；不一致时保留旧 payload 字节并停止。工具已计入 DataLoader 创建迭代器时消耗的随机种子。早先保存的结果会补回原收集器，最终顺序仍由原收集器确定。

worker 退出码 3 表示部分进度。torchrun 通常会因此返回 1，应同时保留实际 launcher 状态。普通退出码 1 不能单独作为“按计划暂停”的证据。固定步数诊断会先写齐 16 份 partial 收据，再同步退出；截止时间、信号和 STOP 路径不等待该同步，可能保留有效 journal 而未写齐 16 份 partial 报告。

## 验收与输出

本次两模型 FP32 已完成以下四路线流程及后续第五次正位置检查。复现时使用相同官方 22-anchor 筛选、16 rank，并锁定 checkpoint、源码和镜像；更换输入或精度后应重新留证：

22-token filter 按完整官方 YAML 的原顺序取前 22 项，已导出的配置只替换 test filter；完整标注、历史帧、图像、PDM 和词表均保留。原基线没有 `--expected-anchors` 参数，结束后应核对原验证报告，确认恰好处理了 22 个唯一 anchor。

在受监督的基线容器内，沿用前文环境和挂载，执行真实存在的原入口：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/inference.py \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --output /output/result --precision fp32 --msda-backend cuda-compatible
```

下表每次运行都换全新输出和 UUID。journal 只按表中要求复用；journal 命令沿用前文，诊断时设 `--expected-anchors 22`。

| 执行 | journal | wrapper 参数 |
| --- | --- | --- |
| 原基线 | 不使用 | 使用上面的原入口 |
| 不中断 journal | 新建 A | `--journal-mode fresh --stop-after-steps 0` |
| 提交一步后暂停 | 新建 B | `--journal-mode fresh --stop-after-steps 1` |
| 暂停后恢复 | 保留 B | `--journal-mode resume --stop-after-steps 0` |
| 正位置检查 | 保留已完整的 B | `--journal-mode resume --stop-after-steps 0`，再次换全新输出和 UUID |

本次主机监督程序与 CPU 实测审计器在结果 JSON 中按 SHA 留证，本运行补丁没有把它们作为新的公共可执行工具发布。上面提供的是容器内运行入口，仍需自有容器监督和下述独立 CPU 比较；请用经审查的验证器逐项落实标准。CPU 合成测试通过不能替代新设备运行的验收。

1. 用未插桩的原 `ascend/inference.py` 完整运行，作为独立基线。
2. 单独新建 journal 和输出，以 `--expected-anchors 22` 完整执行一次不中断的 journal 推理。
3. 另建 journal，设置 `--expected-anchors 22 --stop-after-steps 1`。
4. 保留中断的 journal，换全新输出后恢复，去掉 `--stop-after-steps`。
5. 核对全部 16 rank 收据、边界结果与 RNG 严格重播、每 rank 两个含补齐的采样位置；将两条完整 journal 路线的最终 22 条有序结果分别与原基线逐项比较。还应比较两个 journal 的初始与逐步 RNG、全部结果值，并确认暂停前的收据未被覆盖。原未插桩基线没有逐步 RNG 记录，不能声称做过这项比较。数组须检查 shape、dtype、原始字节，标量、指标和词表索引也须全部一致。

单独保存再恢复各行后，pickle 的对象复用方式可能不同，因此整体 pickle SHA 不同不能直接判为数值不同。反过来，仅比较 token 数量也不足以证明等价。

### 正位置恢复验收

上述四次运行还不能证明实际跳过了已完成的采样位置。22 个 anchor、16 rank 时，每个 rank 有两个含补齐的位置；提交一步后暂停，恢复时选择 `R=0`，两个位置都会重新执行。

四次对照通过后，保留已完整提交的中断 journal，使用相同的 22-anchor 配置和精度再做第五次运行。另挂一个全新 `/output` 目录并生成新 UUID；源码、checkpoint、输入 SHA、设备映射及 `/journal` 挂载保持不变。FP32 在相同已批准环境中执行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /workspace/WorldEngine/ascend/tools/journal_inference.py \
  --source-root /workspace/WorldEngine \
  --config /config/inference.py --checkpoint /weights/initial.pth \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --locked-inputs /proof/locked_inputs.json \
  --expected-locked-inputs-sha256 "$INPUT_LOCK_SHA" \
  --precision fp32 --msda-backend cuda-compatible \
  --expected-anchors 22 --journal-root /journal --journal-mode resume \
  --output /output/result --attempt-id "$ATTEMPT_ID" \
  --deadline-utc "$DEADLINE_UTC" --stop-file /journal/STOP
```

`result/journal_execution.json` 中全部 16 rank 必须同时满足 `committed_before=committed_after=2`、`common_resume_step=1`、`forward_steps_this_attempt=1`、`replayed_exact_steps=1`、`new_committed_steps=0`、`all_rank_prefixes=[2]*16`。这才证明位置 0 来自已保存前缀，只重播了位置 1。最终 22 条有序输出仍须与原基线逐字段比较，类型、dtype 和原始数值位全部一致。

第五次运行前先冻结原四次审计的文件清单，结束后核对清单中每个文件的 SHA 均未改变；允许增加本次 attempt 收据，不能改写已有证据。还须核对第五个独立实际容器 ID、exit0、无 OOM、已停止及自有 watchdog 清理成功。每个模型、每种精度分别检查。本次两模型 FP32 的这项 NPU 实测已通过，不能转用于未测配置。

该小集检查不代表完整 NAVTEST、官方评分、收敛、GPU 对照、性能或长期稳定性已通过。

| 输出 | 含义 |
| --- | --- |
| `result/journal_partial_rank*.json` | 部分停止原因与输入合同，不是完整推理结果 |
| `/journal/rank_*/steps/*/{payload.pkl,receipt.json}` | 已提交的原结果、RNG 和哈希链 |
| `/journal/rank_*/attempts/*.json` | 每次运行的提交、重播计数与结果 |
| `result/original_inference_validation.json` | 增补插桩字段前的原验证报告，保留原始字节 |
| `result/inference_validation.json` | 原报告及明确的插桩来源字段 |
| `result/journal_execution.json` | 仅在原配置全部 anchor 与全部 rank 验证完成后生成 |

最终仍须通过原有完整 token 覆盖、有限的 40×3 轨迹、词表一致性与原收集流程。官方 NAVSIM 评分另需完整官方 metric cache。journal、部分停止或某个 rank 的完成都不能替代这些检查。

## CPU 测试与上游夹具

仓库不附带 MMDetection 函数源码夹具。请使用下面的标准库提取器，静态读取固定镜像中实际安装的 `mmdet/apis/test.py`。提取器不导入或执行 Torch、torch_npu、MMDetection、DrivingSDK，因此不需要 NPU 设备、驱动/HAL 共享库或 CANN 初始化。实际模型推理仍需要完整 Ascend 环境。

不要在 CPU 容器中用 `from mmdet.apis import multi_gpu_test` 替代静态提取。该导入可能进入 DrivingSDK 的设备名称查询并初始化 NPU 运行时；仅补齐驱动/HAL 库也不能保证这条动态导入链可在无设备环境执行。

固定镜像中实际安装的 MMDetection 版本已静态核实为 **2.24.0**，两个文件的实测 SHA 如下。

| 安装文件 | SHA256 |
| --- | --- |
| `mmdet/apis/test.py` | `33cb4ea46fc0a55d2571be7a47cb8486792c0775c4ab2695dbc508fbd6f62bdd` |
| `mmdet/version.py` | `2c12507eab845d59ca0b7d6c404189ddb12c52fb2c9e26547f898925ecffad09` |

这些值直接读取自固定镜像，没有根据 API 相似性推测版本。整文件 SHA 与两个函数源码 SHA 分别记录。函数 SHA 来自此前独立的镜像内实际 API 检查；静态提取不能冒充又做过一次动态 API 验证。

以下命令在固定镜像内、仓库根目录执行。

```bash
cd /workspace/WorldEngine
export WORLDENGINE_IMAGE_ID=sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48
export PINNED_MMDET_TEST_FILE_SHA=33cb4ea46fc0a55d2571be7a47cb8486792c0775c4ab2695dbc508fbd6f62bdd
export PINNED_MMDET_VERSION_FILE_SHA=2c12507eab845d59ca0b7d6c404189ddb12c52fb2c9e26547f898925ecffad09
python -S ascend/tools/extract_journal_test_fixtures.py \
  --source-file /workspace/bevformer/mmdetection/mmdet/apis/test.py \
  --expected-source-file-sha256 "$PINNED_MMDET_TEST_FILE_SHA" \
  --version-file /workspace/bevformer/mmdetection/mmdet/version.py \
  --expected-version-file-sha256 "$PINNED_MMDET_VERSION_FILE_SHA" \
  --license-file /workspace/bevformer/mmdetection/LICENSE \
  --output /fixtures/new_journal_fixture_set
# 安装分发包含 NOTICE 时，另加 --notice-file /actual/installed/NOTICE。

# NumPy CPU 测试正常运行，不挂载加速卡设备。
export WORLDENGINE_JOURNAL_FIXTURES=/fixtures/new_journal_fixture_set
# 可选：经独立批准的真实结果小样本，不能使用随意下载的 pickle。
# export JOURNAL_ACTUAL_SAMPLE_ROOT=/fixtures/approved_original_rank0_step0
python -m unittest discover -s ascend/tools/tests -p 'test_*journal*.py' -v
```

提取器先核对安装源码、版本文件与两个函数的 SHA，再创建全新夹具目录。它用 AST 定位原始顶层函数，通过标准库源码块读取器保留与实际 `inspect.getsource` 相同的 LF 函数文本；版本只读取字面量 `__version__`，不执行版本文件。实际 LICENSE、可选 NOTICE 原样保留，整文件与函数 SHA 及来源信息一并记录。

上游项目由 [OpenMMLab](https://github.com/open-mmlab/mmdetection) 发布，使用 [Apache-2.0 许可](https://github.com/open-mmlab/mmdetection/blob/v2.24.0/LICENSE)。引用标签与实测安装版本一致，兼容性仍同时核对整文件与函数 SHA。

未设置 `WORLDENGINE_JOURNAL_FIXTURES` 时，两项实际接口集成测试会明确跳过。另有五项真实结果回归测试需要设置 `JOURNAL_ACTUAL_SAMPLE_ROOT`，其中包含经批准的原 `payload.pkl`、`receipt.json`、`header.json`；运行源码不附带该结果样本。payload 的固定 SHA 为 `7eeb89723c34deae06c04c63261404d4c80d91894bff6f4cadf8b1a2eeb51cde`。未提供时这五项明确 skip，显式提供错误样本则报错。其余 NumPy/标准库测试可继续执行。显式指定的路径无效或源码 SHA 不匹配时应报错。带 skip 的执行不算完整 CPU 测试。两组批准的夹具齐全时，公开测试集共 27 项：原循环/journal 13 项、类型值与真实结果回归 10 项、静态提取器 4 项。CPU 合成夹具与静态提取不代表 NPU 实测。

最终提取器（SHA `a56545393e86388b9bf84c28bb2a7c42ef0a1f2458544b86d47f781a6f207001`）已在固定 Linux 镜像中、不挂载设备节点和驱动目录，完成静态提取与当时的 17 项 CPU 测试，全部通过且无 skip。实际执行记录为 `static_public_journal_CPU_v3_actual.json`，整文件 SHA256 为 `62734dd4701cbe31965f3cd6e0faece56de20f836ce0ee6312ac9225e94730bd`，包含提取、测试、容器 exit0 与无 OOM 的证据。原始记录含部署路径，仅随本地验收证据保留；这项 CPU 结果不代表 NPU 中断恢复、完整 NAVTEST 或官方评分已通过。

上述静态提取实测使用的是早期 schema1 journal core，证明未变的提取器及夹具生成链路，不作为 schema2 修复证据。

schema2 core `f92e2fe35f428be30851c272921c450582e5b2c5846297b415385e13f8dfe4c4` 另外通过了 23 项 Linux CPU 测试，全部通过且无 skip：包括原循环/journal 13 项及新增类型值回归 10 项，其中含真实结果小样本在另一进程重开与重播。记录 `journal_v2_core_CPU_actual.json` 的整文件 SHA 为 `884f6015b3da289c5cf6dfa0c69ffef70541dfbf9c1d8ac80646414776c0ba2b`。这是 CPU 回归证据；随后完成的 FP32 NPU 实测见上文，完整 NAVTEST 推理及官方评分尚未在本记录中验收。

完整公开 schema2 候选随后在固定 Linux 镜像中统一复核：先从安装文件静态提取夹具，再执行全部 27 项公开 CPU 测试，全部通过且无 skip。容器使用 `runc`，未挂载加速卡设备或驱动目录，exit0 且无 OOM。记录 `static_public_journal_CPU_v4_actual.json` 的整文件 SHA 为 `1cd9abe0356b6ea8530d1aedbcc3861af4fba982548604a9e324dc7946cd6c32`。该结果只验收公开工具的 CPU 复现路径。较早一次设备尝试在资源门禁处、创建推理容器前停止，记录单独保留；后来通过的 FP32 四路线和正位置检查使用了新 attempt。CPU 测试本身不能替代这些 NPU 结果。
