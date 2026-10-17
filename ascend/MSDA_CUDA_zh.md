# Ascend CUDA-compatible MSDA 后端

`--msda-backend cuda-compatible` 显式选择 Ascend 上的 FP32 多尺度可变形注意力（MSDA）实现，面向原 MMCV 1.6.2 CUDA 实现已测的坐标运算和一阶归约顺序。它使用原生 PyTorch NPU 运算，浮点输入、输出和梯度保留在 NPU，仅在主机读取少量整数空间元数据。

默认仍为 `--msda-backend reference`，沿用通过 NPU `grid_sample` 实现的历史 MMCV CPU32 数学参考路径。两种后端的数值目标分别记录。相关说明见[参考实现](MSDA_zh.md)、[原 CUDA 基线](CUDA_BASELINE_zh.md)及[安装与使用指南](../docs/algengine_ascend_zh.md)。

## 支持范围

| 项目 | CUDA-compatible 后端 |
| --- | --- |
| 计算精度 | FP32 与 BF16 模型模式下，MSDA 核心均使用 FP32；模型 BF16 autocast 不改变该计算精度 |
| 每个注意力头的通道数 | 固定为 32 |
| 空间轴 | 高度、宽度分别为 1 至 4096 的正整数 |
| 求导范围 | value、sampling locations、attention weights 的一阶梯度 |
| query 处理 | 每块最多 2048 个，完整处理所有 query，不截断 query 范围 |
| 常规布局 | 空间层数与投影层数正确对应；value 梯度使用显式连续存储，也覆盖输入 value 非连续的情况 |
| 原 map 特殊布局 | 仅支持连续 raw L4 / spatial L1、8 个头、4 个采样点，并保持原全局 flat-prefix 解释 |
| 浮点输入 | value、weight、上游梯度为有限值；坐标补偿的有限中间值和下溢边界以实现策略为准 |

对原 map 特殊布局，后端先解释完整连续 location/weight 缓冲区的**全局前缀，再进行 query 分块**，不会按每个头或每个 query 单独选取 level 0。梯度映回原始 raw 张量，未使用的连续尾部保持为零。

坐标路径采用 FP32 补偿乘加，不使用 FP64 或浮点 CPU 回退。远距离 padding 及非有限 sampling locations 映射到无效 padding 坐标。该处理不代表支持任意非有限 value、weight 或上游梯度，也不构成通用 IEEE 正确舍入 FMA 或高阶求导保证。value 梯度累加顺序也可能与 CUDA 原子操作不同，应区分容差内一致与逐位相同。

## 数值证据与验收

所有元素必须为有限值，并满足 `abs(actual - reference) <= atol + rtol * abs(reference)`：

| 完整张量 | 绝对容差 | 相对容差 |
| --- | ---: | ---: |
| 前向输出 | `2e-5` | `2e-5` |
| 三种梯度分别比较 | `1e-3` | `1e-4` |

[算子验证记录](results/npu-cuda-msda-20261017.json)标明公版入口实测的源码指纹 `3d46c07f…`（245 个 Python 文件）和公版数值核心 `feb35fc8…`。这些算子输入为数值测试张量，不是驾驶数据样本。固定 CUDA 输入及保存的参考张量通过完整文件和张量哈希传输，NPU 侧未按 seed 重新生成输入。相同 seed 本身不能证明不同 Torch 版本生成的输入字节相同。

| 对象或验证 | 身份与实际范围 | 状态 |
| --- | --- | --- |
| 公版注册入口 | 真实 DrivingSDK MX 和 MMCV 函数；两组完整 normal-L1 几何，报告 `b7054570…` | 64 项 CUDA 比较全部通过，每入口 32 项。16 项 CPU32 比较中 8 项通过、8 项超差。80 项均完整执行，输入、源码、精度、后端未变，资源正常释放 |
| 公版 MMCV 原 map 布局 | 真实 MMCV 六参数注册入口；4 个固定 raw L4 / spatial L1 用例，报告 `a377773f…` | 16 项 CUDA 比较全部通过。16 项 CPU32 比较中 14 项通过、2 项超差。32 项均完整执行，完整 raw 梯度保留，未使用尾部为零。MX 不支持该 raw-L4 接口 |
| 原 CUDA 目标 | H20、Torch 2.0.1+cu118、MMCV 1.6.2；固定输入的完整输出及三种梯度 | 已实测；部分边界结果与保存的 CPU32 参考存在差异 |

以下补充验证分别标明其数值源码，扩展已测数学和布局用例。上表公版入口结果仅覆盖所列 normal-L1 和 MMCV raw L4 / spatial L1 用例，不代表其他 legacy 布局、BF16 模型或训练验收。

| 补充验证 | 数值源码与实际范围 | 结果 |
| --- | --- | --- |
| 私有 NPU 候选 v8 | 数值源码 `264676e4…`；B2/Q40000 与 B1/Q65536，均为 200×200、H8/C32/P4；每例保存 4 次 CUDA 重复结果 | 32 项完整 CUDA 张量比较均在原容差内；保存的 CPU32 超差结果单独保留 |
| 私有 legacy 布局候选 v6 | 数值源码 `6f8e19f9…`；4 个 raw L4 / spatial L1 用例 | 16 项完整 CUDA 张量比较均在原容差内；未使用的 raw 梯度尾部为零 |
| 私有 v8 布局检查 | 6 个用例，包含非连续 value 布局；使用新的可精确表示二进制分数输入及原 Torch 2.7.1 CPU 数学函数 | 24 项完整张量比较通过，与保存的 CUDA/CPU32 张量分别记录 |
| 私有守护坐标检查 | 22 组空间轴、2112 个坐标分量；padding 与可精确表示的二进制分数数学用例 | 坐标检查通过；28 项按需完整张量数学比较通过，目标为相应 Torch 2.7.1 CPU 参考 |
| 私有 v8 固定二进制分数用例 | 报告 `58e60d57…`；10 个保存的 Torch 2.0.1 用例，覆盖 8 种梯度需求组合及另外 2 组多层几何 | 56 项实际存在角色的比较全部通过：保存的 CUDA201 与 CPU201 目标各 28 项；不需要的梯度另行核验 |

原 CUDA 与 CPU32 数学实现可能采用不同的边界运算顺序。通过 CUDA 目标不会改变保留的 CPU32 超差结果，因此公版入口报告保留“完整执行但超出原容差”分类，worker 退出码为 1；其守护控制器正常退出且没有遗留自有任务。整模型训练结果单独记录。这些算子结果不构成整模型重放、跨平台训练等价、收敛或性能结论。

公开的[原CUDA金样导出器](../tools/validation/README_export_cuda_msda_goldens_zh.md)与[NPU验证器](../tools/validation/README_npu_cuda_msda_zh.md)已完成端到端实测：重新导出原CUDA输入和完整参照、传输全部16个文件，再调用两个真实公开NPU入口。64项CUDA张量比较全部通过；独立CPU32目标仍为8项通过、8项超差。验证器对16个输入文件执行了前后完整校验，源码、后端及原容差均未改变。

## FP32 同次连续前缀续训对照（2026 年 10 月 18 日）

[完整结果](results/cuda-compatible-fp32-strict16-resume2-20261018.json)。使用公开源 `3d46c07f4ff0`（245 个 Python 文件）和 `cuda-compatible` MSDA，16 个 rank、每 rank batch 1（全局 batch 16）、workers 0，统一运行两次更新。连续训练保存第 1、2 次更新的检查点；新进程从同次连续训练的第 1 次更新检查点恢复至第 2 次更新。第 1 次更新由两条链共享，本项检验该检查点之后的续训一致性，不检验独立初始化的一致性。

| 模型 | 模型张量：torch.equal / 总数 | 全部比较张量：torch.equal / 总数 | 模型／全部张量最大绝对差 | 最大 loss 差 | 原门槛 |
|---|---:|---:|---:|---:|---|
| VADv2 | 889 / 889 | 1698 / 1698 | 0 / 0 | 0 | 门槛内 |
| HydraMDP | 909 / 909 | 1778 / 1778 | 0 / 0 | 0 | 门槛内 |

原比较器保持张量 `atol=1e-6`、`rtol=1e-5` 和 loss 差严格小于 `1e-4`。计数中的一致性使用 `torch.equal`，不表示原始张量字节或序列化检查点文件字节相同。公共模型和 Adam 状态由 rank 0 保存；全部张量计数另包含 16 个 rank 的最终 CPU／NPU RNG 张量，并非每个 rank 全部最终模型值的转储。每个完成的模型核验了 16 次完整状态恢复、64 个分类后的四类 RNG 恢复事件和 48 条下载的训练记录；原参与集合的梯度及 Adam 状态逐参数名核对形状、优化器分组位置、FP32 类型与有限值。

数据配置包含 139 个训练锚点，实际当前锚点去重数为 VADv2: 32, HydraMDP: 32，已完成模型的并集为 32。计数仅取历史队列的最后一个 token。本项未新增推理或 NAVSIM 评分；两次更新仍处于原 500 次更新的 warmup 内，不构成收敛、质量、独立初始化复现或八次更新差异修复的验收。历史八次更新的完整超容差结果继续单独保留。

完整状态续训须满足检查点记录的源版本、精度、拓扑和训练总更新数合同。本次设备训练保持上述实测源身份。后续验证器或命令行修订即使不改变模型公式，也可能改变整个检出版本的源指纹，因此不能直接视作可继续恢复旧检查点的同一版本。只读 CPU 结果比较可使用另行标明身份的验证工具版本。

另以单独标识版本的补强 CPU 验证器复核同一批保存结果，检查形状与 dtype、张量及记录有限性、完整更新序列和与合同 world size 对应的 RNG 覆盖。新回执及检查点前后不变的哈希与原比较器结果分开列明。

本次实测通过元数据观察器（`8148ff59`）调用原公开训练入口，核对 RNG 恢复及逐参数梯度／Adam 覆盖；数值函数和检查点合同保持原样，但观察过程可能改变执行时序。公开启动命令不包含该观察器及其完整诊断记录，尚未验证无观察器运行也能得到相同的 `torch.equal` 结果；请按 [CPU 检查点比较](../docs/algengine_ascend_zh.md#5-保存和恢复完整训练状态)独立检查每次新运行保存的产物。


### 运行同前缀训练／恢复协议

为所选模型设置相匹配的 official100 配置和初始化权重。按实测训练／恢复协议运行时使用其记录的源码／镜像／数据版本，并在同一检出版本内完成两臂。下列命令新建连续前缀，不加载其他源版本产生的检查点。

```bash
set -euo pipefail
: "${WORLDENGINE_ROOT:?Set the public checkout}"
: "${PREPARED139:?Set the validated NAVTRAIN139 prepared directory}"
: "${VOCAB:?Set test_8192_kmeans.npy}"
: "${MAPS:?Set the NuPlan maps directory}"
: "${IMAGE:?Set the dependency image}"
: "${EXPECTED_IMAGE_ID:?Set the verified image ID}"
: "${CONFIG:?Set the VADv2 or HydraMDP official100 config}"
: "${INITIALIZER:?Set the matching official100 checkpoint}"
: "${OUTPUT_ROOT:?Set a fresh absolute output directory}"
: "${MEMORY:?Set the validated container memory limit}"
: "${CPUS:?Set the validated container CPU limit}"
: "${PIDS_LIMIT:?Set the validated container task limit}"
: "${SHM_SIZE:?Set the validated shared-memory limit}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$WORLDENGINE_ROOT"
continuous="$OUTPUT_ROOT/continuous0to2"
resumed="$OUTPUT_ROOT/sameprefix_own1to2"
launcher=(python3 ascend/run_container.py
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --mini-prepared "$PREPARED139" --vocab "$VOCAB" --maps "$MAPS"
  --image "$IMAGE" --expected-image-id "$EXPECTED_IMAGE_ID"
  --memory "$MEMORY" --cpus "$CPUS" --pids-limit "$PIDS_LIMIT" --shm-size "$SHM_SIZE"
  --precision fp32 --msda-backend cuda-compatible --deterministic)
train=(--task train --config "$CONFIG" --checkpoint "$INITIALIZER"
  --runtime --upstream-schedule --batch-size 1 --workers 0
  --steps 2 --checkpoint-interval 1 --check-interval 1)
"${launcher[@]}" "${train[@]}" --output "$continuous"
test -s "$continuous/iter_1.pth"
test -s "$continuous/iter_2.pth"
"${launcher[@]}" "${train[@]}" --resume "$continuous/iter_1.pth" --output "$resumed"
test -s "$resumed/iter_2.pth"
```

## 源版本与检查点合同

验证器、命令行和数据路径修订的源指纹为 `40997952fdc9`（245 个 Python 文件）；上述设备对照实测源为 `3d46c07f4ff0`（245 个文件）。MSDA 核心不变，但六个已评审源文件的变化使完整状态续训合同不同。完整续训须使用检查点记录的源版本、精度、拓扑与训练总更新数；官方规划权重仍可用于新训练的初始化。只读 CPU 比较另用 `ba855451…` 验证器，其六项公开 CPU 回归测试可[独立运行](../tools/validation/test_validation_guards.py)。

实测 `3d46` 源码对应提交 [4946f15](https://github.com/ZhengDeL/WorldEngine/tree/4946f15dd2073b7c0974bc2335091246a4ec9023)。复现或恢复上述运行时，请单独检出该提交。更新后的 `4099` 源码可初始化新训练，其追加的 VADv2 单次更新检查见[源码回归报告](results/source-review-regression-20261018.json)。16 rank 实测容器限额为 `MEMORY=256g`、`CPUS=128`、`PIDS_LIMIT=8192`、`SHM_SIZE=8g`；这些是测试限额，不代表最低硬件要求。

## 训练、自身检查点恢复与推理

2026年10月17日，VADv2、HydraMDP 的 FP32/BF16 四组均通过公开 `cuda-compatible` MSDA 入口，实际完成训练、自身 checkpoint 续训、原 DDP4 推理及独立官方 CPU 评分。四个互斥分组各使用 4 个 rank，每 rank batch1，每组全局 batch4。每个配置从原官方 100%规划权重初始化，先完成第 1 次更新，再用自己的 checkpoint 恢复到第 2 次更新。原 backbone 冻结设置、几何计算、损失、梯度裁剪和 AdamW 均保留。实测显式使用 `--msda-backend cuda-compatible`，默认精度仍是 FP32。BF16 禁用 autocast 权重缓存，实际记录到 Conv/Linear 的 BF16 激活，以及 FP32 主权重、参与更新的梯度和 Adam 状态。

训练集配置包含从原官方数据确定性派生的 139 个锚点；本轮每组在两次更新中实际消费 8 个不同的当前锚点，四组锚点集合相同，合并后仍为 8 个。历史队列帧未额外算作当前锚点。

| 模型 | 精度 | rank × 每 rank batch | 更新与自身续训 | 独立评估锚点 | 官方 PDMS |
|---|---|---|---|---|---|
| VADv2 | FP32 | 4 × 1 | 1 → 自身 checkpoint → 2 | 22 | 0.871938963172 |
| VADv2 | BF16 | 4 × 1 | 1 → 自身 checkpoint → 2 | 22 | 0.908986383163 |
| HydraMDP | FP32 | 4 × 1 | 1 → 自身 checkpoint → 2 | 22 | 0.953343369212 |
| HydraMDP | BF16 | 4 × 1 | 1 → 自身 checkpoint → 2 | 22 | 0.953676351654 |

四组使用同一批 22 个独立评估锚点，不是 88 个独立场景。每组推理都导出原 22 个唯一 token 对应的 40×3 有限轨迹，并通过原词表检查。原 NAVSIM 评分器使用真实官方 cache payload 独立评分：每组 22 个 token 均有效，完整七项指标和逐 token CSV 均保留。[完整结果与哈希](results/cuda-compatible-navtrain139-short2-20261017.json)给出全精度数值、源代码/输入/checkpoint/收据哈希及实测范围。

原 8 个训练容器全部 exit0，32 份 rank 阶段收据通过；16 份自身续训 rank 收据均精确恢复原完整模型及 Adam 状态，并按真实调用来源核对 Python/NumPy/CPU/NPU 四类 RNG 恢复一致。VADv2、HydraMDP 参与更新且具有梯度和 Adam 状态的参数张量数分别为 259、279；完整具名参数核验分母分别为 730、750，不代表每个可训练参数每次都参与更新。4 个推理容器及 4 个 CPU 评分容器也均 exit0、正常释放，原源代码和输入未变。

这 2 次更新仍处于原 500 次更新 warmup 内。表中分数是有界功能量测，分数差异不能作为质量提升或 FP32/BF16 质量等价结论；本轮不替代原 16 rank/global16严格重放、全量数据训练、长期收敛及未测的真实合成数据 producer 配置。既有严格数值重放限制仍保留。

下面按顺序复现各个实测四 rank 配置；实际 campaign 是四个互斥分组并行执行。先按依赖环境、官方数据准备指南设置绝对目录和已验收镜像 ID。每条命令使用原公开数值入口，并应保留其 runtime、checkpoint、推理和官方评分记录。

```bash
set -euo pipefail
# Use absolute paths. OUTPUT_ROOT must be a new campaign directory.
: "${WORLDENGINE_ROOT:?Set the public repository root}"
: "${PREPARED139:?Set the official NAVTRAIN139/NAVTEST22 prepared directory}"
: "${VAD100_CKPT:?Set the official100 VADv2 checkpoint}"
: "${HYDRA100_CKPT:?Set the official100 HydraMDP checkpoint}"
: "${VOCAB:?Set test_8192_kmeans.npy}"
: "${MAPS:?Set the NuPlan maps directory}"
: "${OFFICIAL_CACHE22:?Set the genuine official metric-cache directory}"
: "${IMAGE:?Set the CPU-accepted dependency image tag or ID}"
: "${EXPECTED_IMAGE_ID:?Set its verified docker image ID}"
: "${OUTPUT_ROOT:?Set a new absolute output directory}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$WORLDENGINE_ROOT"

run_case() {
  local model=$1 precision=$2 devices=$3 cfg initializer stem fresh resumed inference score
  case "$model" in
    vad)
      cfg="$WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py"
      initializer="$VAD100_CKPT" ;;
    hydra)
      cfg="$WORLDENGINE_ROOT/projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py"
      initializer="$HYDRA100_CKPT" ;;
    *) return 2 ;;
  esac
  stem=${cfg##*/}; stem=${stem%.py}
  fresh="$OUTPUT_ROOT/${model}_${precision}_fresh1"
  resumed="$OUTPUT_ROOT/${model}_${precision}_own1to2"
  inference="$OUTPUT_ROOT/${model}_${precision}_eval22"
  score="$OUTPUT_ROOT/${model}_${precision}_official_score22"
  local -a launcher train
  launcher=(python3 ascend/run_container.py
    --devices "$devices" --mini-prepared "$PREPARED139"
    --vocab "$VOCAB" --maps "$MAPS"
    --image "$IMAGE" --expected-image-id "$EXPECTED_IMAGE_ID"
    --precision "$precision" --msda-backend cuda-compatible --deterministic
    --memory 64g --cpus 32 --pids-limit 8192 --shm-size 8g)
  train=(--task train --config "$cfg" --checkpoint "$initializer"
    --runtime --upstream-schedule --batch-size 1 --workers 0
    --steps 2 --checkpoint-interval 1 --check-interval 1)

  # Horizon stays at two updates in both processes; only the first stops early.
  "${launcher[@]}" "${train[@]}" --stop-after 1 --output "$fresh"
  test -s "$fresh/iter_1.pth"
  "${launcher[@]}" "${train[@]}" --resume "$fresh/iter_1.pth" --output "$resumed"
  test -s "$resumed/iter_2.pth"

  # Original public DDP4 inference; the public exporter requires exactly22 tokens.
  "${launcher[@]}" --task inference \
    --config "$resumed/${stem}_rank0.py" --checkpoint "$resumed/iter_2.pth" \
    --output "$inference"

  # Independent original NAVSIM scoring, CPU-only, no accelerator devices.
  docker run --rm --runtime runc --network none \
    --memory 64g --cpus 8 --pids-limit 2048 \
    -e CUDA_VISIBLE_DEVICES= -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "$WORLDENGINE_ROOT:/source:ro" \
    -v "$OUTPUT_ROOT:$OUTPUT_ROOT:rw" \
    -v "$OFFICIAL_CACHE22:$OFFICIAL_CACHE22:ro" -w /source \
    "$EXPECTED_IMAGE_ID" /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
      --predictions "$inference/predictions.pkl" \
      --metric-cache "$OFFICIAL_CACHE22" --output "$score"
}

# Sequential reproduction of each tested four-rank configuration.
run_case vad fp32 0,1,2,3
run_case hydra fp32 4,5,6,7
run_case vad bf16 8,9,10,11
run_case hydra bf16 12,13,14,15
```

训练、完整状态恢复和推理必须选择相同的 `--msda-backend cuda-compatible`。后端策略、源码、镜像、精度、world size、数据和调度共同构成恢复合同。旧源码或另一后端的检查点不能直接作为完整状态的延续。与独立连续训练的数值重放对照需另行执行，原张量 `atol=1e-6, rtol=1e-5`、loss差 `<1e-4` 的门槛保持不变。

## 参考路径与回滚

省略 `--msda-backend`，或显式选择 `--msda-backend reference`，即可使用历史参考路径。使用新输出目录和恢复合同匹配的完整检查点，或从兼容权重初始化新的优化器。原 CUDA 目标、CPU32 目标和整模型比较报告分别保留。
