# AlgEngine 在 Ascend 910C 上的安装与使用

本文是[原 AlgEngine 使用指南](algengine_usage.md)的 NPU 操作版，按数据准备、训练、评价、稀有场景筛选和微调的顺序说明。命令在 Linux NPU 服务器执行，默认当前目录是仓库根目录。原 GPU 脚本保留；NPU 使用 `ascend/run_container.py` 或新增的 `_npu.sh` 前端。

选择 `--msda-backend cuda-compatible` 时，训练、完整状态恢复和推理应保持相同后端；省略该参数则使用默认 `reference`。具体命令及适用范围见[MSDA 后端指南](../ascend/MSDA_CUDA_zh.md)。

WorldEngine 有两个模块：**AlgEngine** 负责模型训练与评价，**SimEngine** 负责渲染、仿真和 rollout 生成。这里操作 AlgEngine；已有的真实 SimEngine 导出可以作为训练输入，本指南不部署其渲染器。

## 常用入口

| 操作 | 入口 |
| --- | --- |
| 构建运行环境 | [公开父镜像](../ascend/docker/README_zh.md)，再构建[应用与编译工具依赖环境](../ascend/docker/dependency_repair/README_zh.md) |
| 选择性下载官方数据和权重 | `ascend/tools/download_algengine_assets.py` |
| 固定官方数据子集 | [NAVTRAIN139 / NAVTEST22](../ascend/data_tools/navtrain139/README_zh.md)；原 NAVTEST-only mini 路径在第3节保留 |
| 完整 NAVTRAIN 输入的 CPU 检查 | `ascend/tools/verify_navtrain_inputs.py`；见[完整输入检查](#完整-navtrain-输入的-cpu-检查) |
| 完整输入短周期训练 | [4 rank、FP32/BF16、8 次更新及自身 4→8 恢复](#完整输入短周期训练) |
| 按原配置训练 | `ascend/run_container.py --config ...` |
| NPU 训练与评价脚本 | `projects/AlgEngine/scripts/e2e_dist_train_npu.sh`、`e2e_dist_eval_npu.sh` |
| 完整断点续训 | 原训练参数，加 `--resume ...` 和新输出目录 |
| 推理 | `ascend/run_container.py --task inference --config <训练导出的配置>` |
| 独立官方评分 | 已有真实 metric-cache 时执行 `ascend/tools/score_cached.py`；需先从日志/地图生成时用 `ascend/official_score.py` |
| 稀有场景筛选 | 原 `projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py` |
| 真实 SimScale 数据准备 | [数据工具说明](../ascend/data_tools/README_zh.md) |
| 复跑实测的 16 芯片同步候选 | [八次更新的专用步骤](../ascend/execution/README_zh.md) |

## 1. 克隆仓库并准备环境

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git rev-parse HEAD
git submodule status
```

先构建[公开父镜像](../ascend/docker/README_zh.md)，再完成[依赖环境及三项 CPU 检查](../ascend/docker/dependency_repair/README_zh.md)。宿主需要可用的 Ascend 驱动、Docker、Python 3 和 NPU 字符设备；镜像提供 PyTorch/Torch-NPU、OpenMMLab、DrivingSDK 和地图处理依赖。不要在此镜像上覆盖安装原仓的 CUDA wheel。

按安装说明取得镜像标签和**实际 image ID**。下文显式传递这两个值，不依赖 launcher 的历史默认镜像。开始新的训练可使用自行构建的镜像；恢复已有断点必须使用原训练的镜像与源码。

以下变量在当前宿主 shell 设置一次。`DATA` 指向同时包含 `alg_engine/` 和 `raw/` 的数据目录，不是 `data/alg_engine/`。`WORK` 放在源码目录之外，每个训练输出选择新目录。

```bash
set -euo pipefail
SOURCE="$PWD"
DATA=/datasets/worldengine/data
WORK=/experiments/worldengine_npu
RUNS="$WORK/runs"
MAPS="$DATA/raw/nuplan/dataset/maps"
LOGS="$DATA/raw/navsim_logs/test"
: "${IMAGE:?Set IMAGE using the container installation guide}"
: "${IMAGE_ID:?Set the verified Docker image ID}"
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$IMAGE_ID"
mkdir -p "$DATA" "$RUNS"
npu-smi info
```

`--devices` 填宿主的**逻辑芯片编号**。实测 910C 服务器的 8 张双芯物理卡提供 16 个逻辑芯片，每个芯片 64 GiB HBM；运行前用 `npu-smi info` 核对本机映射及空闲设备。launcher 仅挂载指定芯片，容器内进程编号从 0 开始。

## 2. 数据下载、权重和目录

所有下载、校验和解压在服务器执行。目录名称以[原仓数据组织说明](data_organization.md)为准，按需准备 AlgEngine 数据即可，无需下载全部约 5 TB 发布资产。

先下载官方 NAVTEST 标注、PDM、8192 词表和已发布初始化权重：

```bash
python3 ascend/tools/download_algengine_assets.py \
  --group navtest --data-root "$DATA" --source hf-mirror
python3 ascend/tools/download_algengine_assets.py \
  --group weights --data-root "$DATA" --source hf-mirror
WEIGHTS="$DATA/alg_engine/ckpts"
VOCAB="$DATA/alg_engine/test_8192_kmeans.npy"
VAD_INIT="$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth"
VAD_BASE="$WEIGHTS/vadv2/e2e_vadv2_50pct_ep8.pth"
HYDRA_BASE="$WEIGHTS/hydramdp/e2e_hydramdp_50pct_ep8.pth"
```

内容锁也包含官方 100% 规划基座，两份权重已完成[整文件大小和 SHA-256 校验](../ascend/evidence/planning-checkpoints-20261016.json)。以这些基座开始新的微调任务时，通过 `--checkpoint` 指定对应文件；训练数据范围仍由配置和 filter 决定。

```bash
VAD_BASE_100="$WEIGHTS/vadv2/e2e_vadv2_100pct_ep8.pth"
HYDRA_BASE_100="$WEIGHTS/hydramdp/e2e_hydramdp_100pct_ep8.pth"
```

Hugging Face及hf-mirror线路请求固定发布revision。魔搭线路请求其 `master`，按同一内容锁的文件大小和SHA256校验，不宣称向魔搭请求了Hugging Face的commit。下载记录分别保存请求revision和内容参考revision。国内线路可选 `--source modelscope`，官方线路用 `--source huggingface`；有效旧文件会复用。工具不下载相机包、地图、原始日志或SimEngine资产，换线路时保留哈希校验。

NAVTEST 标注和 PDM 合计约 **3.64 GB**，相机、词表、权重和地图另计。小样本最终只需实际引用的 JPEG，但这些真实图片需要已经从对应 OpenScene 数据或提供的小数据包中取得。仓库不承诺存在一个恰好包含所选 JPEG 的独立公开下载包。

### 相机图像、场景日志与地图

[原 NAVSIM 测试下载脚本](../ascend/third_party/navsim/download/download_test.sh)提供 OpenScene metadata 和32个相机分片的地址。按所需日志选择分片；下面的 `SHARD=0` 是下载示例，其覆盖范围未确认包含制作 mini 所需的三个有效日志。原整套脚本还下载全部相机和 LiDAR，小样本验证应按引用选择分片。示例使用[输入锁](../ascend/data_tools/INPUT_LOCKS.json)记录的 OpenScene revision；这两个 test 包的下载命令尚未实际执行验证。

```bash
DOWNLOADS="$WORK/downloads"
EXTRACT="$DATA/raw/openscene_download_test"
SHARD=0   # Select a required shard; the original test camera shards are numbered 0..31.
OS_REV=a76f840b65e972bc45e56c2adced897498e9a026
OS_URL="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/$OS_REV/openscene-v1.1"
mkdir -p "$DOWNLOADS"
test ! -e "$EXTRACT"
mkdir -p "$EXTRACT"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/openscene_metadata_test.tgz.part" \
  "$OS_URL/openscene_metadata_test.tgz"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" \
  "$OS_URL/openscene_sensor_test_camera/openscene_sensor_test_camera_${SHARD}.tgz"
tar -tzf "$DOWNLOADS/openscene_metadata_test.tgz.part" > "$DOWNLOADS/metadata_members.txt"
tar -tzf "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" > "$DOWNLOADS/camera_members.txt"
sed -n '1,20p' "$DOWNLOADS/metadata_members.txt"
sed -n '1,20p' "$DOWNLOADS/camera_members.txt"
tar -xzf "$DOWNLOADS/openscene_metadata_test.tgz.part" -C "$EXTRACT"
tar -xzf "$DOWNLOADS/openscene_sensor_test_camera_${SHARD}.tgz.part" -C "$EXTRACT"
test -d "$EXTRACT/openscene-v1.1/meta_datas"
test -d "$EXTRACT/openscene-v1.1/sensor_blobs"
mkdir -p "$DATA/raw/openscene-v1.1/sensor_blobs" "$DATA/raw/navsim_logs"
test ! -e "$DATA/raw/openscene-v1.1/sensor_blobs/test"
test ! -e "$DATA/raw/navsim_logs/test"
# Relative links remain inside DATA when it is mounted as /data/full.
ln -s ../../openscene_download_test/openscene-v1.1/sensor_blobs \
  "$DATA/raw/openscene-v1.1/sensor_blobs/test"
ln -s ../openscene_download_test/openscene-v1.1/meta_datas "$DATA/raw/navsim_logs/test"
sha256sum "$DOWNLOADS"/*.tgz.part > "$DOWNLOADS/transfer_SHA256SUMS"
```

先检查压缩包成员和最终解析路径。缺少其他日志时，只补下载对应分片并解压到该 OpenScene 树，避免覆盖已验证的 JPEG。`transfer_SHA256SUMS` 记录本次下载文件的内容，不是预先核验的发布者哈希。准备工具会实际解码所选 JPEG；覆盖不足时应根据报告补数据。

地图来源是[原 maps 下载脚本](../ascend/third_party/navsim/download/download_maps.sh)中的官方地址。已有完整地图直接设置 `MAPS`，无需重复下载：

```bash
MAP_STAGE="$DATA/raw/nuplan/dataset"
mkdir -p "$MAP_STAGE"
test ! -e "$MAP_STAGE/nuplan-maps-v1.0"
test ! -e "$MAP_STAGE/maps"
curl --fail --location --retry 3 --continue-at - \
  --output "$DOWNLOADS/nuplan-maps-v1.1.zip.part" \
  https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip -l "$DOWNLOADS/nuplan-maps-v1.1.zip.part"
unzip "$DOWNLOADS/nuplan-maps-v1.1.zip.part" -d "$MAP_STAGE"
test -f "$MAP_STAGE/nuplan-maps-v1.0/nuplan-maps-v1.0.json"
ln -s nuplan-maps-v1.0 "$MAP_STAGE/maps"
MAPS="$MAP_STAGE/maps"
LOGS="$DATA/raw/navsim_logs/test"
```

URL 的 archive 版本与解压后的 `nuplan-maps-v1.0` 目录名沿用原脚本。保留地图 JSON 及各地区 `.gpkg`。完整 NAVTRAIN 相机包由[原 NAVTRAIN 下载脚本](../ascend/third_party/navsim/download/download_navtrain.sh)给出，只安排 AlgEngine 所需的训练图片，保持 `trainval` 目录；模型训练无需 SimEngine 场景资产。

完整 NAVTRAIN 还需以下两个元数据文件：

```bash
python3 ascend/tools/download_algengine_assets.py \
  --group navtrain --data-root "$DATA" --source hf-mirror
```

合计 **39,862,469,866 字节**。图像、地图和原始日志单独安排。相机压缩包必须解压，让标注中的 `cams[*].data_path` 能在相机根目录下读取；把 `.tar.gz` 放在目录里并不能直接训练。

```text
DATA/
├── alg_engine/
│   ├── ckpts/
│   │   ├── vadv2/e2e_vadv2_50pct_ep8.pth
│   │   ├── hydramdp/e2e_hydramdp_50pct_ep8.pth
│   │   └── track_map_nuplan_r50_navtrain_{50,100}pct_bs1x8.pth
│   ├── merged_infos_navformer/nuplan_openscene_{navtrain,navtest}.pkl
│   ├── pdms_cache/pdm_8192_gt_cache_{navtrain,navtest}.pkl
│   └── test_8192_kmeans.npy
└── raw/
    ├── openscene-v1.1/sensor_blobs/{trainval,test}/
    ├── nuplan/dataset/maps/nuplan-maps-v1.0.json
    │                         # plus the corresponding regional .gpkg maps
    └── navsim_logs/test/      # genuine logs for independent CPU rescoring
```

地图除 `nuplan-maps-v1.0.json` 外，还需对应区域的真实 `.gpkg`。`LOGS` 应指向与选中 token 相符的 NAVSIM 原始日志目录，独立 CPU 评分时使用。

## 推荐流程：NAVTRAIN139 / NAVTEST22

固定小数据验证使用[公开准备工具](../ascend/data_tools/navtrain139/README_zh.md)：从原 NAVTRAIN 日志保留139个训练锚点，从另一条 NAVTEST 日志保留22个评价锚点，原标签和8192候选 PDM 保持。这是从官方数据确定性派生的验证子集，不是官方单独发布的标准 mini，也不代表完整基准。

### 准备环境、原始输入和权重

先构建公开 post8 父镜像，再按[应用与编译工具依赖环境](../ascend/docker/dependency_repair/README_zh.md)构建最终镜像，执行其中三条 CPU 验收命令。将 `IMAGE` 和 `IMAGE_ID` 设置为最终镜像。实测修复环境的两个原始 `pip check` 及重新加载 CANN 后的 API 检查已通过；CPU 环境检查不替代模型验收。

按第2节下载标注、PDM和官方权重，包含 `--group navtrain`。准备器的[输入锁](../ascend/data_tools/navtrain139/INPUTS.lock.json)列出六个原文件的准确地址、revision、大小和完整哈希，已有文件经校验后直接复用。两个相机包必须是 **mini camera 12** 和 **test camera 30**。无缓存时，六个文件共47,913,227,335字节。[独立官方 API 核验](../ascend/evidence/official-data-origins-20261016.json)于2026-10-16逐项确认六文件的固定 revision、路径、大小和 LFS 哈希；该查询只取元信息，不下载文件内容。无需下载整套相机或 SimEngine 资产。

执行数据工具文档中的[完整 CPU 准备命令](../ascend/data_tools/navtrain139/README_zh.md#在-linux-服务器执行)，设置 `OUTPUT_PARENT="$WORK/navtrain139"`，先创建父目录，保持其中 `prepared/` 不存在。工具直接读取两个原始相机包，无需先解压；最终只复制1832张实际引用的原 JPEG，并输出 launcher 可直接读取的训练/评价 split。此输出不再调用 `split_mini.py`。原文件哈希、CPU 重建、完整元信息比较与图像检查见[VALIDATION.json](../ascend/data_tools/navtrain139/VALIDATION.json)。

第2节定义的 `VOCAB`、`VAD_BASE_100` 和 `HYDRA_BASE_100` 分别指向官方词表和100%训练的规划基座。下方 `--mini-prepared` 选择139/22验证数据；权重名称不改变训练数据范围。`MAPS` 需指向完整地图目录。

### 短周期验证：八次更新

有限设备时段优先使用此路径。为 VADv2/HydraMDP 和 FP32/BF16 四种组合分别执行，输出目录彼此独立。连续分支运行8次更新；另一独立分支仍设置 `--steps 8`，仅用 `--stop-after 4` 提前停止，再从自身第4次更新断点恢复至8。不能把中断分支改成 `--steps 4`，否则学习率调度合同不同。此流程使用 IterBasedRunner，不代表八个 epoch 或收敛。

```bash
PREPARED="$WORK/navtrain139/prepared"
test -f "$PREPARED/manifest.json"
test -f "$PREPARED/preparation.json"
test -f "$MAPS/nuplan-maps-v1.0.json"
MODEL=vad                 # vad or hydra
PRECISION=fp32            # repeat separately with bf16
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
case "$MODEL" in
  vad) CONFIG=worldengine/e2e_vadv2_100pct.py; INITIALIZER="$VAD_BASE_100" ;;
  hydra) CONFIG=hydramdp/e2e_hydramdp_100pct.py; INITIALIZER="$HYDRA_BASE_100" ;;
  *) printf 'Select MODEL=vad or MODEL=hydra\n' >&2; exit 2 ;;
esac
STEM=$(basename "$CONFIG" .py)
PREFIX="$RUNS/${MODEL}_navtrain139_${PRECISION}_updates8"
SHORT_COMMON=(--devices "$DEVICES" --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$SOURCE/projects/AlgEngine/configs/$CONFIG"
  --mini-prepared "$PREPARED" --checkpoint "$INITIALIZER" --vocab "$VOCAB"
  --maps "$MAPS" --runtime --steps 8 --batch-size 1 --workers 0
  --precision "$PRECISION" --deterministic --upstream-schedule
  --checkpoint-interval 4 --check-interval 1 --memory 256g --cpus 128
  --pids-limit 8192)

python3 ascend/run_container.py "${SHORT_COMMON[@]}" --output "${PREFIX}_continuous"
python3 ascend/run_container.py "${SHORT_COMMON[@]}" --stop-after 4 \
  --output "${PREFIX}_partial"
python3 ascend/run_container.py "${SHORT_COMMON[@]}" \
  --resume "${PREFIX}_partial/iter_4.pth" --output "${PREFIX}_resumed"
END_CHECKPOINT=iter_8.pth
END_UPDATES=8
```

选定这一路径后，跳过下一节八个 epoch 命令，继续执行共用的 CPU 数值比较、独立22个 token 推理和官方评分。使用刚设置的 `PREFIX`、`END_CHECKPOINT` 和 `END_UPDATES`。

### 原生 epoch 流程：八个 epoch

需要原生 epoch 验证时，选择这一替代路径；不要把它与上面的八次更新断点混用。每次使用新输出目录；连续训练、中断训练和恢复训练均保留八个 epoch 的相同调度终点。

```bash
PREPARED="$WORK/navtrain139/prepared"
test -f "$PREPARED/manifest.json"
test -f "$PREPARED/preparation.json"
test -f "$MAPS/nuplan-maps-v1.0.json"
MODEL=vad                 # vad or hydra
PRECISION=fp32            # bf16 is an experimental, separate run
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
case "$MODEL" in
  vad) CONFIG=worldengine/e2e_vadv2_100pct.py; INITIALIZER="$VAD_BASE_100" ;;
  hydra) CONFIG=hydramdp/e2e_hydramdp_100pct.py; INITIALIZER="$HYDRA_BASE_100" ;;
  *) printf 'Select MODEL=vad or MODEL=hydra\n' >&2; exit 2 ;;
esac
STEM=$(basename "$CONFIG" .py)
PREFIX="$RUNS/${MODEL}_navtrain139_${PRECISION}_epochs8"
COMMON=(--devices "$DEVICES" --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$SOURCE/projects/AlgEngine/configs/$CONFIG"
  --mini-prepared "$PREPARED" --checkpoint "$INITIALIZER" --vocab "$VOCAB"
  --maps "$MAPS" --runtime --epochs 8 --batch-size 1 --workers 0
  --precision "$PRECISION" --deterministic --upstream-schedule
  --checkpoint-interval 4 --check-interval 1 --memory 256g --cpus 128
  --pids-limit 8192)

python3 ascend/run_container.py "${COMMON[@]}" --output "${PREFIX}_continuous"
python3 ascend/run_container.py "${COMMON[@]}" --stop-after-epochs 4 \
  --output "${PREFIX}_partial"
python3 ascend/run_container.py "${COMMON[@]}" \
  --resume "${PREFIX}_partial/epoch_4.pth" --output "${PREFIX}_resumed"
END_CHECKPOINT=epoch_8.pth
END_UPDATES=72
```

139个锚点在16 rank、每 rank B1时，每 epoch 为9次更新。连续训练与恢复终点为72次更新，中断分支停在36次。这验证小子集上的原生 epoch 训练流程，不代表原103,288锚点训练集已收敛。256g是此子集的训练内存分配，不能用于推断全量训练容量。设备列表表示同一宿主上的16个逻辑芯片，运行前核对物理卡映射与空闲状态。

默认 `fp32` 关闭公开 Matmul/Conv HF32 开关。`bf16` 使用原生NPU前向autocast并关闭历史/当前帧共享权重缓存，主参数、梯度、优化器状态、MSDA和规划损失保持FP32，不使用loss scaler。几何与原BatchNorm/SyncBatchNorm在FP32岛内计算，保留原模块、运行统计和collectives。源码`e14dd6b2…`/244个Python文件及实测`3522…`镜像下，两类官方100%规划配置均完成[16rank八次BF16更新、自身恢复及独立22token推理/评分](../ascend/results/bf16-v7-navtrain139-short8-20261016.json)。[全部rank和保存的Adam状态](../ascend/results/bf16-v7-gradient-coverage-20261016.json)验证此前每模型138个覆盖缺项恢复，但两组完整数值重放仍超未放宽的原容差。8次更新在500次warmup内，不证明收敛、质量提升或数值等价。测试此显式试验模式时设`PRECISION=bf16`，使用独立输出和自身断点；FP32断点不能按BF16合同继续恢复。

v6 执行记录不再作为训练验收依据：[匹配检查点确认两个 BF16 模型相对 FP32 均缺少138个优化器状态ID](../ascend/results/bf16-v6-optimizer-coverage-20261016.json)。[真实 CPU 缓存实验](../ascend/results/bf16-autocast-history-cache-cpu-20261016.json)还复现了历史 `no_grad` 后共享层缺梯度，而 loss 仍有限、head 仍更新的现象。CPU机制和旧检查点覆盖缺陷均已确认；v7的覆盖修复另经真实NPU运行及保存状态验证，不能据此解释旧FP32严格重放差异或宣布通用数值修复。

[已完成的FP32／BF16对照](../ascend/results/fp32-bf16-v7-matched8-planning-20261016.json)包含四个模型／阶段组合，同一22个独立锚点、每组全部40pose。相同官方权重的推理比较与各精度独立训练8次更新后的比较分开解释；量测未设质量等价阈值，也不是GPU对照。

在无 NPU 的 CPU 容器内，对两个完整终点执行原比较器：

```bash
COMPARISON="${PREFIX}_comparison"
mkdir -p "$COMPARISON"
COMPARE_EXIT=0
docker run --rm --runtime runc --network none --memory 24g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$RUNS:$RUNS:ro" \
  -v "$COMPARISON:/checks:rw" -w /source "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
' --continuous "${PREFIX}_continuous/$END_CHECKPOINT" \
  --resumed "${PREFIX}_resumed/$END_CHECKPOINT" \
  --continuous-run "${PREFIX}_continuous" --partial-run "${PREFIX}_partial" \
  --resumed-run "${PREFIX}_resumed" --output /checks/comparison.json \
  || COMPARE_EXIT=$?
python3 - "$COMPARE_EXIT" "$COMPARISON/comparison.json" "$END_UPDATES" <<'PY'
import json, sys
from pathlib import Path
code = int(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text())
expected = {0: "pass", 1: "fail"}
assert code in expected and report["result"] == expected[code]
assert report["tensor_tolerance"] == {"atol": 1e-6, "rtol": 1e-5}
assert report["loss_absolute_tolerance"] == 1e-4
assert report["world_size"] == 16 and report["completed_updates"] == int(sys.argv[3])
print("Original numerical comparison:", report["result"])
PY
```

比较覆盖模型/优化器张量、各 rank RNG、batch token、学习率和损失。原张量容差为 `atol=1e-6, rtol=1e-5`，最大损失差需小于 `1e-4`。对于已完成且超出原容差的对照，命令保留原 JSON 的 `result=fail` 和退出1，并检查报告完整性，让后续独立推理/评分继续；报告缺失或不一致仍会中止流程。即使载入与训练都正常退出，比较超限仍须保留为数值结果，不得绕过源码、数据、镜像或精度合同。

### 独立推理和官方 CPU 评分

使用所选短周期或 epoch 分支导出的配置与自身终点断点。`END_CHECKPOINT` 对应 `iter_8.pth` 或 `epoch_8.pth`。若评价连续分支，替换相应路径，并选择另一个输出目录。

```bash
INFERENCE="${PREFIX}_resumed_eval"
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "${PREFIX}_resumed/${STEM}_rank0.py" \
  --checkpoint "${PREFIX}_resumed/$END_CHECKPOINT" --mini-prepared "$PREPARED" \
  --vocab "$VOCAB" --maps "$MAPS" --precision "$PRECISION" --deterministic \
  --memory 64g --cpus 16 --pids-limit 2048 --output "$INFERENCE"

CACHE=/datasets/worldengine/official_metric_cache
SCORE="${PREFIX}_resumed_score"
mkdir -p "$SCORE"
docker run --rm --runtime runc --network none --memory 24g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$INFERENCE:/predictions:ro" \
  -v "$CACHE:$CACHE:ro" -v "$SCORE:/output:rw" -w /source "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
  --predictions /predictions/predictions.pkl --metric-cache "$CACHE" \
  --output /output/score
cat "$SCORE/score/official_score.json"
```

评分需要覆盖22个预测 token 的原官方 NAVSIM metric-cache。训练用 PDM 标签不能代替它。工具仅选取本次预测的 token，不改缓存内容和评分数学。若缓存使用软链，需在容器内保留目标可达；必要时将目标根目录按同一绝对路径只读挂载。没有官方缓存时，按第6节从真实日志与地图生成。

分别检查 `inference_validation.json`、22个 token 的官方评分及原数值比较 JSON。独立的[官方100权重GPU／NPU FP32对照](../ascend/results/gpu-npu-official100-eval22-20261016.json)限同一22个评价锚点，不证明完整NAVTRAIN收敛、完整NAVTEST质量、GPU训练或通用16 rank数值复现。当前模型结果见[验证报告](../ascend/VALIDATION_zh.md)，不能用环境或数据准备的 CPU 通过结果代替模型通过。

可使用[公开CPU比较工具](../tools/validation/README.md)从原预测和官方CSV重算全部位姿及七项评分差分；同一接口可用于FP32／BF16或GPU／NPU输出，不设置质量等价阈值。

## 3. 准备真实小数据，完成单芯片八次更新

这是另一条 NAVTEST-only smoke 数据路径。需要真实 NAVTRAIN 训练锚点和固定独立评价集时，使用上面的 NAVTRAIN139/NAVTEST22 推荐流程。

此流程从真实 NAVTEST 标注、PDM 和可读取的相机图像中抽取小样本，保留原标签及 8192 项奖励向量。训练、评价按整段驾驶日志分开，用于适配验证；它不是完整官方 NAVTRAIN/NAVTEST 基准。

数据准备在镜像内用 CPU 执行，不挂载 NPU。`WORK` 在容器内外使用相同绝对路径，后续 split 创建的相机链接才能继续使用。

```bash
RARE="$SOURCE/projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml"
MINI="$WORK/mini_real_data"
MINI_RARE="$WORK/prepared/worldengine_navtest_smoke_rare.yaml"
test -d "$DATA/raw/openscene-v1.1/sensor_blobs/test"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$DATA:$DATA:ro" -v "$WORK:$WORK:rw" \
  -w /source "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python ascend/tools/prepare_real_mini.py \
      --annotations "$1/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl" \
      --pdm "$1/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl" \
      --cameras "$1/raw/openscene-v1.1/sensor_blobs/test" \
      --rare-split /source/projects/AlgEngine/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml \
      --camera-mode copy \
      --output "$2/prepared"
    python ascend/split_mini.py --prepared "$2/prepared" \
      --output "$2/mini_real_data" \
      --rare-split "$2/prepared/worldengine_navtest_smoke_rare.yaml"
  ' bash "$DATA" "$WORK"
cat "$MINI/manifest.json"
```

`WORK/prepared` 和 `MINI` 必须是新目录。准备器默认复制选中 JPEG，要求至少一个真实 rare 锚点、同日志内可检索的非 rare 锚点，以及独立的普通日志，并检查当前及历史图像。完整 `RARE` 文件仅用于准备时定义稀有场景；生成的 `MINI_RARE` 只列出选中且原队列完整的 rare 锚点。拆分和 mini 微调都使用 `MINI_RARE`，避免仅作为历史帧保留的其他 rare token 被当作训练锚点。如果图像缺失或无法形成日志独立的划分，先补齐输入，不能用随机图片或伪造样本替代。24g 是此 NAVTEST 小数据准备的资源示例，不是完整 NAVTRAIN 内存预算。

新准备器默认选择三个日志，每日志最多 22 个锚点，拆分前合计最多 66 个。实际训练／评价数量和哈希以本次 `manifest.json` 为准，不等同于此前实测包固定的 42／22 个锚点。用新数据开始新训练，并恢复自身断点；不能因文件名相同就引用旧包的数值或质量结果。

用一个逻辑芯片、原 VAD 配置和官方感知初始化完成八次更新：

```bash
python3 ascend/run_container.py --devices 0 \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/vad_mini8"
```

`--steps 8` 表示八次优化器更新，保留原 optimizer LR/warmup 参数，但 horizon 为八次 iteration；它不会完成原配置的八个 epoch。`fp32` 关闭公开 Matmul 和 Conv HF32 开关。成功后应产生 `exit_code`、`runtime_rank0.json/jsonl`、`iter_4.pth`、`iter_8.pth` 和 `e2e_vadv2_50pct_rank0.py`。

Hydra 使用 `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct.py`，初始化仍为同一感知权重，并换新输出。感知模型用 `projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py`，初始化改为 `ckpts/bevformerv2-r50-t1-base_epoch_48.pth`，增加 `--maps "$MAPS"`。

若习惯原仓的 shell 方式，可在新输出执行同一小数据检查。前端参数顺序是 `<config> <num_chips> [resume_checkpoint]`：

```bash
export IMAGE IMAGE_ID VOCAB
export INIT_CHECKPOINT="$VAD_INIT" MINI_PREPARED="$MINI"
export RUN_OUTPUT="$RUNS/vad_mini8_shell" CPU_MEMORY=64g
bash projects/AlgEngine/scripts/e2e_dist_train_npu.sh \
  configs/worldengine/e2e_vadv2_50pct.py 1 -- \
  --steps 8 --checkpoint-interval 4 --check-interval 1 --deterministic
```

该前端调用同一 NPU launcher，在单台服务器使用指定数量的芯片，不会把 16 芯片解释成两台 8 GPU 服务器。实际芯片不是从 0 连续编号时，设置 `DEVICE_IDS`。完整数据训练应取消 `MINI_PREPARED`，改为导出 `DATA_ROOT="$DATA"`；两者只能选一个。

## 4. 按原 NAVTRAIN 50%／100% 配置训练

### 完整 NAVTRAIN 输入的 CPU 检查

[`verify_navtrain_inputs.py`](../ascend/tools/verify_navtrain_inputs.py)仅检查原始 NAVTRAIN 100% split。它复用所选源码的原 `inspect_phase`，核对全部锚点、当前帧与前三帧历史、8 个相机，以及各 PDM 数组的 8192 项形状和有限值。标注、PDM、词表整文件按官方锁定值校验；每张所需 JPEG 均计算内容 SHA-256，再执行 verify、重新打开和完整解码。公开 CLI 还要求原始 1920×1080 RGB JPEG。该工具不加载模型，也不检查 NAVTEST、地图或独立评分 cache；评价输入仍需执行下方 preflight。

输出目录必须全新，并放在源码和数据目录之外。以下容器仅使用 CPU，没有任何加速器设备挂载，镜像 ID 沿用第 1 节的核验值。环境需有 `preflight.py` 使用的 MMCV、NumPy、PyYAML，以及解码所需的 Pillow。

```bash
: "${DEADLINE_UTC:?Set a future UTC cutoff, including Z or +00:00}"
QC_DIR="$WORK/navtrain100_inputs"
mkdir "$QC_DIR"

docker run --rm --runtime runc --network none --memory 128g --cpus 8 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" -v "$QC_DIR:/output:rw" -w /source \
  "$IMAGE_ID" /opt/conda/envs/bevformer/bin/python ascend/tools/verify_navtrain_inputs.py \
    --source /source --data-root /data \
    --config projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py \
    --decode-workers 4 --output /output/navtrain100.json \
    --deadline-utc "$DEADLINE_UTC" --stop-file /output/STOP
```

同时检查退出码与 `navtrain100.json`。通过条件包括 `actual_exit=0`、`result=pass_complete_original_NAVTRAIN_inputs_and_required_JPEG_decode_CPU_only`、错误列表为空、请求与选中锚点一致、标注/图像/PDM 无缺失或无效项，以及 `decoded=decode_passed=unique_required_JPEGs`、`decode_failed=0`。完成的 `navtrain100.json.images.jsonl` 逐文件记录相对路径、内容 SHA-256 和解码结果；报告还保存该清单的摘要、源码/配置/filter/资产身份，以及前后内存和设备状态。残留 `.part` 或中断/拒绝报告不能作为完整覆盖结果。

截止时间必填，必须为未来的显式 UTC 时间，不再限制到某个固定日期。创建 `$QC_DIR/STOP` 可请求协同停止。4 个解码 worker 最多排队 8 个 future。容器的 128g 限额用于输入检查；此前检查器峰值 RSS 约 51.99 GB 仅作参考，不能作为训练内存预算。

[10 月 20 日完整数据结果](../ascend/results/full-navtrain-inputs-20261020.json)覆盖 103,288 个锚点和 1,219,960 张所需 JPEG，实际使用的是原私有检查器，其 SHA-256 为 `453d8e893ccef3720b4323e12e6c862f0fdee5020b85384c3ef37e1b3c31ab0b`，原报告 SHA-256 为 `882d5a98a5476b7ae135cd9b33a3c37f6c65fa7d3ef3d5a7c36d6d7bfdf72284`。公开 CLI 保留检查算法，目前只完成 14 项本地合成 CPU 夹具，尚未重复全量运行。新增截止控制、CPU 门禁和显式图像尺寸门禁，不计作新的全量数据或 NPU 训练验收。夹具命令为 `python -m unittest discover -s ascend/tools -p test_verify_navtrain_inputs.py -v`。

### 完整输入短周期训练

[10 月 20 日实测](../ascend/results/full-navtrain-short8-20261020.json)使用完整的 103,288 个 NAVTRAIN 锚点 loader 及[检查通过的 1,219,960 张 JPEG](../ascend/results/full-navtrain-inputs-20261020.json)。VADv2、HydraMDP 分别完成 FP32 和 BF16 模仿训练，每组 4 rank、8 次优化器更新，并精确恢复自身第 4 次更新的模型和优化器状态。32 份 rank 报告、128 条逐 rank 更新记录均通过。各组实际使用同一组 32 个当前锚点、128 个历史／当前输入帧，不代表完整 epoch 或收敛。两个 `e2e_*_100pct` 配置均未启用 HydraMDP 策略梯度微调。

服务器为 8 张双芯片物理 NPU 卡、16 个芯片；本次使用物理卡 0–1 的芯片 0–3。实测运行源码为 `0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922`／245 个 Python 文件，镜像为 `sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`。配置中的 split YAML 在 runtime 哈希前按 AlgEngine 目录解析；旧源码合同的 checkpoint 不能跨此次源码变更恢复完整状态，应初始化新训练后使用自身断点。

下面一次执行一个 `MODEL`／`PRECISION` 组合，其余组合使用新输出重复。`DATA` 指完整数据根目录，不传 `--mini-prepared`。512g／32 CPU 是本次 4 rank 实测分配，不是最低资源要求。原 optimizer 和 LR 选项保留，horizon 显式设为 8 次更新；此时 500 次更新的 warmup 尚未结束。

```bash
MODEL=vadv2                         # or hydramdp
PRECISION=fp32                      # or bf16; keep unchanged on resume
DEVICES=0,1,2,3                     # verify actual host ownership/mapping
case "$MODEL" in
  vadv2) CONFIG=projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py
         BASE="$VAD_BASE_100" ;;
  hydramdp) CONFIG=projects/AlgEngine/configs/hydramdp/e2e_hydramdp_100pct.py
            BASE="$HYDRA_BASE_100" ;;
  *) exit 2 ;;
esac
FULL_RUN="$RUNS/${MODEL}_${PRECISION}_fullinput"

python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$CONFIG" --data-root "$DATA" --checkpoint "$BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --stop-after 4 --upstream-schedule \
  --checkpoint-interval 4 --check-interval 1 --batch-size 1 --workers 0 \
  --precision "$PRECISION" --msda-backend cuda-compatible \
  --memory 512g --cpus 32 --output "${FULL_RUN}_first4"

python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$CONFIG" --data-root "$DATA" --checkpoint "$BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --upstream-schedule \
  --checkpoint-interval 4 --check-interval 1 --batch-size 1 --workers 0 \
  --precision "$PRECISION" --msda-backend cuda-compatible \
  --resume "${FULL_RUN}_first4/latest.pth" \
  --memory 512g --cpus 32 --output "${FULL_RUN}_resume8"
```

两条命令均须退出 0。逐 rank 核对 0→4／4→8、各 4 次更新、原始 loss 有限、所监控参数更新并同步；恢复阶段均须报告 `pass_exact_saved_state_restore`。`--check-interval 1` 对每次更新已存在的梯度执行有限值门禁。同步检查只覆盖一个规划参数，使用 `atol=1e-6, rtol=1e-5`，不代表完整模型逐位对照。独立 CPU 检查确认 8 份 checkpoint 保存的模型和 Adam Tensor 均有限；恢复数量分别为 VADv2 889／777、HydraMDP 909／837。精确保存状态恢复与连续 8 步／中断后 4+4 步的数值重放等价分别验收。

BF16 记录到 Conv2d／Linear 的真实 BF16 激活，主参数、已有梯度和 Adam 状态保持 FP32；MSDA、几何、规划 loss 和归一化在 FP32 区域执行。训练结束时的命名元数据覆盖 VADv2／HydraMDP 的 730／750 个参数，其中 259／279 个有梯度记录，各有 310 个可训练参数未记录到梯度。保存的优化器参数组、状态 ID 和 schema 与 FP32 覆盖一致；这说明成员关系和 dtype，不是完整梯度值对照。

独立[单芯片 FP32 实测](../ascend/results/single-chip-capacity-20261020.json)对两模型各运行 4 次更新，batch=1、workers=0，分配 192g／16 CPU。单进程峰值 RSS 为 50.69／50.79 GiB；NPU allocator 峰值 allocated 为 13.05／13.07 GiB、reserved 为 19.81／19.81 GiB。复现该范围时只执行上面的第一条命令，选择一个可用宿主芯片，设置 `PRECISION=fp32`、`--memory 192g --cpus 16` 并换新输出。该实测不覆盖单芯片 BF16、续训、完整 epoch、最低内存或吞吐；进程 RSS 与 allocator reserved 也不代表主机总内存或设备 HBM 总占用。

[完整 NAVTEST 输入检查](../ascend/results/full-navtest-inputs-20261020.json)另行通过 12,146 个锚点、136 个日志和 142,976 张所需 JPEG，包含前三帧历史和 8 路相机。该输入证据可供六组评价复用；完整模型推理、官方 metric-cache 校验与官方评分仍分别提供结果。同机并行作业的时间记录不作为性能基准。

### 训练与评价 preflight

完整训练前先检查标注、当前／历史图片、PDM、词表、地图及独立 NAVTEST 评分 cache。`CACHE` 必须覆盖配置所需的 NAVTEST token，包含官方 NAVSIM payload 和 `metadata/*.csv` 索引。`alg_engine/pdms_cache/` 下的标签 PDM 是另一类输入，不能替代这个独立评分 cache。

```bash
CACHE=/datasets/worldengine/metric_cache_navtest_v1
PREFLIGHT="$WORK/preflight_vad50"
mkdir -p "$PREFLIGHT"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 128g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" -v "$MAPS:/maps:ro" \
  -v "$CACHE:/metric_cache:ro" -v "$PREFLIGHT:/output:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export WORLDENGINE_ROOT=/source
    python ascend/preflight.py \
      --config /source/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
      --data-root /data --maps /maps --metric-cache /metric_cache \
      --require-official-score --hash-assets --ranks 16 --output /output/preflight.json
  '
cat "$PREFLIGHT/preflight.json"
```

没有完整评分 cache 时，从真实 NAVSIM 日志和地图按锁定的 [NAVSIM cache 文档](../ascend/third_party/navsim/docs/cache.md)生成。[原缓存脚本](../ascend/third_party/navsim/navsim/planning/script/run_metric_caching.py)可在 CPU 使用 `train_test_split=navtest worker=sequential`，路径须指向真实数据；这一过程无需 SimEngine 渲染器。

新 cache 目录的 CPU 命令如下：

```bash
mkdir -p "$CACHE"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 64g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$LOGS:/logs:ro" -v "$MAPS:/maps:ro" \
  -v "$CACHE:/output:rw" -w /source "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:${PYTHONPATH:-}
    export NAVSIM_EXP_ROOT=/output OPENSCENE_DATA_ROOT=/unused NUPLAN_MAPS_ROOT=/maps
    python ascend/third_party/navsim/navsim/planning/script/run_metric_caching.py \
      train_test_split=navtest worker=sequential navsim_log_path=/logs cache.cache_path=/output
  '
```

使用原官方 split 和 cache generator。64g 为资源示例，完整 cache 生成的预算尚未实测。检查实际 payload 数、日志及 preflight token 覆盖；上游 cache 脚本在记录异常 entry 后仍可能以0退出，需要核对每个 token 的结果。

preflight 检查图片文件存在性、大小、历史队列、token 覆盖和 PDM 数值，不逐张解码图片。检查报告无错误后，再确定完整训练内存。实测 16 份原 NAVTRAIN dataset 常驻对象的合计 RSS 约 **834 GB**，尚未计入完整图像、模型／optimizer、worker 和评价。`TRAIN_MEMORY` 要根据宿主可用 RAM 和训练实测预算设置，不能沿用 mini 的 64g。

原规划配置为 8 GPU，每 GPU batch=2，global batch=16。下例用 **16 个逻辑芯片，每芯片 batch=1**，保持样本总数；分组方式发生变化，尚无 GPU 数值对照。4 张物理卡／8 芯片 batch=2 的容量尚未实测。

```bash
: "${TRAIN_MEMORY:?Set an adequate full-training Docker memory limit}"
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --data-root "$DATA" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --batch-size 1 --workers 0 --precision fp32 --deterministic \
  --maps "$MAPS" --official-metric-cache "$CACHE" --validate \
  --memory "$TRAIN_MEMORY" --cpus 64 --pids-limit 8192 \
  --output "$RUNS/vad50_full"
```

不传 `--steps`、`--epochs`，保留原配置八个 epoch、optimizer 和 LR horizon。迁移入口的 `--validate` 每个 epoch 执行评价。100% 训练将配置改为 `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py`，保留原配置的 `track_map_nuplan_r50_navtrain_50pct_bs1x8.pth` 初始化器，并换新输出目录。原 50% filter 为 50,000 个锚点，完整 split 为 103,288；完整输入下的 8 次更新训练已完成，见上方记录；完整 epoch 训练及收敛仍待验收。

单独的[同步候选步骤](../ascend/execution/README_zh.md)只允许两个已测规划配置和八次更新，不用于完整 epoch 训练。后续四组独立重复已完成，其中两组通过原容差，两组超出原容差；同步设置尚不能作为通用修复。

## 5. 保存和恢复完整训练状态

`--checkpoint` 是模型初始化权重。`--resume` 恢复模型、optimizer、每个 rank 的 RNG、采样进度和 LR。官方发布的仅权重模型不能当作迁移入口的完整断点。

接第 3 节，从**同一次连续训练**的 `iter_4.pth` 恢复到原第八次更新，输出换新目录：

```bash
python3 ascend/run_container.py --devices 0 \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$VAD_INIT" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --resume "$RUNS/vad_mini8/iter_4.pth" \
  --output "$RUNS/vad_mini8_resumed"
```

演练 epoch 中断时，完整训练命令增加 `--stop-after-epochs 2`，仍保留原八个 epoch horizon。退出并生成 `epoch_2.pth` 后，使用相同参数去掉停止选项，增加 `--resume "$RUNS/vad50_partial/epoch_2.pth"`，再换新输出；初次输出需命名为 `vad50_partial`。不要把 LR horizon 改成两个 epoch。

精确状态恢复要求 `--workers 0`，源码／配置字节、镜像、精度及初始环境、world size、数据／过滤／词表、batch、seed 和 schedule 保持一致。合同不匹配时定位差异，不能绕过检查。保存原训练 checkout 和数据。launcher 必须显式传 `--resume`，不会因目录已有 `latest.pth` 自动续训。

以下 CPU 命令比较两个第八次更新的终点。使用安装的官方 Torch-NPU 序列化入口，不挂载 NPU：

```bash
CHECK="$WORK/vad_mini8_comparison"
mkdir -p "$CHECK"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$RUNS:/runs:ro" -v "$CHECK:/checks:rw" \
  -w /source "$IMAGE" /opt/conda/envs/bevformer/bin/python -c '
import runpy, sys, torch, torch_npu
from torch_npu.utils import serialization
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = ["ascend/compare_resume.py"] + sys.argv[1:]
runpy.run_path("ascend/compare_resume.py", run_name="__main__")
' \
  --continuous /runs/vad_mini8/iter_8.pth --resumed /runs/vad_mini8_resumed/iter_8.pth \
  --continuous-run /runs/vad_mini8 --partial-run /runs/vad_mini8 \
  --resumed-run /runs/vad_mini8_resumed --output /checks/comparison.json
cat "$CHECK/comparison.json"
```

比较器按原容差检查 model／optimizer Tensor、loss、tokens、LR 和最终 RNG。精确载入通过后，后续计算仍可能产生差异；已完成的默认16芯片严格对照仍有部分结果超出原容差，需保留并查看[当前结果](../ascend/results/strict-replay-post8-20261014.json)，不能仅凭载入成功判断连续训练数值相同。

## 6. 推理、原评价和独立官方评分

推理使用**训练导出的配置**。它保存实际 mini／完整数据路径、词表和 head 设置；launcher 的推理入口不会替原 GPU 配置自动改写数据路径。

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$RUNS/vad_mini8/e2e_vadv2_50pct_rank0.py" \
  --mini-prepared "$MINI" --checkpoint "$RUNS/vad_mini8/iter_8.pth" \
  --vocab "$VOCAB" --precision fp32 --deterministic \
  --output "$RUNS/vad_mini8_eval"
```

结果包括 `predictions.pkl`、`metrics.csv` 和 `inference_validation.json`。规划入口检查配置 token 是否各出现一次、真实 `40×3` 轨迹是否有限；选择式 head 还检查 8192 词表选取是否一致。感知配置执行原检测／地图评价器，其指标同时查看 `console.log`。

对应的 NPU shell 评价命令：

```bash
export RUN_OUTPUT="$RUNS/vad_mini8_eval_shell" CPU_MEMORY=64g
bash projects/AlgEngine/scripts/e2e_dist_eval_npu.sh \
  "$RUNS/vad_mini8/e2e_vadv2_50pct_rank0.py" "$RUNS/vad_mini8/iter_8.pth" 1 -- \
  --deterministic
```

使用第 3 节 shell 示例导出的 `IMAGE`、`IMAGE_ID`、`VOCAB` 和 `MINI_PREPARED`。完整数据应改成 `DATA_ROOT` 与对应完整训练导出的配置，评价也选择独立输出目录。

独立小数据评分需要真实 scene logs 和地图，在 CPU 执行。下例中的宿主路径按相同名称挂入容器：

```bash
SCORE="$WORK/vad_mini8_official"
mkdir -p "$SCORE"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" \
  -v "$LOGS:/logs:ro" -v "$MAPS:/maps:ro" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:${PYTHONPATH:-}
    python ascend/official_score.py --prepared "$1" \
      --logs /logs --maps /maps --cache "$2/cache" \
      --predictions "$3/predictions.pkl" --output "$2/official_score.json"
  ' bash "$MINI" "$SCORE" "$RUNS/vad_mini8_eval"
```

评分使用锁定的 NAVSIM v1.1、原 40 点到 8 点转换 `[4::5]`、cache generation、simulator 和 scorer。模型内部选取的 cache 标签不能当作独立评分。CPU scorer 不使用 NPU；24g 仅为该小样本的资源示例。

已有覆盖对应 token 的官方 cache payload 时，可直接评分，无需重新生成 payload 或挂载 logs／maps。公开辅助器从输入 cache 中精确选取本次 prediction token，拒绝缺失或重复匹配，再以搬迁后的索引调用已有 `ascend.scoring.score_predictions`。多余 cache token 不参加评分，原 payload 字节和评分数学保持不动：

```bash
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 24g --cpus 8 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" /opt/conda/envs/bevformer/bin/python ascend/tools/score_cached.py \
  --predictions "$RUNS/vad_mini8_eval/predictions.pkl" \
  --metric-cache "$SCORE/cache" --output "$WORK/vad_mini8_from_cache"
```

`--output` 必须是尚不存在的新目录，每个 prediction token 必须有真实官方 payload。保留评分 JSON 和逐 token CSV，镜像导入成功或容器退出不能替代评分结果。

完整 NAVTEST 推理用完整训练导出的配置，传 `--data-root "$DATA" --memory "$EVAL_MEMORY"`，不传 mini。`EVAL_MEMORY` 根据评价容量设置。测试 filter 为原 `navtest.yaml` 时，原评价器导出 `metrics_navsim_submission.pkl`。使用完整官方 cache 和原 CPU 包装脚本重新评分：

```bash
FULL_EVAL="$RUNS/vad50_full_eval"
: "${EVAL_MEMORY:?Set the full NAVTEST evaluation memory limit}"
python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$RUNS/vad50_full/e2e_vadv2_50pct_rank0.py" \
  --data-root "$DATA" --memory "$EVAL_MEMORY" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --maps "$MAPS" --precision fp32 --output "$FULL_EVAL"

docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 64g --cpus 16 \
  -v "$SOURCE:/source:ro" -v "$FULL_EVAL:/predictions:ro" \
  -v "$CACHE:$CACHE:ro" -v "$WORK:$WORK:rw" -w /source/projects/AlgEngine \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export NAVSIM_DEVKIT_ROOT=/source/ascend/third_party/navsim
    bash scripts/e2e_navsim_official_rescore.sh \
      /predictions/metrics_navsim_submission.pkl "$1" "$2/vad50_full_score"
  ' bash "$CACHE" "$WORK"
```

64g 是完整评分命令的初始资源示例，尚非完整基准的实测预算。cache payload 必须可按 metadata 的路径读取；搬迁后重新建立索引或生成 cache。DiffusionDrive 等生成式轨迹模型必须经过官方重新评分，不能把推理 CSV 当成最终 PDMS。已有GPU／NPU对照限两个官方100基座的22锚点FP32推理，不是完整基准或GPU训练。

<a id="x86-cpu-scoring-of-gpu-exports"></a>

### 在x86 CPU上重算GPU导出的官方评分

公开`ascend/scoring.py`与[原CUDA对照实测](../ascend/results/gpu-npu-official100-eval22-20261016.json)使用的评分源码逐字节一致。输入为原`predictions.pkl`列表和匹配的官方NAVSIM metric-cache；原评分从`trajectory[4::5]`选择8点，[CPU比较工具](../tools/validation/README.md)另行量测全部40个导出位姿。训练PDM标签和预测中内嵌的指标不能替代官方cache及独立重评分。

实测Xeon Gold6430上，NumPy1.23.4／OpenBLAS0.3.20默认分派导致LQR计算异常和0分；在新CPU进程设置`OPENBLAS_CORETYPE=Haswell`后，未改包和评分公式即可恢复。该条件仅针对已测CPU／软件组合，**不要设为NPU全局环境，也不外推到其它BLAS构建**。相关依据见[NumPy上游问题](https://github.com/numpy/numpy/issues/24903)。

在本仓根目录，使用已有CUDA Python环境及原NAVSIM／NuPlan依赖，并检出NAVSIM子模块。以下路径指向本次自己完整导出的22-token预测、对应官方cache和一个新输出目录：

```bash
GPU_PREDICTIONS=/path/to/your_GPU_run/predictions.pkl
OFFICIAL_CACHE22=/path/to/official22/metric_cache
GPU_SCORE_OUT=/path/to/new_GPU_export_CPU_score
CUDA_VISIBLE_DEVICES='' OPENBLAS_CORETYPE=Haswell \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python - "$GPU_PREDICTIONS" "$OFFICIAL_CACHE22" "$GPU_SCORE_OUT" <<'PY'
import pickle
import sys
from pathlib import Path
from ascend.scoring import score_predictions

predictions, cache, output = map(Path, sys.argv[1:])
if output.exists() or output.is_symlink():
    raise SystemExit("Use a fresh CPU-score output directory")
with predictions.open("rb") as stream:
    records = pickle.load(stream)
if len(records) != 22 or len({row["token"] for row in records}) != 22:
    raise SystemExit("This example requires the complete 22-token export")
payloads = list(cache.rglob("metric_cache.pkl"))
if len(payloads) != 22 or {path.parent.name for path in payloads} != {row["token"] for row in records}:
    raise SystemExit("Provide exactly the 22 matching official cache payloads")
print(score_predictions(records, cache, output))
PY
```

保留`official_score.json`及`official_scores.csv`，异常尝试也不覆盖。两模型实测中，CPU环境纠正后，全部40×3轨迹值和每个token的七项官方指标均与NPU参照相同。两侧源码／框架栈不同，相同JPEG不证明预处理Tensor逐位一致；该结果限有界推理输出，未设质量等价阈值，也不是纯硬件归因。

## 7. 在 NAVTRAIN 上评价并筛选稀有场景

稀有场景筛选需要训练 split 的真实逐场景结果。原 GPU 分块脚本不能直接当作 NPU launcher。下面从完整训练导出配置另写一个推理配置，把 test 阶段指向对应 NAVTRAIN，原训练配置和断点保持不动。

```bash
NAVTRAIN_CFG="$WORK/navtrain_inference.py"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 8g --cpus 2 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python - "$1" "$2" <<'"'"'PY'"'"'
import sys
from mmcv import Config
cfg = Config.fromfile(sys.argv[1], import_custom_modules=False)
for phase in ("val", "test"):
    ds = cfg.data[phase]
    ds.ann_file = "/data/full/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
    ds.nav_filter_path = "/workspace/WorldEngine/projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain_50pct.yaml"
    ds.train_pdm_path = "/data/full/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain"
    ds.test_pdm_path = ds.train_pdm_path
    ds.pipeline[0].img_root = "/data/full/raw/openscene-v1.1/sensor_blobs/trainval"
cfg.dump(sys.argv[2])
PY
  ' bash "$RUNS/vad50_full/e2e_vadv2_50pct_rank0.py" "$NAVTRAIN_CFG"

python3 ascend/run_container.py --devices 0 --task inference \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config "$NAVTRAIN_CFG" --data-root "$DATA" --memory "$EVAL_MEMORY" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --maps "$MAPS" --output "$RUNS/vad50_navtrain_eval"
```

本节命令及下节完整数据微调示例使用 50% split。筛选整个训练集合时，要使用对应 100% 训练导出配置和 planner 断点，把推理 filter 和采样脚本的 `--base-split` 都改为 `navtrain.yaml`。合并文件改为 `navtrain_collision.yaml`、`navtrain_off_road.yaml`、`navtrain_ep_1pct.yaml`；前缀来自 base split 文件名，不来自模型名。此结果集使用独立的新输出。

仓库发布的是50% rare-log微调配置，没有独立的 `100pct_rlft_rare_log.py`。派生全量数据微调时，须在已有rare-log配置上显式传 `--train-filter "$SOURCE/projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain.yaml"`，同时使用本次完整split生成的 `--rare-split`。数据覆盖设置和初始化权重分别记录，并换新输出；它不是已发布或已验收的100%微调配方。复现100%基座实验时使用对应planner断点；同结构的50%权重也可初始化全量数据微调，但要明确记录其来源，不能称为100%训练的基座。只改推理filter不会改变微调数据集。

大集合可按同样方式拆成 token YAML 分块，在新输出逐块评价，再合并无重复 token 的结果；保持原 token。原 annotation／PDM 仍需完整载入，减少 filter 不会消除这部分内存。

从 CSV 去掉 `average` 汇总行，再调用原稀有场景脚本；把碰撞、驶出可行驶区域、低进度三类 token 合并成一个显式 `--rare-split` 输入：

```bash
RARE_OUT="$WORK/vad50_rare"
mkdir -p "$RARE_OUT"
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --memory 8g --cpus 2 \
  -v "$SOURCE:/source:ro" -v "$WORK:$WORK:rw" -w /source \
  "$IMAGE" bash -lc '
    set -euo pipefail
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    python - "$1" "$2" <<'"'"'PY'"'"'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[df.token != "average"]
assert len(df) and not df.token.duplicated().any()
df.to_csv(sys.argv[2] + "/navtrain.csv", index=False)
PY
    python projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py \
      --pdm-result "$2/navtrain.csv" \
      --base-split projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain_50pct.yaml \
      --output-dir "$2" --ep-percentile 1
    python - "$2" <<'"'"'PY'"'"'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
files = [root / ("navtrain_50pct_" + name + ".yaml")
         for name in ("collision", "off_road", "ep_1pct")]
splits = [yaml.safe_load(path.read_text()) for path in files]
merged = dict(splits[0])
merged["tokens"] = sorted(set().union(*(set(row["tokens"]) for row in splits)))
assert merged["tokens"], "No rare tokens selected; inspect the evaluation results"
(root / "navtrain_rare_union.yaml").write_text(yaml.safe_dump(merged))
PY
  ' bash "$RUNS/vad50_navtrain_eval/metrics.csv" "$RARE_OUT"
```

原规则提取碰撞、off-road 和安全样本中进度最低的 1%。输入过小或没有正进度的安全样本时，原百分位计算可能失败，应提供有代表性的真实评价集合并检查结果，不修改规则强行产生 token。

## 8. 真实日志微调和合成数据配置

[10月16日 BF16 v7 真实日志微调记录](../ascend/results/real-log-ft37-bf16-v7-short8-20261016.json)与 NAVTRAIN139 分开：两个原37/22配置均完成8次更新、自身第4次更新恢复及逐位一致的续训对照。该记录包含完整 VAD 评价，以及在日志进度14/22时中断、尚未评分的 Hydra 推理。[10月17日 Hydra 补充验证](../ascend/results/hydra-ft37-bf16-eval22-20261017.json)使用同一自身第8次更新检查点，重新完成全部22锚点推理及原 NAVSIM CPU 评分，实测 PDMS 为0.9338557270555728；原中断记录继续保留。这些功能结果不增加 producer 合成数据覆盖，也不代表收敛。

VAD 的 rare **真实日志**微调使用对应 planner 权重初始化，并传入上节 YAML；这一配置无需 SimEngine rollout：

```bash
python3 ascend/run_container.py --devices "$DEVICES" \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
  --data-root "$DATA" --rare-split "$RARE_OUT/navtrain_rare_union.yaml" \
  --checkpoint "$RUNS/vad50_full/epoch_8.pth" --vocab "$VOCAB" \
  --runtime --batch-size 1 --workers 0 --precision fp32 \
  --maps "$MAPS" --official-metric-cache "$CACHE" --validate \
  --memory "$TRAIN_MEMORY" --cpus 64 --pids-limit 8192 \
  --output "$RUNS/vad50_rare_ft"
```

也可用已发布的 `VAD_BASE` 初始化。原 VAD 配置使用 LoRA 和奖励微调，`rl_finetuning=False`，policy gradient 未启用。Hydra 配置 `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct_rlft_rare_log.py` 从 base 启用 policy gradient。使用 `HYDRA_BASE` 或对应训练 Hydra 权重、Hydra 自己筛选的 rare 数据和新输出。新的微调任务重新建立 optimizer／schedule，使用 `--checkpoint` 初始化；`--resume` 用于恢复原任务的完整状态。

先做小数据功能检查时，分别执行下面两条命令。`ascend/train.py` 的 profile 映射选择各自原微调配置。切换到 Hydra 时不能保留 VAD 的 `--config`，因为显式配置优先于 profile。八步命令不带 `--validate`，逐 epoch 评价不能与 `--steps` 同时使用。

```bash
python3 ascend/run_container.py --devices 0 --profile vadv2_ft \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --rare-split "$MINI_RARE" \
  --checkpoint "$VAD_BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/vad_rare_mini8"

python3 ascend/run_container.py --devices 0 --profile hydramdp_rl \
  --image "$IMAGE" --expected-image-id "$IMAGE_ID" \
  --mini-prepared "$MINI" --rare-split "$MINI_RARE" \
  --checkpoint "$HYDRA_BASE" --vocab "$VOCAB" \
  --runtime --steps 8 --batch-size 1 --workers 0 --precision fp32 \
  --deterministic --upstream-schedule --checkpoint-interval 4 --check-interval 1 \
  --memory 64g --cpus 16 --output "$RUNS/hydra_rare_mini8"
```

mini 必须包含真实 rare 锚点，以及这些日志内可检索的非 rare normal 候选；每个候选都要保留原四帧队列和相机图片。另一个训练日志中的 normal 锚点不满足原微调 dataset 的同日志检索规则。通过原 dataset 检查实际返回的 token 和两条分支。这两条八步命令只做功能检查，不作为完整微调或收敛结果。

原仓共 44 个配置，[配置指南](config_guide.md)说明其关系。实际配置通过 `--config` 传入；两个感知配置原 batch=1，42 个规划配置原 batch=2。NPU 包装入口保留模型／head／dataset 分支，batch 和 schedule 的改动通过参数明确设置。

八个配置需要真实 rollout／replay／BWM／SimScale 输入，通过 `--synthetic-root` 提供。不能把普通真实 mini 冒充合成数据。[SimScale 准备工具](../ascend/data_tools/README_zh.md)用约 47.50 GB 锁定输入，以及另计的地图、镜像和输出，准备两项已测配置；其余六项需要对应生产流程的真实导出。公开 16384 候选奖励不等于训练的 8192 标签，按工具流程准备，不裁剪或改名替代。闭环仿真和新 rollout 渲染按[原 SimEngine 指南](simengine_usage.md)在适用的 GPU 环境执行。

## 9. 输出、日志和常见问题

每次运行只读挂载源码和数据，输出目录可写。完整 runtime 输出一般包含：

```text
run/
├── launch.json                  # exact image, device selection and command
├── console.log
├── train_exit_code, exit_code
├── <config_stem>_rank0.py       # actual configuration for later inference
├── runtime_rank<N>.json         # per-rank final checks and precision/state details
├── runtime_rank<N>.jsonl        # update/loss/tokens/LR/resource stream
├── iter_<N>.pth or epoch_<N>.pth
├── latest.pth
└── validation/epoch_<N>/        # with --validate on epoch training
```

默认前台等待执行完成。设备使用时段和监控已安排时可加 `--detach`；此时返回容器 ID 仅表示启动，不能据此判断训练完成。

```bash
cat "$RUNS/vad_mini8/exit_code"
tail -n 60 "$RUNS/vad_mini8/console.log"
cat "$RUNS/vad_mini8/runtime_rank0.json"
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  worldengine-npu-vad_mini8
```

| 现象 | 检查与处理 |
| --- | --- |
| image ID 不一致 | 检查标签实际指向的镜像。新训练记录自己的 ID，续训必须使用原镜像。 |
| 输出目录已存在 | 换新输出，保留之前的日志和断点。 |
| 缺失标注、图片／历史队列或 8192 PDM | 修正真实数据目录及所选图片覆盖，再执行准备／preflight。 |
| 完整 PKL 要求传 `--memory` | 按每 rank 数据常驻、训练、评价及 worker 预算设置上限。 |
| `can't start new thread`／EAGAIN | 检查容器 PID／线程数。默认上限为 `max(2048, 512×rank数)`，16 rank 示例设置 8192。 |
| `EL0020`、设备占用、`507899` | 暂停在该芯片启动新作业，核对具体设备／容器归属；不能仅凭空进程列表判断驱动可用。 |
| resume contract 不匹配 | 核对源码、镜像、数据／filter、精度环境、world size 和 schedule，恢复一致输入。 |
| NaN／Inf | 保留运行目录，排查原始 loss、gradient 和训练状态。 |
| 已完成对照，结果超出原容差 | 保留原数值报告和断点，按原门槛判定；载入成功不代表后续数值重放相同。 |
| 官方评分找不到 cache payload | 核对 token 覆盖及 `metadata/*.csv` 路径，搬迁后重建索引或 cache。 |

小数据训练、恢复和评分的实测条件、版本及结果见[验证报告](../ascend/VALIDATION_zh.md)。公开配方重建、两类微调的 16 芯片训练与续训、独立日志推理及官方评分已有实测。公开同步入口四组基座训练重复均完成，其中两组超容差，数值问题仍待解决；完整 NAVTRAIN 收敛、完整独立 NAVTEST、其余六个真实合成配置和目标环境持续训练尚未验收。

公开提交 `ec153c1` 的实际复现覆盖 37 个训练候选、22 个独立评价锚点；新镜像两类 LoRA 微调的最终比较全部逐位相同。[复现证据](../ascend/results/actual-public-reproduction-16chip-20261014.json)记录配置、镜像、输入和边界。使用原始相机目录层级，不能将平铺 JPEG 目录直接作为原 NAVTEST 标注的相机根目录。

## 10. FP32 规划推理优化选项

推理默认沿用原实现。`--inference-optimization` 只在本次推理进程中启用优化，不改训练、权重、模型参数、测试数据处理和评分。使用时须指定 `--precision fp32 --msda-backend cuda-compatible`，保持 Matmul/Conv HF32 关闭。

| 选项 | 执行方式及适用范围 |
| --- | --- |
| `none` | 原实现，查询分块为 2048。 |
| `point-parallel` | 合并计算相互独立的 MSDA 采样点，保留原补偿坐标算法和逐层、逐点累加顺序。`--msda-query-chunk` 可选 2048、4096、8192，较大分块会增加临时内存。 |
| `point-parallel-last-frame` | 在上一项基础上，去掉 NAVFormer 规划推理中输出被丢弃的前三次外层计算。仅适用于 batch 1、三帧历史加当前帧。保留的最后一次原调用仍处理全部四帧，BEV 帧计算次数由 11 次减为 4 次，并补齐被省去调用中的六次 NumPy 随机数消耗。 |

最后一帧优化启用前会核对原模型方法和 GridMask 源码。训练模式、启用梯度、感知任务、其他历史帧配置或源码变化时，回到原帧处理路径并记录原因。带梯度的 MSDA 输入使用原前向和反向实现。推理结束或异常退出都会恢复原函数。查看实际调用计数，确认优化确实生效。

两次运行使用同一份已导出的推理配置、官方权重、数据、镜像、精度、确定性设置和逻辑芯片。配置中的路径须与第 6 节的挂载一致，输出目录分别新建。下面的 `DEVICES=0,...,15` 表示 16 个逻辑芯片，即 8 张双芯物理卡。

```bash
COMMON=(--task inference --devices "$DEVICES"
  --image "$IMAGE" --expected-image-id "$IMAGE_ID"
  --config "$INFERENCE_CONFIG" --data-root "$DATA"
  --checkpoint "$CHECKPOINT" --vocab "$VOCAB" --maps "$MAPS"
  --memory "$EVAL_MEMORY" --cpus 64 --pids-limit 8192
  --precision fp32 --msda-backend cuda-compatible --deterministic)
python3 ascend/run_container.py "${COMMON[@]}" \
  --output "$RUNS/fp32_original"
python3 ascend/run_container.py "${COMMON[@]}" \
  --inference-optimization point-parallel-last-frame --msda-query-chunk 8192 \
  --output "$RUNS/fp32_optimized"
```

在同一套已安装环境内比较两个结果目录：

```bash
python ascend/tools/compare_inference.py \
  --baseline /results/fp32_original --candidate /results/fp32_optimized \
  --output /results/fp32_comparison.json
```

比较工具核对源码、配置、权重、运行环境、精度、设备配置和 token 顺序，并逐字段检查输出类型及原始数值是否一致。退出码为 0 还要求所选优化确实执行。`inference_validation.json` 保存选项、分块、实际调用次数及回退原因，计数来自 rank 0。官方 NAVSIM 独立评分仍按第 6 节执行。

性能测试需在相同设备和数据上重复交替运行，分别报告完整作业耗时和预热后的模型耗时。原 `elapsed_seconds` 从执行原测试脚本前开始，包含模型与权重加载、推理和结果收集，不含更早的 Python/Torch 导入及配置、精度设置；容器总耗时还包含这些前置阶段。两者都不能当作纯算子时延。带审计或 profiler 的运行不能与普通运行直接比较。小集验证只证明该范围内的功能及数值一致性，不代表完整 NAVTEST 质量或训练收敛。需要回退时，去掉优化参数，或指定 `--inference-optimization none` 并保留分块 2048。

[实测记录](validation/algengine_fp32_optimization_20261029.json)：两模型通过公开 `ascend/inference.py` 完成原实现与优化实现的 16 rank 对照，使用 22 个官方真实 NAVTEST 样本。全部输出字段的类型及原始数值一致，优化调用计数非零。记录另列单算子和 139 样本辅助审计，并保留完整集、GPU 对照及训练验收的边界。
