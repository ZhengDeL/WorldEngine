# AlgEngine 在 Ascend 910C 上的安装与使用

本文是[原 AlgEngine 使用指南](algengine_usage.md)的 NPU 操作版，按数据准备、训练、评价、稀有场景筛选和微调的顺序说明。命令在 Linux NPU 服务器执行，默认当前目录是仓库根目录。原 GPU 脚本保留；NPU 使用 `ascend/run_container.py` 或新增的 `_npu.sh` 前端。

WorldEngine 有两个模块：**AlgEngine** 负责模型训练与评价，**SimEngine** 负责渲染、仿真和 rollout 生成。这里操作 AlgEngine；已有的真实 SimEngine 导出可以作为训练输入，本指南不部署其渲染器。

## 常用入口

| 操作 | 入口 |
| --- | --- |
| 构建、导入运行环境 | [容器安装说明](../ascend/docker/README_zh.md) |
| 选择性下载官方数据和权重 | `ascend/tools/download_algengine_assets.py` |
| 准备真实小数据 | `ascend/tools/prepare_real_mini.py`，再执行 `ascend/split_mini.py` |
| 按原配置训练 | `ascend/run_container.py --config ...` |
| 原仓风格的训练、评价前端 | `projects/AlgEngine/scripts/e2e_dist_train_npu.sh`、`e2e_dist_eval_npu.sh` |
| 完整断点续训 | 原训练参数，加 `--resume ...` 和新输出目录 |
| 推理 | `ascend/run_container.py --task inference --config <训练导出的配置>` |
| 小数据独立评分 | CPU 执行 `ascend/official_score.py` |
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

先完成[容器安装](../ascend/docker/README_zh.md)。宿主需要可用的 Ascend 驱动、Docker、Python 3 和 NPU 字符设备；镜像提供 PyTorch/Torch-NPU、OpenMMLab、DrivingSDK 和地图处理依赖。不要在此镜像上覆盖安装原仓的 CUDA wheel。

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

Hugging Face及hf-mirror线路请求固定发布revision。魔搭线路请求其 `master`，按同一内容锁的文件大小和SHA256校验，不宣称向魔搭请求了Hugging Face的commit。下载记录分别保存请求revision和内容参考revision。国内线路可选 `--source modelscope`，官方线路用 `--source huggingface`；有效旧文件会复用。工具不下载相机包、地图、原始日志或SimEngine资产，换线路时保留哈希校验。

NAVTEST 标注和 PDM 合计约 **3.64 GB**，相机、词表、权重和地图另计。小样本最终只需实际引用的 JPEG，但这些真实图片需要已经从对应 OpenScene 数据或提供的小数据包中取得。仓库不承诺存在一个恰好包含所选 JPEG 的独立公开下载包。

### 相机、原始日志和地图从哪里获取

[原 NAVSIM 测试下载脚本](../ascend/third_party/navsim/download/download_test.sh)给出 OpenScene metadata 和 32 个相机分片的地址。按所需日志选择分片；下面 `SHARD=0` 仅示范下载，**不保证**包含制作 mini 所需的三个有效日志。小检查不要直接执行原整套脚本，它还会下载全部相机和 LiDAR。这里使用[输入锁](../ascend/data_tools/INPUT_LOCKS.json)记录的 OpenScene revision；本次文档更新没有实际执行这两个新的 test 包下载。

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

## 3. 准备真实小数据，完成单芯片八次更新

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

使用原官方 split 和 cache generator。64g 是初始资源示例，本次未实测完整 cache 生成的预算。检查实际 payload 数、日志及 preflight token 覆盖；上游 cache 脚本可能记录失败 entry 后仍以 0 退出，不能只看进程退出码。

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

不传 `--steps`、`--epochs`，保留原配置八个 epoch、optimizer 和 LR horizon。迁移入口的 `--validate` 每个 epoch 执行评价。100% 训练将配置改为 `projects/AlgEngine/configs/worldengine/e2e_vadv2_100pct.py`，初始化使用 `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth`，并换新输出目录。原 50% filter 为 50,000 个锚点，完整 split 为 103,288；完整图像训练及收敛尚未完成。

单独的[同步候选步骤](../ascend/execution/README_zh.md)只允许两个已测规划配置和八次更新，不用于完整 epoch 训练。该候选每模型各通过一次严格连续／恢复对照，重复可靠性仍需检查。

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

比较器按原容差检查 model／optimizer Tensor、loss、tokens、LR 和最终 RNG。精确载入通过后，后续计算仍可能产生差异；默认 16 芯片严格对照目前有失败记录，需保留并查看[当前结果](../ascend/results/strict-replay-post8-20261014.json)，不能仅凭载入成功判断连续训练数值相同。

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

对应的原仓风格 shell 评价命令：

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

64g 是完整评分命令的初始资源示例，尚非完整基准的实测预算。cache payload 必须可按 metadata 的路径读取；搬迁后重新建立索引或生成 cache。DiffusionDrive 等生成式轨迹模型必须经过官方重新评分，不能把推理 CSV 当成最终 PDMS。GPU/NPU 相同模型轨迹对照尚未执行。

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

也可用已发布的 `VAD_BASE` 初始化。这份原 VAD 配置启用 LoRA 和奖励微调，但 `rl_finetuning=False`，不能写成启用了 policy gradient。Hydra 配置 `projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct_rlft_rare_log.py` 从 base 启用 policy gradient。使用 `HYDRA_BASE` 或对应训练 Hydra 权重、Hydra 自己筛选的 rare 数据和新输出。不同训练的微调需重新建立 optimizer／schedule，使用 `--checkpoint` 初始化，不使用 `--resume`。

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
| NaN／Inf 或严格比较失败 | 保留运行目录，检查原始 loss、gradient 和状态记录，不放宽门槛强行通过。 |
| 官方评分找不到 cache payload | 核对 token 覆盖及 `metadata/*.csv` 路径，搬迁后重建索引或 cache。 |

已测的小数据训练、恢复和评分可以按记录输入复跑。完整 NAVTRAIN 收敛、完整独立 NAVTEST、六项真实合成导出和最终目标配置持续运行仍未完成。各配置的当前实测范围与结果见 [Ascend 操作及结果说明](../ascend/README_zh.md)。
