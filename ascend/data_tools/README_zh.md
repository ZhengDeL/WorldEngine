# 真实 SimScale 子集准备与复现

三个 SHA 锁定的 CLI 已于 2026 年 10 月 14 日 04:22:57（北京时间）完成服务器 CPU 复现，转换 26 个 clip、解码 832 个历史 JPEG、重新计算 8192 候选奖励、导出 104 个 payload。四阶段均退出 0、无 OOM，原始输入 SHA 未变，来源与收据 SHA 见 `CPU_REPRODUCTION_VERIFIED.json`。

工具及其 SHA 清单随仓库发布。新导出数据已完成原 dataset 的 CPU 检查，以及 post8 单 rank 和双 rank 的真实训练/恢复对照。

| 阶段 | 完成时间（北京时间） | 实测结果 |
| --- | --- | --- |
| 原 dataset CPU getitem | 2026 年 10 月 14 日 | 两个原配置、synthetic/normal 四次请求；每次 prepare 一次，None=0、retry=0，token、四帧图像与六项 8192 评分一致 |
| 单 rank 连续/恢复训练 | 2026 年 10 月 14 日 04:54:02 | 两配置各连续 8 更新，再从自身 iter4 恢复到 8；Finetune 模型/完整 Tensor 数 989/1291，Cotrain 为 909/1748，均逐位相同 |
| 双 rank 连续/恢复训练 | 2026 年 10 月 14 日 05:00:51 | 两配置各连续 8 更新、自身 iter4 恢复到 8；全部 8 份 rank 功能收据通过；Finetune 为 989/1293，Cotrain 为 909/1750，均逐位相同 |

完整 Tensor 计数包含 rank RNG。两轮训练的参数最大差和 loss 差均为 0，tokens、LR 和最终 RNG 相同。对应收据为 `GETITEM_CPU_VERIFIED.json`、`NPU_SINGLE_RANK_VERIFIED.json` 和 `NPU_TWO_RANK_VERIFIED.json`。CPU 检查使用 SDK 导入期设备名元数据桥：两次查询取得此前实测设备名后，原函数恢复且 NPU 未初始化。这不是无桥 CPU 导入验收。

范围是一个真实场景、每配置一个明确 normal 锚点及原 WE 派生奖励协议。其他六个 rollout/replay/BWM 配置所需真实导出尚未齐备；完整 NAVTRAIN 收敛、GPU 对照及 16rank 严格数值验收未通过。前两个工具的转换、评分和输入适配函数 AST 与已实测参考 helper 相同；导出工具还通过 18 项本机队列、tar 安全和 dry-run 检查。验证范围及版本记录如下。

这一流程使用一个真实 SimScale 场景、真实 NAVTRAIN 正常样本和原 WorldEngine 四秒 CPU 评分。它不复现 SimScale v2 的 16384 奖励，不部署 SimEngine 渲染器，也不验收完整 NAVTRAIN 收敛。原始图像、标定、GT、词表顺序和正常样本 PDM 不修改。

## 输入和容量

所有数据下载、校验和提取在服务器执行。具体原路径、固定 revision、bytes、SHA256、官方与 hf-mirror 地址见 `INPUT_LOCKS.json`。已准备的文件直接复用，不重复下载。

| 输入 | 完整下载字节数 | 用途 |
|---|---:|---|
| NAVTRAIN annotation | 16,125,493,728 | 从原 246165 帧选择两个正常锚点的原四帧 queue |
| NAVTRAIN 8192 PDM | 23,736,976,138 | 保留两个正常锚点的原 8192 评分 entry |
| SimScale round0 metadata | 3,320,437,526 | 原 hist65 索引对应 26 个 clip 的原转换，最后只训练一个真实场景 |
| SimScale hist65 相机 | 241,575,885 | 26 个 clip 的 832 个原历史 JPEG；训练 subset 使用选中场景 32 个 |
| OpenScene mini12 相机 | 2,781,004,086 | 两个实际正常锚点各 32 个原 JPEG |
| OpenScene raw mini metadata | 534,344,190 | 读取一个完整原日志并绑定连续 12 帧记录交通 |
| 8192 词表与 hist 索引 | 3,932,288 + 1,824,152 | 原候选顺序、hist65 原成员选择 |
| 正式 finetune/cotrain 初始化 | 434,885,463 + 319,958,048 | 后续原两配置的 NPU 验证，不参与 CPU 奖励计算 |

以上合计 **47,500,431,504 B，约 47.50 GB**，maps、源码、镜像和新输出另计。annotation 与 PDM 合计 39,862,469,866 B；虽然最终只取八条原 normal 记录及两个 PDM entry，发布者提供的是完整 PKL，仍需要这两个完整文件。此流程不要求 449 GB 完整训练图像、90 GB 官方 mini 全相机、5 TB 资产、LiDAR 或 future 相机。

先前检查的 19,305,089,518 B 发布奖励与原 8192 训练协议不兼容：16384 候选、字段不同，目标 8192 的完整轨迹行与 16384 没有精确对应。它仅用于历史兼容性调查，**不是本流程训练输入或必下载文件**。

使用真实 `nuplan-maps-v1.0.json` 与对应 `.gpkg` 地图；已有官方 maps 可复用。地图独立下载总量尚未统计，不包含在 47.50 GB 合计中。CPU 运行使用已锁镜像 `local/worldengine:26.1.0-a3-clean-20261013`，ID `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`，保留镜像原 PYTHONPATH 和依赖目录。

## 初始化权重下载与校验

以下两份权重固定到 WorldEngine 数据仓 revision `d0e88d858533ca407a3387a26577c083aa7b3617`。下载只在服务器执行。已有完整文件时直接按 `INPUT_LOCKS.json` 校验并复用，不重复下载；以下命令仅供新下载目录使用，运行前启用 `set -euo pipefail`，任一大小或 SHA 失败立即停止，保留 `.part`，不作为训练初始化。

| 用途 | 文件 | 字节数 | SHA256 |
| --- | --- | ---: | --- |
| Finetune 显式 override | `e2e_vadv2_100pct_ep8.pth` | 434,885,463 | `fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a` |
| Cotrain 原初始化 | `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth` | 319,958,048 | `9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a` |

官方 HF 和国内 hf-mirror 使用相同固定 revision、发布路径和内容 SHA，完整 URL 见 `INPUT_LOCKS.json`。Finetune 的 100pct 权重是明确的 override；原配置指定的 `e2e_vadv2_ep8.pth` 尚未找到，不得把下载文件改名冒充它。

```bash
# Server host only. Use a fresh input directory for downloads.
set -euo pipefail
INPUTS='<host locked public input directory>'
mkdir -p "$INPUTS"
PUBLISHER_REV=d0e88d858533ca407a3387a26577c083aa7b3617
HF_BASE="https://huggingface.co/datasets/OpenDriveLab/WorldEngine/resolve/$PUBLISHER_REV"
# If the official route is unavailable, select this exact-revision mirror instead:
# HF_BASE="https://hf-mirror.com/datasets/OpenDriveLab/WorldEngine/resolve/$PUBLISHER_REV"

test ! -e "$INPUTS/e2e_vadv2_100pct_ep8.pth"
curl --fail --location --retry 3 --continue-at - \
  --output "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" \
  "$HF_BASE/data/alg_engine/ckpts/vadv2/e2e_vadv2_100pct_ep8.pth"
test "$(stat -c %s "$INPUTS/e2e_vadv2_100pct_ep8.pth.part")" = 434885463
printf '%s  %s\n' fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a \
  "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" | sha256sum --check -
mv --no-clobber -- "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" "$INPUTS/e2e_vadv2_100pct_ep8.pth"

test ! -e "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
curl --fail --location --retry 3 --continue-at - \
  --output "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" \
  "$HF_BASE/data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
test "$(stat -c %s "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part")" = 319958048
printf '%s  %s\n' 9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a \
  "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" | sha256sum --check -
mv --no-clobber -- "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" \
  "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
```

## 源码和工具身份

工具位于仓内 `ascend/data_tools/`，使用独立 SHA 清单。现有训练指纹明确只包括 `projects/AlgEngine/**/*.py` 和 `ascend/*.py`；嵌套的纯数据准备工具独立使用 `TOOLS_MANIFEST.json` 闭环。记录这一范围的理由是区分运行时数值代码和数据生产代码，不是忽略数据变更。训练合同仍锁定实际 annotation、filter、PDM 和 vocab 内容 SHA。

数据准备工具记录运行时身份，不配置训练 CANN 开关或改变恢复合同。调用者必须提供真实 `--expected-runtime-sha256`。工具另记录原 converter 及四份 SimEngine 评分/转换源码 SHA、抽取方法 AST、maps SHA、发布输入与生成输出 SHA。结果适用于记录中的工具及运行时版本；AST 一致性检查不能替代实际 CPU 执行。

三个 CLI 默认 dry-run；加 `--execute` 才执行。每阶段使用新输出，不覆盖已有证据；history 阶段只向已通过 conversion 的目录追加此前不存在的图像及 history receipt。整个 tar 先审查绝对路径、`..`、链接、特殊成员和重复文件名，然后只取实际引用，绝不 extractall。

## 分步操作

下面的变量由运行者填写；数据文件必须先匹配 `INPUT_LOCKS.json`。在 CPU 容器内挂载源码只读、输入只读、tools 只读和独立 output 可写，不挂载 NPU 设备。

```bash
WE_SOURCE=/workspace/WorldEngine
TOOLS=/tools
RUNTIME_FP='<所选源码实际指纹>'
INPUTS=/inputs
MAPS=/maps
OUT=/output/new_reproduction
DEADLINE='<本次授权的未来UTC时间>'
```

第 1 步使用 CPU8GiB/4CPU，沿原 converter 的默认 2Hz、1.5 秒 past、4 秒 future、`filter_boxes=False`处理 26 个 clip。调用原 `parse_log`，不改原转换数学；核对 frame identity、标定、三步过去及八步未来的 GT/mask。输出路径不能含 `meta_datas` 字符串，因为原 converter 按原路径替换该目录名。

```bash
python "$TOOLS/simscale_convert.py" --mode convert \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_meta_datas.tar.gz" \
  --hist-index "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist.json" \
  --output "$OUT/converted" --deadline "$DEADLINE"
```

先检查计划，再对同命令追加 `--execute`。通过后，仍使用 CPU8GiB/4CPU 执行 history 阶段，逐个 JPEG 保持原 byte 并解码；不下载 future 相机。

```bash
python "$TOOLS/simscale_convert.py" --mode history \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist_65.tar.gz" \
  --output "$OUT/converted" --deadline "$DEADLINE" --execute
```

第 2 步使用 CPU64GiB/16CPU 和原八个评分 worker，生成选中场景 8192 条原序候选的奖励。

```bash
python "$TOOLS/worldengine_8192_logged_reward.py" \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --metadata "$INPUTS/openscene_metadata_mini.tgz" \
  --conversion-root "$OUT/converted" --vocab "$INPUTS/test_8192_kmeans.npy" \
  --maps "$MAPS" --scene-id fff002cb15c15a7a-000 \
  --original-log 2021.06.09.17.23.18_veh-38_00773_01140 \
  --folder-name synthetic_reaction_pdm_v1.0-0 \
  --expected-anchor-token ab75b7addff3512d-000 \
  --output "$OUT/reward" --deadline "$DEADLINE"
```

检查计划后加 `--execute`。输出为独立 `worldengine_8192_logged_traffic_derived_rewards.pkl`，绝不覆盖或截取原 16384 奖励。原日志 734 帧中，十二个 literal synthetic token 对应连续原 index213–224；scene 名称与 frame3 token 不同。合成当前 ego 与原记录交通不假定物理状态相同。

该派生协议保留原 WorldEngine 四秒评分、原 8192 词表、原八个 worker 和每候选相对 PDM reference 的进度归一化。reference planner 只接收实际四帧 history，评分接收当前加八帧记录交通；路线为原/合成一致的发布 roadblock IDs；原 NAVSIM 转换提供 global detections 并包含记录 ego。零加速度、零转角沿原 WE ego converter。它不是 SimScale v2、完整引擎生命周期或原生 nuPlanDB 加载等价协议。

第 3 步使用已实现的 `export_simscale_minimum.py`，CPU 容器上限 128GiB/4CPU，宿主至少保留 192GiB 共享余量。它直接读取已锁定的官方 annotation、PDM 和 mini12 相机包，核对当前 conversion、history、reward 收据及原数学源码锁后导出；直接核对发布输入和可复现的准备阶段记录。默认 dry-run，检查计划后加 `--execute`。该 CLI 已完成服务器真实 CPU 重跑，保留 64 个 normal 和 32 个 synthetic 原 JPEG、原 12 帧 synthetic 记录与原正常样本 PDM。导出进程峰值 RSS 为 28,364,746,752 B；完整 annotation、PDM 和 mini12 相机包前后 SHA 一致。

```bash
python "$TOOLS/export_simscale_minimum.py" \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --annotation "$INPUTS/nuplan_openscene_navtrain.pkl" \
  --pdm "$INPUTS/pdm_8192_gt_cache_navtrain.pkl" \
  --camera-archive "$INPUTS/openscene_sensor_mini_camera_12.tgz" \
  --conversion-root "$OUT/converted" --reward-root "$OUT/reward" \
  --scene-id fff002cb15c15a7a-000 --folder-name synthetic_reaction_pdm_v1.0-0 \
  --original-log 2021.06.09.17.23.18_veh-38_00773_01140 \
  --expected-anchor-token ab75b7addff3512d-000 \
  --finetune-normal-token 4b96572a4b9d5ea7 \
  --cotrain-normal-token fff002cb15c15a7a \
  --output "$OUT/subsets" --deadline "$DEADLINE"
```

导出必须从完整原 annotation 按原前驱 index 取四帧，而不是精确 500000 微秒查找。保留原 record、GT、calibration、image path 和 PDM entry；两个 token 都属于原 NAVTRAIN。两个配置各输出 35 个 payload，共享 synthetic 输出 34 个，共 104 个 payload。显式 one-anchor normal 白名单有意缩小检索池，不等价于全数据随机检索。原 retrieval、folder_name、train PDM 选择、head/loss 分支留给原 dataset 运行，数据身份由原输入和当前导出记录核对。

第 4 步仅在 subset CLI 真实重跑及来源核对后，使用原 `ascend/run_container.py` 分别运行两个 SimScale 原配置。每配置自己的 data root 和 `filters/navtrain_actual_normal_anchor.yaml`，共享真正 synthetic root，分别初始化 8 更新并从自己的 iter4 完整恢复到 8。Tensor 门槛仍 1e-6/1e-5，loss 差小于 1e-4；原 CPU loader 和比较器不改。

Finetune 明确使用发布的 `e2e_vadv2_100pct_ep8.pth` 作为 checkpoint override；原配置的 `e2e_vadv2_ep8.pth` basename 未找到，不能静默改名冒充。Cotrain 使用原 `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth`。运行时 flags/source 指纹和设备按选定的实测版本记录，数据工具不配置训练开关。

## 数据与训练验证记录

服务器 CPU 实测已完成 conversion、832 个原 history 图像、CPU8192 评分和 subset export，核对原数学 AST/源码锁、输入前后 SHA、104 个 payload、两个原 dataset 的四条真实 queue 和无 fallback。保存命令、工具 manifest、image、实际退出/OOM、数据 manifest、源/权重 SHA 及结果；未完成或未通过的记录均保留。

版本化记录包含这组真实数据的单 rank 和两 rank 两配置 NPU8 更新和自身 iter4 恢复，均逐位相同。其范围是一个场景、每配置一个 normal 锚点、原 WE 派生奖励协议，不能自动转给新的 CLI 或推广到更多 rank。cc91/post8 默认 16rank 对照仍有 Tensor 超出原容差；[同步执行候选](../execution/README_zh.md)另有每模型一次的正例，重复可靠性待验；其他六个 rollout/replay/BWM 真实导出仍缺；本准备流程不覆盖它们。

下面是原 launcher 的复跑命令，路径均为服务器宿主路径。CPU 导出已实测；新导出数据通过原 dataset 四分支检查后，再确认目标芯片空闲；每次使用新目录，并由独立监督器实施 UTC 运行时间上限。一个连续/恢复对必须保持相同源、镜像与数据，不能混用不同版本的 checkpoint。命令用于新的复跑；每次执行仍需独立核对自身收据。

```bash
# Run on the server host after the CPU export and original dataset checks pass.
# Fill host paths; SOURCE/MAPS/INPUTS below are host paths, not container paths.
SOURCE='<host checkout of the verified runtime>'
SUBSETS='<host output of export_simscale_minimum.py>'
MAPS='<host official maps directory>'
INPUTS='<host locked public input directory>'
RUNS='<host fresh evidence directory>'
DEVICE='<one confirmed idle logical chip>'
IMAGE='<verified training image tag>'
IMAGE_ID='<verified training image ID>'

for CASE in finetune cotrain; do
  if [ "$CASE" = finetune ]; then
    NAME=e2e_vadv2_simscale_finetune_pdm_1round_8epoch
    INIT="$INPUTS/e2e_vadv2_100pct_ep8.pth"
  else
    NAME=e2e_vadv2_simscale_co_train_pdm_1round_8epoch
    INIT="$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
  fi
  COMMON=(--devices "$DEVICE" --config "$SOURCE/projects/AlgEngine/configs/simscale/$NAME.py"
    --checkpoint "$INIT" --vocab "$INPUTS/test_8192_kmeans.npy"
    --data-root "$SUBSETS/$NAME/data"
    --train-filter "$SUBSETS/$NAME/filters/navtrain_actual_normal_anchor.yaml"
    --synthetic-root "$SUBSETS/synthetic" --maps "$MAPS"
    --runtime --steps 8 --checkpoint-interval 4 --check-interval 1
    --batch-size 1 --workers 0 --deterministic --precision fp32
    --upstream-schedule --memory 64g --cpus 12 --pids-limit 4096
    --image "$IMAGE" --expected-image-id "$IMAGE_ID")
  python "$SOURCE/ascend/run_container.py" "${COMMON[@]}" \
    --output "$RUNS/${CASE}_continuous"
  python "$SOURCE/ascend/run_container.py" "${COMMON[@]}" \
    --resume "$RUNS/${CASE}_continuous/iter_4.pth" \
    --output "$RUNS/${CASE}_resumed"
done
```

逐项核对真实退出码、原 checkpoint 参数匹配、synthetic/normal 分支、有限 loss/抽查 gradient、完整保存状态恢复，以及实际前后精度收据。两个配置分别用原 `ascend/compare_resume.py` 对照 continuous/resumed 的 `iter_8.pth` 与原 JSONL。CPU 容器的官方 TorchNPU 预导入/runpy 命令见[运行时操作文档](../execution/README.md#original-numerical-comparison-on-cpu)，不得初始化 NPU、修改 loader 或放宽数值门槛。Tensor 门槛保持 `atol=1e-6, rtol=1e-5`，loss 门槛 `1e-4`。此流程不提供完整 NAVTRAIN 收敛或独立质量验收。
