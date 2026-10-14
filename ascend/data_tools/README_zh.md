# 真实SimScale小样本复现工具：私有暂存稿

三个CLI已于2026-10-14 04:22:57（北京时间）完成服务器真实CPU全链复跑：转换26个clip、解码832个历史JPEG、重新计算8192候选奖励、导出104个payload。四阶段均退出0、无OOM，原始输入SHA未变，来源与收据SHA见 `CPU_REPRODUCTION_VERIFIED.json`。

本目录尚未公开发布。新导出数据已完成原dataset的CPU检查，以及post8单rank和双rank的真实训练/恢复对照。

| 阶段 | 完成时间（北京时间） | 实测结果 |
| --- | --- | --- |
| 原dataset CPU getitem | 2026年10月14日 | 两个原配置、synthetic/normal四次请求；每次prepare一次，None=0、retry=0，token、四帧图像与六项8192评分一致 |
| 单rank连续/恢复训练 | 2026年10月14日04:54:02 | 两配置各连续8更新，再从自身iter4恢复到8；Finetune模型/完整Tensor数989/1291，Cotrain为909/1748，均逐位相同 |
| 双rank连续/恢复训练 | 2026年10月14日05:00:51 | 两配置各连续8更新、自身iter4恢复到8；全部8份rank功能收据通过；Finetune为989/1293，Cotrain为909/1750，均逐位相同 |

完整Tensor计数包含rank RNG。两轮训练的参数最大差和loss差均为0，tokens、LR和最终RNG相同。对应收据为 `GETITEM_CPU_VERIFIED.json`、`NPU_SINGLE_RANK_VERIFIED.json` 和 `NPU_TWO_RANK_VERIFIED.json`。CPU检查使用SDK导入期设备名元数据桥：两次查询取得此前实测设备名后，原函数恢复且NPU未初始化。这不是无桥CPU导入验收。

范围是一个真实场景、每配置一个明确normal锚点及原WE派生奖励协议。其他六个rollout/replay/BWM配置所需真实导出尚未齐备；完整NAVTRAIN收敛、GPU对照及16rank严格数值验收未通过。前两个工具的转换、评分和输入适配函数AST与已实测参考helper相同；导出工具还通过18项本机队列、tar安全和dry-run检查。公开发布仍待最终审核。

这一流程使用一个真实SimScale场景、真实NAVTRAIN正常样本和原WorldEngine四秒CPU评分。它不复现SimScale v2的16384奖励，不部署SimEngine渲染器，也不验收完整NAVTRAIN收敛。原始图像、标定、GT、词表顺序和正常样本PDM不修改。

## 输入和容量

所有数据下载、校验和提取在服务器执行。具体原路径、固定revision、bytes、SHA256、官方与hf-mirror地址见 `INPUT_LOCKS.json`。已准备的文件直接复用，不重复下载。

| 输入 | 完整下载字节数 | 用途 |
|---|---:|---|
| NAVTRAIN annotation | 16,125,493,728 | 从原246165帧选择两个正常锚点的原四帧queue |
| NAVTRAIN 8192 PDM | 23,736,976,138 | 保留两个正常锚点的原8192评分entry |
| SimScale round0 metadata | 3,320,437,526 | 原hist65索引对应26个clip的原转换，最后只训练一个真实场景 |
| SimScale hist65相机 | 241,575,885 | 26个clip的832个原历史JPEG；训练subset使用选中场景32个 |
| OpenScene mini12相机 | 2,781,004,086 | 两个实际正常锚点各32个原JPEG |
| OpenScene raw mini metadata | 534,344,190 | 读取一个完整原日志并绑定连续12帧记录交通 |
| 8192词表与hist索引 | 3,932,288 + 1,824,152 | 原候选顺序、hist65原成员选择 |
| 正式finetune/cotrain初始化 | 434,885,463 + 319,958,048 | 后续原两配置的NPU验证，不参与CPU奖励计算 |

以上合计 **47,500,431,504 B，约47.50 GB**，maps、源码、镜像和新输出另计。annotation与PDM合计39,862,469,866 B；虽然最终只取八条原normal记录及两个PDM entry，发布者提供的是完整PKL，仍需要这两个完整文件。此流程不要求449 GB完整训练图像、90 GB官方mini全相机、5 TB资产、LiDAR或future相机。

先前检查的19,305,089,518 B发布奖励与原8192训练协议不兼容：16384候选、字段不同，目标8192的完整轨迹行与16384没有精确对应。它仅用于历史兼容性调查，**不是本流程训练输入或必下载文件**。

使用真实 `nuplan-maps-v1.0.json` 与对应 `.gpkg` 地图；已有官方maps可复用。本次没有取得独立全地图下载大小，不能把maps包含在47.50 GB合计中。CPU运行使用已锁镜像 `local/worldengine:26.1.0-a3-clean-20261013`，ID `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`，保留镜像原PYTHONPATH和依赖目录。

## 初始化权重下载与校验

以下两份权重固定到WorldEngine数据仓revision `d0e88d858533ca407a3387a26577c083aa7b3617`。下载只在服务器执行。已有完整文件时直接按 `INPUT_LOCKS.json` 校验并复用，不重复下载；以下命令仅供新下载目录使用，运行前启用 `set -euo pipefail`，任一大小或SHA失败立即停止，保留 `.part`，不作为训练初始化。

| 用途 | 文件 | 字节数 | SHA256 |
| --- | --- | ---: | --- |
| Finetune显式override | `e2e_vadv2_100pct_ep8.pth` | 434,885,463 | `fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a` |
| Cotrain原初始化 | `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth` | 319,958,048 | `9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a` |

官方HF和国内hf-mirror使用相同固定revision、发布路径和内容SHA，完整URL见 `INPUT_LOCKS.json`。Finetune的100pct权重是明确的override；原配置指定的 `e2e_vadv2_ep8.pth` 尚未找到，不得把下载文件改名冒充它。

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

拟将工具放在仓内 `ascend/data_tools/`。现有训练指纹明确只包括 `projects/AlgEngine/**/*.py` 和 `ascend/*.py`；嵌套的纯数据准备工具独立使用 `TOOLS_MANIFEST.json` 闭环。记录这一范围的理由是区分运行时数值代码和数据生产代码，不是忽略数据变更。训练合同仍锁定实际annotation、filter、PDM和vocab内容SHA。

本暂存稿没有修改运行时的五份Python、CANN开关或任何源码合同。调用者必须提供真实 `--expected-runtime-sha256`。工具另记录原converter及四份SimEngine评分/转换源码SHA、抽取方法AST、maps SHA、发布输入与生成输出SHA。不能把旧858f工具的结果改写成新cc91工具实测，也不能把新的暂存CLI的AST通过写成CPU运行通过。

三份暂存CLI都是默认dry-run；加 `--execute` 才执行。每阶段使用新输出，不覆盖已有证据；history阶段只向已通过conversion的目录追加此前不存在的图像及history receipt。整个tar先审查绝对路径、`..`、链接、特殊成员和重复文件名，然后只取实际引用，绝不extractall。

## 分步操作

下面的变量由运行者填写；数据文件必须先匹配 `INPUT_LOCKS.json`。在CPU容器内挂载源码只读、输入只读、tools只读和独立output可写，不挂载NPU设备。

```bash
WE_SOURCE=/workspace/WorldEngine
TOOLS=/tools
RUNTIME_FP='<所选源码实际指纹>'
INPUTS=/inputs
MAPS=/maps
OUT=/output/new_reproduction
DEADLINE='<本次授权的未来UTC时间>'
```

第1步使用CPU8GiB/4CPU，沿原converter的默认2Hz、1.5秒past、4秒future、`filter_boxes=False`处理26个clip。调用原 `parse_log`，不改原转换数学；核对frame identity、标定、三步过去及八步未来的GT/mask。输出路径不能含 `meta_datas` 字符串，因为原converter按原路径替换该目录名。

```bash
python "$TOOLS/simscale_convert.py" --mode convert \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_meta_datas.tar.gz" \
  --hist-index "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist.json" \
  --output "$OUT/converted" --deadline "$DEADLINE"
```

先检查计划，再对同命令追加 `--execute`。通过后，仍使用CPU8GiB/4CPU执行history阶段，逐个JPEG保持原byte并解码；不下载future相机。

```bash
python "$TOOLS/simscale_convert.py" --mode history \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist_65.tar.gz" \
  --output "$OUT/converted" --deadline "$DEADLINE" --execute
```

第2步使用CPU64GiB/16CPU和原八个评分worker，生成选中场景8192条原序候选的奖励。

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

检查计划后加 `--execute`。输出为独立 `worldengine_8192_logged_traffic_derived_rewards.pkl`，绝不覆盖或截取原16384奖励。原日志734帧中，十二个literal synthetic token对应连续原index213–224；scene名称与frame3 token不同。合成当前ego与原记录交通不假定物理状态相同。

该派生协议保留原WorldEngine四秒评分、原8192词表、原八个worker和每候选相对PDM reference的进度归一化。reference planner只接收实际四帧history，评分接收当前加八帧记录交通；路线为原/合成一致的发布roadblock IDs；原NAVSIM转换提供global detections并包含记录ego。零加速度、零转角沿原WE ego converter。它不是SimScale v2、完整引擎生命周期或原生nuPlanDB加载等价协议。

第3步使用已实现的 `export_simscale_minimum.py`，CPU容器上限128GiB/4CPU，宿主至少保留192GiB共享余量。它直接读取已锁定的官方annotation、PDM和mini12相机包，核对当前conversion、history、reward收据及原数学源码锁后导出；不依赖此前私有getitem或图像提取收据。默认dry-run，检查计划后加 `--execute`。该CLI已完成服务器真实CPU重跑，保留64个normal和32个synthetic原JPEG、原12帧synthetic记录与原正常样本PDM。导出进程峰值RSS为28,364,746,752 B；完整annotation、PDM和mini12相机包前后SHA一致。

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

导出必须从完整原annotation按原前驱index取四帧，而不是精确500000微秒查找。保留原record、GT、calibration、image path和PDM entry；两个token都属于原NAVTRAIN。两个配置各输出35个payload，共享synthetic输出34个，共104个payload。显式one-anchor normal白名单有意缩小检索池，不等价于全数据随机检索。原retrieval、folder_name、train PDM选择、head/loss分支留给原dataset运行，不伪造私有历史getitem收据来满足门禁。

第4步仅在subset CLI真实重跑及来源核对后，使用原 `ascend/run_container.py` 分别运行两个SimScale原配置。每配置自己的data root和 `filters/navtrain_actual_normal_anchor.yaml`，共享真正synthetic root，分别初始化8更新并从自己的iter4完整恢复到8。Tensor门槛仍1e-6/1e-5，loss差小于1e-4；原CPU loader和比较器不改。

Finetune明确使用发布的 `e2e_vadv2_100pct_ep8.pth` 作为checkpoint override；原配置的 `e2e_vadv2_ep8.pth` basename未找到，不能静默改名冒充。Cotrain使用原 `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth`。运行时flags/source指纹及设备由最终验收版本与执行方指定，数据工具不设置或“修复”训练开关。

## 发布前重跑

已在实际服务器以暂存CLI重新完成conversion、832个原history图像、CPU8192评分和subset export，核对原数学AST/源码锁、输入前后SHA、104个payload、两个原dataset的四条真实queue和无fallback。保存命令、工具manifest、image、实际退出/OOM、数据manifest、源/权重SHA及结果；任何失败保留。

先前私有工具已经完成这组真实数据的单rank和两rank两配置NPU8更新和自身iter4恢复，均逐位相同。其范围是一个场景、每配置一个normal锚点、原WE派生奖励协议，不能自动转给新的CLI或推广到更多rank。新cc91的16rank严格重放仍有失败，包括post8首个VAD对照；其他六个rollout/replay/BWM真实导出仍缺；本准备流程不覆盖它们。

下面是原launcher的复跑命令，路径均为服务器宿主路径。CPU导出已实测；新导出数据通过原dataset四分支检查后，再确认目标芯片空闲；每次使用新目录，并由只管理本次任务的监督器落实授权截止时间。一个连续/恢复对必须保持相同源、镜像与数据，不能混用不同版本的checkpoint。命令用于新的复跑；每次执行仍需独立核对自身收据。

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

逐项核对真实退出码、原checkpoint参数匹配、synthetic/normal分支、有限loss/抽查gradient、完整保存状态恢复，以及实际前后精度收据。两个配置分别用原 `ascend/compare_resume.py` 对照continuous/resumed的 `iter_8.pth` 与原JSONL。CPU容器的官方TorchNPU预导入/runpy命令见[运行时操作文档](../README.md#run-save-and-resume)，不得初始化NPU、修改loader或放宽数值门槛。Tensor门槛保持 `atol=1e-6, rtol=1e-5`，loss门槛 `1e-4`。此流程不提供完整NAVTRAIN收敛或独立质量验收。
