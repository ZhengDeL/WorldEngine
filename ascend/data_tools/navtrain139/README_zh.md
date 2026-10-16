# 确定性准备 NAVTRAIN139 / NAVTEST22

本工具从官方 WorldEngine 注释/PDM 和 OpenScene 图像派生固定验证子集，不是官方发布的“139/22标准小包”。它不下载数据，不依赖私有中间数据包，也不查询硬件。官方地址、固定版本、原文件完整大小及 SHA 见 `INPUTS.lock.json`。

已有实测数据由一个真实 NAVTRAIN 日志的139个锚点和另一个 NAVTEST 日志的22个锚点组成，来源及逐项证据见 `DATA_PROVENANCE.json`。公开 CLI 已于2026-10-16使用六个锁定的官方完整缓存文件实际重建。两个 CPU 阶段均退出0、未挂 NPU：1832张原 JPEG 全部通过完整 SHA 和解码检查，六份派生元信息既通过完整嵌套值及候选顺序比较，也与先前子集逐字节相同。实测见[CPU 重建记录](VALIDATION.json)。本项只验证数据准备，不据此推断模型训练或质量通过。

## 原始缓存输入

需要四个官方完整 ANN/PDM PKL、mini相机12包和test相机30包。若无任何缓存，六文件共47,913,227,335字节；当前直接复用已有校验缓存，不安排完整图像集或仿真资产下载。最终只复制1832张实际队列引用的原 JPEG，不生成随机数据、奖励，不缩放或重新编码图像。

WorldEngine版本为 `d0e88d858533ca407a3387a26577c083aa7b3617`，OpenScene版本为 `a76f840b65e972bc45e56c2adced897498e9a026`；原 split 来源提交为 `ec13257a95690a3e74d201f3b94064313a9369f8`。

训练保留原连续284帧、该日志全部139个原 split 锚点及其原生四帧队列。评测保留51帧和既有独立分支的22个固定官方数据锚点。公开筛选清单只含标识符和原图哈希，不含新标签。PDM 原完整条目、8192候选值与顺序保持，不附加 future-mask 或 SimScale 排除条件。

准备器只执行原 `load_annotations` 的未修改 AST，并使用真实 pickle IO，保留原索引和两个原生 scene 字段赋值；不构造完整 dataset、不运行 pipeline、不导入硬件 SDK、不建模型。评测仅将相机路径改成 `log_name/原文件名`，其他字段值保持。

## 在 Linux 服务器执行

使用已验收的依赖修复镜像，或先按公开配方构建。CPU 准备需要 NumPy1.23.5、PyYAML、Pillow；预留128GiB内存、8CPU及至少2GiB新输出空间。容器不挂 NPU、驱动或任何设备。六个原文件可以留在不同缓存目录，不必再复制40GB元信息到同一目录。

```bash
set -euo pipefail
SOURCE=/path/to/WorldEngine
TOOLS="$SOURCE/ascend/data_tools/navtrain139"
NAVTRAIN_ANN=/cached/nuplan_openscene_navtrain.pkl
NAVTRAIN_PDM=/cached/pdm_8192_gt_cache_navtrain.pkl
NAVTEST_ANN=/cached/nuplan_openscene_navtest.pkl
NAVTEST_PDM=/cached/pdm_8192_gt_cache_navtest.pkl
TRAIN_CAMERAS=/cached/openscene_sensor_mini_camera_12.tgz
EVAL_CAMERAS=/cached/openscene_sensor_test_camera_30.tgz
OUTPUT_PARENT=/path/to/new/outputs
IMAGE_ID=sha256:REPLACE_WITH_ACTUAL_ACCEPTED_IMAGE_ID
DEADLINE=$(date -u -d '+15 minutes' '+%Y-%m-%dT%H:%M:%S+00:00')
docker run --rm --runtime runc --network none --memory 128g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$SOURCE:/workspace/WorldEngine:ro" -v "$TOOLS:/tools:ro" \
  -v "$NAVTRAIN_ANN:/inputs/navtrain.pkl:ro" \
  -v "$NAVTRAIN_PDM:/inputs/navtrain_pdm.pkl:ro" \
  -v "$NAVTEST_ANN:/inputs/navtest.pkl:ro" \
  -v "$NAVTEST_PDM:/inputs/navtest_pdm.pkl:ro" \
  -v "$TRAIN_CAMERAS:/inputs/train_cameras.tgz:ro" \
  -v "$EVAL_CAMERAS:/inputs/eval_cameras.tgz:ro" \
  -v "$OUTPUT_PARENT:/output:rw" "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python /tools/prepare_navtrain139.py \
  --source /workspace/WorldEngine \
  --navtrain-annotations /inputs/navtrain.pkl --navtrain-pdm /inputs/navtrain_pdm.pkl \
  --navtest-annotations /inputs/navtest.pkl --navtest-pdm /inputs/navtest_pdm.pkl \
  --train-cameras /inputs/train_cameras.tgz --eval-cameras /inputs/eval_cameras.tgz \
  --output /output/prepared --deadline "$DEADLINE" --max-seconds 900 --execute
```

事先创建 `OUTPUT_PARENT`，保持其下 `prepared` 不存在。不加 `--execute` 时只输出计划，不读完整数据、不创建输出。六个完整原输入前后都核完整 SHA；全量 tar 扫描拒绝路径逃逸、链接、特殊文件及重复来源，只复制引用图像，逐张原始字节哈希及真实解码必须通过。

输出含六个派生元信息文件、`cameras/`、兼容公开启动器的 `manifest.json`、`preparation.json`、`result.json`。运行时传 `--mini-prepared "$OUTPUT_PARENT/prepared"`；训练/评测 PDM 文件保留原加载器需要的 navtrain/navtest 命名。

这次实测重建的六份元信息与历史派生文件逐字节相同，收据也独立核对完整嵌套字段和候选顺序。以后重放若使用不同的对象编码，文件字节是否相同仍单列记录，不用语义一致替代原始文件哈希。每个新输出都有自己的数据哈希合同，必须单独验证模型、推理与评分，不能借用另一准备目录的断点兼容或 NPU 结果。该子集不能证明完整 NAVTRAIN 收敛、完整 NAVTEST 质量。失败保留原收据与部分输出，重试使用新目录。
