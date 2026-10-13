# WorldEngine AlgEngine：Ascend 910C 部署、训练与验收操作文档

**2026-10-13 MSDA 更新：**当前 FP32 原生 NPU 采样实现、严格梯度、新模型回归与资源结果见 [MSDA 修复补充](MSDA_zh.md)及其[版本化实测结果](evidence/msda-precision-20261013.json)。下文 `96a7bef…` 指纹、旧一小时吞吐/内存和旧门禁失败均属于此前 fused 路径历史版本；不要将这些旧数字归给新源码。

本文与 [英文操作文档](FULL_ALGENGINE.md)、[已有三个规划入口的复跑说明](README.md) 使用同一份代码和实测证据。当前结论是**真实小数据功能部分通过，完整训练与质量尚未验收**。迁移代码已发布，不等于保证整套框架、全部数据和全部精度模式没有问题。

## 1. 先明确模块、版本与验收对象

WorldEngine 有两个模块：**SimEngine** 负责仿真与场景生成，继续使用 GPU，本次没有部署；**AlgEngine** 负责感知预训练、端到端轨迹规划、训练和微调，是本次 NPU 适配对象。AlgEngine 中消费 SimEngine/SimScale 导出数据的训练配置也在评估范围内，但这些配置仍需要真正的导出图像、标注和奖励。

| 固定项 | 值 |
|---|---|
| 上游 WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` |
| 迁移分支 | [ascend-910c-algengine](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine) |
| 已发布的实测代码提交 | `f804a8754e6c08d07e5302599dda6b068a41586c`；之后的文档提交不改变下面的 Python 指纹 |
| AlgEngine/Ascend Python 指纹 | `96a7bef17fd27a681b5f90c367507a3a9f72767e5698fbe514d077d20ec0f997`，239 个 Python 文件，路径排序且 CRLF 规范为 LF |
| 官方 NAVSIM | v1.1，`3e8291bfa89ff247231e0227778840cd0a036896` |
| 实测镜像 | `local/worldengine:26.1.0-a3-clean-20261013` |
| 实测镜像 ID | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` |
| 机器可读验收结果 | [full-algengine-20261013.json](evidence/full-algengine-20261013.json) |
| 原始 44 个配置的契约 | [original-configs-20261013.json](evidence/original-configs-20261013.json) |

实测设备为 Atlas 800I A3 推理系列、Ascend910_9362，8 张物理卡、16 个逻辑芯片，每个逻辑芯片 64 GiB HBM。实验中完成反向传播，不代表厂商已经正式支持该推理 SKU 的训练。厂商支持、模型质量和代码功能分别验收。

## 2. 已经跑到什么程度

保留原模型结构：200×200 BEV、900 个 tracking queries、8 相机；规划训练使用 4 帧时序。感知训练原配置使用 3 帧队列，评价补充历史后处理 37 帧。使用真实 JPEG、原标注与 PDM，训练 42 个锚点来自两段日志，独立评价 22 个锚点来自第三段日志，整段日志没有重叠。真实小数据覆盖功能，不能代表完整 NAVTRAIN 分布。

| 验证项 | 实际结果 | 结论边界 |
|---|---|---|
| 44 个公开配置 | 全部在固定环境解析；36 个真实数据配置各完成 3 次原模型优化器更新，使用各自原始初始化权重 | 覆盖 VADv2、HydraMDP、DiffusionDrive、两种 UniAD 感知配置；B1/workers0、小数据过滤及短 LR 是明确覆盖项，原数据比例与收敛未验收 |
| 8 个合成数据配置 | 缺真正导出数据，未执行训练 | 没有用真实 mini 冒充合成数据通过 |
| 原生感知 epoch 与续训 | 2 rank，epoch 1 每 rank 21 次更新；完整恢复后到 42；两个 epoch 都完成原始 detection/mapping 评价 | 恢复 1393 个 model、3594 个 optimizer Tensor，载入后逐项相同；mAP 为 0，质量不通过 |
| 规划 epoch 与独立评分 | 22/22 输出由原 NAVSIM CPU 评分；VAD reward/LoRA 两个 epoch 的 PDMS 为 0.845583、0.885092 | 小数据上升不等于完整基准提升；本次独立集 rare token 数为 0 |
| 控制变量数值续训 | 从同一连续训练的断点恢复，VAD reward/LoRA 的 model 最大差 7.45e-9、model/optimizer 最大差 2.98e-8、loss 差 0 | 通过既有 atol=1e-6、rtol=1e-5、loss=1e-4 门禁；历史 base/Hydra 的失败仍保留 |
| DiffusionDrive 8 rank | 每 rank 3 次更新；22/22 有限的 40×3 轨迹，官方 PDMS 0.377859，ADE/FDE 10.96562/19.69085 | 规划 head 仅短训练，不能作为成熟模型质量 |
| 感知 8 rank 持续训练 | 每 rank 428 次更新，最短 3610.270 秒，全部正常退出且没有 OOM | 重复 42 个训练锚点的一小时实测，不是完整数据吞吐、24 小时或内存泄漏排除 |
| 一小时资源 | 每芯片峰值 allocated/reserved 43.68/47.73 GiB；观察到 CPU cgroup 高水位 63.23 GiB，容器上限 128 GiB | cgroup 含被计费缓存；完整 NAVTRAIN CPU 内存尚未测量 |
| 多卡参数同步 | 8 rank、3 次原生更新后，1233 个参数 Tensor、79,777,383 个元素逐位相同 | 只声明参数一致，不声明 rank 本地非参数 buffer 一致 |
| 实际 8 rank 断点续训 | 从一小时训练的 iter_100 完整断点恢复，100→103；所有 rank 恢复 model/optimizer 和 8 份 RNG | Adam 条件参数合法计数为 [91,99,100]，不是每个参数都等于全局更新数；没有宣称全部步骤严格数值重放 |
| checkpoint 独立检查 | 40 个功能断点及 5 个一小时断点通过 CPU 序列化、有限性、计数/RNG 检查 | 不替代后续完整数据收敛验收 |
| 独立 CLI 感知推理 | epoch 续训模型和一小时最终模型均处理 37 帧，370 个实际预测字段有限，原评价器执行 | 空 map union 和原协议不适用项记 null 并保留覆盖数，不用 0 冒充有效指标 |

一小时端到端吞吐为 **0.94808 锚点曝光/秒**，包括真实取数、训练、监测和断点开销；前 20 次更新之后为 0.95762。428×8=3424 次曝光包含重复样本和 sampler padding。已记录的 update p50/p95 为 6.21183/7.00874 秒，其计时不包含取 batch，不能与端到端速率混为同一指标。没有 GPU 设备，当前没有 GPU/NPU 性能或模型轨迹对照。

## 3. 干净环境与镜像复现

以下命令在 **Linux 服务器**执行，示例目录可按实际交付路径替换。数据、权重下载与解压均留在服务器，不需要在 Windows 下载 5 TB。

```bash
git clone --branch ascend-910c-algengine --recurse-submodules \
  https://github.com/ZhengDeL/WorldEngine.git
cd WorldEngine
git rev-parse HEAD
git submodule status
npu-smi info
docker version

IMAGE=local/worldengine:26.1.0-a3-clean-20261013
IMAGE_ID=sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f
```

复现已测试镜像，优先使用交付的 `worldengine_clean_20261013.tar`。镜像没有宣称上传公开镜像仓库；离线包应由交付位置拷贝至服务器。其大小为 **23,444,539,904 bytes**，SHA256 为 `5c730fd509d9dea6ac1599094c7e1e60ab0fad78996031fcb46e7571988c5cac`。

```bash
sha256sum worldengine_clean_20261013.tar
docker load -i worldengine_clean_20261013.tar
docker image inspect --format '{{.Id}}' "$IMAGE"
# 必须与 IMAGE_ID 一致；launcher 也会校验。
```

有镜像仓库和 Python/Conda 下载访问条件时，可以按已公开 recipe 从固定官方 ARM64 基镜像重新构建：

```bash
docker build --no-cache --platform linux/arm64 \
  -f ascend/docker/Dockerfile -t local/worldengine:rebuild ascend/docker
docker image inspect --format '{{.Id}}' local/worldengine:rebuild
```

recipe 固定基镜像 manifest 和主要依赖，但重新解析软件包渠道不保证新 image ID 与离线包逐位相同。新镜像应记录实际 ID、检查包版本并重新跑功能门禁；使用 `--image` 和 `--expected-image-id` 指定它，不能直接把旧镜像的验收结论改写成新镜像已验收。

实测栈：驱动 26.1.1、CANN 9.1.0、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post6、mx_driving 26.1.0.post2、MMCV full 1.7.2、MMDet 2.24.0、MMDet3D 1.0.0rc4、NumPy 1.23.5。`+cpu` 是此 NPU 发行栈的版本标识，不表示训练跑在 CPU。训练 Python 为 3.10.21，单独 geospatial prefix 为 3.10.22；版本快照见 [docker/versions-20261013.json](docker/versions-20261013.json)。模型 Tensor 使用 FP32，matmul HF32 禁用，convolution HF32 允许；没有把它宣称为 GPU FP32 等价或 AMP/BF16 验收。

启动器将源码、数据、地图和初始权重只读挂载，将日志、checkpoint、CANN 编译和缓存写入独立 output。它不修改宿主共享包、不更改共享服务、不自动挂载 NAS。每次运行必须使用不存在的新 output，便于保留失败和对照。

## 4. 准备哪些数据、权重与存储

只运行小数据功能验证时，使用已经交付的 `mini_real_data.tar.gz`、原权重、8192 词表、地图及 22 个官方 scenario cache。mini 约 242.3 MB 压缩，包含 1120 张原始引用 JPEG；它不是完整训练集。移动数据时保留标注的相对相机路径；重新打包改变标注哈希后，应开始新训练。

完整数据目录约定如下：

```text
/datasets/worldengine/data/
  alg_engine/
    merged_infos_navformer/
      nuplan_openscene_navtrain.pkl
      nuplan_openscene_navtest.pkl
    pdms_cache/
      pdm_8192_gt_cache_navtrain.pkl
      pdm_8192_gt_cache_navtest.pkl
    test_8192_kmeans.npy
    ckpts/
  raw/
    openscene-v1.1/
      sensor_blobs/trainval/       # 所选当前帧和历史帧的 8 相机文件
      sensor_blobs/test/
      meta_datas/trainval/         # 官方 scenario cache 生成所需真实元数据
      meta_datas/test/
    nuplan/dataset/maps/           # nuplan-maps-v1.0.json 及引用的 .gpkg
```

| 输入 | 已核实大小或用途 |
|---|---|
| NAVTRAIN 标注 + PDM 两个 PKL | 39,862,469,866 bytes，序列化文件大小，不等于内存需求 |
| NAVTEST 标注 + PDM 两个 PKL | 3,639,831,385 bytes |
| 8192×40×3 轨迹词表 | 3,932,288 bytes；SHA256 见 lock |
| 五个本次用到的官方权重 | 合计 1,769,585,126 bytes |
| 定向 NAVTRAIN 图像下载候选 | 官方 NAVSIM v1.1 脚本列出的 8 个 current/history 包，HEAD 观察合计 449,046,313,693 bytes 压缩；尚未整包下载 |
| NAVTEST 图像、真实 metadata、地图 | 必须补齐独立评价和历史引用；图像包文件名存在不代表标注引用完整 |
| 合成/SimScale 消费配置 | 还要真正导出图像、转换后的 metadata、plan_idx、匹配的 rewards；见第 9 节 |

固定文件大小与 SHA256 见 [assets.lock.json](assets.lock.json)。VADv2/HydraMDP 选择式训练需要所选 token 的六组真实 8192 维 PDM 指标。感知和 DiffusionDrive 训练不需要选择式 PDM targets，但规划质量仍需要**官方 scenario metric cache**。这两个 cache 是不同输入，不能互相替代。

相机 pipeline 不读取原始 LiDAR。可以按标注、时序和 8 相机引用定向拷贝，而不必立即下载整个 5 TB 发布包。磁盘预算必须同时包含压缩包、解压图像、NAVTEST、标注、cache、权重和 checkpoint；不能只给 449 GB 的压缩包空间。

### 国内来源与仅在服务器下载

已验证服务器可访问发布者的 [WorldEngine 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)、[OpenScene 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene)、[SimScale 魔搭目录](https://www.modelscope.cn/datasets/OpenDriveLab/SimScale)。hf-mirror 已实际下载两个新增官方初始化权重并通过原发布 SHA256；魔搭还完成词表 byte-range 前缀比对，这不等于魔搭整个大数据集已下载校验。

```bash
WEIGHTS=/datasets/worldengine/weights
python3 ascend/fetch_assets.py --group all --source hf-mirror --output "$WEIGHTS"
# 指定源不可用时，只下载同一组已锁定权重：
python3 ascend/fetch_assets.py --group all --source modelscope --output "$WEIGHTS"
```

`--group planners` 只取两个已发布 epoch-8 规划模型；`warmstarts` 取 BEVFormer 和 50%/100% 感知初始化；`all` 取五个文件。工具支持 Range 续传和最终大小/SHA 校验，输出 `official_planners.json`。不会自动下载图片、完整 PKL 或 SimEngine 资产。镜像构建/下载网络确实受限时，可用已提供的服务器代理，限定到当前进程，例如：

```bash
PROXY_URL=http://proxy-host:1080  # 替换为运维提供的实际代理
http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" \
  python3 ascend/fetch_assets.py --group warmstarts --source hf-mirror --output "$WEIGHTS"
```

这不更改系统或 Git 全局代理。hf-mirror 固定 WorldEngine 数据 revision `d0e88d858533ca407a3387a26577c083aa7b3617`；魔搭 master 仍用同一组原发布 SHA 限定内容。

## 5. 先做 CPU 输入门禁与内存检查

完整 NAVTRAIN 尚未交付，以下是数据到位后的操作模板，**不是已执行的完整数据验收结果**。首先在单个 CPU 容器同时保留 train/eval 反序列化数据，确认真实 RSS，再决定 8 rank 的训练内存预算。

```bash
DATA=/datasets/worldengine/data
MAPS="$DATA/raw/nuplan/dataset/maps"
CACHE=/datasets/worldengine/metric_cache_navtest
RUNS=/experiments/worldengine
mkdir -p "$RUNS/preflight"
: "${PREFLIGHT_MEMORY_LIMIT:?先设置经过容量核对的单进程检查内存上限}"

docker run --rm --memory "$PREFLIGHT_MEMORY_LIMIT" --cpus 8 --entrypoint bash \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v "$PWD:/workspace/WorldEngine:ro" -v "$DATA:/data/full:ro" \
  -v "$MAPS:/data/maps:ro" -v "$CACHE:/data/cache:ro" \
  -v "$RUNS/preflight:/output:rw" "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    cd /workspace/WorldEngine
    python ascend/preflight.py --config configs/worldengine/e2e_vadv2_50pct.py \
      --data-root /data/full --maps /data/maps --metric-cache /data/cache \
      --require-official-score --hash-assets --ranks 8 --output /output/preflight.json
  '
```

检查 `result=pass_complete_inputs` 和退出码 0；缺文件、token、history、8 相机、PDM 维度/有限性、train/eval 日志重叠或官方 cache 覆盖时，报告具体原因并返回失败。`--hash-assets` 用于未改变的完整发布 PKL/词表，不用于重新打包的 mini。

`rss_bytes` 和 `peak_rss_bytes` 是实际观察；`cpu_data_resident_rank_projection_bytes` 只是该检查进程 RSS×rank 数，不能当作训练实测内存。还须加入模型/optimizer、评价、进程/runtime、worker 和实际余量。图像检查确认引用文件与大小，**没有逐图像内容哈希或完整解码验收**；地图检查也不能代替每个地区完整评价。真正合成内容需要另做 dataset build，第 9 节有具体条件。

## 6. 小数据训练、微调与一小时复跑

下面使用交付的真实 mini 和固定 image。设备编号指宿主逻辑芯片，例子 0,2,...,14 使用每张物理卡的一个芯片；先确认 `npu-smi info` 无冲突任务。

```bash
MINI=/datasets/worldengine/mini_real_data
VOCAB=/datasets/worldengine/test_8192_kmeans.npy
VAD="$WEIGHTS/e2e_vadv2_50pct_ep8.pth"
HYDRA="$WEIGHTS/e2e_hydramdp_50pct_ep8.pth"

# 单芯片 imitation 功能检查，原感知初始化：
python3 ascend/run_container.py --devices 0 \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --mini-prepared "$MINI" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth" --vocab "$VOCAB" \
  --steps 3 --batch-size 1 --workers 0 --output "$RUNS/vad_short" --detach

# 已发布规划基座的 reward/LoRA 微调：
python3 ascend/run_container.py --devices 0 --profile vadv2_ft --runtime \
  --mini-prepared "$MINI" --checkpoint "$VAD" --vocab "$VOCAB" \
  --steps 30 --batch-size 2 --workers 0 --deterministic \
  --checkpoint-interval 10 --output "$RUNS/vad_ft" --detach

# Hydra 原配置包含实际 policy gradient；使用它自己的基座：
python3 ascend/run_container.py --devices 0 --profile hydramdp_rl --runtime \
  --mini-prepared "$MINI" --checkpoint "$HYDRA" --vocab "$VOCAB" \
  --steps 30 --batch-size 2 --workers 0 --checkpoint-interval 10 \
  --output "$RUNS/hydra_ft" --detach
```

微调 mini 按原 rare/common 语义构造训练选取；VAD reward/LoRA 原配置 PG 禁用，不能宣称两者均启用 PG。旧小数据独立评价中，VAD 基座/微调 PDMS 0.910051/0.876269，Hydra 0.934006/0.934006：功能运行成功，质量收益未验收，失败或退化结果都保留。

感知完整 backbone+tracking+map 路径，使用原 BEVFormer 初始化：

```bash
python3 ascend/run_container.py --devices 0 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 3 --batch-size 1 --workers 0 \
  --output "$RUNS/perception_short" --detach

# 感知 8 rank 一小时压力检查：10000 是 LR 总 horizon，3600 是实测时长停止条件。
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --steps 10000 --upstream-schedule \
  --duration-seconds 3600 --batch-size 1 --workers 0 --memory 128g --cpus 64 \
  --checkpoint-interval 100 --check-interval 20 --output "$RUNS/perception_hour" --detach
```

128g 是本次 mini 一小时的实测容器配置，不是全数据预算。保留上游 optimizer LR/warmup 时，iter-based 压力 horizon 仍区别于原始 epoch 训练。单芯片短感知训练实测峰值约 43.61 GiB allocated；B1 适用，DDP 增加芯片不会自动减少每芯片模型内存。

原始 epoch 流程用 `--epochs` 而非 `--steps`。例如用 2 rank 保留 2-epoch horizon、在第一轮后停下：

```bash
python3 ascend/run_container.py --devices 0,2 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --epochs 2 --stop-after-epochs 1 --validate \
  --batch-size 1 --workers 0 --memory 64g --cpus 32 \
  --output "$RUNS/perception_partial" --detach
```

等待运行结束后再续训或复用设备。`--detach` 只表示启动容器；完整成功需同时检查退出码、真实更新、梯度/参数与推理评价，见第 10 节。

## 7. 完整状态断点续训

`--checkpoint` 是初始权重；`--resume` 才恢复 model、optimizer、每 rank RNG、采样进度、LR 与训练契约。旧的仅权重 checkpoint 不能替代完整状态。当前精确状态续训要求 workers=0，源代码、image、数据/过滤、world size、batch、seed、optimizer 和 LR 总 horizon 不变，output 必须换新目录。

```bash
# 接第 6 节的感知原生 epoch；保留 epochs=2 和其余参数，去掉 stop-after-epochs：
python3 ascend/run_container.py --devices 0,2 \
  --config projects/AlgEngine/configs/worldengine/track_map_nuplan_r50_navtrain_50pct.py \
  --mini-prepared "$MINI" --checkpoint "$WEIGHTS/bevformerv2-r50-t1-base_epoch_48.pth" \
  --vocab "$VOCAB" --maps "$MAPS" --epochs 2 --validate \
  --batch-size 1 --workers 0 --memory 64g --cpus 32 \
  --resume "$RUNS/perception_partial/epoch_1.pth" \
  --output "$RUNS/perception_resumed" --detach
```

iter 训练同理：保持 `--steps 10000`，从一小时断点 `iter_100.pth` 恢复，测试可加 `--stop-after 103`，不保留 `--duration-seconds`。训练参数与原 run 相同。实际 8 rank 已完成该恢复及 3 次后续更新。

载入后的全部 model/optimizer Tensor 检查应为 exact；条件分支参数的 Adam 计数允许各不相同，但必须合法且不大于全局更新数。完整状态恢复通过，和后续每一步数值严格重放通过，是两个独立结论。

做严格数值对照时，先完成一个连续 run 并在中间保存 checkpoint，再从**该连续 run 自己的 checkpoint**起点续训到相同终点，使用 [compare_resume.py](compare_resume.py) 对比 model/optimizer、loss、tokens、LR、final RNG；具体 CPU 容器命令见 [英文复跑说明](README.md#run-save-and-resume)。不能用两次独立初始化前缀的差异证明断点实现有误。

迁移工具同时核对原配置文件的原始字节哈希。恢复本次旧断点应使用交付的精确源码 tar 或实际训练源码目录；从 Git 在不同系统重新 checkout 可能改变配置换行，虽然规范化 Python 指纹相同，也可能被 resume 契约拒绝。不要删掉该检查来绕过数据/源码漂移。新 clone 可开始新训练。

## 8. 推理和官方独立评分

推理使用训练输出的 `*_rank0.py`，其中包含已确定的实际数据路径与 head 类型。感知输出真实 boxes/scores/maps；DiffusionDrive 检查它生成的 40×3 轨迹；选择式规划还检查词表选取一致。不要把模型内部 PDM cache targets 当成独立质量结论。

```bash
python3 ascend/run_container.py --devices 0 --task inference \
  --config "$RUNS/perception_resumed/track_map_nuplan_r50_navtrain_50pct_rank0.py" \
  --mini-prepared "$MINI" --maps "$MAPS" \
  --checkpoint "$RUNS/perception_resumed/epoch_2.pth" --vocab "$VOCAB" \
  --output "$RUNS/perception_eval" --detach

# DiffusionDrive 的实际生成轨迹与 ADE/FDE：
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 \
  --config projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_100pct.py \
  --mini-prepared "$MINI" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth" --vocab "$VOCAB" \
  --steps 3 --batch-size 1 --workers 0 --inference-after \
  --memory 128g --cpus 64 --output "$RUNS/dd_short" --detach
```

为规划结果生成官方 CPU 评分，可使用固定 image、真实 logs/maps 与未改动的 NAVSIM v1.1：

```bash
SCORING=/datasets/worldengine/scoring  # 内含 navsim_logs/ 和 maps/
mkdir -p "$RUNS/dd_cpu_score"
docker run --rm --memory 24g --cpus 8 --entrypoint bash \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/source:ro" -v "$MINI:/mini:ro" -v "$SCORING:/scoring:ro" \
  -v "$RUNS/dd_short/inference:/predictions:ro" -v "$RUNS/dd_cpu_score:/output:rw" \
  "$IMAGE" -lc '
    export PATH=/opt/conda/envs/bevformer/bin:$PATH
    export PYTHONPATH=/source/ascend/third_party/navsim:$PYTHONPATH
    cd /source
    python ascend/official_score.py --prepared /mini \
      --logs /scoring/navsim_logs --maps /scoring/maps --cache /output/cache \
      --predictions /predictions/predictions.pkl --output /output/official_score.json
  '
```

该评分按上游 `[4::5]` 从 40 个 pose 取 8 个，使用官方 cache generation、simulator 与 scorer。上例 24g 是小数据 CPU 评分模板，不能当作全数据评分内存预算。已经生成官方 metric cache 时，原生 epoch 的 `--official-metric-cache` 可以只读使用 payload，并在 output 重建搬迁后的 metadata 路径；每个 epoch 保留 predictions、逐 token 分数和整体/可用 rare 指标。

感知评价的协议不适用 TP 指标和真实 zero-union 项记 null，并保留 category 覆盖；其他非有限预测/指标返回失败。规划缺 token、非有限轨迹或官方 cache 不完整也返回失败。GPU 当前不可用，CPU 官方打分能验证 NPU 轨迹的业务指标，但不能代替 GPU/NPU 相同模型输出对照。

## 9. 真正合成数据与完整 NAVTRAIN 如何继续

8 个真实合成/SimScale 消费配置的现状是**输入不足、训练未实测**。需要 `--synthetic-root`，WorldEngine rollouts 包含 `sensor_blobs/`、`pdms_pkl/`、`meta_datas/`、`plan_idx.csv`；SimScale 使用 `meta_datas_navformer/`、per-folder reward PKL 与 split YAML。保留原配置的 folder_name 和过滤条件；缺根目录会提前失败。

额外前置问题也必须解决：Hydra BWM 原配置包含 `/path/to/synthetic/rollouts/...` 占位项，应按实际导出目录做可审查映射；SimScale fine-tune 的 `e2e_vadv2_ep8.pth` 原始文件/别名未确认，不能悄悄替换为 50% planner；公开 reward 文件名提示 16384 轨迹，而配置使用 8192 词表，须验证实际维度、候选顺序和转换。把 reward 改名不会让它自动兼容。详细配置列表在 [原配置契约](evidence/original-configs-20261013.json) 和 [结果 JSON](evidence/full-algengine-20261013.json) 中。

完整 NAVTRAIN 收敛不依赖下载 SimEngine 的全部 5 TB，但确实依赖完整所选图像、标注、PDM、独立 NAVTEST 与官方评分 cache。数据到位后的顺序是：

1. 拷贝/下载到服务器，核对 lock、history、8 相机与独立日志划分；CPU preflight 不通过先修输入。
2. 实测完整 PKL 反序列化 RSS，再运行代表性完整数据训练，记录 rank RSS、容器 cgroup、HBM、I/O；由这些实际值设置训练内存上限。
3. 按各原配置完成全部 horizon。规划通常 8 epochs，感知 20/40，DiffusionDrive 100；每轮保留 checkpoint 与独立指标，验证中断恢复。
4. 核对 common/rare 覆盖与独立完整 NAVTEST 指标，不以训练 loss 下降代替质量。质量阈值与最终目标配置确定后才作正式通过判断。
5. 在目标 batch、worker、芯片布局与完整数据上持续训练，确认要求的运行时长、吞吐、内存和失败恢复；小数据一小时不能自动覆盖这些条件。

例如原始 VAD 50% NAVTRAIN 训练，保留原 8-epoch horizon、原 batch/optimizer/LR；以下必须等完整输入与容量检查通过后执行：

```bash
: "${MEASURED_CPU_MEMORY_LIMIT:?先设置依据完整数据实测的训练内存上限}"
python3 ascend/run_container.py --devices 0,2,4,6,8,10,12,14 --runtime \
  --config projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
  --data-root "$DATA" \
  --checkpoint "$WEIGHTS/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth" \
  --vocab "$DATA/alg_engine/test_8192_kmeans.npy" --maps "$MAPS" \
  --official-metric-cache "$CACHE" --validate --workers 0 \
  --memory "$MEASURED_CPU_MEMORY_LIMIT" --cpus 64 \
  --output "$RUNS/navtrain_vad50" --detach
```

要演练第 2 个 epoch 后中断，第一次增加 `--stop-after-epochs 2`；续训保持原 horizon、validate 与全部契约不变，用 `--resume .../epoch_2.pth` 和新 output。此处不使用 `--mini-prepared`、`--steps` 或三步短 LR。原始 50% filter 为 50,000 anchors，完整 NAVTRAIN YAML 为 103,288；小数据反复训练不能作为这些集合的收敛。

## 10. 怎样确认跑成功、需要几台设备

容器名为 `worldengine-npu-<output目录名>`。检查具体 run，不删除失败输出：

```bash
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  worldengine-npu-perception_hour
cat "$RUNS/perception_hour/exit_code"
tail -n 50 "$RUNS/perception_hour/console.log"
ls "$RUNS/perception_hour"/runtime_rank*.json
npu-smi info
```

`launch.json` 固定镜像、设备与命令；`runtime_rank*.json/jsonl` 提供真实更新数、原始 loss、梯度检查、tokens/LR、RSS/HBM、恢复结果。验证还看 `validation/epoch_*/evaluation.json`、官方评分目录与 `inference_validation.json`。单看 Docker Up、loss 日志一行或退出码 0 不能证明全部验收通过。检测到 NaN/Inf 会记录失败，不用原模型 nan_to_num 掩盖原始 loss 问题。

| 目的 | 当前有证据的设备/资源 | 尚待测量 |
|---|---|---|
| 小数据单模型功能、B1 感知 | 1 个 64 GiB 逻辑芯片即可运行完整结构；峰值约 43.61 GiB allocated | 更大 batch、AMP、其他芯片容量 |
| 原始 8 rank global batch 布局及一小时 | 1 台含 8 个合适逻辑芯片的服务器，本次每物理卡取一个芯片 | 16 芯片、不同卡内布局或多机扩展 |
| 完整 NAVTRAIN | 当前只确定模型/小数据可在一台运行 | 全数据 CPU object 膨胀、数据 I/O、完整训练耗时/收敛和截止时间内所需机器数 |

8 rank 不等于 8 台服务器。现有一台服务器能够运行已实测的 1/2/8 rank 情形；不能据此保证一台就满足完整数据训练的内存和交付时间。DDP 每 rank 仍有模型副本与自己的数据对象；39.86 GB 的两个训练 PKL 不是完整训练峰值 RAM。若 CPU/I/O 成为瓶颈，多买 NPU 也不会自动解决。

## 11. 运行优势、代价与尚未通过项

这次适配提供统一的原配置入口、HCCL 多卡、完整状态恢复、原始评价与独立官方 CPU 打分，可以在没有 GPU 对照机时先确认 NPU 功能与官方指标。真实小数据路线避免等待 5 TB 完整下载；保存镜像、数据和源码哈希后能区分版本漂移与算子/数据错误。CPU 打分与 NPU 训练分开执行，保留上游评分语义。

代价是旧 OpenMMLab/CUDA 依赖要固定到已验证的 NPU 软件栈；unfrozen backbone 的 fused MaxPool 无 backward，需要 PyTorch NPU 路径。当前 FP32 MSDA 使用原生 NPU grid_sample 前后向、query 分块及后向重算，完整模型仍有矩阵求逆的 CPU fallback。新实现的带监测一小时吞吐为 0.35381 锚点曝光/秒，性能优化与完整数据容量仍需实测。容器输出必须可写用于 CANN 编译/cache，源代码只读不能直接存这些文件。CPU 大 PKL、地图和 I/O 仍需单独容量设计。

**当前严格 MSDA 门禁已通过：**`858f9d…` FP32 路径在原 CPU32 参考、原容差下通过正常、特殊布局及完整 Q40000 几何的前向与三梯度检查，原 seed29 超差元素已由 35/19200 变为 0。两组单 rank VADv2/Hydra 受控严格续训也已通过。历史 `96a7bef…` fused 路径曾有 35/19200 元素失败、最大差 0.0309448、relative L2 2.73677e-6，原 atomic accumulation 观察和历史多卡续训失败保留为旧源码证据，不再列作当前位置梯度缺项。八 rank 全状态恢复已通过，目标多卡严格数值重放仍需独立验证。详细条件见 [MSDA 修复实测](MSDA_zh.md)。

完整 NAVTRAIN 收敛、完整独立 NAVTEST common/rare 质量、真正合成 8 配置、目标多卡/长训练严格数值重放、完整数据 CPU RAM/I/O/工期、目标 batch/workers 的性能和持续稳定性仍需验收。GPU 模型对照和具体机型的厂商正式训练支持需要外部条件；AMP/BF16、二阶梯度、16 芯片/多机是扩展能力，不能列为已测 FP32 单机路径的运行失败。当前发布物包含可执行的迁移代码、操作方法和有边界的实测证据，完整数据正式验收尚未完成。
