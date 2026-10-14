# 规划基座同步执行候选

[English](README.md) · [AlgEngine 操作文档](../README_zh.md)

适配器 SHA256：`b6c5d067cc7ba54efa4650e8278c721d6d4187a65bc91870a8b4b1fca4405b5f`。

可选的 `ascend/execution/synchronous_launcher.py` 只覆盖原 VAD/Hydra 基座配置、
16 个逻辑芯片、B1、workers0、FP32、8 次更新及每 4 次保存断点。它校验 cc91 运行
指纹（243 个 Python 文件）、原启动器、两个原配置及锁定的 post8 子镜像，再执行
未修改的原启动器。唯一新增的 Docker 参数是初始环境
`TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=1`。原 `launch.json` 不改写，
`effective_launch.json` 单独记录生效命令和实际 Docker 退出码。

公开适配器完成本地命令构造检查后，已按发布文件的相同 SHA256 在 NPU 实际
执行。重复对照包含 VAD 和 Hydra 各两对 continuous8/own4/resume8 流程，四对均
完成，其中两对通过原数值门槛、两对超出，门槛没有调整。因此，适配器已完成
设备执行验证，同步设置的普遍确定性问题仍未解决。

[公开适配器重复对照](../results/public-sync-repeated-16chip-20261014.json)
保留了四对完整比较结果。

此前独立单次正例继续保留：VAD 全部逐位相同，Hydra 最大差
`2.9802322387695312e-8`，但并非全部逐位相同。应同时查看这些历史正例和本次
重复对照的通过、失败结果。

以下示例复现已在设备执行的公开适配器流程。将路径替换为服务器上的真实资产，
保存通过和失败的比较结果，继续使用原数值门槛。`MINI` 为规划
对照使用的 NAVTEST 派生真实子集，42 个训练锚点、22 个独立评价锚点，不是单场景
合成导出。初始化权重使用官方 track/map 100pct，词表使用原 8192 版本。示例本地
镜像标签不代表公开镜像仓地址。执行需要 16 个空闲逻辑芯片；原启动器检查字符
设备节点，不能据此认定驱动空闲或能初始化。

不加 `--execute` 时只生成新方案并执行原只读镜像、设备和路径检查，不启动
Docker 训练。方案也会占用其输出目录，试运行与正式运行须使用不同的新目录。
下例明确执行前台训练；如只审阅方案，移除 `--execute`，另选新输出目录。

版本化设备实测结果：[VAD](../results/sync-vad-16chip-20261014.json)、[Hydra](../results/sync-hydra-16chip-20261014.json)。

## 连续训练与自身断点重放

以下代码块在同一个 Bash 会话执行。

```bash
set -euo pipefail

SOURCE=/path/WorldEngine
IMAGE=local/worldengine:26.1.0-a3-post8-candidate-20261014
IMAGE_ID=sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45
MINI=/path/data/mini_real_data
MAPS=/path/assets/maps
INIT=/path/checkpoints/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth
VOCAB=/path/assets/test_8192_kmeans.npy
RUNS=/path/runs/synchronous_$(date -u +%Y%m%dT%H%M%SZ)
CONFIG="$SOURCE/projects/AlgEngine/configs/navformer/e2e_vadv2.py"
COLD="$RUNS/vad_continuous"
RESUMED="$RUNS/vad_resumed"

test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$IMAGE_ID"
printf '%s  %s\n' \
  9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a "$INIT" \
  cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35 "$VOCAB" \
  | sha256sum --check

SYNC_ARGS=(
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --profile imitation --task train --config "$CONFIG"
  --checkpoint "$INIT" --vocab "$VOCAB" --mini-prepared "$MINI" --maps "$MAPS"
  --runtime --steps 8 --checkpoint-interval 4 --check-interval 1
  --batch-size 1 --workers 0 --deterministic --precision fp32
  --memory 256g --cpus 128 --pids-limit 8192
  --image "$IMAGE" --expected-image-id "$IMAGE_ID"
)

python3 "$SOURCE/ascend/execution/synchronous_launcher.py" \
  --source "$SOURCE" --execute -- "${SYNC_ARGS[@]}" --output "$COLD"
test -f "$COLD/iter_4.pth"
test -f "$COLD/iter_8.pth"

python3 "$SOURCE/ascend/execution/synchronous_launcher.py" \
  --source "$SOURCE" --execute -- "${SYNC_ARGS[@]}" \
  --output "$RESUMED" --resume "$COLD/iter_4.pth"
test -f "$RESUMED/iter_8.pth"
```

Hydra 复跑改用 `e2e_hydramdp.py` 及新的 `hydra_continuous`、`hydra_resumed`
目录，其余设置不变。适配器检查 `iter_4.pth` 文件名；调用方须传入该次连续运行
自身的文件，完整恢复兼容性及状态载入由未修改的原训练器检查。

## 原数值比较器的 CPU 执行

CPU 比较使用新输出目录、相同锁定 post8 镜像，不挂 NPU 设备，源码及运行目录只读。
比较目标保持原 `ascend/compare_resume.py`，SHA 为上述 `9f2cb1…`。此前实测使用
单独锁定的官方加载器审计包装 `2eb274e…`。下面内联命令使用相同的已安装
官方 `torch_npu` 注册机制，与锁定的审计脚本并非逐字节相同，也没有单独的执行验收记录。

```bash
CHECKS="$RUNS/cpu_compare"
mkdir "$CHECKS"
docker run --rm --network none --memory 32g --cpus 8 --pids-limit 512 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$SOURCE:/workspace/WorldEngine:ro" -v "$RUNS:/runs:ro" \
  -v "$CHECKS:/output:rw" -w /workspace/WorldEngine "$IMAGE" \
  /opt/conda/envs/bevformer/bin/python -c '
import hashlib, runpy, sys
from pathlib import Path
import torch, torch_npu
from torch_npu.utils import serialization
target = Path("ascend/compare_resume.py")
expected = "9f2cb1cd4a2638cba9a4be24d8a56242909c638ca0bde616445435982ec35330"
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
assert torch_npu.__version__ == "2.7.1.post8"
assert torch_npu.version.git_version == "a783f11e456f27d7451a1ea745f1556823357c95"
assert torch.load is serialization.load
assert not torch.npu.is_initialized()
sys.argv = [str(target)] + sys.argv[1:]
try:
    runpy.run_path(str(target), run_name="__main__")
finally:
    assert not torch.npu.is_initialized()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
' \
  --continuous /runs/vad_continuous/iter_8.pth \
  --resumed /runs/vad_resumed/iter_8.pth \
  --continuous-run /runs/vad_continuous --partial-run /runs/vad_continuous \
  --resumed-run /runs/vad_resumed --output /output/comparison.json
```

Hydra 比较替换全部 VAD 目录名。以连续运行自身作为保存前缀的控制，不另做独立
初始化。保留实际退出码和比较 JSON，容差保持不变。完整比较中 Tensor 超容差属于数值不通过；调用中断且没有完整比较时，不作数值判定。只有实际比较器在原
`atol=1e-6, rtol=1e-5`、loss `1e-4` 下得到 `result="pass"`，并同时核对实际
16 份 rank 运行及恢复结果，才支持该次运行的有限范围结论；生成方案或 Docker 退出 0
本身不构成模型验收。
