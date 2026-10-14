# 规划基座同步候选的复现操作

[English](README.md) · [AlgEngine操作文档](../README_zh.md)

适配器SHA256：`b6c5d067cc7ba54efa4650e8278c721d6d4187a65bc91870a8b4b1fca4405b5f`。

可选的 `ascend/execution/synchronous_launcher.py` 只覆盖原VAD/Hydra基座配置、
16个逻辑芯片、B1、workers0、FP32、8次更新及每4次保存断点。它校验cc91运行
指纹（243个Python文件）、原启动器、两个原配置及锁定的post8子镜像，再执行
未修改的原启动器。唯一新增的Docker参数是初始环境
`TASK_QUEUE_ENABLE=0`、`ASCEND_LAUNCH_BLOCKING=1`。原 `launch.json` 不改写，
`effective_launch.json` 单独记录生效命令和实际Docker退出码。

公开适配器已在本机调用真实原启动器，通过命令构造检查；测试中的镜像查询、
设备节点为明确的stub。**该公开适配器本身还没有在NPU执行。** 此前私有控制器
使用相同原训练入口和这两个初始环境设置，VAD、Hydra各一对自身断点重放通过
原数值门槛：VAD全部逐位相同；Hydra最大差 `2.9802322387695312e-8`，并非全部
逐位相同。每项仅一次重复，不能写成普遍确定性修复。

以下为未执行的复跑示例。将路径替换为服务器上已有的真实资产。`MINI` 为规划
对照使用的NAVTEST派生真实子集，42个训练锚点、22个独立评价锚点，不是单场景
合成导出。初始化权重使用官方track/map 100pct，词表使用原8192版本。示例本地
镜像标签不代表公开镜像仓地址。运行前须安排16个芯片可用；原启动器检查字符
设备节点，不能据此认定驱动空闲或能初始化。

不加 `--execute` 时只生成新方案并执行原只读镜像、设备和路径检查，不启动
Docker训练。方案也会占用其输出目录，试运行与正式运行须使用不同的新目录。
下例明确执行前台训练；如只审阅方案，移除 `--execute`，另选新输出目录。

私有控制器实测结果：[VAD](../results/sync-vad-16chip-20261014.json)、[Hydra](../results/sync-hydra-16chip-20261014.json)。

## 连续训练与自身断点重放

以下代码块在同一个Bash会话执行。

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

Hydra复跑改用 `e2e_hydramdp.py` 及新的 `hydra_continuous`、`hydra_resumed`
目录，其余设置不变。适配器检查 `iter_4.pth` 文件名；调用方须传入本次连续运行
自身的文件，完整恢复兼容性及状态载入由未修改的原训练器检查。

## 原数值比较器的CPU执行

CPU比较使用新输出目录、相同锁定post8镜像，不挂NPU设备，源码及运行目录只读。
比较目标保持原 `ascend/compare_resume.py`，SHA为上述 `9f2cb1…`。此前实测使用
单独锁定的私有官方加载器审计包装 `2eb274e…`。下面内联命令使用相同的已安装
官方 `torch_npu` 注册机制，不声称它与私有审计脚本逐字节相同，也不是新实测。

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

Hydra比较替换全部VAD目录名。以连续运行自身作为保存前缀的控制，不另做独立
初始化。非零退出及失败JSON应保留，不能放宽门槛。只有实际比较器在原
`atol=1e-6, rtol=1e-5`、loss `1e-4` 下得到 `result="pass"`，并同时核对实际
16份rank运行及恢复结果，才支持本次有限范围的结论；生成方案或Docker退出0
本身不构成模型验收。
