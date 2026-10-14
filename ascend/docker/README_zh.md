# AlgEngine NPU 环境安装

[English](README.md) · [使用指南](../../docs/algengine_ascend_zh.md)

以下命令在Linux ARM64昇腾服务器执行，工作目录为仓库根目录。宿主机需已有
可用的昇腾驱动、Docker、Python3及pip。先检查 `uname -m`、`npu-smi info` 和
`docker info`。实测栈使用CANN9.1、driver26.1.1，镜像内为Python3.10。
此环境用于AlgEngine，不需要安装SimEngine环境或原仓CUDA依赖。

## 从公开基础镜像构建

基础镜像固定到DrivingSDK的公开ARM64 manifest。原 `Dockerfile` 构建post6
父镜像，新 `Dockerfile.post8` 在bevformer环境安装实测所用的官方post8 wheel。

```bash
set -euo pipefail
test "$(uname -m)" = aarch64
BASE='swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801'
PARENT=local/worldengine:algengine-post6-user
IMAGE=local/worldengine:algengine-post8-user

docker pull --platform linux/arm64 "$BASE"
docker build --no-cache --build-arg BASE_IMAGE="$BASE" \
  -f ascend/docker/Dockerfile -t "$PARENT" ascend/docker

python3 -m pip download --no-deps --only-binary=:all: \
  --implementation cp --python-version 3.10 --abi cp310 \
  --platform manylinux_2_28_aarch64 --index-url https://pypi.org/simple \
  --dest ascend/docker torch-npu==2.7.1.post8
printf '%s  %s\n' \
  3b65991505f215af76ca84ca05da7f64e942eedb0f6fd638f59a0976f6685f68 \
  ascend/docker/torch_npu-2.7.1.post8-cp310-cp310-manylinux_2_28_aarch64.whl \
  | sha256sum --check

docker build --pull=false --network none --no-cache \
  --build-arg BASE_IMAGE="$PARENT" -f ascend/docker/Dockerfile.post8 \
  -t "$IMAGE" ascend/docker
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
export IMAGE IMAGE_ID
printf 'IMAGE=%s\nIMAGE_ID=%s\n' "$IMAGE" "$IMAGE_ID"
```

保存 `IMAGE` 和 `IMAGE_ID`，运行时显式传入
`--image "$IMAGE" --expected-image-id "$IMAGE_ID"`。重建的ID可能与实测镜像不同，
Conda依赖解析和镜像层时间会影响ID。先按使用指南运行单芯片真实数据检查，
核对实际版本、精度和退出报告，再扩大设备规模。公开 post8 子镜像配方已完成
本地审阅，完整干净服务器构建与实机验证尚未执行。

离线转移时执行 `docker save -o worldengine_npu.tar "$IMAGE"`，保存
`sha256sum worldengine_npu.tar`。目标机执行 `docker load -i worldengine_npu.tar`
后核对原 `IMAGE_ID`。旧post6归档实测为23.44GB，不代表新归档的大小。

## 检查安装版本

```bash
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$IMAGE" \
  /opt/conda/envs/bevformer/bin/python -c '
from importlib.metadata import version
for package in ("torch", "torch-npu", "mmcv-full", "mmdet", "mmdet3d", "numpy"):
    print(package, version(package))
assert version("torch-npu") == "2.7.1.post8"
'
```

原父镜像版本记录见 [`versions-20261013.json`](versions-20261013.json)。原镜像
`pip check`有16项依赖发现；导入或构建成功不表示这些项目已解决。新镜像的实际
行为需要单芯片训练检查。Docker配方不会安装宿主机驱动。

## 使用已安装的实测镜像

如已有实测镜像或经过校验的镜像归档，可以直接使用对应标签和ID。
`local/...`是本地标签，不是公开下载地址。

| 镜像 | 记录的ID | 用途 |
| --- | --- | --- |
| 原post6 clean | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` | 历史实测；[原环境锁](../runtime.lock.json) |
| post8候选 | `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45` | 两个16芯片单次严格续训正例 |

受限的[同步重放辅助器](../execution/README_zh.md)要求记录的post8 ID和源码指纹，
不能换成自建ID后冒用既有结论。通用启动器可显式使用自建ID，新运行记录单独保存。
