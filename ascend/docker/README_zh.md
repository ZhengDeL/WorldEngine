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
docker build --network host --no-cache \
  --build-arg HTTP_PROXY="${http_proxy:-}" \
  --build-arg HTTPS_PROXY="${https_proxy:-}" \
  --build-arg BASE_IMAGE="$BASE" \
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
核对实际版本、精度和退出报告，再扩大设备规模。本次从 GitHub 新克隆的
`ec153c12559a63a446f4913c060198ab050b8caf` 提交重建父、子镜像，三个配方文件
保持原字节。新子镜像
`sha256:1541450982f8c2938f3c387b929ad0ba8a90024d73712b3eabe49631812d12ce`
用于两个原真实日志微调配置的 16 NPU 训练、自身断点恢复、原 CPU 严格比较、
独立推理和 CPU 评分。公开流程的 14 个阶段均 exit 0，其中前四个输入和数据准备
CPU 阶段使用已核实的历史 post8 镜像，后十个训练与评价阶段使用本次新镜像。
64 份 rank 运行记录和 32 份 rank 恢复记录均通过。VAD 的 989 个模型 Tensor 和
1,321 个完整对照 Tensor 全部逐位相同；Hydra 对应为 989 和 1,261 个，完整对照
包含各 rank 的 RNG Tensor。Tensor 和 loss 最大差均为 0，继续使用原门槛
（`atol=1e-6`、`rtol=1e-5`、loss `1e-4`）及默认 queue/blocking 环境。
两项独立推理和官方 CPU 评分分别覆盖 22 个独立日志锚点。

这些结果证明两个实测微调路径的小数据、短训练功能。完整配置验收、收敛、GPU
对照、普遍确定性修复及新镜像持续训练仍需分别验证。
[公开复现证据](../results/actual-public-reproduction-16chip-20261014.json)
记录了源码、镜像、数据及各阶段结果。

离线转移时执行 `docker save -o worldengine_npu.tar "$IMAGE"`，保存
`sha256sum worldengine_npu.tar`。目标机执行 `docker load -i worldengine_npu.tar`
后核对原 `IMAGE_ID`。旧 post6 归档为 23.44GB；本次新镜像包实测为
23,593,312,256 字节，SHA256 为
`bbf56ecafda424d37dc89ab47f8f9d975be76c316b953648b5919429036bc71d`。

## 构建网络与官方基座缓存

本次父镜像在默认桥接网络下无法访问局域网代理，实际报出 `Network is
unreachable`。重试时为父镜像构建增加 `--network host`；需要代理时，设置本站的
`http_proxy`/`https_proxy`，上面的命令通过构建参数传入，不修改 daemon 全局代理。
子镜像保持 `--network none`。构建的 host 网络选项不会配置 daemon 的仓库网络。

实测宿主机还缓存了同一官方基座镜像，虽然其 `RepoDigests` 列表为空，但固定
manifest 的实际查询结果证明 config digest 为
`sha256:1a63c030ea4036467cfc486bef006acf6f1cfbb1cc3b0dc362d07f86b84349bd`，
与本地镜像 ID 和 Linux/ARM64 平台一致。本次将该已核实的缓存作为 `BASE_IMAGE`，
没有换用旧 AlgEngine 应用镜像。采用这一方式时，应先查询官方固定清单并核对
config ID 和本地平台，再用缓存 ID 设置 `BASE`、替代首次拉取，保存校验记录。
仅有相似的本地标签不足以证明基座相同。

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

原父镜像版本记录见 [`versions-20261013.json`](versions-20261013.json)。历史镜像与本次新镜像的
`pip check` 均报告 16 项问题，新构建仍返回 exit 1。原始输出保留，包含缺失依赖
声明、MMCV 要求 `protobuf==4.25.8` 而实际为 `protobuf 7.36.2` 的版本冲突，以及
`te` 平台提示。本次 VAD/Hydra 路径通过，不表示
这些问题已解决或所有包和配置均已验证。新增配置应分别运行检查。Docker配方不会安装宿主机驱动。

## 使用已安装的实测镜像

如已有实测镜像或经过校验的镜像归档，可以直接使用对应标签和ID。
`local/...`是本地标签，不是公开下载地址。

| 镜像 | 记录的ID | 用途 |
| --- | --- | --- |
| 原post6 clean | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` | 历史实测；[原环境锁](../runtime.lock.json) |
| post8 候选 | `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45` | 历史单次正例；公开适配器重复对照完成四对，其中两对在原门槛内，两对超出 |
| 公开配方重建 | `sha256:1541450982f8c2938f3c387b929ad0ba8a90024d73712b3eabe49631812d12ce` | 已完成 14 阶段公开流程中的十个训练与评价阶段；使用独立镜像和数据记录 |

受限的[同步重放辅助器](../execution/README_zh.md)要求记录的post8 ID和源码指纹，
不能换成自建ID后冒用既有结论。通用启动器可显式使用自建ID，新运行记录单独保存。
