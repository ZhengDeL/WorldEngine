# AlgEngine 应用与编译工具环境

本目录用一个 Dockerfile 依次构建运行依赖修复层和独立 CANN 工具环境。基础镜像由公开的 `ascend/docker/Dockerfile`、`Dockerfile.post8` 构建，不依赖临时中间镜像、服务器私有目录或凭据。

该合并 Dockerfile 已于2026-10-16直接继承公开 post8 配方产物构建成功。模型与工具两个真实解释器分别通过原样 `pip check` 和 CPU API 验收；第三项在重新 sourcing 原 CANN `set_env.sh`、导入实际框架后，路径与编译解释器检查也通过。实测镜像为 `sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`。构建逐文件复用完整哈希验证的 wheel，没有继承私有修复中间镜像。实际范围和输入哈希见[构建验证记录](BUILD_VALIDATION.json)。

这是 CPU 环境验收；新镜像的冷缓存 NPU 训练、恢复和独立评分仍需各自实测。依赖报告通过不代表分布式数值恢复问题已经解决。

## 环境划分

模型应用保留 CPython3.10、Torch2.7.1、TorchNPU2.7.1.post8、MMCV1.7.2，并使用 protobuf4.25.8。独立编译与分析工具使用系统 CPython3.11 和 protobuf5.29.5，暴露九个原 SDK 组件的真实文件及未修改的元数据。两套解释器分别执行原样 `pip check` 和实际 API 检查；其他可选 SDK 工具不在本次组件范围内。

原 headless OpenCV 的文件与元数据保留，完整 OpenCV 真 wheel 安装在独立前缀。这是为了满足实际 SDK 同时声明两个发行包的要求。检查覆盖两发行包的真实 RECORD、不同前缀、实际加载来源和 JPEG/resize 功能，不通过删除元数据消除提示。

基镜像有两处原 RECORD 路径缺失：TE 许可和 Profiler 命令行脚本。对应原始字节已经在 SDK 其他目录中找到。精确重定位锁只接受这两条声明、位置及哈希；原缺失位置和原元数据保持，其他缺项直接失败。新工具入口用真实 Python3.11 执行原 Profiler 脚本，不修改其 shebang。

## 服务器下载与构建

在 Linux ARM64 服务器执行。先按 `ascend/docker/README_zh.md` 构建公开 post8 基础镜像，再将其真实标签或 ID 传给 `BASE_IMAGE`。独立重建的镜像 ID 可以不同，包版本、原生文件及 SDK 内容门禁仍须通过。

60个唯一官方 wheel 共151,981,835字节。下载器默认只显示计划，加 `--execute` 才下载；代理读取站点的标准环境变量，失败后尝试直连，不关闭 TLS 校验。

公开下载器已于2026-10-16在服务器全新目录独立实测：60个官方 wheel 全部下载完成，完整大小和锁定哈希均通过。这次下载在镜像构建之后进行；前面的实测构建使用已有且完整哈希校验的缓存，两项证据分别记录，见 `BUILD_VALIDATION.json`。

```bash
set -euo pipefail
REPAIR="$PWD/ascend/docker/dependency_repair"
BASE_IMAGE=local/worldengine:algengine-post8-user
IMAGE=local/worldengine:algengine-dependencies-user
DEADLINE=$(date -u -d '+45 minutes' '+%Y-%m-%dT%H:%M:%S+00:00')
LOCK_SHA=$(sha256sum "$REPAIR/WHEELS.lock.json" | cut -d ' ' -f 1)
python3 "$REPAIR/download_wheels.py" --output "$REPAIR/wheelhouse" \
  --expected-lock-sha256 "$LOCK_SHA" --deadline "$DEADLINE"
python3 "$REPAIR/download_wheels.py" --output "$REPAIR/wheelhouse" \
  --expected-lock-sha256 "$LOCK_SHA" --deadline "$DEADLINE" --execute
docker build --network host --no-cache \
  --build-arg HTTP_PROXY="${http_proxy:-}" \
  --build-arg HTTPS_PROXY="${https_proxy:-}" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f "$REPAIR/Dockerfile" -t "$IMAGE" "$REPAIR"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
export IMAGE IMAGE_ID
```

可重复添加 `--reuse-wheelhouse /path/to/verified/cache`，逐文件完整校验后复制已有缓存，原文件前后哈希必须相同；未知缓存不覆盖。每次使用新的输出目录，失败保留原收据和部分文件。

OS 包管理器可能需要站点代理，因此构建使用 host 网络，并以构建参数传递代理。它不修改 Docker daemon 的注册表路由。wheel 离线按完整哈希安装，不自动解析新依赖；OS 包仓库和原公开 Conda 配方的解析仍可能变化，不能承诺镜像字节一致。

## CPU 验收

以下容器不挂任何 NPU 设备。使用新镜像的真实 ID，保留两个原始 pip 输出和三份 JSON，三条命令都必须返回0。

```bash
CHECKS=$(mktemp -d "$PWD/worldengine-environment-checks.XXXXXX")
docker run --rm --runtime runc --network none --memory 16g --cpus 4 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$CHECKS:/output:rw" "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python \
  /opt/worldengine-environments/application_CPU_acceptance.py \
  --output /output/application.json
docker run --rm --runtime runc --network none --memory 16g --cpus 4 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$CHECKS:/output:rw" "$IMAGE_ID" \
  /opt/worldengine-environments/compiler_python311.sh \
  /opt/worldengine-environments/sdk_tools_CPU_acceptance.py \
  --output /output/SDK_tools.json
docker run --rm --runtime runc --network none --memory 16g --cpus 4 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$CHECKS:/output:rw" "$IMAGE_ID" bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec /opt/conda/envs/bevformer/bin/python \
  /opt/worldengine-environments/verify_launch_environment.py \
  --output /output/original_CANN_source_environment.json
'
```

应用解释器每次启动时，`.pth` 模块清除其 CANN Python 搜索目录；原启动器再次 `source set_env.sh` 后启动该解释器，也执行同一检查。启动模块不预导入 Torch、不调用设备。`HI_PYTHON` 选择独立工具解释器。第三个命令实际核对 CANN sourcing 和框架导入后的解释器选择、路径及 SDK 元数据可见性；它仍是 CPU 检查。

应用检查包括 OpenCV、protobuf4 和三个地理库的四个 nuPlan 城市 UTM 往返、独立 WebMercator 公式。`PROJ_DATA`、`PROJ_LIB` 显式选择匹配的 Conda 数据库。工具检查包括 TE、SymPy/mpmath 真实符号运算、protobuf5 和原 compiler/Profiler 帮助入口；帮助命令成功不等于完成内核编译。

## NPU 后续验收

CPU 全部通过后，使用公开 AlgEngine 启动入口，显式传 `--image "$IMAGE" --expected-image-id "$IMAGE_ID"`，使用新的输出目录。启动器为该任务新建 `ASCEND_CACHE_PATH`、`ASCEND_WORK_PATH`；它不清理共享宿主或驱动缓存。模型、损失、精度设置和原比较阈值保持，分别检验真实 forward/backward、恢复、独立推理与 CPU 评分。CPU 结果不能证明所有 CANN 编译调用均走 `HI_PYTHON`，也不代表16-rank严格回放、完整 NAVTRAIN 收敛、GPU 等价或全部可选 SDK 工具已经通过。
