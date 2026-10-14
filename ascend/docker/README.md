# AlgEngine NPU environment

[中文](README_zh.md) · [Usage guide](../../docs/algengine_ascend.md)

Run these commands on the Linux ARM64 Ascend server, from the repository root.
The host needs a working Ascend driver, CANN-compatible Docker access, Python3
and pip. Check `uname -m`, `npu-smi info` and `docker info` before starting.
The measured stack used CANN9.1/driver26.1.1 and Python3.10 inside the image.
Do not install the upstream CUDA requirements over this environment.

## Build from the public base

The base is the pinned public DrivingSDK ARM64 manifest. The existing
`Dockerfile` constructs the post6 parent; the new `Dockerfile.post8` installs
the exact official wheel used by the post8 experiments in the bevformer prefix.
No GPU toolkit or SimEngine environment is needed for this AlgEngine workflow.

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

Save these two values with the run. The launcher accepts this rebuilt image
using explicit `--image "$IMAGE" --expected-image-id "$IMAGE_ID"`. The build
does not reproduce the previous Docker image ID: Conda/channel resolution and
layer timestamps can differ. Start with the single-chip real-data check in the
usage guide and inspect its actual package/precision reports before expanding.
The new public child Dockerfile has been reviewed locally; this exact public
build command has not been run on a server in this delivery.

For offline transfer, run `docker save -o worldengine_npu.tar "$IMAGE"`, record
`sha256sum worldengine_npu.tar`, then `docker load -i worldengine_npu.tar` on the
target and compare the original `IMAGE_ID`. The old measured post6 archive
was23.44GB; that size is not the new archive's measured size.

## Verify installed versions without training

```bash
docker run --rm --network none -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$IMAGE" \
  /opt/conda/envs/bevformer/bin/python -c '
from importlib.metadata import version
for package in ("torch", "torch-npu", "mmcv-full", "mmdet", "mmdet3d", "numpy"):
    print(package, version(package))
assert version("torch-npu") == "2.7.1.post8"
'
```

The package snapshot for the original parent is in
[`versions-20261013.json`](versions-20261013.json). The original image's
`pip check` contains16 known dependency findings; a successful import or build
does not mean these are resolved. Use the NPU training check to establish the
behavior of the newly built image. Host drivers are not installed by Docker.

## Reuse an existing measured image

If the measured image is already installed or supplied as a verified archive,
use its actual tag and expected ID directly. `local/...` tags are local names,
not public download URLs. The recorded images are:

| Image | Recorded ID | Use |
| --- | --- | --- |
| Original post6 clean | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` | Historical measurements; [runtime lock](../runtime.lock.json) |
| Post8 candidate | `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45` | Two single positive16-chip replay pairs |

The restricted [synchronous replay adapter](../execution/README.md) requires
the recorded post8 ID and original runtime fingerprint. It cannot accept a
different self-built image by substituting its ID. The general launcher uses
the explicit self-built ID; its new results must be retained separately.
