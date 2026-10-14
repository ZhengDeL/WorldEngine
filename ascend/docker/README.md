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

Save these two values with the run. The launcher accepts this rebuilt image
using explicit `--image "$IMAGE" --expected-image-id "$IMAGE_ID"`. The build
does not reproduce the previous Docker image ID: Conda/channel resolution and
layer timestamps can differ. Start with the single-chip real-data check in the
usage guide and inspect its actual package/precision reports before expanding.
The public parent and post8 child recipes were rebuilt from a fresh GitHub
checkout at `ec153c12559a63a446f4913c060198ab050b8caf`, with all three recipe
files unchanged. The new child image
`sha256:1541450982f8c2938f3c387b929ad0ba8a90024d73712b3eabe49631812d12ce`
ran the two original real-log fine-tuning configs on 16 logical NPUs, their own
checkpoint restorations, original strict CPU comparisons, independent inference
and CPU scoring. All 14 workflow stages exited 0; the first four CPU input and
preparation stages used the previously verified post8 image, while the ten
training and evaluation stages used this rebuild. The 64 rank runtime records
and 32 rank restoration records passed. The VAD comparison was bitwise equal
for all 989 model tensors and 1,321 complete comparison tensors; Hydra was equal
for all 989 and 1,261 respectively, including the rank RNG tensors in each
complete comparison. Tensor and loss differences were 0. Both comparisons used
the original tolerances (`atol=1e-6`, `rtol=1e-5`, loss `1e-4`) and the default
queue/blocking environment. Each independent inference and official CPU score
covered 22 held-out anchors.

These are short, small-data functional results for the two measured fine-tuning
paths. They do not establish all-configuration acceptance, full convergence,
GPU equivalence, a general determinism repair or sustained training in the
rebuilt image. [Public reproduction evidence](../results/actual-public-reproduction-16chip-20261014.json)
records the source, images, data and individual stage results.

For offline transfer, run `docker save -o worldengine_npu.tar "$IMAGE"`, record
`sha256sum worldengine_npu.tar`, then `docker load -i worldengine_npu.tar` on the
target and compare the original `IMAGE_ID`. The old measured post6 archive
was23.44GB. The newly built image archive was 23,593,312,256 bytes, with
SHA256 `bbf56ecafda424d37dc89ab47f8f9d975be76c316b953648b5919429036bc71d`.

## Build networking and verified official cache

The measured parent build needed `--network host` to reach the configured LAN
proxy during `RUN` steps. The default bridge had returned `Network is
unreachable`. Set the site's `http_proxy`/`https_proxy` when needed; the commands
above pass them as build arguments without changing the daemon globally. The
child build retains `--network none`. Host build networking does not configure
the daemon's registry route.

The measured host also had the exact official base cached by image ID, despite
an empty `RepoDigests` list. A query of the pinned manifest proved its config
digest to be
`sha256:1a63c030ea4036467cfc486bef006acf6f1cfbb1cc3b0dc362d07f86b84349bd`,
which matched the actual local image ID and Linux/ARM64 platform. That proved
cache was used as `BASE_IMAGE`; it was not a previous AlgEngine application
image. If using this route, replace the initial pull with a registry manifest
query and exact config-ID/platform checks before setting `BASE` to the cached
ID. Preserve the proof. An unrelated local tag or image is insufficient.

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
[`versions-20261013.json`](versions-20261013.json). Both the historical image and this rebuilt image reported 16 `pip check`
findings; the rebuild returned exit 1. The original output is retained, including
missing requirements, MMCV's `protobuf==4.25.8` requirement against the installed
`protobuf 7.36.2`, and the `te` platform finding. Successful measured VAD/Hydra paths do not resolve those findings or
certify every package/configuration. Use a separate run to verify any additional
configuration. Host drivers are not installed by Docker.

## Reuse an existing measured image

If the measured image is already installed or supplied as a verified archive,
use its actual tag and expected ID directly. `local/...` tags are local names,
not public download URLs. The recorded images are:

| Image | Recorded ID | Use |
| --- | --- | --- |
| Original post6 clean | `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f` | Historical measurements; [runtime lock](../runtime.lock.json) |
| Post8 candidate | `sha256:3fef77b906c6b1e5cdf68c08d43135dc7b0509fb938a4665bb0d478bd8b64e45` | Historical single positives; the public adapter repeat control completed four pairs, two within and two outside the original tolerance |
| Public recipe rebuild | `sha256:1541450982f8c2938f3c387b929ad0ba8a90024d73712b3eabe49631812d12ce` | Ten training/evaluation stages in the completed 14-stage public workflow; separate image and data records |

The restricted [synchronous replay adapter](../execution/README.md) requires
the recorded post8 ID and original runtime fingerprint. It cannot accept a
different self-built image by substituting its ID. The general launcher uses
the explicit self-built ID; its new results must be retained separately.
