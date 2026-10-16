# AlgEngine application and compiler environments

This directory builds the runtime dependency repair and the separate CANN tool
environment in one image. `BASE_IMAGE` is an image built with the public
`ascend/docker/Dockerfile` and `Dockerfile.post8` recipes. No private intermediate
image, server path or credential is required.

The combined Dockerfile was built on 2026-10-16 from the public-recipe post8
image. Both real interpreters passed their original `pip check` and CPU API
gates. The third check passed after sourcing CANN's original `set_env.sh` and
importing the actual application framework. The measured new image was
`sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48`.
The build reused complete hash-verified wheels; it did not use a private repair
image as its parent. [CPU build validation](BUILD_VALIDATION.json) records the
actual scope and input hashes.

These are CPU environment results. Cold-cache NPU model training, restoration
and independent scoring in this new image require separate evidence. A clean
dependency report does not resolve the distributed numerical replay issue.

## Layout and requirements

Use a Linux ARM64 server with the documented DrivingSDK/CANN9.1 base, a CPU-only
Docker build runtime, and enough space for a fresh image and build layers. Build
the public post8 parent using `ascend/docker/README.md`, then set `BASE_IMAGE` to
that actual tag or ID. An independently rebuilt base may have a different image
ID. Package, native-file and SDK-content gates must still pass.

The model application retains CPython3.10, Torch2.7.1, TorchNPU2.7.1.post8,
MMCV1.7.2 and protobuf4.25.8. The compiler/profiler uses a real separate
CPython3.11 environment and protobuf5.29.5. Nine original SDK distributions and
their metadata remain unchanged. Other optional SDK tools are outside this
component set. Neither environment is accepted by filtering `pip check` output.

The original headless OpenCV distribution stays installed. The genuine full
OpenCV wheel is installed in a distinct prefix because the actual SDK declares
both distributions. Checks verify both original RECORD payloads, distinct
prefixes, the active full-OpenCV source and real JPEG/resize operations.

The SDK baseline has two absent RECORD locations: TE's license and the Profiler
console script. The exact publisher bytes already exist at other SDK locations.
The relocation lock allows only those exact hashes and paths; it leaves the
original missing locations and metadata untouched. All other missing payloads
fail. The independent profiler entrypoint runs the original script with the
real CPython3.11 interpreter.

## Download and build

Run binary downloads on the deployment server. The 60 unique official wheels
total 151,981,835 bytes. Proxy settings come from the site's normal environment;
the downloader falls back to a direct HTTPS route and never disables TLS checks.
The default invocation prints a plan. `--execute` is required to download.

The public downloader was separately tested in a fresh server directory on
2026-10-16: all 60 official wheels, 151,981,835 bytes, passed their complete
locked hashes. This download followed the measured image build, which used
verified existing caches; see `BUILD_VALIDATION.json`.

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

`--reuse-wheelhouse /path/to/verified/cache` may be repeated on the download
command. Each reused file is checked against the complete original size and
SHA before and after copying. Unknown cache files remain untouched. Use a fresh
output directory for each attempt. Retain its receipt and partial files on failure.

The build may need the configured LAN proxy for the OS package manager. Host
networking applies only to the build; it does not change the Docker daemon's
registry route. All wheel installations use local hash-locked files without
dependency resolution. OS package-provider resolution and the original public
Conda build do not guarantee bit-identical images.

## CPU acceptance

These checks expose no NPU devices. Use the actual new image ID. Keep both raw
pip outputs and every JSON report; all three commands must exit0.

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

The application `.pth` startup module removes CANN Python package directories
each time the actual CPython3.10 model interpreter starts, including after the
launcher sources CANN's original `set_env.sh`. It does not import Torch or call
a device. `HI_PYTHON` selects the separate compiler interpreter. The third check
verifies the actual selection and paths after the normal CANN sourcing sequence
and original framework imports. The original SDK files remain installed.

Application checks include real OpenCV/protobuf operations and pyproj, Rasterio
and Fiona CRS transforms for four nuPlan cities, plus an independent WebMercator
formula. The matching Conda PROJ database is selected explicitly with
`PROJ_DATA` and `PROJ_LIB`. Tool checks include TE, its runtime SymPy/mpmath
dependencies, real symbolic operations, protobuf5 and the original compiler and
Profiler `--help` commands. A successful help command is not kernel compilation.

## NPU validation

After CPU acceptance, run the normal public AlgEngine launcher with explicit
`--image "$IMAGE" --expected-image-id "$IMAGE_ID"` and a fresh kernel-cache
directory. Preserve the original model, loss, precision and comparison thresholds.
The launcher's fresh output supplies new `ASCEND_CACHE_PATH` and
`ASCEND_WORK_PATH` directories; it does not reset shared host or driver caches.
Verify real forward/backward, checkpoint restoration, independent inference and
CPU scoring. A CPU pass does not establish that every CANN runtime compilation
path uses `HI_PYTHON`, strict16-rank numerical replay, full NAVTRAIN convergence,
GPU equivalence or all optional SDK tools. Record each result independently.
