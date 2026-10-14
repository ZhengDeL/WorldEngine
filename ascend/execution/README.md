# Synchronous planner replay candidate

[中文](README_zh.md) · [AlgEngine operation guide](../README.md)

Adapter SHA256: `b6c5d067cc7ba54efa4650e8278c721d6d4187a65bc91870a8b4b1fca4405b5f`.

The optional `ascend/execution/synchronous_launcher.py` adapter is restricted to
the original VAD/Hydra base-planner configurations, 16 logical chips, B1,
workers0, FP32, eight updates and checkpoint interval4. It verifies the cc91
runtime fingerprint (243 Python files), the original launcher, both original
configuration hashes and the locked post8 child image. It invokes the unchanged
launcher and adds only `TASK_QUEUE_ENABLE=0` and `ASCEND_LAUNCH_BLOCKING=1` to
the initial Docker command. The original `launch.json` remains unchanged;
`effective_launch.json` records the effective command and actual Docker exit.

The public adapter passed local command-construction checks and was then
executed on the NPU with its exact published SHA256. The repeated device control
ran two VAD and two Hydra continuous8/own4/resume8 pairs. All four pairs completed;
two passed and two exceeded the unchanged original numerical tolerance. The
adapter is therefore device-executed, but synchronous settings do not establish
a general determinism repair.

[Repeated public-adapter results](../results/public-sync-repeated-16chip-20261014.json)
retain all four completed comparisons.

Earlier independent single pairs remain valid historical results: VAD was fully
bitwise equal, while Hydra passed with maximum difference
`2.9802322387695312e-8` and was not fully bitwise equal. Those single positives
must be read alongside the mixed repeated control.

The examples describe the device-executed public adapter protocol. Replace the
paths with existing server assets and retain both passing and failing comparison
results; the original tolerances remain unchanged.
`MINI` is the prepared genuine 42-training/22-held-out-anchor NAVTEST-derived
subset used by the planning controls, not the one-scene synthetic export.
Use the official track/map 100pct initializer and original 8192 vocabulary.
The local image tag is not a public registry download location. Reserve the
16 chips before execution: the original launcher checks character device nodes,
which does not establish that a driver device is idle or usable.

The default adapter mode creates fresh plans and performs original read-only
image/device/path checks without starting Docker training. It also consumes its
output directory, so a plan and an execution must use different fresh outputs.
The examples below explicitly execute foreground training. For a plan, omit
`--execute` and choose a separate output directory.

Versioned device results: [VAD](../results/sync-vad-16chip-20261014.json), [Hydra](../results/sync-hydra-16chip-20261014.json).

## Continuous training and own-checkpoint replay

Run these blocks in the same Bash session.

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

For Hydra, use `e2e_hydramdp.py` and fresh `hydra_continuous` / `hydra_resumed`
directories, retaining all other settings. The adapter checks an `iter_4.pth`
filename; the operator supplies the continuous run's own file. Complete resume
compatibility/restoration is enforced by the unchanged original trainer.

## Original numerical comparison on CPU

Use a new CPU output directory, the same locked post8 image, no NPU device
mounts and read-only source/run inputs. The target remains the original
`ascend/compare_resume.py`, SHA
`9f2cb1cd4a2638cba9a4be24d8a56242909c638ca0bde616445435982ec35330`.
The earlier actual tests used the separately pinned official-loader
audit wrapper SHA
`2eb274e8fd6ae73d2a81c8c62df3d4c59d882307cac26e496ff9a432691bb474`.
The inline command below uses the same installed official `torch_npu`
registration mechanism; it is a documentation example, not the byte-identical
pinned audit wrapper or a separately executed acceptance record.

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

For Hydra replace all VAD run names. The continuous run itself is the prefix
control. Preserve the actual exit and comparison JSON. A complete comparison with tensors
outside the original tolerance is a numerical failure; an invocation interrupted
before comparison completion has no numerical verdict.
Only an actual comparator `result="pass"` at `atol=1e-6, rtol=1e-5` and loss
`1e-4`, together with actual sixteen-rank runtime/restoration reports, supports
that run's bounded result. A launcher plan or Docker exit alone does not.
