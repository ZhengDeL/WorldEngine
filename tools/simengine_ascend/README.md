# Prepare an official SimEngine scene

`prepare_scene.py` creates a reproducible one-scene input from the original
WorldEngine publication. It preserves one complete scenario dictionary value
and extracts the matching complete asset directory. It does not generate
synthetic Gaussians, alter image frames, run a renderer, or accept a closed-loop
simulation.

## Inputs

Download these two files on the server. The adjacent
[`official_scene_assets.lock.json`](official_scene_assets.lock.json) fixes their
publisher revisions, paths, lengths, and SHA256 values:

| File | Size | SHA256 |
| --- | ---: | --- |
| `all_scenarios.pkl` | 911,923,264 B | `27203d52454f2538e6a70e95136b85ee8df35822f7707dc0091b4943ca7ad41d` |
| `part003.tar.gz` | 27,886,330,535 B | `102d16b9f608ac8f2314aef2ba3e1907cee9db7e2a3d094b1833b4ff60de0d05` |

The total is **28.80 GB** in decimal units. Full WorldEngine assets are not
required. This is the smallest of the three published `navtest_failures` asset
packages; it is not a claim about the smallest individual scene.

Fixed official ModelScope URLs:

```text
https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine/resolve/092ac5f3f3f828698c207b0ad75ca258c9f07ad2/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl
https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine/resolve/092ac5f3f3f828698c207b0ad75ca258c9f07ad2/data/sim_engine/assets/navtest_failures/assets/part003.tar.gz
```

The same paths are published at
[`OpenDriveLab/WorldEngine`](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)
revision `d0e88d858533ca407a3387a26577c083aa7b3617`. Their HF LFS metadata and
fixed ModelScope metadata agree on both the length and SHA256.

Use a compatible SimEngine Python environment. The preparation script imports
standard-library modules; unpickling the original official file also needs the
modules used by that file, including NumPy. It does not need gsplat to prepare
the scene and asset. Do not bypass the original-file hash check when a Python
dependency is missing.

## Prepare

From the repository root, choose a new output directory whose parent already
exists and has no symlink components:

```bash
python tools/simengine_ascend/prepare_scene.py \
  --scenario-pkl /data/worldengine-inputs/all_scenarios.pkl \
  --asset-archive /data/worldengine-inputs/part003.tar.gz \
  --output /data/worldengine-prepared/official-scene-001
```

The script verifies both complete official files before loading the scenario
pickle or extracting an archive member. Repeating a preparation rechecks the
original inputs. `--asset-lock` can select an alternate path for the same exact
supported lock; it cannot authorize different hashes, branches, or unofficial
test data. Tests exercise internal functions with clearly labeled CPU fixtures;
there is no fixture/bypass mode in the public CLI.

Without `--scene-id`, the tool searches the original dictionary for a scene that
maps to the first actual asset confirmed in part003:

```text
2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4
```

If several original entries map to that asset, the first in the publisher
dictionary order is selected and all matching keys are recorded. To choose a
different original dictionary key explicitly, add `--scene-id <exact-key>`. Its
asset must actually exist in this same part003 package; otherwise preparation
is rejected.

Mapping follows the original renderer: use
`scene['metadata'].get('digitaltwin_asset_id', scene['id'])`, then strip the last
four characters only when `asset_id[-4:].startswith('-')`. The original code
does not require those final three characters to be digits. The receipt records
the dictionary key, internal scene ID, and normalized asset ID separately.

Optional controls:

```bash
  --deadline-utc 2026-10-20T10:50:00+00:00 \
  --stop-file /data/worldengine-control/STOP
```

A stop request, signal, or deadline interrupts preparation and preserves its
private staging directory. Choose a fresh output for the next attempt; stages
are not automatically adopted or deleted. Python unpickling can perform long
operations between checks, so an external container/job deadline remains
appropriate when a strict server reservation must be enforced.

## Outputs and archive safety

```text
official-scene-001/
├── scene.pkl
├── assets/
│   └── <normalized-asset-id>/
│       ├── background/<normalized-asset-id>.ckpt
│       └── ...                         # Other actual files in this asset only
├── prepare_receipt.json
└── checkpoint_inventory.*              # Only with --inspect-checkpoint
```

`scene.pkl` contains `{original_dictionary_key: original_scene_value}`. Camera
calibration, tracks, metadata, map features, and history/future arrays are kept
as provided by the publisher. There is no frame selection or image replacement.
The tool serializes this unchanged value into a new pickle and records the
derived SHA256; a derived subset does not have the full original pickle's SHA.

Only the selected complete asset root is written. The gzip archive is still
read sequentially to check its full member inventory for duplicates/conflicts;
preparation therefore scans the package beyond the selected first checkpoint.
Other assets are not extracted. Links, sparse/special members, traversal,
absolute paths, ambiguous/case-alias paths, duplicate names, file/directory
conflicts, and unsupported declared sizes are rejected. Each selected file is
written once and hashed in a private stage. The final directory is published
atomically without replacing an existing output. Linux requires `renameat2`
with `RENAME_NOREPLACE`; an unsupported platform fails while retaining its stage.

The receipt records both official input hashes, the lock hash, selected IDs,
the derived pickle hash, and each extracted file's path/size/SHA256. Successful
preparation has `pass_official_single_scene_asset_preparation`; it does not mean
that a checkpoint has been loaded or rendered. Exit 2 indicates an incomplete
stop/deadline, and exit 1 indicates a rejected input or an error. An error after
staging starts is recorded in that stage's `prepare_error.json`.

## Optional checkpoint inventory

Add `--inspect-checkpoint` to load the verified, extracted checkpoint in a
separate CPU process. This needs PyTorch and the project dependencies used by
the checkpoint. Set the SimEngine import path when needed:

```bash
export PYTHONPATH="$PWD/projects/SimEngine:${PYTHONPATH:-}"
python tools/simengine_ascend/prepare_scene.py \
  --scenario-pkl /data/worldengine-inputs/all_scenarios.pkl \
  --asset-archive /data/worldengine-inputs/part003.tar.gz \
  --output /data/worldengine-prepared/official-scene-002 \
  --inspect-checkpoint
```

The child uses `torch.load(map_location='cpu', weights_only=False)` only after
both complete official files have passed their fixed hashes. It disables device
backend autoload and checks that CUDA/NPU were not initialized. For enforced
device isolation, run the preparation job in a CPU-only container without
accelerator device mounts.

`checkpoint_inventory.json` lists actual submodel types and state tensor shapes,
dtypes, CPU placement, and finiteness. Missing inspection dependencies or invalid
checkpoint structure reject the requested inspection and retain the stage.
Unsupported model types are listed explicitly. Inventory does not execute the
Gaussian models or prove their rendering contracts. In particular, the filename
`background/*.ckpt` alone does not establish that skybox, rigid, or mirrored
models are present. Read the actual inventory before choosing renderer tests.

## Use the prepared data

The original runner accepts these paths:

```text
data_file_path=/data/worldengine-prepared/official-scene-001/scene.pkl
asset_folder_path=/data/worldengine-prepared/official-scene-001/assets
```

Its replay, renderer, metric, and planner settings must be selected separately.
Simulation metrics/dense reward additionally need the selected original scene's
nuPlan map. Planner-in-the-loop tests additionally need the matching AlgEngine
configuration/checkpoint and a working frame/trajectory client. Raw AlgEngine
JPEG subsets and random Gaussians are not substitutes for this official scene.

## CPU fixture checks

```bash
python -m unittest discover -s tools/simengine_ascend/tests -v
```

These tests use small in-memory fixture dictionaries and tar files. They check
hash gates, exact original normalization, unchanged one-item derivation,
matching-asset extraction, unsafe archive rejection, failed-stage retention,
input mutation, stop/deadline controls, and no-overwrite publication. They do
not download official data, initialize an accelerator, inspect the real published
checkpoint, or establish SimEngine runtime acceptance.

## Build the pinned CANN backend

[`build_backend.py`](build_backend.py) reproduces the isolated compatibility
build. Its input is the official
[`cann/gauss-splat`](https://gitcode.com/cann/gauss-splat) repository at commit
`8c0445ebcca91514fce3a7050241f41ed132c340`. The source checkout stays unchanged;
the tool copies its tracked regular files into a new build directory. Tracked
changes, a different commit, symlinks, or worktree bytes that differ from the
pinned Git blobs are rejected. Untracked files are not copied.

The tool requires an existing isolated venv with the compatible PyTorch,
Torch-NPU, pip, setuptools, wheel, and native build prerequisites. It does not
create or repair that environment. Source the CANN environment before running
the tool. A venv created with `--system-site-packages` from the compatible base
Python may reuse the installed PyTorch/Torch-NPU stack. Keep this environment
separate from an existing production service.

Example preparation on the server:

```bash
git clone https://gitcode.com/cann/gauss-splat.git /data/sources/gauss-splat
git -C /data/sources/gauss-splat checkout --detach 8c0445ebcca91514fce3a7050241f41ed132c340

CANN_ROOT=/usr/local/Ascend/cann-9.1.0
source "$CANN_ROOT/set_env.sh"
# If needed, create the venv explicitly from your compatible base Python:
# /path/to/compatible/python -m venv --system-site-packages /data/venvs/gauss-splat-compat

: "${BUILD_DEADLINE_UTC:?Set an authorized future UTC deadline, including +00:00}"
python tools/simengine_ascend/build_backend.py \
  --source /data/sources/gauss-splat \
  --output /data/builds/gauss-splat-compat-001 \
  --python /data/venvs/gauss-splat-compat/bin/python \
  --jobs 8 \
  --cann-root "$CANN_ROOT" \
  --cann-python-path "$CANN_ROOT/python/site-packages" \
  --deadline-utc "$BUILD_DEADLINE_UTC" \
  --stop-file /data/build-control/STOP
```

Create `/data/builds` beforehand. The chosen output must be new and disjoint
from the source and venv. The tool does not reuse a previous build directory.
An interrupted or rejected build remains available with its logs and receipt;
select a new output for the next attempt.

The copied source receives the checked-in
[`apply_compat_patch.py`](../../projects/SimEngine/worldengine/render/ascend/patches/apply_compat_patch.py)
inference-only patch and these three operational adjustments:

1. Bound both original `multiprocessing.cpu_count()` build concurrency calls to
   `min(jobs, multiprocessing.cpu_count())`; `--jobs` accepts 1 through 16.
2. Set the CMake preset to `ASCEND_COMPUTE_UNIT=ascend910_93` and
   `ENABLE_TEST=False`.
3. Locate Torch-NPU through
   `importlib.util.find_spec('torch_npu').origin`. The original
   `site.getsitepackages()[0]` query selected the venv's empty site-packages
   directory in the tested system-site-packages environment and missed the
   actual Torch-NPU headers.

The selected venv's bin directory is placed first on PATH. Build children disable
device backend autoload, remove inherited Python source-shadow paths and pip
destination settings, disable inherited pip configuration, require a venv, bound
compiler parallelism, and set `PIP_NO_INDEX=1`. All build dependencies must
already be installed. The wheel command is:

```text
<selected-venv-python> -m pip wheel --no-build-isolation --no-deps . --wheel-dir <new-output>/wheels
```

CANN's own Python directories may be necessary for `import acl`. The builder
retains an inherited PYTHONPATH entry only if it is an existing absolute
directory inside the declared SDK installation and does not overlap the original
source or new build output. Other inherited Python paths are discarded. The SDK
root defaults to `ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME`; use
`--cann-root` to select it explicitly. The example uses this run's installed
`/usr/local/Ascend/cann-9.1.0` root. Replace it with the actual installation path
on another server. Source `set_env.sh`, select the SDK root, and resolve pyACL
from the same installation. A stale `ascend-toolkit` symlink pointing to another
version can cause the installation preflight to reject the pyACL path.
The root must contain a supported `libascendcl.so` installation marker. This is
a path-scope check, not verification of a vendor signature. An unusual CANN
layout can specify an additional `--cann-python-path <directory>`; it must still
resolve inside that same SDK root. Installation preflight checks that pyACL can
be resolved there, and the final real import records its origin and SHA256.

The builder sets `BUILD_WITHOUT_SHA=1`. The copied source has no `.git` directory;
otherwise upstream setup.py may label its wheel with an unrelated ancestor
repository's HEAD, or `gitUnknown`. The exact CANN source commit remains fixed
and recorded in the build receipt instead of being inferred from a parent repo.

The default stops after creating and hashing exactly one `gauss_splat` wheel.
To install it into this same selected venv, add `--install`. Installation runs
from the separate `control` directory using the exact produced wheel path:

```text
<selected-venv-python> -m pip install --force-reinstall --no-deps <exact-wheel>
```

The force reinstall and external working directory are intentional: ordinary
installation from the source working directory reported an already installed
distribution because the source tree's egg-info was visible in the tested build.
After installation, a separate import check requires both the Python package
and native extension to resolve inside the selected venv, verifies the
compatibility contract string, and records the actual extension SHA256. The
builder never installs into base/global Python.

`build_receipt.json` records the original commit, all tracked original source
SHA256 values through `source_manifest.json`, compatibility patch hashes and its
before/after receipt, operational before/after hashes, the complete patched
tracked-file inventory, exact commands/cwd/logs, environment versions, and wheel
SHA256. With installation enabled it also records the imported package and
extension paths/hash. The original source and tracked copied inputs are checked
again before success. Compiler and CANN environment details remain relevant;
the tool does not promise bitwise-identical wheels across different stacks.

`--dry-run` checks/copies/patches the source and records the intended wheel command
without querying the venv runtime, compiling, or installing. It produces
`dry_run_source_prepared_no_wheel_built`; it cannot be combined with `--install`.
Exit 2 means a deadline/stop request; exit 1 means a rejected input or error.
Only `pass_wheel_build_only` or `pass_wheel_build_and_install_import_only`
indicates the corresponding build stage completed.

Every build child starts a separate process group. On Linux the tool uses
`waitid(..., WNOWAIT)` to retain the leader PID until its original group has been
terminated and reaped, including compiler descendants that remain in that group
after a failed pip process. It does not enumerate or stop host services. Run the build in a
CPU-only container without NPU device mounts when device isolation is required.
The absolute deadline is mandatory; a process cleanup grace can extend several
seconds beyond it, so use a cutoff before the end of the server reservation.

A successful wheel build or import is **not** a numerical or NPU rendering pass.
This tool does not produce an approved renderer validation manifest. Run the
separate device and real-scene validation before accepting the backend.

Builder fixtures are included in the same unittest discovery command above.
They exercise local Git/clean-source rules, tracked-only copying, the exact three
operational changes, environment selection, owned subprocess failure/deadline
cleanup, foreign-process preservation, and stop-before-start. They do not compile
CANN kernels, download source, install a wheel, or establish a device result.
