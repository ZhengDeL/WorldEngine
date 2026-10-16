# Deterministic NAVTRAIN139 / NAVTEST22 preparation

This tool derives a fixed validation subset from original public WorldEngine
annotations/PDM and OpenScene cameras. It is not an official139/22 mini package.
It downloads nothing and requires no private intermediate dataset or hardware
metadata. All input hashes, publisher URLs and revisions are in `INPUTS.lock.json`.

The existing measured data contains139 training anchors from one genuine
NAVTRAIN log and22 independent NAVTEST anchors from a different log. Its
metadata, image and label lineage is documented in `DATA_PROVENANCE.json`.
The public CLI was replayed on the six pinned original cached files on
2026-10-16. Both CPU stages exited0 without NPU devices: all1832 raw JPEGs
passed complete SHA and decoding checks, and all six derived metadata payloads
matched the prior subset byte for byte and by complete nested values/candidate
order. See [CPU reproduction record](VALIDATION.json). This preparation result
does not imply model training or quality acceptance.

## Cached inputs

Supply the four complete original annotation/PDM PKLs, mini camera archive12,
and test camera archive30. They total47,913,227,335B if none are cached. Reuse
verified server caches; do not download all camera archives or simulation
assets. The tool copies only1832 referenced original JPEGs. It never substitutes
random data, new rewards or resized/re-encoded images.

WorldEngine revision: `d0e88d858533ca407a3387a26577c083aa7b3617`.
OpenScene revision: `a76f840b65e972bc45e56c2adced897498e9a026`.
Original split revision: `ec13257a95690a3e74d201f3b94064313a9369f8`.

Training retains284 consecutive original records, all139 original split anchors
in the selected log, and native four-record queues. Evaluation retains51
records and the22 fixed published-data anchors from the existing held-out
branch. The public reference filter contains identifiers and original image
hashes, not labels. Original selected PDM entries and all8192 candidates remain
unchanged. No future-mask or SimScale exclusion is added.

The preparation executes only the original unchanged `load_annotations` method
AST with real pickle IO for metadata indexing and the native scene assignments.
It does not construct a dataset, run its pipeline, import an accelerator SDK or
build a model. Evaluation camera paths use the documented portable
`log_name/original_basename` form; all other field values are retained.

## Run on the Linux server

Use the accepted dependency-repair image or build that public recipe first.
The CPU preparation requires NumPy1.23.5, PyYAML and Pillow. Allow128GiB RAM,
8CPU cores and at least2GiB for new outputs. No NPU devices or driver mounts are
needed. Pass the actual original resolved file paths; they can remain in
different cache directories, without copying40GB of metadata into one directory.

```bash
set -euo pipefail
SOURCE=/path/to/WorldEngine
TOOLS="$SOURCE/ascend/data_tools/navtrain139"
NAVTRAIN_ANN=/cached/nuplan_openscene_navtrain.pkl
NAVTRAIN_PDM=/cached/pdm_8192_gt_cache_navtrain.pkl
NAVTEST_ANN=/cached/nuplan_openscene_navtest.pkl
NAVTEST_PDM=/cached/pdm_8192_gt_cache_navtest.pkl
TRAIN_CAMERAS=/cached/openscene_sensor_mini_camera_12.tgz
EVAL_CAMERAS=/cached/openscene_sensor_test_camera_30.tgz
OUTPUT_PARENT=/path/to/new/outputs
IMAGE_ID=sha256:REPLACE_WITH_ACTUAL_ACCEPTED_IMAGE_ID
DEADLINE=$(date -u -d '+15 minutes' '+%Y-%m-%dT%H:%M:%S+00:00')

docker run --rm --runtime runc --network none --memory 128g --cpus 8 \
  --pids-limit 2048 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$SOURCE:/workspace/WorldEngine:ro" -v "$TOOLS:/tools:ro" \
  -v "$NAVTRAIN_ANN:/inputs/navtrain.pkl:ro" \
  -v "$NAVTRAIN_PDM:/inputs/navtrain_pdm.pkl:ro" \
  -v "$NAVTEST_ANN:/inputs/navtest.pkl:ro" \
  -v "$NAVTEST_PDM:/inputs/navtest_pdm.pkl:ro" \
  -v "$TRAIN_CAMERAS:/inputs/train_cameras.tgz:ro" \
  -v "$EVAL_CAMERAS:/inputs/eval_cameras.tgz:ro" \
  -v "$OUTPUT_PARENT:/output:rw" "$IMAGE_ID" \
  /opt/conda/envs/bevformer/bin/python /tools/prepare_navtrain139.py \
  --source /workspace/WorldEngine \
  --navtrain-annotations /inputs/navtrain.pkl --navtrain-pdm /inputs/navtrain_pdm.pkl \
  --navtest-annotations /inputs/navtest.pkl --navtest-pdm /inputs/navtest_pdm.pkl \
  --train-cameras /inputs/train_cameras.tgz --eval-cameras /inputs/eval_cameras.tgz \
  --output /output/prepared --deadline "$DEADLINE" --max-seconds 900 --execute
```

Create `OUTPUT_PARENT` before running and keep `/output/prepared` nonexistent.
Omit `--execute` to print a plan without reading dataset files or creating output.
Every full input is verified before and after processing. The complete archive
scan rejects unsafe paths, links, special files and duplicate image sources;
selected raw JPEG bytes must match the fixed reference hashes and decode.

The output contains the six derived metadata files, `cameras/`, a launcher-ready
`manifest.json`, `preparation.json` and `result.json`. Use
`--mini-prepared "$OUTPUT_PARENT/prepared"` with the public launcher. Both
training and evaluation PDM prefixes retain the required navtrain/navtest names.

The measured replay reproduced all six metadata files byte for byte. The
receipt also checks complete nested values and candidate order independently;
it records file identity separately from semantic equality for future replays. Any new output has its own data
hash contract and must undergo its own model/inference/score validation; it cannot
borrow checkpoint compatibility or NPU acceptance from a different prepared run.
Do not use this subset as evidence of complete NAVTRAIN convergence or full
NAVTEST quality. Preserve failures and all partial outputs; rerun in a new directory.
