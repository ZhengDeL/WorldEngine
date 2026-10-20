# Full NAVTEST input audit

`ascend/tools/verify_navtest_inputs.py` checks the official 12,146 NAVTEST anchors,
136 logs, current plus three original history frames, eight cameras, and all
142,976 required 1920×1080 RGB JPEGs. It uses the unchanged public
`ascend.preflight.inspect_phase`, then independently reconstructs the 17,872
required queue frames. This is CPU input acceptance, not model inference,
training, convergence, GPU agreement or official trajectory scoring.

On October 20, 2026, the public CLI completed all 142,976 required JPEG
decodes and official 32-archive byte-linkage checks in the fixed image, with
zero errors and exit code 0. A separate ledger audit then reconstructed and
verified every image/member record and ordered digest without decoding the
images again. See the [measured record](navtest_public_input_results.md) for
source, original-summary and audit-report identities. The earlier private
worker `aad3c0875a015dd4c51204dcf2534a080f03ef6c4bdb51534189c44c8ec5ca79`
remains a separate result.

## Inputs and fixed identities

The selected AlgEngine source must have the original 245-file fingerprint
`0488bfb748904ab0853f2c86ad868a1bf84c19991ad383ca1a15e346f7e40922`.
The extra nested tools have their own hashes and do not change that scope.
Both original 100% model test configurations must describe the same full
`NavSimOpenSceneE2E` test split, queue4, history3 and load_interval1.

The data root must contain these exact publisher assets:

| Relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl` | 848516627 | `59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b` |
| `alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl` | 2791314758 | `58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316` |
| `alg_engine/test_8192_kmeans.npy` | 3932288 | `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35` |

Images belong under `raw/openscene-v1.1/sensor_blobs/test/<log>/<camera>/<file>.jpg`.
Model checkpoints and generated metric caches are not inputs to this CPU check.
Only trusted, publisher-SHA-locked annotation/PDM pickles are loaded.

Publisher pixel provenance additionally requires all 32 original test-camera
archives, 127,882,665,618 compressed bytes, under one archive root. The public
lock is `ascend/data_tools/navtest12146/ARCHIVES.lock.json`, SHA256
`c62488d01a9fe43c254085aa009a06f1e001e5011f4081a56069d209b7e6ed54`.
It contains only filenames, fixed-revision download URLs, sizes, hashes and
publisher revisions. The hashes come from official Hugging Face LFS metadata at
revision `a76f840b65e972bc45e56c2adced897498e9a026`; the ModelScope URLs use the
matching per-file revision `9439af9e240a950dd0b6862ce3bc34233c5d8690`.
The recorded ModelScope metadata query used `master`; the fixed revision comes
from its returned file records, not from a claim that the query was pinned.

## CPU-only execution

Use a Linux container with NumPy, Pillow, PyYAML and the existing MMCV Config
dependency. Do not mount accelerator devices or request an Ascend runtime.
The frozen image below has the existing CPU-inspection dependencies. Disable
Torch backend autoload before any package import; MMCV may import Torch but no
accelerator backend is initialized by this checker. Device nodes or initialized
CUDA/NPU backends cause rejection before and after the audit.

Set `SOURCE`, `DATA`, `ARCHIVES`, `QC_PARENT`, `JOB_NAME` and a future, authorized
UTC `DEADLINE_UTC`. `QC_PARENT` must exist; its `run1` child must not exist.
The container name must also be fresh. Preserve actual image/container identity,
mounts, exit/OOM state and the external owned-container deadline watchdog.

```bash
docker run --name "$JOB_NAME" --runtime runc --network none \
  --memory 64g --cpus 8 --pids-limit 1024 \
  -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e CUDA_VISIBLE_DEVICES= -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" \
  -v "$ARCHIVES:/archives:ro" -v "$QC_PARENT:/qc:rw" \
  --entrypoint /opt/conda/envs/bevformer/bin/python \
  sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48 \
  /source/ascend/tools/verify_navtest_inputs.py \
  --source /source --data-root /data --output-dir /qc/run1 \
  --archive-root /archives \
  --archive-lock /source/ascend/data_tools/navtest12146/ARCHIVES.lock.json \
  --archive-lock-sha256 c62488d01a9fe43c254085aa009a06f1e001e5011f4081a56069d209b7e6ed54 \
  --deadline-utc "$DEADLINE_UTC" --stop-file /qc/STOP --decode-workers 4
```

The tested allocation was 64 GiB RAM and eight CPUs. The checker process
reported a peak RSS of 5,001,781,248 bytes (about 4.66 GiB). This is neither total
container memory nor model-training capacity; smaller allocations have not been
validated by this result. Decode queues hold at most twice the selected 1–4
worker count. Archive streaming also uses at most four workers. No image
payload collection is retained in memory.

## What is verified

1. Official annotations, PDM, vocabulary, split, source and configurations are
   locked. The log-token IDs are mapped to the 136 official log names through
   the locked annotations; these two naming systems are not compared directly.
2. Every required JPEG is hashed, checked with Pillow `verify()`, reopened and
   fully decoded with `load()`. Its geometry/RGB mode and immediate post-decode
   SHA are checked. Missing history, camera channels, nonfinite PDM or bad JPEGs
   reject acceptance.
3. All 32 gzip archives are read to EOF. Their complete compressed byte size/SHA
   and gzip CRC/length are checked. Every tar header is checked for path escape,
   case alias, duplicate names, file/directory conflicts, links, sparse/special
   files, and bounded member/metadata sizes. Nothing is extracted.
4. Every required JPEG member's uncompressed size/SHA must match the decoded
   dataset ledger. Unselected files still receive safe-header checks and belong
   to the verified whole archive; their payloads need not be individually hashed.
   Equal JPEG duplicates across archives are allowed only when their bytes match
   the same required ledger entry. The proof establishes at least one publisher
   source per required image, not a unique source archive.
5. After archive inspection, all required dataset JPEGs are rehashed against
   the decode ledger. Annotation/PDM/vocabulary/source/split/lock identities are
   checked again. These checks cover the recorded read windows; a read-only
   container mount does not prevent later writes by an external host process.

`summary.json` must report
`pass_complete_original_NAVTEST_inputs_and_required_JPEG_decode_CPU_only`,
actual_exit0, zero decode failures, 142,976 passed images, and accepted archive
pixel provenance. Preserve `images.jsonl`, the per-archive member ledgers and
their whole hashes with the summary. Exit1 is rejection; exit2 is incomplete
STOP/signal/deadline. Neither is a full input pass.

Without original archives, explicit `--assembled-only` may check the metadata
and assembled JPEG tree. Omit all three archive options in that mode. Its result
is `pass_NAVTEST_metadata_and_JPEG_decode_without_publisher_pixel_provenance_CPU_only`
and both provenance flags remain false. It cannot establish official camera
pixel origin and must not be substituted for the formal archive-linked result.

For an interrupted run, a fresh output may reuse the exact same public worker's
SHA-locked previous summary and JPEG ledger with all four options:
`--resume-report`, `--resume-report-sha256`, `--resume-ledger`,
`--resume-ledger-sha256`. Passing old JPEGs are rehashed before reuse; malformed,
duplicate or changed rows reject reuse. Archive streams are rechecked. The old
private aad3 report cannot be silently reused as a report from this new worker.

Run local CPU fixtures with:

```bash
python -m unittest discover -s ascend/tools/tests -p 'test_*navtest*.py' -v
```

Synthetic tar/JPEG fixtures test the mechanisms only. Full official-data
acceptance requires a separately retained actual run of this exact public source.
