# Export original CUDA MSDA goldens

This optional companion creates the schema-1 files consumed by the [NPU verifier](README_npu_cuda_msda.md). It uses the separately published `validate_original_cuda_msda.py` at SHA256 `af7309e616f9a02648f9a5b35f8960ea97f51e108b1915bca7843448bd733fea`; the input generator and original MMCV CPU32/CUDA formulas are unchanged. It requires `mmcv-full==1.6.2` with its actual CUDA extension, NumPy, and a compatible PyTorch CUDA environment.

The two normal-L1 full shapes and four same-storage repetitions are fixed. The exporter records the actual Torch/CUDA versions and complete tensor hashes. The seed does not promise identical values across Torch versions. CPU32 comparisons and repeated bitwise agreement remain separately recorded; successful export does not require CPU32 numerical agreement.

```bash
python tools/validation/export_original_cuda_msda_goldens.py \
  --validator tools/validation/validate_original_cuda_msda.py \
  --output "$NEW_GOLDEN_DIRECTORY" --device cuda:0 \
  --maximum-seconds 270
```

The command previews without importing Torch or creating output. Execute it in a dedicated process with an external timeout:

```bash
timeout --signal=TERM --kill-after=10s 300s \
  python tools/validation/export_original_cuda_msda_goldens.py \
    --validator tools/validation/validate_original_cuda_msda.py \
    --output "$NEW_GOLDEN_DIRECTORY" --device cuda:0 \
    --maximum-seconds 270 --execute
```

`--maximum-seconds` checks the internal budget but cannot guarantee interruption of an in-flight CUDA or serialization call. Keep the external timeout or an equivalent process supervisor. A timeout or incomplete manifest is not a usable export. Use a fresh directory under an existing canonical parent, at least 4 GiB of currently free device memory, and approximately 2 GB for files in addition to the enforced 8 GiB disk reserve. All complete tensors remain on that host. Limit `CUDA_VISIBLE_DEVICES` to the intended device before running it.

On completion, copy the whole golden directory to the NPU host using your normal authorized transfer process. Obtain the SHA256 of `manifest.json` on the producer and independently compare it after transfer. Use that digest with the NPU verifier's `--manifest` argument. The manifest binds all 12 complete tensor files and two case manifests, including the complete six-input tuple and all four output roles.

The additional immutable `source_proof.json` records the actual original source and precision before execution. For compatibility with schema 1, payload field `original_prior_report_SHA256` binds this pre-execution source report; it is explicitly not a previous numerical measurement. No private report is needed. Keep this source proof with the export. A complete portable export has exactly 16 files: the root manifest and source proof, two case manifests, and 12 tensor files. The NPU verifier verifies the proof digest and its source/precision/case identity, in addition to the complete file and tensor identities.

Exporter exit 0 means complete, source-bound artifact creation; it does not mean all CPU32 tolerance checks passed. Exit 2 means blocked or incomplete export. Do not use an incomplete manifest. The NPU verifier independently retains CUDA versus CPU32 numerical outcomes and returns 1 for a complete matrix with tolerance failures.

The [backend guide](../../ascend/MSDA_CUDA.md) explains NPU backend selection; the [original CUDA baseline](../../ascend/CUDA_BASELINE.md) records measured operator and training boundaries. Producing these two normal-L1 cases does not cover every shape or a full model.
