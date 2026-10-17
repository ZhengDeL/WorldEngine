# Verify NPU MSDA against saved CUDA tensors

This validator calls the actual registered MX and MMCV MSDA entries through the selected WorldEngine backend. It compares every forward element and all three input gradients with four saved original CUDA executions and a separately saved CPU32 reference. It generates neither random inputs nor a new reference.

The required golden directory is not included in the repository. Supply a trusted, complete schema-1 CUDA export and its independently obtained manifest SHA256. The current verifier accepts the two normal-L1 cases below, with four CUDA repetitions per case:

| Case | Batch | Queries | Spatial shape | Heads / channels / points |
|---|---:|---:|---|---|
| Temporal | 2 | 40,000 | 200 × 200 | 8 / 32 / 4 |
| Query | 1 | 65,536 | 200 × 200 | 8 / 32 / 4 |

For each case, the export contains `inputs.pt`, `original_CPU32_reference.pt`, `cuda_repeat_000.pt` through `cuda_repeat_003.pt`, and `case_manifest.json`. The root `manifest.json` records each file's size, SHA256, tensor metadata and raw tensor SHA256. The validator checks these identities before and after execution and uses Torch's restricted `weights_only=True` loader. A portable export also contains `source_proof.json`, giving 16 files in total: one root manifest, one source proof, two case manifests and 12 tensor files. The verifier checks the source-proof whole SHA256 and joins its validator, source, precision, versions and case definitions to the root manifest. Each tensor payload is bound to the same proof digest. Legacy schema-1 exports without a `prior_report_file` retain their original 15-file interface.

Create a portable export on a CUDA host with the [CUDA golden exporter](README_export_cuda_msda_goldens.md), then transfer the whole directory and independently verify the producer manifest digest. Golden tensors are not downloaded by this tool.

Run inside the supported WorldEngine Ascend environment, with `torch_npu`, MMCV and MX Driving available. Set `$WORLDENGINE_ROOT`, `$GOLDENS`, `$GOLDEN_SHA256` and a fresh `$REPORT` path outside the source and golden directories. The report's parent directory must exist. Device numbers refer to visible logical devices.

```bash
python tools/validation/verify_npu_cuda_msda.py \
  --source "$WORLDENGINE_ROOT" \
  --goldens "$GOLDENS" --manifest "$GOLDEN_SHA256" \
  --output "$REPORT" --device npu:0 \
  --backend cuda-compatible --maximum-seconds 180
```

The command previews the selection without importing Torch, initializing a device or writing a report. Execute the same selection in a dedicated process with an external timeout:

```bash
timeout --signal=TERM --kill-after=10s 210s \
  python tools/validation/verify_npu_cuda_msda.py \
    --source "$WORLDENGINE_ROOT" \
    --goldens "$GOLDENS" --manifest "$GOLDEN_SHA256" \
    --output "$REPORT" --device npu:0 \
    --backend cuda-compatible --maximum-seconds 180 --execute
```

`--maximum-seconds` is an internal budget, not a guarantee that an in-flight native operation can be interrupted. Python signal handling may wait for the native call to return. Keep the external timeout or an equivalent process supervisor; its termination exit status is not a completed numerical result. Both registered entries are selected by default, yielding 80 complete tensor comparisons: 64 against CUDA and 16 against CPU32. `--entries mmcv` selects only the six-argument MMCV entry; `--backend reference` measures the default historical route against the same fixed targets.

The original tolerances remain forward `atol=2e-5, rtol=2e-5` and gradients `atol=1e-3, rtol=1e-4`. The report separates CUDA and CPU32 counts, raw tensor-byte equality (SHA256, distinct from serialized file bytes) and tolerance agreement. It records actual source hashes, backend policy, versions, FP32/disabled-autocast/HF32-off settings and strict deterministic flag readback. A strict global flag does not establish deterministic registration of every external kernel.

| Exit | Meaning |
|---:|---|
| 0 | Complete matrix; every CUDA and CPU32 comparison is within its original tolerance |
| 1 | Complete matrix; one or more comparisons are outside tolerance |
| 2 | Invalid input, blocked/incomplete execution, or integrity failure |

CPU32 and original CUDA differ at some interpolation boundaries. CUDA comparisons can all pass while CPU32 comparisons cause exit 1; inspect the separate target counts. Bitwise inequality alone does not cause exit 1. These are bounded operator results, without a claim about full-model replay, convergence, quality, performance or the cause of other NPU numerical differences. Cross-version RNG seeds are not evidence of identical inputs: use the complete saved tensor hashes.

See the [backend guide](../../ascend/MSDA_CUDA.md) for model CLI usage and checkpoint contracts, and the [original CUDA baseline](../../ascend/CUDA_BASELINE.md) for measured scope and target differences.
