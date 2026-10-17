# Original CUDA MSDA validation

[中文](README_cuda_msda_zh.md) · [Measured CUDA baseline](../../ascend/CUDA_BASELINE.md)

This standalone Linux tool checks the original `mmcv-full==1.6.2` CUDA MSDA
extension against its unchanged CPU FP32 reference and against itself. It
changes no model or installed operator. The published [actual result](../../ascend/results/cuda-msda-original-20261017.json)
includes complete CPU32 negatives; execution completion is not a numeric PASS.

Use an existing CUDA environment with a compatible CUDA-enabled MMCV 1.6.2
build, PyTorch and NumPy. The measured stack was PyTorch 2.0.1+cu118, CUDA 11.8,
NumPy 1.23.4 on H20 GPUs. The tool installs nothing. It pins the original MMCV Python
reference and forward/backward source hashes, and records the full native DSO
SHA and versions. A different DSO build is allowed but does not inherit the
published run's provenance. Other Torch versions can generate different input
bytes despite the same seed: compare the six input and four CPU-reference hashes.

Run from the repository root. The parent output directory must exist; the
report itself must be fresh. Default preview imports no framework and writes
no file:

```bash
mkdir -p /tmp/msda-result
python tools/validation/validate_original_cuda_msda.py \
  --output /tmp/msda-result/report.json
```

Execute on an idle selected GPU. Keep an external owned-process timeout for a
CUDA C++ call that does not return:

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=20s 360s \
  python tools/validation/validate_original_cuda_msda.py \
  --output /tmp/msda-result/report.json --device cuda:0 \
  --maximum-seconds 300 --execute
```

Default coverage is both normal-L1 full shapes, four executions on the same
storages per case, `seed=29`, four CPU threads, strict deterministic flags and
TF32 off. `--case temporal40000` or `--case query65536` selects one case;
`--repetitions` accepts 4–8. Each comparison checks every element of forward,
value gradient, location gradient and attention-weight gradient. All repetitions
finish before the retained GPU results are copied to CPU. No Tensor payload is
exported or sent away from the execution host.

Forward tolerances remain `atol=2e-5, rtol=2e-5`; gradient tolerances remain
`atol=1e-3, rtol=1e-4`. Repeat raw-byte equality is a separate field from these
tolerance checks. A global strict flag does not certify an external CUDA
extension's deterministic implementation. The two cases are not the historical
raw-L4/spatial-L1 layout experiment.

Exit code `0` means complete comparisons within both CPU32 and repeat tolerance gates;
exit code `1` means complete comparisons outside at least one gate; exit code `2` means blocked
or incomplete execution. A timeout command can return its own code. Retain the
JSON for exit code `1` rather than treating it as a missing result. This is an operator
baseline, not model-training, convergence, NPU-cause or cross-platform parity
acceptance. Actual runtime paths remain in local reports; the public evidence
redacts only installed module paths and library-search paths.
