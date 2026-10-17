# Planning-output comparison

This CPU analysis tool compares the original AlgEngine exported planning
trajectories and, optionally, two original NAVSIM score CSVs. It supports
FP32/BF16 and GPU/NPU result pairs without importing a model or an accelerator
framework. It does not run inference, invoke the official scorer, change
training, or decide quality equivalence.

Run from the repository root with Python 3.9+ and NumPy. The validation replay
used NumPy 1.23.4 on the NPU server; the public analysis tool has also recomputed
the retained outputs on a separate CPU host. Inputs are the original
`predictions.pkl` exports: a list of unique-token records, each containing a
finite floating NumPy `trajectory` of shape `(40, 3)` and integer `chosen_ind`.
The restricted loader rejects Torch storages and arbitrary pickle globals.
Use trusted experiment exports, and retain their original hashes.

## Original CUDA MSDA baseline

The separate [CUDA MSDA validator](README_cuda_msda.md) ([中文](README_cuda_msda_zh.md))
checks complete operator outputs/gradients with the original CPU32 and repeated
native gates. See the [measured CUDA operator and training baseline](../../ascend/CUDA_BASELINE.md)
([中文](../../ascend/CUDA_BASELINE_zh.md)); numerical negatives remain explicit.

## Compare trajectories and original scores

First produce each arm's predictions with the
[AlgEngine inference workflow](../../docs/algengine_ascend.md#independent-inference-and-official-cpu-scoring),
then run the original official CPU scorer for **each exact prediction file**.
The scorer writes `official_scores.csv`. This tool reaggregates that CSV;
it cannot substitute for official rescoring of new trajectories.

```bash
set -euo pipefail
LEFT=/absolute/path/to/fp32/inference22
RIGHT=/absolute/path/to/bf16/inference22
LEFT_SCORE=/absolute/path/to/fp32/score22/official_scores.csv
RIGHT_SCORE=/absolute/path/to/bf16/score22/official_scores.csv
OUTPUT=/absolute/path/to/new/trajectory_comparison.json

python tools/validation/compare_planning_outputs.py \
  --left-predictions "$LEFT/predictions.pkl" \
  --right-predictions "$RIGHT/predictions.pkl" \
  --left-score-csv "$LEFT_SCORE" --right-score-csv "$RIGHT_SCORE" \
  --left-label 'NPU FP32' --right-label 'NPU BF16' \
  --comparison-kind separately_trained \
  --output "$OUTPUT"
```

For the same official checkpoint under two inference modes, set
`--comparison-kind same_checkpoint_inference`. Verify checkpoint identity in
the original inference receipts separately; this label does not prove it.
For GPU/NPU analysis, pass the original GPU export on the left and NPU export
on the right, and label the arms accordingly. The same weights, tokens,
candidate vocabulary, geometry and inference settings must be checked in the
experiment provenance before interpreting a platform difference.

Both prediction files must contain exactly the same unique token set. The
tool does not select an intersection or silently omit failures. With CSVs,
both must cover that exact set with valid rows and the seven original metric
columns. Supply both score CSVs or neither. `--provenance existing.json` records
an optional provenance file's hash; its contents are not independently verified
by this measurement tool. The output is a new JSON file and is never overwritten.

## Measurements and interpretation

The output records input sizes and SHA256 before/after analysis, the analysis
script hash, token-by-token chosen-index agreement, exact trajectory equality,
and every exported pose's position/heading difference. It also reports mean,
median, 95th percentile and maximum across all poses, plus final-pose position
statistics. Differences are computed as **right minus left** in FP64:

- Position distance: Euclidean norm of `(dx, dy)`, in metres.
- Heading distance: `abs(atan2(sin(dheading), cos(dheading)))`, in radians.
- Official metrics: the seven means and per-token/mean right-minus-left
  differences from the already generated original CSVs.

The default is 40 poses and 8192 candidates. `--poses` and `--candidates` make
different export dimensions explicit; they do not resample trajectories.
The original official NAVSIM scorer selects eight poses from a 40-pose export;
this tool measures all 40. It does not reinterpret the scorer's formula.

The [recorded FP32/BF16 comparison](../../ascend/results/fp32-bf16-v7-matched8-planning-20261016.json)
uses the **same 22 held-out anchors**, four groups of 880 poses, 3520 pose
comparisons in total. Same-checkpoint groups isolate inference-mode changes;
adapted groups also include the two separately trained eight-update runs.
The retained eight prediction files and eight official CSVs were recomputed
with this public CLI: position values, aggregate statistics, selected-index
counts and all seven score means/deltas matched the published record exactly.
Four individual circular-heading values differed only in the final FP64 bit
on the separate host (maximum 2.78e-17 rad), while aggregate values matched.

The measurement has **no quality-equivalence threshold** and does not emit
an equivalence PASS. Completion with finite values is not convergence,
quality improvement, repeated determinism or GPU/NPU parity. In particular,
the recorded maximum position difference is 6.578134 m; it must not be described
as uniformly negligible.

This nested result-analysis directory is outside the recorded runtime source
fingerprint, which covers `projects/AlgEngine/**/*.py` and `ascend/*.py`.
Adding the analysis tool does not change the measured `e14dd6b2…`/244-file
runtime identity. The analysis tool has its own SHA256 in each output.
