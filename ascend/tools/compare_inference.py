"""Compare same-source FP32 inference A/B outputs without relaxing tolerances."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ascend.tools.rank_journal import semantic_digest


def compare(baseline, candidate):
    a = json.loads((baseline / 'inference_validation.json').read_text())
    b = json.loads((candidate / 'inference_validation.json').read_text())
    identities = ('image_id', 'source_sha256', 'source_config_sha256', 'checkpoint_sha256',
                  'torch_version', 'torch_npu_version', 'torch_npu_git_version', 'world_size', 'precision', 'device_provenance')
    required = ('image_id', 'source_sha256', 'source_config_sha256', 'checkpoint_sha256',
                'torch_version', 'torch_npu_version', 'world_size', 'precision', 'device_provenance')
    identity_mismatches = [key for key in identities if a.get(key) != b.get(key)]
    identity_mismatches += ['missing_' + key for key in required if a.get(key) is None or b.get(key) is None]
    for role, report in (('baseline', a), ('candidate', b)):
        if report.get('result') != 'pass_real_planning_inference':
            identity_mismatches.append(role + '_not_validated_planning_inference')
    for role, report in (('baseline', a), ('candidate', b)):
        state = report.get('precision', {}).get('before_model', {})
        if state.get('mode') != 'fp32' or state.get('public') != dict(matmul_allow_hf32=False, conv_allow_hf32=False):
            identity_mismatches.append(role + '_not_strict_FP32')
    with (baseline / 'predictions.pkl').open('rb') as stream:
        left = pickle.load(stream)
    with (candidate / 'predictions.pkl').open('rb') as stream:
        right = pickle.load(stream)
    for role, results, report in (('baseline', left, a), ('candidate', right, b)):
        if not results or len(results) != report.get('anchors'):
            identity_mismatches.append(role + '_incomplete_anchor_count')
    order_equal = [item['token'] for item in left] == [item['token'] for item in right]
    mismatched = []
    if order_equal:
        mismatched = [x['token'] for x, y in zip(left, right)
                      if semantic_digest(x) != semantic_digest(y)]
    unique = len({item['token'] for item in left}) == len(left)
    opt = b.get('inference_optimization', {})
    requested = opt.get('requested', 'none')
    optimization_used = (requested == 'none' or opt.get('point_parallel_calls', 0) > 0)
    if requested == 'point-parallel-last-frame':
        optimization_used = optimization_used and opt.get('last_frame_calls', 0) > 0
    exact = not identity_mismatches and order_equal and unique and not mismatched
    return dict(result='exact' if exact else 'requires_diagnosis',
                identity_mismatches=identity_mismatches, anchors=len(left),
                ordered_tokens_equal=order_equal, unique_tokens=unique,
                typed_outputs_raw_exact=order_equal and not mismatched,
                mismatched_tokens=mismatched, requested_optimization_actually_used=optimization_used,
                baseline_elapsed_seconds=a.get('elapsed_seconds'),
                candidate_elapsed_seconds=b.get('elapsed_seconds'),
                timing_scope='Original test-runner walltime after Python/Torch imports and config setup; includes model/checkpoint loading, inference and collection. Use repeated A/B with identical devices, data and warm state for performance acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.baseline, args.candidate)
    if args.output.exists():
        raise RuntimeError('Fresh comparison output required')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['result'] == 'exact' and report['requested_optimization_actually_used'] else 3)


if __name__ == '__main__':
    main()
