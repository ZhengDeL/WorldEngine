"""Score exported 40-pose trajectories through the existing official CPU path."""
import argparse
import json
import os
from pathlib import Path
import pickle
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--metric-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    predictions = args.predictions.resolve(strict=True)
    cache = args.metric_cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error('Select a fresh output directory')
    os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    sys.path.insert(0, str(ROOT))
    from ascend.scoring import score_predictions
    with predictions.open('rb') as stream:
        results = pickle.load(stream)
    if not isinstance(results, list) or not results:
        raise RuntimeError('Predictions must be a nonempty original result list')
    tokens = [row.get('token') if isinstance(row, dict) else None for row in results]
    if any(not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', token) for token in tokens):
        raise RuntimeError('Prediction tokens must be nonempty path-safe identifiers')
    if len(set(tokens)) != len(tokens):
        raise RuntimeError('Duplicate prediction tokens')
    requested = set(tokens)
    payloads = {}
    for payload in cache.rglob('metric_cache.pkl'):
        token = payload.parent.name
        if token not in requested:
            continue
        resolved = payload.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError('Cache payload is not a regular file: '+str(payload))
        if token in payloads:
            raise RuntimeError('Ambiguous duplicate cache token: '+token)
        payloads[token] = resolved
    missing = sorted(requested - set(payloads))
    if missing:
        raise RuntimeError('Official metric-cache payloads missing prediction tokens: '+str(missing[:20]))
    # The pinned scorer iterates every cache token. Limit its index to exactly
    # this submission while keeping original payload bytes and mathematics.
    selected = output/'selected_payloads'
    selected.mkdir(parents=True)
    for token, payload in sorted(payloads.items()):
        folder = selected/token
        folder.mkdir()
        (folder/'metric_cache.pkl').symlink_to(payload)
    selection = dict(prediction_tokens=sorted(requested), selected_cache_tokens=sorted(payloads),
                     exact_token_coverage=True, source_metric_cache=str(cache),
                     payload_bytes_modified=False)
    (output/'cache_selection.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf8')
    metrics = score_predictions(results, selected, output)
    print(json.dumps(dict(result='pass_original_CPU_scoring', metrics=metrics,
                          output=str(output)), indent=2))


if __name__ == '__main__':
    main()
