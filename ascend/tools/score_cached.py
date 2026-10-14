"""Score exported 40-pose trajectories through the existing official CPU path."""
import argparse
import json
import os
from pathlib import Path
import pickle
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
    metrics = score_predictions(results, cache, output)
    print(json.dumps(dict(result='pass_original_CPU_scoring', metrics=metrics,
                          output=str(output)), indent=2))


if __name__ == '__main__':
    main()
