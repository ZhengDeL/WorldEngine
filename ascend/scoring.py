"""Strict trajectory coverage and unmodified official NAVSIM CPU scoring."""
import json,pickle,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def score_predictions(results,cache,output):
    cache,output=Path(cache),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    nav=ROOT/'ascend/third_party/navsim'
    sys.path.insert(0,str(nav))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.script.run_pdm_score_from_submission import run_pdm_score
    # Relocate metadata paths without changing the cache payload or scorer.
    index=output/'cache_index';metadata=index/'metadata';metadata.mkdir(parents=True,exist_ok=True)
    payloads=sorted(cache.rglob('metric_cache.pkl'))
    if not payloads:raise RuntimeError('No official metric-cache payloads')
    (metadata/'metric_cache.csv').write_text('file_name\n'+'\n'.join(str(p.resolve()) for p in payloads)+'\n')
    loader=MetricCacheLoader(index)
    tokens=[row['token'] for row in results]
    if len(set(tokens))!=len(tokens) or not set(tokens).issubset(set(loader.tokens)):
        raise RuntimeError('Duplicate predictions or incomplete official metric cache')
    predictions={}
    for row in results:
        trajectory=np.asarray(row['trajectory'],dtype=np.float64)
        if trajectory.shape!=(40,3) or not np.isfinite(trajectory).all():raise RuntimeError('Invalid exported trajectory')
        predictions[row['token']]=Trajectory(trajectory[4::5])
    submission=output/'submission.pkl';pickle.dump(dict(predictions=[predictions]),submission.open('wb'))
    config=OmegaConf.load(nav/'navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
    scored=run_pdm_score(submission,instantiate(config.simulator),instantiate(config.scorer),index)
    if scored is None or len(scored)!=len(tokens) or set(scored['token'])!=set(tokens) or not scored['valid'].all():
        raise RuntimeError('Official scoring failed or omitted tokens')
    metrics=scored.drop(columns=['token','valid']).mean().to_dict()
    if not np.isfinite(list(metrics.values())).all():raise RuntimeError('Nonfinite official scores')
    scored.to_csv(output/'official_scores.csv',index=False)
    (output/'official_score.json').write_text(json.dumps(dict(valid_tokens=len(tokens),metrics=metrics,scorer_commit='3e8291bfa89ff247231e0227778840cd0a036896'),indent=2))
    return metrics
