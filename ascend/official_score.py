"""Cache and rescore disjoint real trajectories using unmodified NAVSIM v1.1."""
import argparse,json,os,pickle,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True)
    p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['NUPLAN_MAPS_ROOT']=str(a.maps)
    from navsim.common.dataloader import SceneLoader,MetricCacheLoader
    from navsim.common.dataclasses import SceneFilter,SensorConfig,Trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.script.run_pdm_score_from_submission import run_pdm_score
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    assert manifest['train_eval_token_overlap']==0
    split=yaml.safe_load((a.prepared/manifest['splits']['eval']['filter']).read_text())
    tokens=set(split['tokens'])
    a.cache.mkdir(exist_ok=True,parents=True)
    started=time.monotonic()
    loader=SceneLoader(a.logs,None,SceneFilter(num_history_frames=4,num_future_frames=10,
                       frame_interval=1,has_route=True,log_names=split['log_names'],tokens=list(tokens)),SensorConfig.build_no_sensors())
    if set(loader.tokens)!=tokens:raise RuntimeError('Official scene loader does not cover all heldout tokens')
    processor=MetricCacheProcessor(str(a.cache),force_feature_computation=False)
    rows=[]
    for token in sorted(tokens):
        scene=loader.get_scene_from_token(token)
        scenario=NavSimScenario(scene,map_root=str(a.maps),map_version='nuplan-maps-v1.0')
        entry=processor.compute_metric_cache(scenario)
        if entry is None:raise RuntimeError('Failed cache: '+token)
        rows.append(str(entry.file_name))
        print('CACHED',token,flush=True)
    metadata=a.cache/'metadata'
    metadata.mkdir(exist_ok=True)
    (metadata/'metric_cache.csv').write_text('file_name\n'+'\n'.join(rows)+'\n')
    report=dict(result='pass_cache',tokens=len(tokens),cache_seconds=time.monotonic()-started,
                scorer='unmodified official NAVSIM v1.1 at 3e8291bfa89ff247231e0227778840cd0a036896',
                dataset_scope=manifest['dataset_scope'],gpu_comparison=False)
    if a.predictions:
        results=pickle.load(a.predictions.open('rb'))
        if {r['token'] for r in results}!=tokens or len(results)!=len(tokens):raise RuntimeError('Incomplete predictions')
        submission=a.output.with_suffix('.submission.pkl')
        # Match the original WorldEngine submission conversion exactly: take
        # poses at 0.5, 1.0, ..., 4.0 seconds from its 40-point 10Hz output.
        pickle.dump({'predictions':[{r['token']:Trajectory(np.asarray(r['trajectory'],dtype=np.float64)[4::5]) for r in results}]},submission.open('wb'))
        cfg=OmegaConf.load(Path(__file__).parent/'third_party/navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
        simulator,scorer=instantiate(cfg.simulator),instantiate(cfg.scorer)
        scored=run_pdm_score(submission,simulator,scorer,a.cache)
        if scored is None or len(scored)!=len(tokens) or not scored['valid'].all():raise RuntimeError('Official scoring failed')
        metrics=scored.drop(columns=['token','valid']).mean().to_dict()
        if not np.isfinite(list(metrics.values())).all():raise RuntimeError('Nonfinite official scores')
        scored.to_csv(a.output.with_suffix('.csv'),index=False)
        report.update(result='pass_official_small_data_score',valid_tokens=int(scored['valid'].sum()),metrics=metrics,
                      acceptance='Independent real log subset scored with official CPU simulator/scorer; no GPU parity or full benchmark acceptance')
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
