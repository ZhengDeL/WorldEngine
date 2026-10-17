"""Split a prepared official real subset by whole driving logs.

Only functional/small-data evaluation is claimed. This does not create a full
official NAVTRAIN/NAVTEST benchmark, and the planner vocab/labels stay intact.
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rare-split',type=Path,required=True)
    args=p.parse_args()
    source=args.prepared.resolve(strict=True)
    if args.output.exists():raise RuntimeError('Select a new output')
    args.output.mkdir(parents=True)
    output=args.output.resolve()
    frames=pickle.load(open(source/'worldengine_navtest_smoke.pkl','rb'))['infos']
    pdm=pickle.load(open(source/'worldengine_navtest_smoke_pdm.pkl','rb'))
    original=yaml.safe_load((source/'worldengine_navtest_smoke.yaml').read_text())
    anchors=set(original['tokens'])
    rare=set(yaml.safe_load(args.rare_split.read_text())['tokens'])
    logs=sorted({x['log_name'] for x in frames})
    rare_logs={x['log_name'] for x in frames if x['token'] in anchors & rare}
    eligible=[log for log in logs if log not in rare_logs]
    if not eligible:raise RuntimeError('Need a separate non-rare log for heldout evaluation')
    heldout=eligible[-1]
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['splits']={}
    token_sets={}
    for phase in ('train','eval'):
        subset=[x for x in frames if (x['log_name']==heldout)==(phase=='eval')]
        tokens=[x['token'] for x in subset if x['token'] in anchors]
        if not tokens:raise RuntimeError('Empty split')
        token_sets[phase]=set(tokens)
        stem=f'worldengine_navtest_{phase}'
        pickle.dump({'infos':subset,'metadata':{'source':'official real subset','phase':phase}},open(output/f'{stem}.pkl','wb'),protocol=4)
        pickle.dump({t:pdm[t] for t in tokens},open(output/f'{stem}_pdm.pkl','wb'),protocol=4)
        config=copy.deepcopy(original)
        config['tokens']=tokens
        config['log_names']=sorted({x['log_name'] for x in subset})
        (output/f'{stem}.yaml').write_text(yaml.safe_dump(config))
        manifest['splits'][phase]=dict(annotations=stem+'.pkl',filter=stem+'.yaml',pdm_prefix=stem+'_pdm',
                                     anchors=len(tokens),frames=len(subset),logs=config['log_names'])
    assert not token_sets['train'] & token_sets['eval']
    if not token_sets['train'] & rare:raise RuntimeError('No genuine rare training anchor')
    manifest.update(training_anchors=len(token_sets['train']),heldout_samples=len(token_sets['eval']),
                    heldout_log=heldout,split_by='whole driving log',train_eval_token_overlap=0,
                    train_rare_tokens=sorted(token_sets['train'] & rare),
                    dataset_scope='Real NAVTEST subset split by log for adaptation training and independent small-data evaluation; not full official benchmark')
    # Keep the old baseline files for provenance; phase-specific files drive runs.
    for name in ('worldengine_navtest_smoke.pkl','worldengine_navtest_smoke.yaml','worldengine_navtest_smoke_pdm.pkl'):
        (output/name).symlink_to(source/name)
    (output/'cameras').symlink_to((source/'cameras').resolve(strict=True),target_is_directory=True)
    manifest['split_files_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in output.iterdir() if x.is_file() and not x.is_symlink()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
