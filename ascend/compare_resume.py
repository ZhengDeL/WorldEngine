"""Compare uninterrupted and resumed training artifacts on CPU."""
import argparse,json,math
from pathlib import Path
import torch


def finite_numeric(value):
    if type(value) in (int, float):return math.isfinite(value)
    if isinstance(value, dict):return all(finite_numeric(item) for item in value.values())
    if isinstance(value, (list, tuple)):return all(finite_numeric(item) for item in value)
    return False


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):return None
    if isinstance(value, dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):return [json_safe(item) for item in value]
    return value


def raw_tensor_equal(first, second):
    """Compare logical tensor bytes after the original shape/dtype checks."""
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    first_bytes = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    second_bytes = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(first_bytes, second_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--continuous',type=Path,required=True)
    p.add_argument('--resumed',type=Path,required=True)
    p.add_argument('--continuous-run',type=Path,required=True)
    p.add_argument('--partial-run',type=Path,required=True)
    p.add_argument('--resumed-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comparison-updates',type=int,help='Compare a saved prefix for an independent-repeat control')
    a=p.parse_args()
    first=torch.load(a.continuous,map_location='cpu',weights_only=False)
    second=torch.load(a.resumed,map_location='cpu',weights_only=False)
    stats=dict(tensors=0,bitwise_equal_tensors=0,max_absolute_difference=0.,failures=[])
    def compare(x,y,path):
        if torch.is_tensor(x) or torch.is_tensor(y):
            stats['tensors']+=1
            if not (torch.is_tensor(x) and torch.is_tensor(y)):
                stats['failures'].append(path+': tensor type');return
            if x.shape!=y.shape:stats['failures'].append(path+': shape');return
            if x.dtype!=y.dtype:stats['failures'].append(path+': dtype');return
            if not (bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())):
                stats['failures'].append(path+': nonfinite');return
            equal=torch.equal(x,y)
            stats['bitwise_equal_tensors']+=int(raw_tensor_equal(x,y))
            if x.is_floating_point():
                diff=float((x-y).abs().max()) if x.numel() else 0.
                stats['max_absolute_difference']=max(stats['max_absolute_difference'],diff)
                okay=torch.allclose(x,y,atol=1e-6,rtol=1e-5)
            else:okay=equal
            if not okay:stats['failures'].append(path)
        elif isinstance(x,dict):
            if set(x)!=set(y):stats['failures'].append(path+': keys');return
            for k in x:compare(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):stats['failures'].append(path+': length');return
            for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not (math.isfinite(x) and math.isfinite(y)):stats['failures'].append(path+': nonfinite scalar')
            elif x!=y:stats['failures'].append(path)
        elif x!=y:stats['failures'].append(path)
    compare(first['state_dict'],second['state_dict'],'model')
    model_stats=dict(stats,failures=list(stats['failures']))
    compare(first['optimizer'],second['optimizer'],'optimizer')
    states=[x['meta']['ascend_resume'] for x in (first,second)]
    for key in ('contract','completed_updates','completed_epochs'):compare(states[0][key],states[1][key],'resume/'+key)
    rng_coverage=True
    worlds=[]
    for side,state in zip(('continuous','resumed'),states):
        world=state['contract'].get('world_size')
        rngs=state.get('rng_by_rank')
        valid=type(world) is int and world>0 and isinstance(rngs,list) and len(rngs)==world
        worlds.append(world)
        if not valid:
            stats['failures'].append('rng/'+side+': incomplete contract world_size coverage')
            rng_coverage=False
    if rng_coverage and worlds[0]!=worlds[1]:
        stats['failures'].append('rng: world_size differs')
        rng_coverage=False
    rng_before=len(stats['failures'])
    if rng_coverage:
        for state in states:
            for rng in state['rng_by_rank']:
                numpy=rng['numpy']
                if not isinstance(numpy,dict):
                    rng['numpy']=dict(generator=numpy[0],keys=numpy[1].tolist(),position=numpy[2],has_gauss=numpy[3],cached_gaussian=numpy[4])
        compare(states[0]['rng_by_rank'],states[1]['rng_by_rank'],'rng')
    rng_match=rng_coverage and len(stats['failures'])==rng_before
    completed=states[1]['completed_updates']
    target=completed if a.comparison_updates is None else a.comparison_updates
    horizon_valid=type(completed) is int and completed>0 and type(target) is int and 1<=target<=completed
    if not horizon_valid:stats['failures'].append('logs: comparison_updates must be 1..completed_updates')
    expected_sequence=list(range(1,target+1)) if horizon_valid else []
    sequences_complete=rng_coverage and horizon_valid
    loss_max=0.
    token_match,lr_match=True,True
    invalid_numeric_records=[]
    for rank in range(worlds[0] if rng_coverage and horizon_valid else 0):
        def read(root):return [json.loads(x) for x in (root/f'runtime_rank{rank}.jsonl').read_text().splitlines()]
        baseline=read(a.continuous_run)
        tail=read(a.resumed_run)
        if a.comparison_updates is not None:
            baseline=[x for x in baseline if x['iter']<=a.comparison_updates]
            tail=[x for x in tail if x['iter']<=a.comparison_updates]
        if not tail:
            stats['failures'].append(f'logs/rank{rank}: empty resumed tail')
            sequences_complete=False
            continue
        resumed=[x for x in read(a.partial_run) if x['iter']<tail[0]['iter']]+tail
        if [x['iter'] for x in baseline]!=expected_sequence or [x['iter'] for x in resumed]!=expected_sequence:
            stats['failures'].append(f'logs/rank{rank}: incomplete full update sequence')
            sequences_complete=False
        for u,v in zip(baseline,resumed):
            token_match=token_match and u['tokens']==v['tokens']
            lr_match=lr_match and u['lr']==v['lr']
            valid_loss=True
            for side,record in (('continuous',u),('resumed',v)):
                for field in ('loss','lr'):
                    okay=(type(record[field]) in (int,float) and math.isfinite(record[field])) if field=='loss' else finite_numeric(record[field])
                    if not okay:
                        stats['failures'].append(f'logs/rank{rank}/iter{record["iter"]}/{side}/{field}: nonfinite or nonnumeric')
                        invalid_numeric_records.append(dict(rank=rank,iteration=record['iter'],side=side,field=field))
                        if field=='loss':valid_loss=False
            if valid_loss:loss_max=max(loss_max,abs(u['loss']-v['loss']))
    row=dict(result='pass' if not stats['failures'] and token_match and lr_match and loss_max<1e-4 else 'fail',
             model=model_stats,model_and_optimizer=stats,max_loss_difference=loss_max,
             identical_batch_tokens=token_match,identical_learning_rates=lr_match,
             identical_final_rng_states=rng_match,
             completed_updates=states[1]['completed_updates'],world_size=len(states[1]['rng_by_rank']) if isinstance(states[1].get('rng_by_rank'),list) else None,
             all_rank_RNG_coverage_complete=rng_coverage,full_update_sequences_complete=sequences_complete,required_update_sequence=expected_sequence,
             tensor_tolerance=dict(atol=1e-6,rtol=1e-5),loss_absolute_tolerance=1e-4,
             all_loss_and_learning_rate_records_finite=not invalid_numeric_records,invalid_numeric_records=invalid_numeric_records,
             nonfinite_statistics_encoded_as_null=True)
    a.output.write_text(json.dumps(json_safe(row),indent=2,allow_nan=False))
    print(json.dumps(json_safe(row),indent=2,allow_nan=False))
    if row['result']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
