"""Compare uninterrupted and resumed training artifacts on CPU."""
import argparse,json
from pathlib import Path
import torch


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
        if torch.is_tensor(x):
            stats['tensors']+=1
            if x.shape!=y.shape:stats['failures'].append(path+': shape');return
            equal=torch.equal(x,y)
            stats['bitwise_equal_tensors']+=int(equal)
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
        elif x!=y:stats['failures'].append(path)
    compare(first['state_dict'],second['state_dict'],'model')
    model_stats=dict(stats)
    compare(first['optimizer'],second['optimizer'],'optimizer')
    states=[x['meta']['ascend_resume'] for x in (first,second)]
    for key in ('contract','completed_updates','completed_epochs'):compare(states[0][key],states[1][key],'resume/'+key)
    for state in states:
        for rng in state['rng_by_rank']:
            numpy=rng['numpy']
            if not isinstance(numpy,dict):
                rng['numpy']=dict(generator=numpy[0],keys=numpy[1].tolist(),position=numpy[2],has_gauss=numpy[3],cached_gaussian=numpy[4])
    rng_before=len(stats['failures'])
    compare(states[0]['rng_by_rank'],states[1]['rng_by_rank'],'rng')
    rng_match=len(stats['failures'])==rng_before
    loss_max=0.
    token_match,lr_match=True,True
    for rank in range(len(states[0]['rng_by_rank'])):
        def read(root):return [json.loads(x) for x in (root/f'runtime_rank{rank}.jsonl').read_text().splitlines()]
        baseline=read(a.continuous_run)
        tail=read(a.resumed_run)
        if a.comparison_updates:
            baseline=[x for x in baseline if x['iter']<=a.comparison_updates]
            tail=[x for x in tail if x['iter']<=a.comparison_updates]
        resumed=[x for x in read(a.partial_run) if x['iter']<tail[0]['iter']]+tail
        assert [x['iter'] for x in baseline]==[x['iter'] for x in resumed]
        for u,v in zip(baseline,resumed):
            token_match=token_match and u['tokens']==v['tokens']
            lr_match=lr_match and u['lr']==v['lr']
            loss_max=max(loss_max,abs(u['loss']-v['loss']))
    row=dict(result='pass' if not stats['failures'] and token_match and lr_match and loss_max<1e-4 else 'fail',
             model=model_stats,model_and_optimizer=stats,max_loss_difference=loss_max,
             identical_batch_tokens=token_match,identical_learning_rates=lr_match,
             identical_final_rng_states=rng_match,
             completed_updates=states[1]['completed_updates'],world_size=len(states[1]['rng_by_rank']),
             tensor_tolerance=dict(atol=1e-6,rtol=1e-5),loss_absolute_tolerance=1e-4)
    a.output.write_text(json.dumps(row,indent=2))
    print(json.dumps(row,indent=2))
    if row['result']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
