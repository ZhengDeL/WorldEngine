"""Selectively download pinned official planners; resume ranges and verify hashes."""
import argparse,hashlib,json,time,urllib.request,urllib.parse,sys
from pathlib import Path

REVISION='d0e88d858533ca407a3387a26577c083aa7b3617'
PLANNERS=[('vadv2','e2e_vadv2_50pct_ep8.pth',434885591,'3f8efa98c9874c54ad1e8e547136cf6f4c92fd2c24454d4dbe39b538e8c64b98'),
          ('hydramdp','e2e_hydramdp_50pct_ep8.pth',450763271,'7e39dcd2d64e3bca32cbeca3a3e127de736a505c220ef47a24d854b4cf093065')]
WARMSTARTS=[('', 'bevformerv2-r50-t1-base_epoch_48.pth',244021691,'2a00b5656aa14c578896ac9a6ccac84c80587b17b3a7250c7e7f5a55869eb1a8'),
            ('','track_map_nuplan_r50_navtrain_50pct_bs1x8.pth',319956525,'2a81595fcee9df0ef93e9f76d7e735752786802f8a907c0c06949afb03ecea7a'),
            ('','track_map_nuplan_r50_navtrain_100pct_bs1x8.pth',319958048,'9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a')]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(url,destination,size,sha):
    destination=Path(destination)
    destination.parent.mkdir(exist_ok=True,parents=True)
    partial=destination.with_suffix('.part')
    if destination.exists():
        if destination.stat().st_size==size and (sha is None or digest(destination)==sha):return destination
        if not partial.exists() and destination.stat().st_size<size:
            destination.replace(partial)
        else:destination.replace(destination.with_suffix('.rejected'))
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise RuntimeError('Partial exceeds official file size')
    failures=0
    while offset<size:
        end=min(offset+32*1024*1024,size)-1
        # Public resolve endpoint, with a fresh redirect per requested range.
        request_url=url+('?download=true&offset='+str(offset) if 'huggingface.co/' in url else '')
        request=urllib.request.Request(request_url,headers={'Range':f'bytes={offset}-{end}'})
        previous=offset
        try:
            with urllib.request.urlopen(request,timeout=120) as src:
                if src.status==200 and offset==0:
                    end=size-1
                elif src.status!=206 or not src.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Server did not honor the requested byte range')
                with partial.open('ab') as out:
                    while offset<=end:
                        block=src.read(min(1024*1024,end-offset+1))
                        if not block:break
                        out.write(block)
                        offset+=len(block)
            if offset==previous:raise RuntimeError('Empty range response')
            failures=0
            print(destination.name,offset,'/',size,flush=True)
        except Exception as e:
            failures+=1
            print(destination.name,'range_retry',type(e).__name__,str(e),flush=True)
            if failures>=5:raise
            time.sleep(min(failures*2,10))
    actual=digest(partial)
    if sha is not None and actual!=sha:raise RuntimeError('Publisher checksum mismatch: '+destination.name)
    partial.replace(destination)
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group',choices=['planners','warmstarts','all'],default='planners')
    p.add_argument('--source',choices=['hf-mirror','modelscope','huggingface'],default='hf-mirror')
    a=p.parse_args()
    if not sys.platform.startswith('linux'):p.error('Run dataset/weight downloads on the designated Linux server')
    rows=[]
    assets=PLANNERS if a.group=='planners' else WARMSTARTS if a.group=='warmstarts' else PLANNERS+WARMSTARTS
    for family,name,size,sha in assets:
        asset='data/alg_engine/ckpts/'+(family+'/' if family else '')+name
        if a.source=='modelscope':
            query=urllib.parse.urlencode(dict(Source='SDK',Revision='master',FilePath=asset,View=False))
            url='https://www.modelscope.cn/api/v1/datasets/OpenDriveLab/WorldEngine/repo?'+query
        else:
            host='hf-mirror.com' if a.source=='hf-mirror' else 'huggingface.co'
            url=f'https://{host}/datasets/OpenDriveLab/WorldEngine/resolve/{REVISION}/{asset}'
        path=fetch(url,a.output/name,size,sha)
        rows.append(dict(name=name,url=url,bytes=path.stat().st_size,sha256=digest(path)))
    (a.output/'official_planners.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
