"""Measure deserialization RAM of available official full NAVTEST PKLs."""
import argparse,gc,json,os,pickle,resource,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--pdm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    def record(stage,obj=None):
        rows.append(dict(stage=stage,rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         items=len(obj) if obj is not None else None,at=time.time()))
        print(json.dumps(rows[-1]),flush=True)
    record('baseline')
    start=time.monotonic()
    annotations=pickle.load(a.annotations.open('rb'))
    record('annotations_loaded',annotations.get('infos',annotations) if isinstance(annotations,dict) else annotations)
    pdm=pickle.load(a.pdm.open('rb'))
    record('annotations_and_pdm_loaded',pdm)
    report=dict(rows=rows,seconds=time.monotonic()-start,annotation_bytes=a.annotations.stat().st_size,
                pdm_bytes=a.pdm.stat().st_size,scope='Single CPU process loading complete available NAVTEST caches; NAVTRAIN and concurrent full training remain unmeasured')
    a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
