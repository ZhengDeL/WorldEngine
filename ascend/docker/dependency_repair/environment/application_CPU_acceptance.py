"""Accept the actual separated application environment only; tool env has a separate gate."""
import argparse
import hashlib
from importlib import metadata
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('Fresh acceptance receipt required')
    if list(Path('/dev').glob('davinci[0-9]*')):raise RuntimeError('CPU-only container required')
    result={'result':'running','scope':'Actual application Python environment; not compiler/profiler/NPU acceptance',
      'sys_path':list(sys.path),'modules':{},'failures':[],
      'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        import worldengine_application_environment as env
        result['environment_startup']=env.STATUS
        if not env.STATUS['active'] or any(env.sdk_python_path(x) for x in sys.path):raise RuntimeError('Application environment not actually separated')
        if any(x.startswith(('torch','mx_driving')) for x in sys.modules):raise RuntimeError('Unexpected framework preload before actual acceptance imports')
        for name in ('te','ms-service-profiler','affinity-sched'):
            try:metadata.distribution(name)
            except metadata.PackageNotFoundError:continue
            raise RuntimeError('Compiler/profiler SDK distribution is still in the application environment: '+name)
        for name in ('numpy','cv2','google.protobuf','google.auth','diffusers','huggingface_hub',
          'safetensors','hf_xet','tensorboard','pyproj','rasterio','fiona','shapely','geopandas','nuplan','torch','torch_npu','mmcv','mmcv._ext'):
            try:
                m=importlib.import_module(name)
                result['modules'][name]={'result':'imported','file':getattr(m,'__file__',None),'version':getattr(m,'__version__',None)}
            except BaseException as error:
                result['modules'][name]={'result':'failed','error':repr(error)}
                result['failures'].append('Runtime import: '+name)
        import torch
        import torch_npu
        result['NPU_initialized']=torch_npu.npu.is_initialized();result['CUDA_initialized']=torch.cuda.is_initialized()
        if result['NPU_initialized'] or result['CUDA_initialized']:raise RuntimeError('Unexpected device initialization')
        result['actual_versions']={n:metadata.version(n) for n in ('torch','torch-npu','mx-driving','mmcv-full','numpy','protobuf','opencv-python')}
        expected={'torch':'2.7.1+cpu','torch-npu':'2.7.1.post8','mx-driving':'26.1.0.post2','mmcv-full':'1.7.2','numpy':'1.23.5','protobuf':'4.25.8','opencv-python':'4.11.0.86'}
        if result['actual_versions']!=expected:raise RuntimeError('Protected/application versions differ')
        original=json.loads(Path('/opt/worldengine-dependency-repair/apply_receipt.json').read_text())
        if not original['protected_identical'] or original['protected_before']!=original['protected_after']:
            raise RuntimeError('Parent dependency repair original protection did not pass')
        files={path:digest for item in original['protected_after'].values() for path,digest in item['files'].items()}
        for filename,wanted in files.items():
            value=hashlib.sha256()
            with Path(filename).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
            if value.hexdigest()!=wanted:raise RuntimeError('Original protected source/native file changed: '+filename)
        result['original_protected_payloads_unchanged']={'files':len(files),'pass':True}
        # Reuse the frozen, exact full/headless layout and real RECORD payload gate.
        import minimal_dependency_reference as original_gate
        result['OpenCV_distinct_prefix_payloads']=original_gate.opencv_layout()
        import numpy as np
        import cv2
        pixels=np.arange(48,dtype=np.uint8).reshape(4,4,3)
        ok,encoded=cv2.imencode('.jpg',pixels)
        if not ok or cv2.imdecode(encoded,cv2.IMREAD_COLOR).shape!=(4,4,3):raise RuntimeError('Actual cv2 JPEG roundtrip failed')
        if cv2.resize(pixels,(2,2)).shape!=(2,2,3):raise RuntimeError('Actual cv2 resize failed')
        from google.protobuf.struct_pb2 import Struct
        one=Struct();one.update({'probe':1});two=Struct();two.ParseFromString(one.SerializeToString())
        if one!=two:raise RuntimeError('Actual protobuf4 roundtrip failed')
        result['actual_OpenCV_protobuf_roundtrips']=True
        import geospatial_CPU_acceptance
        result['actual_geospatial_APIs']=geospatial_CPU_acceptance.run_checks()
    except BaseException as error:result['failures'].append(repr(error));result['traceback']=traceback.format_exc()
    try:
        done=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=60)
        result['pip_check']={'actual_exit':done.returncode,'stdout':done.stdout,'stderr':done.stderr}
        if done.returncode:result['failures'].append('Actual application pip check failed')
    except BaseException as error:result['failures'].append('pip check incomplete: '+repr(error))
    result['result']='pass_actual_application_environment_CPU_only' if not result['failures'] else 'failed_application_environment_acceptance'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'result':result['result'],'failures':result['failures']}))
    return 0 if not result['failures'] else 1

if __name__=='__main__':raise SystemExit(main())
