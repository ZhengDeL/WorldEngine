"""Actual independent SDK tool interpreter acceptance; no NPU/kernel compilation."""
import argparse
from email.parser import Parser
import hashlib
from importlib import metadata
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Fresh receipt required')
 r={'result':'running','scope':'Nine declared SDK compiler/tool components, actual imports and pip; not all SDK optional tools or NPU kernels',
  'python':sys.version,'prefix':sys.prefix,'sys_path':list(sys.path),'modules':{},'failures':[],
  'helper_sha256':sha(Path(__file__))}
 try:
  if list(Path('/dev').glob('davinci[0-9]*')):raise RuntimeError('CPU-only container required')
  if sys.version_info[:2]!=(3,11) or sys.prefix!='/opt/worldengine-sdk-tools311':raise RuntimeError('Actual independent CPython3.11 tool interpreter required')
  if any('/opt/conda/envs/bevformer' in x or '/opt/worldengine-dependency-fix' in x for x in sys.path):raise RuntimeError('Application packages leaked into tool environment')
  from packaging.tags import parse_tag,sys_tags
  te=metadata.distribution('te');raw=te.read_text('WHEEL')
  declared=Parser().parsestr(raw or '').get_all('Tag',[])
  tags=set().union(*(parse_tag(x) for x in declared)) if declared else set()
  r['TE_wheel_tags']={'declared':declared,'current_compatible':bool(tags&set(sys_tags())),
   'WHEEL_sha256':hashlib.sha256((raw or '').encode()).hexdigest()}
  if not r['TE_wheel_tags']['current_compatible']:raise RuntimeError('Actual TE wheel tag and tool interpreter do not match')
  lock=json.loads(Path('/opt/worldengine-environments/SDK_TOOLS_WHEELS.lock.json').read_text())
  versions={x['name']:x['version'] for x in lock['artifacts']}
  versions.update({'te': '0.4.0', 'ms-service-profiler': '26.0.0', 'affinity-sched': '26.1.0', 'asc-opc-tool': '0.1.0', 'asc-op-compile-base': '0.1.0', 'opc-tool': '0.1.0', 'op-compile-tool': '0.1.0', 'auto-tune': '0.1.0', 'schedule-search': '0.0.1'})
  r['actual_versions']={name:metadata.version(name) for name in versions}
  if r['actual_versions']!=versions:raise RuntimeError('Tool package versions differ from explicit lock')
  r['SDK_metadata_origins']={name:str(Path(getattr(metadata.distribution(name),'_path','')).resolve())
   for name in ('te', 'ms-service-profiler', 'affinity-sched', 'asc-opc-tool', 'asc-op-compile-base', 'opc-tool', 'op-compile-tool', 'auto-tune', 'schedule-search')}
  for path in r['SDK_metadata_origins'].values():
   if not Path(path).is_relative_to(Path('/usr/local/Ascend/cann-9.1.0').resolve()):raise RuntimeError('SDK distribution did not resolve to unchanged installed SDK metadata')
  for name in ('numpy','scipy','sympy','mpmath','te','te.tvm','tvm','tbe','ms_service_profiler',
   'msguard','pydantic','loguru','openpyxl','asc_op_compile_base','asc_opc_tool','opc_tool','op_compile_tool','auto_tune','schedule_search','opentelemetry.exporter.otlp.proto.grpc.trace_exporter',
   'opentelemetry.exporter.otlp.proto.http.trace_exporter'):
   try:
    m=importlib.import_module(name)
    r['modules'][name]={'result':'imported','file':getattr(m,'__file__',None),'version':getattr(m,'__version__',None)}
   except BaseException as error:
    r['modules'][name]={'result':'failed','error':repr(error),'traceback':traceback.format_exc()};r['failures'].append('Tool import failed: '+name)
  r['sys_path_after_runtime_imports']=list(sys.path)
  if any('/opt/conda/envs/bevformer' in x or '/opt/worldengine-dependency-fix' in x for x in sys.path):raise RuntimeError('Application packages leaked during SDK tool imports')
  r['visible_SDK_metadata_after_imports']=[{'name':d.metadata['Name'],'version':d.version,'path':str(getattr(d,'_path',''))}for d in metadata.distributions()if str(getattr(d,'_path','')).startswith(('/usr/local/Ascend','/opt/worldengine-sdk-components'))]
  import sympy
  import mpmath
  variable=sympy.Symbol('worldengine_compiler_probe')
  if sympy.expand((variable+1)**2)!=variable**2+2*variable+1 or mpmath.sqrt(4)!=2:raise RuntimeError('Actual symbolic/multiprecision tool API failed')
  r['actual_symbolic_APIs']={'sympy':sympy.__version__,'mpmath':mpmath.__version__,'polynomial_identity':True,'multiprecision_sqrt':True}
  from msguard import Rule,validate_params,validate_args
  r['msguard_official_consumer_API']={'Rule':bool(Rule),'validate_params':callable(validate_params),'validate_args':callable(validate_args)}
  from google.protobuf import __version__ as protobuf_actual
  from google.protobuf.struct_pb2 import Struct
  if protobuf_actual!='5.29.5':raise RuntimeError('Actual tool protobuf module differs from tool lock')
  one=Struct();one.update({'probe':1});two=Struct();two.ParseFromString(one.SerializeToString())
  if one!=two:raise RuntimeError('Real SDK tool protobuf5 message roundtrip failed')
  r['actual_tool_protobuf_roundtrip']={'version':protobuf_actual,'pass':True}
  import argparse as std_argparse
  import sysconfig
  if Path(std_argparse.__file__).resolve()!=Path(sysconfig.get_path('stdlib')).resolve()/'argparse.py':raise RuntimeError('Backport argparse shadows actual stdlib')
  if any(x.startswith(('torch','torch_npu','mx_driving')) for x in sys.modules):raise RuntimeError('SDK CPU acceptance unexpectedly imported accelerator framework')
  before=json.loads(Path('/opt/worldengine-environments/sdk_component_view.json').read_text())
  for record in before['original_before'].values():
   for path,item in record['files'].items():
    source=Path(path)
    if source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:raise RuntimeError('Original SDK payload bytes changed: '+path)
  r['original_SDK_components_unchanged']=True
  r['original_RECORD_relocations']=before['RECORD_relocation_policy']
  for item in before['RECORD_relocation_policy']['relocations'].values():
   if Path(item['original_declared_location']).exists() or sha(Path(item['actual_path']))!=item['sha256']:raise RuntimeError('Original SDK missing-path or verified relocation baseline changed')
  profiler=before['independent_profiler_launcher'];profiler_path=Path(profiler['path'])
  if sha(profiler_path)!=profiler['sha256']:raise RuntimeError('Independent profiler launcher source changed')
  profiler_done=subprocess.run([str(profiler_path),'--help'],text=True,capture_output=True,timeout=45)
  r['actual_profiler_entrypoint_help']={'command':[str(profiler_path),'--help'],'actual_exit':profiler_done.returncode,'stdout':profiler_done.stdout,'stderr':profiler_done.stderr,
   'original_publisher_script':profiler['original_script'],'original_RECORD_path_still_missing':True,'scope':'Independent real3.11 launcher invokes exact original publisher script; no profiler collection acceptance'}
  if profiler_done.returncode:raise RuntimeError('Actual relocated original profiler entrypoint --help failed')
  opc=Path('/usr/local/Ascend/cann-9.1.0/compiler/bin/asc_opc')
  if sha(opc)!='4400fcf5a7628e549787f84852ad54680bc5ff2217a02e17a76a24be4ba3bf2c':raise RuntimeError('Original asc_opc source identity changed')
  if os.environ.get('HI_PYTHON')!='/opt/worldengine-environments/compiler_python311.sh':raise RuntimeError('Compiler interpreter selection differs')
  command=[str(opc),'--help']
  help_done=subprocess.run(command,text=True,capture_output=True,timeout=45)
  r['original_asc_opc_help']={'command':command,'actual_exit':help_done.returncode,
   'stdout':help_done.stdout,'stderr':help_done.stderr,'original_source_sha256':sha(opc),
   'HI_PYTHON':os.environ.get('HI_PYTHON'),'compiled_kernel':False}
  if help_done.returncode:raise RuntimeError('Actual original compiler --help failed')
 except BaseException as error:r['failures'].append(repr(error));r['traceback']=traceback.format_exc()
 try:
  proc=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True,timeout=60)
  r['pip_check']={'actual_exit':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
  if proc.returncode:r['failures'].append('Actual SDK tool pip check failed')
 except BaseException as error:r['failures'].append('pip check incomplete: '+repr(error))
 r['result']='pass_actual_selected_SDK_tools_CPU_acceptance' if not r['failures'] else 'failed_selected_SDK_tools_CPU_acceptance'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'result':r['result'],'failures':r['failures']}))
 return bool(r['failures'])

if __name__=='__main__':raise SystemExit(main())
