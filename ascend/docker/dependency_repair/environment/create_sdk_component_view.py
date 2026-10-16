"""Expose nine complete installed SDK compiler/profiler components in a separate real tool environment."""
import csv
import hashlib
from importlib import metadata
import json
import shlex
from pathlib import Path
import sys

SDK_BASE=Path('/usr/local/Ascend/cann-9.1.0')
SDK=SDK_BASE/'python/site-packages'
VIEW=Path('/opt/worldengine-sdk-components')
RECEIPT=Path('/opt/worldengine-environments/sdk_component_view.json')
SELECTED={'te-0.4.0.dist-info': ('te', '0.4.0', '685dccfba66b97995e4c437f841d825f04082085d2770dae3fd884a88da3a5ff'), 'ms_service_profiler-26.0.0.dist-info': ('ms-service-profiler', '26.0.0', 'e887dee77efde2972915ac695d99fac0267d2a9c71f3bbcef8f6cb8b3fd868bd'), 'affinity_sched-26.1.0.dist-info': ('affinity-sched', '26.1.0', 'b80991a50e648694dd5b6eba84a61fb577fe7b260a0e024641070949bb6bf23e'), 'asc_opc_tool-0.1.0.dist-info': ('asc-opc-tool', '0.1.0', 'ff7afcac97367bd7ef62bf891a0574e3f969923413246a7234fda9500db0d27c'), 'asc_op_compile_base-0.1.0.dist-info': ('asc-op-compile-base', '0.1.0', '21a8b6eec4066fd47730d7f6f1a11d09f2d1f2903a7a99ce1cabdc751b4166a4'), 'opc_tool-0.1.0.dist-info': ('opc-tool', '0.1.0', '606c9f4997740a66e5a937f6cd53aaf52a4b67421ac6f6868295c00f9b03d731'), 'op_compile_tool-0.1.0.dist-info': ('op-compile-tool', '0.1.0', 'c5561cf2c479e2eaab5ac5c6e700884e1c6a48a836534551861cbd5c1f6a7bfd'), 'auto_tune-0.1.0.dist-info': ('auto-tune', '0.1.0', '9ff676801b47fbb8629a3781b1fd35e3bdfc2dcaa7efa035c1a43a37f01cd2a5'), 'schedule_search-0.0.1.dist-info': ('schedule-search', '0.0.1', '8e4513dc93c3ad77cefd6779fd3046fe6223151a9df81b57bf48e40fe794d7e0')}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

POLICY=json.loads(Path('/opt/worldengine-environments/SDK_RECORD_RELOCATIONS.lock.json').read_text())

def resolve_original_payload(directory,relative,source,declared_hash,declared_size):
 key=directory+'::'+relative
 expected=POLICY['relocations'].get(key)
 if expected is None:
  if not source.is_file():raise RuntimeError('Unknown original SDK payload missing: '+str(source))
  return source,None
 if (source.exists() or str(source.resolve())!=expected['original_declared_location'] or declared_hash!=expected['declared_hash'] or declared_size!=expected['declared_size']):
  raise RuntimeError('Exact original missing RECORD declaration/baseline changed: '+key)
 actual=Path(expected['actual_path'])
 if not actual.resolve().is_relative_to(SDK_BASE.resolve()) or not actual.is_file() or actual.stat().st_size!=expected['bytes'] or sha(actual)!=expected['sha256']:
  raise RuntimeError('Exact relocated original publisher payload changed: '+key)
 return actual.resolve(),expected

def original_payload():
 result={};tops=set()
 for directory,(name,version,digest) in SELECTED.items():
  root=SDK/directory
  if sha(root/'METADATA')!=digest:raise RuntimeError('Installed SDK metadata identity changed: '+name)
  d=metadata.Distribution.at(root)
  record=d.read_text('RECORD')
  if d.version!=version or not record:raise RuntimeError('Complete installed SDK component inventory required: '+name)
  declarations={row[0]:(row[1]if len(row)>1 else '',row[2]if len(row)>2 else '')for row in csv.reader(record.splitlines())if row and row[0]}
  paths=[row[0] for row in csv.reader(record.splitlines()) if row and row[0]]
  if not paths or len(paths)!=len(set(paths)):raise RuntimeError('Complete unique SDK RECORD paths required: '+name)
  files={};relocated={}
  # Some Python metadata versions omit missing payloads from d.files. Read every
  # original RECORD row so a missing installed SDK file cannot disappear silently.
  for relative in paths:
   raw=Path(str(relative))
   if '__pycache__' in raw.parts or raw.suffix=='.pyc':continue
   source=Path(d.locate_file(relative)).resolve()
   if not source.is_relative_to(SDK_BASE.resolve()):
    raise RuntimeError('Unexpected installed SDK component payload outside SDK: '+str(source))
   source,relocation=resolve_original_payload(directory,str(relative),source,*declarations[relative])
   if relocation is not None:relocated[str(relative)]=relocation
   files[str(source)]={'bytes':source.stat().st_size,'sha256':sha(source)}
   if raw.parts and raw.parts[0] not in ('.','..'):
    top=raw.parts[0]
    if top.endswith(('.dist-info','.egg-info')) and top!=directory:
     raise RuntimeError('Component inventory includes foreign metadata: '+top)
    tops.add(top)
  if not files:raise RuntimeError('No actual SDK payload files: '+name)
  result[name]={'version':version,'original_metadata':str(root),'files':files,'verified_record_relocations':relocated}
  tops.add(directory)
 return result,tops

def main():
 if VIEW.exists():raise RuntimeError('Fresh SDK component view required')
 before,tops=original_payload();VIEW.mkdir()
 for name in sorted(tops):
  source=SDK/name
  if not source.exists():raise RuntimeError('SDK top-level component missing: '+name)
  (VIEW/name).symlink_to(source,target_is_directory=source.is_dir())
 selected=sorted(d.metadata['Name'].lower().replace('_','-') for d in metadata.distributions(path=[str(VIEW)]))
 if selected!=sorted(x[0] for x in SELECTED.values()):raise RuntimeError('Tool environment exposes an unexpected SDK distribution')
 console=Path('/opt/worldengine-sdk-tools311/bin/msserviceprofiler')
 if console.exists():raise RuntimeError('Fresh independent profiler launcher required')
 original=POLICY['relocations']['ms_service_profiler-26.0.0.dist-info::../../bin/msserviceprofiler']['actual_path']
 launcher='#!/bin/sh\nset -eu\nexec /opt/worldengine-environments/compiler_python311.sh '+shlex.quote(original)+' "$@"\n'
 console.write_bytes(launcher.encode());console.chmod(0o755)
 after,_=original_payload()
 if before!=after:raise RuntimeError('SDK source bytes changed while creating environment')
 result={'result':'component_view_created_original_SDK_bytes_unchanged',
  'scope':'Nine original compiler/TE/profiler/affinity components; other optional SDK tools remain outside this environment',
  'RECORD_relocation_policy':POLICY,'independent_profiler_launcher':{'path':str(console),'sha256':sha(console),'original_script':original},'selected_distributions':selected,'view':str(VIEW),'links':{x:str((VIEW/x).readlink()) for x in sorted(tops)},
  'original_before':before,'original_after':after,'SDK_unchanged':True,
  'framework_modules_imported':[x for x in sys.modules if x.startswith(('torch','mx_driving'))]}
 if result['framework_modules_imported']:raise RuntimeError('Unexpected framework preload')
 RECEIPT.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
