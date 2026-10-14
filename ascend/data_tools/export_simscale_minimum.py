"""Export a genuine, explicitly filtered SimScale/NAVTRAIN train-only subset.

Default invocation prints a plan. CPU execution copies original records, camera
bytes and PDM entries. Derived labels use the original WE four-second logged
traffic protocol; no equivalence with published SimScale v2 is claimed.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import signal
import sys
import tarfile
import time

GIB=1024**3
FOLDER='synthetic_reaction_pdm_v1.0-0'
LOG='2021.06.09.17.23.18_veh-38_00773_01140'
METRICS=['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score']
CONFIGS={'finetune':'e2e_vadv2_simscale_finetune_pdm_1round_8epoch','cotrain':'e2e_vadv2_simscale_co_train_pdm_1round_8epoch'}
LOCKS={
 'annotation':(16125493728,'ccfdf2484c147c05246fe0bb3dbd51f8cc14b8d70e8f3df09f10057febf1c92b'),
 'pdm':(23736976138,'6258895288d276bd4f57550400ec599df2ad0a6f2f10d5661c6589e9991834ca'),
 'camera_archive':(2781004086,'4b7f59cfdedede8405cd7ae2828179a7a79e96b26c7b1d6da86dad698b31677e'),
}
META_SHA='a7a6d725d926d1200410e5e9b8c39368b5b519bba6535816bf528bcc965331da'
HISTORY_SHA='25ab91a08a38e7a7270ed983b51e01fc6d6f02bbb955ef8f4abdfcccfc9fc48c'
VOCAB_SHA='cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35'
RAWMINI_SHA='c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()


def fingerprint(source):
 h=hashlib.sha256()
 for p in sorted([*(source/'projects/AlgEngine').rglob('*.py'),*(source/'ascend').glob('*.py')],key=lambda x:x.relative_to(source).as_posix()):
  h.update(p.relative_to(source).as_posix().encode());h.update(p.read_bytes().replace(b'\r\n',b'\n'))
 return h.hexdigest()


def relative(raw):
 p=PurePosixPath(raw)
 if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in raw:raise ValueError('Unsafe relative input path')
 return p


def safe_file(path):
 if path.is_symlink() or not path.is_file():raise ValueError('Expected regular input file')
 return path


def persist(path,row):
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)


def output_record(path,output):
 return dict(path=path.relative_to(output).as_posix(),bytes=path.stat().st_size,sha256=sha(path))


def write_pickle(path,value,output):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:pickle.dump(value,f,protocol=4)
 return output_record(path,output)


def copy_verified(source,dest,digest,output):
 safe_file(source)
 if sha(source)!=digest or dest.exists():raise ValueError('Source lock or fresh copy destination differs')
 dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as left,dest.open('xb') as right:shutil.copyfileobj(left,right,8*1024**2)
 if sha(source)!=digest or sha(dest)!=digest:raise ValueError('Copy changed original bytes')
 return output_record(dest,output)


def chain(args):
 conversion_path=safe_file(args.conversion_root/'conversion_manifest.json')
 history_path=safe_file(args.conversion_root/'history_manifest.json')
 reward_path=safe_file(args.reward_root/'reward_pilot.json')
 conversion=json.loads(conversion_path.read_text());history=json.loads(history_path.read_text());reward=json.loads(reward_path.read_text())
 tools=Path(__file__).resolve().parent
 reward_tool=tools/'worldengine_8192_logged_reward.py'
 expected_math=ast.literal_eval(next(node.value for node in ast.parse(reward_tool.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='LOCKS' for target in node.targets)))
 actual_math={name:hashlib.sha256((args.source/'projects/SimEngine/worldengine'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in expected_math}
 if actual_math!=expected_math or reward.get('source_locks')!=expected_math:raise ValueError('Original scoring/simulation math source differs from the current reward receipt')
 if any(q.get('helper_sha256')!=sha(tools/name) for q,name in [(conversion,'simscale_convert.py'),(history,'simscale_convert.py'),(reward,'worldengine_8192_logged_reward.py')]):raise ValueError('Current preparation tools do not match input receipts')
 for q in [conversion,history]:
  if any(q.get(key)!=args.expected_runtime_sha256 for key in ['source_python_sha256','source_python_sha256_after']):raise ValueError('Conversion/history runtime source differs')
 if conversion.get('result')!='pass_original_converter_26_real_clips' or len(conversion.get('clips',[]))!=26 or conversion.get('archive_sha256_after')!=META_SHA or conversion.get('inputs',{}).get('metadata',{}).get('sha256')!=META_SHA:raise ValueError('Locked genuine conversion receipt missing')
 if history.get('result')!='pass_all_real_history_camera_references' or history.get('actual_image_count')!=832 or history.get('archive_sha256_after')!=HISTORY_SHA or history.get('inputs',{}).get('history',{}).get('sha256')!=HISTORY_SHA or history.get('inputs',{}).get('conversion_manifest_sha256')!=sha(conversion_path):raise ValueError('Genuine history receipt does not bind current conversion')
 selected=[c for c in conversion['clips'] if c.get('scene_id')==args.scene_id]
 if len(selected)!=1:raise ValueError('One exact genuine scene required')
 clip=selected[0]
 if clip.get('original_log_name')!=args.original_log or clip.get('anchor_token')!=args.expected_anchor_token or clip.get('frame_count')!=12:raise ValueError('Scene/log/anchor identity differs')
 converted=args.conversion_root/relative(clip['converted_relative_path']).as_posix();safe_file(converted)
 if sha(converted)!=clip['converted_sha256']:raise ValueError('Original converter output changed')
 with converted.open('rb') as f:data=pickle.load(f)
 infos=data['infos']
 if not isinstance(infos,list) or len(infos)!=12 or infos[3]['token']!=args.expected_anchor_token:raise ValueError('Converted genuine twelve-frame identity differs')
 if reward.get('result')!='pass_original_WE_scoring_derived_protocol_not_SimScale_v2_equivalence' or any(reward.get(k)!=args.expected_runtime_sha256 for k in ['runtime_source_sha256_before','runtime_source_sha256_after']):raise ValueError('Accepted current WE reward/runtime receipt required')
 if reward.get('metadata_sha256_after')!=RAWMINI_SHA or reward.get('vocab_sha256_after')!=VOCAB_SHA or reward.get('map_input_manifest_before')!=reward.get('map_input_manifest_after') or not reward.get('map_input_manifest_before'):raise ValueError('Reward original metadata/vocabulary/maps locks differ')
 if reward.get('input_binding',{}).get('synthetic_metadata_sha256')!=clip['raw_sha256'] or reward.get('input_binding',{}).get('scene_id')!=args.scene_id or reward.get('reward_token')!=args.expected_anchor_token or reward.get('candidates')!=8192 or reward.get('all_metrics_finite') is not True or reward.get('candidate_row_order')!='unchanged original test_8192_kmeans.npy':raise ValueError('Reward scene or unchanged candidate row order differs')
 protocol=reward.get('protocol',{})
 if protocol.get('proposal_seconds')!=4 or protocol.get('simscale_v2_equivalence_claimed') is not False or protocol.get('full_SimEngine_execution') is not False:raise ValueError('Derived protocol must retain its stated scope')
 if any(reward.get(key)!=dict(npu=False,cuda=False) for key in ['device_initialization_before','device_initialization_after']):raise ValueError('Reward CPU-only initialization receipt missing')
 labels=safe_file(args.reward_root/'worldengine_8192_logged_traffic_derived_rewards.pkl')
 if sha(labels)!=reward.get('output_sha256'):raise ValueError('Current derived reward bytes changed')
 refs=clip['history_camera_references'];images={r['data_path']:r for r in history['images']}
 if len(refs)!=32 or len({r['data_path'] for r in refs})!=32:raise ValueError('Exactly four genuine history frames/eight cameras required')
 return dict(clip=clip,converted=converted,infos=infos,labels=labels,images=images,refs=refs,reward=reward,
  receipt_sha256={k:sha(v) for k,v in [('conversion',conversion_path),('history',history_path),('reward',reward_path)]})


def queues(infos,tokens,log):
 found={token:[] for token in tokens}
 for idx,row in enumerate(infos):
  if row['token'] in found:found[row['token']].append(idx)
 result={}
 for token,indices in found.items():
  if len(indices)!=1 or indices[0]<3:raise ValueError('Selected normal anchor must occur once with original predecessors')
  idx=indices[0];queue=infos[idx-3:idx+1]
  if any(r['log_name']!=log for r in queue) or any(int(b['timestamp'])<=int(a['timestamp']) for a,b in zip(queue,queue[1:])):raise ValueError('Original contiguous queue log/time order differs')
  if any(len(r['cams'])!=8 for r in queue):raise ValueError('Eight original cameras per queue frame required')
  refs=[dict(token=r['token'],camera=k,data_path=v['data_path']) for r in queue for k,v in r['cams'].items()]
  if len({r['data_path'] for r in refs})!=32 or any(relative(r['data_path']).parts[0]!=log for r in refs):raise ValueError('Original real camera identity differs')
  result[token]=dict(raw_index=idx,raw_indices=list(range(idx-3,idx+1)),records=queue,references=refs)
 return result


def audit_archive(path):
 seen=set();count=dirs=expanded=0;h=hashlib.sha256()
 with tarfile.open(path,'r|gz') as archive:
  for member in archive:
   name=relative(member.name).as_posix()
   if not (member.isfile() or member.isdir()):raise ValueError('Links or special archive member forbidden')
   if member.isfile():
    if name in seen:raise ValueError('Duplicate normalized archive file')
    seen.add(name);count+=1;expanded+=member.size
   else:dirs+=1
   h.update((name+'\0'+str(member.size)+'\0'+str(member.type)+'\n').encode())
 return dict(files=count,directories=dirs,expanded_bytes=expanded,inventory_sha256=h.hexdigest(),links_or_special_members=False)


def camera_bytes(path,wanted,destinations,output):
 import cv2
 import numpy as np
 found={}
 with tarfile.open(path,'r|gz') as archive:
  for member in archive:
   if not member.isfile():continue
   hits=[p for p in wanted if member.name==p or member.name.endswith('/'+p)]
   if not hits:continue
   if len(hits)!=1 or hits[0] in found or not 0<member.size<=32*1024**2:raise ValueError('Ambiguous, duplicate or oversized normal image')
   key=hits[0];payload=archive.extractfile(member).read(member.size+1)
   image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
   if len(payload)!=member.size or image is None or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Incomplete original normal JPEG or decode failure')
   records=[]
   for dest in destinations[key]:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(payload)
    records.append(output_record(dest,output))
   found[key]=dict(raw_member=member.name,sha256=hashlib.sha256(payload).hexdigest(),decoded_shape=list(image.shape),files=records)
 if set(found)!=set(wanted):raise ValueError('Published camera archive lacks selected original queue images')
 return found


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['source','annotation','pdm','camera-archive','conversion-root','reward-root','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--expected-runtime-sha256',required=True);p.add_argument('--deadline',required=True)
 p.add_argument('--max-seconds',type=int,default=1200);p.add_argument('--reserve-gib',type=int,default=192)
 p.add_argument('--scene-id',default='fff002cb15c15a7a-000');p.add_argument('--original-log',default=LOG)
 p.add_argument('--folder-name',default=FOLDER);p.add_argument('--expected-anchor-token',default='ab75b7addff3512d-000')
 p.add_argument('--finetune-normal-token',default='4b96572a4b9d5ea7');p.add_argument('--cotrain-normal-token',default='fff002cb15c15a7a')
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not re.fullmatch('[0-9a-f]{64}',a.expected_runtime_sha256) or not 1<=a.max_seconds<=3600 or a.reserve_gib<192 or a.folder_name!=FOLDER:p.error('Exact runtimeSHA, supported folder, bounded duration and >=192GiB reserve required')
 for value in [a.scene_id,a.original_log,a.expected_anchor_token,a.finetune_normal_token,a.cotrain_normal_token]:
  if not re.fullmatch('[A-Za-z0-9_.-]+',value):p.error('Literal safe scene/log/token identities required')
 deadline=datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
 if deadline.tzinfo is None:p.error('Timezone-aware authorized deadline required')
 plan=dict(result='dry_run_not_executed',output=str(a.output),helper_sha256=sha(Path(__file__)),expected_runtime_sha256=a.expected_runtime_sha256,
  scope='One genuine scene; one explicit NAVTRAIN normal anchor per config. Original four-frame records/images/PDM; narrowed normal candidate pool.',
  normal_candidate_pool_changed=True,full_data_sampling_equivalent=False,training_accepted=False,full_navtrain_convergence_accepted=False,
  no_navtest_directories_created=True,published_SimScale_v2_equivalent=False,expected_payload_files=104,subsets=[])
 if not a.execute:print(json.dumps(plan,indent=2));return 0
 if not sys.platform.startswith('linux') or a.output.exists() or list(Path('/dev').glob('davinci[0-9]*')):p.error('Fresh Linux CPU-only output required')
 cutoff=min(deadline.timestamp()-120,time.time()+a.max_seconds)
 if cutoff-time.time()<300:p.error('Sufficient authorized execution time required')
 cap=Path('/sys/fs/cgroup/memory.max');cap=cap if cap.exists() else Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
 limit=cap.read_text().strip();available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 if limit=='max' or int(limit)>128*GIB or available<(128+a.reserve_gib)*GIB:raise MemoryError('Finite <=128GiB container and shared host reserve required')
 a.output.mkdir(parents=True);plan.update(result='running',started_at_utc=datetime.now(timezone.utc).isoformat())
 def expired(_s,_f):raise TimeoutError('Owned subset export deadline')
 signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(max(1,int(cutoff-time.time())))
 try:
  import numpy as np
  if fingerprint(a.source)!=a.expected_runtime_sha256:raise ValueError('Selected runtime source changed')
  plan['input_locks_before']={}
  for key,(size,digest) in LOCKS.items():
   path=safe_file(getattr(a,key))
   if path.stat().st_size!=size or sha(path)!=digest:raise ValueError('Whole publisher input lock differs: '+key)
   plan['input_locks_before'][key]=dict(bytes=size,sha256=digest)
  binding=chain(a);plan['preparation_receipt_sha256']=binding['receipt_sha256']
  plan['tools_sha256']={name:sha(Path(__file__).resolve().parent/name) for name in ['simscale_convert.py','worldengine_8192_logged_reward.py','export_simscale_minimum.py']}
  plan['camera_archive_safety']=audit_archive(a.camera_archive)
  with a.annotation.open('rb') as f:annotation=pickle.load(f)
  infos=annotation['infos'] if isinstance(annotation,dict) else annotation
  if not isinstance(infos,list) or len(infos)!=246165:raise ValueError('Complete original NAVTRAIN frame count differs')
  tokens={'finetune':a.finetune_normal_token,'cotrain':a.cotrain_normal_token}
  normal=queues(infos,set(tokens.values()),a.original_log)
  del annotation,infos;gc.collect()
  with a.pdm.open('rb') as f:pdms=pickle.load(f)
  if not isinstance(pdms,dict) or len(pdms)!=103288:raise ValueError('Complete original NAVTRAIN PDM key count differs')
  destinations={}
  for branch,token in tokens.items():
   entry=pdms[token]
   for metric in METRICS:
    if np.asarray(entry[metric]).shape!=(8192,) or not np.isfinite(np.asarray(entry[metric])).all():raise ValueError('Original normal PDM candidate/finite gate differs')
   selected=normal[token];root=a.output/CONFIGS[branch];data=root/'data'
   records=selected['records'];record_hashes=[hashlib.sha256(pickle.dumps(r,protocol=4)).hexdigest() for r in records]
   item=dict(config=CONFIGS[branch]+'.py',normal_token=token,original_raw_indices=selected['raw_indices'],original_frame_tokens=[r['token'] for r in records],
    original_record_pickle_sha256=record_hashes,original_normal_entry_pickle_sha256=hashlib.sha256(pickle.dumps(entry,protocol=4)).hexdigest(),normal_pool_scope='one explicitly filtered normal anchor; original retrieval still required',files=[])
   anno_path=data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl';pdm_path=data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl'
   item['files'].append(write_pickle(anno_path,records,a.output));item['files'].append(write_pickle(pdm_path,{token:entry},a.output))
   with anno_path.open('rb') as f:again=pickle.load(f)
   with pdm_path.open('rb') as f:again_pdm=pickle.load(f)
   if [hashlib.sha256(pickle.dumps(r,protocol=4)).hexdigest() for r in again]!=record_hashes or list(again_pdm)!=[token] or hashlib.sha256(pickle.dumps(again_pdm[token],protocol=4)).hexdigest()!=item['original_normal_entry_pickle_sha256']:raise ValueError('Original records/PDM serialization changed')
   filt=root/'filters/navtrain_actual_normal_anchor.yaml';filt.parent.mkdir(parents=True)
   with filt.open('x') as f:f.write('log_names:\n  - '+a.original_log+'\ntokens:\n  - '+token+'\n')
   item['files'].append(output_record(filt,a.output))
   for ref in selected['references']:
    dest=data/'raw/openscene-v1.1/sensor_blobs/trainval'/relative(ref['data_path']).as_posix()
    destinations.setdefault(ref['data_path'],[]).append(dest)
   plan['subsets'].append(item)
  del pdms;gc.collect()
  if len(destinations)!=64:raise ValueError('This two-anchor reproduction expects 64 distinct original JPEGs')
  images=camera_bytes(a.camera_archive,set(destinations),destinations,a.output);plan['normal_images']=images
  for item in plan['subsets']:
   prefix=CONFIGS['finetune' if item['normal_token']==a.finetune_normal_token else 'cotrain']+'/'
   item['files'].extend(r for image in images.values() for r in image['files'] if r['path'].startswith(prefix))
   if len(item['files'])!=35:raise ValueError('Each original configuration requires exactly 35 payload files')
  with binding['labels'].open('rb') as f:labels=pickle.load(f)
  if not isinstance(labels,dict) or list(labels)!=[a.expected_anchor_token] or any(np.asarray(v).shape!=(8192,) or not np.isfinite(np.asarray(v)).all() for v in labels[a.expected_anchor_token].values()) or any(m not in labels[a.expected_anchor_token] for m in METRICS):raise ValueError('Unchanged derived labels must contain finite8192 candidates and six original loader metrics')
  shared=a.output/'synthetic';synthetic_files=[]
  converted_dest=shared/'meta_datas_navformer'/a.folder_name/binding['converted'].name
  synthetic_files.append(copy_verified(binding['converted'],converted_dest,binding['clip']['converted_sha256'],a.output))
  for ref in binding['refs']:
   key=relative(ref['data_path']).as_posix();image=binding['images'][key]
   source=a.conversion_root/'synthetic_root/sensor_blobs'/a.folder_name/key
   synthetic_files.append(copy_verified(source,shared/'sensor_blobs'/a.folder_name/key,image['sha256'],a.output))
  synthetic_files.append(copy_verified(binding['labels'],shared/'pdms_pkl'/(a.folder_name+'.pkl'),binding['reward']['output_sha256'],a.output))
  if len(synthetic_files)!=34:raise ValueError('One genuine synthetic clip/32 images/derived reward required')
  plan['synthetic_files']=synthetic_files;plan['derived_reward_sha256']=binding['reward']['output_sha256']
  plan['input_sha256_after']={key:sha(getattr(a,key)) for key in LOCKS}
  if any(plan['input_sha256_after'][key]!=digest for key,(_size,digest) in LOCKS.items()) or fingerprint(a.source)!=a.expected_runtime_sha256 or chain(a)['receipt_sha256']!=binding['receipt_sha256']:raise ValueError('Original inputs/runtime/preparation chain changed during export')
  payloads=[r for item in plan['subsets'] for r in item['files']]+synthetic_files
  if len(payloads)!=104 or len(list(a.output.rglob('*.jpg')))!=96:raise ValueError('104 payloads/96 unchanged JPEGs required')
  plan.update(result='pass_two_genuine_filtered_subset_exports_not_training',source_sha256_after=a.expected_runtime_sha256,payload_files=104,
   original_normal_jpeg_count=64,original_synthetic_jpeg_count=32,original_synthetic_frames=12,publisher_revision='d0e88d858533ca407a3387a26577c083aa7b3617')
 except BaseException as error:
  import traceback
  plan.update(result='failed_retained',error=repr(error),traceback=traceback.format_exc())
 finally:
  import resource
  signal.alarm(0);plan.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   npu_or_cuda_modules_imported=bool('torch_npu' in sys.modules or 'torch.cuda' in sys.modules))
  if plan['npu_or_cuda_modules_imported']:plan.update(result='failed_retained',error='Raw CPU export imported accelerator modules')
  persist(a.output/'subset_export.json',plan)
 print(json.dumps(dict(result=plan['result'],output=str(a.output),error=plan.get('error'))),flush=True)
 return 0 if plan['result'].startswith('pass_') else 1


if __name__=='__main__':raise SystemExit(main())
