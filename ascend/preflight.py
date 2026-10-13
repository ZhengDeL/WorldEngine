"""Fail closed on incomplete AlgEngine inputs, before loading a model on NPU.

Run in the locked CPU environment. Large PKLs are deliberately inspected in one
process so the report records actual deserialization RSS rather than file size.
"""
import argparse,csv,gc,hashlib,json,os,pickle,resource,time
from pathlib import Path
import numpy as np
import yaml
from mmcv import Config

ROOT=Path(__file__).resolve().parents[1]
ALG=ROOT/'projects/AlgEngine'
PDM_KEYS=('no_at_fault_collisions','drivable_area_compliance','ego_progress',
          'time_to_collision_within_bound','comfort','score')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def memory():
    return dict(rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def inspect_phase(ds,ann,split,image_root,pdm_path,errors,retain,include_test_history=False):
    phase=dict(annotations=str(ann),filter=str(split),image_root=str(image_root))
    if not ann.is_file() or not split.is_file():return phase
    filt=yaml.safe_load(split.read_text())
    requested=set(filt.get('tokens',filt.get('scenario_tokens',[])))
    if not requested:errors.append('Empty token filter: '+str(split))
    started=time.monotonic()
    obj=pickle.load(ann.open('rb'))
    infos=obj['infos'] if isinstance(obj,dict) else obj
    retain.append(obj)
    selected=[i for i,info in enumerate(infos) if info['token'] in requested]
    present={infos[i]['token'] for i in selected}
    missing=requested-present
    if missing:errors.append(f'{split.name}: {len(missing)} requested tokens absent from annotations; examples {sorted(missing)[:10]}')
    if len(present)!=len(selected):errors.append('Duplicate selected annotation tokens: '+str(ann))
    # Training queues and Det evaluation use three history frames. Planning
    # evaluation also builds the configured temporal queue from these indices.
    history=int(ds.get('queue_length',4))-1
    if include_test_history:history=max(history,int(ds.get('history_frame_num',3)))
    needed=set()
    bad_history=[]
    for i in selected:
        queue=range(i-history,i+1)
        if i-history<0 or any(infos[j]['log_token']!=infos[i]['log_token'] for j in queue):
            bad_history.append(infos[i]['token']);continue
        needed.update(queue)
    if bad_history:errors.append(f'{split.name}: {len(bad_history)} anchors lack same-log history; examples {bad_history[:10]}')
    images=set();missing_images=[];missing_count=0;image_bytes=0
    for i in sorted(needed):
        cameras=infos[i].get('cams',{})
        if len(cameras)!=8:errors.append(f"{infos[i]['token']}: expected eight cameras, found {len(cameras)}")
        for cam in cameras.values():
            path=Path(cam['data_path'])
            if not path.is_absolute():path=image_root/path
            images.add(str(path))
    ref_hash=hashlib.sha256()
    for name in sorted(images):
        path=Path(name)
        try:
            size=path.stat().st_size
            if not path.is_file() or size==0:raise FileNotFoundError(name)
            image_bytes+=size
            ref_hash.update((name+'\0'+str(size)+'\n').encode())
        except OSError:
            missing_count+=1
            if len(missing_images)<20:missing_images.append(name)
    if missing_count:errors.append(f'{split.name}: {missing_count}/{len(images)} required current/history camera files missing; examples {missing_images}')
    phase.update(annotation_frames=len(infos),requested_anchors=len(requested),selected_anchors=len(selected),
                 missing_annotation_tokens=len(missing),history_frames=len(needed),camera_files=len(images),
                 missing_camera_files=missing_count,camera_bytes=image_bytes,camera_reference_size_sha256=ref_hash.hexdigest(),
                 camera_content_hash_verified=False,load_and_check_seconds=time.monotonic()-started,**memory())
    if pdm_path:
        if pdm_path.is_file():
            pdm=pickle.load(pdm_path.open('rb'));retain.append(pdm)
            missing_pdm=present-set(pdm)
            invalid=[]
            for token in sorted(present-missing_pdm):
                row=pdm[token]
                if any(key not in row or np.asarray(row[key]).shape!=(8192,) or
                       not np.isfinite(np.asarray(row[key])).all() for key in PDM_KEYS):
                    invalid.append(token)
            if missing_pdm:errors.append(f'{split.name}: missing genuine PDM scores for {len(missing_pdm)} anchors; examples {sorted(missing_pdm)[:10]}')
            if invalid:errors.append(f'{split.name}: invalid/nonfinite 8192-entry PDM rows for {len(invalid)} anchors; examples {invalid[:10]}')
            phase.update(pdm_tokens=len(pdm),missing_pdm_tokens=len(missing_pdm),invalid_pdm_rows=len(invalid),**memory())
    return phase,present,{infos[i]['log_token'] for i in needed}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--data-root',type=Path);g.add_argument('--mini-prepared',type=Path)
    p.add_argument('--maps',type=Path);p.add_argument('--synthetic-root',type=Path)
    p.add_argument('--metric-cache',type=Path,help='Existing official NAVTEST metric cache with metadata/*.csv')
    p.add_argument('--require-official-score',action='store_true')
    p.add_argument('--hash-assets',action='store_true',help='Verify original full annotation/PDM/vocabulary publisher SHA256')
    p.add_argument('--ranks',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.monotonic();errors=[];retain=[]
    if a.ranks<1:p.error('--ranks must be positive')
    os.environ['WORLDENGINE_ROOT']=str(ROOT)
    config=a.config if a.config.is_absolute() else ALG/a.config
    cfg=Config.fromfile(str(config),import_custom_modules=False)
    head=cfg.model.get('planning_head')
    needs_pdm=bool(head and head.type!='DiffusionPlanningHead')
    perception='Det' in cfg.data.train.type
    synthetic=any(x in cfg.data.train.type for x in ('Synthetic','SimScale'))
    rows={};sets={};files={}
    manifest=json.loads((a.mini_prepared/'manifest.json').read_text()) if a.mini_prepared else None
    for phase in ('train','eval'):
        ds=cfg.data.train if phase=='train' else cfg.data.val
        if manifest:
            spec=manifest['splits'][phase]
            ann=a.mini_prepared/spec['annotations'];split=a.mini_prepared/spec['filter']
            image_root=a.mini_prepared/'cameras'
            pdm=a.mini_prepared/(spec['pdm_prefix']+'.pkl') if needs_pdm else None
        else:
            name='navtrain' if phase=='train' else 'navtest'
            ann=a.data_root/f'alg_engine/merged_infos_navformer/nuplan_openscene_{name}.pkl'
            split=Path(ds.nav_filter_path)
            if not split.is_absolute():split=ALG/split
            image_root=a.data_root/'raw/openscene-v1.1/sensor_blobs'/('trainval' if phase=='train' else 'test')
            pdm=a.data_root/f'alg_engine/pdms_cache/pdm_8192_gt_cache_{name}.pkl' if needs_pdm else None
        for file in (ann,split,pdm):
            if file is None:continue
            if not file.is_file():errors.append('Missing file: '+str(file))
            else:files[str(file)]=dict(bytes=file.stat().st_size,sha256=None)
        if ann.is_file() and split.is_file():
            rows[phase],tokens,logs=inspect_phase(ds,ann,split,image_root,pdm,errors,retain,
                                                 include_test_history=perception and phase=='eval')
            sets[phase]=(tokens,logs)
    overlap={}
    if len(sets)==2:
        overlap=dict(token_overlap=len(sets['train'][0]&sets['eval'][0]),
                     history_log_overlap=len(sets['train'][1]&sets['eval'][1]))
        if any(overlap.values()):errors.append('Train/eval tokens or whole logs overlap: '+str(overlap))
    if perception or a.require_official_score:
        if not a.maps or not (a.maps/'nuplan-maps-v1.0.json').is_file():errors.append('Complete nuPlan maps required: --maps/<nuplan-maps-v1.0.json and all referenced .gpkg files>')
        elif not list(a.maps.rglob('*.gpkg')):errors.append('No .gpkg map data under '+str(a.maps))
    if synthetic:
        names=('sensor_blobs','pdms_pkl','meta_datas_navformer' if 'SimScale' in cfg.data.train.type else 'meta_datas')
        for name in names:
            if not a.synthetic_root or not (a.synthetic_root/name).is_dir():errors.append('Missing genuine synthetic export directory: '+name)
        # Directory presence cannot establish compatible camera/plan/PDM contents.
        errors.append('Synthetic content acceptance requires dataset build and genuine exported sequences; this real-data preflight cannot certify it')
    score_ready=False
    if a.metric_cache and len(sets)==2:
        paths={}
        for table in sorted((a.metric_cache/'metadata').glob('*.csv')):
            with table.open() as f:
                for row in csv.DictReader(f):
                    path=Path(row['file_name']);paths[path.parent.name]=path
        for payload in a.metric_cache.rglob('metric_cache.pkl'):
            paths[payload.parent.name]=payload
        missing=[token for token in sets['eval'][0] if token not in paths or not paths[token].is_file()]
        score_ready=not missing
        if missing:errors.append(f'Official metric cache missing {len(missing)} eval tokens; examples {sorted(missing)[:10]}')
    if a.require_official_score and not score_ready:errors.append('Complete official NAVTEST metric cache required for quality acceptance; generate it from genuine metadata and maps first')
    if a.hash_assets:
        if manifest:errors.append('--hash-assets expects unchanged publisher full PKLs, not a repacked mini')
        else:
            lock=json.loads((ROOT/'ascend/assets.lock.json').read_text())
            for asset in lock['assets']:
                if '/merged_infos_navformer/' not in asset['path'] and '/pdms_cache/' not in asset['path'] and not asset['path'].endswith('.npy'):continue
                if '/pdms_cache/' in asset['path'] and not needs_pdm:continue
                path=a.data_root/Path(asset['path']).relative_to('data')
                if not path.is_file():errors.append('Missing locked asset: '+str(path));continue
                digest=sha256(path);files[str(path)]=dict(bytes=path.stat().st_size,sha256=digest)
                if digest!=asset['sha256'] or path.stat().st_size!=asset['bytes']:errors.append('Publisher checksum/size mismatch: '+str(path))
    observed=memory()
    report=dict(result='pass_complete_inputs' if not errors else 'blocked_incomplete_inputs',errors=errors,
                config=str(config),config_sha256=sha256(config),dataset_type=cfg.data.train.type,
                original_epochs=cfg.get('total_epochs'),planning_head=head.type if head else None,
                needs_pdm=needs_pdm,perception=perception,synthetic=synthetic,
                data_scope=manifest['dataset_scope'] if manifest else 'Original configured full NAVTRAIN and independent NAVTEST',
                phases=rows,overlap=overlap,files=files,official_metric_cache_ready=score_ready,
                elapsed_seconds=time.monotonic()-started,**observed,
                cpu_data_resident_rank_projection_bytes=observed['rss_bytes']*a.ranks,
                projection_scope='Only this CPU inspection process times ranks. Add model/optimizer, validation, workers and measured runtime headroom; this is not measured full training RAM.',
                convergence_accepted=False,gpu_parity_accepted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if not errors else 2)

if __name__=='__main__':main()
