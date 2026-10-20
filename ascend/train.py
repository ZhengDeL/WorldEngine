"""Run an original AlgEngine configuration on Ascend, with optional mini data."""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config
from ascend.compat import enable_training_compat
from ascend.precision import precision_state, device_provenance

ROOT = Path(__file__).resolve().parents[1]
ALG = ROOT / 'projects/AlgEngine'
PROFILES = {
    'imitation': 'worldengine/e2e_vadv2_50pct.py',
    'vadv2_ft': 'worldengine/e2e_vadv2_50pct_rlft_rare_log.py',
    'hydramdp_rl': 'hydramdp/e2e_hydramdp_50pct_rlft_rare_log.py',
}




def _resolve_dataset_filter_paths(cfg, alg_root):
    """Resolve configured filters before runtime hashing or changing cwd."""
    alg_root = Path(alg_root).resolve(strict=True)

    def resolved_file(value, field):
        if not isinstance(value, (str, Path)):
            raise TypeError('%s must contain a file path' % field)
        if isinstance(value, str) and not value.strip():
            raise ValueError('%s must not be empty' % field)
        path = Path(value)
        path = (path if path.is_absolute() else alg_root / path).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError('%s is not a regular file: %s' % (field, path))
        return str(path)

    for phase in ('train', 'val', 'test'):
        dataset = cfg.data[phase]
        if dataset.get('nav_filter_path') is not None:
            dataset['nav_filter_path'] = resolved_file(dataset['nav_filter_path'], phase+'.nav_filter_path')
        rare = dataset.get('finetune_yaml')
        if rare is not None:
            if isinstance(rare, (str, Path)):
                rare = [rare]
            elif not isinstance(rare, (list, tuple)):
                raise TypeError('%s.finetune_yaml must be a path or path list' % phase)
            dataset['finetune_yaml'] = [resolved_file(value, phase+'.finetune_yaml') for value in rare]


def _validate_stop_options(args, parser):
    """Reject stop settings that the selected runner cannot honor."""
    import math
    if args.stop_after is not None:
        if args.steps is None:
            parser.error('--stop-after requires --steps')
        if args.stop_after < 1 or args.stop_after > args.steps:
            parser.error('--stop-after must be between 1 and --steps')
    if args.duration_seconds is not None:
        if args.steps is None:
            parser.error('--duration-seconds requires --steps')
        if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
            parser.error('--duration-seconds must be finite and positive')
    if args.stop_after_epochs is not None:
        if args.steps is not None:
            parser.error('--stop-after-epochs requires an epoch schedule without --steps')
        if args.stop_after_epochs < 1:
            parser.error('--stop-after-epochs must be positive')
        if args.epochs is not None and args.stop_after_epochs > args.epochs:
            parser.error('--stop-after-epochs must not exceed --epochs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='imitation')
    parser.add_argument('--config', type=Path, help='Any published AlgEngine config, relative to projects/AlgEngine or absolute')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--train-filter', type=Path, help='Explicit NAVTRAIN anchor filter for --data-root; hashed in the resume contract')
    parser.add_argument('--mini-prepared', type=Path)
    parser.add_argument('--rare-split', type=Path)
    parser.add_argument('--synthetic-root', type=Path, help='Existing genuine SimEngine/SimScale exported dataset')
    parser.add_argument('--validate', action='store_true', help='Run the original epoch validation hook; requires complete evaluation inputs')
    parser.add_argument('--official-metric-cache',type=Path,help='Unmodified official NAVTEST cache for per-epoch CPU rescoring')
    parser.add_argument('--steps', type=int, help='Bounded optimizer updates; omit for original epoch training')
    parser.add_argument('--batch-size', type=int, help='Per-rank batch; default 1 for bounded tests, upstream value for full runs')
    parser.add_argument('--workers', type=int, help='Default 0 for bounded tests, upstream value for full runs')
    parser.add_argument('--lr', type=float, help='Override learning rate; bounded tests default to 5e-5')
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--runtime', action='store_true', help='Native optimizer, resource observation and complete resume checkpoints')
    parser.add_argument('--resume', type=Path, help='Complete runtime checkpoint, including optimizer and every rank RNG')
    parser.add_argument('--epochs', type=int, help='Epoch schedule horizon; compatible with --runtime')
    parser.add_argument('--stop-after', type=int, help='Stop at this global update while preserving --steps LR horizon')
    parser.add_argument('--stop-after-epochs', type=int, help='Stop at an epoch boundary while preserving the epoch LR horizon')
    parser.add_argument('--duration-seconds', type=float, help='Stop iter training after this measured duration')
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--check-interval', type=int, default=10)
    parser.add_argument('--upstream-schedule', action='store_true', help='Keep original optimizer LR and warmup parameters')
    parser.add_argument('--deterministic', action='store_true', help='Request deterministic PyTorch/CANN algorithms for numerical replay checks')
    parser.add_argument('--precision', choices=('fp32','fp32-hf32-conv','bf16'), default='fp32',
                        help='fp32 disables Matmul/Conv HF32; fp32-hf32-conv reproduces legacy Conv HF32; bf16 uses native forward/loss autocast with FP32 master parameters, optimizer state and MSDA, without a scaler (implies --runtime)')
    parser.add_argument('--debug-anomaly', action='store_true', help='Record the forward origin of a failing backward operation')
    parser.add_argument('--msda-backend', choices=('reference', 'cuda-compatible'), default='reference',
                        help='FP32 sampling backend; cuda-compatible matches the measured original CUDA arithmetic')
    args = parser.parse_args()
    _validate_stop_options(args, parser)
    from ascend.msda import configure_backend
    configure_backend(args.msda_backend)
    if args.train_filter and (not args.data_root or args.mini_prepared):
        parser.error('--train-filter requires --data-root without --mini-prepared')
    if args.train_filter:
        args.train_filter=args.train_filter.resolve()
        if not args.train_filter.is_file() or 'navtrain' not in args.train_filter.name or 'navtest' in args.train_filter.name:
            parser.error('--train-filter must be a NAVTRAIN file whose resolved basename contains navtrain and excludes navtest; the original dataset selects its PDM cache by this name')
    if args.precision=='bf16':args.runtime=True
    if args.resume or args.stop_after or args.duration_seconds or args.epochs or args.stop_after_epochs:
        args.runtime = True
    if args.steps and args.epochs:parser.error('Choose a step or epoch schedule')
    if args.resume and (args.workers or 0)!=0:parser.error('Exact resume requires --workers 0')
    if args.checkpoint_interval<1 or args.check_interval<1:parser.error('Intervals must be positive')
    if args.runtime and args.workers is None:args.workers=0
    if args.epochs is not None and args.epochs<1:parser.error('--epochs must be positive')
    if not (args.mini_prepared or args.data_root):
        parser.error('Provide --mini-prepared or --data-root with raw/ and alg_engine/')
    if (args.batch_size is not None and args.batch_size < 1) or (args.steps is not None and args.steps < 1):
        parser.error('Batch size and bounded steps must be positive')
    for p in (args.checkpoint, args.vocab):
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.deterministic and (os.environ.get('CLOSE_MATMUL_K_SHIFT')!='1' or
                               os.environ.get('HCCL_DETERMINISTIC')!='true'):
        parser.error('--deterministic requires CLOSE_MATMUL_K_SHIFT=1 and HCCL_DETERMINISTIC=true before the Torch process starts; use ascend/run_container.py or export both first')
    os.environ['WORLDENGINE_ROOT'] = str(ROOT)
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    # Bind this rank before public runtime setters, and set precision before
    # original model construction. Tensor storage/config/loss remain unchanged.
    torch.npu.matmul.allow_hf32=False
    torch.npu.conv.allow_hf32=args.precision=='fp32-hf32-conv'
    enable_training_compat()
    if args.debug_anomaly:torch.autograd.set_detect_anomaly(True)
    if args.deterministic:torch.use_deterministic_algorithms(True)
    actual_precision=precision_state(args.precision)
    if actual_precision['public']!=dict(matmul_allow_hf32=False,conv_allow_hf32=args.precision=='fp32-hf32-conv'):
        raise RuntimeError('Public NPU precision settings differ from --precision')
    sys.path[:0] = [str(ALG), str(ROOT)]
    config = args.config or Path('configs') / PROFILES[args.profile]
    config = config if config.is_absolute() else ALG / config
    if args.config:
        args.profile=config.stem
        args.runtime=True
        if args.workers is None:args.workers=0
    cfg = Config.fromfile(str(config), import_custom_modules=False)
    head=cfg.model.get('planning_head')
    train_type=cfg.data.train.type
    needs_pdm=head is not None and head.type!='DiffusionPlanningHead'
    synthetic=any(name in train_type for name in ('Synthetic','SimScale'))
    if synthetic and not args.synthetic_root:
        parser.error('This configuration requires genuine --synthetic-root exports; real mini logs cannot replace them')
    cfg.dist_params.backend = 'hccl'
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.resume_from = None
    if args.resume:
        if not args.resume.is_file():raise FileNotFoundError(args.resume)
        cfg.resume_from=str(args.resume.resolve())
    if head is not None and 'vocab_path' in head:
        head.vocab_path = str(args.vocab.resolve())
    cfg.data.samples_per_gpu = args.batch_size if args.batch_size is not None else (1 if args.steps else cfg.data.samples_per_gpu)
    cfg.data.workers_per_gpu = args.workers if args.workers is not None else (0 if args.steps else cfg.data.workers_per_gpu)
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    elif args.steps is not None and not args.upstream_schedule:
        cfg.optimizer.lr = 5e-5
    cfg.optimizer.foreach = False
    cfg.pop('fp16', None)
    cfg.work_dir = str(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    scope = 'official full train/val assets; epoch/accuracy acceptance required separately'
    if args.mini_prepared:
        mini = args.mini_prepared.resolve()
        manifest = json.loads((mini/'manifest.json').read_text())
        names=['worldengine_navtest_smoke.pkl','worldengine_navtest_smoke.yaml','worldengine_navtest_smoke_pdm.pkl']
        if manifest.get('splits'):
            names=[name for spec in manifest['splits'].values() for name in
                   (spec['annotations'],spec['filter'],spec['pdm_prefix']+'.pkl')]
        for name in names:
            if not (mini/name).is_file():raise FileNotFoundError(mini/name)
        if manifest.get('trajectory_labels_changed'):
            raise RuntimeError('Unexpected functional dataset manifest')
        scope = manifest['dataset_scope']
        for phase in ('train', 'val', 'test'):
            ds = cfg.data[phase]
            spec=manifest.get('splits',{}).get('train' if phase=='train' else 'eval',{})
            ds.ann_file = str(mini/spec.get('annotations','worldengine_navtest_smoke.pkl'))
            ds.nav_filter_path = str(mini/spec.get('filter','worldengine_navtest_smoke.yaml'))
            ds.test_pdm_path = str(mini/spec.get('pdm_prefix','worldengine_navtest_smoke_pdm'))
            ds.train_pdm_path = ds.test_pdm_path
            ds.data_root = str(mini/'cameras')
            ds.pipeline[0].img_root = str(mini/'cameras')
        train = cfg.data.train
        if 'FineTune' in train_type:
            # Genuine rare tokens from the official navtest failures split.
            split = args.rare_split or ALG/'configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml'
            if not split.is_file():
                raise FileNotFoundError(split)
            train.finetune_yaml = [str(split.resolve())]
            # Keep common-only versus rare+normal semantics of each original config.
            if not train.get('normal_only',False):train.normal_ratio = 1
    else:
        data = args.data_root.resolve()
        # Match scripts/train.py's actual --no-validate behavior. A second
        # workflow dataset is still built even when the eval hook is disabled.
        script_validation = args.validate or not (args.steps is not None or args.runtime)
        workflow_validation = len(cfg.workflow) == 2 or any(phase == 'val' for phase, _ in cfg.workflow)
        requires_navtest = script_validation or workflow_validation
        image_roots = [data/'raw/openscene-v1.1/sensor_blobs/trainval']
        pickle_files = [data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl']
        if needs_pdm:
            pickle_files.append(data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl')
        if requires_navtest:
            image_roots.append(data/'raw/openscene-v1.1/sensor_blobs/test')
            pickle_files.append(data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl')
            if needs_pdm:
                pickle_files.append(data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl')
        missing = ([str(path) for path in image_roots if not path.is_dir()] +
                   [str(path) for path in pickle_files if not path.is_file()])
        if missing:
            raise RuntimeError('Missing full assets: '+json.dumps(missing))
        if not requires_navtest:
            scope = ('Official NAVTRAIN metadata/PDM paths with selected training images; '
                     'complete selected-image coverage and full convergence require separate verification')
        for phase in ('train', 'val', 'test'):
            ds = cfg.data[phase]
            split = 'trainval' if phase == 'train' else 'test'
            ds.data_root = str(data/'raw/openscene-v1.1')
            ds.ann_file = str(data/f'alg_engine/merged_infos_navformer/nuplan_openscene_nav{"train" if phase == "train" else "test"}.pkl')
            ds.train_pdm_path = str(data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain')
            ds.test_pdm_path = str(data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest')
            ds.pipeline[0].img_root = str(data/'raw/openscene-v1.1/sensor_blobs'/split)
        if args.rare_split:
            cfg.data.train.finetune_yaml = [str(args.rare_split.resolve())]
        if args.train_filter:
            cfg.data.train.nav_filter_path = str(args.train_filter.resolve(strict=True))
            scope = ('Explicit selected NAVTRAIN anchors with original metadata and PDM schema; '
                     'subset coverage and full convergence require separate verification')
    if synthetic:
        root=args.synthetic_root.resolve(strict=True)
        needed=['sensor_blobs','pdms_pkl','meta_datas_navformer' if 'SimScale' in train_type else 'meta_datas']
        missing=[str(root/name) for name in needed if not (root/name).is_dir()]
        if missing:raise RuntimeError('Missing genuine synthetic exports: '+json.dumps(missing))
        cfg.data.train.synthetic_root=str(root)
    for phase in ('train','val','test'):
        cfg.data[phase].strict_pdm=needs_pdm
    _resolve_dataset_filter_paths(cfg, ALG)
    if cfg.get('evaluation', {}).get('pipeline'):
        cfg.evaluation.pipeline[0].img_root = cfg.data.test.pipeline[0].img_root
    if args.steps is not None:
        cfg.pop('total_epochs', None)
        cfg.runner = dict(type='IterBasedRunner', max_iters=args.steps)
        cfg.checkpoint_config = dict(by_epoch=False, interval=args.steps, save_optimizer=True, max_keep_ckpts=1)
        if args.upstream_schedule:
            cfg.lr_config.by_epoch=False
        else:
            cfg.lr_config = dict(policy='CosineAnnealing', by_epoch=False, warmup='linear', warmup_iters=min(2,args.steps), warmup_ratio=1/3, min_lr_ratio=0.001)
        cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
        cfg.optimizer_config = dict(type='AscendValidationHook', profile=args.profile, expected_steps=args.steps,
                                    expected_pg=args.profile=='hydramdp_rl', dataset_scope=scope, initialization_checkpoint=cfg.load_from,
                                    grad_clip=dict(max_norm=35,norm_type=2))
        cfg.custom_imports = dict(imports=['mmdet3d_plugin','ascend.validation_hook'], allow_failed_imports=False)
        cfg.evaluation.interval = 100000
    if args.runtime:
        if args.epochs:
            cfg.total_epochs=args.epochs
            cfg.runner=dict(type='EpochBasedRunner',max_epochs=args.epochs)
        def digest(path):
            h=hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            return h.hexdigest()
        contract=dict(contract_version=2,precision=actual_precision,
                      profile=args.profile,world_size=int(os.environ.get('WORLD_SIZE','1')),
                      batch=cfg.data.samples_per_gpu,workers=cfg.data.workers_per_gpu,seed=args.seed,
                      annotation=digest(cfg.data.train.ann_file),filter=digest(cfg.data.train.nav_filter_path),
                      pdm=digest(cfg.data.train.test_pdm_path+'.pkl' if args.mini_prepared else cfg.data.train.train_pdm_path+'.pkl') if needs_pdm else None,
                      vocab=digest(args.vocab),optimizer=dict(cfg.optimizer),lr=dict(cfg.lr_config),runner=dict(cfg.runner),
                      original_config=digest(config),dataset_type=train_type,
                      expected_pg=bool(head and head.get('rl_finetuning',False)),validate=args.validate)
        fingerprint=hashlib.sha256()
        for file in sorted(list(ALG.rglob('*.py'))+list((ROOT/'ascend').glob('*.py'))):
            fingerprint.update(str(file.relative_to(ROOT)).encode())
            fingerprint.update(file.read_bytes().replace(b'\r\n',b'\n'))
        contract.update(source_sha256=fingerprint.hexdigest(),image_id=os.environ.get('WORLDENGINE_IMAGE_ID'),
                        torch_version=str(torch.__version__),torch_npu_version=torch_npu.__version__,
                        torch_npu_git_version=getattr(getattr(torch_npu,'version',None),'git_version',None),
                        deterministic=args.deterministic,
                        rare_filters=[digest(x) for x in (cfg.data.train.get('finetune_yaml') or [])])
        cfg.runner.type='AscendIterRunner' if args.steps else 'AscendEpochRunner'
        cfg.runner.resume_contract=contract
        cfg.optimizer_config=dict(grad_clip=dict(max_norm=35,norm_type=2))
        cfg.checkpoint_config=None
        cfg.custom_imports=dict(imports=['mmdet3d_plugin','ascend.runtime'],allow_failed_imports=False)
        cfg.custom_hooks=[dict(type='AscendRuntimeHook',priority='LOWEST',profile=args.profile,
                              initialization_checkpoint=None if args.resume else cfg.load_from,
                              checkpoint_interval=args.checkpoint_interval,stop_after=args.stop_after,
                              stop_after_epochs=args.stop_after_epochs,duration_seconds=args.duration_seconds,
                              check_interval=args.check_interval,device_provenance=device_provenance())]
        cfg.log_config=dict(interval=10,hooks=[dict(type='TextLoggerHook',by_epoch=not bool(args.steps))])
    if args.precision=='bf16':
        cfg.custom_imports['imports'].append('ascend.autocast')
        cfg.custom_hooks.insert(0,dict(type='AscendBFloat16Hook',priority='LOW'))
    cfg.find_unused_parameters = True
    if args.validate:
        cfg.ascend_epoch_evaluation=True
        cfg.evaluation.interval=1
        if args.official_metric_cache:
            cfg.evaluation.official_metric_cache=str(args.official_metric_cache.resolve(strict=True))
        if head is not None and not args.mini_prepared and not args.official_metric_cache:
            parser.error('Full quality acceptance requires --official-metric-cache')
        if args.steps:parser.error('Per-epoch validation requires an epoch schedule, not --steps')
    path = args.output/f'{args.profile}_rank{os.environ.get("LOCAL_RANK","0")}.py'
    cfg.dump(str(path))
    sys.argv = ['scripts/train.py', str(path), '--launcher', 'pytorch', '--work-dir', str(args.output), '--seed', str(args.seed)]
    if args.deterministic:sys.argv.append('--deterministic')
    if (args.steps is not None or args.runtime) and not args.validate:
        sys.argv.append('--no-validate')
    os.chdir(ALG)
    if args.precision=='bf16':
        from ascend.autocast import training_precision_cleanup
        with training_precision_cleanup(args.precision):
            runpy.run_path(str(ALG/'scripts/train.py'), run_name='__main__')
    else:
        runpy.run_path(str(ALG/'scripts/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
