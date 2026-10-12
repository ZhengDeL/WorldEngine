"""Run an original AlgEngine configuration on Ascend, with optional mini data."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
ALG = ROOT / 'projects/AlgEngine'
PROFILES = {
    'imitation': 'worldengine/e2e_vadv2_50pct.py',
    'vadv2_ft': 'worldengine/e2e_vadv2_50pct_rlft_rare_log.py',
    'hydramdp_rl': 'hydramdp/e2e_hydramdp_50pct_rlft_rare_log.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='imitation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--mini-prepared', type=Path)
    parser.add_argument('--rare-split', type=Path)
    parser.add_argument('--steps', type=int, help='Bounded optimizer updates; omit for original epoch training')
    parser.add_argument('--batch-size', type=int, help='Per-rank batch; default 1 for bounded tests, upstream value for full runs')
    parser.add_argument('--workers', type=int, help='Default 0 for bounded tests, upstream value for full runs')
    parser.add_argument('--lr', type=float, help='Override learning rate; bounded tests default to 5e-5')
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    if not (args.mini_prepared or args.data_root):
        parser.error('Provide --mini-prepared or --data-root with raw/ and alg_engine/')
    if (args.batch_size is not None and args.batch_size < 1) or (args.steps is not None and args.steps < 1):
        parser.error('Batch size and bounded steps must be positive')
    for p in (args.checkpoint, args.vocab):
        if not p.is_file():
            raise FileNotFoundError(p)
    os.environ['WORLDENGINE_ROOT'] = str(ROOT)
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    sys.path[:0] = [str(ALG), str(ROOT)]
    cfg = Config.fromfile(str(ALG / 'configs' / PROFILES[args.profile]), import_custom_modules=False)
    cfg.dist_params.backend = 'hccl'
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.resume_from = None
    cfg.model.planning_head.vocab_path = str(args.vocab.resolve())
    cfg.data.samples_per_gpu = args.batch_size if args.batch_size is not None else (1 if args.steps else cfg.data.samples_per_gpu)
    cfg.data.workers_per_gpu = args.workers if args.workers is not None else (0 if args.steps else cfg.data.workers_per_gpu)
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    elif args.steps is not None:
        cfg.optimizer.lr = 5e-5
    cfg.optimizer.foreach = False
    cfg.pop('fp16', None)
    cfg.work_dir = str(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    scope = 'official full train/val assets; epoch/accuracy acceptance required separately'
    if args.mini_prepared:
        mini = args.mini_prepared.resolve()
        for name in ('worldengine_navtest_smoke.pkl', 'worldengine_navtest_smoke.yaml', 'worldengine_navtest_smoke_pdm.pkl', 'manifest.json'):
            if not (mini/name).is_file():
                raise FileNotFoundError(mini/name)
        manifest = json.loads((mini/'manifest.json').read_text())
        if manifest.get('trajectory_labels_changed') or manifest.get('heldout_samples') != 0:
            raise RuntimeError('Unexpected functional dataset manifest')
        scope = manifest['dataset_scope']
        for phase in ('train', 'val', 'test'):
            ds = cfg.data[phase]
            ds.ann_file = str(mini/'worldengine_navtest_smoke.pkl')
            ds.nav_filter_path = str(mini/'worldengine_navtest_smoke.yaml')
            ds.test_pdm_path = str(mini/'worldengine_navtest_smoke_pdm')
            ds.data_root = str(mini/'cameras')
            ds.pipeline[0].img_root = str(mini/'cameras')
        train = cfg.data.train
        if args.profile != 'imitation':
            # Genuine rare tokens from the official navtest failures split.
            split = args.rare_split or ALG/'configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml'
            if not split.is_file():
                raise FileNotFoundError(split)
            train.finetune_yaml = [str(split.resolve())]
            train.normal_only = False
            train.normal_ratio = 1
    else:
        data = args.data_root.resolve()
        required = [data/'raw/openscene-v1.1/sensor_blobs/trainval', data/'raw/openscene-v1.1/sensor_blobs/test',
                    data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl',
                    data/'alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl',
                    data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl',
                    data/'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl']
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError('Missing full assets: '+json.dumps(missing))
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
    if cfg.get('evaluation', {}).get('pipeline'):
        cfg.evaluation.pipeline[0].img_root = cfg.data.test.pipeline[0].img_root
    if args.steps is not None:
        cfg.pop('total_epochs', None)
        cfg.runner = dict(type='IterBasedRunner', max_iters=args.steps)
        cfg.checkpoint_config = dict(by_epoch=False, interval=args.steps, save_optimizer=True, max_keep_ckpts=1)
        cfg.lr_config = dict(policy='CosineAnnealing', by_epoch=False, warmup='linear', warmup_iters=min(2,args.steps), warmup_ratio=1/3, min_lr_ratio=0.001)
        cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
        cfg.optimizer_config = dict(type='AscendValidationHook', profile=args.profile, expected_steps=args.steps,
                                    expected_pg=args.profile=='hydramdp_rl', dataset_scope=scope, initialization_checkpoint=cfg.load_from,
                                    grad_clip=dict(max_norm=35,norm_type=2))
        cfg.custom_imports = dict(imports=['mmdet3d_plugin','ascend.validation_hook'], allow_failed_imports=False)
        cfg.evaluation.interval = 100000
    cfg.find_unused_parameters = True
    path = args.output/f'{args.profile}_rank{os.environ.get("LOCAL_RANK","0")}.py'
    cfg.dump(str(path))
    sys.argv = ['scripts/train.py', str(path), '--launcher', 'pytorch', '--work-dir', str(args.output), '--seed', str(args.seed)]
    if args.steps is not None:
        sys.argv.append('--no-validate')
    os.chdir(ALG)
    runpy.run_path(str(ALG/'scripts/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
