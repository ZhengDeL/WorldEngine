"""FP32 planning inference recovery with bounded shared-filesystem collection.

Retains the pinned model loop and journal; uses an explicitly locked CPU filesystem collector.
The legacy journal entry remains unchanged. New runs bind their requested
optimization and query chunk; existing journals cannot be adopted implicitly.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
import uuid

from filesystem_journal_observer import (COLLECT_SHA, Guard, IMAGE, JournalPause, MULTI_SHA,
                               SOURCE_SHA, fingerprint, observed_test, parse_deadline)
from rank_journal import canonical_json, need, rng_pack, rng_unpack, sha, sync_directory, write_file

OPTIMIZED_SOURCE_SHA = '7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb'
MODES = ('none', 'point-parallel', 'point-parallel-last-frame')
CHUNKS = (2048, 4096, 8192)


def optimization_contract(expected_source, actual_source, mode, chunk, backend):
    """Reject changed source/options before model construction or journal writes."""
    need(expected_source in (SOURCE_SHA, OPTIMIZED_SOURCE_SHA), 'Unsupported pinned runtime source')
    need(actual_source == expected_source, 'Runtime source fingerprint differs')
    need(mode in MODES and type(chunk) is int and chunk in CHUNKS, 'Unsupported optimization/chunk')
    need(backend in ('reference', 'cuda-compatible'), 'Unsupported MSDA backend')
    need(mode != 'none' or chunk == 2048, 'Original path requires query chunk 2048')
    need(expected_source != SOURCE_SHA or (mode == 'none' and chunk == 2048),
         'Original migration source supports only its original inference path')
    need(mode == 'none' or backend == 'cuda-compatible', 'Optimized inference requires cuda-compatible MSDA')
    return dict(mode=mode, query_chunk=chunk, scope='FP32 planning inference only')


def claim_attempt_output(output, claim, rank, guard):
    """Publish a complete attempt claim before any peer can observe its path."""
    final = output / 'journal_attempt_start.json'
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        temporary = output / ('journal_attempt_start.' + uuid.uuid4().hex + '.tmp')
        write_file(temporary, canonical_json(claim))
        os.replace(temporary, final)
        sync_directory(output)
    else:
        while not final.exists():
            guard.check()
            time.sleep(.1)
        need(json.loads(final.read_text()) == claim, 'Attempt output claim belongs to another launch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'config', 'checkpoint', 'output', 'journal-root', 'locked-inputs'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--expected-world-size', type=int, choices=(4, 12, 16), required=True)
    p.add_argument('--collection-plan', type=Path, required=True)
    p.add_argument('--expected-collection-plan-sha256', required=True)
    p.add_argument('--journal-mode', choices=('fresh', 'resume'), required=True)
    p.add_argument('--precision', choices=('fp32',), default='fp32')
    p.add_argument('--msda-backend', choices=('reference', 'cuda-compatible'), default='cuda-compatible')
    p.add_argument('--inference-optimization', choices=MODES, default='none')
    p.add_argument('--msda-query-chunk', type=int, choices=CHUNKS, default=2048)
    p.add_argument('--deterministic', action='store_true')
    for name in ('config', 'checkpoint', 'source', 'locked-inputs'):
        p.add_argument('--expected-' + name + '-sha256', required=True)
    p.add_argument('--expected-image-id', default=IMAGE)
    p.add_argument('--expected-anchors', type=int, default=22)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--stop-after-steps', type=int, default=0)
    args = p.parse_args()
    need(args.expected_anchors > 0 and args.stop_after_steps >= 0, 'Invalid anchor/step limit')
    root = args.source_root.resolve(strict=True)
    config = args.config.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    locked_path = args.locked_inputs.resolve(strict=True)
    output = args.output.resolve()
    journal_root = args.journal_root.resolve()
    need(output != journal_root and journal_root not in output.parents and output not in journal_root.parents,
         'Journal and attempt output must be separate')
    need(len(args.attempt_id) == 32 and all(c in '0123456789abcdef' for c in args.attempt_id),
         'New lowercase UUID hex required')
    need(args.expected_source_sha256 == OPTIMIZED_SOURCE_SHA, 'Filesystem entry requires the pinned merged FP32 inference source')
    optimization = optimization_contract(args.expected_source_sha256, fingerprint(root),
                                         args.inference_optimization, args.msda_query_chunk, args.msda_backend)
    need(os.environ.get('WORLDENGINE_IMAGE_ID') == args.expected_image_id == IMAGE,
         'Frozen original image required')
    need(int(os.environ.get('WORLD_SIZE', '0')) == args.expected_world_size, 'Actual world differs from the explicit contract')
    need(sha(config) == args.expected_config_sha256 and sha(checkpoint) == args.expected_checkpoint_sha256,
         'Config/checkpoint SHA differs')
    need(sha(locked_path) == args.expected_locked_inputs_sha256, 'Locked input manifest changed')
    locked = json.loads(locked_path.read_text())
    need(set(locked) == {'annotation', 'pdm', 'vocabulary', 'filter', 'input_qc'},
         'Five externally locked input files required')
    for item in locked.values():
        path = Path(item['path'])
        need(path.is_absolute() and path.resolve(strict=True) == path and path.stat().st_size == item['bytes']
             and len(item['sha256']) == 64, 'Locked input path/size/SHA invalid')
    guard = Guard(parse_deadline(args.deadline_utc), args.stop_file)
    guard.check()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: setattr(guard, 'signalled', True))
    here = Path(__file__).resolve().parent
    need(sha(here / 'journal_inference.py') == '89e24b8d733d3a2e07ba17e4b5a8b9973d2f1428db3334014b85dcc11a359f37', 'Pinned legacy observer source differs')
    need(sha(here / 'rank_journal.py') == 'f92e2fe35f428be30851c272921c450582e5b2c5846297b415385e13f8dfe4c4', 'Pinned journal source differs')
    need(sha(here / 'file_collection.py') == '6a5488bee8d53eb4fd6501702e80eb058653846f2d07ff5961c5bc074420542c', 'Pinned filesystem collector source differs')
    need(sha(here / 'filesystem_journal_observer.py') == 'e29725530dc26c563c7bec4450266ad019cf5b507d945dd35919fdb084a20e82', 'Pinned world-explicit observer source differs')
    need(Path(inspect.getfile(Guard)).resolve() == here / 'journal_inference.py', 'Foreign legacy observer import')
    instrumentation = dict(wrapper_sha256=sha(Path(__file__)),
                           original_observer_sha256='89e24b8d733d3a2e07ba17e4b5a8b9973d2f1428db3334014b85dcc11a359f37',
                           world_explicit_observer_sha256=sha(here / 'filesystem_journal_observer.py'),
                           filesystem_collector_sha256=sha(here / 'file_collection.py'),
                           journal_sha256=sha(here / 'rank_journal.py'),
                           original_multi_gpu_test_sha256=MULTI_SHA,
                           original_collect_results_cpu_sha256=COLLECT_SHA)
    from file_collection import collect_results_file, _load_contract
    need(sha(args.collection_plan) == args.expected_collection_plan_sha256, 'Collection plan SHA differs')
    collection_plan = json.loads(args.collection_plan.read_text())
    need(set(collection_plan) == {'run_id', 'run_contract_sha256', 'results', 'reports'}, 'Unexpected collection plan fields')
    for name, size in [('results', args.expected_anchors), ('reports', args.expected_world_size)]:
        value = collection_plan[name]
        need(set(value) == {'root', 'contract_sha256'}, 'Unexpected collection plan entry')
        header = _load_contract(Path(value['root']), collection_plan['run_id'], value['contract_sha256'], size, args.expected_world_size)
        need(header['source_sha256'] == collection_plan['run_contract_sha256'], 'Collection host run contract differs')
        need(header['sample_field'] == ('token' if name == 'results' else None), 'Collection identity field differs')
    contract = dict(schema_version=3, world_size=args.expected_world_size, anchors=args.expected_anchors,
                    source_sha256=args.expected_source_sha256, image_id=IMAGE,
                    config_sha256=args.expected_config_sha256, checkpoint_sha256=args.expected_checkpoint_sha256,
                    precision=args.precision, msda_backend=args.msda_backend, deterministic=args.deterministic,
                    inference_optimization=optimization, locked_inputs=locked,
                    locked_inputs_sha256=sha(locked_path), instrumentation=instrumentation,
                    result_collection=dict(mode='atomic_shared_filesystem', implementation_sha256=sha(here / 'file_collection.py'),
                                           host_run_contract_sha256=collection_plan['run_contract_sha256']),
                    execution_environment={key: os.environ.get(key) for key in (
                        'CLOSE_MATMUL_K_SHIFT', 'HCCL_DETERMINISTIC', 'HCCL_CONNECT_TIMEOUT', 'HCCL_EXEC_TIMEOUT', 'TASK_QUEUE_ENABLE',
                        'ASCEND_LAUNCH_BLOCKING', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'WORLDENGINE_HOST_DEVICE_IDS', 'ASCEND_RT_VISIBLE_DEVICES')})
    os.environ['WORLDENGINE_ROOT'] = str(root)
    sys.path[:0] = [str(root), str(root / 'projects/AlgEngine')]
    import torch
    import mmdet.apis
    from mmcv import Config
    original = mmdet.apis.multi_gpu_test
    need(hashlib.sha256(inspect.getsource(original).encode()).hexdigest() == MULTI_SHA,
         'Installed original multi_gpu_test changed')
    need(hashlib.sha256(inspect.getsource(original.__globals__['collect_results_cpu']).encode()).hexdigest() == COLLECT_SHA,
         'Installed original CPU collector changed')
    cfg = Config.fromfile(str(config), import_custom_modules=False)
    need(cfg.model.get('process_perception', False) is False
         and cfg.model.get('planning_head', {}).get('type') in ('TrajScoringHead', 'TrajScoringHeadRL'),
         'Only original vocabulary planning configurations are supported')
    paths = dict(annotation=cfg.data.test.ann_file, vocabulary=cfg.model.planning_head.vocab_path,
                 filter=cfg.data.test.nav_filter_path)
    need(all(Path(value).resolve(strict=True) == Path(locked[key]['path']) for key, value in paths.items()),
         'Actual config paths differ from locked inputs')
    claim = dict(attempt_id=args.attempt_id, contract_sha256=hashlib.sha256(canonical_json(contract)).hexdigest())
    claim_attempt_output(output, claim, int(os.environ.get('RANK', '0')), guard)
    reports = []
    original_collect = original.__globals__['collect_results_cpu']
    def filesystem_collect(result_part, size, tmpdir=None):
        guard.check()
        entry = collection_plan['results']
        return collect_results_file(result_part, size, rank=torch.distributed.get_rank(),
            world_size=args.expected_world_size, root=entry['root'], run_id=collection_plan['run_id'],
            contract_sha256=entry['contract_sha256'], deadline_utc=args.deadline_utc, stop_file=args.stop_file, guard_callback=guard.check)

    def wrapped(model, loader, tmpdir=None, gpu_collect=False):
        guard.check()
        from ascend.runtime import rng_state, restore_rng
        get_rng = lambda: rng_pack(rng_state())
        restore = lambda value: restore_rng(rng_unpack(value, torch))
        before = get_rng()
        need(Path(inspect.getfile(type(model.module))).resolve() == root / 'projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py',
             'Foreign NAVFormer plugin shadows pinned source')
        verified = [None]
        if torch.distributed.get_rank() == 0:
            for item in locked.values():
                guard.check()
                need(sha(item['path']) == item['sha256'], 'Locked data/QC SHA differs')
            verified[0] = hashlib.sha256(canonical_json(locked)).hexdigest()
        torch.distributed.broadcast_object_list(verified, src=0)
        need(verified[0] == hashlib.sha256(canonical_json(locked)).hexdigest(), 'Rank0 locked-input check differs')
        restore(before)
        need(Path(str(loader.dataset.pdm_path) + '.pkl').resolve() == Path(locked['pdm']['path']),
             'Actual selected PDM differs from input lock')
        tokens = [loader.dataset.data_infos[index]['token'] for index in loader.dataset.index_map]
        entry = collection_plan['results']
        header = _load_contract(Path(entry['root']), collection_plan['run_id'], entry['contract_sha256'], len(tokens), args.expected_world_size)
        need(header['expected_ids'] == tokens, 'Host expected IDs differ from the actual dataset index_map order')
        original.__globals__['collect_results_cpu'] = filesystem_collect
        result, report = observed_test(original, model, loader, tmpdir, gpu_collect, torch=torch,
            contract=contract, journal_root=journal_root, mode=args.journal_mode, guard=guard,
            get_rng=get_rng, restore_rng=restore, stop_after_steps=args.stop_after_steps)
        reports.append(report)
        return result

    mmdet.apis.multi_gpu_test = wrapped
    argv = [str(root / 'ascend/inference.py'), '--config', str(config), '--checkpoint', str(checkpoint),
            '--output', str(output), '--precision', args.precision, '--msda-backend', args.msda_backend]
    if args.expected_source_sha256 == OPTIMIZED_SOURCE_SHA:
        argv += ['--inference-optimization', args.inference_optimization, '--msda-query-chunk', str(args.msda_query_chunk)]
    if args.deterministic:
        argv += ['--deterministic']
    sys.argv = argv
    try:
        runpy.run_path(str(root / 'ascend/inference.py'), run_name='__main__')
        guard.check()
        need(len(reports) == 1, 'Exactly one original planning test required')
        entry = collection_plan['reports']
        gathered = collect_results_file([reports[0]], args.expected_world_size,
            rank=torch.distributed.get_rank(), world_size=args.expected_world_size,
            root=entry['root'], run_id=collection_plan['run_id'], contract_sha256=entry['contract_sha256'],
            deadline_utc=args.deadline_utc, stop_file=args.stop_file, guard_callback=guard.check)
        if torch.distributed.get_rank() == 0:
            need([item['rank'] for item in gathered] == list(range(args.expected_world_size)) and all(
                 item['result'] == 'pass_original_padded_rank_sequence_and_exact_replay'
                 and item['committed_after'] == (args.expected_anchors + args.expected_world_size - 1) // args.expected_world_size for item in gathered),
                 'Every original rank must finish its padded sequence')
            need(fingerprint(root) == args.expected_source_sha256 and sha(config) == args.expected_config_sha256
                 and sha(checkpoint) == args.expected_checkpoint_sha256, 'Source/config/checkpoint changed')
            for item in locked.values():
                guard.check()
                need(sha(item['path']) == item['sha256'], 'Locked data/QC changed')
            validation = output / 'inference_validation.json'
            original_raw = validation.read_bytes()
            original_report = json.loads(original_raw)
            need(original_report['result'] == 'pass_real_planning_inference'
                 and original_report['anchors'] == args.expected_anchors
                 and original_report['world_size'] == args.expected_world_size, 'Original complete inference validation required')
            write_file(output / 'original_inference_validation.json', original_raw)
            original_report['execution_instrumentation'] = dict(
                **instrumentation, contract_sha256=hashlib.sha256(canonical_json(contract)).hexdigest(),
                journal_mode=args.journal_mode, rank_attempts=gathered,
                original_report_sha256=hashlib.sha256(original_raw).hexdigest(),
                acceptance='Configured-anchor recovery with an explicit optimization contract; no scoring, GPU parity or performance claim.')
            temporary = output / ('inference_validation.' + uuid.uuid4().hex + '.tmp')
            write_file(temporary, canonical_json(original_report))
            os.replace(temporary, validation)
            record = dict(result='pass_full_configured_optimized_inference_with_journal',
                          contract=contract, rank_attempts=gathered,
                          acceptance='Configured-anchor recovery only; no full NAVTEST quality, GPU parity, scoring or performance claim.')
            write_file(output / 'journal_execution.json', canonical_json(record))
    except JournalPause as exc:
        output.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get('RANK', '0'))
        write_file(output / ('journal_partial_rank%d.json' % rank), canonical_json(dict(
            result='partial_committed_journal_not_complete_inference', reason=str(exc), contract=contract)))
        if str(exc).startswith('Controlled committed-step stop'):
            torch.distributed.barrier()
        raise SystemExit(3)
    finally:
        mmdet.apis.multi_gpu_test = original
        original.__globals__['collect_results_cpu'] = original_collect


if __name__ == '__main__':
    main()
