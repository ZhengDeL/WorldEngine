"""CPU pilot: original WorldEngine 4s scoring on genuine OpenScene logged traffic.

Derived protocol, not a reproduction of published SimScale v2 rewards. Original
scoring/simulation methods are executed unchanged; no renderer or engine runs.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import logging
from multiprocessing import Pool
import os
from pathlib import Path, PurePosixPath
import pickle
import signal
import sys
import tarfile
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import time
from functools import partial


META_SIZE = 534344190
META_SHA = 'c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be'
VOCAB_SHA = 'cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35'
LOG = '2021.06.09.17.23.18_veh-38_00773_01140'
SCENE_ID = 'fff002cb15c15a7a-000'
FOLDER = 'synthetic_reaction_pdm_v1.0-0'
GIB = 1024 ** 3
LOCKS = {
    'manager/dense_reward_manager.py': '193b88b91fa1c35c2b93bd809993c8ad7a9393eba6565cb9486a65342efeb537',
    'components/agents/policy/pdm_planner/utils/WE2pdm_utils.py': '802612b3e4be06ffad2a97955d0cb9202935949a3e57ad2855812edced6b3524',
    'components/agents/policy/pdm_planner/scoring/pdm_scorer.py': '55b696a6b0e5efeabf11f32a8de0064d50b7869e2d63eae37983158c9d7cb025',
    'components/agents/policy/pdm_planner/simulation/pdm_simulator.py': '377e21fe2b0adf83e1bcf5617e1df9c71137e1f42c216f2b486ebdc8ba8f5536',
}
logger = logging.getLogger(__name__)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, row):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def cpu_worker_init():
    # The supervisor's timeout exception handler must not replace the original
    # terminating SIGTERM behavior in owned Pool children.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def stop_owned_pool(pool):
    children = list(pool._pool)
    shutdown = threading.Thread(target=pool.terminate, daemon=True)
    shutdown.start()
    shutdown.join(5)
    for child in children:
        if child.is_alive():
            child.kill()
    until = time.monotonic() + 5
    for child in children:
        child.join(max(0, until-time.monotonic()))
    if any(child.is_alive() for child in children):
        raise TimeoutError('Owned CPU scoring workers did not exit within10seconds')


def extract_methods(path, class_name, method_names, target_name, functions=()):
    text = path.read_text()
    module = ast.parse(text)
    original = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    if {m.name for m in methods} != set(method_names):
        raise ValueError('Original scoring methods missing')
    outputs = [ast.ClassDef(name=target_name, bases=[], keywords=[], body=methods, decorator_list=[])]
    outputs.extend(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in functions)
    if len(outputs) != 1 + len(functions):
        raise ValueError('Original helper functions missing')
    tree = ast.fix_missing_locations(ast.Module(body=outputs, type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), globals())
    return dict(path=str(path), canonical_source_sha256=hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest(),
        selected_ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        unmodified_methods=sorted(method_names), unmodified_functions=list(functions))


def inputs(args, row):
    manifest = json.loads((args.conversion_root / 'conversion_manifest.json').read_text())
    entry = next(c for c in manifest['clips'] if c['scene_id'] == SCENE_ID)
    syn_path = args.conversion_root / 'synthetic_root/meta_datas' / FOLDER / PurePosixPath(entry['raw_member']).name
    if sha(syn_path) != entry['raw_sha256']:
        raise ValueError('Original synthetic metadata SHA differs')
    with syn_path.open('rb') as stream:
        synthetic = pickle.load(stream)
    original_payload = None
    members = 0
    with tarfile.open(args.metadata, 'r|gz') as archive:
        for member in archive:
            members += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Original archive unsafe member')
            if member.isfile() and path.name == LOG + '.pkl':
                if original_payload is not None or member.size > 64 * 1024 * 1024:
                    raise ValueError('Ambiguous/oversized original log')
                original_payload = archive.extractfile(member).read(member.size + 1)
    if original_payload is None:
        raise ValueError('Original mini metadata lacks exact source log')
    original = pickle.loads(original_payload)
    if not isinstance(original, list) or len(synthetic) != 12:
        raise TypeError('Unexpected original raw metadata schema')
    index = {f['token']: i for i, f in enumerate(original)}
    ids = [index.get(f['token'][:-4]) if f['token'].endswith('-000') else None for f in synthetic]
    if any(i is None for i in ids) or ids != list(range(ids[0], ids[0] + 12)):
        raise ValueError('Twelve literal frame tokens do not map to consecutive original frames')
    source = original[ids[0]:ids[0] + 12]
    if any(f['log_name'] != LOG for f in source) or any(int(b['timestamp']) <= int(a['timestamp']) for a, b in zip(source, source[1:])):
        raise ValueError('Original log/time order differs')
    deltas = [int(b['timestamp'])-int(a['timestamp']) for a,b in zip(source, source[1:])]
    if any(not 400000 <= delta <= 600000 for delta in deltas):
        raise ValueError('Original raw frames are not a genuine approximately2Hz contiguous sequence')
    if source[3]['map_location'] != synthetic[3]['map_location'] or source[3]['roadblock_ids'] != synthetic[3]['roadblock_ids']:
        raise ValueError('Actual original and synthetic map/route identity differs')
    (args.output / 'original_complete_log_unchanged.pkl').write_bytes(original_payload)
    row['input_binding'] = dict(scene_id=SCENE_ID, synthetic_metadata_sha256=sha(syn_path),
        original_log_member_sha256=hashlib.sha256(original_payload).hexdigest(), original_log_frame_count=len(original),
        original_indices=ids, literal_frame_tokens_all_correspond=True, original_route_ids=source[3]['roadblock_ids'],
        source_time_deltas_microseconds=deltas,
        source_frame_timestamps=[int(f['timestamp']) for f in source],
        synthetic_frame_timestamps=[int(f['timestamp']) for f in synthetic],
        nominal_scoring_interval_seconds=.5, real_sensor_timestamp_jitter_preserved=True,
        physical_states_assumed_equal=False, reactive_future_reconstructed=False, metadata_archive_members=members)
    return source, synthetic


def imports(args):
    global TrajectorySampling, PlannerInput, PlannerInitialization, SimulationIteration, SimulationHistoryBuffer
    global EgoState, TimePoint, StateSE2, StateVector2D, Agent, StaticObject, SceneObjectMetadata
    global OrientedBox, TrackedObjects, DetectionsTracks, PDMClosedPlanner, BatchIDMPolicy
    global PDMSimulator, PDMDrivableMap, PDMPath, MultiMetricIndex, WeightedMetricIndex, get_pacifica_parameters
    sys.path.extend([str(args.source / 'projects/SimEngine'), str(args.source / 'ascend/third_party/navsim')])
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import TimePoint, StateSE2, StateVector2D
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
    from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
    from worldengine.components.agents.policy.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
    from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters


def score(args, source, synthetic, row):
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    scene = Scene.from_scene_dict_list(source, Path('/no-sensors'), 4, 8, SensorConfig.build_no_sensors())
    scenario = NavSimScenario(scene, str(args.maps), 'nuplan-maps-v1.0')
    observations = []
    for iteration in range(-3, 9):
        ego = scenario.get_ego_state_at_iteration(iteration)
        observation = scenario.get_tracked_objects_at_iteration(iteration)
        if any(obj.track_token == 'ego' for obj in observation.tracked_objects):
            raise ValueError('Original annotation unexpectedly already contains ego')
        observed_ego = Agent(TrackedObjectType.VEHICLE, ego.car_footprint.oriented_box,
            ego.dynamic_car_state.center_velocity_2d,
            SceneObjectMetadata(ego.time_us, token=SCENE_ID, track_id='ego', track_token='ego'))
        observations.append(DetectionsTracks(TrackedObjects(list(observation.tracked_objects) + [observed_ego])))
    kernel = OriginalScoringKernel()
    kernel.current_step = 3
    kernel.buffer_size = 9
    kernel._map_radius = 100
    kernel._num_scoring_workers = 8
    kernel.use_cuda = False
    kernel.engine = SimpleNamespace(global_config=dict(reward_sampling_poses=8))
    kernel.map_api = scenario.map_api
    kernel._cached_roadblock_ids = list(source[3]['roadblock_ids'])
    kernel._pdm_closed = PDMClosedPlanner(TrajectorySampling(num_poses=41, interval_length=.1),
        TrajectorySampling(num_poses=40, interval_length=.1),
        BatchIDMPolicy(speed_limit_fraction=[.15, .25, .5, .75, 1.0], fallback_target_velocity=30.,
            min_gap_to_lead_agent=10., headway_time=1.5, accel_max=4.5, decel_max=3.), [-3.5, 3.5], 100)
    kernel.gt_array = np.load(args.vocab)
    if kernel.gt_array.shape != (8192, 40, 3) or not np.isfinite(kernel.gt_array).all():
        raise ValueError('Original 8192 trajectory vocabulary shape/finite differs')
    # Synthetic ego coordinates are original publisher global rear-axle poses.
    origin = np.asarray(source[0]['ego2global_translation'][:2], dtype=np.float64)
    adapter_scene = dict(metadata={'old_origin_in_current_coordinate': -origin},
        base_timestamp=int(synthetic[0]['timestamp']), sample_rate=10, sdc_id='ego', object_track={'ego': {'state': {}}})
    agent = SimpleNamespace(navigation=SimpleNamespace(checkpoint_lanes=[SimpleNamespace(roadblock_id=x) for x in source[3]['roadblock_ids']]))
    converter = OriginalEgoConverter(adapter_scene, agent, kernel.engine)
    from pyquaternion import Quaternion
    kernel.ego_states_list = []
    for frame in synthetic[:4]:
        heading = Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]
        local_velocity = np.asarray(frame['ego_dynamic_state'][:2], dtype=np.float64)
        c, s = np.cos(heading), np.sin(heading)
        agent.current_velocity = np.array([c*local_velocity[0]-s*local_velocity[1], s*local_velocity[0]+c*local_velocity[1]])
        agent.current_heading = heading
        agent.rear_vehicle = SimpleNamespace(current_position=np.asarray(frame['ego2global_translation'][:2])-origin)
        ego = converter.convert_to_current_ego_state(len(kernel.ego_states_list))
        if not np.allclose(ego.rear_axle.array, frame['ego2global_translation'][:2], atol=1e-9, rtol=0):
            raise ValueError('Original WE global rear axle conversion differs')
        if not np.allclose(ego.dynamic_car_state.rear_axle_velocity_2d.array, local_velocity, atol=1e-9, rtol=0):
            raise ValueError('Original WE local velocity conversion differs')
        kernel.ego_states_list.append(ego)
    kernel.agent = agent
    # Supply actual history to the reference planner. Passing the complete
    # future list to a bounded history deque would select future observations.
    # This explicit input adapter differs from the manager's engine lifecycle.
    kernel.observations_list = observations[:4]
    # Equivalent metadata conversion of publisher traffic-light tuples through original NAVSIM scenario.
    converter.convert_to_traffic_lights = lambda _i: list(scenario.get_traffic_light_status_at_iteration(0))
    kernel.converter = converter
    we_scene = dict(base_timestamp=int(synthetic[0]['timestamp']), sample_rate=10)
    kernel.planner_input, initialization = kernel._get_planner_inputs(we_scene)
    if kernel.planner_input.history.current_state[1] is not observations[3]:
        raise ValueError('Reference planner history did not end at actual current observation')
    kernel._pdm_closed.initialize(initialization)
    reference = kernel._pdm_closed.compute_planner_trajectory(kernel.planner_input)
    kernel._map_api = initialization['map_api']
    kernel.observations_list = observations
    kernel._scoring_pool = Pool(8, initializer=cpu_worker_init)
    try:
        result = kernel.compute_pdm_scores(reference)
    except BaseException:
        stop_owned_pool(kernel._scoring_pool)
        raise
    else:
        kernel._scoring_pool.close()
        kernel._scoring_pool.join()
    for name, value in result.items():
        if np.asarray(value).shape != (8192,) or not np.isfinite(value).all():
            raise ValueError('Original scoring output shape/nonfinite: ' + name)
    output = {synthetic[3]['token']: result}
    with (args.output / 'worldengine_8192_logged_traffic_derived_rewards.pkl').open('wb') as stream:
        pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    row.update(reward_token=synthetic[3]['token'], candidates=8192, all_metrics_finite=True,
        metrics={k: dict(shape=list(v.shape), minimum=float(np.min(v)), maximum=float(np.max(v))) for k,v in result.items()},
        candidate_row_order='unchanged original test_8192_kmeans.npy', reference_removed_once_by_original_score_impl=True,
        output_sha256=sha(args.output / 'worldengine_8192_logged_traffic_derived_rewards.pkl'),
        protocol=dict(traffic='original raw OpenScene global detections plus logged ego vehicle',
            ego='publisher synthetic rear axle; unchanged WE converter uses zero acceleration/steering',
            proposal_seconds=4, observations=9, interpolated_observations=41,
            route='identical publisher original/synthetic roadblock IDs; original WE PDM route correction',
            progress='unchanged WE batch=True: each candidate normalized against PDM reference',
            scoring_workers=8, simscale_v2_equivalence_claimed=False, full_SimEngine_execution=False,
            raw_nuplan_database_input_equivalence_claimed=False))
    row['input_adapter_differences_from_engine_lifecycle'] = [
        'Reference planner receives only four actual history observations, not full future list.',
        'Published original/synthetic identical roadblock IDs supply the route; no engine navigation runs.',
        'Original raw OpenScene annotations converted by unchanged NAVSIM replace nuPlan database object loading.',
        'No engine lifecycle, renderer, or claim of full DenseRewardManager replay identity.'
    ]



def runtime_fingerprint(source):
    value=hashlib.sha256()
    files=sorted([*(source/'projects/AlgEngine').rglob('*.py'),*(source/'ascend').glob('*.py')],key=lambda p:p.relative_to(source).as_posix())
    for path in files:value.update(path.relative_to(source).as_posix().encode());value.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return value.hexdigest()

def maps_manifest(root):
    version=root/'nuplan-maps-v1.0.json';paths=[version,*sorted(root.rglob('*.gpkg'))]
    if not version.is_file() or len(paths)<2:raise ValueError('Original nuPlan maps v1.0 and GPKG payloads required')
    return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'metadata', 'conversion-root', 'vocab', 'maps', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--scene-id', default=SCENE_ID)
    parser.add_argument('--original-log', default=LOG)
    parser.add_argument('--folder-name', default=FOLDER)
    parser.add_argument('--expected-anchor-token', default='ab75b7addff3512d-000')
    parser.add_argument('--execute', action='store_true', help='Execute CPU scoring; omitted means a read-only plan')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(result='dry_run_not_executed', source=str(args.source), expected_runtime_sha256=args.expected_runtime_sha256,
            scene_id=args.scene_id, original_log=args.original_log, folder_name=args.folder_name, anchor_token=args.expected_anchor_token,
            output=str(args.output), protocol='Original WE 4-second logged-traffic derived labels; not SimScale v2 equivalence'), indent=2))
        return
    global np, torch
    import numpy as np
    import torch
    globals().update(SCENE_ID=args.scene_id, LOG=args.original_log, FOLDER=args.folder_name)
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    cutoff = min(deadline.timestamp()-120, time.time()+args.max_seconds)
    if deadline.tzinfo is None or not 1 <= args.max_seconds <= 3600 or cutoff <= time.time() or args.output.exists():
        parser.error('New output and authorized future deadline required')
    if list(Path('/dev').glob('davinci*')):
        parser.error('CPU only container required')
    args.output.mkdir(parents=True)
    row = dict(result='running', started_unix_seconds=time.time(), helper_sha256=sha(Path(__file__)),
        original_inputs_modified=False, published_reward_modified=False, fake_rewards_created=False)
    def expired(_signal, _frame):
        raise TimeoutError('Owned CPU reward pilot deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1,int(cutoff-time.time())))
    try:
        root = Path('/sys/fs/cgroup')
        limitpath = root / 'memory.max' if (root / 'memory.max').exists() else root / 'memory/memory.limit_in_bytes'
        limit = limitpath.read_text().strip()
        if limit == 'max' or int(limit) > 64*GIB:
            raise MemoryError('Finite <=64GiB container cap required')
        if args.metadata.stat().st_size != META_SIZE or sha(args.metadata) != META_SHA or sha(args.vocab) != VOCAB_SHA:
            raise ValueError('Official whole metadata/vocabulary size/SHA differs')
        row['runtime_source_sha256_before'] = runtime_fingerprint(args.source)
        if row['runtime_source_sha256_before'] != args.expected_runtime_sha256:raise ValueError('Selected runtime source differs')
        row['map_input_manifest_before'] = maps_manifest(args.maps)
        os.environ['NUPLAN_MAPS_ROOT'] = str(args.maps)
        import torch_npu
        row['device_initialization_before'] = dict(npu=torch.npu.is_initialized(), cuda=torch.cuda.is_initialized())
        if any(row['device_initialization_before'].values()):
            raise RuntimeError('Unexpected accelerator initialization before CPU reward pilot')
        imports(args)
        we = args.source / 'projects/SimEngine/worldengine'
        row['source_locks'] = {}
        for name, expected in LOCKS.items():
            actual = hashlib.sha256((we/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            if actual != expected:
                raise ValueError('Original WE scoring source differs: ' + name)
            row['source_locks'][name] = actual
        row['kernel'] = extract_methods(we/'manager/dense_reward_manager.py', 'DenseRewardManager',
            ['compute_pdm_scores','_interpolate_observations','_interpolate_single_observation','batched_global_transform',
             '_get_planner_inputs','get_route_roadblocks_ids','get_current_roadblock_ids'], 'OriginalScoringKernel',
            ('wrap_to_pi','_score_proposals_impl'))
        row['ego_converter'] = extract_methods(we/'components/agents/policy/pdm_planner/utils/WE2pdm_utils.py',
            'WE2NuPlanConverter', ['__init__','convert_to_current_ego_state'], 'OriginalEgoConverter',
            ('denormalize_from_ego_center',))
        source, synthetic = inputs(args, row)
        if synthetic[3]['token'] != args.expected_anchor_token:raise ValueError('Original synthetic anchor identity differs')
        save(args.output/'reward_pilot.json', row)
        score(args, source, synthetic, row)
        row['device_initialization_after'] = dict(npu=torch.npu.is_initialized(), cuda=torch.cuda.is_initialized())
        if any(row['device_initialization_after'].values()):
            raise RuntimeError('Unexpected accelerator initialization during CPU reward pilot')
        row['metadata_sha256_after'] = sha(args.metadata)
        row['vocab_sha256_after'] = sha(args.vocab)
        if row['metadata_sha256_after'] != META_SHA or row['vocab_sha256_after'] != VOCAB_SHA:
            raise ValueError('Original input changed')
        row['runtime_source_sha256_after'] = runtime_fingerprint(args.source)
        row['map_input_manifest_after'] = maps_manifest(args.maps)
        if row['runtime_source_sha256_after'] != args.expected_runtime_sha256 or row['map_input_manifest_after'] != row['map_input_manifest_before']:raise ValueError('Runtime or maps changed')
        row['result'] = 'pass_original_WE_scoring_derived_protocol_not_SimScale_v2_equivalence'
    except BaseException as error:
        import traceback
        row.update(result='failed_retained', error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        row['completed_unix_seconds'] = time.time()
        save(args.output/'reward_pilot.json', row)
    print(json.dumps(dict(result=row['result'], error=row.get('error')), indent=2), flush=True)
    raise SystemExit(0 if row['result'].startswith('pass_') else 1)


if __name__ == '__main__':
    main()
