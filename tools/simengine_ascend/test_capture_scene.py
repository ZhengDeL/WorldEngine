"""Local synthetic fixtures only; no official pickle/device execution proof."""
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_scene as tool

spec = importlib.util.spec_from_file_location('capture_fixture_writer',
    tool.SIMULATION / 'worldengine/render/ascend/capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_fixture(root):
    """Exercise receipt checks with deliberately synthetic local file bytes."""
    prepared = root / 'prepared'
    prepared.mkdir()
    scene = prepared / 'scene.pkl'
    scene.write_bytes(b'synthetic fixture, never unpickled')
    asset_id = tool.preparation.DEFAULT_ID
    name = 'assets/%s/background/%s.ckpt' % (asset_id, asset_id)
    checkpoint = prepared / name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint, never torch-loaded')
    lock, lock_sha = tool.preparation.load_official_lock(tool.preparation.HERE / 'official_scene_assets.lock.json')
    receipt = dict(schema_version=1, result='pass_official_single_scene_asset_preparation',
        publisher=lock['repository'], huggingface_revision=lock['huggingface_revision'],
        modelscope_revision=lock['modelscope_revision'], asset_lock_sha256=lock_sha,
        sources=deepcopy(lock['files']), renderer_execution_accepted=False, complete_SimEngine_acceptance=False,
        selected=dict(dictionary_key='fixture_scene_key', scene_internal_id=asset_id, normalized_asset_id=asset_id),
        derived_scene=dict(path='scene.pkl', size=scene.stat().st_size, sha256=digest(scene), original_value_preserved=True),
        asset_root='assets', checkpoint=name,
        asset_files=[dict(output_path=name, archive_member=tool.preparation.PREFIX+'/'+asset_id+'/background/'+asset_id+'.ckpt',
                          size=checkpoint.stat().st_size, sha256=digest(checkpoint))])
    path = prepared / 'prepare_receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    return prepared, receipt, digest(path)


class TaggedCPUTensor:
    """A dtype/device tag fixture; this is explicitly not an accelerator tensor."""
    def __init__(self, array):
        self.array = np.asarray(array, np.float32)
        self.dtype, self.device = 'torch.float32', 'npu:0'
    def __len__(self):
        return len(self.array)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.array


def hook_fixture(output):
    n, c, width, height = 5, 2, 32, 16
    means = np.zeros((n, 3), np.float32)
    means[:, 2] = np.arange(n)+2
    arrays = dict(means=means, quats=np.tile([1, 0, 0, 0], (n, 1)), scales=np.full((n, 3), .2),
        opacities=np.full(n, .4), colors=np.full((c, n, 3), .3),
        viewmats=np.tile(np.eye(4), (c, 1, 1)), Ks=np.tile([[20, 0, 16.5], [0, 20, 8.5], [0, 0, 1]], (c, 1, 1)))
    tagged = {name: TaggedCPUTensor(value) for name, value in arrays.items()}
    renderer = SimpleNamespace(sensor_mapping={'CAM_A': 0, 'CAM_B': 1}, timestamp=12345678,
        collected_gaussians={name: tagged[name] for name in ('means', 'quats', 'scales', 'opacities')})
    renderer.collected_gaussians['rgbs'] = tagged['colors']
    scene = dict(cameras={name: dict(width=width, height=height) for name in renderer.sensor_mapping})
    proof = dict(selected=dict(dictionary_key='fixture', scene_internal_id='fixture_internal', normalized_asset_id='fixture_asset'),
        checkpoint=dict(name='assets/fixture/background/fixture.ckpt', sha256='1'*64),
        prepare_receipt_sha256='2'*64, asset_lock_sha256='3'*64, scene_sha256='4'*64)
    request = dict(output=str(output), device='npu:0', reserve_free_bytes=0, source_commit='a'*40,
                   source=dict(sha256='b'*64), reaction='NR')
    kwargs = dict(tagged, width=width, height=height, tile_size=16, packed=False, near_plane=.01, far_plane=1e10,
        render_mode='RGB+ED', sparse_grad=False, absgrad=True, rasterize_mode='antialiased', radius_clip=0.)
    return renderer, scene, proof, request, kwargs


class CaptureSceneTests(unittest.TestCase):
    def test_receipt_bound_prepared_scene_and_full_checkpoint_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, receipt, receipt_sha = prepared_fixture(Path(temp))
            proof = tool.verify_prepared(prepared, receipt_sha, lambda: None)
            self.assertEqual(proof['all_extracted_asset_files_rehashed'], 1)
            self.assertEqual(proof['checkpoint']['sha256'], receipt['asset_files'][0]['sha256'])
            self.assertFalse(proof['publisher_containers_rehashed_by_capture_tool'])

    def test_receipt_checkpoint_mutation_and_extra_asset_rejected(self):
        for fault in ('receipt', 'checkpoint', 'extra'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                prepared, receipt, expected = prepared_fixture(Path(temp))
                if fault == 'receipt':
                    (prepared / 'prepare_receipt.json').write_text('{}')
                elif fault == 'checkpoint':
                    (prepared / receipt['checkpoint']).write_bytes(b'changed')
                else:
                    (prepared / 'assets' / 'unrecorded.bin').write_bytes(b'extra')
                with self.assertRaises(ValueError):
                    tool.verify_prepared(prepared, expected, lambda: None)

    def test_false_official_revision_preservation_and_traversal_rejected(self):
        for fault in ('revision', 'preservation', 'traversal'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                prepared, receipt, _ = prepared_fixture(Path(temp))
                if fault == 'revision':
                    receipt['huggingface_revision'] = '0'*40
                elif fault == 'preservation':
                    receipt['derived_scene']['original_value_preserved'] = False
                else:
                    receipt['asset_files'][0]['output_path'] = '../outside.ckpt'
                path = prepared / 'prepare_receipt.json'
                path.write_text(json.dumps(receipt))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    tool.verify_prepared(prepared, digest(path), lambda: None)

    def test_exact_hook_exports_all_original_arrays_and_cameras(self):
        with tempfile.TemporaryDirectory() as temp:
            state = hook_fixture(Path(temp))
            renderer, scene, proof, request, kwargs = state
            hook = tool.CaptureOnce(renderer, scene, proof, request, capture.write_capture, lambda: None)
            with self.assertRaises(tool.CaptureComplete):
                hook(**kwargs)
            self.assertEqual(hook.calls, 1)
            arrays, manifest = capture.load_capture(hook.path)
            for name in tool.ARRAYS:
                np.testing.assert_array_equal(arrays[name], kwargs[name].array)
            self.assertEqual(manifest['shape'], dict(gaussians=5, cameras=2))
            self.assertEqual(manifest['render']['width'], 32)
            self.assertEqual(manifest['render']['height'], 16)
            self.assertEqual(manifest['render']['render_mode'], 'RGB+ED')
            self.assertEqual(manifest['provenance']['camera_names'], ['CAM_A', 'CAM_B'])
            self.assertEqual(manifest['provenance']['official_asset_manifest_sha256'], '2'*64)
            with self.assertRaisesRegex(RuntimeError, 'exactly once'):
                hook(**kwargs)

    def test_changed_full_arrays_camera_resolution_device_and_mode_rejected(self):
        for fault in ('array', 'resolution', 'camera', 'device', 'dtype', 'mode'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                renderer, scene, proof, request, kwargs = hook_fixture(Path(temp))
                if fault == 'array':
                    kwargs['means'] = TaggedCPUTensor(kwargs['means'].array[:1])
                elif fault == 'resolution':
                    kwargs['width'] = 16
                elif fault == 'camera':
                    renderer.sensor_mapping = {'CAM_B': 0, 'CAM_A': 1}
                elif fault == 'device':
                    kwargs['Ks'].device = 'cpu'
                elif fault == 'dtype':
                    kwargs['Ks'].dtype = 'torch.float16'
                else:
                    kwargs['render_mode'] = 'RGB'
                hook = tool.CaptureOnce(renderer, scene, proof, request, capture.write_capture, lambda: None)
                with self.assertRaises(ValueError):
                    hook(**kwargs)
                self.assertFalse((Path(temp) / 'capture').exists())

    def test_disk_guard_and_real_export_exception_do_not_complete(self):
        for fault in ('space', 'writer'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                renderer, scene, proof, request, kwargs = hook_fixture(Path(temp))
                if fault == 'space':
                    request['reserve_free_bytes'] = 10**30
                def writer(*args):
                    raise OSError('real export failed')
                hook = tool.CaptureOnce(renderer, scene, proof, request, writer, lambda: None)
                with self.assertRaises(OSError):
                    hook(**kwargs)
                self.assertIsNone(hook.path)

    def test_stop_guard_before_and_after_export_remains_error(self):
        for after_export in (False, True):
            with self.subTest(after_export=after_export), tempfile.TemporaryDirectory() as temp:
                renderer, scene, proof, request, kwargs = hook_fixture(Path(temp))
                calls = []
                def guard():
                    calls.append(None)
                    if len(calls) >= (2 if after_export else 1):
                        raise TimeoutError('deadline')
                hook = tool.CaptureOnce(renderer, scene, proof, request, capture.write_capture, guard)
                with self.assertRaises(TimeoutError):
                    hook(**kwargs)
                self.assertEqual((Path(temp) / 'capture/capture.json').exists(), after_export)

    def test_initialization_is_original_replay_and_explicit_RGBED(self):
        request = dict(maps_root='/maps', output='/new', device='npu:2', small_manifest='/small.json',
                       cpu_threads=8, reaction='R')
        proof = dict(scene_path='/scene.pkl', asset_root='/assets')
        values = dict(row.lstrip('+').split('=', 1) for row in tool.initialization_overrides(request, proof))
        self.assertEqual(values['renderer.render_depth'], 'true')
        self.assertEqual(values['renderer.device'], '"npu:2"')
        self.assertEqual(values['agent_policy'], '"idm_policy"')
        self.assertEqual(values['use_planner_actions'], 'false')
        self.assertEqual(values['ego_policy'], '"trajectory_policy"')
        request['device'] = 'cuda:1'
        values = dict(row.lstrip('+').split('=', 1) for row in tool.initialization_overrides(request, proof))
        self.assertNotIn('renderer.rasterization_backend', values)
        self.assertNotIn('renderer.npu_compat_manifest', values)
        self.assertEqual(values['renderer.device'], '"cuda:1"')

    def test_dry_run_never_imports_torch_or_executes_pickle(self):
        torch_was_imported = 'torch' in sys.modules
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared, _, receipt_sha = prepared_fixture(root)
            maps = root / 'maps'
            maps.mkdir()
            manifest = root / 'small.json'
            manifest.write_text('{}')
            output = root / 'capture_output'
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                status = tool.main(['--prepared-dir', str(prepared), '--prepare-receipt-sha256', receipt_sha,
                    '--maps-root', str(maps), '--small-manifest', str(manifest), '--device', 'npu:0',
                    '--source-commit', 'a'*40, '--output', str(output),
                    '--deadline-utc', '2099-01-01T00:00:00+00:00', '--dry-run'])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue())['result'], 'capture_prepared_not_executed')
            self.assertFalse(output.exists())
            self.assertEqual('torch' in sys.modules, torch_was_imported)

    def test_duplicate_json_and_nonexplicit_device_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.json'
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                tool.read_json(path)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(['--prepared-dir', '.', '--prepare-receipt-sha256', '0'*64, '--maps-root', '.',
                '--device', 'npu', '--source-commit', 'a'*40, '--output', 'new', '--deadline-utc', '2099-01-01T00:00:00Z'])


if __name__ == '__main__':
    unittest.main()
