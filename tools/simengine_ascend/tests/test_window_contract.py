"""Synthetic CPU window/CLI checks; no model, renderer or device execution."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from simengine_ascend import run_simulation as simulation
from simengine_ascend import window_contract as window


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def scene(self, length=24):
        return {'official-key': dict(log_length=length, cameras={c: {} for c in window.CAMERAS},
            metadata={'openscene_data_infos_dict': {i: dict(token='real-%d'%i,
                cams={c: dict(data_path='source/%s/%d.jpg'%(c, i)) for c in window.CAMERAS}) for i in range(length)}})}

    def write_scene(self, value):
        p = self.root/'scene.pkl'; p.write_bytes(pickle.dumps(value))
        return p, hashlib.sha256(p.read_bytes()).hexdigest()

    def check_scene(self, value, future=20):
        p, sha = self.write_scene(value)
        return window.inspect_scene_window(p, sha, ['official-key'], 4, future)

    def test_exact_12_default_and_24_indices_without_mutating_scene(self):
        self.assertEqual(window.window_contract()['consumed_control_indices'], list(range(4, 12)))
        self.assertEqual(window.window_contract(4, 20)['consumed_control_indices'], list(range(4, 24)))
        value = self.scene(); before = pickle.dumps(value)
        result = self.check_scene(value)
        self.assertEqual(result['configured_window']['final_frame'], 23)
        self.assertEqual(result['scenes'][0]['checked_real_frames'], 24)
        self.assertEqual(result['padded_frames'], 0)
        self.assertEqual(before, pickle.dumps(value))

    def test_invalid_history_future_bool_and_noninteger_rejected(self):
        for h, f in [(3, 20), (5, 20), (4, 19), (4, 21), (True, 20), (4, True), (4., 20), (4, 8.)]:
            with self.subTest(history=h, future=f), self.assertRaises(ValueError): window.window_contract(h, f)

    def test_short_log_missing_real_frames_camera_and_duplicate_token_rejected(self):
        for kind in ('short-log', 'short-metadata', 'missing-camera', 'missing-scene-camera', 'duplicate-token'):
            value = self.scene(); row = value['official-key']
            if kind == 'short-log': row['log_length'] = 23
            if kind == 'short-metadata': row['metadata']['openscene_data_infos_dict'].pop(23)
            if kind == 'missing-scene-camera': row['cameras'].pop('CAM_F0')
            if kind == 'missing-camera': row['metadata']['openscene_data_infos_dict'][23]['cams'].pop('CAM_F0')
            if kind == 'duplicate-token': row['metadata']['openscene_data_infos_dict'][23]['token'] = 'real-0'
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.check_scene(value)

    def test_wrong_hash_never_loads_pickle_and_extra_bytes_rejected(self):
        p, sha = self.write_scene(self.scene())
        with mock.patch.object(window.pickle, 'load', side_effect=AssertionError('must not load')):
            with self.assertRaises(ValueError): window.inspect_scene_window(p, '0'*64, ['official-key'], 4, 20)
        p.write_bytes(p.read_bytes()+b'extra'); sha = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaises(ValueError): window.inspect_scene_window(p, sha, ['official-key'], 4, 20)

    def test_24_dry_command_requires_trusted_SHA_and_keeps_dense_horizon_guard(self):
        p, sha = self.write_scene(self.scene())
        assets=self.root/'assets'; maps=self.root/'maps'; assets.mkdir(); maps.mkdir()
        manifest=self.root/'manifest.json'; manifest.write_text('{}')
        ids=self.root/'ids.json'; ids.write_text('["official-key"]')
        args=['--scene-pkl', str(p), '--asset-root', str(assets), '--maps-root', str(maps),
              '--validated-manifest',str(manifest),'--expected-scene-ids',str(ids),'--device','npu:0',
              '--output',str(self.root/'new-output'),'--deadline-utc',(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),
              '--mode','replay','--num-future','20','--dry-run']
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): simulation.main(args)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): simulation.main(args+['--scene-sha256',sha,'--dense-rewards'])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): simulation.main(args+['--scene-sha256',sha,'--pdm-metrics'])
        stream=io.StringIO(); before=set(sys.modules)
        with redirect_stdout(stream): self.assertEqual(simulation.main(args+['--scene-sha256',sha]),0)
        report=json.loads(stream.getvalue()); self.assertIn('num_history=4',report['command']);self.assertIn('num_future=20',report['command'])
        self.assertEqual(report['configured_window']['frames'],24)
        self.assertEqual(report['scene_window_preflight']['padded_frames'],0)
        self.assertFalse((self.root/'new-output').exists())
        self.assertNotIn('torch_npu',set(sys.modules)-before)


if __name__=='__main__':unittest.main(verbosity=2)
