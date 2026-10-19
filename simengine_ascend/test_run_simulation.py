"""CPU command, image and owned-process fixtures for the simulation wrapper."""
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from PIL import Image

from simengine_ascend import run_simulation as simulation


class SimulationWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='WE_simulator_CPU_fixture_')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.scene = self.folder / 'scene.pkl'
        self.scene.write_bytes(b'CPU fixture: no pickle loading is performed')
        self.manifest = self.folder / 'manifest.json'
        self.manifest.write_text('{"fixture_only":true}')
        self.ids = self.folder / 'scene_ids.json'
        self.ids.write_text('["official_source_dictionary_key_fixture"]')
        self.assets = self.folder / 'assets'
        self.maps = self.folder / 'maps'
        self.assets.mkdir()
        self.maps.mkdir()
        self.deadline = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()

    def args(self, mode='replay', reaction='NR'):
        return SimpleNamespace(
            scene_pkl=self.scene, asset_root=self.assets, maps_root=self.maps,
            validated_manifest=self.manifest, device='npu:1', output=self.folder / 'new run',
            mode=mode, reaction=reaction, cpu_threads=8, pdm_metrics=False,
            dense_rewards=False, planner_data_path=self.folder / 'plans',
            planner_client_folder=self.folder / 'frames', deadline_utc=self.deadline,
            stop_file=self.folder / 'STOP',
        )

    def values(self, args):
        return dict(item.lstrip('+').split('=', 1) for item in simulation.build_overrides(args))

    def test_replay_uses_original_logged_policy_and_explicit_NPU(self):
        values = self.values(self.args())
        self.assertEqual(values['distributed_mode'], '"SINGLE_NODE"')
        self.assertEqual(values['gpu'], 'false')
        self.assertEqual(values['renderer.device'], '"npu:1"')
        self.assertEqual(values['renderer.render_depth'], 'true')
        self.assertEqual(values['renderer.rasterization_backend'], '"cann"')
        self.assertEqual(values['ego_policy'], '"trajectory_policy"')
        self.assertEqual(values['ego_controller'], '"log_play_controller"')
        self.assertEqual(values['use_planner_actions'], 'false')
        self.assertNotIn('planner_data_path', values)
        self.assertEqual(values['debug_static_world'], 'false')

    def test_closed_loop_and_reactive_options_follow_original_testing_script(self):
        values = self.values(self.args('closed-loop', 'R'))
        self.assertEqual(values['ego_policy'], '"env_input_policy"')
        self.assertEqual(values['ego_client'], '"navformer_client"')
        self.assertEqual(values['agent_policy'], '"idm_policy"')
        self.assertEqual(values['agent_navigation'], '"idm_navigation"')
        self.assertEqual(values['use_planner_actions'], 'true')
        self.assertEqual(json.loads(values['planner_data_path']), str(self.folder / 'plans'))

    def test_paths_are_single_quoted_Hydra_scalars_without_shell_expansion(self):
        value = self.folder / '中文 folder $(literal) a=b [1]'
        self.assertEqual(json.loads(simulation.hydra_value(value)), str(value))
        self.assertIn('中文', simulation.hydra_value(value))

    def test_environment_keeps_CPU_PDM_and_explicit_shared_stop(self):
        environment = simulation.runner_environment(self.args())
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(environment['WORLDENGINE_CLOSED_LOOP_STOP_FILE'], str(self.folder / 'STOP'))
        self.assertEqual(environment['WORLDENGINE_ROOT'], str(simulation.ROOT))
        self.assertIn(str(simulation.SIMULATION), environment['PYTHONPATH'])

    def test_dry_run_does_not_load_Torch_pickle_or_create_output(self):
        output = self.folder / 'dry_output'
        arguments = [
            '--scene-pkl', str(self.scene), '--asset-root', str(self.assets),
            '--maps-root', str(self.maps), '--validated-manifest', str(self.manifest),
            '--device', 'npu:0', '--output', str(output), '--deadline-utc', self.deadline,
            '--expected-scene-ids', str(self.ids), '--mode', 'replay', '--dry-run',
        ]
        before = set(sys.modules)
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(simulation.main(arguments), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report['result'], 'command_prepared_not_executed')
        self.assertFalse(report['simulation_quality_accepted'])
        self.assertFalse(output.exists())
        self.assertNotIn('torch', set(sys.modules) - before)
        self.assertNotIn('torch_npu', set(sys.modules) - before)

    def test_existing_output_and_duplicate_scene_ids_rejected(self):
        with self.assertRaises(ValueError):
            simulation.fresh_output(self.assets)
        self.ids.write_text('["same","same"]')
        with self.assertRaises(ValueError):
            simulation.load_scene_ids(self.ids)

    def test_saved_JPEG_decode_and_corruption_checks(self):
        output = self.folder / 'image_job'
        folder = output / 'openscene_format/sensor_blobs/log/CAM_F0'
        folder.mkdir(parents=True)
        path = folder / 'frame.jpg'
        Image.new('RGB', (9, 7), color=(60, 90, 120)).save(path)
        result = simulation.validate_images(output, lambda: None)
        self.assertEqual(result['decoded_jpegs'], 1)
        self.assertFalse(result['full_camera_frame_coverage_accepted'])
        path.write_bytes(b'corrupt JPEG')
        with self.assertRaises(Exception):
            simulation.validate_images(output, lambda: None)

    def test_owned_process_success_error_cutoff_and_foreign_process(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'])
        try:
            simulation.run_owned([sys.executable, '-c', 'print("actual child")'],
                                 self.folder, dict(simulation.os.environ), self.folder / 'ok.log', lambda: None)
            self.assertIn('actual child', (self.folder / 'ok.log').read_text())
            with self.assertRaises(RuntimeError):
                simulation.run_owned([sys.executable, '-c', 'raise RuntimeError("fixture")'],
                                     self.folder, dict(simulation.os.environ), self.folder / 'bad.log', lambda: None)
            until = time.monotonic() + .3
            def guard():
                if time.monotonic() >= until:
                    raise TimeoutError('CPU fixture cutoff')
            with self.assertRaises(TimeoutError):
                simulation.run_owned([sys.executable, '-c', 'import time;time.sleep(10)'],
                                     self.folder, dict(simulation.os.environ), self.folder / 'cutoff.log', guard)
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait()


if __name__ == '__main__':
    unittest.main(verbosity=2)
