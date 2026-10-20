"""CPU filesystem/process fixtures; no planner model, renderer or accelerator."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'projects/AlgEngine')]
from closed_loop.monitor import AsyncFileMonitor
from simengine_ascend import file_protocol as protocol
from simengine_ascend.closed_loop_runtime import ClosedLoopRuntime,algengine_identity,assert_module_origins
from simengine_ascend.launch_closed_loop import launch,validate_runner_reports


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
    def tearDown(self):self.directory.cleanup()

    def test_atomic_pickle_and_numpy_roundtrip(self):
        path=self.root/'frame_0.pkl';protocol.atomic_pickle(path,{'index':0})
        with path.open('rb') as stream:self.assertEqual(pickle.load(stream),{'index':0})
        output=self.root/'scene_4.npy';value=np.zeros((40,3),dtype=np.float32)
        protocol.atomic_numpy(output,value)
        np.testing.assert_array_equal(protocol.wait_numpy(output,.1),value)
        self.assertEqual(len(list(self.root.glob('*.tmp'))),0)

    def test_atomic_publication_preserves_existing_files_on_error(self):
        path=self.root/'owned.bin';path.write_bytes(b'original')
        def failing(stream):
            stream.write(b'partial');raise RuntimeError('fixture writer error')
        with self.assertRaises(RuntimeError):protocol.atomic_write(path,failing)
        self.assertEqual(path.read_bytes(),b'original')
        with self.assertRaises(FileExistsError):protocol.atomic_write(path,lambda stream:stream.write(b'foreign'),replace=False)
        self.assertEqual(path.read_bytes(),b'original')
        self.assertEqual(list(self.root.glob('.*.tmp')),[])

    def test_wait_rejects_corruption_nonfinite_and_times_out(self):
        with self.assertRaises(TimeoutError):protocol.wait_numpy(self.root/'absent.npy',.02)
        path=self.root/'corrupt.npy';path.write_bytes(b'broken')
        with self.assertRaises(ValueError):protocol.wait_numpy(path,.02)
        bad=np.zeros((40,3),dtype=np.float32);bad[2,0]=np.nan
        with self.assertRaises(ValueError):protocol.validate_trajectory(bad)
        with self.assertRaises(ValueError):protocol.validate_trajectory(np.zeros((8,3)))
        stop=self.root/'STOP';stop.touch()
        with self.assertRaises(InterruptedError):protocol.wait_numpy(self.root/'absent.npy',1,stop_file=stop)

    def test_monitor_consecutive_queue_scene_switch_and_error_propagation(self):
        for scene in ('b','a'):
            for index in range(4):protocol.atomic_pickle(self.root/('%s_%d.pkl'%(scene,index)),{'frame':index})
        (self.root/'ignored_bad.pkl').write_bytes(b'x')
        monitor=AsyncFileMonitor(self.root,max_wait_time=.1,max_frame_per_scene=5)
        self.assertEqual(monitor.prefix,'a')
        self.assertEqual(len(monitor.get_file_queue()),4)
        self.assertEqual(monitor.get_current_queue(),[])
        protocol.atomic_pickle(self.root/'a_4.pkl',{'frame':4})
        self.assertTrue(monitor.get_current_queue()[-1].endswith('a_4.pkl'))
        self.assertTrue(monitor._reinitialize_queue());self.assertEqual(monitor.prefix,'b')
        empty=self.root/'empty';empty.mkdir()
        with self.assertRaises(TimeoutError):AsyncFileMonitor(empty,max_wait_time=.02,check_interval=.005)
        with self.assertRaises(ValueError):AsyncFileMonitor(empty,max_wait_time=float('inf'))

    def test_cuda_default_does_not_load_external_ascend(self):
        before=set(sys.modules)
        args=SimpleNamespace(device='cuda:0',precision=None)
        runtime=ClosedLoopRuntime(args,ROOT)
        self.assertEqual(runtime.root,ROOT)
        self.assertEqual(set(sys.modules)-before,set())

    def test_npu_requires_explicit_identity_and_rejects_shadow(self):
        args=SimpleNamespace(device='npu:0',precision=None)
        with self.assertRaises(ValueError):ClosedLoopRuntime(args,ROOT)
        selected=self.root/'selected_alg'
        (selected/'ascend').mkdir(parents=True)
        (selected/'projects/AlgEngine/mmdet3d_plugin').mkdir(parents=True)
        fake=self.root/'foreign.py';fake.write_text('')
        previous=sys.modules.get('mmdet3d_plugin')
        sys.modules['mmdet3d_plugin']=SimpleNamespace(__file__=str(fake))
        try:
            with self.assertRaises(RuntimeError):assert_module_origins(selected)
        finally:
            if previous is None:del sys.modules['mmdet3d_plugin']
            else:sys.modules['mmdet3d_plugin']=previous

    def test_launcher_success_failure_deadline_and_foreign_process(self):
        foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
        try:
            for name,code,expected,seconds in [('success','pass',0,4),('failure','raise RuntimeError("fixture")',1,4),
                                                ('deadline','import time;time.sleep(10)',2,.4)]:
                plan=self.root/(name+'.json')
                plan.write_text(json.dumps({phase:dict(argv=[sys.executable,'-c',code],cwd=str(self.root))
                                            for phase in ('simulation','planner')}))
                deadline=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat()
                self.assertEqual(launch(plan,self.root/name,deadline),expected)
                self.assertIsNone(foreign.poll())
                report=json.loads((self.root/name/'launcher_report.json').read_text())
                self.assertTrue(all(row['actual_exit'] is not None for row in report['children'].values()))
                self.assertFalse(report['real_closed_loop_accepted'])
        finally:foreign.terminate();foreign.wait()

    def test_launcher_cleans_its_children_when_stop_publication_fails(self):
        plan=self.root/'plan.json'
        plan.write_text(json.dumps({phase:dict(argv=[sys.executable,'-c','import time;time.sleep(10)'],cwd=str(self.root))
                                    for phase in ('simulation','planner')}))
        deadline=(datetime.now(timezone.utc)+timedelta(seconds=.4)).isoformat()
        def simulated_disk_failure(path,*args,**kwargs):
            if Path(path).name=='STOP':raise OSError('fixture full disk')
            return protocol.atomic_write(path,*args,**kwargs)
        with mock.patch('simengine_ascend.launch_closed_loop.atomic_write',side_effect=simulated_disk_failure):
            self.assertEqual(launch(plan,self.root/'disk_failure',deadline),2)
        report=json.loads((self.root/'disk_failure/launcher_report.json').read_text())
        self.assertEqual(report['result'],'cleanup_error')
        self.assertTrue(report['cleanup_errors'])
        self.assertTrue(all(row['actual_exit'] is not None for row in report['children'].values()))

    def test_original_runner_report_is_required_and_failures_are_not_swallowed(self):
        with self.assertRaises(ValueError):validate_runner_reports(self.root,1)
        path=self.root/'runner_report_fixture.json'
        row=dict(scenario_name='scene',log_name='log',planner_name=None,
                 succeeded=True,duration_s=.25,error_message=None)
        path.write_text(json.dumps([row]))
        self.assertEqual(validate_runner_reports(self.root,1)['completed_unique_scenarios'],1)
        self.assertEqual(validate_runner_reports(self.root,1,['scene'])['expected_scene_ids'],['scene'])
        with self.assertRaises(ValueError):validate_runner_reports(self.root,1,['another_scene'])
        with self.assertRaises(ValueError):validate_runner_reports(self.root,2)
        path.write_text(json.dumps([row,row]))
        with self.assertRaises(ValueError):validate_runner_reports(self.root,2)
        for changed in ({'succeeded':False},{'error_message':'actual scene error'},{'duration_s':float('nan')}):
            path.write_text(json.dumps([dict(row,**changed)]))
            with self.assertRaises(ValueError):validate_runner_reports(self.root,1)


if __name__=='__main__':unittest.main(verbosity=2)
