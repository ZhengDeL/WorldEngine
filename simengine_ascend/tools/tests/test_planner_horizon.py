"""CPU queue/termination fixtures, not real scene or accelerator acceptance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('planner_horizon_fixture',ROOT/'projects/AlgEngine/closed_loop/sim_test.py')
planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)


class PlannerHorizonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.frames=self.root/'frames';self.frames.mkdir()
        self.flag=self.root/'simulation_completed.flag';self.report=self.root/'simulation_launch_report.json'

    def producer(self,history=4,frames=12):
        value=dict(result='running_original_simulation',configured_window=dict(num_history=history,num_future=frames-history,
            frames=frames,final_frame=frames-1,consumed_control_indices=list(range(history,frames))),
            scene_window_preflight=dict(original_log_length=24))
        self.report.write_text(json.dumps(value));return value

    def monitor(self,frames=12,prefix='sceneA'):
        for i in range(frames):(self.frames/(prefix+'_'+str(i)+'.pkl')).write_bytes(b'CPU monitor fixture')
        return planner.AsyncFileMonitor(self.frames,max_history=4,max_frame_per_scene=frames,max_wait_time=1,
                                       stop_file_path=str(self.flag))

    def final_queue(self,monitor):
        current=list(monitor.file_history)
        while monitor.next_idx<monitor.max_frame_per_scene:current=monitor.get_current_queue()
        return current

    def test_bind_rendered12_not_original24_context(self):
        self.producer();value=planner.planner_window(4,12,self.report,True)
        self.assertEqual(value['frames'],12);self.assertEqual(value['num_future'],8)
        self.assertEqual(value['source'],'simulation_launch_report.configured_window')

    def test_mismatched_renderer24_planner12_is_rejected(self):
        self.producer(frames=24)
        with self.assertRaises(ValueError):planner.planner_window(4,12,self.report,True)

    def test_mismatched_history_or_typed_counts_rejected(self):
        value=self.producer();value['configured_window']['num_history']=4.0;self.report.write_text(json.dumps(value))
        with self.assertRaises(ValueError):planner.planner_window(4,12,self.report,True)
        self.producer()
        with self.assertRaises(ValueError):planner.planner_window(8,12,self.report,True)

    def test_controlled_NPU_missing_report_rejected(self):
        with self.assertRaises(ValueError):planner.planner_window(4,12,self.report,True)

    def test_explicit_standalone_replay_keeps_configuration_fallback(self):
        value=planner.planner_window(4,12,self.report,False)
        self.assertEqual(value['source'],'planner_configuration_without_producer_report')

    def test_last_required_plan_is_not_dropped_even_if_flag_arrives(self):
        self.flag.write_text('completed')
        monitor=SimpleNamespace(max_history=4,max_frame_per_scene=12,next_idx=11)
        queue=['frame7','frame8','frame9','frame10']
        current,step,active=planner.control_queue(monitor,queue,7,4,self.flag)
        self.assertTrue(active);self.assertEqual(step+4,11);self.assertIs(current,queue)

    def test_real_monitor32ms_completion_window_has_no_extra_prediction(self):
        monitor=self.monitor();current=self.final_queue(monitor)
        timer=threading.Timer(.032,lambda:self.flag.write_text('completed'));timer.start()
        try:_,_,active=planner.control_queue(monitor,current,8,4,self.flag)
        finally:timer.join()
        self.assertFalse(active);self.assertEqual(self.flag.read_text(),'completed')
        self.assertEqual(len(list(self.frames.glob('*.pkl'))),12)

    def test_late_completion_waits_without_model_work(self):
        monitor=self.monitor();current=self.final_queue(monitor)
        timer=threading.Timer(.18,lambda:self.flag.write_text('late complete'));timer.start()
        try:_,_,active=planner.control_queue(monitor,current,8,4,self.flag)
        finally:timer.join()
        self.assertFalse(active);self.assertTrue(self.flag.is_file())

    def test_missing_completion_and_next_scene_times_out_instead_of_passing(self):
        monitor=self.monitor();monitor.max_wait_time=.025;current=self.final_queue(monitor)
        with self.assertRaises(TimeoutError):planner.control_queue(monitor,current,8,4,self.flag)

    def test_pending_next_scene_queue_is_not_dropped(self):
        monitor=self.monitor();current=self.final_queue(monitor)
        for i in range(4):(self.frames/('sceneB_'+str(i)+'.pkl')).write_bytes(b'next CPU scene')
        queue,step,active=planner.control_queue(monitor,current,8,4,self.flag)
        self.assertTrue(active);self.assertEqual(step,0);self.assertEqual(monitor.prefix,'sceneB')
        self.assertEqual(monitor.next_idx,4);self.assertTrue(all('sceneB_'in p for p in queue))
        self.assertFalse(self.flag.exists())

    def test_late_next_scene_queue_is_waited_for(self):
        monitor=self.monitor();current=self.final_queue(monitor)
        def publish():
            for i in range(4):(self.frames/('sceneB_'+str(i)+'.pkl')).write_bytes(b'next CPU scene')
        timer=threading.Timer(.032,publish);timer.start()
        try:queue,step,active=planner.control_queue(monitor,current,8,4,self.flag)
        finally:timer.join()
        self.assertTrue(active);self.assertEqual(step,0);self.assertEqual(len(queue),4)
        self.assertEqual(monitor.prefix,'sceneB');self.assertFalse(self.flag.exists())

    def test_original24_frames_execute_twenty_required_controls(self):
        monitor=self.monitor(frames=24);current=list(monitor.file_history);planned=[]
        for scene_step in range(20):
            current,step,active=planner.control_queue(monitor,current,scene_step,4,self.flag)
            self.assertTrue(active);planned.append(step+4);current=monitor.get_current_queue()
        self.flag.write_text('completed')
        _,_,active=planner.control_queue(monitor,current,20,4,self.flag)
        self.assertFalse(active);self.assertEqual(planned,list(range(4,24)))

    def test_counter_mismatch_cannot_discard_the_last_necessary_control(self):
        monitor=SimpleNamespace(max_history=4,max_frame_per_scene=12,next_idx=11)
        with self.assertRaises(ValueError):planner.control_queue(monitor,['x']*4,8,4,self.flag)

    def test_overflow_or_incomplete_queue_rejected(self):
        monitor=SimpleNamespace(max_history=4,max_frame_per_scene=12,next_idx=13)
        with self.assertRaises(ValueError):planner.control_queue(monitor,['x']*4,9,4,self.flag)
        monitor.next_idx=4
        with self.assertRaises(ValueError):planner.control_queue(monitor,['x']*3,0,4,self.flag)


if __name__=='__main__':unittest.main(verbosity=2)
