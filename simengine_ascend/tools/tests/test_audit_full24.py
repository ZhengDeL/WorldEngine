"""CPU-only checks for the explicit current-source full24 audit contract."""
import unittest
from simengine_ascend.tools import audit_full24 as audit


class Full24ContractTests(unittest.TestCase):
    def test_planner_must_explicitly_monitor_real_24_frames(self):
        command=['python','sim_test.py','--cfg-options','data.test.queue_length=4','sim.max_frame_per_scene=24']
        self.assertTrue(audit.validate_planner_window_command(command,24)['explicit_planner_window_required'])
        for invalid in [[],['--cfg-options','sim.max_frame_per_scene=12'],
                        command+['sim.max_frame_per_scene=24'],command+['--cfg-options','x=1']]:
            with self.subTest(command=invalid),self.assertRaises(ValueError):
                audit.validate_planner_window_command(invalid,24)

    def test_current_route_checks_actual_backend_and_recorded_root(self):
        runtime={'AlgEngine_runtime':dict(source_sha256=audit.ALG_SOURCE,python_files=245),
                 'precision':'fp32','precision_before':{'msda':{'backend':'cuda-compatible'}},
                 'precision_after':{'msda':{'backend':'cuda-compatible'}}}
        command=['python','--precision','fp32','--algengine-root','/code/algorithm',
                 '--expected-algengine-source-sha256',audit.ALG_SOURCE,'--msda-backend','cuda-compatible']
        self.assertEqual(audit.validate_current_planner_contract(runtime,command,'/code/algorithm')['precision'],'fp32')
        with self.assertRaises(ValueError):audit.validate_current_planner_contract(runtime,command,'/foreign')
        runtime['precision_after']['msda']['backend']='reference'
        with self.assertRaises(ValueError):audit.validate_current_planner_contract(runtime,command,'/code/algorithm')

    def test_distinct_explicit_host_pair_required(self):
        self.assertEqual(audit.host_mapping([0,1])['WORLDENGINE_HOST_DEVICE_IDS'],'0,1')
        for pair in ([0,0],[-1,1],[0,16]):
            with self.subTest(pair=pair),self.assertRaises(ValueError):audit.host_mapping(pair)


if __name__=='__main__':unittest.main(verbosity=2)
