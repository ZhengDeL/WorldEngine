"""CPU contract fixtures; no real scene, model, scoring formula or NPU run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from simengine_ascend.planner_optimization import IMPLEMENTATION
from simengine_ascend.tools import audit_optimized_default8 as audit


class OptimizedDefault8Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'reference.json'
        self.inventory = dict(model='vadv2', reaction='NR', original_scene_sha256=audit.SCENE_SHA,
                              frames=[dict(step=i, sha256='1'*64) for i in range(12)],
                              images=[dict(step=i//8, camera=str(i%8), sha256='2'*64) for i in range(96)],
                              controls=[dict(step=i, plan_sha256='3'*64, input_sha256='4'*64) for i in range(4,12)],
                              dense_rows=[dict(step=i, pickle_sha256='5'*64, metrics={'score': {'raw_sha256': '6'*64}}) for i in range(3,12)],
                              PDM_CSV_sha256='7'*64, PDM_metrics={'score': .79057}, PDM_CSV_precision_decimals=5)
        self.reference = dict(schema_version=1, kind='public212_default8_persisted_prefix_reference',
                              reference_runtime=dict(sha256=audit.REFERENCE_SOURCE,files=212),
                              reference_auditor_sha256=audit.REFERENCE_AUDITOR_SHA,
                              reference_creator_sha256=audit.sha(audit.__file__),
                              reference_audit_sha256='8'*64, inventory=self.inventory)

    def write(self):
        self.path.write_text(json.dumps(self.reference),encoding='utf-8')
        return audit.sha(self.path)

    def read(self): return audit.read_reference(self.path,self.write(),'vadv2','NR')

    def report(self):
        return dict(result='pass_original_default8_scored_control_window',
                    auditor_sha256=audit.REFERENCE_AUDITOR_SHA, model='vadv2',reaction='NR',precision='fp32',
                    source=dict(sha256=audit.REFERENCE_SOURCE,files=212),prepare_receipt_sha256=audit.PREPARE_SHA,
                    formal_checkpoint_sha256=audit.MODELS['vadv2'][2],
                    planner_provenance=dict(AlgEngine_runtime=dict(source_sha256=audit.REFERENCE_ALG_SOURCE,python_files=245),
                                           AlgEngine_declared_commit='55efdef54139c63ab9afce9178ebb3bd130717ef', image_id=audit.IMAGE),
                    camera_frame_coverage=dict(decoded_jpegs=96),consumption=dict(consumed_control_indices=list(range(4,12))),
                    scoring=dict(all_dense_arrays_finite=True), acceptance=dict(original_PDM_and_dense_artifacts=True))

    def runtime(self,predictions=9):
        evidence=dict(requested='point-parallel-last-frame',query_chunk=8192,point_parallel_calls=81,
                      original_msda_fallback_calls=0,original_frame_fallback_calls=0,fallback_reasons={},
                      last_frame_calls=predictions,last_frame_source_contract='pinned_original_methods',
                      preserved_NumPy_draws=6*predictions)
        runtime=dict(AlgEngine_runtime=dict(source_sha256=audit.ALG_SOURCE,python_files=249),precision='fp32',
                     precision_before=dict(msda=dict(backend='cuda-compatible')),precision_after=dict(msda=dict(backend='cuda-compatible')),
                     inference_optimization_source=dict(source_sha256=audit.ALG_SOURCE,python_files=249,helpers=dict(IMPLEMENTATION)),
                     inference_optimization=evidence)
        command=['--precision','fp32','--algengine-root','/alg','--expected-algengine-source-sha256',audit.ALG_SOURCE,
                 '--msda-backend','cuda-compatible','--inference-optimization','point-parallel-last-frame','--msda-query-chunk','8192']
        return runtime,command

    def test_complete_portable_reference_and_raw_inventory(self):
        reference=self.read();result=audit.compare_reference(reference,copy.deepcopy(self.inventory))
        self.assertTrue(result['all_full_dense_pickle_bytes_equal'])
        self.assertFalse(result['GPU_equivalence'])
        self.assertFalse(result['formal_end_to_end_performance'])

    def test_wrong_whole_reference_SHA_rejected(self):
        self.write()
        with self.assertRaises(ValueError): audit.read_reference(self.path,'0'*64,'vadv2','NR')

    def test_wrong_creator_rejected(self):
        self.reference['reference_creator_sha256']='0'*64
        with self.assertRaises(ValueError): self.read()

    def test_foreign_source_and_old_auditor_rejected(self):
        for key in ('reference_runtime','reference_auditor_sha256'):
            before=copy.deepcopy(self.reference[key]);self.reference[key]={} if isinstance(before,dict) else '0'*64
            with self.subTest(key=key),self.assertRaises(ValueError): self.read()
            self.reference[key]=before

    def test_wrong_model_reaction_scene_rejected(self):
        for key,value in (('model','hydramdp'),('reaction','R'),('original_scene_sha256','0'*64)):
            before=self.inventory[key];self.inventory[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.read()
            self.inventory[key]=before

    def test_missing_frame_image_control_dense_rejected(self):
        for key in ('frames','images','controls','dense_rows'):
            before=self.inventory[key];self.inventory[key]=before[:-1]
            with self.subTest(key=key),self.assertRaises(ValueError): self.read()
            self.inventory[key]=before

    def test_complete_persisted_differences_rejected(self):
        for key in ('frames','images','controls','dense_rows','PDM_CSV_sha256','PDM_metrics'):
            current=copy.deepcopy(self.inventory)
            if isinstance(current[key],list): current[key][0]['changed']='different persisted bytes'
            elif isinstance(current[key],dict): current[key]['score']=0
            else: current[key]='0'*64
            with self.subTest(key=key),self.assertRaises(ValueError): audit.compare_reference(self.reference,current)

    def test_complete_original_report_contract(self):
        audit.validate_reference_report(self.report(),'vadv2','NR')

    def test_old_unscored_or_foreign_source_report_rejected(self):
        for key,value in (('result','pass_configured_real_closed_loop_consumption_window'),('auditor_sha256','0'*64),
                          ('source',dict(sha256=audit.SIM_SOURCE,files=213))):
            report=self.report();report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): audit.validate_reference_report(report,'vadv2','NR')

    def test_report_missing_actual_controls_or_scoring_rejected(self):
        for key in ('consumption','scoring','acceptance'):
            report=self.report()
            if key=='consumption': report[key]['consumed_control_indices']=list(range(4,11))
            elif key=='scoring': report[key]['all_dense_arrays_finite']=False
            else: report[key]['original_PDM_and_dense_artifacts']=False
            with self.subTest(key=key),self.assertRaises(ValueError): audit.validate_reference_report(report,'vadv2','NR')

    def test_actual_generated_prediction_count_not_fixed_to_controls(self):
        for count in (8,9):
            runtime,command=self.runtime(count)
            result=audit.validate_current_planner_contract(runtime,command,'/alg',count)
            self.assertEqual(result['predictions'],count)
            self.assertEqual(result['preserved_NumPy_draws'],runtime['inference_optimization']['preserved_NumPy_draws'])
            self.assertNotIn('original_NumPy_draws',result)

    def test_optimization_fallback_or_missing_draws_rejected(self):
        for key,value in (('original_msda_fallback_calls',1),('preserved_NumPy_draws',48),('last_frame_calls',8)):
            runtime,command=self.runtime();runtime['inference_optimization'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError): audit.validate_current_planner_contract(runtime,command,'/alg',9)

    def test_wrong_actual_command_source_or_chunk_rejected(self):
        for option,value in (('--expected-algengine-source-sha256','0'*64),('--inference-optimization','none'),('--msda-query-chunk','2048')):
            runtime,command=self.runtime();command[command.index(option)+1]=value
            with self.subTest(key=option),self.assertRaises(ValueError): audit.validate_current_planner_contract(runtime,command,'/alg',9)

    def test_both_original_scorers_and_exact_profile_required(self):
        source=dict(sha256=audit.SIM_SOURCE,files=213)
        for key in ('pdm_metrics_CPU','dense_rewards_CPU'):
            wrapper=dict(pdm_metrics_CPU=True,dense_rewards_CPU=True);wrapper[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError): audit.validate_simulation_contract(wrapper,{},source,{},'key',Path('/run/job'),'NR')


if __name__=='__main__': unittest.main(verbosity=2)
