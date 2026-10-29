"""CPU artifact fixtures, not real-scene or NPU acceptance."""
import csv
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
from simengine_ascend.tools import audit_scored_default8 as audit


class ScoredDefault8Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.job = Path(self.tmp.name)/'job'
        self.folder = self.job/'WE_output/openscene_format'
        self.pkls = self.folder/'pdms_pkl'; self.pkls.mkdir(parents=True)
        self.recorded = Path('/original/job')
        self.scene = dict(id='fixture-scene', token='fixture-token', log_length=24,
                          metadata={'openscene_data_infos_dict': {i: {} for i in range(24)}},
                          object_track={'actor': {'state': {k: np.zeros((24, 2)) for k in ('position', 'velocity', 'heading')}}})
        self.pdm = self.folder/'all_scenes_pdm_averages_NR.csv'
        self.columns = ['token', 'step']+list(audit.METRICS)+['first_violation_step']
        self.rows = [dict(token=token, step='11', **{k: '1.00000' for k in audit.METRICS}, first_violation_step='')
                     for token in (self.scene['id'], 'overall_average')]
        self.write_pdm()
        self.values = {k: np.ones(8192, dtype='bool' if k in audit.METRICS[:2] else 'float64') for k in audit.DENSE}
        self.values.update(IL_plan_idx=1, target_traj=np.zeros((8, 2)))
        for step in range(3, 12): self.write_dense(step)
        self.index = self.folder/'all_scenes_pdm_pkl_paths_NR.csv'
        self.indices = [dict(token=self.scene['token'], step=str(i), pkl_path=str(self.recorded/'WE_output/openscene_format/pdms_pkl'/self.name(i))) for i in range(3, 12)]
        self.write_index()

    def name(self, step): return self.scene['id']+'_step_'+str(step)+'_scores.pkl'
    def write_pdm(self):
        with self.pdm.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, self.columns); writer.writeheader(); writer.writerows(self.rows)
    def write_index(self):
        with self.index.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, ['token', 'step', 'pkl_path']); writer.writeheader(); writer.writerows(self.indices)
    def write_dense(self, step):
        with (self.pkls/self.name(step)).open('wb') as stream: pickle.dump(self.values, stream)
    def check(self): return audit.audit_scoring(self.job, self.recorded, self.scene, 'NR')

    def test_complete_finite_original_layout(self):
        result = self.check()
        self.assertEqual(result['dense_positions'], list(range(3, 12)))
        self.assertEqual(len(result['dense_rows']), 9)
        self.assertEqual(result['PDM_times100'], 100)
        self.assertFalse(result['GPU_score_parity'])
        self.assertFalse(result['formulas_executed_by_this_audit'])

    def test_zero_score_is_preserved_as_a_quality_observation(self):
        for row in self.rows:
            row.update(no_at_fault_collisions='0.00000', score='0.00000', first_violation_step='11.00000')
        self.write_pdm(); result = self.check()
        self.assertEqual(result['PDM_times100'], 0)
        self.assertEqual(result['original_first_NC_DAC_violation_step'], 11)

    def test_wrong_reaction_is_rejected(self):
        with self.assertRaises(ValueError): audit.audit_scoring(self.job, self.recorded, self.scene, 'R')

    def test_nonfinite_PDM_is_rejected(self):
        self.rows[0]['score'] = 'nan'; self.write_pdm()
        with self.assertRaises(ValueError): self.check()

    def test_average_mismatch_is_rejected(self):
        self.rows[1]['score'] = '0.5'; self.write_pdm()
        with self.assertRaises(ValueError): self.check()

    def test_short_original_context_is_rejected(self):
        self.scene['log_length'] = 12
        with self.assertRaises(ValueError): self.check()

    def test_missing_actor_future_is_rejected(self):
        self.scene['object_track']['actor']['state']['position'] = np.zeros((19, 2))
        with self.assertRaises(ValueError): self.check()

    def test_missing_dense_position_is_rejected(self):
        (self.pkls/self.name(11)).unlink()
        with self.assertRaises(ValueError): self.check()

    def test_wrong_dense_shape_is_rejected(self):
        self.values['score'] = np.ones(8191); self.write_dense(3)
        with self.assertRaises(ValueError): self.check()

    def test_nonfinite_dense_array_is_rejected(self):
        self.values['score'][123] = np.inf; self.write_dense(3)
        with self.assertRaises(ValueError): self.check()

    def test_wrong_dense_dtype_is_rejected(self):
        self.values['score'] = np.ones(8192, dtype='float32'); self.write_dense(3)
        with self.assertRaises(ValueError): self.check()

    def test_bool_candidate_index_is_rejected(self):
        self.values['IL_plan_idx'] = True; self.write_dense(3)
        with self.assertRaises(ValueError): self.check()

    def test_short_target_is_rejected(self):
        self.values['target_traj'] = np.zeros((7, 2)); self.write_dense(3)
        with self.assertRaises(ValueError): self.check()

    def test_trailing_dense_pickle_is_rejected(self):
        with (self.pkls/self.name(3)).open('ab') as stream: stream.write(b'extra')
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_dense_CSV_position_is_rejected(self):
        self.indices[-1] = dict(self.indices[0]); self.write_index()
        with self.assertRaises(ValueError): self.check()

    def test_foreign_recorded_run_is_rejected(self):
        self.indices[-1]['pkl_path'] = '/different/job/dense.pkl'; self.write_index()
        with self.assertRaises(ValueError): self.check()

    def test_unscored_or_foreign_source_profile_is_rejected(self):
        exact = dict(sha256=audit.SIM_SOURCE, files=212)
        for source in (dict(exact, files=213), dict(exact, sha256='0'*64)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                audit.validate_simulation_contract({}, {}, source, {}, 'scene', self.recorded, 'NR')
        for name in ('pdm_metrics_CPU', 'dense_rewards_CPU'):
            wrapper = dict(pdm_metrics_CPU=True, dense_rewards_CPU=True); wrapper[name] = False
            with self.subTest(field=name), self.assertRaises(ValueError):
                audit.validate_simulation_contract(wrapper, {}, exact, {}, 'scene', self.recorded, 'NR')
        for name in ('with_metric_manager', 'with_dense_reward_manager'):
            cfg = dict(with_metric_manager=True, with_dense_reward_manager=True); cfg[name] = False
            with self.subTest(field=name), self.assertRaises(ValueError):
                audit.validate_simulation_contract(dict(pdm_metrics_CPU=True, dense_rewards_CPU=True), cfg, exact, {}, 'scene', self.recorded, 'NR')


if __name__ == '__main__': unittest.main(verbosity=2)
