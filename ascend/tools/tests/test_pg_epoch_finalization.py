"""PG finalization guards plus installed MMCV/CPU AdamW replay; no NPU jobs.

The logical transport fixtures exercise the complete source after_run method.
The installed-runner tests execute toy CPU optimizer updates and compare their
complete state; actual distributed HydraPG acceptance remains a separate gate.
"""
import ast
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_epoch_resume as base


class PolicyGradientFinalizationLogic(unittest.TestCase):
    def prepared(self, root):
        return base.FinalEpochFinalization().prepared(root)[1]

    def test_saved_final_batch_PG_reports_no_new_PG_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            restored=self.prepared(Path(folder))
            row=base.actual_after_run(restored,expected_pg=True)
            self.assertEqual(row['result'],'pass_epoch_finalization_only_no_optimizer_updates')
            self.assertTrue(row['expected_pg']);self.assertFalse(row['pg_observed_nonzero'])
            self.assertFalse(row['pg_executed_this_run']);self.assertTrue(row['pg_zero_update_epoch_finalization'])
            self.assertEqual(row['updates_this_run'],0);self.assertFalse(row['parameter_updated'])

    def test_PG_finalization_requires_the_restored_PG_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            restored=self.prepared(Path(folder));base.actual_after_run(restored,expected_pg=True)
            for declaration in (False,None,1):
                restored.runner.resume_contract['expected_pg']=declaration
                with self.subTest(declaration=declaration),self.assertRaises(RuntimeError):
                    restored.runtime['epoch_finalization_only'](restored.runner,restored.runner.hook)

    def test_ordinary_PG_updates_still_require_observed_nonzero_PG(self):
        with tempfile.TemporaryDirectory() as folder:
            restored=self.prepared(Path(folder));base.actual_after_run(restored,expected_pg=True)
            runner=restored.runner;hook=runner.hook
            del runner._ascend_epoch_finalization
            hook.steps=1;hook.watch=type(hook.watch)(2.)
            with self.assertRaisesRegex(RuntimeError,'No nonzero real policy-gradient loss'):
                restored.runtime['after_run'](hook,runner)

    def test_PG_zero_update_cannot_hide_a_changed_parameter(self):
        with tempfile.TemporaryDirectory() as folder:
            restored=self.prepared(Path(folder))
            with self.assertRaisesRegex(RuntimeError,'Parameter changed during zero-update'):
                base.actual_after_run(restored,changed=True,expected_pg=True)


class InstalledMMCVPolicyGradientFinalization(base.InstalledMMCVWarmupResume):
    def finished(self):
        _,expected,_=self.run_cpu(original_loop=True)
        _,_,pending=self.run_cpu(pause=8)
        runner,actual,_=self.run_cpu(saved=pending)
        self.assert_raw_state(actual,expected)
        self.assertEqual((pending['completed_updates'],pending['completed_epochs']),(8,7))
        self.assertEqual((actual['updates'],actual['epochs']),(8,8))
        # The CPU fixture manually restored and compared the complete toy state.
        # NPU source/checkpoint restoration is tested by the separate NN gate.
        runner._ascend_restoration=dict(result='pass_exact_saved_state_restore')
        runner.resume_contract.update(precision=dict(mode='fp32'),expected_pg=True)
        hook=SimpleNamespace(steps=0,initial_iter=8,initial_epoch=7,expected_pg=True)
        gate=type(runner).train.__globals__['epoch_finalization_only']
        return runner,hook,gate

    def test_installed_MMCV_AdamW_full_CPU_state_is_unchanged_by_PG_finalization(self):
        runner,hook,gate=self.finished()
        before=copy.deepcopy((runner.model.state_dict(),runner.optimizer.state_dict(),self.rng()))
        self.assertTrue(gate(runner,hook))
        self.assert_raw_state((runner.model.state_dict(),runner.optimizer.state_dict(),self.rng()),before)

    def test_original_c468_source_rejects_the_same_finished_PG_state(self):
        runner,hook,gate=self.finished();self.assertTrue(gate(runner,hook))
        path=Path(__file__).parent/'fixtures/pg_epoch_finalization/runtime_before.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'c468cc632cbcbe4dea200e775451baac71ea67f5d203c6cf24a6d50d532d4a20')
        node=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='epoch_finalization_only')
        namespace=dict(EpochBasedRunner=self.Base)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        with self.assertRaisesRegex(RuntimeError,'Zero-update epoch finalization'):
            namespace['epoch_finalization_only'](runner,hook)

    def test_no_exact_restoration_cannot_accept_zero_update_PG(self):
        runner,hook,gate=self.finished();del runner._ascend_restoration
        with self.assertRaises(RuntimeError):gate(runner,hook)

    def test_nonfinal_epoch_resume_executes_the_next_real_optimizer_update(self):
        _,_,pending=self.run_cpu(pause=4)
        runner,actual,_=self.run_cpu(saved=pending,pause=5)
        self.assertEqual((actual['updates'],actual['epochs']),(5,4))
        runner.resume_contract.update(precision=dict(mode='fp32'),expected_pg=True)
        hook=SimpleNamespace(steps=1,initial_iter=4,initial_epoch=3,expected_pg=True)
        self.assertFalse(type(runner).train.__globals__['epoch_finalization_only'](runner,hook))


def load_tests(loader, _tests, _pattern):
    # Include only the new tests; the seven inherited real-MMCV cases are already
    # executed by test_epoch_resume.py in the complete qualification.
    result=loader.loadTestsFromTestCase(PolicyGradientFinalizationLogic)
    for name in sorted(InstalledMMCVPolicyGradientFinalization.__dict__):
        if name.startswith('test_'):result.addTest(InstalledMMCVPolicyGradientFinalization(name))
    return result


if __name__=='__main__':unittest.main()
