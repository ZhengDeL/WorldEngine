"""Offline builder contracts; no CANN compilation, network or installation."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_backend_under_test', HERE / 'build_backend.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.guard=m.Guard(datetime.fromtimestamp(time.time()+20,timezone.utc))

    def tearDown(self):self.temporary.cleanup()

    def operational_source(self):
        source=self.root/'source';source.mkdir()
        (source/'setup.py').write_text('a=multiprocessing.cpu_count()\nb=multiprocessing.cpu_count()\n')
        (source/'CMakePresets.json').write_text(json.dumps({'configurePresets':[{'cacheVariables':{
            'ASCEND_COMPUTE_UNIT':{'value':'ascend910b;ascend910_93'},'ENABLE_TEST':{'value':'True'}}}]}))
        path=source/'gauss_splat/csrc';path.mkdir(parents=True)
        (path/'CMakeLists.txt').write_text("import os; import site; print(site.getsitepackages()[0] + '/torch_npu')")
        return source

    def test_operational_patches_are_exact_and_record_before_after_hashes(self):
        source=self.operational_source();before={p.relative_to(source).as_posix():p.read_bytes() for p in source.rglob('*') if p.is_file()}
        rows=m.operational_patches(source,3)
        self.assertEqual(len(rows),3)
        self.assertEqual((source/'setup.py').read_text().count('min(3, multiprocessing.cpu_count())'),2)
        cache=json.loads((source/'CMakePresets.json').read_text())['configurePresets'][0]['cacheVariables']
        self.assertEqual(cache['ASCEND_COMPUTE_UNIT']['value'],'ascend910_93')
        self.assertEqual(cache['ENABLE_TEST']['value'],'False')
        cmake=(source/'gauss_splat/csrc/CMakeLists.txt').read_text()
        self.assertIn("importlib.util.find_spec('torch_npu').origin",cmake)
        self.assertNotIn('site.getsitepackages',cmake)
        for row in rows:
            self.assertEqual(row['before_sha256'],hashlib.sha256(before[row['path']]).hexdigest())
            self.assertEqual(row['after_sha256'],hashlib.sha256((source/row['path']).read_bytes()).hexdigest())

    def test_jobs_outside_one_to_sixteen_rejected(self):
        for jobs in (0,17):
            with self.assertRaises(ValueError):m.operational_patches(self.root,jobs)

    def test_unexpected_original_markers_rejected(self):
        source=self.operational_source();(source/'setup.py').write_text('unexpected source')
        with self.assertRaises(ValueError):m.operational_patches(source,8)

    def test_copy_only_manifest_files_without_source_mutation(self):
        source=self.root/'original';source.mkdir();(source/'keep').write_bytes(b'original');(source/'untracked').write_text('ignore')
        files=[{'path':'keep','size':8,'sha256':m.sha(b'original'),'mode':'100644'}]
        output=self.root/'copy';m.copy_tracked(source,output,files,self.guard)
        self.assertEqual((output/'keep').read_bytes(),b'original')
        self.assertFalse((output/'untracked').exists())
        self.assertEqual((source/'keep').read_bytes(),b'original')
        with self.assertRaises(FileExistsError):m.copy_tracked(source,output,files,self.guard)

    def test_source_mutation_detected_during_copy(self):
        source=self.root/'original';source.mkdir();(source/'keep').write_bytes(b'changed')
        files=[{'path':'keep','size':8,'sha256':m.sha(b'original'),'mode':'100644'}]
        with self.assertRaises(RuntimeError):m.copy_tracked(source,self.root/'copy',files,self.guard)

    @unittest.skipUnless(shutil.which('git'),'local git is required for this offline source fixture')
    def test_git_exact_commit_clean_tracked_and_untracked_exclusion(self):
        source=self.root/'git';source.mkdir()
        def git(*args):return subprocess.check_output(['git','-C',str(source),*args],stderr=subprocess.DEVNULL)
        git('init','--quiet');git('config','core.autocrlf','false')
        (source/'file').write_bytes(b'canonical\n');git('add','file')
        git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','offline fixture')
        commit=git('rev-parse','HEAD').decode().strip()
        (source/'ignored-untracked').write_text('not copied')
        inventory=m.source_inventory(source,self.guard,expected_commit=commit)
        self.assertEqual([row['path'] for row in inventory],['file'])
        with self.assertRaises(ValueError):m.source_inventory(source,self.guard,expected_commit='0'*40)
        (source/'file').write_text('changed tracked file')
        with self.assertRaises(ValueError):m.source_inventory(source,self.guard,expected_commit=commit)

    def test_child_environment_uses_selected_venv_and_offline_pip(self):
        python=self.root/'venv/bin/python'
        with mock.patch.dict(os.environ,{'PYTHONPATH':'unsafe-shadow','PYTHONHOME':'unsafe-home','PIP_TARGET':'unsafe-global','PIP_PREFIX':'unsafe-prefix','PIP_USER':'1'}):
            env=m.child_environment(python,8)
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('PYTHONHOME',env)
        self.assertEqual(env['PATH'].split(os.pathsep)[0],str(python.parent))
        self.assertEqual(env['TORCH_DEVICE_BACKEND_AUTOLOAD'],'0');self.assertEqual(env['PIP_NO_INDEX'],'1')
        self.assertNotIn('PIP_TARGET',env);self.assertNotIn('PIP_PREFIX',env);self.assertNotIn('PIP_USER',env)
        self.assertEqual(env['PIP_CONFIG_FILE'],os.devnull);self.assertEqual(env['PIP_REQUIRE_VIRTUALENV'],'1')
        self.assertEqual(env['BUILD_WITHOUT_SHA'],'1')
        self.assertEqual(env['MAX_JOBS'],'8');self.assertEqual(env['CMAKE_BUILD_PARALLEL_LEVEL'],'8')

    def test_owned_command_success_records_actual_command_and_exit(self):
        commands=[]
        m.run_owned([sys.executable,'-c','print("offline child")'],self.root,dict(os.environ),self.root/'run.log',self.guard,commands,'fixture')
        self.assertEqual(commands[0]['exit_code'],0)
        self.assertIn('offline child',(self.root/'run.log').read_text())

    def test_deadline_stops_owned_process_and_spares_unrelated_process(self):
        foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        commands=[];guard=m.Guard(datetime.fromtimestamp(time.time()+0.4,timezone.utc))
        try:
            with self.assertRaises(m.Paused):m.run_owned([sys.executable,'-c','import time;time.sleep(30)'],self.root,dict(os.environ),self.root/'deadline.log',guard,commands,'fixture_deadline')
            self.assertIsNotNone(commands[0]['exit_code']);self.assertIsNone(foreign.poll())
        finally:foreign.terminate();foreign.wait(timeout=5)

    def test_nonzero_command_is_recorded_and_rejected(self):
        commands=[]
        with self.assertRaises(RuntimeError):m.run_owned([sys.executable,'-c','raise SystemExit(7)'],self.root,dict(os.environ),self.root/'failure.log',self.guard,commands,'fixture_failure')
        self.assertEqual(commands[0]['exit_code'],7)

    def test_stop_file_prevents_command_start(self):
        stop=self.root/'STOP';stop.touch();guard=m.Guard(datetime.fromtimestamp(time.time()+10,timezone.utc),stop)
        commands=[]
        with self.assertRaises(m.Paused):m.run_owned([sys.executable,'-c','pass'],self.root,dict(os.environ),self.root/'never.log',guard,commands,'never')
        self.assertFalse(commands);self.assertFalse((self.root/'never.log').exists())

    def test_cann_python_whitelist_discards_project_relative_missing_and_parent_paths(self):
        root=self.root/'sdk';(root/'lib64').mkdir(parents=True)
        (root/'lib64/libascendcl.so').write_bytes(b'fixture library, not real CANN')
        sdk_python=root/'python/site-packages';sdk_python.mkdir(parents=True)
        project=self.root/'project';project.mkdir()
        inherited=os.pathsep.join([str(project),'.',str(root/'missing'),str(sdk_python)])
        paths,receipt=m.validated_cann_python_paths(root,inherited,excluded=(project,self.root/'output'))
        self.assertEqual(paths,[str(sdk_python.resolve())])
        self.assertFalse(receipt['vendor_signature_verified'])
        env=m.child_environment(self.root/'venv/bin/python',8,paths)
        self.assertEqual(env['PYTHONPATH'],str(sdk_python.resolve()))
        with self.assertRaises(ValueError):m.validated_cann_python_paths(root,'',explicit=(project,))
        with self.assertRaises(ValueError):m.validated_cann_python_paths(None,'',explicit=(sdk_python,))

    def test_cann_root_without_installation_marker_rejected(self):
        root=self.root/'not-sdk';root.mkdir()
        with self.assertRaises(ValueError):m.validated_cann_python_paths(root,'')

    @unittest.skipUnless(os.name=='posix' and hasattr(os,'WNOWAIT') and hasattr(os,'waitid'),'real WNOWAIT requires POSIX support')
    def test_linux_zero_exit_remains_zero_and_owned_descendant_group_is_terminated(self):
        pid_file=self.root/'descendant.pid'
        program="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(str(p.pid))"
        commands=[]
        m.run_owned([sys.executable,'-c',program,str(pid_file)],self.root,dict(os.environ),self.root/'descendant.log',self.guard,commands,'fixture_parent_zero')
        self.assertEqual(commands[0]['exit_code'],0)
        pid=int(pid_file.read_text())
        proc=Path('/proc')/str(pid)/'stat'
        until=time.monotonic()+2
        while proc.exists() and time.monotonic()<until:
            if proc.read_text().split(') ',1)[1].split()[0]=='Z':break
            time.sleep(0.02)
        if proc.exists():self.assertEqual(proc.read_text().split(') ',1)[1].split()[0],'Z')

    def test_full_dry_run_fixture_never_builds_installs_or_changes_original(self):
        source=self.operational_source()
        for name in m.COMPATIBILITY_FILES:
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture original')
        files=[{'path':p.relative_to(source).as_posix(),'size':p.stat().st_size,'sha256':m.sha(p.read_bytes()),'mode':'100644'}
               for p in source.rglob('*') if p.is_file()]
        before={row['path']:(source/row['path']).read_bytes() for row in files}
        patch_root=self.root/'fixture_patches';patch_root.mkdir()
        (patch_root/'calc_render_mtgs_compat.cpp').write_text('fixture replacement')
        fake='''import argparse,hashlib,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--apply',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()\nfiles=FILES\nbefore={n:hashlib.sha256((a.source/n).read_bytes()).hexdigest() for n in files}\nfor n in files:(a.source/n).write_text('fixture patched')\nafter={n:hashlib.sha256((a.source/n).read_bytes()).hexdigest() for n in files}\na.receipt.write_text(json.dumps({'canonical_commit':COMMIT,'contract':CONTRACT,'applied':True,'inference_only':True,'original_sha256':before,'patched_sha256':after}))\n'''
        fake=fake.replace('FILES',repr(sorted(m.COMPATIBILITY_FILES))).replace('COMMIT',repr(m.COMMIT)).replace('CONTRACT',repr(m.CONTRACT))
        (patch_root/'apply_compat_patch.py').write_text(fake)
        class Args:pass
        args=Args();args.source=source;args.output=self.root/'dry-build';args.python=Path(sys.executable)
        args.jobs=8;args.deadline_utc=datetime.fromtimestamp(time.time()+15,timezone.utc).isoformat()
        args.stop_file=None;args.install=False;args.dry_run=True
        with mock.patch.object(m,'PATCH_ROOT',patch_root),mock.patch.object(m,'source_inventory',return_value=files):
            self.assertEqual(m.build(args),0)
        receipt=json.loads((args.output/'build_receipt.json').read_text())
        self.assertEqual(receipt['result'],'dry_run_source_prepared_no_wheel_built')
        self.assertFalse(receipt['device_validation_passed'])
        self.assertEqual([row['phase'] for row in receipt['commands']],['apply_compatibility_patch'])
        self.assertEqual(list((args.output/'wheels').iterdir()),[])
        self.assertEqual(before,{row['path']:(source/row['path']).read_bytes() for row in files})


if __name__=='__main__':unittest.main()
