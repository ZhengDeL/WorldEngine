"""CPU fixtures for preparation mechanics, without official data or rendering."""
import copy
import hashlib
import importlib.util
import io
import json
import os
import pickle
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_scene_under_test', HERE / 'prepare_scene.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def expectation(path):
    return {'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def archive_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, body, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(body) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = '/outside'
            archive.addfile(info, io.BytesIO(body) if info.isfile() else None)
    return buffer.getvalue()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.guard = m.Guard()
        self.asset = 'actual-log-token'
        self.member = m.PREFIX + '/' + self.asset + '/background/' + self.asset + '.ckpt'

    def tearDown(self):
        self.doCleanups()  # Close verified descriptors before Windows removes fixture files.
        self.temporary.cleanup()

    def verified(self, path):
        source = m.VerifiedInput(path)
        source.verify(expectation(path), self.guard)
        self.addCleanup(source.stream.close)
        return source

    def write_archive(self, entries):
        path = self.root / 'assets.tar.gz'
        path.write_bytes(archive_bytes(entries))
        return self.verified(path)

    def test_official_lock_and_all_mutations_rejected(self):
        good, digest = m.load_official_lock(HERE / 'official_scene_assets.lock.json')
        self.assertEqual(good, m.EXPECTED_LOCK)
        for change in ('hash', 'schema_type', 'revision', 'extra'):
            bad = copy.deepcopy(good)
            if change == 'hash': bad['files']['assets']['sha256'] = '0' * 64
            if change == 'schema_type': bad['schema_version'] = True
            if change == 'revision': bad['modelscope_revision'] = 'master'
            if change == 'extra': bad['test_mode'] = True
            path = self.root / (change + '.json');path.write_text(json.dumps(bad), encoding='utf-8')
            with self.subTest(change=change), self.assertRaises(ValueError):m.load_official_lock(path)
        duplicate = self.root / 'duplicate.json';duplicate.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):m.load_official_lock(duplicate)

    def test_unverified_inputs_never_deserialized_or_extracted(self):
        path=self.root/'fake';path.write_bytes(b'not pickle or gzip')
        with m.VerifiedInput(path) as source:
            with mock.patch.object(m.pickle,'load',side_effect=AssertionError('unsafe load')) as load:
                with self.assertRaises(RuntimeError):m.select_scene(source,None,self.asset,self.guard)
                load.assert_not_called()
            with self.assertRaises(RuntimeError):m.extract_selected_asset(source,self.asset,self.root,self.guard)

    def test_wrong_whole_hash_does_not_mark_verified(self):
        path=self.root/'input';path.write_bytes(b'abc')
        with m.VerifiedInput(path) as source:
            with self.assertRaises(ValueError):source.verify({'size':3,'sha256':'0'*64},self.guard)
            self.assertFalse(source.verified)

    def test_original_normalization_uses_internal_id_and_exact_four_character_rule(self):
        self.assertEqual(m.normalized_asset_id({'id':'real-id-abc','metadata':{}}),'real-id')
        self.assertEqual(m.normalized_asset_id({'id':'ignored','metadata':{'digitaltwin_asset_id':'wanted-001'}}),'wanted')
        self.assertEqual(m.normalized_asset_id({'id':'intact-token','metadata':{}}),'intact-token')
        with self.assertRaises(ValueError):m.normalized_asset_id({'id':'../escape','metadata':{}})

    def test_real_mapping_preserves_entire_selected_value_and_extracts_only_its_asset(self):
        selected={'id':'internal-other','metadata':{'digitaltwin_asset_id':self.asset+'-001','digitaltwin_ego2globals':[[1,2,3]]},
                  'cameras':{'front':{'intrinsic':[[1,2],[3,4]],'history':[b'original']}},
                  'object_track':{'ego':{'state':[1,2,3]}},'map_features':{'lane':{'polyline':[(1,2)]}},'log_length':23}
        scenes={'unrelated':{'id':'another-id','metadata':{}},'original-key':selected}
        scenario_path=self.root/'all.pkl';scenario_path.write_bytes(pickle.dumps(scenes))
        scenario=self.verified(scenario_path)
        archive=self.write_archive([(self.member,b'actual fixture checkpoint bytes',tarfile.REGTYPE),
            (m.PREFIX+'/'+self.asset+'/road_height_map/map.npy',b'original ancillary bytes',tarfile.REGTYPE),
            (m.PREFIX+'/other/background/other.ckpt',b'never extracted',tarfile.REGTYPE)])
        lock=copy.deepcopy(m.EXPECTED_LOCK);lock['default_asset_id']=self.asset
        lock['files']['scenario'].update(expectation(scenario_path));lock['files']['assets'].update(expectation(archive.path))
        output=self.root/'prepared'
        receipt=m.prepare(scenario,archive,lock,'fixture-only-no-official-data',output,None,False,self.guard)
        self.assertEqual(receipt['result'],'pass_official_single_scene_asset_preparation')
        self.assertEqual(pickle.loads((output/'scene.pkl').read_bytes()),{'original-key':selected})
        self.assertEqual((output/'assets'/self.asset/'background'/(self.asset+'.ckpt')).read_bytes(),b'actual fixture checkpoint bytes')
        self.assertFalse((output/'assets'/'other').exists())
        self.assertEqual(len(receipt['asset_files']),2)
        self.assertFalse(receipt['renderer_execution_accepted'])
        self.assertEqual(receipt['checkpoint_inspection']['result'],'not_requested')

    def test_requested_scene_must_have_actual_checkpoint_in_package(self):
        source=self.write_archive([(self.member,b'checkpoint',tarfile.REGTYPE)])
        stage=self.root/'stage';stage.mkdir()
        with self.assertRaisesRegex(ValueError,'no matching'):m.extract_selected_asset(source,'different-asset',stage,self.guard)

    def test_unsafe_tar_paths_links_duplicates_and_conflicts_rejected(self):
        variants=[('../escape',b'',tarfile.REGTYPE),('/absolute',b'',tarfile.REGTYPE),
                  (m.PREFIX+'/bad/link',b'',tarfile.SYMTYPE),(m.PREFIX+'//empty',b'',tarfile.REGTYPE)]
        for index,entry in enumerate(variants):
            with self.subTest(entry=entry[0]):
                source=self.write_archive([entry]);stage=self.root/('unsafe'+str(index));stage.mkdir()
                with self.assertRaises(ValueError):m.extract_selected_asset(source,self.asset,stage,self.guard)
                source.stream.close()
        for index,entries in enumerate([
            [(self.member,b'a',tarfile.REGTYPE),(self.member,b'b',tarfile.REGTYPE)],
            [(m.PREFIX+'/'+self.asset+'/background',b'file',tarfile.REGTYPE),(self.member,b'child',tarfile.REGTYPE)],
            [(m.PREFIX+'/'+self.asset+'/Mixed/one',b'a',tarfile.REGTYPE),
             (m.PREFIX+'/'+self.asset+'/mixed/two',b'b',tarfile.REGTYPE)]
        ]):
            source=self.write_archive(entries);stage=self.root/('duplicate'+str(index));stage.mkdir()
            with self.assertRaises(ValueError):m.extract_selected_asset(source,self.asset,stage,self.guard)
            source.stream.close()

    def test_atomic_publication_never_overwrites_existing_directory(self):
        stage=self.root/'stage';stage.mkdir();(stage/'new').write_text('new')
        final=self.root/'final';final.mkdir();(final/'keep').write_text('keep')
        with self.assertRaises(FileExistsError):m.publish_directory_noreplace(stage,final)
        self.assertEqual((final/'keep').read_text(),'keep')
        self.assertTrue((stage/'new').exists())

    def test_input_change_after_hash_is_detected(self):
        path=self.root/'input';path.write_bytes(b'abc');source=self.verified(path)
        path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):source.unchanged()

    def test_failed_preparation_retains_stage_and_never_publishes_output(self):
        scene={'id':self.asset,'metadata':{}}
        path=self.root/'scene.pkl';path.write_bytes(pickle.dumps({'original':scene}))
        scenario=self.verified(path)
        archive=self.write_archive([(self.member,b'one',tarfile.REGTYPE),(self.member,b'two',tarfile.REGTYPE)])
        lock=copy.deepcopy(m.EXPECTED_LOCK);lock['default_asset_id']=self.asset
        output=self.root/'must-not-exist'
        with self.assertRaises(ValueError):m.prepare(scenario,archive,lock,'fixture-only',output,None,False,self.guard)
        self.assertFalse(output.exists())
        stages=list(self.root.glob('.must-not-exist.stage-*'))
        self.assertEqual(len(stages),1)
        receipt=json.loads((stages[0]/'prepare_error.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['result'],'blocked_or_rejected')
        self.assertFalse(receipt['renderer_execution_accepted'])

    def test_stop_and_expired_deadline_are_enforced(self):
        stop=self.root/'STOP';stop.touch()
        with self.assertRaises(m.Paused):m.Guard(stop_file=stop).check()
        with self.assertRaises(m.Paused):m.Guard(datetime(2000,1,1,tzinfo=timezone.utc)).check()


if __name__=='__main__':
    unittest.main()
