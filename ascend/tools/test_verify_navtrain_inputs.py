"""Small synthetic CPU fixtures for the public inspector, not NAVTRAIN acceptance."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageFile


SOURCE = Path(__file__).with_name('verify_navtrain_inputs.py')
SPEC = importlib.util.spec_from_file_location('public_navtrain_input_inspector', SOURCE)
QC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QC)


def future():
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


class InputInspectorFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        self.addCleanup(setattr, ImageFile, 'LOAD_TRUNCATED_IMAGES', previous)
        self.guard = QC.Guard(future())

    def jpeg(self, name='camera.jpg', size=(1920, 1080), mode='RGB', color=None):
        path = self.root / name
        image = Image.new(mode, size, color or ('navy' if mode == 'RGB' else 50))
        image.save(path, format='JPEG', quality=85)
        image.close()
        return path

    def arguments(self, deadline=None):
        values = ['--source', str(self.root), '--data-root', str(self.root),
                  '--config', 'worldengine/e2e_vadv2_100pct.py',
                  '--output', str(self.root/'qc.json')]
        if deadline is not None:
            values += ['--deadline-utc', deadline]
        return values

    def test_valid_original_size_JPEG_is_hashed_verified_and_fully_loaded(self):
        path = self.jpeg()
        result = QC.decode_jpeg(path, 'log/CAM_F0/camera.jpg', self.guard)
        self.assertEqual(result['result'], 'pass')
        self.assertTrue(result['verified'])
        self.assertTrue(result['loaded'])
        self.assertEqual((result['width'], result['height'], result['mode']), (1920, 1080, 'RGB'))
        self.assertEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['bytes'], path.stat().st_size)

    def test_truncated_JPEG_header_is_not_sufficient_for_acceptance(self):
        path = self.jpeg()
        path.write_bytes(path.read_bytes()[:-64])
        result = QC.decode_jpeg(path, 'log/CAM_F0/camera.jpg', self.guard)
        self.assertEqual(result['result'], 'error')
        self.assertFalse(result['loaded'])

    def test_wrong_dimensions_are_rejected(self):
        result = QC.decode_jpeg(self.jpeg(size=(960, 540)), 'camera.jpg', self.guard)
        self.assertEqual(result['result'], 'error')
        self.assertIn('1920x1080', result['error'])
        self.assertFalse(result['loaded'])

    def test_grayscale_and_non_JPEG_are_rejected(self):
        result = QC.decode_jpeg(self.jpeg(mode='L'), 'camera.jpg', self.guard)
        self.assertEqual(result['result'], 'error')
        png = self.root/'disguised.jpg'
        Image.new('RGB', (1920, 1080)).save(png, format='PNG')
        result = QC.decode_jpeg(png, 'camera.jpg', self.guard)
        self.assertEqual(result['result'], 'error')
        self.assertIn('not a JPEG', result['error'])

    def test_content_hash_changes_even_if_size_and_mtime_are_preserved(self):
        path = self.root/'content.bin'
        path.write_bytes(b'A'*4096)
        stamp = path.stat()
        before = QC.digest_file(path, self.guard)
        path.write_bytes(b'B'*4096)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        after = QC.digest_file(path, self.guard)
        self.assertEqual(before['bytes'], after['bytes'])
        self.assertNotEqual(before['sha256'], after['sha256'])

    def test_valid_JPEG_replaced_after_hash_is_rejected_after_decode(self):
        path = self.jpeg()
        replacement = self.jpeg('replacement.jpg', color='yellow').read_bytes()
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        original_digest = QC.digest_file

        def replace_after_hash(filename, guard):
            result = original_digest(filename, guard)
            stamp = filename.stat()
            filename.write_bytes(replacement)
            os.utime(filename, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1_000_000_000))
            return result

        with mock.patch.object(QC, 'digest_file', side_effect=replace_after_hash):
            result = QC.decode_jpeg(path, 'camera.jpg', self.guard)
        self.assertTrue(result['loaded'])
        self.assertEqual(result['result'], 'error')
        self.assertIn('changed during hash', result['error'])
        self.assertEqual(result['sha256'], old_hash)
        self.assertNotEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_STOP_prevents_input_access(self):
        stop = self.root/'STOP'
        stop.touch()
        guard = QC.Guard(future(), stop)
        with self.assertRaises(QC.Paused):
            QC.decode_jpeg(self.root/'not_opened.jpg', 'camera.jpg', guard)

    def test_same_length_same_mtime_replacement_after_hash_is_rejected(self):
        path = self.jpeg()
        replacement = self.jpeg('replacement.jpg', color='yellow').read_bytes()
        original_bytes = path.read_bytes()
        size = max(len(original_bytes), len(replacement))
        original_bytes += b'\0' * (size-len(original_bytes))
        replacement += b'\0' * (size-len(replacement))
        path.write_bytes(original_bytes)
        old_hash = hashlib.sha256(original_bytes).hexdigest()
        original_digest = QC.digest_file
        changed = False

        def replace_once_after_hash(filename, guard):
            nonlocal changed
            result = original_digest(filename, guard)
            if not changed:
                stamp = filename.stat()
                filename.write_bytes(replacement)
                os.utime(filename, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                changed = True
            return result

        with mock.patch.object(QC, 'digest_file', side_effect=replace_once_after_hash):
            result = QC.decode_jpeg(path, 'camera.jpg', self.guard)
        self.assertTrue(result['verified'] and result['loaded'])
        self.assertEqual(result['result'], 'error')
        self.assertIn('content changed', result['error'])
        self.assertEqual(result['sha256'], old_hash)
        self.assertNotEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_deadline_is_required_UTC_and_has_no_historical_date_cap(self):
        for value in (None, '2099-01-01T00:00:00', '2099-01-01T00:00:00+08:00',
                      '2000-01-01T00:00:00+00:00', 'invalid'):
            with self.subTest(deadline=value), mock.patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    QC.parse_args(self.arguments(value))
                self.assertEqual(error.exception.code, 2)
        args = QC.parse_args(self.arguments('2099-01-01T00:00:00Z'))
        self.assertEqual(QC.parse_UTC_deadline(args.deadline_utc).utcoffset(), timedelta(0))

    def test_elapsed_deadline_and_signal_stop_propagate(self):
        guard = QC.Guard('2000-01-01T00:00:00+00:00')
        with self.assertRaises(QC.Paused):
            guard.check()
        self.guard.stop('SIGTERM')
        with self.assertRaises(QC.Paused):
            QC.digest_file(self.root/'not_opened.bin', self.guard)

    def test_NPU_CUDA_nodes_and_initialized_states_are_rejected(self):
        clean = dict(device_nodes=[], CUDA_initialized=False, NPU_initialized=False)
        QC.validate_CPU_only(clean)
        for changed in (dict(device_nodes=['/dev/davinci0']), dict(device_nodes=['/dev/nvidia0']),
                        dict(NPU_initialized=True), dict(CUDA_initialized=True)):
            state = dict(clean, **changed)
            with self.subTest(state=state), self.assertRaises(QC.Rejected):
                QC.validate_CPU_only(state)

    def test_NPU_device_discovery_rejects_without_loading_Torch(self):
        def devices(_path, pattern):
            return [self.root/'davinci7'] if pattern == 'davinci*' else []

        with mock.patch.object(QC.Path, 'glob', autospec=True, side_effect=devices), \
                mock.patch.object(QC.Path, 'exists', return_value=False), \
                mock.patch.dict(QC.sys.modules, {'torch': None, 'torch_npu': None}):
            state = QC.accelerator_state()
        self.assertEqual(len(state['device_nodes']), 1)
        self.assertIn('davinci7', state['device_nodes'][0])
        self.assertFalse(state['NPU_initialized'])
        with self.assertRaises(QC.Rejected):
            QC.validate_CPU_only(state)

    def test_missing_preflight_tuple_and_invalid_PDM_are_not_accepted(self):
        with self.assertRaises(QC.Rejected):
            QC.validate_inspection({'missing': 'annotations'}, [])
        phase = {key: 0 for key in ('missing_annotation_tokens', 'history_frames', 'camera_files',
                 'missing_camera_files', 'camera_bytes', 'missing_pdm_tokens')}
        phase.update(requested_anchors=1, selected_anchors=1, pdm_tokens=1, invalid_pdm_rows=1)
        with self.assertRaises(QC.Rejected):
            QC.validate_inspection((phase, {'token'}, {'log'}), [])

    def queue(self):
        names = ('CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0')
        return [dict(token='token%d'%i, log_token='original-log', log_name='official-log',
                     timestamp=1_600_000_000_000_000+i*500_000,
                     cams={name: {'data_path': 'official-log/%s/frame%d.jpg'%(name,i)}
                           for name in names}) for i in range(4)]

    def test_original_four_frame_eight_camera_reference_queue(self):
        references, counts = QC.referenced_images(self.queue(), {'token3'}, self.root, 4, self.guard)
        self.assertEqual(len(references), 32)
        self.assertEqual(counts, dict(selected_anchors=1, queue_frames=4, camera_reference_occurrences=32))

    def test_missing_camera_history_and_nonfinite_time_are_rejected(self):
        for kind in ('camera', 'history', 'time'):
            infos = self.queue()
            if kind == 'camera':
                del infos[0]['cams']['CAM_B0']
            elif kind == 'history':
                infos[0]['log_token'] = 'different-log'
            else:
                infos[1]['timestamp'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(QC.Rejected):
                QC.referenced_images(infos, {'token3'}, self.root, 4, self.guard)


if __name__ == '__main__':
    unittest.main()
