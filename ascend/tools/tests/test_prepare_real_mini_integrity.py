"""Temporary synthetic CPU inputs; no published dataset or model is used."""
import argparse
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml


class CachedDecodeIntegrity(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'prepare_real_mini.py'
        spec = importlib.util.spec_from_file_location('mini_integrity', path)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.temp = tempfile.TemporaryDirectory(prefix='mini-integrity-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        cameras = self.root / 'cameras'
        infos, pdm = [], {}
        channels = ('CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0')
        for log in ('logA', 'logB', 'logC'):
            for index in range(5):
                cams = {}
                for channel in channels:
                    relative = '%s/%s/%d.jpg' % (log, channel, index)
                    image = cameras / relative
                    image.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (2, 2)).save(image, format='JPEG')
                    cams[channel] = {'data_path': relative}
                token = log + str(index)
                infos.append(dict(token=token, log_name=log, log_token=log, timestamp=index, cams=cams))
                if index >= 3:
                    pdm[token] = {metric: np.zeros(8192, dtype=np.float32) for metric in self.helper.METRICS}
        annotation, rewards, rare = (self.root / name for name in ('annotations.pkl', 'pdm.pkl', 'rare.yaml'))
        annotation.write_bytes(pickle.dumps({'infos': infos}, protocol=4))
        rewards.write_bytes(pickle.dumps(pdm, protocol=4))
        rare.write_text(yaml.safe_dump({'tokens': ['logA3'], 'log_names': ['logA']}))
        self.args = argparse.Namespace(output=self.root / 'prepared', logs=3, anchors_per_log=2,
                                       annotations=annotation, pdm=rewards, rare_split=rare,
                                       cameras=cameras, camera_mode='copy')
        self.changed = cameras / 'logA/CAM_F0/0.jpg'

    def test_unchanged_decoded_bytes_are_copied_exactly(self):
        expected = self.helper.sha256(self.changed)
        report = self.helper.prepare(self.args)
        self.assertTrue(report['jpeg_decode_verified'])
        target = self.args.output / 'cameras/logA/CAM_F0/0.jpg'
        self.assertEqual(self.helper.sha256(target), expected)
        with Image.open(target) as image:
            image.load()

    def mutate_on_hash(self, occurrence):
        original = self.helper.sha256
        calls = []

        def altered(path):
            if Path(path) == self.changed:
                calls.append(True)
                if len(calls) == occurrence:
                    self.changed.write_bytes(b'Invalid JPEG in a negative temporary fixture\n')
            return original(path)

        return altered

    def test_change_between_decode_and_copy_rejects_without_manifest(self):
        # The third hash is the copy-stage check, after pre/post decode hashes.
        with patch.object(self.helper, 'sha256', self.mutate_on_hash(3)):
            with self.assertRaisesRegex(ValueError, 'changed after cached decode'):
                self.helper.prepare(self.args)
        self.assertFalse((self.args.output / 'manifest.json').exists())

    def test_change_during_decode_is_rejected(self):
        with patch.object(self.helper, 'sha256', self.mutate_on_hash(2)):
            with self.assertRaises(ValueError):
                self.helper.prepare(self.args)
        self.assertFalse((self.args.output / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
