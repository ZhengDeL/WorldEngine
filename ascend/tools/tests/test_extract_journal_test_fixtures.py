"""CPU tests for static fixture export; no upstream source copies/imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
import extract_journal_test_fixtures as export


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'mmdetection/mmdet/apis/test.py';self.source.parent.mkdir(parents=True)
        self.functions={'multi_gpu_test':b'def multi_gpu_test():\n    return None\n',
                        'collect_results_cpu':b'def collect_results_cpu():\n    return None\n'}
        # Executing/importing either source/version file would fail. Static
        # extraction must inspect their bytes and AST without executing them.
        self.source.write_bytes(b'import torch\nraise RuntimeError("MUST NOT EXECUTE INSTALLED SOURCE")\n'+
                                b'\n'.join(self.functions.values()))
        self.version=self.root/'mmdetection/mmdet/version.py';self.version.write_bytes(b'import torch_npu\n__version__ = "fixture-only"\nraise RuntimeError("MUST NOT EXECUTE VERSION")\n')
        self.license=self.root/'mmdetection/LICENSE';self.license.write_text('Synthetic fixture license marker: Apache License\nVersion 2.0\n')
        self.rows={name:dict(function_source=raw,provenance={'installed_source_file':str(self.source),
            'function_source_sha256':export.digest(raw),'extraction':'synthetic-static-fixture'}) for name,raw in self.functions.items()}
    def tearDown(self):self.tmp.cleanup()
    def write(self,output,notice=None,license=None):
        return export.write_fixtures(output,self.rows,license or self.license,notice,self.version,export.digest(self.version.read_bytes()))
    def test_export_exact_license_notice_version_and_no_overwrite(self):
        notice=self.root/'mmdetection/NOTICE';notice.write_text('Synthetic local notice')
        output=self.root/'export';result=self.write(output,notice)
        self.assertEqual((output/'MMDETECTION_LICENSE').read_bytes(),self.license.read_bytes())
        self.assertEqual((output/'MMDETECTION_NOTICE').read_bytes(),notice.read_bytes())
        self.assertEqual((output/'MMDETECTION_VERSION.py.txt').read_bytes(),self.version.read_bytes())
        receipt=json.loads((output/'provenance.json').read_text())
        for name,item in receipt['files'].items():self.assertEqual(export.digest((output/name).read_bytes()),item['sha256'])
        self.assertFalse(receipt['dynamic_API_import_executed']);self.assertFalse(receipt['torch_or_NPU_import_executed'])
        with self.assertRaises(FileExistsError):self.write(output,notice)
        self.assertEqual(receipt,result)
    def test_foreign_license_and_wrong_source_rejected(self):
        other=self.root/'LICENSE';other.write_text('Apache License\nVersion 2.0')
        with self.assertRaisesRegex(RuntimeError,'same installed'):self.write(self.root/'foreign',license=other)
        with self.assertRaisesRegex(RuntimeError,'function source SHA differs'):export.extract_functions(self.source,export.digest(self.source.read_bytes()))
        with self.assertRaisesRegex(RuntimeError,'whole-file SHA differs'):export.extract_functions(self.source,'0'*64)
        self.assertFalse((self.root/'foreign').exists())
    def test_invalid_license_or_nonliteral_version_leaves_output_absent(self):
        self.version.write_bytes(b'__version__ = forbidden_call()\n')
        with self.assertRaisesRegex(RuntimeError,'never execute'):self.write(self.root/'version_bad')
        self.license.write_text('wrong license')
        with self.assertRaisesRegex(RuntimeError,'Apache-2.0'):self.write(self.root/'bad')
        self.assertFalse((self.root/'bad').exists());self.assertFalse((self.root/'version_bad').exists())
    def test_AST_extracts_exact_source_without_import_or_execution(self):
        expected={name:export.digest(raw) for name,raw in self.functions.items()}
        with mock.patch.object(export,'EXPECTED',expected):rows=export.extract_functions(self.source,export.digest(self.source.read_bytes()))
        for name in expected:self.assertEqual(rows[name]['function_source'],self.functions[name])
        version,raw=export.static_version(self.version,export.digest(self.version.read_bytes()))
        self.assertEqual(version,'fixture-only');self.assertEqual(raw,self.version.read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
