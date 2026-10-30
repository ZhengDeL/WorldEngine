"""Source-bound epoch candidate contracts; CPU/static, not device acceptance."""
import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import optimized_journal_inference as optimized
import journal_inference as legacy
import verify_navtest_inputs as qc
EXPECTED='a0d8820b69d5bce3d61a8a6cc6b8135c1c3a4ffbd8623e3cd10b14784d989a81'
class EpochSourceContractTests(unittest.TestCase):
 def test_old_checkpoint_source_contracts_refused_before_tensor_restore(self):
  tree=ast.parse((SOURCE/'ascend/runtime.py').read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ResumeMixin')
  saved={'optimizer':{},'meta':{'ascend_resume':{'contract':{}}}}
  namespace={'torch':SimpleNamespace(load=lambda *a,**kw:saved)}
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'<exact ResumeMixin AST>','exec'),namespace)
  runner=namespace['ResumeMixin']();runner.resume_contract={'source_sha256':EXPECTED}
  for source in ['216499c4268afe5e6ed32564765616b636bde1723b38c47b924ebae8ca9a0e5c','4ef8c8533134646254f0821cc8736051e91c7e8c54ea0398682d3990cd4acccb','7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb','9da112331978fef0fc666d463194cc21e80bf2a0c42798ee246fcba62cf3830f', 'e0487cb0130db6f99491bba76120b75ae83b1fd8f3044025e615df01fb7e346e', '87560d390f1addcc6910e98b5344133d1499cb2cf31361b3ff89d9cbbba3906d']:
   with self.subTest(source=source):
    saved['meta']['ascend_resume']['contract']={'source_sha256':source}
    with self.assertRaisesRegex(RuntimeError,'Resume contract changed'):runner.resume('/generated_checkpoint_not_read')
 def test_new_optimized_and_QC_routes_bound_legacy_route_preserved(self):
  self.assertEqual(legacy.fingerprint(SOURCE),EXPECTED)
  self.assertEqual(optimized.OPTIMIZED_SOURCE_SHA,EXPECTED)
  self.assertEqual(qc.RUNTIME_SHA,EXPECTED)
  self.assertEqual(legacy.SOURCE_SHA,'76c51f77d6d0c17f33652ea307a654bee5057c76ccd6f122b34e1fab66a1d3c5')
  for mode in optimized.MODES:
   self.assertEqual(optimized.optimization_contract(EXPECTED,EXPECTED,mode,2048,'cuda-compatible')['mode'],mode)
  with self.assertRaisesRegex(RuntimeError,'Unsupported pinned runtime source'):optimized.optimization_contract('7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb',EXPECTED,'none',2048,'cuda-compatible')
  with self.assertRaisesRegex(RuntimeError,'fingerprint differs'):optimized.optimization_contract(legacy.SOURCE_SHA,EXPECTED,'none',2048,'cuda-compatible')
 def test_exact_epoch_overlay_and_locked_original_MSDA_reference(self):
  pins={'ascend/runtime.py':'83b33ee9b546fa06637d4b8d29ccd17f6367d78b7b3465dba635e2f9c16154ee','ascend/epoch_cursor.py':'f83615141f95468838064a27a1b18890843d8c220116871c52d4a4986e198916','ascend/tools/tests/fixtures/msda_training/msda_cuda_compatible_before.py':'feb35fc82b67dec1fbf8d24c0f5f60d6ff30356e8f419cf37e90d92481e854d9','ascend/train.py':'296c09bf33d227e7b2121ef8622396c88dc773c37d0498312cce9fc4b8d68a9a'}
  for name,pin in pins.items():self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),pin,name)
if __name__=='__main__':unittest.main()
