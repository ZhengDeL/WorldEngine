"""Test native-stream device scope without importing Torch or an accelerator."""
import ast
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import unittest


class FakeNPU:
    def __init__(self):self.current=0;self.entered=[]
    def device(self,target):
        state=self
        class Scope:
            def __enter__(self):
                self.previous=state.current;state.current=target.index;state.entered.append(target.index)
            def __exit__(self,*_):state.current=self.previous
        return Scope()


class DeviceGuardTests(unittest.TestCase):
    def setUp(self):
        self.npu=FakeNPU()
        tree=ast.parse((Path(__file__).resolve().parents[1]/'cann.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_with_npu_device')
        namespace={'torch':SimpleNamespace(npu=self.npu),'wraps':wraps}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_device_guard','exec'),namespace)
        self.guard=namespace['_with_npu_device']

    def tensor(self,index=1,kind='npu'):
        return SimpleNamespace(device=SimpleNamespace(type=kind,index=index))

    def test_current_zero_input_one_scopes_native_call_and_restores(self):
        observed=[]
        @self.guard
        def native(_,**kwargs):observed.append(self.npu.current);return 7
        self.assertEqual(native(None,means=self.tensor()),7)
        self.assertEqual(observed,[1]);self.assertEqual(self.npu.current,0)

    def test_exception_still_restores_original_device(self):
        @self.guard
        def native(_,**kwargs):raise RuntimeError('native fixture failure')
        with self.assertRaises(RuntimeError):native(None,means=self.tensor())
        self.assertEqual(self.npu.current,0)

    def test_cpu_input_rejected_before_native_scope(self):
        @self.guard
        def native(_,**kwargs):self.fail('CPU input reached native code')
        with self.assertRaises(ValueError):native(None,means=self.tensor(kind='cpu'))
        self.assertEqual(self.npu.entered,[])

    def test_missing_input_rejected(self):
        @self.guard
        def native(_,**kwargs):self.fail('Missing input reached native code')
        with self.assertRaises(ValueError):native(None)


if __name__=='__main__':unittest.main()
