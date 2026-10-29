"""True Torch CPU contracts for explicit training Gv layout; no NPU claim."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
sys.path.insert(0,str(SOURCE))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
try:
    import torch
except ImportError:
    torch=None
from ascend import msda_value_gradient as helper


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def raw(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


class StaticTrainingLayoutContracts(unittest.TestCase):
    def test_options_reject_precision_backend_runtime_or_task_changes(self):
        helper.validate_training_option('none',precision='bf16',backend='reference',task='inference',runtime=False)
        helper.validate_training_option('head-major',precision='fp32',backend='cuda-compatible',runtime=True)
        for kw in [dict(precision='bf16'),dict(precision='fp32-hf32-conv'),dict(backend='reference'),dict(task='inference'),dict(runtime=False)]:
            options=dict(precision='fp32',backend='cuda-compatible',task='train',runtime=True);options.update(kw)
            with self.subTest(options=options),self.assertRaises(ValueError):helper.validate_training_option('head-major',**options)
        with self.assertRaises(ValueError):helper.validate_training_option('auto',precision='fp32',backend='cuda-compatible')

    def test_actual_launcher_rejects_nontraining_and_unsupported_precision_before_devices(self):
        common=[sys.executable,'-B',str(SOURCE/'ascend/run_container.py'),'--devices','0','--checkpoint','/generated_missing','--vocab','/generated_missing','--output','/generated_missing','--msda-value-grad-layout','head-major']
        for args,message in [(['--task','inference','--precision','fp32','--msda-backend','cuda-compatible'],'requires training'),(['--task','train','--precision','bf16','--msda-backend','cuda-compatible'],'requires FP32'),(['--task','train','--precision','fp32','--msda-backend','reference'],'requires FP32')]:
            with self.subTest(args=args):
                r=subprocess.run(common+args,capture_output=True,text=True,timeout=30)
                self.assertEqual(r.returncode,2,r.stderr);self.assertIn(message,r.stderr)

    def test_callback_is_the_only_MSDA_math_change_and_default_call_exact(self):
        original=ast.parse((HERE/'fixtures/msda_training/msda_cuda_compatible_before.py').read_bytes())
        candidate=ast.parse((SOURCE/'ascend/msda_cuda_compatible.py').read_bytes())
        fn=next(n for n in candidate.body if isinstance(n,ast.FunctionDef)and n.name=='make_function')
        self.assertEqual([x.arg for x in fn.args.args],['torch','grad_value_accumulator'])
        self.assertIsInstance(fn.args.defaults[-1],ast.Constant);self.assertIsNone(fn.args.defaults[-1].value)
        fn.args.args.pop();fn.args.defaults.pop()
        class StripCallback(ast.NodeTransformer):
            count=0
            def visit_If(self,node):
                if isinstance(node.test,ast.Compare)and isinstance(node.test.left,ast.Name)and node.test.left.id=='grad_value_accumulator':
                    self.count+=1;return node.body
                return self.generic_visit(node)
        strip=StripCallback();candidate=strip.visit(candidate);self.assertEqual(strip.count,1)
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(candidate,include_attributes=False))
        text=(SOURCE/'ascend/msda_value_gradient.py').read_text()
        self.assertNotIn('Tensor.index_add_ =',text)

    def test_training_flag_forwarded_explicitly_and_inference_child_does_not_inherit(self):
        tree=ast.parse((SOURCE/'ascend/run_container.py').read_text());lists=[n for n in ast.walk(tree)if isinstance(n,ast.List)]
        training=[n for n in lists if n.elts and isinstance(n.elts[0],ast.Constant)and n.elts[0].value=='ascend/train.py']
        inference=[n for n in lists if n.elts and isinstance(n.elts[0],ast.Constant)and n.elts[0].value=='ascend/inference.py']
        self.assertEqual(len(training),1);self.assertGreaterEqual(len(inference),1)
        self.assertIn('--msda-value-grad-layout',ast.unparse(training[0]))
        for n in inference:self.assertNotIn('--msda-value-grad-layout',ast.unparse(n))


@unittest.skipIf(torch is None,'Real Torch CPU is required on the target; zero skips mandatory there')
class TorchCPUTrainingLayoutContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.before_index_add=torch.Tensor.index_add_
        self.old_backend=os.environ.get('WORLDENGINE_MSDA_BACKEND')
        self.original=load('training_original_feb',HERE/'fixtures/msda_training/msda_cuda_compatible_before.py')
        self.candidate=load('training_candidate_callback',SOURCE/'ascend/msda_cuda_compatible.py')
    def tearDown(self):
        self.assertIs(torch.Tensor.index_add_,self.before_index_add)
        if self.old_backend is None:os.environ.pop('WORLDENGINE_MSDA_BACKEND',None)
        else:os.environ['WORLDENGINE_MSDA_BACKEND']=self.old_backend
    def arrays(self,legacy=False,noncontiguous=False):
        generator=torch.Generator().manual_seed(20261012)
        b,s,h,c=(2,12,8,32)if legacy else(2,13,2,32)
        q,l,p=(9,4,4)if legacy else(17,2,3)
        def make(shape):
            if noncontiguous:
                out=torch.rand((*shape[:-1],shape[-1]*2),generator=generator,dtype=torch.float32)[...,::2]
                self.assertFalse(out.is_contiguous());return out
            return torch.rand(shape,generator=generator,dtype=torch.float32)
        return make((b,s,h,c)),make((b,q,h,l,p,2)),make((b,q,h,l,p)),make((b,q,h*c)),((3,4),)if legacy else((3,4),(1,1))
    def execute(self,implementation,arrays,needs,callback=None,legacy=False):
        v,l,a,go,pairs=arrays
        class Context:
            needs_input_grad=tuple(needs)+(False,False,False)
            def save_for_backward(self,*saved):self.saved_tensors=saved
        ctx=Context();fn=implementation.make_function(torch,grad_value_accumulator=callback)if implementation is self.candidate else implementation.make_function(torch)
        with torch.no_grad():
            out=fn.forward(ctx,v,l,a,pairs,legacy,2048);grads=fn.backward(ctx,go)
        return out,grads
    def assert_outputs(self,left,right):
        self.assertTrue(raw(left[0],right[0]))
        for a,b in zip(left[1],right[1]):
            if a is None:self.assertIsNone(b)
            else:self.assertTrue(raw(a,b))
    def test_head_mapping_B1_B2_B8_nonzero_destination_collisions_all_channels(self):
        for batch,heads in [(1,1),(2,8),(8,8)]:
            spatial,queries,channels=11,7,32
            loc=(torch.arange(batch*queries*heads).reshape(batch,queries,heads)//heads)%3
            index=(torch.arange(batch).view(batch,1,1)*spatial*heads+loc*heads+torch.arange(heads).view(1,1,heads)).reshape(-1)
            source=((torch.arange(index.numel()*channels)%17)-8).reshape(-1,channels).float()/8
            target=((torch.arange(batch*spatial*heads*channels)%9)-4).reshape(-1,channels).float()/8
            expected=target.clone().index_add_(0,index,source);got=target.clone()
            returned=helper.head_major_index_add_(got,index,source,layout=(batch,spatial,heads,channels),tensor_api=torch)
            self.assertIs(returned,got);self.assertTrue(raw(expected,got))
    def test_helper_rejects_dtype_stride_and_geometry_without_implicit_conversion(self):
        target=torch.zeros(12,32);index=torch.zeros(4,dtype=torch.int64);source=torch.zeros(4,32)
        for d,i,s,layout in [(target,index.int(),source,(1,12,1,32)),(target.double(),index,source.double(),(1,12,1,32)),(target,index,source[:,::2],(1,12,1,32)),(target[:,::2],index,source,(1,12,1,32)),(target,index,source,(1,12,2,32))]:
            with self.assertRaises(ValueError):helper.head_major_index_add_(d,i,s,layout=layout,tensor_api=torch)
    def test_original_default_forward_and_all_gradients_raw_unchanged(self):
        arrays=self.arrays();self.assert_outputs(self.execute(self.original,arrays,(True,True,True)),self.execute(self.candidate,arrays,(True,True,True)))
    def test_callback_all_three_gradients_raw_unchanged(self):
        arrays=self.arrays();self.assert_outputs(self.execute(self.original,arrays,(True,True,True)),self.execute(self.candidate,arrays,(True,True,True),helper.make_accumulator(torch)))
    def test_callback_all_eight_needs_combinations_and_none_positions(self):
        arrays=self.arrays()
        for bits in range(8):
            needs=tuple(bool(bits&(1<<i))for i in range(3))
            with self.subTest(needs=needs):self.assert_outputs(self.execute(self.original,arrays,needs),self.execute(self.candidate,arrays,needs,helper.make_accumulator(torch)))
    def test_noncontiguous_original_inputs_and_grad_output(self):
        arrays=self.arrays(noncontiguous=True);self.assert_outputs(self.execute(self.original,arrays,(True,True,True)),self.execute(self.candidate,arrays,(True,True,True),helper.make_accumulator(torch)))
    def test_legacy_entire_prefix_and_unused_tail_all_three_gradients(self):
        arrays=self.arrays(legacy=True);got=self.execute(self.candidate,arrays,(True,True,True),helper.make_accumulator(torch),legacy=True)
        self.assert_outputs(self.execute(self.original,arrays,(True,True,True),legacy=True),got)
        for grad in got[1][1:3]:self.assertEqual(int(torch.count_nonzero(grad.reshape(-1)[grad.numel()//4:])),0)
    def test_bad_callback_return_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'preserve its destination'):
            self.execute(self.candidate,self.arrays(),(True,False,False),lambda dest,*args:dest.clone())
    def test_policy_cache_refuses_layout_or_backend_change_and_environment_cannot_enable(self):
        previous=os.environ.get('WORLDENGINE_MSDA_TRAINING_OPTIMIZATION');os.environ['WORLDENGINE_MSDA_TRAINING_OPTIMIZATION']='head-major'
        try:
            original_factory=load('original_none_factory_compatibility',SOURCE/'ascend/msda.py')
            os.environ['WORLDENGINE_MSDA_BACKEND']='reference'
            none_function=original_factory._cuda_function()
            original_factory.configure_backend('cuda-compatible')
            self.assertIs(original_factory._cuda_function(),none_function)
            self.assertEqual(original_factory.backend_policy()['training_value_gradient']['mode'],'none')
            m=load('fresh_training_msda_policy',SOURCE/'ascend/msda.py');m.configure_backend('cuda-compatible')
            self.assertEqual(m.backend_policy()['training_value_gradient']['mode'],'none')
            m.configure_backend('cuda-compatible',value_gradient_layout='head-major');fn=m._cuda_function();policy=m.backend_policy()
            self.assertEqual(policy['training_value_gradient']['mode'],'head-major');self.assertEqual(len(policy['training_value_gradient']['helper_source_sha256']),64)
            self.assertIs(m._cuda_function(),fn)
            for backend,layout in [('cuda-compatible','none'),('reference','none')]:
                with self.assertRaises(RuntimeError):m.configure_backend(backend,value_gradient_layout=layout)
            os.environ['WORLDENGINE_MSDA_BACKEND']='reference'
            with self.assertRaises(RuntimeError):m.backend_policy()
        finally:
            if previous is None:os.environ.pop('WORLDENGINE_MSDA_TRAINING_OPTIMIZATION',None)
            else:os.environ['WORLDENGINE_MSDA_TRAINING_OPTIMIZATION']=previous
    def test_runtime_exact_resume_refuses_layout_change_before_tensor_restore(self):
        tree=ast.parse((SOURCE/'ascend/runtime.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ResumeMixin')
        old={'source_sha256':'same_generated_source','precision':{'msda':{'training_value_gradient':{'mode':'none'}}}}
        saved={'optimizer':{},'meta':{'ascend_resume':{'contract':old}}};namespace={'torch':SimpleNamespace(load=lambda *a,**k:saved)}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'<actual runtime ResumeMixin>','exec'),namespace)
        instance=namespace['ResumeMixin']();instance.resume_contract=copy.deepcopy(old);instance.resume_contract['precision']['msda']['training_value_gradient']['mode']='head-major'
        with self.assertRaisesRegex(RuntimeError,'Resume contract changed'):instance.resume('/generated_not_read')

if __name__=='__main__':unittest.main()
