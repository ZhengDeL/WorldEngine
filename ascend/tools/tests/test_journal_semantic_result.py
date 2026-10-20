"""Actual-result CPU reproduction plus strict typed-value/integrity negatives.

The real sample is an immutable original official22 NPU result, not a new NPU
execution. All writes are to a TemporaryDirectory; original evidence is read-only.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
from rank_journal import RankJournal,semantic_digest,sha,RESULT_DIGEST_FORMAT

HERE=Path(__file__).resolve().parent
SAMPLE=Path(os.environ['JOURNAL_ACTUAL_SAMPLE_ROOT']).resolve(strict=True) if os.environ.get('JOURNAL_ACTUAL_SAMPLE_ROOT') else None
SAMPLE_SHA='7eeb89723c34deae06c04c63261404d4c80d91894bff6f4cadf8b1a2eeb51cde'


class Semantic(unittest.TestCase):
    def test_dictionary_order_and_alias_graph_are_not_value_changes(self):
        a=np.arange(12,dtype=np.float32).reshape(3,4)
        left=dict(a=a,b=[a,np.float32(1.5)],c=('x',True,None))
        right=dict(c=('x',True,None),b=[a.copy(),np.float32(1.5)],a=a.copy())
        self.assertEqual(semantic_digest(left),semantic_digest(right))
        self.assertEqual(semantic_digest(left),semantic_digest(pickle.loads(pickle.dumps(left,protocol=4))))

    def test_python_numpy_scalar_type_dtype_shape_and_signed_zero(self):
        for left,right in [(0.0,-0.0),(np.float32(0.),np.float32(-0.)),(1,True),(1,1.),(np.float32(1),np.float64(1)),
                           ([1,2],(1,2)),({'1':2},{1:2}),(b'a',bytearray(b'a')),
                           (np.zeros((2,3),dtype=np.float32),np.zeros((3,2),dtype=np.float32)),
                           (np.zeros((2,3),dtype=np.float32),np.zeros((2,3),dtype=np.float64)),
                           (np.array([1],dtype='<i4'),np.array([1],dtype='>i4'))]:
            with self.subTest(left=repr(left),right=repr(right)):self.assertNotEqual(semantic_digest(left),semantic_digest(right))

    def test_noncontiguous_arrays_keep_every_logical_c_order_byte(self):
        view=np.arange(24,dtype=np.float32).reshape(4,6)[:,::2]
        self.assertFalse(view.flags.c_contiguous)
        self.assertEqual(semantic_digest(view),semantic_digest(view.copy(order='C')))
        changed=view.copy();changed[0,0]=np.nextafter(changed[0,0],np.float32(1))
        self.assertNotEqual(semantic_digest(view),semantic_digest(changed))

    def test_structured_dtype_fields_and_metadata_are_not_dropped(self):
        a=np.zeros(2,dtype=[('x','f4'),('y','i4')]);b=np.zeros(2,dtype=[('other','f4'),('y','i4')])
        self.assertNotEqual(semantic_digest(a),semantic_digest(b))
        c=np.zeros(2,dtype=np.dtype('f4',metadata=dict(unit='m')))
        d=np.zeros(2,dtype=np.dtype('f4',metadata=dict(unit='km')))
        self.assertNotEqual(semantic_digest(c),semantic_digest(d))

    def test_unsupported_object_dtype_cycles_and_depth_fail_closed(self):
        cyclic=[];cyclic.append(cyclic)
        for value in (np.array([object()],dtype=object),object(),cyclic):
            with self.subTest(kind=type(value).__name__),self.assertRaises(RuntimeError):semantic_digest(value)
        deep=[];cursor=deep
        for _ in range(70):child=[];cursor.append(child);cursor=child
        with self.assertRaises(RuntimeError):semantic_digest(deep)


@unittest.skipUnless(SAMPLE is not None,'Set JOURNAL_ACTUAL_SAMPLE_ROOT to the approved SHA-locked real-result sample; it is not bundled')
class ActualResult(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sha(SAMPLE/'payload.pkl'),SAMPLE_SHA,'Original actual sample must be available without modification')
        self.sample_before={name:sha(SAMPLE/name) for name in ('payload.pkl','receipt.json','header.json')}
        self.actual=pickle.loads((SAMPLE/'payload.pkl').read_bytes())
        self.result=self.actual['result'];self.rng=self.actual['rng_after']
        receipt=json.loads((SAMPLE/'receipt.json').read_text());self.assertEqual(receipt['payload_sha256'],SAMPLE_SHA)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.tokens=[self.result['token']]+['cpu_fixture_%02d'%i for i in range(1,22)]
        self.contract=dict(world_size=16,source='actual-result-CPU-fixture',original_sample_sha256=SAMPLE_SHA)
    def tearDown(self):
        self.assertEqual({name:sha(SAMPLE/name) for name in self.sample_before},self.sample_before)
    def journal(self,mode='fresh'):
        return RankJournal(self.root,self.contract,self.tokens,16,0,mode)
    def create(self):
        with self.journal() as journal:
            journal.initial_rng(self.rng);journal.append(0,self.result,self.rng)
    def test_actual_numpy_dtype_alias_pickle_bytes_differ_values_exact(self):
        native=dict(self.result,trajectory=np.asarray(self.result['trajectory'].tolist(),dtype=self.result['trajectory'].dtype.str))
        self.assertEqual(self.result['trajectory'].tobytes(),native['trajectory'].tobytes())
        self.assertEqual(self.result['trajectory'].dtype,native['trajectory'].dtype)
        self.assertNotEqual(pickle.dumps(self.result,protocol=4),pickle.dumps(native,protocol=4))
        self.assertEqual(semantic_digest(self.result),semantic_digest(native))
        self.assertIs(type(self.result['ade_4s']),np.float32)
    def test_actual_result_fresh_and_cross_process_reopen_replay(self):
        self.create();path=self.root/'rank_00/steps/000000/payload.pkl';before=sha(path)
        source='''
import json,pickle,sys,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from rank_journal import RankJournal,semantic_digest
actual=pickle.loads(Path(sys.argv[3]).read_bytes())
result=dict(actual['result'],trajectory=np.asarray(actual['result']['trajectory'].tolist(),dtype=actual['result']['trajectory'].dtype.str))
tokens=json.loads(sys.argv[4]);contract=json.loads(sys.argv[5])
with RankJournal(Path(sys.argv[2]),contract,tokens,16,0,'resume') as journal:
    assert journal.committed==1
    outcome=journal.append(0,result,actual['rng_after'])
    assert outcome=='replayed_exactly_no_overwrite'
    print(json.dumps(dict(outcome=outcome,digest=semantic_digest(journal.read(0)['result']),numpy=np.__version__)))
'''
        result=subprocess.run([sys.executable,'-c',source,str(TOOLS),str(self.root),str(SAMPLE/'payload.pkl'),
            json.dumps(self.tokens),json.dumps(self.contract)],capture_output=True,text=True,timeout=30,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['digest'],semantic_digest(self.result));self.assertEqual(sha(path),before)
    def test_actual_scalar_and_rng_signedzero_replay_are_rejected(self):
        changed_rng=copy.deepcopy(self.rng);changed_rng['numpy']['cached_gaussian']=-0.0
        reference=copy.deepcopy(self.rng);reference['numpy']['cached_gaussian']=0.0
        with self.journal() as journal:
            journal.initial_rng(reference);journal.append(0,self.result,reference)
            changed=dict(self.result,ade_4s=np.float64(self.result['ade_4s']))
            with self.assertRaisesRegex(RuntimeError,'differs exactly'):journal.append(0,changed,reference)
            with self.assertRaisesRegex(RuntimeError,'differs exactly'):journal.append(0,self.result,changed_rng)
    def test_raw_payload_corruption_and_repaired_raw_hash_semantic_change_rejected(self):
        self.create();folder=self.root/'rank_00/steps/000000';payload=folder/'payload.pkl';receipt=folder/'receipt.json'
        old=payload.read_bytes();payload.write_bytes(old+b'changed')
        with self.assertRaisesRegex(RuntimeError,'hash chain'):self.journal('resume')
        value=pickle.loads(old);value['result']['trajectory'][0,0]=np.nextafter(value['result']['trajectory'][0,0],np.float32(np.inf))
        payload.write_bytes(pickle.dumps(value,protocol=4));record=json.loads(receipt.read_text());record['payload_sha256']=sha(payload);receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError,'typed values changed'):self.journal('resume')
    def test_old_header_or_receipt_schema_is_explicitly_rejected(self):
        self.create();path=self.root/'rank_00/header.json';old=path.read_bytes();header=json.loads(old);header['schema_version']=1;path.write_text(json.dumps(header))
        with self.assertRaisesRegex(RuntimeError,'Unsupported journal schema'):self.journal('resume')
        path.write_bytes(old);receipt=self.root/'rank_00/steps/000000/receipt.json';record=json.loads(receipt.read_text());self.assertEqual(record['result_digest_format'],RESULT_DIGEST_FORMAT)
        record['schema_version']=1;receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError,'Unsupported receipt schema'):self.journal('resume')


if __name__=='__main__':unittest.main()
