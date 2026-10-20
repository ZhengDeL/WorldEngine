"""Atomic original-result/RNG journal; CPU-testable, with no model arithmetic.

Each committed step is a nonempty atomically renamed directory. Uncommitted
attempt directories are retained and never counted or removed on resume.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import stat
import struct
import uuid

import numpy as np

JOURNAL_SCHEMA=2
RESULT_DIGEST_FORMAT='typed-result-v1'


def need(value,message):
    if not value: raise RuntimeError(message)


def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def semantic_digest(value):
    """Hash exact typed values, independent of pickle memo/dtype alias graphs.

    Arrays retain dtype descriptors/metadata, shape and every C-order byte.
    Python and NumPy scalars retain type and bits, including signed zero.
    Dict key/value types are included; dict insertion order and object sharing
    are not numerical result values. Lists/tuples retain type and order.
    Unsupported objects, object dtypes, cycles and excessive input fail closed.
    """
    active=set();nodes=[0]
    def frame(tag,raw):return tag+struct.pack('>Q',len(raw))+raw
    def dtype_bytes(dtype,depth):
        need(not dtype.hasobject,'Object dtype has no stable numerical raw-byte contract')
        return encode((dtype.str,dtype.descr,dtype.itemsize,bool(dtype.isalignedstruct),
                       dict(dtype.metadata) if dtype.metadata is not None else None),depth+1)
    def encode(item,depth=0):
        nodes[0]+=1;need(depth<=64 and nodes[0]<=100000,'Typed result exceeds bounded depth/node count')
        kind=type(item)
        if item is None:return frame(b'N',b'')
        if kind is bool:return frame(b'B',b'1' if item else b'0')
        if kind is int:return frame(b'I',str(item).encode('ascii'))
        if kind is float:return frame(b'F',struct.pack('>d',item))
        if kind is str:return frame(b'S',item.encode('utf8','surrogatepass'))
        if kind is bytes:return frame(b'Y',item)
        if kind is bytearray:return frame(b'Z',bytes(item))
        if kind is np.ndarray:
            raw=item.tobytes(order='C');need(len(raw)<=4*1024**2,'Unbounded result array')
            return frame(b'A',frame(b'D',dtype_bytes(item.dtype,depth))+encode(tuple(item.shape),depth+1)+frame(b'R',raw))
        if isinstance(item,np.generic):
            name=kind.__module__+'.'+kind.__qualname__
            return frame(b'G',encode(name,depth+1)+frame(b'D',dtype_bytes(item.dtype,depth))+frame(b'R',item.tobytes()))
        need(kind in (dict,list,tuple),'Unsupported exact result value type: '+kind.__module__+'.'+kind.__qualname__)
        marker=id(item);need(marker not in active,'Cyclic result values are unsupported');active.add(marker)
        try:
            if kind is dict:
                pairs=[(encode(key,depth+1),encode(child,depth+1)) for key,child in item.items()]
                pairs.sort(key=lambda pair:pair[0]);raw=b''.join(frame(b'K',key)+frame(b'V',child) for key,child in pairs)
                return frame(b'M',raw)
            return frame(b'L' if kind is list else b'T',b''.join(encode(child,depth+1) for child in item))
        finally:active.remove(marker)
    raw=encode(value);need(len(raw)<=8*1024**2,'Unbounded typed result encoding')
    return hashlib.sha256(RESULT_DIGEST_FORMAT.encode('ascii')+b'\0'+raw).hexdigest()


def sha(path):
    path=Path(path);before=path.lstat();need(stat.S_ISREG(before.st_mode),'Regular journal file required')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    after=path.lstat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Journal file changed')
    return digest.hexdigest()


def sync_directory(path):
    if os.name=='posix':
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def write_file(path,raw):
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def rank_indices(size,world,rank):
    need(type(size) is int and size>0 and type(world) is int and world>0 and type(rank) is int and 0<=rank<world,'Invalid sampler dimensions')
    count=math.ceil(size/world);total=count*world
    return [(rank+step*world)%size for step in range(count)]


def validate_result(row,token):
    need(isinstance(row,dict) and row.get('token')==token,'Original result token differs from exact rank sampler position')
    trajectory=row.get('trajectory')
    need(isinstance(trajectory,np.ndarray) and trajectory.shape==(40,3) and trajectory.dtype.kind=='f'
         and np.isfinite(trajectory).all(),'Original result must contain finite floating[40,3] trajectory')
    chosen=row.get('chosen_ind')
    need(type(chosen) is int and 0<=chosen<8192,'Original selected vocabulary index invalid')
    keys=('ade_4s','fde_4s','no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score')
    need(all(key in row and isinstance(row[key],(int,float,np.number)) and not isinstance(row[key],(bool,np.bool_))
             and math.isfinite(float(row[key])) for key in keys),'Original result metrics missing/nonfinite')


def rng_pack(state):
    """Normalize only RNG state bytes; no activation/model values are copied."""
    result=dict(state)
    if isinstance(result.get('numpy'),tuple):
        algorithm,keys,position,has_gauss,cached=result['numpy']
        result['numpy']=dict(generator=algorithm,keys=np.asarray(keys,dtype=np.uint32).tolist(),
                             position=int(position),has_gauss=int(has_gauss),cached_gaussian=float(cached))
    for key in ('cpu','npu'):
        tensor=state[key]
        need(str(tensor.dtype)=='torch.uint8' and len(tensor.shape)==1,'Expected original byte RNG state')
        result[key]=bytes(tensor.cpu().tolist())
    need(set(result)=={'python','numpy','cpu','npu'},'Original four RNG sources required')
    return result


def validate_rng(value):
    need(isinstance(value,dict) and set(value)=={'python','numpy','cpu','npu'},'Four original RNG sources required')
    need(all(isinstance(value[key],bytes) and value[key] for key in ('cpu','npu')),'Nonempty byte RNG states required')
    numpy=value['numpy']
    need(isinstance(numpy,dict) and set(numpy)=={'generator','keys','position','has_gauss','cached_gaussian'}
         and isinstance(numpy['keys'],list) and all(type(k) is int and 0<=k<2**32 for k in numpy['keys']),
         'Normalized original NumPy RNG state required')


def rng_unpack(state,torch):
    result=dict(state)
    for key in ('cpu','npu'):result[key]=torch.tensor(list(state[key]),dtype=torch.uint8,device='cpu')
    return result


class RankJournal:
    def __init__(self,root,contract,tokens,world,rank,mode):
        self.root=Path(root);self.world=world;self.rank=rank;self.tokens=list(tokens)
        need(self.root.is_absolute() and self.root.resolve()==self.root,'Journal root must be absolute/canonical before any write')
        need(mode in ('fresh','resume') and len(set(tokens))==len(tokens),'Journal mode/unique dataset tokens invalid')
        self.indices=rank_indices(len(tokens),world,rank);self.contract=dict(contract)
        need(contract.get('world_size')==world,'Contract world differs')
        self.contract_sha=hashlib.sha256(canonical_json(contract)).hexdigest()
        self.folder=self.root/('rank_%02d'%rank)
        if mode=='fresh':self.folder.mkdir(parents=True,exist_ok=False)
        else:need(self.folder.is_dir() and not self.folder.is_symlink(),'Missing canonical prior rank journal')
        need(self.folder.resolve(strict=True)==self.folder,'Journal path must be canonical')
        lock=self.folder/'writer.lock'
        if lock.exists() or lock.is_symlink():need(stat.S_ISREG(lock.lstat().st_mode),'Regular nonsymlink writer lock required')
        self.lock=lock.open('a+b')
        if os.name=='posix':
            import fcntl
            try:fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:self.lock.close();raise RuntimeError('Another writer owns this rank journal') from exc
        else:
            import msvcrt
            try:
                self.lock.seek(0)
                if self.lock.read(1)==b'':self.lock.write(b'0');self.lock.flush()
                self.lock.seek(0);msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:self.lock.close();raise RuntimeError('Another writer owns this rank journal') from exc
        self.steps=self.folder/'steps';self.staging=self.folder/'uncommitted'
        header=dict(schema_version=JOURNAL_SCHEMA,contract=contract,contract_sha256=self.contract_sha,rank=rank,world_size=world,
                    dataset_size=len(tokens),dataset_order_sha256=hashlib.sha256(canonical_json(tokens)).hexdigest(),
                    sampler_indices=self.indices,sampler_tokens=[tokens[index] for index in self.indices])
        header_path=self.folder/'header.json'
        try:
            if mode=='fresh':
                self.steps.mkdir();self.staging.mkdir();write_file(header_path,canonical_json(header));sync_directory(self.folder)
            else:
                need(all(path.is_dir() and not path.is_symlink() for path in (self.steps,self.staging)),
                     'Canonical owned steps/uncommitted directories required')
                need(stat.S_ISREG(header_path.lstat().st_mode),'Regular journal header required')
                stored=json.loads(header_path.read_text())
                need(stored.get('schema_version')==JOURNAL_SCHEMA,'Unsupported journal schema; v1 is preserved and requires a fresh v2 journal')
                need(stored==header,'Source/checkpoint/config/image/precision/rank/token contract changed')
            directories=sorted(self.steps.iterdir(),key=lambda path:path.name)
            need(all(path.is_dir() and not path.is_symlink() and re.fullmatch('[0-9]{6}',path.name) for path in directories),'Unexpected committed journal member')
            need([path.name for path in directories]==['%06d'%n for n in range(len(directories))],'Committed journal has a hole')
            need(len(directories)<=len(self.indices),'Too many committed steps')
            self.committed=len(directories);self._receipt_hashes=[]
            for step in range(self.committed):self.read(step,verify_chain=True)
        except BaseException:self.close();raise

    def close(self):
        if getattr(self,'lock',None) is not None:self.lock.close();self.lock=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def read(self,step,verify_chain=False):
        need(0<=step<self.committed,'Uncommitted journal step requested')
        folder=self.steps/('%06d'%step);receipt_path=folder/'receipt.json';payload_path=folder/'payload.pkl'
        need(set(path.name for path in folder.iterdir())=={'receipt.json','payload.pkl'},'Incomplete atomic journal commit')
        receipt=json.loads(receipt_path.read_text());need(payload_path.stat().st_size<=4*1024**2,'Unbounded journal step')
        need(receipt.get('schema_version')==JOURNAL_SCHEMA and receipt.get('result_digest_format')==RESULT_DIGEST_FORMAT,
             'Unsupported receipt schema/digest; v1 journals are not adopted')
        previous=self._receipt_hashes[step-1] if verify_chain and step else (sha(self.steps/('%06d'%(step-1))/'receipt.json') if step else None)
        need(receipt.get('step')==step and receipt.get('rank')==self.rank and receipt.get('contract_sha256')==self.contract_sha
             and receipt.get('dataset_index')==self.indices[step] and receipt.get('token')==self.tokens[self.indices[step]]
             and receipt.get('previous_receipt_sha256')==previous and sha(payload_path)==receipt.get('payload_sha256'),'Journal identity/hash chain differs')
        with payload_path.open('rb') as stream:
            value=pickle.load(stream);need(not stream.read(1),'Extra journal payload bytes')
        validate_result(value['result'],receipt['token'])
        validate_rng(value['rng_after'])
        need(semantic_digest(value['result'])==receipt.get('original_result_semantic_sha256'),'Original result typed values changed')
        if verify_chain:self._receipt_hashes.append(sha(receipt_path))
        return value

    def initial_rng(self,value=None):
        folder=self.folder/'initial_rng';path=folder/'state.pkl'
        if folder.exists():
            need(folder.is_dir() and not folder.is_symlink() and set(p.name for p in folder.iterdir())=={'state.pkl','receipt.json'},'Incomplete initial RNG commit')
            expected=json.loads((folder/'receipt.json').read_text())
            need(sha(path)==expected['sha256'] and expected['contract_sha256']==self.contract_sha,'Initial RNG identity differs')
            with path.open('rb') as stream:stored=pickle.load(stream);need(not stream.read(1),'Extra RNG payload bytes')
            validate_rng(stored);return stored
        need(value is not None and self.committed==0,'Initial RNG missing for a committed rank')
        validate_rng(value);stage=self.staging/uuid.uuid4().hex;stage.mkdir()
        raw=pickle.dumps(value,protocol=4);write_file(stage/'state.pkl',raw)
        write_file(stage/'receipt.json',canonical_json(dict(sha256=hashlib.sha256(raw).hexdigest(),contract_sha256=self.contract_sha)))
        sync_directory(stage);os.rename(stage,folder)
        sync_directory(self.folder);return value

    def append(self,step,result,rng_after):
        need(0<=step<len(self.indices),'Step outside original sampler')
        validate_result(result,self.tokens[self.indices[step]])
        validate_rng(rng_after)
        result_sha=semantic_digest(result)
        if step<self.committed:
            previous=self.read(step)
            need(semantic_digest(previous['result'])==result_sha and semantic_digest(previous['rng_after'])==semantic_digest(rng_after),
                 'Replayed committed result/RNG differs exactly; preserve old bytes and stop')
            return 'replayed_exactly_no_overwrite'
        need(step==self.committed,'Cannot append a noncontiguous journal step')
        stage=self.staging/uuid.uuid4().hex;stage.mkdir()
        payload=pickle.dumps(dict(result=result,rng_after=rng_after),protocol=4)
        receipt=dict(schema_version=JOURNAL_SCHEMA,result_digest_format=RESULT_DIGEST_FORMAT,
            step=step,rank=self.rank,contract_sha256=self.contract_sha,dataset_index=self.indices[step],
            token=self.tokens[self.indices[step]],original_result_semantic_sha256=result_sha,payload_sha256=hashlib.sha256(payload).hexdigest(),
            previous_receipt_sha256=self._receipt_hashes[-1] if self._receipt_hashes else None)
        write_file(stage/'payload.pkl',payload);write_file(stage/'receipt.json',canonical_json(receipt));sync_directory(stage)
        destination=self.steps/('%06d'%step)
        need(not destination.exists(),'Refusing to replace an existing step')
        os.rename(stage,destination);sync_directory(self.steps)
        self.committed+=1;self._receipt_hashes.append(sha(destination/'receipt.json'))
        return 'committed'


def common_start(counts,expected_world):
    need(len(counts)==expected_world and all(type(value) is int and value>=0 for value in counts),'Complete integer rank-prefix inventory required')
    # Replay one agreed boundary step, including a completed journal. This proves
    # exact continuity and executes the existing per-attempt precision observers.
    return max(0,min(counts)-1)


def merge_original_order(parts,size,world):
    expected=math.ceil(size/world)
    need(len(parts)==world and all(len(part)==expected for part in parts),'Every original padded rank result required')
    return [row for batch in zip(*parts) for row in batch][:size]
