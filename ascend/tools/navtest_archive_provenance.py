"""Read-only compressed-archive identity and required JPEG byte provenance.

No extraction, filesystem replacement, model import or device call is performed.
Tar path/type/size rules follow the earlier safe streaming extractor. A match is
accepted only after complete gzip EOF/CRC and compressed whole-file SHA checks.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import threading


class Rejected(Exception):pass


def need(value,message):
    if not value:raise Rejected(message)


def parts(name,directory=False):
    need(isinstance(name,str) and name and not name.startswith('/')
         and not any(value in name for value in ('\\',':','\0')),'Unsafe archive member path')
    while name.startswith('./'):name=name[2:]
    if directory:name=name.rstrip('/')
    result=tuple(name.split('/'))
    need(len(name)<=2048 and len(result)<=64 and all(x not in ('','.','..') and len(x)<=255 for x in result),
         'Noncanonical archive member path')
    return result


def relative(member,prefix):
    value=parts(member.name,member.isdir())
    if value[:len(prefix)]==prefix:
        tail=value[len(prefix):];need(tail or member.isdir(),'File occupies archive root');return value,tail
    need(member.isdir() and prefix[:len(value)]==value,'Archive member outside official test prefix')
    return value,()


class DigestReader:
    def __init__(self,stream,guard):self.stream=stream;self.guard=guard;self.count=0;self.digest=hashlib.sha256()
    def read(self,size=-1):
        self.guard.check();need(0<=size<=4*1024**2,'Compressed reads must remain bounded')
        value=self.stream.read(size);self.count+=len(value);self.digest.update(value);return value
    def tell(self):return self.count
    def readable(self):return True
    def seekable(self):return False


def bounded_info(guard,maximum):
    class Info(tarfile.TarInfo):
        def _proc_member(self,archive):
            guard.check();need(self.size>=0,'Negative archive member size')
            metadata=(tarfile.XHDTYPE,tarfile.XGLTYPE,tarfile.GNUTYPE_LONGNAME,tarfile.GNUTYPE_LONGLINK)
            need(self.size<=(65536 if self.type in metadata else maximum),'Archive member/metadata too large')
            return super()._proc_member(archive)
    return Info


def file_stamp(path):
    value=path.lstat();need(stat.S_ISREG(value.st_mode),'Canonical regular archive required')
    need(path.is_absolute() and path.resolve(strict=True)==path,'Archive cannot use symlink aliases')
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns


def audit_archive(path,entry,required,guard,ledger_path,prefix='openscene-v1.1/sensor_blobs/test',
                  max_member_bytes=64*1024**2,max_members=1000000):
    """Fixture API may use a small archive; the public caller enforces official32."""
    path=Path(path);guard.check();before=file_stamp(path)
    need(before[2]==entry['bytes'] and path.name==entry['file'],'Archive byte size/name differs from publisher lock')
    need(re.fullmatch('[0-9a-f]{64}',entry['sha256']) is not None,'Archive SHA required')
    prefix=parts(prefix,True);seen=set();seen_case=set();objects={};case_paths={};matched=set();expanded=members=files=0
    ledger_hash=hashlib.sha256();rows=0
    with Path(ledger_path).open('xb') as ledger, path.open('rb') as source:
        reader=DigestReader(source,guard)
        with gzip.GzipFile(fileobj=reader,mode='rb') as compressed:
            with tarfile.open(fileobj=compressed,mode='r|',ignore_zeros=True,
                              tarinfo=bounded_info(guard,max_member_bytes)) as archive:
                for member in archive:
                    guard.check();members+=1
                    need(members<=max_members,'Archive member count limit exceeded')
                    need(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE,tarfile.DIRTYPE)
                         and member.sparse is None,'Links, sparse and special members are forbidden')
                    need(0<=member.size<=max_member_bytes and (not member.isdir() or member.size==0),'Invalid member size')
                    normalized,tail=relative(member,prefix);name='/'.join(normalized)
                    need(name not in seen and name.casefold() not in seen_case,'Duplicate normalized archive member')
                    seen.add(name);seen_case.add(name.casefold())
                    for depth in range(1,len(normalized)+1):
                        node='/'.join(normalized[:depth]);fold=node.casefold()
                        need(fold not in case_paths or case_paths[fold]==node,'Case alias in an archive ancestor')
                        case_paths[fold]=node
                        directory=depth<len(normalized) or member.isdir()
                        need(node not in objects or (objects[node]=='directory' and directory),
                             'Archive file/directory ancestor conflict')
                        objects[node]='directory' if directory else 'file'
                    if member.isdir():continue
                    files+=1;expanded+=member.size
                    need(expanded<=max(entry['bytes']*4,64*1024**2),'Expanded archive size limit exceeded')
                    key='/'.join(tail)
                    if key not in required:continue
                    expected=required[key]
                    need(member.size==expected['bytes'],'Publisher member size differs from decoded dataset JPEG: '+key)
                    payload=archive.extractfile(member);need(payload is not None,'Missing regular member payload')
                    digest=hashlib.sha256();count=0
                    while count<member.size:
                        guard.check();block=payload.read(min(1024**2,member.size-count))
                        need(block,'Truncated member payload');count+=len(block);digest.update(block)
                    need(not payload.read(1) and count==member.size,'Member payload length differs')
                    need(digest.hexdigest()==expected['sha256'],'Publisher member SHA differs from decoded dataset JPEG: '+key)
                    row=dict(archive=entry['file'],member=member.name,relative=key,bytes=count,sha256=digest.hexdigest())
                    raw=(json.dumps(row,sort_keys=True)+'\n').encode();ledger.write(raw);ledger_hash.update(raw)
                    rows+=1;matched.add(key)
            # This validates gzip trailer CRC/length, concatenated streams and
            # includes every compressed byte in the publisher SHA.
            while compressed.read(1024**2):guard.check()
        while reader.read(1024**2):guard.check()
        need(reader.count==entry['bytes'] and reader.digest.hexdigest()==entry['sha256'],'Complete compressed archive SHA differs')
        need(file_stamp(path)==before,'Archive changed during the read-only audit')
        ledger.flush();os.fsync(ledger.fileno())
    return dict(file=entry['file'],bytes=reader.count,sha256=reader.digest.hexdigest(),members=members,
                regular_files=files,expanded_bytes=expanded,matched_JPEG_rows=rows,
                matched_unique_JPEGs=len(matched),member_ledger=str(ledger_path),member_ledger_sha256=ledger_hash.hexdigest(),
                gzip_EOF_CRC_checked=True,whole_archive_SHA256_verified=True),matched


def audit_all(archive_root,entries,required,guard,output,workers=4):
    need(1<=workers<=4,'Use1..4 bounded archive workers')
    archive_root=Path(archive_root);need(archive_root.is_absolute() and archive_root.resolve(strict=True)==archive_root,'Canonical archive root required')
    output=Path(output);output.mkdir(exist_ok=False)
    futures=[];summaries=[];matched=set();duplicates=0
    cancelled=threading.Event()
    class PoolGuard:
        def check(self):
            need(not cancelled.is_set(),'Archive batch cancelled after another archive failed')
            guard.check()
    pool_guard=PoolGuard()
    pool=ThreadPoolExecutor(max_workers=workers)
    try:
        for index,entry in enumerate(entries):
            guard.check();path=archive_root/entry['file']
            futures.append(pool.submit(audit_archive,path,entry,required,pool_guard,output/('%02d_members.jsonl'%index)))
        for future in as_completed(futures):
            guard.check();row,found=future.result();duplicates+=len(found & matched);matched.update(found);summaries.append(row)
    except BaseException:
        # Do not mark the caller's STOP/deadline state on an integrity error:
        # its SIGALRM must not relabel a rejection as a voluntary pause while
        # these bounded workers finish. True STOP/deadline still propagates.
        cancelled.set()
        for future in futures:future.cancel()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    order={entry['file']:index for index,entry in enumerate(entries)}
    summaries.sort(key=lambda row:order[row['file']])
    need(matched==set(required),'Official archives do not cover the exact required decoded JPEG set')
    return dict(result='pass_all_publisher_archives_and_required_decoded_JPEG_bytes_match',archives=summaries,
                archive_count=len(summaries),whole_archive_bytes=sum(x['bytes'] for x in summaries),
                required_JPEGs=len(required),matched_unique_JPEGs=len(matched),duplicate_equal_JPEG_matches=duplicates,
                image_pixel_provenance_accepted=True,archive_contents_extracted=False)
