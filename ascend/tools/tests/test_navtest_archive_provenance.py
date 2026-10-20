"""Small synthetic tar fixtures; not official data or full NAVTEST acceptance."""
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_CPU_fixture',TOOLS/'navtest_archive_provenance.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
PREFIX='openscene-v1.1/sensor_blobs/test'
KEY='official_log/CAM_F0/image.jpg'


class Guard:
    def __init__(self):self.stopped=False
    def check(self):
        if self.stopped:raise RuntimeError('fixture STOP')
    def stop(self,reason):self.stopped=True


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.guard=Guard();self.payload=b'original JPEG fixture bytes'
        self.required={KEY:dict(bytes=len(self.payload),sha256=hashlib.sha256(self.payload).hexdigest())};self.counter=0
    def make(self,members=None,trailer=b''):
        self.counter+=1;path=self.root/('archive%d.tgz'%self.counter)
        members=members if members is not None else [(PREFIX+'/',None,tarfile.DIRTYPE),(PREFIX+'/'+KEY,self.payload,tarfile.REGTYPE),
                                                     (PREFIX+'/unused/CAM_F0/other.jpg',b'unselected bytes',tarfile.REGTYPE)]
        raw=io.BytesIO()
        with tarfile.open(fileobj=raw,mode='w') as stream:
            for name,payload,kind in members:
                item=tarfile.TarInfo(name);item.type=kind;item.size=len(payload) if payload is not None else 0
                if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):item.linkname='/foreign'
                stream.addfile(item,io.BytesIO(payload) if payload is not None else None)
        path.write_bytes(gzip.compress(raw.getvalue(),mtime=0)+trailer)
        entry=dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return path,entry
    def audit(self,path,entry,required=None,**kwargs):
        self.counter+=1
        return A.audit_archive(path,entry,self.required if required is None else required,self.guard,
                               self.root/('ledger%d.jsonl'%self.counter),**kwargs)
    def test_whole_archive_EOF_and_required_member_bind_decoded_dataset_bits(self):
        path,entry=self.make(trailer=b'\0'*97);before=path.read_bytes();row,matched=self.audit(path,entry)
        self.assertEqual(matched,{KEY});self.assertEqual(row['bytes'],len(before));self.assertEqual(row['sha256'],entry['sha256'])
        self.assertTrue(row['gzip_EOF_CRC_checked'] and row['whole_archive_SHA256_verified']);self.assertEqual(path.read_bytes(),before)
        self.assertEqual(row['matched_unique_JPEGs'],1);self.assertEqual(row['regular_files'],2)
    def test_substituted_dataset_JPEG_bytes_or_size_rejected(self):
        path,entry=self.make()
        for bad in (dict(bytes=len(self.payload),sha256='0'*64),dict(bytes=len(self.payload)+1,sha256=self.required[KEY]['sha256'])):
            with self.subTest(bad=bad),self.assertRaises(A.Rejected):self.audit(path,entry,{KEY:bad})
    def test_compressed_whole_SHA_mismatch_rejected_even_with_matching_member(self):
        path,entry=self.make()
        with self.assertRaisesRegex(A.Rejected,'compressed archive SHA'):self.audit(path,dict(entry,sha256='0'*64))
    def test_path_escape_wrong_prefix_and_case_duplicate_rejected(self):
        for name in ('../escape.jpg','/absolute.jpg','C:/escape.jpg',PREFIX+'/../escape.jpg',PREFIX+'//bad.jpg','foreign/file.jpg'):
            path,entry=self.make([(name,b'x',tarfile.REGTYPE)])
            with self.subTest(name=name),self.assertRaises(A.Rejected):self.audit(path,entry,{})
        for other in (PREFIX+'/'+KEY,'./'+PREFIX+'/'+KEY,(PREFIX+'/'+KEY).upper()):
            path,entry=self.make([(PREFIX+'/'+KEY,self.payload,tarfile.REGTYPE),(other,self.payload,tarfile.REGTYPE)])
            with self.subTest(other=other),self.assertRaises(A.Rejected):self.audit(path,entry)
    def test_links_and_special_members_rejected_even_when_unselected(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.FIFOTYPE,tarfile.CHRTYPE,tarfile.BLKTYPE):
            path,entry=self.make([(PREFIX+'/unselected',None,kind)])
            with self.subTest(kind=kind),self.assertRaises(A.Rejected):self.audit(path,entry,{})
    def test_file_directory_ancestor_conflicts_and_implicit_case_alias_rejected(self):
        for members in (
            [(PREFIX+'/log',b'x',tarfile.REGTYPE),(PREFIX+'/log/cam/x.jpg',b'x',tarfile.REGTYPE)],
            [(PREFIX+'/log/cam/x.jpg',b'x',tarfile.REGTYPE),(PREFIX+'/log',b'x',tarfile.REGTYPE)],
            [(PREFIX+'/Log/cam/a.jpg',b'x',tarfile.REGTYPE),(PREFIX+'/log/cam/b.jpg',b'x',tarfile.REGTYPE)]):
            path,entry=self.make(members)
            with self.subTest(members=members),self.assertRaises(A.Rejected):self.audit(path,entry,{})
    def test_truncated_gzip_or_corrupt_CRC_rejected(self):
        for kind in ('truncated','crc'):
            path,entry=self.make();raw=bytearray(path.read_bytes())
            if kind=='truncated':raw=raw[:-8]
            else:raw[-8]^=1
            path.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            with self.subTest(kind=kind),self.assertRaises(Exception):self.audit(path,entry)
    def test_member_count_size_and_extended_metadata_are_bounded(self):
        path,entry=self.make()
        with self.assertRaises(A.Rejected):self.audit(path,entry,max_member_bytes=8)
        with self.assertRaises(A.Rejected):self.audit(path,entry,max_members=1)
        path,entry=self.make([('pax',b'x'*65537,tarfile.XHDTYPE)])
        with self.assertRaises(A.Rejected):self.audit(path,entry,{})
    def test_STOP_prevents_archive_read_and_success(self):
        path,entry=self.make();self.guard.stop('STOP')
        with self.assertRaisesRegex(RuntimeError,'STOP'):self.audit(path,entry)
    def test_all_archives_exact_required_coverage_and_missing_reference_rejected(self):
        path,entry=self.make();out=self.root/'all'
        result=A.audit_all(self.root,[entry],self.required,self.guard,out,workers=1)
        self.assertTrue(result['image_pixel_provenance_accepted']);self.assertFalse(result['archive_contents_extracted'])
        missing=dict(self.required,missing=dict(bytes=1,sha256='0'*64))
        with self.assertRaisesRegex(A.Rejected,'exact required'):A.audit_all(self.root,[entry],missing,self.guard,self.root/'missing',1)
    def test_later_archive_failure_stops_the_slow_first_archive(self):
        started=threading.Event();cancelled=threading.Event()
        def simulated(path,entry,required,guard,ledger):
            if entry['file']=='first':
                started.set();until=time.monotonic()+2
                while time.monotonic()<until:
                    try:guard.check()
                    except Exception:cancelled.set();raise
                    time.sleep(.001)
                self.fail('Failure did not stop the first archive promptly')
            self.assertTrue(started.wait(1));raise A.Rejected('later archive rejected')
        began=time.monotonic()
        with patch.object(A,'audit_archive',simulated),self.assertRaisesRegex(A.Rejected,'later archive'):
            A.audit_all(self.root,[dict(file='first'),dict(file='second')],{},self.guard,self.root/'parallel',2)
        self.assertTrue(cancelled.is_set());self.assertFalse(self.guard.stopped)
        self.assertLess(time.monotonic()-began,1)


if __name__=='__main__':unittest.main()
