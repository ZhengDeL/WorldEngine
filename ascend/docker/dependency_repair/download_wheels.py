"""Fetch genuine locked wheels on a Linux server; default prints a plan only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import urllib.parse
import urllib.request


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def validate_lock(path):
    lock = json.loads(path.read_text())
    names = set()
    for item in lock['artifacts']:
        name = item['filename'];url = urllib.parse.urlsplit(item['url'])
        if (name in names or not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', name)
                or url.scheme != 'https' or url.hostname != 'files.pythonhosted.org'
                or url.username or url.password or url.query or url.fragment
                or url.path.rsplit('/', 1)[-1] != name
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or type(item['bytes']) is not int or not 0 < item['bytes'] < 100 * 1024 ** 2):
            raise ValueError('Invalid, duplicate or non-publisher artifact in lock')
        names.add(name)
    if len(names) != 60 or sum(row['bytes'] for row in lock['artifacts']) != lock['total_bytes']:
        raise ValueError('Complete 60-wheel lock and exact byte total required')
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('WHEELS.lock.json'))
    parser.add_argument('--expected-lock-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-wheelhouse', type=Path, action='append', default=[])
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    declared = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if declared.tzinfo is None:
        parser.error('The deadline must include a timezone')
    lock_path = args.lock.resolve(strict=True)
    if sha(lock_path) != args.expected_lock_sha256:
        raise ValueError('Original lock SHA differs')
    lock = validate_lock(lock_path)
    report = {'result': 'dry_run_no_binary_download', 'artifacts': len(lock['artifacts']),
              'bytes': lock['total_bytes'], 'lock_sha256': args.expected_lock_sha256,
              'routes': ['site_proxy_environment', 'direct_no_proxy'],
              'scope': 'Official HTTPS wheels only; no installation, Docker or device calls'}
    if not args.execute:
        print(json.dumps(report, indent=2));return 0
    if not sys.platform.startswith('linux'):
        parser.error('Download binary artifacts on the Linux deployment server')
    deadline = min(declared.timestamp(), time.time() + 3600)
    if deadline - time.time() < 60:
        parser.error('At least 60 seconds must remain')
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error('Use a new wheelhouse inside an existing directory')
    if shutil.disk_usage(output.parent).free < lock['total_bytes'] + 1024 ** 3:
        raise RuntimeError('Keep at least 1 GiB additional disk space')
    caches = [path.resolve(strict=True) for path in args.reuse_wheelhouse]
    if any(not path.is_dir() for path in caches):
        raise ValueError('Cache inputs must be existing directories')
    output.mkdir();parts = output / '.parts';parts.mkdir()
    rows = [];report.update(result='running', started_UTC=datetime.now(timezone.utc).isoformat(),
                           files=rows, helper_sha256=sha(Path(__file__)))

    def publish():
        (output / 'download_receipt.json').write_text(json.dumps(report, indent=2) + '\n')

    def alarm(*_):
        if time.time() >= deadline:
            raise TimeoutError('Overall wheel download cutoff')
        raise TimeoutError('Whole request/range cutoff')

    old = signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
        for item in lock['artifacts']:
            row = {'filename': item['filename'], 'expected_sha256': item['sha256'],
                   'expected_bytes': item['bytes']};rows.append(row);publish()
            target = output / item['filename'];partial = parts / (item['filename'] + '.part')
            reused = False
            for cache in caches:
                candidate = cache / item['filename']
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                if candidate.stat().st_size != item['bytes'] or sha(candidate) != item['sha256']:
                    continue
                with candidate.open('rb') as source, partial.open('xb') as destination:
                    shutil.copyfileobj(source, destination, 1024 ** 2)
                if sha(candidate) != item['sha256'] or sha(partial) != item['sha256']:
                    raise RuntimeError('Read-only cache or copied bytes changed')
                reused = True;break
            if not reused:
                with partial.open('xb'):
                    pass
                artifact_cutoff = min(deadline, time.time() + 600)
                while partial.stat().st_size < item['bytes']:
                    if time.time() >= artifact_cutoff:
                        raise TimeoutError('Ten-minute artifact cutoff')
                    start = partial.stat().st_size;end = min(item['bytes'] - 1, start + 2 * 1024 ** 2 - 1)
                    error = None
                    for mode in ('environment', 'direct'):
                        bound = min(deadline, artifact_cutoff, time.time() + 60)
                        signal.setitimer(signal.ITIMER_REAL, max(.001, bound - time.time()))
                        handler = urllib.request.ProxyHandler() if mode == 'environment' else urllib.request.ProxyHandler({})
                        opener = urllib.request.build_opener(handler)
                        request = urllib.request.Request(item['url'], headers={
                            'Accept-Encoding': 'identity', 'Range': 'bytes=%d-%d' % (start, end)})
                        try:
                            with opener.open(request, timeout=min(15, max(.1, bound - time.time()))) as response:
                                expected = 'bytes %d-%d/%d' % (start, end, item['bytes'])
                                if (response.status != 206 or response.headers.get('Content-Range') != expected
                                        or int(response.headers.get('Content-Length', -1)) != end - start + 1
                                        or urllib.parse.urlsplit(response.geturl()).hostname != 'files.pythonhosted.org'):
                                    raise ValueError('Publisher response does not match exact range')
                                data = response.read(end - start + 2)
                                if len(data) != end - start + 1:
                                    raise ValueError('Short or oversized publisher range')
                            with partial.open('ab') as stream:
                                if stream.tell() != start:
                                    raise RuntimeError('Own partial offset changed')
                                stream.write(data)
                            error = None;break
                        except (OSError, ValueError) as failure:
                            error = repr(failure);row.setdefault('failed_routes', []).append({'mode': mode, 'error': error})
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()));publish()
                    if error is not None:
                        raise RuntimeError('Both publisher routes failed: ' + error)
            if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']:
                raise RuntimeError('Complete wheel bytes/SHA mismatch')
            # Atomic non-overwrite publication. The directory belongs to this run.
            os.link(partial, target);partial.unlink()
            row.update(result='verified_cache_reused' if reused else 'downloaded_whole_SHA_verified',
                       actual_sha256=sha(target));publish()
        if sha(lock_path) != args.expected_lock_sha256:
            raise RuntimeError('Input lock changed')
        report['result'] = 'complete_all_locked_wheels_verified_no_install'
    except BaseException as failure:
        report.update(result='incomplete_download_evidence_preserved', error=repr(failure))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0);signal.signal(signal.SIGALRM, old)
        report['finished_UTC'] = datetime.now(timezone.utc).isoformat();publish()
    print(json.dumps({'result': report['result'], 'output': str(output)}))
    return 0 if report['result'] == 'complete_all_locked_wheels_verified_no_install' else 1


if __name__ == '__main__':
    raise SystemExit(main())
