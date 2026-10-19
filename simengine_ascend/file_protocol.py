"""Atomic, bounded file exchange used by both closed-loop processes."""
from datetime import datetime
import os
from pathlib import Path
import pickle
import time
import uuid


def deadline_epoch(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Closed-loop deadline must include a timezone')
    return parsed.timestamp()


def check_stop(stop_file=None, deadline_utc=None):
    stop_file = stop_file or os.environ.get('WORLDENGINE_CLOSED_LOOP_STOP_FILE')
    deadline_utc = deadline_utc or os.environ.get('WORLDENGINE_CLOSED_LOOP_DEADLINE_UTC')
    if stop_file and Path(stop_file).exists():
        raise InterruptedError('Closed-loop stop file: '+str(stop_file))
    deadline = deadline_epoch(deadline_utc)
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError('Closed-loop authorization deadline reached')


def atomic_write(path, writer, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Protocol output must not be a symlink: '+str(path))
    temporary = path.with_name('.'+path.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary,path)
        else:
            os.link(temporary,path)
            temporary.unlink()
        if os.name == 'posix':
            descriptor=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
    finally:
        # This exact random path belongs to this call, not a glob of shared files.
        if temporary.exists():temporary.unlink()


def atomic_pickle(path, value):
    atomic_write(path,lambda stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL))


def atomic_numpy(path, value, replace=True):
    import numpy as np
    array=np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError('Object arrays are forbidden in the trajectory protocol')
    atomic_write(path,lambda stream:np.save(stream,array,allow_pickle=False),replace=replace)


def validate_trajectory(value):
    import numpy as np
    array=np.asarray(value)
    if array.shape!=(40,3) or array.dtype.kind!='f' or not np.isfinite(array).all():
        raise ValueError('Closed-loop trajectory must be finite floating[40,3]')
    return array


def wait_numpy(path,timeout_seconds,stop_file=None,deadline_utc=None,poll_seconds=.1):
    import math
    import numpy as np
    if not math.isfinite(timeout_seconds) or timeout_seconds<=0:
        raise ValueError('Trajectory wait timeout must be finite and positive')
    path=Path(path)
    end=time.monotonic()+timeout_seconds
    while True:
        check_stop(stop_file,deadline_utc)
        if path.is_symlink():
            raise ValueError('Trajectory input must not be a symlink')
        try:
            # A published filename is complete: corruption is an error, not an
            # unbounded retry that hides a permanently invalid producer result.
            with path.open('rb') as stream:
                value=np.load(stream,allow_pickle=False)
                if stream.read(1):raise ValueError('Unexpected trailing trajectory bytes')
            return validate_trajectory(value)
        except FileNotFoundError:
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Timed out waiting for trajectory: '+str(path))
            time.sleep(min(poll_seconds,remaining))
