"""Keep a Linux session leader allocated until its owned group is stopped.

Callers must use peek_exit() instead of Popen.poll()/wait() until cleanup_owned.
WNOWAIT retains an exited leader, so its PID/PGID cannot be reused while TERM
and KILL are sent to its original process group. No service enumeration occurs.
Platforms without waitid/WNOWAIT support direct-child cleanup only.
"""
import os
import signal
import subprocess
import sys
import time


def retains_leader():
    return (sys.platform.startswith('linux') and hasattr(os, 'waitid')
            and all(hasattr(os, name) for name in ('WNOWAIT', 'WEXITED', 'WNOHANG', 'P_PID')))


def peek_exit(child):
    """Return the original exit status without reaping the Linux leader."""
    if not retains_leader():
        return child.poll()
    if child.returncode is not None:
        raise RuntimeError('Owned leader was reaped before process-group cleanup')
    result = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is None:
        return None
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def cleanup_owned(child, grace_seconds=3):
    """Stop the owned original group before reaping, even after leader exit0."""
    if retains_leader():
        if child.returncode is not None:
            raise RuntimeError('Cannot establish group ownership after its leader was reaped')
        # waitid establishes that the unreaped PID is still our direct child.
        peek_exit(child)
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Owned child is no longer the expected session/group leader')
        def send(number):
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass
        send(signal.SIGTERM)
        until = time.monotonic()+grace_seconds
        while peek_exit(child) is None and time.monotonic() < until:
            time.sleep(.05)
        # Always kill remaining group members while the leader is retained.
        # Its original exit status is preserved by wait(), including exit0.
        send(signal.SIGKILL)
        return child.wait(timeout=3)
    if child.poll() is None:
        child.terminate()
        try:
            return child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            return child.wait(timeout=3)
    return child.returncode
