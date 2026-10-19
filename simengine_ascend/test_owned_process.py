"""Linux process fixtures: exited leaders, stubborn owned children and isolation."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

from simengine_ascend.launch_closed_loop import launch
from simengine_ascend.owned_process import cleanup_owned, peek_exit, retains_leader
from simengine_ascend.run_simulation import run_owned


def command_for_orphan(ready, marker, leader_exit=0, leader_sleep=False):
    child_code = ('import os,signal,time;from pathlib import Path;'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
        'marker=%r;Path(%r).write_text(str(os.getpid()));time.sleep(120)' % (marker, str(ready)))
    leader_code = ('import subprocess,sys,time;from pathlib import Path;'
        'child=subprocess.Popen([sys.executable,"-c",%r]);'
        'ready=Path(%r);end=time.monotonic()+10\n'
        'while not ready.exists() and time.monotonic()<end:time.sleep(.01)\n'
        'if not ready.exists():raise RuntimeError("fixture child did not become ready")\n'
        '%s\nsys.exit(%d)' % (child_code, str(ready), 'time.sleep(120)' if leader_sleep else '', leader_exit))
    return [sys.executable, '-c', leader_code]


def not_running(pid):
    try:
        text = Path('/proc/%d/stat' % pid).read_text()
    except FileNotFoundError:
        return True
    return text[text.rfind(')')+2:].split()[0] in ('Z', 'X')


@unittest.skipUnless(retains_leader(), 'Requires Linux waitid/WNOWAIT and /proc; no mocked group-ownership proof')
class OwnedGroupLinuxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.marker = 'owned-fixture-'+uuid.uuid4().hex
        self.ready = self.root / 'owned_child.pid'
        self.foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(120)'], start_new_session=True)

    def tearDown(self):
        # Only a matching fixture command may be stopped after a test failure.
        if self.ready.exists():
            pid = int(self.ready.read_text())
            try:
                command = Path('/proc/%d/cmdline' % pid).read_bytes()
                if self.marker.encode() in command and not not_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if self.foreign.poll() is None:
            self.foreign.terminate()
        self.foreign.wait(timeout=5)
        self.temporary.cleanup()

    def assert_owned_stopped_foreign_alive(self):
        self.assertTrue(self.ready.is_file(), 'Actual same-group child must have installed SIGTERM ignore')
        pid = int(self.ready.read_text())
        until = time.monotonic()+3
        while not not_running(pid) and time.monotonic() < until:
            time.sleep(.02)
        self.assertTrue(not_running(pid), 'Owned same-group child survived cleanup')
        self.assertIsNone(self.foreign.poll(), 'Unrelated sentinel was signalled')

    def test_nonreaping_peek_preserves_exit0_until_group_cleanup(self):
        with (self.root / 'leader.log').open('wb') as stream:
            child = subprocess.Popen(command_for_orphan(self.ready, self.marker), stdout=stream, stderr=stream, start_new_session=True)
            try:
                until = time.monotonic()+10
                while peek_exit(child) is None and time.monotonic() < until:
                    time.sleep(.02)
                self.assertEqual(peek_exit(child), 0)
                self.assertIsNone(child.returncode, 'Leader must remain unreaped until its group is stopped')
                self.assertEqual(cleanup_owned(child), 0)
                self.assertEqual(child.returncode, 0)
            finally:
                if child.returncode is None:
                    cleanup_owned(child)
        self.assert_owned_stopped_foreign_alive()

    def test_run_owned_leader_exit0_stops_ignoring_child_and_preserves_zero(self):
        status = run_owned(command_for_orphan(self.ready, self.marker), self.root, dict(os.environ),
                           self.root / 'run.log', lambda: None)
        self.assertEqual(status, 0)
        self.assert_owned_stopped_foreign_alive()

    def test_run_owned_deadline_stops_child_after_TERM_exits_leader(self):
        startup_until = time.monotonic()+10
        deadline = []
        def guard():
            if self.ready.exists() and not deadline:
                deadline.append(time.monotonic()+.2)
            if not deadline and time.monotonic() >= startup_until:
                raise RuntimeError('Fixture child did not install its SIGTERM handler')
            if deadline and time.monotonic() >= deadline[0]:
                raise TimeoutError('Linux fixture deadline')
        with self.assertRaisesRegex(TimeoutError, 'Linux fixture deadline'):
            run_owned(command_for_orphan(self.ready, self.marker, leader_sleep=True), self.root,
                      dict(os.environ), self.root / 'cutoff.log', guard)
        self.assert_owned_stopped_foreign_alive()

    def test_closed_loop_launcher_cleans_zero_exit_leader_group(self):
        plan = dict(planner=dict(argv=command_for_orphan(self.ready, self.marker), cwd=str(self.root)),
                    simulation=dict(argv=[sys.executable, '-c', 'print("fixture simulation leader")'], cwd=str(self.root)))
        path = self.root / 'plan.json'
        path.write_text(json.dumps(plan))
        output = self.root / 'job'
        deadline = (datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()
        self.assertEqual(launch(path, output, deadline), 0)
        report = json.loads((output / 'launcher_report.json').read_bytes())
        self.assertEqual(report['children']['planner']['actual_exit'], 0)
        self.assertEqual(report['children']['simulation']['actual_exit'], 0)
        self.assertFalse(report['real_closed_loop_accepted'])
        self.assert_owned_stopped_foreign_alive()


if __name__ == '__main__':
    unittest.main(verbosity=2)
