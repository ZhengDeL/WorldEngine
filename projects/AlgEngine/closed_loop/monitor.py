"""Finite, deterministic monitor for complete published frame metadata."""
import asyncio
from collections import deque
import logging
import math
from pathlib import Path
import re
import time


class AsyncFileMonitor:
    def __init__(self,folder_path,pattern='{}_{}.pkl',check_interval=.1,max_history=4,
                 max_frame_per_scene=12,wait_for_files=True,max_wait_time=60,logger=None,
                 stop_file_path=None,cancellation_file=None,deadline_utc=None):
        if not math.isfinite(max_wait_time) or max_wait_time<=0:raise ValueError('Finite positive frame wait timeout required')
        if not math.isfinite(check_interval) or check_interval<=0:raise ValueError('Positive polling interval required')
        if max_history<1 or max_frame_per_scene<max_history:raise ValueError('Invalid frame/history limits')
        self.folder_path=Path(folder_path);self.pattern=pattern;self.check_interval=check_interval
        self.max_history=max_history;self.max_frame_per_scene=max_frame_per_scene
        self.wait_for_files=wait_for_files;self.max_wait_time=max_wait_time
        self.logger=logger or logging.getLogger(__name__);self.stop_file_path=stop_file_path
        self.cancellation_file=cancellation_file;self.deadline_utc=deadline_utc
        self.file_history=deque(maxlen=max_history);self.his_prefix=[];self.scene_idx=0
        self.prefix=None;self.next_idx=0;self._running=False
        self._initialize_queue()

    def _check_cancel(self):
        from simengine_ascend.file_protocol import check_stop
        check_stop(self.cancellation_file,self.deadline_utc)

    def _completed(self):return bool(self.stop_file_path and Path(self.stop_file_path).exists())

    def _wait_for_sufficient_files(self):
        end=time.monotonic()+self.max_wait_time
        while True:
            self._check_cancel()
            if self._completed():return 'stop_iteration'
            groups={}
            for path in sorted(self.folder_path.glob('*.pkl')):
                match=re.fullmatch(r'(.+)_(\d+)\.pkl',path.name)
                if not match:continue
                if path.is_symlink() or not path.is_file():raise ValueError('Unsafe published frame metadata: '+str(path))
                groups.setdefault(match.group(1),set()).add(int(match.group(2)))
            for prefix,indices in groups.items():
                if prefix not in self.his_prefix and set(range(self.max_history)).issubset(indices):
                    self.prefix=prefix;return True
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Timed out waiting for an initial consecutive frame queue in '+str(self.folder_path))
            time.sleep(min(self.check_interval,remaining))

    def _initialize_queue(self):
        if self.wait_for_files and self._wait_for_sufficient_files()=='stop_iteration':return False
        if self.prefix is None:raise ValueError('Scene prefix was not selected')
        paths=[self.folder_path/self.pattern.format(self.prefix,index) for index in range(self.max_history)]
        if not all(path.is_file() and not path.is_symlink() for path in paths):raise ValueError('Initial frame queue is incomplete')
        self.file_history=deque(map(str,paths),maxlen=self.max_history)
        self.next_idx=self.max_history;self.his_prefix.append(self.prefix)
        return True

    def _reinitialize_queue(self,force_reinitialize=False):
        self._check_cancel()
        if self._completed():return False
        if not self.file_history:return self._initialize_queue()
        if self.next_idx>=self.max_frame_per_scene or force_reinitialize:
            self.scene_idx+=1;self.file_history.clear();self.prefix=None
            return self._initialize_queue()
        return False

    def get_current_queue(self):
        self._check_cancel()
        if self.prefix is None:return []
        path=self.folder_path/self.pattern.format(self.prefix,self.next_idx)
        if not path.exists():return []
        if path.is_symlink() or not path.is_file():raise ValueError('Unsafe next frame metadata: '+str(path))
        self.file_history.append(str(path));self.next_idx+=1
        return list(self.file_history)

    def get_file_queue(self):return list(self.file_history)

    def update_history(self,new_file):
        if new_file not in self.file_history:self.file_history.append(new_file)

    async def monitor(self,callback=None):
        self._running=True
        while self._running:
            queue=self.get_current_queue()
            if queue and callback is not None:callback(queue)
            await asyncio.sleep(self.check_interval)

    def stop(self):self._running=False

    @property
    def latest_file(self):return self.file_history[-1] if self.file_history else None
