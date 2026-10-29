"""Explicit real-frame windows; no simulation or scoring implementation."""
import hashlib
import numbers
import os
from pathlib import Path
import pickle
import stat


CAMERAS = {'CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2'}


def window_contract(num_history=4, num_future=8):
    if type(num_history) is not int or num_history != 4:
        raise ValueError('The unchanged planner requires exactly four real history frames')
    if type(num_future) is not int or num_future not in (8, 20):
        raise ValueError('Only the reviewed 12-frame default and explicit 24-frame candidate are supported')
    return dict(num_history=num_history, num_future=num_future, frames=num_history+num_future,
                final_frame=num_history+num_future-1,
                consumed_control_indices=list(range(num_history, num_history+num_future)))


def inspect_scene_window(path, expected_sha256, scene_ids, num_history=4, num_future=8, guard=lambda: None):
    """Read only a trusted, whole-SHA checked prepared scene; never extend it.

    The expected hash must come from trusted official preparation. This helper
    checks window availability, not the publisher of an arbitrary supplied hash.
    Default 12-frame wrapper execution does not call this new pickle preflight.
    """
    window = window_contract(num_history, num_future)
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError('Prepared scene must be a regular nonsymlink file')
    def digest(stream):
        stream.seek(0)
        h = hashlib.sha256()
        while True:
            guard(); block = stream.read(8*1024**2)
            if not block: break
            h.update(block)
        return h.hexdigest()
    def identity(value):
        result = (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)
        # Python 3.12 Windows fstat/stat expose different legacy ctime fields;
        # Linux deployment also locks ctime. Whole bytes are checked on both.
        return result+(value.st_ctime_ns,) if os.name == 'posix' else result
    guard()
    with p.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or digest(stream) != expected_sha256:
            raise ValueError('Prepared scene whole SHA differs; pickle was not loaded')
        stream.seek(0)
        scenes = pickle.load(stream)
        if stream.read(1):
            raise ValueError('Unexpected trailing bytes in prepared scene')
        after = os.fstat(stream.fileno())
        if (digest(stream) != expected_sha256 or identity(before) != identity(after)
                or identity(after) != identity(p.stat()) or p.is_symlink()):
            raise ValueError('Prepared scene changed during preflight')
    if not isinstance(scenes, dict) or set(scenes) != set(scene_ids) or len(scenes) != len(scene_ids):
        raise ValueError('Prepared scene keys differ from the explicit scene selection')
    observed = []
    for key in scene_ids:
        guard(); scene = scenes[key]
        if (not isinstance(scene, dict) or not isinstance(scene.get('log_length'), numbers.Integral)
                or isinstance(scene['log_length'], bool) or scene['log_length'] < window['frames']):
            raise ValueError('Original scene log_length is shorter than the requested real window')
        original = scene.get('metadata', {}).get('openscene_data_infos_dict')
        if not isinstance(original, dict) or len(original) < window['frames'] or set(scene.get('cameras', {})) != CAMERAS:
            raise ValueError('Original metadata/camera coverage is shorter than the requested real window')
        frames = list(original.values())[:window['frames']]
        tokens = []
        for frame in frames:
            guard()
            if not isinstance(frame, dict) or not isinstance(frame.get('token'), str) or not frame['token']:
                raise ValueError('Real source frame token is missing')
            tokens.append(frame['token'])
            cams = frame.get('cams')
            if not isinstance(cams, dict) or set(cams) != CAMERAS or any(
                not isinstance(cam, dict) or not isinstance(cam.get('data_path'), str) or not cam['data_path']
                for cam in cams.values()
            ):
                raise ValueError('A real source frame omits original camera metadata')
        if len(set(tokens)) != window['frames']:
            raise ValueError('Real source frame tokens are duplicated')
        observed.append(dict(dictionary_key=key, original_log_length=int(scene['log_length']),
                             original_frame_metadata_count=len(original), checked_real_frames=len(frames),
                             original_scene_unchanged=True))
    return dict(result='pass_original_real_window_available', scene_sha256=expected_sha256,
                configured_window=window, scenes=observed, padded_frames=0,
                publisher_independently_verified=False, scoring_accepted=False)
