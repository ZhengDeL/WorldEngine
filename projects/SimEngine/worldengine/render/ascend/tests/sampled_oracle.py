"""Independent float64 reference at selected pixels of a complete Gaussian set.

This diagnostic never allocates a full image and is not a rendering backend.
Every Gaussian is projected for every camera. Complete rectangular-tile
candidate sets are retained, then ordered by depth and original Gaussian ID.
The candidate limit raises an error; it never changes the rendered scene.
Only the requested pixels are compared, so this is not full-image validation.
"""
import math

import numpy as np


_ARRAYS = ('means', 'quats', 'scales', 'opacities', 'colors', 'viewmats', 'Ks')
_RENDER = frozenset(('width', 'height', 'tile_size', 'near_plane', 'far_plane',
                     'render_mode', 'rasterize_mode', 'radius_clip', 'eps2d'))


def _check(guard):
    if guard is not None:
        guard()


def _validate(inputs, render, pixels, chunk_size, max_candidates, guard):
    _check(guard)
    if type(chunk_size) is not int or not 1 <= chunk_size <= 65536:
        raise ValueError('chunk_size must be an integer in 1..65536')
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('max_candidates must be a positive integer')
    if not isinstance(inputs, dict) or set(inputs) != set(_ARRAYS):
        raise ValueError('The exact seven Gaussian/camera arrays are required')
    if not isinstance(render, dict) or set(render) != _RENDER:
        raise ValueError('The exact nine render settings are required')
    for name in ('width', 'height', 'tile_size'):
        if type(render[name]) is not int or render[name] < 1:
            raise ValueError('Positive integer render dimensions are required')
    if render['tile_size'] not in (16, 32):
        raise ValueError('Only the validated tile sizes 16 and 32 are supported')
    if render['render_mode'] not in ('RGB', 'RGB+ED') or render['rasterize_mode'] not in ('classic', 'antialiased'):
        raise ValueError('Unsupported render mode')
    for name in ('near_plane', 'far_plane', 'eps2d', 'radius_clip'):
        value = render[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite scalar projection settings are required')
    if not (0 < render['near_plane'] < render['far_plane'] and render['eps2d'] > 0 and render['radius_clip'] >= 0):
        raise ValueError('Invalid projection settings')
    near, far = (float(np.float32(render[name])) for name in ('near_plane', 'far_plane'))
    if not (0 < near < far and math.isfinite(far)):
        raise ValueError('Clipping settings must remain finite and distinct in FP32')
    if any(not isinstance(inputs[name], np.ndarray) for name in _ARRAYS):
        raise ValueError('Reference inputs must be NumPy arrays')
    means, views = inputs['means'], inputs['viewmats']
    if means.ndim != 2 or views.ndim != 3:
        raise ValueError('Invalid Gaussian or camera rank')
    n, cameras = len(means), len(views)
    if cameras < 1:
        raise ValueError('At least one camera is required')
    shapes = ((n, 3), (n, 4), (n, 3), (n,), (cameras, n, 3), (cameras, 4, 4), (cameras, 3, 3))
    for name, shape in zip(_ARRAYS, shapes):
        value = inputs[name]
        if value.shape != shape or value.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Unexpected floating input shape: ' + name)
        # Validate finite values in bounded slices, including unsampled cameras.
        slices = (value[camera] for camera in range(cameras)) if name == 'colors' else (value,)
        for array in slices:
            for start in range(0, len(array), chunk_size):
                _check(guard)
                if not np.isfinite(array[start:start + chunk_size]).all():
                    raise ValueError('Nonfinite reference input: ' + name)
    for start in range(0, n, chunk_size):
        _check(guard)
        end = start + chunk_size
        norm = np.linalg.norm(inputs['quats'][start:end].astype(np.float64), axis=-1)
        if (np.any(norm == 0) or not np.isfinite(norm).all()
                or np.any(inputs['scales'][start:end] <= 0)
                or np.any(inputs['opacities'][start:end] < 0)
                or np.any(inputs['opacities'][start:end] > 1)):
            raise ValueError('Expected nonzero quaternions and activated scales/opacities')
    if np.any(inputs['Ks'][:, 0, 0] <= 0) or np.any(inputs['Ks'][:, 1, 1] <= 0):
        raise ValueError('Positive focal lengths are required')
    try:
        coordinates = list(pixels)
    except TypeError as error:
        raise ValueError('Sample coordinates must be an iterable') from error
    if not coordinates:
        raise ValueError('At least one sample pixel is required')
    result = []
    for coordinate in coordinates:
        _check(guard)
        if (not isinstance(coordinate, (tuple, list, np.ndarray)) or len(coordinate) != 3
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in coordinate)):
            raise ValueError('Sample coordinates must contain integer (camera, y, x) triples')
        camera, y, x = map(int, coordinate)
        if not (0 <= camera < cameras and 0 <= y < render['height'] and 0 <= x < render['width']):
            raise ValueError('Sample pixel outside its camera image')
        result.append((camera, y, x))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate sample coordinates are not allowed')
    return n, cameras, result


def _project_chunk(means, quats, scales, view, intrinsic, render):
    """Float64 projection equations; no runtime/native implementation is used."""
    means, quats, scales = (np.asarray(value, np.float64) for value in (means, quats, scales))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
        w, x, y, z = quats.T
        rotation = np.stack((
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), axis=-1).reshape(-1, 3, 3)
        matrix = rotation * scales[:, None, :]
        covariance = matrix @ matrix.transpose(0, 2, 1)
        points = means @ view[:3, :3].T + view[:3, 3]
        camera_covariance = view[None, :3, :3] @ covariance @ view[None, :3, :3].transpose(0, 2, 1)
        tx, ty, depth = points.T
        safe_z = np.where(depth != 0, depth, 1)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        width, height = render['width'], render['height']
        tanx, tany = .5*width/fx, .5*height/fy
        clamped_x = safe_z * np.clip(tx/safe_z, -cx/fx - .3*tanx, (width-cx)/fx + .3*tanx)
        clamped_y = safe_z * np.clip(ty/safe_z, -cy/fy - .3*tany, (height-cy)/fy + .3*tany)
        zero = np.zeros_like(depth)
        jacobian = np.stack((fx/safe_z, zero, -fx*clamped_x/safe_z**2,
                             zero, fy/safe_z, -fy*clamped_y/safe_z**2), axis=-1).reshape(-1, 2, 3)
        covariance2 = jacobian @ camera_covariance @ jacobian.transpose(0, 2, 1)
        original_det = np.linalg.det(covariance2)
        covariance2 += np.eye(2)[None] * render['eps2d']
        a, b, c = covariance2[:, 0, 0], covariance2[:, 0, 1], covariance2[:, 1, 1]
        determinant = a*c - b*b
        safe_det = np.maximum(determinant, 1e-10)
        conics = np.stack((c, -b, a), axis=-1) / safe_det[:, None]
        compensation = np.sqrt(np.maximum(0, original_det/safe_det))
        midpoint = (a+c)*.5
        radius = np.ceil(3*np.sqrt(midpoint + np.sqrt(np.maximum(.01, midpoint*midpoint - determinant))))
        xy = np.stack((fx*tx/safe_z + cx, fy*ty/safe_z + cy), axis=-1)
        near = float(np.float32(render['near_plane']))
        far = float(np.float32(render['far_plane']))
        valid = ((depth >= near) & (depth <= far) & (determinant > 0)
                 & (radius > render['radius_clip'])
                 & (xy[:, 0]+radius > 0) & (xy[:, 0]-radius < width)
                 & (xy[:, 1]+radius > 0) & (xy[:, 1]-radius < height))
        radius = np.where(valid, radius, 0)
        # Keep integer-valued bounds in float64 to avoid unsafe int64 casts for
        # extreme but finite input. They are compared only to valid pixel tiles.
        low = np.floor((xy-radius[:, None])/render['tile_size'])
        high = np.ceil((xy+radius[:, None])/render['tile_size'])
    for value in (depth, xy, conics, compensation, radius, low, high):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite float64 projected geometry')
    return depth, xy, conics, compensation, radius, low, high


def _composite_pixel(values, x, y, chunk_size, guard):
    """Depth-sorted rows: depth, xy[2], conic[3], effective opacity, RGB[3]."""
    transmittance, weighted_depth = 1., 0.
    rgb = np.zeros(3, np.float64)
    evaluated, contributing, exhausted = 0, 0, False
    for start in range(0, len(values), chunk_size):
        _check(guard)
        rows = values[start:start + chunk_size]
        dx, dy = rows[:, 1]-(x+.5), rows[:, 2]-(y+.5)
        with np.errstate(over='raise', invalid='raise'):
            sigma = .5*(rows[:, 3]*dx*dx + rows[:, 5]*dy*dy) + rows[:, 4]*dx*dy
        if not np.isfinite(sigma).all():
            raise ValueError('Nonfinite float64 pixel exponent')
        alpha = np.zeros(len(rows), np.float64)
        valid = sigma >= 0
        alpha[valid] = np.minimum(.999, rows[valid, 6]*np.exp(-sigma[valid]))
        alpha[alpha < 1/255] = 0
        after = transmittance * np.cumprod(1-alpha)
        crossing = np.flatnonzero((alpha > 0) & (after <= 1e-4))
        limit = int(crossing[0]) if len(crossing) else len(rows)
        # The crossing Gaussian is excluded, exactly as in the forward kernel.
        before = np.empty(len(rows), np.float64)
        before[0] = transmittance
        before[1:] = after[:-1]
        weights = before[:limit] * alpha[:limit]
        rgb += np.sum(weights[:, None]*rows[:limit, 7:10], axis=0)
        weighted_depth += float(np.sum(weights*rows[:limit, 0]))
        contributing += int(np.count_nonzero(weights))
        evaluated += limit + bool(len(crossing))
        if limit:
            transmittance = float(after[limit-1])
        if len(crossing):
            exhausted = True
            break
    alpha_result = 1-transmittance
    depth = weighted_depth/max(alpha_result, 1e-10)
    return rgb, alpha_result, depth, evaluated, contributing, exhausted


def sampled_reference(inputs, render, sample_pixels, *, chunk_size=4096,
                      max_candidates=2_000_000, guard=None):
    """Evaluate selected (camera,y,x) pixels from all original Gaussians.

    ``max_candidates`` bounds the sum of retained Gaussian/tile pairs in one
    camera. A Gaussian intersecting two sampled tiles counts twice. Candidates
    include zero-opacity and subthreshold values; only the actual compositing
    equations can skip their contributions. ``guard`` may raise to stop work.
    """
    n, cameras, pixels = _validate(inputs, render, sample_pixels, chunk_size, max_candidates, guard)
    rgb = np.zeros((len(pixels), 3), np.float64)
    alpha = np.zeros((len(pixels), 1), np.float64)
    expected_depth = np.zeros_like(alpha)
    statistics = dict(gaussians=n, cameras=cameras, sample_pixels=len(pixels),
                      chunk_size=chunk_size, max_candidates=max_candidates,
                      all_gaussians_projected_per_camera=True,
                      gaussian_subsampling=False, candidate_truncation=False,
                      full_image_comparison=False, retained_camera_tile_pairs=0,
                      peak_camera_tile_pairs=0, camera_statistics=[])
    for camera in range(cameras):
        _check(guard)
        tiles = {}
        for sample_index, (sample_camera, y, x) in enumerate(pixels):
            if sample_camera == camera:
                tiles.setdefault((x//render['tile_size'], y//render['tile_size']), []).append(sample_index)
        chunks = {tile: [] for tile in tiles}
        retained, projected = 0, 0
        view = np.asarray(inputs['viewmats'][camera], np.float64)
        intrinsic = np.asarray(inputs['Ks'][camera], np.float64)
        for start in range(0, n, chunk_size):
            _check(guard)
            end = min(n, start+chunk_size)
            depth, xy, conics, compensation, radius, low, high = _project_chunk(
                inputs['means'][start:end], inputs['quats'][start:end], inputs['scales'][start:end],
                view, intrinsic, render)
            projected += end-start
            effective_opacity = inputs['opacities'][start:end].astype(np.float64)
            if render['rasterize_mode'] == 'antialiased':
                effective_opacity *= compensation
            for tile, pieces in chunks.items():
                _check(guard)
                tx, ty = tile
                eligible = ((radius > 0) & (low[:, 0] <= tx) & (tx < high[:, 0])
                            & (low[:, 1] <= ty) & (ty < high[:, 1]))
                indices = np.flatnonzero(eligible)
                count = len(indices)
                if retained+count > max_candidates:
                    raise ValueError('Complete camera/tile candidate budget exceeded: camera=%d count>%d; no truncation allowed'
                                     % (camera, max_candidates))
                if not count:
                    continue
                retained += count
                values = np.column_stack((depth[indices], xy[indices], conics[indices],
                                          effective_opacity[indices], inputs['colors'][camera, start:end][indices]))
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite retained candidate')
                pieces.append((indices.astype(np.int64)+start, values))
        camera_stats = dict(camera=camera, gaussians_projected=projected, sampled_tiles=len(tiles),
                            retained_tile_pairs=retained, tile_candidates=[], samples=[])
        for tile, pieces in chunks.items():
            _check(guard)
            if pieces:
                ids = np.concatenate([piece[0] for piece in pieces])
                values = np.concatenate([piece[1] for piece in pieces])
                pieces.clear()
                order = np.lexsort((ids, values[:, 0]))
                values = values[order]
                del ids, order
            else:
                values = np.empty((0, 10), np.float64)
            camera_stats['tile_candidates'].append(dict(tile_x=tile[0], tile_y=tile[1], count=len(values)))
            for sample_index in tiles[tile]:
                _check(guard)
                _, y, x = pixels[sample_index]
                result = _composite_pixel(values, x, y, chunk_size, guard)
                rgb[sample_index], alpha[sample_index, 0], expected_depth[sample_index, 0] = result[:3]
                camera_stats['samples'].append(dict(sample_index=sample_index, complete_candidates=len(values),
                    evaluated_until_physical_termination=result[3], contributing=result[4],
                    physical_transmittance_termination=result[5]))
        statistics['retained_camera_tile_pairs'] += retained
        statistics['peak_camera_tile_pairs'] = max(statistics['peak_camera_tile_pairs'], retained)
        statistics['camera_statistics'].append(camera_stats)
    _check(guard)
    if not all(np.isfinite(value).all() for value in (rgb, alpha, expected_depth)):
        raise ValueError('Nonfinite sampled reference result')
    return dict(rgb=rgb, alpha=alpha, expected_depth=expected_depth, statistics=statistics)
