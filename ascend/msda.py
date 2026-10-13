"""FP32 MSDA using native NPU grid-sampler forward and backward operations.

Only the small integer spatial metadata moves to the host. All floating-point
values, outputs and gradients remain on the NPU. Sampling intermediates are
recomputed in backward so each decoder layer retains only its original inputs.
The coordinate, zero-padding and boundary rules follow the unmodified MMCV
PyTorch reference; this does not establish GPU fused-kernel numerical parity.
"""
import torch
import torch.nn.functional as F


def _sample(value, shape_pairs, locations, weights, legacy_cuda_layout=False):
    if legacy_cuda_layout:
        # The original map head projects four levels but supplies one BEV.
        # MMCV 1.7.2 CUDA addresses the whole contiguous buffer using the
        # actual spatial level count, including its query/batch strides.
        # Preserve that flat-prefix rule and the original checkpoint shapes.
        # A per-head or per-query level slice would change its semantics.
        batch, queries, heads, _, points, _ = locations.shape
        count = batch * queries * heads * len(shape_pairs) * points
        locations = locations.reshape(-1, 2)[:count].reshape(
            batch, queries, heads, len(shape_pairs), points, 2)
        weights = weights.reshape(-1)[:count].reshape(
            batch, queries, heads, len(shape_pairs), points)
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = locations.shape
    values = value.split([height * width for height, width in shape_pairs], dim=1)
    features = []
    for level, (height, width) in enumerate(shape_pairs):
        features.append(values[level].flatten(2).transpose(1, 2).reshape(
            batch * heads, channels, height, width).contiguous())
    outputs = []
    # Queries are independent. Bound transient list/stack/product allocations
    # for camera-rebatched Q=40000 instead of building them all at once.
    for begin in range(0, queries, 2048):
        end = min(begin + 2048, queries)
        grids = 2 * locations[:, begin:end] - 1
        sampled = []
        for level, feature in enumerate(features):
            grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            sampled.append(F.grid_sample(feature, grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False))
        attention = weights[:, begin:end].transpose(1, 2).reshape(
            batch * heads, 1, end - begin, levels * points)
        output = (torch.stack(sampled, dim=-2).flatten(-2) * attention).sum(-1)
        outputs.append(output.view(batch, heads * channels, end - begin).transpose(1, 2))
    return torch.cat(outputs, dim=1).contiguous()


def reference_forward(ctx, value, shapes, starts, locations, weights,
                      allow_legacy_map_layout=False):
    if value.device.type != 'npu' or value.dtype != torch.float32:
        raise ValueError('The reference-compatible MSDA path is FP32 NPU only')
    if locations.device != value.device or weights.device != value.device:
        raise ValueError('MSDA floating inputs must be on the same NPU')
    if value.ndim != 4 or shapes.ndim != 2 or starts.ndim != 1:
        raise ValueError('Invalid MSDA value/spatial metadata dimensions')
    if shapes.dtype not in (torch.int32, torch.int64) or starts.dtype not in (torch.int32, torch.int64):
        raise ValueError('MSDA spatial metadata must be integer tensors')
    pairs = shapes.detach().cpu().tolist()
    offsets = starts.detach().cpu().tolist()
    if not pairs or any(len(pair) != 2 or min(pair) < 1 for pair in pairs):
        raise ValueError('Invalid MSDA spatial shapes')
    sizes = [height * width for height, width in pairs]
    expected, total = [], 0
    for size in sizes:
        expected.append(total)
        total += size
    if offsets != expected or value.shape[1] != total:
        raise ValueError('MSDA level starts/value length disagree with spatial shapes')
    if locations.ndim != 6 or locations.shape[-1] != 2:
        raise ValueError('Invalid MSDA sampling-location shape')
    legacy_layout = (allow_legacy_map_layout and len(pairs) == 1 and
                     locations.shape[3] == 4 and locations.shape[2] == 8 and
                     locations.shape[4] == 4 and value.shape[3] == 32)
    if locations.shape[3] != len(pairs) and not legacy_layout:
        raise ValueError('MSDA projected/spatial level counts disagree; only the '
                         'original MMCV map L4/spatial L1 layout is supported')
    if legacy_layout and not all(item.is_contiguous() for item in (value, locations, weights)):
        raise ValueError('The original MMCV CUDA map layout requires contiguous floating inputs')
    if locations.shape[0] != value.shape[0] or locations.shape[2] != value.shape[2]:
        raise ValueError('MSDA locations/value batch or head dimensions disagree')
    if locations.shape[1] < 1 or locations.shape[4] < 1:
        raise ValueError('MSDA requires nonempty queries and sampling points')
    if tuple(weights.shape) != tuple(locations.shape[:-1]):
        raise ValueError('MSDA attention weights and locations disagree')
    locations, weights = locations.type_as(value), weights.type_as(value)
    ctx._worldengine_reference_msda = True
    ctx._worldengine_msda_shapes = pairs
    ctx._worldengine_msda_legacy_layout = legacy_layout
    ctx.save_for_backward(value, locations, weights)
    with torch.autocast(device_type='npu', enabled=False):
        return _sample(value, pairs, locations, weights, legacy_layout)


def reference_backward(ctx, grad_output):
    saved = ctx.saved_tensors
    needed = [ctx.needs_input_grad[index] for index in (0, 3, 4)]
    if not any(needed):
        return None, None, None, None, None
    with torch.enable_grad(), torch.autocast(device_type='npu', enabled=False):
        inputs = [item.detach().requires_grad_(flag) for item, flag in zip(saved, needed)]
        output = _sample(inputs[0], ctx._worldengine_msda_shapes, inputs[1], inputs[2],
                         ctx._worldengine_msda_legacy_layout)
        active = [item for item, flag in zip(inputs, needed) if flag]
        gradients = iter(torch.autograd.grad(output, active,
                                             grad_output.contiguous(),
                                             create_graph=False, allow_unused=False))
        value, locations, weights = [next(gradients) if flag else None for flag in needed]
    return value, None, None, locations, weights
