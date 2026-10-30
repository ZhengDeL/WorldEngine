"""Ordered FP32 MSDA value-gradient accumulation in a head-major temporary.

This internal helper consumes the original canonical B,Q,H flattened addresses.
Spatial/head ranges are proved by the unchanged MSDA samples() generator. It
never changes input tensors, point/corner call order, or gradient-channel sums.
CPU is supported for contract tests; model execution remains FP32 NPU only.
"""
MEASURED_PRIMITIVE_SHA256 = '4b4e7acf6c716a902c7863acab1790625f3f909152f168b6669450f1a4b94b87'
MODES = ('none', 'head-major')

def require(value, message):
    if not value:
        raise ValueError(message)

def validate_training_option(mode, *, precision, backend, task='train', runtime=True):
    require(mode in MODES, 'Unsupported MSDA value-gradient layout')
    if mode != 'none':
        require(task == 'train', 'MSDA value-gradient layout requires training')
        require(precision == 'fp32' and backend == 'cuda-compatible',
                'MSDA value-gradient layout requires FP32 cuda-compatible MSDA')
        require(runtime, 'MSDA value-gradient layout requires the complete runtime checkpoint contract')

def head_major_index_add_(target, index, source, *, layout, tensor_api):
    """Update and return the original contiguous B*S*H,C destination in place.

    Only the caller's already validated original B,Q,H address stream is valid.
    Address/head decomposition uses integer operations; all values stay FP32.
    """
    require(type(layout) is tuple and len(layout) == 4 and
            all(type(x) is int and x > 0 for x in layout), 'Positive exact B/S/H/C layout required')
    batch, spatial, heads, channels = layout
    require(channels == 32 and tuple(target.shape) == (batch * spatial * heads, channels),
            'Original contiguous value-gradient geometry required')
    require(index.ndim == 1 and index.numel() > 0 and index.numel() % (batch * heads) == 0,
            'Original B,Q,H address stream required')
    require(tuple(source.shape) == (index.numel(), channels), 'Original B,Q,H,C contributions required')
    require(index.dtype == tensor_api.int64 and target.dtype == source.dtype == tensor_api.float32,
            'Original int64 addresses and FP32 values required')
    require(target.device == source.device == index.device, 'All accumulator tensors must share a device')
    require(all(x.is_contiguous() for x in (target, index, source)), 'Contiguous original accumulator inputs required')
    queries = index.numel() // (batch * heads)
    addresses = index.view(batch, queries, heads)
    values = source.view(batch, queries, heads, channels)
    destination = target.view(batch, spatial, heads, channels)
    batch_offset = tensor_api.arange(batch, device=index.device, dtype=index.dtype).view(batch, 1, 1) * (spatial * heads)
    head_offset = tensor_api.arange(heads, device=index.device, dtype=index.dtype).view(1, 1, heads)
    spatial_index = (addresses - batch_offset - head_offset) // heads
    expanded = spatial_index.permute(0, 2, 1).unsqueeze(-1).expand(batch, heads, queries, channels)
    head_destination = destination.permute(0, 2, 1, 3).contiguous()
    head_source = values.permute(0, 2, 1, 3).contiguous()
    head_destination.scatter_add_(2, expanded, head_source)
    target.copy_(head_destination.permute(0, 2, 1, 3).contiguous().view_as(target))
    return target

def make_accumulator(tensor_api):
    def accumulate(target, index, source, layout):
        return head_major_index_add_(target, index, source, layout=layout, tensor_api=tensor_api)
    return accumulate
