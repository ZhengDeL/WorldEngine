"""FP32/C32 multi-scale deformable attention with CUDA-compatible pixel arithmetic.

The compensated coordinates and first-order gradients use native PyTorch NPU
operations. Only integer spatial metadata is supplied by the caller on CPU.
See MSDA_CUDA.md for supported shapes, measured tolerances and limitations.
"""
POLICY = {'name': 'cuda_compatible_fp32_c32_v1', 'coordinate': 'FP32 split4097 product residual + TwoSum(product,-0.5) residual + final compensation; integer scale1..4096 with split-scale TwoProd when needed; actual NPU output must be measured', 'forward_order': 'ascending spatial level then sampling point, FP32', 'gradient_channel_order': 'ascending0..31 as original blocksize32 reduce_v1', 'value_gradient_storage': 'Explicit contiguous logical B,S,H,C buffer; input value/layout unchanged', 'value_gradient_order': 'querychunk,level,point,corner device index_add_; differs from native atomic interleaving', 'legacy_layout': 'entire original contiguous raw flat-prefix interpreted before query chunking; unused tail zero', 'coordinate_scope': 'positive integer axes1..4096; finite inputs/intermediates without product-error underflow; local exact-family and actual NPU domains stated separately', 'coordinate_FP64_or_CPU_fallback': False, 'far_padding_guard': 'FP32 finite clamp[-4,4]; nonfinite loc->4, inactive for every positive shape; finite values/weights/GO assumed', 'nonfinite_policy_source': 'MMCVv1.6.2 ms_deform_attn_cuda_kernel.cuh active-window conjunction before bilerp/gradient, lines219..227 and280..307', 'compute_dtype': 'torch.float32', 'channels_per_head': 32}
REFERENCE_NUMERICAL_IMPLEMENTATION_SHA256 = '264676e4c67097de7b939a467da83e48f1e58fc430923813c1720dc2e9477582'

def require(value, message):
    if not value:
        raise ValueError(message)

def canonical(value, locations, weights, pairs, legacy):
    require(value.ndim == 4 and locations.ndim == 6 and (locations.shape[-1] == 2), 'MSDA dimensions differ')
    b, s, h, c = value.shape
    require(c == 32, 'The CUDA-compatible backend requires 32 channels per head')
    require(locations.shape[0] == b and locations.shape[2] == h and (tuple(weights.shape) == tuple(locations.shape[:-1])), 'MSDA geometry differs')
    require(pairs and all((len(p) == 2 and all((type(x) is int and x > 0 for x in p)) for p in pairs)), 'Positive exact host integer shape tuples required')
    require(s == sum((y * x for y, x in pairs)), 'Full value spatial extent differs')
    require(0 < locations.shape[1] and locations.shape[4] > 0, 'Nonempty full queries/points required')
    require(value.dtype == locations.dtype == weights.dtype and str(value.dtype) == 'torch.float32', 'Only original FP32 arithmetic allowed')
    require(value.device == locations.device == weights.device, 'All floating inputs must share the actual device')
    if locations.shape[3] != len(pairs):
        require(legacy and len(pairs) == 1 and (locations.shape[3:5] == (4, 4)) and (h == 8), 'Only explicit original rawL4/spatialL1 flat-prefix mismatch allowed')
        require(all((x.is_contiguous() for x in (value, locations, weights))), 'Original raw legacy buffer must be contiguous')
        count = b * locations.shape[1] * h * len(pairs) * locations.shape[4]
        return (locations.reshape(-1, 2)[:count].reshape(b, locations.shape[1], h, len(pairs), locations.shape[4], 2), weights.reshape(-1)[:count].reshape(b, locations.shape[1], h, len(pairs), locations.shape[4]))
    require(not legacy, 'Legacy flag must describe an actual rawL4/spatialL1 mismatch')
    return (locations, weights)

def integer_scale_parts(scale):
    """Split positive shape metadata into12-significand-bit high plus exact low.

    This is integer-only metadata arithmetic, not a Tensor/FP64/CPU fallback.
    """
    require(type(scale) is int and 1 <= scale <= 4096, 'CUDA-compatible coordinates require integer axis sizes from 1 to 4096')
    shift = max(0, scale.bit_length() - 12)
    if shift == 0:
        return (scale, 0)
    unit = 1 << shift
    high, remainder = divmod(scale, unit)
    half = unit >> 1
    if remainder > half or (remainder == half and high & 1):
        high += 1
    high *= unit
    return (high, scale - high)

def finite_padding_coordinate(torch, x):
    """Protect inactive far-padding coordinates before split4097/floor.

    For any positive integer shape, abs(loc)>4 is outside the native bilinear
    support. NaN/Inf also fail the original CUDA active-window conjunction.
    Map these cases to finite inactive coordinates, preserving zero padding and
    manual zero gradients for finite values/weights/GO. This is not a claim that
    arbitrary nonfinite value/weight tensors are supported.
    """
    return torch.where(torch.isfinite(x), torch.clamp(x, min=-4, max=4), torch.full_like(x, 4))

def compensated_scale_offset_FP32(torch, x, scale):
    """FP32 TwoProd residual and TwoSum compensation; no FP64/host fallback.

    Shape integers1..4096 are supported. Inputs/intermediates must remain finite
    with no underflow affecting the product residual. Local exact finite-family
    checks and actual NPU readbacks establish only their stated tested domains;
    this callable does not claim a general correctly-rounded IEEE FMA primitive.
    """
    scale_high, scale_low = integer_scale_parts(scale)
    x = finite_padding_coordinate(torch, x)
    offset = torch.full_like(x, -0.5)
    product = x * scale
    split = x * 4097
    high = split - (split - x)
    low = x - high
    if scale_low == 0:
        product_error = high * scale - product + low * scale
    else:
        residual1 = product - high * scale_high
        residual2 = residual1 - low * scale_high
        residual3 = residual2 - high * scale_low
        product_error = low * scale_low - residual3
    total = product + offset
    recovered = total - product
    sum_error = product - (total - recovered) + (offset - recovered)
    return total + (product_error + sum_error)

def pixel_coordinates(torch, xy, height, width):
    """The actual FP32 expression used by sampling and coordinate readback."""
    x = compensated_scale_offset_FP32(torch, xy[..., 0], width)
    y = compensated_scale_offset_FP32(torch, xy[..., 1], height)
    return (x, y)

def make_function(torch, grad_value_accumulator=None):
    """Create an isolated autograd Function; never replace registered operators."""

    def samples(value, locations, weights, pairs, chunk):
        b, s, h, c = value.shape
        q = locations.shape[1]
        flat = value.reshape(b * s * h, c)
        batch = torch.arange(b, device=value.device).view(b, 1, 1)
        head = torch.arange(h, device=value.device).view(1, 1, h)
        for begin in range(0, q, chunk):
            end = min(q, begin + chunk)
            start = 0
            for level, (height, width) in enumerate(pairs):
                for point in range(locations.shape[4]):
                    xy = locations[:, begin:end, :, level, point]
                    x, y = pixel_coordinates(torch, xy, height, width)
                    x0 = torch.floor(x).to(torch.long)
                    y0 = torch.floor(y).to(torch.long)
                    lx = x - x0.to(value.dtype)
                    ly = y - y0.to(value.dtype)
                    hx = 1 - lx
                    hy = 1 - ly
                    active = (x > -1) & (y > -1) & (x < width) & (y < height)
                    pixels = []
                    addresses = []
                    masks = []
                    for yy, xx in ((y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)):
                        valid = active & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                        address = (batch * s + start + yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)) * h + head
                        pixels.append(flat.index_select(0, address.reshape(-1)).reshape(b, end - begin, h, c) * valid.unsqueeze(-1))
                        addresses.append(address)
                        masks.append(valid)
                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx
                    v1, v2, v3, v4 = pixels
                    val = w1.unsqueeze(-1) * v1 + w2.unsqueeze(-1) * v2
                    val = val + w3.unsqueeze(-1) * v3
                    val = val + w4.unsqueeze(-1) * v4
                    yield (begin, end, level, point, (height, width), pixels, (hx, hy, lx, ly), (w1, w2, w3, w4), addresses, masks, val, weights[:, begin:end, :, level, point])
                start += height * width

    def serial_channels(tensor):
        output = tensor[..., 0].clone()
        for channel in range(1, 32):
            output = output + tensor[..., channel]
        return output

    class DirectPixel(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value, locations, weights, pairs, legacy=False, chunk=2048):
            require(type(chunk) is int and 1 <= chunk <= 2048, 'Explicit bounded full-query chunk required')
            locations_used, weights_used = canonical(value, locations, weights, pairs, legacy)
            ctx.pairs = tuple((tuple(p) for p in pairs))
            ctx.legacy = legacy
            ctx.chunk = chunk
            ctx.save_for_backward(value, locations, weights)
            b, _, h, c = value.shape
            q = locations.shape[1]
            output = value.new_zeros((b, q, h, c))
            for begin, end, level, point, hw, pixels, fraction, coefficients, addresses, masks, val, weight in samples(value, locations_used, weights_used, pairs, chunk):
                output[:, begin:end].add_(val * weight.unsqueeze(-1))
            return output.reshape(b, q, h * c)

        @staticmethod
        def backward(ctx, grad_output):
            value, rawloc, rawweight = ctx.saved_tensors
            needed = ctx.needs_input_grad[:3]
            if not any(needed):
                return (None, None, None, None, None, None)
            locations, weights = canonical(value, rawloc, rawweight, ctx.pairs, ctx.legacy)
            b, s, h, c = value.shape
            q = locations.shape[1]
            require(tuple(grad_output.shape) == (b, q, h * c) and grad_output.dtype == value.dtype, 'Original GO geometry/dtype differs')
            go = grad_output.contiguous().reshape(b, q, h, c)
            gv = torch.zeros_like(value, memory_format=torch.contiguous_format) if needed[0] else None
            gl = torch.zeros_like(locations) if needed[1] else None
            ga = torch.zeros_like(weights) if needed[2] else None
            for begin, end, level, point, (height, width), pixels, (hx, hy, lx, ly), coef, addresses, masks, val, weight in samples(value, locations, weights, ctx.pairs, ctx.chunk):
                top = go[:, begin:end]
                tg = top * weight.unsqueeze(-1)
                if needed[0]:
                    flat = gv.view(b * s * h, c)
                    for w, addr, mask in zip(coef, addresses, masks):
                        contribution = w.unsqueeze(-1) * tg
                        contribution = contribution * mask.unsqueeze(-1)
                        if grad_value_accumulator is None:
                            flat.index_add_(0, addr.reshape(-1), contribution.reshape(-1, c))
                        else:
                            updated = grad_value_accumulator(flat, addr.reshape(-1),
                                contribution.reshape(-1, c), (b, s, h, c))
                            require(updated is flat, 'Value-gradient accumulator must preserve its destination')
                if needed[1]:
                    v1, v2, v3, v4 = pixels
                    gh = -hx.unsqueeze(-1) * v1
                    gw = -hy.unsqueeze(-1) * v1
                    gh = gh - lx.unsqueeze(-1) * v2
                    gw = gw + hy.unsqueeze(-1) * v2
                    gh = gh + hx.unsqueeze(-1) * v3
                    gw = gw - ly.unsqueeze(-1) * v3
                    gh = gh + lx.unsqueeze(-1) * v4
                    gw = gw + ly.unsqueeze(-1) * v4
                    dx = width * gw * tg
                    dy = height * gh * tg
                    gl[:, begin:end, :, level, point, 0] = serial_channels(dx)
                    gl[:, begin:end, :, level, point, 1] = serial_channels(dy)
                if needed[2]:
                    ga[:, begin:end, :, level, point] = serial_channels(top * val)
            if ctx.legacy:
                if gl is not None:
                    full = torch.zeros_like(rawloc)
                    full.reshape(-1)[:gl.numel()].copy_(gl.reshape(-1))
                    gl = full
                if ga is not None:
                    full = torch.zeros_like(rawweight)
                    full.reshape(-1)[:ga.numel()].copy_(ga.reshape(-1))
                    ga = full
            return (gv, gl, ga, None, None, None)
    return DirectPixel
