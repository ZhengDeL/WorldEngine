"""Explicit FP32 training MSDA with parallel point arithmetic and ordered accumulation.

The arithmetic factory below is preserved from the measured research candidate.
CPU supports contract tests; the model entry remains restricted to FP32 NPU.
The existing head-major helper and CUDA-compatible coordinate functions are
reused without altering their arithmetic or the point/corner update sequence.
"""
from ascend.msda_cuda_compatible import canonical, pixel_coordinates, require
from ascend.msda_value_gradient import (MODES as ORIGINAL_MODES, make_accumulator,
                                      validate_training_option as _validate_original)

MODE = 'point-parallel-head-major'
MODES = ORIGINAL_MODES + (MODE,)
MEASURED_COMBINED_SHA256 = '135f2d062ee7704ae9fb6a4765a1ba326723ecf97de8967e53d22d03bed73a8f'
MEASURED_HEAD_HELPER_SHA256 = '3ea227d93d4a01050c5f83b719cd8b650435d7e94b14f004f58006d42415e03b'


def validate_training_option(mode, *, precision, backend, task='train', runtime=True):
    # The old modes retain their original validator and default behavior.
    _validate_original('head-major' if mode == MODE else mode,
                       precision=precision, backend=backend, task=task, runtime=runtime)


def make_function(torch):
    """Private point-parallel/head-major study; original ordered accumulation."""
    grad_value_accumulator = make_accumulator(torch)

    def samples(value, locations, weights, pairs, chunk):
        b, s, h, c = value.shape
        q = locations.shape[1]
        flat = value.reshape(b * s * h, c)
        batch = torch.arange(b, device=value.device).view(b, 1, 1, 1)
        head = torch.arange(h, device=value.device).view(1, 1, h, 1)
        for begin in range(0, q, chunk):
            end = min(q, begin + chunk)
            start = 0
            for level, (height, width) in enumerate(pairs):
                # P remains a batch dimension through independent arithmetic.
                # canonical() has already interpreted the ENTIRE legacy prefix.
                xy = locations[:, begin:end, :, level]
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
                    pixels.append(flat.index_select(0, address.reshape(-1)).reshape(b, end - begin, h, locations.shape[4], c) * valid.unsqueeze(-1))
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
                yield (begin, end, level, (height, width), pixels, (hx, hy, lx, ly), (w1, w2, w3, w4), addresses, masks, val, weights[:, begin:end, :, level])
                start += height * width

    def serial_channels(tensor):
        output = tensor[..., 0].clone()
        for channel in range(1, 32):
            output = output + tensor[..., channel]
        return output

    class PointParallelCPU(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value, locations, weights, pairs, legacy=False, chunk=2048):
            require(value.device.type in ('cpu', 'npu'), 'Isolated CPU/NPU diagnostic only')
            require(type(chunk) is int and 1 <= chunk <= 2048, 'Explicit bounded full-query chunk required')
            locations_used, weights_used = canonical(value, locations, weights, pairs, legacy)
            ctx.pairs = tuple((tuple(p) for p in pairs))
            ctx.legacy = legacy
            ctx.chunk = chunk
            ctx.save_for_backward(value, locations, weights)
            b, _, h, c = value.shape
            q = locations.shape[1]
            output = value.new_zeros((b, q, h, c))
            for begin, end, level, hw, pixels, fraction, coefficients, addresses, masks, val, weight in samples(value, locations_used, weights_used, pairs, chunk):
                # Do NOT sum P or fuse these updates: baseline level/point order.
                for point in range(locations_used.shape[4]):
                    output[:, begin:end].add_(val[..., point, :] * weight[..., point].unsqueeze(-1))
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
            for begin, end, level, (height, width), pixels, (hx, hy, lx, ly), coef, addresses, masks, val, weight in samples(value, locations, weights, ctx.pairs, ctx.chunk):
                top = go[:, begin:end].unsqueeze(-2)
                tg = top * weight.unsqueeze(-1)
                if needed[0]:
                    flat = gv.view(b * s * h, c)
                    # Point THEN corner, and B/Q/H address flattening unchanged.
                    for point in range(locations.shape[4]):
                        for w, addr, mask in zip(coef, addresses, masks):
                            contribution = w[..., point].unsqueeze(-1) * tg[..., point, :]
                            contribution = contribution * mask[..., point].unsqueeze(-1)
                            returned = grad_value_accumulator(flat, addr[..., point].reshape(-1).contiguous(), contribution.reshape(-1, c).contiguous(), (b, s, h, c))
                            require(returned is flat, 'Head accumulator must return the original value-gradient buffer')
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
                    gl[:, begin:end, :, level, :, 0] = serial_channels(dx)
                    gl[:, begin:end, :, level, :, 1] = serial_channels(dy)
                if needed[2]:
                    ga[:, begin:end, :, level] = serial_channels(top * val)
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
    return PointParallelCPU
