"""FP32 MSDA point batching with unchanged level/point accumulation order.

Used only by the explicit inference optimization context. Training retains the
original CUDA-compatible implementation, including its backward path.
"""
from ascend.msda_cuda_compatible import canonical, pixel_coordinates, require


def make_function(torch, query_chunk=2048):
    """Keep per-point arithmetic and all accumulation loops in baseline order."""

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

    require(type(query_chunk) is int and query_chunk in (2048, 4096, 8192),
            'Inference query chunk must be 2048, 4096 or 8192')

    class PointParallelInference(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value, locations, weights, pairs, legacy=False, chunk=2048):
            require(value.device.type in ('cpu', 'npu'), 'Only FP32 CPU/NPU point-batched inference supported')
            require(type(chunk) is int and 1 <= chunk <= 2048, 'Explicit bounded full-query chunk required')
            chunk = query_chunk
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
            raise RuntimeError('Point-parallel MSDA is inference-only; use the original backend for training')
    return PointParallelInference
