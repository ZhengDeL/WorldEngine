"""Real SH bases in the gsplat 1.4 ordering, evaluated on the input device.

The polynomial convention is the one used by gsplat 1.4.0 and Peter-Pike Sloan's
Efficient Spherical Harmonic Evaluation (JCGT 2013).  This module does not apply
MTGS's sigmoid for degree zero, +0.5 offset, or RGB clipping; the original model
continues to apply those operations.
"""
import torch


def spherical_harmonics(degree, directions, coefficients):
    """Evaluate degrees 0--4 without moving tensors to another device."""
    if not isinstance(degree, int) or not 0 <= degree <= 4:
        raise ValueError('The Ascend SH implementation supports integer degrees 0 through 4')
    count = (degree + 1) ** 2
    if directions.shape[-1] != 3 or coefficients.shape[-2] < count or coefficients.shape[-1] != 3:
        raise ValueError('Invalid SH directions or coefficient dimensions')
    if directions.shape[:-1] != coefficients.shape[:-2] or directions.device != coefficients.device:
        raise ValueError('SH directions and coefficients must have matching shape and device')
    # gsplat 1.4 spherical_harmonics.cuh normalizes again for positive degree,
    # even when MTGS has supplied unit directions. Keep its rsqrt convention
    # without epsilon; a zero direction is nonfinite for positive degree, while
    # degree zero never reads a normalized direction.
    if degree >= 1:
        directions = directions * torch.rsqrt((directions*directions).sum(-1,keepdim=True))
    x, y, z = directions.unbind(-1)
    bases = [torch.ones_like(x) * 0.2820947917738781]
    if degree >= 1:
        bases += [-0.48860251190292*y, 0.48860251190292*z, -0.48860251190292*x]
    if degree >= 2:
        z2 = z*z
        c1, s1 = x*x-y*y, 2*x*y
        bases += [0.5462742152960395*s1, -1.092548430592079*z*y,
                  0.9461746957575601*z2-0.3153915652525201,
                  -1.092548430592079*z*x, 0.5462742152960395*c1]
    if degree >= 3:
        c2, s2 = x*c1-y*s1, x*s1+y*c1
        a = -2.285228997322329*z2+0.4570457994644658
        bases += [-0.5900435899266435*s2, 1.445305721320277*z*s1, a*y,
                  z*(1.865881662950577*z2-1.119528997770346), a*x,
                  1.445305721320277*z*c1, -0.5900435899266435*c2]
    if degree >= 4:
        c3, s3 = x*c2-y*s2, x*s2+y*c2
        a = z*(-4.683325804901025*z2+2.007139630671868)
        b = 3.31161143515146*z2-0.47308734787878
        bases += [0.6258357354491763*s3, -1.770130769779931*z*s2, b*s1, a*y,
                  1.984313483298443*z2*(1.865881662950577*z2-1.119528997770346)
                  -1.006230589874905*(0.9461746957575601*z2-0.3153915652525201),
                  a*x, b*c1, -1.770130769779931*z*c2, 0.6258357354491763*c3]
    return (torch.stack(bases, dim=-1).unsqueeze(-1)*coefficients[..., :count, :]).sum(-2)
