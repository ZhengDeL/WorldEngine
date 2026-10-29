"""MTGS's gsplat-1.4 forward contract over patched CANN low-level operators.

This does not call CANN's high-level Rasterizer, gaussian_filter or FlashGS mask:
their filtering contracts differ. Gaussian projection and pixel compositing use
CANN operators. Radius construction, tile membership and stable sorting stay on
the same NPU in Torch. Only small integer scheduling counts are copied to CPU,
as required by the canonical scheduler. No CPU rendering fallback is provided.
"""
import importlib
import math
import time
from functools import wraps
import torch
from .manifest import CONTRACT, validate_manifest


def _with_npu_device(function):
    """Canonical raw bindings use the current stream without a DeviceGuard."""
    @wraps(function)
    def guarded(self,*args,**kwargs):
        means=kwargs.get('means')
        if means is None or means.device.type!='npu':
            raise ValueError('An explicit NPU means tensor is required')
        with torch.npu.device(means.device):
            return function(self,*args,**kwargs)
    return guarded


def runtime_identity():
    import acl
    import torch_npu
    version=acl.get_version()
    if len(version)!=4 or version[-1]!=0:
        raise RuntimeError('Cannot establish the active CANN ACL version')
    return {'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),
            'cann_acl':'.'.join(str(v) for v in version[:3]),
            'npu_device_name':torch.npu.get_device_name(torch.npu.current_device())}


def _stable_tile_order(keys,grid_tiles):
    # Int64 ArgSort uses AiCPU on the supported NPU stack. Every tile ID is
    # exactly representable in FP32 within this grid bound; stable=True keeps
    # depth and Gaussian-ID ties unchanged. Larger grids retain integer keys.
    if grid_tiles<=2**24:
        keys=keys.to(torch.float32)
    return torch.argsort(keys,stable=True)


def tile_intersections(means2d, radii, depths, width, height, tile_size, maximum=16_000_000):
    """gsplat 1.4 rectangular tile membership, stable by depth then Gaussian ID."""
    nx,ny=math.ceil(width/tile_size),math.ceil(height/tile_size)
    low=torch.floor((means2d-radii[:,None])/tile_size).to(torch.int64)
    high=torch.ceil((means2d+radii[:,None])/tile_size).to(torch.int64)
    low[:,0].clamp_(0,nx);low[:,1].clamp_(0,ny)
    high[:,0].clamp_(0,nx);high[:,1].clamp_(0,ny)
    spans=high-low
    counts=spans[:,0]*spans[:,1]*(radii>0).to(torch.int64)
    total=int(counts.sum().item())
    if total>maximum:
        raise ValueError('Tile intersection allocation exceeds the explicit safety bound')
    if not total:
        return (torch.empty(0,dtype=torch.int64,device=radii.device),
                torch.zeros(nx*ny,dtype=torch.int64,device=radii.device))
    ids=torch.repeat_interleave(torch.arange(len(radii),device=radii.device),counts,output_size=total)
    starts=torch.cumsum(counts,0)-counts
    # ids already expands each Gaussian by counts; gathering its start offset
    # avoids a second identical repeat_interleave over the same counts.
    relative=torch.arange(total,device=radii.device)-starts[ids]
    tile_x=low[ids,0]+relative%spans[ids,0]
    tile_y=low[ids,1]+relative//spans[ids,0]
    tiles=tile_y*nx+tile_x
    # Two stable sorts preserve increasing Gaussian ID for equal depths, like
    # gsplat's stable sort over its original intersection expansion order.
    order=torch.argsort(depths[ids],stable=True)
    order=order[_stable_tile_order(tiles[order],nx*ny)]
    totals=torch.zeros(nx*ny,dtype=torch.int64,device=radii.device)
    totals.scatter_add_(0,tiles,torch.ones_like(tiles))
    return ids[order].to(torch.int64).contiguous(),totals


def pixel_coordinates(width,height,tile_size,device):
    nx,ny=math.ceil(width/tile_size),math.ceil(height/tile_size)
    ys,xs=torch.meshgrid(torch.arange(ny*tile_size,device=device,dtype=torch.float32),
                         torch.arange(nx*tile_size,device=device,dtype=torch.float32),indexing='ij')
    xy=torch.stack((xs+.5,ys+.5),dim=-1)
    return xy.reshape(ny,tile_size,nx,tile_size,2).permute(0,2,4,1,3).reshape(nx*ny,2,-1).contiguous()


def tiles_to_image(tensor,width,height,tile_size):
    nx,ny=math.ceil(width/tile_size),math.ceil(height/tile_size)
    return tensor.permute(1,2,0).reshape(ny,nx,tile_size,tile_size,-1).permute(0,2,1,3,4).reshape(ny*tile_size,nx*tile_size,-1)[:height,:width]


def _covariances_without_tail(native,quats,scales):
    """Avoid the canonical covariance kernel's unsafe nonaligned tail.

    At the pinned revision, its tail path atomically copies eight float slots
    into planes whose stride is the unaligned Gaussian count. Invalid padded
    quaternions can also produce NaN, which remains NaN after masking by zero.
    Duplicate a real, validated input to align the native call, then discard
    those extra covariance columns before any projection or tile membership.
    This preserves the logical Gaussian set and does not normalize it again.
    """
    n=len(quats)
    if not n:
        raise ValueError('The empty scene must bypass native covariance')
    padding=(-n)%8
    if padding:
        quats=torch.cat((quats,quats[:1].expand(padding,-1)),dim=0)
        scales=torch.cat((scales,scales[:1].expand(padding,-1)),dim=0)
    covars=native.quat_scales_to_covars(quats.T[None].contiguous(),scales.T[None].contiguous())
    covars=covars[...,:n].contiguous()
    if not bool(torch.isfinite(covars).all()):
        raise RuntimeError('The CANN covariance kernel returned nonfinite values')
    return covars


class CANNRasterizer:
    """Inference-only renderer; a passing manifest is mandatory for MTGS use."""
    def __init__(self,manifest,*,_validation_only=False,_validation_lease=None):
        import torch_npu  # Register PrivateUse1 before loading the native extension.
        self.native=importlib.import_module('gauss_splat.gsplat_c')
        if getattr(self.native,'mtgs_compatibility_contract',lambda:None)()!=CONTRACT:
            raise RuntimeError('Unpatched CANN gauss-splat cannot implement the MTGS contract')
        if _validation_lease is not None:
            if manifest is not None or _validation_only:
                raise ValueError('A real-input validation lease cannot be combined with another manifest')
            from .real_validation import validate_lease
            self.manifest,self._validation_deadline=validate_lease(_validation_lease,self.native.__file__,runtime_identity())
            self.validation_only=True
            self._real_input_lease=True
        else:
            self.manifest=validate_manifest(manifest,self.native.__file__,runtime_identity(),_validation_only)
            self.validation_only=_validation_only
            self._validation_deadline=None
            self._real_input_lease=False

    @classmethod
    def for_validation_only(cls,build_manifest):
        """Small device probe before approval; never selected by MTGS config."""
        return cls(build_manifest,_validation_only=True)

    @classmethod
    def for_bounded_real_input_validation(cls,lease):
        """Explicit diagnostic lease; never exposed by the MTGS configuration."""
        return cls(None,_validation_lease=lease)

    def _check_deadline(self):
        if getattr(self,'_validation_deadline',None) is not None and time.time()>=self._validation_deadline:
            raise TimeoutError('The real-input device validation lease expired')

    @torch.no_grad()
    @_with_npu_device
    def __call__(self,*,means,quats,scales,opacities,colors,viewmats,Ks,width,height,
                 tile_size=16,packed=False,near_plane=.01,far_plane=1e10,
                 render_mode='RGB',sparse_grad=False,absgrad=True,
                 rasterize_mode='antialiased',radius_clip=0.,eps2d=.3):
        self._check_deadline()
        if packed or sparse_grad or tile_size not in (16,32) or render_mode not in ('RGB','RGB+ED'):
            raise ValueError('Only MTGS unpacked FP32 RGB/RGB+ED forward with tile 16/32 is supported')
        if rasterize_mode not in ('classic','antialiased') or not 0<near_plane<far_plane:
            raise ValueError('Invalid rasterization mode or clipping planes')
        tensors=(means,quats,scales,opacities,colors,viewmats,Ks)
        if any(t.device.type!='npu' or t.device!=means.device or t.dtype!=torch.float32 for t in tensors):
            raise ValueError('All CANN rendering inputs must be FP32 on the same explicit NPU')
        n,c=len(means),len(viewmats)
        if self.manifest.get('evidence_type') is not None:
            from .real_validation import check_render_contract
            check_render_contract(self.manifest['validated_render_settings'],width=width,height=height,
                tile_size=tile_size,near_plane=near_plane,far_plane=far_plane,render_mode=render_mode,
                rasterize_mode=rasterize_mode,radius_clip=radius_clip,eps2d=eps2d)
            if getattr(self,'_real_input_lease',False):
                shape=self.manifest['capture_binding']['shape']
                if (n,c)!=(shape['gaussians'],shape['cameras']):
                    raise ValueError('A real-input lease requires the exact captured full Gaussian and camera counts')
        shapes=((n,3),(n,4),(n,3),(n,),(c,n,3),(c,4,4),(c,3,3))
        if any(tuple(t.shape)!=shape for t,shape in zip(tensors,shapes)):
            raise ValueError('MTGS rendering input shape differs from the supported contract')
        limits=self.manifest['validated_limits']
        if (n>limits['max_gaussians'] or c>limits['max_cameras'] or width*height>limits['max_pixels']
                or width<1 or height<1 or c<1):
            raise ValueError('Rendering shape is outside the manifest tested envelope')
        if self.validation_only and not getattr(self,'_real_input_lease',False) and (n>4096 or c>8 or width*height>65536):
            raise ValueError('Unapproved device probes are limited to small synthetic scenes')
        if torch.npu.get_device_name(torch.npu.current_device())!=self.manifest['runtime_versions']['npu_device_name']:
            raise ValueError('The selected NPU differs from the tested device family')
        if any(not bool(torch.isfinite(t).all()) for t in (viewmats,Ks)):
            raise ValueError('Nonfinite camera matrices are not valid rendering inputs')
        if not n:
            return (torch.zeros(c,height,width,4 if render_mode=='RGB+ED' else 3,device=means.device),
                    torch.zeros(c,height,width,1,device=means.device),
                    dict(contract=CONTRACT,backend='cann',inference_only=True,
                         device_validation_accepted=not self.validation_only,
                         radii=torch.empty(c,0,dtype=torch.int64,device=means.device)))
        if (any(not bool(torch.isfinite(t).all()) for t in tensors[:5])
                or bool((scales<=0).any()) or bool(((opacities<0)|(opacities>1)).any())):
            raise ValueError('Invalid finite/activated Gaussian parameters')
        if bool((quats.norm(dim=-1)==0).any()):
            raise ValueError('Zero quaternion is not a valid Gaussian orientation')
        # Both gsplat's CUDA kernel and CANN's covariance kernel normalize.
        # Do not introduce an additional normalization between MTGS and CANN.
        covars=_covariances_without_tail(self.native,quats,scales)
        coordinates=pixel_coordinates(width,height,tile_size,means.device)
        images=[];alphas=[];radii_by_camera=[]
        import acl
        vector_count,status=acl.get_device_capability(torch.npu.current_device(),1)
        if status or vector_count<1:
            raise RuntimeError('Cannot establish the selected NPU vector-core count')
        for cam in range(c):
            self._check_deadline()
            if n:
                projected=self.native.projection_three_dims_gaussian_forward(
                    means.T[None].contiguous(),covars,opacities[None].contiguous(),
                    viewmats[None,cam:cam+1].contiguous(),Ks[None,cam:cam+1].contiguous(),
                    width,height,eps2d,rasterize_mode=='antialiased','pinhole')
                xy,depth,conics,comp,det,_,cov2=projected
                xy=xy[0,0].T;depth=depth[0,0,0];conics=conics[0,0].T
                det=det[0,0,0];cov2=cov2[0,0]
                b=(cov2[0]+cov2[2])*.5
                radius=torch.ceil(3*torch.sqrt(b+torch.sqrt(torch.clamp(b*b-det,min=.01))))
                valid=((det>0)&(depth>=near_plane)&(depth<=far_plane)&(radius>radius_clip)
                       &(xy[:,0]+radius>0)&(xy[:,0]-radius<width)
                       &(xy[:,1]+radius>0)&(xy[:,1]-radius<height))
                radius=torch.where(valid,radius,0).to(torch.int64)
                xy=torch.where(valid[:,None],xy,0)
                ids,counts=tile_intersections(xy,radius,depth,width,height,tile_size)
            else:
                radius=torch.empty(0,dtype=torch.int64,device=means.device)
                ids=torch.empty(0,dtype=torch.int64,device=means.device)
            radii_by_camera.append(radius)
            if len(ids):
                effective=opacities*comp[0,0,0] if rasterize_mode=='antialiased' else opacities
                attributes=torch.cat((xy,conics,effective[:,None],colors[cam],depth[:,None]),dim=-1).contiguous()
                # Canonical scheduling is explicitly CPU integer bookkeeping;
                # geometry, Gaussian payloads and rendered pixels never move CPU.
                schedule=self.native.get_render_schedule(counts.cpu(),vector_count).to(means.device)
                rgb,weighted_depth,log_t,*_=self.native.calc_render_fwd_double_clip_gsids(
                    attributes,coordinates,schedule,ids)
                alpha=-torch.expm1(log_t)
                rgb=tiles_to_image(rgb,width,height,tile_size)
                alpha=tiles_to_image(alpha[None],width,height,tile_size)
                weighted_depth=tiles_to_image(weighted_depth,width,height,tile_size)
            else:
                rgb=torch.zeros(height,width,3,device=means.device)
                alpha=torch.zeros(height,width,1,device=means.device)
                weighted_depth=torch.zeros_like(alpha)
            if render_mode=='RGB+ED':
                expected_depth=weighted_depth/alpha.clamp_min(1e-10)
                rgb=torch.cat((rgb,expected_depth),dim=-1)
            images.append(rgb);alphas.append(alpha)
        self._check_deadline()
        return torch.stack(images),torch.stack(alphas),dict(
            contract=CONTRACT,backend='cann',inference_only=True,
            device_validation_accepted=not self.validation_only,radii=torch.stack(radii_by_camera))
