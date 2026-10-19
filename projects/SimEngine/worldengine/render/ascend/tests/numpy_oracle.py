"""Small-scene numerical oracle, never a runtime rendering backend.

Equations follow gsplat v1.4.0 fully_fused_projection_fwd.cu and
rasterize_to_pixels_fwd.cu.  Calculations here use NumPy float64 to provide an
independent target for the FP32 device implementation. This intentionally slow
oracle is limited to 4096 Gaussians and 65536 output pixels per camera.
"""
import math
import numpy as np


def sh_bases(degree,directions):
    """Independent real SH via associated Legendre recurrence, not polynomials.

    Index = l*l+l+m; Condon--Shortley phase gives gsplat's signs. This includes
    the SH degree-zero basis; MTGS's degree-zero color activation is separate.
    """
    if degree not in range(5):raise ValueError('Expected degree 0..4')
    directions=np.asarray(directions,np.float64)
    directions=directions/np.linalg.norm(directions,axis=-1,keepdims=True)
    x,y,z=np.moveaxis(directions,-1,0)
    angle=np.arctan2(y,x);radial=np.sqrt(np.maximum(0,1-z*z))
    result=np.zeros(directions.shape[:-1]+((degree+1)**2,))
    for ell in range(degree+1):
        for order in range(ell+1):
            p=np.ones_like(z)
            for j in range(1,order+1):p*=-(2*j-1)*radial
            if ell>order:
                previous=p;p=z*(2*order+1)*p
                for j in range(order+2,ell+1):
                    previous,p=p,((2*j-1)*z*p-(j+order-1)*previous)/(j-order)
            norm=math.sqrt((2*ell+1)/(4*math.pi)*math.factorial(ell-order)/math.factorial(ell+order))
            if order==0:result[...,ell*ell+ell]=norm*p
            else:
                factor=math.sqrt(2)*norm*p
                result[...,ell*ell+ell-order]=factor*np.sin(order*angle)
                result[...,ell*ell+ell+order]=factor*np.cos(order*angle)
    return result


def project(means,quats,scales,views,intrinsics,width,height,eps=.3,near=.01,far=1e10,radius_clip=0):
    means,quats,scales,views,intrinsics=[np.asarray(v,dtype=np.float64) for v in (means,quats,scales,views,intrinsics)]
    near,far=float(np.float32(near)),float(np.float32(far))
    if len(means)>4096 or width*height>65536:
        raise ValueError('The numerical oracle is restricted to small validation inputs')
    quats=quats/np.linalg.norm(quats,axis=-1,keepdims=True)
    w,x,y,z=quats.T
    rotation=np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                       2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                       2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(-1,3,3)
    matrix=rotation*scales[:,None,:]
    covariance=matrix@matrix.transpose(0,2,1)
    outputs=[]
    for view,k in zip(views,intrinsics):
        points=means@view[:3,:3].T+view[:3,3]
        cam_cov=view[None,:3,:3]@covariance@view[None,:3,:3].transpose(0,2,1)
        tx,ty,depth=points.T
        safe_z=np.where(depth!=0,depth,1)
        fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
        tanx,tany=.5*width/fx,.5*height/fy
        clamped_x=safe_z*np.clip(tx/safe_z,-cx/fx-.3*tanx,(width-cx)/fx+.3*tanx)
        clamped_y=safe_z*np.clip(ty/safe_z,-cy/fy-.3*tany,(height-cy)/fy+.3*tany)
        zero=np.zeros_like(depth)
        jacobian=np.stack((fx/safe_z,zero,-fx*clamped_x/(safe_z**2),
                           zero,fy/safe_z,-fy*clamped_y/(safe_z**2)),axis=-1).reshape(-1,2,3)
        covariance2=jacobian@cam_cov@jacobian.transpose(0,2,1)
        original_det=np.linalg.det(covariance2)
        covariance2+=np.eye(2)[None]*eps
        a,b,c=covariance2[:,0,0],covariance2[:,0,1],covariance2[:,1,1]
        det=a*c-b*b
        conics=np.stack((c,-b,a),axis=-1)/np.maximum(det,1e-10)[:,None]
        compensation=np.sqrt(np.maximum(0,original_det/np.maximum(det,1e-10)))
        midpoint=(a+c)*.5
        radius=np.ceil(3*np.sqrt(midpoint+np.sqrt(np.maximum(.01,midpoint*midpoint-det))))
        xy=np.stack((fx*tx/safe_z+cx,fy*ty/safe_z+cy),axis=-1)
        valid=((depth>=near)&(depth<=far)&(det>0)&(radius>radius_clip)
               &(xy[:,0]+radius>0)&(xy[:,0]-radius<width)
               &(xy[:,1]+radius>0)&(xy[:,1]-radius<height))
        radius=np.where(valid,radius,0).astype(np.int64)
        outputs.append(dict(xy=xy,depth=depth,conics=conics,compensation=compensation,radius=radius))
    return outputs


def composite(projections,opacities,colors,width,height,tile_size=16,antialiased=False,expected_depth=False):
    if width*height>65536:
        raise ValueError('The numerical oracle is restricted to small validation images')
    if len(opacities)>4096:
        raise ValueError('The numerical oracle is restricted to small validation Gaussian sets')
    y,x=np.mgrid[:height,:width]
    pixel_tiles=np.stack((x//tile_size,y//tile_size),axis=-1)
    outputs=[];alphas=[]
    for camera,p in enumerate(projections):
        trans=np.ones((height,width));done=np.zeros_like(trans,dtype=bool)
        rgb=np.zeros((height,width,3));depth=np.zeros_like(trans)
        for g in np.argsort(p['depth'],kind='stable'):
            radius=p['radius'][g]
            if radius<=0:continue
            center=p['xy'][g]
            low=np.floor((center-radius)/tile_size)
            high=np.ceil((center+radius)/tile_size)
            eligible=np.all((pixel_tiles>=low)&(pixel_tiles<high),axis=-1)&~done
            dx,dy=center[0]-(x+.5),center[1]-(y+.5)
            a,b,c=p['conics'][g]
            sigma=.5*(a*dx*dx+c*dy*dy)+b*dx*dy
            opacity=float(opacities[g])*(p['compensation'][g] if antialiased else 1)
            alpha=np.minimum(.999,opacity*np.exp(-sigma))
            eligible&=(sigma>=0)&(alpha>=1/255)
            next_trans=trans*(1-alpha)
            exhausted=eligible&(next_trans<=1e-4)
            done|=exhausted
            used=eligible&~exhausted
            weight=np.where(used,trans*alpha,0)
            rgb+=weight[...,None]*colors[camera,g]
            depth+=weight*p['depth'][g]
            trans=np.where(used,next_trans,trans)
        alpha=1-trans
        if expected_depth:
            rgb=np.concatenate((rgb,(depth/np.maximum(alpha,1e-10))[...,None]),axis=-1)
        outputs.append(rgb);alphas.append(alpha[...,None])
    return np.stack(outputs),np.stack(alphas)
