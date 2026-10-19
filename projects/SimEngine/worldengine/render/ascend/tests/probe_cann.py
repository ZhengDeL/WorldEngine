"""Bounded single-NPU device validation against the independent NumPy oracle.

Run inside an isolated Ascend environment with only the explicitly authorized
logical device exposed. A host watchdog is still required for a device/runtime
hang. This measures small synthetic operator inputs, not scene quality or GPU
parity, full SimEngine acceptance, or a performance comparison.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from numpy_oracle import project,composite,sh_bases
from worldengine.render.ascend.manifest import CONTRACT,CANONICAL_COMMIT,IDENTITY_FIELDS,digest,REQUIRED_CASES


def make_build_manifest(native,versions,limits):
    extension=Path(native.__file__).resolve(strict=True)
    opp=extension.parent/'packages/vendors/customize'
    api=opp/'op_api/lib/libcust_opapi.so'
    kernels=[p for p in (opp/'op_impl').rglob('*') if p.is_file() and p.suffix in ('.o','.so','.json')]
    if not kernels:
        raise ValueError('No compiled operator kernel files found in the selected package')
    files=[dict(role='python_extension',path=str(extension),sha256=digest(extension)),
           dict(role='operator_api',path=str(api.resolve()),sha256=digest(api))]
    files += [dict(role='operator_kernel',path=str(p.resolve()),sha256=digest(p)) for p in sorted(kernels)]
    here=Path(__file__).resolve().parents[1]
    return dict(contract=CONTRACT,canonical_commit=CANONICAL_COMMIT,
                runtime_versions=versions,
                runtime_files=files,adapter_sha256={n:digest(here/n) for n in ('cann.py','sh.py','manifest.py')},
                validation_sources_sha256={n:digest(here/n) for n in ('tests/probe_cann.py','tests/numpy_oracle.py')},
                compatibility_kernel_sha256=digest(here/'patches/calc_render_mtgs_compat.cpp'),
                validated_limits=limits,validation=dict(result='not_run'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',required=True,help='Explicit isolated logical NPU, e.g. npu:0')
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--gaussians',type=int,default=256)
    parser.add_argument('--cameras',type=int,default=8)
    parser.add_argument('--side',type=int,default=64)
    args=parser.parse_args()
    if not (8<=args.gaussians<=4096 and 2<=args.cameras<=8 and 16<=args.side<=256):
        parser.error('Use 8..4096 Gaussians, 2..8 cameras, side 16..256; required tail cases include 7 Gaussians')
    deadline_value=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if deadline_value.tzinfo is None:
        parser.error('--deadline-utc must contain an explicit UTC offset')
    deadline=deadline_value.timestamp()
    def guard():
        if time.time()>=deadline:raise TimeoutError('Authorized device validation deadline reached')
    guard()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(result='running',scope='small synthetic CANN forward validation, no real-scene/GPU/performance acceptance',
                started_utc=datetime.now(timezone.utc).isoformat(),passed_cases=[],measurements=[])
    report_path=args.output/'device_validation.json'
    manifest=None
    try:
        import torch
        import torch_npu
        from worldengine.render.ascend.cann import CANNRasterizer,runtime_identity
        from worldengine.render.ascend.sh import spherical_harmonics
        device=torch.device(args.device)
        if device.type!='npu':raise ValueError('This is an explicitly selected NPU test')
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32=False
        native=importlib.import_module('gauss_splat.gsplat_c')
        if getattr(native,'mtgs_compatibility_contract',lambda:None)()!=CONTRACT:
            raise RuntimeError('The selected CANN package is not the patched forward build')
        limits=dict(max_gaussians=args.gaussians,max_cameras=args.cameras,max_pixels=max(args.side**2,71*37))
        manifest=make_build_manifest(native,runtime_identity(),limits)
        build_path=args.output/'build_manifest.json'
        build_path.write_text(json.dumps(manifest,indent=2)+'\n')
        renderer=CANNRasterizer.for_validation_only(build_path)
        report['identity']={key:manifest[key] for key in IDENTITY_FIELDS}
        report['device_name']=torch.npu.get_device_name(device)
        side=args.side
        rng=np.random.default_rng(20261020)
        def run_case(name,means,scales,opacity,colors=None,views=None,k=None,antialias=False,near=.01,far=10,radius_clip=0,quats=None,width=None,height=None,tile_size=16):
            guard()
            means=np.asarray(means,np.float32).reshape(-1,3);n=len(means)
            w,h=side if width is None else width,side if height is None else height
            scales=np.broadcast_to(np.asarray(scales,np.float32),(n,3)).copy()
            opacity=np.broadcast_to(np.asarray(opacity,np.float32),(n,)).copy()
            quats=np.tile([1.,0,0,0],(n,1)).astype(np.float32) if quats is None else np.asarray(quats,np.float32)
            views=np.eye(4,dtype=np.float32)[None] if views is None else np.asarray(views,np.float32)
            c=len(views)
            k=np.tile([[w*.7,0,w*.5+.5],[0,h*.8,h*.5+.5],[0,0,1]],(c,1,1)).astype(np.float32) if k is None else np.asarray(k,np.float32)
            colors=np.tile([[1.,.2,.1]],(c,n,1)).astype(np.float32) if colors is None else np.asarray(colors,np.float32)
            projected=project(means,quats,scales,views,k,w,h,near=near,far=far,radius_clip=radius_clip)
            expected_rgb,expected_alpha=composite(projected,opacity,colors,w,h,tile_size=tile_size,antialiased=antialias,expected_depth=True)
            def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).to(device=device,dtype=torch.float32)
            inputs={name:tensor(value) for name,value in dict(means=means,quats=quats,scales=scales,opacities=opacity,
                      colors=colors,viewmats=views,Ks=k).items()}
            started=time.monotonic()
            actual_rgb,actual_alpha,meta=renderer(**inputs,width=w,height=h,tile_size=tile_size,
                      rasterize_mode='antialiased' if antialias else 'classic',render_mode='RGB+ED',
                      near_plane=near,far_plane=far,radius_clip=radius_clip)
            torch.npu.synchronize();elapsed=time.monotonic()-started
            actual_rgb=actual_rgb.cpu().numpy();actual_alpha=actual_alpha.cpu().numpy()
            if not np.isfinite(actual_rgb).all() or not np.isfinite(actual_alpha).all():raise AssertionError('Nonfinite CANN image')
            np.testing.assert_allclose(actual_rgb[...,:3],expected_rgb[...,:3],atol=3e-4,rtol=1e-3)
            np.testing.assert_allclose(actual_rgb[...,3:],expected_rgb[...,3:],atol=2e-3,rtol=1e-3)
            np.testing.assert_allclose(actual_alpha,expected_alpha,atol=3e-4,rtol=1e-3)
            np.testing.assert_array_equal(meta['radii'].cpu().numpy(),np.stack([p['radius'] for p in projected]))
            if meta['device_validation_accepted']:raise AssertionError('Unapproved probe was mislabeled as accepted')
            report['passed_cases'].append(name)
            report['measurements'].append(dict(case=name,gaussians=n,cameras=c,pixels=w*h,width=w,height=h,tile_size=tile_size,
                rgb_max_abs=float(np.max(np.abs(actual_rgb[...,:3]-expected_rgb[...,:3]))),
                alpha_max_abs=float(np.max(np.abs(actual_alpha-expected_alpha))),
                expected_depth_max_abs=float(np.max(np.abs(actual_rgb[...,3:]-expected_rgb[...,3:]))),
                forward_elapsed_seconds=elapsed,performance_acceptance=False))
            guard()
        with torch.no_grad():
            run_case('pixel_center',[[0,0,2]],.2,.4)
            for count in (1,2,3,7):
                tail_means=np.stack((np.linspace(-.25,.25,count),np.zeros(count),np.linspace(2,3,count)),axis=-1)
                run_case('covariance_tail_'+str(count),tail_means,.12,.4)
            report['passed_cases'].append('covariance_nonmultiple_of_eight')
            run_case('alpha_threshold',[[-1,0,3],[0,0,3],[1,0,3]],.12,[1/256,.005,1/255])
            run_case('exclusive_transmittance',[[0,0,2],[0,0,3],[0,0,4]],1.,.999,colors=np.eye(3)[None])
            run_case('empty_tiles',[[1000,0,2],[0,0,-1],[0,0,0]],.1,.5)
            run_case('zero_gaussians',np.empty((0,3)),.1,.5)
            run_case('tile32_nonsquare_partial_tiles',[[0,0,2],[.8,.2,3],[-.5,-.3,4]],.2,.5,
                     width=71,height=37,tile_size=32,antialias=True)
            run_case('depth_expected',[[0,0,2],[0,0,4]],.2,[.4,.5])
            run_case('per_camera_rgb',[[0,0,2]],.2,.5,
                     colors=np.array([[[1.,0,0]],[[0,1.,0]]]),views=np.tile(np.eye(4),(2,1,1)))
            run_case('near_far_radius',[[0,0,z] for z in [.005,.01,.02,9.99,10,11]],.005,.7)
            run_case('radius_clip',[[0,0,2],[1,0,3]],.01,.5,radius_clip=3)
            run_case('antialias_projection',[[0,0,2],[.4,.1,3],[-.2,.1,4]],[[.001,.002,.003],[.1,.2,.1],[.01,.03,.02]],.7,antialias=True)
            off=np.array([[[side*.7,0,side*.27],[0,side*.8,side*.61],[0,0,1]]],np.float32)
            run_case('offcenter_projection',[[0,0,2],[3,.2,3],[-2,-.4,2]],[[.1,.3,.2],[.3,.2,.1],[.1,.2,.3]],.6,k=off,antialias=True)
            angle=.15
            rotated=np.eye(4,dtype=np.float32)
            rotated[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
            rotated[0,3]=.1
            run_case('rotated_gaussians_and_cameras',[[0,0,2],[.3,.1,3]],[[.05,.2,.1],[.2,.04,.1]],.5,
                     views=rotated[None],quats=[[.8,.1,.2,.3],[.7,-.2,.1,.4]],antialias=True)
            directions=rng.normal(size=(64,3)).astype(np.float32)
            directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
            features=rng.normal(size=(64,25,3)).astype(np.float32)
            for degree in range(5):
                bases=sh_bases(degree,directions)
                # Test each of all 25 coefficient positions, including unused
                # positions for lower degrees, against independent recurrence.
                onehot=np.broadcast_to(np.eye(25,dtype=np.float32)[:,None,:,None],(25,64,25,3)).copy()
                rays=np.broadcast_to(directions,(25,64,3)).copy()
                expected=np.zeros((25,64,3));expected[:(degree+1)**2]=bases.T[...,None]
                actual=spherical_harmonics(degree,torch.from_numpy(rays).to(device),torch.from_numpy(onehot).to(device)).cpu().numpy()
                np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)
                # The CUDA entry normalizes positive-degree directions again;
                # test that contract explicitly rather than unit rays alone.
                nonunit=directions*np.linspace(.125,8.,len(directions),dtype=np.float32)[:,None]
                nonunit_rays=np.broadcast_to(nonunit,(25,64,3)).copy()
                nonunit_basis=sh_bases(degree,nonunit)
                expected_nonunit=np.zeros((25,64,3));expected_nonunit[:(degree+1)**2]=nonunit_basis.T[...,None]
                actual_nonunit=spherical_harmonics(degree,torch.from_numpy(nonunit_rays).to(device),torch.from_numpy(onehot).to(device)).cpu().numpy()
                np.testing.assert_allclose(actual_nonunit,expected_nonunit,atol=3e-5,rtol=3e-5)
                if degree:
                    expected=np.clip(np.sum(bases[...,None]*features[:,:bases.shape[-1]],axis=-2)+.5,0,1)
                    actual=torch.clamp(spherical_harmonics(degree,torch.from_numpy(directions).to(device),torch.from_numpy(features).to(device))+.5,0,1).cpu().numpy()
                    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)
            expected=1/(1+np.exp(-features[:,0]))
            actual=torch.sigmoid(torch.from_numpy(features[:,0]).to(device)).cpu().numpy()
            np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)
            report['passed_cases'].append('sh_degree_0_4')
            n,c=args.gaussians,args.cameras
            means=rng.uniform([-1.5,-1,2],[1.5,1,9],size=(n,3)).astype(np.float32)
            views=np.tile(np.eye(4),(c,1,1));views[:,0,3]=np.linspace(-.1,.1,c)
            run_case('bounded_multi_camera_scene',means,rng.uniform(.03,.2,(n,3)),rng.uniform(.01,.8,n),
                     colors=rng.uniform(0,1,(c,n,3)),views=views,antialias=True,quats=rng.normal(size=(n,4)))
        if not REQUIRED_CASES.issubset(set(report['passed_cases'])):raise AssertionError('Required validation case missing')
        report['result']='pass'
    except BaseException as error:
        report.update(result='error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['completed_utc']=datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if report['result']=='pass':
            manifest['validation']=dict(result='pass',passed_cases=report['passed_cases'],
                   evidence_path=str(report_path.resolve()),evidence_sha256=digest(report_path))
            approved=args.output/'validated_manifest.json'
            approved.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'result':report['result'],'report':str(report_path)},indent=2))


if __name__=='__main__':main()
