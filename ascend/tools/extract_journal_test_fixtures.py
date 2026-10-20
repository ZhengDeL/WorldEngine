"""Statically extract SHA-locked installed MMDetection CPU fixtures.

Uses the standard library only. It does not import Torch, torch_npu, MMDetection
or DrivingSDK and never executes the installed source. Original live-interface
inspection is a separate prerequisite; this helper verifies its recorded source
hashes against the installed file and retains license/version provenance.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat

IMAGE='sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48'
MULTI_SHA='1c17038cc42177a369516d2ca6520d5716c061b83c854e54b577d6d8301c82a7'
COLLECT_SHA='b1bd40c200c4629cee65b59cc38eaa47aa459ff9a269b948052b8755e57bbe98'
EXPECTED={'multi_gpu_test':MULTI_SHA,'collect_results_cpu':COLLECT_SHA}


def require(value,message):
    if not value:raise RuntimeError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def regular(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode),
            'Absolute regular nonsymlink source/license/version file required')
    return path


def stable_bytes(path,expected=None):
    path=regular(path);before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Installed file changed during read')
    if expected is not None:require(digest(raw)==expected,'Installed whole-file SHA differs')
    return raw


def extract_functions(source_file,expected_source_sha256):
    source_file=regular(source_file);raw=stable_bytes(source_file,expected_source_sha256)
    # inspect.getsource uses universal newlines. Preserve this exact LF function
    # representation while separately binding original whole-file bytes.
    text=raw.decode('utf-8').replace('\r\n','\n');tree=ast.parse(text,filename=str(source_file));lines=text.splitlines(keepends=True);rows={}
    for name,expected in EXPECTED.items():
        nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
        require(len(nodes)==1 and isinstance(nodes[0],ast.FunctionDef),'One original top-level function required: '+name)
        node=nodes[0];start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
        source=''.join(inspect.getblock(lines[start:])).encode('utf-8')
        require(digest(source)==expected,'Installed function source SHA differs: '+name)
        rows[name]=dict(function_source=source,provenance=dict(extraction='stdlib_AST_and_inspect_getblock_no_execution',
            module_reference='mmdet.apis.test',installed_source_file=str(source_file),
            installed_source_file_sha256=digest(raw),function_source_sha256=expected,first_line=start+1))
    return rows


def static_version(version_file,expected_version_sha256):
    version_file=regular(version_file);raw=stable_bytes(version_file,expected_version_sha256)
    tree=ast.parse(raw.decode('utf-8'));versions=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='__version__' for target in node.targets):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError,SyntaxError):raise RuntimeError('Version must be a literal string; never execute version source')
            require(isinstance(value,str) and value,'Literal nonempty installed __version__ required');versions.append(value)
    require(len(versions)==1,'One literal installed __version__ assignment required')
    return versions[0],raw


def write_fixtures(output,rows,license_file,notice_file,version_file,expected_version_sha256):
    output=Path(output)
    require(output.is_absolute() and output.resolve()==output,'Absolute canonical fresh fixture output required')
    license_file=regular(license_file);license_raw=stable_bytes(license_file)
    require(all(Path(row['provenance']['installed_source_file']).parents[2]==license_file.parent for row in rows.values()),
            'License must belong to the same installed MMDetection source root')
    require(b'Apache License' in license_raw and b'Version 2.0' in license_raw,
            'The supplied installed MMDetection license is not the expected Apache-2.0 text')
    version_file=regular(version_file)
    require(version_file.parent==license_file.parent/'mmdet','Version must belong to the same installed source root')
    version,version_raw=static_version(version_file,expected_version_sha256)
    payloads={'MMDETECTION_LICENSE':license_raw,'MMDETECTION_VERSION.py.txt':version_raw}
    for name,row in rows.items():payloads['installed_'+name+'.py.txt']=row['function_source']
    if notice_file is not None:
        notice_file=regular(notice_file)
        require(notice_file.parent==license_file.parent,'NOTICE must belong to the same installed source root')
        payloads['MMDETECTION_NOTICE']=stable_bytes(notice_file)
    provenance=dict(schema_version=2,purpose='Local statically extracted CPU fixtures; no package import or device execution',
        publisher='OpenMMLab',upstream_project='https://github.com/open-mmlab/mmdetection',
        upstream_license_reference='https://github.com/open-mmlab/mmdetection/blob/v2.24.0/LICENSE',
        upstream_api_reference='https://github.com/open-mmlab/mmdetection/blob/v2.24.0/mmdet/apis/test.py',
        license='Apache-2.0',installed_license_file=str(license_file),installed_package_version=version,
        installed_version_file=str(version_file),installed_version_file_sha256=expected_version_sha256,
        image_id=IMAGE,dynamic_API_import_executed=False,torch_or_NPU_import_executed=False,
        live_interface_proof='Function SHA constants were obtained by separate original live-image interface inspection; this extraction is static',
        functions={name:row['provenance'] for name,row in rows.items()},
        files={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in payloads.items()})
    output.mkdir(parents=False,exist_ok=False)
    for name,raw in payloads.items():
        with (output/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    with (output/'provenance.json').open('xb') as stream:
        stream.write(json.dumps(provenance,indent=2,allow_nan=False).encode());stream.flush();os.fsync(stream.fileno())
    return provenance


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-file','license-file','version-file'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-source-file-sha256','expected-version-file-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--notice-file',type=Path,help='Original NOTICE, when present in the installed distribution')
    args=parser.parse_args(argv)
    require(os.environ.get('WORLDENGINE_IMAGE_ID')==IMAGE,'Approved immutable image ID required')
    require(not list(Path('/dev').glob('davinci[0-9]*')) and not list(Path('/dev').glob('nvidia[0-9]*')),
            'Run fixture extraction without accelerator device mounts')
    rows=extract_functions(args.source_file,args.expected_source_file_sha256)
    provenance=write_fixtures(args.output,rows,args.license_file,args.notice_file,args.version_file,args.expected_version_file_sha256)
    print(json.dumps(dict(result='generated_local_static_SHA_locked_MMDetection_CPU_fixtures',
                         image_id=provenance['image_id'],files=provenance['files']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
