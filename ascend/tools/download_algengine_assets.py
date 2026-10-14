"""Download selected SHA-pinned AlgEngine inputs on a Linux server."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]


def selected(group, asset):
    name = asset['path']
    if group == 'all':
        return True
    if group == 'weights':
        return '/ckpts/' in name
    return name.endswith('.npy') or (
        ('/merged_infos_navformer/' in name or '/pdms_cache/' in name)
        and ('_'+group+'.pkl') in name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root containing alg_engine/ and raw/; not the source checkout')
    parser.add_argument('--group', choices=('navtest', 'navtrain', 'weights', 'all'), default='navtest',
                        help='Explicit all includes both full metadata/PDM splits; no camera or SimEngine download')
    parser.add_argument('--source', choices=('hf-mirror', 'modelscope', 'huggingface'), default='hf-mirror')
    args = parser.parse_args(argv)
    if not sys.platform.startswith('linux'):
        parser.error('Run data and weight downloads on the designated Linux server')
    lock = json.loads((ROOT/'ascend/assets.lock.json').read_text(encoding='utf8'))
    rows = [asset for asset in lock['assets'] if selected(args.group, asset)]
    if not rows:
        raise RuntimeError('No locked assets match the selected group')
    spec = importlib.util.spec_from_file_location('worldengine_pinned_fetch', ROOT/'ascend/fetch_assets.py')
    fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetcher)
    destination = args.data_root.resolve()
    records = []
    for asset in rows:
        relative = Path(asset['path']).relative_to('data')
        target = destination/relative
        if args.source == 'modelscope':
            query = urllib.parse.urlencode(dict(Source='SDK', Revision='master', FilePath=asset['path'], View=False))
            url = 'https://www.modelscope.cn/api/v1/datasets/OpenDriveLab/WorldEngine/repo?'+query
        else:
            host = 'hf-mirror.com' if args.source == 'hf-mirror' else 'huggingface.co'
            url = f"https://{host}/datasets/{lock['dataset']}/resolve/{lock['revision']}/{asset['path']}"
        fetcher.fetch(url, target, asset['bytes'], asset['sha256'])
        actual = fetcher.digest(target)
        if actual != asset['sha256'] or target.stat().st_size != asset['bytes']:
            raise RuntimeError('Locked asset verification failed: '+str(relative))
        records.append(dict(relative_path=relative.as_posix(), bytes=asset['bytes'], sha256=actual, url=url))
    receipt = dict(result='pass_selected_locked_assets', group=args.group, source=args.source,
                   dataset=lock['dataset'],
                   requested_revision='master' if args.source == 'modelscope' else lock['revision'],
                   content_reference_revision=lock['revision'], files=records,
                   cameras_downloaded=False, SimEngine_assets_downloaded=False)
    receipt_path = destination/'alg_engine'/('download_'+args.group+'.json')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
