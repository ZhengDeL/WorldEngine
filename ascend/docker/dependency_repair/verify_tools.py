"""Verify the independent preparation-tool manifest without importing a framework."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'TOOLS_MANIFEST.json').read_text())
    for relative, wanted in manifest['files'].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Missing or unsafe tool source: ' + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise RuntimeError('Tool source SHA mismatch: ' + relative)
    print(json.dumps({'result': 'verified_local_tool_sources_not_build_or_NPU_acceptance',
                      'files': len(manifest['files'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
