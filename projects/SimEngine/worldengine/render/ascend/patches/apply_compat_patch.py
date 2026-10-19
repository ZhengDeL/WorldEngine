"""Apply the inference-only patch to an isolated, exact CANN source checkout.

Default is read-only.  --apply writes only the three explicitly checked files.
An applied patch is build provenance, not a passing numerical/device validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

COMMIT = '8c0445ebcca91514fce3a7050241f41ed132c340'
CONTRACT = 'mtgs-gsplat14-forward-v1'
ORIGINAL = {
    'kernels/calc_render_fwd_double_clip_gsids/op_kernel/calc_render_fwd_double_clip_gsids.cpp':
        'ddd1b5d279b0e5e2e6d22123864932576e849271d3cb137904f371b5bbf788ae',
    'gauss_splat/csrc/pybind.cpp': '6e28322a8ceb21ad0f7f6b4095dddacc5801272235f2b126fcd668f6288b5842',
    'gauss_splat/csrc/CalcRender.cpp': '3422839ebfa608aaac12ce4b18f0541fc7210e9ea883e1380697520e135f4a58',
}


def transformations(source):
    replacements = {}
    for name, expected in ORIGINAL.items():
        path = source/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Expected unmodified canonical source: '+name)
        replacements[name] = path.read_text()
    kernel = next(name for name in ORIGINAL if name.startswith('kernels/'))
    replacements[kernel] = Path(__file__).with_name('calc_render_mtgs_compat.cpp').read_text()
    name = 'gauss_splat/csrc/pybind.cpp'
    marker = '    m.def("_init_op_api_so_path", &init_op_api_so_path);'
    if replacements[name].count(marker) != 1:
        raise ValueError('Unexpected canonical binding')
    replacements[name] = replacements[name].replace(marker, marker+'\n    m.def("mtgs_compatibility_contract", []() { return "'+CONTRACT+'"; });')
    name = 'gauss_splat/csrc/CalcRender.cpp'
    marker = '    TORCH_CHECK(vColor.device().type() == at::kPrivateUse1, "Invalid device.");'
    if replacements[name].count(marker) != 1:
        raise ValueError('Unexpected canonical backward wrapper')
    replacements[name] = replacements[name].replace(marker,
        '    TORCH_CHECK(false, "MTGS compatibility build is inference-only; backward is unavailable.");\n'+marker)
    return {name: text.replace('\r\n','\n').encode() for name,text in replacements.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise ValueError('Select a fresh patch receipt path')
    source=args.source.resolve(strict=True)
    changed=transformations(source)
    receipt=dict(canonical_commit=COMMIT,contract=CONTRACT,inference_only=True,
                 applied=args.apply,device_validation_passed=False,
                 original_sha256=ORIGINAL,
                 patched_sha256={name:hashlib.sha256(b).hexdigest() for name,b in changed.items()})
    if args.apply:
        for name,b in changed.items():
            (source/name).write_bytes(b)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
