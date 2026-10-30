"""Explicit bridge to the independently pinned AlgEngine inference context.

This module contains no model or operator implementation. The default retains
the original planner path; enabled paths require independent device acceptance.
"""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path

MODES = ('none', 'point-parallel', 'point-parallel-last-frame')
CHUNKS = (2048, 4096, 8192)
ALG_SOURCE = 'cbe546949820096ae12755c95e24e39b39f4fa6e04ec6264b5afa3e78c053610'
ALG_FILES = 249
IMPLEMENTATION = {
    'ascend/inference_optimization.py': '0f9795fba6eb58e29670f7c5d096640cc1194c3718974f256c411858db3761ea',
    'ascend/msda_point_parallel.py': 'cd1b6243f49ef19aecaa1041b142304df0cd127e95e48d231652eae128829a39',
    'ascend/msda.py': '2443500dc267dc57913ed618c3ed778579e3e98c9ecb11a6c948d9ded15963b4',
}


def options(args, backend):
    mode = getattr(args, 'inference_optimization', 'none')
    chunk = getattr(args, 'msda_query_chunk', 2048)
    if mode not in MODES or type(chunk) is not int or chunk not in CHUNKS:
        raise ValueError('Unsupported planner inference optimization or query chunk')
    if mode == 'none':
        if chunk != 2048:
            raise ValueError('A nondefault query chunk requires a planner inference optimization')
    elif (backend != 'npu' or args.precision != 'fp32'
          or getattr(args, 'msda_backend', 'reference') != 'cuda-compatible'):
        raise ValueError('Planner optimizations require NPU FP32 and cuda-compatible MSDA')
    return mode, chunk


def bind_source(root, identity, mode):
    if mode == 'none':
        return None
    if identity != dict(source_sha256=ALG_SOURCE, python_files=ALG_FILES):
        raise ValueError('Planner optimization requires the separately pinned AlgEngine source')
    root = Path(root).resolve(strict=True)
    for name, digest in IMPLEMENTATION.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Planner optimization helper identity differs: ' + name)
    return dict(source_sha256=ALG_SOURCE, python_files=ALG_FILES, helpers=dict(IMPLEMENTATION))


def validate_actual(value, mode, chunk, predictions):
    if type(predictions) is not int or predictions < 1:
        raise RuntimeError('No actual planner predictions for optimization acceptance')
    if (value.get('requested') != mode or value.get('query_chunk') != chunk
            or type(value.get('point_parallel_calls')) is not int or value['point_parallel_calls'] < 1
            or value.get('original_msda_fallback_calls') != 0
            or value.get('original_frame_fallback_calls') != 0 or value.get('fallback_reasons') != {}):
        raise RuntimeError('Requested planner optimization did not execute without fallback')
    if mode == 'point-parallel-last-frame' and (
            value.get('last_frame_calls') != predictions
            or value.get('last_frame_source_contract') != 'pinned_original_methods'
            or value.get('preserved_NumPy_draws') != 6 * predictions):
        raise RuntimeError('Original last-frame/GridMask execution contract is incomplete')


@contextmanager
def activate(runtime):
    mode, chunk = runtime.optimization_options
    if mode == 'none':
        yield
        return
    # Preparation already establishes the external package boundary. Import the
    # tested implementation from that checkout, rather than copying its math.
    from ascend.inference_optimization import optimized_inference
    with optimized_inference(mode, chunk, precision='fp32', backend='cuda-compatible',
                             source_root=runtime.root) as evidence:
        try:
            yield
        finally:
            runtime.provenance['inference_optimization'] = copy.deepcopy(evidence)
