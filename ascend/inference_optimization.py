"""Explicit FP32 inference optimizations with original-path fallbacks.

The default is unchanged. This module is activated only by ascend/inference.py;
training does not import it. Arithmetic and frame elimination are separately
selectable so each change can be compared against the original implementation.
"""
import copy
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import warnings

MODES = ('none', 'point-parallel', 'point-parallel-last-frame')
CHUNKS = (2048, 4096, 8192)
_ACTIVE = False
_METHODS = {
    'extract_img_feat': '2cdf81e063c4a534a6c1da624db91becf9eeb997154cecde3cc727ba864df200',
    'get_history_bev': '7293d561869754019abedf5fc20f1898c2f0ff2145a4e71fed0c3f21f913827c',
    'get_bevs': '595600aae48c92a18e19a499fc1c5411461ff98d0c758a659cd4d6c84d970b42',
    '_forward_single_frame': '7950acd55781acf2f4321d989ffdac1a73ba400b30bff828028113bfb6e33f4f',
    'forward_track_test': 'f180d2098ec74a0f655177b28aef3ed6496e5abe1234a78cb3d835ce1c033b29',
}
_GRID_MASK = '14bc011b8855ab200058845d8db7a840a6836c1652d69812d556b38226a2b516'


def validate_options(mode, query_chunk, precision, backend):
    if mode not in MODES or type(query_chunk) is not int or query_chunk not in CHUNKS:
        raise ValueError('Unsupported inference optimization or query chunk')
    if mode == 'none':
        if query_chunk != 2048:
            raise ValueError('A nondefault query chunk requires an inference optimization')
    elif precision != 'fp32' or backend != 'cuda-compatible':
        raise ValueError('Inference optimizations require --precision fp32 --msda-backend cuda-compatible')


def _source_digest(method):
    return hashlib.sha256(inspect.getsource(method).replace('\r\n', '\n').encode()).hexdigest()


def _source_contract(cls, root):
    expected = Path(root) / 'projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py'
    if cls.__name__ != 'NAVFormer' or Path(inspect.getfile(cls)).resolve() != expected.resolve():
        return 'different_NAVFormer_source'
    for name, wanted in _METHODS.items():
        if _source_digest(getattr(cls, name)) != wanted:
            return 'changed_method_' + name
    return None


def _frame_fallback(module, torch, img):
    if module.training or torch.is_grad_enabled():
        return 'requires_eval_no_grad'
    if module.process_perception is not False:
        return 'perception_enabled'
    if any(part.training for part in module.modules()):
        return 'submodule_training'
    if img.ndim != 6 or img.shape[0] != 1 or img.shape[1] != 4:
        return 'requires_B1_T4_queue'
    if module.use_grid_mask:
        grid = module.grid_mask
        if type(grid).__name__ != 'GridMask' or _source_digest(type(grid).forward) != _GRID_MASK:
            return 'changed_GridMask_random_draw_contract'
    return None


@contextmanager
def optimized_inference(mode='none', query_chunk=2048, *, precision='fp32',
                        backend='reference', source_root=None):
    """Temporarily activate requested paths and report actual use per rank.

    Point batching preserves the original ascending level/point accumulation.
    Last-frame execution retains all three historical images and the current
    image in the final original call. Its discarded eval GridMask draws are
    consumed in their original NumPy RNG order. Unknown model shapes/code and
    gradient-bearing MSDA inputs execute the original path instead.
    """
    global _ACTIVE
    validate_options(mode, query_chunk, precision, backend)
    evidence = dict(requested=mode, query_chunk=query_chunk,
                    observation_rank=int(os.environ.get('RANK', '0')),
                    point_parallel_calls=0, original_msda_fallback_calls=0,
                    last_frame_calls=0, original_frame_fallback_calls=0,
                    preserved_NumPy_draws=0, fallback_reasons={},
                    last_frame_source_contract=None)
    if mode == 'none':
        yield evidence
        return
    if _ACTIVE:
        raise RuntimeError('Nested inference optimization contexts are unsupported')
    import numpy as np
    import torch
    from ascend import msda
    from ascend.msda_point_parallel import make_function

    def fallback(reason):
        if reason not in evidence['fallback_reasons']:
            warnings.warn('WorldEngine inference optimization uses original path: ' + reason,
                          RuntimeWarning, stacklevel=3)
        counts = evidence['fallback_reasons']
        counts[reason] = counts.get(reason, 0) + 1

    original_function = msda._cuda_function()
    candidate = make_function(torch, query_chunk=query_chunk)

    class InferenceMSDA:
        @staticmethod
        def forward(ctx, value, locations, weights, pairs, legacy=False, chunk=2048):
            # Custom autograd Function.forward itself runs without grad. The
            # input requires_grad flags, not is_grad_enabled(), protect training.
            if any(item.requires_grad for item in (value, locations, weights)):
                evidence['original_msda_fallback_calls'] += 1
                fallback('gradient_bearing_MSDA_input')
                return original_function.forward(ctx, value, locations, weights, pairs, legacy, chunk)
            evidence['point_parallel_calls'] += 1
            return candidate.forward(ctx, value, locations, weights, pairs, legacy, chunk)

        backward = staticmethod(original_function.backward)

    cls = original_frame = patched_frame = None
    previous_function = msda._CUDA_FUNCTION
    try:
        _ACTIVE = True
        msda._CUDA_FUNCTION = InferenceMSDA
        if mode == 'point-parallel-last-frame':
            from mmdet3d_plugin.navformer.detectors.navformer import NAVFormer
            cls = NAVFormer
            contract = _source_contract(cls, source_root)
            evidence['last_frame_source_contract'] = contract or 'pinned_original_methods'
            if contract:
                fallback(contract)
            else:
                original_frame = cls.forward_track_test

                def final_original_call(self, img, gt_bboxes_3d, gt_labels_3d, gt_inds,
                                        l2g_t, l2g_r_mat, img_metas, timestamp):
                    reason = _frame_fallback(self, torch, img)
                    if reason:
                        evidence['original_frame_fallback_calls'] += 1
                        fallback(reason)
                        return original_frame(self, img, gt_bboxes_3d, gt_labels_3d, gt_inds,
                                              l2g_t, l2g_r_mat, img_metas, timestamp)
                    track_instances = self._generate_empty_tracks()
                    if self.use_grid_mask:
                        for _ in range(6):
                            np.random.rand()
                        evidence['preserved_NumPy_draws'] += 6
                    i = img.size(1) - 1
                    prev_img = img[:, :i, ...]
                    prev_img_metas = copy.deepcopy(img_metas)
                    img_single = torch.stack([image[i] for image in img], dim=0)
                    img_metas_single = [copy.deepcopy(meta[i]) for meta in img_metas]
                    frame_res = self._forward_single_frame(
                        img_single, img_metas_single, track_instances, prev_img, prev_img_metas,
                        l2g_r_mat[0][i], l2g_t[0][i], None, None, None, [], [], [], [])
                    evidence['last_frame_calls'] += 1
                    return {key: frame_res[key] for key in ('bev_embed', 'bev_pos')}

                patched_frame = final_original_call
                cls.forward_track_test = patched_frame
        yield evidence
    finally:
        msda._CUDA_FUNCTION = previous_function
        if cls is not None and patched_frame is not None:
            cls.forward_track_test = original_frame
        _ACTIVE = False
