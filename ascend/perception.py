"""Validate actual perception predictions and retain empty-union map coverage."""
import numpy as np
import torch

def validate_perception_outputs(results):
    fields=('boxes_3d_det','scores_3d_det','boxes_3d','scores_3d','track_scores','sdc_scores_3d',
            'mask_semantic_flat','mask_semantic_flat_gt','mask_query_panoptic')
    checked=0
    counts={name+suffix:0. for name in ('drivable','lanes','divider','crossing','contour')
            for suffix in ('_intersection','_union')}
    def array(value):
        value=value.tensor if hasattr(value,'tensor') else value
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    for row in results:
        if 'boxes_3d_det' not in row or 'scores_3d_det' not in row or 'ret_iou' not in row:
            raise RuntimeError('Missing actual detection or mapping prediction')
        values={key:row[key] for key in fields if key in row}
        values.update({'soft/'+key:value for key,value in row.get('soft',{}).items()})
        for key,value in values.items():
            if not np.isfinite(array(value)).all():raise RuntimeError('Nonfinite perception output: '+key)
            checked+=1
        for key in counts:
            if key not in row['ret_iou']:raise RuntimeError('Missing original mapping count: '+key)
            value=float(array(row['ret_iou'][key]).item())
            if not np.isfinite(value) or value<0:raise RuntimeError('Invalid mapping count: '+key)
            counts[key]+=value
        for name in ('drivable','lanes','divider','crossing','contour'):
            if float(array(row['ret_iou'][name+'_intersection']).item())>float(array(row['ret_iou'][name+'_union']).item()):
                raise RuntimeError('Mapping intersection exceeds union: '+name)
    if not checked:raise RuntimeError('Empty perception predictions')
    empty=[name for name in ('drivable','lanes','divider','crossing','contour') if counts[name+'_union']==0]
    iou={name:None if name in empty else counts[name+'_intersection']/counts[name+'_union']
         for name in ('drivable','lanes','divider','crossing','contour')}
    return dict(finite_prediction_fields=checked,mapping_counts=counts,mapping_iou=iou,empty_union_categories=empty)

