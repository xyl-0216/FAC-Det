"""GT-free fixed-quality inference, applied to the live matrix before top-k."""
import torch
from util.facb_list_quality import FEATURES, QualityResidual, make_context, logit
from util.facb_detection_replay import DenseDetections


def validate(saved):
    if saved['features']!=FEATURES or not saved['include_same']:
        raise ValueError('Unsupported fixed-quality feature schema')
    model=QualityResidual(); model.load_state_dict(saved['state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in saved['state_dict'].values()):
        raise ValueError('Nonfinite fixed-quality weights')
    for k in ['mean','std']:
        if saved[k].shape!=(len(FEATURES),) or not torch.isfinite(saved[k]).all():
            raise ValueError('Invalid fixed-quality normalization')
    if (saved['std']<=0).any(): raise ValueError('Invalid std')
    return saved


def audit_rows(records, audits):
    rows=[]
    for a in audits:
        i,q=a['image_id'],a['query_index']; r=records[i]
        cx,cy,w,h=r['boxes'][q].unbind(); height,width=r['size'].unbind()
        box=torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2))*torch.stack((width,height,width,height))
        rows.append(dict(image_id=i,query_index=q,box=box.tolist(),
            old_global_id=a['old_top_global_id'],selected_global_id=a['p0_selected_global_id'],
            **{k:a[k] for k in ['old_score','selected_original_score','itm_class_probability',
                'itm_shape_probability','itm_symbol_probability','fused_margin_over_old','fused_top1_margin']},
            tier='unassigned',bucket='inference',gt_id=None))
    return rows


def apply(saved, records, audits, targets, current):
    """current is the actual live D0 score tensor, batch order equals records."""
    validate(saved)
    if not audits:
        return current
    dense=DenseDetections(records)
    ctx=make_context(dense,audit_rows(records,audits),set(records),set(targets),saved['threshold'],True)
    for b,i in enumerate(records):
        if not torch.equal(ctx['view'].records[i]['scores'],current[b].detach().cpu()):
            raise ValueError('Actual D0 writer differs from fixed-quality reference')
    hs=ctx['hypotheses']; model=QualityResidual(); model.load_state_dict(saved['state_dict'])
    if not hs or all(torch.count_nonzero(p).item()==0 for p in model.parameters()): return current
    with torch.inference_mode(), torch.autocast(device_type='cpu',enabled=False):
        x=torch.tensor([h['feature'] for h in hs],dtype=torch.float32)
        base=torch.tensor([logit(h['base']) for h in hs],dtype=torch.float32)
        values=model((x-saved['mean'])/saved['std'],base)[0].sigmoid().tolist()
    result=current.clone(); batches={i:b for b,i in enumerate(records)}
    positions={i:{int(c):j for j,c in enumerate(r['labels'])} for i,r in records.items()}
    for h,v in zip(hs,values):
        i,q,c=h['key']; result[batches[i],q,positions[i][c]]=v
    return result
