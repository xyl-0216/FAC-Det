"""Fixed intrinsic box/class targets plus list ranking and D0 protection.

Training GT labels every eligible hypothesis independently of output truncation.
Intrinsic positives can include duplicates; only list ranking defines TP ownership.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from util.facb_list_quality import (FEATURES, QualityResidual, overlap, predict,
                                    supervision, logit)


def intrinsic_targets(coco, context):
    values=[]
    for h in context['hypotheses']:
        i,q,c=h['key']; r=context['view'].records[i]
        cx,cy,w,height=r['boxes'][q].tolist(); ih,iw=r['size'].tolist()
        box=[(cx-w/2)*iw,(cy-height/2)*ih,(cx+w/2)*iw,(cy+height/2)*ih]
        regular=[]; ignored=[]
        for a in coco.imgToAnns.get(i,[]):
            if a['category_id']!=c: continue
            x,y,bw,bh=a['bbox']; gt=[x,y,x+bw,y+bh]
            if a.get('iscrowd',0):
                inter=max(0,min(box[2],gt[2])-max(box[0],gt[0]))*max(0,min(box[3],gt[3])-max(box[1],gt[1]))
                ignored.append(inter/max((box[2]-box[0])*(box[3]-box[1]),1e-12))
            elif a.get('ignore',0): ignored.append(overlap(box,gt))
            else: regular.append(overlap(box,gt))
        # A regular same-class match wins over an ignored region at each IoU.
        labels=[1. if max(regular,default=0)>=t-1e-7 else
                np.nan if max(ignored,default=0)>=t-1e-7 else 0.
                for t in np.linspace(.5,.95,10)]
        valid=[v for v in labels if np.isfinite(v)]
        values.append(float(np.mean(valid)) if valid else np.nan)
    indices=[j for j,v in enumerate(values) if np.isfinite(v)]
    classes=[context['hypotheses'][j]['key'][2] for j in indices]
    counts={c:classes.count(c) for c in set(classes)}
    return dict(indices=torch.tensor(indices,dtype=torch.long),
        values=torch.tensor([values[j] for j in indices],dtype=torch.float32),
        weights=torch.tensor([1/counts[c]/max(len(counts),1) for c in classes],dtype=torch.float32),
        summary=dict(fixed_rows=len(values),fixed_valid=len(indices),
            fixed_positive=sum(v>0 for v in values if np.isfinite(v)),
            fixed_zero=sum(v==0 for v in values if np.isfinite(v)),fixed_ignored=len(values)-len(indices)))


def preservation(logits, baseline):
    """Hinge on baseline TP scores and initially positive TP/lower-quality gaps.

    Endpoints may be fixed references. This is a training penalty, not a guarantee.
    """
    full=baseline['fixed'].index_copy(0,baseline['output_positions'],logits[baseline['hypothesis_positions']])
    positive=baseline['supervised'][baseline['quality']>0]
    score=F.relu(baseline['fixed'][positive]-full[positive]).square().mean() if len(positive) else logits.sum()*0
    pair=baseline['pairs']
    if len(pair):
        gap=baseline['fixed'][pair[:,0]]-baseline['fixed'][pair[:,1]]
        active=gap>0
        current=full[pair[:,0]]-full[pair[:,1]]
        rank=F.relu(gap[active]-current[active]).square().mean() if active.any() else logits.sum()*0
    else: rank=logits.sum()*0
    return score+rank


def train(coco, context, steps, protect):
    torch.manual_seed(42); model=QualityResidual(); hs=context['hypotheses']
    x=torch.tensor([h['feature'] for h in hs],dtype=torch.float32).reshape(-1,len(FEATURES))
    mean=x.mean(0) if hs else torch.zeros(len(FEATURES))
    std=x.std(0,unbiased=False).clamp_min(1e-5) if hs else torch.ones(len(FEATURES))
    base=torch.tensor([logit(h['base']) for h in hs],dtype=torch.float32)
    snapshots={0:copy.deepcopy(model.state_dict())}; history=[]
    if not hs or not steps: return model,mean,std,snapshots,history
    fixed=intrinsic_targets(coco,context)
    baseline=supervision(coco,context,context['view'],True) if protect else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    for step in range(1,steps+1):
        if (step-1)%20==0:
            target=supervision(coco,context,predict(context,model,mean,std),True)
            history.append(dict(refresh_after_step=step-1,**target['summary'],**fixed['summary'],
                protected_tp_rows=int((baseline['quality']>0).sum()) if protect else 0))
        logits,delta=model((x-mean)/std,base)
        quality=(F.binary_cross_entropy_with_logits(logits[fixed['indices']],fixed['values'],reduction='none')*fixed['weights']).sum()
        full=target['fixed'].index_copy(0,target['output_positions'],logits[target['hypothesis_positions']])
        pair=target['pairs']
        rank=(F.softplus(full[pair[:,1]]-full[pair[:,0]])*target['pair_weights']).mean() if len(pair) else logits.sum()*0
        guard=preservation(logits,baseline) if protect else logits.sum()*0
        loss=quality+rank+guard+.01*delta.square().mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step%20==0 or step==steps: snapshots[step]=copy.deepcopy(model.state_dict())
    return model,mean,std,snapshots,history
