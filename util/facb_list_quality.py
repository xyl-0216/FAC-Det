"""Training primitives for D0-anchored hypothesis quality/ranking.

GT never enters eligibility, features, or replay. Only accepted selected and
optional in-scope same hypotheses are mutable; old stays at its D0 score.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.facb_detection_replay import DenseDetections, replay_record
from util import facb_quality_features as d
from util.facb_quality_targets import accepted
from util.facb_quality_targets import coco_targets, ranking_pairs, logit

FEATURES = ['d0_logit', *d.FEATURES, 'neighbor_score', 'neighbor_iou', 'log_neighbor_count']


class QualityResidual(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(len(FEATURES), 1)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)

    def forward(self, x, base):
        delta = 2 * torch.tanh(self.linear(x).squeeze(-1))
        return base + delta, delta


def overlap(a, b):
    inter = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    union = max(0.,a[2]-a[0])*max(0.,a[3]-a[1]) + max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/max(union, 1e-12)


def make_context(dense, rows, ids, targets, threshold, include_same=False):
    ids = set(ids)
    assert {r['image_id'] for r in rows} == ids
    assert len({(r['image_id'],r['query_index']) for r in rows}) == len(rows)
    records = {i:dict(dense.records[i], scores=dense.records[i]['scores'].clone()) for i in sorted(ids)}
    maps = {i:{int(g):(j,int(records[i]['labels'][j])) for j,g in enumerate(records[i]['global_ids'])} for i in ids}
    accepted = [r for r in rows if accepted(r, targets, threshold)]
    for r in accepted:
        i,q = r['image_id'],r['query_index']; old=maps[i][r['old_global_id']][0]; new=maps[i][r['selected_global_id']][0]
        scores=records[i]['scores']; scores[q,old],scores[q,new]=scores[q,new].clone(),scores[q,old].clone()
    view = DenseDetections(records)
    hypotheses=[]
    for r in rows:
        change=accepted(r,targets,threshold)
        same=include_same and r['old_global_id']==r['selected_global_id'] and r['selected_global_id'] in targets
        if not change and not same: continue
        i,q=r['image_id'],r['query_index']; pos,category=maps[i][r['selected_global_id']]
        original=dense.records[i]['scores'][q]
        if int(original.argmax()) != maps[i][r['old_global_id']][0]:
            raise ValueError('OOF detector old-top drift')
        cx,cy,w,h=dense.records[i]['boxes'][q].unbind()
        height,width=dense.records[i]['size'].unbind()
        box=torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2))*torch.stack((width,height,width,height))
        if max(abs(a-b) for a,b in zip(box.tolist(),r['box']))>.001:
            raise ValueError('OOF crop box drift')
        if abs(float(original[maps[i][r['old_global_id']][0]])-r['old_score'])>1e-6:
            raise ValueError('OOF feature/detector drift')
        if abs(float(original[pos])-r['selected_original_score'])>1e-6:
            raise ValueError('OOF selected score drift')
        score=float(records[i]['scores'][q,pos])
        # Retained and mutable D0 neighbors both describe predicted competition.
        neighbors=[(overlap(r['box'],n['box']),n['score']) for n in view[i]
                   if n['query_index']!=q and n['category_id']==category]
        close=[s for v,s in neighbors if v>=.5]
        x=[logit(score), *d.row_feature(r), max(close,default=0.),
           max((v for v,s in neighbors),default=0.), math.log1p(len(close))]
        hypotheses.append(dict(key=(i,q,category), feature=x, base=score,
            role='selected' if change else 'same', tier=r['tier'], gt_id=r.get('gt_id'),
            bucket=r['bucket']))
    assert len({h['key'] for h in hypotheses})==len(hypotheses)
    return dict(view=view, hypotheses=hypotheses, ids=ids, threshold=threshold, include_same=include_same)


def predict(context, model, mean, std):
    hs=context['hypotheses']; view=context['view']
    if not hs or all(torch.count_nonzero(p).item()==0 for p in model.parameters()):
        return {i:copy.deepcopy(rs) for i,rs in view.items()}
    with torch.inference_mode():
        x=torch.tensor([h['feature'] for h in hs],dtype=torch.float32)
        base=torch.tensor([logit(h['base']) for h in hs],dtype=torch.float32)
        scores=model((x-mean)/std,base)[0].sigmoid().tolist()
    updates={i:[] for i in view}
    for h,s in zip(hs,scores):
        i,q,c=h['key']; updates[i].append((q,c,s))
    return {i:replay_record(r,updates[i]) for i,r in view.records.items()}


def supervision(coco, context, output, use_rank):
    hs=context['hypotheses']; lookup={h['key']:j for j,h in enumerate(hs)}
    flat=[]; indices=[]
    for i in sorted(context['ids']):
        for r in output[i]:
            index=lookup.get((i,r['query_index'],r['category_id']),-1)
            flat.append(dict(r,image_id=i,_mutable=index>=0)); indices.append(index)
    detail=coco_targets(coco,flat,context['ids'],return_details=True)
    targets=detail['quality']
    pairs=ranking_pairs(flat,targets,42,8) if use_rank else []
    if len(pairs)>20000:
        take=np.random.default_rng(42).choice(len(pairs),20000,replace=False)
        pairs=[pairs[j] for j in take]
    mutable=[j for j,k in enumerate(indices) if k>=0 and np.isfinite(targets[j])]
    classes={flat[j]['category_id'] for j in mutable}
    count={c:sum(flat[j]['category_id']==c for j in mutable) for c in classes}
    weights=torch.tensor([1/count[flat[j]['category_id']]/max(len(classes),1) for j in mutable],dtype=torch.float32)
    kinds=dict(mutable_rows=len(hs), evaluated_mutable=len(mutable),
        undefined_mutable=sum(k>=0 and not np.isfinite(t) for k,t in zip(indices,targets)),
        outside_list_mutable=len(hs)-sum(k>=0 for k in indices),
        positive_mutable=sum(targets[j]>0 for j in mutable),fp_mutable=sum(targets[j]==0 for j in mutable),
        ranking_pairs=len(pairs), fixed_reference_pairs=sum(not(flat[a]['_mutable'] and flat[b]['_mutable']) for a,b in pairs),
        same_image_cross_query_pairs=sum(flat[a]['image_id']==flat[b]['image_id'] and flat[a]['query_index']!=flat[b]['query_index'] for a,b in pairs))
    return dict(fixed=torch.tensor([logit(r['score']) for r in flat],dtype=torch.float32),
        output_positions=torch.tensor([j for j,k in enumerate(indices) if k>=0],dtype=torch.long),
        hypothesis_positions=torch.tensor([k for k in indices if k>=0],dtype=torch.long),
        supervised=torch.tensor(mutable,dtype=torch.long),
        quality=torch.tensor([targets[j] for j in mutable],dtype=torch.float32),weights=weights,
        pairs=torch.tensor(pairs,dtype=torch.long).reshape(-1,2),
        pair_weights=torch.tensor([targets[a]-targets[b] for a,b in pairs],dtype=torch.float32),
        summary={k:int(v) for k,v in kinds.items()})


def train(coco, context, steps, use_rank):
    torch.manual_seed(42); model=QualityResidual(); hs=context['hypotheses']
    x=torch.tensor([h['feature'] for h in hs],dtype=torch.float32).reshape(-1,len(FEATURES))
    mean=x.mean(0) if len(hs) else torch.zeros(len(FEATURES))
    std=x.std(0,unbiased=False).clamp_min(1e-5) if len(hs) else torch.ones(len(FEATURES))
    base=torch.tensor([logit(h['base']) for h in hs],dtype=torch.float32)
    snapshots={0:copy.deepcopy(model.state_dict())}; history=[]
    if not hs or not steps: return model,mean,std,snapshots,history
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    for step in range(1,steps+1):
        if (step-1)%20==0:
            target=supervision(coco,context,predict(context,model,mean,std),use_rank)
            history.append(dict(refresh_after_step=step-1,**target['summary']))
        logits,delta=model((x-mean)/std,base)
        full=target['fixed'].index_copy(0,target['output_positions'],logits[target['hypothesis_positions']])
        quality=(F.binary_cross_entropy_with_logits(full[target['supervised']],target['quality'],reduction='none')*target['weights']).sum()
        pair=target['pairs']
        rank=(F.softplus(full[pair[:,1]]-full[pair[:,0]])*target['pair_weights']).mean() if len(pair) else logits.sum()*0
        loss=quality+rank+.01*delta.square().mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step%20==0 or step==steps: snapshots[step]=copy.deepcopy(model.state_dict())
    return model,mean,std,snapshots,history


def coverage(context):
    groups={}
    for h in context['hypotheses']:
        group=groups.setdefault(h['tier'],{}).setdefault(h['role'],dict(rows=0,gt=set(),old_correct=0,old_correct_gt=set()))
        group['rows']+=1
        valid=h['gt_id'] is not None and h['gt_id']>=0
        if valid: group['gt'].add((h['key'][0],h['gt_id']))
        if h['bucket'].startswith('old_correct'):
            group['old_correct']+=1
            if valid: group['old_correct_gt'].add((h['key'][0],h['gt_id']))
    return {t:{r:dict(rows=v['rows'],unique_gt=len(v['gt']),old_correct_rows=v['old_correct'],
                     old_correct_unique_gt=len(v['old_correct_gt'])) for r,v in rs.items()} for t,rs in groups.items()}
