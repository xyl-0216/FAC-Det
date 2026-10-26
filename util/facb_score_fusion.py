"""F1: frozen-gate score mapping and exact standard postprocess replay."""
import hashlib
from pathlib import Path
import torch
from torchvision.ops import batched_nms

SCHEMA = 'facb_score_fusion_v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fused_score(q, old, abc):
    """Identity coefficients return old bit-for-bit, including boundary values."""
    if not isinstance(abc, torch.Tensor):
        abc = old.new_tensor(abc)
    if not abc.requires_grad and torch.equal(abc, abc.new_tensor([0.,1.,0.])):
        return old.clone()
    a,b,c = abc.unbind()
    return (a*torch.logit(q.clamp(1e-6,1-1e-6)) +
            b*torch.logit(old.clamp(1e-6,1-1e-6)) + c).sigmoid()


def load_fusion(path, gate_path, reader_id):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if obj.get('schema') != SCHEMA:
        raise ValueError('Unsupported F1 checkpoint')
    a,b,c = obj['abc']
    if not all(torch.isfinite(torch.tensor([a,b,c]))) or a<0 or not 0<=b<=1:
        raise ValueError('Invalid F1 coefficients')
    if obj['gate_sha256'] != sha256(gate_path) or obj['reader_id'] != reader_id:
        raise ValueError('F1 requires exactly the gate and reader used in collection')
    return obj


def capture_chunk(directory, index, scores, boxes, labels, sizes, outputs,
                  changes, num_select, nms, gate_sha, reader_id):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for batch in range(len(scores)):
        image_id = int(outputs['image_ids'][batch])
        record = dict(schema=SCHEMA,image_id=image_id,
            scores=scores[batch].detach().float().cpu(),
            boxes=boxes[batch].detach().float().cpu(),labels=labels.detach().cpu(),
            size=sizes[batch].detach().cpu(),num_select=int(num_select),nms=float(nms),
            changes=[v for v in changes if v['batch']==batch],
            gate_sha256=gate_sha,reader_id=reader_id,
            protocol=outputs.get('facb_speedhr_inference',{}).get('proposal_protocol'),
            image_path=(outputs.get('_facb_f1_image_paths') or [None]*len(scores))[batch])
        path=directory/f'{image_id}_{index:06d}_{batch}.pt'
        if path.exists():
            raise FileExistsError(f'F1 capture would overwrite {path}')
        torch.save(record,path)


def chunk_hypotheses(record, abc, pool=False):
    scores=record['scores'].clone()
    mutable=torch.zeros_like(scores,dtype=torch.bool)
    quality=torch.zeros_like(scores)
    anchor=torch.zeros_like(scores)
    for change in record['changes']:
        q,c=change['query'],change['selected']
        if mutable[q,c]: raise ValueError('Duplicate mutable query/class in F1 cache')
        mutable[q,c]=True
        quality[q,c]=change['quality']; anchor[q,c]=change['old_score']
        scores[q,c]=fused_score(scores.new_tensor(change['quality']),
                                scores.new_tensor(change['old_score']),abc)
    n=min(record['num_select'],scores.numel())
    if pool:
        fixed=scores.flatten().masked_fill(mutable.flatten(),-float('inf'))
        fixed_ids=torch.topk(fixed,min(n,int((~mutable).sum()))).indices
        ids=torch.cat([fixed_ids,mutable.flatten().nonzero().flatten()])
    else:
        ids=torch.topk(scores.flatten(),n).indices
    query=ids//scores.shape[1]; cls=ids%scores.shape[1]
    cx,cy,w,h=record['boxes'][query].unbind(-1)
    height,width=record['size']
    boxes=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)*torch.tensor([width,height,width,height])
    values=scores.flatten()[ids]; labels=record['labels'][cls]
    if not pool and record['nms']>0:
        keep=batched_nms(boxes,values,labels,record['nms'])
        ids,boxes,values,labels=ids[keep],boxes[keep],values[keep],labels[keep]
    result=[]
    for flat,box,score,label in zip(ids.tolist(),boxes.tolist(),values.tolist(),labels.tolist()):
        result.append(dict(image_id=record['image_id'],category_id=int(label),
            bbox=[box[0],box[1],box[2]-box[0],box[3]-box[1]],score=score,
            mutable=bool(mutable.flatten()[flat]),q=float(quality.flatten()[flat]),
            old=float(anchor.flatten()[flat]),key=f"{record['_name']}:{flat}"))
    return result


def replay(records, abc, pool=False):
    groups={}
    limits={}
    counts={}
    for r in records:
        groups.setdefault(r['image_id'],[]).extend(chunk_hypotheses(r,abc,pool))
        counts[r['image_id']]=counts.get(r['image_id'],0)+1
        if r['image_id'] in limits and limits[r['image_id']]!=r['num_select']:
            raise ValueError('Mixed num_select')
        limits[r['image_id']]=r['num_select']
    out=[]
    for image_id,rows in groups.items():
        if pool or counts[image_id]==1:
            out.extend(rows)
        else:
            values=torch.tensor([r['score'] for r in rows])
            ids=torch.topk(values,min(limits[image_id],len(rows))).indices.tolist()
            out.extend(rows[i] for i in ids)
    return out
