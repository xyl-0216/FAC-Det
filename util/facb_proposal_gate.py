"""frozen-reader proposal gate. No detector or GT-dependent selection here."""
import numpy as np
import torch
from torch import nn
from PIL import Image

SCHEMA = "facb_proposal_pair_v1"


def pack_features(crop, query, selected, old, scalars):
    """Order is shared verbatim by online inference and cached-feature fitting."""
    parts = [crop, query, selected, old, selected - old, selected * old]
    if any(p.ndim != 1 or p.numel() != crop.numel() for p in parts):
        raise ValueError("requires equal-width crop/query/candidate states")
    scalars = crop.new_tensor(scalars)
    if scalars.numel() != 8:
        raise ValueError("expects eight scalar context values")
    result = torch.cat(parts + [scalars]).float()
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite representation")
    return result


class ProposalGate(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, learn_quality=False):
        super().__init__()
        self.state_dim = int(state_dim)
        self.validity = nn.Sequential(nn.Linear(2 * state_dim + 8, hidden_dim),
                                      nn.GELU(), nn.Linear(hidden_dim, 1))
        self.override = nn.Sequential(nn.Linear(6 * state_dim + 8, hidden_dim),
                                      nn.GELU(), nn.Linear(hidden_dim, 3))
        self.learn_quality = learn_quality
        if learn_quality:
            self.visual_projection = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            self.text_projection = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim))
            self.logit_scale = nn.Parameter(torch.tensor(2.3))
            self.attribute_gate = nn.Sequential(nn.Linear(state_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
            self.quality_head = nn.Sequential(nn.Linear(6 * state_dim + 8 + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        if x.shape[-1] != 6 * self.state_dim + 8:
            raise ValueError("feature width mismatch")
        validity_x = torch.cat([x[..., :2 * self.state_dim], x[..., -8:]], -1)
        return self.validity(validity_x).squeeze(-1), self.override(x)

    def quality(self, x, candidate_text, facet_logits, selected_slots):
        """Local contrastive alignment + sample-dependent attribute weighting.

        Text/crop features are frozen-reader features. Only these projections,
        the attribute gate and quality head are fitted in the first milestone.
        The contrastive probability is an input, never the final detection score.
        """
        if not self.learn_quality or facet_logits.shape[-1] != 3:
            raise ValueError('Quality requires group/shape/symbol evidence')
        crop = x[..., :self.state_dim]
        visual = torch.nn.functional.normalize(self.visual_projection(crop), dim=-1)
        text = torch.nn.functional.normalize(self.text_projection(candidate_text), dim=-1)
        match = torch.einsum('bd,bkd->bk', visual, text) * self.logit_scale.exp().clamp(1.,100.)
        attribute_probs = facet_logits.sigmoid()
        weights = self.attribute_gate(torch.cat([crop[:,None].expand(-1,candidate_text.shape[1],-1),attribute_probs],-1)).softmax(-1)
        attribute_match = (weights * attribute_probs).sum(-1)
        local_logits = match + attribute_match
        batch = torch.arange(x.shape[0],device=x.device)
        selected_local = local_logits.softmax(-1)[batch,selected_slots,None]
        selected_attributes = attribute_probs[batch,selected_slots]
        weighted_attribute = attribute_match[batch,selected_slots,None]
        quality_input = torch.cat([x,selected_local,selected_attributes,weighted_attribute],-1)
        return self.quality_head(quality_input).squeeze(-1),local_logits,weights


def stage_score(validity_prob, useful_prob, thresholds):
    """0.5 iff both stages pass; this is a decision score, not TP probability."""
    p = torch.stack([validity_prob, useful_prob], -1).clamp(1e-6, 1-1e-6)
    t = p.new_tensor(thresholds).clamp(1e-6, 1-1e-6)
    return (torch.logit(p) - torch.logit(t)).amin(-1).sigmoid()


def raw_predicted_crops(image_paths, batch_ids, boxes, scale, crop_size, device):
    """Eval-only normalized predicted boxes; caller must use resize-only geometry.

    GT annotations are deliberately absent from this API. Original pixels are
    cropped with PIL and normalized identically for collection and deployment.
    """
    images, crops = {}, []
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    for batch_id, box in zip(batch_ids.tolist(), boxes.detach().float().cpu()):
        path = image_paths[batch_id] if image_paths is not None else None
        if not path:
            raise ValueError("raw_image FACB inference requires an original image path")
        if batch_id not in images:
            with Image.open(path) as src:
                images[batch_id] = src.convert("RGB")
        img = images[batch_id]
        cx, cy, bw, bh = box.tolist()
        bw, bh = min(max(bw * scale, 1e-6), 1.), min(max(bh * scale, 1e-6), 1.)
        x1, y1 = max(0., min(1., cx-bw/2)), max(0., min(1., cy-bh/2))
        x2, y2 = max(0., min(1., cx+bw/2)), max(0., min(1., cy+bh/2))
        w, h = img.size
        crop = img.crop((x1*w, y1*h, max(x2*w, x1*w+1), max(y2*h, y1*h+1)))
        arr = np.array(crop.resize((crop_size, crop_size), resampling), dtype=np.float32)/255.
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        crops.append((tensor-tensor.new_tensor([.485,.456,.406])[:,None,None]) /
                     tensor.new_tensor([.229,.224,.225])[:,None,None])
    batch = torch.stack(crops).to(device)
    rois = batch.new_zeros((len(crops), 5))
    rois[:, 0] = torch.arange(len(crops), device=device)
    rois[:, 3:] = crop_size
    return batch, rois
