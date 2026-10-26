"""Trainable crop/text candidate selection with a separate reject logit."""
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'facb_local_selector_v1'


def selector_probabilities(model, crop, text, facets):
    """Use the same FP32 scoring precision as cached CPU fitting."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.autocast(device_type=crop.device.type,enabled=False):
            return model(crop.float(),text.float(),facets.float()).softmax(-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class LocalSelector(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.text = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.facet_gate = nn.Sequential(nn.Linear(dim+3, hidden), nn.GELU(), nn.Linear(hidden,3))
        self.reject = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, crop, text, facets):
        visual = F.normalize(self.visual(crop),dim=-1)
        words = F.normalize(self.text(text),dim=-1)
        logits = torch.einsum('nd,nkd->nk',visual,words)*self.log_scale.exp().clamp(1,100)
        evidence = facets.sigmoid()
        weight = self.facet_gate(torch.cat([crop[:,None].expand(-1,text.shape[1],-1),evidence],-1)).softmax(-1)
        logits = logits + (weight*evidence).sum(-1)
        return torch.cat([logits,self.reject(crop)],-1)


def load_selector(path, device):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema') != SCHEMA:
        raise ValueError('Not a local selector checkpoint')
    model = LocalSelector(state['dim'],state['hidden']).to(device)
    model.load_state_dict(state['model_state'],strict=True)
    model.eval()
    model.requires_grad_(False)
    return model,state
