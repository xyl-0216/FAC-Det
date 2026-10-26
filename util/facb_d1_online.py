"""Evaluation-only D1 inference from frozen nested-fold weights; no GT inputs."""
import math
import torch
from torch.nn import functional as F


def score_pair(saved, old_score, selected_score, itm_probabilities, margin_old, margin_top):
    def logit(x):
        x = min(max(float(x), 1e-6), 1 - 1e-6)
        return math.log(x / (1 - x))
    feature = [logit(old_score), logit(selected_score),
               *[logit(p) for p in itm_probabilities], float(margin_old), float(margin_top)]
    state = saved['state_dict']
    if all(torch.count_nonzero(v).item() == 0 for v in state.values()):
        return dict(selected=float(old_score), old=float(selected_score), p=1.,
                    q=float(old_score), feature=feature)
    # The trained head is FP32/CPU; keep this tiny inference on the same backend.
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        x = (torch.tensor([feature], dtype=torch.float32) - saved['mean']) / saved['std']
        dq = F.linear(x, state['quality_delta.weight'], state['quality_delta.bias']).squeeze()
        p = (1 + F.linear(x, state['takeover_delta.weight'], state['takeover_delta.bias']).squeeze()).clamp(0, 1)
        sold = torch.tensor(float(old_score), dtype=torch.float32)
        original = torch.tensor(float(selected_score), dtype=torch.float32)
        q = torch.sigmoid(torch.logit(sold.clamp(1e-6, 1-1e-6)) + dq)
        selected = (1-p)*original + p*q
        old = (1-p)*sold + p*original
    return dict(selected=float(selected), old=float(old), p=float(p), q=float(q), feature=feature)


def validate_score(saved):
    expected = {'quality_delta.weight': (1, 7), 'quality_delta.bias': (1,),
                'takeover_delta.weight': (1, 7), 'takeover_delta.bias': (1,)}
    if set(saved['state_dict']) != set(expected):
        raise ValueError('Unsupported D1 weights')
    for name, shape in expected.items():
        value = saved['state_dict'][name]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('Invalid D1 weights')
    for name in ['mean', 'std']:
        if saved[name].shape != (7,) or not torch.isfinite(saved[name]).all():
            raise ValueError('Invalid D1 normalization')
    if (saved['std'] <= 0).any():
        raise ValueError('Invalid D1 std')
    return saved
