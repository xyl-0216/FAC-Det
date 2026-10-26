"""GT-free, fixed-width candidate expansion for frozen-reader collection/eval."""
import torch


def fixed_bank_candidates(semantic, local_to_global, fixed_ids, outside_topk=10):
    fixed = torch.as_tensor(fixed_ids, device=semantic.device, dtype=torch.long)
    if fixed.numel() != torch.unique(fixed).numel():
        raise ValueError('Duplicate fixed candidate IDs')
    matches = fixed[:, None] == local_to_global[None, :]
    if not bool((matches.sum(1) == 1).all()):
        raise ValueError('Fixed candidate bank requires every fixed class in the current prompt; use a full-class prompt')
    positions = matches.long().argmax(1)
    if not bool(torch.isfinite(semantic[:, positions]).all()):
        raise ValueError('Fixed candidate text is unavailable/invalid')
    outside = semantic.clone()
    outside[:, positions] = -torch.inf
    if int(torch.isfinite(outside).sum(1).min()) < outside_topk:
        raise ValueError('Insufficient outside-bank candidates')
    external = outside.topk(outside_topk, dim=1).indices
    selected = torch.cat([positions.expand(len(semantic), -1), external], 1)
    # Preserve the detector-top1 slot and semantic ordering expected downstream.
    values = semantic.gather(1, selected)
    order = values.argsort(dim=1, descending=True, stable=True)
    return local_to_global[selected.gather(1, order)]
