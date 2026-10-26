"""Detector-only candidates; duplicate padding must be masked in the reader."""
import torch


def unique_candidate_mask(ids):
    equal = ids[:, :, None] == ids[:, None, :]
    earlier = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).tril(-1)
    return ~(equal & earlier).any(-1)


def union_candidates(semantic, local_to_global, fixed_ids, outside_topk=10):
    """Return a padded fixed-pool/semantic-top-k union for every query.

    The padded width is derived from the supplied candidate set and top-k size.
    """
    fixed_ids = [int(value) for value in fixed_ids]
    outside_topk = min(max(int(outside_topk), 0), int(semantic.shape[1]))
    assert fixed_ids and len(set(fixed_ids)) == len(fixed_ids)
    assert len(local_to_global) == semantic.shape[1]
    assert len(local_to_global.unique()) == len(local_to_global)
    valid_ids = set(int(value) for value in local_to_global.tolist())
    assert set(fixed_ids).issubset(valid_ids)
    natural = semantic.argsort(dim=1, descending=True, stable=True)[:, :outside_topk]
    padded_width = len(fixed_ids) + outside_topk
    result = []
    for scores, top in zip(semantic, natural):
        ids = set(fixed_ids) | set(local_to_global[top].tolist())
        ordered = [int(local_to_global[i]) for i in scores.argsort(descending=True, stable=True).tolist() if int(local_to_global[i]) in ids]
        assert len(ordered) == len(ids) and bool(torch.isfinite(scores).all())
        result.append(ordered + [ordered[0]] * (padded_width-len(ordered)))
    return torch.tensor(result, device=semantic.device)
