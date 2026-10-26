import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from groundingdino.util.misc import is_dist_avail_and_initialized


def build_category_text_prototypes(text_features, category_token_masks, text_token_mask=None):
    """Pool token-level text features into category-level prototypes."""
    category_prototypes = []
    prototype_valid_masks = []
    text_length = text_features.shape[1]

    for batch_idx, category_mask in enumerate(category_token_masks):
        category_mask = category_mask[:, :text_length]
        if text_token_mask is not None:
            category_mask = category_mask & text_token_mask[batch_idx].unsqueeze(0)

        weights = category_mask.float()
        token_counts = weights.sum(dim=-1, keepdim=True)
        normalized_weights = weights / token_counts.clamp(min=1.0)
        prototypes = normalized_weights @ text_features[batch_idx]

        category_prototypes.append(prototypes)
        prototype_valid_masks.append(token_counts.squeeze(-1) > 0)

    return category_prototypes, prototype_valid_masks


class AppearanceGuidedGroupRegularizer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_groups=3,
        ema_momentum=0.9,
        use_ema=True,
        consistency_loss=False,
        positive_only=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.ema_momentum = ema_momentum
        self.use_ema = use_ema
        self.consistency_loss = consistency_loss
        self.positive_only = positive_only

        self.register_buffer("ema_anchors", torch.zeros(num_groups, hidden_dim))
        self.register_buffer("ema_initialized", torch.zeros(num_groups, dtype=torch.bool))

    def _empty_loss(self):
        return self.ema_anchors.new_tensor(0.0)

    def _reduce_group_statistics(self, group_sums, group_counts):
        if not is_dist_avail_and_initialized():
            return group_sums, group_counts
        dist.all_reduce(group_sums)
        dist.all_reduce(group_counts)
        return group_sums, group_counts

    @torch.no_grad()
    def _update_ema(self, batch_anchors, batch_valid):
        if not self.use_ema:
            return
        for group_idx in range(self.num_groups):
            if not batch_valid[group_idx]:
                continue
            anchor = batch_anchors[group_idx].detach()
            if self.ema_initialized[group_idx]:
                self.ema_anchors[group_idx].mul_(self.ema_momentum).add_(
                    anchor, alpha=1.0 - self.ema_momentum
                )
            else:
                self.ema_anchors[group_idx].copy_(anchor)
                self.ema_initialized[group_idx] = True

    def forward(
        self,
        category_prototypes,
        category_group_ids,
        positive_label_indices,
        prototype_valid_masks=None,
    ):
        if not category_prototypes:
            zero = self._empty_loss()
            return {"loss_aggr_orth": zero}

        device = category_prototypes[0].device
        group_sums = torch.zeros(self.num_groups, self.hidden_dim, device=device)
        group_counts = torch.zeros(self.num_groups, device=device)

        for sample_idx, prototypes in enumerate(category_prototypes):
            if prototypes.numel() == 0:
                continue
            group_ids = category_group_ids[sample_idx].to(device)
            valid_mask = None
            if prototype_valid_masks is not None:
                valid_mask = prototype_valid_masks[sample_idx].to(device)

            if self.positive_only:
                selected = torch.unique(positive_label_indices[sample_idx].to(device))
            else:
                selected = torch.arange(prototypes.shape[0], device=device)

            selected = selected[selected >= 0]
            selected = selected[selected < prototypes.shape[0]]
            selected = selected[selected < group_ids.shape[0]]
            if valid_mask is not None:
                selected = selected[valid_mask[selected]]
            if selected.numel() == 0:
                continue

            selected_groups = group_ids[selected]
            selected_prototypes = prototypes[selected]

            for group_idx in range(self.num_groups):
                group_mask = selected_groups == group_idx
                if not group_mask.any():
                    continue
                group_sums[group_idx] += selected_prototypes[group_mask].sum(dim=0)
                group_counts[group_idx] += group_mask.sum()

        group_sums, group_counts = self._reduce_group_statistics(group_sums, group_counts)
        batch_valid = group_counts > 0
        batch_anchors = group_sums / group_counts.clamp(min=1.0).unsqueeze(-1)
        batch_anchors = F.normalize(batch_anchors, dim=-1, eps=1e-6)

        ema_reference = self.ema_anchors.detach().clone()
        ema_valid = self.ema_initialized.detach().clone()
        effective_anchors = batch_anchors.clone()
        fallback_mask = (~batch_valid) & ema_valid
        if fallback_mask.any():
            effective_anchors[fallback_mask] = F.normalize(
                ema_reference[fallback_mask], dim=-1, eps=1e-6
            )

        active_mask = batch_valid | fallback_mask
        if active_mask.sum() < 2:
            losses = {"loss_aggr_orth": self._empty_loss()}
        else:
            active_anchors = effective_anchors[active_mask]
            gram = active_anchors @ active_anchors.t()
            identity = torch.eye(active_anchors.shape[0], device=device)
            losses = {"loss_aggr_orth": (gram - identity).pow(2).mean()}

        if self.consistency_loss:
            consistency_mask = batch_valid & ema_valid
            if consistency_mask.any():
                losses["loss_aggr_consistency"] = F.mse_loss(
                    batch_anchors[consistency_mask],
                    F.normalize(ema_reference[consistency_mask], dim=-1),
                )
            else:
                losses["loss_aggr_consistency"] = self._empty_loss()

        self._update_ema(batch_anchors, batch_valid)
        return losses
