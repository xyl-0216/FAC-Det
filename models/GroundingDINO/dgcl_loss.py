"""Loss utilities for Dynamic Gated Contrastive Learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cxcywh_to_xyxy(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _aligned_iou(boxes1, boxes2):
    boxes1 = _cxcywh_to_xyxy(boxes1)
    boxes2 = _cxcywh_to_xyxy(boxes2)
    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0).prod(dim=-1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-6)


def matched_category_contrastive_loss(
    token_logits,
    pred_boxes,
    label_map_list,
    targets,
    indices,
    *,
    baseline_token_logits=None,
    margin=0.2,
    topk=5,
    iou_threshold=0.5,
    ce_weight=1.0,
    rank_weight=0.5,
    hard_only=False,
    baseline_margin_threshold=0.5,
    preserve_weight=0.0,
    class_weights=None,
):
    """Rank each matched ground-truth category above prompted hard negatives."""
    zero = token_logits.new_zeros((), dtype=torch.float32)
    ce_terms = []
    rank_terms = []
    preserve_terms = []
    valid_rows = 0
    matched_rows = 0
    preserved_rows = 0
    weight_means = []

    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        label_map = label_map_list[batch_idx].to(
            device=token_logits.device, dtype=torch.float32
        )
        if label_map.numel() == 0 or label_map.shape[0] < 2:
            continue
        label_map = label_map / label_map.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # Invalid/padded positions are outside every category span. Replacing
        # infinities avoids (-inf * 0) -> nan during category aggregation.
        sample_token_logits = torch.nan_to_num(
            token_logits[batch_idx].float(),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        category_scores = sample_token_logits @ label_map.transpose(0, 1)
        matched_scores = category_scores[src_idx]
        baseline_scores = None
        if baseline_token_logits is not None:
            sample_baseline_logits = torch.nan_to_num(
                baseline_token_logits[batch_idx].detach().float(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            baseline_scores = (
                sample_baseline_logits @ label_map.transpose(0, 1)
            )[src_idx]
        target_labels = targets[batch_idx]["labels"][tgt_idx].to(
            device=matched_scores.device, dtype=torch.long
        )
        row_weights = None

        valid = (target_labels >= 0) & (
            target_labels < matched_scores.shape[-1]
        )
        if iou_threshold > 0.0 and valid.any():
            matched_iou = _aligned_iou(
                pred_boxes[batch_idx, src_idx].float(),
                targets[batch_idx]["boxes"][tgt_idx].to(
                    device=pred_boxes.device, dtype=torch.float32
                ),
            )
            valid = valid & torch.isfinite(matched_iou) & (
                matched_iou >= iou_threshold
            )

        label_loss_mask = targets[batch_idx].get("caption_label_loss_mask")
        if label_loss_mask is not None:
            label_loss_mask = label_loss_mask.to(
                device=matched_scores.device, dtype=torch.bool
            ).view(-1)
            if label_loss_mask.numel() == matched_scores.shape[-1]:
                valid = valid & label_loss_mask[
                    target_labels.clamp(
                        min=0, max=matched_scores.shape[-1] - 1
                    )
                ]
                matched_scores = matched_scores.masked_fill(
                    ~label_loss_mask.unsqueeze(0), -1e4
                )
                if baseline_scores is not None:
                    baseline_scores = baseline_scores.masked_fill(
                        ~label_loss_mask.unsqueeze(0), -1e4
                    )

        if not valid.any():
            continue
        matched_scores = matched_scores[valid]
        target_labels = target_labels[valid]
        if class_weights is not None:
            global_target_labels = target_labels
            caption_label_ids = targets[batch_idx].get("caption_label_ids")
            if caption_label_ids is not None:
                try:
                    caption_label_ids = [
                        int(label_id) for label_id in caption_label_ids
                    ]
                    caption_label_ids = torch.as_tensor(
                        caption_label_ids,
                        device=matched_scores.device,
                        dtype=torch.long,
                    )
                except (TypeError, ValueError):
                    caption_label_ids = None
                if (
                    caption_label_ids is not None
                    and caption_label_ids.numel() == matched_scores.shape[-1]
                ):
                    global_target_labels = caption_label_ids[
                        target_labels.clamp(
                            min=0, max=caption_label_ids.numel() - 1
                        )
                    ]
            sample_class_weights = class_weights.to(
                device=matched_scores.device, dtype=torch.float32
            )
            in_range = (global_target_labels >= 0) & (
                global_target_labels < sample_class_weights.numel()
            )
            row_weights = torch.ones_like(
                target_labels, dtype=torch.float32
            )
            if in_range.any():
                row_weights[in_range] = sample_class_weights[
                    global_target_labels[in_range]
                ]
        if baseline_scores is not None:
            baseline_scores = baseline_scores[valid]
        finite = torch.isfinite(matched_scores).all(dim=-1)
        if baseline_scores is not None:
            finite = finite & torch.isfinite(baseline_scores).all(dim=-1)
        if not finite.any():
            continue
        matched_scores = matched_scores[finite]
        target_labels = target_labels[finite]
        if row_weights is not None:
            row_weights = row_weights[finite]
        if baseline_scores is not None:
            baseline_scores = baseline_scores[finite]
        matched_rows += int(matched_scores.shape[0])

        if hard_only and baseline_scores is not None:
            baseline_positive = baseline_scores.gather(
                1, target_labels.unsqueeze(1)
            ).squeeze(1)
            baseline_negative = baseline_scores.clone()
            baseline_negative.scatter_(
                1, target_labels.unsqueeze(1), -1e4
            )
            baseline_hardest = baseline_negative.max(dim=-1).values
            baseline_prediction = baseline_scores.argmax(dim=-1)
            hard_mask = (
                baseline_prediction != target_labels
            ) | (
                baseline_positive - baseline_hardest
                < float(baseline_margin_threshold)
            )
            easy_mask = ~hard_mask
            if preserve_weight > 0.0 and easy_mask.any():
                preserve_terms.append(
                    F.smooth_l1_loss(
                        matched_scores[easy_mask],
                        baseline_scores[easy_mask],
                    )
                )
                preserved_rows += int(easy_mask.sum().item())
            if not hard_mask.any():
                continue
            matched_scores = matched_scores[hard_mask]
            target_labels = target_labels[hard_mask]
            if row_weights is not None:
                row_weights = row_weights[hard_mask]

        if row_weights is None:
            row_weights = torch.ones_like(target_labels, dtype=torch.float32)
        row_weights = row_weights.clamp_min(0.0)
        row_weight_den = row_weights.sum().clamp_min(1e-6)
        weight_means.append(row_weights.detach().mean())
        ce_terms.append(
            (
                F.cross_entropy(
                    matched_scores, target_labels, reduction="none"
                )
                * row_weights
            ).sum()
            / row_weight_den
        )
        positive_scores = matched_scores.gather(
            1, target_labels.unsqueeze(1)
        ).squeeze(1)
        negative_scores = matched_scores.clone()
        negative_scores.scatter_(1, target_labels.unsqueeze(1), -1e4)
        hard_negative_count = min(
            max(int(topk), 1), negative_scores.shape[-1] - 1
        )
        hard_negative_scores = negative_scores.topk(
            k=hard_negative_count, dim=-1
        ).values
        rank_terms.append(
            (
                F.softplus(
                    hard_negative_scores
                    - positive_scores.unsqueeze(1)
                    + float(margin)
                ).mean(dim=-1)
                * row_weights
            ).sum()
            / row_weight_den
        )
        valid_rows += int(matched_scores.shape[0])

    if not ce_terms and not preserve_terms:
        return zero, {
            "valid_rows": 0,
            "matched_rows": matched_rows,
            "preserved_rows": 0,
        }
    ce_loss = torch.stack(ce_terms).mean() if ce_terms else zero
    rank_loss = torch.stack(rank_terms).mean() if rank_terms else zero
    preserve_loss = (
        torch.stack(preserve_terms).mean() if preserve_terms else zero
    )
    result = (
        float(ce_weight) * ce_loss
        + float(rank_weight) * rank_loss
        + float(preserve_weight) * preserve_loss
    )
    if not torch.isfinite(result):
        return zero, {
            "valid_rows": valid_rows,
            "matched_rows": matched_rows,
            "preserved_rows": preserved_rows,
        }
    return result, {
        "valid_rows": valid_rows,
        "matched_rows": matched_rows,
        "preserved_rows": preserved_rows,
        "ce": ce_loss.detach(),
        "rank": rank_loss.detach(),
        "preserve": preserve_loss.detach(),
        "weight_mean": (
            torch.stack(weight_means).mean().detach()
            if weight_means
            else zero.detach()
        ),
    }


def matched_category_logits_contrastive_loss(
    category_logits_list,
    pred_boxes,
    targets,
    indices,
    *,
    label_map_list=None,
    baseline_token_logits=None,
    margin=0.2,
    topk=5,
    iou_threshold=0.5,
    ce_weight=1.0,
    rank_weight=0.5,
    hard_only=False,
    baseline_margin_threshold=0.5,
    preserve_weight=0.0,
    class_weights=None,
    zero_like=None,
    use_inference_fused_logits=False,
    category_residual_weight=1.0,
    category_residual_logit_space=False,
    category_residual_match_baseline_distribution=False,
    error_targeted=False,
    preserve_correct=False,
):
    """Rank matched categories using DGCL category logits directly.

    This variant closes the training/inference loop for category residual
    inference: the same category logits used by PostProcess receive a direct
    CE/ranking signal, instead of being supervised only through token-level
    broadcast logits.
    """
    if zero_like is None:
        first_tensor = None
        if isinstance(category_logits_list, torch.Tensor):
            first_tensor = category_logits_list
        elif isinstance(category_logits_list, (list, tuple)):
            for value in category_logits_list:
                if isinstance(value, torch.Tensor):
                    first_tensor = value
                    break
        if first_tensor is None:
            raise ValueError("category_logits_list must contain a tensor")
        zero = first_tensor.new_zeros((), dtype=torch.float32)
    else:
        zero = zero_like.new_zeros((), dtype=torch.float32)

    ce_terms = []
    rank_terms = []
    preserve_terms = []
    valid_rows = 0
    matched_rows = 0
    preserved_rows = 0
    weight_means = []

    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        if isinstance(category_logits_list, torch.Tensor):
            if batch_idx >= category_logits_list.shape[0]:
                continue
            sample_category_logits = category_logits_list[batch_idx]
        elif isinstance(category_logits_list, (list, tuple)):
            if batch_idx >= len(category_logits_list):
                continue
            sample_category_logits = category_logits_list[batch_idx]
        else:
            continue
        if not isinstance(sample_category_logits, torch.Tensor):
            continue
        if sample_category_logits.numel() == 0 or sample_category_logits.dim() != 2:
            continue

        sample_category_logits = torch.nan_to_num(
            sample_category_logits.float(),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        score_logits = sample_category_logits
        baseline_scores = None
        if baseline_token_logits is not None and label_map_list is not None:
            label_map = label_map_list[batch_idx].to(
                device=sample_category_logits.device, dtype=torch.float32
            )
            if (
                label_map.numel() > 0
                and label_map.shape[0] == sample_category_logits.shape[-1]
            ):
                label_map = label_map / label_map.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0)
                sample_baseline_logits = torch.nan_to_num(
                    baseline_token_logits[batch_idx].detach().float(),
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
                if (
                    use_inference_fused_logits
                    and category_residual_logit_space
                ):
                    baseline_prob = (
                        sample_baseline_logits.sigmoid()
                        @ label_map.transpose(0, 1)
                    ).clamp(min=1e-6, max=1.0 - 1e-6)
                    baseline_scores_all = torch.logit(baseline_prob)
                    calibrated_logits = sample_category_logits
                    if category_residual_match_baseline_distribution:
                        src = sample_category_logits.reshape(-1)
                        ref = baseline_scores_all.reshape(-1)
                        src_std = src.std(unbiased=False).clamp_min(1e-6)
                        ref_std = ref.std(unbiased=False).clamp_min(1e-6)
                        calibrated_logits = (
                            (sample_category_logits - src.mean())
                            / src_std
                            * ref_std
                            + ref.mean()
                        )
                    score_logits = (
                        baseline_scores_all
                        + float(category_residual_weight)
                        * (calibrated_logits - baseline_scores_all.detach())
                    )
                    baseline_scores = baseline_scores_all[src_idx]
                else:
                    baseline_scores_all = (
                        sample_baseline_logits @ label_map.transpose(0, 1)
                    )
                    baseline_scores = baseline_scores_all[src_idx]

        matched_scores = score_logits[src_idx]

        target_labels = targets[batch_idx]["labels"][tgt_idx].to(
            device=matched_scores.device, dtype=torch.long
        )
        row_weights = None

        valid = (target_labels >= 0) & (
            target_labels < matched_scores.shape[-1]
        )
        if iou_threshold > 0.0 and valid.any():
            matched_iou = _aligned_iou(
                pred_boxes[batch_idx, src_idx].float(),
                targets[batch_idx]["boxes"][tgt_idx].to(
                    device=pred_boxes.device, dtype=torch.float32
                ),
            )
            valid = valid & torch.isfinite(matched_iou) & (
                matched_iou >= iou_threshold
            )

        label_loss_mask = targets[batch_idx].get("caption_label_loss_mask")
        if label_loss_mask is not None:
            label_loss_mask = label_loss_mask.to(
                device=matched_scores.device, dtype=torch.bool
            ).view(-1)
            if label_loss_mask.numel() == matched_scores.shape[-1]:
                valid = valid & label_loss_mask[
                    target_labels.clamp(
                        min=0, max=matched_scores.shape[-1] - 1
                    )
                ]
                matched_scores = matched_scores.masked_fill(
                    ~label_loss_mask.unsqueeze(0), -1e4
                )
                if baseline_scores is not None:
                    baseline_scores = baseline_scores.masked_fill(
                        ~label_loss_mask.unsqueeze(0), -1e4
                    )

        if not valid.any():
            continue
        matched_scores = matched_scores[valid]
        target_labels = target_labels[valid]
        if class_weights is not None:
            global_target_labels = target_labels
            caption_label_ids = targets[batch_idx].get("caption_label_ids")
            if caption_label_ids is not None:
                try:
                    caption_label_ids = [
                        int(label_id) for label_id in caption_label_ids
                    ]
                    caption_label_ids = torch.as_tensor(
                        caption_label_ids,
                        device=matched_scores.device,
                        dtype=torch.long,
                    )
                except (TypeError, ValueError):
                    caption_label_ids = None
                if (
                    caption_label_ids is not None
                    and caption_label_ids.numel() == matched_scores.shape[-1]
                ):
                    global_target_labels = caption_label_ids[
                        target_labels.clamp(
                            min=0, max=caption_label_ids.numel() - 1
                        )
                    ]
            sample_class_weights = class_weights.to(
                device=matched_scores.device, dtype=torch.float32
            )
            in_range = (global_target_labels >= 0) & (
                global_target_labels < sample_class_weights.numel()
            )
            row_weights = torch.ones_like(
                target_labels, dtype=torch.float32
            )
            if in_range.any():
                row_weights[in_range] = sample_class_weights[
                    global_target_labels[in_range]
                ]
        if baseline_scores is not None:
            baseline_scores = baseline_scores[valid]
        finite = torch.isfinite(matched_scores).all(dim=-1)
        if baseline_scores is not None:
            finite = finite & torch.isfinite(baseline_scores).all(dim=-1)
        if not finite.any():
            continue
        matched_scores = matched_scores[finite]
        target_labels = target_labels[finite]
        if row_weights is not None:
            row_weights = row_weights[finite]
        if baseline_scores is not None:
            baseline_scores = baseline_scores[finite]
        matched_rows += int(matched_scores.shape[0])

        if error_targeted and baseline_scores is not None:
            baseline_prediction = baseline_scores.argmax(dim=-1)
            error_mask = baseline_prediction != target_labels
            correct_mask = ~error_mask
            if preserve_correct and preserve_weight > 0.0 and correct_mask.any():
                preserve_terms.append(
                    F.smooth_l1_loss(
                        matched_scores[correct_mask],
                        baseline_scores[correct_mask],
                    )
                )
                preserved_rows += int(correct_mask.sum().item())
            if not error_mask.any():
                continue
            matched_scores = matched_scores[error_mask]
            target_labels = target_labels[error_mask]
            if row_weights is not None:
                row_weights = row_weights[error_mask]

        if hard_only and baseline_scores is not None:
            baseline_positive = baseline_scores.gather(
                1, target_labels.unsqueeze(1)
            ).squeeze(1)
            baseline_negative = baseline_scores.clone()
            baseline_negative.scatter_(
                1, target_labels.unsqueeze(1), -1e4
            )
            baseline_hardest = baseline_negative.max(dim=-1).values
            baseline_prediction = baseline_scores.argmax(dim=-1)
            hard_mask = (
                baseline_prediction != target_labels
            ) | (
                baseline_positive - baseline_hardest
                < float(baseline_margin_threshold)
            )
            easy_mask = ~hard_mask
            if preserve_weight > 0.0 and easy_mask.any():
                preserve_terms.append(
                    F.smooth_l1_loss(
                        matched_scores[easy_mask],
                        baseline_scores[easy_mask],
                    )
                )
                preserved_rows += int(easy_mask.sum().item())
            if not hard_mask.any():
                continue
            matched_scores = matched_scores[hard_mask]
            target_labels = target_labels[hard_mask]
            if row_weights is not None:
                row_weights = row_weights[hard_mask]

        if row_weights is None:
            row_weights = torch.ones_like(target_labels, dtype=torch.float32)
        row_weights = row_weights.clamp_min(0.0)
        row_weight_den = row_weights.sum().clamp_min(1e-6)
        weight_means.append(row_weights.detach().mean())
        ce_terms.append(
            (
                F.cross_entropy(
                    matched_scores, target_labels, reduction="none"
                )
                * row_weights
            ).sum()
            / row_weight_den
        )
        positive_scores = matched_scores.gather(
            1, target_labels.unsqueeze(1)
        ).squeeze(1)
        negative_scores = matched_scores.clone()
        negative_scores.scatter_(1, target_labels.unsqueeze(1), -1e4)
        hard_negative_count = min(
            max(int(topk), 1), negative_scores.shape[-1] - 1
        )
        hard_negative_scores = negative_scores.topk(
            k=hard_negative_count, dim=-1
        ).values
        rank_terms.append(
            (
                F.softplus(
                    hard_negative_scores
                    - positive_scores.unsqueeze(1)
                    + float(margin)
                ).mean(dim=-1)
                * row_weights
            ).sum()
            / row_weight_den
        )
        valid_rows += int(matched_scores.shape[0])

    if not ce_terms and not preserve_terms:
        return zero, {
            "valid_rows": 0,
            "matched_rows": matched_rows,
            "preserved_rows": 0,
        }
    ce_loss = torch.stack(ce_terms).mean() if ce_terms else zero
    rank_loss = torch.stack(rank_terms).mean() if rank_terms else zero
    preserve_loss = (
        torch.stack(preserve_terms).mean() if preserve_terms else zero
    )
    result = (
        float(ce_weight) * ce_loss
        + float(rank_weight) * rank_loss
        + float(preserve_weight) * preserve_loss
    )
    if not torch.isfinite(result):
        return zero, {
            "valid_rows": valid_rows,
            "matched_rows": matched_rows,
            "preserved_rows": preserved_rows,
        }
    return result, {
        "valid_rows": valid_rows,
        "matched_rows": matched_rows,
        "preserved_rows": preserved_rows,
        "ce": ce_loss.detach(),
        "rank": rank_loss.detach(),
        "preserve": preserve_loss.detach(),
        "weight_mean": (
            torch.stack(weight_means).mean().detach()
            if weight_means
            else zero.detach()
        ),
    }
