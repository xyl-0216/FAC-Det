# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
import json
from typing import Iterable
from pathlib import Path

from util.utils import to_device
from util import box_ops
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops.boxes import batched_nms

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from models.GroundingDINO.groundingdino import create_positive_map
from util.caption_builder import build_caption_from_labels


def _normalize_positive_map(pos_map, device):
    pos_map = pos_map.to(device=device).clone()
    denominator = pos_map.sum(dim=1, keepdim=True)
    valid_rows = denominator.squeeze(-1) > 0
    pos_map[valid_rows] = pos_map[valid_rows] / denominator[valid_rows]
    return pos_map


def _dump_detection_results_jsonl(path, targets, results):
    if not path:
        return
    dump_path = Path(path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("a", encoding="utf-8") as handle:
        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            labels = target.get("labels")
            boxes = target.get("boxes_xyxy_abs", target.get("boxes"))
            record = {
                "image_id": image_id,
                "scores": [float(x) for x in result["scores"].detach().cpu().tolist()],
                "labels": [int(x) for x in result["labels"].detach().cpu().tolist()],
                "boxes": [
                    [float(v) for v in box]
                    for box in result["boxes"].detach().cpu().tolist()
                ],
                "gt_labels": [int(x) for x in labels.detach().cpu().tolist()] if labels is not None else [],
                "gt_boxes": [
                    [float(v) for v in box]
                    for box in boxes.detach().cpu().tolist()
                ] if isinstance(boxes, torch.Tensor) else [],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _zscore_1d(values):
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-6)


def _apply_post_nms_facb_rerank(results, outputs, bbox_postprocessor, args):
    if not bool(getattr(args, "use_appearance_post_nms_rerank", False)):
        return results
    if (
        "appearance_local_evidence_logits" not in outputs
        or "appearance_local_candidate_ids" not in outputs
        or "query_ids" not in results[0]
    ):
        return results

    local_logits = outputs["appearance_local_evidence_logits"]
    local_candidate_ids = outputs["appearance_local_candidate_ids"]
    local_candidate_mask = outputs.get("appearance_local_candidate_mask", None)
    pred_logits = outputs["pred_logits"]
    device = pred_logits.device
    positive_map = _normalize_positive_map(bbox_postprocessor.positive_map, device)
    semantic_prob = pred_logits.sigmoid().float() @ positive_map.float().T
    label_ids = bbox_postprocessor.label_ids.to(device=device, dtype=torch.long)
    global_to_position = {int(g.item()): int(i) for i, g in enumerate(label_ids)}

    min_score = float(getattr(args, "appearance_post_nms_min_score", 0.02))
    max_score = float(getattr(args, "appearance_post_nms_max_score", -1.0))
    max_margin = float(getattr(args, "appearance_post_nms_max_semantic_margin", 0.10))
    lam = float(getattr(args, "appearance_post_nms_zscore_lambda", 2.0))
    debug_enabled = os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") == "1"
    debug = {
        "detections": 0,
        "eligible": 0,
        "changed": 0,
        "missing_evidence": 0,
    }

    for batch_idx, result in enumerate(results):
        if batch_idx >= local_logits.shape[0]:
            continue
        labels = result["labels"].clone()
        query_ids = result.get("query_ids", None)
        if query_ids is None:
            continue
        for det_idx in range(labels.numel()):
            debug["detections"] += 1
            det_score = float(result["scores"][det_idx].detach().item())
            if det_score < min_score:
                continue
            if max_score >= 0.0 and det_score > max_score:
                continue
            query_idx = int(query_ids[det_idx].detach().item())
            if query_idx < 0 or query_idx >= local_logits.shape[1]:
                continue

            ev_row = local_logits[batch_idx, query_idx].to(device=device).float()
            cid_row = local_candidate_ids[batch_idx, query_idx].to(device=device, dtype=torch.long)
            if local_candidate_mask is not None:
                mask_row = local_candidate_mask[batch_idx, query_idx].to(device=device, dtype=torch.bool)
            else:
                mask_row = cid_row >= 0
            valid = mask_row & torch.isfinite(ev_row) & (cid_row >= 0)
            if valid.sum().item() < 2:
                debug["missing_evidence"] += 1
                continue

            valid_global_ids = [int(x) for x in cid_row[valid].detach().cpu().tolist()]
            valid_positions = [
                global_to_position[gid]
                for gid in valid_global_ids
                if gid in global_to_position
            ]
            if len(valid_positions) < 2:
                debug["missing_evidence"] += 1
                continue
            position_tensor = torch.as_tensor(valid_positions, device=device, dtype=torch.long)
            app_logits = []
            aligned_global_ids = []
            for gid, score in zip(valid_global_ids, ev_row[valid]):
                if gid in global_to_position:
                    app_logits.append(score)
                    aligned_global_ids.append(gid)
            if len(app_logits) < 2:
                continue
            app_logits = torch.stack(app_logits).float()
            sem = semantic_prob[batch_idx, query_idx, position_tensor].float()
            sem_top2 = sem.topk(k=2, dim=0).values
            sem_margin = float((sem_top2[0] - sem_top2[1]).detach().item())
            if max_margin >= 0.0 and sem_margin > max_margin:
                continue

            debug["eligible"] += 1
            fused = _zscore_1d(sem) + lam * _zscore_1d(app_logits)
            winner = int(fused.argmax().item())
            winner_global_id = int(aligned_global_ids[winner])
            winner_position = global_to_position.get(winner_global_id, None)
            if winner_position is None:
                continue
            new_label = int(label_ids[winner_position].detach().item())
            if int(labels[det_idx].detach().item()) != new_label:
                labels[det_idx] = labels.new_tensor(new_label)
                debug["changed"] += 1
        result["labels"] = labels

    if debug_enabled and utils.is_main_process() and not getattr(_apply_post_nms_facb_rerank, "_printed", False):
        print(
            "[facb-debug][post-nms] "
            f"detections={debug['detections']} eligible={debug['eligible']} "
            f"changed={debug['changed']} missing_evidence={debug['missing_evidence']} "
            f"min_score={min_score} max_margin={max_margin} lambda={lam}",
            flush=True,
        )
        _apply_post_nms_facb_rerank._printed = True
    return results


def _extract_self_facb_roi_tokens(srcs, boxes_cxcywh, levels, roi_size):
    if not srcs or boxes_cxcywh.numel() == 0:
        return None
    device = boxes_cxcywh.device
    num_boxes = boxes_cxcywh.shape[0]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh.detach().clamp(0.0, 1.0))
    chunks = []
    for lvl in levels:
        lvl = int(lvl)
        if lvl < 0 or lvl >= len(srcs):
            continue
        fmap = srcs[lvl]
        _, _, fh, fw = fmap.shape
        b = boxes_xyxy.clone()
        scale = b.new_tensor([fw, fh, fw, fh])
        b = b * scale
        b[..., 0::2] = b[..., 0::2].clamp(0.0, float(fw - 1))
        b[..., 1::2] = b[..., 1::2].clamp(0.0, float(fh - 1))
        b[..., 2] = torch.maximum(b[..., 2], b[..., 0] + 1.0).clamp(max=float(fw))
        b[..., 3] = torch.maximum(b[..., 3], b[..., 1] + 1.0).clamp(max=float(fh))
        batch_idx = torch.zeros(num_boxes, 1, device=device, dtype=b.dtype)
        rois = torch.cat([batch_idx, b], dim=1)
        feats = roi_align(
            fmap.detach(),
            rois,
            output_size=int(roi_size),
            spatial_scale=1.0,
            aligned=True,
        )
        chunks.append(feats.flatten(2).permute(0, 2, 1))
    if not chunks:
        return None
    return torch.cat(chunks, dim=1)


def _expand_self_facb_boxes_cxcywh(boxes_cxcywh, scale):
    scale = float(scale)
    if scale <= 1.0:
        return boxes_cxcywh
    boxes = boxes_cxcywh.detach().clone()
    boxes[..., 2:] = (boxes[..., 2:] * scale).clamp(min=1e-6, max=1.0)
    boxes[..., :2] = boxes[..., :2].clamp(0.0, 1.0)
    return boxes


def _attach_pre_nms_self_facb_crm(
    outputs,
    bbox_postprocessor,
    model,
    args,
):
    """Attach bounded Self-FACB category scores before global top-k/NMS.

    Candidate mining and routing use only detached baseline scores.  The helper
    never writes into token logits or the text projection, and unprocessed
    queries remain bitwise-identical to the baseline category scores.
    """
    if not bool(getattr(args, "use_self_facb_crm", False)):
        return outputs
    if not bool(getattr(args, "enable_self_facb_crm_inference", True)):
        return outputs
    if (
        "self_facb_srcs" not in outputs
        or "self_facb_proto_bank" not in outputs
        or "pred_boxes" not in outputs
    ):
        return outputs

    model_ref = model.module if hasattr(model, "module") else model
    head = getattr(model_ref, "self_facb_head", None)
    if head is None or not hasattr(head, "forward_residual"):
        return outputs

    pred_logits = outputs["pred_logits"]
    device = pred_logits.device
    positive_map = _normalize_positive_map(bbox_postprocessor.positive_map, device)
    semantic_prob = torch.nan_to_num(
        pred_logits.sigmoid().float() @ positive_map.float().T,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    category_residual = semantic_prob.new_zeros(semantic_prob.shape)
    label_ids = bbox_postprocessor.label_ids.to(device=device, dtype=torch.long)
    proto_bank = outputs["self_facb_proto_bank"].to(
        device=device, dtype=pred_logits.dtype
    )
    proto_mask = outputs.get("self_facb_proto_bank_mask", None)
    if proto_mask is not None:
        proto_mask = proto_mask.to(device=device, dtype=torch.bool)

    num_global = int(proto_bank.shape[0])
    num_classes = min(int(semantic_prob.shape[-1]), int(label_ids.numel()))
    if num_global <= 0 or num_classes < 2:
        return outputs
    label_ids = label_ids[:num_classes]
    valid_class = (label_ids >= 0) & (label_ids < num_global)
    if proto_mask is not None:
        bank_valid = proto_mask[
            label_ids.clamp(min=0, max=max(num_global - 1, 0))
        ]
        if bank_valid.dim() > 1:
            bank_valid = bank_valid.all(dim=-1)
        valid_class = valid_class & bank_valid.bool()
    if int(valid_class.sum().item()) < 2:
        return outputs

    topk = min(max(int(getattr(args, "self_facb_crm_topk", 5)), 2), num_classes)
    query_cap = max(int(getattr(args, "self_facb_crm_query_topk", 150)), 1)
    min_score = float(getattr(args, "self_facb_crm_min_score", 0.02))
    max_score = float(getattr(args, "self_facb_crm_max_score", 0.20))
    max_margin = float(getattr(args, "self_facb_crm_max_margin", 0.10))
    levels = getattr(args, "self_facb_levels", outputs.get("self_facb_levels", [0]))
    if isinstance(levels, int):
        levels = [levels]
    roi_size = int(
        getattr(args, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7))
    )
    processed = 0
    lifted = 0
    residual_abs_sum = 0.0
    residual_abs_max = 0.0
    lift_abs_sum = 0.0
    lift_abs_max = 0.0
    lc_mode = bool(getattr(args, "use_self_facb_lc", False))

    for batch_idx in range(int(semantic_prob.shape[0])):
        sem = semantic_prob[batch_idx, :, :num_classes]
        ranked = sem.masked_fill(~valid_class.view(1, -1), float("-inf"))
        top_values, top_positions = torch.topk(ranked, k=topk, dim=-1)
        top1 = top_values[:, 0]
        margin = top_values[:, 0] - top_values[:, 1]
        eligible = torch.isfinite(top1) & (top1 >= min_score)
        if max_score >= 0.0:
            eligible = eligible & (top1 <= max_score)
        if max_margin >= 0.0:
            eligible = eligible & (margin <= max_margin)
        query_indices = eligible.nonzero(as_tuple=False).flatten()
        if query_indices.numel() == 0:
            continue
        if query_indices.numel() > query_cap:
            cap_order = top1[query_indices].topk(k=query_cap, dim=0).indices
            query_indices = query_indices[cap_order]

        roi_tokens = _extract_self_facb_roi_tokens(
            [src[batch_idx:batch_idx + 1] for src in outputs["self_facb_srcs"]],
            outputs["pred_boxes"][batch_idx, query_indices].detach(),
            levels,
            roi_size,
        )
        if lc_mode:
            context_scale = float(getattr(args, "self_facb_lc_context_scale", 1.0))
            if context_scale > 1.0:
                context_tokens = _extract_self_facb_roi_tokens(
                    [src[batch_idx:batch_idx + 1] for src in outputs["self_facb_srcs"]],
                    _expand_self_facb_boxes_cxcywh(
                        outputs["pred_boxes"][batch_idx, query_indices].detach(),
                        context_scale,
                    ),
                    levels,
                    roi_size,
                )
                if context_tokens is not None and context_tokens.numel() > 0:
                    roi_tokens = (
                        context_tokens
                        if roi_tokens is None or roi_tokens.numel() == 0
                        else torch.cat([roi_tokens, context_tokens], dim=1)
                    )
        if roi_tokens is None or roi_tokens.numel() == 0:
            continue
        candidate_positions = top_positions[query_indices]
        candidate_global = label_ids[candidate_positions]
        candidate_mask = torch.isfinite(top_values[query_indices])
        candidate_prototypes = proto_bank[
            candidate_global.clamp(min=0, max=max(num_global - 1, 0))
        ]
        if lc_mode and hasattr(head, "forward_lc"):
            selected_values = top_values[query_indices].clamp(0.0, 1.0)
            num_candidates = int(selected_values.shape[1])
            top1_values = selected_values[:, :1].expand(-1, num_candidates)
            if num_candidates > 1:
                margin_values = (selected_values[:, :1] - selected_values[:, 1:2]).expand(
                    -1, num_candidates
                )
                rank_feature = torch.linspace(
                    0.0,
                    1.0,
                    steps=num_candidates,
                    device=device,
                    dtype=torch.float32,
                ).view(1, -1).expand_as(selected_values)
            else:
                margin_values = selected_values.new_zeros(selected_values.shape)
                rank_feature = selected_values.new_zeros(selected_values.shape)
            score_features = torch.stack(
                [
                    selected_values.float(),
                    top1_values.float(),
                    margin_values.float(),
                    rank_feature.float(),
                ],
                dim=-1,
            )
            lc_out = head.forward_lc(
                roi_tokens,
                candidate_prototypes,
                candidate_mask,
                score_features=score_features,
                class_delta_bound=float(
                    getattr(args, "self_facb_lc_infer_class_delta_bound", 0.5)
                ),
                lift_bound=float(getattr(args, "self_facb_lc_infer_lift_bound", 2.5)),
            )
            class_delta = lc_out["class_delta"].float().masked_fill(~candidate_mask, 0.0)
            lift = lc_out["lift"].float().masked_fill(~candidate_mask, 0.0)
            lift_confidence = lc_out["lift_confidence"].float().masked_fill(
                ~candidate_mask, 0.0
            )
            raw_scores = lc_out["raw_scores"].float().masked_fill(
                ~candidate_mask, float("-inf")
            )
            base_logits = torch.logit(
                selected_values.clamp(min=1e-5, max=1.0 - 1e-5)
            ).masked_fill(~candidate_mask, float("-inf"))
            rerank_logits = (base_logits + class_delta).masked_fill(
                ~candidate_mask, float("-inf")
            )
            winner_slot = rerank_logits.argmax(dim=1)
            row_index = torch.arange(winner_slot.numel(), device=device)
            app_margin = raw_scores[row_index, winner_slot] - raw_scores[:, 0]
            winner_conf = lift_confidence[row_index, winner_slot]
            evidence_gate = (
                (app_margin >= float(getattr(args, "self_facb_lc_app_margin_thr", 0.0)))
                & (
                    winner_conf
                    >= float(getattr(args, "self_facb_lc_lift_confidence_thr", 0.5))
                )
            )
            residual = class_delta.clone()
            winner_lift = lift[row_index, winner_slot]
            residual[row_index, winner_slot] = residual[row_index, winner_slot] + torch.where(
                evidence_gate,
                winner_lift,
                winner_lift.new_zeros(winner_lift.shape),
            )
            residual = residual.masked_fill(~candidate_mask, 0.0)
            lifted += int(evidence_gate.sum().item())
            gated_lift = winner_lift[evidence_gate].abs()
            if gated_lift.numel() > 0:
                lift_abs_sum += float(gated_lift.sum().item())
                lift_abs_max = max(lift_abs_max, float(gated_lift.max().item()))
        else:
            residual, _, _ = head.forward_residual(
                roi_tokens,
                candidate_prototypes,
                candidate_mask,
            )
            residual = residual.float().masked_fill(~candidate_mask, 0.0)
        category_residual[
            batch_idx, query_indices.unsqueeze(1), candidate_positions
        ] = residual

        processed += int(query_indices.numel())
        abs_residual = residual[candidate_mask].abs()
        if abs_residual.numel() > 0:
            residual_abs_sum += float(abs_residual.sum().item())
            residual_abs_max = max(
                residual_abs_max, float(abs_residual.max().item())
            )

    outputs["self_facb_crm_category_residual"] = category_residual.to(
        dtype=pred_logits.dtype
    )
    outputs["self_facb_crm_processed"] = pred_logits.new_tensor(float(processed))
    outputs["self_facb_lc_lifted"] = pred_logits.new_tensor(float(lifted))
    outputs["self_facb_lc_lift_abs_mean"] = pred_logits.new_tensor(
        lift_abs_sum / max(lifted, 1)
    )
    outputs["self_facb_lc_lift_abs_max"] = pred_logits.new_tensor(lift_abs_max)
    denom = max(processed * topk, 1)
    outputs["self_facb_crm_residual_abs_mean"] = pred_logits.new_tensor(
        residual_abs_sum / denom
    )
    outputs["self_facb_crm_residual_abs_max"] = pred_logits.new_tensor(
        residual_abs_max
    )
    if (
        os.environ.get("GROUNDINGDINO_FACB_CRM_DEBUG", "0") == "1"
        and utils.is_main_process()
        and not getattr(_attach_pre_nms_self_facb_crm, "_printed", False)
    ):
        print(
            "[facb-crm][pre-nms] "
            f"processed={processed} topk={topk} query_cap={query_cap} "
            f"score_band=[{min_score},{max_score}] max_margin={max_margin} "
            f"lc={lc_mode} lifted={lifted} "
            f"residual_abs_mean={residual_abs_sum / denom:.6f} "
            f"residual_abs_max={residual_abs_max:.6f} "
            f"lift_abs_mean={lift_abs_sum / max(lifted, 1):.6f} "
            f"lift_abs_max={lift_abs_max:.6f}",
            flush=True,
        )
        _attach_pre_nms_self_facb_crm._printed = True
    return outputs


# Detector-native Self-FACB post-NMS rerank helpers.  The branch only verifies
# semantic top-M candidates for already selected detections; it does not create
# boxes or run a standalone classifier.
def _new_self_facb_debug():
    return {
        "detections": 0,
        "eligible": 0,
        "changed": 0,
        "score_updated": 0,
        "missing": 0,
        "low_score": 0,
        "high_score": 0,
        "high_margin": 0,
        "low_semantic_top1": 0,
        "high_semantic_top1": 0,
        "invalid_query": 0,
        "invalid_proto": 0,
        "same_group_filtered": 0,
        "current_missing": 0,
        "guarded": 0,
        "anchor_guarded": 0,
        "rank_filtered": 0,
        "cand_sum": 0,
        "cand_max": 0,
        "lambda_sum": 0.0,
    }


def _new_self_facb_change_stats():
    return {
        "detections": 0,
        "matched": 0,
        "changed": 0,
        "unchanged": 0,
        "before_correct": 0,
        "after_correct": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "wrong_to_wrong": 0,
        "right_to_right_changed": 0,
        "unchanged_right": 0,
        "unchanged_wrong": 0,
        "no_match": 0,
        "changed_gt_in_candidates": 0,
        "changed_gt_missing_candidates": 0,
        "changed_semantic_margin_sum": 0.0,
        "changed_app_margin_sum": 0.0,
        "changed_fused_margin_sum": 0.0,
        "top1_bins": {},
        "score_bins": {},
        "winner_semantic_bins": {},
    }


def _self_facb_gt_boxes_abs(target, device, dtype):
    boxes = target.get("boxes_xyxy_abs", None)
    if isinstance(boxes, torch.Tensor):
        return boxes.to(device=device, dtype=dtype)
    boxes = target.get("boxes", None)
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return None
    boxes = boxes.to(device=device, dtype=dtype)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    size = target.get("orig_size", target.get("size", None))
    if not isinstance(size, torch.Tensor):
        return boxes_xyxy
    h, w = size.to(device=device, dtype=dtype)
    scale = boxes_xyxy.new_tensor([w, h, w, h]).clamp(min=1.0)
    return boxes_xyxy * scale


def _match_self_facb_detection_to_gt(det_box, target, iou_thr):
    gt_boxes = _self_facb_gt_boxes_abs(target, det_box.device, det_box.dtype)
    gt_labels = target.get("labels", None)
    if (
        gt_boxes is None
        or gt_labels is None
        or gt_boxes.numel() == 0
        or gt_labels.numel() == 0
    ):
        return None, None, 0.0
    gt_labels = gt_labels.to(device=det_box.device, dtype=torch.long)
    ious = box_ops.box_iou(det_box.view(1, 4), gt_boxes)[0].squeeze(0)
    best_iou, best_idx = ious.max(dim=0)
    best_iou_float = float(best_iou.detach().item())
    if best_iou_float < float(iou_thr):
        return None, None, best_iou_float
    idx = int(best_idx.detach().item())
    return idx, int(gt_labels[idx].detach().item()), best_iou_float


def _update_self_facb_change_stats(
    stats,
    records,
    target,
    before_labels,
    after_labels,
    boxes,
    scores,
    query_ids,
    changed_indices,
    iou_thr,
    jsonl_path=None,
):
    if stats is None:
        return
    records = records or {}
    image_id = int(target["image_id"].item()) if "image_id" in target else -1
    changed_set = set(int(x) for x in changed_indices)
    writer = None
    if jsonl_path and utils.is_main_process():
        dump_path = Path(jsonl_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        writer = dump_path.open("a", encoding="utf-8")
    try:
        for det_idx in range(before_labels.numel()):
            stats["detections"] += 1
            gt_idx, gt_label, iou = _match_self_facb_detection_to_gt(
                boxes[det_idx], target, iou_thr
            )
            if gt_label is None:
                stats["no_match"] += 1
                continue
            stats["matched"] += 1
            before = int(before_labels[det_idx].detach().item())
            after = int(after_labels[det_idx].detach().item())
            before_ok = before == gt_label
            after_ok = after == gt_label
            stats["before_correct"] += int(before_ok)
            stats["after_correct"] += int(after_ok)
            changed = det_idx in changed_set or before != after
            if changed:
                stats["changed"] += 1
                extra = records.get(int(det_idx), {}) if isinstance(records, dict) else {}
                if isinstance(extra, dict):
                    candidate_labels = [int(x) for x in extra.get("candidate_labels", [])]
                    if candidate_labels:
                        if int(gt_label) in candidate_labels:
                            stats["changed_gt_in_candidates"] += 1
                        else:
                            stats["changed_gt_missing_candidates"] += 1
                    for stat_key, record_key in (
                        ("changed_semantic_margin_sum", "semantic_margin"),
                        ("changed_app_margin_sum", "app_margin"),
                        ("changed_fused_margin_sum", "fused_margin"),
                    ):
                        value = extra.get(record_key, None)
                        if value is not None:
                            stats[stat_key] += float(value)
                    top1_value = extra.get("semantic_top1", None)
                    if top1_value is not None:
                        top1 = float(top1_value)
                        if top1 < 0.02:
                            bin_key = "lt002"
                        elif top1 < 0.05:
                            bin_key = "002_005"
                        elif top1 < 0.10:
                            bin_key = "005_010"
                        elif top1 < 0.20:
                            bin_key = "010_020"
                        else:
                            bin_key = "ge020"
                    else:
                        bin_key = "missing"
                    score_value = float(scores[det_idx].detach().item())
                    if score_value < 0.02:
                        score_bin_key = "lt002"
                    elif score_value < 0.05:
                        score_bin_key = "002_005"
                    elif score_value < 0.10:
                        score_bin_key = "005_010"
                    elif score_value < 0.20:
                        score_bin_key = "010_020"
                    else:
                        score_bin_key = "ge020"
                    winner_sem_value = None
                    winner_local = extra.get("winner_local", None)
                    candidate_semantic_scores = extra.get("candidate_semantic_scores", None)
                    if (
                        winner_local is not None
                        and isinstance(candidate_semantic_scores, list)
                        and 0 <= int(winner_local) < len(candidate_semantic_scores)
                    ):
                        winner_sem_value = float(candidate_semantic_scores[int(winner_local)])
                    if winner_sem_value is None:
                        winner_sem_bin_key = "missing"
                    elif winner_sem_value < 0.02:
                        winner_sem_bin_key = "lt002"
                    elif winner_sem_value < 0.05:
                        winner_sem_bin_key = "002_005"
                    elif winner_sem_value < 0.10:
                        winner_sem_bin_key = "005_010"
                    elif winner_sem_value < 0.20:
                        winner_sem_bin_key = "010_020"
                    else:
                        winner_sem_bin_key = "ge020"
                if (not before_ok) and after_ok:
                    stats["wrong_to_right"] += 1
                    flow = "wrong_to_right"
                elif before_ok and (not after_ok):
                    stats["right_to_wrong"] += 1
                    flow = "right_to_wrong"
                elif (not before_ok) and (not after_ok):
                    stats["wrong_to_wrong"] += 1
                    flow = "wrong_to_wrong"
                else:
                    stats["right_to_right_changed"] += 1
                    flow = "right_to_right_changed"
                if isinstance(extra, dict):
                    bins = stats.setdefault("top1_bins", {})
                    bucket = bins.setdefault(
                        bin_key,
                        {
                            "changed": 0,
                            "wrong_to_right": 0,
                            "right_to_wrong": 0,
                            "wrong_to_wrong": 0,
                            "right_to_right_changed": 0,
                        },
                    )
                    bucket["changed"] += 1
                    bucket[flow] += 1
                    score_bins = stats.setdefault("score_bins", {})
                    score_bucket = score_bins.setdefault(
                        score_bin_key,
                        {
                            "changed": 0,
                            "wrong_to_right": 0,
                            "right_to_wrong": 0,
                            "wrong_to_wrong": 0,
                            "right_to_right_changed": 0,
                        },
                    )
                    score_bucket["changed"] += 1
                    score_bucket[flow] += 1
                    winner_semantic_bins = stats.setdefault("winner_semantic_bins", {})
                    winner_semantic_bucket = winner_semantic_bins.setdefault(
                        winner_sem_bin_key,
                        {
                            "changed": 0,
                            "wrong_to_right": 0,
                            "right_to_wrong": 0,
                            "wrong_to_wrong": 0,
                            "right_to_right_changed": 0,
                        },
                    )
                    winner_semantic_bucket["changed"] += 1
                    winner_semantic_bucket[flow] += 1
                if writer is not None:
                    record = {
                        "image_id": image_id,
                        "det_idx": int(det_idx),
                        "query_id": int(query_ids[det_idx].detach().item()) if query_ids is not None else -1,
                        "score": float(scores[det_idx].detach().item()),
                        "before_label": before,
                        "after_label": after,
                        "gt_label": int(gt_label),
                        "gt_idx": int(gt_idx),
                        "iou": float(iou),
                        "flow": flow,
                        "box": [float(x) for x in boxes[det_idx].detach().cpu().tolist()],
                    }
                    if isinstance(extra, dict):
                        record.update(extra)
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                stats["unchanged"] += 1
                if before_ok:
                    stats["unchanged_right"] += 1
                else:
                    stats["unchanged_wrong"] += 1
    finally:
        if writer is not None:
            writer.close()


def _print_self_facb_change_stats(stats, prefix="[facb-debug][change]"):
    if not stats or not utils.is_main_process():
        return
    matched = max(int(stats.get("matched", 0)), 1)
    changed_total = int(stats.get("changed", 0))
    changed = max(changed_total, 1)
    before_acc = stats.get("before_correct", 0) / matched
    after_acc = stats.get("after_correct", 0) / matched
    avg_sem_margin = stats.get("changed_semantic_margin_sum", 0.0) / changed
    avg_app_margin = stats.get("changed_app_margin_sum", 0.0) / changed
    avg_fused_margin = stats.get("changed_fused_margin_sum", 0.0) / changed
    print(
        f"{prefix} detections={stats.get('detections', 0)} matched={stats.get('matched', 0)} "
        f"no_match={stats.get('no_match', 0)} changed={stats.get('changed', 0)} "
        f"unchanged={stats.get('unchanged', 0)} before_acc={before_acc:.4f} "
        f"after_acc={after_acc:.4f} delta={after_acc - before_acc:.4f} "
        f"wrong_to_right={stats.get('wrong_to_right', 0)} "
        f"right_to_wrong={stats.get('right_to_wrong', 0)} "
        f"wrong_to_wrong={stats.get('wrong_to_wrong', 0)} "
        f"right_to_right_changed={stats.get('right_to_right_changed', 0)} "
        f"unchanged_right={stats.get('unchanged_right', 0)} "
        f"unchanged_wrong={stats.get('unchanged_wrong', 0)} "
        f"changed_precision={stats.get('wrong_to_right', 0) / changed:.4f} "
        f"gt_in_changed={stats.get('changed_gt_in_candidates', 0)}/{changed_total} "
        f"avg_sem_margin={avg_sem_margin:.4f} avg_app_margin={avg_app_margin:.4f} "
        f"avg_fused_margin={avg_fused_margin:.4f}",
        flush=True,
    )
    bins = stats.get("top1_bins", {}) or {}
    for key in ("lt002", "002_005", "005_010", "010_020", "ge020", "missing"):
        bucket = bins.get(key, None)
        if not bucket:
            continue
        changed = max(int(bucket.get("changed", 0)), 1)
        print(
            f"{prefix}-top1-bin bucket={key} changed={bucket.get('changed', 0)} "
            f"wrong_to_right={bucket.get('wrong_to_right', 0)} "
            f"right_to_wrong={bucket.get('right_to_wrong', 0)} "
            f"wrong_to_wrong={bucket.get('wrong_to_wrong', 0)} "
            f"right_to_right_changed={bucket.get('right_to_right_changed', 0)} "
            f"changed_precision={bucket.get('wrong_to_right', 0) / changed:.4f}",
            flush=True,
        )
    score_bins = stats.get("score_bins", {}) or {}
    for key in ("lt002", "002_005", "005_010", "010_020", "ge020"):
        bucket = score_bins.get(key, None)
        if not bucket:
            continue
        changed = max(int(bucket.get("changed", 0)), 1)
        print(
            f"{prefix}-score-bin bucket={key} changed={bucket.get('changed', 0)} "
            f"wrong_to_right={bucket.get('wrong_to_right', 0)} "
            f"right_to_wrong={bucket.get('right_to_wrong', 0)} "
            f"wrong_to_wrong={bucket.get('wrong_to_wrong', 0)} "
            f"right_to_right_changed={bucket.get('right_to_right_changed', 0)} "
            f"changed_precision={bucket.get('wrong_to_right', 0) / changed:.4f}",
            flush=True,
        )
    winner_semantic_bins = stats.get("winner_semantic_bins", {}) or {}
    for key in ("lt002", "002_005", "005_010", "010_020", "ge020", "missing"):
        bucket = winner_semantic_bins.get(key, None)
        if not bucket:
            continue
        changed = max(int(bucket.get("changed", 0)), 1)
        print(
            f"{prefix}-winner-sem-bin bucket={key} changed={bucket.get('changed', 0)} "
            f"wrong_to_right={bucket.get('wrong_to_right', 0)} "
            f"right_to_wrong={bucket.get('right_to_wrong', 0)} "
            f"wrong_to_wrong={bucket.get('wrong_to_wrong', 0)} "
            f"right_to_right_changed={bucket.get('right_to_right_changed', 0)} "
            f"changed_precision={bucket.get('wrong_to_right', 0) / changed:.4f}",
            flush=True,
        )


def _print_self_facb_skip(debug_enabled, reason):
    if not debug_enabled or not utils.is_main_process():
        return
    if getattr(_print_self_facb_skip, "_printed", False):
        return
    print(f"[facb-debug][self-post-nms] skipped reason={reason}", flush=True)
    _print_self_facb_skip._printed = True


def _resolve_self_facb_post_nms_inputs(results, outputs, model, args):
    if not bool(getattr(args, "use_self_facb", False)):
        return None, "use_self_facb_false"
    model_ref = getattr(model, "module", model)
    if model_ref is None or not hasattr(model_ref, "self_facb_head"):
        return None, "missing_self_facb_head"
    if not results or "query_ids" not in results[0]:
        return None, "missing_query_ids"
    if "self_facb_srcs" not in outputs or "self_facb_proto_bank" not in outputs:
        return None, "missing_self_facb_features"
    return model_ref, None


def _valid_self_facb_proto_mask(gids, proto_mask):
    valid = gids >= 0
    if proto_mask is None:
        return valid
    if proto_mask.dim() == 2:
        num_proto = proto_mask.shape[0]
        gid_safe = gids.clamp(0, num_proto - 1)
        return valid & (gids < num_proto) & proto_mask[gid_safe].all(dim=-1)
    return valid & (gids < proto_mask.numel()) & proto_mask[
        gids.clamp(0, proto_mask.numel() - 1)
    ]


def _select_self_facb_candidates(
    sem_row,
    label_ids,
    proto_mask,
    topm,
    max_margin,
    max_top1,
    debug,
    require_low_top1=False,
):
    k = min(int(topm), int(sem_row.numel()))
    if k < 2:
        return None

    top_scores, top_pos = torch.topk(sem_row, k=k, dim=0)
    sem_top1 = float(top_scores[0].detach().item())
    sem_margin = float((top_scores[0] - top_scores[1]).detach().item())
    low_top1 = max_top1 >= 0.0 and sem_top1 <= max_top1
    if low_top1:
        debug["low_semantic_top1"] += 1
    if bool(require_low_top1) and not low_top1:
        debug["high_semantic_top1"] += 1
        return None
    if max_margin >= 0.0 and sem_margin > max_margin and not low_top1:
        debug["high_margin"] += 1
        return None

    gids = label_ids[top_pos]
    valid = _valid_self_facb_proto_mask(gids, proto_mask)
    if valid.sum().item() < 2:
        debug["missing"] += 1
        debug["invalid_proto"] += 1
        return None

    return {
        "global_ids": gids[valid],
        "semantic_scores": top_scores[valid],
        "semantic_top1": sem_top1,
        "semantic_margin": sem_margin,
    }


def _filter_self_facb_same_group(selected, current_label, group_ids, debug):
    if group_ids is None:
        return selected
    if current_label < 0 or current_label >= group_ids.numel():
        return selected
    gids = selected["global_ids"]
    valid_gid = (gids >= 0) & (gids < group_ids.numel())
    if valid_gid.sum().item() < 2:
        return selected
    current_group = group_ids[current_label]
    same_group = valid_gid & (group_ids[gids.clamp(0, group_ids.numel() - 1)] == current_group)
    if same_group.sum().item() < 2:
        debug["same_group_filtered"] += 1
        return None
    filtered = dict(selected)
    filtered["global_ids"] = selected["global_ids"][same_group]
    filtered["semantic_scores"] = selected["semantic_scores"][same_group]
    return filtered


def _pack_self_facb_candidates(
    roi_tokens, proto_bank, candidate_global_ids, semantic_rows
):
    max_k = max(int(x.numel()) for x in candidate_global_ids)
    if proto_bank.dim() == 3:
        candidate_prototypes = roi_tokens.new_zeros(
            (len(candidate_global_ids), max_k, proto_bank.shape[1], proto_bank.shape[-1])
        )
    else:
        candidate_prototypes = roi_tokens.new_zeros(
            (len(candidate_global_ids), max_k, proto_bank.shape[-1])
        )
    candidate_mask = torch.zeros(
        (len(candidate_global_ids), max_k), device=roi_tokens.device, dtype=torch.bool
    )
    semantic_matrix = roi_tokens.new_zeros((len(candidate_global_ids), max_k))

    for row_idx, (gids, sem) in enumerate(zip(candidate_global_ids, semantic_rows)):
        k = int(gids.numel())
        candidate_prototypes[row_idx, :k] = proto_bank[gids]
        candidate_mask[row_idx, :k] = True
        semantic_matrix[row_idx, :k] = sem.to(dtype=semantic_matrix.dtype)

    return candidate_prototypes, candidate_mask, semantic_matrix


def _run_self_facb_head(
    head, roi_tokens, candidate_prototypes, candidate_mask, proto_bank, aggregation
):
    old_aggregation = getattr(head, "facet_aggregation", None)
    if aggregation and hasattr(head, "facet_aggregation"):
        head.facet_aggregation = head._normalize_facet_aggregation(aggregation)
    try:
        return head(
            roi_tokens.to(dtype=proto_bank.dtype),
            candidate_prototypes,
            candidate_mask,
        ).float()
    finally:
        if aggregation and old_aggregation is not None:
            head.facet_aggregation = old_aggregation


def _fuse_self_facb_scores(semantic_scores, facb_scores, fusion_mode, lam):
    if fusion_mode in {"facb_only", "appearance_only", "app_only"}:
        return facb_scores
    if fusion_mode in {"facb_zscore", "appearance_zscore", "app_zscore"}:
        return _zscore_1d(facb_scores)
    if semantic_scores.numel() < 3:
        sem_delta = semantic_scores - semantic_scores.mean()
        app_delta = facb_scores - facb_scores.mean()
        return sem_delta + float(lam) * app_delta
    return _zscore_1d(semantic_scores) + float(lam) * _zscore_1d(facb_scores)


def _self_facb_replacement_score(
    semantic_scores,
    facb_scores,
    fused_scores,
    winner_local,
    fusion_mode,
    current_score=None,
    score_update_mode="legacy",
    score_update_alpha=0.5,
    score_update_cap=-1.0,
):
    sem_score = semantic_scores[winner_local].clamp(min=0.0, max=1.0)
    facb_prob = facb_scores[winner_local].sigmoid().clamp(min=0.0, max=1.0)
    if fused_scores is not None and fused_scores.numel() > 0:
        fused_conf = torch.softmax(fused_scores.float(), dim=0)[winner_local].clamp(min=0.0, max=1.0)
    else:
        fused_conf = sem_score

    mode = str(score_update_mode or "legacy").strip().lower()
    alpha = max(0.0, min(1.0, float(score_update_alpha)))
    if mode in {"legacy", "semantic"}:
        if fusion_mode in {
            "facb_only",
            "appearance_only",
            "app_only",
            "facb_zscore",
            "appearance_zscore",
            "app_zscore",
        }:
            return facb_prob
        return sem_score
    if mode in {"facb", "facb_sigmoid", "appearance", "appearance_sigmoid"}:
        return facb_prob
    if mode in {"blend", "semantic_facb_blend", "sem_facb_blend"}:
        return ((1.0 - alpha) * sem_score + alpha * facb_prob).clamp(min=0.0, max=1.0)
    if mode in {"max", "semantic_facb_max", "sem_facb_max"}:
        return torch.maximum(sem_score, facb_prob)
    if mode in {"fused", "fused_softmax", "fusion_softmax"}:
        return fused_conf
    if mode in {"sem_anchor_max", "semantic_anchor_max", "reanchor_max"} and current_score is not None:
        current = current_score.to(device=sem_score.device, dtype=sem_score.dtype).clamp(min=0.0, max=1.0)
        return torch.maximum(current, sem_score)
    if mode in {"bounded_boost", "capped_boost", "det_sem_facb_bounded"} and current_score is not None:
        current = current_score.to(device=sem_score.device, dtype=sem_score.dtype).clamp(min=0.0, max=1.0)
        target = torch.maximum(sem_score, facb_prob)
        boosted = current + alpha * (target - current).clamp(min=0.0)
        cap = float(score_update_cap)
        if cap >= 0.0:
            boosted = torch.minimum(boosted, current + current.new_tensor(cap))
        return boosted.clamp(min=0.0, max=1.0)
    if mode in {"current_max", "original_max", "det_sem_facb_max"} and current_score is not None:
        current = current_score.to(device=sem_score.device, dtype=sem_score.dtype).clamp(min=0.0, max=1.0)
        return torch.maximum(current, torch.maximum(sem_score, facb_prob))
    return sem_score


def _print_self_facb_debug(
    debug,
    topm,
    levels,
    roi_size,
    min_score,
    max_margin,
    max_top1,
    max_rank,
    lam,
    aggregation,
    fusion_mode,
    update_score,
):
    if os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") != "1":
        return
    if not utils.is_main_process():
        return
    if getattr(_print_self_facb_debug, "_printed", False):
        return
    avg_candidates = debug["cand_sum"] / max(debug["eligible"], 1)
    avg_lambda = debug["lambda_sum"] / max(debug["eligible"], 1)
    print(
        "[facb-debug][self-post-nms] "
        f"detections={debug['detections']} eligible={debug['eligible']} "
        f"changed={debug['changed']} score_updated={debug['score_updated']} "
        f"missing={debug['missing']} low_score={debug['low_score']} "
        f"high_score={debug['high_score']} "
        f"high_margin={debug['high_margin']} "
        f"low_semantic_top1={debug['low_semantic_top1']} "
        f"high_semantic_top1={debug['high_semantic_top1']} "
        f"invalid_query={debug['invalid_query']} "
        f"invalid_proto={debug['invalid_proto']} "
        f"same_group_filtered={debug['same_group_filtered']} "
        f"current_missing={debug['current_missing']} guarded={debug['guarded']} "
        f"anchor_guarded={debug['anchor_guarded']} "
        f"rank_filtered={debug['rank_filtered']} "
        f"avg_candidates={avg_candidates:.2f} max_candidates={debug['cand_max']} "
        f"topm={topm} levels={list(levels)} roi_size={roi_size} "
        f"min_score={min_score} max_margin={max_margin} max_top1={max_top1} "
        f"max_rank={max_rank} lambda={lam} "
        f"avg_lambda={avg_lambda:.3f} "
        f"aggregation={aggregation} fusion_mode={fusion_mode} "
        f"update_score={update_score}",
        flush=True,
    )
    _print_self_facb_debug._printed = True


def _apply_post_nms_self_facb_rerank(
    results,
    outputs,
    bbox_postprocessor,
    model,
    args,
    orig_target_sizes,
    targets=None,
):
    debug_enabled = os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") == "1"
    model_ref, skip_reason = _resolve_self_facb_post_nms_inputs(
        results, outputs, model, args
    )
    if skip_reason is not None:
        _print_self_facb_skip(debug_enabled, skip_reason)
        return results

    pred_logits = outputs["pred_logits"]
    device = pred_logits.device
    srcs = outputs["self_facb_srcs"]
    proto_bank = outputs["self_facb_proto_bank"].to(
        device=device, dtype=pred_logits.dtype
    )
    proto_mask = outputs.get("self_facb_proto_bank_mask", None)
    if proto_mask is not None:
        proto_mask = proto_mask.to(device=device, dtype=torch.bool)

    levels = getattr(args, "self_facb_levels", None)
    if levels is None:
        levels = outputs.get(
            "self_facb_levels",
            getattr(args, "appearance_local_evidence_levels", [0]),
        )
    if isinstance(levels, int):
        levels = [levels]
    roi_size = int(getattr(args, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7)))
    topm = int(getattr(args, "self_facb_topm", getattr(args, "appearance_local_evidence_topm", 10)))
    min_score = float(getattr(args, "self_facb_min_score", 0.02))
    max_score = float(getattr(args, "self_facb_max_score", -1.0))
    max_margin = float(getattr(args, "self_facb_max_semantic_margin", 0.10))
    max_top1 = float(getattr(args, "self_facb_max_semantic_top1", -1.0))
    require_low_top1 = bool(getattr(args, "self_facb_require_low_semantic_top1", False))
    max_rank = int(getattr(args, "self_facb_max_semantic_rank", -1))
    lam = float(getattr(args, "self_facb_lambda", 2.0))
    min_app_margin = float(getattr(args, "self_facb_min_app_margin", -1.0))
    min_anchor_app_margin = float(getattr(args, "self_facb_min_anchor_app_margin", -1.0))
    require_current = bool(getattr(args, "self_facb_require_current_label", False))
    same_group_only = bool(getattr(args, "self_facb_same_group_only", False))
    dynamic_lambda = bool(getattr(args, "self_facb_dynamic_lambda", False))
    dynamic_lambda_min = float(getattr(args, "self_facb_dynamic_lambda_min", 0.25))
    update_score = bool(getattr(args, "self_facb_update_score", False))
    score_update_mode = str(getattr(args, "self_facb_score_update_mode", "legacy") or "legacy")
    score_update_alpha = float(getattr(args, "self_facb_score_update_alpha", 0.5))
    score_update_cap = float(getattr(args, "self_facb_score_update_cap", -1.0))
    update_score_on_change_only = bool(
        getattr(args, "self_facb_update_score_on_change_only", False)
    )
    change_stats = getattr(_apply_post_nms_self_facb_rerank, "_change_stats", None)
    change_jsonl = str(getattr(args, "self_facb_change_analysis_jsonl", "") or "")
    change_iou_thr = float(getattr(args, "self_facb_change_iou_thr", 0.5))
    inference_aggregation = str(
        getattr(args, "self_facb_inference_aggregation", "") or ""
    ).strip()
    fusion_mode = str(
        getattr(args, "self_facb_fusion_mode", "semantic_fusion")
    ).strip().lower()

    positive_map = _normalize_positive_map(bbox_postprocessor.positive_map, device)
    semantic_prob = pred_logits.sigmoid().float() @ positive_map.float().T
    label_ids = bbox_postprocessor.label_ids.to(device=device, dtype=torch.long)
    global_to_position = {int(g.item()): int(i) for i, g in enumerate(label_ids)}
    group_ids = getattr(args, "appearance_global_fine_group_ids", None)
    if group_ids is not None:
        group_ids = torch.as_tensor(group_ids, device=device, dtype=torch.long)

    head = model_ref.self_facb_head.to(device)
    head.eval()
    debug = _new_self_facb_debug()

    with torch.no_grad():
        for batch_idx, result in enumerate(results):
            query_ids = result.get("query_ids", None)
            if query_ids is None or result["labels"].numel() == 0:
                continue

            before_labels = result["labels"].clone()
            before_scores = result["scores"].clone()
            labels = result["labels"].clone()
            boxes_abs = result["boxes"].to(device=device, dtype=pred_logits.dtype)
            orig_h, orig_w = orig_target_sizes[batch_idx].to(
                device=device, dtype=pred_logits.dtype
            )
            scale = boxes_abs.new_tensor([orig_w, orig_h, orig_w, orig_h]).clamp(min=1.0)
            boxes_norm_xyxy = boxes_abs / scale
            boxes_norm_cxcywh = box_ops.box_xyxy_to_cxcywh(
                boxes_norm_xyxy
            ).clamp(0.0, 1.0)

            eligible_det = []
            candidate_global_ids = []
            semantic_rows = []
            changed_indices = []
            change_records = {}
            for det_idx in range(labels.numel()):
                debug["detections"] += 1
                score = float(result["scores"][det_idx].detach().item())
                if score < min_score:
                    debug["low_score"] += 1
                    continue
                if max_score >= 0.0 and score >= max_score:
                    debug["high_score"] += 1
                    continue

                query_idx = int(query_ids[det_idx].detach().item())
                if query_idx < 0 or query_idx >= semantic_prob.shape[1]:
                    debug["invalid_query"] += 1
                    continue

                selected = _select_self_facb_candidates(
                    semantic_prob[batch_idx, query_idx].float(),
                    label_ids,
                    proto_mask,
                    topm,
                    max_margin,
                    max_top1,
                    debug,
                    require_low_top1=require_low_top1,
                )
                if selected is None:
                    continue
                current_label = int(labels[det_idx].detach().item())
                if require_current and not bool((selected["global_ids"] == current_label).any().item()):
                    debug["current_missing"] += 1
                    continue
                if same_group_only:
                    selected = _filter_self_facb_same_group(
                        selected,
                        current_label,
                        group_ids,
                        debug,
                    )
                    if selected is None:
                        continue

                eligible_det.append(det_idx)
                candidate_global_ids.append(selected["global_ids"])
                semantic_rows.append(selected["semantic_scores"])

            if not eligible_det:
                result["labels"] = labels
                continue

            roi_tokens = _extract_self_facb_roi_tokens(
                [s[batch_idx:batch_idx + 1] for s in srcs],
                boxes_norm_cxcywh[eligible_det],
                levels,
                roi_size,
            )
            if roi_tokens is None:
                debug["missing"] += len(eligible_det)
                result["labels"] = labels
                continue

            candidate_prototypes, candidate_mask, semantic_matrix = (
                _pack_self_facb_candidates(
                    roi_tokens,
                    proto_bank,
                    candidate_global_ids,
                    semantic_rows,
                )
            )
            app_scores = _run_self_facb_head(
                head,
                roi_tokens,
                candidate_prototypes,
                candidate_mask,
                proto_bank,
                inference_aggregation,
            )
            semantic_matrix = semantic_matrix.float().masked_fill(~candidate_mask, float("-inf"))
            app_scores = app_scores.masked_fill(~candidate_mask, float("-inf"))

            for row_idx, det_idx in enumerate(eligible_det):
                valid = candidate_mask[row_idx]
                sem = semantic_matrix[row_idx, valid]
                app = app_scores[row_idx, valid]
                if sem.numel() < 2 or not torch.isfinite(app).any():
                    continue

                debug["eligible"] += 1
                debug["cand_sum"] += int(sem.numel())
                debug["cand_max"] = max(debug["cand_max"], int(sem.numel()))
                row_lam = lam
                if dynamic_lambda and max_margin > 0.0:
                    sem_sorted = torch.sort(sem, descending=True).values
                    sem_margin = float((sem_sorted[0] - sem_sorted[1]).detach().item())
                    uncertainty = max(0.0, min(1.0, (max_margin - sem_margin) / max(max_margin, 1e-6)))
                    row_lam = lam * max(dynamic_lambda_min, uncertainty)
                debug["lambda_sum"] += float(row_lam)
                fused = _fuse_self_facb_scores(sem, app, fusion_mode, row_lam)
                winner_local = int(fused.argmax().item())
                sem_sorted = torch.sort(sem, descending=True).values
                app_sorted = torch.sort(app, descending=True).values
                fused_sorted = torch.sort(fused, descending=True).values
                sem_margin = (
                    float((sem_sorted[0] - sem_sorted[1]).detach().item())
                    if sem_sorted.numel() >= 2 else 0.0
                )
                app_margin = (
                    float((app_sorted[0] - app_sorted[1]).detach().item())
                    if app_sorted.numel() >= 2 else 0.0
                )
                fused_margin = (
                    float((fused_sorted[0] - fused_sorted[1]).detach().item())
                    if fused_sorted.numel() >= 2 else 0.0
                )
                candidate_labels = [
                    int(x) for x in candidate_global_ids[row_idx].detach().cpu().tolist()
                ]
                change_records[int(det_idx)] = {
                    "candidate_labels": candidate_labels,
                    "candidate_semantic_scores": [
                        float(x) for x in sem.detach().float().cpu().tolist()
                    ],
                    "candidate_app_scores": [
                        float(x) for x in app.detach().float().cpu().tolist()
                    ],
                    "candidate_fused_scores": [
                        float(x) for x in fused.detach().float().cpu().tolist()
                    ],
                    "semantic_margin": sem_margin,
                    "semantic_top1": float(selected.get("semantic_top1", 0.0)),
                    "app_margin": app_margin,
                    "fused_margin": fused_margin,
                    "winner_local": winner_local,
                    "winner_semantic_rank": int(winner_local + 1),
                    "winner_label": int(candidate_labels[winner_local]) if candidate_labels else -1,
                    "row_lambda": float(row_lam),
                }
                if max_rank > 0 and winner_local >= max_rank:
                    debug["rank_filtered"] += 1
                    continue
                if min_app_margin >= 0.0:
                    if app_sorted.numel() >= 2:
                        if app_margin < min_app_margin:
                            debug["guarded"] += 1
                            continue
                if min_anchor_app_margin >= 0.0:
                    current_label = int(labels[det_idx].detach().item())
                    current_slot = -1
                    for cand_idx, cand_label in enumerate(candidate_labels):
                        if int(cand_label) == current_label:
                            current_slot = int(cand_idx)
                            break
                    if current_slot >= 0 and current_slot != winner_local:
                        anchor_margin = float(
                            (app[winner_local] - app[current_slot]).detach().item()
                        )
                        if anchor_margin < min_anchor_app_margin:
                            debug["anchor_guarded"] += 1
                            continue
                winner_gid = int(candidate_global_ids[row_idx][winner_local].detach().item())
                winner_pos = global_to_position.get(winner_gid)
                if winner_pos is None:
                    continue

                new_label = int(label_ids[winner_pos].detach().item())
                if int(labels[det_idx].detach().item()) != new_label:
                    labels[det_idx] = labels.new_tensor(new_label)
                    debug["changed"] += 1
                    changed_indices.append(int(det_idx))

                if update_score and (
                    not update_score_on_change_only
                    or int(before_labels[det_idx].detach().item()) != int(labels[det_idx].detach().item())
                ):
                    new_score = _self_facb_replacement_score(
                        sem,
                        app,
                        fused,
                        winner_local,
                        fusion_mode,
                        current_score=result["scores"][det_idx],
                        score_update_mode=score_update_mode,
                        score_update_alpha=score_update_alpha,
                        score_update_cap=score_update_cap,
                    )
                    result["scores"][det_idx] = result["scores"].new_tensor(
                        float(new_score.detach().item())
                    )
                    debug["score_updated"] += 1

            result["labels"] = labels
            if targets is not None and batch_idx < len(targets):
                _update_self_facb_change_stats(
                    stats=change_stats,
                    records=change_records,
                    target=targets[batch_idx],
                    before_labels=before_labels,
                    after_labels=labels,
                    boxes=boxes_abs,
                    scores=before_scores.to(device=device),
                    query_ids=query_ids.to(device=device) if query_ids is not None else None,
                    changed_indices=changed_indices,
                    iou_thr=change_iou_thr,
                    jsonl_path=change_jsonl,
                )

    _print_self_facb_debug(
        debug,
        topm,
        levels,
        roi_size,
        min_score,
        max_margin,
        max_top1,
        max_rank,
        lam,
        inference_aggregation or getattr(head, "facet_aggregation", "default"),
        fusion_mode,
        update_score,
    )
    return results


def _build_dynamic_positive_map(tokenizer, max_text_len, caption, cat_list, device):
    tokenized = tokenizer(
        caption,
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    label_ids = torch.arange(len(cat_list))
    pos_map = create_positive_map(
        tokenized,
        label_ids,
        cat_list,
        caption,
        max_text_len=max_text_len,
    )
    return _normalize_positive_map(pos_map, device)


def _resolve_stage2_rerank_label_texts(args, base_cat_list):
    prompt_source = str(getattr(args, "stage2_candidate_rerank_prompt_source", "semantic")).strip()
    semantic_label_list = list(getattr(args, "semantic_label_list", []) or [])
    description_label_list = list(getattr(args, "description_label_list", []) or [])
    eval_label_list = list(getattr(args, "eval_label_list", []) or [])
    base_cat_list = list(base_cat_list or [])

    if prompt_source == "semantic":
        return semantic_label_list or eval_label_list or base_cat_list
    if prompt_source == "description":
        return description_label_list or semantic_label_list or eval_label_list or base_cat_list
    if prompt_source == "eval":
        return eval_label_list or base_cat_list
    if prompt_source == "base":
        return base_cat_list
    raise ValueError(
        "Unsupported stage2_candidate_rerank_prompt_source: "
        f"{prompt_source}"
    )


@torch.no_grad()
def _rerank_stage1_results_locally(
    stage1_results,
    base_prob_to_label,
    alt_prob_to_label,
    bbox_postprocessor,
    class_topk=10,
    query_topk=None,
    base_weight=0.8,
    alt_weight=0.2,
):
    if query_topk is None:
        query_topk = int(bbox_postprocessor.num_select)
    else:
        query_topk = int(query_topk)

    class_topk = max(1, int(class_topk))
    base_weight = float(base_weight)
    alt_weight = float(alt_weight)
    label_ids = bbox_postprocessor.label_ids.to(base_prob_to_label.device)
    reranked_results = []

    for batch_idx, stage1_result in enumerate(stage1_results):
        scores = stage1_result["scores"]
        labels = stage1_result["labels"]
        boxes = stage1_result["boxes"]
        query_ids = stage1_result.get("query_ids", None)
        class_positions = stage1_result.get("class_positions", None)

        if (
            scores.numel() == 0
            or query_ids is None
            or class_positions is None
        ):
            reranked_results.append(stage1_result)
            continue

        new_scores = scores.clone()
        fused_query_class_scores = []
        for det_idx in range(scores.shape[0]):
            query_idx = int(query_ids[det_idx].item())
            class_position = int(class_positions[det_idx].item())
            base_row = base_prob_to_label[batch_idx, query_idx]
            alt_row = alt_prob_to_label[batch_idx, query_idx]
            k = min(class_topk, base_row.shape[0])
            top_positions = base_row.topk(k=k, dim=0).indices
            if not (top_positions == int(class_position)).any().item():
                fused_score = base_row[class_position]
            else:
                local_idx = torch.nonzero(top_positions == int(class_position), as_tuple=False).flatten()
                if local_idx.numel() == 0:
                    fused_score = base_row[class_position]
                else:
                    local_idx = int(local_idx[0].item())
                    fused_local_scores = base_weight * base_row[top_positions] + alt_weight * alt_row[top_positions]
                    fused_score = fused_local_scores[local_idx]
            new_scores[det_idx] = fused_score
            fused_query_class_scores.append(fused_score)

        keep_topk = min(query_topk, new_scores.shape[0])
        if keep_topk > 0:
            top_scores, top_idx = torch.topk(new_scores, keep_topk)
            reranked_results.append(
                {
                    "scores": top_scores,
                    "labels": labels[top_idx],
                    "boxes": boxes[top_idx],
                    "query_ids": query_ids[top_idx],
                    "class_positions": class_positions[top_idx],
                }
            )
        else:
            reranked_results.append(
                {
                    "scores": new_scores,
                    "labels": labels,
                    "boxes": boxes,
                    "query_ids": query_ids,
                    "class_positions": class_positions,
                }
            )

    return reranked_results


@torch.no_grad()
def _postprocess_stage2_mixed_prompt(
    outputs,
    target_sizes,
    stage2_caption_lists,
    stage2_class_positions,
    bbox_postprocessor,
    sem_weight=0.7,
    desc_weight=0.3,
):
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    prob_to_token = out_logits.sigmoid()
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes_xyxy = boxes_xyxy * scale_fct[:, None, :]

    num_select = int(bbox_postprocessor.num_select)
    label_ids = bbox_postprocessor.label_ids.to(prob_to_token.device)
    results = []

    for batch_idx, cat_list in enumerate(stage2_caption_lists):
        caption, normalized_cat_list = build_caption_from_labels(cat_list)
        pos_map = _build_dynamic_positive_map(
            bbox_postprocessor.tokenizer,
            bbox_postprocessor.max_text_len,
            caption,
            normalized_cat_list,
            prob_to_token.device,
        )
        text_scores = prob_to_token[batch_idx] @ pos_map.T

        class_count = len(stage2_class_positions[batch_idx])
        if class_count == 0:
            results.append(
                {
                    "scores": torch.empty(0, device=prob_to_token.device),
                    "labels": torch.empty(0, dtype=torch.long, device=prob_to_token.device),
                    "boxes": torch.empty(0, 4, device=prob_to_token.device),
                }
            )
            continue

        pair_scores = text_scores.view(text_scores.shape[0], class_count, 2)
        fused_scores = sem_weight * pair_scores[:, :, 0] + desc_weight * pair_scores[:, :, 1]
        topk_count = min(num_select, fused_scores.numel())
        topk_values, topk_indexes = torch.topk(fused_scores.reshape(-1), topk_count, dim=0)
        topk_boxes = torch.div(topk_indexes, class_count, rounding_mode="trunc")
        topk_local_classes = topk_indexes % class_count

        class_positions = stage2_class_positions[batch_idx].to(prob_to_token.device)
        labels = label_ids[class_positions[topk_local_classes]]
        boxes = boxes_xyxy[batch_idx][topk_boxes]

        if bbox_postprocessor.nms_iou_threshold > 0:
            keep = batched_nms(
                boxes,
                topk_values,
                labels,
                iou_threshold=bbox_postprocessor.nms_iou_threshold,
            )
            topk_values = topk_values[keep]
            labels = labels[keep]
            boxes = boxes[keep]

        results.append({"scores": topk_values, "labels": labels, "boxes": boxes})

    return results


@torch.no_grad()
def _postprocess_stage2_semantic_prompt(
    outputs,
    target_sizes,
    stage2_caption_lists,
    stage2_class_positions,
    bbox_postprocessor,
):
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    prob_to_token = out_logits.sigmoid()
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes_xyxy = boxes_xyxy * scale_fct[:, None, :]

    num_select = int(bbox_postprocessor.num_select)
    label_ids = bbox_postprocessor.label_ids.to(prob_to_token.device)
    results = []

    for batch_idx, cat_list in enumerate(stage2_caption_lists):
        caption, normalized_cat_list = build_caption_from_labels(cat_list)
        pos_map = _build_dynamic_positive_map(
            bbox_postprocessor.tokenizer,
            bbox_postprocessor.max_text_len,
            caption,
            normalized_cat_list,
            prob_to_token.device,
        )
        class_scores = prob_to_token[batch_idx] @ pos_map.T

        class_count = len(stage2_class_positions[batch_idx])
        if class_count == 0:
            results.append(
                {
                    "scores": torch.empty(0, device=prob_to_token.device),
                    "labels": torch.empty(0, dtype=torch.long, device=prob_to_token.device),
                    "boxes": torch.empty(0, 4, device=prob_to_token.device),
                }
            )
            continue

        topk_count = min(num_select, class_scores.numel())
        topk_values, topk_indexes = torch.topk(class_scores.reshape(-1), topk_count, dim=0)
        topk_boxes = torch.div(topk_indexes, class_count, rounding_mode="trunc")
        topk_local_classes = topk_indexes % class_count

        class_positions = stage2_class_positions[batch_idx].to(prob_to_token.device)
        labels = label_ids[class_positions[topk_local_classes]]
        boxes = boxes_xyxy[batch_idx][topk_boxes]

        if bbox_postprocessor.nms_iou_threshold > 0:
            keep = batched_nms(
                boxes,
                topk_values,
                labels,
                iou_threshold=bbox_postprocessor.nms_iou_threshold,
            )
            topk_values = topk_values[keep]
            labels = labels[keep]
            boxes = boxes[keep]

        results.append({"scores": topk_values, "labels": labels, "boxes": boxes})

    return results


def _build_stage2_mixed_prompt_chunks(
    top_positions,
    label_ids,
    semantic_label_list,
    description_label_list,
    tokenizer,
    max_text_len,
):
    chunks = []
    current_positions = []
    current_texts = []

    for position in top_positions:
        class_id = int(label_ids[int(position)].item())
        if class_id >= len(semantic_label_list) or class_id >= len(description_label_list):
            continue

        pair_texts = [
            str(semantic_label_list[class_id]),
            str(description_label_list[class_id]),
        ]
        candidate_texts = current_texts + pair_texts
        candidate_caption, _ = build_caption_from_labels(candidate_texts)
        candidate_token_len = len(tokenizer(candidate_caption)["input_ids"])

        if current_positions and candidate_token_len > max_text_len:
            caption, normalized_texts = build_caption_from_labels(current_texts)
            chunks.append(
                {
                    "positions": current_positions,
                    "texts": normalized_texts,
                    "caption": caption,
                }
            )
            current_positions = [int(position)]
            current_texts = pair_texts
        else:
            current_positions.append(int(position))
            current_texts = candidate_texts

    if current_positions:
        caption, normalized_texts = build_caption_from_labels(current_texts)
        chunks.append(
            {
                "positions": current_positions,
                "texts": normalized_texts,
                "caption": caption,
            }
        )

    return chunks


def _build_stage2_semantic_prompt_chunks(
    top_positions,
    label_ids,
    semantic_label_list,
    tokenizer,
    max_text_len,
):
    chunks = []
    current_positions = []
    current_texts = []

    for position in top_positions:
        class_id = int(label_ids[int(position)].item())
        if class_id >= len(semantic_label_list):
            continue

        candidate_text = str(semantic_label_list[class_id])
        candidate_texts = current_texts + [candidate_text]
        candidate_caption, _ = build_caption_from_labels(candidate_texts)
        candidate_token_len = len(tokenizer(candidate_caption)["input_ids"])

        if current_positions and candidate_token_len > max_text_len:
            caption, normalized_texts = build_caption_from_labels(current_texts)
            chunks.append(
                {
                    "positions": current_positions,
                    "texts": normalized_texts,
                    "caption": caption,
                }
            )
            current_positions = [int(position)]
            current_texts = [candidate_text]
        else:
            current_positions.append(int(position))
            current_texts = candidate_texts

    if current_positions:
        caption, normalized_texts = build_caption_from_labels(current_texts)
        chunks.append(
            {
                "positions": current_positions,
                "texts": normalized_texts,
                "caption": caption,
            }
        )

    return chunks


def _merge_stage2_chunk_results(chunk_results_by_image, num_select, nms_iou_threshold, device):
    merged_results = []
    for img_results in chunk_results_by_image:
        img_results = [r for r in img_results if r["scores"].numel() > 0]
        if not img_results:
            merged_results.append(
                {
                    "scores": torch.empty(0, device=device),
                    "labels": torch.empty(0, dtype=torch.long, device=device),
                    "boxes": torch.empty(0, 4, device=device),
                }
            )
            continue

        merged_scores = torch.cat([r["scores"] for r in img_results])
        merged_labels = torch.cat([r["labels"] for r in img_results])
        merged_boxes = torch.cat([r["boxes"] for r in img_results])

        if nms_iou_threshold > 0:
            keep = batched_nms(
                merged_boxes,
                merged_scores,
                merged_labels,
                iou_threshold=nms_iou_threshold,
            )
            merged_scores = merged_scores[keep]
            merged_labels = merged_labels[keep]
            merged_boxes = merged_boxes[keep]

        topk = min(int(num_select), merged_scores.shape[0])
        if topk > 0:
            topk_scores, topk_idx = torch.topk(merged_scores, topk)
            merged_results.append(
                {
                    "scores": topk_scores,
                    "labels": merged_labels[topk_idx],
                    "boxes": merged_boxes[topk_idx],
                }
            )
        else:
            merged_results.append(
                {
                    "scores": merged_scores,
                    "labels": merged_labels,
                    "boxes": merged_boxes,
                }
            )

    return merged_results


def _init_stage2_candidate_diagnostics(topk_values=(1, 5, 10, 20, 50)):
    return {
        "topk_values": tuple(int(k) for k in topk_values),
        "gt_total": 0,
        "wrong_total": 0,
        "wrong_seen_total": 0,
        "wrong_unseen_total": 0,
        "gt_hits": {int(k): 0 for k in topk_values},
        "wrong_hits": {int(k): 0 for k in topk_values},
        "wrong_seen_hits": {int(k): 0 for k in topk_values},
        "wrong_unseen_hits": {int(k): 0 for k in topk_values},
    }


@torch.no_grad()
def _update_stage2_candidate_diagnostics(
    diagnostics,
    per_image_scores,
    stage1_results,
    targets,
    seen_category_ids=None,
    iou_threshold=0.5,
):
    if diagnostics is None:
        return

    seen_category_ids = set(int(cat_id) for cat_id in (seen_category_ids or []))
    max_requested_topk = max(diagnostics["topk_values"])
    num_classes = per_image_scores.shape[1]
    max_topk = min(max_requested_topk, num_classes)
    if max_topk <= 0:
        return

    top_positions = per_image_scores.topk(k=max_topk, dim=1).indices

    for batch_idx, target in enumerate(targets):
        gt_labels = target.get("labels")
        gt_boxes = target.get("boxes")
        if gt_labels is None or gt_boxes is None or gt_labels.numel() == 0:
            continue

        gt_labels = gt_labels.to(device=top_positions.device, dtype=torch.long).view(-1)
        gt_boxes = gt_boxes.to(device=top_positions.device)
        gt_label_set = {int(label.item()) for label in gt_labels}
        diagnostics["gt_total"] += len(gt_label_set)

        image_top_positions = top_positions[batch_idx]
        for k in diagnostics["topk_values"]:
            k_eff = min(int(k), image_top_positions.numel())
            if k_eff <= 0:
                continue
            topk_set = {int(label.item()) for label in image_top_positions[:k_eff]}
            diagnostics["gt_hits"][int(k)] += sum(
                1 for gt_label in gt_label_set if gt_label in topk_set
            )

        stage1_result = stage1_results[batch_idx]
        pred_boxes = stage1_result.get("boxes")
        pred_labels = stage1_result.get("labels")
        if pred_boxes is None or pred_labels is None or pred_boxes.numel() == 0:
            continue

        pred_boxes = pred_boxes.to(device=top_positions.device)
        pred_labels = pred_labels.to(device=top_positions.device, dtype=torch.long).view(-1)
        gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        gt_size = target.get("orig_size", None)
        if gt_size is not None:
            gt_size = gt_size.to(device=top_positions.device)
            img_h, img_w = gt_size.unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            gt_boxes_xyxy = gt_boxes_xyxy * scale_fct

        pairwise_iou, _ = box_ops.box_iou(pred_boxes, gt_boxes_xyxy)
        if pairwise_iou.numel() == 0:
            continue

        best_pred_iou, best_pred_idx = pairwise_iou.max(dim=0)
        for gt_idx, (best_iou, pred_idx) in enumerate(zip(best_pred_iou, best_pred_idx)):
            if float(best_iou.item()) < float(iou_threshold):
                continue

            gt_label = int(gt_labels[gt_idx].item())
            pred_label = int(pred_labels[pred_idx].item())
            if pred_label == gt_label:
                continue

            diagnostics["wrong_total"] += 1
            is_seen = gt_label in seen_category_ids
            if is_seen:
                diagnostics["wrong_seen_total"] += 1
            else:
                diagnostics["wrong_unseen_total"] += 1

            for k in diagnostics["topk_values"]:
                k_eff = min(int(k), image_top_positions.numel())
                if k_eff <= 0:
                    continue
                gt_in_topk = bool((image_top_positions[:k_eff] == gt_label).any().item())
                if not gt_in_topk:
                    continue
                diagnostics["wrong_hits"][int(k)] += 1
                if is_seen:
                    diagnostics["wrong_seen_hits"][int(k)] += 1
                else:
                    diagnostics["wrong_unseen_hits"][int(k)] += 1


def _finalize_stage2_candidate_diagnostics(diagnostics):
    if diagnostics is None:
        return {}

    if utils.is_dist_avail_and_initialized():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        counts = [
            diagnostics["gt_total"],
            diagnostics["wrong_total"],
            diagnostics["wrong_seen_total"],
            diagnostics["wrong_unseen_total"],
        ]
        for k in diagnostics["topk_values"]:
            k = int(k)
            counts.extend(
                [
                    diagnostics["gt_hits"][k],
                    diagnostics["wrong_hits"][k],
                    diagnostics["wrong_seen_hits"][k],
                    diagnostics["wrong_unseen_hits"][k],
                ]
            )
        count_tensor = torch.tensor(counts, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(count_tensor)
        reduced_counts = [int(round(float(value))) for value in count_tensor.cpu().tolist()]
        cursor = 0
        diagnostics["gt_total"] = reduced_counts[cursor]
        cursor += 1
        diagnostics["wrong_total"] = reduced_counts[cursor]
        cursor += 1
        diagnostics["wrong_seen_total"] = reduced_counts[cursor]
        cursor += 1
        diagnostics["wrong_unseen_total"] = reduced_counts[cursor]
        cursor += 1
        for k in diagnostics["topk_values"]:
            k = int(k)
            diagnostics["gt_hits"][k] = reduced_counts[cursor]
            cursor += 1
            diagnostics["wrong_hits"][k] = reduced_counts[cursor]
            cursor += 1
            diagnostics["wrong_seen_hits"][k] = reduced_counts[cursor]
            cursor += 1
            diagnostics["wrong_unseen_hits"][k] = reduced_counts[cursor]
            cursor += 1

    stats = {}
    gt_total = max(int(diagnostics["gt_total"]), 1)
    wrong_total = max(int(diagnostics["wrong_total"]), 1)
    wrong_seen_total = max(int(diagnostics["wrong_seen_total"]), 1)
    wrong_unseen_total = max(int(diagnostics["wrong_unseen_total"]), 1)

    stats["stage2_candidate_gt_total"] = int(diagnostics["gt_total"])
    stats["stage2_candidate_wrong_total"] = int(diagnostics["wrong_total"])
    stats["stage2_candidate_wrong_seen_total"] = int(diagnostics["wrong_seen_total"])
    stats["stage2_candidate_wrong_unseen_total"] = int(diagnostics["wrong_unseen_total"])

    for k in diagnostics["topk_values"]:
        k = int(k)
        stats[f"stage2_candidate_gt_recall_top{k}"] = diagnostics["gt_hits"][k] / gt_total
        stats[f"stage2_candidate_wrong_gt_recall_top{k}"] = (
            diagnostics["wrong_hits"][k] / wrong_total
        )
        stats[f"stage2_candidate_wrong_seen_gt_recall_top{k}"] = (
            diagnostics["wrong_seen_hits"][k] / wrong_seen_total
        )
        stats[f"stage2_candidate_wrong_unseen_gt_recall_top{k}"] = (
            diagnostics["wrong_unseen_hits"][k] / wrong_unseen_total
        )

    return stats


def _resolve_dataset_manifest_path(datasets_arg):
    if not datasets_arg:
        return None
    manifest_path = Path(datasets_arg)
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    return manifest_path


def _resolve_annotation_path(path_value, base_dir):
    ann_path = Path(path_value)
    if not ann_path.is_absolute():
        ann_path = (base_dir / ann_path).resolve()
    return ann_path


def _extract_seen_category_ids_from_annotation(ann_path):
    ann_path = Path(ann_path)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    seen_cat_ids = set()
    suffix = ann_path.suffix.lower()
    if suffix == ".jsonl":
        with ann_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                for instance in sample.get("detection", {}).get("instances", []):
                    label = instance.get("label")
                    if label is not None:
                        seen_cat_ids.add(int(label))
        return seen_cat_ids

    if suffix == ".json":
        with ann_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for annotation in data.get("annotations", []):
            category_id = annotation.get("category_id")
            if category_id is not None:
                seen_cat_ids.add(int(category_id))
        return seen_cat_ids

    raise ValueError(f"Unsupported annotation format for seen/unseen split: {ann_path}")


def _extract_category_instance_counts_from_annotation(ann_path):
    ann_path = Path(ann_path)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    counts = {}
    suffix = ann_path.suffix.lower()
    if suffix == ".jsonl":
        with ann_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                for instance in sample.get("detection", {}).get("instances", []):
                    label = instance.get("label")
                    if label is None:
                        continue
                    label = int(label)
                    counts[label] = counts.get(label, 0) + 1
        return counts

    if suffix == ".json":
        with ann_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for annotation in data.get("annotations", []):
            category_id = annotation.get("category_id")
            if category_id is None:
                continue
            category_id = int(category_id)
            counts[category_id] = counts.get(category_id, 0) + 1
        return counts

    raise ValueError(f"Unsupported annotation format for category frequency split: {ann_path}")


def _extract_seen_category_ids_from_split_meta(split_meta_path):
    split_meta_path = Path(split_meta_path)
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {split_meta_path}")

    with split_meta_path.open("r", encoding="utf-8-sig") as handle:
        split_meta = json.load(handle)

    seen_ids = split_meta.get("seen_orig_ids")
    if seen_ids is not None:
        return {int(cat_id) for cat_id in seen_ids}

    class_table = split_meta.get("class_table", [])
    extracted = set()
    for row in class_table:
        if row.get("split") != "seen":
            continue
        if row.get("orig_id") is not None:
            extracted.add(int(row["orig_id"]))
        elif row.get("new_id") is not None:
            extracted.add(int(row["new_id"]))
    return extracted


def _infer_seen_unseen_category_subsets(args, coco_gt):
    datasets_manifest = _resolve_dataset_manifest_path(getattr(args, "datasets", None))
    if datasets_manifest is None or not datasets_manifest.exists():
        return {}

    try:
        with datasets_manifest.open("r", encoding="utf-8-sig") as handle:
            dataset_config = json.load(handle)
    except Exception as exc:
        print(f"Warning: failed to load dataset manifest for seen/unseen split: {exc}")
        return {}

    train_entries = dataset_config.get("train", [])
    if not train_entries:
        return {}

    explicit_seen_ids = set()
    explicit_seen_source = None
    explicit_seen_meta = getattr(args, "seen_category_meta_path", None)
    if explicit_seen_meta:
        explicit_seen_source = Path(explicit_seen_meta)
        try:
            explicit_seen_ids = _extract_seen_category_ids_from_split_meta(explicit_seen_source)
        except Exception as exc:
            print(
                f"Warning: failed to parse seen-category metadata from {explicit_seen_source}: {exc}"
            )
            explicit_seen_ids = set()

    if not explicit_seen_ids:
        for candidate_name in (
            "tt100k_noleak_rich_split_meta.json",
            "tt100k_noleak_split_meta.json",
            "tt100k_small_split_meta.json",
            "tt100k_debug_split_meta.json",
        ):
            try:
                candidate_paths = list(datasets_manifest.parent.rglob(candidate_name))
            except Exception:
                candidate_paths = []
            for candidate_path in candidate_paths:
                try:
                    candidate_seen_ids = _extract_seen_category_ids_from_split_meta(candidate_path)
                except Exception:
                    continue
                if not candidate_seen_ids:
                    continue
                explicit_seen_ids = candidate_seen_ids
                explicit_seen_source = candidate_path
                break
            if explicit_seen_ids:
                break

    if explicit_seen_ids:
        eval_annotations = getattr(coco_gt, "dataset", {}).get("annotations", [])
        eval_cat_ids = {
            int(annotation["category_id"])
            for annotation in eval_annotations
            if annotation.get("category_id") is not None
        }
        seen_eval_cat_ids = sorted(eval_cat_ids.intersection(explicit_seen_ids))
        unseen_eval_cat_ids = sorted(eval_cat_ids.difference(explicit_seen_ids))
        print(
            "Evaluation category split (original-id metadata): seen={} unseen={} source={}".format(
                len(seen_eval_cat_ids),
                len(unseen_eval_cat_ids),
                explicit_seen_source,
            )
        )
        return {"seen": seen_eval_cat_ids, "unseen": unseen_eval_cat_ids}

    seen_cat_ids = set()
    for entry in train_entries:
        ann_relpath = entry.get("anno")
        if not ann_relpath:
            continue
        ann_path = _resolve_annotation_path(ann_relpath, datasets_manifest.parent)
        try:
            seen_cat_ids.update(_extract_seen_category_ids_from_annotation(ann_path))
        except Exception as exc:
            print(f"Warning: failed to parse training annotations for seen/unseen split from {ann_path}: {exc}")

    if not seen_cat_ids:
        return {}

    eval_annotations = getattr(coco_gt, "dataset", {}).get("annotations", [])
    eval_cat_ids = {
        int(annotation["category_id"])
        for annotation in eval_annotations
        if annotation.get("category_id") is not None
    }

    seen_eval_cat_ids = sorted(eval_cat_ids.intersection(seen_cat_ids))
    unseen_eval_cat_ids = sorted(eval_cat_ids.difference(seen_cat_ids))
    if not seen_eval_cat_ids and not unseen_eval_cat_ids:
        return {}

    print(
        "Evaluation category split: seen={} unseen={}".format(
            len(seen_eval_cat_ids), len(unseen_eval_cat_ids)
        )
    )
    return {"seen": seen_eval_cat_ids, "unseen": unseen_eval_cat_ids}


def _parse_multi_bucket_defs(bucket_spec):
    """Parse a multi-bucket spec into [(label, lo, hi), ...].

    Format: ``"5-9,10-19,20-49,50-99,100-"`` where each segment is
    ``lo-hi`` (inclusive) or ``lo-`` (open upper bound).
    Buckets are evaluated in order; the first matching bucket wins.

    Accepts both a comma-separated string and a list (the latter
    occurs when the value reaches us via SLConfig DictAction, which
    splits on commas before forwarding).
    """
    if not bucket_spec:
        return []
    if isinstance(bucket_spec, list):
        # DictAction already split on commas → join back into a string.
        segments = [str(item).strip() for item in bucket_spec if str(item).strip()]
    elif isinstance(bucket_spec, str):
        segments = [seg.strip() for seg in bucket_spec.split(",") if seg.strip()]
    else:
        return []
    buckets = []
    for segment in segments:
        if segment.endswith("-"):
            lo_str = segment[:-1].strip()
            if not lo_str:
                continue
            lo = int(lo_str)
            buckets.append((segment, lo, None))
        elif "-" in segment:
            parts = segment.split("-", 1)
            lo = int(parts[0].strip())
            hi = int(parts[1].strip())
            label = "{}-{}".format(lo, hi)
            buckets.append((label, lo, hi))
    return buckets


def _infer_seen_frequency_category_subsets(args, coco_gt, seen_cat_ids=None):
    datasets_manifest = _resolve_dataset_manifest_path(getattr(args, "datasets", None))
    if datasets_manifest is None or not datasets_manifest.exists():
        return {}

    try:
        with datasets_manifest.open("r", encoding="utf-8-sig") as handle:
            dataset_config = json.load(handle)
    except Exception as exc:
        print("Warning: failed to load dataset manifest for seen-frequency split: {}".format(exc))
        return {}

    train_entries = dataset_config.get("train", [])
    if not train_entries:
        return {}

    instance_counts = {}
    for entry in train_entries:
        ann_relpath = entry.get("anno")
        if not ann_relpath:
            continue
        ann_path = _resolve_annotation_path(ann_relpath, datasets_manifest.parent)
        try:
            ann_counts = _extract_category_instance_counts_from_annotation(ann_path)
        except Exception as exc:
            print(
                "Warning: failed to parse training annotations for seen-frequency split from {}: {}".format(
                    ann_path, exc
                )
            )
            continue
        for cat_id, count in ann_counts.items():
            instance_counts[int(cat_id)] = instance_counts.get(int(cat_id), 0) + int(count)

    if not instance_counts:
        return {}

    eval_annotations = getattr(coco_gt, "dataset", {}).get("annotations", [])
    eval_cat_ids = {
        int(annotation["category_id"])
        for annotation in eval_annotations
        if annotation.get("category_id") is not None
    }
    if seen_cat_ids is None:
        seen_cat_ids = set(instance_counts)
    else:
        seen_cat_ids = {int(cat_id) for cat_id in seen_cat_ids}

    candidate_ids = eval_cat_ids.intersection(seen_cat_ids)

    # ---- multi-bucket mode ----
    multi_bucket_spec = getattr(args, "seen_freq_multi_buckets", None)
    if multi_bucket_spec:
        bucket_defs = _parse_multi_bucket_defs(multi_bucket_spec)
        if bucket_defs:
            subsets = {}
            assigned = set()
            summary_parts = []
            for label, lo, hi in bucket_defs:
                if hi is not None:
                    ids = sorted(
                        cat_id
                        for cat_id in candidate_ids
                        if lo <= instance_counts.get(cat_id, 0) <= hi
                    )
                else:
                    ids = sorted(
                        cat_id
                        for cat_id in candidate_ids
                        if instance_counts.get(cat_id, 0) >= lo
                    )
                subset_key = "freq_{}".format(label.replace("-", "_"))
                subsets[subset_key] = ids
                assigned.update(ids)
                summary_parts.append("{}={}".format(label, len(ids)))

            # Catch-all for categories that fell through the cracks
            # (e.g. < min(lo) when buckets don't start at 0).
            rest_ids = sorted(set(candidate_ids) - assigned)
            if rest_ids:
                subsets["freq_lt{}".format(bucket_defs[0][1])] = rest_ids
                summary_parts.insert(0, "lt{}={}".format(bucket_defs[0][1], len(rest_ids)))

            print(
                "Evaluation seen-frequency multi-bucket: {} seen_eval={} counted_classes={}".format(
                    " ".join(summary_parts),
                    len(candidate_ids),
                    len(instance_counts),
                )
            )
            return subsets

    # ---- legacy two-bucket mode ----
    high_threshold = int(getattr(args, "seen_freq_high_threshold", 100))
    low_threshold = int(getattr(args, "seen_freq_low_threshold", 50))
    high_ids = sorted(
        cat_id for cat_id in candidate_ids if instance_counts.get(cat_id, 0) >= high_threshold
    )
    low_ids = sorted(
        cat_id for cat_id in candidate_ids if instance_counts.get(cat_id, 0) < low_threshold
    )

    subsets = {}
    if high_ids:
        subsets["seen_train_ge100"] = high_ids
    if low_ids:
        subsets["seen_train_lt50"] = low_ids

    print(
        "Evaluation seen-frequency split: ge{}={} lt{}={} seen_eval={} counted_classes={}".format(
            high_threshold,
            len(high_ids),
            low_threshold,
            len(low_ids),
            len(candidate_ids),
            len(instance_counts),
        )
    )
    return subsets


def _infer_freqtier_combined_category_subsets(args, coco_gt):
    split_meta_path = getattr(args, "seen_category_meta_path", None)
    if not split_meta_path:
        return {}

    split_meta_path = Path(split_meta_path)
    if not split_meta_path.exists():
        return {}

    try:
        with split_meta_path.open("r", encoding="utf-8-sig") as handle:
            split_meta = json.load(handle)
    except Exception as exc:
        print("Warning: failed to load freqtier split metadata from {}: {}".format(split_meta_path, exc))
        return {}

    if split_meta.get("protocol") != "tt100k_freqtier_combined_fewshot":
        return {}

    eval_annotations = getattr(coco_gt, "dataset", {}).get("annotations", [])
    eval_cat_ids = {
        int(annotation["category_id"])
        for annotation in eval_annotations
        if annotation.get("category_id") is not None
    }
    if not eval_cat_ids:
        return {}

    def clean_ids(values):
        return sorted(eval_cat_ids.intersection({int(cat_id) for cat_id in (values or [])}))

    fewshot_groups = split_meta.get("fewshot_groups", {}) or {}

    def fewshot_class_ids(group_name):
        group = fewshot_groups.get(group_name, {})
        if isinstance(group, dict):
            return group.get("classes", [])
        return group

    subsets = {}
    head_ids = clean_ids(split_meta.get("head_orig_ids") or split_meta.get("seen_orig_ids"))
    mid_ids = clean_ids(split_meta.get("mid_candidate_ids"))
    tail_1shot_ids = clean_ids(fewshot_class_ids("tail_1shot"))
    tail_3shot_ids = clean_ids(fewshot_class_ids("tail_3shot"))
    mid_5shot_ids = clean_ids(fewshot_class_ids("mid_5shot"))
    mid_10shot_ids = clean_ids(fewshot_class_ids("mid_10shot"))

    if head_ids:
        subsets["head_ge100"] = head_ids
    if mid_ids:
        subsets["mid_50_99_all"] = mid_ids
    if tail_1shot_ids:
        subsets["tail_1shot"] = tail_1shot_ids
    if tail_3shot_ids:
        subsets["tail_3shot"] = tail_3shot_ids
    if mid_5shot_ids:
        subsets["mid_5shot"] = mid_5shot_ids
    if mid_10shot_ids:
        subsets["mid_10shot"] = mid_10shot_ids
    smallshot_3_5_10_ids = sorted(
        set(tail_3shot_ids) | set(mid_5shot_ids) | set(mid_10shot_ids)
    )
    if smallshot_3_5_10_ids:
        subsets["smallshot_3_5_10"] = smallshot_3_5_10_ids

    train_ids = set()
    train_records = split_meta.get("train_records", {}) or {}
    instances_by_class = train_records.get("instances_by_class", {}) or {}
    for cat_id in instances_by_class:
        train_ids.add(int(cat_id))
    if not train_ids:
        train_ids.update(int(cat_id) for cat_id in (split_meta.get("head_orig_ids") or []))
        train_ids.update(int(cat_id) for cat_id in (split_meta.get("unused_mid_full_orig_ids") or []))
        for group in fewshot_groups.values():
            group_ids = group.get("classes", []) if isinstance(group, dict) else group
            train_ids.update(int(cat_id) for cat_id in (group_ids or []))

    zero_shot_ids = sorted(eval_cat_ids.difference(train_ids))
    if zero_shot_ids:
        subsets["zero_shot"] = zero_shot_ids

    print(
        "Evaluation freqtier combined split: {} source={}".format(
            " ".join("{}={}".format(name, len(ids)) for name, ids in subsets.items()),
            split_meta_path,
        )
    )
    return subsets


def _filter_eval_category_subsets(args, category_subsets):
    subset_spec = getattr(args, "eval_category_subsets", None)
    if not subset_spec:
        return category_subsets
    if isinstance(subset_spec, str):
        requested = [item.strip() for item in subset_spec.split(",") if item.strip()]
    elif isinstance(subset_spec, (list, tuple, set)):
        requested = [str(item).strip() for item in subset_spec if str(item).strip()]
    else:
        return category_subsets
    if not requested:
        return category_subsets

    filtered = {
        name: category_subsets[name]
        for name in requested
        if name in category_subsets and category_subsets[name]
    }
    missing = [name for name in requested if name not in category_subsets]
    print(
        "Evaluation category subsets filtered: kept={} missing={}".format(
            ",".join(filtered.keys()) if filtered else "none",
            ",".join(missing) if missing else "none",
        )
    )
    return filtered


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_amp_dtype():
    amp_dtype = os.environ.get("GROUNDINGDINO_AMP_DTYPE", "fp16").strip().lower()
    if amp_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _amp_autocast(enabled: bool, amp_dtype: torch.dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=amp_dtype)
    return torch.cuda.amp.autocast(enabled=enabled)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _collect_non_finite_params(model: torch.nn.Module, max_items: int = 8):
    bad = []
    for name, param in model.named_parameters():
        if param is None:
            continue
        data = param.detach().data
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        if nan_count or inf_count:
            bad.append((name, int(nan_count), int(inf_count), data.numel()))
            if len(bad) >= max_items:
                break
    return bad


def _collect_non_finite_grads(model: torch.nn.Module, max_items: int = 8):
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        nan_count = torch.isnan(grad).sum().item()
        inf_count = torch.isinf(grad).sum().item()
        if nan_count or inf_count:
            bad.append((name, int(nan_count), int(inf_count), grad.numel()))
            if len(bad) >= max_items:
                break
    return bad


def _print_dgcl_grad_debug(model: torch.nn.Module):
    key_patterns = (
        "sail_last_class_embed.log_temperature",
        "sail_last_class_embed.query_adapter.out_proj",
        "sail_last_class_embed.text_adapter",
        "sail_last_class_embed.query_gate_proj",
        "sail_last_class_embed.text_gate_proj",
        "sail_last_class_embed.gate_score",
        "sail_last_class_embed.word_cocontribution",
    )
    trainable = 0
    with_grad = 0
    nonzero_grad = 0
    total_grad_norm_sq = 0.0
    samples = []
    key_samples = []
    for name, param in model.named_parameters():
        if "sail_last_class_embed" not in name:
            continue
        if param.requires_grad:
            trainable += 1
        if param.grad is None:
            continue
        with_grad += 1
        grad = torch.nan_to_num(param.grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = float(grad.norm().cpu())
        if grad_norm > 0.0:
            nonzero_grad += 1
        total_grad_norm_sq += grad_norm * grad_norm
        if len(samples) < 12:
            samples.append((name, grad_norm, bool(param.requires_grad)))
        if any(pattern in name for pattern in key_patterns):
            key_samples.append((name, grad_norm, bool(param.requires_grad)))
    print(
        "[dgcl-grad-debug] "
        f"trainable_sail_params={trainable} "
        f"with_grad={with_grad} nonzero_grad={nonzero_grad} "
        f"total_grad_norm={math.sqrt(total_grad_norm_sq):.6e}"
    )
    for name, grad_norm, requires_grad in samples:
        print(
            "[dgcl-grad-debug] "
            f"{name} requires_grad={requires_grad} grad_norm={grad_norm:.6e}"
        )
    for name, grad_norm, requires_grad in key_samples:
        print(
            "[dgcl-grad-debug][key] "
            f"{name} requires_grad={requires_grad} grad_norm={grad_norm:.6e}"
        )


def _print_self_facb_grad_debug(model: torch.nn.Module):
    key_patterns = (
        "feat_map.",
        "class_embed",
        "transformer.decoder.layers.4",
        "transformer.decoder.layers.5",
        "facb_symbol_align_proj",
        "self_facb_head",
    )
    trainable = 0
    with_grad = 0
    nonzero_grad = 0
    total_grad_norm_sq = 0.0
    samples = []
    key_samples = []
    group_patterns = {
        "feat_map": ("feat_map.",),
        "class_embed": ("class_embed",),
        "decoder4": ("transformer.decoder.layers.4",),
        "decoder5": ("transformer.decoder.layers.5",),
        "symbol_proj": ("facb_symbol_align_proj",),
        "self_facb_head": ("self_facb_head",),
    }
    group_stats = {
        group: {"trainable": 0, "with_grad": 0, "nonzero_grad": 0, "grad_norm_sq": 0.0}
        for group in group_patterns
    }
    for name, param in model.named_parameters():
        if not any(pattern in name for pattern in key_patterns):
            continue
        if param.requires_grad:
            trainable += 1
            for group, patterns in group_patterns.items():
                if any(pattern in name for pattern in patterns):
                    group_stats[group]["trainable"] += 1
        if param.grad is None:
            continue
        with_grad += 1
        grad = torch.nan_to_num(param.grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = float(grad.norm().cpu())
        if grad_norm > 0.0:
            nonzero_grad += 1
        total_grad_norm_sq += grad_norm * grad_norm
        for group, patterns in group_patterns.items():
            if any(pattern in name for pattern in patterns):
                group_stats[group]["with_grad"] += 1
                if grad_norm > 0.0:
                    group_stats[group]["nonzero_grad"] += 1
                group_stats[group]["grad_norm_sq"] += grad_norm * grad_norm
        if len(samples) < 16:
            samples.append((name, grad_norm, bool(param.requires_grad)))
        if any(pattern in name for pattern in key_patterns):
            key_samples.append((name, grad_norm, bool(param.requires_grad)))
    print(
        "[facb-grad-debug] "
        f"trainable_facb_related_params={trainable} "
        f"with_grad={with_grad} nonzero_grad={nonzero_grad} "
        f"total_grad_norm={math.sqrt(total_grad_norm_sq):.6e}"
    )
    for group, stats in group_stats.items():
        print(
            "[facb-grad-debug][group] "
            f"{group} trainable={stats['trainable']} "
            f"with_grad={stats['with_grad']} "
            f"nonzero_grad={stats['nonzero_grad']} "
            f"grad_norm={math.sqrt(stats['grad_norm_sq']):.6e}"
        )
    for name, grad_norm, requires_grad in samples:
        print(
            "[facb-grad-debug] "
            f"{name} requires_grad={requires_grad} grad_norm={grad_norm:.6e}"
        )
    for name, grad_norm, requires_grad in key_samples[:32]:
        print(
            "[facb-grad-debug][key] "
            f"{name} requires_grad={requires_grad} grad_norm={grad_norm:.6e}"
        )


def _print_facb_loss_debug(loss_dict, weight_dict):
    keys = [
        "loss_facb_symbol_align",
        "facb_symbol_align_nce",
        "facb_symbol_align_pairwise",
        "facb_symbol_align_rows",
        "facb_symbol_align_topm_valid_frac",
        "facb_symbol_align_valid_cands",
        "facb_symbol_align_score_std",
        "facb_symbol_align_roi_norm",
        "facb_symbol_align_proto_norm",
        "facb_symbol_align_acc",
        "loss_self_facb",
    ]
    parts = []
    for key in keys:
        value = loss_dict.get(key, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                text = f"{float(value.detach().float().cpu().item()):.6g}"
            else:
                text = f"tensor{tuple(value.shape)}"
        else:
            text = str(value)
        scale = weight_dict.get(key, None)
        if scale is not None:
            text = f"{text}*{scale}"
        parts.append(f"{key}={text}")
    if parts:
        print("[facb-loss-debug] " + " ".join(parts))


def _print_facb_direct_grad_debug(model: torch.nn.Module, loss_dict):
    loss = loss_dict.get("loss_facb_symbol_align", None)
    if loss is None or not isinstance(loss, torch.Tensor):
        return
    print(
        "[facb-direct-grad-debug] "
        f"loss_requires_grad={bool(loss.requires_grad)} "
        f"loss_grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn is not None else 'None'}"
    )
    if not loss.requires_grad:
        return
    model_without_ddp = _get_model_without_ddp(model)
    group_patterns = {
        "feat_map": ("feat_map.",),
        "symbol_proj": ("facb_symbol_align_proj",),
        "class_embed": ("class_embed",),
        "decoder5": ("transformer.decoder.layers.5",),
    }
    params = []
    owners = []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        for group, patterns in group_patterns.items():
            if any(pattern in name for pattern in patterns):
                params.append(param)
                owners.append(group)
                break
    if not params:
        return
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    stats = {
        group: {"params": 0, "with_grad": 0, "nonzero": 0, "norm_sq": 0.0}
        for group in group_patterns
    }
    for group, grad in zip(owners, grads):
        stats[group]["params"] += 1
        if grad is None:
            continue
        stats[group]["with_grad"] += 1
        grad = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(grad.norm().cpu())
        if norm > 0.0:
            stats[group]["nonzero"] += 1
        stats[group]["norm_sq"] += norm * norm
    for group, item in stats.items():
        print(
            "[facb-direct-grad-debug][group] "
            f"{group} params={item['params']} with_grad={item['with_grad']} "
            f"nonzero={item['nonzero']} grad_norm={math.sqrt(item['norm_sq']):.6e}"
        )


def _facb_symbol_align_params(model: torch.nn.Module):
    model_without_ddp = _get_model_without_ddp(model)
    params = []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("feat_map.") or "facb_symbol_align_proj" in name:
            params.append((name, param))
    return params


def _compute_facb_symbol_align_manual_grads(model: torch.nn.Module, loss_dict, weight_dict):
    loss = loss_dict.get("loss_facb_symbol_align", None)
    if loss is None or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return []
    named_params = _facb_symbol_align_params(model)
    if not named_params:
        return []
    params = [param for _, param in named_params]
    weighted_loss = loss * weight_dict.get("loss_facb_symbol_align", 1.0)
    grads = torch.autograd.grad(weighted_loss, params, retain_graph=True, allow_unused=True)
    manual_grads = [(name, param, grad) for (name, param), grad in zip(named_params, grads) if grad is not None]
    if os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1":
        stats = {
            "feat_map": {"params": 0, "nonzero": 0, "norm_sq": 0.0},
            "symbol_proj": {"params": 0, "nonzero": 0, "norm_sq": 0.0},
        }
        for name, _, grad in manual_grads:
            group = "feat_map" if name.startswith("feat_map.") else "symbol_proj"
            grad = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            norm = float(grad.norm().cpu())
            stats[group]["params"] += 1
            stats[group]["norm_sq"] += norm * norm
            if norm > 0.0:
                stats[group]["nonzero"] += 1
        for group, item in stats.items():
            print(
                "[facb-manual-grad-debug][computed] "
                f"{group} params={item['params']} nonzero={item['nonzero']} "
                f"grad_norm={math.sqrt(item['norm_sq']):.6e}"
            )
    return manual_grads


def _apply_facb_symbol_align_manual_grads(manual_grads):
    if not manual_grads:
        return
    stats = {
        "feat_map": {"params": 0, "nonzero": 0, "norm_sq": 0.0, "nonfinite_before": 0},
        "symbol_proj": {"params": 0, "nonzero": 0, "norm_sq": 0.0, "nonfinite_before": 0},
    }
    for name, param, grad in manual_grads:
        group = "feat_map" if name.startswith("feat_map.") else "symbol_proj"
        grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).to(
            device=param.device,
            dtype=param.dtype,
        )
        if param.grad is None:
            param.grad = grad.clone()
        else:
            existing = param.grad.detach()
            nonfinite = int((~torch.isfinite(existing)).sum().detach().cpu().item())
            stats[group]["nonfinite_before"] += nonfinite
            existing = torch.nan_to_num(existing, nan=0.0, posinf=0.0, neginf=0.0)
            param.grad = existing + grad
        current = torch.nan_to_num(param.grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(current.norm().cpu())
        stats[group]["params"] += 1
        stats[group]["norm_sq"] += norm * norm
        if norm > 0.0:
            stats[group]["nonzero"] += 1
    if os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1":
        for group, item in stats.items():
            print(
                "[facb-manual-grad-debug][applied] "
                f"{group} params={item['params']} nonzero={item['nonzero']} "
                f"grad_norm={math.sqrt(item['norm_sq']):.6e} "
                f"nonfinite_before={item['nonfinite_before']}"
            )


def _get_model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def _get_query_selector_stats(model):
    model_without_ddp = _get_model_without_ddp(model)
    transformer = getattr(model_without_ddp, "transformer", None)
    if transformer is None:
        return {}

    raw_stats = getattr(transformer, "latest_query_selector_stats", {}) or {}
    stats = {}
    for key, value in raw_stats.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                stats[key] = value.detach().item()
            else:
                stats[key] = value.detach().float().mean().item()
        elif isinstance(value, (int, float)):
            stats[key] = float(value)
    return stats


def _get_similarity_calibration_module(model):
    model_without_ddp = _get_model_without_ddp(model)
    class_embed = getattr(model_without_ddp, "class_embed", None)
    if class_embed is None or len(class_embed) == 0:
        return None
    return class_embed[0]


def _get_similarity_calibration_stats(model):
    contrastive_embed = _get_similarity_calibration_module(model)
    if contrastive_embed is None:
        return {}

    raw_stats = getattr(contrastive_embed, "latest_stats", {}) or {}
    stats = {}
    for key, value in raw_stats.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                stats[key] = value.detach().item()
            else:
                stats[key] = value.detach().float().mean().item()
        elif isinstance(value, (int, float)):
            stats[key] = float(value)
    return stats


def _compute_query_selector_grad_norm(model):
    model_without_ddp = _get_model_without_ddp(model)
    transformer = getattr(model_without_ddp, "transformer", None)
    query_selector = getattr(transformer, "query_selector", None) if transformer is not None else None
    if query_selector is None:
        return {}

    total_sq = 0.0
    score_head_sq = 0.0
    has_any = False
    score_head_has_any = False
    alpha_grad = None

    for name, param in query_selector.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq += grad_norm ** 2
        has_any = True
        if name.startswith("score_head."):
            score_head_sq += grad_norm ** 2
            score_head_has_any = True
        if name == "alpha_logit":
            alpha_grad = param.grad.detach().item()

    stats = {}
    if has_any:
        stats["query_selector_grad_norm"] = total_sq ** 0.5
    if score_head_has_any:
        stats["query_selector_score_head_grad_norm"] = score_head_sq ** 0.5
    if alpha_grad is not None:
        stats["query_selector_alpha_grad"] = alpha_grad
    return stats


def _compute_similarity_calibration_grad_norm(model):
    contrastive_embed = _get_similarity_calibration_module(model)
    if contrastive_embed is None:
        return {}

    total_sq = 0.0
    has_any = False
    if getattr(contrastive_embed, "similarity_calibration_mode", "") == "residual_gate":
        for name in ("gate_scale", "gate_logit_scale", "gate_logit_bias"):
            param = getattr(contrastive_embed, name, None)
            if param is None or param.grad is None:
                continue
            grad_norm = param.grad.detach().data.norm(2).item()
            total_sq += grad_norm ** 2
            has_any = True
    else:
        similarity_head = getattr(contrastive_embed, "similarity_head", None)
        if similarity_head is None:
            return {}
        for param in similarity_head.parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().data.norm(2).item()
            total_sq += grad_norm ** 2
            has_any = True

    if not has_any:
        return {}
    return {"similarity_head_grad_norm": total_sq ** 0.5}


def _build_caption_label_map(tokenized_batch, batch_index, label_texts, caption, max_text_len, device, dtype):
    if not label_texts:
        return None

    label_maps = []
    for label_idx in range(len(label_texts)):
        label_id = torch.tensor([label_idx], dtype=torch.long)
        per_label = create_positive_map(
            tokenized_batch[batch_index],
            label_id,
            label_texts,
            caption,
            max_text_len=max_text_len,
        )
        label_maps.append(per_label.squeeze(0))

    label_map = torch.stack(label_maps, dim=0).to(device=device, dtype=dtype)
    label_map = label_map / label_map.sum(dim=1, keepdim=True).clamp(min=1.0)
    return label_map


def _compute_zs_preserve_distill_loss(student_outputs, teacher_outputs, cap_list, captions, args):
    coef = float(getattr(args, "zs_preserve_distill_coef", 0.0) or 0.0)
    if coef <= 0:
        return None

    temperature = float(getattr(args, "zs_preserve_distill_temperature", 1.0) or 1.0)
    temperature = max(temperature, 1e-6)
    query_threshold = float(getattr(args, "zs_preserve_distill_query_threshold", 0.05) or 0.0)

    max_text_len = min(
        student_outputs["pred_logits"].shape[-1],
        teacher_outputs["pred_logits"].shape[-1],
    )
    student_probs_token = student_outputs["pred_logits"][..., :max_text_len].sigmoid()
    teacher_probs_token = teacher_outputs["pred_logits"][..., :max_text_len].sigmoid()

    total_loss = student_probs_token.new_tensor(0.0)
    total_weight = student_probs_token.new_tensor(0.0)

    for batch_index, label_texts in enumerate(cap_list):
        supervision_token_batch = student_outputs.get("supervision_token", student_outputs["token"])
        label_map = _build_caption_label_map(
            tokenized_batch=supervision_token_batch,
            batch_index=batch_index,
            label_texts=label_texts,
            caption=captions[batch_index],
            max_text_len=max_text_len,
            device=student_probs_token.device,
            dtype=student_probs_token.dtype,
        )
        if label_map is None:
            continue

        student_label_scores = student_probs_token[batch_index] @ label_map.T
        teacher_label_scores = teacher_probs_token[batch_index] @ label_map.T

        teacher_query_conf = teacher_label_scores.max(dim=-1).values.detach()
        if query_threshold > 0:
            selected_queries = teacher_query_conf >= query_threshold
        else:
            selected_queries = torch.ones_like(teacher_query_conf, dtype=torch.bool)

        if not selected_queries.any():
            selected_queries = teacher_query_conf.topk(k=1, dim=0).indices
            selected_queries_mask = torch.zeros_like(teacher_query_conf, dtype=torch.bool)
            selected_queries_mask[selected_queries] = True
            selected_queries = selected_queries_mask

        student_selected = student_label_scores[selected_queries]
        teacher_selected = teacher_label_scores[selected_queries].detach()
        query_weights = teacher_query_conf[selected_queries]

        teacher_dist = F.softmax(teacher_selected / temperature, dim=-1)
        student_log_dist = F.log_softmax(student_selected / temperature, dim=-1)
        per_query_kl = F.kl_div(student_log_dist, teacher_dist, reduction="none").sum(dim=-1)
        per_query_kl = per_query_kl * (temperature ** 2)

        total_loss = total_loss + (per_query_kl * query_weights).sum()
        total_weight = total_weight + query_weights.sum()

    if total_weight.item() <= 0:
        return student_probs_token.new_tensor(0.0)
    return total_loss / total_weight.clamp(min=1e-6)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None,
                    teacher_model: torch.nn.Module = None):
    amp_dtype = _resolve_amp_dtype()
    use_grad_scaler = args.amp and amp_dtype == torch.float16
    scaler = _make_grad_scaler(enabled=use_grad_scaler)
    nan_debug = _env_flag("GROUNDINGDINO_NAN_DEBUG")
    if nan_debug or _env_flag("GROUNDINGDINO_AUTOGRAD_ANOMALY"):
        torch.autograd.set_detect_anomaly(True)


    model.train()
    criterion.train()
    setattr(criterion, "current_epoch", int(epoch))
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('query_selector_alpha', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_prob_max', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_prob_entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_selected_prob_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_selected_score_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_ranking_score_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_ranking_score_std', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_score_head_grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('query_selector_alpha_grad', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_raw_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_raw_std', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_logit_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_logit_std', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_delta_mean', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_delta_std', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('similarity_head_grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(1, int(getattr(args, "train_print_freq", 50)))

    _cnt = 0
    model_without_ddp = _get_model_without_ddp(model)
    transformer = getattr(model_without_ddp, "transformer", None)
    query_selector = getattr(transformer, "query_selector", None) if transformer is not None else None
    if query_selector is not None:
        routing_start_epoch = int(getattr(args, "query_score_routing_start_epoch", 1))
        warmup_epochs = max(1, int(getattr(args, "query_score_alpha_warmup_epochs", 3)))
        alpha_target = float(getattr(args, "query_score_alpha_target", 0.3))
        if epoch < routing_start_epoch:
            query_selector.set_route_with_selector(False)
            query_selector.set_runtime_alpha(
                float(getattr(args, "query_score_pretrain_alpha", 0.0))
            )
        else:
            progress = min(
                1.0,
                float(epoch - routing_start_epoch + 1) / float(warmup_epochs),
            )
            query_selector.set_route_with_selector(True)
            query_selector.set_runtime_alpha(alpha_target * progress)


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        if nan_debug:
            pre_bad = _collect_non_finite_params(_get_model_without_ddp(model))
            if pre_bad:
                print("[nan-debug] Non-finite parameters found before forward:")
                for name, n_nan, n_inf, total in pre_bad:
                    print(f"[nan-debug] param={name} nan={n_nan} inf={n_inf} total={total}")
                raise RuntimeError("Model parameters are non-finite before forward")

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        forward_captions = [t.get("forward_caption", t["caption"]) for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        distill_captions = [t.get("distill_caption", t["caption"]) for t in targets]
        distill_cap_list = [t.get("distill_cap_list", t["cap_list"]) for t in targets]
        query_contrastive_captions = [
            t.get("query_contrastive_caption", t["caption"]) for t in targets
        ]
        aux_token_ce_captions = [
            t.get("aux_token_ce_caption", t["caption"]) for t in targets
        ]
        aux_token_ce_cap_list = [
            t.get("aux_token_ce_cap_list", t["cap_list"]) for t in targets
        ]
        appearance_roi_captions = [
            t.get("appearance_roi_caption", t["caption"]) for t in targets
        ]
        appearance_roi_cap_list = [
            t.get("appearance_roi_cap_list", t["cap_list"]) for t in targets
        ]
        appearance_caption_label_ids = [
            t.get("caption_label_ids", None) for t in targets
        ]
        if (
            _cnt == 0
            and epoch == 0
            and _env_flag("GROUNDINGDINO_DUAL_PROMPT_DEBUG")
            and utils.is_main_process()
        ):
            print("[dual-prompt-debug] forward_caption:", forward_captions[0][:1000])
            print("[dual-prompt-debug] supervision_caption:", captions[0][:1000])
            print("[dual-prompt-debug] distill_caption:", distill_captions[0][:1000])
            if cap_list and cap_list[0]:
                print("[dual-prompt-debug] supervision cap_list[:2]:", cap_list[0][:2])
        if (
            _cnt == 0
            and epoch == 0
            and _env_flag("GROUNDINGDINO_APPEARANCE_TEXT_DEBUG")
            and utils.is_main_process()
        ):
            print("[appearance-text-debug] appearance_roi_caption:", appearance_roi_captions[0])
            print(
                "[appearance-text-debug] appearance_roi_cap_list:",
                appearance_roi_cap_list[0],
            )
            print(
                "[appearance-text-debug] caption_label_ids:",
                appearance_caption_label_ids[0],
            )
            print("[appearance-text-debug] supervision_caption:", captions[0])
            print("[appearance-text-debug] forward_caption:", forward_captions[0])
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
            outputs = model(
                samples,
                captions=forward_captions,
                supervision_captions=captions,
                supervision_caption_lists=cap_list,
                query_contrastive_captions=query_contrastive_captions,
                appearance_captions=appearance_roi_captions,
                appearance_caption_label_ids=appearance_caption_label_ids,
                gt_targets_for_roi=targets,
            )
            appearance_alpha = outputs.get("appearance_alpha", None)
            appearance_logits = outputs.get("pred_logits_appearance", None)
            semantic_logits = outputs.get("pred_logits_semantic", None)
            if appearance_alpha is not None:
                metric_logger.update(appearance_alpha=appearance_alpha.mean())
            if appearance_logits is not None and semantic_logits is not None:
                appearance_alpha_scaled = appearance_logits
                if appearance_alpha is not None:
                    appearance_alpha_scaled = appearance_alpha * appearance_logits
                appearance_abs = torch.nan_to_num(
                    appearance_alpha_scaled.abs().mean(), nan=0.0, posinf=0.0, neginf=0.0
                )
                semantic_abs = torch.nan_to_num(
                    semantic_logits.abs().mean(), nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(1e-6)
                metric_logger.update(
                    appearance_logit_abs=appearance_abs,
                    semantic_logit_abs=semantic_abs,
                    appearance_sem_ratio=appearance_abs / semantic_abs,
                )
            proto_cls_scores = outputs.get("proto_cls_scores", None)
            if proto_cls_scores is not None:
                valid_scores = [
                    score.abs().mean() for score in proto_cls_scores if score.numel() > 0
                ]
                if valid_scores:
                    metric_logger.update(proto_cls_logit_abs=torch.stack(valid_scores).mean())
            loss_dict = criterion(
                outputs,
                targets,
                cap_list,
                captions,
                aux_cat_list=aux_token_ce_cap_list,
                aux_caption=aux_token_ce_captions,
            )
            if teacher_model is not None and bool(getattr(args, "use_zs_preserve_distill", False)):
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        samples,
                        captions=distill_captions,
                        supervision_captions=distill_captions,
                        appearance_captions=distill_captions,
                    )
                if distill_captions == forward_captions and distill_captions == captions:
                    student_distill_outputs = outputs
                else:
                    student_distill_outputs = model(
                        samples,
                        captions=distill_captions,
                        supervision_captions=distill_captions,
                        query_contrastive_captions=distill_captions,
                        appearance_captions=distill_captions,
                    )
                distill_loss = _compute_zs_preserve_distill_loss(
                    student_outputs=student_distill_outputs,
                    teacher_outputs=teacher_outputs,
                    cap_list=distill_cap_list,
                    captions=distill_captions,
                    args=args,
                )
                if distill_loss is not None:
                    loss_dict["loss_zs_preserve"] = distill_loss

            weight_dict = criterion.weight_dict
            if "loss_zs_preserve" in loss_dict:
                weight_dict = dict(weight_dict)
                weight_dict["loss_zs_preserve"] = float(
                    getattr(args, "zs_preserve_distill_coef", 0.0) or 0.0
                )

            # FACB symbol alignment is a training-time adaptation loss.  Give it
            # priority over the legacy isolated appearance-branch path so the
            # online symbol prototypes actually participate in backward().
            if (
                getattr(args, "use_facb_symbol_align_loss", False)
                and "loss_facb_symbol_align" in loss_dict
                and isinstance(loss_dict["loss_facb_symbol_align"], torch.Tensor)
                and loss_dict["loss_facb_symbol_align"].requires_grad
            ):
                symbol_only = os.environ.get("GROUNDINGDINO_FACB_SYMBOL_ALIGN_ONLY", "0") == "1"
                manual_symbol_grad = os.environ.get("GROUNDINGDINO_FACB_SYMBOL_ALIGN_MANUAL_GRAD", "0") == "1"
                if symbol_only:
                    backward_keys = ["loss_facb_symbol_align"]
                else:
                    backward_keys = [
                        k for k in loss_dict.keys()
                        if (
                            k in weight_dict
                            and isinstance(loss_dict[k], torch.Tensor)
                            and loss_dict[k].requires_grad
                            and not (manual_symbol_grad and k == "loss_facb_symbol_align")
                        )
                    ]
                    if "loss_facb_symbol_align" not in backward_keys:
                        backward_keys.append("loss_facb_symbol_align")
                if manual_symbol_grad:
                    backward_keys = [k for k in backward_keys if k != "loss_facb_symbol_align"]
                if backward_keys:
                    losses = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in backward_keys)
                else:
                    losses = loss_dict["loss_facb_symbol_align"] * 0.0
                if (
                    os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1"
                    and not getattr(train_one_epoch, "_facb_backward_debug_printed", False)
                ):
                    coeffs = ",".join(f"{k}:{float(weight_dict.get(k, 1.0)):.4g}" for k in backward_keys)
                    print(
                        "[facb-backward-debug] "
                        f"symbol_only={int(symbol_only)} manual_symbol_grad={int(manual_symbol_grad)} "
                        f"keys={','.join(backward_keys) if backward_keys else 'none'} coeffs={coeffs}"
                    )
                    train_one_epoch._facb_backward_debug_printed = True
            elif (
                getattr(args, "use_roi_appearance_branch", False)
                or getattr(args, "use_appearance_contrastive_head", False)
                or getattr(args, "self_facb_branch_only", False)
            ):
                active_key = (
                    "loss_self_facb"
                    if getattr(args, "self_facb_branch_only", False)
                    else "loss_appearance_contrastive"
                )
                if active_key in loss_dict and loss_dict[active_key].requires_grad:
                    losses = loss_dict[active_key] * weight_dict.get(active_key, 1.0)
                else:
                    losses = None  # no valid contrastive samples this step
            else:
                losses = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys() if k in weight_dict
                )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward pass — skip if no active losses have grad_fn
        # (e.g. branch-only step with no matched contrastive queries).
        if losses is not None and losses.requires_grad:
            if (
                os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1"
                and not getattr(train_one_epoch, "_facb_direct_grad_debug_printed", False)
            ):
                _print_facb_loss_debug(loss_dict, weight_dict)
                _print_facb_direct_grad_debug(model, loss_dict)
                train_one_epoch._facb_direct_grad_debug_printed = True
            # amp backward function
            if use_grad_scaler:
                optimizer.zero_grad()
                facb_symbol_manual_grads = None
                if (
                    getattr(args, "use_facb_symbol_align_loss", False)
                    and os.environ.get("GROUNDINGDINO_FACB_SYMBOL_ALIGN_MANUAL_GRAD", "0") == "1"
                ):
                    facb_symbol_manual_grads = _compute_facb_symbol_align_manual_grads(model, loss_dict, weight_dict)
                scaler.scale(losses).backward()
                grads_unscaled = False
                if facb_symbol_manual_grads:
                    scaler.unscale_(optimizer)
                    grads_unscaled = True
                    _apply_facb_symbol_align_manual_grads(facb_symbol_manual_grads)
                if (
                    os.environ.get("GROUNDINGDINO_DGCL_GRAD_DEBUG", "0") == "1"
                    and not getattr(train_one_epoch, "_dgcl_grad_debug_printed", False)
                ):
                    scaler.unscale_(optimizer)
                    grads_unscaled = True
                    _print_dgcl_grad_debug(_get_model_without_ddp(model))
                    train_one_epoch._dgcl_grad_debug_printed = True
                    if os.environ.get("GROUNDINGDINO_DGCL_GRAD_DEBUG_EXIT", "0") == "1":
                        raise SystemExit(0)
                if (
                    os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1"
                    and not getattr(train_one_epoch, "_facb_grad_debug_printed", False)
                ):
                    if not grads_unscaled:
                        scaler.unscale_(optimizer)
                        grads_unscaled = True
                    _print_facb_loss_debug(loss_dict, weight_dict)
                    _print_self_facb_grad_debug(_get_model_without_ddp(model))
                    train_one_epoch._facb_grad_debug_printed = True
                    if os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG_EXIT", "0") == "1":
                        raise SystemExit(0)
                if nan_debug:
                    bad_grads = _collect_non_finite_grads(_get_model_without_ddp(model))
                    if bad_grads:
                        print("[nan-debug] Non-finite gradients found before optimizer step:")
                        for name, n_nan, n_inf, total in bad_grads:
                            print(f"[nan-debug] grad={name} nan={n_nan} inf={n_inf} total={total}")
                        raise RuntimeError("Non-finite gradients detected")
                if max_norm > 0:
                    if not grads_unscaled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # original backward function
                optimizer.zero_grad()
                facb_symbol_manual_grads = None
                if (
                    getattr(args, "use_facb_symbol_align_loss", False)
                    and os.environ.get("GROUNDINGDINO_FACB_SYMBOL_ALIGN_MANUAL_GRAD", "0") == "1"
                ):
                    facb_symbol_manual_grads = _compute_facb_symbol_align_manual_grads(model, loss_dict, weight_dict)
                losses.backward()
                if facb_symbol_manual_grads:
                    _apply_facb_symbol_align_manual_grads(facb_symbol_manual_grads)
                if (
                    os.environ.get("GROUNDINGDINO_DGCL_GRAD_DEBUG", "0") == "1"
                    and not getattr(train_one_epoch, "_dgcl_grad_debug_printed", False)
                ):
                    _print_dgcl_grad_debug(_get_model_without_ddp(model))
                    train_one_epoch._dgcl_grad_debug_printed = True
                    if os.environ.get("GROUNDINGDINO_DGCL_GRAD_DEBUG_EXIT", "0") == "1":
                        raise SystemExit(0)
                if (
                    os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG", "0") == "1"
                    and not getattr(train_one_epoch, "_facb_grad_debug_printed", False)
                ):
                    _print_facb_loss_debug(loss_dict, weight_dict)
                    _print_self_facb_grad_debug(_get_model_without_ddp(model))
                    train_one_epoch._facb_grad_debug_printed = True
                    if os.environ.get("GROUNDINGDINO_FACB_GRAD_DEBUG_EXIT", "0") == "1":
                        raise SystemExit(0)
                if nan_debug:
                    bad_grads = _collect_non_finite_grads(_get_model_without_ddp(model))
                    if bad_grads:
                        print("[nan-debug] Non-finite gradients found before optimizer step:")
                        for name, n_nan, n_inf, total in bad_grads:
                            print(f"[nan-debug] grad={name} nan={n_nan} inf={n_inf} total={total}")
                        raise RuntimeError("Non-finite gradients detected")
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
        else:
            optimizer.zero_grad()

        if nan_debug:
            post_bad = _collect_non_finite_params(_get_model_without_ddp(model))
            if post_bad:
                print("[nan-debug] Non-finite parameters found after optimizer step:")
                for name, n_nan, n_inf, total in post_bad:
                    print(f"[nan-debug] param={name} nan={n_nan} inf={n_inf} total={total}")
                raise RuntimeError("Model parameters became non-finite after optimizer step")

        if args.onecyclelr:
            lr_scheduler.step()

        query_selector_stats = _get_query_selector_stats(model)
        query_selector_grad_stats = _compute_query_selector_grad_norm(model)
        similarity_calibration_stats = _get_similarity_calibration_stats(model)
        similarity_calibration_grad_stats = _compute_similarity_calibration_grad_norm(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if query_selector_stats:
            metric_logger.update(**query_selector_stats)
        if query_selector_grad_stats:
            metric_logger.update(**query_selector_grad_stats)
        if similarity_calibration_stats:
            metric_logger.update(**similarity_calibration_stats)
        if similarity_calibration_grad_stats:
            metric_logger.update(**similarity_calibration_grad_stats)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def _split_cat_list_into_chunks(cat_list, tokenizer, max_text_len):
    """Split cat_list so each chunk's caption fits within max_text_len tokens.

    Greedy left-to-right packing: keeps adding labels until the tokenized
    caption would exceed max_text_len, then starts a new chunk.
    """
    if not cat_list:
        return [cat_list]

    from util.caption_builder import build_caption_from_labels

    chunks = []
    i = 0
    while i < len(cat_list):
        chunk = [cat_list[i]]
        i += 1
        while i < len(cat_list):
            candidate = chunk + [cat_list[i]]
            caption, _ = build_caption_from_labels(candidate)
            token_ids = tokenizer(caption)["input_ids"]
            if len(token_ids) > max_text_len:
                break
            chunk = candidate
            i += 1
        chunks.append(chunk)
    return chunks


def _infer_same_group_category_subset(args, coco_gt):
    """Build a 'same_group' subset: classes that share a fine-group with >=1 sibling.

    Uses args.appearance_global_fine_group_ids (indexed by GLOBAL class index, which
    aligns positionally with sorted(coco.getCatIds()) — the same ordering PostProcess
    assumes).  A class is in the subset iff its fine-group id is shared by 2+ classes
    AMONG the eval categories (singletons are excluded — they have no same-group
    confuser, so they can't benefit from discriminative-token reweighting).

    Returns {} when no fine-group vector is available.
    """
    fine_group_ids = getattr(args, "appearance_global_fine_group_ids", None)
    if not fine_group_ids:
        return {}
    try:
        cat_ids = sorted(int(c) for c in coco_gt.getCatIds())
    except Exception:
        return {}
    if len(cat_ids) != len(fine_group_ids):
        print(
            f"Warning: same-group subset skipped — fine_group_ids ({len(fine_group_ids)}) "
            f"!= eval categories ({len(cat_ids)}); ordering cannot be aligned."
        )
        return {}
    # Count fine-group membership across eval categories.
    from collections import Counter
    group_count = Counter(int(g) for g in fine_group_ids)
    same_group_cat_ids = [
        cat_ids[idx]
        for idx, g in enumerate(fine_group_ids)
        if group_count[int(g)] >= 2
    ]
    if not same_group_cat_ids:
        return {}
    print(
        f"Evaluation same-group subset: {len(same_group_cat_ids)} classes in "
        f"multi-member fine-groups (of {len(cat_ids)} total)"
    )
    return {"same_group": sorted(same_group_cat_ids)}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()
    model_without_ddp = _get_model_without_ddp(model)
    transformer = getattr(model_without_ddp, "transformer", None)
    query_selector = getattr(transformer, "query_selector", None) if transformer is not None else None
    if query_selector is not None:
        # Runtime routing state is not stored in a checkpoint. Standalone --eval
        # must therefore enable the configured LGPR path explicitly.
        query_selector.set_route_with_selector(True)
        query_selector.set_runtime_alpha(
            float(getattr(args, "query_score_alpha_target", 1.0))
        )
        if utils.is_main_process():
            print(
                "[LGPR] evaluation routing enabled: "
                f"fusion={float(getattr(args, 'query_score_alpha_target', 1.0)):.3f}",
                flush=True,
            )
    amp_dtype = _resolve_amp_dtype()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    detection_dump_path = os.environ.get("GROUNDINGDINO_DUMP_DETECTIONS_JSON", "").strip()
    if detection_dump_path and utils.is_main_process():
        dump_path = Path(detection_dump_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("", encoding="utf-8")
    crm_baseline_dump_path = os.environ.get(
        "GROUNDINGDINO_DUMP_CRM_BASELINE_JSON", ""
    ).strip()
    if crm_baseline_dump_path and utils.is_main_process():
        crm_dump_path = Path(crm_baseline_dump_path)
        crm_dump_path.parent.mkdir(parents=True, exist_ok=True)
        crm_dump_path.write_text("", encoding="utf-8")
    facb_change_enabled = (
        os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") == "1"
        or bool(str(getattr(args, "self_facb_change_analysis_jsonl", "") or ""))
    )
    if facb_change_enabled:
        _apply_post_nms_self_facb_rerank._change_stats = _new_self_facb_change_stats()
        change_jsonl = str(getattr(args, "self_facb_change_analysis_jsonl", "") or "")
        if change_jsonl and utils.is_main_process():
            change_path = Path(change_jsonl)
            change_path.parent.mkdir(parents=True, exist_ok=True)
            change_path.write_text("", encoding="utf-8")
    else:
        _apply_post_nms_self_facb_rerank._change_stats = None
    eval_print_freq = max(1, int(getattr(args, "eval_print_freq", 50)))
    eval_compute_loss = bool(getattr(args, "eval_compute_loss", False))
    eval_compute_loss_semantic = bool(getattr(args, "eval_compute_loss_semantic", False))
    eval_compute_loss_train_text = bool(getattr(args, "eval_compute_loss_train_text", False))
    eval_loss_sources = []
    if eval_compute_loss:
        if eval_compute_loss_semantic or eval_compute_loss_train_text:
            if eval_compute_loss_semantic:
                eval_loss_sources.append("semantic")
            if eval_compute_loss_train_text:
                eval_loss_sources.append("train_text")
        else:
            eval_loss_sources.append("semantic")

    def _resolve_eval_loss_source_targets(loss_source, batch_size):
        def _pick_text(target, keys, default=None):
            for key in keys:
                value = target.get(key)
                if value is not None:
                    return value
            return default

        # Try to read per-sample fields from targets first; fall back to eval-level lists
        captions = [
            _pick_text(t, ("caption",), None)
            for t in targets
        ]
        cap_lists = [
            _pick_text(t, ("cap_list",), None)
            for t in targets
        ]
        if any(c is None for c in captions) or any(c is None for c in cap_lists):
            captions = [_eval_caption] * batch_size
            cap_lists = [_eval_cat_list[:] for _ in range(batch_size)]

        # For train_text source, use query_contrastive fields if available
        if loss_source == "train_text":
            qc_captions = [
                _pick_text(t, ("query_contrastive_caption",), None)
                for t in targets
            ]
            qc_cap_lists = [
                _pick_text(t, ("query_contrastive_cap_list",), None)
                for t in targets
            ]
            if all(c is not None for c in qc_captions) and all(c is not None for c in qc_cap_lists):
                captions = qc_captions
                cap_lists = qc_cap_lists

        return {"captions": captions, "cap_list": cap_lists}

    def _resolve_eval_appearance_targets(batch_size):
        captions = [t.get("appearance_roi_caption", None) for t in targets]
        if all(c is not None for c in captions):
            return captions

        appearance_eval_label_list = list(
            getattr(args, "appearance_eval_label_list", []) or getattr(args, "label_list", []) or []
        )
        if appearance_eval_label_list:
            appearance_eval_caption, _ = build_caption_from_labels(appearance_eval_label_list)
            return [appearance_eval_caption] * batch_size
        return [_eval_caption] * batch_size

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    category_subsets = _infer_seen_unseen_category_subsets(args, base_ds)
    category_subsets.update(
        _infer_seen_frequency_category_subsets(
            args, base_ds, seen_cat_ids=category_subsets.get("seen", None)
        )
    )
    category_subsets.update(_infer_freqtier_combined_category_subsets(args, base_ds))
    # Method A diagnostic: add a same-group subset so per-subset AP isolates whether
    # discriminative-token reweighting helps fine-grained (same-group) confusion
    # specifically, vs a generic global rerank.  No-op without a fine-group vector.
    category_subsets.update(_infer_same_group_category_subset(args, base_ds))
    category_subsets = _filter_eval_category_subsets(args, category_subsets)
    coco_evaluator = CocoGroundingEvaluator(
        base_ds, iou_types, useCats=useCats, category_subsets=category_subsets
    )
    stage1_coco_evaluator = None
    stage2_candidate_diagnostics = None



    _cnt = 0
    output_state_dict = {} # for debug only
    # Prefer args.label_list so evaluation prompt stays consistent with
    # training prompt text (e.g. fine-grained appearance descriptions).
    cat_list = list(getattr(args, "label_list", []) or [])
    if not cat_list:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [
            str(item.get("semantic", "")).strip() or str(item["name"]).strip()
            for item in category_dict
        ]
    caption, cat_list = build_caption_from_labels(cat_list)
    # Fallback for eval loss computation when per-sample targets lack caption/cap_list
    _eval_caption = caption
    _eval_cat_list = cat_list
    appearance_eval_label_list = list(
        getattr(args, "appearance_eval_label_list", []) or _eval_cat_list
    )
    appearance_eval_caption, _ = build_caption_from_labels(appearance_eval_label_list)
    # ---- Dual-prompt eval (scheme ②): forward=semantic, scoring=appearance ----
    # Training routes forward(cross-attn)=semantic and scoring/loss=appearance.  To
    # keep eval consistent, we feed a SEMANTIC caption as the forward stream while the
    # postprocessor (and supervision_captions) stay on appearance text — so the
    # decoder cross-attention sees the same semantic text it did at train time, and
    # pred_logits is still scored on the appearance tokens via dual_prompt_class_embed.
    # cat_list here is the appearance label_list (eval_prompt_text_source=visual_cue_text),
    # so the postprocessor positive_map already matches the appearance-scored logits.
    _dual_prompt_eval = bool(getattr(args, "use_dual_prompt_training", False))
    _semantic_eval_label_list = list(getattr(args, "semantic_label_list", []) or cat_list)
    if _dual_prompt_eval and len(_semantic_eval_label_list) == len(cat_list):
        _semantic_eval_caption, _ = build_caption_from_labels(_semantic_eval_label_list)
    else:
        _dual_prompt_eval = False
        _semantic_eval_caption = caption
        _semantic_eval_label_list = list(cat_list)
    print("Input text prompt:", caption)
    if _dual_prompt_eval and utils.is_main_process():
        print(
            "[dual-prompt-eval] forward(semantic)[:1]=", _semantic_eval_label_list[:1],
            "| scoring(appearance)[:1]=", cat_list[:1],
        )
    if _env_flag("GROUNDINGDINO_PROMPT_DEBUG") and utils.is_main_process():
        print(
            "[prompt-debug] eval_prompt_text_source:",
            getattr(args, "eval_prompt_text_source", None),
        )
        print("[prompt-debug] eval cat_list[:2]:", cat_list[:2])
    bbox_postprocessor = postprocessors['bbox']

    # ---- Method A (Phase 1): precompute GLOBAL token-DF over the full class set ----
    # Must run on the complete cat_list BEFORE any chunking so DF reflects all classes,
    # not a chunk-local subset.  No-op when postprocess_token_idf_lambda == 0.
    if float(getattr(bbox_postprocessor, "postprocess_token_idf_lambda", 0.0)) != 0.0:
        df = bbox_postprocessor.set_global_token_df(cat_list)
        if utils.is_main_process():
            print(
                f"[token-idf] lambda={bbox_postprocessor.postprocess_token_idf_lambda} "
                f"gamma={getattr(bbox_postprocessor, 'postprocess_token_idf_gamma', 1.0)} "
                f"global DF over {len(cat_list)} classes: {len(df)} distinct token ids"
            )

    _chunked_cat_lists = _split_cat_list_into_chunks(
        cat_list,
        bbox_postprocessor.tokenizer,
        bbox_postprocessor.max_text_len,
    )
    _chunked_appearance_cat_lists = []
    _appearance_chunk_offset = 0
    for _chunk_cat_list in _chunked_cat_lists:
        _chunk_len = len(_chunk_cat_list)
        _chunk_app = appearance_eval_label_list[
            _appearance_chunk_offset : _appearance_chunk_offset + _chunk_len
        ]
        if len(_chunk_app) != _chunk_len:
            _chunk_app = _chunk_cat_list
        _chunked_appearance_cat_lists.append(_chunk_app)
        _appearance_chunk_offset += _chunk_len
    # Semantic chunks aligned to the SAME class offsets, for the dual-prompt forward
    # stream (scheme ②).  cat_list is appearance, so chunks are sized by appearance
    # tokens; the semantic forward caption per chunk just needs the same class slice.
    _chunked_semantic_cat_lists = []
    _semantic_chunk_offset = 0
    for _chunk_cat_list in _chunked_cat_lists:
        _chunk_len = len(_chunk_cat_list)
        _chunk_sem = _semantic_eval_label_list[
            _semantic_chunk_offset : _semantic_chunk_offset + _chunk_len
        ]
        if len(_chunk_sem) != _chunk_len:
            _chunk_sem = _chunk_cat_list
        _chunked_semantic_cat_lists.append(_chunk_sem)
        _semantic_chunk_offset += _chunk_len
    _use_chunked_eval = len(_chunked_cat_lists) > 1
    stage2_candidate_rerank_enabled = False
    stage2_semantic_topk_enabled = False
    if _use_chunked_eval:
        if bool(getattr(args, "use_self_facb_crm", False)):
            raise RuntimeError(
                "FACB-CRM requires an unchunked full-class evaluation caption so "
                "natural top-K is computed over the complete category space. "
                "Increase max_text_len or shorten the evaluation labels."
            )
        print(
            f"Chunked eval: {len(cat_list)} labels split into "
            f"{len(_chunked_cat_lists)} chunks "
            f"(max_text_len={bbox_postprocessor.max_text_len})"
        )
        # stage2 rerank is incompatible with chunked eval
        stage2_rerank_enabled = False
        stage2_candidate_rerank_enabled = False
        stage2_semantic_topk_enabled = False
    else:
        bbox_postprocessor.update_cat_list(cat_list)
        if _env_flag("GROUNDINGDINO_PROMPT_DEBUG") and utils.is_main_process():
            pos_map = bbox_postprocessor.positive_map
            zero_rows = int((pos_map.sum(dim=1) == 0).sum().item())
            print(
                "[prompt-debug] bbox_postprocessor positive_map shape:",
                tuple(pos_map.shape),
                "zero_rows=",
                zero_rows,
            )
        stage2_rerank_enabled = bool(getattr(args, "use_stage2_mixed_prompt_inference", False))
        stage2_candidate_rerank_enabled = bool(
            getattr(args, "use_stage2_candidate_rerank", False)
        )
        stage2_semantic_topk_enabled = bool(
            getattr(args, "use_stage2_semantic_topk_inference", False)
        )

    stage2_topk = max(1, int(getattr(args, "stage2_mixed_prompt_topk", 20)))
    stage2_sem_weight = float(getattr(args, "stage2_mixed_prompt_sem_weight", 0.7))
    stage2_desc_weight = float(getattr(args, "stage2_mixed_prompt_desc_weight", 0.3))
    semantic_label_list = list(getattr(args, "semantic_label_list", []) or cat_list)
    description_label_list = list(getattr(args, "description_label_list", []) or semantic_label_list)
    stage2_candidate_rerank_topk = max(
        1, int(getattr(args, "stage2_candidate_rerank_topk", 10))
    )
    stage2_candidate_rerank_query_topk = max(
        int(bbox_postprocessor.num_select),
        int(getattr(args, "stage2_candidate_rerank_query_topk", bbox_postprocessor.num_select)),
    )
    stage2_candidate_rerank_label_texts = _resolve_stage2_rerank_label_texts(args, cat_list)
    stage2_candidate_rerank_base_weight = float(
        getattr(args, "stage2_candidate_rerank_base_weight", 0.8)
    )
    stage2_candidate_rerank_alt_weight = float(
        getattr(args, "stage2_candidate_rerank_alt_weight", 0.2)
    )
    if stage2_rerank_enabled:
        print(
            "Stage2 mixed prompt inference: "
            f"topk={stage2_topk}, sem_weight={stage2_sem_weight}, desc_weight={stage2_desc_weight}"
        )
        stage1_coco_evaluator = CocoGroundingEvaluator(
            base_ds, iou_types, useCats=useCats, category_subsets=category_subsets
        )
        stage2_candidate_diagnostics = _init_stage2_candidate_diagnostics()
    elif stage2_semantic_topk_enabled:
        print(f"Stage2 semantic topk inference: topk={stage2_topk}")
        stage1_coco_evaluator = CocoGroundingEvaluator(
            base_ds, iou_types, useCats=useCats, category_subsets=category_subsets
        )
        stage2_candidate_diagnostics = _init_stage2_candidate_diagnostics()
    elif stage2_candidate_rerank_enabled:
        print(
            "Stage2 candidate rerank: "
            f"class_topk={stage2_candidate_rerank_topk}, "
            f"query_topk={stage2_candidate_rerank_query_topk}, "
            f"prompt_source={getattr(args, 'stage2_candidate_rerank_prompt_source', 'semantic')}, "
            f"base_weight={stage2_candidate_rerank_base_weight}, "
            f"alt_weight={stage2_candidate_rerank_alt_weight}"
        )
        stage1_coco_evaluator = CocoGroundingEvaluator(
            base_ds, iou_types, useCats=useCats, category_subsets=category_subsets
        )
        stage2_candidate_diagnostics = _init_stage2_candidate_diagnostics()

    for samples, targets in metric_logger.log_every(data_loader, eval_print_freq, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_image_ids = torch.as_tensor(
            [
                int(
                    t["image_id"].item()
                    if hasattr(t.get("image_id", None), "item")
                    else t.get("image_id", -1)
                )
                for t in targets
            ],
            dtype=torch.long,
        )

        if _use_chunked_eval:
            # --- chunked: run model once per chunk, merge results ---
            num_select = bbox_postprocessor.num_select
            all_chunk_results = [[] for _ in range(bs)]

            for chunk_cat_list, chunk_appearance_cat_list, chunk_semantic_cat_list in zip(
                _chunked_cat_lists, _chunked_appearance_cat_lists, _chunked_semantic_cat_lists
            ):
                chunk_caption, _norm = build_caption_from_labels(chunk_cat_list)
                chunk_appearance_caption, _ = build_caption_from_labels(
                    chunk_appearance_cat_list
                )
                chunk_semantic_caption, _ = build_caption_from_labels(
                    chunk_semantic_cat_list
                )
                bbox_postprocessor.update_cat_list(chunk_cat_list)
                input_captions = [chunk_caption] * bs

                with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                    if _dual_prompt_eval:
                        # Scheme ②: forward=semantic chunk, scoring=appearance chunk.
                        chunk_outputs = model(
                            samples,
                            facb_image_paths=[t.get("image_path") for t in targets],
                            captions=[chunk_semantic_caption] * bs,
                            supervision_captions=input_captions,
                            supervision_caption_lists=[chunk_cat_list] * bs,
                            appearance_captions=[chunk_appearance_caption] * bs,
                            appearance_caption_label_ids=[list(range(len(chunk_cat_list)))] * bs,
                        )
                    else:
                        chunk_outputs = model(
                            samples,
                            facb_image_paths=[t.get("image_path") for t in targets],
                            captions=input_captions,
                            supervision_caption_lists=[chunk_cat_list] * bs,
                            appearance_captions=[chunk_appearance_caption] * bs,
                            appearance_caption_label_ids=[list(range(len(chunk_cat_list)))] * bs,
                        )
                chunk_outputs["image_ids"] = batch_image_ids

                chunk_results = bbox_postprocessor(chunk_outputs, orig_target_sizes)
                if _env_flag("GROUNDINGDINO_PROMPT_DEBUG") and utils.is_main_process() and _cnt == 0:
                    sample_labels = []
                    if chunk_results and chunk_results[0]["labels"].numel() > 0:
                        sample_labels = chunk_results[0]["labels"][:5].detach().cpu().tolist()
                    print(
                        "[prompt-debug] chunked eval:",
                        f"use_coco_eval={getattr(args, 'use_coco_eval', False)}",
                        f"chunk_size={len(chunk_cat_list)}",
                        f"sample_labels={sample_labels}",
                        f"label_ids_head={bbox_postprocessor.label_ids[:5].detach().cpu().tolist()}",
                    )

                for img_idx, r in enumerate(chunk_results):
                    all_chunk_results[img_idx].append(r)

            # Merge per-chunk results: concat and take top num_select by score
            results = []
            for img_results in all_chunk_results:
                merged_scores = torch.cat([r["scores"] for r in img_results])
                merged_labels = torch.cat([r["labels"] for r in img_results])
                merged_boxes = torch.cat([r["boxes"] for r in img_results])
                topk = min(num_select, merged_scores.shape[0])
                if topk > 0:
                    topk_scores, topk_idx = torch.topk(merged_scores, topk)
                    results.append({
                        "scores": topk_scores,
                        "labels": merged_labels[topk_idx],
                        "boxes": merged_boxes[topk_idx],
                    })
                else:
                    results.append({
                        "scores": merged_scores,
                        "labels": merged_labels,
                        "boxes": merged_boxes,
                    })
        else:
            input_captions = [caption] * bs
            with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                if _dual_prompt_eval:
                    # Scheme ②: forward(cross-attn)=semantic, scoring=appearance.
                    # supervision_captions drives pred_logits via dual_prompt_class_embed;
                    # the postprocessor positive_map (built from cat_list=appearance)
                    # already matches that scoring space.
                    outputs = model(
                        samples,
                        facb_image_paths=[t.get("image_path") for t in targets],
                        captions=[_semantic_eval_caption] * bs,
                        supervision_captions=input_captions,
                        supervision_caption_lists=[cat_list] * bs,
                        appearance_captions=[appearance_eval_caption] * bs,
                        appearance_caption_label_ids=[list(range(len(cat_list)))] * bs,
                    )
                else:
                    outputs = model(
                        samples,
                        facb_image_paths=[t.get("image_path") for t in targets],
                        captions=input_captions,
                        supervision_caption_lists=[cat_list] * bs,
                        appearance_captions=[appearance_eval_caption] * bs,
                        appearance_caption_label_ids=[list(range(len(cat_list)))] * bs,
                    )
            outputs["image_ids"] = batch_image_ids

            if (
                crm_baseline_dump_path
                and bool(getattr(args, "use_self_facb_crm", False))
                and utils.is_main_process()
            ):
                baseline_results = bbox_postprocessor(outputs, orig_target_sizes)
                _dump_detection_results_jsonl(
                    crm_baseline_dump_path,
                    targets,
                    baseline_results,
                )
            with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                outputs = _attach_pre_nms_self_facb_crm(
                    outputs,
                    bbox_postprocessor,
                    model,
                    args,
                )
            outputs["image_ids"] = batch_image_ids
            results = bbox_postprocessor(outputs, orig_target_sizes)

        for eval_loss_source in eval_loss_sources:
            loss_inputs = _resolve_eval_loss_source_targets(eval_loss_source, bs)
            loss_captions = loss_inputs["captions"]
            loss_cap_list = loss_inputs["cap_list"]
            if not any(len(cap_list) > 0 for cap_list in loss_cap_list):
                continue
            loss_aux_captions = loss_captions
            loss_aux_cap_list = loss_cap_list
            with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                loss_outputs = model(
                    samples,
                    captions=loss_captions,
                    supervision_captions=loss_captions,
                    query_contrastive_captions=loss_captions,
                    appearance_captions=_resolve_eval_appearance_targets(bs),
                    appearance_caption_label_ids=[t.get("caption_label_ids", None) for t in targets],
                    return_loss_aux_outputs=True,
                )
                loss_dict = criterion(
                    loss_outputs,
                    targets,
                    loss_cap_list,
                    loss_captions,
                    aux_cat_list=loss_aux_cap_list,
                    aux_caption=loss_aux_captions,
                )
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f"{eval_loss_source}_{k}_unscaled": v for k, v in loss_dict_reduced.items()
            }
            weight_dict = criterion.weight_dict
            loss_dict_reduced_scaled = {
                f"{eval_loss_source}_{k}": v * weight_dict[k]
                for k, v in loss_dict_reduced.items()
                if k in weight_dict
            }
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            metric_logger.update(
                **{f"eval_loss_{eval_loss_source}": losses_reduced_scaled},
                **loss_dict_reduced_scaled,
                **loss_dict_reduced_unscaled,
            )

        stage1_results = results
        if stage2_rerank_enabled or stage2_candidate_rerank_enabled or stage2_semantic_topk_enabled:
            stage1_res = {
                target['image_id'].item(): output
                for target, output in zip(targets, stage1_results)
            }
            stage1_coco_evaluator.update(stage1_res)

            prob_to_token = outputs["pred_logits"].sigmoid()
            base_pos_map = _normalize_positive_map(
                bbox_postprocessor.positive_map,
                prob_to_token.device,
            )
            prob_to_label = prob_to_token @ base_pos_map.T
            per_image_scores = prob_to_label.amax(dim=1)
            _update_stage2_candidate_diagnostics(
                diagnostics=stage2_candidate_diagnostics,
                per_image_scores=per_image_scores,
                stage1_results=stage1_results,
                targets=targets,
                seen_category_ids=category_subsets.get("seen", []),
                iou_threshold=float(getattr(args, "stage2_candidate_diag_iou", 0.5)),
            )
            if stage2_rerank_enabled:
                stage2_chunks_by_image = []
                max_stage2_chunks = 0

                for batch_idx in range(bs):
                    k = min(stage2_topk, per_image_scores.shape[1])
                    top_positions = per_image_scores[batch_idx].topk(k=k, dim=0).indices.cpu()
                    chunks = _build_stage2_mixed_prompt_chunks(
                        top_positions=top_positions.tolist(),
                        label_ids=bbox_postprocessor.label_ids,
                        semantic_label_list=semantic_label_list,
                        description_label_list=description_label_list,
                        tokenizer=bbox_postprocessor.tokenizer,
                        max_text_len=bbox_postprocessor.max_text_len,
                    )
                    stage2_chunks_by_image.append(chunks)
                    max_stage2_chunks = max(max_stage2_chunks, len(chunks))

                if max_stage2_chunks > 1 and _cnt == 0:
                    print(
                        "Stage2 mixed prompt chunked rerank: "
                        f"max_chunks_per_image={max_stage2_chunks}, topk={stage2_topk}, "
                        f"max_text_len={bbox_postprocessor.max_text_len}"
                    )

                stage2_chunk_results_by_image = [[] for _ in range(bs)]
                for chunk_idx in range(max_stage2_chunks):
                    active_batch_indices = []
                    stage2_caption_lists = []
                    stage2_class_positions = []
                    stage2_input_captions = []

                    for batch_idx, image_chunks in enumerate(stage2_chunks_by_image):
                        if chunk_idx >= len(image_chunks):
                            continue
                        chunk = image_chunks[chunk_idx]
                        active_batch_indices.append(batch_idx)
                        stage2_caption_lists.append(chunk["texts"])
                        stage2_class_positions.append(
                            torch.tensor(chunk["positions"], dtype=torch.long)
                        )
                        stage2_input_captions.append(chunk["caption"])

                    if not active_batch_indices:
                        continue

                    active_samples = utils.NestedTensor(
                        samples.tensors[active_batch_indices],
                        samples.mask[active_batch_indices] if samples.mask is not None else None,
                    )
                    active_target_sizes = orig_target_sizes[active_batch_indices]
                    with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                        stage2_outputs = model(
                            active_samples,
                            captions=stage2_input_captions,
                            appearance_captions=[appearance_eval_caption] * len(active_batch_indices),
                        )
                    chunk_results = _postprocess_stage2_mixed_prompt(
                        stage2_outputs,
                        active_target_sizes,
                        stage2_caption_lists,
                        stage2_class_positions,
                        bbox_postprocessor,
                        sem_weight=stage2_sem_weight,
                        desc_weight=stage2_desc_weight,
                    )
                    for local_idx, batch_idx in enumerate(active_batch_indices):
                        stage2_chunk_results_by_image[batch_idx].append(chunk_results[local_idx])

                results = _merge_stage2_chunk_results(
                    stage2_chunk_results_by_image,
                    bbox_postprocessor.num_select,
                    bbox_postprocessor.nms_iou_threshold,
                    orig_target_sizes.device,
                )
            elif stage2_semantic_topk_enabled:
                stage2_chunks_by_image = []
                max_stage2_chunks = 0

                for batch_idx in range(bs):
                    k = min(stage2_topk, per_image_scores.shape[1])
                    top_positions = per_image_scores[batch_idx].topk(k=k, dim=0).indices.cpu()
                    chunks = _build_stage2_semantic_prompt_chunks(
                        top_positions=top_positions.tolist(),
                        label_ids=bbox_postprocessor.label_ids,
                        semantic_label_list=semantic_label_list,
                        tokenizer=bbox_postprocessor.tokenizer,
                        max_text_len=bbox_postprocessor.max_text_len,
                    )
                    stage2_chunks_by_image.append(chunks)
                    max_stage2_chunks = max(max_stage2_chunks, len(chunks))

                if max_stage2_chunks > 1 and _cnt == 0:
                    print(
                        "Stage2 semantic chunked rerank: "
                        f"max_chunks_per_image={max_stage2_chunks}, topk={stage2_topk}, "
                        f"max_text_len={bbox_postprocessor.max_text_len}"
                    )

                stage2_chunk_results_by_image = [[] for _ in range(bs)]
                for chunk_idx in range(max_stage2_chunks):
                    active_batch_indices = []
                    stage2_caption_lists = []
                    stage2_class_positions = []
                    stage2_input_captions = []

                    for batch_idx, image_chunks in enumerate(stage2_chunks_by_image):
                        if chunk_idx >= len(image_chunks):
                            continue
                        chunk = image_chunks[chunk_idx]
                        active_batch_indices.append(batch_idx)
                        stage2_caption_lists.append(chunk["texts"])
                        stage2_class_positions.append(
                            torch.tensor(chunk["positions"], dtype=torch.long)
                        )
                        stage2_input_captions.append(chunk["caption"])

                    if not active_batch_indices:
                        continue

                    active_samples = utils.NestedTensor(
                        samples.tensors[active_batch_indices],
                        samples.mask[active_batch_indices] if samples.mask is not None else None,
                    )
                    active_target_sizes = orig_target_sizes[active_batch_indices]
                    with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                        stage2_outputs = model(active_samples, captions=stage2_input_captions)
                    chunk_results = _postprocess_stage2_semantic_prompt(
                        stage2_outputs,
                        active_target_sizes,
                        stage2_caption_lists,
                        stage2_class_positions,
                        bbox_postprocessor,
                    )
                    for local_idx, batch_idx in enumerate(active_batch_indices):
                        stage2_chunk_results_by_image[batch_idx].append(chunk_results[local_idx])

                results = _merge_stage2_chunk_results(
                    stage2_chunk_results_by_image,
                    bbox_postprocessor.num_select,
                    bbox_postprocessor.nms_iou_threshold,
                    orig_target_sizes.device,
                )
            else:
                alt_caption, alt_cat_list = build_caption_from_labels(stage2_candidate_rerank_label_texts)
                if alt_caption:
                    with _amp_autocast(enabled=args.amp, amp_dtype=amp_dtype):
                        alt_outputs = model(
                            samples,
                            captions=[alt_caption] * bs,
                            appearance_captions=[appearance_eval_caption] * bs,
                        )
                    alt_prob_to_token = alt_outputs["pred_logits"].sigmoid()
                    alt_pos_map = _build_dynamic_positive_map(
                        bbox_postprocessor.tokenizer,
                        bbox_postprocessor.max_text_len,
                        alt_caption,
                        alt_cat_list,
                        alt_prob_to_token.device,
                    )
                    alt_prob_to_label = alt_prob_to_token @ alt_pos_map.T
                    results = _rerank_stage1_results_locally(
                        stage1_results=stage1_results,
                        base_prob_to_label=prob_to_label,
                        alt_prob_to_label=alt_prob_to_label,
                        bbox_postprocessor=bbox_postprocessor,
                        class_topk=stage2_candidate_rerank_topk,
                        query_topk=stage2_candidate_rerank_query_topk,
                        base_weight=stage2_candidate_rerank_base_weight,
                        alt_weight=stage2_candidate_rerank_alt_weight,
                    )
                else:
                    results = stage1_results
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        target_sizes = None
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        enable_facb_engine_hooks = bool(
            getattr(args, "enable_facb_engine_hooks", False)
            or getattr(args, "enable_self_facb_engine_hooks", False)
        )
        if enable_facb_engine_hooks:
            results = _apply_post_nms_facb_rerank(
                results, outputs, bbox_postprocessor, args
            )
            results = _apply_post_nms_self_facb_rerank(
                results, outputs, bbox_postprocessor, model, args, orig_target_sizes, targets
            )
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if _env_flag("GTSRB_EVAL_DEBUG") and utils.is_main_process() and _cnt == 0:
            _t0 = targets[0]
            _r0 = results[0]
            _sc = _r0["scores"].detach().cpu()
            _lb = _r0["labels"].detach().cpu()
            _bx = _r0["boxes"].detach().cpu()
            _order = _sc.argsort(descending=True)[:5]
            print("[gtsrb-debug] === first image ===")
            print("[gtsrb-debug] orig_size(h,w)=", _t0["orig_size"].tolist(),
                  "image_id=", int(_t0["image_id"].item()))
            print("[gtsrb-debug] GT labels=", _t0["labels"].cpu().tolist())
            print("[gtsrb-debug] GT boxes(xyxy abs)=", _t0.get("boxes_xyxy_abs", "n/a"),
                  "| GT boxes(raw target)=", _t0["boxes"].cpu().tolist())
            print("[gtsrb-debug] pred top5 scores=", _sc[_order].tolist())
            print("[gtsrb-debug] pred top5 labels=", _lb[_order].tolist())
            print("[gtsrb-debug] pred top5 boxes(xyxy abs)=", _bx[_order].tolist())
            print("[gtsrb-debug] cat_list[:5]=", cat_list[:5], "n_cat=", len(cat_list))

        if coco_evaluator is not None:
            if detection_dump_path and utils.is_main_process():
                _dump_detection_results_jsonl(detection_dump_path, targets, results)
            coco_evaluator.update(res)

        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if stage1_coco_evaluator is not None:
        stage1_coco_evaluator.synchronize_between_processes()

    if (
        args.save_results
        and coco_evaluator is not None
        and utils.is_main_process()
        and output_dir
    ):
        output_dir_path = Path(output_dir)
        for iou_type, merged_results in getattr(coco_evaluator, "coco_results_merged", {}).items():
            if not merged_results:
                continue
            save_json_path = output_dir_path / f"{iou_type}_predictions_all.json"
            with save_json_path.open("w", encoding="utf-8") as handle:
                json.dump(merged_results, handle)
            print(f"Saved merged {iou_type} predictions to {save_json_path}")

    # accumulate predictions from all images
    if stage1_coco_evaluator is not None:
        print("Stage1-only evaluator before stage2 rerank:")
        stage1_coco_evaluator.accumulate()
        stage1_coco_evaluator.summarize()
    if coco_evaluator is not None:
        if stage1_coco_evaluator is not None:
            print("Stage2 mixed-prompt evaluator after rerank:")
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    facb_change_stats = getattr(_apply_post_nms_self_facb_rerank, "_change_stats", None)
    if facb_change_enabled and facb_change_stats is not None:
        _print_self_facb_change_stats(facb_change_stats)
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stage2_candidate_stats = _finalize_stage2_candidate_diagnostics(stage2_candidate_diagnostics)
    if stage2_candidate_stats and utils.is_main_process():
        print(
            "Stage2 candidate diagnostics: "
            f"gt_total={stage2_candidate_stats['stage2_candidate_gt_total']}, "
            f"wrong_total={stage2_candidate_stats['stage2_candidate_wrong_total']}"
        )
        for k in (1, 5, 10, 20, 50):
            gt_key = f"stage2_candidate_gt_recall_top{k}"
            wrong_key = f"stage2_candidate_wrong_gt_recall_top{k}"
            if gt_key not in stage2_candidate_stats:
                continue
            print(
                f"  top{k}: "
                f"gt_recall={stage2_candidate_stats[gt_key]:.4f}, "
                f"wrong_gt_recall={stage2_candidate_stats[wrong_key]:.4f}"
            )
    stats.update(stage2_candidate_stats)
    if facb_change_enabled and facb_change_stats is not None:
        for key, value in facb_change_stats.items():
            stats[f"self_facb_change_{key}"] = value
        matched = max(int(facb_change_stats.get("matched", 0)), 1)
        changed = max(int(facb_change_stats.get("changed", 0)), 1)
        stats["self_facb_change_before_acc"] = facb_change_stats.get("before_correct", 0) / matched
        stats["self_facb_change_after_acc"] = facb_change_stats.get("after_correct", 0) / matched
        stats["self_facb_change_delta_acc"] = (
            stats["self_facb_change_after_acc"] - stats["self_facb_change_before_acc"]
        )
        stats["self_facb_change_precision"] = facb_change_stats.get("wrong_to_right", 0) / changed
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            for subset_name, subset_stats in getattr(coco_evaluator, "subset_stats", {}).get("bbox", {}).items():
                stats[f'coco_eval_bbox_{subset_name}'] = subset_stats
            if stage1_coco_evaluator is not None:
                stats['coco_eval_bbox_stage1'] = (
                    stage1_coco_evaluator.coco_eval['bbox'].stats.tolist()
                )
                stats['coco_eval_bbox_stage2'] = stats['coco_eval_bbox']
                for subset_name, subset_stats in getattr(
                    stage1_coco_evaluator, "subset_stats", {}
                ).get("bbox", {}).items():
                    stats[f'coco_eval_bbox_stage1_{subset_name}'] = subset_stats
                for subset_name, subset_stats in getattr(
                    coco_evaluator, "subset_stats", {}
                ).get("bbox", {}).items():
                    stats[f'coco_eval_bbox_stage2_{subset_name}'] = subset_stats
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()



    return stats, coco_evaluator
