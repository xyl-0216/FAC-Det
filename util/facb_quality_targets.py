"""Shared method utilities."""
import contextlib


import io


import math


import numpy as np


from pycocotools.cocoeval import COCOeval


def logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def accepted(row, target_ids, threshold):
    return (
        int(row["selected_global_id"]) != int(row["old_global_id"])
        and int(row["selected_global_id"]) in target_ids
        and float(row["itm_class_probability"]) >= threshold
    )


def coco_targets(coco, detections, image_ids, return_details=False):
    if not detections:
        thresholds = COCOeval().params.iouThrs.copy()
        quality = np.empty(0, dtype=np.float32)
        if return_details:
            return dict(quality=quality, matched=np.zeros((0, len(thresholds)), dtype=bool),
                        valid=np.zeros((0, len(thresholds)), dtype=bool), iou_thresholds=thresholds)
        return quality
    records = []
    for row in detections:
        x1, y1, x2, y2 = row["box"]
        records.append({
            "image_id": row["image_id"], "category_id": row["category_id"],
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": row["score"],
        })
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(records), "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
    # NaN is undefined supervision, never a negative label. Preserve the
    # per-threshold mask so new scorers can distinguish ignore from FP.
    matches = np.zeros((len(records), len(evaluator.params.iouThrs)), dtype=bool)
    valid_mask = np.zeros_like(matches)
    targets = np.zeros(len(records), dtype=np.float32)
    counts = np.zeros(len(records), dtype=np.float32)
    for item in evaluator.evalImgs:
        if item is None or item["aRng"] != evaluator.params.areaRng[0]:
            continue
        for column, detection_id in enumerate(item["dtIds"]):
            valid = ~item["dtIgnore"][:, column]
            if valid.any():
                index = int(detection_id) - 1
                matches[index] = item["dtMatches"][:, column] > 0
                valid_mask[index] = valid
                targets[index] += float((item["dtMatches"][:, column][valid] > 0).mean())
                counts[index] += 1.0
    mask = counts > 0
    targets[mask] /= counts[mask]
    targets[~mask] = np.nan
    if return_details:
        return dict(quality=targets, matched=matches, valid=valid_mask,
                    iou_thresholds=evaluator.params.iouThrs.copy())
    return targets


def ranking_pairs(rows, targets, seed, negatives_per_positive):
    groups = {}
    for index, (row, target) in enumerate(zip(rows, targets)):
        if np.isfinite(target):
            groups.setdefault(int(row["category_id"]), []).append(index)
    rng = np.random.default_rng(seed)
    pairs = []
    for indices in groups.values():
        positives = [i for i in indices if targets[i] > 0]
        for positive in positives:
            # Include lower-quality positives as well as FP, but never a pair
            # with two fixed endpoints (it has no learning gradient).
            eligible = [i for i in indices if targets[i] < targets[positive]
                        and (rows[positive].get("_mutable") or rows[i].get("_mutable"))]
            if not eligible:
                continue
            chosen = rng.choice(eligible, min(negatives_per_positive, len(eligible)), replace=False)
            pairs.extend((positive, int(negative)) for negative in chosen)
    return pairs


