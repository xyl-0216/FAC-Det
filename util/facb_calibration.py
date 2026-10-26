"""Shared method utilities."""
import contextlib


import copy


import io


import json


from collections import Counter, defaultdict


from pycocotools.cocoeval import COCOeval


TARGET_TIERS = {"tail_3shot", "mid_5shot", "mid_10shot"}


def load_split(path):
    report = json.loads(path.read_text(encoding="utf8"))
    fit = set(report["fit_images"])
    calibration = set(report["calibration_images"])
    assert fit and calibration and not fit.intersection(calibration)
    return fit, calibration


def outcome(row, target_ids):
    changed = row["selected_global_id"] != row["old_global_id"]
    in_scope = row["selected_global_id"] in target_ids
    useful = (
        changed
        and in_scope
        and row["tier"] in TARGET_TIERS
        and row["bucket"] == "old_wrong_gt_present"
        and row["selected_global_id"] == row["gt_class"]
    )
    harm = (
        changed
        and in_scope
        and row["bucket"] in {"old_correct_low_score", "old_correct_regular_score"}
        and row["selected_global_id"] != row["gt_class"]
    )
    return changed, in_scope, useful, harm


def load_baseline(pool, image_ids):
    baseline = {}
    for shard in sorted(pool.glob("images_*")):
        path = shard / "cache" / "final_detections.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf8"):
            row = json.loads(line)
            if row["image_id"] not in image_ids:
                continue
            assert row["image_id"] not in baseline
            baseline[row["image_id"]] = [
                dict(box=box, score=score, category_id=category_id)
                for box, score, category_id in zip(row["boxes"], row["scores"], row["labels"])
            ]
    assert set(baseline) == image_ids
    assert all(len(detections) == 300 for detections in baseline.values())
    return baseline


def apply_score_conserving_swaps(
    baseline, rows, calibration_ids, target_ids, threshold,
    score_key="fused_margin_over_old",
):
    result = copy.deepcopy(baseline)
    edits = []
    occupied = set()
    for row in rows:
        if row["image_id"] not in calibration_ids:
            continue
        changed, in_scope, useful, harm = outcome(row, target_ids)
        if not (changed and in_scope and row[score_key] >= threshold):
            continue
        image_id = row["image_id"]
        detections = result[image_id]
        same_box = [
            i for i, det in enumerate(detections)
            if max(abs(x - y) for x, y in zip(det["box"], row["box"])) <= 0.001
        ]
        old_slots = [
            i for i in same_box
            if detections[i]["category_id"] == row["old_global_id"]
            and abs(detections[i]["score"] - row["old_score"]) < 1e-6
        ]
        selected_slots = [
            i for i in same_box if detections[i]["category_id"] == row["selected_global_id"]
        ]
        assert len(old_slots) <= 1 and len(selected_slots) <= 1
        if not old_slots:
            edits.append(dict(**row, useful=useful, harm=harm, status="old_outside_final300"))
            continue
        old_slot = old_slots[0]
        assert (image_id, old_slot) not in occupied
        occupied.add((image_id, old_slot))
        if selected_slots:
            selected_slot = selected_slots[0]
            assert (image_id, selected_slot) not in occupied
            occupied.add((image_id, selected_slot))
            detections[selected_slot]["category_id"] = row["old_global_id"]
        else:
            selected_slot = None
        detections[old_slot]["category_id"] = row["selected_global_id"]
        edits.append(
            dict(
                **row,
                useful=useful,
                harm=harm,
                status="swapped",
                old_detection_slot=old_slot,
                selected_detection_slot=selected_slot,
            )
        )
    for image_id in baseline:
        before = [(d["box"], d["score"]) for d in baseline[image_id]]
        after = [(d["box"], d["score"]) for d in result[image_id]]
        assert before == after
    return result, edits


def evaluate(coco, image_ids, detections, groups):
    records = []
    for image_id, rows in detections.items():
        for row in rows:
            x1, y1, x2, y2 = row["box"]
            records.append(
                dict(
                    image_id=image_id,
                    category_id=row["category_id"],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=row["score"],
                )
            )
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(records), "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    mean = lambda values: float(values[values >= 0].mean()) if (values >= 0).any() else None
    result = {"overall_AP": mean(precision)}
    for name, category_ids in groups.items():
        indices = [i for i, category_id in enumerate(evaluator.params.catIds) if category_id in category_ids]
        result[name + "_AP"] = mean(precision[:, :, indices])
    return result


def summarize_edits(edits):
    applied = [row for row in edits if row["status"] == "swapped"]
    useful = [row for row in applied if row["useful"]]
    harm = [row for row in applied if row["harm"]]
    return dict(
        accepted_rows=len(edits),
        applied_rows=len(applied),
        outside_final300=len(edits) - len(applied),
        useful_rows=len(useful),
        useful_unique_gt=len({row["gt_id"] for row in useful}),
        harm_rows=len(harm),
        harm_unique_gt=len({row["gt_id"] for row in harm}),
        other_rows=len(applied) - len(useful) - len(harm),
        accepted_tiers=dict(Counter(row["tier"] for row in applied)),
        selected_classes=dict(Counter(str(row["selected_global_id"]) for row in applied)),
    )


