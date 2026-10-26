"""Shared method utilities."""
import json


import math


import numpy as np


import torch


from util.facb_dual_facet_itm import DualFacetITM


def overlap(a, b):
    return bool(set(a).intersection(b))


def choose_slots(raw, label, scored, class_targets, max_pairs):
    ids = raw["candidate_global_ids"]
    selected = int(scored["selected_slot"])
    old = ids.index(label["old_global_id"])
    slots = [selected, old]
    valid = label.get("gt_class") is not None and float(label.get("best_iou", 0.0)) >= 0.5
    gt = label.get("gt_class")
    if valid and gt in ids:
        gt_slot = ids.index(gt)
        slots.append(gt_slot)
        gt_shape = class_targets[gt][1]
        gt_symbol = class_targets[gt][2]
        same_shape = [
            i for i, candidate in enumerate(ids)
            if candidate != gt and overlap(class_targets[candidate][1], gt_shape)
            and not overlap(class_targets[candidate][2], gt_symbol)
        ]
        same_shape.sort(key=lambda i: float(raw["candidate_facet_logits"][i][2]), reverse=True)
        slots.extend(same_shape[:2])
    # A high semantic rival is useful for background and for classes whose GT is absent.
    rivals = sorted(range(len(ids)), key=lambda i: float(raw["candidate_semantic_scores"][i]), reverse=True)
    slots.extend(rivals[:3])
    unique = []
    for slot in slots:
        if slot not in unique:
            unique.append(slot)
        if len(unique) == max_pairs:
            break
    return unique


def load_pair_data(pool, p0_rows_path, facet_config, max_pairs, allowed_image_ids=None, packed_cache=None):
    labels = {
        (row["image_id"], row["query_index"]): row
        for row in map(json.loads, (pool / "labels.jsonl").open(encoding="utf8"))
    }
    scored = {
        (row["image_id"], row["query_index"]): row
        for row in map(json.loads, p0_rows_path.open(encoding="utf8"))
    }
    if allowed_image_ids is not None:
        labels = {key: row for key, row in labels.items() if key[0] in allowed_image_ids}
        scored = {key: row for key, row in scored.items() if key[0] in allowed_image_ids}
    assert labels.keys() == scored.keys()
    config = json.loads(facet_config.read_text(encoding="utf8"))
    names = config["facb_speedhr_shared_facet_names"]
    assert names == ["group", "shape", "symbol"], names
    class_targets = config["facb_speedhr_shared_facet_class_targets"]

    records = []
    decoder_states, text_states, raw_facets = [], [], []
    selected_pair_index = {}
    for shard in sorted(pool.glob("images_*")):
        path = shard / "cache" / "proposals.jsonl"
        if not path.exists():
            continue
        source = torch.load(packed_cache / (shard.name + '.pt'), map_location='cpu', weights_only=False) if packed_cache else path.open(encoding="utf8")
        for line in source:
            raw = line if isinstance(line, dict) else json.loads(line)
            key = (raw["image_id"], raw["query_index"])
            if key not in labels:
                continue
            label, p0 = labels[key], scored[key]
            assert raw["candidate_global_ids"] == label["candidate_global_ids"] == p0["candidate_global_ids"]
            slots = choose_slots(raw, label, p0, class_targets, max_pairs)
            valid = label.get("gt_class") is not None and float(label.get("best_iou", 0.0)) >= 0.5
            gt = label.get("gt_class")
            for slot in slots:
                candidate = raw["candidate_global_ids"][slot]
                shape_match = valid and overlap(class_targets[candidate][1], class_targets[gt][1])
                symbol_match = valid and overlap(class_targets[candidate][2], class_targets[gt][2])
                class_match = valid and candidate == gt
                record = dict(
                    image_id=key[0], query_index=key[1], split=label["split"], slot=slot,
                    candidate_global_id=candidate, gt_class=gt, valid=valid,
                    class_label=int(class_match), shape_label=int(shape_match), symbol_label=int(symbol_match),
                    selected_pair=slot == int(p0["selected_slot"]),
                    dangerous=int(
                        label["bucket"] in {"old_correct_low_score", "old_correct_regular_score"}
                        and candidate != label["old_global_id"]
                    ),
                    duplicate_weight=float(label.get("gt_normalized_weight", 1.0)) if valid else 1.0,
                )
                index = len(records)
                records.append(record)
                decoder_states.append(raw["candidate_states"][slot])
                text_states.append(raw["candidate_text_states"][slot])
                # Raw logits, before the saturating sigmoid: shape index 1, symbol index 2.
                raw_facets.append([raw["candidate_facet_logits"][slot][1], raw["candidate_facet_logits"][slot][2]])
                if record["selected_pair"]:
                    assert key not in selected_pair_index
                    selected_pair_index[key] = index
        print("loaded", shard.name, "pairs", len(records), flush=True)
    assert labels.keys() == selected_pair_index.keys()
    return dict(
        records=records,
        decoder=torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in decoder_states]),
        text=torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in text_states]),
        facets=torch.tensor(raw_facets, dtype=torch.float32),
        selected_pair_index=selected_pair_index,
        labels=labels,
        scored=scored,
    )


def balanced_weights(records, indices, label_name, dangerous_multiplier):
    labels = torch.tensor([records[i][label_name] for i in indices], dtype=torch.float32)
    duplicate = torch.tensor([records[i]["duplicate_weight"] for i in indices], dtype=torch.float32)
    danger = torch.tensor([records[i]["dangerous"] for i in indices], dtype=torch.float32)
    raw = duplicate * (1.0 + (dangerous_multiplier - 1.0) * danger * (1.0 - labels))
    result = torch.zeros_like(raw)
    for value in (0.0, 1.0):
        mask = labels == value
        if mask.any():
            result[mask] = 0.5 * raw[mask] / raw[mask].sum()
    return labels, result


def binary_ap(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def predict(model, data, indices, mean, std, batch_size):
    outputs = []
    with torch.inference_mode():
        for chunk in torch.tensor(indices).split(batch_size):
            values = model(data["decoder"][chunk], data["text"][chunk], data["facets"][chunk], mean, std)
            outputs.append(torch.stack(values, dim=1).sigmoid())
    return torch.cat(outputs)


def threshold_grid(rows):
    ordered = sorted(rows, key=lambda row: row["itm_class_probability"], reverse=True)
    budgets = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    thresholds = [math.inf]
    for budget in budgets:
        if 0 < budget <= len(ordered):
            thresholds.append(float(ordered[budget - 1]["itm_class_probability"]))
    thresholds.extend([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    return sorted(set(thresholds), reverse=True)


