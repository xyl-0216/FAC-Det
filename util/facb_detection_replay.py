"""Complete pre-top-k detector capture and score-only replay for FACB.

One full-class prompt per image; no GT input. Scores precede D0 writes.
The replay uses the capture device by default to preserve torch.topk tie
behaviour. CPU fallback for CUDA captures must be requested explicitly.
"""
import copy
from pathlib import Path

import torch

SCHEMA = "facb_pre_topk_scores_v1"


def validate_record(record):
    if record.get("schema") != SCHEMA:
        raise ValueError("Need a complete FACB pre-top-k cache, not final top-300 detections")
    scores, boxes, labels = record["scores"], record["boxes"], record["labels"]
    if scores.ndim != 2 or boxes.shape != (scores.shape[0], 4):
        raise ValueError("Invalid query/class score or box dimensions")
    if labels.ndim != 1 or len(labels) != scores.shape[1] or labels.unique().numel() != len(labels):
        raise ValueError("Category identities must be unique and match score columns")
    global_ids = record.get("global_ids", labels)
    if global_ids.shape != labels.shape or global_ids.unique().numel() != len(labels):
        raise ValueError("Global candidate IDs must uniquely map score columns")
    if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Expected finite detector probabilities in [0,1]")
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] < 0).any():
        raise ValueError("Expected finite normalized cxcywh boxes")
    if int(record["num_select"]) <= 0 or int(record["num_select"]) > scores.numel():
        raise ValueError("Invalid top-k budget")
    if torch.as_tensor(record["size"]).shape != (2,) or (torch.as_tensor(record["size"]) <= 0).any():
        raise ValueError("Invalid image size")


def capture_pre_topk(directory, scores, boxes, labels, global_ids, sizes, outputs,
                     num_select, nms, provenance):
    """Write one tensor file per image, preserving dtype and full score matrix."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not provenance:
        raise ValueError("Capture requires an explicit detector/provenance ID")
    if len(labels) != 232 or set(global_ids.tolist()) != set(range(232)):
        raise ValueError("FACB score capture requires a single full 232-class prompt")
    infer = outputs.get("facb_speedhr_inference") or {}
    for batch in range(len(scores)):
        image_id = int(outputs["image_ids"][batch])
        path = directory / f"{image_id}.pt"
        if path.exists():
            raise FileExistsError(f"Refusing duplicate image/chunk or overwrite: {path}")
        record = dict(schema=SCHEMA, image_id=image_id,
            scores=scores[batch].detach().cpu().clone(),
            boxes=boxes[batch].detach().cpu().clone(), labels=labels.detach().cpu().clone(),
            global_ids=global_ids.detach().cpu().clone(), size=sizes[batch].detach().cpu().clone(),
            num_select=int(num_select), nms=float(nms), score_device=str(scores.device),
            provenance=str(provenance), protocol=infer.get("proposal_protocol"),
            capture_stage="before_facb_write_and_topk")
        batch_ids = infer.get("batch_indices")
        if isinstance(batch_ids, torch.Tensor):
            mask = batch_ids == batch
            record["proposals"] = {key: value[mask].detach().cpu().clone()
                for key in ("query_indices", "candidate_ids", "candidate_mask",
                            "candidate_states", "candidate_text_states", "candidate_facet_scores",
                            "crop_states", "query_states")
                if isinstance((value := infer.get(key)), torch.Tensor)}
        validate_record(record)
        torch.save(record, path)


def replay_record(record, updates=(), device=None):
    """Updates are (query_id, category_id, score), with duplicate writes rejected."""
    validate_record(record)
    device = torch.device(device or record.get("score_device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA capture needs CUDA replay for tie fidelity; explicit device='cpu' is diagnostic only")
    scores = record["scores"].to(device).clone()
    labels = record["labels"].to(device)
    positions = {int(c): p for p, c in enumerate(labels.tolist())}
    seen = set()
    for query, category, value in updates:
        query, category, value = int(query), int(category), float(value)
        key = (query, category)
        if key in seen or not 0 <= query < len(scores) or category not in positions:
            raise ValueError(f"Invalid/duplicate score update {key}")
        if not 0 <= value <= 1:
            raise ValueError("Scores must be finite probabilities")
        seen.add(key)
        scores[query, positions[category]] = value
    values, flat = torch.topk(scores.flatten(), int(record["num_select"]))
    queries, columns = flat // scores.shape[1], flat % scores.shape[1]
    boxes = record["boxes"].to(device)
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2), -1)[queries]
    height, width = record["size"].to(device).unbind()
    xyxy = xyxy * torch.stack((width, height, width, height))[None]
    categories = labels[columns]
    if record["nms"] > 0:
        from torchvision.ops import batched_nms
        keep = batched_nms(xyxy, values, categories, float(record["nms"]))
        flat, queries, categories, xyxy, values = [t[keep] for t in (flat, queries, categories, xyxy, values)]
    return [dict(box=b, score=s, category_id=c, query_index=q, flat_index=f)
            for b, s, c, q, f in zip(xyxy.cpu().tolist(), values.cpu().tolist(),
                                   categories.cpu().tolist(), queries.cpu().tolist(), flat.cpu().tolist())]


class DenseDetections(dict):
    """Detection view with complete immutable source matrices for exact writes."""
    def __init__(self, records, device=None):
        self.records = records
        self.device = device
        super().__init__({i: replay_record(r, device=device) for i, r in records.items()})

    def subset(self, ids):
        ids = set(ids)
        if not ids <= self.records.keys():
            raise ValueError("Missing replay images")
        result = copy.copy(self)
        result.records = {i: self.records[i] for i in sorted(ids)}
        result.clear()
        result.update({i: copy.deepcopy(self[i]) for i in sorted(ids)})
        return result

    @classmethod
    def load(cls, directory, ids, device=None):
        records = {}
        for image_id in sorted(ids):
            path = Path(directory) / f"{image_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing complete pre-top-k cache: {path}")
            record = torch.load(path, map_location="cpu", weights_only=False)
            validate_record(record)
            if int(record["image_id"]) != image_id:
                raise ValueError("Cache image identity mismatch")
            records[image_id] = record
        signatures = {(r.get("provenance"), r["num_select"], r["nms"]) for r in records.values()}
        if len(signatures) != 1 or not next(iter(signatures))[0]:
            raise ValueError("Mixed or missing detector provenance/postprocess settings")
        return cls(records, device=device)

    def apply_pair_scores(self, selected, old):
        result = {}
        for image_id, rows in self.items():
            updates = [(r["query_index"], r["category_id"],
                        selected[r["_pair"]] if r["_role"] == "selected" else old[r["_pair"]])
                       for r in rows if "_pair" in r]
            result[image_id] = replay_record(self.records[image_id], updates, self.device)
        return result


def prepare_d1_pairs(baseline, rows, image_ids, target_ids, threshold, accepted, feature, target_tiers):
    """D1 compatibility adapter: stable query/class identities, including outside top-k."""
    result = baseline.subset(image_ids)
    specs, occupied = [], set()
    for row in rows:
        i, q = int(row["image_id"]), int(row["query_index"])
        if i not in image_ids or not accepted(row, target_ids, threshold):
            continue
        record = result.records[i]
        global_ids = record.get("global_ids", record["labels"]).tolist()
        positions = {int(c): p for p, c in enumerate(global_ids)}
        old_id, selected_id = int(row["old_global_id"]), int(row["selected_global_id"])
        if old_id == selected_id or (i, q) in occupied:
            raise ValueError("D1 expects unique changed queries")
        occupied.add((i, q))
        old_pos, selected_pos = positions[old_id], positions[selected_id]
        if not 0 <= q < len(record["scores"]):
            raise ValueError("Feature query ID is absent from the complete cache")
        cx, cy, w, h = record["boxes"][q].unbind()
        height, width = record["size"].unbind()
        cached_box = torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2)) * torch.stack((width,height,width,height))
        if max(abs(a-b) for a,b in zip(cached_box.tolist(), row["box"])) > .001:
            raise ValueError("Prediction box drift: feature crop does not match replay query")
        if int(record["scores"][q].argmax()) != old_pos:
            raise ValueError("Cached detector old-top differs from feature source")
        for position, expected in ((old_pos, row["old_score"]), (selected_pos, row["selected_original_score"])):
            if abs(float(record["scores"][q, position]) - float(expected)) > 1e-6:
                raise ValueError("Detector score drift: regenerate aligned features/cache")
        pair_id = len(specs)
        for position, role in ((old_pos, "old"), (selected_pos, "selected")):
            cid = int(record["labels"][position])
            existing = next((r for r in result[i] if r["query_index"] == q and r["category_id"] == cid), None)
            if existing is None:
                existing = dict(query_index=q, category_id=cid)
                result[i].append(existing)
            existing.update(_pair=pair_id, _role=role)
        specs.append(dict(key=(i, q), feature=feature(row), old_score=float(record["scores"][q, old_pos]),
            selected_original=float(record["scores"][q, selected_pos]),
            useful=bool(row["bucket"] == "old_wrong_gt_present" and row["tier"] in target_tiers
                        and selected_id == row["gt_class"]),
            old_correct=row["bucket"] in {"old_correct_low_score", "old_correct_regular_score"},
            bucket=row["bucket"], tier=row["tier"], gt_id=row.get("gt_id"),
            selected_global_id=selected_id, old_global_id=old_id))
    return result, specs, []


def scoring_hypotheses(query, old_class, selected_class, accepted_change, target_ids, include_same=True):
    """GT-free scoring eligibility, separate from D0's changed-only accept gate."""
    if accepted_change:
        if old_class == selected_class or selected_class not in target_ids:
            raise ValueError("Inconsistent accepted D0 change")
        return [(query, selected_class, "selected"), (query, old_class, "old")]
    if include_same and selected_class == old_class and old_class in target_ids:
        return [(query, old_class, "same")]
    return []
