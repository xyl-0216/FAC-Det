# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import copy
import datetime
import hashlib
import json
import pickletools
import platform
import random
import shutil
import socket
import time
import zipfile
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, WeightedRandomSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.appearance_prompt import (
    FACB_COMPOSITIONAL_FACET_NAMES,
    build_facb_compositional_facet_texts,
    build_facb_shared_facet_schema,
    build_label_text_map,
    build_fine_group_ids,
    build_multi_facet_texts,
)
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict

_COCO_EVAL_STAT_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _named_coco_stats(values):
    values = _json_safe(values)
    if not isinstance(values, list):
        return values
    return {
        name: values[idx] if idx < len(values) else None
        for idx, name in enumerate(_COCO_EVAL_STAT_NAMES)
    }


def _save_eval_results(output_dir, test_stats, args, epoch=None, mode="eval"):
    if not output_dir or not utils.is_main_process() or test_stats is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_bbox = {}
    for key, value in test_stats.items():
        if key == "coco_eval_bbox":
            subset = "overall"
        elif key.startswith("coco_eval_bbox_"):
            subset = key[len("coco_eval_bbox_"):]
        else:
            continue
        coco_bbox[subset] = _named_coco_stats(value)

    record = {
        "mode": mode,
        "epoch": epoch,
        "now_time": str(datetime.datetime.now()),
        "config_file": str(getattr(args, "config_file", "")),
        "datasets": str(getattr(args, "datasets", "")),
        "resume": str(getattr(args, "resume", "")),
        "pretrain_model_path": str(getattr(args, "pretrain_model_path", "")),
        "postprocess_token_idf_lambda": _json_safe(
            getattr(args, "postprocess_token_idf_lambda", None)
        ),
        "matcher_token_idf_lambda": _json_safe(
            getattr(args, "matcher_token_idf_lambda", None)
        ),
        "use_postprocess_score_norm": _json_safe(
            getattr(args, "use_postprocess_score_norm", None)
        ),
        "postprocess_score_norm_type": _json_safe(
            getattr(args, "postprocess_score_norm_type", None)
        ),
        "coco_eval_bbox": coco_bbox,
        "raw_stats": _json_safe(test_stats),
    }

    with (output_dir / "latest_eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with (output_dir / "eval_results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _metric(subset, name):
        value = coco_bbox.get(subset, {}).get(name, None)
        return "nan" if value is None else f"{float(value):.4f}"

    parts = [
        f"time={record['now_time']}",
        f"mode={mode}",
        f"epoch={epoch}",
        f"idf={record['postprocess_token_idf_lambda']}",
        f"matcher_idf={record['matcher_token_idf_lambda']}",
        f"score_norm={record['use_postprocess_score_norm']}",
        f"norm_type={record['postprocess_score_norm_type']}",
    ]
    summary_subsets = (
        "overall",
        "seen",
        "unseen",
        "head_ge100",
        "mid_50_99_all",
        "tail_1shot",
        "tail_3shot",
        "mid_5shot",
        "mid_10shot",
        "zero_shot",
        "same_group",
    )
    for subset in summary_subsets:
        if subset not in coco_bbox:
            continue
        parts.extend([
            f"{subset}_AP={_metric(subset, 'AP')}",
            f"{subset}_AP50={_metric(subset, 'AP50')}",
            f"{subset}_AP75={_metric(subset, 'AP75')}",
            f"{subset}_AR100={_metric(subset, 'AR100')}",
        ])
    summary_line = " | ".join(parts)
    with (output_dir / "latest_eval_summary.txt").open("w", encoding="utf-8") as f:
        f.write(summary_line + "\n")
    with (output_dir / "eval_summary.txt").open("a", encoding="utf-8") as f:
        f.write(summary_line + "\n")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_checkpoint_compat(path, map_location="cpu"):
    """Load trusted training checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_checkpoint_metadata(path):
    metadata = {
        "exists": False,
        "size_bytes": None,
        "is_torch_zip": False,
        "has_optimizer": False,
        "has_scheduler": False,
        "has_args": False,
        "epoch": None,
        "config_file": None,
        "text_encoder_type": None,
        "use_text_adapter_ffn": None,
        "use_aggr": None,
    }
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return metadata

    metadata["exists"] = True
    metadata["size_bytes"] = checkpoint_path.stat().st_size
    if not zipfile.is_zipfile(checkpoint_path):
        return metadata

    metadata["is_torch_zip"] = True
    with zipfile.ZipFile(checkpoint_path) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not pickle_names:
            return metadata
        operations = list(pickletools.genops(archive.read(pickle_names[0])))

    value_targets = {
        "optimizer": "has_optimizer",
        "lr_scheduler": "has_scheduler",
        "args": "has_args",
        "epoch": "epoch",
        "config_file": "config_file",
        "text_encoder_type": "text_encoder_type",
        "use_text_adapter_ffn": "use_text_adapter_ffn",
        "use_aggr": "use_aggr",
    }
    for index, (_op, argument, _pos) in enumerate(operations):
        if not isinstance(argument, str):
            continue
        output_key = value_targets.get(argument)
        if output_key is None:
            continue
        if output_key in {"has_optimizer", "has_scheduler", "has_args"}:
            metadata[output_key] = True
            continue
        for next_operation, next_argument, _ in operations[index + 1:index + 10]:
            if next_operation.name in {"BINUNICODE", "SHORT_BINUNICODE"}:
                metadata[output_key] = next_argument
                break
            if next_operation.name in {"BININT", "BININT1", "BININT2", "LONG1"}:
                metadata[output_key] = int(next_argument)
                break
            if next_operation.name == "NEWTRUE":
                metadata[output_key] = True
                break
            if next_operation.name == "NEWFALSE":
                metadata[output_key] = False
                break
            if next_operation.name == "NONE":
                metadata[output_key] = None
                break
    return metadata


def _classify_checkpoint_source(source_name, metadata):
    has_training_state = bool(
        metadata.get("has_optimizer")
        or metadata.get("has_scheduler")
        or metadata.get("has_args")
        or metadata.get("epoch") is not None
    )
    if source_name == "resume":
        return "resume_training_checkpoint" if has_training_state else "resume_model_only_checkpoint"
    return "warm_start_task_checkpoint" if has_training_state else "pretrain_initialization"


def _fingerprint_path(path_value):
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    record = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "kind": "missing",
    }
    if resolved.is_file():
        record["kind"] = "file"
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = _sha256_file(resolved)
    elif resolved.is_dir():
        record["kind"] = "directory"
    return record


def _build_run_manifest(args, dataset_meta):
    manifest = {
        "created_at": str(datetime.datetime.now()),
        "command": sys.argv,
        "git": utils.get_sha(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        },
        "runtime": {
            "seed": getattr(args, "seed", None),
            "amp": bool(getattr(args, "amp", False)),
            "distributed": bool(getattr(args, "distributed", False)),
            "world_size": getattr(args, "world_size", None),
            "device": getattr(args, "device", None),
        },
        "config": {
            "config_file": _fingerprint_path(getattr(args, "config_file", None)),
            "datasets": _fingerprint_path(getattr(args, "datasets", None)),
            "output_dir": str(Path(getattr(args, "output_dir", "")).resolve()) if getattr(args, "output_dir", "") else "",
            "baseline_protocol": getattr(args, "baseline_protocol", None),
            "pure_baseline": bool(getattr(args, "pure_baseline", False)),
        },
        "dataset_meta": dataset_meta,
        "resolved_files": {},
        "checkpoint_sources": {},
    }

    for split_name, entries in dataset_meta.items():
        if not isinstance(entries, list):
            continue
        manifest["resolved_files"][split_name] = []
        for entry in entries:
            record = {}
            for key in ("root", "anno", "label_map", "appearance_prompt_meta_path", "seen_category_meta_path"):
                record[key] = _fingerprint_path(entry.get(key))
            record["dataset_mode"] = entry.get("dataset_mode")
            manifest["resolved_files"][split_name].append(record)

    for source_name in ("resume", "pretrain_model_path"):
        source_value = getattr(args, source_name, "")
        if not source_value:
            continue
        source_record = _fingerprint_path(source_value)
        metadata = _inspect_checkpoint_metadata(source_value)
        source_record["checkpoint_metadata"] = metadata
        source_record["source_type"] = _classify_checkpoint_source(source_name, metadata)
        manifest["checkpoint_sources"][source_name] = source_record

    return manifest


def _save_run_manifest(args, dataset_meta, logger=None):
    if not args.output_dir or not utils.is_main_process():
        return

    manifest = _build_run_manifest(args, dataset_meta)
    output_path = Path(args.output_dir) / "run_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if logger is not None:
        logger.info(f"Run manifest saved to {output_path}")


def resolve_dataset_meta_paths(dataset_meta, datasets_json_path):
    base_dir = Path(datasets_json_path).resolve().parent
    resolved_meta = copy.deepcopy(dataset_meta)
    path_keys = (
        "root",
        "anno",
        "label_map",
        "appearance_prompt_meta_path",
        "seen_category_meta_path",
    )

    for split_name, entries in resolved_meta.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in path_keys:
                value = entry.get(key)
                if not value or not isinstance(value, str):
                    continue
                candidate = Path(value).expanduser()
                if candidate.is_absolute():
                    entry[key] = str(candidate)
                else:
                    entry[key] = str((base_dir / candidate).resolve())

    return resolved_meta


def _compute_repeat_factors_from_coco_dataset(dataset, repeat_thr, repeat_power, repeat_max):
    if not hasattr(dataset, "coco") or not hasattr(dataset, "ids"):
        return [1.0] * len(dataset)

    image_ids = list(dataset.ids)
    num_images = max(len(image_ids), 1)
    image_categories = []
    category_image_count = {}

    for image_id in image_ids:
        anns = dataset.coco.imgToAnns.get(image_id, [])
        category_ids = {
            int(ann["category_id"])
            for ann in anns
            if int(ann.get("iscrowd", 0)) == 0
        }
        image_categories.append(category_ids)
        for category_id in category_ids:
            category_image_count[category_id] = category_image_count.get(category_id, 0) + 1

    category_repeat = {}
    for category_id, image_count in category_image_count.items():
        freq = float(image_count) / float(num_images)
        if freq <= 0:
            repeat = float(repeat_max)
        elif freq < repeat_thr:
            repeat = (float(repeat_thr) / freq) ** float(repeat_power)
        else:
            repeat = 1.0
        category_repeat[category_id] = min(float(repeat), float(repeat_max))

    repeat_factors = []
    for category_ids in image_categories:
        if not category_ids:
            repeat_factors.append(1.0)
            continue
        repeat_factors.append(max(category_repeat.get(category_id, 1.0) for category_id in category_ids))
    return repeat_factors


def _compute_repeat_factors_from_odvg_dataset(dataset, repeat_thr, repeat_power, repeat_max):
    if not hasattr(dataset, "metas"):
        return [1.0] * len(dataset)

    metas = list(dataset.metas)
    num_images = max(len(metas), 1)
    image_categories = []
    category_image_count = {}

    for meta in metas:
        detection = meta.get("detection", {}) if isinstance(meta, dict) else {}
        category_ids = {
            str(instance.get("label"))
            for instance in detection.get("instances", [])
            if instance.get("label") is not None
        }
        image_categories.append(category_ids)
        for category_id in category_ids:
            category_image_count[category_id] = category_image_count.get(category_id, 0) + 1

    category_repeat = {}
    for category_id, image_count in category_image_count.items():
        freq = float(image_count) / float(num_images)
        if freq <= 0:
            repeat = float(repeat_max)
        elif freq < repeat_thr:
            repeat = (float(repeat_thr) / freq) ** float(repeat_power)
        else:
            repeat = 1.0
        category_repeat[category_id] = min(float(repeat), float(repeat_max))

    repeat_factors = []
    for category_ids in image_categories:
        if not category_ids:
            repeat_factors.append(1.0)
            continue
        repeat_factors.append(max(category_repeat.get(category_id, 1.0) for category_id in category_ids))
    return repeat_factors


def _compute_repeat_factors(dataset, repeat_thr, repeat_power, repeat_max):
    if isinstance(dataset, ConcatDataset):
        repeat_factors = []
        for sub_dataset in dataset.datasets:
            if hasattr(sub_dataset, "coco") and hasattr(sub_dataset, "ids"):
                sub_repeat_factors = _compute_repeat_factors_from_coco_dataset(
                    sub_dataset,
                    repeat_thr=repeat_thr,
                    repeat_power=repeat_power,
                    repeat_max=repeat_max,
                )
            elif hasattr(sub_dataset, "metas"):
                sub_repeat_factors = _compute_repeat_factors_from_odvg_dataset(
                    sub_dataset,
                    repeat_thr=repeat_thr,
                    repeat_power=repeat_power,
                    repeat_max=repeat_max,
                )
            else:
                sub_repeat_factors = [1.0] * len(sub_dataset)
            repeat_factors.extend(sub_repeat_factors)
        return repeat_factors
    if hasattr(dataset, "coco") and hasattr(dataset, "ids"):
        return _compute_repeat_factors_from_coco_dataset(
            dataset,
            repeat_thr=repeat_thr,
            repeat_power=repeat_power,
            repeat_max=repeat_max,
        )
    if hasattr(dataset, "metas"):
        return _compute_repeat_factors_from_odvg_dataset(
            dataset,
            repeat_thr=repeat_thr,
            repeat_power=repeat_power,
            repeat_max=repeat_max,
        )
    return [1.0] * len(dataset)


def _dataset_image_category_ids(dataset):
    """Return one set of global category ids per dataset image."""
    if isinstance(dataset, ConcatDataset):
        rows = []
        for sub_dataset in dataset.datasets:
            rows.extend(_dataset_image_category_ids(sub_dataset))
        return rows
    if hasattr(dataset, "coco") and hasattr(dataset, "ids"):
        return [
            {
                int(ann["category_id"])
                for ann in dataset.coco.imgToAnns.get(image_id, [])
                if int(ann.get("iscrowd", 0)) == 0
            }
            for image_id in dataset.ids
        ]
    if hasattr(dataset, "metas"):
        rows = []
        for meta in dataset.metas:
            detection = meta.get("detection", {}) if isinstance(meta, dict) else {}
            category_ids = set()
            for instance in detection.get("instances", []):
                value = instance.get("label")
                if value is None:
                    continue
                try:
                    category_ids.add(int(value))
                except (TypeError, ValueError):
                    continue
            rows.append(category_ids)
        return rows
    return [set() for _ in range(len(dataset))]


def _load_facb_smallshot_ids(args):
    meta_value = getattr(args, "facb_speedhr_split_meta_path", None)
    if not meta_value:
        return set()
    path = Path(str(meta_value)).expanduser()
    if not path.is_absolute():
        cwd_path = (Path.cwd() / path).resolve()
        config_path = (Path(args.config_file).resolve().parent / path).resolve()
        path = cwd_path if cwd_path.exists() else config_path
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)
    except Exception:
        return set()
    min_count = int(getattr(args, "facb_speedhr_agf_min_train_count", 1))
    max_count = int(getattr(args, "facb_speedhr_agf_max_train_count", 10))
    selected = set()
    for group in (meta.get("fewshot_groups", {}) or {}).values():
        if not isinstance(group, dict):
            continue
        shot = int(group.get("shot", 0) or 0)
        if shot < min_count or shot > max_count:
            continue
        selected.update(int(class_id) for class_id in group.get("classes", []) or [])
    return selected


class FACBSmallshotBalancedSampler(torch.utils.data.Sampler):
    """Arrange indices so every batch contains small-shot and anchor images."""

    def __init__(
        self,
        dataset,
        smallshot_ids,
        batch_size,
        smallshot_per_batch=1,
        smallshot_anchor_cycle=0,
    ):
        self.num_samples = (len(dataset) // int(batch_size)) * int(batch_size)
        self.batch_size = int(batch_size)
        self.smallshot_per_batch = max(
            1, min(int(smallshot_per_batch), self.batch_size - 1)
        )
        self.smallshot_anchor_cycle = max(int(smallshot_anchor_cycle), 0)
        image_categories = _dataset_image_category_ids(dataset)
        smallshot_ids = {int(value) for value in smallshot_ids}
        self.small_indices = [
            idx for idx, categories in enumerate(image_categories)
            if categories & smallshot_ids
        ]
        self.anchor_indices = [
            idx for idx, categories in enumerate(image_categories)
            if not (categories & smallshot_ids)
        ]
        if not self.anchor_indices:
            self.anchor_indices = list(range(len(dataset)))

    def __iter__(self):
        if not self.small_indices or self.num_samples <= 0:
            yield from torch.randperm(self.num_samples).tolist()
            return
        if self.batch_size <= 1:
            for sample_idx in range(self.num_samples):
                use_anchor = False
                if self.smallshot_anchor_cycle > 0 and self.anchor_indices:
                    use_anchor = (sample_idx + 1) % (self.smallshot_anchor_cycle + 1) == 0
                else:
                    use_anchor = sample_idx % 2 == 1 and bool(self.anchor_indices)
                pool = (
                    self.anchor_indices
                    if use_anchor
                    else self.small_indices
                )
                choice = int(torch.randint(len(pool), (1,)).item())
                yield pool[choice]
            return
        num_batches = self.num_samples // self.batch_size
        for _ in range(num_batches):
            small_choice = torch.randint(
                len(self.small_indices), (self.smallshot_per_batch,)
            ).tolist()
            anchor_count = self.batch_size - self.smallshot_per_batch
            anchor_choice = torch.randint(
                len(self.anchor_indices), (anchor_count,)
            ).tolist()
            batch = [self.small_indices[idx] for idx in small_choice]
            batch.extend(self.anchor_indices[idx] for idx in anchor_choice)
            order = torch.randperm(len(batch)).tolist()
            yield from (batch[idx] for idx in order)

    def __len__(self):
        return self.num_samples


def _compute_small_object_boosts_from_coco_dataset(dataset, small_area_thr, small_boost):
    if not hasattr(dataset, "coco") or not hasattr(dataset, "ids"):
        return [1.0] * len(dataset)

    boosts = []
    for image_id in dataset.ids:
        anns = dataset.coco.imgToAnns.get(image_id, [])
        has_small = False
        for ann in anns:
            if int(ann.get("iscrowd", 0)) != 0:
                continue
            area = float(ann.get("area", 0.0))
            if area > 0.0 and area <= float(small_area_thr):
                has_small = True
                break
        boosts.append(float(small_boost) if has_small else 1.0)
    return boosts


def _compute_small_object_boosts_from_odvg_dataset(dataset, small_area_thr, small_boost):
    if not hasattr(dataset, "metas"):
        return [1.0] * len(dataset)

    boosts = []
    for meta in dataset.metas:
        detection = meta.get("detection", {}) if isinstance(meta, dict) else {}
        has_small = False
        for instance in detection.get("instances", []):
            bbox = instance.get("bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            try:
                x0, y0, x1_or_w, y1_or_h = map(float, bbox[:4])
            except (TypeError, ValueError):
                continue
            # ODVG TT100K annotations store boxes in xyxy pixel format.
            # Keep a conservative fallback for xywh-style boxes to avoid
            # silently breaking other ODVG datasets.
            if x1_or_w > x0 and y1_or_h > y0:
                area = (x1_or_w - x0) * (y1_or_h - y0)
            else:
                area = x1_or_w * y1_or_h
            if area > 0.0 and area <= float(small_area_thr):
                has_small = True
                break
        boosts.append(float(small_boost) if has_small else 1.0)
    return boosts


def _compute_small_object_boosts(dataset, small_area_thr, small_boost):
    if isinstance(dataset, ConcatDataset):
        boosts = []
        for sub_dataset in dataset.datasets:
            if hasattr(sub_dataset, "coco") and hasattr(sub_dataset, "ids"):
                sub_boosts = _compute_small_object_boosts_from_coco_dataset(
                    sub_dataset, small_area_thr=small_area_thr, small_boost=small_boost
                )
            elif hasattr(sub_dataset, "metas"):
                sub_boosts = _compute_small_object_boosts_from_odvg_dataset(
                    sub_dataset, small_area_thr=small_area_thr, small_boost=small_boost
                )
            else:
                sub_boosts = [1.0] * len(sub_dataset)
            boosts.extend(sub_boosts)
        return boosts
    if hasattr(dataset, "coco") and hasattr(dataset, "ids"):
        return _compute_small_object_boosts_from_coco_dataset(
            dataset, small_area_thr=small_area_thr, small_boost=small_boost
        )
    if hasattr(dataset, "metas"):
        return _compute_small_object_boosts_from_odvg_dataset(
            dataset, small_area_thr=small_area_thr, small_boost=small_boost
        )
    return [1.0] * len(dataset)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument(
        '--eval_interval',
        default=1,
        type=int,
        help='run evaluation every N epochs during training; the final epoch is always evaluated',
    )

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument(
        '--pin_memory',
        dest='pin_memory',
        action='store_true',
        help='Pin host memory in dataloaders. Enabled by default on CUDA.',
    )
    parser.add_argument(
        '--no_pin_memory',
        dest='pin_memory',
        action='store_false',
        help='Disable pinned host memory in dataloaders.',
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        '--prefetch_factor',
        default=None,
        type=int,
        help='Dataloader prefetch factor per worker. Uses PyTorch default when unset.',
    )
    parser.add_argument(
        '--disable_persistent_workers',
        action='store_true',
        help='Disable persistent dataloader workers between epochs.',
    )
    parser.add_argument(
        '--val_batch_size',
        default=None,
        type=int,
        help='Validation batch size. Defaults to 4 to preserve current behavior.',
    )
    parser.add_argument(
        '--disable_tf32',
        action='store_true',
        help='Disable TF32 matmul/convolution on supported NVIDIA GPUs.',
    )
    parser.add_argument(
        '--disable_cudnn_benchmark',
        action='store_true',
        help='Disable cudnn benchmark autotuning.',
    )
    return parser


def _get_parser_default_map():
    parser = get_args_parser()
    default_map = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        default_map[dest] = action.default
    return default_map


_PURE_BASELINE_FORBIDDEN_FLAGS = (
    "use_text_adapter_ffn",
    "use_query_score_selector",
    "use_modality_projection",
    "use_similarity_calibration",
    "use_quality_head",
    "use_lgpr_survival",
    "use_lgpr_loc_preserve",
    "use_lgpr_box_refine",
    "use_query_contrastive",
    "use_global_image_text_sigmoid_loss",
    "use_aux_token_ce",
    "use_aux_token_hard_neg",
    "use_appearance_cls_loss",
    "use_aggr",
    "use_sail_last_class_embed",
    "use_local_roi_query_fusion",
    "use_appearance_roi_fusion",
    "use_appearance_query_bias",
    "use_appearance_rescoring",
    "use_appearance_text_alignment",
    "use_appearance_text_rerank",
    "use_visual_confidence",
    "use_appearance_conditioned_gate",
    "use_appearance_token_branch",
    "use_appearance_contrastive_head",
    "use_appearance_contrastive_loss",
    "use_appearance_uncertainty_rerank",
    "use_proto_cls_head",
    "use_appearance_prompt",
    "use_appearance_class_fusion",
    "use_appearance_prototype_bank",
    "use_appearance_discriminative_residual",
    "use_appearance_local_evidence",
    "use_appearance_clip_evidence",
    "use_appearance_local_evidence_rerank",
    "use_appearance_local_evidence_zscore_fusion",
    "use_appearance_local_evidence_gated_replace",
    "use_appearance_local_evidence_winner_take_top1",
    "use_stage2_mixed_prompt_training",
    "use_stage2_mixed_prompt_inference",
    "use_stage2_candidate_rerank",
    "use_roi_appearance_branch",
    "use_appearance_prototype_fusion",
    "use_appearance_proto_rerank",
    "use_appearance_same_group_fusion",
    "use_appearance_seen_only_fusion",
    "use_appearance_neutral_fusion",
    "use_semantic_group_margin",
    "use_self_facb_loss",
    "use_self_facb_multi_facet",
    "use_zs_preserve_distill",
)


def _validate_pure_baseline_config(args):
    if not bool(getattr(args, "pure_baseline", False)):
        return

    enabled = [
        name for name in _PURE_BASELINE_FORBIDDEN_FLAGS
        if bool(getattr(args, name, False))
    ]
    if enabled:
        raise ValueError(
            "pure_baseline=True but research modules are enabled: "
            + ", ".join(enabled)
        )

    text_encoder_type = str(getattr(args, "text_encoder_type", "")).lower()
    if "modernbert" not in text_encoder_type:
        raise ValueError(
            "The controlled pure baseline requires the selected ModernBERT "
            "text encoder; "
            f"got {text_encoder_type!r}."
        )


def _infer_category_meta_path(dataset_meta):
    for split_name in ("val", "train"):
        for entry in dataset_meta.get(split_name, []):
            candidate = entry.get("anno")
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.suffix.lower() == ".json" and candidate_path.exists():
                return str(candidate_path)
    return None


def _refresh_tt100k_label_list(args, dataset_meta, logger=None):
    label_map_path = None
    for entry in dataset_meta.get("train", []):
        candidate = entry.get("label_map")
        if candidate:
            label_map_path = Path(candidate)
            break
    if label_map_path is None or not label_map_path.exists():
        return

    category_meta_path = (
        getattr(args, "category_meta_path", None)
        or getattr(args, "appearance_prompt_meta_path", None)
    )
    if not category_meta_path:
        category_meta_path = _infer_category_meta_path(dataset_meta)
        if category_meta_path is not None:
            args.appearance_prompt_meta_path = category_meta_path
            if getattr(args, "category_meta_path", None) is None:
                args.category_meta_path = category_meta_path
            if logger is not None:
                logger.info(f"Resolved TT100K category metadata from {category_meta_path}")

    with label_map_path.open("r", encoding="utf-8-sig") as handle:
        raw_label_map = json.load(handle)
    if logger is not None:
        logger.info(f"TT100K train label_map resolved to {label_map_path}")

    eval_prompt_text_source = getattr(
        args,
        "eval_prompt_text_source",
        getattr(args, "supervision_prompt_text_source", "semantic"),
    )
    label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source=eval_prompt_text_source,
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    args.label_list = [str(label_text_map[str(idx)]) for idx in range(len(raw_label_map))]
    semantic_label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source="semantic",
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    description_label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source="description",
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    args.semantic_label_list = [
        str(semantic_label_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    args.description_label_list = [
        str(description_label_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    appearance_label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source=getattr(args, "appearance_roi_text_source", "visual_cue_text"),
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    args.appearance_eval_label_list = [
        str(appearance_label_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    if getattr(args, "facb_speedhr_candidate_multi_facet", False):
        facet_label_map_path = getattr(args, "facb_speedhr_facet_label_map", None)
        facet_raw_label_map = raw_label_map
        if facet_label_map_path:
            facet_label_map_path = Path(facet_label_map_path)
            if not facet_label_map_path.is_absolute():
                facet_label_map_path = (
                    Path(args.config_file).resolve().parent / facet_label_map_path
                ).resolve()
            with facet_label_map_path.open("r", encoding="utf-8-sig") as handle:
                facet_raw_label_map = json.load(handle)
            if logger is not None:
                logger.info(
                    f"FACB compositional facet label_map resolved to {facet_label_map_path}"
                )
        args.facb_speedhr_multi_facet_label_list = (
            build_facb_compositional_facet_texts(
                facet_raw_label_map,
                category_meta_path=category_meta_path,
            )
        )
        args.facb_speedhr_facet_names = list(FACB_COMPOSITIONAL_FACET_NAMES)
        if logger is not None:
            preview = args.facb_speedhr_multi_facet_label_list[0]
            logger.info(
                "Built FACB compositional facets: "
                f"classes={len(args.facb_speedhr_multi_facet_label_list)} "
                f"facets={args.facb_speedhr_facet_names} preview={preview}"
            )
    if getattr(args, "facb_speedhr_shared_compositional_facets", False):
        shared_label_map_path = getattr(args, "facb_speedhr_facet_label_map", None)
        shared_raw_label_map = raw_label_map
        if shared_label_map_path:
            shared_label_map_path = Path(shared_label_map_path)
            if not shared_label_map_path.is_absolute():
                shared_label_map_path = (
                    Path(args.config_file).resolve().parent / shared_label_map_path
                ).resolve()
            with shared_label_map_path.open("r", encoding="utf-8-sig") as handle:
                shared_raw_label_map = json.load(handle)
        shared_schema = build_facb_shared_facet_schema(
            shared_raw_label_map,
            category_meta_path=category_meta_path,
            facet_names=getattr(
                args,
                "facb_speedhr_shared_facet_schema_names",
                None,
            ),
            include_composite_symbol=bool(
                getattr(
                    args,
                    "facb_speedhr_shared_facet_include_composite_symbol",
                    False,
                )
            ),
            composite_symbol_only=bool(
                getattr(
                    args,
                    "facb_speedhr_shared_facet_composite_symbol_only",
                    False,
                )
            ),
        )
        args.facb_speedhr_shared_facet_names = shared_schema["facet_names"]
        args.facb_speedhr_shared_facet_vocab_texts = shared_schema["vocab_texts"]
        args.facb_speedhr_shared_facet_class_targets = shared_schema[
            "class_target_ids"
        ]
        if logger is not None:
            vocab_sizes = [
                len(vocab) for vocab in args.facb_speedhr_shared_facet_vocab_texts
            ]
            example_id = min(225, len(shared_schema["class_facet_texts"]) - 1)
            logger.info(
                "Built FACB shared compositional schema: "
                f"facets={args.facb_speedhr_shared_facet_names} "
                f"vocab_sizes={vocab_sizes} example_class={example_id} "
                f"targets={shared_schema['class_facet_texts'][example_id]}"
            )
    if getattr(args, "use_self_facb_multi_facet", False):
        self_facb_label_map_path = getattr(args, "self_facb_label_map", None)
        self_facb_raw_label_map = raw_label_map
        if self_facb_label_map_path:
            self_facb_label_map_path = Path(self_facb_label_map_path)
            if not self_facb_label_map_path.is_absolute():
                self_facb_label_map_path = (Path(args.config_file).resolve().parent / self_facb_label_map_path).resolve()
            with self_facb_label_map_path.open("r", encoding="utf-8-sig") as handle:
                self_facb_raw_label_map = json.load(handle)
            if logger is not None:
                logger.info(f"Self-FACB facet label_map resolved to {self_facb_label_map_path}")
        facet_sources = getattr(
            args,
            "self_facb_facet_sources",
            ("facet_shape", "facet_symbol"),
        )
        if isinstance(facet_sources, str):
            facet_sources = [
                item.strip() for item in facet_sources.split(",") if item.strip()
            ]
        args.appearance_multi_facet_label_list = build_multi_facet_texts(
            self_facb_raw_label_map,
            category_meta_path=category_meta_path,
            facet_sources=facet_sources,
            allow_semantic_fallback=bool(getattr(args, "self_facb_allow_semantic_fallback", False)),
        )
    # Dual prototype: shared (structural) text source, e.g. attr_text.  Built as a
    # second [num_classes] list aligned to the same class-id order as
    # appearance_eval_label_list (the discriminative source).
    appearance_shared_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source=getattr(args, "appearance_shared_text_source", "attr_text"),
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    args.appearance_shared_label_list = [
        str(appearance_shared_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    infer_label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source="infer_text",
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    train_label_text_map = build_label_text_map(
        raw_label_map,
        enabled=getattr(args, "use_appearance_prompt", False),
        text_source="train_text",
        separator=getattr(args, "appearance_prompt_separator", ", "),
        category_meta_path=category_meta_path,
        deduplicate=True,
    )
    args.infer_label_list = [
        str(infer_label_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    args.train_label_list = [
        str(train_label_text_map[str(idx)]) for idx in range(len(raw_label_map))
    ]
    # Global [num_classes] fine-group id vector (class id order), used by the
    # prototype-bank appearance loss to mine same-fine-group hard negatives.
    try:
        fine_group_ids, fine_group_key_to_id = build_fine_group_ids(
            raw_label_map,
            category_meta_path=category_meta_path,
        )
        args.appearance_global_fine_group_ids = fine_group_ids
        if logger is not None:
            logger.info(
                f"TT100K appearance fine-groups: {len(fine_group_key_to_id)} groups "
                f"over {len(fine_group_ids)} classes."
            )
    except Exception as exc:  # noqa: BLE001 - non-fatal: bank loss falls back to all-class negatives
        args.appearance_global_fine_group_ids = None
        if logger is not None:
            logger.warning(f"Failed to build appearance fine-groups: {exc}")
    if logger is not None and args.label_list:
        logger.info(f"TT100K prompt label preview: {args.label_list[:2]}")
    if _env_flag("GROUNDINGDINO_PROMPT_DEBUG") and logger is not None:
        logger.info(
            "TT100K prompt debug: "
            f"eval_prompt_text_source={eval_prompt_text_source}, "
            f"use_appearance_prompt={getattr(args, 'use_appearance_prompt', False)}"
        )
        logger.info(f"TT100K prompt debug: label_list[:2]={args.label_list[:2]}")
        logger.info(
            "TT100K prompt debug: "
            f"train_label_list[:2]={args.train_label_list[:2]}"
        )
        logger.info(
            "TT100K prompt debug: "
            f"semantic_label_list[:2]={args.semantic_label_list[:2]}"
        )


def _restore_tt100k_prompt_lists_from_checkpoint(args, checkpoint, logger=None):
    checkpoint_args = _checkpoint_args_to_dict(
        checkpoint.get("args") if isinstance(checkpoint, dict) else None
    )
    if not checkpoint_args:
        return False

    base_label_list = checkpoint_args.get("label_list", None)
    if not base_label_list:
        return False

    base_label_list = [str(item) for item in base_label_list]
    args.label_list = base_label_list
    args.eval_label_list = [
        str(item) for item in checkpoint_args.get("eval_label_list", base_label_list)
    ]
    args.semantic_label_list = [
        str(item) for item in checkpoint_args.get("semantic_label_list", base_label_list)
    ]
    args.description_label_list = [
        str(item) for item in checkpoint_args.get("description_label_list", base_label_list)
    ]
    # Prototype-bank mode: the appearance bank text + fine-group vector are derived
    # from the CURRENT label map / category meta, not from the resumed checkpoint
    # (phase7_best was trained without appearance descriptions, so its args carry
    # only semantic labels).  Preserve the freshly-built values from
    # _refresh_tt100k_label_list instead of clobbering them with checkpoint args.
    _preserve_appearance_for_bank = bool(
        getattr(args, "use_appearance_prototype_bank", False)
    )
    _fresh_appearance_eval = getattr(args, "appearance_eval_label_list", None)
    _fresh_appearance_shared = getattr(args, "appearance_shared_label_list", None)
    _fresh_semantic = getattr(args, "semantic_label_list", None)
    _fresh_fine_groups = getattr(args, "appearance_global_fine_group_ids", None)
    args.appearance_eval_label_list = [
        str(item) for item in checkpoint_args.get("appearance_eval_label_list", base_label_list)
    ]
    if _preserve_appearance_for_bank and _fresh_appearance_eval:
        args.appearance_eval_label_list = list(_fresh_appearance_eval)
        args.appearance_global_fine_group_ids = _fresh_fine_groups
        if _fresh_appearance_shared:
            args.appearance_shared_label_list = list(_fresh_appearance_shared)
        # Local-evidence candidate ranking builds a semantic bank from this list;
        # keep it aligned to the CURRENT label map, not the resumed checkpoint.
        if _fresh_semantic:
            args.semantic_label_list = list(_fresh_semantic)
        if logger is not None:
            logger.info(
                "Prototype-bank mode: preserved freshly-built appearance label list "
                "and fine-group vector across checkpoint restore."
            )
    args.infer_label_list = [
        str(item) for item in checkpoint_args.get("infer_label_list", base_label_list)
    ]
    args.train_label_list = [
        str(item) for item in checkpoint_args.get("train_label_list", base_label_list)
    ]
    if logger is not None:
        logger.info(
            "Restored prompt label lists from resume checkpoint "
            f"({len(base_label_list)} labels)."
        )
    return True


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def _checkpoint_args_to_dict(checkpoint_args):
    if checkpoint_args is None:
        return {}
    if isinstance(checkpoint_args, dict):
        return checkpoint_args
    if hasattr(checkpoint_args, "__dict__"):
        return vars(checkpoint_args)
    return {}


def _build_teacher_args_from_checkpoint(args, checkpoint):
    teacher_args = copy.deepcopy(args)
    checkpoint_args = _checkpoint_args_to_dict(checkpoint.get("args") if isinstance(checkpoint, dict) else None)
    if not checkpoint_args:
        return teacher_args, False

    for key, value in checkpoint_args.items():
        if str(key).startswith("_"):
            continue
        setattr(teacher_args, key, value)

    # Keep runtime placement/distributed fields from the current process.
    for key in ("device", "rank", "local_rank", "distributed", "gpu"):
        if hasattr(args, key):
            setattr(teacher_args, key, getattr(args, key))
    return teacher_args, True


def build_teacher_model_if_needed(args, device, logger=None):
    if not bool(getattr(args, "use_zs_preserve_distill", False)):
        return None

    teacher_path = getattr(args, "zs_preserve_teacher_path", None)
    if not teacher_path:
        teacher_path = getattr(args, "pretrain_model_path", None)
    if not teacher_path:
        raise ValueError(
            "Zero-shot preservation distillation is enabled, but no teacher checkpoint path "
            "was provided. Set `zs_preserve_teacher_path` or `pretrain_model_path`."
        )

    checkpoint = load_checkpoint_compat(teacher_path, map_location="cpu")
    teacher_args, used_checkpoint_args = _build_teacher_args_from_checkpoint(args, checkpoint)
    teacher_model, _, _ = build_model_main(teacher_args)
    teacher_model.to(device)

    teacher_state = checkpoint.get("model", checkpoint)
    load_output = teacher_model.load_state_dict(clean_state_dict(teacher_state), strict=False)

    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    if logger is not None:
        logger.info(f"Loaded zero-shot teacher from {teacher_path}")
        logger.info(f"Teacher args loaded from checkpoint: {used_checkpoint_args}")
        logger.info(f"Teacher load status: {load_output}")

    return teacher_model


def _summarize_trainable_parameters(model):
    trainable_params = {name: param.numel() for name, param in model.named_parameters() if param.requires_grad}
    return {
        "num_parameter_tensors": len(trainable_params),
        "num_parameters": sum(trainable_params.values()),
    }


def _summarize_optimizer_param_groups(param_dicts, default_lr, default_weight_decay):
    summaries = []
    for idx, group in enumerate(param_dicts):
        params = group.get("params", [])
        if not params:
            continue
        summaries.append(
            {
                "group_index": idx,
                "lr": group.get("lr", default_lr),
                "weight_decay": group.get("weight_decay", default_weight_decay),
                "num_parameter_tensors": len(params),
                "num_parameters": sum(p.numel() for p in params),
            }
        )
    return summaries


def _apply_text_encoder_partial_unfreeze(model, args, logger=None):
    unfreeze_last_n = int(getattr(args, "text_encoder_unfreeze_last_n_layers", 0) or 0)
    unfreeze_embeddings = bool(getattr(args, "text_encoder_unfreeze_embeddings", False))
    if unfreeze_last_n <= 0 and not unfreeze_embeddings:
        return

    model_without_ddp = model.module if hasattr(model, "module") else model
    bert = getattr(model_without_ddp, "bert", None)
    if bert is None:
        raise AttributeError("Partial text encoder unfreeze requested, but model has no `bert` module.")

    trainable_tensors = 0
    trainable_params = 0
    if unfreeze_embeddings:
        for parameter in bert.embeddings.parameters():
            parameter.requires_grad_(True)
            trainable_tensors += 1
            trainable_params += parameter.numel()

    if unfreeze_last_n > 0:
        encoder_layers = getattr(bert, "layers", None) or getattr(
            getattr(bert, "encoder", None), "layer", None
        )
        if encoder_layers is None:
            raise AttributeError(
                "text_encoder_unfreeze_last_n_layers requires a ModernBERT "
                ".layers or BERT .encoder.layer module."
            )
        for layer in encoder_layers[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
                trainable_tensors += 1
                trainable_params += parameter.numel()

    if hasattr(model_without_ddp, "text_encoder_has_trainable_layers"):
        model_without_ddp.text_encoder_has_trainable_layers = True

    if logger is not None:
        logger.info(
            "partially unfroze text encoder: "
            f"last_n_layers={unfreeze_last_n}, embeddings={unfreeze_embeddings}, "
            f"parameter_tensors={trainable_tensors}, parameters={trainable_params}"
        )


def main(args):
    

    utils.setup_distributed(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        if bool(getattr(cfg, "skip_config_pretty_dump", False)):
            shutil.copyfile(args.config_file, save_cfg_path)
        else:
            try:
                cfg.dump(save_cfg_path)
            except Exception as exc:
                print(f"Warning: cfg.dump failed ({exc}). Falling back to raw config copy.")
                shutil.copyfile(args.config_file, save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    parser_defaults = _get_parser_default_map()
    option_keys = set()
    if isinstance(getattr(args, "options", None), dict):
        option_keys = set(args.options.keys())
    for k, v in cfg_dict.items():
        if k.startswith('_'):
            continue
        should_override = (
            k not in args_vars
            or k in option_keys
            or (k in parser_defaults and args_vars.get(k) == parser_defaults[k])
        )
        if should_override:
            setattr(args, k, v)

    if not hasattr(args, "modelname"):
        args.modelname = getattr(cfg, "modelname", "groundingdino")

    _validate_pure_baseline_config(args)

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    logger_output = None
    if args.output_dir and bool(getattr(args, "save_info_log", True)):
        logger_output = os.path.join(args.output_dir, 'info.txt')
    logger = setup_logger(output=logger_output, distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0 and bool(getattr(args, "save_config_snapshot", True)):
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets, encoding="utf-8-sig") as f:
        dataset_meta = json.load(f)
    dataset_meta = resolve_dataset_meta_paths(dataset_meta, args.datasets)
    seen_category_meta_path = None
    for entry in dataset_meta.get("train", []):
        candidate = entry.get("seen_category_meta_path")
        if candidate:
            seen_category_meta_path = candidate
            break
    if seen_category_meta_path:
        args.seen_category_meta_path = seen_category_meta_path
    if getattr(args, "use_coco_eval", False):
        args.coco_val_path = dataset_meta["val"][0]["anno"]
    for _key in ("eval_prompt_text_source", "supervision_prompt_text_source", "forward_prompt_text_source"):
        if getattr(args, _key, None) is None and hasattr(cfg, _key):
            setattr(args, _key, getattr(cfg, _key))
    _refresh_tt100k_label_list(args, dataset_meta, logger=logger)
    resume_checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            resume_checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_checkpoint = load_checkpoint_compat(args.resume, map_location='cpu')
        # Cross-dataset eval (e.g. GTSRB with a TT100K-trained checkpoint): the
        # checkpoint stores the TRAINING label list (232 TT100K classes).  Restoring
        # it here would overwrite the eval dataset's own label list, so the model
        # would be prompted with the wrong class space and AP collapses to 0.  Skip
        # the restore when the config opts out via skip_resume_label_restore=True.
        if getattr(args, "skip_resume_label_restore", False):
            if logger is not None:
                logger.info(
                    "skip_resume_label_restore=True: keeping the eval dataset's own "
                    "label lists (not the checkpoint's training labels)."
                )
        else:
            _restore_tt100k_prompt_lists_from_checkpoint(args, resume_checkpoint, logger=logger)
    _eval_label_list_file = getattr(args, "eval_label_list_file", None)
    if _eval_label_list_file:
        with open(_eval_label_list_file, "r", encoding="utf-8-sig") as _f:
            args.eval_label_list = json.load(_f)
        if logger is not None:
            logger.info(f"Loaded eval_label_list ({len(args.eval_label_list)} items) from {_eval_label_list_file}")
    if args.rank == 0 and bool(getattr(args, "save_config_snapshot", True)):
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        logger.info("Full config updated to {}".format(save_json_path))
    if bool(getattr(args, "save_config_snapshot", True)):
        _save_run_manifest(args, dataset_meta, logger=logger)

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    is_cuda_device = device.type == "cuda" and torch.cuda.is_available()
    if args.pin_memory is None:
        args.pin_memory = is_cuda_device
    if args.val_batch_size is None:
        args.val_batch_size = 4
    if is_cuda_device:
        if not args.disable_tf32:
            if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        if not args.disable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")

    # ---- Phase 2: matcher class-cost IDF reweighting ----
    # Precompute the GLOBAL token-DF over the full class list and hand it to the
    # criterion, so the matcher's class cost can IDF-weight a class's tokens (same
    # DF table / mechanism as PostProcess scoring).  No-op when the lambda is 0.
    _matcher_idf_lambda = float(getattr(args, "matcher_token_idf_lambda", 0.0))
    criterion.matcher_token_idf_lambda = _matcher_idf_lambda
    if _matcher_idf_lambda != 0.0:
        from models.GroundingDINO.groundingdino import compute_token_df_from_cat_list
        from groundingdino.util import get_tokenlizer
        _full_cat_list = list(getattr(args, "label_list", []) or [])
        if _full_cat_list:
            _tok = get_tokenlizer.get_tokenlizer(args.text_encoder_type)
            _df, _ndf = compute_token_df_from_cat_list(
                _tok, _full_cat_list, getattr(args, "max_text_len", 256)
            )
            criterion._matcher_token_df = _df
            criterion._matcher_token_df_num_classes = _ndf
            logger.info(
                f"[matcher-idf] lambda={_matcher_idf_lambda} global DF over "
                f"{_ndf} classes: {len(_df)} distinct token ids"
            )
        else:
            logger.warning(
                "[matcher-idf] lambda>0 but args.label_list is empty; "
                "matcher IDF disabled (falls back to uniform)."
            )


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("trainable params before freezing: " + json.dumps(_summarize_trainable_parameters(model)))

    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break
    _apply_text_encoder_partial_unfreeze(model, args, logger=logger)

    train_only_keywords = getattr(args, "train_only_keywords", None)
    if train_only_keywords:
        if isinstance(train_only_keywords, str):
            train_only_keywords = [train_only_keywords]
        else:
            train_only_keywords = [str(k) for k in train_only_keywords if str(k)]
        logger.info("train_only_keywords active: " + json.dumps(train_only_keywords))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(keyword in name for keyword in train_only_keywords))
        if bool(getattr(args, "sail_last_class_embed_train_blend", True)) is False:
            blend_freeze_keywords = (
                "sail_last_class_embed.output_blend_logit",
                "sail_last_class_embed.output_query_gate",
                "sail_last_class_embed.output_residual_log_scale",
            )
            frozen_blend_params = 0
            for name, parameter in model.named_parameters():
                if any(keyword in name for keyword in blend_freeze_keywords):
                    parameter.requires_grad_(False)
                    frozen_blend_params += int(parameter.numel())
            logger.info(
                "sail_last_class_embed token blend frozen: "
                f"params={frozen_blend_params}"
            )

    if train_only_keywords and bool(
        getattr(args, "train_only_eval_frozen_modules", False)
    ):
        original_train = model.train

        def _train_with_frozen_modules_in_eval(mode=True):
            original_train(mode)
            if mode:
                for module in model.modules():
                    if not any(
                        parameter.requires_grad
                        for parameter in module.parameters(recurse=True)
                    ):
                        module.eval()
            return model

        model.train = _train_with_frozen_modules_in_eval
        model.train(True)
        frozen_eval_modules = sum(
            1
            for module in model.modules()
            if not any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=True)
            )
        )
        logger.info(
            "train-only frozen modules forced to eval mode: "
            f"modules={frozen_eval_modules}"
        )

    # Visual confidence / appearance gate / old appearance branches:
    # freeze all except trainable heads AND force eval mode to silence dropout/bn.
    # These branches share parameters with the main detector path.
    if (
        getattr(args, "use_visual_confidence", False)
        or getattr(args, "use_appearance_conditioned_gate", False)
        or getattr(args, "use_appearance_token_branch", False)
    ):
        _freeze_patterns = []
        if getattr(args, "use_visual_confidence", False):
            _freeze_patterns.append("visual_confidence")
        if getattr(args, "use_appearance_conditioned_gate", False):
            _freeze_patterns.append("appearance_gate")
        if getattr(args, "use_appearance_token_branch", False):
            _freeze_patterns.append("appearance_token")
        for name, parameter in model.named_parameters():
            if not any(p in name for p in _freeze_patterns):
                parameter.requires_grad_(False)
        # Put non-trainable modules in eval mode so dropout/bn are frozen.
        # engine.py calls model.train() each epoch, so patch model.train().
        _orig_train = model.train
        def _freezing_train(mode: bool = True):
            _orig_train(mode)
            if mode:
                for _mn, _m in model.named_modules():
                    if not any(p in _mn for p in _freeze_patterns):
                        _m.eval()
        model.train = _freezing_train
        # Apply once now.
        for module_name, module in model.named_modules():
            if not any(p in module_name for p in _freeze_patterns):
                module.eval()

    # ROI appearance branch: freeze all except roi_appearance_*.
    # DIFFERENT from above: we do NOT force eval mode on frozen modules.
    # These branches are fully isolated (detached inputs + no shared
    # parameters), so the frozen detector can stay in train mode without
    # leaking gradient.
    _iso_patterns = []
    if getattr(args, "use_roi_appearance_branch", False):
        _iso_patterns.append("roi_appearance")
    if getattr(args, "use_appearance_contrastive_head", False):
        _iso_patterns.append("appearance_contrastive_head")
        _iso_patterns.append("appearance_concat_fusion")
        _iso_patterns.append("appearance_roi_ca_attn")
        _iso_patterns.append("appearance_roi_ca_norm")
        _iso_patterns.append("appearance_roi_ca_proj")
        _iso_patterns.append("appearance_roi_spatial_attn")
        # Per-class learnable discriminative residual lives directly on the model
        # as `appearance_discriminative_residual` (no shared substring with the
        # patterns above), so it MUST be whitelisted explicitly or it gets frozen
        # to requires_grad=False and never updates (residual stays all-zero).
        _iso_patterns.append("appearance_discriminative_residual")
        _iso_patterns.append("appearance_visual_centroids")
        # Candidate-conditioned local evidence head (Wq/Wk/Wv + logit scale) lives
        # at model.appearance_local_evidence_head.* — whitelist it or it gets frozen.
        _iso_patterns.append("appearance_local_evidence_head")
        if getattr(args, "appearance_gradient_to_backbone", False):
            _iso_patterns.append("backbone")
            _iso_patterns.append("input_proj")
    if getattr(args, "use_appearance_clip_evidence", False):
        # CLIP crop evidence head (logit scale + optional visual projection).  CLIP
        # towers inside are frozen by the module itself; only these tiny params train.
        # Independent of use_appearance_contrastive_head, so whitelist separately.
        _iso_patterns.append("appearance_clip_evidence_head")
    if getattr(args, "self_facb_branch_only", False):
        _iso_patterns.append("self_facb_head")
    # Method B (same-group token-margin) is NOT an isolated branch: it sharpens the
    # detector's OWN query-text alignment, so its gradient must reach class_embed /
    # decoder / bert.  If B is on we must NOT freeze the backbone into an isolated
    # appearance-only run — doing so would silently zero B's gradient (the trap that
    # killed earlier per-class-residual / local-evidence attempts).  So when B is the
    # active method, skip the isolation freeze entirely and let the normal
    # freeze_keywords / lr settings govern what trains.
    if getattr(args, "use_semantic_group_margin", False) and not getattr(
        args, "force_appearance_isolation_with_group_margin", False
    ):
        if _iso_patterns:
            logger.info(
                "Method B (semantic_group_margin) is ON: skipping appearance isolation "
                "freeze so gradient can reach the detector's own alignment "
                "(class_embed/decoder). Isolated appearance branches are not frozen-trained."
            )
        _iso_patterns = []
    if _iso_patterns:
        for name, parameter in model.named_parameters():
            if not any(p in name for p in _iso_patterns):
                parameter.requires_grad_(False)
        if getattr(args, "self_facb_branch_only", False):
            scorer_type = str(getattr(args, "self_facb_scorer_type", "local_evidence") or "local_evidence").strip().lower()
            for name, parameter in model.named_parameters():
                if "self_facb_head.scorer." in name and scorer_type == "decoder":
                    parameter.requires_grad_(False)
                if "self_facb_head.decoder." in name and scorer_type != "decoder":
                    parameter.requires_grad_(False)

    logger.info("trainable params after freezing: " + json.dumps(_summarize_trainable_parameters(model)))

    param_dicts = get_param_dict(args, model_without_ddp)
    logger.info(
        "optimizer param groups: "
        + json.dumps(_summarize_optimizer_param_groups(param_dicts, args.lr, args.weight_decay))
    )

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    logger.debug(f'number of validation samples: {len(dataset_val)}')

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            if getattr(args, "use_facb_smallshot_balanced_sampling", False):
                smallshot_ids = _load_facb_smallshot_ids(args)
                sampler_train = FACBSmallshotBalancedSampler(
                    dataset_train,
                    smallshot_ids=smallshot_ids,
                    batch_size=args.batch_size,
                    smallshot_per_batch=int(
                        getattr(args, "facb_smallshot_images_per_batch", 1)
                    ),
                    smallshot_anchor_cycle=int(
                        getattr(args, "facb_smallshot_anchor_cycle", 0)
                    ),
                )
                logger.info(
                    "Using FACB small-shot balanced sampling: "
                    f"smallshot_ids={len(smallshot_ids)}, "
                    f"smallshot_images={len(sampler_train.small_indices)}, "
                    f"anchor_images={len(sampler_train.anchor_indices)}, "
                    f"smallshot_per_batch={sampler_train.smallshot_per_batch}, "
                    f"smallshot_anchor_cycle={sampler_train.smallshot_anchor_cycle}"
                )
            elif getattr(args, "use_repeat_factor_sampling", False):
                repeat_factors = _compute_repeat_factors(
                    dataset_train,
                    repeat_thr=float(getattr(args, "repeat_factor_thr", 0.01)),
                    repeat_power=float(getattr(args, "repeat_factor_power", 0.5)),
                    repeat_max=float(getattr(args, "repeat_factor_max", 5.0)),
                )
                sampler_weights = np.asarray(repeat_factors, dtype=np.float64)
                if getattr(args, "use_small_object_oversampling", False):
                    small_boosts = np.asarray(
                        _compute_small_object_boosts(
                            dataset_train,
                            small_area_thr=float(getattr(args, "small_object_area_thr", 32.0 * 32.0)),
                            small_boost=float(getattr(args, "small_object_repeat_boost", 1.5)),
                        ),
                        dtype=np.float64,
                    )
                    sampler_weights = sampler_weights * small_boosts
                    logger.info(
                        "Using small object oversampling: "
                        f"area_thr={getattr(args, 'small_object_area_thr', 32.0 * 32.0)}, "
                        f"boost={getattr(args, 'small_object_repeat_boost', 1.5)}, "
                        f"avg_small_boost={float(np.mean(small_boosts)):.3f}, "
                        f"max_small_boost={float(np.max(small_boosts)):.3f}"
                    )
                sampler_train = WeightedRandomSampler(
                    weights=torch.as_tensor(sampler_weights, dtype=torch.double),
                    num_samples=len(sampler_weights),
                    replacement=True,
                )
                logger.info(
                    "Using repeat factor sampling: "
                    f"thr={getattr(args, 'repeat_factor_thr', 0.01)}, "
                    f"power={getattr(args, 'repeat_factor_power', 0.5)}, "
                    f"max={getattr(args, 'repeat_factor_max', 5.0)}, "
                    f"avg_repeat={float(np.mean(repeat_factors)):.3f}, "
                    f"max_repeat={float(np.max(repeat_factors)):.3f}, "
                    f"avg_weight={float(np.mean(sampler_weights)):.3f}, "
                    f"max_weight={float(np.max(sampler_weights)):.3f}"
                )
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        train_loader_kwargs = dict(
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if args.num_workers > 0:
            if args.prefetch_factor is not None:
                train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
            train_loader_kwargs["persistent_workers"] = not args.disable_persistent_workers
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            **train_loader_kwargs,
        )

    val_loader_kwargs = dict(
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        if args.prefetch_factor is not None:
            val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["persistent_workers"] = not args.disable_persistent_workers
    data_loader_val = DataLoader(
        dataset_val,
        args.val_batch_size,
        sampler=sampler_val,
        drop_last=False,
        **val_loader_kwargs,
    )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = load_checkpoint_compat(args.frozen_weights, map_location='cpu')
        frozen_load_output = model_without_ddp.detr.load_state_dict(
            clean_state_dict(checkpoint['model']),
            strict=False,
        )
        logger.info(f"Loaded frozen weights with status: {frozen_load_output}")

    output_dir = Path(args.output_dir)
    auto_resume = getattr(args, "auto_resume", True)
    resume_load_optimizer = getattr(args, "resume_load_optimizer", True)
    resume_load_lr_scheduler = getattr(args, "resume_load_lr_scheduler", True)
    resume_keep_epoch = getattr(args, "resume_keep_epoch", True)

    if auto_resume and (not args.resume) and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        logger.info(f"Auto-resume enabled: found existing checkpoint at {args.resume}")
    if args.resume:
        checkpoint = resume_checkpoint
        if checkpoint is None:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = load_checkpoint_compat(args.resume, map_location='cpu')
        resume_load_output = model_without_ddp.load_state_dict(
            clean_state_dict(checkpoint['model']),
            strict=False,
        )
        logger.info(f"Resumed model weights from {args.resume}")
        logger.info(f"Resume load status: {resume_load_output}")


        if not args.eval and 'epoch' in checkpoint and resume_keep_epoch:
            args.start_epoch = checkpoint['epoch'] + 1

        if not args.eval and resume_load_optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Loaded optimizer state from checkpoint")
        elif args.resume and not args.eval:
            logger.info("Skipped optimizer state when resuming; using a fresh optimizer")

        if not args.eval and resume_load_lr_scheduler and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("Loaded lr scheduler state from checkpoint")
        elif args.resume and not args.eval:
            logger.info("Skipped lr scheduler state when resuming; using the current config schedule")

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = load_checkpoint_compat(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        if "modernbert" in str(getattr(model_without_ddp.bert.config, "model_type", "")).lower():
            _ignorekeywordlist = list(_ignorekeywordlist) + ["bert."]
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        logger.info(
            "Ignored {} pretrained keys (first 20): {}".format(
                len(ignorelist), json.dumps(ignorelist[:20])
            )
        )

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        missing_bert_keys = [
            key for key in _load_output.missing_keys if key.startswith("bert.")
        ]
        frozen_text_encoder = bool(getattr(args, "freeze_text_encoder", False))
        text_encoder_partially_trainable = (
            int(getattr(args, "text_encoder_unfreeze_last_n_layers", 0) or 0) > 0
            or bool(getattr(args, "text_encoder_unfreeze_embeddings", False))
        )
        allow_mismatch = bool(
            getattr(args, "allow_frozen_text_encoder_mismatch", False)
        )
        if (
            missing_bert_keys
            and frozen_text_encoder
            and not text_encoder_partially_trainable
            and not allow_mismatch
        ):
            preview = ", ".join(missing_bert_keys[:5])
            raise RuntimeError(
                "Pretrained checkpoint did not initialize the configured text "
                f"encoder ({len(missing_bert_keys)} missing BERT keys; first: "
                f"{preview}). The text encoder is frozen, so training cannot "
                "repair this cross-modal mismatch. Use a checkpoint matching "
                "text_encoder_type, unfreeze an explicit subset of the text "
                "encoder for alignment, or set "
                "allow_frozen_text_encoder_mismatch=True for a diagnostic run."
            )
        if missing_bert_keys and text_encoder_partially_trainable:
            logger.warning(
                "The detector checkpoint does not contain the configured text "
                "encoder weights. This is expected for the ModernBERT baseline; "
                "cross-modal alignment must be learned through the explicitly "
                "unfrozen ModernBERT layers and feat_map."
            )

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)

        log_stats = {**{f'eval_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "train.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            _save_eval_results(output_dir, test_stats, args, epoch=None, mode="eval")

        return



    print("Start training")
    start_time = time.time()
    teacher_model = build_teacher_model_if_needed(args, device, logger=logger)
    best_map_holder = BestMetricHolder(use_ema=False)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), teacher_model=teacher_model)

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            # Always save latest per-epoch checkpoint (overwrites), plus best separately.
            # Total: only two checkpoint files at any time.
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        eval_interval = int(getattr(args, "eval_interval", 0) or 0)
        should_eval = (
            eval_interval > 0
            and (
                ((epoch + 1) % eval_interval == 0)
                or ((epoch + 1) == args.epochs)
            )
        )
        test_stats = None
        coco_evaluator = None
        if should_eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in test_stats.items()},
            }
            _save_eval_results(output_dir, test_stats, args, epoch=epoch, mode="train_eval")
        else:
            if eval_interval > 0:
                logger.info(
                    f"Skipping evaluation at epoch {epoch + 1}; "
                    f"next eval every {eval_interval} epochs."
                )
            else:
                logger.info(
                    f"Skipping evaluation at epoch {epoch + 1}; "
                    "training-time evaluation is disabled (eval_interval=0)."
                )
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
            }


        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process() and bool(getattr(args, "save_epoch_log", True)):
            with (output_dir / "train.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                save_eval_interval = int(getattr(args, "save_eval_interval", 0) or 0)
                if not bool(getattr(args, "save_eval_artifacts", True)):
                    save_eval_interval = 0
                should_save_eval = (
                    save_eval_interval > 0
                    and (
                        ((epoch + 1) % save_eval_interval == 0)
                        or ((epoch + 1) == args.epochs)
                    )
                )
                if should_save_eval and "bbox" in coco_evaluator.coco_eval:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    eval_name = f'epoch{epoch + 1:04d}.pth'
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        output_dir / "eval" / eval_name,
                    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
