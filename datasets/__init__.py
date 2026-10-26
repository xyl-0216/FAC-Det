# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

import torch.utils.data
import torchvision
from pycocotools.coco import COCO

from .coco import build as build_coco


def _resolve_odvg_category_name(raw_entry):
    if isinstance(raw_entry, dict):
        for key in ("semantic", "name", "label", "text", "category", "description"):
            value = str(raw_entry.get(key, "")).strip()
            if value:
                return value
        return ""
    return str(raw_entry).strip()


def _build_coco_api_from_odvg(dataset):
    if getattr(dataset, "dataset_mode", None) != "OD":
        return None

    cached = getattr(dataset, "coco", None)
    if cached is not None:
        return cached

    raw_label_map = getattr(dataset, "raw_label_map", {}) or {}
    categories = []
    for label_id in sorted(raw_label_map.keys(), key=lambda item: int(item)):
        categories.append(
            {
                "id": int(label_id),
                "name": _resolve_odvg_category_name(raw_label_map[label_id]),
            }
        )

    images = []
    annotations = []
    ann_id = 1
    for image_id, meta in enumerate(getattr(dataset, "metas", [])):
        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        images.append(
            {
                "id": int(image_id),
                "file_name": str(meta.get("filename", "")),
                "width": width,
                "height": height,
            }
        )

        detection = meta.get("detection", {}) or {}
        for instance in detection.get("instances", []):
            bbox = instance.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": int(ann_id),
                    "image_id": int(image_id),
                    "category_id": int(instance.get("label", -1)),
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco.createIndex()
    dataset.coco = coco
    return coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if getattr(dataset, "dataset_mode", None) == "OD":
        return _build_coco_api_from_odvg(dataset)


def build_dataset(image_set, args, datasetinfo):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco(image_set, args, datasetinfo)
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg
        return build_odvg(image_set, args, datasetinfo)
    raise ValueError(f'dataset {args.dataset_file} not supported')
