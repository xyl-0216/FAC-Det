# ------------------------------------------------------------------------
# Grounding DINO. Midified by Shilong Liu.
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from groundingdino.util.misc import all_gather, all_gather_cpu


class CocoGroundingEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True, category_subsets=None):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats
        self.category_subsets = category_subsets or {}
        self.subset_stats = {k: {} for k in iou_types}
        # Keep full-eval detections across all update() calls; subset AP should
        # be computed from the whole validation set, not only the last batch.
        self.coco_results = {k: [] for k in iou_types}
        self.coco_results_merged = {k: [] for k in iou_types}
        self.coco_dt_full = {}
        self.subset_debug = {k: {} for k in iou_types}
        self.debug_subset_eval = os.environ.get("GROUNDINGDINO_EVAL_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self.coco_results[iou_type].extend(results)

    def synchronize_between_processes(self):
        gathered_img_ids = all_gather_cpu(self.img_ids)
        merged_img_ids = []
        for rank_img_ids in gathered_img_ids:
            merged_img_ids.extend(rank_img_ids)
        if merged_img_ids:
            merged_img_ids = list(np.unique(np.asarray(merged_img_ids)))
        self.img_ids = merged_img_ids

        for iou_type in self.iou_types:
            gathered_results = all_gather_cpu(self.coco_results[iou_type])
            merged_results = []
            for rank_results in gathered_results:
                merged_results.extend(rank_results)
            self.coco_results_merged[iou_type] = merged_results

            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco_dt_full[iou_type] = COCO.loadRes(self.coco_gt, merged_results) if merged_results else COCO()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = self.coco_dt_full[iou_type]
            coco_eval.params.imgIds = list(self.img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)
            create_common_coco_eval(coco_eval, img_ids, eval_imgs)
            self.eval_imgs[iou_type] = eval_imgs

            # The full-image evaluation objects are no longer needed after the
            # consolidated COCOeval pass.
            self.coco_results[iou_type] = []

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            self.subset_stats[iou_type] = {}
            if os.environ.get("GROUNDINGDINO_SKIP_SUBSET_EVAL", "").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                continue
            for subset_name, subset_cat_ids in self.category_subsets.items():
                subset_cat_ids = sorted(set(int(cat_id) for cat_id in subset_cat_ids))
                if not subset_cat_ids:
                    continue
                subset_eval = self._build_subset_eval(coco_eval, iou_type, subset_cat_ids)
                print("IoU metric: {} ({} class)".format(iou_type, subset_name))
                subset_eval.summarize()
                self.subset_stats[iou_type][subset_name] = subset_eval.stats.tolist()
                self.subset_debug[iou_type][subset_name] = {
                    "images": len(subset_eval.params.imgIds),
                    "categories": len(subset_cat_ids),
                    "gt_annotations": len(getattr(subset_eval.cocoGt, "dataset", {}).get("annotations", [])),
                    "dt_annotations": len(getattr(subset_eval.cocoDt, "dataset", {}).get("annotations", [])),
                }
                if self.debug_subset_eval:
                    debug = self.subset_debug[iou_type][subset_name]
                    print(
                        "[SubsetDebug][{}][{}] images={} categories={} gt_anns={} dt_anns={}".format(
                            iou_type,
                            subset_name,
                            debug["images"],
                            debug["categories"],
                            debug["gt_annotations"],
                            debug["dt_annotations"],
                        )
                    )

    def _build_subset_coco(self, coco_api, subset_cat_ids, img_ids):
        source_dataset = getattr(coco_api, "dataset", {}) or {}
        gt_dataset = getattr(self.coco_gt, "dataset", {}) or {}

        subset_cat_ids = set(int(cat_id) for cat_id in subset_cat_ids)
        img_id_set = set(int(img_id) for img_id in img_ids)

        source_images = source_dataset.get("images", []) or gt_dataset.get("images", [])
        source_categories = source_dataset.get("categories", []) or gt_dataset.get("categories", [])
        source_annotations = source_dataset.get("annotations", [])

        base_dataset = {
            "images": [
                dict(img) for img in source_images
                if int(img.get("id", -1)) in img_id_set
            ],
            "categories": [
                dict(cat) for cat in source_categories
                if int(cat.get("id", -1)) in subset_cat_ids
            ],
            "annotations": [
                dict(ann) for ann in source_annotations
                if int(ann.get("category_id", -1)) in subset_cat_ids
                and int(ann.get("image_id", -1)) in img_id_set
            ],
        }

        for optional_key in ("info", "licenses"):
            if optional_key in source_dataset:
                base_dataset[optional_key] = copy.deepcopy(source_dataset[optional_key])
            elif optional_key in gt_dataset:
                base_dataset[optional_key] = copy.deepcopy(gt_dataset[optional_key])

        subset_coco = COCO()
        subset_coco.dataset = base_dataset
        subset_coco.createIndex()
        return subset_coco

    def _build_subset_eval(self, coco_eval, iou_type, subset_cat_ids):
        img_ids = [int(img_id) for img_id in coco_eval.params.imgIds]
        subset_gt = self._build_subset_coco(self.coco_gt, subset_cat_ids, img_ids)
        dt_source = self.coco_dt_full.get(iou_type, None)
        if dt_source is None:
            dt_source = coco_eval.cocoDt
        subset_dt = self._build_subset_coco(dt_source, subset_cat_ids, img_ids)

        subset_eval = COCOeval(subset_gt, subset_dt, iouType=iou_type)
        subset_eval.useCats = self.useCats
        subset_eval.params.imgIds = img_ids
        subset_eval.params.useCats = self.useCats
        subset_eval.params.maxDets = list(coco_eval.params.maxDets)
        subset_eval.params.areaRng = copy.deepcopy(coco_eval.params.areaRng)
        subset_eval.params.areaRngLbl = copy.deepcopy(coco_eval.params.areaRngLbl)
        if self.useCats:
            subset_eval.params.catIds = subset_cat_ids

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                subset_eval.evaluate()
        subset_eval.accumulate()
        return subset_eval

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": int(original_id),
                        "category_id": int(labels[k]),
                        "bbox": box,
                        "score": float(scores[k]),
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] 
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')

    # import pdb;pdb.set_trace()
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId) 
        for imgId in p.imgIds 
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet) 
        for catId in catIds 
        for areaRng in p.areaRng 
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
