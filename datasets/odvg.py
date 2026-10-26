from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
import re
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import random
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import datasets.transforms as T
from datasets.appearance_prompt import (
    AppearancePromptBuilder,
    build_label_text_map,
    normalize_raw_label_map,
)
from util.caption_builder import build_caption_from_labels, normalize_caption_label_text
from groundingdino.util import get_tokenlizer


GENERIC_DESCRIPTOR_STOPWORDS = {
    "a", "an", "and", "ahead", "all", "area", "at", "by", "for", "from", "in",
    "into", "of", "on", "or", "the", "to", "with",
}

GENERIC_DESCRIPTOR_TEMPLATE_WORDS = {
    "background", "barred", "black", "blue", "bold", "border", "bordered",
    "center", "centered", "circular", "circle", "contains", "depicts", "diagonal",
    "edge", "equilateral", "face", "features", "graphic", "indicates", "indicating",
    "inside", "letter", "numeral", "pointing", "red", "shape", "shaped", "shows",
    "sign", "silhouette", "symbol", "traffic", "triangular", "triangle", "warning",
    "white",
}


def _normalize_meta_text(value):
    if value is None:
        return ""
    return str(value).strip()

class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        category_meta_path: str = None,
        use_appearance_prompt: bool = False,
        appearance_prompt_text_source: str = "semantic",
        use_dual_prompt_training: bool = False,
        use_stage2_mixed_prompt_training: bool = False,
        use_component_full_text_supervision: bool = False,
        component_supervision_main_text_source: str = "train_text",
        component_supervision_attr_text_source: str = "attr_text",
        query_contrastive_text_source: str = "train_text",
        appearance_roi_text_source: str = "visual_cue_text",
        forward_prompt_text_source: str = "description",
        supervision_prompt_text_source: str = "semantic",
        appearance_roi_num_neg: int = 8,
        appearance_prompt_separator: str = ", ",
        appearance_prompt_meta_path: str = None,
        text_encoder_type: str = None,
        max_text_len: int = 256,
        max_label_tokens: int = None,
        use_hard_negative_sampling: bool = True,
        hard_negative_ratio: float = 0.6,
        hard_negative_min_score: float = 1.0,
        ignore_unseen_negative_loss: bool = False,
        negative_label_pool: str = "all",
        use_full_label_caption_training: bool = False,
        prioritize_positive_caption_labels: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        self.category_meta_path = category_meta_path or appearance_prompt_meta_path
        self.use_dual_prompt_training = bool(use_dual_prompt_training)
        self.use_stage2_mixed_prompt_training = bool(use_stage2_mixed_prompt_training)
        self.use_component_full_text_supervision = bool(use_component_full_text_supervision)
        self.appearance_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=appearance_prompt_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.forward_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=forward_prompt_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.supervision_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=supervision_prompt_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.component_main_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=component_supervision_main_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.component_attr_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=component_supervision_attr_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.query_contrastive_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=query_contrastive_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.appearance_roi_prompt_builder = AppearancePromptBuilder(
            enabled=use_appearance_prompt,
            text_source=appearance_roi_text_source,
            separator=appearance_prompt_separator,
            category_meta_path=self.category_meta_path,
        )
        self.appearance_roi_num_neg = max(0, int(appearance_roi_num_neg))
        self.text_encoder_type = text_encoder_type
        self.max_text_len = max_text_len
        self.max_label_tokens = max_label_tokens
        self.use_hard_negative_sampling = use_hard_negative_sampling
        self.hard_negative_ratio = max(0.0, min(1.0, float(hard_negative_ratio)))
        self.hard_negative_min_score = float(hard_negative_min_score)
        self.ignore_unseen_negative_loss = bool(ignore_unseen_negative_loss)
        self.negative_label_pool = str(negative_label_pool).lower()
        self.use_full_label_caption_training = bool(use_full_label_caption_training)
        self.prioritize_positive_caption_labels = bool(
            prioritize_positive_caption_labels
        )
        if self.negative_label_pool not in {"all", "present"}:
            raise ValueError(
                f"Unsupported negative_label_pool: {negative_label_pool}. "
                "Expected one of {'all', 'present'}."
            )
        self.text_tokenizer = None
        self.caption_special_tokens = 0
        self.label_token_lengths = {}
        self.supervision_label_token_lengths = {}
        self.label_descriptors = {}
        self.caption_negative_pool = set()
        self.present_label_ids = set()
        self.label_groups = {}
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
            self._init_text_tokenizer()
            self._init_label_descriptors()
            self._log_group_statistics()
        self._load_metas(anno)
        if self.dataset_mode == "OD":
            self._init_present_label_ids()
            self._init_caption_negative_pool()
            self._log_caption_loss_supervision()
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        # Use utf-8-sig so Windows-generated JSON files with BOM
        # can be parsed reliably.
        with open(label_map_anno, 'r', encoding='utf-8-sig') as file:
            raw_label_map = json.load(file)
        self.raw_label_map = normalize_raw_label_map(raw_label_map)
        self.label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.appearance_prompt_builder.enabled,
            text_source=self.appearance_prompt_builder.text_source,
            separator=self.appearance_prompt_builder.separator,
            category_meta_path=self.appearance_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        self.forward_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.forward_prompt_builder.enabled,
            text_source=self.forward_prompt_builder.text_source,
            separator=self.forward_prompt_builder.separator,
            category_meta_path=self.forward_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        self.supervision_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.supervision_prompt_builder.enabled,
            text_source=self.supervision_prompt_builder.text_source,
            separator=self.supervision_prompt_builder.separator,
            category_meta_path=self.supervision_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        self.component_main_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.component_main_prompt_builder.enabled,
            text_source=self.component_main_prompt_builder.text_source,
            separator=self.component_main_prompt_builder.separator,
            category_meta_path=self.component_main_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        self.component_attr_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.component_attr_prompt_builder.enabled,
            text_source=self.component_attr_prompt_builder.text_source,
            separator=self.component_attr_prompt_builder.separator,
            category_meta_path=self.component_attr_prompt_builder.category_meta_path,
            deduplicate=False,
        )
        self.query_contrastive_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.query_contrastive_prompt_builder.enabled,
            text_source=self.query_contrastive_prompt_builder.text_source,
            separator=self.query_contrastive_prompt_builder.separator,
            category_meta_path=self.query_contrastive_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        self.appearance_roi_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=self.appearance_roi_prompt_builder.enabled,
            text_source=self.appearance_roi_prompt_builder.text_source,
            separator=self.appearance_roi_prompt_builder.separator,
            category_meta_path=self.appearance_roi_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        # Distillation should preserve the Phase1 semantic prompt space even when
        # the Stage2 training caption is expanded with descriptions.
        self.distill_label_map = build_label_text_map(
            self.raw_label_map,
            enabled=False,
            text_source="semantic",
            separator=self.supervision_prompt_builder.separator,
            category_meta_path=self.supervision_prompt_builder.category_meta_path,
            deduplicate=True,
        )
        if not self.use_dual_prompt_training:
            self.forward_label_map = self.label_map
            self.supervision_label_map = self.label_map
        if self.use_stage2_mixed_prompt_training:
            self.forward_label_map = self.label_map
        self.label_index = set(self.label_map.keys())

    def _init_text_tokenizer(self):
        if not self.text_encoder_type:
            return

        self.text_tokenizer = get_tokenlizer.get_tokenlizer(self.text_encoder_type)
        self.caption_special_tokens = self.text_tokenizer.num_special_tokens_to_add(pair=False)
        for label_id in self.label_map.keys():
            candidate_texts = [self.label_map[label_id]]
            if self.use_dual_prompt_training or self.use_stage2_mixed_prompt_training:
                candidate_texts.extend(
                    [
                        self.forward_label_map[label_id],
                        self.supervision_label_map[label_id],
                    ]
                )
            if self.use_component_full_text_supervision:
                candidate_texts.extend(
                    [
                        self.component_main_label_map.get(label_id, ""),
                        self.component_attr_label_map.get(label_id, ""),
                    ]
                )
            candidate_texts.append(self.query_contrastive_label_map.get(label_id, ""))
            if self.use_stage2_mixed_prompt_training:
                candidate_texts.append(
                    self.forward_prompt_builder.build(label_id, self.raw_label_map[label_id])
                )

            token_lengths = []
            for label_text in dict.fromkeys(candidate_texts):
                tokenized = self.text_tokenizer(
                    f"{label_text} .",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                token_lengths.append(len(tokenized["input_ids"]))
            if self.use_stage2_mixed_prompt_training:
                self.label_token_lengths[label_id] = sum(token_lengths)
            else:
                self.label_token_lengths[label_id] = max(token_lengths) if token_lengths else 0
            supervision_text = self.supervision_label_map.get(label_id, self.label_map[label_id])
            supervision_tokenized = self.text_tokenizer(
                f"{supervision_text} .",
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            self.supervision_label_token_lengths[label_id] = len(
                supervision_tokenized["input_ids"]
            )

    def _get_label_token_budget(self):
        if self.max_label_tokens is not None:
            return int(self.max_label_tokens)
        if self.max_text_len is not None and self.max_text_len > 0:
            return int(self.max_text_len) - int(self.caption_special_tokens)
        return None

    def _resolve_label_text(self, label_id):
        label_entry = self.raw_label_map[str(label_id)]
        if isinstance(label_entry, dict):
            for key in ("description", "semantic", "name", "label", "text"):
                value = str(label_entry.get(key, "")).strip()
                if value:
                    return value
            return ""
        return str(label_entry).strip()

    def _iter_descriptor_texts(self, label_id):
        texts = []
        for mapping in (
            getattr(self, "label_map", {}),
            getattr(self, "forward_label_map", {}),
            getattr(self, "supervision_label_map", {}),
            getattr(self, "distill_label_map", {}),
        ):
            value = str(mapping.get(label_id, "")).strip()
            if value:
                texts.append(value)

        fallback_text = self._resolve_label_text(label_id)
        if fallback_text:
            texts.append(fallback_text)

        return list(dict.fromkeys(texts))

    def _resolve_label_entry(self, label_id):
        return self.raw_label_map[str(label_id)]

    def _resolve_label_meta_text(self, label_id, key):
        label_entry = self._resolve_label_entry(label_id)
        if not isinstance(label_entry, dict):
            return ""
        return _normalize_meta_text(label_entry.get(key))

    def _tokenize_descriptor_words(self, text):
        words = set()
        for token in re.findall(r"[a-z]+", str(text).lower()):
            if token in GENERIC_DESCRIPTOR_STOPWORDS:
                continue
            if token in GENERIC_DESCRIPTOR_TEMPLATE_WORDS:
                continue
            if len(token) == 1 and token not in {"m", "t"}:
                continue
            words.add(token)
        return words

    def _extract_descriptor_measurements(self, text):
        value = str(text).lower()
        return set(
            re.findall(
                r"\d+(?:\.\d+)?\s*(?:km/?h|kmh|m|t)",
                value,
            )
        )

    def _init_label_descriptors(self):
        for label_id in self.label_map.keys():
            texts = self._iter_descriptor_texts(label_id)
            words = set()
            numbers = set()
            measurements = set()
            attr_words = set()
            attr_measurements = set()
            for label_text in texts:
                words.update(self._tokenize_descriptor_words(label_text))
                numbers.update(re.findall(r"\d+(?:\.\d+)?", label_text.lower()))
                measurements.update(self._extract_descriptor_measurements(label_text))
            attr_text = self._resolve_label_meta_text(label_id, "attr_text")
            if attr_text:
                attr_words.update(self._tokenize_descriptor_words(attr_text))
                attr_measurements.update(self._extract_descriptor_measurements(attr_text))
            group_id = self.appearance_prompt_builder.get_group_id(
                label_id, self.raw_label_map[label_id]
            )
            appearance_group = self.appearance_prompt_builder.get_appearance_group(
                label_id, self.raw_label_map[label_id]
            )
            self.label_groups[label_id] = {
                "group_id": group_id,
                "appearance_group": appearance_group,
            }
            self.label_descriptors[label_id] = {
                "words": words,
                "numbers": numbers,
                "measurements": measurements,
                "attr_words": attr_words,
                "attr_measurements": attr_measurements,
                "group_id": group_id,
                "appearance_group": appearance_group,
            }

    def _log_group_statistics(self):
        group_counts = {group_name: 0 for group_name in ("indication", "prohibition", "warning")}
        invalid_count = 0

        for label_id in self.label_map.keys():
            group_name = self.appearance_prompt_builder.get_group_name(
                label_id, self.raw_label_map[label_id]
            )
            if group_name in group_counts:
                group_counts[group_name] += 1
            else:
                invalid_count += 1

        summary = ", ".join(
            [f"{group_name}={count}" for group_name, count in group_counts.items()]
            + [f"invalid={invalid_count}"]
        )
        print(f"  == category groups: {summary}")
        if invalid_count == len(self.label_map):
            print(
                "  == warning: no valid category groups were resolved; "
                "set `category_meta_path` to TT100K category metadata so AGGR can take effect."
            )

    def _init_present_label_ids(self):
        present_label_ids = set()
        for meta in self.metas:
            detection = meta.get("detection", {})
            for instance in detection.get("instances", []):
                label_id = str(instance.get("label"))
                if label_id in self.label_index:
                    present_label_ids.add(label_id)

        self.present_label_ids = self.label_index.intersection(present_label_ids)
        print(
            "  == labels present in annotations: "
            f"{len(self.present_label_ids)}/{len(self.label_index)}"
        )

    def _init_caption_negative_pool(self):
        if self.negative_label_pool == "all":
            self.caption_negative_pool = set(self.label_index)
            print(f"  == negative label pool: all labels ({len(self.caption_negative_pool)})")
            return

        self.caption_negative_pool = set(self.present_label_ids)
        if not self.caption_negative_pool:
            self.caption_negative_pool = set(self.label_index)
            print(
                "  == warning: no present-label negative pool could be resolved; "
                f"falling back to all labels ({len(self.caption_negative_pool)})"
            )
            return

        print(
            "  == negative label pool: present labels "
            f"({len(self.caption_negative_pool)}/{len(self.label_index)})"
        )

    def _log_caption_loss_supervision(self):
        if not self.ignore_unseen_negative_loss:
            return

        if not self.present_label_ids:
            print(
                "  == warning: ignore_unseen_negative_loss is enabled, but no present labels "
                "were resolved; keeping all caption labels supervised."
            )
            return

        ignored_label_count = len(self.label_index.difference(self.present_label_ids))
        print(
            "  == caption loss supervision: positives + present labels only "
            f"(ignoring negative loss for {ignored_label_count} absent labels)"
        )

    def _hard_negative_score(self, pos_label_id, neg_label_id):
        pos_desc = self.label_descriptors.get(pos_label_id)
        neg_desc = self.label_descriptors.get(neg_label_id)
        if pos_desc is None or neg_desc is None:
            return 0.0

        score = 0.0
        pos_group_id = pos_desc["group_id"]
        neg_group_id = neg_desc["group_id"]
        if pos_group_id >= 0 and pos_group_id == neg_group_id:
            score += 1.5
        if (
            pos_desc.get("appearance_group")
            and pos_desc.get("appearance_group") == neg_desc.get("appearance_group")
        ):
            score += 2.0

        shared_measurements = pos_desc["measurements"].intersection(neg_desc["measurements"])
        if shared_measurements:
            score += 4.0 * len(shared_measurements)

        shared_numbers = pos_desc["numbers"].intersection(neg_desc["numbers"])
        if shared_numbers:
            score += 2.5 * len(shared_numbers)

        shared_words = pos_desc["words"].intersection(neg_desc["words"])
        if shared_words:
            union_words = pos_desc["words"].union(neg_desc["words"])
            score += 1.0 + 3.0 * (len(shared_words) / max(len(union_words), 1))

        shared_attr_words = pos_desc["attr_words"].intersection(neg_desc["attr_words"])
        if shared_attr_words:
            score -= 1.0 + 2.0 * (
                len(shared_attr_words)
                / max(len(pos_desc["attr_words"].union(neg_desc["attr_words"])), 1)
            )

        shared_attr_measurements = pos_desc["attr_measurements"].intersection(
            neg_desc["attr_measurements"]
        )
        if shared_attr_measurements:
            score -= 3.0 * len(shared_attr_measurements)

        return score

    def _group_negative_labels(self, pos_labels, candidate_neg_labels):
        same_appearance = []
        same_group = []
        other = []
        pos_labels = list(pos_labels)

        for neg_label_id in candidate_neg_labels:
            neg_meta = self.label_groups.get(neg_label_id, {})
            placed = False
            for pos_label_id in pos_labels:
                pos_meta = self.label_groups.get(pos_label_id, {})
                if (
                    pos_meta.get("appearance_group")
                    and pos_meta.get("appearance_group") == neg_meta.get("appearance_group")
                ):
                    same_appearance.append(neg_label_id)
                    placed = True
                    break
            if placed:
                continue

            for pos_label_id in pos_labels:
                pos_meta = self.label_groups.get(pos_label_id, {})
                if (
                    pos_meta.get("group_id", -1) >= 0
                    and pos_meta.get("group_id") == neg_meta.get("group_id")
                ):
                    same_group.append(neg_label_id)
                    placed = True
                    break
            if placed:
                continue

            other.append(neg_label_id)

        return same_appearance, same_group, other

    def _rank_negative_labels_for_pos(self, pos_label_id, candidate_neg_labels):
        shuffled_neg_labels = list(candidate_neg_labels)
        random.shuffle(shuffled_neg_labels)

        scored_neg_labels = []
        for neg_label_id in shuffled_neg_labels:
            score = self._hard_negative_score(pos_label_id, neg_label_id)
            scored_neg_labels.append((neg_label_id, score))

        scored_neg_labels.sort(key=lambda item: item[1], reverse=True)
        return scored_neg_labels

    def _rank_negative_labels(self, pos_labels, candidate_neg_labels):
        shuffled_neg_labels = list(candidate_neg_labels)
        random.shuffle(shuffled_neg_labels)

        scored_neg_labels = []
        for neg_label_id in shuffled_neg_labels:
            max_score = 0.0
            for pos_label_id in pos_labels:
                max_score = max(max_score, self._hard_negative_score(pos_label_id, neg_label_id))
            scored_neg_labels.append((neg_label_id, max_score))

        scored_neg_labels.sort(key=lambda item: item[1], reverse=True)
        return scored_neg_labels

    def _select_hard_negative_labels(self, pos_labels, neg_labels, max_count):
        if max_count <= 0:
            return []

        pos_order = list(pos_labels)
        random.shuffle(pos_order)
        selected = []
        selected_set = set()

        ranked_by_pos = {}
        for pos_label_id in pos_order:
            ranked_by_pos[pos_label_id] = [
                neg_label_id
                for neg_label_id, score in self._rank_negative_labels_for_pos(pos_label_id, neg_labels)
                if score >= self.hard_negative_min_score
            ]

        cursors = {pos_label_id: 0 for pos_label_id in pos_order}
        while len(selected) < max_count:
            added_this_round = False
            for pos_label_id in pos_order:
                ranked_candidates = ranked_by_pos[pos_label_id]
                cursor = cursors[pos_label_id]
                while cursor < len(ranked_candidates) and ranked_candidates[cursor] in selected_set:
                    cursor += 1
                cursors[pos_label_id] = cursor
                if cursor >= len(ranked_candidates):
                    continue

                neg_label_id = ranked_candidates[cursor]
                selected.append(neg_label_id)
                selected_set.add(neg_label_id)
                cursors[pos_label_id] = cursor + 1
                added_this_round = True
                if len(selected) >= max_count:
                    break

            if not added_this_round:
                break

        if len(selected) >= max_count:
            return selected

        ranked_global = self._rank_negative_labels(pos_labels, neg_labels)
        for neg_label_id, score in ranked_global:
            if score < self.hard_negative_min_score:
                break
            if neg_label_id in selected_set:
                continue
            selected.append(neg_label_id)
            selected_set.add(neg_label_id)
            if len(selected) >= max_count:
                break

        return selected

    def _try_add_label(self, selected_labels, selected_label_set, used_tokens, label_id, token_budget):
        if label_id in selected_label_set:
            return used_tokens, False

        label_tokens = self.label_token_lengths.get(label_id, 0)
        if token_budget is not None and used_tokens + label_tokens > token_budget:
            return used_tokens, False

        selected_labels.append(label_id)
        selected_label_set.add(label_id)
        return used_tokens + label_tokens, True

    def _sample_caption_labels(self, pos_labels):
        if self.use_full_label_caption_training:
            return self._sample_full_label_caption_labels(pos_labels)

        pos_labels = list(pos_labels)
        random.shuffle(pos_labels)

        selected_labels = list(pos_labels)
        selected_label_set = set(selected_labels)
        if len(selected_labels) >= self.max_labels:
            random.shuffle(selected_labels)
            return selected_labels

        token_budget = self._get_label_token_budget()
        used_tokens = self.caption_special_tokens
        for label_id in selected_labels:
            used_tokens += self.supervision_label_token_lengths.get(
                label_id, self.label_token_lengths.get(label_id, 0)
            )

        negative_pool = self.caption_negative_pool if self.caption_negative_pool else self.label_index
        neg_labels = list(negative_pool.difference(set(pos_labels)))
        same_appearance_neg, same_group_neg, other_neg = self._group_negative_labels(
            pos_labels,
            neg_labels,
        )
        # Hard negatives are prioritized below, but the remaining class pool must
        # stay available. Dropping `other_neg` silently turns an "all" negative
        # pool into same-group-only supervision and creates a train/eval mismatch.
        neg_labels = same_appearance_neg + same_group_neg + other_neg

        remaining_slots = self.max_labels - len(selected_labels)
        hard_negative_quota = 0
        hard_negative_candidates = []
        if self.use_hard_negative_sampling and pos_labels and remaining_slots > 0:
            hard_negative_quota = min(
                remaining_slots,
                int(round(remaining_slots * self.hard_negative_ratio)),
            )
            hard_negative_candidates = self._select_hard_negative_labels(
                pos_labels,
                neg_labels,
                hard_negative_quota,
            )

        for label_id in hard_negative_candidates:
            if len(selected_labels) >= self.max_labels or hard_negative_quota <= 0:
                break
            used_tokens, added = self._try_add_label(
                selected_labels,
                selected_label_set,
                used_tokens,
                label_id,
                token_budget,
            )
            if added:
                hard_negative_quota -= 1

        remaining_neg_labels = [label_id for label_id in neg_labels if label_id not in selected_label_set]
        same_appearance_remaining, same_group_remaining, other_remaining = self._group_negative_labels(
            pos_labels,
            remaining_neg_labels,
        )
        random.shuffle(same_appearance_remaining)
        random.shuffle(same_group_remaining)
        random.shuffle(other_remaining)
        remaining_neg_labels = (
            same_appearance_remaining + same_group_remaining + other_remaining
        )

        for label_id in remaining_neg_labels:
            if len(selected_labels) >= self.max_labels:
                break
            used_tokens, _ = self._try_add_label(
                selected_labels,
                selected_label_set,
                used_tokens,
                label_id,
                token_budget,
            )

        if not self.prioritize_positive_caption_labels:
            random.shuffle(selected_labels)
        return selected_labels

    def _sample_full_label_caption_labels(self, pos_labels):
        pos_labels = list(pos_labels)
        random.shuffle(pos_labels)

        selected_labels = list(pos_labels)
        selected_label_set = set(selected_labels)
        if len(selected_labels) >= self.max_labels:
            random.shuffle(selected_labels)
            return selected_labels

        token_budget = self._get_label_token_budget()
        used_tokens = self.caption_special_tokens
        for label_id in selected_labels:
            used_tokens += self.label_token_lengths.get(label_id, 0)

        # DGCL benefits from a training prompt whose class competition resembles
        # the full eval vocabulary. Positives stay first so they are never lost to
        # the token budget; negatives are randomized to avoid fixed class order.
        candidate_labels = list(self.label_index.difference(selected_label_set))
        random.shuffle(candidate_labels)
        for label_id in candidate_labels:
            if len(selected_labels) >= self.max_labels:
                break
            if label_id in selected_label_set:
                continue
            label_tokens = self.supervision_label_token_lengths.get(
                label_id, self.label_token_lengths.get(label_id, 0)
            )
            if token_budget is not None and used_tokens + label_tokens > token_budget:
                continue
            selected_labels.append(label_id)
            selected_label_set.add(label_id)
            used_tokens += label_tokens

        if not self.prioritize_positive_caption_labels:
            random.shuffle(selected_labels)
        return selected_labels

    def _build_stage2_mixed_caption(self, vg_labels, pos_labels):
        caption_list = []
        expanded_label_ids = []
        label_to_primary_index = {}
        label_to_positive_indices = {}
        caption_label_loss_mask = []

        for label_id in vg_labels:
            sem_text = self.supervision_label_map[label_id]
            desc_text = self.forward_prompt_builder.build(label_id, self.raw_label_map[label_id])

            current_indices = []
            for text in (sem_text, desc_text):
                if not text:
                    continue
                text = str(text).strip()
                if text in caption_list:
                    text_index = caption_list.index(text)
                else:
                    text_index = len(caption_list)
                    caption_list.append(text)
                    expanded_label_ids.append(label_id)
                    if self.ignore_unseen_negative_loss and self.present_label_ids:
                        caption_label_loss_mask.append(
                            (label_id in pos_labels) or (label_id in self.present_label_ids)
                        )
                current_indices.append(text_index)

            if not current_indices:
                continue
            label_to_primary_index[label_id] = current_indices[0]
            label_to_positive_indices[label_id] = sorted(set(current_indices))

        caption, caption_list = build_caption_from_labels(caption_list)
        cap_group_ids = [
            self.appearance_prompt_builder.get_group_id(label_id, self.raw_label_map[label_id])
            for label_id in expanded_label_ids
        ]
        return (
            caption,
            caption_list,
            label_to_primary_index,
            label_to_positive_indices,
            cap_group_ids,
            caption_label_loss_mask or None,
        )

    def _build_component_full_text_supervision_caption(self, vg_labels, pos_labels):
        caption_list = []
        expanded_label_ids = []
        cap_group_ids = []
        label_to_primary_index = {}
        label_to_positive_indices = {}
        caption_label_loss_mask = []
        attr_component_index = {}

        def _component_keep(label_id):
            if not (self.ignore_unseen_negative_loss and self.present_label_ids):
                return None
            return (label_id in pos_labels) or (label_id in self.present_label_ids)

        def _append_component(text, label_id, group_id):
            text = str(text).strip()
            if not text:
                return None
            if text in caption_list:
                idx = caption_list.index(text)
            else:
                idx = len(caption_list)
                caption_list.append(text)
                expanded_label_ids.append(label_id)
                cap_group_ids.append(group_id)
                keep_flag = _component_keep(label_id)
                if keep_flag is not None:
                    caption_label_loss_mask.append(bool(keep_flag))
            keep_flag = _component_keep(label_id)
            if keep_flag is not None:
                caption_label_loss_mask[idx] = bool(caption_label_loss_mask[idx] or keep_flag)
            return idx

        for label_id in vg_labels:
            label_meta = self.label_groups.get(label_id, {})
            group_id = int(label_meta.get("group_id", -1))
            appearance_group = str(label_meta.get("appearance_group", "") or "").strip()

            current_indices = []

            main_text = self.component_main_label_map.get(label_id, "")
            main_idx = _append_component(main_text, label_id, group_id)
            if main_idx is not None:
                label_to_primary_index[label_id] = main_idx
                current_indices.append(main_idx)

            attr_text = str(self.component_attr_label_map.get(label_id, "") or "").strip()
            if attr_text:
                attr_key = appearance_group or f"__attr__::{attr_text}"
                if attr_key in attr_component_index:
                    attr_idx = attr_component_index[attr_key]
                    keep_flag = _component_keep(label_id)
                    if keep_flag is not None:
                        caption_label_loss_mask[attr_idx] = bool(
                            caption_label_loss_mask[attr_idx] or keep_flag
                        )
                else:
                    attr_idx = _append_component(attr_text, label_id, group_id)
                    if attr_idx is not None:
                        attr_component_index[attr_key] = attr_idx
                if attr_idx is not None:
                    current_indices.append(attr_idx)

            if not current_indices:
                continue

            if label_id not in label_to_primary_index:
                label_to_primary_index[label_id] = current_indices[0]
            label_to_positive_indices[label_id] = sorted(set(current_indices))

        caption, caption_list = build_caption_from_labels(caption_list)
        return (
            caption,
            caption_list,
            label_to_primary_index,
            label_to_positive_indices,
            cap_group_ids,
            caption_label_loss_mask or None,
        )

    def _load_metas(self, anno):
        with open(anno, 'r', encoding='utf-8-sig') as f:
            self.metas = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            global_classes = torch.tensor(
                [int(obj["label"]) for obj in instances], dtype=torch.int64
            )
            pos_labels = set(ori_classes)
            vg_labels = self._sample_caption_labels(pos_labels)
            distill_caption_list = [self.distill_label_map[lb] for lb in vg_labels]
            distill_caption, distill_caption_list = build_caption_from_labels(distill_caption_list)
            query_contrastive_caption_list = [
                self.query_contrastive_label_map[lb] for lb in vg_labels
            ]
            query_contrastive_caption, query_contrastive_caption_list = build_caption_from_labels(
                query_contrastive_caption_list
            )
            appearance_roi_caption_list = [
                self.appearance_roi_label_map[lb] for lb in vg_labels
            ]
            appearance_roi_caption, appearance_roi_caption_list = build_caption_from_labels(
                appearance_roi_caption_list
            )
            aux_token_ce_caption, aux_token_ce_caption_list = (
                query_contrastive_caption,
                query_contrastive_caption_list,
            )
            caption_label_loss_mask = None
            if (
                self.ignore_unseen_negative_loss
                and self.present_label_ids
                and not self.use_stage2_mixed_prompt_training
            ):
                caption_label_loss_mask = [
                    (label_id in pos_labels) or (label_id in self.present_label_ids)
                    for label_id in vg_labels
                ]

            label_positive_map = None
            if self.use_stage2_mixed_prompt_training:
                (
                    caption,
                    caption_list,
                    label_to_index,
                    label_to_positive_indices,
                    cap_group_ids,
                    caption_label_loss_mask,
                ) = self._build_stage2_mixed_caption(vg_labels, pos_labels)
                forward_caption, forward_caption_list = caption, caption_list
                missing_pos_labels = pos_labels.difference(label_to_index.keys())
                if missing_pos_labels:
                    raise ValueError(
                        "Stage2 mixed prompt has empty text for positive labels: "
                        f"{sorted(missing_pos_labels)}"
                    )
                classes = [label_to_index[str(obj["label"])] for obj in instances]
                label_positive_map = [
                    label_to_positive_indices[str(obj["label"])] for obj in instances
                ]
            elif self.use_component_full_text_supervision:
                (
                    caption,
                    caption_list,
                    label_to_index,
                    label_to_positive_indices,
                    cap_group_ids,
                    caption_label_loss_mask,
                ) = self._build_component_full_text_supervision_caption(vg_labels, pos_labels)
                missing_pos_labels = pos_labels.difference(label_to_index.keys())
                if missing_pos_labels:
                    raise ValueError(
                        "Component full-text supervision has empty text for positive labels: "
                        f"{sorted(missing_pos_labels)}"
                    )
                classes = [label_to_index[str(obj["label"])] for obj in instances]
                label_positive_map = [
                    label_to_positive_indices[str(obj["label"])] for obj in instances
                ]
                forward_caption, forward_caption_list = caption, caption_list
                if self.use_dual_prompt_training:
                    forward_caption_list = []
                    for lb in vg_labels:
                        if lb in pos_labels:
                            forward_caption_list.append(self.forward_label_map[lb])
                        else:
                            forward_caption_list.append(self.component_main_label_map.get(lb, self.supervision_label_map[lb]))
                    forward_caption, forward_caption_list = build_caption_from_labels(forward_caption_list)
            else:
                supervision_caption_list = [self.supervision_label_map[lb] for lb in vg_labels]
                caption, caption_list = build_caption_from_labels(supervision_caption_list)
                forward_caption, forward_caption_list = caption, caption_list
                if self.use_dual_prompt_training:
                    forward_caption_list = []
                    for lb in vg_labels:
                        if lb in pos_labels:
                            forward_caption_list.append(self.forward_label_map[lb])
                        else:
                            forward_caption_list.append(self.supervision_label_map[lb])
                    forward_caption, forward_caption_list = build_caption_from_labels(forward_caption_list)
                cap_group_ids = [
                    self.appearance_prompt_builder.get_group_id(lb, self.raw_label_map[lb])
                    for lb in vg_labels
                ]
                label_to_index = {label_id: index for index, label_id in enumerate(vg_labels)}
                classes = [label_to_index[str(obj["label"])] for obj in instances]

            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            if classes.numel() > 0:
                min_class = int(classes.min().item())
                max_class = int(classes.max().item())
                if min_class < 0 or max_class >= len(caption_list):
                    raise IndexError(
                        "ODVG local class index out of bounds for "
                        f"{rel_path}: min={min_class}, max={max_class}, "
                        f"num_caption_labels={len(caption_list)}, "
                        f"max_labels={self.max_labels}, "
                        f"pos_labels={sorted(pos_labels)}"
                    )
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list  = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption, caption_list = build_caption_from_labels(uni_caption_list)
            forward_caption, forward_caption_list = caption, caption_list
            distill_caption, distill_caption_list = caption, caption_list
            query_contrastive_caption, query_contrastive_caption_list = caption, caption_list
            aux_token_ce_caption, aux_token_ce_caption_list = caption, caption_list
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            global_classes = classes.clone()
            if classes.numel() > 0:
                min_class = int(classes.min().item())
                max_class = int(classes.max().item())
                if min_class < 0 or max_class >= len(caption_list):
                    raise IndexError(
                        "VG local class index out of bounds for "
                        f"{rel_path}: min={min_class}, max={max_class}, "
                        f"num_caption_labels={len(caption_list)}"
                    )
        target = {}
        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        target["image_path"] = abs_path
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["forward_cap_list"] = forward_caption_list if self.dataset_mode == "OD" else caption_list
        target["forward_caption"] = forward_caption if self.dataset_mode == "OD" else caption
        target["distill_cap_list"] = distill_caption_list if self.dataset_mode == "OD" else caption_list
        target["distill_caption"] = distill_caption if self.dataset_mode == "OD" else caption
        target["query_contrastive_cap_list"] = (
            query_contrastive_caption_list if self.dataset_mode == "OD" else caption_list
        )
        target["query_contrastive_caption"] = (
            query_contrastive_caption if self.dataset_mode == "OD" else caption
        )
        target["aux_token_ce_cap_list"] = (
            aux_token_ce_caption_list if self.dataset_mode == "OD" else caption_list
        )
        target["aux_token_ce_caption"] = (
            aux_token_ce_caption if self.dataset_mode == "OD" else caption
        )
        target["appearance_roi_cap_list"] = (
            appearance_roi_caption_list if self.dataset_mode == "OD" else caption_list
        )
        target["appearance_roi_caption"] = (
            appearance_roi_caption if self.dataset_mode == "OD" else caption
        )
        target["caption_label_ids"] = list(vg_labels) if self.dataset_mode == "OD" else list(caption_list)
        target["boxes"] = boxes
        target["labels"] = classes
        # Keep object-level dataset ids independent from the dynamically sampled
        # caption slots. FACB uses these ids for few-shot membership diagnostics.
        target["global_labels"] = global_classes
        if boxes.numel() > 0:
            area = boxes[:, 2] * boxes[:, 3]
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((classes.shape[0],), dtype=torch.int64)
        if self.dataset_mode == "OD":
            target["cap_group_ids"] = torch.tensor(cap_group_ids, dtype=torch.int64)
            if label_positive_map is not None:
                max_positive_count = max(len(indices) for indices in label_positive_map)
                padded_positive_map = [
                    indices + [indices[0]] * (max_positive_count - len(indices))
                    for indices in label_positive_map
                ]
                target["label_positive_map"] = torch.tensor(
                    padded_positive_map, dtype=torch.int64
                )
            if caption_label_loss_mask is not None:
                target["caption_label_loss_mask"] = torch.tensor(
                    caption_label_loss_mask, dtype=torch.bool
                )
        # size, cap_list, caption, bboxes, labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)
    hflip_prob = getattr(args, 'train_hflip_prob', 0.5)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(p=hflip_prob),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = ODVGDataset(
            img_folder,
            ann_file,
            label_map,
            max_labels=args.max_labels,
            category_meta_path=getattr(args, "category_meta_path", None),
            use_appearance_prompt=getattr(args, "use_appearance_prompt", False),
            appearance_prompt_text_source=getattr(args, "appearance_prompt_text_source", "semantic"),
            use_dual_prompt_training=getattr(args, "use_dual_prompt_training", False) if image_set == "train" else False,
            use_stage2_mixed_prompt_training=(
                getattr(args, "use_stage2_mixed_prompt_training", False) if image_set == "train" else False
            ),
            use_component_full_text_supervision=(
                getattr(args, "use_component_full_text_supervision", False) if image_set == "train" else False
            ),
            component_supervision_main_text_source=getattr(
                args, "component_supervision_main_text_source", "train_text"
            ),
            component_supervision_attr_text_source=getattr(
                args, "component_supervision_attr_text_source", "attr_text"
            ),
            query_contrastive_text_source=getattr(
                args, "query_contrastive_text_source", "train_text"
            ),
            appearance_roi_text_source=getattr(
                args, "appearance_roi_text_source", "visual_cue_text"
            ),
            appearance_roi_num_neg=getattr(args, "appearance_roi_num_neg", 8),
            forward_prompt_text_source=getattr(args, "forward_prompt_text_source", "description"),
            supervision_prompt_text_source=getattr(args, "supervision_prompt_text_source", "semantic"),
            appearance_prompt_separator=getattr(args, "appearance_prompt_separator", ", "),
            appearance_prompt_meta_path=getattr(args, "appearance_prompt_meta_path", None),
            text_encoder_type=getattr(args, "text_encoder_type", None),
            max_text_len=getattr(args, "max_text_len", 256),
            max_label_tokens=getattr(args, "max_label_tokens", None),
            use_hard_negative_sampling=getattr(args, "use_hard_negative_sampling", True),
            hard_negative_ratio=getattr(args, "hard_negative_ratio", 0.6),
            hard_negative_min_score=getattr(args, "hard_negative_min_score", 1.0),
            ignore_unseen_negative_loss=(
                getattr(args, "ignore_unseen_negative_loss", False) if image_set == "train" else False
            ),
            negative_label_pool=(
                getattr(args, "negative_label_pool", "all") if image_set == "train" else "all"
            ),
            use_full_label_caption_training=(
                getattr(args, "use_full_label_caption_training", False) if image_set == "train" else False
            ),
            prioritize_positive_caption_labels=(
                getattr(args, "prioritize_positive_caption_labels", False)
                if image_set == "train"
                else False
            ),
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
    )
    return dataset


if __name__=="__main__":
    dataset_vg = ODVGDataset("path/GRIT-20M/data/","path/GRIT-20M/anno/grit_odvg_10k.jsonl",)
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)] 
    print(data)
    dataset_od = ODVGDataset("pathl/V3Det/",
        "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
        "path/V3Det/annotations/v3det_label_map.json",
    )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)] 
    print(data)
