# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import json
import math
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision.ops import roi_align
from torchvision.ops.boxes import batched_nms, nms

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span
from util.caption_builder import build_caption_from_labels
from util.facb_proposal_gate import ProposalGate, pack_features, raw_predicted_crops, stage_score, SCHEMA
from util.facb_score_fusion import fused_score, load_fusion, capture_chunk, sha256
from util.facb_candidate_bank import fixed_bank_candidates
from util.facb_local_selector import LocalSelector, load_selector, selector_probabilities
from util.facb_pool_v2 import union_candidates, unique_candidate_mask
from util.facb_dual_facet_itm import DualFacetITM, dual_facet_logits
from util.facb_d1_online import score_pair as facb_d1_score_pair, validate_score as validate_facb_d1_score

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    ModernBertWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .aggr import AppearanceGuidedGroupRegularizer, build_category_text_prototypes
from .dgcl_loss import (
    matched_category_contrastive_loss,
    matched_category_logits_contrastive_loss,
)
from .transformer import build_transformer
from .utils import (
    MLP,
    ContrastiveEmbed,
    SAILAlignmentHead,
    SAILLastClassEmbed,
    sigmoid_focal_loss,
    CandidateConditionedLocalEvidence,
    CLIPCropEvidence,
)
from .self_facb import SelfFACBHead

from .matcher import build_matcher




class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        use_aggr=False,
        freeze_text_encoder=False,
        text_encoder_unfreeze_last_n_layers=0,
        text_encoder_unfreeze_embeddings=False,
        use_text_adapter_ffn=False,
        text_adapter_hidden_dim=256,
        text_adapter_dropout=0.0,
        use_similarity_calibration=False,
        similarity_calibration_type="star",
        similarity_calibration_hidden_dim=None,
        similarity_calibration_dropout=0.0,
        similarity_calibration_residual=True,
        similarity_calibration_mode="additive",
        class_logit_norm=False,
        class_logit_scale=10.0,
        class_logit_bias=0.0,
        use_modality_projection=False,
        modality_projection_type="star",
        modality_projection_hidden_dim=None,
        modality_projection_dropout=0.0,
        modality_projection_width_factor=2,
        modality_projection_residual=True,
        dual_prompt_class_logit_norm=False,
        dual_prompt_class_logit_scale=10.0,
        dual_prompt_class_logit_bias=0.0,
        use_quality_head=False,
        use_lgpr_survival=False,
        lgpr_survival_hidden_dim=None,
        use_lgpr_box_refine=False,
        lgpr_box_refine_hidden_dim=None,
        lgpr_box_refine_scale=0.05,
        use_lgpr_box_refine_inference=True,
        use_query_contrastive=False,
        use_global_image_text_sigmoid_loss=False,
        use_sail_alignment_head=False,
        sail_alignment_target_dim=None,
        sail_alignment_width_factor=8,
        sail_alignment_logit_scale=20.0,
        sail_alignment_logit_bias=-10.0,
        use_sail_last_class_embed=False,
        sail_last_class_embed_width_factor=8,
        sail_last_class_embed_alpha_init=0.2,
        sail_last_class_embed_use_similarity_gate=True,
        sail_last_class_embed_similarity_gate_bottleneck_dim=64,
        sail_last_class_embed_similarity_gate_initial_bias=0.0,
        sail_last_class_embed_use_text_gate=False,
        sail_last_class_embed_text_gate_hidden_dim=None,
        sail_last_class_embed_text_gate_strength_init=0.05,
        sail_last_class_embed_gate_hidden_dim=128,
        sail_last_class_embed_adapter_scale_init=0.1,
        sail_last_class_embed_temperature_init=0.2,
        sail_last_class_embed_use_frequency_prior=True,
        sail_last_class_embed_non_category_logit=-10.0,
        sail_last_class_embed_use_baseline_residual=True,
        sail_last_class_embed_use_main_token_blend=True,
        sail_last_class_embed_output_blend_init=0.1,
        sail_last_class_embed_match_baseline_distribution=True,
        sail_last_class_embed_use_margin_gate=False,
        sail_last_class_embed_margin_gate_tau=1.0,
        sail_last_class_embed_margin_gate_gamma=0.2,
        sail_last_class_embed_use_word_cocontribution=False,
        sail_last_class_embed_word_cocontribution_similarity_scale_init=0.0,
        sail_last_class_embed_word_cocontribution_completeness_init=0.95,
        sail_last_class_embed_word_cocontribution_discriminative_init=0.0,
        use_aux_token_ce=False,
        use_local_roi_query_fusion=False,
        local_roi_query_fusion_output_size=7,
        use_appearance_roi_fusion=False,
        use_feature_level_fusion=False,
        appearance_roi_output_size=7,
        appearance_roi_temperature=0.07,
        appearance_roi_alpha=0.5,
        appearance_roi_semantic_topk=0,
        use_learnable_appearance_proto=False,
        use_appearance_query_bias=False,
        appearance_num_classes=0,
        use_proto_cls_head=False,
        proto_cls_temperature=0.07,
        use_appearance_rescoring=False,
        appearance_rescoring_temperature=1.0,
        appearance_rescoring_bias=0.0,
        appearance_rescoring_mode="max",
        use_appearance_text_alignment=False,
        appearance_text_loss_coef=0.5,
        use_appearance_text_rerank=False,
        appearance_text_rerank_weight=0.25,
        use_visual_confidence=False,
        visual_confidence_roi_output_size=7,
        use_appearance_conditioned_gate=False,
        appearance_gate_temperature=0.07,
        appearance_gate_feature_source="backbone",
        appearance_logit_bias_weight=0.0,
        appearance_logit_bias_detach=False,
        appearance_logit_bias_norm="tanh",
        use_appearance_token_branch=False,
        appearance_token_branch_weight=0.2,
        appearance_token_branch_temperature=0.07,
        appearance_token_branch_scale_init=-2.0,
        use_roi_appearance_branch=False,
        roi_appearance_output_size=7,
        roi_appearance_temperature=0.07,
        roi_appearance_scale_init=-2.0,
        use_appearance_contrastive_head=False,
        appearance_contrastive_head_class_logit_norm=None,
        appearance_contrastive_head_class_logit_scale=None,
        appearance_contrastive_head_class_logit_bias=None,
        appearance_contrastive_head_use_similarity_calibration=None,
        appearance_contrastive_head_similarity_calibration_type=None,
        appearance_contrastive_head_similarity_calibration_hidden_dim=None,
        appearance_contrastive_head_similarity_calibration_dropout=None,
        appearance_contrastive_head_similarity_calibration_residual=None,
        appearance_contrastive_head_similarity_calibration_mode=None,
        appearance_contrastive_head_use_modality_projection=None,
        appearance_contrastive_head_modality_projection_type=None,
        appearance_contrastive_head_modality_projection_hidden_dim=None,
        appearance_contrastive_head_modality_projection_dropout=None,
        appearance_contrastive_head_modality_projection_width_factor=None,
        appearance_contrastive_head_modality_projection_residual=None,
        use_appearance_prototype_bank=False,
        appearance_prototype_texts=None,
        appearance_shared_prototype_texts=None,
        use_appearance_discriminative_residual=False,
        appearance_discriminative_mode="residual",
        appearance_contrastive_head_feature_source="decoder_query",
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present
        self.use_aggr = use_aggr
        self._prompt_debug_printed = False
        self._appearance_debug_printed = False
        self.freeze_text_encoder = freeze_text_encoder
        self.text_encoder_unfreeze_last_n_layers = max(
            0, int(text_encoder_unfreeze_last_n_layers or 0)
        )
        self.text_encoder_unfreeze_embeddings = bool(text_encoder_unfreeze_embeddings)
        self.text_encoder_has_trainable_layers = False
        self.class_logit_norm = bool(class_logit_norm)
        self.class_logit_scale = float(class_logit_scale)
        self.class_logit_bias = float(class_logit_bias)
        self.dual_prompt_class_logit_norm = bool(dual_prompt_class_logit_norm)
        self.dual_prompt_class_logit_scale = float(dual_prompt_class_logit_scale)
        self.dual_prompt_class_logit_bias = float(dual_prompt_class_logit_bias)
        self.appearance_contrastive_head_class_logit_norm = (
            class_logit_norm
            if appearance_contrastive_head_class_logit_norm is None
            else bool(appearance_contrastive_head_class_logit_norm)
        )
        self.appearance_contrastive_head_class_logit_scale = (
            class_logit_scale
            if appearance_contrastive_head_class_logit_scale is None
            else float(appearance_contrastive_head_class_logit_scale)
        )
        self.appearance_contrastive_head_class_logit_bias = (
            class_logit_bias
            if appearance_contrastive_head_class_logit_bias is None
            else float(appearance_contrastive_head_class_logit_bias)
        )
        self.appearance_contrastive_head_use_similarity_calibration = (
            use_similarity_calibration
            if appearance_contrastive_head_use_similarity_calibration is None
            else bool(appearance_contrastive_head_use_similarity_calibration)
        )
        self.appearance_contrastive_head_similarity_calibration_type = (
            similarity_calibration_type
            if appearance_contrastive_head_similarity_calibration_type is None
            else appearance_contrastive_head_similarity_calibration_type
        )
        self.appearance_contrastive_head_similarity_calibration_hidden_dim = (
            similarity_calibration_hidden_dim
            if appearance_contrastive_head_similarity_calibration_hidden_dim is None
            else appearance_contrastive_head_similarity_calibration_hidden_dim
        )
        self.appearance_contrastive_head_similarity_calibration_dropout = (
            similarity_calibration_dropout
            if appearance_contrastive_head_similarity_calibration_dropout is None
            else float(appearance_contrastive_head_similarity_calibration_dropout)
        )
        self.appearance_contrastive_head_similarity_calibration_residual = (
            similarity_calibration_residual
            if appearance_contrastive_head_similarity_calibration_residual is None
            else bool(appearance_contrastive_head_similarity_calibration_residual)
        )
        self.appearance_contrastive_head_similarity_calibration_mode = (
            similarity_calibration_mode
            if appearance_contrastive_head_similarity_calibration_mode is None
            else appearance_contrastive_head_similarity_calibration_mode
        )
        self.appearance_contrastive_head_use_modality_projection = (
            use_modality_projection
            if appearance_contrastive_head_use_modality_projection is None
            else bool(appearance_contrastive_head_use_modality_projection)
        )
        self.appearance_contrastive_head_modality_projection_type = (
            modality_projection_type
            if appearance_contrastive_head_modality_projection_type is None
            else appearance_contrastive_head_modality_projection_type
        )
        self.appearance_contrastive_head_modality_projection_hidden_dim = (
            modality_projection_hidden_dim
            if appearance_contrastive_head_modality_projection_hidden_dim is None
            else appearance_contrastive_head_modality_projection_hidden_dim
        )
        self.appearance_contrastive_head_modality_projection_dropout = (
            modality_projection_dropout
            if appearance_contrastive_head_modality_projection_dropout is None
            else float(appearance_contrastive_head_modality_projection_dropout)
        )
        self.appearance_contrastive_head_modality_projection_width_factor = (
            modality_projection_width_factor
            if appearance_contrastive_head_modality_projection_width_factor is None
            else max(int(appearance_contrastive_head_modality_projection_width_factor), 1)
        )
        self.appearance_contrastive_head_modality_projection_residual = (
            modality_projection_residual
            if appearance_contrastive_head_modality_projection_residual is None
            else bool(appearance_contrastive_head_modality_projection_residual)
        )
        self.use_query_contrastive = bool(use_query_contrastive)
        self.use_global_image_text_sigmoid_loss = bool(use_global_image_text_sigmoid_loss)
        self.use_sail_alignment_head = bool(use_sail_alignment_head)
        self.use_sail_last_class_embed = bool(use_sail_last_class_embed)
        self.sail_last_class_embed_use_main_token_blend = bool(
            sail_last_class_embed_use_main_token_blend
        )
        self.use_aux_token_ce = bool(use_aux_token_ce)
        self.use_local_roi_query_fusion = bool(use_local_roi_query_fusion)
        self.local_roi_query_fusion_output_size = max(
            1, int(local_roi_query_fusion_output_size)
        )
        self.use_appearance_roi_fusion = bool(use_appearance_roi_fusion)
        self.use_feature_level_fusion = bool(use_feature_level_fusion)
        self.appearance_roi_output_size = max(1, int(appearance_roi_output_size))
        self.appearance_roi_temperature = max(float(appearance_roi_temperature), 1e-6)
        self.appearance_roi_alpha = float(appearance_roi_alpha)
        self.appearance_roi_semantic_topk = max(int(appearance_roi_semantic_topk), 0)
        self.use_learnable_appearance_proto = bool(use_learnable_appearance_proto)
        self.use_appearance_query_bias = bool(use_appearance_query_bias)
        self.appearance_num_classes = max(int(appearance_num_classes), 0)
        self.use_proto_cls_head = bool(use_proto_cls_head)
        self.proto_cls_temperature = max(float(proto_cls_temperature), 1e-6)
        self.use_appearance_rescoring = bool(use_appearance_rescoring)
        self.appearance_rescoring_temperature = float(appearance_rescoring_temperature)
        self.appearance_rescoring_bias = float(appearance_rescoring_bias)
        self.appearance_rescoring_mode = str(appearance_rescoring_mode).strip().lower()
        self.use_appearance_text_alignment = bool(use_appearance_text_alignment)
        self.appearance_text_loss_coef = float(appearance_text_loss_coef)
        self.use_appearance_text_rerank = bool(use_appearance_text_rerank)
        self.appearance_text_rerank_weight = float(appearance_text_rerank_weight)
        self.query_score_pos_iou_thr = 0.5
        self.query_score_neg_iou_thr = 0.1
        self.query_score_num_pos = 16
        self.query_score_num_neg = 32
        self.query_score_rank_margin = 0.2
        if self.use_sail_alignment_head:
            self.sail_alignment_head = SAILAlignmentHead(
                embed_dim=hidden_dim,
                target_dim=sail_alignment_target_dim,
                width_factor=sail_alignment_width_factor,
                logit_scale=sail_alignment_logit_scale,
                logit_bias=sail_alignment_logit_bias,
            )
        else:
            self.sail_alignment_head = None
        if self.use_sail_last_class_embed:
            self.sail_last_class_embed = SAILLastClassEmbed(
                embed_dim=hidden_dim,
                max_text_len=max_text_len,
                width_factor=sail_last_class_embed_width_factor,
                gate_hidden_dim=sail_last_class_embed_gate_hidden_dim,
                adapter_scale_init=sail_last_class_embed_adapter_scale_init,
                temperature_init=sail_last_class_embed_temperature_init,
                use_frequency_prior=sail_last_class_embed_use_frequency_prior,
                non_category_logit=sail_last_class_embed_non_category_logit,
                use_baseline_residual=sail_last_class_embed_use_baseline_residual,
                output_blend_init=sail_last_class_embed_output_blend_init,
                match_baseline_distribution=(
                    sail_last_class_embed_match_baseline_distribution
                ),
                use_margin_gate=sail_last_class_embed_use_margin_gate,
                margin_gate_tau=sail_last_class_embed_margin_gate_tau,
                margin_gate_gamma=sail_last_class_embed_margin_gate_gamma,
                use_word_cocontribution=sail_last_class_embed_use_word_cocontribution,
                word_cocontribution_similarity_scale_init=(
                    sail_last_class_embed_word_cocontribution_similarity_scale_init
                ),
                word_cocontribution_completeness_init=(
                    sail_last_class_embed_word_cocontribution_completeness_init
                ),
                word_cocontribution_discriminative_init=(
                    sail_last_class_embed_word_cocontribution_discriminative_init
                ),
            )
        else:
            self.sail_last_class_embed = None

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # text encoder (bert / modernbert / roberta)
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        has_pooler = hasattr(self.bert, "pooler") and self.bert.pooler is not None
        if has_pooler:
            self.bert.pooler.dense.weight.requires_grad_(False)
            self.bert.pooler.dense.bias.requires_grad_(False)
            self.bert = BertModelWarper(bert_model=self.bert)
        else:
            # ModernBERT / RoBERTa (no pooler)
            self.bert = ModernBertWarper(modernbert_model=self.bert)
        if self.freeze_text_encoder:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)
        self._unfreeze_text_encoder_layers_if_requested()

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        if use_text_adapter_ffn:
            self.text_adapter_ffn = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, text_adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(text_adapter_dropout),
                nn.Linear(text_adapter_hidden_dim, self.hidden_dim),
                nn.Dropout(text_adapter_dropout),
            )
            nn.init.zeros_(self.text_adapter_ffn[4].weight)
            nn.init.zeros_(self.text_adapter_ffn[4].bias)
        else:
            self.text_adapter_ffn = None

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed(
            embed_dim=hidden_dim,
            max_text_len=max_text_len,
            use_similarity_calibration=use_similarity_calibration,
            similarity_calibration_type=similarity_calibration_type,
            similarity_calibration_hidden_dim=similarity_calibration_hidden_dim,
            similarity_calibration_dropout=similarity_calibration_dropout,
            similarity_calibration_residual=similarity_calibration_residual,
            similarity_calibration_mode=similarity_calibration_mode,
            class_logit_norm=self.class_logit_norm,
            class_logit_scale=self.class_logit_scale,
            class_logit_bias=self.class_logit_bias,
            use_modality_projection=use_modality_projection,
            modality_projection_type=modality_projection_type,
            modality_projection_hidden_dim=modality_projection_hidden_dim,
            modality_projection_dropout=modality_projection_dropout,
            modality_projection_width_factor=modality_projection_width_factor,
            modality_projection_residual=modality_projection_residual,
        )

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        if self.dual_prompt_class_logit_norm:
            dual_prompt_class_embed = copy.deepcopy(_class_embed)
            dual_prompt_class_embed.class_logit_norm = True
            dual_prompt_class_embed.class_logit_scale = self.dual_prompt_class_logit_scale
            dual_prompt_class_embed.class_logit_bias = self.dual_prompt_class_logit_bias
            self.dual_prompt_class_embed = nn.ModuleList(
                [dual_prompt_class_embed for _ in range(transformer.num_decoder_layers)]
            )
        else:
            self.dual_prompt_class_embed = None
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # Appearance contrastive head: mirrors self.class_embed but reads
        # fine-grained appearance descriptions instead of semantic labels.
        # Same ContrastiveEmbed architecture, independent weights.
        # Only the last layer is used at inference (same as semantic head).
        if use_appearance_contrastive_head:
            _appearance_contrastive_head = ContrastiveEmbed(
                embed_dim=hidden_dim,
                max_text_len=max_text_len,
                use_similarity_calibration=self.appearance_contrastive_head_use_similarity_calibration,
                similarity_calibration_type=self.appearance_contrastive_head_similarity_calibration_type,
                similarity_calibration_hidden_dim=self.appearance_contrastive_head_similarity_calibration_hidden_dim,
                similarity_calibration_dropout=self.appearance_contrastive_head_similarity_calibration_dropout,
                similarity_calibration_residual=self.appearance_contrastive_head_similarity_calibration_residual,
                similarity_calibration_mode=self.appearance_contrastive_head_similarity_calibration_mode,
                class_logit_norm=self.appearance_contrastive_head_class_logit_norm,
                class_logit_scale=self.appearance_contrastive_head_class_logit_scale,
                class_logit_bias=self.appearance_contrastive_head_class_logit_bias,
                use_modality_projection=self.appearance_contrastive_head_use_modality_projection,
                modality_projection_type=self.appearance_contrastive_head_modality_projection_type,
                modality_projection_hidden_dim=self.appearance_contrastive_head_modality_projection_hidden_dim,
                modality_projection_dropout=self.appearance_contrastive_head_modality_projection_dropout,
                modality_projection_width_factor=self.appearance_contrastive_head_modality_projection_width_factor,
                modality_projection_residual=self.appearance_contrastive_head_modality_projection_residual,
            )
            self.appearance_contrastive_head = nn.ModuleList(
                [copy.deepcopy(_appearance_contrastive_head) for _ in range(transformer.num_decoder_layers)]
            )
            # Lightweight fusion for decoder_query + ROI concatenation.
            # Projects 2×hidden_dim → hidden_dim before the ContrastiveEmbed.
            # Zero-init bias → starts near identity on the decoder_query half.
            self.appearance_concat_fusion = nn.Linear(
                hidden_dim * 2, hidden_dim
            )
            nn.init.xavier_uniform_(self.appearance_concat_fusion.weight)
            nn.init.constant_(self.appearance_concat_fusion.bias, 0.0)
            # ---- roi_cross_attn feature source (direction A) ----
            # ROI-pooled backbone features attend over prototype bank entries so
            # the visual query is shaped by appearance descriptions, not semantic
            # classification.  One MHA layer + LN + residual, then a projection
            # to hidden_dim.  Kept in a ModuleDict so it only trains when used.
            self.appearance_roi_ca_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=max(1, nheads),
                dropout=0.0,
                batch_first=True,
            )
            self.appearance_roi_ca_norm = nn.LayerNorm(hidden_dim)
            self.appearance_roi_ca_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.xavier_uniform_(self.appearance_roi_ca_proj[0].weight)
            nn.init.constant_(self.appearance_roi_ca_proj[0].bias, 0.0)
            nn.init.xavier_uniform_(self.appearance_roi_ca_proj[3].weight)
            nn.init.constant_(self.appearance_roi_ca_proj[3].bias, 0.0)
            # ---- Spatial attention pooling for ROI features (14×14 → D) ----
            # Replaces mean-pool: a single Linear(D,1)+softmax over spatial tokens
            # lets the branch focus on the most discriminative region (e.g. the
            # digit area in pl40/pl60) instead of averaging everything flat.
            self.appearance_roi_spatial_attn = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.appearance_roi_spatial_attn.weight)
            nn.init.zeros_(self.appearance_roi_spatial_attn.bias)
            # ---- Candidate-conditioned local evidence head ----
            # Replaces the failed single-vector matching paradigm: keep the ROI
            # spatial tokens and let each candidate prototype read the region it
            # cares about.  Shared Wq/Wk/Wv only (no per-class params), so it stays
            # a conditional evidence branch and transfers across appearance styles.
            self.appearance_local_evidence_head = CandidateConditionedLocalEvidence(
                embed_dim=hidden_dim,
                logit_scale=float(
                    getattr(self, "appearance_contrastive_head_class_logit_scale", 10.0)
                    or 10.0
                ),
            )
        else:
            self.appearance_contrastive_head = None
            self.appearance_concat_fusion = None
            self.appearance_roi_ca_attn = None
            self.appearance_roi_ca_norm = None
            self.appearance_roi_ca_proj = None
            self.appearance_roi_spatial_attn = None
            self.appearance_local_evidence_head = None

        # ---- Prototype-bank mode for the appearance head ----
        # Instead of contrasting against per-image padded text tokens, score
        # queries against a fixed [num_classes, hidden_dim] bank of class
        # appearance prototypes (frozen-BERT text features pooled per class).
        # Set by main() / build via appearance_prototype_texts; the bank tensor
        # itself is built lazily on first forward (needs BERT on the right device)
        # and cached as a non-persistent buffer.
        self.use_appearance_prototype_bank = bool(use_appearance_prototype_bank)
        self.appearance_prototype_texts = (
            list(appearance_prototype_texts)
            if appearance_prototype_texts is not None
            else None
        )
        self.appearance_contrastive_head_feature_source = str(
            appearance_contrastive_head_feature_source or "decoder_query"
        )
        # ---- Candidate-conditioned local evidence config (overridden in build) ----
        # Defaults are safe (path off); build() sets these from args.
        self.use_appearance_local_evidence = False
        self.appearance_local_evidence_topm = 8
        self.appearance_local_evidence_levels = [0]
        self.appearance_local_evidence_roi_size = 7
        # Detector-native post-NMS FACB.  This is intentionally independent from
        # the older appearance/CLIP paths: it reads backbone ROI tokens and BERT
        # text prototypes, then reranks only semantic top-M candidates.
        self.use_self_facb = False
        self.return_self_facb_features = False
        self.self_facb_head = SelfFACBHead(
            embed_dim=hidden_dim,
            logit_scale=float(
                getattr(self, "appearance_contrastive_head_class_logit_scale", 10.0)
                or 10.0
            ),
            text_proj_hidden_dim=512,
        )
        self.facb_symbol_align_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.xavier_uniform_(self.facb_symbol_align_proj[1].weight)
        nn.init.zeros_(self.facb_symbol_align_proj[1].bias)
        nn.init.xavier_uniform_(self.facb_symbol_align_proj[3].weight)
        nn.init.zeros_(self.facb_symbol_align_proj[3].bias)
        self.use_facb_symbol_align = False
        self.facb_symbol_align_texts = None
        self.facb_symbol_align_fallback_texts = None
        self.facb_symbol_align_semantic_texts = None
        self.facb_symbol_align_extract_from_semantic = False
        self.facb_symbol_align_facet_index = 1
        self.facb_symbol_align_online_chunk = 64
        # Method B: same-group token-margin (sharpens the detector's OWN alignment).
        self.use_semantic_group_margin = False
        # Semantic prototype texts (== args.semantic_label_list), aligned to global
        # class id order.  Used ONLY to rank candidates for the local-evidence path,
        # so cand_ids come from true semantic full-space — NOT from the semantic head
        # scored against the appearance bank (which is a different space and was the
        # candidate-source bug).  Set in build().
        self.appearance_semantic_prototype_texts = None
        self.register_buffer("appearance_semantic_prototype_bank", None, persistent=False)
        self.register_buffer("appearance_semantic_prototype_bank_mask", None, persistent=False)
        self.facb_speedhr_candidate_multi_facet = False
        self.facb_speedhr_multi_facet_texts = None
        self.facb_speedhr_facet_names = None
        self.register_buffer(
            "facb_speedhr_facet_prototype_bank", None, persistent=False
        )
        self.register_buffer(
            "facb_speedhr_facet_prototype_bank_mask", None, persistent=False
        )
        self.facb_speedhr_shared_compositional_facets = False
        self.facb_speedhr_shared_facet_names = None
        self.facb_speedhr_shared_facet_vocab_texts = None
        self.register_buffer(
            "facb_speedhr_shared_facet_prototype_bank", None, persistent=False
        )
        self.register_buffer(
            "facb_speedhr_shared_facet_prototype_mask", None, persistent=False
        )
        self.register_buffer(
            "facb_speedhr_shared_facet_class_targets", None, persistent=False
        )
        self.register_buffer(
            "facb_speedhr_shared_facet_class_target_mask", None, persistent=False
        )
        # ---- CLIP crop evidence config (overridden in build) ----
        # Original-pixel crop -> frozen CLIP vision, scored by cosine against frozen
        # CLIP text embeddings of candidate appearance descriptions.  Replaces the
        # detector-ROI-token local evidence (which failed: feature-map lacks the
        # discriminative pixels).  The head + CLIP-text bank are built in build()
        # once the weights path is known.
        self.use_appearance_clip_evidence = False
        self.appearance_clip_model_path = None
        self.appearance_clip_text_model_path = None
        self.appearance_clip_crop_size = 224
        self.appearance_clip_topm = 8
        self.appearance_clip_eval_select_mode = "detections"
        self.appearance_clip_min_box_area = 0.0
        self.appearance_clip_min_det_score = 0.0
        self.appearance_clip_evidence_head = None
        self.appearance_clip_text_prototype_texts = None   # candidate appearance texts (global id order)
        self.register_buffer("appearance_clip_text_bank", None, persistent=False)
        # Lazily populated [num_classes, hidden_dim] prototype bank + valid mask.
        self.register_buffer("appearance_prototype_bank", None, persistent=False)
        self.register_buffer("appearance_prototype_bank_mask", None, persistent=False)

        # ---- Dual prototype: shared (structural) bank ----
        # A second bank built from attr_text (shared structure).  In v1 it is
        # built but NOT used for scoring — it is reserved for later shared/discr
        # fusion and for diagnostics.  Pairwise loss acts only on the
        # discriminative bank (appearance_prototype_bank + residual below).
        self.appearance_shared_prototype_texts = (
            list(appearance_shared_prototype_texts)
            if appearance_shared_prototype_texts is not None
            else None
        )
        self.register_buffer("appearance_shared_prototype_bank", None, persistent=False)
        self.register_buffer("appearance_shared_prototype_bank_mask", None, persistent=False)

        # ---- Per-class learnable discriminative residual ----
        # The frozen prototype bank is a no_grad buffer, so it carries NO per-class
        # capacity to separate near-collinear same-group prototypes (pl40 vs pl60).
        # This residual is the ONLY trainable per-class term: p_discr = normalize(
        # base + delta).  Zero-initialized so training starts exactly at the frozen
        # text anchor and the residual only grows where the pairwise loss needs it.
        # Kept small (zero init + weight decay via the default optimizer group) to
        # avoid degenerating into a standalone classifier prototype.
        self.use_appearance_discriminative_residual = bool(
            use_appearance_discriminative_residual
        )
        self.appearance_discriminative_mode = str(
            appearance_discriminative_mode or "residual"
        ).strip().lower()
        if self.appearance_discriminative_mode not in ("residual", "centroid"):
            self.appearance_discriminative_mode = "residual"
        if (
            self.use_appearance_discriminative_residual
            and self.appearance_prototype_texts
            and self.appearance_discriminative_mode == "centroid"
        ):
            self.appearance_visual_centroids = nn.Parameter(
                torch.empty(len(self.appearance_prototype_texts), hidden_dim)
            )
            nn.init.normal_(self.appearance_visual_centroids, mean=0.0, std=0.02)
            self.appearance_discriminative_residual = None
        elif self.use_appearance_discriminative_residual and self.appearance_prototype_texts:
            self.appearance_discriminative_residual = nn.Parameter(
                torch.zeros(len(self.appearance_prototype_texts), hidden_dim)
            )
            self.appearance_visual_centroids = None
        else:
            self.appearance_discriminative_residual = None
            self.appearance_visual_centroids = None
        # Populated each forward when the residual is active (monitoring only).
        self.appearance_residual_stats = {}

        self.use_quality_head = use_quality_head
        if use_quality_head:
            _quality_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            nn.init.constant_(_quality_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_quality_embed.layers[-1].bias.data, 0)
            self.quality_embed = nn.ModuleList(
                [_quality_embed for _ in range(transformer.num_decoder_layers)]
            )
        else:
            self.quality_embed = None

        # Recall-oriented decoder-query survival head.  Decoder queries are
        # already language-conditioned by GroundingDINO's text/image decoder;
        # this head predicts a single class-agnostic retention utility per query.
        # A zero-initialized final layer makes a freshly attached head neutral.
        self.use_lgpr_survival = bool(use_lgpr_survival)
        if self.use_lgpr_survival:
            survival_hidden_dim = int(lgpr_survival_hidden_dim or hidden_dim)
            self.lgpr_survival_head = MLP(
                hidden_dim, survival_hidden_dim, 1, 3
            )
            nn.init.constant_(self.lgpr_survival_head.layers[-1].weight.data, 0)
            nn.init.constant_(self.lgpr_survival_head.layers[-1].bias.data, 0)
        else:
            self.lgpr_survival_head = None

        # Zero-init residual box refinement.  The base decoder/bbox heads stay
        # untouched; this head predicts a small inverse-sigmoid box correction.
        self.use_lgpr_box_refine = bool(use_lgpr_box_refine)
        self.use_lgpr_box_refine_inference = bool(use_lgpr_box_refine_inference)
        self.lgpr_box_refine_scale = float(lgpr_box_refine_scale)
        if self.use_lgpr_box_refine:
            refine_hidden_dim = int(lgpr_box_refine_hidden_dim or hidden_dim)
            self.lgpr_box_refine_head = MLP(hidden_dim, refine_hidden_dim, 4, 3)
            nn.init.constant_(self.lgpr_box_refine_head.layers[-1].weight.data, 0)
            nn.init.constant_(self.lgpr_box_refine_head.layers[-1].bias.data, 0)
        else:
            self.lgpr_box_refine_head = None

        if use_query_contrastive or use_aux_token_ce:
            self.aux_query_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.aux_query_proj = None
        if self.use_local_roi_query_fusion:
            self.local_roi_query_value_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.local_roi_query_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.local_roi_query_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.local_roi_query_value_proj = None
            self.local_roi_query_fc = None
            self.local_roi_query_gate = None

        if self.use_appearance_roi_fusion:
            self.appearance_roi_token_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.appearance_roi_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.appearance_roi_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.appearance_roi_query_norm = nn.LayerNorm(hidden_dim)
            self.appearance_roi_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=max(1, nheads),
                dropout=0.0,
                batch_first=True,
            )
            if self.use_appearance_query_bias:
                self.appearance_query_bias = nn.Embedding(num_queries, hidden_dim)
            else:
                self.appearance_query_bias = None
            self.appearance_roi_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            if self.use_learnable_appearance_proto and self.appearance_num_classes > 0:
                self.appearance_class_embed = nn.Embedding(self.appearance_num_classes, hidden_dim)
            else:
                self.appearance_class_embed = None
            gate_linear_layers = [m for m in self.appearance_roi_gate if isinstance(m, nn.Linear)]
            if gate_linear_layers:
                nn.init.xavier_uniform_(gate_linear_layers[0].weight)
                if gate_linear_layers[0].bias is not None:
                    nn.init.constant_(gate_linear_layers[0].bias, 0)
                if gate_linear_layers[-1].bias is not None:
                    nn.init.constant_(gate_linear_layers[-1].bias, -2.0)
        else:
            self.appearance_roi_token_proj = None
            self.appearance_roi_proj = None
            self.appearance_roi_query_proj = None
            self.appearance_roi_query_norm = None
            self.appearance_roi_cross_attn = None
            self.appearance_query_bias = None
            self.appearance_roi_gate = None
            self.appearance_class_embed = None

        if self.use_proto_cls_head:
            self.proto_cls_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proto_cls_proto_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.proto_cls_query_proj = None
            self.proto_cls_proto_gate = None

        if self.use_appearance_text_alignment:
            self.appearance_text_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.appearance_text_query_proj = None

        # class-agnostic visual confidence head (backbone ROI features → scalar quality).
        # ROI features are mean-pooled spatially, so input dim is just hidden_dim (256).
        self.use_visual_confidence = use_visual_confidence
        self.visual_confidence_roi_output_size = int(visual_confidence_roi_output_size)
        if use_visual_confidence:
            self.visual_confidence_roi_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.constant_(self.visual_confidence_roi_proj[-1].weight.data, 0)
            nn.init.constant_(self.visual_confidence_roi_proj[-1].bias.data, 0)
        else:
            self.visual_confidence_roi_proj = None

        # v2: class-wise appearance-conditioned gate (backbone ROI or decoder query × appearance desc)
        self.use_appearance_conditioned_gate = use_appearance_conditioned_gate
        self.appearance_gate_feature_source = appearance_gate_feature_source
        self.appearance_logit_bias_weight = float(appearance_logit_bias_weight)
        self.appearance_logit_bias_detach = bool(appearance_logit_bias_detach)
        self.appearance_logit_bias_norm = str(appearance_logit_bias_norm).strip().lower()
        self.use_appearance_token_branch = bool(use_appearance_token_branch)
        self.appearance_token_branch_weight = float(appearance_token_branch_weight)
        self.appearance_token_branch_scale_init = float(appearance_token_branch_scale_init)
        self.use_roi_appearance_branch = bool(use_roi_appearance_branch)
        self.roi_appearance_output_size = max(1, int(roi_appearance_output_size))
        self.roi_appearance_scale_init = float(roi_appearance_scale_init)
        self.use_appearance_contrastive_head = bool(use_appearance_contrastive_head)
        self.appearance_gate_temperature = nn.Parameter(
            torch.tensor(float(appearance_gate_temperature))
        )
        self.appearance_token_branch_temperature = nn.Parameter(
            torch.tensor(float(appearance_token_branch_temperature))
        )
        if use_appearance_conditioned_gate:
            self.appearance_gate_roi_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.appearance_gate_roi_proj = None

        if self.use_appearance_token_branch:
            self.appearance_token_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.appearance_token_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=max(1, nheads),
                dropout=0.0,
                batch_first=True,
            )
            self.appearance_token_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.appearance_token_branch_logit_scale = nn.Parameter(
                torch.tensor(self.appearance_token_branch_scale_init, dtype=torch.float32)
            )
            self.appearance_token_class_embed = None
            self.appearance_token_query_norm = nn.LayerNorm(hidden_dim)
        else:
            self.appearance_token_query_proj = None
            self.appearance_token_cross_attn = None
            self.appearance_token_ffn = None
            self.appearance_token_branch_logit_scale = None
            self.appearance_token_class_embed = None
            self.appearance_token_query_norm = None

        # ---- ROI-based appearance branch ----
        # Uses backbone visual features directly (ROI Align) instead of decoder
        # queries.  The decoder output is optimised for semantic text alignment;
        # fine-grained appearance details (color, border, symbol shape) are better
        # preserved in the backbone feature maps before semantic compression.
        if self.use_roi_appearance_branch:
            # Lightweight spatial encoder: refines ROI features before pooling.
            # Two conv layers with residual preserve spatial structure (numbers,
            # arrows, border thickness) that mean-pooling would discard.
            self.roi_appearance_spatial_encoder = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.roi_appearance_spatial_proj = nn.Linear(hidden_dim, hidden_dim)
            # Learnable query for attention pooling over spatial positions.
            self.roi_appearance_pool_query = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )
            self.roi_appearance_pool_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=max(1, nheads),
                dropout=0.0, batch_first=True,
            )
            self.roi_appearance_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.roi_appearance_text_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=max(1, nheads),
                dropout=0.0, batch_first=True,
            )
            self.roi_appearance_text_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.roi_appearance_norm = nn.LayerNorm(hidden_dim)
            self.roi_appearance_logit_scale = nn.Parameter(
                torch.tensor(self.roi_appearance_scale_init, dtype=torch.float32)
            )
            self.roi_appearance_temperature = nn.Parameter(
                torch.tensor(float(roi_appearance_temperature))
            )
        else:
            self.roi_appearance_spatial_encoder = None
            self.roi_appearance_spatial_proj = None
            self.roi_appearance_pool_query = None
            self.roi_appearance_pool_attn = None
            self.roi_appearance_query_proj = None
            self.roi_appearance_text_cross_attn = None
            self.roi_appearance_text_ffn = None
            self.roi_appearance_norm = None
            self.roi_appearance_logit_scale = None
            self.roi_appearance_temperature = None

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            if self.dual_prompt_class_logit_norm:
                self.dual_prompt_enc_out_class_embed = copy.deepcopy(_class_embed)
                self.dual_prompt_enc_out_class_embed.class_logit_norm = True
                self.dual_prompt_enc_out_class_embed.class_logit_scale = (
                    self.dual_prompt_class_logit_scale
                )
                self.dual_prompt_enc_out_class_embed.class_logit_bias = (
                    self.dual_prompt_class_logit_bias
                )
            else:
                self.dual_prompt_enc_out_class_embed = None

            self.refpoint_embed = None

        self._reset_parameters()

    def _unfreeze_text_encoder_layers_if_requested(self):
        if self.text_encoder_unfreeze_embeddings:
            for parameter in self.bert.embeddings.parameters():
                parameter.requires_grad_(True)
            self.text_encoder_has_trainable_layers = True

        if self.text_encoder_unfreeze_last_n_layers <= 0:
            return

        # BertModelWarper → .encoder.layer, ModernBertWarper → .layers
        encoder_layers = getattr(self.bert, "layers", None) or getattr(
            getattr(self.bert, "encoder", None), "layer", None
        )
        if encoder_layers is None:
            raise AttributeError(
                "text_encoder_unfreeze_last_n_layers requires an encoder.layer or .layers module"
            )

        for layer in encoder_layers[-self.text_encoder_unfreeze_last_n_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        self.text_encoder_has_trainable_layers = True

    def _encode_captions(self, captions, device):
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            if "token_type_ids" in tokenized:
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        if self.freeze_text_encoder and not self.text_encoder_has_trainable_layers:
            with torch.no_grad():
                bert_output = self.bert(**tokenized_for_encoder)
        else:
            bert_output = self.bert(**tokenized_for_encoder)

        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        if self.text_adapter_ffn is not None:
            encoded_text = encoded_text + self.text_adapter_ffn(encoded_text)
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "tokenized": one_hot_token,
        }
        if "token_type_ids" in tokenized:
            text_dict["token_type_ids"] = tokenized["token_type_ids"]
        return text_dict, one_hot_token, cate_to_token_mask_list

    @torch.no_grad()
    def _build_appearance_prototype_bank(self, device, texts=None):
        """Encode a [num_classes] description list into a pooled prototype bank.

        Each class description is tokenized, run through the (frozen) BERT +
        feat_map, and mean-pooled over its valid (non-special) tokens to produce
        one [hidden_dim] prototype.  The result is a [C, hidden_dim] buffer aligned
        to global class id order (texts[i] == class i, matching
        main._refresh_tt100k_label_list / args.appearance_eval_label_list).

        `texts` defaults to self.appearance_prototype_texts (the appearance bank);
        pass args.semantic_label_list to build the semantic ranking bank instead.

        BERT is frozen and this runs under no_grad, so the bank carries no gradient.
        The trainable text_projection inside ContrastiveEmbed is applied LATER, per
        step, in forward_against_bank — so the bank itself can be safely cached.
        Encoded one class per call to keep each caption's token space independent
        (avoids cross-class attention bleed from concatenated captions).
        """
        if texts is None:
            texts = self.appearance_prototype_texts
        if not texts:
            return None, None
        is_multi_facet = bool(texts and isinstance(texts[0], (list, tuple)))
        if is_multi_facet:
            facet_banks = []
            facet_masks = []
            num_facets = max(len(row) for row in texts)
            for facet_idx in range(num_facets):
                facet_texts = [
                    str(row[facet_idx] if facet_idx < len(row) else row[-1])
                    for row in texts
                ]
                bank_f, mask_f = self._build_appearance_prototype_bank(device, texts=facet_texts)
                facet_banks.append(bank_f)
                facet_masks.append(mask_f)
            return torch.stack(facet_banks, dim=1), torch.stack(facet_masks, dim=1)

        protos = []
        valid = []
        for text in texts:
            caption = str(text).strip()
            if not caption:
                protos.append(torch.zeros(self.hidden_dim, device=device))
                valid.append(False)
                continue
            text_dict, _, _ = self._encode_captions([caption], device)
            encoded = text_dict["encoded_text"][0]  # [L, hidden_dim]
            mask = text_dict["text_token_mask"][0].to(encoded.device)  # [L]
            # Drop [CLS]/[SEP]-style special tokens at the ends; keep content
            # tokens.  Mean-pool over valid content tokens.
            token_mask = mask.clone()
            valid_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 2:
                # Exclude the first ([CLS]) and last ([SEP]) valid tokens.
                token_mask[valid_idx[0]] = False
                token_mask[valid_idx[-1]] = False
            denom = token_mask.sum().clamp(min=1).to(encoded.dtype)
            proto = (encoded * token_mask.unsqueeze(-1).to(encoded.dtype)).sum(0) / denom
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            protos.append(proto)
            valid.append(bool(token_mask.any().item()))

        bank = torch.stack(protos, dim=0).to(device)  # [C, hidden_dim]
        bank_mask = torch.tensor(valid, dtype=torch.bool, device=device)
        return bank, bank_mask

    def _get_semantic_prototype_bank(self, device):
        """Cached semantic-text prototype bank for local-evidence candidate ranking.

        Built from self.appearance_semantic_prototype_texts (== args.semantic_label_list),
        aligned to global class id order.  Scoring fused_last_query against THIS bank
        gives a true semantic full-space ranking, so cand_ids reflect the semantic
        head's actual class preference — not its response to appearance descriptions.
        """
        texts = self.appearance_semantic_prototype_texts
        if not texts:
            return None, None
        bank = self.appearance_semantic_prototype_bank
        if bank is None or bank.device != device:
            bank, bank_mask = self._build_appearance_prototype_bank(device, texts=texts)
            self.appearance_semantic_prototype_bank = bank
            self.appearance_semantic_prototype_bank_mask = bank_mask
        return self.appearance_semantic_prototype_bank, self.appearance_semantic_prototype_bank_mask

    def _get_facb_speedhr_facet_prototype_bank(self, device):
        """Return the cached [classes, facets, hidden] FACB text bank."""
        texts = self.facb_speedhr_multi_facet_texts
        if not self.facb_speedhr_candidate_multi_facet or not texts:
            return None, None
        bank = self.facb_speedhr_facet_prototype_bank
        if bank is None or bank.device != device:
            num_facets = max(len(row) for row in texts)
            facet_banks = []
            facet_masks = []
            for facet_idx in range(num_facets):
                facet_texts = [
                    str(row[facet_idx] if facet_idx < len(row) else "")
                    for row in texts
                ]
                bank_f, mask_f = self._build_text_prototype_bank_batched(
                    facet_texts, device=device, chunk_size=64
                )
                facet_banks.append(bank_f)
                facet_masks.append(mask_f)
            bank = torch.stack(facet_banks, dim=1)
            bank_mask = torch.stack(facet_masks, dim=1)
            self.facb_speedhr_facet_prototype_bank = bank
            self.facb_speedhr_facet_prototype_bank_mask = bank_mask
        return (
            self.facb_speedhr_facet_prototype_bank,
            self.facb_speedhr_facet_prototype_bank_mask,
        )

    def _get_facb_speedhr_shared_facet_prototype_bank(self, device):
        """Return padded unique attribute prototypes [facet, value, hidden]."""

        vocab_texts = self.facb_speedhr_shared_facet_vocab_texts
        if not self.facb_speedhr_shared_compositional_facets or not vocab_texts:
            return None, None
        bank = self.facb_speedhr_shared_facet_prototype_bank
        if bank is None or bank.device != device:
            max_values = max((len(vocab) for vocab in vocab_texts), default=0)
            if max_values <= 0:
                return None, None
            facet_banks = []
            facet_masks = []
            for vocab in vocab_texts:
                bank_f, mask_f = self._build_text_prototype_bank_batched(
                    list(vocab), device=device, chunk_size=64
                )
                pad = max_values - int(bank_f.shape[0])
                if pad > 0:
                    bank_f = torch.cat(
                        [bank_f, bank_f.new_zeros((pad, bank_f.shape[-1]))], dim=0
                    )
                    mask_f = torch.cat(
                        [
                            mask_f,
                            torch.zeros(pad, device=device, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                facet_banks.append(bank_f)
                facet_masks.append(mask_f)
            self.facb_speedhr_shared_facet_prototype_bank = torch.stack(
                facet_banks, dim=0
            )
            self.facb_speedhr_shared_facet_prototype_mask = torch.stack(
                facet_masks, dim=0
            )
        return (
            self.facb_speedhr_shared_facet_prototype_bank,
            self.facb_speedhr_shared_facet_prototype_mask,
        )

    @torch.no_grad()
    def _build_text_prototype_bank_batched(self, texts, device, chunk_size=64):
        """Encode independent captions in batches and content-mean-pool them."""
        protos = []
        valid = []
        chunk_size = max(int(chunk_size), 1)
        for start in range(0, len(texts), chunk_size):
            raw_chunk = [str(x).strip() for x in texts[start : start + chunk_size]]
            nonempty = [bool(x) for x in raw_chunk]
            captions = [x if x else "missing facet" for x in raw_chunk]
            text_dict, _, _ = self._encode_captions(captions, device)
            encoded_batch = text_dict["encoded_text"]
            mask_batch = text_dict["text_token_mask"].to(encoded_batch.device)
            for row_idx, is_valid in enumerate(nonempty):
                if not is_valid:
                    protos.append(torch.zeros(self.hidden_dim, device=device))
                    valid.append(False)
                    continue
                encoded = encoded_batch[row_idx]
                token_mask = mask_batch[row_idx].clone()
                valid_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 2:
                    token_mask[valid_idx[0]] = False
                    token_mask[valid_idx[-1]] = False
                denom = token_mask.sum().clamp(min=1).to(encoded.dtype)
                proto = (
                    encoded * token_mask.unsqueeze(-1).to(encoded.dtype)
                ).sum(0) / denom
                protos.append(
                    torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
                )
                valid.append(bool(token_mask.any().item()))
        return (
            torch.stack(protos, dim=0).to(device),
            torch.tensor(valid, dtype=torch.bool, device=device),
        )

    def _pool_content_tokens(self, encoded_text, text_token_mask):
        if encoded_text is None or text_token_mask is None:
            return None, None
        token_mask = text_token_mask.to(dtype=torch.bool, device=encoded_text.device).clone()
        if token_mask.dim() != 2 or encoded_text.dim() != 3:
            return None, None
        for batch_idx in range(token_mask.shape[0]):
            valid_idx = torch.nonzero(token_mask[batch_idx], as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 2:
                token_mask[batch_idx, valid_idx[0]] = False
                token_mask[batch_idx, valid_idx[-1]] = False
        valid_rows = token_mask.any(dim=1)
        weights = token_mask.unsqueeze(-1).to(encoded_text.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        pooled = (encoded_text * weights).sum(dim=1) / denom
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled, valid_rows

    def _get_facb_symbol_align_texts(self):
        if bool(getattr(self, "facb_symbol_align_extract_from_semantic", False)):
            semantic_texts = self.facb_symbol_align_semantic_texts
            if semantic_texts:
                return [
                    self._extract_facb_symbol_text(str(text))
                    for text in semantic_texts
                ]
        texts = self.facb_symbol_align_texts
        if texts is None:
            texts = self.appearance_prototype_texts
        if not texts:
            return None
        fallback_texts = self.facb_symbol_align_fallback_texts
        if isinstance(texts[0], (list, tuple)):
            facet_idx = max(0, int(getattr(self, "facb_symbol_align_facet_index", 1)))
            out = []
            for row_idx, row in enumerate(texts):
                if not row:
                    value = ""
                elif facet_idx < len(row):
                    value = str(row[facet_idx]).strip()
                else:
                    value = str(row[-1]).strip()
                if not value:
                    if fallback_texts is not None and row_idx < len(fallback_texts):
                        value = str(fallback_texts[row_idx]).strip()
                if not value:
                    for item in row:
                        value = str(item).strip()
                        if value:
                            break
                out.append(value)
            return out
        return [str(item) for item in texts]

    @staticmethod
    def _extract_facb_symbol_text(label):
        text = str(label).strip()
        if not text:
            return ""
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            return " ".join(numbers)
        lowered = text.lower()
        direction_words = []
        for word in ("left", "right", "straight", "u-turn", "uturn", "upward", "downward"):
            if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lowered):
                direction_words.append(word)
        if direction_words:
            return " ".join(direction_words)
        return text

    def _encode_facb_symbol_prototypes_online(self, class_ids, device, dtype):
        texts = self._get_facb_symbol_align_texts()
        if not texts:
            return None, None
        flat = class_ids.reshape(-1).to(device=device, dtype=torch.long)
        valid_flat = (flat >= 0) & (flat < len(texts))
        if flat.numel() == 0:
            return None, None
        chunk = max(1, int(getattr(self, "facb_symbol_align_online_chunk", 64)))
        proto_chunks = []
        valid_chunks = []
        for start in range(0, int(flat.numel()), chunk):
            ids = flat[start : start + chunk]
            valid = valid_flat[start : start + chunk]
            captions = [
                str(texts[int(idx.item())]).strip() if bool(ok.item()) else ""
                for idx, ok in zip(ids, valid)
            ]
            text_dict, _, _ = self._encode_captions(captions, device)
            pooled, valid_rows = self._pool_content_tokens(
                text_dict.get("encoded_text", None),
                text_dict.get("text_token_mask", None),
            )
            if pooled is None or valid_rows is None:
                proto_chunks.append(
                    torch.zeros(ids.numel(), self.hidden_dim, device=device, dtype=dtype)
                )
                valid_chunks.append(torch.zeros(ids.numel(), device=device, dtype=torch.bool))
                continue
            pooled = torch.nan_to_num(
                pooled.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
            )
            proto_chunks.append(pooled)
            valid_chunks.append(valid_rows.to(device=device, dtype=torch.bool) & valid)

        protos = torch.cat(proto_chunks, dim=0).view(*class_ids.shape, self.hidden_dim)
        valid = torch.cat(valid_chunks, dim=0).view_as(class_ids)
        return protos, valid

    def _build_global_image_text_alignment_outputs(self, query_features, text_dict):
        if (
            query_features is None
            or text_dict is None
            or self.sail_alignment_head is None
        ):
            return None
        encoded_text = text_dict.get("encoded_text", None)
        text_token_mask = text_dict.get("text_token_mask", None)
        pooled_text, valid_text = self._pool_content_tokens(encoded_text, text_token_mask)
        if pooled_text is None or valid_text is None:
            return None
        pooled_query = torch.nan_to_num(
            query_features.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0
        )
        pooled_text = pooled_text.to(dtype=pooled_query.dtype, device=pooled_query.device)
        aligned = self.sail_alignment_head(pooled_query, pooled_text)
        return {
            "global_alignment_image_features": aligned["image_features"],
            "global_alignment_text_features": aligned["text_features"],
            "global_alignment_text_valid_mask": valid_text.to(device=pooled_query.device),
            "global_alignment_logit_scale": aligned["logit_scale"].to(
                device=pooled_query.device, dtype=pooled_query.dtype
            ),
            "global_alignment_logit_bias": aligned["logit_bias"].to(
                device=pooled_query.device, dtype=pooled_query.dtype
            ),
        }

    def _forward_appearance_local_evidence(
        self, out, srcs, roi_boxes, fused_last_query,
        class_embed_layers, roi_query_gt_labels,
        pred_logits=None, supervision_text_dict=None,
        supervision_cate_to_token_mask_list=None,
    ):
        """Candidate-conditioned local evidence forward.

        EVERYTHING runs in per-image CAPTION-LOCAL class space, because that is the
        space the detector scores in and the space GT labels live in:
          - target["labels"] (== roi_query_gt_labels) are caption-LOCAL indices
            (datasets/odvg.py: label_to_index over the per-image sampled vg_labels),
            NOT global class ids.
          - The detector's semantic ranking is pred_logits (token space) aggregated
            to caption classes via the supervision positive-map — the SAME quantity
            analyze reports as semantic_acc (~0.86).  Candidates MUST come from this,
            not from a 232-class prototype bank (different space → GT never lands in
            the top-M → gt_in_candidate_rate ~ random; the bug across prior versions).

        Candidate set = detector semantic top-M over the caption classes (GT injected
        at train).  Appearance prototypes are gathered from the global appearance
        bank by mapping caption-local -> global via caption_label_ids.

        Writes to `out`:
            appearance_local_evidence_logits : [bs, nq, M]
            appearance_local_candidate_ids   : [bs, nq, M]  GLOBAL class id per slot
            appearance_local_candidate_mask  : [bs, nq, M]  bool valid
            appearance_local_gt_slot         : [bs, nq]     index of GT in M (train), else -1
        """
        device = roi_boxes.device
        bs, nq = roi_boxes.shape[0], roi_boxes.shape[1]
        M = int(getattr(self, "appearance_local_evidence_topm", 8))

        # ---- frozen appearance prototype bank [C, D] (candidate EVIDENCE) ----
        bank, bank_mask = self._get_appearance_prototype_bank(device)
        if bank is None or pred_logits is None or supervision_text_dict is None:
            return
        bank = bank.to(dtype=fused_last_query.dtype)
        C, D = bank.shape

        # ---- detector semantic scores in caption-local class space ----
        # pred_logits: [bs, nq, max_text_len] token logits.  Aggregate to caption
        # classes with the supervision positive-map (same as analyze's semantic_acc).
        supervision_caption_lists = supervision_text_dict.get("caption_lists", None)
        if supervision_caption_lists is not None:
            sem_pos_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict, supervision_caption_lists, self.max_text_len, device,
            )
        else:
            sem_pos_maps = self._build_caption_positive_map_from_masks(
                supervision_cate_to_token_mask_list, self.max_text_len, device,
            )
        # caption_label_ids[b]: caption-local index -> global class id (== vg_labels).
        cap_label_ids = supervision_text_dict.get("caption_label_ids", None)

        out["appearance_is_local_evidence"] = True
        evidence_full = torch.full((bs, nq, M), float("-inf"), device=device, dtype=fused_last_query.dtype)
        cand_global_full = torch.full((bs, nq, M), -1, dtype=torch.long, device=device)
        cand_mask_full = torch.zeros((bs, nq, M), dtype=torch.bool, device=device)
        gt_slot_full = torch.full((bs, nq), -1, dtype=torch.long, device=device)
        # Per-query detector semantic margin (top1 - top2 over caption classes), in
        # the SAME space as candidate ranking.  Lets the loss apply the low-margin
        # filter without a separate sem_logits_bank (which no longer exists).
        sem_margin_full = torch.full((bs, nq), float("inf"), device=device, dtype=torch.float32)

        # Precompute multi-scale ROI tokens once per image (shared across candidates).
        roi_tokens_bnq = self._extract_multiscale_roi_tokens(srcs, roi_boxes, bs, nq, device)
        if roi_tokens_bnq is None:
            return
        roi_tokens_bnq = roi_tokens_bnq.view(bs, nq, roi_tokens_bnq.shape[1], roi_tokens_bnq.shape[2])

        for b in range(bs):
            pos_map = sem_pos_maps[b]                       # [n_cap, max_text_len]
            if pos_map is None or pos_map.shape[0] == 0:
                continue
            n_cap = pos_map.shape[0]
            topm = min(M, n_cap)
            # detector semantic class scores [nq, n_cap]
            with torch.no_grad():
                sem_cls = pred_logits[b].sigmoid().float() @ pos_map.float().T
                sem_cls = torch.nan_to_num(sem_cls, nan=0.0, posinf=20.0, neginf=-20.0)
                # Per-query semantic margin (top1 - top2) for the loss's low-margin filter.
                if n_cap > 1:
                    s2 = torch.topk(sem_cls, k=2, dim=-1).values
                    sem_margin_full[b] = (s2[:, 0] - s2[:, 1]).to(sem_margin_full.dtype)
                else:
                    sem_margin_full[b] = float("inf")
            cand_local = torch.topk(sem_cls, k=topm, dim=-1).indices   # [nq, topm] caption-local

            # GT injection (train): GT is a caption-local index already.
            gt_b = None
            if self.training and roi_query_gt_labels is not None:
                gt_b = roi_query_gt_labels[b]               # [nq], -1 or caption-local
                valid_gt = (gt_b >= 0) & (gt_b < n_cap)
                in_set = (cand_local == gt_b.unsqueeze(-1)) & valid_gt.unsqueeze(-1)
                already = in_set.any(dim=-1)
                need = valid_gt & (~already)
                if need.any():
                    last = cand_local[:, -1]
                    cand_local[:, -1] = torch.where(need, gt_b, last)
                eq = (cand_local == gt_b.unsqueeze(-1)) & valid_gt.unsqueeze(-1)
                has = eq.any(dim=-1)
                gt_slot_full[b, has] = eq.float().argmax(dim=-1)[has]

            # caption-local -> global ids for this image
            if cap_label_ids is not None and b < len(cap_label_ids) and cap_label_ids[b] is not None:
                g = torch.as_tensor(
                    [int(x) for x in cap_label_ids[b]], dtype=torch.long, device=device
                )
            else:
                # Fallback: assume caption-local == global (only valid if caption is full-class).
                g = torch.arange(n_cap, dtype=torch.long, device=device)
            g = g.clamp(0, C - 1)
            cand_global = g[cand_local]                     # [nq, topm] global ids
            cand_proto = bank[cand_global]                  # [nq, topm, D] appearance prototypes

            evidence_b = self.appearance_local_evidence_head(
                roi_tokens_bnq[b],                          # [nq, L, D]
                cand_proto,                                 # [nq, topm, D]
                None,
            )                                               # [nq, topm]
            evidence_full[b, :, :topm] = evidence_b.to(evidence_full.dtype)
            cand_global_full[b, :, :topm] = cand_global
            cand_mask_full[b, :, :topm] = True

        out["appearance_local_evidence_logits"] = evidence_full
        out["appearance_local_candidate_ids"] = cand_global_full
        out["appearance_local_candidate_mask"] = cand_mask_full
        out["appearance_local_gt_slot"] = gt_slot_full
        out["appearance_local_sem_margin"] = sem_margin_full
        return

    def _extract_multiscale_roi_tokens(self, srcs, roi_boxes, bs, nq, device):
        """Multi-scale ROI spatial tokens -> [bs*nq, L, D] (detached, no backbone grad)."""
        levels = getattr(self, "appearance_local_evidence_levels", None) or [0]
        out_size = int(getattr(self, "appearance_local_evidence_roi_size", 7))
        roi_boxes_n = box_ops.box_cxcywh_to_xyxy(roi_boxes.detach().clamp(0.0, 1.0))
        token_chunks = []
        for lvl in levels:
            if lvl < 0 or lvl >= len(srcs):
                continue
            fmap = srcs[lvl]
            _, _, fh, fw = fmap.shape
            b = roi_boxes_n.clone()
            sc = b.new_tensor([fw, fh, fw, fh])
            b = b * sc
            b[..., 0::2] = b[..., 0::2].clamp(0.0, float(fw - 1))
            b[..., 1::2] = b[..., 1::2].clamp(0.0, float(fh - 1))
            b[..., 2] = torch.maximum(b[..., 2], b[..., 0] + 1.0).clamp(max=float(fw))
            b[..., 3] = torch.maximum(b[..., 3], b[..., 1] + 1.0).clamp(max=float(fh))
            bidx = torch.arange(bs, device=device, dtype=b.dtype)[:, None].expand(bs, nq).reshape(-1, 1)
            flat = torch.cat([bidx, b.reshape(-1, 4)], dim=1)
            feats = roi_align(
                fmap.detach(), flat, output_size=out_size, spatial_scale=1.0, aligned=True,
            )
            token_chunks.append(feats.flatten(2).permute(0, 2, 1))
        if not token_chunks:
            return None
        return torch.cat(token_chunks, dim=1)               # [bs*nq, L, D]

    @torch.no_grad()
    def _get_clip_text_bank(self, device):
        """Cached [num_classes, d_clip] L2-normalized CLIP text-embedding bank.

        Built once from self.appearance_clip_text_prototype_texts (candidate
        appearance descriptions, global class id order) using a frozen CLIP text
        tower.  Shares CLIP's joint space with the vision crops -> no modality gap.
        """
        texts = self.appearance_clip_text_prototype_texts
        if not texts:
            return None
        bank = self.appearance_clip_text_bank
        if bank is not None and bank.device == device:
            return bank
        from transformers import CLIPTextModelWithProjection, AutoTokenizer
        text_path = self.appearance_clip_text_model_path or self.appearance_clip_model_path
        tok = AutoTokenizer.from_pretrained(text_path)
        model = CLIPTextModelWithProjection.from_pretrained(text_path).to(device)
        model.eval()
        embeds = []
        bs_enc = 64
        for i in range(0, len(texts), bs_enc):
            chunk = [str(t) for t in texts[i:i + bs_enc]]
            enc = tok(chunk, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            out = model(**enc)
            embeds.append(out.text_embeds.float())
        bank = torch.cat(embeds, dim=0)                      # [C, d]
        bank = torch.nn.functional.normalize(bank, p=2, dim=-1, eps=1e-6)
        # Free the text tower; only the cached bank is needed afterwards.
        del model, tok
        self.appearance_clip_text_bank = bank
        return self.appearance_clip_text_bank

    def _forward_appearance_clip_evidence(
        self, out, samples, roi_boxes, fused_last_query,
        roi_query_gt_labels, pred_logits, supervision_text_dict,
        supervision_cate_to_token_mask_list,
    ):
        """CLIP crop evidence forward.

        Same candidate machinery as the local-evidence path (detector caption-local
        top-M, GT injected at train, low-margin gate), but evidence = cosine between
        a frozen-CLIP embedding of the ORIGINAL-PIXEL crop and frozen-CLIP text
        embeddings of the candidate appearance descriptions.

        To bound CLIP ViT cost, crops are encoded ONLY for queries that participate:
        GT-matched queries at train, all matched queries at eval (via `indices` is not
        available here, so train uses roi_query_gt_labels>=0; eval encodes the top
        scoring queries — handled by the caller passing matched-only boxes is complex,
        so v1 encodes per-image matched-by-GT at train and falls back to all queries
        capped at a max at eval).
        """
        device = roi_boxes.device
        bs, nq = roi_boxes.shape[0], roi_boxes.shape[1]
        M = int(getattr(self, "appearance_clip_topm", 8))
        head = self.appearance_clip_evidence_head
        if head is None or pred_logits is None or supervision_text_dict is None:
            return
        text_bank = self._get_clip_text_bank(device)
        if text_bank is None:
            return
        text_bank = text_bank.to(device)
        C, d_clip = text_bank.shape

        # caption-local semantic scoring (same as local-evidence path)
        supervision_caption_lists = supervision_text_dict.get("caption_lists", None)
        if supervision_caption_lists is not None:
            sem_pos_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict, supervision_caption_lists, self.max_text_len, device,
            )
        else:
            sem_pos_maps = self._build_caption_positive_map_from_masks(
                supervision_cate_to_token_mask_list, self.max_text_len, device,
            )
        cap_label_ids = supervision_text_dict.get("caption_label_ids", None)

        out["appearance_is_local_evidence"] = True   # reuse same analyze/loss path
        evidence_full = torch.full((bs, nq, M), float("-inf"), device=device, dtype=torch.float32)
        cand_global_full = torch.full((bs, nq, M), -1, dtype=torch.long, device=device)
        cand_mask_full = torch.zeros((bs, nq, M), dtype=torch.bool, device=device)
        gt_slot_full = torch.full((bs, nq), -1, dtype=torch.long, device=device)
        sem_margin_full = torch.full((bs, nq), float("inf"), device=device, dtype=torch.float32)
        debug_enabled = os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") == "1"
        trace_limit = int(os.environ.get("GROUNDINGDINO_FACB_TRACE_LIMIT", "0") or 0)
        trace_records = []
        debug_selected_queries = 0
        debug_encoded_crops = 0
        debug_valid_evidence_queries = 0

        # Input image tensor [bs,3,Hp,Wp] in detector-normalized space (Hp,Wp are
        # the batch-PADDED dims).  CRITICAL: roi_boxes are normalized to each image's
        # UNPADDED size (transforms.Normalize divides by the per-image resized h,w
        # BEFORE collate pads to the batch max).  So we must de-normalize with each
        # image's true (unpadded) size, derived from samples.mask (1 = padded pixel),
        # NOT the padded Hp,Wp — otherwise every image except the largest is cropped
        # from the wrong pixels (systematic offset/scale), which looks like random.
        img = samples.tensors if hasattr(samples, "tensors") else samples
        _, _, Hp, Wp = img.shape
        smask = getattr(samples, "mask", None)   # [bs, Hp, Wp], True/1 = padded
        max_eval_q = int(getattr(self, "appearance_clip_max_eval_queries", 64))

        for b in range(bs):
            pos_map = sem_pos_maps[b]
            if pos_map is None or pos_map.shape[0] == 0:
                continue
            n_cap = pos_map.shape[0]
            topm = min(M, n_cap)
            with torch.no_grad():
                sem_cls = pred_logits[b].sigmoid().float() @ pos_map.float().T   # [nq, n_cap]
                sem_cls = torch.nan_to_num(sem_cls, nan=0.0, posinf=20.0, neginf=-20.0)
                if n_cap > 1:
                    s2 = torch.topk(sem_cls, k=2, dim=-1).values
                    sem_margin_full[b] = (s2[:, 0] - s2[:, 1]).to(sem_margin_full.dtype)
            cand_local = torch.topk(sem_cls, k=topm, dim=-1).indices            # [nq, topm]

            # ---- choose which queries to actually encode through CLIP ----
            if self.training and roi_query_gt_labels is not None:
                sel = torch.nonzero(roi_query_gt_labels[b] >= 0, as_tuple=False).squeeze(-1)
            else:
                # Eval should verify detected regions, not arbitrary raw queries.
                # Approximate postprocess by selecting high scoring query-class
                # detections from the flattened semantic score table, then dedupe
                # to query ids. The legacy query-confidence mode is kept for
                # diagnostics.
                select_mode = str(
                    getattr(self, "appearance_clip_eval_select_mode", "detections")
                ).strip().lower()
                if select_mode == "query_conf":
                    conf = sem_cls.max(dim=-1).values
                    k = min(max_eval_q, nq)
                    sel = torch.topk(conf, k=k).indices
                else:
                    flat_scores = sem_cls.reshape(-1)
                    min_det_score = float(getattr(self, "appearance_clip_min_det_score", 0.0))
                    if min_det_score > 0.0:
                        valid_flat = flat_scores >= min_det_score
                        if valid_flat.any():
                            flat_scores_for_topk = flat_scores.masked_fill(~valid_flat, float("-inf"))
                        else:
                            flat_scores_for_topk = flat_scores
                    else:
                        flat_scores_for_topk = flat_scores
                    if flat_scores.numel() == 0:
                        continue
                    finite_flat = torch.isfinite(flat_scores_for_topk)
                    if not finite_flat.any():
                        continue
                    top_flat_k = min(max(int(max_eval_q) * 4, int(max_eval_q)), int(flat_scores.numel()))
                    flat_idx = torch.topk(flat_scores_for_topk, k=top_flat_k).indices
                    if min_det_score > 0.0:
                        flat_idx = flat_idx[flat_scores[flat_idx] >= min_det_score]
                    if flat_idx.numel() == 0:
                        continue
                    query_ids = torch.div(flat_idx, n_cap, rounding_mode="trunc")
                    seen_q = set()
                    sel_list = []
                    for q_item in query_ids.detach().cpu().tolist():
                        q_int = int(q_item)
                        if q_int in seen_q:
                            continue
                        seen_q.add(q_int)
                        sel_list.append(q_int)
                        if len(sel_list) >= max_eval_q:
                            break
                    if sel_list:
                        sel = torch.as_tensor(sel_list, dtype=torch.long, device=device)
                    else:
                        sel = torch.empty(0, dtype=torch.long, device=device)
                min_area = float(getattr(self, "appearance_clip_min_box_area", 0.0))
                if min_area > 0.0 and sel.numel() > 0:
                    box_wh = roi_boxes[b, sel, 2:4].detach().clamp(min=0.0, max=1.0)
                    area = box_wh[:, 0] * box_wh[:, 1]
                    sel = sel[area >= min_area]
            if sel.numel() == 0:
                continue
            debug_selected_queries += int(sel.numel())

            # GT injection (caption-local) for selected queries
            gt_b = roi_query_gt_labels[b] if (self.training and roi_query_gt_labels is not None) else None
            if gt_b is not None:
                valid_gt = (gt_b >= 0) & (gt_b < n_cap)
                in_set = (cand_local == gt_b.unsqueeze(-1)) & valid_gt.unsqueeze(-1)
                already = in_set.any(dim=-1)
                need = valid_gt & (~already)
                if need.any():
                    cand_local[:, -1] = torch.where(need, gt_b, cand_local[:, -1])
                eq = (cand_local == gt_b.unsqueeze(-1)) & valid_gt.unsqueeze(-1)
                has = eq.any(dim=-1)
                gt_slot_full[b, has] = eq.float().argmax(dim=-1)[has]

            # caption-local -> global
            if cap_label_ids is not None and b < len(cap_label_ids) and cap_label_ids[b] is not None:
                g = torch.as_tensor([int(x) for x in cap_label_ids[b]], dtype=torch.long, device=device)
            else:
                g = torch.arange(n_cap, dtype=torch.long, device=device)
            g = g.clamp(0, C - 1)

            # ---- crop selected boxes from the input image tensor ----
            # Per-image UNPADDED size from the mask (mask True = padded pixel).
            if smask is not None:
                valid = ~smask[b].bool()                 # [Hp, Wp], True = real pixel
                Hb = int(valid.any(dim=1).sum().item()) or Hp
                Wb = int(valid.any(dim=0).sum().item()) or Wp
            else:
                Hb, Wb = Hp, Wp
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes[b, sel].detach().clamp(0.0, 1.0))
            scale = boxes_xyxy.new_tensor([Wb, Hb, Wb, Hb])
            boxes_pix = (boxes_xyxy * scale).round().long()
            crops = []
            keep = []
            for j in range(sel.numel()):
                x0, y0, x1, y1 = [int(v) for v in boxes_pix[j].tolist()]
                x0 = max(0, min(x0, Wb - 1)); y0 = max(0, min(y0, Hb - 1))
                x1 = max(x0 + 1, min(x1, Wb)); y1 = max(y0 + 1, min(y1, Hb))
                crop = img[b:b + 1, :, y0:y1, x0:x1]
                if crop.shape[-1] < 1 or crop.shape[-2] < 1:
                    continue
                crops.append(F.interpolate(crop, size=(head.crop_size, head.crop_size),
                                           mode="bilinear", align_corners=False))
                keep.append(j)
            if not crops:
                continue
            crops = torch.cat(crops, dim=0)                  # [n_sel, 3, crop, crop]
            debug_encoded_crops += int(crops.shape[0])
            # ---- Probe A: optional crop visualization dump (env-gated, one-shot) ----
            # GROUNDINGDINO_CROP_DUMP=<dir> saves the first few de-normalized crops to
            # PNG so we can eyeball whether boxes land on the sign body, then disables
            # itself.  No effect on training/eval numerics.
            _dump_dir = os.environ.get("GROUNDINGDINO_CROP_DUMP", "").strip()
            if _dump_dir and not getattr(self, "_clip_crop_dumped", False):
                try:
                    from torchvision.utils import save_image as _save_image
                    os.makedirs(_dump_dir, exist_ok=True)
                    _mean = crops.new_tensor(CLIPCropEvidence.IMAGENET_MEAN).view(1, 3, 1, 1)
                    _std = crops.new_tensor(CLIPCropEvidence.IMAGENET_STD).view(1, 3, 1, 1)
                    _vis = (crops[: min(16, crops.shape[0])].float() * _std + _mean).clamp(0, 1)
                    for _i in range(_vis.shape[0]):
                        _save_image(_vis[_i], os.path.join(_dump_dir, f"crop_b{b}_{_i}.png"))
                    self._clip_crop_dumped = True
                    print(f"[clip-crop-dump] saved {_vis.shape[0]} crops to {_dump_dir}")
                except Exception as _e:  # noqa: BLE001 - visualization must never break training
                    print(f"[clip-crop-dump] skipped: {_e}")
            img_embeds = head.encode_crops(crops)            # [n_sel, d] L2

            keep_sel = sel[torch.as_tensor(keep, device=device, dtype=torch.long)]
            cand_local_sel = cand_local[keep_sel]            # [n_sel, topm]
            cand_global_sel = g[cand_local_sel]              # [n_sel, topm]
            cand_text = text_bank[cand_global_sel]           # [n_sel, topm, d]
            ev = head.score(img_embeds, cand_text, None)     # [n_sel, topm]

            evidence_full[b, keep_sel, :topm] = ev.to(evidence_full.dtype)
            cand_global_full[b, keep_sel, :topm] = cand_global_sel
            cand_mask_full[b, keep_sel, :topm] = True
            debug_valid_evidence_queries += int(keep_sel.numel())
            if trace_limit > 0 and len(trace_records) < trace_limit:
                ev_cpu = ev.detach().float().cpu()
                cand_global_cpu = cand_global_sel.detach().cpu()
                cand_local_cpu = cand_local_sel.detach().cpu()
                sem_cls_cpu = sem_cls.detach().float().cpu()
                keep_sel_cpu = keep_sel.detach().cpu()
                boxes_norm_cpu = roi_boxes[b, keep_sel].detach().float().cpu()
                boxes_pix_cpu = boxes_pix[
                    torch.as_tensor(keep, dtype=torch.long, device=boxes_pix.device)
                ].detach().cpu()
                for row in range(int(keep_sel_cpu.numel())):
                    if len(trace_records) >= trace_limit:
                        break
                    q_idx = int(keep_sel_cpu[row].item())
                    logits_row = ev_cpu[row, :topm]
                    probs_row = torch.softmax(logits_row, dim=-1)
                    app_best_slot = int(probs_row.argmax().item())
                    sem_scores = [
                        float(sem_cls_cpu[q_idx, int(cand_local_cpu[row, slot].item())].item())
                        for slot in range(topm)
                    ]
                    trace_records.append(
                        {
                            "batch": int(b),
                            "query": q_idx,
                            "box_cxcywh_norm": [round(float(x), 6) for x in boxes_norm_cpu[row].tolist()],
                            "box_xyxy_pix": [int(x) for x in boxes_pix_cpu[row].tolist()],
                            "candidate_global_ids": [
                                int(cand_global_cpu[row, slot].item()) for slot in range(topm)
                            ],
                            "semantic_scores": [round(float(x), 6) for x in sem_scores],
                            "appearance_logits": [
                                round(float(logits_row[slot].item()), 6) for slot in range(topm)
                            ],
                            "appearance_probs": [
                                round(float(probs_row[slot].item()), 6) for slot in range(topm)
                            ],
                            "semantic_best_global_id": int(cand_global_cpu[row, 0].item()),
                            "appearance_best_global_id": int(cand_global_cpu[row, app_best_slot].item()),
                            "appearance_best_slot": app_best_slot,
                        }
                    )

        out["appearance_local_evidence_logits"] = evidence_full
        out["appearance_local_candidate_ids"] = cand_global_full
        out["appearance_local_candidate_mask"] = cand_mask_full
        out["appearance_local_gt_slot"] = gt_slot_full
        out["appearance_local_sem_margin"] = sem_margin_full
        if debug_enabled and not getattr(self, "_facb_clip_debug_printed", False):
            finite_count = int(torch.isfinite(evidence_full).sum().item())
            valid_slots = int(cand_mask_full.sum().item())
            print(
                "[facb-debug][clip] "
                f"bs={bs} nq={nq} topm={M} "
                f"selected_queries={debug_selected_queries} "
                f"encoded_crops={debug_encoded_crops} "
                f"valid_evidence_queries={debug_valid_evidence_queries} "
                f"valid_slots={valid_slots} finite_logits={finite_count} "
                f"text_bank_shape={tuple(text_bank.shape)} "
                f"pairwise_head={head.pairwise_head is not None}",
                flush=True,
            )
            self._facb_clip_debug_printed = True
        if trace_limit > 0 and not getattr(self, "_facb_clip_trace_printed", False):
            for record in trace_records:
                print(
                    "[facb-trace][clip] " + json.dumps(record, ensure_ascii=False),
                    flush=True,
                )
            self._facb_clip_trace_printed = True
        return

    def _get_appearance_prototype_bank(self, device):
        """Return the cached prototype bank, building it on first use."""
        if not self.use_appearance_prototype_bank:
            return None, None
        bank = self.appearance_prototype_bank
        if bank is None or bank.device != device:
            bank, bank_mask = self._build_appearance_prototype_bank(device)
            self.appearance_prototype_bank = bank
            self.appearance_prototype_bank_mask = bank_mask
        return self.appearance_prototype_bank, self.appearance_prototype_bank_mask

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.local_roi_query_fc is not None:
            for module in self.local_roi_query_fc:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.local_roi_query_gate is not None:
            linear_layers = [m for m in self.local_roi_query_gate if isinstance(m, nn.Linear)]
            for idx, module in enumerate(linear_layers):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if linear_layers and linear_layers[-1].bias is not None:
                nn.init.constant_(linear_layers[-1].bias, -2.0)
        if self.appearance_roi_proj is not None:
            for module in self.appearance_roi_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_roi_query_proj is not None:
            for module in self.appearance_roi_query_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_roi_token_proj is not None:
            for module in self.appearance_roi_token_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_roi_gate is not None:
            linear_layers = [m for m in self.appearance_roi_gate if isinstance(m, nn.Linear)]
            for idx, module in enumerate(linear_layers):
                if idx == len(linear_layers) - 1:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if linear_layers and linear_layers[-1].bias is not None:
                nn.init.constant_(linear_layers[-1].bias, -2.0)
        if self.appearance_query_bias is not None:
            nn.init.normal_(self.appearance_query_bias.weight, mean=0.0, std=0.02)
        if self.appearance_class_embed is not None:
            nn.init.normal_(self.appearance_class_embed.weight, mean=0.0, std=0.02)
        if self.proto_cls_query_proj is not None:
            for module in self.proto_cls_query_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.proto_cls_proto_gate is not None:
            linear_layers = [m for m in self.proto_cls_proto_gate if isinstance(m, nn.Linear)]
            for idx, module in enumerate(linear_layers):
                if idx == len(linear_layers) - 1:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if linear_layers and linear_layers[-1].bias is not None:
                nn.init.constant_(linear_layers[-1].bias, -2.0)
        if self.appearance_text_query_proj is not None:
            for module in self.appearance_text_query_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_token_query_proj is not None:
            for module in self.appearance_token_query_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_token_ffn is not None:
            for module in self.appearance_token_ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.appearance_token_branch_logit_scale is not None:
            nn.init.constant_(self.appearance_token_branch_logit_scale, self.appearance_token_branch_scale_init)
        if self.roi_appearance_spatial_encoder is not None:
            for module in self.roi_appearance_spatial_encoder:
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.roi_appearance_spatial_proj is not None:
            nn.init.xavier_uniform_(self.roi_appearance_spatial_proj.weight)
            nn.init.constant_(self.roi_appearance_spatial_proj.bias, 0)
        if self.roi_appearance_query_proj is not None:
            for module in self.roi_appearance_query_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.roi_appearance_text_ffn is not None:
            for module in self.roi_appearance_text_ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        if self.roi_appearance_logit_scale is not None:
            nn.init.constant_(self.roi_appearance_logit_scale, self.roi_appearance_scale_init)

    def _build_caption_positive_map_from_masks(self, category_token_masks, max_text_len, device):
        positive_maps = []
        for category_mask in category_token_masks:
            positive_map = torch.zeros(
                (category_mask.shape[0], max_text_len),
                dtype=torch.float32,
                device=device,
            )
            valid_length = min(category_mask.shape[1], max_text_len)
            if valid_length > 0:
                positive_map[:, :valid_length] = category_mask[:, :valid_length].to(
                    device=device, dtype=torch.float32
                )
            denominator = positive_map.sum(dim=1, keepdim=True)
            valid_rows = denominator.squeeze(-1) > 0
            positive_map[valid_rows] = positive_map[valid_rows] / denominator[valid_rows]
            positive_maps.append(positive_map)
        return positive_maps

    def _build_caption_positive_map_from_text(
        self,
        text_dict,
        caption_lists,
        max_text_len,
        device,
    ):
        positive_maps = []
        tokenized = text_dict.get("tokenized", None)

        class _BatchCharToToken:
            def __init__(self, batch_encoding, batch_idx):
                self.batch_encoding = batch_encoding
                self.batch_idx = batch_idx

            def char_to_token(self, char_idx):
                try:
                    return self.batch_encoding.char_to_token(self.batch_idx, char_idx)
                except TypeError:
                    return self.batch_encoding.char_to_token(char_idx)

        for batch_idx, cat_list in enumerate(caption_lists):
            if not cat_list:
                positive_maps.append(
                    torch.zeros((0, max_text_len), dtype=torch.float32, device=device)
                )
                continue
            if tokenized is None or not hasattr(tokenized, "char_to_token"):
                positive_maps.append(
                    torch.zeros((len(cat_list), max_text_len), dtype=torch.float32, device=device)
                )
                continue

            per_label = create_positive_map(
                _BatchCharToToken(tokenized, batch_idx),
                torch.arange(len(cat_list), dtype=torch.long),
                cat_list,
                None,
                max_text_len=max_text_len,
            )
            per_label = per_label.to(device=device, dtype=torch.float32)
            per_label = per_label / per_label.sum(dim=1, keepdim=True).clamp(min=1.0)
            positive_maps.append(per_label)
        return positive_maps

    def _build_appearance_fusion_outputs(
        self,
        query_features,
        pred_boxes,
        feature_map,
        semantic_pred_logits,
        supervision_text_dict,
        appearance_text_dict,
        appearance_cate_to_token_mask_list,
        supervision_cate_to_token_mask_list,
    ):
        if (
            not self.use_appearance_roi_fusion
            or feature_map is None
            or self.appearance_roi_proj is None
            or self.appearance_roi_token_proj is None
            or self.appearance_roi_query_proj is None
            or self.appearance_roi_cross_attn is None
            or self.appearance_roi_gate is None
            or query_features is None
            or pred_boxes is None
        ):
            return None

        bs, num_queries, hidden_dim = query_features.shape
        _, _, feat_h, feat_w = feature_map.shape
        if bs == 0 or num_queries == 0 or feat_h <= 0 or feat_w <= 0:
            return None

        roi_boxes = pred_boxes.detach().clamp(0.0, 1.0)
        roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes)
        scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
        roi_boxes_xyxy = roi_boxes_xyxy * scale
        roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(0.0, max(float(feat_w - 1), 0.0))
        roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(0.0, max(float(feat_h - 1), 0.0))
        roi_boxes_xyxy[..., 2] = torch.maximum(
            roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
        ).clamp(max=float(feat_w))
        roi_boxes_xyxy[..., 3] = torch.maximum(
            roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
        ).clamp(max=float(feat_h))

        batch_indices = torch.arange(bs, device=query_features.device, dtype=roi_boxes_xyxy.dtype)
        batch_indices = batch_indices[:, None].expand(bs, num_queries).reshape(-1, 1)
        flat_rois = torch.cat([batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1)

        roi_feats = roi_align(
            feature_map,
            flat_rois,
            output_size=self.appearance_roi_output_size,
            spatial_scale=1.0,
            aligned=True,
        )
        roi_tokens = roi_feats.flatten(2).transpose(1, 2)
        roi_tokens = self.appearance_roi_token_proj(roi_tokens)

        query_tokens = self.appearance_roi_query_proj(
            query_features.detach().reshape(-1, hidden_dim)
        ).view(bs * num_queries, 1, hidden_dim)
        if self.appearance_query_bias is not None:
            query_bias = self.appearance_query_bias.weight.unsqueeze(0).expand(bs, -1, -1)
            query_tokens = query_tokens + query_bias.reshape(bs * num_queries, 1, hidden_dim)
        attn_output, _ = self.appearance_roi_cross_attn(
            query=query_tokens,
            key=roi_tokens,
            value=roi_tokens,
            need_weights=False,
        )
        app_query = self.appearance_roi_query_norm(query_tokens + attn_output).squeeze(1)
        app_query = self.appearance_roi_proj(app_query).view(bs, num_queries, hidden_dim)
        app_query = F.normalize(app_query, dim=-1, eps=1e-6)
        debug_enabled = (
            not self._appearance_debug_printed
            and os.environ.get("APPEARANCE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        roi_tokens_view = roi_tokens.view(bs, num_queries, -1, hidden_dim)
        attn_output_view = attn_output.view(bs, num_queries, hidden_dim)

        base_query = F.normalize(query_features, dim=-1, eps=1e-6)
        gate_input = torch.cat([base_query, app_query], dim=-1)
        alpha = torch.sigmoid(self.appearance_roi_gate(gate_input))
        alpha = alpha * self.appearance_roi_alpha

        appearance_prototypes, appearance_valid_masks = build_category_text_prototypes(
            appearance_text_dict["encoded_text"],
            appearance_cate_to_token_mask_list,
            appearance_text_dict["text_token_mask"],
        )
        if self.appearance_class_embed is not None:
            caption_label_ids = appearance_text_dict.get("caption_label_ids", None)
            if caption_label_ids is not None:
                learned_prototypes = []
                learned_valid_masks = []
                for batch_idx, label_ids in enumerate(caption_label_ids):
                    local_ids = []
                    for label_id in label_ids:
                        try:
                            local_ids.append(int(label_id))
                        except (TypeError, ValueError):
                            continue
                    if local_ids:
                        index_tensor = torch.as_tensor(
                            local_ids, device=query_features.device, dtype=torch.long
                        )
                        index_tensor = index_tensor.clamp(min=0, max=max(self.appearance_num_classes - 1, 0))
                        learned_prototypes.append(self.appearance_class_embed(index_tensor))
                        learned_valid_masks.append(
                            torch.ones(len(local_ids), device=query_features.device, dtype=torch.bool)
                        )
                    else:
                        learned_prototypes.append(
                            query_features.new_zeros((0, hidden_dim))
                        )
                        learned_valid_masks.append(
                            torch.zeros((0,), device=query_features.device, dtype=torch.bool)
                        )
                appearance_prototypes = learned_prototypes
                appearance_valid_masks = learned_valid_masks
        semantic_prototypes, semantic_valid_masks = build_category_text_prototypes(
            supervision_text_dict["encoded_text"],
            supervision_cate_to_token_mask_list,
            supervision_text_dict["text_token_mask"],
        )
        supervision_caption_lists = supervision_text_dict.get("caption_lists", None)
        if supervision_caption_lists is not None:
            semantic_pos_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict,
                supervision_caption_lists,
                self.max_text_len,
                query_features.device,
            )
        else:
            semantic_pos_maps = self._build_caption_positive_map_from_masks(
                supervision_cate_to_token_mask_list,
                self.max_text_len,
                query_features.device,
            )

        appearance_token_logits = torch.zeros(
            (bs, num_queries, self.max_text_len),
            dtype=query_features.dtype,
            device=query_features.device,
        )
        appearance_text_logits = torch.zeros(
            (bs, num_queries, self.max_text_len),
            dtype=query_features.dtype,
            device=query_features.device,
        )
        appearance_class_scores = []
        proto_cls_scores = [] if self.use_proto_cls_head else None
        for batch_idx in range(bs):
            proto = appearance_prototypes[batch_idx]
            sem_proto = semantic_prototypes[batch_idx]
            app_valid_mask = appearance_valid_masks[batch_idx]
            sem_valid_mask = semantic_valid_masks[batch_idx]
            sem_pos_map = semantic_pos_maps[batch_idx]
            class_count = min(proto.shape[0], app_valid_mask.shape[0])
            sem_class_count = min(
                sem_pos_map.shape[0],
                sem_proto.shape[0],
                sem_valid_mask.shape[0],
            )
            if debug_enabled and batch_idx == 0:
                caption_label_ids_dbg = appearance_text_dict.get("caption_label_ids", None)
                caption_label_count = (
                    len(caption_label_ids_dbg[batch_idx])
                    if caption_label_ids_dbg is not None and batch_idx < len(caption_label_ids_dbg)
                    and caption_label_ids_dbg[batch_idx] is not None
                    else None
                )
                print(
                    "[appearance-count-debug] "
                    f"batch={batch_idx} "
                    f"use_learnable_proto={self.use_learnable_appearance_proto} "
                    f"has_class_embed={self.appearance_class_embed is not None} "
                    f"proto_count={proto.shape[0]} "
                    f"app_valid_count={app_valid_mask.shape[0]} "
                    f"sem_pos_count={sem_pos_map.shape[0]} "
                    f"class_count={class_count} "
                    f"sem_class_count={sem_class_count} "
                    f"caption_label_count={caption_label_count}"
                )
            appearance_scores_full = query_features.new_zeros((num_queries, class_count))
            proto_scores_full = (
                query_features.new_zeros((num_queries, class_count))
                if proto_cls_scores is not None
                else None
            )

            if proto.numel() == 0 or class_count == 0:
                if debug_enabled and batch_idx == 0:
                    print(
                        "[appearance-debug] skipped-empty "
                        f"roi_tokens_norm={roi_tokens_view[0].norm(dim=-1).mean().item():.4f} "
                        f"attn_output_norm={attn_output_view[0].norm(dim=-1).mean().item():.4f} "
                        f"app_query_norm={app_query[0].norm(dim=-1).mean().item():.4f} "
                        f"class_count={class_count} "
                        f"proto_shape={tuple(proto.shape)} "
                        f"sem_proto_shape={tuple(sem_proto.shape)}"
                    )
                    self._appearance_debug_printed = True
                appearance_class_scores.append(appearance_scores_full)
                if proto_cls_scores is not None:
                    proto_cls_scores.append(proto_scores_full)
                continue

            proto = proto[:class_count]
            app_valid_mask = app_valid_mask[:class_count].to(query_features.device)
            if not bool(app_valid_mask.any()):
                if debug_enabled and batch_idx == 0:
                    print(
                        "[appearance-debug] skipped-no-app-valid "
                        f"roi_tokens_norm={roi_tokens_view[0].norm(dim=-1).mean().item():.4f} "
                        f"attn_output_norm={attn_output_view[0].norm(dim=-1).mean().item():.4f} "
                        f"app_query_norm={app_query[0].norm(dim=-1).mean().item():.4f} "
                        f"class_count={class_count} "
                        f"sem_class_count={sem_class_count}"
                    )
                    self._appearance_debug_printed = True
                appearance_class_scores.append(appearance_scores_full)
                if proto_cls_scores is not None:
                    proto_cls_scores.append(proto_scores_full)
                continue

            app_proto_valid = F.normalize(proto[app_valid_mask], dim=-1, eps=1e-6)
            app_proto_valid = torch.nan_to_num(app_proto_valid, nan=0.0, posinf=0.0, neginf=0.0)
            app_scores_valid = (
                app_query[batch_idx] @ app_proto_valid.transpose(0, 1)
            ) / self.appearance_roi_temperature
            app_scores_valid = torch.nan_to_num(app_scores_valid, nan=0.0, posinf=20.0, neginf=-20.0)
            app_scores_valid = app_scores_valid.clamp(min=-20.0, max=20.0)
            appearance_scores_full[:, app_valid_mask] = app_scores_valid.to(
                dtype=appearance_scores_full.dtype
            )

            if debug_enabled and batch_idx == 0:
                top1_mean = appearance_scores_full.max(dim=-1).values.mean().item()
                class_std_mean = (
                    appearance_scores_full.std(dim=-1, unbiased=False).mean().item()
                    if appearance_scores_full.shape[-1] > 0
                    else 0.0
                )
                print(
                    "[appearance-debug] "
                    f"roi_tokens_norm={roi_tokens_view[0].norm(dim=-1).mean().item():.4f} "
                    f"attn_output_norm={attn_output_view[0].norm(dim=-1).mean().item():.4f} "
                    f"app_query_norm={app_query[0].norm(dim=-1).mean().item():.4f} "
                    f"app_proto_norm={app_proto_valid.norm(dim=-1).mean().item():.4f} "
                    f"class_count={class_count} "
                    f"sem_class_count={sem_class_count} "
                    f"class_top1_mean={top1_mean:.4f} "
                    f"class_std_mean={class_std_mean:.4f} "
                    f"alpha_mean={alpha[0].mean().item():.4f}"
                )
                self._appearance_debug_printed = True
            appearance_class_scores.append(appearance_scores_full)

            if self.use_appearance_text_alignment and self.appearance_text_query_proj is not None:
                text_query = base_query[batch_idx] + alpha[batch_idx] * app_query[batch_idx]
                text_query = self.appearance_text_query_proj(text_query)
                text_query = F.normalize(text_query, dim=-1, eps=1e-6)
                text_query = torch.nan_to_num(text_query, nan=0.0, posinf=0.0, neginf=0.0)
                text_logits = text_query @ supervision_text_dict["encoded_text"][batch_idx].transpose(0, 1)
                text_logits = torch.nan_to_num(text_logits, nan=0.0, posinf=20.0, neginf=-20.0)
                text_logits = text_logits.clamp(min=-20.0, max=20.0)
                valid_token_len = min(text_logits.shape[-1], self.max_text_len)
                if valid_token_len > 0:
                    appearance_text_logits[batch_idx, :, :valid_token_len] = text_logits[:, :valid_token_len].to(
                        dtype=appearance_text_logits.dtype
                    )

            token_overlap = min(class_count, sem_class_count)
            if token_overlap > 0:
                sem_pos_map_overlap = sem_pos_map[:token_overlap]
                sem_proto_overlap = sem_proto[:token_overlap]
                sem_valid_overlap = sem_valid_mask[:token_overlap].to(query_features.device)
                app_valid_overlap = app_valid_mask[:token_overlap]
                token_valid_mask = app_valid_overlap & sem_valid_overlap
                if bool(token_valid_mask.any()):
                    appearance_scores_for_token = appearance_scores_full[:, :token_overlap]
                    if (
                        semantic_pred_logits is not None
                        and self.appearance_roi_semantic_topk > 0
                        and token_overlap > self.appearance_roi_semantic_topk
                    ):
                        semantic_class_scores = semantic_pred_logits[batch_idx] @ sem_pos_map_overlap.transpose(0, 1).to(
                            device=query_features.device, dtype=query_features.dtype
                        )
                        topk = min(token_overlap, self.appearance_roi_semantic_topk)
                        topk_indices = semantic_class_scores.topk(k=topk, dim=-1).indices
                        topk_mask = torch.zeros_like(appearance_scores_for_token, dtype=torch.bool)
                        topk_mask.scatter_(1, topk_indices, True)
                        appearance_scores_for_token = appearance_scores_for_token.masked_fill(
                            ~topk_mask, 0.0
                        )
                    appearance_token_logits[batch_idx] = appearance_scores_for_token[:, token_valid_mask] @ sem_pos_map_overlap[
                        token_valid_mask
                    ].to(device=query_features.device, dtype=query_features.dtype)

            if proto_cls_scores is not None:
                proto_overlap = min(class_count, sem_class_count)
                if proto_overlap > 0:
                    sem_proto_overlap = sem_proto[:proto_overlap]
                    sem_valid_overlap = sem_valid_mask[:proto_overlap].to(query_features.device)
                    app_valid_overlap = app_valid_mask[:proto_overlap]
                    proto_valid_mask = app_valid_overlap & sem_valid_overlap
                    if bool(proto_valid_mask.any()):
                        sem_proto_valid = F.normalize(sem_proto_overlap[proto_valid_mask], dim=-1, eps=1e-6)
                        sem_proto_valid = torch.nan_to_num(sem_proto_valid, nan=0.0, posinf=0.0, neginf=0.0)
                        app_proto_overlap = F.normalize(proto[:proto_overlap][proto_valid_mask], dim=-1, eps=1e-6)
                        app_proto_overlap = torch.nan_to_num(app_proto_overlap, nan=0.0, posinf=0.0, neginf=0.0)
                        proto_gate = torch.sigmoid(
                            self.proto_cls_proto_gate(torch.cat([sem_proto_valid, app_proto_overlap], dim=-1))
                        )
                        proto_gate = torch.nan_to_num(proto_gate, nan=0.0, posinf=1.0, neginf=0.0)
                        fused_proto = F.normalize(
                            (1.0 - proto_gate) * sem_proto_valid + proto_gate * app_proto_overlap,
                            dim=-1,
                            eps=1e-6,
                        )
                        fused_proto = torch.nan_to_num(fused_proto, nan=0.0, posinf=0.0, neginf=0.0)
                        proto_query = base_query[batch_idx] + alpha[batch_idx] * app_query[batch_idx]
                        proto_query = self.proto_cls_query_proj(proto_query)
                        proto_query = F.normalize(proto_query, dim=-1, eps=1e-6)
                        proto_query = torch.nan_to_num(proto_query, nan=0.0, posinf=0.0, neginf=0.0)
                        proto_scores_valid = (
                            proto_query @ fused_proto.transpose(0, 1)
                        ) / self.proto_cls_temperature
                        proto_scores_valid = torch.nan_to_num(proto_scores_valid, nan=0.0, posinf=20.0, neginf=-20.0)
                        proto_scores_valid = proto_scores_valid.clamp(min=-20.0, max=20.0)
                        proto_scores_overlap = proto_scores_full[:, :proto_overlap]
                        proto_scores_overlap[:, proto_valid_mask] = proto_scores_valid.to(
                            dtype=proto_scores_full.dtype
                        )
                proto_cls_scores.append(proto_scores_full)

        outputs = {
            "appearance_token_logits": appearance_token_logits,
            "appearance_text_logits": appearance_text_logits,
            "appearance_class_scores": appearance_class_scores,
            "appearance_alpha": alpha,
            "app_query": app_query,
        }
        caption_label_ids = appearance_text_dict.get("caption_label_ids", None)
        if caption_label_ids is not None:
            outputs["appearance_caption_label_ids"] = caption_label_ids
        if proto_cls_scores is not None:
            outputs["proto_cls_scores"] = proto_cls_scores
        return outputs

    def _fuse_local_roi_query_features(self, query_features, pred_boxes, feature_map):
        if (
            not self.use_local_roi_query_fusion
            or feature_map is None
            or query_features is None
            or pred_boxes is None
        ):
            return query_features

        bs, num_queries, hidden_dim = query_features.shape
        _, _, feat_h, feat_w = feature_map.shape
        if bs == 0 or num_queries == 0 or feat_h <= 0 or feat_w <= 0:
            return query_features

        roi_boxes = pred_boxes.detach().clamp(0.0, 1.0)
        roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes)
        scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
        roi_boxes_xyxy = roi_boxes_xyxy * scale
        roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(0.0, max(float(feat_w - 1), 0.0))
        roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(0.0, max(float(feat_h - 1), 0.0))
        roi_boxes_xyxy[..., 2] = torch.maximum(
            roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
        ).clamp(max=float(feat_w))
        roi_boxes_xyxy[..., 3] = torch.maximum(
            roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
        ).clamp(max=float(feat_h))

        batch_indices = torch.arange(bs, device=query_features.device, dtype=roi_boxes_xyxy.dtype)
        batch_indices = batch_indices[:, None].expand(bs, num_queries).reshape(-1, 1)
        flat_rois = torch.cat(
            [batch_indices, roi_boxes_xyxy.reshape(-1, 4)],
            dim=1,
        )

        roi_feats = roi_align(
            feature_map,
            flat_rois,
            output_size=self.local_roi_query_fusion_output_size,
            spatial_scale=1.0,
            aligned=True,
        )
        roi_feats = self.local_roi_query_value_proj(roi_feats)
        roi_feats = roi_feats.mean(dim=(-1, -2))
        roi_feats = self.local_roi_query_fc(roi_feats).view(bs, num_queries, hidden_dim)
        gate = torch.sigmoid(
            self.local_roi_query_gate(torch.cat([query_features, roi_feats], dim=-1))
        )
        return (1.0 - gate) * query_features + gate * roi_feats

    def _extract_roi_visual_features(self, feature_map, pred_boxes, output_size=7):
        """Pure visual ROI features from backbone feature map for class-agnostic quality.

        Returns [bs, num_queries, 1] raw logits (NOT sigmoided), or None if disabled.
        """
        if (not self.use_visual_confidence
            or feature_map is None
            or pred_boxes is None
            or self.visual_confidence_roi_proj is None):
            return None

        bs, num_queries, _ = pred_boxes.shape
        _, _, feat_h, feat_w = feature_map.shape
        if bs == 0 or num_queries == 0 or feat_h <= 0 or feat_w <= 0:
            return None

        roi_boxes = pred_boxes.detach().clamp(0.0, 1.0)
        roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes)
        scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
        roi_boxes_xyxy = roi_boxes_xyxy * scale
        roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(
            0.0, float(feat_w - 1))
        roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(
            0.0, float(feat_h - 1))
        roi_boxes_xyxy[..., 2] = torch.maximum(
            roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
        ).clamp(max=float(feat_w))
        roi_boxes_xyxy[..., 3] = torch.maximum(
            roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
        ).clamp(max=float(feat_h))

        batch_indices = torch.arange(
            bs, device=pred_boxes.device, dtype=roi_boxes_xyxy.dtype
        )[:, None].expand(bs, num_queries).reshape(-1, 1)
        flat_rois = torch.cat([batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1)

        roi_feats = roi_align(
            feature_map,
            flat_rois,
            output_size=output_size,
            spatial_scale=1.0,
            aligned=True,
        )
        # Mean-pool spatial dims: [bs*nq, 256, 7, 7] → [bs*nq, 256]
        roi_feats = roi_feats.mean(dim=(-1, -2))
        visual_scores = self.visual_confidence_roi_proj(roi_feats)
        visual_scores = torch.nan_to_num(
            visual_scores, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        return visual_scores.view(bs, num_queries, 1)

    def _compute_appearance_gate_scores(
        self, feature_map, pred_boxes, appearance_text_dict,
        appearance_cate_to_token_mask_list, output_size=7,
        decoder_query=None,
    ):
        """Class-wise appearance compatibility × appearance prototypes.

        When decoder_query is provided (feature_source="decoder"), uses decoder query
        features directly — they have attended to text tokens via cross-attention.
        Otherwise ROI Align on backbone feature map.

        Returns [bs, num_queries, num_classes] raw similarity logits, or None if disabled.
        """
        if (not self.use_appearance_conditioned_gate
            or self.appearance_gate_roi_proj is None
            or appearance_text_dict is None
            or appearance_cate_to_token_mask_list is None):
            return None

        if decoder_query is not None:
            bs, num_queries, hidden_dim = decoder_query.shape
            if bs == 0 or num_queries == 0:
                return None
            vis_feats = decoder_query.reshape(bs * num_queries, hidden_dim)
        else:
            if feature_map is None or pred_boxes is None:
                return None
            bs, num_queries, _ = pred_boxes.shape
            _, _, feat_h, feat_w = feature_map.shape
            if bs == 0 or num_queries == 0 or feat_h <= 0 or feat_w <= 0:
                return None

            # ---- ROI Align on backbone feature map ----
            roi_boxes = pred_boxes.detach().clamp(0.0, 1.0)
            roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes)
            scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
            roi_boxes_xyxy = roi_boxes_xyxy * scale
            roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(
                0.0, float(feat_w - 1))
            roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(
                0.0, float(feat_h - 1))
            roi_boxes_xyxy[..., 2] = torch.maximum(
                roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
            ).clamp(max=float(feat_w))
            roi_boxes_xyxy[..., 3] = torch.maximum(
                roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
            ).clamp(max=float(feat_h))

            batch_indices = torch.arange(
                bs, device=pred_boxes.device, dtype=roi_boxes_xyxy.dtype
            )[:, None].expand(bs, num_queries).reshape(-1, 1)
            flat_rois = torch.cat([batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1)

            roi_feats = roi_align(
                feature_map, flat_rois, output_size=output_size,
                spatial_scale=1.0, aligned=True,
            )
            vis_feats = roi_feats.mean(dim=(-1, -2))  # [bs*nq, hidden_dim]

        # ---- Project + L2 normalize ----
        vis_projected = self.appearance_gate_roi_proj(vis_feats)
        vis_normed = F.normalize(vis_projected, dim=-1, eps=1e-6)

        # ---- Build appearance description text prototypes ----
        appearance_prototypes, _appearance_valid = build_category_text_prototypes(
            appearance_text_dict["encoded_text"],
            appearance_cate_to_token_mask_list,
            appearance_text_dict["text_token_mask"],
        )

        # ---- Cosine similarity per batch item ----
        num_classes = max(proto.shape[0] for proto in appearance_prototypes)
        device = vis_normed.device
        all_scores = vis_normed.new_zeros(bs * num_queries, num_classes)
        temp = self.appearance_gate_temperature.clamp(min=0.01, max=100.0)

        for batch_idx in range(bs):
            proto = appearance_prototypes[batch_idx]  # [n_cats, hidden_dim]
            if proto.shape[0] == 0:
                continue
            proto_normed = F.normalize(proto, dim=-1, eps=1e-6)
            q_slice = slice(batch_idx * num_queries, (batch_idx + 1) * num_queries)
            sim = torch.mm(vis_normed[q_slice], proto_normed.T) / temp
            all_scores[q_slice, :sim.shape[1]] = sim

        return all_scores.view(bs, num_queries, num_classes)

    def _build_appearance_token_branch_logits(
        self,
        query_features,
        appearance_text_dict,
        appearance_cate_to_token_mask_list,
        supervision_text_dict,
        supervision_cate_to_token_mask_list,
    ):
        if (
            not self.use_appearance_token_branch
            or self.appearance_token_query_proj is None
            or self.appearance_token_cross_attn is None
            or self.appearance_token_ffn is None
            or query_features is None
            or appearance_text_dict is None
            or supervision_text_dict is None
            or appearance_cate_to_token_mask_list is None
            or supervision_cate_to_token_mask_list is None
        ):
            return None

        # Detach text features to prevent the appearance contrastive loss
        # from corrupting the shared feat_map (text projection layer).
        # The feat_map is used by both semantic and appearance text encoding;
        # gradients flowing back through appearance prototypes would distort
        # semantic text representations and cause CE loss explosion.
        app_tokens = appearance_text_dict.get("encoded_text", None)
        app_token_mask = appearance_text_dict.get("text_token_mask", None)
        if app_tokens is None or app_token_mask is None:
            return None
        app_tokens = app_tokens.detach()
        app_token_mask = app_token_mask.detach()

        bs, num_queries, hidden_dim = query_features.shape
        if bs == 0 or num_queries == 0:
            return None

        query = self.appearance_token_query_proj(query_features)
        key_padding_mask = ~app_token_mask.to(dtype=torch.bool, device=query_features.device)
        attn_out, _ = self.appearance_token_cross_attn(
            query=query,
            key=app_tokens,
            value=app_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        app_query = self.appearance_token_query_norm(query_features + attn_out)
        app_query = self.appearance_token_ffn(app_query)
        app_query = F.normalize(app_query, dim=-1, eps=1e-6)

        # Build prototypes from the SAME detached text features used for
        # cross-attention, so the contrastive loss cannot reach feat_map via
        # the prototype path either.
        appearance_prototypes, appearance_valid_masks = build_category_text_prototypes(
            app_tokens,
            appearance_cate_to_token_mask_list,
            app_token_mask,
        )
        supervision_caption_lists = supervision_text_dict.get("caption_lists", None)
        if supervision_caption_lists is not None:
            semantic_pos_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict,
                supervision_caption_lists,
                self.max_text_len,
                query_features.device,
            )
        else:
            semantic_pos_maps = self._build_caption_positive_map_from_masks(
                supervision_cate_to_token_mask_list,
                self.max_text_len,
                query_features.device,
            )

        token_logits = query_features.new_zeros((bs, num_queries, self.max_text_len))
        class_scores = []
        temp = self.appearance_token_branch_temperature.clamp(min=0.01, max=100.0)

        for batch_idx in range(bs):
            proto = appearance_prototypes[batch_idx]
            valid_mask = appearance_valid_masks[batch_idx].to(
                device=query_features.device, dtype=torch.bool
            )
            sem_pos_map = semantic_pos_maps[batch_idx]
            class_count = min(proto.shape[0], valid_mask.shape[0], sem_pos_map.shape[0])
            if class_count <= 0:
                class_scores.append(query_features.new_zeros((num_queries, 0)))
                continue

            proto = proto[:class_count]
            valid_mask = valid_mask[:class_count]
            sem_pos_map = sem_pos_map[:class_count].to(
                device=query_features.device, dtype=query_features.dtype
            )
            if not bool(valid_mask.any()):
                class_scores.append(query_features.new_zeros((num_queries, class_count)))
                continue

            proto = F.normalize(proto, dim=-1, eps=1e-6)
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            score = app_query[batch_idx] @ proto.transpose(0, 1)
            score = torch.nan_to_num(score / temp, nan=0.0, posinf=20.0, neginf=-20.0)
            score = score.clamp(min=-20.0, max=20.0)
            score = score.masked_fill(~valid_mask.view(1, -1), 0.0)
            class_scores.append(score)

            if class_count > 0:
                token_logits[batch_idx] = score @ sem_pos_map

        return {
            "appearance_token_branch_logits": token_logits,
            "appearance_token_branch_class_scores": class_scores,
        }

    def _build_roi_appearance_branch(
        self,
        feature_map,
        pred_boxes,
        appearance_text_dict,
        appearance_cate_to_token_mask_list,
        supervision_text_dict,
        supervision_cate_to_token_mask_list,
    ):
        """ROI-based appearance branch: backbone visual features → appearance class scores.

        Key differences from the decoder-query-based token branch:
        1. Input is backbone ROI visual features, NOT decoder queries.
           Backbone features preserve fine-grained visual patterns (color, texture,
           border shape, symbols) that decoder cross-attention to semantic text
           tends to compress away.
        2. A lightweight spatial encoder (2-layer conv) refines ROI features
           before pooling, so spatial details (numbers, arrows, edge thickness)
           are not destroyed by aggressive mean-pooling.
        3. Learned attention pooling replaces mean-pooling — a trainable query
           attends to spatial positions, learning to focus on discriminative regions.
        4. All text features are detached to completely isolate the branch from
           the shared feat_map, preventing gradient interference with CE loss.
        """
        if (
            not self.use_roi_appearance_branch
            or feature_map is None
            or pred_boxes is None
            or self.roi_appearance_spatial_encoder is None
        ):
            return None

        # Detach feature_map to prevent appearance gradients from flowing
        # into input_proj or backbone.  Without this the main detector's
        # visual feature projections can shift, corrupting the classification
        # surface even when lr_backbone=0.0.
        feature_map = feature_map.detach()

        bs, num_queries, hidden_dim_box = pred_boxes.shape
        _, _, feat_h, feat_w = feature_map.shape
        if bs == 0 or num_queries == 0 or feat_h <= 0 or feat_w <= 0:
            return None

        # ---- 1. ROI Align on high-resolution backbone feature map ----
        roi_boxes = pred_boxes.detach().clamp(0.0, 1.0)
        roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes)
        scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
        roi_boxes_xyxy = roi_boxes_xyxy * scale
        roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(0.0, float(feat_w - 1))
        roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(0.0, float(feat_h - 1))
        roi_boxes_xyxy[..., 2] = torch.maximum(
            roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
        ).clamp(max=float(feat_w))
        roi_boxes_xyxy[..., 3] = torch.maximum(
            roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
        ).clamp(max=float(feat_h))

        batch_indices = torch.arange(bs, device=pred_boxes.device, dtype=roi_boxes_xyxy.dtype)
        batch_indices = batch_indices[:, None].expand(bs, num_queries).reshape(-1, 1)
        flat_rois = torch.cat([batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1)

        roi_feats = roi_align(
            feature_map, flat_rois,
            output_size=self.roi_appearance_output_size,
            spatial_scale=1.0, aligned=True,
        )
        # roi_feats: [bs * nq, hidden_dim, roi_h, roi_w]

        # ---- 2. Spatial encoder: refine before pooling ----
        roi_feats = self.roi_appearance_spatial_encoder(roi_feats)
        # [bs * nq, hidden_dim, roi_h, roi_w]

        # ---- 3. Flatten spatial dims → sequence, project ----
        roi_feats = roi_feats.flatten(2).transpose(1, 2)
        # [bs * nq, roi_h * roi_w, hidden_dim]
        roi_feats = self.roi_appearance_spatial_proj(roi_feats)
        roi_feats = F.relu(roi_feats)

        # ---- 4. Learned attention pooling over spatial positions ----
        pool_query = self.roi_appearance_pool_query.expand(bs * num_queries, -1, -1)
        pooled, _ = self.roi_appearance_pool_attn(
            query=pool_query, key=roi_feats, value=roi_feats, need_weights=False,
        )
        pooled = pooled.squeeze(1)  # [bs * nq, hidden_dim]
        visual_query = self.roi_appearance_query_proj(pooled)
        visual_query = visual_query.view(bs, num_queries, self.hidden_dim)

        # ---- 5. Cross-attn to appearance text (fully detached) ----
        app_tokens = appearance_text_dict.get("encoded_text", None)
        app_token_mask = appearance_text_dict.get("text_token_mask", None)
        if app_tokens is None or app_token_mask is None:
            return None
        app_tokens = app_tokens.detach()
        app_token_mask = app_token_mask.detach()

        key_padding_mask = ~app_token_mask.to(dtype=torch.bool, device=visual_query.device)
        attn_out, _ = self.roi_appearance_text_cross_attn(
            query=visual_query, key=app_tokens, value=app_tokens,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        app_query = self.roi_appearance_norm(visual_query + attn_out)
        app_query = self.roi_appearance_text_ffn(app_query)
        app_query = F.normalize(app_query, dim=-1, eps=1e-6)

        # ---- 6. Build prototypes from detached text ----
        appearance_prototypes, appearance_valid_masks = build_category_text_prototypes(
            app_tokens, appearance_cate_to_token_mask_list, app_token_mask,
        )
        supervision_caption_lists = supervision_text_dict.get("caption_lists", None)
        if supervision_caption_lists is not None:
            semantic_pos_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict, supervision_caption_lists,
                self.max_text_len, visual_query.device,
            )
        else:
            semantic_pos_maps = self._build_caption_positive_map_from_masks(
                supervision_cate_to_token_mask_list, self.max_text_len,
                visual_query.device,
            )

        # ---- 7. Compute class scores (same pattern as token branch) ----
        class_scores = []
        temp = self.roi_appearance_temperature.clamp(min=0.01, max=100.0)

        for batch_idx in range(bs):
            proto = appearance_prototypes[batch_idx]
            valid_mask = appearance_valid_masks[batch_idx].to(
                device=visual_query.device, dtype=torch.bool
            )
            sem_pos_map = semantic_pos_maps[batch_idx]
            class_count = min(proto.shape[0], valid_mask.shape[0], sem_pos_map.shape[0])
            if class_count <= 0:
                class_scores.append(visual_query.new_zeros((num_queries, 0)))
                continue

            proto = proto[:class_count]
            valid_mask = valid_mask[:class_count]
            sem_pos_map = sem_pos_map[:class_count].to(
                device=visual_query.device, dtype=visual_query.dtype
            )
            if not bool(valid_mask.any()):
                class_scores.append(visual_query.new_zeros((num_queries, class_count)))
                continue

            proto = F.normalize(proto, dim=-1, eps=1e-6)
            proto = torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)
            score = app_query[batch_idx] @ proto.transpose(0, 1)
            score = torch.nan_to_num(score / temp, nan=0.0, posinf=20.0, neginf=-20.0)
            score = score.clamp(min=-20.0, max=20.0)
            score = score.masked_fill(~valid_mask.view(1, -1), 0.0)
            class_scores.append(score)

        return {
            "roi_appearance_class_scores": class_scores,
        }

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    @torch.no_grad()
    def _run_facb_speedhr_inference(
        self,
        samples,
        pred_logits,
        pred_boxes,
        category_token_masks,
        caption_lists=None,
        supervision_text_dict=None,
        image_paths=None,
        query_states=None,
        semantic_override=None,
    ):
        head = getattr(self, "facb_speedhr_head", None)
        if (
            head is None
            or not bool(getattr(self, "enable_facb_speedhr_inference", False))
            or not bool(getattr(self, "facb_speedhr_candidate_matching", False))
        ):
            return None
        facet_bank, facet_valid = self._get_facb_speedhr_facet_prototype_bank(
            pred_logits.device
        )
        if facet_bank is None or facet_valid is None:
            return None
        use_shared_facets = bool(
            getattr(self, "facb_speedhr_shared_compositional_facets", False)
        )
        shared_bank = None
        shared_valid = None
        shared_class_targets = None
        shared_class_target_mask = None
        if use_shared_facets:
            shared_bank, shared_valid = (
                self._get_facb_speedhr_shared_facet_prototype_bank(
                    pred_logits.device
                )
            )
            shared_class_targets = self.facb_speedhr_shared_facet_class_targets
            shared_class_target_mask = (
                self.facb_speedhr_shared_facet_class_target_mask
            )
            if (
                shared_bank is None
                or shared_valid is None
                or shared_class_targets is None
            ):
                return None

        topk = max(int(getattr(self, "facb_speedhr_infer_candidate_topk", 10)), 2)
        max_rows = max(int(getattr(self, "facb_speedhr_infer_max_rows_per_image", 32)), 1)
        chunk_size = max(int(getattr(self, "facb_speedhr_infer_chunk_size", 16)), 1)
        crop_scale = float(getattr(self, "facb_speedhr_crop_scale", 1.2))
        min_semantic_score = float(
            getattr(self, "facb_speedhr_infer_min_semantic_score", -1.0)
        )
        max_semantic_score = float(
            getattr(self, "facb_speedhr_infer_max_semantic_score", -1.0)
        )
        allowed_ids = getattr(self, "facb_speedhr_infer_allowed_global_ids", None)
        if allowed_ids is not None:
            allowed_ids = torch.as_tensor(
                allowed_ids, device=pred_logits.device, dtype=torch.long
            ).view(-1)

        batch_rows = []
        query_rows = []
        candidate_rows = []
        roi_rows = []
        # ModernBERT tokenizes the separator in "label . label" as both `.`
        # and the whitespace-prefixed `Ġ.` token.  The legacy special-token
        # boundary masks therefore exposed only a prefix of the 232 categories
        # to FACB (27 classes in the TT100K prompt).  Build class/token maps from
        # deterministic character spans, matching the detector postprocessor.
        caption_positive_maps = None
        if supervision_text_dict is not None and caption_lists is not None:
            caption_positive_maps = self._build_caption_positive_map_from_text(
                supervision_text_dict,
                caption_lists,
                self.max_text_len,
                pred_logits.device,
            )
        for batch_idx, category_mask in enumerate(category_token_masks):
            token_dim = min(int(category_mask.shape[-1]), int(pred_logits.shape[-1]))
            if token_dim <= 0:
                continue
            if (
                caption_positive_maps is not None
                and batch_idx < len(caption_positive_maps)
                and caption_positive_maps[batch_idx] is not None
            ):
                category_weights = caption_positive_maps[batch_idx].to(
                    device=pred_logits.device, dtype=torch.float32
                )[:, :token_dim]
            else:
                category_weights = category_mask.to(
                    device=pred_logits.device, dtype=torch.float32
                )[:, :token_dim]
            valid_local = category_weights.sum(dim=-1) > 0
            num_local = int(category_weights.shape[0])
            if num_local < 2:
                continue
            category_weights = category_weights[:num_local]
            valid_local = valid_local[:num_local]

            # Evaluation splits the 232 labels into text-length-safe prompt chunks.
            # Keep semantic columns local, but map every selected candidate back to
            # its global TT100K id before indexing the global facet bank.
            local_to_global = torch.arange(
                num_local, device=pred_logits.device, dtype=torch.long
            )
            if caption_lists is not None and batch_idx < len(caption_lists):
                current_labels = list(caption_lists[batch_idx] or [])
                global_labels = list(
                    getattr(self, "facb_speedhr_label_texts", None) or []
                )
                if current_labels and global_labels:
                    def _label_key(value):
                        return " ".join(str(value).strip().lower().split())

                    current_keys = [
                        _label_key(label)
                        for label in current_labels[:num_local]
                    ]
                    global_keys = [_label_key(label) for label in global_labels]
                    mapped = None
                    for start in range(
                        max(len(global_keys) - len(current_keys) + 1, 0)
                    ):
                        if global_keys[start : start + len(current_keys)] == current_keys:
                            mapped = list(range(start, start + len(current_keys)))
                            break
                    if mapped is None:
                        # Fallback for non-contiguous custom prompts. Duplicate
                        # names remain unresolved here, but standard eval chunks
                        # always take contiguous slices and use the path above.
                        global_lookup = {}
                        for global_id, label in enumerate(global_labels):
                            global_lookup.setdefault(_label_key(label), global_id)
                        mapped = [
                            global_lookup.get(key, -1) for key in current_keys
                        ]
                    if len(mapped) < num_local:
                        mapped.extend([-1] * (num_local - len(mapped)))
                    local_to_global = torch.as_tensor(
                        mapped, device=pred_logits.device, dtype=torch.long
                    )
            weights = category_weights
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pred_token_scores = pred_logits[batch_idx, :, :token_dim].float().sigmoid()
            semantic = torch.nan_to_num(
                pred_token_scores @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            if semantic_override is not None:
                # Deployment-aligned pool-v2 selection must use the same class
                # scores that the detector postprocessor ranks.  Selecting rows
                # earlier from a separately reconstructed token map changes the
                # proposal population even when the detector weights are frozen.
                semantic = semantic_override[
                    batch_idx, :, local_to_global
                ].float().clone()
                # The override already contains PostProcess-validated class
                # columns.  Reapplying the model-side ModernBERT token mask here
                # incorrectly removes valid categories and changes the top-16
                # proposal population.
                valid_local = (local_to_global >= 0) & (
                    local_to_global < semantic_override.shape[-1]
                )
            elif getattr(self, "facb_proposal_pool_v2", False):
                # Collection hook for the original pool-v2 cache.
                semantic = self.facb_pool_v2_semantic_override[
                    batch_idx, :, local_to_global
                ].float().clone()
                valid_local = (local_to_global >= 0) & (
                    local_to_global < self.facb_pool_v2_semantic_override.shape[-1]
                )
            semantic[:, ~valid_local] = float("-inf")
            valid_count = int(valid_local.sum().item())
            if valid_count < 2:
                continue
            k = min(topk, valid_count)
            top = torch.topk(semantic, k=k, dim=-1)
            candidate_local_ids = top.indices
            candidate_ids = local_to_global[candidate_local_ids]
            if allowed_ids is not None and allowed_ids.numel() > 0:
                eligible = (
                    candidate_ids[:, :, None] == allowed_ids[None, None, :]
                ).any(dim=(1, 2))
                allowed_local_mask = (
                    local_to_global[:, None] == allowed_ids[None, :]
                ).any(dim=1)
                allowed_local = torch.nonzero(
                    allowed_local_mask, as_tuple=False
                ).flatten()
                allowed_score = semantic[:, allowed_local]
                if allowed_score.numel() == 0:
                    continue
                rank_score = allowed_score.max(dim=-1).values
            else:
                eligible = torch.ones(
                    semantic.shape[0], device=semantic.device, dtype=torch.bool
                )
                rank_score = top.values[:, 0]
            chunk_top_score = top.values[:, 0]
            if min_semantic_score >= 0.0:
                eligible = eligible & (chunk_top_score >= min_semantic_score)
            if max_semantic_score >= 0.0:
                eligible = eligible & (chunk_top_score <= max_semantic_score)
            eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
            if eligible_idx.numel() == 0:
                continue
            keep_count = min(max_rows, int(eligible_idx.numel()))
            keep_order = torch.topk(
                rank_score[eligible_idx], k=keep_count, dim=0
            ).indices
            query_idx = eligible_idx[keep_order]
            if os.environ.get("GROUNDINGDINO_FACB_PROPOSAL_DEBUG", "0") == "1":
                expected_query_idx = semantic.max(dim=-1).values.topk(
                    k=min(max_rows, semantic.shape[0])
                ).indices
                print(
                    "[facb-speedhr][proposal-debug] "
                    f"override={semantic_override is not None} "
                    f"expected={expected_query_idx.detach().cpu().tolist()} "
                    f"selected={query_idx.detach().cpu().tolist()} "
                    f"selected_scores={rank_score[query_idx].detach().cpu().tolist()}",
                    flush=True,
                )
            candidate_ids = candidate_ids[query_idx]
            fixed_ids = getattr(self, "facb_speedhr_fixed_candidate_ids", [])
            candidate_policy = str(
                getattr(self, "facb_speedhr_candidate_policy", "fixed_bank_plus_outside_topk_v1")
            ).strip().lower()
            if fixed_ids:
                if candidate_policy in {
                    "fixed36_union_natural_top10_v2",
                    "fixed_pool_union_natural_topk_v2",
                }:
                    candidate_ids = union_candidates(
                        semantic[query_idx], local_to_global, fixed_ids, topk
                    )
                else:
                    candidate_ids = fixed_bank_candidates(
                        semantic[query_idx], local_to_global, fixed_ids, topk
                    )

            valid_global_candidates = (
                (candidate_ids >= 0) & (candidate_ids < facet_bank.shape[0])
            )
            if not bool(valid_global_candidates.all()):
                keep_rows = valid_global_candidates.all(dim=1)
                query_idx = query_idx[keep_rows]
                candidate_ids = candidate_ids[keep_rows]
                if query_idx.numel() == 0:
                    continue

            boxes = pred_boxes[batch_idx, query_idx].detach().float().clone()
            boxes[:, 2:] = (boxes[:, 2:] * crop_scale).clamp(
                min=1e-6, max=1.0
            )
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
            if samples.mask is not None:
                valid_pixels = ~samples.mask[batch_idx]
                valid_h = max(int(valid_pixels.any(dim=1).sum().item()), 1)
                valid_w = max(int(valid_pixels.any(dim=0).sum().item()), 1)
            else:
                valid_h = int(samples.tensors.shape[-2])
                valid_w = int(samples.tensors.shape[-1])
            scale = boxes_xyxy.new_tensor(
                [float(valid_w), float(valid_h), float(valid_w), float(valid_h)]
            )
            boxes_xyxy = boxes_xyxy * scale
            boxes_xyxy[:, 2] = torch.maximum(
                boxes_xyxy[:, 2], boxes_xyxy[:, 0] + 1.0
            )
            boxes_xyxy[:, 3] = torch.maximum(
                boxes_xyxy[:, 3], boxes_xyxy[:, 1] + 1.0
            )
            rois = torch.cat(
                [
                    boxes_xyxy.new_full((boxes_xyxy.shape[0], 1), float(batch_idx)),
                    boxes_xyxy,
                ],
                dim=1,
            )
            batch_rows.append(
                torch.full_like(query_idx, batch_idx, dtype=torch.long)
            )
            query_rows.append(query_idx)
            candidate_rows.append(candidate_ids)
            roi_rows.append(rois)

        if not roi_rows:
            return None
        batch_ids = torch.cat(batch_rows, dim=0)
        query_ids = torch.cat(query_rows, dim=0)
        candidate_ids = torch.cat(candidate_rows, dim=0)
        rois = torch.cat(roi_rows, dim=0)
        if use_shared_facets:
            candidate_text = facet_bank[candidate_ids, 0]
            candidate_valid = facet_valid[candidate_ids, 0]
            candidate_shared_targets = shared_class_targets[candidate_ids]
            candidate_shared_target_mask = shared_class_target_mask[candidate_ids]
        else:
            candidate_text = facet_bank[candidate_ids]
            candidate_valid = facet_valid[candidate_ids]
            candidate_shared_targets = None
            candidate_shared_target_mask = None
        candidate_policy = str(
            getattr(self, "facb_speedhr_candidate_policy", "fixed_bank_plus_outside_topk_v1")
        ).strip().lower()
        if getattr(self, "facb_proposal_pool_v2", False) or candidate_policy in {
            "fixed36_union_natural_top10_v2",
            "fixed_pool_union_natural_topk_v2",
        }:
            # Padded duplicate IDs are never valid attention keys.
            from util.facb_pool_v2 import unique_candidate_mask
            pool_mask = unique_candidate_mask(candidate_ids)
            candidate_valid = candidate_valid & pool_mask.reshape(
                *pool_mask.shape, *([1] * (candidate_valid.ndim - 2)))
        logits_chunks = []
        facet_score_chunks = []
        state_chunks, crop_state_chunks, text_state_chunks = [], [], []
        export_states = bool(getattr(self, "facb_speedhr_export_states", False))
        pixel_source = getattr(self, "facb_speedhr_infer_pixel_source", "tensor")
        head_images = samples.tensors
        if pixel_source == "raw_image":
            head_images, rois = raw_predicted_crops(
                image_paths, batch_ids, pred_boxes[batch_ids, query_ids],
                crop_scale, int(head.crop_size), pred_logits.device,
            )
        for start in range(0, rois.shape[0], chunk_size):
            stop = min(start + chunk_size, rois.shape[0])
            head_out = head(
                head_images,
                rois[start:stop],
                return_states=export_states,
                candidate_text=candidate_text[start:stop],
                candidate_mask=candidate_valid[start:stop],
                shared_facet_prototypes=shared_bank,
                shared_facet_valid=shared_valid,
                candidate_shared_targets=(
                    candidate_shared_targets[start:stop]
                    if candidate_shared_targets is not None
                    else None
                ),
                candidate_shared_target_mask=(
                    candidate_shared_target_mask[start:stop]
                    if candidate_shared_target_mask is not None
                    else None
                ),
            )
            chunk_facet_scores = None
            if isinstance(head_out, tuple):
                head_out = head_out[0]
            elif isinstance(head_out, dict):
                if export_states:
                    state_chunks.append(head_out["candidate_states"].float())
                    text_state_chunks.append(head_out["candidate_text_states"].float())
                    crop_state_chunks.append(head_out["crop_state"].float())
                chunk_facet_scores = head_out.get("candidate_facet_scores", None)
                head_out = head_out["candidate_logits"]
            logits_chunks.append(head_out.float())
            if isinstance(chunk_facet_scores, torch.Tensor):
                facet_score_chunks.append(chunk_facet_scores.float())
        candidate_logits = torch.cat(logits_chunks, dim=0)
        candidate_facet_scores = None
        if facet_score_chunks:
            candidate_facet_scores = torch.cat(facet_score_chunks, dim=0)
        # Do not apply the small-shot scope using the classifier winner here.
        # Composition modes may select a different slot later.  Filtering at
        # this point made the proposal population depend on candidate-head
        # argmax while the actual write used the composition argmax.  The final
        # selected global id is scope-checked in _apply_facb_speedhr_inference.
        proposal_protocol = {
            "schema": SCHEMA, "pixel_source": pixel_source,
            **({"candidate_policy": candidate_policy,
                "fixed_candidate_ids": list(self.facb_speedhr_fixed_candidate_ids)}
               if getattr(self, "facb_speedhr_fixed_candidate_ids", []) else {}),
            "topk": topk, "max_rows_per_chunk": max_rows,
            "crop_scale": crop_scale, "crop_size": int(head.crop_size),
            "min_semantic_score": min_semantic_score,
            "max_semantic_score": max_semantic_score,
        }
        if semantic_override is not None or getattr(self, "facb_proposal_pool_v2", False):
            proposal_protocol["selection_score_source"] = "postprocess_prob_to_label"
        return {
            "batch_indices": batch_ids,
            "query_indices": query_ids,
            "candidate_ids": candidate_ids,
            "candidate_logits": candidate_logits,
            "candidate_facet_scores": candidate_facet_scores,
            "candidate_mask": candidate_valid,
            "candidate_states": torch.cat(state_chunks) if state_chunks else None,
            "candidate_text_states": torch.cat(text_state_chunks) if text_state_chunks else None,
            "crop_states": torch.cat(crop_state_chunks) if crop_state_chunks else None,
            "query_states": query_states[batch_ids, query_ids].float() if export_states else None,
            "proposal_protocol": proposal_protocol,
            # PostProcess owns the final composition winner selection, so pass
            # the deployment scope along with the proposal tensors.  Keeping
            # this only on the model object silently lost the scope at the
            # model/PostProcess boundary.
            "allowed_global_ids": allowed_ids,
        }

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if "captions" in kw:
            captions = kw["captions"]
        elif targets is not None:
            captions = [t["caption"] for t in targets]
        else:
            captions = kw["captions"]
        supervision_captions = kw.get("supervision_captions", captions)
        supervision_caption_lists_kw = kw.get("supervision_caption_lists", None)
        query_contrastive_captions = kw.get("query_contrastive_captions", supervision_captions)
        appearance_captions = kw.get("appearance_captions", supervision_captions)
        appearance_caption_label_ids = kw.get("appearance_caption_label_ids", None)

        text_dict, one_hot_token, cate_to_token_mask_list = self._encode_captions(
            captions, samples.device
        )
        supervision_text_dict = text_dict
        supervision_token = one_hot_token
        supervision_cate_to_token_mask_list = cate_to_token_mask_list
        query_contrastive_text_dict = supervision_text_dict
        query_contrastive_token = supervision_token
        query_contrastive_cate_to_token_mask_list = supervision_cate_to_token_mask_list
        use_dual_prompt_supervision = supervision_captions != captions
        if supervision_captions != captions:
            supervision_text_dict, supervision_token, supervision_cate_to_token_mask_list = self._encode_captions(
                supervision_captions, samples.device
            )
        supervision_caption_lists = supervision_caption_lists_kw
        if supervision_caption_lists is None and targets is not None:
            supervision_caption_lists = [t["cap_list"] for t in targets]
        appearance_caption_lists = (
            [t.get("appearance_roi_cap_list", t["cap_list"]) for t in targets]
            if targets is not None else None
        )
        if supervision_caption_lists is not None:
            supervision_text_dict["caption_lists"] = supervision_caption_lists
        # caption-local index -> global class id for the SUPERVISION caption
        # (== per-image sampled vg_labels).  Needed by the local-evidence path to map
        # detector candidate ranking (caption-local) to appearance bank rows (global).
        if targets is not None and len(targets) > 0 and "caption_label_ids" in targets[0]:
            supervision_text_dict["caption_label_ids"] = [
                t.get("caption_label_ids", None) for t in targets
            ]
        if query_contrastive_captions != supervision_captions:
            (
                query_contrastive_text_dict,
                query_contrastive_token,
                query_contrastive_cate_to_token_mask_list,
            ) = self._encode_captions(query_contrastive_captions, samples.device)
        else:
            query_contrastive_text_dict = supervision_text_dict
            query_contrastive_token = supervision_token
            query_contrastive_cate_to_token_mask_list = supervision_cate_to_token_mask_list
        appearance_text_dict = supervision_text_dict
        appearance_token = supervision_token
        appearance_cate_to_token_mask_list = supervision_cate_to_token_mask_list
        if appearance_captions != supervision_captions:
            (
                appearance_text_dict,
                appearance_token,
                appearance_cate_to_token_mask_list,
            ) = self._encode_captions(appearance_captions, samples.device)
        if appearance_caption_lists is not None:
            appearance_text_dict["caption_lists"] = appearance_caption_lists
        if appearance_caption_label_ids is not None:
            appearance_text_dict["caption_label_ids"] = appearance_caption_label_ids
        if (
            self.training
            and not self._prompt_debug_printed
            and os.environ.get("GROUNDINGDINO_PROMPT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        ):
            print("[prompt-debug] forward_caption:", captions[0][:1000] if captions else "")
            print(
                "[prompt-debug] supervision_caption:",
                supervision_captions[0][:1000] if supervision_captions else "",
            )
            print(
                "[prompt-debug] text_mask shapes:",
                tuple(text_dict["text_token_mask"].shape),
                tuple(supervision_text_dict["text_token_mask"].shape),
            )
            print(
                "[prompt-debug] token counts:",
                int(text_dict["text_token_mask"][0].sum().item()) if text_dict["text_token_mask"].numel() else 0,
                int(supervision_text_dict["text_token_mask"][0].sum().item()) if supervision_text_dict["text_token_mask"].numel() else 0,
            )
            self._prompt_debug_printed = True


        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        lgpr_box_refine_base = None
        lgpr_box_refine_delta = None
        if (
            self.use_lgpr_box_refine
            and self.lgpr_box_refine_head is not None
            and (self.training or self.use_lgpr_box_refine_inference)
        ):
            lgpr_box_refine_base = outputs_coord_list[-1]
            lgpr_box_refine_delta = torch.tanh(
                self.lgpr_box_refine_head(hs[-1].detach())
            )
            refined_unsig = (
                inverse_sigmoid(lgpr_box_refine_base.detach())
                + self.lgpr_box_refine_scale * lgpr_box_refine_delta
            )
            refined_boxes = refined_unsig.sigmoid()
            outputs_coord_list = outputs_coord_list.clone()
            outputs_coord_list[-1] = refined_boxes

        lgpr_survival_logits_all = None
        if self.use_lgpr_survival and self.lgpr_survival_head is not None:
            lgpr_survival_logits_all = torch.stack(
                [self.lgpr_survival_head(layer_hs.detach()) for layer_hs in hs]
            )

        fused_last_query = hs[-1]
        if self.use_local_roi_query_fusion:
            fused_last_query = self._fuse_local_roi_query_features(
                query_features=hs[-1],
                pred_boxes=outputs_coord_list[-1],
                feature_map=srcs[0] if srcs else None,
            )

        # Refine predicted boxes with GT boxes for appearance ROI Align during training.
        roi_boxes = outputs_coord_list[-1]
        gt_targets_for_roi = kw.get("gt_targets_for_roi", None)
        # Per-query GT label (CAPTION-LOCAL index, == target["labels"], NOT global)
        # for queries whose box matches a GT (IoU > 0.5).  -1 = no GT match.  Used by
        # the local-evidence path to inject the positive into each query's candidate
        # set (which is also caption-local) for listwise CE.
        roi_query_gt_labels = None
        if self.training and gt_targets_for_roi is not None:
            roi_boxes = roi_boxes.clone()
            _bs_q = roi_boxes.shape[0]
            roi_query_gt_labels = torch.full(
                (_bs_q, roi_boxes.shape[1]), -1, dtype=torch.long, device=roi_boxes.device
            )
            for b_idx, gt_tgt in enumerate(gt_targets_for_roi):
                gt_boxes = gt_tgt.get("boxes", None)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue
                pred_xyxy = box_ops.box_cxcywh_to_xyxy(roi_boxes[b_idx])
                gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes.to(roi_boxes.device))
                iou = box_ops.box_iou(pred_xyxy, gt_xyxy)[0]
                best_iou, best_idx = iou.max(dim=1)
                mask = best_iou > 0.5
                if mask.any():
                    roi_boxes[b_idx, mask] = gt_boxes[best_idx[mask]].to(dtype=roi_boxes.dtype)
                    gt_labels = gt_tgt.get("labels", None)
                    if gt_labels is not None and gt_labels.numel() > 0:
                        gt_labels = gt_labels.to(device=roi_boxes.device, dtype=torch.long)
                        roi_query_gt_labels[b_idx, mask] = gt_labels[best_idx[mask]]

        class_embed_layers = self.class_embed
        if (
            self.dual_prompt_class_embed is not None
            and use_dual_prompt_supervision
        ):
            class_embed_layers = self.dual_prompt_class_embed

        semantic_pred_logits_for_appearance = None
        if self.use_appearance_roi_fusion and self.appearance_roi_semantic_topk > 0:
            semantic_pred_logits_for_appearance = class_embed_layers[-1](
                fused_last_query, supervision_text_dict
            )

        # Feature-level appearance fusion: enrich query features before class_embed
        appearance_fusion_outputs = self._build_appearance_fusion_outputs(
            query_features=fused_last_query,
            pred_boxes=roi_boxes,
            feature_map=srcs[0] if srcs else None,
            semantic_pred_logits=semantic_pred_logits_for_appearance,
            supervision_text_dict=supervision_text_dict,
            appearance_text_dict=appearance_text_dict,
            appearance_cate_to_token_mask_list=appearance_cate_to_token_mask_list,
            supervision_cate_to_token_mask_list=supervision_cate_to_token_mask_list,
        )
        if appearance_fusion_outputs is not None and self.use_feature_level_fusion:
            app_query = appearance_fusion_outputs["app_query"]
            appearance_alpha = appearance_fusion_outputs["appearance_alpha"]
            fused_last_query = fused_last_query + appearance_alpha * app_query

        class_hs = list(hs[:-1]) + [fused_last_query]
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, supervision_text_dict)
                for layer_cls_embed, layer_hs in zip(class_embed_layers, class_hs)
            ]
        )
        pred_logits = outputs_class[-1]
        baseline_last_logits = pred_logits
        if self.use_sail_last_class_embed:
            sail_category_masks = supervision_cate_to_token_mask_list
            if supervision_caption_lists is not None:
                sail_category_masks = self._build_caption_positive_map_from_text(
                    supervision_text_dict,
                    supervision_caption_lists,
                    self.max_text_len,
                    fused_last_query.device,
                )
            sail_last_logits = self.sail_last_class_embed(
                fused_last_query,
                supervision_text_dict,
                sail_category_masks,
            )
            sail_category_logits = self.sail_last_class_embed.latest_gate_cache.get(
                "category_logits_tensor", None
            )
            sail_category_token_gate = (
                self.sail_last_class_embed.latest_gate_cache.get(
                    "category_token_gate_tensor", None
                )
            )
            if self.sail_last_class_embed_use_main_token_blend:
                pred_logits = self.sail_last_class_embed.blend_with_baseline(
                    sail_last_logits,
                    baseline_last_logits,
                    supervision_text_dict["text_token_mask"],
                    query_features=fused_last_query,
                    category_token_masks=sail_category_masks,
                )
            else:
                pred_logits = baseline_last_logits
            out = {
                "pred_logits_semantic": pred_logits,
                "pred_boxes": outputs_coord_list[-1],
                "pred_logits_dgcl": pred_logits,
                "pred_logits_dgcl_raw": sail_last_logits,
                "pred_logits_baseline": baseline_last_logits,
            }
            if sail_category_logits is not None:
                out["pred_logits_dgcl_category_raw"] = sail_category_logits
            if sail_category_token_gate is not None:
                out["pred_logits_dgcl_token_gate"] = sail_category_token_gate
        else:
            out = {
                "pred_logits_semantic": pred_logits,
                "pred_boxes": outputs_coord_list[-1],
            }
        if lgpr_box_refine_base is not None:
            out["pred_boxes_lgpr_base"] = lgpr_box_refine_base.detach()
            out["lgpr_box_refine_delta"] = lgpr_box_refine_delta
        if self.training and self.use_global_image_text_sigmoid_loss:
            global_alignment_outputs = self._build_global_image_text_alignment_outputs(
                query_features=fused_last_query,
                text_dict=supervision_text_dict,
            )
            if global_alignment_outputs is not None:
                out.update(global_alignment_outputs)
        if appearance_fusion_outputs is not None:
            if self.use_feature_level_fusion:
                appearance_token_logits = appearance_fusion_outputs["appearance_token_logits"]
                appearance_alpha = appearance_fusion_outputs["appearance_alpha"]
                pred_logits = pred_logits + appearance_alpha * appearance_token_logits
                out["pred_logits_appearance"] = appearance_token_logits
                out["appearance_alpha"] = appearance_alpha
            out["appearance_class_scores"] = appearance_fusion_outputs["appearance_class_scores"]
            if "appearance_text_logits" in appearance_fusion_outputs:
                out["appearance_text_logits"] = appearance_fusion_outputs["appearance_text_logits"]
            if "appearance_caption_label_ids" in appearance_fusion_outputs:
                out["appearance_caption_label_ids"] = appearance_fusion_outputs["appearance_caption_label_ids"]
            if "proto_cls_scores" in appearance_fusion_outputs:
                out["proto_cls_scores"] = appearance_fusion_outputs["proto_cls_scores"]

        if self.use_appearance_token_branch:
            # Detach query_features so the appearance contrastive loss trains
            # only the appearance-branch parameters (projection / cross-attn / FFN),
            # NOT the main decoder.  Without detach the gradient flows back into
            # the decoder and corrupts the semantic CE alignment — the same problem
            # as direct logit injection, but through the gradient path.
            appearance_token_branch_outputs = self._build_appearance_token_branch_logits(
                query_features=fused_last_query.detach(),
                appearance_text_dict=appearance_text_dict,
                appearance_cate_to_token_mask_list=appearance_cate_to_token_mask_list,
                supervision_text_dict=supervision_text_dict,
                supervision_cate_to_token_mask_list=supervision_cate_to_token_mask_list,
            )
            if appearance_token_branch_outputs is not None:
                appearance_token_branch_logits = appearance_token_branch_outputs[
                    "appearance_token_branch_logits"
                ]
                appearance_token_branch_class_scores = appearance_token_branch_outputs[
                    "appearance_token_branch_class_scores"
                ]
                appearance_token_branch_scale = torch.sigmoid(
                    self.appearance_token_branch_logit_scale
                )
                appearance_token_branch_logits = (
                    appearance_token_branch_logits * appearance_token_branch_scale
                )
                out["pred_logits_appearance_token_branch"] = appearance_token_branch_logits
                out["appearance_token_branch_class_scores"] = appearance_token_branch_class_scores
                if "appearance_class_scores" not in out:
                    out["appearance_class_scores"] = appearance_token_branch_class_scores
                if "appearance_caption_label_ids" not in out:
                    out["appearance_caption_label_ids"] = appearance_text_dict.get(
                        "caption_label_ids", None
                    )
                out["appearance_token_branch_scale"] = appearance_token_branch_scale.detach()
                out["appearance_token_branch_weight"] = torch.as_tensor(
                    self.appearance_token_branch_weight,
                    device=pred_logits.device,
                    dtype=pred_logits.dtype,
                )

        if self.use_roi_appearance_branch:
            # ROI-based appearance branch: backbone visual features → class scores.
            # Fully independent from decoder — no shared parameters, no gradient
            # interference with CE loss.  Uses GT-refined boxes during training.
            roi_appearance_outputs = self._build_roi_appearance_branch(
                feature_map=srcs[0] if srcs else None,
                pred_boxes=roi_boxes,  # already GT-refined for matched queries
                appearance_text_dict=appearance_text_dict,
                appearance_cate_to_token_mask_list=appearance_cate_to_token_mask_list,
                supervision_text_dict=supervision_text_dict,
                supervision_cate_to_token_mask_list=supervision_cate_to_token_mask_list,
            )
            if roi_appearance_outputs is not None:
                out["appearance_class_scores"] = roi_appearance_outputs[
                    "roi_appearance_class_scores"
                ]
                out["appearance_caption_label_ids"] = appearance_text_dict.get(
                    "caption_label_ids", None
                )
                out["roi_appearance_logit_scale"] = torch.sigmoid(
                    self.roi_appearance_logit_scale
                ).detach()

        # ---- Mirrored ContrastiveEmbed appearance head ----
        # Same architecture as self.class_embed, but reads appearance descriptions
        # instead of semantic labels. Decoder query is detached so contrastive loss
        # does not propagate into the main detection path.
        if self.appearance_contrastive_head is not None:
            # Feature source for appearance head.
            # "decoder_query" (default): fused_last_query.detach() — same as v1/v2.
            # "decoder_layer_N": hs[N].detach() — middle layers may retain more
            #     visual detail before semantic classification shapes the query.
            #     e.g. "decoder_layer_2" uses the 3rd decoder layer output.
            # "roi": backbone ROI Align — raw visual features, no text interaction.
            app_feature_source = str(
                getattr(self, "appearance_contrastive_head_feature_source", "decoder_query")
            )
            # ---- Candidate-conditioned local evidence path ----
            # Replaces ROI->single-vector->bank-matching.  Keeps multi-scale ROI
            # spatial tokens and lets each candidate prototype read the region it
            # cares about, scoring ONLY over a small per-query candidate set
            # (semantic top-M, GT injected at train).  Output: out["appearance_local_*"].
            use_local_evidence = (
                bool(getattr(self, "use_appearance_local_evidence", False))
                and self.appearance_local_evidence_head is not None
                and srcs is not None and len(srcs) > 0
            )
            if use_local_evidence:
                self._forward_appearance_local_evidence(
                    out=out,
                    srcs=srcs,
                    roi_boxes=roi_boxes,
                    fused_last_query=fused_last_query,
                    class_embed_layers=class_embed_layers,
                    roi_query_gt_labels=roi_query_gt_labels,
                    pred_logits=pred_logits,
                    supervision_text_dict=supervision_text_dict,
                    supervision_cate_to_token_mask_list=supervision_cate_to_token_mask_list,
                )
            elif app_feature_source in ("roi", "roi_cross_attn"):
                # Extract ROI visual features from backbone feature map.
                # "roi": plain mean-pooled ROI features (no cross-attn).
                # "roi_cross_attn": ROI features attend over prototype bank so the
                #   visual query is aligned to appearance descriptions before being
                #   scored — avoids the modality gap from decoder_query.
                feature_map = srcs[0] if srcs else None
                if feature_map is None:
                    app_visual_features = fused_last_query.detach()
                else:
                    _, _, feat_h, feat_w = feature_map.shape
                    bs_fm, nq, _ = roi_boxes.shape
                    roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
                        roi_boxes.detach().clamp(0.0, 1.0)
                    )
                    scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
                    roi_boxes_xyxy = roi_boxes_xyxy * scale
                    roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(0.0, float(feat_w - 1))
                    roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(0.0, float(feat_h - 1))
                    roi_boxes_xyxy[..., 2] = torch.maximum(
                        roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
                    ).clamp(max=float(feat_w))
                    roi_boxes_xyxy[..., 3] = torch.maximum(
                        roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
                    ).clamp(max=float(feat_h))
                    batch_indices = torch.arange(
                        bs_fm, device=roi_boxes.device, dtype=roi_boxes_xyxy.dtype
                    )[:, None].expand(bs_fm, nq).reshape(-1, 1)
                    flat_rois = torch.cat(
                        [batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1
                    )
                    feature_map_for_app = (
                        feature_map
                        if bool(getattr(self, "appearance_gradient_to_backbone", False))
                        else feature_map.detach()
                    )
                    roi_feats = roi_align(
                        feature_map_for_app, flat_rois,
                        output_size=14, spatial_scale=1.0, aligned=True,
                    )
                    # Spatial attention pooling: [bs*nq, D, 14, 14] → [bs, nq, D].
                    # Linear(D,1)+softmax over 196 spatial tokens so the branch
                    # can focus on the digit/marker region rather than averaging.
                    bs_nq = bs_fm * nq
                    roi_spatial = roi_feats.flatten(2).permute(0, 2, 1)  # [bs*nq, 196, D]
                    if self.appearance_roi_spatial_attn is not None:
                        attn_w = self.appearance_roi_spatial_attn(roi_spatial)   # [bs*nq, 196, 1]
                        attn_w = torch.softmax(attn_w, dim=1)
                        roi_pooled = (attn_w * roi_spatial).sum(dim=1)           # [bs*nq, D]
                    else:
                        roi_pooled = roi_spatial.mean(dim=1)
                    roi_pooled = roi_pooled.view(bs_fm, nq, self.hidden_dim)
                    if app_feature_source == "roi_cross_attn":
                        # Cross-attend: ROI features (query) over prototype bank
                        # (key/value).  Bank is only available once built; fall back
                        # to plain roi_pooled if not ready yet.
                        _bank_ca, _bank_ca_mask = self._get_appearance_prototype_bank(
                            roi_pooled.device
                        )
                        if (
                            _bank_ca is not None
                            and self.appearance_roi_ca_attn is not None
                        ):
                            _bank_ca = _bank_ca.to(dtype=roi_pooled.dtype)
                            # bank: [C, D] → [1, C, D] → [bs*nq, C, D] not needed;
                            # use batch_first MHA with query=[bs,nq,D], kv=[bs,C,D]
                            _bank_kv = _bank_ca.unsqueeze(0).expand(bs_fm, -1, -1)
                            if _bank_ca_mask is not None:
                                # key_padding_mask: True = ignore; shape [bs, C]
                                _kpm = (~_bank_ca_mask).unsqueeze(0).expand(bs_fm, -1)
                            else:
                                _kpm = None
                            _ca_out, _ = self.appearance_roi_ca_attn(
                                query=roi_pooled,
                                key=_bank_kv,
                                value=_bank_kv,
                                key_padding_mask=_kpm,
                                need_weights=False,
                            )
                            roi_pooled = self.appearance_roi_ca_norm(
                                roi_pooled + _ca_out
                            )
                            roi_pooled = self.appearance_roi_ca_proj(roi_pooled)
                    app_visual_features = roi_pooled
            elif app_feature_source == "concat":
                # Decoder query (alignment) + ROI (visual detail).
                # A single Linear projects the concatenated features back
                # to hidden_dim before feeding into the shared ContrastiveEmbed.
                feature_map = srcs[0] if srcs else None
                if feature_map is not None:
                    _, _, feat_h, feat_w = feature_map.shape
                    bs_fm, nq, _ = roi_boxes.shape
                    roi_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
                        roi_boxes.detach().clamp(0.0, 1.0)
                    )
                    scale = roi_boxes_xyxy.new_tensor([feat_w, feat_h, feat_w, feat_h])
                    roi_boxes_xyxy = roi_boxes_xyxy * scale
                    roi_boxes_xyxy[..., 0::2] = roi_boxes_xyxy[..., 0::2].clamp(0.0, float(feat_w - 1))
                    roi_boxes_xyxy[..., 1::2] = roi_boxes_xyxy[..., 1::2].clamp(0.0, float(feat_h - 1))
                    roi_boxes_xyxy[..., 2] = torch.maximum(
                        roi_boxes_xyxy[..., 2], roi_boxes_xyxy[..., 0] + 1.0
                    ).clamp(max=float(feat_w))
                    roi_boxes_xyxy[..., 3] = torch.maximum(
                        roi_boxes_xyxy[..., 3], roi_boxes_xyxy[..., 1] + 1.0
                    ).clamp(max=float(feat_h))
                    batch_indices = torch.arange(
                        bs_fm, device=roi_boxes.device, dtype=roi_boxes_xyxy.dtype
                    )[:, None].expand(bs_fm, nq).reshape(-1, 1)
                    flat_rois = torch.cat(
                        [batch_indices, roi_boxes_xyxy.reshape(-1, 4)], dim=1
                    )
                    feature_map_for_app = (
                        feature_map
                        if bool(getattr(self, "appearance_gradient_to_backbone", False))
                        else feature_map.detach()
                    )
                    roi_feats = roi_align(
                        feature_map_for_app, flat_rois,
                        output_size=14, spatial_scale=1.0, aligned=True,
                    )
                    roi_spatial = roi_feats.flatten(2).permute(0, 2, 1)  # [bs*nq, 196, D]
                    if self.appearance_roi_spatial_attn is not None:
                        attn_w = self.appearance_roi_spatial_attn(roi_spatial)
                        attn_w = torch.softmax(attn_w, dim=1)
                        roi_pooled = (attn_w * roi_spatial).sum(dim=1)
                    else:
                        roi_pooled = roi_spatial.mean(dim=1)
                    roi_pooled = roi_pooled.view(
                        bs_fm, nq, self.hidden_dim
                    )
                    concat_feats = torch.cat(
                        [fused_last_query.detach(), roi_pooled], dim=-1
                    )  # [bs, nq, 512]
                    app_visual_features = self.appearance_concat_fusion(concat_feats)
                else:
                    app_visual_features = fused_last_query.detach()
            elif app_feature_source.startswith("decoder_layer_"):
                # Use a middle decoder layer: preserves cross-modal interaction
                # from earlier decoder layers while retaining more visual detail
                # than the final fused query (shaped by semantic classification).
                try:
                    layer_idx = int(app_feature_source.split("_")[-1])
                    layer_idx = max(0, min(layer_idx, len(hs) - 1))
                except (ValueError, IndexError):
                    layer_idx = -1
                app_visual_features = hs[layer_idx].detach()
            else:
                # Default: last decoder layer (same as semantic head input).
                app_visual_features = fused_last_query.detach()

            bank, bank_mask = (None, None)
            if not use_local_evidence:
                bank, bank_mask = self._get_appearance_prototype_bank(
                    fused_last_query.device
                )
            if (not use_local_evidence) and self.use_appearance_prototype_bank and bank is not None:
                # ---- Prototype-bank path ----
                # Score every query against ALL class prototypes in one matmul.
                # Output is already global-class-aligned [bs, nq, C] (C == num_classes,
                # bank index i == global class id i), so no positive-map aggregation
                # and no local→global remap is needed downstream.  Hard negatives in
                # the loss are mined from the full class set, not just in-image classes.
                bank = bank.to(dtype=app_visual_features.dtype)
                # Discriminative bank = frozen base + per-class learnable residual.
                # forward_against_bank applies class_logit_norm internally, so adding
                # the residual here is equivalent to normalize(base + delta).  The
                # untouched `bank` is kept for the semantic confuser path below so the
                # appearance residual never leaks into semantic hard-negative mining.
                disc_bank = bank
                if self.appearance_visual_centroids is not None:
                    disc_bank = self.appearance_visual_centroids.to(
                        device=bank.device, dtype=bank.dtype
                    )
                    with torch.no_grad():
                        _cent = self.appearance_visual_centroids
                        _cent_norm = _cent.norm(dim=-1)
                        self.appearance_residual_stats = {
                            "centroid_norm_mean": float(_cent_norm.mean().item()),
                            "centroid_norm_max": float(_cent_norm.max().item()),
                        }
                    out["appearance_centroid_norm_mean"] = _cent_norm.mean().detach()
                    out["appearance_centroid_norm_max"] = _cent_norm.max().detach()
                elif self.appearance_discriminative_residual is not None:
                    disc_bank = bank + self.appearance_discriminative_residual.to(bank.dtype)
                    # Lightweight stat for monitoring residual magnitude (is it
                    # learning too much / drifting toward a standalone classifier?).
                    with torch.no_grad():
                        _res = self.appearance_discriminative_residual
                        _res_norm = _res.norm(dim=-1)            # [C]
                        _base_norm = bank.norm(dim=-1).mean().clamp(min=1e-6)
                        self.appearance_residual_stats = {
                            "residual_norm_mean": float(_res_norm.mean().item()),
                            "residual_norm_max": float(_res_norm.max().item()),
                            "residual_to_base_ratio": float(
                                (_res_norm.mean() / _base_norm).item()
                            ),
                        }
                    # Expose as detached scalars so the criterion can log them as
                    # non-loss metrics alongside the appearance loss.
                    out["appearance_residual_norm_mean"] = _res_norm.mean().detach()
                    out["appearance_residual_norm_max"] = _res_norm.max().detach()
                app_class_logits = self.appearance_contrastive_head[-1].forward_against_bank(
                    app_visual_features, disc_bank, bank_mask=bank_mask
                )  # [bs, nq, C]
                app_class_logits = torch.nan_to_num(
                    app_class_logits, nan=0.0, posinf=20.0, neginf=-20.0
                ).clamp(min=-20.0, max=20.0)
                # Tensor form (not list) signals bank mode to loss/postprocess.
                out["appearance_class_scores"] = app_class_logits
                # Identity mapping: local index == global class id.
                out["appearance_caption_label_ids"] = None
                out["appearance_is_global_bank"] = True
                # Semantic scores in the SAME global-class space (bank-aligned),
                # used by the pairwise loss to pick semantic hard negatives.
                # Always computed in bank mode during training — the pairwise loss
                # needs this regardless of the legacy sem_confuser_mode config key.
                if self.training:
                    with torch.no_grad():
                        sem_bank_logits = class_embed_layers[-1].forward_against_bank(
                            fused_last_query.detach(),
                            bank.to(dtype=fused_last_query.dtype),
                            bank_mask=bank_mask,
                        )
                    out["sem_logits_bank"] = sem_bank_logits
            elif not use_local_evidence:
                app_token_logits = self.appearance_contrastive_head[-1](
                    app_visual_features, appearance_text_dict
                )
                # Aggregate token-level logits → class-level scores via positive maps.
                # Use supervision caption_lists as the canonical class ordering.
                # appearance_text_dict may not have 'caption_lists' after re-encoding
                # when appearance_captions != supervision_captions (e.g. global eval).
                app_caption_lists = (
                    appearance_text_dict.get("caption_lists", None)
                    or supervision_text_dict.get("caption_lists", None)
                )
                if app_caption_lists is not None:
                    pos_maps = self._build_caption_positive_map_from_text(
                        appearance_text_dict, app_caption_lists,
                        self.max_text_len, fused_last_query.device,
                    )
                else:
                    pos_maps = self._build_caption_positive_map_from_masks(
                        appearance_cate_to_token_mask_list,
                        self.max_text_len, fused_last_query.device,
                    )
                app_class_scores = []
                for b in range(app_token_logits.shape[0]):
                    pmap = pos_maps[b].to(device=app_token_logits.device, dtype=app_token_logits.dtype)
                    pmap = torch.nan_to_num(pmap, nan=0.0, posinf=0.0, neginf=0.0)
                    if pmap.shape[0] == 0:
                        app_class_scores.append(app_token_logits.new_zeros((app_token_logits.shape[1], 0)))
                    else:
                        # ContrastiveEmbed pads invalid text positions with -inf.
                        # Class aggregation multiplies by sparse positive maps, so
                        # leave invalid tokens as zero contribution instead of
                        # producing -inf * 0 = NaN.
                        token_logits = torch.nan_to_num(
                            app_token_logits[b],
                            nan=0.0,
                            posinf=20.0,
                            neginf=0.0,
                        ).clamp(min=-20.0, max=20.0)
                        class_scores = token_logits @ pmap.T
                        class_scores = torch.nan_to_num(
                            class_scores,
                            nan=0.0,
                            posinf=20.0,
                            neginf=-20.0,
                        ).clamp(min=-20.0, max=20.0)
                        app_class_scores.append(class_scores)
                out["appearance_class_scores"] = app_class_scores
                out["appearance_caption_label_ids"] = appearance_text_dict.get(
                    "caption_label_ids", None
                )

        out["pred_logits"] = pred_logits
        if lgpr_survival_logits_all is not None:
            out["pred_lgpr_survival"] = lgpr_survival_logits_all[-1]
            # The custom survival loss consumes the last configured decoder
            # layers directly; it is intentionally not routed through the
            # regular Hungarian-matched auxiliary-loss loop.
            out["pred_lgpr_survival_all"] = lgpr_survival_logits_all
        # ---- CLIP crop evidence branch (independent of the contrastive head) ----
        # Original-pixel crop -> frozen CLIP vision vs frozen CLIP text of candidate
        # appearance descriptions.  Reuses the candidate/GT/loss/analyze plumbing
        # (writes the same appearance_local_* keys).  Gated by config; CLIP towers
        # are frozen so this only adds a tiny logit-scale (+ optional projection).
        if (
            bool(getattr(self, "use_appearance_clip_evidence", False))
            and self.appearance_clip_evidence_head is not None
        ):
            self._forward_appearance_clip_evidence(
                out=out,
                samples=samples,
                roi_boxes=roi_boxes,
                fused_last_query=fused_last_query,
                roi_query_gt_labels=roi_query_gt_labels,
                pred_logits=pred_logits,
                supervision_text_dict=supervision_text_dict,
                supervision_cate_to_token_mask_list=supervision_cate_to_token_mask_list,
            )
        # ---- Method B: same-group token-margin (caption-local class scores) ----
        # Store per-query caption-local class scores WITH gradient (unlike the
        # isolated appearance branches, B's purpose is to sharpen the detector's own
        # query-text alignment, so gradient MUST flow back into pred_logits).  The
        # criterion's loss_semantic_group_margin reads these + GT + cap_group_ids and
        # applies a hinge on same-group pairs.  Class scores mirror the detector's
        # actual aggregation: pred_logits.sigmoid() @ uniform-mean positive-map.
        if self.training and bool(getattr(self, "use_semantic_group_margin", False)):
            sup_caption_lists = supervision_text_dict.get("caption_lists", None)
            if sup_caption_lists is not None:
                _b_pos_maps = self._build_caption_positive_map_from_text(
                    supervision_text_dict, sup_caption_lists, self.max_text_len, pred_logits.device,
                )
            else:
                _b_pos_maps = self._build_caption_positive_map_from_masks(
                    supervision_cate_to_token_mask_list, self.max_text_len, pred_logits.device,
                )
            # Per-image [nq, n_cap] class scores; keep as a list (n_cap varies per image).
            # Clean pred_logits the same way the main focal loss does (it carries
            # ±inf on padded text positions); raw sigmoid of nan/inf would poison the
            # class scores -> nan loss.
            _b_scores = []
            _pl_clean = torch.nan_to_num(pred_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            p_tok = _pl_clean.sigmoid()
            for _bi in range(pred_logits.shape[0]):
                pm = _b_pos_maps[_bi]
                if pm is None or pm.shape[0] == 0:
                    _b_scores.append(None)
                    continue
                pm = torch.nan_to_num(pm.to(p_tok.dtype), nan=0.0, posinf=0.0, neginf=0.0)
                _b_scores.append(p_tok[_bi] @ pm.T)   # [nq, n_cap], grad-carrying
            out["semantic_group_margin_scores"] = _b_scores
        return_query_selector_outputs = bool(
            kw.get("return_query_selector_outputs", False)
        )
        if (self.training or return_query_selector_outputs) and self.transformer is not None:
            selector_outputs = getattr(self.transformer, "latest_query_selector_outputs", None) or {}
            if selector_outputs:
                out["query_selector_ranking_scores"] = selector_outputs.get("ranking_scores", None)
                out["query_selector_learned_scores"] = selector_outputs.get("learned_scores", None)
                out["query_selector_candidate_indices"] = selector_outputs.get("candidate_indices", None)
                out["query_selector_selected_candidate_indices"] = selector_outputs.get("selected_candidate_indices", None)
                # Global flattened encoder indices after the *actual* routing
                # policy.  For mixed baseline/LGPR routing this differs from
                # selected_candidate_indices, which describes the selector-only
                # ranking inside the candidate pool.  Keeping both is useful for
                # read-only oracle/coverage diagnostics.
                out["query_selector_selected_indices"] = selector_outputs.get("selected_indices", None)
                out["query_selector_candidate_scores"] = selector_outputs.get("candidate_scores", None)
                out["query_selector_candidate_prior_scores"] = selector_outputs.get("candidate_prior_scores", None)
                out["query_selector_candidate_boxes"] = selector_outputs.get("candidate_boxes", None)
                out["query_selector_candidate_valid_mask"] = selector_outputs.get("candidate_valid_mask", None)
                out["query_selector_candidate_level_ids"] = selector_outputs.get("candidate_level_ids", None)
                out["query_selector_contextual_scores"] = selector_outputs.get("contextual_scores", None)
                out["query_selector_scale_weights"] = selector_outputs.get("scale_weights", None)

        if self.use_quality_head:
            outputs_quality = torch.stack([
                layer_quality_embed(layer_hs)
                for layer_quality_embed, layer_hs in zip(self.quality_embed, hs)
            ])
            out["pred_quality"] = outputs_quality[-1]

        if self.use_visual_confidence:
            visual_confidence = self._extract_roi_visual_features(
                feature_map=srcs[0] if srcs else None,
                pred_boxes=outputs_coord_list[-1],
                output_size=self.visual_confidence_roi_output_size,
            )
            if visual_confidence is not None:
                out["pred_visual_confidence"] = visual_confidence

        if self.use_appearance_conditioned_gate:
            decoder_query = (
                fused_last_query
                if self.appearance_gate_feature_source == "decoder"
                else None
            )
            app_gate_scores = self._compute_appearance_gate_scores(
                feature_map=srcs[0] if srcs else None,
                pred_boxes=outputs_coord_list[-1],
                appearance_text_dict=appearance_text_dict,
                appearance_cate_to_token_mask_list=appearance_cate_to_token_mask_list,
                output_size=self.visual_confidence_roi_output_size,
                decoder_query=decoder_query,
            )
            if app_gate_scores is not None:
                out["pred_appearance_gate_scores"] = app_gate_scores

                # Logit-level bias (pre-sigmoid): stronger than post-hoc gate.
                # Applied BEFORE sigmoid, so it can push classes across the threshold.
                if self.appearance_logit_bias_weight > 0.0:
                    bias = app_gate_scores  # [bs, nq, num_cats]
                    if self.appearance_logit_bias_detach:
                        bias = bias.detach()
                    if self.appearance_logit_bias_norm == "sigmoid_centered":
                        bias = bias.sigmoid() - 0.5
                    elif self.appearance_logit_bias_norm == "tanh":
                        bias = torch.tanh(bias)
                    elif self.appearance_logit_bias_norm != "raw":
                        bias = torch.tanh(bias)
                    pred_logits_ref = out["pred_logits"]  # [bs, nq, num_classes]
                    if bias.shape[-1] > pred_logits_ref.shape[-1]:
                        bias = bias[..., :pred_logits_ref.shape[-1]]
                    elif bias.shape[-1] < pred_logits_ref.shape[-1]:
                        padded = bias.new_zeros(*bias.shape[:-1], pred_logits_ref.shape[-1])
                        padded[..., :bias.shape[-1]] = bias
                        bias = padded
                    out["pred_logits"] = pred_logits_ref + self.appearance_logit_bias_weight * bias

        need_loss_aux_outputs = self.training or bool(kw.get("return_loss_aux_outputs", False))
        if (self.use_aggr or self.use_query_contrastive or self.use_aux_token_ce) and need_loss_aux_outputs:
            if self.use_aggr:
                category_text_prototypes, category_prototype_valid_mask = build_category_text_prototypes(
                    supervision_text_dict["encoded_text"],
                    supervision_cate_to_token_mask_list,
                    supervision_text_dict["text_token_mask"],
                )
                out["category_text_prototypes"] = category_text_prototypes
                out["category_prototype_valid_mask"] = category_prototype_valid_mask
            if self.use_query_contrastive:
                (
                    contrastive_text_prototypes,
                    contrastive_prototype_valid_mask,
                ) = build_category_text_prototypes(
                    query_contrastive_text_dict["encoded_text"],
                    query_contrastive_cate_to_token_mask_list,
                    query_contrastive_text_dict["text_token_mask"],
                )
                out["query_contrastive_text_prototypes"] = contrastive_text_prototypes
                out["query_contrastive_prototype_valid_mask"] = contrastive_prototype_valid_mask
        projected_aux_query = None
        if self.aux_query_proj is not None:
            projected_aux_query = self.aux_query_proj(fused_last_query)
        if self.use_query_contrastive:
            out["query_features"] = projected_aux_query if projected_aux_query is not None else fused_last_query
        if self.use_aux_token_ce:
            aux_query = projected_aux_query if projected_aux_query is not None else fused_last_query
            out["pred_aux_token_logits"] = class_embed_layers[-1](aux_query, query_contrastive_text_dict)
            out["aux_token"] = query_contrastive_token
            aux_bs, aux_len_td = query_contrastive_text_dict["text_token_mask"].shape
            out["aux_text_mask"] = torch.zeros(
                aux_bs, self.max_text_len, dtype=torch.bool, device=samples.device
            )
            for b in range(aux_bs):
                for j in range(aux_len_td):
                    if query_contrastive_text_dict["text_token_mask"][b][j] == True:
                        out["aux_text_mask"][b][j] = True

        # Used to calculate losses
        bs, len_td = supervision_text_dict['text_token_mask'].shape
        out['text_mask']=torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
            samples.device
        )
        for b in range(bs):
            for j in range(len_td):
                if supervision_text_dict['text_token_mask'][b][j] == True:
                    out['text_mask'][b][j] = True

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord_list,
                outputs_quality if self.use_quality_head else None
            )
        out['token']=one_hot_token
        out['supervision_token'] = supervision_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            enc_class_embed = self.transformer.enc_out_class_embed
            if (
                getattr(self, "dual_prompt_enc_out_class_embed", None) is not None
                and use_dual_prompt_supervision
            ):
                enc_class_embed = self.dual_prompt_enc_out_class_embed
            interm_class = enc_class_embed(hs_enc[-1], supervision_text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])


        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])

        if bool(getattr(self, "use_self_facb", False)) or bool(
            getattr(self, "return_self_facb_features", False)
        ):
            out["self_facb_srcs"] = [s.detach() for s in srcs]
            bank, bank_mask = self._get_appearance_prototype_bank(samples.device)
            if bank is not None:
                out["self_facb_proto_bank"] = bank.detach().to(dtype=pred_logits.dtype)
                out["self_facb_proto_bank_mask"] = (
                    bank_mask.detach() if bank_mask is not None else None
                )
            out["self_facb_levels"] = list(
                getattr(
                    self,
                    "self_facb_levels",
                    getattr(self, "appearance_local_evidence_levels", [0]),
                )
                or [0]
            )
            out["self_facb_roi_size"] = int(
                getattr(
                    self,
                    "self_facb_roi_size",
                    getattr(self, "appearance_local_evidence_roi_size", 7),
                )
            )
        if self.training and bool(getattr(self, "use_facb_speedhr_loss", False)):
            out["facb_speedhr_samples"] = samples.tensors.detach()
            (
                facb_speedhr_text_prototypes,
                facb_speedhr_text_valid,
            ) = build_category_text_prototypes(
                supervision_text_dict["encoded_text"],
                supervision_cate_to_token_mask_list,
                supervision_text_dict["text_token_mask"],
            )
            out["facb_speedhr_text_prototypes"] = [
                proto.detach() for proto in facb_speedhr_text_prototypes
            ]
            out["facb_speedhr_text_valid"] = [
                valid.detach() for valid in facb_speedhr_text_valid
            ]
            if bool(getattr(self, "facb_speedhr_candidate_multi_facet", False)):
                facet_bank, facet_bank_mask = (
                    self._get_facb_speedhr_facet_prototype_bank(samples.device)
                )
                if facet_bank is not None:
                    out["facb_speedhr_facet_prototypes"] = facet_bank.detach()
                    out["facb_speedhr_facet_valid"] = facet_bank_mask.detach()
            if bool(
                getattr(
                    self, "facb_speedhr_shared_compositional_facets", False
                )
            ):
                shared_bank, shared_bank_mask = (
                    self._get_facb_speedhr_shared_facet_prototype_bank(
                        samples.device
                    )
                )
                if shared_bank is not None:
                    out["facb_speedhr_shared_facet_prototypes"] = (
                        shared_bank.detach()
                    )
                    out["facb_speedhr_shared_facet_valid"] = (
                        shared_bank_mask.detach()
                    )
        if not self.training and bool(
            getattr(self, "enable_facb_speedhr_inference", False)
        ):
            if bool(getattr(self, "facb_speedhr_defer_to_postprocess", False)):
                # Keep crop extraction after the detector has produced its exact
                # class-score tensor.  The closure is consumed immediately by
                # PostProcess.forward and never enters a checkpoint or cache.
                def _deferred_facb_speedhr_reader(semantic_scores):
                    return self._run_facb_speedhr_inference(
                        samples,
                        out["pred_logits"],
                        out["pred_boxes"],
                        supervision_cate_to_token_mask_list,
                        caption_lists=supervision_caption_lists,
                        supervision_text_dict=supervision_text_dict,
                        image_paths=kw.get("facb_image_paths"),
                        query_states=fused_last_query,
                        semantic_override=semantic_scores,
                    )
                out["_facb_speedhr_deferred_reader"] = _deferred_facb_speedhr_reader
                out["_facb_f1_image_paths"] = kw.get("facb_image_paths")
            else:
                facb_infer = self._run_facb_speedhr_inference(
                    samples,
                    out["pred_logits"],
                    out["pred_boxes"],
                    supervision_cate_to_token_mask_list,
                    caption_lists=supervision_caption_lists,
                    supervision_text_dict=supervision_text_dict,
                    image_paths=kw.get("facb_image_paths"),
                    query_states=fused_last_query,
                )
                if facb_infer is not None:
                    out["facb_speedhr_inference"] = facb_infer
                    out["_facb_f1_image_paths"] = kw.get("facb_image_paths")

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_quality=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_quality is not None:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_quality": q}
                for a, b, q in zip(outputs_class[:-1], outputs_coord[:-1], outputs_quality[:-1])
            ]
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]




class FACBSpeedHRCropHead(nn.Module):
    """Small crop-level reader for speed-limit numerals.

    This is intentionally narrow: it reads high-resolution image crops for
    speed-limit signs and outputs logits over observed speed values.  It does
    not create detector class scores by itself; the detector still learns
    through its native token focal loss.
    """

    def __init__(self, num_classes, crop_size=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.crop_size = int(crop_size)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, image_tensors, rois):
        if image_tensors is None or rois is None or rois.numel() == 0:
            return None
        crops = roi_align(
            image_tensors.float(),
            rois.float(),
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        features = self.net(crops)
        return self.classifier(features)


class FACBGDinoHRCropHead(nn.Module):
    """High-resolution crop reader that reuses the detector visual backbone.

    The GroundingDINO backbone is shared as a frozen reference and is deliberately
    not registered as a child module, so this head only saves/trains its own
    crop-side projection, low-rank adapter, and classifier.  This gives FACB a
    stronger evidence source than the small CNN without perturbing the main
    detector's global visual representation.
    """

    def __init__(
        self,
        crop_backbone,
        feature_dim,
        num_classes,
        crop_size=224,
        hidden_dim=256,
        adapter_rank=16,
        adapter_scale=1.0,
        lora_rank=8,
        lora_scale=1.0,
        lora_last_qkv=2,
        candidate_matching=False,
        candidate_decoder_layers=2,
        candidate_nheads=8,
        candidate_max_facets=5,
        shared_facet_count=0,
        shared_candidate_fusion_init=-2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.__dict__["_crop_backbone_ref"] = crop_backbone
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.crop_size = int(crop_size)
        self.adapter_rank = max(int(adapter_rank), 0)
        self.adapter_scale = float(adapter_scale)
        self.lora_rank = max(int(lora_rank), 0)
        self.lora_scale = float(lora_scale)
        self.candidate_matching = bool(candidate_matching)
        self.candidate_max_facets = max(int(candidate_max_facets), 1)
        self.shared_facet_count = max(int(shared_facet_count), 0)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
        )
        if self.adapter_rank > 0:
            self.adapter_down = nn.Linear(int(hidden_dim), self.adapter_rank, bias=False)
            self.adapter_up = nn.Linear(self.adapter_rank, int(hidden_dim), bias=False)
            nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = None
            self.adapter_up = None
        self._crop_lora_targets = []
        self.crop_lora_down = nn.ModuleList()
        self.crop_lora_up = nn.ModuleList()
        if self.lora_rank > 0 and crop_backbone is not None:
            qkv_modules = [
                module
                for name, module in crop_backbone.named_modules()
                if name.endswith("attn.qkv") and isinstance(module, nn.Linear)
            ]
            if int(lora_last_qkv) > 0:
                qkv_modules = qkv_modules[-int(lora_last_qkv):]
            self._crop_lora_targets = qkv_modules
            for module in qkv_modules:
                down = nn.Linear(module.in_features, self.lora_rank, bias=False)
                up = nn.Linear(self.lora_rank, module.out_features, bias=False)
                nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                nn.init.zeros_(up.weight)
                self.crop_lora_down.append(down)
                self.crop_lora_up.append(up)
        if self.candidate_matching:
            self.candidate_text_proj = nn.Sequential(
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=int(hidden_dim),
                nhead=max(int(candidate_nheads), 1),
                dim_feedforward=4 * int(hidden_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=max(int(candidate_decoder_layers), 1),
                norm=nn.LayerNorm(int(hidden_dim)),
            )
            self.candidate_score = nn.Linear(int(hidden_dim), 1)
            self.candidate_facet_type_embedding = nn.Embedding(
                self.candidate_max_facets, int(hidden_dim)
            )
            self.candidate_facet_weight_logits = nn.Parameter(
                torch.zeros(self.candidate_max_facets)
            )
            self.candidate_facet_gate = nn.Linear(int(hidden_dim), 1)
            nn.init.zeros_(self.candidate_facet_gate.weight)
            nn.init.zeros_(self.candidate_facet_gate.bias)
            self.shared_candidate_space_visual_proj = nn.Sequential(
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.shared_candidate_space_logit_scale = nn.Parameter(
                torch.tensor(math.log(10.0), dtype=torch.float32)
            )
            self.use_shared_candidate_space = False
            if self.shared_facet_count > 0:
                self.shared_facet_query = nn.Embedding(
                    self.shared_facet_count, int(hidden_dim)
                )
                self.shared_facet_visual_proj = nn.Sequential(
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                self.shared_facet_text_proj = nn.Sequential(
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                self.shared_facet_logit_scale = nn.Parameter(
                    torch.tensor(math.log(10.0), dtype=torch.float32)
                )
                self.shared_facet_weight_logits = nn.Parameter(
                    torch.zeros(self.shared_facet_count)
                )
                self.shared_candidate_fusion_logit = nn.Parameter(
                    torch.tensor(
                        float(shared_candidate_fusion_init), dtype=torch.float32
                    )
                )
            else:
                self.shared_facet_query = None
                self.shared_facet_visual_proj = None
                self.shared_facet_text_proj = None
                self.shared_facet_logit_scale = None
                self.shared_facet_weight_logits = None
                self.shared_candidate_fusion_logit = None
        else:
            self.candidate_text_proj = None
            self.candidate_decoder = None
            self.candidate_score = None
            self.candidate_facet_type_embedding = None
            self.candidate_facet_weight_logits = None
            self.candidate_facet_gate = None
            self.shared_candidate_space_visual_proj = None
            self.shared_candidate_space_logit_scale = None
            self.use_shared_candidate_space = False
            self.shared_facet_query = None
            self.shared_facet_visual_proj = None
            self.shared_facet_text_proj = None
            self.shared_facet_logit_scale = None
            self.shared_facet_weight_logits = None
            self.shared_candidate_fusion_logit = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )
        self.score_residual_alpha_logit = nn.Parameter(torch.zeros(()))
        self.shared_symbol_bce = False
        self.shared_facet_names = []
        self.shared_facet_alignment_mode = "decoder"
        self.shared_facet_token_topk = 3
        self.shared_facet_token_temperature = 0.07

    def score_residual_alpha(self):
        # Starts exactly at 0 while preserving a non-zero gradient at init.
        return F.softplus(self.score_residual_alpha_logit) - math.log(2.0)

    @property
    def crop_backbone(self):
        return self.__dict__.get("_crop_backbone_ref", None)

    def _run_crop_backbone(self, crop_backbone, crop_nested):
        handles = []
        for idx, module in enumerate(self._crop_lora_targets):
            down = self.crop_lora_down[idx]
            up = self.crop_lora_up[idx]

            def _hook(_module, inputs, output, down=down, up=up):
                if not inputs:
                    return output
                residual = up(F.gelu(down(inputs[0].float())))
                return output + self.lora_scale * residual.to(dtype=output.dtype)

            handles.append(module.register_forward_hook(_hook))
        try:
            if handles:
                return crop_backbone(crop_nested)
            with torch.no_grad():
                return crop_backbone(crop_nested)
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        image_tensors,
        rois,
        candidate_text=None,
        candidate_mask=None,
        shared_facet_prototypes=None,
        shared_facet_valid=None,
        candidate_shared_targets=None,
        candidate_shared_target_mask=None,
        return_states=False,
    ):
        if image_tensors is None or rois is None or rois.numel() == 0:
            return None
        crop_backbone = self.crop_backbone
        if crop_backbone is None:
            return None
        crops = roi_align(
            image_tensors.float(),
            rois.float(),
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        crop_mask = torch.zeros(
            (crops.shape[0], crops.shape[-2], crops.shape[-1]),
            dtype=torch.bool,
            device=crops.device,
        )
        crop_nested = NestedTensor(crops, crop_mask)
        backbone_out = self._run_crop_backbone(crop_backbone, crop_nested)
        if isinstance(backbone_out, dict):
            last_key = sorted(backbone_out.keys())[-1]
            feat = backbone_out[last_key].tensors
        elif isinstance(backbone_out, (list, tuple)):
            feat = backbone_out[-1]
            if isinstance(feat, NestedTensor):
                feat = feat.tensors
        else:
            feat = backbone_out
        feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
        crop_tokens = feat.flatten(2).transpose(1, 2)
        crop_tokens = self.input_proj(crop_tokens)
        pooled = crop_tokens.mean(dim=1)
        hidden = pooled
        if self.adapter_down is not None and self.adapter_up is not None:
            hidden = hidden + self.adapter_scale * self.adapter_up(
                F.gelu(self.adapter_down(hidden))
            )
        if self.candidate_matching and candidate_text is not None:
            candidate_text = torch.nan_to_num(
                candidate_text.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            is_multi_facet = candidate_text.dim() == 4
            if is_multi_facet:
                num_rows, num_candidates, num_facets, text_dim = candidate_text.shape
                if num_facets > self.candidate_max_facets:
                    raise ValueError(
                        f"FACB candidate facets {num_facets} exceed "
                        f"configured maximum {self.candidate_max_facets}"
                    )
                candidate_text_flat = candidate_text.reshape(
                    num_rows, num_candidates * num_facets, text_dim
                )
                facet_ids = torch.arange(
                    num_facets, device=candidate_text.device, dtype=torch.long
                )
                facet_type = self.candidate_facet_type_embedding(facet_ids)
                facet_type = facet_type.view(1, 1, num_facets, -1).expand(
                    num_rows, num_candidates, -1, -1
                )
                facet_type = facet_type.reshape(
                    num_rows, num_candidates * num_facets, -1
                )
                candidate_query = self.candidate_text_proj(candidate_text_flat)
                candidate_query = candidate_query + facet_type
                candidate_mask_flat = (
                    candidate_mask.reshape(num_rows, num_candidates * num_facets)
                    if candidate_mask is not None
                    else None
                )
            else:
                candidate_query_base = self.candidate_text_proj(candidate_text)
                candidate_query = candidate_query_base
                if (
                    shared_facet_prototypes is not None
                    and self.candidate_facet_type_embedding is not None
                ):
                    # Preserve the semantic-facet query convention learned by
                    # the class-aligned warm start while switching the other
                    # attributes to deduplicated shared vocabularies.
                    candidate_query = candidate_query + self.candidate_facet_type_embedding.weight[
                        0
                    ].view(1, 1, -1)
                candidate_mask_flat = candidate_mask
            candidate_padding_mask = None
            if candidate_mask_flat is not None:
                candidate_padding_mask = ~candidate_mask_flat.to(
                    device=candidate_query.device, dtype=torch.bool
                )
            use_shared_candidate_space = (
                (not is_multi_facet)
                and bool(getattr(self, "use_shared_candidate_space", False))
                and self.shared_candidate_space_visual_proj is not None
            )
            if return_states and (use_shared_candidate_space or is_multi_facet):
                raise ValueError("states require the shared-facet decoder path")
            if use_shared_candidate_space:
                crop_semantic = F.normalize(
                    self.shared_candidate_space_visual_proj(hidden),
                    dim=-1,
                    eps=1e-6,
                )
                text_semantic = F.normalize(
                    candidate_query_base, dim=-1, eps=1e-6
                )
                shared_candidate_scale = (
                    self.shared_candidate_space_logit_scale.exp().clamp(
                        min=1.0, max=100.0
                    )
                )
                decoded_logits = shared_candidate_scale * torch.einsum(
                    "nh,nkh->nk", crop_semantic, text_semantic
                )
            else:
                decoded = self.candidate_decoder(
                    candidate_query,
                    crop_tokens,
                    tgt_key_padding_mask=candidate_padding_mask,
                )
                decoded_logits = self.candidate_score(decoded).squeeze(-1)
            if not is_multi_facet:
                if candidate_mask_flat is not None:
                    decoded_logits = decoded_logits.masked_fill(
                        ~candidate_mask_flat.to(
                            device=decoded_logits.device, dtype=torch.bool
                        ),
                        -20.0,
                    )
                if (
                    self.shared_facet_query is None
                    or shared_facet_prototypes is None
                ):
                    if return_states:
                        return {"candidate_logits": decoded_logits, "candidate_states": decoded, "crop_state": hidden, "candidate_text_states": candidate_query}
                    return decoded_logits

                num_facets = min(
                    self.shared_facet_count,
                    int(shared_facet_prototypes.shape[0]),
                )
                facet_text = F.normalize(
                    self.shared_facet_text_proj(
                        torch.nan_to_num(
                            shared_facet_prototypes[:num_facets].float(),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    ),
                    dim=-1,
                    eps=1e-6,
                )
                shared_scale = self.shared_facet_logit_scale.exp().clamp(
                    min=1.0, max=100.0
                )
                alignment_mode = str(
                    getattr(self, "shared_facet_alignment_mode", "decoder")
                    or "decoder"
                ).strip().lower()
                if alignment_mode in {"token", "token_topk", "localized"}:
                    token_visual = F.normalize(
                        self.shared_facet_visual_proj(crop_tokens),
                        dim=-1,
                        eps=1e-6,
                    )
                    token_logits = shared_scale * torch.einsum(
                        "nth,fvh->nftv", token_visual, facet_text
                    )
                    token_topk = max(
                        1,
                        min(
                            int(getattr(self, "shared_facet_token_topk", 3)),
                            int(token_logits.shape[2]),
                        ),
                    )
                    token_temp = max(
                        float(
                            getattr(
                                self,
                                "shared_facet_token_temperature",
                                0.07,
                            )
                        ),
                        1e-6,
                    )
                    facet_names = list(getattr(self, "shared_facet_names", []) or [])
                    shared_logits_rows = []
                    pooled_visual = F.normalize(
                        self.shared_facet_visual_proj(hidden),
                        dim=-1,
                        eps=1e-6,
                    )
                    global_logits = shared_scale * torch.einsum(
                        "nh,fvh->nfv", pooled_visual, facet_text
                    )
                    for facet_idx in range(num_facets):
                        facet_name = (
                            str(facet_names[facet_idx]).strip().lower()
                            if facet_idx < len(facet_names)
                            else ""
                        )
                        if facet_name == "group":
                            logits_f = global_logits[:, facet_idx]
                        elif alignment_mode == "token_topk":
                            logits_f = token_logits[:, facet_idx].topk(
                                k=token_topk, dim=1
                            ).values.mean(dim=1)
                        else:
                            logits_f = (
                                token_temp
                                * torch.logsumexp(
                                    token_logits[:, facet_idx] / token_temp,
                                    dim=1,
                                )
                            )
                        shared_logits_rows.append(logits_f)
                    shared_logits = torch.stack(shared_logits_rows, dim=1)
                else:
                    facet_query = self.shared_facet_query.weight[:num_facets]
                    facet_query = facet_query.unsqueeze(0).expand(
                        crop_tokens.shape[0], -1, -1
                    )
                    facet_state = self.candidate_decoder(facet_query, crop_tokens)
                    facet_visual = F.normalize(
                        self.shared_facet_visual_proj(facet_state),
                        dim=-1,
                        eps=1e-6,
                    )
                    shared_logits = shared_scale * torch.einsum(
                        "nfh,fvh->nfv", facet_visual, facet_text
                    )
                if shared_facet_valid is not None:
                    shared_valid = shared_facet_valid[:num_facets].to(
                        device=shared_logits.device, dtype=torch.bool
                    )
                    shared_logits = shared_logits.masked_fill(
                        ~shared_valid.unsqueeze(0), -20.0
                    )

                candidate_facet_scores = None
                fusion = torch.sigmoid(self.shared_candidate_fusion_logit)
                if candidate_shared_targets is not None:
                    target_ids = candidate_shared_targets[:, :, :num_facets].to(
                        device=shared_logits.device, dtype=torch.long
                    )
                    target_mask = target_ids >= 0
                    if candidate_shared_target_mask is not None:
                        target_mask = target_mask & candidate_shared_target_mask[
                            :, :, :num_facets
                        ].to(device=shared_logits.device, dtype=torch.bool)
                    safe_ids = target_ids.clamp(min=0)
                    expanded_logits = shared_logits[:, None].expand(
                        -1, target_ids.shape[1], -1, -1
                    )
                    gathered = expanded_logits.gather(3, safe_ids)
                    gathered = gathered.masked_fill(~target_mask, 0.0)
                    candidate_facet_scores = gathered.sum(dim=-1) / target_mask.sum(
                        dim=-1
                    ).clamp(min=1)
                    if bool(getattr(self, "shared_symbol_bce", False)):
                        facet_names = list(getattr(self, "shared_facet_names", []) or [])
                        symbol_indices = [
                            idx
                            for idx, name in enumerate(facet_names[:num_facets])
                            if str(name).strip().lower() == "symbol"
                        ]
                        if symbol_indices:
                            gathered_prob = torch.sigmoid(gathered)
                            symbol_score = gathered_prob.sum(dim=-1) / target_mask.sum(
                                dim=-1
                            ).clamp(min=1)
                            symbol_score = torch.logit(
                                symbol_score.clamp(min=1e-6, max=1.0 - 1e-6)
                            )
                            for symbol_idx in symbol_indices:
                                candidate_facet_scores[:, :, symbol_idx] = torch.where(
                                    target_mask[:, :, symbol_idx].any(dim=-1),
                                    symbol_score[:, :, symbol_idx],
                                    candidate_facet_scores[:, :, symbol_idx],
                                )
                    facet_available = target_mask.any(dim=-1)
                    facet_prior = F.softmax(
                        self.shared_facet_weight_logits[:num_facets], dim=0
                    ).view(1, 1, num_facets)
                    facet_weight = facet_prior * facet_available.float()
                    facet_weight = facet_weight / facet_weight.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-6)
                    composition_score = (
                        candidate_facet_scores * facet_weight
                    ).sum(dim=-1)
                    decoded_logits = decoded_logits + fusion * composition_score

                return {
                    "candidate_logits": decoded_logits,
                    **({"candidate_states": decoded, "crop_state": hidden, "candidate_text_states": candidate_query} if return_states else {}),
                    "shared_facet_logits": shared_logits,
                    "candidate_facet_scores": candidate_facet_scores,
                    "shared_facet_fusion": fusion,
                }

            facet_logits = decoded_logits.reshape(
                num_rows, num_candidates, num_facets
            )
            if candidate_mask is None:
                facet_valid = torch.ones_like(facet_logits, dtype=torch.bool)
            else:
                facet_valid = candidate_mask.to(
                    device=facet_logits.device, dtype=torch.bool
                )
            facet_logits = facet_logits.masked_fill(~facet_valid, -20.0)
            decoded_facets = decoded.reshape(
                num_rows, num_candidates, num_facets, -1
            )
            # A zero-initialized conditional gate starts from equal weighting,
            # then learns which evidence is useful for each crop/candidate pair.
            # The global logits are only a facet prior; the decoded state makes
            # numeric classes favor value evidence and directional classes favor
            # symbol evidence without requiring a hand-written class gate.
            raw_weights = self.candidate_facet_gate(decoded_facets).squeeze(-1)
            raw_weights = raw_weights + self.candidate_facet_weight_logits[
                :num_facets
            ].view(1, 1, num_facets)
            raw_weights = raw_weights.masked_fill(~facet_valid, float("-inf"))
            facet_weights = F.softmax(raw_weights, dim=-1)
            facet_weights = torch.nan_to_num(
                facet_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            candidate_logits = (facet_logits * facet_weights).sum(dim=-1)
            candidate_valid = facet_valid.any(dim=-1)
            candidate_logits = candidate_logits.masked_fill(~candidate_valid, -20.0)
            return candidate_logits, facet_logits, facet_weights
        return self.classifier(hidden)


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        focal_alpha,
        focal_gamma,
        losses,
        aggr_regularizer=None,
        query_contrastive_temperature=0.07,
        query_contrastive_mode="softmax_ce",
        query_contrastive_bias=0.0,
        query_contrastive_topk_neg=5,
        query_contrastive_same_group_only=False,
        query_contrastive_fallback_all=False,
        use_global_image_text_sigmoid_loss=False,
        use_aux_token_hard_neg=False,
        aux_token_hard_neg_pos_weight=1.0,
        aux_token_hard_neg_same_group_weight=4.0,
        aux_token_hard_neg_diff_group_weight=0.1,
        use_proto_cls_head=False,
        use_appearance_cls_loss=False,
        use_appearance_contrastive_loss=False,
        use_appearance_text_alignment=False,
        appearance_contrastive_topk_neg=5,
        appearance_contrastive_same_group_only=True,
        appearance_contrastive_fallback_all=True,
        appearance_global_fine_group_ids=None,
        appearance_contrastive_sem_confuser_mode=False,
        appearance_full_ce_loss_coef=0.0,
        appearance_contrastive_loss_mode="candidate_ce",
        appearance_pairwise_loss_margin=0.0,
        use_self_facb_loss=False,
        self_facb_topk=10,
        self_facb_ce_aux_weight=0.5,
        self_facb_pairwise_loss_weight=1.0,
        self_facb_pairwise_loss_margin=0.0,
        self_facb_force_gt_candidate=True,
        self_facb_train_box_source="gt",
        self_facb_train_same_group_only=False,
        self_facb_semantic_top1_loss_weight=0.0,
        self_facb_semantic_top1_loss_margin=0.2,
        self_facb_hard_case_weight=0.0,
        self_facb_hard_case_semantic_margin=0.03,
        self_facb_same_group_hard_weight=0.0,
        facb_symbol_align_topk=5,
        facb_symbol_align_temperature=0.07,
        facb_symbol_align_pairwise_margin=0.1,
        facb_symbol_align_pairwise_weight=0.5,
        facb_symbol_align_inbatch_weight=0.5,
        facb_symbol_align_hard_weight=1.0,
        facb_symbol_align_easy_weight=0.1,
        facb_symbol_align_error_only=False,
        use_facb_text_orth_loss=False,
        facb_text_orth_margin=0.85,
        facb_text_orth_anchor_weight=0.05,
    ):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma= focal_gamma
        self.aggr_regularizer = aggr_regularizer
        self.query_contrastive_temperature = float(query_contrastive_temperature)
        self.query_contrastive_mode = str(query_contrastive_mode).strip().lower()
        self.query_contrastive_bias = float(query_contrastive_bias)
        self.query_contrastive_topk_neg = max(int(query_contrastive_topk_neg), 0)
        self.query_contrastive_same_group_only = bool(query_contrastive_same_group_only)
        self.query_contrastive_fallback_all = bool(query_contrastive_fallback_all)
        self.use_global_image_text_sigmoid_loss = bool(use_global_image_text_sigmoid_loss)
        self.use_aux_token_hard_neg = bool(use_aux_token_hard_neg)
        self.aux_token_hard_neg_pos_weight = float(aux_token_hard_neg_pos_weight)
        self.aux_token_hard_neg_same_group_weight = float(aux_token_hard_neg_same_group_weight)
        self.aux_token_hard_neg_diff_group_weight = float(aux_token_hard_neg_diff_group_weight)
        self.use_proto_cls_head = bool(use_proto_cls_head)
        self.use_appearance_cls_loss = bool(use_appearance_cls_loss)
        self.use_appearance_contrastive_loss = bool(use_appearance_contrastive_loss)
        self.use_appearance_text_alignment = bool(use_appearance_text_alignment)
        self.appearance_contrastive_topk_neg = max(int(appearance_contrastive_topk_neg), 0)
        self.appearance_full_ce_loss_coef = float(appearance_full_ce_loss_coef)
        self.appearance_contrastive_loss_mode = str(
            appearance_contrastive_loss_mode or "candidate_ce"
        ).strip().lower()
        self.appearance_pairwise_loss_margin = float(appearance_pairwise_loss_margin)
        self.use_self_facb_loss = bool(use_self_facb_loss)
        self.self_facb_topk = max(int(self_facb_topk), 1)
        self.self_facb_ce_aux_weight = float(self_facb_ce_aux_weight)
        self.self_facb_pairwise_loss_weight = float(self_facb_pairwise_loss_weight)
        self.self_facb_pairwise_loss_margin = float(self_facb_pairwise_loss_margin)
        self.self_facb_force_gt_candidate = bool(self_facb_force_gt_candidate)
        self.self_facb_train_box_source = str(self_facb_train_box_source or "gt").strip().lower()
        self.self_facb_train_same_group_only = bool(self_facb_train_same_group_only)
        self.self_facb_semantic_top1_loss_weight = float(self_facb_semantic_top1_loss_weight)
        self.self_facb_semantic_top1_loss_margin = float(self_facb_semantic_top1_loss_margin)
        self.self_facb_hard_case_weight = float(self_facb_hard_case_weight)
        self.self_facb_hard_case_semantic_margin = float(self_facb_hard_case_semantic_margin)
        self.self_facb_same_group_hard_weight = float(self_facb_same_group_hard_weight)
        self.self_facb_head_ref = None
        self.facb_symbol_align_topk = max(int(facb_symbol_align_topk), 1)
        self.facb_symbol_align_temperature = max(float(facb_symbol_align_temperature), 1e-6)
        self.facb_symbol_align_pairwise_margin = float(facb_symbol_align_pairwise_margin)
        self.facb_symbol_align_pairwise_weight = float(facb_symbol_align_pairwise_weight)
        self.facb_symbol_align_inbatch_weight = float(facb_symbol_align_inbatch_weight)
        self.facb_symbol_align_hard_weight = float(facb_symbol_align_hard_weight)
        self.facb_symbol_align_easy_weight = float(facb_symbol_align_easy_weight)
        self.facb_symbol_align_error_only = bool(facb_symbol_align_error_only)
        self.facb_symbol_align_model_ref = None
        self.use_facb_text_orth_loss = bool(use_facb_text_orth_loss)
        self.facb_text_orth_margin = float(facb_text_orth_margin)
        self.facb_text_orth_anchor_weight = float(facb_text_orth_anchor_weight)
        self.facb_text_orth_model_ref = None
        self.facb_text_orth_label_list = None
        self.facb_text_orth_pairs = None
        self.facb_text_orth_pair_names = None
        self.facb_text_orth_anchor_proto = None
        self.use_facb_speedhr_loss = False
        self.facb_speedhr_head_ref = None
        self.facb_speedhr_global_to_digit = None
        self.facb_speedhr_crop_scale = 1.2
        self.facb_speedhr_digit_loss_weight = 1.0
        self.facb_speedhr_agf_loss_weight = 1.0
        self.facb_speedhr_agf_alpha = 2.0
        self.facb_speedhr_agf_min_extra = 0.0
        self.facb_speedhr_agf_weight_cap = 3.0
        self.facb_speedhr_agf_topk = 10
        self.facb_speedhr_agf_min_score = 0.0
        self.facb_speedhr_agf_max_score = 0.20
        self.facb_speedhr_agf_max_margin = 0.10
        self.facb_speedhr_agf_require_semantic_topk = True
        self.facb_speedhr_agf_require_crop_correct = False
        self.facb_speedhr_agf_min_crop_prob = 0.0
        self.facb_speedhr_agf_min_crop_margin = -1.0
        self.facb_speedhr_agf_weight_mode = "prob"
        self.facb_speedhr_allowed_global_ids = None
        self.facb_speedhr_allowed_digit_ids = None
        self.facb_speedhr_candidate_matching = False
        self.facb_speedhr_candidate_topk = 10
        self.facb_speedhr_candidate_smallshot_weight = 5.0
        self.facb_speedhr_candidate_anchor_weight = 0.05
        self.facb_speedhr_candidate_smallshot_hard_weight = 10.0
        self.facb_speedhr_candidate_anchor_hard_weight = 0.10
        self.facb_speedhr_candidate_query_source = "matcher"
        self.facb_speedhr_candidate_min_iou = 0.0
        self.facb_speedhr_candidate_crop_source = "pred"
        self.facb_speedhr_score_residual_loss_weight = 0.0
        self.facb_speedhr_score_residual_floor = 0.20
        self.facb_speedhr_score_residual_min_conf = 0.80
        self.facb_speedhr_score_residual_min_margin = 0.10
        self.facb_speedhr_score_residual_min_semantic_score = 0.0
        self.facb_speedhr_score_residual_max_semantic_score = 0.20
        self.facb_speedhr_score_residual_alpha_reg = 0.001
        self.facb_speedhr_score_residual_alpha_cap = 1.0
        self.facb_speedhr_score_residual_objective = "floor"
        self.facb_speedhr_score_residual_rank_margin = 0.02
        self.facb_speedhr_score_residual_rank_temperature = 1.0
        self.facb_speedhr_score_residual_soft_gate = False
        self.facb_speedhr_score_residual_gate_type = "prob"
        self.facb_speedhr_score_residual_start_epoch = 0
        self.facb_speedhr_score_residual_ramp_epochs = 1
        self.facb_speedhr_shared_facet_supervision_scope = "all"
        self.facb_speedhr_shared_symbol_hard_neg_weight = 0.0
        self.facb_speedhr_shared_symbol_hard_neg_from_candidates = False
        self.current_epoch = 0
        self.appearance_contrastive_same_group_only = bool(appearance_contrastive_same_group_only)
        self.appearance_contrastive_fallback_all = bool(appearance_contrastive_fallback_all)
        # Global [num_classes] fine-group id vector (class id order).  Used by the
        # prototype-bank appearance loss to mine same-fine-group hard negatives
        # (pl40 vs pl50/pl60...) across ALL classes, not just in-image classes.
        if appearance_global_fine_group_ids is not None:
            self.register_buffer(
                "appearance_global_fine_group_ids",
                torch.as_tensor(appearance_global_fine_group_ids, dtype=torch.long),
                persistent=False,
            )
        else:
            self.appearance_global_fine_group_ids = None
        # Pairwise logistic loss: skip queries where the semantic margin already
        # exceeds this threshold (semantic head is confident, no appearance
        # correction needed).  The actual value is injected after build via
        # criterion.appearance_pairwise_margin_filter_thr = getattr(args, ...).
        # Default 1e9 = no filtering (kept for backward compat).  NOTE: do NOT
        # self-reference getattr(self, ...) here — that was a dead no-op that
        # pinned the threshold to 1e9 regardless of config.
        if not hasattr(self, "appearance_pairwise_margin_filter_thr"):
            self.appearance_pairwise_margin_filter_thr = 1e9
        # Method B: same-group token-margin loss config (set after build).
        if not hasattr(self, "semantic_group_margin"):
            self.semantic_group_margin = 0.2
        if not hasattr(self, "semantic_group_margin_topk"):
            self.semantic_group_margin_topk = 1
        self.appearance_contrastive_sem_confuser_mode = bool(
            appearance_contrastive_sem_confuser_mode
        )
        self._appearance_cls_debug_printed = False
        # ---- Phase 2: matcher class-cost IDF reweighting ----
        # When > 0, the label_map passed to the matcher is IDF-weighted (same DF
        # table / mechanism as PostProcess scoring), so the matcher's class cost
        # stops uniform-averaging a class's tokens and amplifies discriminative
        # ones (e.g. "80" vs "60").  λ=0 => binary label_map => exact current
        # matcher behavior.  DF table is injected after build via main.py.
        # IMPORTANT: this affects ONLY the matcher (assignment, no_grad).  The
        # loss one_hot uses the untouched binary label_map_list, so token-level
        # BCE/focal targets are completely unchanged.
        if not hasattr(self, "matcher_token_idf_lambda"):
            self.matcher_token_idf_lambda = float(
                getattr(self, "matcher_token_idf_lambda", 0.0)
            )
        if not hasattr(self, "matcher_token_idf_gamma"):
            self.matcher_token_idf_gamma = float(
                getattr(self, "matcher_token_idf_gamma", 1.0)
            )
        self._matcher_token_df = None
        self._matcher_token_df_num_classes = 0
        self._matcher_idf_debug_printed = False

    def _matcher_label_map(self, binary_label_map, tokenized):
        """Return the label_map to feed the matcher: IDF-weighted when Phase 2 is on.

        binary_label_map: [C, max_text_len] 0/1 membership for this image's caption.
        tokenized: the per-image tokenizer output (token['j']) for token-id lookup.
        λ=0 or no DF table => returns the binary map unchanged (exact baseline matcher).
        Only the matcher (no_grad assignment) sees this; loss one_hot is unaffected.
        """
        lam = float(getattr(self, "matcher_token_idf_lambda", 0.0))
        gamma = float(getattr(self, "matcher_token_idf_gamma", 1.0))
        if lam == 0.0 or self._matcher_token_df is None or binary_label_map.shape[0] == 0:
            return binary_label_map
        if not self._matcher_idf_debug_printed:
            print(
                f"[matcher-idf] lambda={lam} gamma={gamma} active; DF over "
                f"{self._matcher_token_df_num_classes} classes, "
                f"{len(self._matcher_token_df)} distinct token ids"
            )
            self._matcher_idf_debug_printed = True
        return idf_weight_label_map(
            binary_label_map,
            tokenized,
            self._matcher_token_df,
            self._matcher_token_df_num_classes,
            lam,
            gamma,
        )

    @staticmethod
    def _map_labels_to_score_space(target_entry, target_labels, num_score_classes, device):
        target_labels = target_labels.to(device=device, dtype=torch.long)
        if target_labels.numel() == 0:
            return target_labels

        local_mask = (target_labels >= 0) & (target_labels < num_score_classes)
        if bool(local_mask.all()):
            return target_labels

        if os.environ.get("APPEARANCE_CLS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            caption_label_ids_dbg = target_entry.get("caption_label_ids", None)
            print(
                "[appearance-cls-map-debug] "
                f"labels={target_labels.detach().cpu().tolist()} "
                f"num_score_classes={num_score_classes} "
                f"caption_label_ids={list(caption_label_ids_dbg) if caption_label_ids_dbg is not None else None}"
            )

        caption_label_ids = target_entry.get("caption_label_ids", None)
        if not caption_label_ids:
            return target_labels

        global_to_local = {}
        for local_idx, label_id in enumerate(caption_label_ids):
            try:
                global_to_local[int(label_id)] = local_idx
            except (TypeError, ValueError):
                continue

        if not global_to_local:
            return target_labels.new_full(target_labels.shape, -1, dtype=torch.long, device=device)

        mapped = [global_to_local.get(int(label), -1) for label in target_labels.detach().cpu().tolist()]
        return torch.as_tensor(mapped, device=device, dtype=torch.long)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes = torch.nan_to_num(src_boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target_boxes = torch.nan_to_num(target_boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        src_boxes = torch.cat([src_boxes[:, :2], src_boxes[:, 2:].clamp(min=1e-6)], dim=-1)
        target_boxes = torch.cat([target_boxes[:, :2], target_boxes[:, 2:].clamp(min=1e-6)], dim=-1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_quality(self, outputs, targets, indices, num_boxes):
        """L1 loss between predicted quality scores and IoU targets.

        Only matched queries are supervised. Classification remains the main
        semantic signal; the quality head only learns how reliable a matched box
        is for ranking.
        """
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_quality': outputs['pred_quality'].sum() * 0.0}

        quality_pred = outputs['pred_quality'][idx].sigmoid().squeeze(-1)
        quality_target = outputs['quality_target'][idx].squeeze(-1)
        quality_pred = torch.nan_to_num(quality_pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        quality_target = torch.nan_to_num(quality_target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        loss = F.l1_loss(quality_pred, quality_target, reduction='sum')
        return {'loss_quality': loss / num_boxes}

    def loss_lgpr_survival(self, outputs, targets, indices, num_boxes):
        """Class-agnostic per-GT query-survival objective.

        Unlike the legacy quality loss, this objective is not restricted to
        Hungarian-matched queries.  It (1) learns continuous IoU objectness from
        per-GT positives and semantic hard negatives, and (2) pushes at least one
        query for every GT above the approximate final top-K query boundary.
        Detector logits and boxes are detached here: the first LGPR-survival
        stage trains only the small survival head and cannot perturb the baseline.
        """
        survival_all = outputs.get("pred_lgpr_survival_all", None)
        final_survival = outputs.get("pred_lgpr_survival", None)
        if survival_all is None and final_survival is None:
            return {"loss_lgpr_survival": outputs["pred_boxes"].sum() * 0.0}

        if survival_all is None:
            survival_all = final_survival.unsqueeze(0)
        if survival_all.dim() == 3:
            survival_all = survival_all.unsqueeze(0)

        layer_boxes = []
        layer_logits = []
        for aux_output in outputs.get("aux_outputs", []):
            if "pred_boxes" in aux_output and "pred_logits" in aux_output:
                layer_boxes.append(aux_output["pred_boxes"])
                layer_logits.append(aux_output["pred_logits"])
        layer_boxes.append(outputs["pred_boxes"])
        layer_logits.append(outputs["pred_logits"])

        if int(survival_all.shape[0]) > len(layer_boxes):
            survival_all = survival_all[-len(layer_boxes):]

        available_layers = min(
            int(survival_all.shape[0]), len(layer_boxes), len(layer_logits)
        )
        if available_layers <= 0:
            return {"loss_lgpr_survival": survival_all.sum() * 0.0}

        last_n = max(1, int(getattr(self, "lgpr_survival_aux_last_n", 1)))
        start_layer = max(0, available_layers - last_n)
        pos_iou_thr = float(getattr(self, "lgpr_survival_pos_iou_thr", 0.5))
        small_pos_iou_thr = float(
            getattr(self, "lgpr_survival_small_pos_iou_thr", 0.3)
        )
        neg_iou_thr = float(getattr(self, "lgpr_survival_neg_iou_thr", 0.1))
        hard_neg_num = max(
            1, int(getattr(self, "lgpr_survival_num_hard_neg", 64))
        )
        selection_topk = max(
            1, int(getattr(self, "lgpr_survival_selection_topk", 100))
        )
        rank_margin = float(getattr(self, "lgpr_survival_rank_margin", 0.1))
        temperature = max(
            float(getattr(self, "lgpr_survival_temperature", 0.2)), 1e-4
        )
        score_weight = float(
            getattr(self, "lgpr_survival_score_weight", 0.2)
        )
        objectness_coef = float(
            getattr(self, "lgpr_survival_objectness_coef", 0.25)
        )
        coverage_coef = float(
            getattr(self, "lgpr_survival_coverage_coef", 1.0)
        )
        small_gt_weight = float(
            getattr(self, "lgpr_survival_small_gt_weight", 2.0)
        )
        aux_decay = min(
            max(float(getattr(self, "lgpr_survival_aux_decay", 0.5)), 0.0),
            1.0,
        )

        total_loss = survival_all.sum() * 0.0
        total_layer_weight = 0.0
        text_mask = outputs.get("text_mask", None)

        for layer_idx in range(start_layer, available_layers):
            survival_logits = survival_all[layer_idx].squeeze(-1)
            pred_boxes = layer_boxes[layer_idx].detach()
            pred_token_logits = layer_logits[layer_idx].detach()
            layer_objectness = survival_logits.sum() * 0.0
            layer_coverage = survival_logits.sum() * 0.0
            objectness_batches = 0
            coverage_weight_sum = 0.0

            for batch_idx, target in enumerate(targets):
                gt_boxes = target.get("boxes", None)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue
                gt_boxes = gt_boxes.to(
                    device=pred_boxes.device, dtype=pred_boxes.dtype
                ).detach()
                batch_boxes = torch.nan_to_num(
                    pred_boxes[batch_idx], nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                batch_boxes = torch.cat(
                    [batch_boxes[:, :2], batch_boxes[:, 2:].clamp_min(1e-6)],
                    dim=-1,
                )
                gt_boxes = torch.nan_to_num(
                    gt_boxes, nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                gt_boxes = torch.cat(
                    [gt_boxes[:, :2], gt_boxes[:, 2:].clamp_min(1e-6)],
                    dim=-1,
                )
                iou = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(batch_boxes),
                    box_ops.box_cxcywh_to_xyxy(gt_boxes),
                )[0].detach()
                if iou.numel() == 0 or not torch.isfinite(iou).all():
                    continue

                max_iou, best_gt_for_query = iou.max(dim=1)
                batch_survival = survival_logits[batch_idx]
                token_logits = torch.nan_to_num(
                    pred_token_logits[batch_idx],
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ).clamp(-20.0, 20.0)
                if text_mask is not None:
                    valid_tokens = text_mask[batch_idx, : token_logits.shape[-1]].bool()
                    if bool(valid_tokens.any()):
                        token_logits = token_logits.masked_fill(
                            ~valid_tokens.unsqueeze(0), -20.0
                        )
                base_query_score = token_logits.sigmoid().max(dim=-1).values.detach()
                fused_query_score = (
                    torch.log(base_query_score.clamp_min(1e-6))
                    + score_weight * torch.tanh(batch_survival)
                )

                orig_size = target.get("orig_size", None)
                if orig_size is not None and torch.as_tensor(orig_size).numel() >= 2:
                    orig_size_tensor = torch.as_tensor(
                        orig_size, device=gt_boxes.device, dtype=gt_boxes.dtype
                    ).flatten()
                    orig_h, orig_w = orig_size_tensor[0], orig_size_tensor[1]
                    gt_pixel_area = (
                        gt_boxes[:, 2] * orig_w * gt_boxes[:, 3] * orig_h
                    )
                else:
                    gt_pixel_area = gt_boxes[:, 2] * gt_boxes[:, 3] * (1600.0 ** 2)
                small_gt = gt_pixel_area < (32.0 ** 2)
                gt_weights = torch.where(
                    small_gt,
                    gt_pixel_area.new_full(gt_pixel_area.shape, small_gt_weight),
                    gt_pixel_area.new_ones(gt_pixel_area.shape),
                )

                # Continuous objectness on per-GT positives plus semantic hard
                # negatives.  Best-per-GT inclusion prevents an easy object from
                # monopolizing all positives.
                positive_mask = max_iou >= pos_iou_thr
                best_query_per_gt = iou.argmax(dim=0)
                positive_mask[best_query_per_gt] = True
                negative_indices = torch.nonzero(
                    max_iou <= neg_iou_thr, as_tuple=False
                ).flatten()
                if negative_indices.numel() > 0:
                    hard_count = min(hard_neg_num, int(negative_indices.numel()))
                    hard_negative_indices = negative_indices[
                        torch.topk(
                            base_query_score[negative_indices], hard_count
                        ).indices
                    ]
                else:
                    hard_negative_indices = negative_indices
                positive_indices = torch.nonzero(
                    positive_mask, as_tuple=False
                ).flatten()
                selected_indices = torch.unique(
                    torch.cat([positive_indices, hard_negative_indices], dim=0)
                )
                if selected_indices.numel() > 0:
                    selected_iou = max_iou[selected_indices].clamp(0.0, 1.0)
                    objectness_targets = torch.where(
                        selected_iou <= neg_iou_thr,
                        torch.zeros_like(selected_iou),
                        selected_iou,
                    )
                    objectness_weights = torch.ones_like(objectness_targets)
                    selected_positive = max_iou[selected_indices] > neg_iou_thr
                    if bool(selected_positive.any()):
                        selected_gt = best_gt_for_query[selected_indices[selected_positive]]
                        objectness_weights[selected_positive] = gt_weights[selected_gt]
                    objectness_loss = F.binary_cross_entropy_with_logits(
                        batch_survival[selected_indices],
                        objectness_targets,
                        reduction="none",
                    )
                    layer_objectness = layer_objectness + (
                        objectness_loss * objectness_weights
                    ).sum() / objectness_weights.sum().clamp_min(1.0)
                    objectness_batches += 1

                # Approximate the final top-K query boundary.  The semantic base
                # score is detached; gradients change only survival utilities.
                boundary_k = min(selection_topk, int(fused_query_score.numel()))
                boundary = torch.topk(fused_query_score, boundary_k).values[-1]
                query_centers = batch_boxes[:, :2]
                gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                center_inside = (
                    (query_centers[:, None, 0] >= gt_xyxy[None, :, 0])
                    & (query_centers[:, None, 0] <= gt_xyxy[None, :, 2])
                    & (query_centers[:, None, 1] >= gt_xyxy[None, :, 1])
                    & (query_centers[:, None, 1] <= gt_xyxy[None, :, 3])
                )
                for gt_idx in range(gt_boxes.shape[0]):
                    if bool(small_gt[gt_idx]):
                        eligible = (
                            (iou[:, gt_idx] >= small_pos_iou_thr)
                            & center_inside[:, gt_idx]
                        )
                    else:
                        eligible = iou[:, gt_idx] >= pos_iou_thr
                    if not bool(eligible.any()):
                        # A final-stage score head cannot repair a box that no
                        # longer overlaps the GT.  Such decoder-drift cases need
                        # the separate localization-preservation extension.
                        continue
                    positive_scores = fused_query_score[eligible]
                    smooth_positive = temperature * (
                        torch.logsumexp(positive_scores / temperature, dim=0)
                        - math.log(max(int(positive_scores.numel()), 1))
                    )
                    gt_loss = F.softplus(
                        (boundary + rank_margin - smooth_positive) / temperature
                    ) * temperature
                    gt_weight = float(gt_weights[gt_idx].item())
                    layer_coverage = layer_coverage + gt_weight * gt_loss
                    coverage_weight_sum += gt_weight

            if objectness_batches > 0:
                layer_objectness = layer_objectness / float(objectness_batches)
            if coverage_weight_sum > 0.0:
                layer_coverage = layer_coverage / coverage_weight_sum
            layer_loss = (
                objectness_coef * layer_objectness
                + coverage_coef * layer_coverage
            )
            distance_from_final = available_layers - 1 - layer_idx
            layer_weight = aux_decay ** distance_from_final
            total_loss = total_loss + layer_weight * layer_loss
            total_layer_weight += layer_weight

        if total_layer_weight <= 0.0:
            return {"loss_lgpr_survival": survival_all.sum() * 0.0}
        return {"loss_lgpr_survival": total_loss / total_layer_weight}

    def loss_lgpr_loc_preserve(self, outputs, targets, indices, num_boxes):
        """High-IoU localization preservation for decoder queries.

        The diagnostic shows that recall survives to top-100 at IoU@0.5, while
        IoU@0.75 is capped by decoder box quality. This loss selects queries that
        already overlap a GT and refines them toward that GT. Selection is based
        on detached IoU; gradients flow only through the selected predicted boxes.
        """
        layer_boxes = []
        for aux_output in outputs.get("aux_outputs", []):
            if "pred_boxes" in aux_output:
                layer_boxes.append(aux_output["pred_boxes"])
        layer_boxes.append(outputs["pred_boxes"])
        if not layer_boxes:
            return {"loss_lgpr_loc_preserve": outputs["pred_boxes"].sum() * 0.0}

        last_n = max(1, int(getattr(self, "lgpr_loc_aux_last_n", 2)))
        start_layer = max(0, len(layer_boxes) - last_n)
        pos_iou_thr = float(getattr(self, "lgpr_loc_pos_iou_thr", 0.5))
        target_iou = float(getattr(self, "lgpr_loc_target_iou", 0.75))
        max_pos_per_gt = max(1, int(getattr(self, "lgpr_loc_max_pos_per_gt", 3)))
        small_gt_weight = float(getattr(self, "lgpr_loc_small_gt_weight", 2.0))
        lowfreq_gt_weight = float(getattr(self, "lgpr_loc_lowfreq_gt_weight", 1.0))
        l1_coef = float(getattr(self, "lgpr_loc_l1_coef", 1.0))
        giou_coef = float(getattr(self, "lgpr_loc_giou_coef", 1.0))
        iou75_coef = float(getattr(self, "lgpr_loc_iou75_coef", 1.0))
        aux_decay = min(max(float(getattr(self, "lgpr_loc_aux_decay", 0.5)), 0.0), 1.0)
        lowfreq_label_ids = getattr(self, "lgpr_loc_lowfreq_label_ids", None)
        if lowfreq_label_ids is None:
            lowfreq_label_ids = set()
        else:
            if torch.is_tensor(lowfreq_label_ids):
                lowfreq_label_ids = lowfreq_label_ids.detach().cpu().flatten().tolist()
            elif isinstance(lowfreq_label_ids, str):
                lowfreq_label_ids = [
                    item.strip() for item in lowfreq_label_ids.split(",") if item.strip()
                ]
            lowfreq_label_ids = {
                int(value) for value in lowfreq_label_ids if int(value) >= 0
            }

        total_loss = outputs["pred_boxes"].sum() * 0.0
        total_layer_weight = 0.0
        total_selected = 0

        for layer_idx in range(start_layer, len(layer_boxes)):
            pred_boxes = layer_boxes[layer_idx]
            layer_loss = pred_boxes.sum() * 0.0
            layer_weight_sum = pred_boxes.new_tensor(0.0)

            for batch_idx, target in enumerate(targets):
                gt_boxes = target.get("boxes", None)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue
                gt_boxes = gt_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
                batch_boxes = torch.nan_to_num(
                    pred_boxes[batch_idx], nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                batch_boxes = torch.cat(
                    [batch_boxes[:, :2], batch_boxes[:, 2:].clamp_min(1e-6)],
                    dim=-1,
                )
                gt_boxes = torch.nan_to_num(
                    gt_boxes, nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                gt_boxes = torch.cat(
                    [gt_boxes[:, :2], gt_boxes[:, 2:].clamp_min(1e-6)],
                    dim=-1,
                )

                with torch.no_grad():
                    iou_detached = box_ops.box_iou(
                        box_ops.box_cxcywh_to_xyxy(batch_boxes.detach()),
                        box_ops.box_cxcywh_to_xyxy(gt_boxes.detach()),
                    )[0]
                    orig_size = target.get("orig_size", None)
                    if orig_size is not None and torch.as_tensor(orig_size).numel() >= 2:
                        orig_size_tensor = torch.as_tensor(
                            orig_size, device=gt_boxes.device, dtype=gt_boxes.dtype
                        ).flatten()
                        orig_h, orig_w = orig_size_tensor[0], orig_size_tensor[1]
                        gt_pixel_area = gt_boxes[:, 2] * orig_w * gt_boxes[:, 3] * orig_h
                    else:
                        gt_pixel_area = gt_boxes[:, 2] * gt_boxes[:, 3] * (1600.0 ** 2)
                    small_gt = gt_pixel_area < (32.0 ** 2)
                    gt_labels = target.get("labels", None)
                    if gt_labels is not None:
                        gt_labels = gt_labels.to(device=gt_boxes.device, dtype=torch.long)

                selected_query_ids = []
                selected_gt_ids = []
                selected_weights = []
                for gt_idx in range(gt_boxes.shape[0]):
                    eligible = torch.nonzero(
                        iou_detached[:, gt_idx] >= pos_iou_thr,
                        as_tuple=False,
                    ).flatten()
                    if eligible.numel() == 0:
                        continue
                    k = min(max_pos_per_gt, int(eligible.numel()))
                    local = torch.topk(iou_detached[eligible, gt_idx], k=k, dim=0).indices
                    chosen = eligible[local]
                    gt_weight = small_gt_weight if bool(small_gt[gt_idx].item()) else 1.0
                    if (
                        gt_labels is not None
                        and lowfreq_label_ids
                        and int(gt_labels[gt_idx].item()) in lowfreq_label_ids
                    ):
                        gt_weight *= lowfreq_gt_weight
                    selected_query_ids.append(chosen)
                    selected_gt_ids.append(
                        torch.full_like(chosen, int(gt_idx), dtype=torch.long)
                    )
                    selected_weights.append(
                        chosen.new_full((chosen.numel(),), float(gt_weight), dtype=pred_boxes.dtype)
                    )

                if not selected_query_ids:
                    continue
                selected_query_ids = torch.cat(selected_query_ids, dim=0)
                selected_gt_ids = torch.cat(selected_gt_ids, dim=0)
                selected_weights = torch.cat(selected_weights, dim=0).to(
                    device=pred_boxes.device, dtype=pred_boxes.dtype
                )
                src_boxes = batch_boxes[selected_query_ids]
                tgt_boxes = gt_boxes[selected_gt_ids]
                src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
                tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes)
                l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum(dim=-1)
                giou = torch.diag(box_ops.generalized_box_iou(src_xyxy, tgt_xyxy))
                iou = torch.diag(box_ops.box_iou(src_xyxy, tgt_xyxy)[0])
                iou75_loss = F.relu(src_boxes.new_tensor(target_iou) - iou)
                loc_loss = l1_coef * l1_loss + giou_coef * (1.0 - giou) + iou75_coef * iou75_loss
                layer_loss = layer_loss + (loc_loss * selected_weights).sum()
                layer_weight_sum = layer_weight_sum + selected_weights.sum()
                total_selected += int(selected_query_ids.numel())

            if float(layer_weight_sum.detach().item()) > 0.0:
                layer_loss = layer_loss / layer_weight_sum.clamp_min(1.0)
                distance_from_final = len(layer_boxes) - 1 - layer_idx
                layer_weight = aux_decay ** distance_from_final
                total_loss = total_loss + layer_weight * layer_loss
                total_layer_weight += layer_weight

        if total_layer_weight <= 0.0 or total_selected <= 0:
            return {"loss_lgpr_loc_preserve": outputs["pred_boxes"].sum() * 0.0}
        return {"loss_lgpr_loc_preserve": total_loss / total_layer_weight}

    def loss_lgpr_box_refine(self, outputs, targets, indices, num_boxes):
        """Train only the residual box-refine head on medium-quality GT covers.

        Selection uses the frozen baseline final boxes.  Gradients flow through
        the refined boxes only, so the module learns a small correction instead
        of changing query routing, classification, or the shared bbox_embed.
        """
        pred_boxes = outputs.get("pred_boxes", None)
        base_boxes = outputs.get("pred_boxes_lgpr_base", None)
        if pred_boxes is None or base_boxes is None:
            zero_ref = pred_boxes if pred_boxes is not None else outputs["pred_logits"]
            return {"loss_lgpr_box_refine": zero_ref.sum() * 0.0}

        pos_iou_low = float(getattr(self, "lgpr_box_refine_pos_iou_low", 0.5))
        pos_iou_high = float(getattr(self, "lgpr_box_refine_pos_iou_high", 0.85))
        target_iou = float(getattr(self, "lgpr_box_refine_target_iou", 0.75))
        max_pos_per_gt = max(1, int(getattr(self, "lgpr_box_refine_max_pos_per_gt", 2)))
        small_gt_weight = float(getattr(self, "lgpr_box_refine_small_gt_weight", 2.0))
        lowfreq_gt_weight = float(getattr(self, "lgpr_box_refine_lowfreq_gt_weight", 1.0))
        l1_coef = float(getattr(self, "lgpr_box_refine_l1_coef", 1.0))
        giou_coef = float(getattr(self, "lgpr_box_refine_giou_coef", 1.0))
        iou75_coef = float(getattr(self, "lgpr_box_refine_iou75_coef", 1.0))
        lowfreq_label_ids = getattr(self, "lgpr_box_refine_lowfreq_label_ids", None)
        if lowfreq_label_ids is None:
            lowfreq_label_ids = set()
        else:
            if torch.is_tensor(lowfreq_label_ids):
                lowfreq_label_ids = lowfreq_label_ids.detach().cpu().flatten().tolist()
            elif isinstance(lowfreq_label_ids, str):
                lowfreq_label_ids = [
                    item.strip() for item in lowfreq_label_ids.split(",") if item.strip()
                ]
            lowfreq_label_ids = {
                int(value) for value in lowfreq_label_ids if int(value) >= 0
            }

        total_loss = pred_boxes.sum() * 0.0
        total_weight = pred_boxes.new_tensor(0.0)
        total_selected = 0

        for batch_idx, target in enumerate(targets):
            gt_boxes = target.get("boxes", None)
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            gt_boxes = gt_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
            gt_boxes = torch.nan_to_num(
                gt_boxes, nan=0.5, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            gt_boxes = torch.cat(
                [gt_boxes[:, :2], gt_boxes[:, 2:].clamp_min(1e-6)],
                dim=-1,
            )
            refined = torch.nan_to_num(
                pred_boxes[batch_idx], nan=0.5, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            refined = torch.cat(
                [refined[:, :2], refined[:, 2:].clamp_min(1e-6)],
                dim=-1,
            )
            base = torch.nan_to_num(
                base_boxes[batch_idx], nan=0.5, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            base = torch.cat(
                [base[:, :2], base[:, 2:].clamp_min(1e-6)],
                dim=-1,
            )

            with torch.no_grad():
                base_iou = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(base),
                    box_ops.box_cxcywh_to_xyxy(gt_boxes),
                )[0]
                orig_size = target.get("orig_size", None)
                if orig_size is not None and torch.as_tensor(orig_size).numel() >= 2:
                    orig_size_tensor = torch.as_tensor(
                        orig_size, device=gt_boxes.device, dtype=gt_boxes.dtype
                    ).flatten()
                    orig_h, orig_w = orig_size_tensor[0], orig_size_tensor[1]
                    gt_pixel_area = gt_boxes[:, 2] * orig_w * gt_boxes[:, 3] * orig_h
                else:
                    gt_pixel_area = gt_boxes[:, 2] * gt_boxes[:, 3] * (1600.0 ** 2)
                small_gt = gt_pixel_area < (32.0 ** 2)
                gt_labels = target.get("labels", None)
                if gt_labels is not None:
                    gt_labels = gt_labels.to(device=gt_boxes.device, dtype=torch.long)

            selected_query_ids = []
            selected_gt_ids = []
            selected_weights = []
            for gt_idx in range(gt_boxes.shape[0]):
                eligible = torch.nonzero(
                    (base_iou[:, gt_idx] >= pos_iou_low)
                    & (base_iou[:, gt_idx] < pos_iou_high),
                    as_tuple=False,
                ).flatten()
                if eligible.numel() == 0:
                    continue
                k = min(max_pos_per_gt, int(eligible.numel()))
                local = torch.topk(base_iou[eligible, gt_idx], k=k, dim=0).indices
                chosen = eligible[local]
                gt_weight = small_gt_weight if bool(small_gt[gt_idx].item()) else 1.0
                if (
                    gt_labels is not None
                    and lowfreq_label_ids
                    and int(gt_labels[gt_idx].item()) in lowfreq_label_ids
                ):
                    gt_weight *= lowfreq_gt_weight
                selected_query_ids.append(chosen)
                selected_gt_ids.append(torch.full_like(chosen, int(gt_idx), dtype=torch.long))
                selected_weights.append(
                    chosen.new_full((chosen.numel(),), float(gt_weight), dtype=pred_boxes.dtype)
                )

            if not selected_query_ids:
                continue

            selected_query_ids = torch.cat(selected_query_ids, dim=0)
            selected_gt_ids = torch.cat(selected_gt_ids, dim=0)
            selected_weights = torch.cat(selected_weights, dim=0).to(
                device=pred_boxes.device, dtype=pred_boxes.dtype
            )
            src_boxes = refined[selected_query_ids]
            tgt_boxes = gt_boxes[selected_gt_ids]
            src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes)
            l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum(dim=-1)
            giou = torch.diag(box_ops.generalized_box_iou(src_xyxy, tgt_xyxy))
            iou = torch.diag(box_ops.box_iou(src_xyxy, tgt_xyxy)[0])
            iou75_loss = F.relu(src_boxes.new_tensor(target_iou) - iou)
            refine_loss = (
                l1_coef * l1_loss
                + giou_coef * (1.0 - giou)
                + iou75_coef * iou75_loss
            )
            total_loss = total_loss + (refine_loss * selected_weights).sum()
            total_weight = total_weight + selected_weights.sum()
            total_selected += int(selected_query_ids.numel())

        if total_selected <= 0 or float(total_weight.detach().item()) <= 0.0:
            return {"loss_lgpr_box_refine": pred_boxes.sum() * 0.0}
        return {"loss_lgpr_box_refine": total_loss / total_weight.clamp_min(1.0)}

    def loss_visual_confidence(self, outputs, targets, indices, num_boxes):
        """BCE loss for class-agnostic visual confidence head (v1.1).

        Positives: Hungarian-matched queries -> target 1.0.
        Hard negatives: unmatched, max_IoU < 0.1, semantic-score top-k -> target 0.0.
        Ambiguous: unmatched, 0.1 <= max_IoU < 0.5 -> target -1 (ignore).
        """
        visual_confidence = outputs.get("pred_visual_confidence", None)
        pred_boxes = outputs.get("pred_boxes", None)
        if visual_confidence is None or pred_boxes is None:
            return {"loss_visual_confidence": 0.0 * num_boxes}

        bs, num_queries, _ = visual_confidence.shape
        device = visual_confidence.device

        target = torch.full((bs, num_queries, 1), -1.0, device=device)

        # ---- Positives: matched queries ----
        for batch_idx, (src_idx, _tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target[batch_idx, src_idx, 0] = 1.0

        # ---- IoU-based negative filtering ----
        pred_logits = outputs["pred_logits"]
        with torch.no_grad():
            semantic_score = pred_logits.sigmoid().max(dim=-1).values

        for batch_idx in range(bs):
            src_idx = indices[batch_idx][0]
            gt_boxes = targets[batch_idx].get("boxes", None)
            matched = set(src_idx.cpu().tolist()) if len(src_idx) > 0 else set()
            unmatched = [i for i in range(num_queries) if i not in matched]
            if not unmatched:
                continue
            unmatched_tensor = torch.tensor(unmatched, device=device)

            # Compute max IoU between each unmatched query and all GT boxes
            if gt_boxes is not None and gt_boxes.numel() > 0:
                query_boxes = pred_boxes[batch_idx, unmatched_tensor]  # [U, 4]
                iou = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(query_boxes),
                    box_ops.box_cxcywh_to_xyxy(gt_boxes.to(device=device, dtype=query_boxes.dtype)),
                )[0]  # [U, num_gt]
                max_iou_per_unmatched = iou.max(dim=1).values  # [U]
            else:
                max_iou_per_unmatched = query_boxes.new_zeros(len(unmatched))

            # Candidates for negative: IoU < 0.1
            neg_eligible_mask = max_iou_per_unmatched < 0.1
            if neg_eligible_mask.any():
                neg_eligible_indices = unmatched_tensor[neg_eligible_mask]
                neg_eligible_scores = semantic_score[batch_idx, neg_eligible_indices]
                num_pos = len(src_idx)
                k = min(len(neg_eligible_indices), max(num_pos * 2, 1))
                _, topk_local = torch.topk(neg_eligible_scores, k)
                hard_neg_indices = neg_eligible_indices[topk_local]
                target[batch_idx, hard_neg_indices, 0] = 0.0

        # ---- Compute loss ----
        valid_mask = target >= 0.0
        if valid_mask.sum() == 0:
            return {"loss_visual_confidence": visual_confidence.sum() * 0.0}

        valid_pred = visual_confidence[valid_mask].squeeze(-1)
        valid_target = target[valid_mask].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(valid_pred, valid_target, reduction="sum")
        num_pos = sum(len(indices[i][0]) for i in range(bs))
        return {"loss_visual_confidence": loss / max(num_pos, 1)}

    def loss_appearance_gate(self, outputs, targets, indices, num_boxes):
        """Cross-entropy loss for class-wise appearance-conditioned gate scores.

        Only Hungarian-matched queries contribute. Maps GT class labels → prototype
        indices via caption_label_ids. Crops padding classes that arise from
        batch-padding different prototype counts across batch items.
        """
        appearance_scores = outputs.get("pred_appearance_gate_scores", None)
        if appearance_scores is None:
            return {"loss_appearance_gate": 0.0 * num_boxes}

        bs = appearance_scores.shape[0]
        total_loss = appearance_scores.new_tensor(0.0)
        total_valid = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            gt_labels = targets[batch_idx]["labels"][tgt_idx]  # class indices

            # Map GT class labels → prototype indices using caption_label_ids
            label_ids_list = targets[batch_idx].get("caption_label_ids", None)
            if label_ids_list is not None:
                num_proto = len(label_ids_list)
                label_to_proto = {}
                for proto_idx, class_id in enumerate(label_ids_list):
                    label_to_proto[int(class_id)] = proto_idx
                proto_targets = [label_to_proto.get(int(lbl), -1) for lbl in gt_labels.tolist()]
                proto_targets = gt_labels.new_tensor(proto_targets)
            else:
                num_proto = appearance_scores.shape[-1]
                proto_targets = gt_labels.to(dtype=torch.long)

            # Crop to actual prototype count for this item (removes batch-padding zeros).
            # Some appearance texts may be dropped/truncated before prototype construction,
            # so also cap by the score tensor's actual class dimension.
            num_score_classes = min(int(num_proto), int(appearance_scores.shape[-1]))
            if num_score_classes <= 0:
                continue
            scores = appearance_scores[batch_idx, src_idx, :num_score_classes]

            valid = (proto_targets >= 0) & (proto_targets < num_score_classes)
            if not valid.any():
                continue
            loss = F.cross_entropy(
                scores[valid], proto_targets[valid], reduction="sum"
            )
            total_loss = total_loss + loss
            total_valid = total_valid + valid.sum().item()

        if total_valid == 0:
            return {"loss_appearance_gate": appearance_scores.sum() * 0.0}

        return {"loss_appearance_gate": total_loss / max(num_boxes, 1)}

    def loss_query_contrastive(self, outputs, targets, indices, num_boxes):
        """Contrast matched query features against per-class text prototypes."""
        query_features = outputs.get("query_features")
        category_prototypes = outputs.get(
            "query_contrastive_text_prototypes",
            outputs.get("category_text_prototypes"),
        )
        prototype_valid_masks = outputs.get(
            "query_contrastive_prototype_valid_mask",
            outputs.get("category_prototype_valid_mask"),
        )
        if query_features is None or category_prototypes is None or prototype_valid_masks is None:
            zero_ref = outputs["pred_logits"]
            return {"loss_query_contrastive": zero_ref.sum() * 0.0}

        total_loss = query_features.new_tensor(0.0)
        total_matches = 0
        temperature = max(self.query_contrastive_temperature, 1e-6)
        use_sigmoid_mode = self.query_contrastive_mode == "sigmoid"

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            batch_queries = query_features[batch_idx, src_idx]
            batch_labels = targets[batch_idx]["labels"][tgt_idx].to(
                device=batch_queries.device, dtype=torch.long
            )
            batch_prototypes = category_prototypes[batch_idx].to(batch_queries.device)
            batch_valid = prototype_valid_masks[batch_idx].to(
                device=batch_queries.device, dtype=torch.bool
            )
            if batch_prototypes.numel() == 0 or not batch_valid.any():
                continue

            valid_positions = torch.nonzero(batch_valid, as_tuple=False).squeeze(-1)
            valid_position_map = torch.full(
                (batch_valid.shape[0],),
                -1,
                dtype=torch.long,
                device=batch_queries.device,
            )
            valid_position_map[valid_positions] = torch.arange(
                valid_positions.numel(), device=batch_queries.device
            )

            if batch_labels.numel() == 0:
                continue
            legal_mask = (batch_labels >= 0) & (batch_labels < valid_position_map.shape[0])
            legal_count = int(legal_mask.sum().item())
            if legal_count == 0:
                continue

            batch_queries = batch_queries[legal_mask]
            batch_labels = batch_labels[legal_mask]
            target_positions = valid_position_map[batch_labels]
            keep_mask = target_positions >= 0
            keep_count = int(keep_mask.sum().item())
            if keep_count == 0:
                continue

            batch_queries = batch_queries[keep_mask]
            target_positions = target_positions[keep_mask]
            batch_prototypes = batch_prototypes[valid_positions]
            valid_group_ids = None
            cap_group_ids = targets[batch_idx].get("cap_group_ids", None)
            if cap_group_ids is not None:
                cap_group_ids = cap_group_ids.to(device=batch_queries.device, dtype=torch.long)
                if cap_group_ids.numel() == batch_valid.shape[0]:
                    valid_group_ids = cap_group_ids[valid_positions]

            batch_queries = F.normalize(batch_queries, dim=-1, eps=1e-6)
            batch_prototypes = F.normalize(batch_prototypes, dim=-1, eps=1e-6)
            logits = (batch_queries @ batch_prototypes.transpose(0, 1)) / temperature
            logits = logits + self.query_contrastive_bias

            if use_sigmoid_mode:
                target_matrix = torch.zeros_like(logits)
                target_matrix[
                    torch.arange(batch_queries.shape[0], device=logits.device), target_positions
                ] = 1.0

                pair_mask = torch.ones_like(logits, dtype=torch.bool)
                if self.query_contrastive_same_group_only and valid_group_ids is not None:
                    row_group_ids = valid_group_ids[target_positions]
                    pair_mask = valid_group_ids.unsqueeze(0) == row_group_ids.unsqueeze(1)
                    pair_mask[
                        torch.arange(batch_queries.shape[0], device=logits.device), target_positions
                    ] = True

                if pair_mask.any():
                    pair_loss = F.binary_cross_entropy_with_logits(
                        logits, target_matrix, reduction="none"
                    )
                    total_loss = total_loss + pair_loss.masked_select(pair_mask).sum()
                    total_matches += int(pair_mask.sum().item())
                continue

            unique_target_positions = torch.unique(target_positions, sorted=False)
            all_candidate_positions = torch.arange(
                logits.shape[1], dtype=torch.long, device=logits.device
            )
            for query_idx in range(batch_queries.shape[0]):
                pos_position = int(target_positions[query_idx].item())

                candidate_tensor = None
                if self.query_contrastive_same_group_only and valid_group_ids is not None:
                    pos_group = int(valid_group_ids[pos_position].item())
                    same_group_positions = torch.nonzero(
                        valid_group_ids == pos_group, as_tuple=False
                    ).squeeze(-1)
                    if same_group_positions.ndim == 0:
                        same_group_positions = same_group_positions.unsqueeze(0)
                    if same_group_positions.numel() <= 1 and self.query_contrastive_fallback_all:
                        same_group_positions = all_candidate_positions
                    candidate_tensor = same_group_positions.to(
                        device=logits.device, dtype=torch.long
                    )
                else:
                    candidate_positions = [pos_position]

                    for neg_position in unique_target_positions.tolist():
                        neg_position = int(neg_position)
                        if neg_position != pos_position:
                            candidate_positions.append(neg_position)

                    if self.query_contrastive_topk_neg > 0 and logits.shape[1] > 1:
                        logits_row = logits[query_idx]
                        wrong_logits = logits_row.clone()
                        wrong_logits[pos_position] = float("-inf")
                        hard_k = min(self.query_contrastive_topk_neg, wrong_logits.shape[0] - 1)
                        if hard_k > 0:
                            hard_positions = torch.topk(
                                wrong_logits, k=hard_k, dim=0
                            ).indices.tolist()
                            for neg_position in hard_positions:
                                neg_position = int(neg_position)
                                if neg_position != pos_position:
                                    candidate_positions.append(neg_position)

                    dedup_candidate_positions = []
                    seen_positions = set()
                    for candidate_position in candidate_positions:
                        if candidate_position in seen_positions:
                            continue
                        seen_positions.add(candidate_position)
                        dedup_candidate_positions.append(candidate_position)
                    candidate_tensor = torch.tensor(
                        dedup_candidate_positions,
                        dtype=torch.long,
                        device=logits.device,
                    )

                if candidate_tensor.numel() <= 1:
                    continue

                candidate_tensor = torch.unique(candidate_tensor, sorted=False)
                pos_mask = candidate_tensor == pos_position
                if not pos_mask.any():
                    candidate_tensor = torch.cat(
                        [
                            torch.tensor([pos_position], dtype=torch.long, device=logits.device),
                            candidate_tensor,
                        ],
                        dim=0,
                    )
                    pos_mask = candidate_tensor == pos_position
                pos_index = int(torch.nonzero(pos_mask, as_tuple=False)[0].item())
                if pos_index != 0:
                    ordered_positions = [pos_position]
                    ordered_positions.extend(
                        int(candidate)
                        for candidate in candidate_tensor.tolist()
                        if int(candidate) != pos_position
                    )
                    candidate_tensor = torch.tensor(
                        ordered_positions, dtype=torch.long, device=logits.device
                    )

                logits_subset = logits[query_idx, candidate_tensor].unsqueeze(0)
                target_subset = torch.zeros(1, dtype=torch.long, device=logits.device)
                total_loss = total_loss + F.cross_entropy(
                    logits_subset, target_subset, reduction="sum"
                )
                total_matches += 1

        if total_matches == 0:
            return {"loss_query_contrastive": query_features.sum() * 0.0}
        return {"loss_query_contrastive": total_loss / max(total_matches, 1)}

    def loss_global_image_text_sigmoid(self, outputs, targets, indices, num_boxes):
        image_features = outputs.get("global_alignment_image_features", None)
        text_features = outputs.get("global_alignment_text_features", None)
        text_valid_mask = outputs.get("global_alignment_text_valid_mask", None)
        zero_ref = outputs["pred_boxes"]
        if (
            not self.use_global_image_text_sigmoid_loss
            or image_features is None
            or text_features is None
        ):
            return {"loss_global_image_text_sigmoid": zero_ref.sum() * 0.0}

        if text_valid_mask is None:
            text_valid_mask = torch.ones(
                image_features.shape[0], dtype=torch.bool, device=image_features.device
            )
        else:
            text_valid_mask = text_valid_mask.to(device=image_features.device, dtype=torch.bool)
        valid_rows = text_valid_mask
        if valid_rows.sum().item() == 0:
            return {"loss_global_image_text_sigmoid": zero_ref.sum() * 0.0}

        image_features = image_features[valid_rows]
        text_features = text_features[valid_rows]
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-6)

        logit_scale = outputs.get("global_alignment_logit_scale", None)
        if logit_scale is None:
            logit_scale = image_features.new_tensor(1.0)
        else:
            logit_scale = logit_scale.to(device=image_features.device, dtype=image_features.dtype)
        logit_bias = outputs.get("global_alignment_logit_bias", None)
        if logit_bias is None:
            logit_bias = image_features.new_tensor(0.0)
        else:
            logit_bias = logit_bias.to(device=image_features.device, dtype=image_features.dtype)

        logits = logit_scale * (image_features @ text_features.transpose(0, 1)) + logit_bias
        targets = torch.eye(logits.shape[0], dtype=logits.dtype, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return {"loss_global_image_text_sigmoid": loss}

    def loss_aux_token_ce(self, outputs, targets, indices, num_boxes):
        pred_aux_token_logits = outputs.get("pred_aux_token_logits", None)
        aux_one_hot = outputs.get("aux_one_hot", None)
        aux_text_mask = outputs.get("aux_text_mask", None)
        aux_label_map_list = outputs.get("aux_label_map_list", None)
        if pred_aux_token_logits is None or aux_one_hot is None or aux_text_mask is None:
            zero_ref = outputs["pred_logits"]
            return {"loss_aux_token_ce": zero_ref.sum() * 0.0}
        text_mask = outputs.get(
            "aux_query_token_ignore_mask",
            outputs.get("aux_loss_text_mask", aux_text_mask),
        )
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        total_loss = pred_aux_token_logits.new_tensor(0.0)
        total_queries = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            batch_pred = pred_aux_token_logits[batch_idx, src_idx]
            batch_target = aux_one_hot[batch_idx, src_idx].to(batch_pred.device).float()
            batch_mask = text_mask[batch_idx].to(device=batch_pred.device, dtype=torch.bool)
            batch_mask = batch_mask.unsqueeze(0).expand(batch_pred.shape[0], -1)

            batch_pred = batch_pred[batch_mask].view(src_idx.numel(), -1)
            batch_target = batch_target[batch_mask].view(src_idx.numel(), -1)

            p = torch.sigmoid(batch_pred)
            ce_loss = F.binary_cross_entropy_with_logits(
                batch_pred, batch_target, reduction="none"
            )
            p_t = p * batch_target + (1 - p) * (1 - batch_target)
            focal = ce_loss * ((1 - p_t) ** gamma)

            if alpha >= 0:
                alpha_t = alpha * batch_target + (1 - alpha) * (1 - batch_target)
                focal = alpha_t * focal

            if self.use_aux_token_hard_neg and aux_label_map_list is not None:
                batch_weight = torch.full(
                    batch_target.shape,
                    self.aux_token_hard_neg_diff_group_weight,
                    dtype=torch.float32,
                    device=batch_pred.device,
                )
                aux_label_map_b = aux_label_map_list[batch_idx].to(
                    device=batch_pred.device, dtype=torch.float32
                )
                # Apply same text mask used for batch_target so token_mask
                # aligns with batch_weight's column dimension
                aux_text_mask_b = text_mask[batch_idx].to(
                    device=batch_pred.device, dtype=torch.bool
                )
                aux_label_map_b = aux_label_map_b[:, aux_text_mask_b]
                cap_group_ids = targets[batch_idx].get("cap_group_ids", None)
                target_labels = targets[batch_idx]["labels"][tgt_idx].to(
                    device=batch_pred.device, dtype=torch.long
                )

                if cap_group_ids is not None:
                    cap_group_ids = cap_group_ids.to(device=batch_pred.device, dtype=torch.long)
                    for query_row, pos_label in enumerate(target_labels.tolist()):
                        if pos_label < 0 or pos_label >= aux_label_map_b.shape[0]:
                            continue
                        if pos_label >= cap_group_ids.shape[0]:
                            continue

                        pos_group = int(cap_group_ids[pos_label].item())
                        same_group_mask = (cap_group_ids == pos_group)
                        same_group_indices = torch.nonzero(same_group_mask, as_tuple=False).squeeze(-1)
                        for cls_idx in same_group_indices.tolist():
                            if cls_idx < 0 or cls_idx >= aux_label_map_b.shape[0]:
                                continue
                            token_mask = aux_label_map_b[cls_idx] > 0
                            batch_weight[query_row, token_mask] = self.aux_token_hard_neg_same_group_weight

                        pos_token_mask = aux_label_map_b[pos_label] > 0
                        batch_weight[query_row, pos_token_mask] = self.aux_token_hard_neg_pos_weight

                focal = focal * batch_weight
                query_loss = focal.sum(dim=1) / batch_weight.sum(dim=1).clamp_min(1)
            else:
                query_loss = focal.mean(dim=1)
            total_loss = total_loss + query_loss.sum()
            total_queries += int(src_idx.numel())

        if total_queries == 0:
            return {"loss_aux_token_ce": pred_aux_token_logits.sum() * 0.0}
        return {"loss_aux_token_ce": total_loss / max(total_queries, 1)}

    def loss_query_selector(self, outputs, targets, indices, num_boxes):
        learned_scores = outputs.get("query_selector_learned_scores", None)
        ranking_scores = outputs.get("query_selector_ranking_scores", None)
        candidate_boxes = outputs.get("query_selector_candidate_boxes", None)
        score_source = ranking_scores if ranking_scores is not None else learned_scores
        if score_source is None or candidate_boxes is None:
            zero_ref = outputs["pred_logits"]
            return {"loss_query_selector": zero_ref.sum() * 0.0}

        total_loss = score_source.new_zeros(())
        total_batches = 0
        pos_thr = float(getattr(self, "query_score_pos_iou_thr", 0.5))
        neg_thr = float(getattr(self, "query_score_neg_iou_thr", 0.1))
        num_pos = int(getattr(self, "query_score_num_pos", 16))
        num_neg = int(getattr(self, "query_score_num_neg", 32))
        margin = float(getattr(self, "query_score_rank_margin", 0.2))

        for batch_idx in range(len(targets)):
            batch_scores = score_source[batch_idx]
            batch_boxes = candidate_boxes[batch_idx].detach()
            gt_boxes = targets[batch_idx].get("boxes", None)
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            if batch_scores.numel() == 0 or batch_boxes.numel() == 0:
                continue

            pred_xyxy = box_ops.box_cxcywh_to_xyxy(batch_boxes)
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes.to(batch_boxes.device).detach())
            if not torch.isfinite(pred_xyxy).all() or not torch.isfinite(gt_xyxy).all():
                continue
            iou = box_ops.box_iou(pred_xyxy, gt_xyxy)[0]
            if not torch.isfinite(iou).all():
                continue
            best_iou = iou.max(dim=1).values.detach().clamp(0.0, 1.0)
            batch_scores = torch.nan_to_num(batch_scores, nan=0.0, posinf=0.0, neginf=0.0)
            pos_mask = best_iou >= pos_thr
            neg_mask = best_iou <= neg_thr
            if pos_mask.sum() == 0:
                top_pos = torch.topk(best_iou, k=min(num_pos, best_iou.numel())).indices
                pos_mask = torch.zeros_like(best_iou, dtype=torch.bool)
                pos_mask[top_pos] = True
            if neg_mask.sum() == 0:
                low_neg = torch.topk(-best_iou, k=min(num_neg, best_iou.numel())).indices
                neg_mask = torch.zeros_like(best_iou, dtype=torch.bool)
                neg_mask[low_neg] = True

            pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
            neg_idx = torch.nonzero(neg_mask, as_tuple=False).flatten()
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_idx = pos_idx[torch.topk(best_iou[pos_idx], k=min(num_pos, pos_idx.numel())).indices]
            neg_idx = neg_idx[
                torch.topk(batch_scores[neg_idx], k=min(num_neg, neg_idx.numel())).indices
            ]

            pos_scores = batch_scores[pos_idx]
            neg_scores = batch_scores[neg_idx]
            pair_logits = pos_scores[:, None] - neg_scores[None, :]
            batch_loss = F.softplus(margin - pair_logits).mean()
            if torch.isfinite(batch_loss):
                total_loss = total_loss + batch_loss
                total_batches += 1

        if total_batches == 0:
            return {"loss_query_selector": score_source.new_zeros(())}
        return {"loss_query_selector": total_loss / total_batches}

    def loss_proto_cls(self, outputs, targets, indices, num_boxes):
        class_scores = outputs.get("proto_cls_scores", None)
        if class_scores is None or not self.use_proto_cls_head:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_proto_cls": zero, "proto_cls_acc": zero}

        total_loss = outputs["pred_logits"].new_tensor(0.0)
        total_correct = outputs["pred_logits"].new_tensor(0.0)
        total_count = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            batch_scores = class_scores[batch_idx]
            if batch_scores.numel() == 0:
                continue
            matched_scores = batch_scores[src_idx]
            if matched_scores.numel() == 0:
                continue
            target_labels = self._map_labels_to_score_space(
                targets[batch_idx],
                targets[batch_idx]["labels"][tgt_idx],
                matched_scores.shape[-1],
                batch_scores.device,
            )
            if target_labels.numel() == 0:
                continue
            matched_scores = torch.nan_to_num(
                matched_scores, nan=0.0, posinf=20.0, neginf=-20.0
            )
            matched_scores = matched_scores.clamp(min=-20.0, max=20.0)
            valid_mask = (target_labels >= 0) & (target_labels < matched_scores.shape[-1])
            if not bool(valid_mask.any()):
                continue
            if not bool(valid_mask.all()):
                matched_scores = matched_scores[valid_mask]
                target_labels = target_labels[valid_mask]
                if matched_scores.numel() == 0:
                    continue
            finite_row_mask = torch.isfinite(matched_scores).all(dim=-1)
            if not bool(finite_row_mask.any()):
                continue
            if not bool(finite_row_mask.all()):
                matched_scores = matched_scores[finite_row_mask]
                target_labels = target_labels[finite_row_mask]
                if matched_scores.numel() == 0:
                    continue
            total_loss = total_loss + F.cross_entropy(
                matched_scores,
                target_labels,
                reduction="sum",
            )
            total_correct = total_correct + (
                matched_scores.argmax(dim=-1) == target_labels
            ).float().sum()
            total_count += int(src_idx.numel())

        if total_count == 0:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_proto_cls": zero, "proto_cls_acc": zero}

        return {
            "loss_proto_cls": total_loss / max(total_count, 1),
            "proto_cls_acc": total_correct / max(total_count, 1),
        }

    def loss_appearance_cls(self, outputs, targets, indices, num_boxes):
        class_scores = outputs.get("appearance_class_scores", None)
        if class_scores is None or not self.use_appearance_cls_loss:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_cls": zero, "appearance_cls_acc": zero}

        total_loss = outputs["pred_logits"].new_tensor(0.0)
        total_correct = outputs["pred_logits"].new_tensor(0.0)
        total_count = 0
        debug_enabled = (
            os.environ.get("APPEARANCE_CLS_DEBUG", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        debug_rows = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            debug_row = {
                "batch": batch_idx,
                "src": int(src_idx.numel()),
                "tgt": int(tgt_idx.numel()),
            }
            if src_idx.numel() == 0:
                debug_row["skip"] = "empty_src"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue
            batch_scores = class_scores[batch_idx]
            debug_row["score_shape"] = tuple(batch_scores.shape)
            if batch_scores.numel() == 0:
                debug_row["skip"] = "empty_scores"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue

            matched_scores = batch_scores[src_idx]
            debug_row["matched_shape"] = tuple(matched_scores.shape)
            if matched_scores.numel() == 0:
                debug_row["skip"] = "empty_matched_scores"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue

            target_labels_global = targets[batch_idx]["labels"][tgt_idx]
            debug_row["global_targets"] = int(target_labels_global.numel())
            if target_labels_global.numel() == 0:
                debug_row["skip"] = "empty_target_labels"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue

            target_labels = self._map_labels_to_score_space(
                targets[batch_idx],
                target_labels_global,
                matched_scores.shape[-1],
                batch_scores.device,
            )
            debug_row["local_valid_before_mask"] = int(((target_labels >= 0) & (target_labels < matched_scores.shape[-1])).sum().item())

            matched_scores = matched_scores.float()
            matched_scores = torch.nan_to_num(
                matched_scores, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(min=-20.0, max=20.0)
            valid_mask = (target_labels >= 0) & (target_labels < matched_scores.shape[-1])
            if not bool(valid_mask.any()):
                debug_row["skip"] = "no_valid_local_targets"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue
            if not bool(valid_mask.all()):
                matched_scores = matched_scores[valid_mask]
                target_labels = target_labels[valid_mask]
                if matched_scores.numel() == 0:
                    debug_row["skip"] = "empty_after_valid_mask"
                    if debug_enabled:
                        debug_rows.append(debug_row)
                    continue
            finite_row_mask = torch.isfinite(matched_scores).all(dim=-1)
            if not bool(finite_row_mask.any()):
                debug_row["skip"] = "non_finite_rows"
                if debug_enabled:
                    debug_rows.append(debug_row)
                continue
            if not bool(finite_row_mask.all()):
                matched_scores = matched_scores[finite_row_mask]
                target_labels = target_labels[finite_row_mask]
                if matched_scores.numel() == 0:
                    debug_row["skip"] = "empty_after_finite_mask"
                    if debug_enabled:
                        debug_rows.append(debug_row)
                    continue

            with torch.cuda.amp.autocast(enabled=False):
                ce_loss = F.cross_entropy(
                    matched_scores,
                    target_labels,
                    reduction="sum",
                )
            total_loss = total_loss + ce_loss.to(device=outputs["pred_logits"].device)
            total_correct = total_correct + (
                matched_scores.argmax(dim=-1) == target_labels
            ).float().sum()
            total_count += int(target_labels.numel())
            debug_row["used"] = int(target_labels.numel())
            if debug_enabled:
                debug_rows.append(debug_row)

        if total_count == 0:
            if debug_enabled and not self._appearance_cls_debug_printed:
                print(
                    "[appearance-cls-debug] total_count=0 class_scores_type="
                    f"{type(class_scores).__name__}"
                )
                for row in debug_rows:
                    print(f"[appearance-cls-debug] {row}")
                self._appearance_cls_debug_printed = True
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_cls": zero, "appearance_cls_acc": zero}

        if debug_enabled and not self._appearance_cls_debug_printed:
            print(
                f"[appearance-cls-debug] total_count={total_count} "
                f"class_scores_len={len(class_scores) if isinstance(class_scores, list) else 'tensor'}"
            )
            for row in debug_rows:
                print(f"[appearance-cls-debug] {row}")
            self._appearance_cls_debug_printed = True

        return {
            "loss_appearance_cls": total_loss / max(total_count, 1),
            "appearance_cls_acc": total_correct / max(total_count, 1),
        }

    def loss_appearance_contrastive(self, outputs, targets, indices, num_boxes):
        # ---- Candidate-conditioned local evidence path ----
        # listwise CE over each query's candidate set (semantic top-M + GT).
        if outputs.get("appearance_is_local_evidence", False):
            return self._loss_appearance_local_evidence(outputs, targets, indices)

        class_scores = outputs.get("appearance_class_scores", None)
        if class_scores is None or not self.use_appearance_contrastive_loss:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_contrastive": zero, "appearance_contrastive_acc": zero}

        # ---- Prototype-bank path ----
        # In bank mode the head emits a single global [bs, nq, C] tensor where
        # column i == global class id i.  Hard negatives are mined from the global
        # fine-group vector across ALL classes (pl40 vs pl50/pl60...), so every
        # matched query sees its real confusers regardless of what shares the image.
        if outputs.get("appearance_is_global_bank", False) and isinstance(
            class_scores, torch.Tensor
        ):
            return self._loss_appearance_contrastive_bank(
                class_scores, targets, indices, outputs
            )

        total_loss = outputs["pred_logits"].new_tensor(0.0)
        total_correct = outputs["pred_logits"].new_tensor(0.0)
        total_count = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            batch_scores = class_scores[batch_idx]
            if not isinstance(batch_scores, torch.Tensor) or batch_scores.numel() == 0:
                continue

            matched_scores = batch_scores[src_idx].float()
            if matched_scores.numel() == 0:
                continue
            matched_scores = torch.nan_to_num(
                matched_scores, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(min=-20.0, max=20.0)

            target_labels_global = targets[batch_idx]["labels"][tgt_idx]
            target_labels = self._map_labels_to_score_space(
                targets[batch_idx],
                target_labels_global,
                matched_scores.shape[-1],
                matched_scores.device,
            )
            if target_labels.numel() == 0:
                continue

            local_group_ids = targets[batch_idx].get("cap_group_ids", None)
            if local_group_ids is not None:
                local_group_ids = local_group_ids.to(
                    device=matched_scores.device, dtype=torch.long
                )
                if local_group_ids.numel() > matched_scores.shape[-1]:
                    local_group_ids = local_group_ids[: matched_scores.shape[-1]]
            all_positions = torch.arange(
                matched_scores.shape[-1], device=matched_scores.device, dtype=torch.long
            )

            for row_idx in range(matched_scores.shape[0]):
                pos_position = int(target_labels[row_idx].item())
                if pos_position < 0 or pos_position >= matched_scores.shape[-1]:
                    continue

                candidate_positions = None
                if (
                    self.appearance_contrastive_same_group_only
                    and local_group_ids is not None
                    and pos_position < local_group_ids.numel()
                ):
                    pos_group = int(local_group_ids[pos_position].item())
                    same_group_positions = torch.nonzero(
                        local_group_ids == pos_group, as_tuple=False
                    ).squeeze(-1)
                    if same_group_positions.ndim == 0:
                        same_group_positions = same_group_positions.unsqueeze(0)
                    same_group_positions = same_group_positions[same_group_positions != pos_position]
                    if (
                        same_group_positions.numel() == 0
                        and self.appearance_contrastive_fallback_all
                    ):
                        same_group_positions = all_positions[all_positions != pos_position]
                    candidate_positions = same_group_positions
                else:
                    candidate_positions = all_positions[all_positions != pos_position]

                if candidate_positions is None or candidate_positions.numel() == 0:
                    continue

                if self.appearance_contrastive_topk_neg > 0 and candidate_positions.numel() > 0:
                    neg_scores = matched_scores[row_idx, candidate_positions]
                    hard_k = min(
                        self.appearance_contrastive_topk_neg,
                        int(candidate_positions.numel()),
                    )
                    candidate_positions = candidate_positions[
                        torch.topk(neg_scores, k=hard_k, dim=0).indices
                    ]

                ordered_positions = torch.cat(
                    [
                        torch.tensor(
                            [pos_position],
                            device=matched_scores.device,
                            dtype=torch.long,
                        ),
                        candidate_positions,
                    ],
                    dim=0,
                )
                row_logits = matched_scores[row_idx, ordered_positions].unsqueeze(0)
                row_target = torch.zeros((1,), device=matched_scores.device, dtype=torch.long)
                total_loss = total_loss + F.cross_entropy(row_logits, row_target, reduction="sum")
                total_correct = total_correct + (
                    row_logits.argmax(dim=-1) == row_target
                ).float().sum()
                total_count += 1

        if total_count == 0:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_contrastive": zero, "appearance_contrastive_acc": zero}

        return {
            "loss_appearance_contrastive": total_loss / max(total_count, 1),
            "appearance_contrastive_acc": total_correct / max(total_count, 1),
        }

    def _loss_appearance_local_evidence(self, outputs, targets, indices):
        """Listwise CE for candidate-conditioned local evidence.

        For each matched query, the candidate set is its semantic top-M (GT injected
        in forward).  Loss = CrossEntropy(evidence_logits[M], gt_slot).  This trains
        exactly the inference operation — "pick GT out of the semantic candidate set
        from local visual evidence" — instead of a sign-only pairwise preference.

        Honors the same low semantic-margin filter as the bank loss: only queries the
        semantic head is unsure about contribute (the rest don't need appearance).
        """
        evidence = outputs.get("appearance_local_evidence_logits", None)   # [bs,nq,M]
        cand_mask = outputs.get("appearance_local_candidate_mask", None)   # [bs,nq,M]
        gt_slot = outputs.get("appearance_local_gt_slot", None)            # [bs,nq]
        sem_margin = outputs.get("appearance_local_sem_margin", None)      # [bs,nq]
        zero = outputs["pred_boxes"].new_tensor(0.0)
        if evidence is None or gt_slot is None:
            return {"loss_appearance_contrastive": zero, "appearance_contrastive_acc": zero}

        margin_thr = float(getattr(self, "appearance_pairwise_margin_filter_thr", 1e9))
        device = evidence.device

        logits_list = []
        target_list = []
        for batch_idx, (src_idx, _tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device)
            ev = evidence[batch_idx][src_idx]            # [n, M]
            gs = gt_slot[batch_idx][src_idx]             # [n]
            valid = gs >= 0                              # GT present in candidate set
            if cand_mask is not None:
                cm = cand_mask[batch_idx][src_idx]       # [n, M]
                ev = ev.masked_fill(~cm.bool(), float("-inf"))
            # ---- low semantic-margin filter (detector caption-space margin) ----
            if sem_margin is not None and margin_thr < 1e8:
                sm = sem_margin[batch_idx][src_idx].float()   # [n]
                valid = valid & (sm <= margin_thr)
            if valid.any():
                logits_list.append(ev[valid])
                target_list.append(gs[valid])

        if not logits_list:
            return {"loss_appearance_contrastive": zero, "appearance_contrastive_acc": zero}

        logits = torch.cat(logits_list, dim=0).float()   # [N, M]
        tgt = torch.cat(target_list, dim=0).long()        # [N]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        with torch.cuda.amp.autocast(enabled=False):
            ce = F.cross_entropy(logits, tgt, reduction="mean")
        acc = (logits.argmax(dim=-1) == tgt).float().mean()
        return {
            "loss_appearance_contrastive": ce,
            "appearance_contrastive_acc": acc.detach(),
        }

    def _loss_appearance_contrastive_bank(self, class_scores, targets, indices, outputs):
        """Candidate CE loss for the conditional appearance evidence branch.

        For each matched query, classify GT among [GT + confusers].  This aligns
        the training metric with reranking, unlike pairwise losses that can be high
        while full/candidate top-1 remains poor.

        Confuser set (in priority order):
          1. semantic top-k from sem_logits_bank (appearance_contrastive_topk_neg > 0)
             — directly targets the semantic head's current real mistakes
          2. same-fine-group classes as fallback when sem_logits_bank unavailable

        Only queries where the semantic margin is below appearance_margin_filter_thr
        contribute to the loss (high-confidence semantic predictions don't need
        appearance correction and would only add noise).

        Metrics returned:
          loss_appearance_contrastive  — mean candidate CE loss
          appearance_contrastive_acc   — candidate-set top-1 accuracy
        """
        device = class_scores.device
        total_loss = class_scores.new_tensor(0.0)
        total_correct = class_scores.new_tensor(0.0)
        total_rows = 0
        full_top1_correct = class_scores.new_tensor(0.0)
        full_top1_total = 0
        same_group_full_correct = class_scores.new_tensor(0.0)
        same_group_full_total = 0
        pairwise_correct = class_scores.new_tensor(0.0)
        pairwise_total = 0
        pairwise_margin_sum = class_scores.new_tensor(0.0)
        same_group_pairwise_correct = class_scores.new_tensor(0.0)
        same_group_pairwise_total = 0

        num_classes = class_scores.shape[-1]
        group_ids = self.appearance_global_fine_group_ids
        if group_ids is not None:
            group_ids = group_ids.to(device=device, dtype=torch.long)
        all_positions = torch.arange(num_classes, device=device, dtype=torch.long)

        # Semantic logits in global-bank space for confuser selection.
        sem_logits = None
        _sl = outputs.get("sem_logits_bank", None)
        if (
            _sl is not None
            and isinstance(_sl, torch.Tensor)
            and _sl.shape[-1] == num_classes
        ):
            sem_logits = _sl.detach().float()

        # Optional semantic margin filter threshold.
        margin_thr = float(getattr(self, "appearance_pairwise_margin_filter_thr", 1e9))

        topk_neg = max(int(self.appearance_contrastive_topk_neg), 1)
        loss_mode = str(
            getattr(self, "appearance_contrastive_loss_mode", "candidate_ce")
        ).strip().lower()
        use_pairwise_loss = loss_mode in {
            "pairwise",
            "pairwise_logistic",
            "pairwise_margin",
            "gt_vs_confuser",
        }
        pairwise_margin = float(getattr(self, "appearance_pairwise_loss_margin", 0.0))

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            matched_scores = class_scores[batch_idx][src_idx].float()  # [n_match, C]
            if matched_scores.numel() == 0:
                continue
            matched_scores = torch.nan_to_num(
                matched_scores, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(min=-20.0, max=20.0)

            target_labels_local = targets[batch_idx]["labels"][tgt_idx].to(
                device=device, dtype=torch.long
            )
            label_ids_list = targets[batch_idx].get("caption_label_ids", None)
            if label_ids_list:
                local_to_global = torch.as_tensor(
                    [int(x) for x in label_ids_list],
                    dtype=torch.long,
                    device=device,
                )
                valid_local = (
                    (target_labels_local >= 0)
                    & (target_labels_local < local_to_global.numel())
                )
                target_labels = torch.full_like(target_labels_local, -1)
                target_labels[valid_local] = local_to_global[
                    target_labels_local[valid_local]
                ]
            else:
                # COCO-style full-class supervision can already be global.
                target_labels = target_labels_local

            matched_sem = None
            if sem_logits is not None and batch_idx < sem_logits.shape[0]:
                matched_sem = sem_logits[batch_idx][src_idx]  # [n_match, C]

            for row_idx in range(matched_scores.shape[0]):
                pos_position = int(target_labels[row_idx].item())
                if pos_position < 0 or pos_position >= num_classes:
                    continue

                score_gt = matched_scores[row_idx, pos_position]
                full_top1_correct = full_top1_correct + (
                    matched_scores[row_idx].argmax(dim=-1) == pos_position
                ).float()
                full_top1_total += 1
                if group_ids is not None and pos_position < group_ids.numel():
                    pos_group = int(group_ids[pos_position].item())
                    sg_all = torch.nonzero(group_ids == pos_group, as_tuple=False).squeeze(-1)
                    sg_all = sg_all[(sg_all >= 0) & (sg_all < num_classes)]
                    if sg_all.numel() > 1:
                        sg_scores = matched_scores[row_idx, sg_all]
                        sg_pred = int(sg_all[sg_scores.argmax(dim=-1)].item())
                        same_group_full_correct = same_group_full_correct + (
                            matched_scores.new_tensor(float(sg_pred == pos_position))
                        )
                        same_group_full_total += 1

                # ---- Semantic margin filter / weighting ----
                # Skip queries the semantic head is already very confident about.
                margin_weight = 1.0
                if matched_sem is not None and margin_thr < 1e8:
                    sem_row = matched_sem[row_idx].clone()
                    sem_row[pos_position] = float("-inf")
                    sem_top1_confuser_score = sem_row.max()
                    sem_margin = float((matched_sem[row_idx, pos_position] - sem_top1_confuser_score).item())
                    use_margin_weight = bool(
                        getattr(self, "appearance_margin_weight_loss", False)
                    )
                    if use_margin_weight:
                        temperature = max(
                            float(getattr(self, "appearance_margin_weight_temperature", 0.2)),
                            1e-6,
                        )
                        min_weight = float(
                            getattr(self, "appearance_margin_weight_min", 0.25)
                        )
                        raw_weight = torch.sigmoid(
                            matched_scores.new_tensor((margin_thr - sem_margin) / temperature)
                        )
                        margin_weight = float(
                            min_weight + (1.0 - min_weight) * raw_weight.item()
                        )
                    elif sem_margin > margin_thr:
                        continue

                # ---- Build confuser set ----
                # mode: "sem_topk" (default) | "same_group" | "union"
                #     | "same_group_topk" | "same_group_then_sem_topk"
                mode = getattr(self, "appearance_confuser_mode", "sem_topk")

                def _same_group_negs():
                    if group_ids is None or pos_position >= group_ids.numel():
                        return None
                    pos_group = int(group_ids[pos_position].item())
                    sg = torch.nonzero(group_ids == pos_group, as_tuple=False).squeeze(-1)
                    if sg.ndim == 0:
                        sg = sg.unsqueeze(0)
                    return sg[sg != pos_position]

                def _sem_topk_negs():
                    if matched_sem is None:
                        return None
                    sem_row = matched_sem[row_idx].clone()
                    sem_row[pos_position] = float("-inf")
                    k = min(topk_neg, num_classes - 1)
                    return torch.topk(sem_row, k=k).indices

                def _semantic_runnerup_neg():
                    sem = _sem_topk_negs()
                    if sem is None or sem.numel() == 0:
                        return None
                    return sem[:1]

                confuser_parts = []
                if bool(getattr(self, "appearance_force_semantic_runnerup", False)):
                    p = _semantic_runnerup_neg()
                    if p is not None and p.numel() > 0:
                        confuser_parts.append(p)
                if mode == "same_group_topk":
                    # The k same-group classes the semantic head ranks highest —
                    # i.e. same group AND semantically most confusable.  This is the
                    # tightest match to the eval metric (GT vs semantic runner-up in
                    # the same fine-group), so the loss trains exactly the pair the
                    # offline metric scores instead of easy far-apart same-group pairs.
                    sg = _same_group_negs()
                    if sg is not None and sg.numel() > 0 and matched_sem is not None:
                        sg_sem = matched_sem[row_idx][sg]            # sem logits over same-group
                        k = min(topk_neg, sg.numel())
                        top_local = torch.topk(sg_sem, k=k).indices
                        confuser_parts.append(sg[top_local])
                    elif sg is not None and sg.numel() > 0:
                        confuser_parts.append(sg)                    # no sem logits: full same-group
                if mode == "same_group_then_sem_topk":
                    # Keep the fine-grained objective, but avoid starving the
                    # loss when a fine group has only a few siblings: take all
                    # same-group negatives first, then fill the remaining slots
                    # with the detector's semantic top-k confusers.
                    sg = _same_group_negs()
                    if sg is not None and sg.numel() > 0:
                        if matched_sem is not None:
                            sg_sem = matched_sem[row_idx][sg]
                            order = torch.argsort(sg_sem, descending=True)
                            sg = sg[order]
                        confuser_parts.append(sg[:topk_neg])
                    remaining = topk_neg - sum(int(p.numel()) for p in confuser_parts)
                    if remaining > 0:
                        sem = _sem_topk_negs()
                        if sem is not None and sem.numel() > 0:
                            if confuser_parts:
                                used = torch.cat(confuser_parts)
                                sem = sem[~torch.isin(sem, used)]
                            if sem.numel() > 0:
                                confuser_parts.append(sem[:remaining])
                if mode in ("sem_topk", "union"):
                    p = _sem_topk_negs()
                    if p is not None and p.numel() > 0:
                        confuser_parts.append(p)
                if mode in ("same_group", "union"):
                    p = _same_group_negs()
                    if p is not None and p.numel() > 0:
                        confuser_parts.append(p)

                if confuser_parts:
                    # Union + dedup.  semantic_topk aligns with the head's real
                    # confusions; same_group guarantees the hardest fine-grained
                    # negatives are always trained against.
                    confuser_positions = torch.unique(torch.cat(confuser_parts))
                    if confuser_positions.numel() > topk_neg:
                        confuser_positions = confuser_positions[:topk_neg]
                elif matched_sem is not None:
                    # Fallback: sem top-k (e.g. same_group mode but no group info).
                    sem_row = matched_sem[row_idx].clone()
                    sem_row[pos_position] = float("-inf")
                    confuser_positions = torch.topk(
                        sem_row, k=min(topk_neg, num_classes - 1)
                    ).indices
                else:
                    # No semantic logits at all: same-group or all-class fallback.
                    if (
                        self.appearance_contrastive_same_group_only
                        and group_ids is not None
                        and pos_position < group_ids.numel()
                    ):
                        sg = _same_group_negs()
                        confuser_positions = (
                            sg if sg is not None and sg.numel() > 0
                            else all_positions[all_positions != pos_position]
                        )
                        if (
                            confuser_positions.numel() == 0
                            and self.appearance_contrastive_fallback_all
                        ):
                            confuser_positions = all_positions[all_positions != pos_position]
                    else:
                        confuser_positions = all_positions[all_positions != pos_position]
                    if confuser_positions.numel() > topk_neg:
                        confuser_positions = confuser_positions[:topk_neg]

                if confuser_positions.numel() == 0:
                    continue

                confuser_positions = confuser_positions[confuser_positions != pos_position]
                if confuser_positions.numel() == 0:
                    continue

                if use_pairwise_loss:
                    confuser_scores = matched_scores[row_idx, confuser_positions]
                    pair_margins = score_gt - confuser_scores
                    if pair_margins.numel() == 0:
                        continue
                    pair_losses = F.softplus(
                        matched_scores.new_tensor(pairwise_margin) - pair_margins
                    )
                    candidate_loss = pair_losses.mean()
                    if margin_weight != 1.0:
                        candidate_loss = candidate_loss * matched_scores.new_tensor(margin_weight)
                    total_loss = total_loss + candidate_loss
                    correct_mask = pair_margins > 0
                    pairwise_correct = pairwise_correct + correct_mask.float().sum()
                    pairwise_total += int(pair_margins.numel())
                    pairwise_margin_sum = pairwise_margin_sum + pair_margins.detach().sum()
                    total_correct = total_correct + correct_mask.float().mean()
                    total_rows += 1
                    if group_ids is not None and pos_position < group_ids.numel():
                        pos_group = int(group_ids[pos_position].item())
                        valid_group_ids = confuser_positions[
                            confuser_positions < group_ids.numel()
                        ]
                        if valid_group_ids.numel() > 0:
                            sg_mask = group_ids[valid_group_ids] == pos_group
                            if bool(sg_mask.any()):
                                sg_margins = score_gt - matched_scores[
                                    row_idx, valid_group_ids[sg_mask]
                                ]
                                same_group_pairwise_correct = (
                                    same_group_pairwise_correct
                                    + (sg_margins > 0).float().sum()
                                )
                                same_group_pairwise_total += int(sg_margins.numel())
                    continue

                ordered_positions = torch.cat(
                    [
                        torch.tensor([pos_position], device=device, dtype=torch.long),
                        confuser_positions,
                    ],
                    dim=0,
                )
                ordered_positions = torch.unique(ordered_positions, sorted=False)
                pos_mask = ordered_positions == pos_position
                if not pos_mask.any():
                    continue
                if int(torch.nonzero(pos_mask, as_tuple=False)[0].item()) != 0:
                    ordered_positions = torch.cat(
                        [
                            torch.tensor([pos_position], device=device, dtype=torch.long),
                            ordered_positions[~pos_mask],
                        ],
                        dim=0,
                    )

                row_logits = matched_scores[row_idx, ordered_positions].unsqueeze(0)
                row_target = torch.zeros(1, device=device, dtype=torch.long)
                candidate_loss = F.cross_entropy(row_logits, row_target, reduction="sum")
                if self.appearance_full_ce_loss_coef > 0.0:
                    full_target = torch.tensor([pos_position], device=device, dtype=torch.long)
                    full_loss = F.cross_entropy(
                        matched_scores[row_idx].unsqueeze(0),
                        full_target,
                        reduction="sum",
                    )
                    candidate_loss = candidate_loss + self.appearance_full_ce_loss_coef * full_loss
                if margin_weight != 1.0:
                    candidate_loss = candidate_loss * matched_scores.new_tensor(margin_weight)
                total_loss = total_loss + candidate_loss
                total_correct = total_correct + (
                    row_logits.argmax(dim=-1) == row_target
                ).float().sum()
                total_rows += 1

        if total_rows == 0:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            result = {
                "loss_appearance_contrastive": zero,
                "appearance_contrastive_acc": zero,
                "appearance_full_top1_acc": zero,
                "appearance_same_group_full_acc": zero,
                "appearance_pairwise_acc": zero,
                "appearance_pairwise_margin": zero,
                "appearance_same_group_pairwise_acc": zero,
            }
            self._attach_residual_stats(result, outputs)
            return result

        result = {
            "loss_appearance_contrastive": total_loss / max(total_rows, 1),
            "appearance_contrastive_acc": total_correct / max(total_rows, 1),
            "appearance_full_top1_acc": full_top1_correct / max(full_top1_total, 1),
            "appearance_same_group_full_acc": (
                same_group_full_correct / max(same_group_full_total, 1)
            ),
            "appearance_pairwise_acc": pairwise_correct / max(pairwise_total, 1),
            "appearance_pairwise_margin": pairwise_margin_sum / max(pairwise_total, 1),
            "appearance_same_group_pairwise_acc": (
                same_group_pairwise_correct / max(same_group_pairwise_total, 1)
            ),
        }
        self._attach_residual_stats(result, outputs)
        return result

    @staticmethod
    def _attach_residual_stats(result, outputs):
        # Surface the discriminative-residual magnitude as non-loss metrics so it
        # is visible in training logs (watch for runaway growth = degenerating
        # into a standalone classifier).  Not weighted into the total loss.
        for key in (
            "appearance_residual_norm_mean",
            "appearance_residual_norm_max",
            "appearance_centroid_norm_mean",
            "appearance_centroid_norm_max",
        ):
            value = outputs.get(key, None)
            if value is not None:
                result[key] = value.detach()

    @staticmethod
    def _normalize_self_facb_levels(levels, num_levels):
        if levels is None:
            levels = [0]
        elif isinstance(levels, int):
            levels = [levels]
        result = []
        for level in levels:
            level = int(level)
            if 0 <= level < num_levels:
                result.append(level)
        return result or [0]

    def _extract_self_facb_roi_tokens_for_loss(
        self,
        srcs,
        boxes_cxcywh,
        batch_indices,
        levels,
        roi_size,
    ):
        if not srcs or boxes_cxcywh.numel() == 0:
            return None
        device = boxes_cxcywh.device
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes_cxcywh.detach().clamp(0.0, 1.0)
        )
        batch_indices = batch_indices.to(device=device)
        chunks = []
        for level in self._normalize_self_facb_levels(levels, len(srcs)):
            fmap = srcs[level].detach()
            _, _, feat_h, feat_w = fmap.shape
            boxes_l = boxes_xyxy.clone()
            boxes_l = boxes_l * boxes_l.new_tensor([feat_w, feat_h, feat_w, feat_h])
            boxes_l[..., 0::2] = boxes_l[..., 0::2].clamp(0.0, float(feat_w - 1))
            boxes_l[..., 1::2] = boxes_l[..., 1::2].clamp(0.0, float(feat_h - 1))
            boxes_l[..., 2] = torch.maximum(
                boxes_l[..., 2], boxes_l[..., 0] + 1.0
            ).clamp(max=float(feat_w))
            boxes_l[..., 3] = torch.maximum(
                boxes_l[..., 3], boxes_l[..., 1] + 1.0
            ).clamp(max=float(feat_h))
            rois = torch.cat(
                [batch_indices.to(dtype=boxes_l.dtype).view(-1, 1), boxes_l],
                dim=1,
            )
            feats = roi_align(
                fmap,
                rois,
                output_size=int(roi_size),
                spatial_scale=1.0,
                aligned=True,
            )
            chunks.append(feats.flatten(2).transpose(1, 2))
        if not chunks:
            return None
        return torch.cat(chunks, dim=1)

    @staticmethod
    def _expand_self_facb_boxes_cxcywh(boxes_cxcywh, scale):
        scale = float(scale)
        if scale <= 1.0:
            return boxes_cxcywh
        boxes = boxes_cxcywh.detach().clone()
        boxes[..., 2:] = (boxes[..., 2:] * scale).clamp(min=1e-6, max=1.0)
        boxes[..., :2] = boxes[..., :2].clamp(0.0, 1.0)
        return boxes

    @staticmethod
    def _local_to_global_label_ids(target, num_local, device):
        label_ids = target.get("caption_label_ids", None)
        if label_ids:
            values = []
            for idx in range(num_local):
                if idx < len(label_ids):
                    try:
                        values.append(int(label_ids[idx]))
                    except (TypeError, ValueError):
                        values.append(-1)
                else:
                    values.append(-1)
            return torch.as_tensor(values, device=device, dtype=torch.long)
        return torch.arange(num_local, device=device, dtype=torch.long)

    def loss_facb_symbol_align(self, outputs, targets, indices, num_boxes):
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        model_ref = getattr(self, "facb_symbol_align_model_ref", None)
        srcs = outputs.get("self_facb_srcs", None)
        label_map_list = outputs.get("_criterion_label_map_list", None)
        if model_ref is None or srcs is None or label_map_list is None:
            return {"loss_facb_symbol_align": zero}
        if not hasattr(model_ref, "facb_symbol_align_proj"):
            return {"loss_facb_symbol_align": zero}

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs.get("pred_boxes", None)
        device = pred_logits.device
        topk = max(int(getattr(self, "facb_symbol_align_topk", 5)), 1)
        boxes = []
        batch_rows = []
        candidate_ids = []
        candidate_masks = []
        target_slots = []
        row_weights = []
        smallshot_flags = []
        semantic_top1_slots = []
        hard_rows = []
        semantic_total = 0
        semantic_correct = pred_logits.new_tensor(0.0, dtype=torch.float32)

        texts = model_ref._get_facb_symbol_align_texts()
        num_global = len(texts) if texts else 0
        if num_global <= 0:
            return {"loss_facb_symbol_align": zero}

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0 or batch_idx >= len(label_map_list):
                continue
            label_map = label_map_list[batch_idx]
            if label_map is None or label_map.numel() == 0:
                continue
            label_map = label_map.to(device=device, dtype=torch.float32)
            num_local = int(label_map.shape[0])
            if num_local <= 0:
                continue
            weights = label_map / label_map.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sem_scores = torch.nan_to_num(
                pred_logits[batch_idx].detach().sigmoid().float() @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            local_to_global = self._local_to_global_label_ids(
                targets[batch_idx], num_local, device
            )
            valid_local = (local_to_global >= 0) & (local_to_global < num_global)
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            gt_local_all = targets[batch_idx]["labels"].to(device=device, dtype=torch.long)
            gt_boxes_all = targets[batch_idx]["boxes"].to(device=device, dtype=pred_logits.dtype)

            for row_pos in range(src_idx.numel()):
                query_idx = int(src_idx[row_pos].item())
                target_idx = int(tgt_idx[row_pos].item())
                if target_idx < 0 or target_idx >= gt_local_all.numel():
                    continue
                gt_local = int(gt_local_all[target_idx].item())
                if gt_local < 0 or gt_local >= num_local or not bool(valid_local[gt_local].item()):
                    continue
                sem_row = sem_scores[query_idx].clone()
                sem_row[~valid_local] = float("-inf")
                valid_count = int(valid_local.sum().item())
                if valid_count <= 0:
                    continue
                k = min(topk, valid_count)
                top_local = torch.topk(sem_row, k=k, dim=0).indices
                if not bool((top_local == gt_local).any().item()):
                    gt_tensor = torch.tensor([gt_local], device=device, dtype=torch.long)
                    if top_local.numel() >= topk:
                        top_local = torch.cat([top_local[:-1], gt_tensor], dim=0)
                    else:
                        top_local = torch.cat([top_local, gt_tensor], dim=0)

                cand_global = local_to_global[top_local]
                target_pos = torch.nonzero(
                    cand_global == int(local_to_global[gt_local].item()),
                    as_tuple=False,
                )
                if target_pos.numel() == 0:
                    continue
                target_slot = int(target_pos[0, 0].item())
                cand_mask = torch.ones(cand_global.numel(), device=device, dtype=torch.bool)
                if cand_global.numel() < topk:
                    pad = topk - cand_global.numel()
                    cand_global = torch.cat([cand_global, cand_global.new_zeros(pad)], dim=0)
                    cand_mask = torch.cat(
                        [cand_mask, torch.zeros(pad, device=device, dtype=torch.bool)],
                        dim=0,
                    )
                else:
                    cand_global = cand_global[:topk]
                    cand_mask = cand_mask[:topk]
                if target_slot >= topk or not bool(cand_mask[target_slot].item()):
                    continue

                top1_slot = -1
                semantic_wrong = False
                if top_local.numel() > 0:
                    top1_global = int(local_to_global[int(top_local[0].item())].item())
                    gt_global = int(local_to_global[gt_local].item())
                    semantic_wrong = top1_global != gt_global
                    semantic_correct = semantic_correct + pred_logits.new_tensor(
                        float(not semantic_wrong), dtype=torch.float32
                    )
                    semantic_total += 1
                    top1_match = torch.nonzero(cand_global == top1_global, as_tuple=False)
                    if top1_match.numel() > 0:
                        top1_slot = int(top1_match[0, 0].item())

                if pred_boxes is not None:
                    roi_box = pred_boxes[batch_idx, query_idx].detach()
                else:
                    roi_box = gt_boxes_all[target_idx].detach()
                boxes.append(roi_box)
                batch_rows.append(torch.tensor(batch_idx, device=device, dtype=torch.long))
                candidate_ids.append(cand_global)
                candidate_masks.append(cand_mask)
                target_slots.append(torch.tensor(target_slot, device=device, dtype=torch.long))
                semantic_top1_slots.append(torch.tensor(top1_slot, device=device, dtype=torch.long))
                hard_rows.append(torch.tensor(semantic_wrong, device=device, dtype=torch.bool))

        if not boxes:
            return {
                "loss_facb_symbol_align": zero,
                "facb_symbol_align_nce": zero.detach(),
                "facb_symbol_align_pairwise": zero.detach(),
                "facb_symbol_align_rows": zero.detach(),
            }

        boxes = torch.stack(boxes, dim=0).to(device=device, dtype=pred_logits.dtype)
        batch_rows = torch.stack(batch_rows, dim=0).to(device=device, dtype=torch.long)
        candidate_ids = torch.stack(candidate_ids, dim=0).to(device=device, dtype=torch.long)
        candidate_masks = torch.stack(candidate_masks, dim=0).to(device=device, dtype=torch.bool)
        target_slots = torch.stack(target_slots, dim=0).to(device=device, dtype=torch.long)
        semantic_top1_slots = torch.stack(semantic_top1_slots, dim=0).to(device=device, dtype=torch.long)
        hard_rows = torch.stack(hard_rows, dim=0).to(device=device, dtype=torch.bool)

        symbol_prototypes, symbol_valid = model_ref._encode_facb_symbol_prototypes_online(
            candidate_ids, device=device, dtype=torch.float32
        )
        if symbol_prototypes is None or symbol_valid is None:
            return {"loss_facb_symbol_align": zero}
        candidate_masks = candidate_masks & symbol_valid.to(device=device, dtype=torch.bool)
        target_valid = candidate_masks.gather(1, target_slots.view(-1, 1)).squeeze(1)
        if not bool(target_valid.any().item()):
            return {"loss_facb_symbol_align": zero}
        boxes = boxes[target_valid]
        batch_rows = batch_rows[target_valid]
        candidate_ids = candidate_ids[target_valid]
        candidate_masks = candidate_masks[target_valid]
        target_slots = target_slots[target_valid]
        semantic_top1_slots = semantic_top1_slots[target_valid]
        hard_rows = hard_rows[target_valid]
        symbol_prototypes = symbol_prototypes[target_valid]

        if bool(getattr(self, "facb_symbol_align_error_only", False)):
            if not bool(hard_rows.any().item()):
                return {
                    "loss_facb_symbol_align": zero,
                    "facb_symbol_align_nce": zero.detach(),
                    "facb_symbol_align_pairwise": zero.detach(),
                    "facb_symbol_align_acc": zero.detach(),
                    "facb_symbol_align_pairwise_acc": zero.detach(),
                    "facb_symbol_align_hard_frac": zero.detach(),
                    "facb_symbol_align_rows": zero.detach(),
                    "facb_symbol_align_topm_valid_frac": zero.detach(),
                    "facb_symbol_align_valid_cands": zero.detach(),
                    "facb_symbol_align_score_std": zero.detach(),
                    "facb_symbol_align_roi_norm": zero.detach(),
                    "facb_symbol_align_proto_norm": zero.detach(),
                    "facb_symbol_align_semantic_acc": (
                        semantic_correct / max(semantic_total, 1)
                        if semantic_total > 0
                        else zero
                    ).detach(),
                }
            boxes = boxes[hard_rows]
            batch_rows = batch_rows[hard_rows]
            candidate_ids = candidate_ids[hard_rows]
            candidate_masks = candidate_masks[hard_rows]
            target_slots = target_slots[hard_rows]
            semantic_top1_slots = semantic_top1_slots[hard_rows]
            symbol_prototypes = symbol_prototypes[hard_rows]
            hard_rows = hard_rows[hard_rows]

        roi_tokens = self._extract_self_facb_roi_tokens_for_loss(
            srcs=srcs,
            boxes_cxcywh=boxes,
            batch_indices=batch_rows,
            levels=getattr(self, "self_facb_levels", outputs.get("self_facb_levels", [0])),
            roi_size=int(getattr(self, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7))),
        )
        if roi_tokens is None or roi_tokens.numel() == 0:
            return {"loss_facb_symbol_align": zero}

        with torch.cuda.amp.autocast(enabled=False):
            roi_vec = roi_tokens.float().mean(dim=1)
            roi_proj = model_ref.facb_symbol_align_proj(roi_vec)
            roi_proj = torch.nan_to_num(roi_proj.float(), nan=0.0, posinf=0.0, neginf=0.0)
            roi_proj = F.normalize(roi_proj, dim=-1, eps=1e-6)
            symbol_proj = torch.nan_to_num(
                symbol_prototypes.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            symbol_proj = F.normalize(symbol_proj, dim=-1, eps=1e-6)
            scores = (roi_proj.unsqueeze(1) * symbol_proj).sum(dim=-1)
            scores = scores / float(getattr(self, "facb_symbol_align_temperature", 0.07))
            scores = scores.masked_fill(~candidate_masks, float("-inf"))
            valid_counts = candidate_masks.sum(dim=1)

            row_weights = torch.where(
                hard_rows,
                scores.new_tensor(float(getattr(self, "facb_symbol_align_hard_weight", 1.0))),
                scores.new_tensor(float(getattr(self, "facb_symbol_align_easy_weight", 0.1))),
            )
            row_weights = row_weights / row_weights.mean().clamp(min=1e-6)
            topm_valid = valid_counts >= 2
            nce_terms = []
            if bool(topm_valid.any().item()):
                nce_per_row = F.cross_entropy(
                    scores[topm_valid], target_slots[topm_valid], reduction="none"
                )
                nce_terms.append((nce_per_row * row_weights[topm_valid]).mean())
            inbatch_weight = float(getattr(self, "facb_symbol_align_inbatch_weight", 0.5))
            if inbatch_weight > 0.0 and roi_proj.shape[0] >= 2:
                pos_ids = target_slots.view(-1, 1, 1).expand(-1, 1, symbol_prototypes.shape[-1])
                pos_proto = symbol_prototypes.gather(1, pos_ids).squeeze(1)
                pos_proto = torch.nan_to_num(
                    pos_proto.float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                pos_proto = F.normalize(pos_proto, dim=-1, eps=1e-6)
                inbatch_scores = torch.matmul(roi_proj, pos_proto.t())
                inbatch_scores = inbatch_scores / float(
                    getattr(self, "facb_symbol_align_temperature", 0.07)
                )
                inbatch_target = torch.arange(inbatch_scores.shape[0], device=device)
                inbatch_per_row = F.cross_entropy(
                    inbatch_scores, inbatch_target, reduction="none"
                )
                if bool((~topm_valid).any().item()):
                    fallback = (inbatch_per_row[~topm_valid] * row_weights[~topm_valid]).mean()
                    nce_terms.append(inbatch_weight * fallback)
                elif not nce_terms:
                    nce_terms.append(inbatch_weight * (inbatch_per_row * row_weights).mean())
            if nce_terms:
                nce = torch.stack(nce_terms).sum()
            else:
                nce = scores.sum() * 0.0

            top1_valid = (
                (semantic_top1_slots >= 0)
                & (semantic_top1_slots < scores.shape[1])
                & candidate_masks.gather(
                    1, semantic_top1_slots.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
                ).squeeze(1)
                & (semantic_top1_slots != target_slots)
            )
            if bool(top1_valid.any().item()):
                pos = scores.gather(1, target_slots.view(-1, 1)).squeeze(1)
                top1 = scores.gather(
                    1, semantic_top1_slots.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
                ).squeeze(1)
                margins = pos[top1_valid] - top1[top1_valid]
                pairwise_per_row = F.softplus(
                    scores.new_tensor(float(getattr(self, "facb_symbol_align_pairwise_margin", 0.1)))
                    - margins
                )
                pairwise = (pairwise_per_row * row_weights[top1_valid]).mean()
                pairwise_acc = (margins > 0).float().mean().detach()
            else:
                pairwise = nce.new_zeros(())
                pairwise_acc = zero.detach()

        loss = nce + float(getattr(self, "facb_symbol_align_pairwise_weight", 0.5)) * pairwise
        pred = scores.argmax(dim=1)
        semantic_acc = (
            semantic_correct / max(semantic_total, 1)
            if semantic_total > 0
            else zero
        )
        return {
            "loss_facb_symbol_align": loss,
            "facb_symbol_align_nce": nce.detach(),
            "facb_symbol_align_pairwise": pairwise.detach(),
            "facb_symbol_align_acc": (pred == target_slots).float().mean().detach(),
            "facb_symbol_align_pairwise_acc": pairwise_acc,
            "facb_symbol_align_hard_frac": hard_rows.float().mean().detach(),
            "facb_symbol_align_rows": scores.new_tensor(float(scores.shape[0])).detach(),
            "facb_symbol_align_topm_valid_frac": topm_valid.float().mean().detach(),
            "facb_symbol_align_valid_cands": valid_counts.float().mean().detach(),
            "facb_symbol_align_score_std": torch.nan_to_num(
                scores[torch.isfinite(scores)].float().std()
                if bool(torch.isfinite(scores).any().item())
                else scores.new_zeros(()),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).detach(),
            "facb_symbol_align_roi_norm": roi_proj.norm(dim=-1).mean().detach(),
            "facb_symbol_align_proto_norm": symbol_proj.norm(dim=-1).mean().detach(),
            "facb_symbol_align_semantic_acc": semantic_acc.detach(),
        }

    def loss_self_facb(self, outputs, targets, indices, num_boxes):
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        if not self.use_self_facb_loss:
            return {"loss_self_facb": zero}
        head = getattr(self, "self_facb_head_ref", None)
        srcs = outputs.get("self_facb_srcs", None)
        proto_bank = outputs.get("self_facb_proto_bank", None)
        label_map_list = outputs.get("_criterion_label_map_list", None)
        if head is None or srcs is None or proto_bank is None or label_map_list is None:
            return {"loss_self_facb": zero}
        head_zero = zero
        for parameter in head.parameters():
            if parameter.requires_grad:
                head_zero = parameter.reshape(-1)[0].float() * 0.0
                break

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs.get("pred_boxes", None)
        device = pred_logits.device
        proto_bank = proto_bank.to(device=device, dtype=pred_logits.dtype)
        if proto_bank.dim() not in (2, 3):
            return {"loss_self_facb": zero}
        num_global = int(proto_bank.shape[0])
        if num_global <= 0:
            return {"loss_self_facb": zero}

        proto_mask = outputs.get("self_facb_proto_bank_mask", None)
        if proto_mask is not None:
            proto_mask = proto_mask.to(device=device, dtype=torch.bool)

        topk = max(int(getattr(self, "self_facb_topk", 10)), 1)
        boxes = []
        batch_rows = []
        candidate_ids = []
        candidate_masks = []
        semantic_candidate_scores = []
        target_slots = []
        semantic_top1_slots = []
        semantic_margins = []
        same_group_confusers = []
        row_is_background = []
        row_is_ooc_keep = []
        crm_mode = bool(getattr(self, "self_facb_crm_mode", False))
        semantic_correct = pred_logits.new_tensor(0.0, dtype=torch.float32)
        semantic_total = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            if batch_idx >= len(label_map_list):
                continue
            label_map = label_map_list[batch_idx]
            if label_map is None or label_map.numel() == 0:
                continue
            label_map = label_map.to(device=device, dtype=torch.float32)
            num_local = int(label_map.shape[0])
            if num_local <= 0:
                continue
            weights = label_map / label_map.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sem_scores = torch.nan_to_num(
                pred_logits[batch_idx].detach().sigmoid().float() @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            local_to_global = self._local_to_global_label_ids(
                targets[batch_idx], num_local, device
            )
            valid_local = (local_to_global >= 0) & (local_to_global < num_global)
            if proto_mask is not None:
                valid_proto = proto_mask[local_to_global.clamp(min=0, max=max(num_global - 1, 0))]
                if valid_proto.dim() > 1:
                    valid_proto = valid_proto.all(dim=-1)
                valid_local = valid_local & valid_proto.to(device=device, dtype=torch.bool)

            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            gt_local_all = targets[batch_idx]["labels"].to(device=device, dtype=torch.long)
            gt_boxes_all = targets[batch_idx]["boxes"].to(device=device, dtype=pred_logits.dtype)

            for row_pos in range(src_idx.numel()):
                query_idx = int(src_idx[row_pos].item())
                target_idx = int(tgt_idx[row_pos].item())
                if target_idx < 0 or target_idx >= gt_local_all.numel():
                    continue
                gt_local = int(gt_local_all[target_idx].item())
                if gt_local < 0 or gt_local >= num_local or not bool(valid_local[gt_local].item()):
                    continue
                sem_row = sem_scores[query_idx].clone()
                candidate_valid = valid_local.clone()
                if bool(getattr(self, "self_facb_train_same_group_only", False)):
                    group_ids = targets[batch_idx].get("cap_group_ids", None)
                    if group_ids is not None:
                        group_ids = group_ids.to(device=device, dtype=torch.long)
                        if gt_local < int(group_ids.numel()):
                            same_group = torch.zeros_like(candidate_valid)
                            n_group = min(int(group_ids.numel()), num_local)
                            same_group[:n_group] = group_ids[:n_group] == group_ids[gt_local]
                            if int((candidate_valid & same_group).sum().item()) >= 2:
                                candidate_valid = candidate_valid & same_group
                sem_row[~candidate_valid] = float("-inf")
                valid_count = int(candidate_valid.sum().item())
                if valid_count <= 0:
                    continue
                k = min(topk, valid_count)
                top_local = torch.topk(sem_row, k=k, dim=0).indices
                if crm_mode:
                    routed_top1 = float(sem_row[int(top_local[0].item())].item())
                    routed_margin = 0.0
                    if top_local.numel() >= 2:
                        routed_margin = float(
                            (
                                sem_row[int(top_local[0].item())]
                                - sem_row[int(top_local[1].item())]
                            ).item()
                        )
                    route_min = float(
                        getattr(self, "self_facb_crm_min_score", 0.02)
                    )
                    route_max = float(
                        getattr(self, "self_facb_crm_max_score", 0.20)
                    )
                    route_max_margin = float(
                        getattr(self, "self_facb_crm_max_margin", 0.10)
                    )
                    if routed_top1 < route_min:
                        continue
                    if route_max >= 0.0 and routed_top1 > route_max:
                        continue
                    if route_max_margin >= 0.0 and routed_margin > route_max_margin:
                        continue
                gt_tensor = torch.tensor([gt_local], device=device, dtype=torch.long)
                gt_in_top = bool((top_local == gt_local).any().item())
                if bool(getattr(self, "self_facb_force_gt_candidate", True)):
                    if not gt_in_top:
                        if top_local.numel() >= topk:
                            top_local = torch.cat([top_local[:-1], gt_tensor], dim=0)
                        else:
                            top_local = torch.cat([top_local, gt_tensor], dim=0)
                elif not gt_in_top:
                    include_ooc_keep = bool(
                        getattr(self, "self_facb_crm_keep_out_of_candidate", False)
                    )
                    if not (crm_mode and include_ooc_keep):
                        continue

                    cand_global = local_to_global[top_local]
                    cand_semantic = sem_row[top_local]
                    cand_mask = torch.ones(
                        cand_global.numel(), device=device, dtype=torch.bool
                    )
                    if cand_global.numel() < topk:
                        pad = topk - cand_global.numel()
                        cand_global = torch.cat(
                            [cand_global, cand_global.new_zeros(pad)], dim=0
                        )
                        cand_semantic = torch.cat(
                            [
                                cand_semantic,
                                cand_semantic.new_full((pad,), float("-inf")),
                            ],
                            dim=0,
                        )
                        cand_mask = torch.cat(
                            [
                                cand_mask,
                                torch.zeros(pad, device=device, dtype=torch.bool),
                            ],
                            dim=0,
                        )
                    else:
                        cand_global = cand_global[:topk]
                        cand_semantic = cand_semantic[:topk]
                        cand_mask = cand_mask[:topk]

                    sem_margin = 0.0
                    if top_local.numel() >= 2:
                        sem0 = sem_row[int(top_local[0].item())]
                        sem1 = sem_row[int(top_local[1].item())]
                        if torch.isfinite(sem0) and torch.isfinite(sem1):
                            sem_margin = float((sem0 - sem1).detach().item())
                    if self.self_facb_train_box_source == "pred" and pred_boxes is not None:
                        roi_box = pred_boxes[batch_idx, query_idx].detach()
                    else:
                        roi_box = gt_boxes_all[target_idx].detach()
                    boxes.append(roi_box)
                    batch_rows.append(torch.tensor(batch_idx, device=device, dtype=torch.long))
                    candidate_ids.append(cand_global)
                    candidate_masks.append(cand_mask)
                    semantic_candidate_scores.append(cand_semantic)
                    target_slots.append(torch.tensor(-1, device=device, dtype=torch.long))
                    semantic_top1_slots.append(torch.tensor(0, device=device, dtype=torch.long))
                    semantic_margins.append(
                        pred_logits.new_tensor(float(sem_margin), dtype=torch.float32)
                    )
                    same_group_confusers.append(
                        torch.tensor(False, device=device, dtype=torch.bool)
                    )
                    row_is_background.append(
                        torch.tensor(False, device=device, dtype=torch.bool)
                    )
                    row_is_ooc_keep.append(
                        torch.tensor(True, device=device, dtype=torch.bool)
                    )
                    continue

                cand_global = local_to_global[top_local]
                target_pos = torch.nonzero(cand_global == int(local_to_global[gt_local].item()), as_tuple=False)
                if target_pos.numel() == 0:
                    continue
                target_slot = int(target_pos[0, 0].item())
                cand_mask = torch.ones(cand_global.numel(), device=device, dtype=torch.bool)
                if cand_global.numel() < topk:
                    pad = topk - cand_global.numel()
                    cand_semantic = sem_row[top_local]
                    cand_global = torch.cat(
                        [cand_global, cand_global.new_zeros(pad)],
                        dim=0,
                    )
                    cand_semantic = torch.cat(
                        [cand_semantic, cand_semantic.new_full((pad,), float("-inf"))],
                        dim=0,
                    )
                    cand_mask = torch.cat(
                        [cand_mask, torch.zeros(pad, device=device, dtype=torch.bool)],
                        dim=0,
                    )
                else:
                    cand_global = cand_global[:topk]
                    cand_semantic = sem_row[top_local[:topk]]
                    cand_mask = cand_mask[:topk]
                if target_slot >= topk or not bool(cand_mask[target_slot].item()):
                    continue

                top1_slot = -1
                sem_margin = 0.0
                same_group_confuser = False
                if top_local.numel() > 0:
                    semantic_pred_global = int(local_to_global[int(top_local[0].item())].item())
                    semantic_correct = semantic_correct + pred_logits.new_tensor(
                        float(semantic_pred_global == int(local_to_global[gt_local].item())),
                        dtype=torch.float32,
                    )
                    semantic_total += 1
                    top1_global = int(local_to_global[int(top_local[0].item())].item())
                    top1_match = torch.nonzero(cand_global == top1_global, as_tuple=False)
                    if top1_match.numel() > 0:
                        top1_slot = int(top1_match[0, 0].item())
                    if top_local.numel() >= 2:
                        sem0 = sem_row[int(top_local[0].item())]
                        sem1 = sem_row[int(top_local[1].item())]
                        if torch.isfinite(sem0) and torch.isfinite(sem1):
                            sem_margin = float((sem0 - sem1).detach().item())
                    group_ids_for_weight = targets[batch_idx].get("cap_group_ids", None)
                    if (
                        top1_slot >= 0
                        and top1_global != int(local_to_global[gt_local].item())
                        and group_ids_for_weight is not None
                    ):
                        group_ids_for_weight = group_ids_for_weight.to(device=device, dtype=torch.long)
                        if gt_local < int(group_ids_for_weight.numel()):
                            top1_local = int(top_local[0].item())
                            if top1_local < int(group_ids_for_weight.numel()):
                                same_group_confuser = bool(
                                    group_ids_for_weight[top1_local].item()
                                    == group_ids_for_weight[gt_local].item()
                                )

                if self.self_facb_train_box_source == "pred" and pred_boxes is not None:
                    roi_box = pred_boxes[batch_idx, query_idx].detach()
                else:
                    roi_box = gt_boxes_all[target_idx].detach()
                boxes.append(roi_box)
                batch_rows.append(torch.tensor(batch_idx, device=device, dtype=torch.long))
                candidate_ids.append(cand_global)
                candidate_masks.append(cand_mask)
                semantic_candidate_scores.append(cand_semantic)
                target_slots.append(torch.tensor(target_slot, device=device, dtype=torch.long))
                semantic_top1_slots.append(torch.tensor(top1_slot, device=device, dtype=torch.long))
                semantic_margins.append(
                    pred_logits.new_tensor(float(sem_margin), dtype=torch.float32)
                )
                same_group_confusers.append(
                    torch.tensor(same_group_confuser, device=device, dtype=torch.bool)
                )
                row_is_background.append(
                    torch.tensor(False, device=device, dtype=torch.bool)
                )
                row_is_ooc_keep.append(
                    torch.tensor(False, device=device, dtype=torch.bool)
                )

            if crm_mode and pred_boxes is not None:
                background_per_image = max(
                    int(getattr(self, "self_facb_crm_background_rows_per_image", 8)),
                    0,
                )
                if background_per_image > 0:
                    unmatched = torch.ones(
                        pred_logits.shape[1], device=device, dtype=torch.bool
                    )
                    unmatched[src_idx] = False
                    background_max_iou = float(
                        getattr(self, "self_facb_crm_background_max_iou", 0.3)
                    )
                    gt_boxes_for_bg = targets[batch_idx].get("boxes", None)
                    if (
                        background_max_iou >= 0.0
                        and isinstance(gt_boxes_for_bg, torch.Tensor)
                        and gt_boxes_for_bg.numel() > 0
                    ):
                        pred_xyxy_for_bg = box_ops.box_cxcywh_to_xyxy(
                            pred_boxes[batch_idx].detach().float()
                        )
                        gt_xyxy_for_bg = box_ops.box_cxcywh_to_xyxy(
                            gt_boxes_for_bg.to(device=device).detach().float()
                        )
                        max_iou_to_gt = box_ops.box_iou(
                            pred_xyxy_for_bg, gt_xyxy_for_bg
                        )[0].max(dim=1).values
                        unmatched = unmatched & (
                            max_iou_to_gt <= background_max_iou
                        )
                    ranked_bg = sem_scores.clone()
                    ranked_bg[:, ~valid_local] = float("-inf")
                    bg_top2 = ranked_bg.topk(
                        k=min(2, ranked_bg.shape[1]), dim=1
                    ).values
                    bg_top1 = bg_top2[:, 0]
                    bg_margin = (
                        bg_top2[:, 0] - bg_top2[:, 1]
                        if bg_top2.shape[1] > 1
                        else bg_top2[:, 0]
                    )
                    bg_min_score = float(
                        getattr(self, "self_facb_crm_min_score", 0.02)
                    )
                    bg_max_score = float(
                        getattr(self, "self_facb_crm_max_score", 0.20)
                    )
                    bg_valid = unmatched & torch.isfinite(bg_top1) & (
                        bg_top1 >= bg_min_score
                    )
                    if bg_max_score >= 0.0:
                        bg_valid = bg_valid & (bg_top1 <= bg_max_score)
                    bg_max_margin = float(
                        getattr(self, "self_facb_crm_max_margin", 0.10)
                    )
                    if bg_max_margin >= 0.0:
                        bg_valid = bg_valid & (bg_margin <= bg_max_margin)
                    bg_queries = bg_valid.nonzero(as_tuple=False).flatten()
                    if bg_queries.numel() > background_per_image:
                        selected = bg_top1[bg_queries].topk(
                            k=background_per_image, dim=0
                        ).indices
                        bg_queries = bg_queries[selected]

                    for query_tensor in bg_queries:
                        query_idx = int(query_tensor.item())
                        sem_row = ranked_bg[query_idx]
                        valid_count = int(torch.isfinite(sem_row).sum().item())
                        if valid_count < 2:
                            continue
                        k = min(topk, valid_count)
                        top_local = torch.topk(sem_row, k=k, dim=0).indices
                        cand_global = local_to_global[top_local]
                        cand_semantic = sem_row[top_local]
                        cand_mask = torch.ones(
                            cand_global.numel(), device=device, dtype=torch.bool
                        )
                        if cand_global.numel() < topk:
                            pad = topk - cand_global.numel()
                            cand_global = torch.cat(
                                [cand_global, cand_global.new_zeros(pad)], dim=0
                            )
                            cand_semantic = torch.cat(
                                [
                                    cand_semantic,
                                    cand_semantic.new_full((pad,), float("-inf")),
                                ],
                                dim=0,
                            )
                            cand_mask = torch.cat(
                                [
                                    cand_mask,
                                    torch.zeros(
                                        pad, device=device, dtype=torch.bool
                                    ),
                                ],
                                dim=0,
                            )
                        boxes.append(pred_boxes[batch_idx, query_idx].detach())
                        batch_rows.append(
                            torch.tensor(batch_idx, device=device, dtype=torch.long)
                        )
                        candidate_ids.append(cand_global[:topk])
                        candidate_masks.append(cand_mask[:topk])
                        semantic_candidate_scores.append(cand_semantic[:topk])
                        target_slots.append(
                            torch.tensor(-1, device=device, dtype=torch.long)
                        )
                        semantic_top1_slots.append(
                            torch.tensor(0, device=device, dtype=torch.long)
                        )
                        semantic_margins.append(
                            pred_logits.new_tensor(0.0, dtype=torch.float32)
                        )
                        same_group_confusers.append(
                            torch.tensor(False, device=device, dtype=torch.bool)
                        )
                        row_is_background.append(
                            torch.tensor(True, device=device, dtype=torch.bool)
                        )
                        row_is_ooc_keep.append(
                            torch.tensor(False, device=device, dtype=torch.bool)
                        )

        if not boxes:
            return {
                "loss_self_facb": head_zero,
                "self_facb_acc": zero,
                "self_facb_pairwise_acc": zero,
                "self_facb_semantic_acc": zero,
            }

        boxes = torch.stack(boxes, dim=0).to(device=device, dtype=pred_logits.dtype)
        batch_rows = torch.stack(batch_rows, dim=0).to(device=device, dtype=torch.long)
        candidate_ids = torch.stack(candidate_ids, dim=0).to(device=device, dtype=torch.long)
        candidate_masks = torch.stack(candidate_masks, dim=0).to(device=device, dtype=torch.bool)
        semantic_candidate_scores = torch.stack(semantic_candidate_scores, dim=0).to(
            device=device, dtype=torch.float32
        )
        target_slots = torch.stack(target_slots, dim=0).to(device=device, dtype=torch.long)
        semantic_top1_slots = torch.stack(semantic_top1_slots, dim=0).to(device=device, dtype=torch.long)
        semantic_margins = torch.stack(semantic_margins, dim=0).to(device=device, dtype=torch.float32)
        same_group_confusers = torch.stack(same_group_confusers, dim=0).to(device=device, dtype=torch.bool)
        row_is_background = torch.stack(row_is_background, dim=0).to(
            device=device, dtype=torch.bool
        )
        row_is_ooc_keep = torch.stack(row_is_ooc_keep, dim=0).to(
            device=device, dtype=torch.bool
        )

        if proto_mask is not None:
            gathered_mask = proto_mask[candidate_ids.clamp(min=0, max=max(num_global - 1, 0))]
            if gathered_mask.dim() > 2:
                gathered_mask = gathered_mask.all(dim=-1)
            candidate_masks = candidate_masks & gathered_mask.to(device=device, dtype=torch.bool)
        supervised_rows = ~(row_is_background | row_is_ooc_keep)
        target_valid = supervised_rows & candidate_masks.gather(
            1, target_slots.clamp(min=0).view(-1, 1)
        ).squeeze(1)
        row_valid = row_is_background | row_is_ooc_keep | target_valid
        row_valid = row_valid & (candidate_masks.sum(dim=1) >= 2)
        if not bool(row_valid.any().item()):
            return {
                "loss_self_facb": head_zero,
                "self_facb_acc": zero,
                "self_facb_pairwise_acc": zero,
                "self_facb_semantic_acc": zero,
            }
        boxes = boxes[row_valid]
        batch_rows = batch_rows[row_valid]
        candidate_ids = candidate_ids[row_valid]
        candidate_masks = candidate_masks[row_valid]
        semantic_candidate_scores = semantic_candidate_scores[row_valid]
        target_slots = target_slots[row_valid]
        semantic_top1_slots = semantic_top1_slots[row_valid]
        semantic_margins = semantic_margins[row_valid]
        same_group_confusers = same_group_confusers[row_valid]
        row_is_background = row_is_background[row_valid]
        row_is_ooc_keep = row_is_ooc_keep[row_valid]

        roi_tokens = self._extract_self_facb_roi_tokens_for_loss(
            srcs=srcs,
            boxes_cxcywh=boxes,
            batch_indices=batch_rows,
            levels=getattr(self, "self_facb_levels", outputs.get("self_facb_levels", [0])),
            roi_size=int(getattr(self, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7))),
        )
        if bool(getattr(self, "self_facb_lc_mode", False)):
            context_scale = float(getattr(self, "self_facb_lc_context_scale", 1.0))
            if context_scale > 1.0:
                context_tokens = self._extract_self_facb_roi_tokens_for_loss(
                    srcs=srcs,
                    boxes_cxcywh=self._expand_self_facb_boxes_cxcywh(boxes, context_scale),
                    batch_indices=batch_rows,
                    levels=getattr(self, "self_facb_levels", outputs.get("self_facb_levels", [0])),
                    roi_size=int(getattr(self, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7))),
                )
                if context_tokens is not None and context_tokens.numel() > 0:
                    roi_tokens = (
                        context_tokens
                        if roi_tokens is None or roi_tokens.numel() == 0
                        else torch.cat([roi_tokens, context_tokens], dim=1)
                    )
        if roi_tokens is None or roi_tokens.numel() == 0:
            return {
                "loss_self_facb": head_zero,
                "self_facb_acc": zero,
                "self_facb_pairwise_acc": zero,
                "self_facb_semantic_acc": zero,
            }

        candidate_prototypes = proto_bank[candidate_ids]
        if crm_mode and bool(getattr(self, "self_facb_lc_mode", False)):
            eps = 1e-5
            base_prob = semantic_candidate_scores.clamp(min=eps, max=1.0 - eps)
            base_logits = torch.logit(base_prob).masked_fill(
                ~candidate_masks, float("-inf")
            )
            num_candidates = int(candidate_masks.shape[1])
            top1_probs = semantic_candidate_scores[:, :1].expand(-1, num_candidates)
            if num_candidates > 1:
                margin_vals = (
                    semantic_candidate_scores[:, :1]
                    - semantic_candidate_scores[:, 1:2]
                ).expand(-1, num_candidates)
                rank_feature = torch.linspace(
                    0.0,
                    1.0,
                    steps=num_candidates,
                    device=device,
                    dtype=torch.float32,
                ).view(1, -1).expand_as(semantic_candidate_scores)
            else:
                margin_vals = semantic_candidate_scores.new_zeros(semantic_candidate_scores.shape)
                rank_feature = semantic_candidate_scores.new_zeros(semantic_candidate_scores.shape)
            score_features = torch.stack(
                [
                    semantic_candidate_scores.float(),
                    top1_probs.float(),
                    margin_vals.float(),
                    rank_feature.float(),
                ],
                dim=-1,
            )
            lc_out = head.forward_lc(
                roi_tokens,
                candidate_prototypes,
                candidate_masks,
                score_features=score_features,
                class_delta_bound=float(
                    getattr(
                        self,
                        "self_facb_lc_train_class_delta_bound",
                        getattr(self, "self_facb_crm_train_residual_bound", 2.0),
                    )
                ),
                lift_bound=float(getattr(self, "self_facb_lc_train_lift_bound", 4.0)),
            )
            class_delta = lc_out["class_delta"].float().masked_fill(~candidate_masks, 0.0)
            lift = lc_out["lift"].float().masked_fill(~candidate_masks, 0.0)
            lift_confidence = lc_out["lift_confidence"].float().masked_fill(~candidate_masks, 0.0)
            raw_scores = lc_out["raw_scores"].float().masked_fill(~candidate_masks, float("-inf"))
            facet_scores = lc_out["facet_scores"]
            fused_logits = (base_logits + class_delta + lift).masked_fill(
                ~candidate_masks, float("-inf")
            )
            class_logits = (base_logits + class_delta).masked_fill(
                ~candidate_masks, float("-inf")
            )
            supervised_rows = ~(row_is_background | row_is_ooc_keep)
            semantic_top1_valid = (
                supervised_rows
                & (semantic_top1_slots >= 0)
                & (semantic_top1_slots < fused_logits.shape[1])
                & candidate_masks.gather(
                    1,
                    semantic_top1_slots.clamp(
                        min=0, max=fused_logits.shape[1] - 1
                    ).view(-1, 1),
                ).squeeze(1)
            )
            top1_wrong = semantic_top1_valid & (semantic_top1_slots != target_slots)
            top1_correct = semantic_top1_valid & (semantic_top1_slots == target_slots)
            top1_semantic_scores = semantic_candidate_scores.gather(
                1,
                semantic_top1_slots.clamp(
                    min=0, max=fused_logits.shape[1] - 1
                ).view(-1, 1),
            ).squeeze(1)
            ambiguous_mode = bool(
                getattr(self, "self_facb_crm_ambiguous_supervision", False)
            )
            ambiguous_margin = float(
                getattr(self, "self_facb_crm_ambiguous_margin", 0.10)
            )
            ambiguous_min_score = float(
                getattr(self, "self_facb_crm_ambiguous_min_score", -1.0)
            )
            ambiguous_max_score = float(
                getattr(self, "self_facb_crm_ambiguous_max_score", -1.0)
            )
            ambiguous_score_gate = torch.ones_like(top1_correct)
            if ambiguous_min_score >= 0.0:
                ambiguous_score_gate = ambiguous_score_gate & (
                    top1_semantic_scores >= ambiguous_min_score
                )
            if ambiguous_max_score >= 0.0:
                ambiguous_score_gate = ambiguous_score_gate & (
                    top1_semantic_scores <= ambiguous_max_score
                )
            ambiguous_correct = (
                top1_correct
                & (semantic_margins <= ambiguous_margin)
                & ambiguous_score_gate
                if ambiguous_mode
                else fused_logits.new_zeros(fused_logits.shape[0], dtype=torch.bool)
            )
            lc_train_rows = top1_wrong | ambiguous_correct
            easy_rows = supervised_rows & ~lc_train_rows

            pos = fused_logits.gather(
                1, target_slots.clamp(min=0).view(-1, 1)
            ).squeeze(1)
            neg_scores = fused_logits.masked_fill(~candidate_masks, float("-inf")).clone()
            neg_scores.scatter_(1, target_slots.clamp(min=0).view(-1, 1), float("-inf"))
            competitor_scores = neg_scores.max(dim=1).values
            if bool(top1_wrong.any().item()):
                top1_score = fused_logits.gather(
                    1,
                    semantic_top1_slots.clamp(
                        min=0, max=fused_logits.shape[1] - 1
                    ).view(-1, 1),
                ).squeeze(1)
                competitor_scores[top1_wrong] = top1_score[top1_wrong]
            valid_pairwise_rows = lc_train_rows & torch.isfinite(competitor_scores)

            if bool(lc_train_rows.any().item()):
                ce_per_row = F.cross_entropy(
                    class_logits[lc_train_rows],
                    target_slots[lc_train_rows],
                    reduction="none",
                )
                train_weights = class_logits.new_full(
                    (int(lc_train_rows.sum().item()),),
                    float(getattr(self, "self_facb_crm_ambiguous_correct_weight", 0.25)),
                )
                train_wrong = top1_wrong[lc_train_rows]
                train_weights[train_wrong] = float(
                    getattr(self, "self_facb_crm_hard_wrong_weight", 1.0)
                )
                hard_ce = (ce_per_row * train_weights).sum() / train_weights.sum().clamp(min=1e-6)
            else:
                hard_ce = class_delta.sum() * 0.0

            if bool(valid_pairwise_rows.any().item()):
                pairwise_margin = pos[valid_pairwise_rows] - competitor_scores[valid_pairwise_rows]
                pairwise_per_row = F.softplus(
                    fused_logits.new_tensor(
                        float(getattr(self, "self_facb_crm_pairwise_margin", 0.2))
                    )
                    - pairwise_margin
                )
                pairwise_weights = fused_logits.new_full(
                    (int(valid_pairwise_rows.sum().item()),),
                    float(getattr(self, "self_facb_crm_ambiguous_correct_weight", 0.25)),
                )
                valid_wrong = top1_wrong[valid_pairwise_rows]
                pairwise_weights[valid_wrong] = float(
                    getattr(self, "self_facb_crm_hard_wrong_weight", 1.0)
                )
                hard_pairwise = (
                    pairwise_per_row * pairwise_weights
                ).sum() / pairwise_weights.sum().clamp(min=1e-6)
                hard_pairwise_acc = (pairwise_margin > 0).float().mean().detach()
            else:
                hard_pairwise = class_delta.sum() * 0.0
                hard_pairwise_acc = zero

            if bool(lc_train_rows.any().item()):
                floor = fused_logits.new_tensor(
                    float(getattr(self, "self_facb_lc_score_floor", 0.10))
                ).clamp(min=eps, max=1.0 - eps)
                floor_logit = torch.logit(floor)
                lift_pos = fused_logits.gather(
                    1, target_slots.clamp(min=0).view(-1, 1)
                ).squeeze(1)
                lift_loss = F.softplus(floor_logit - lift_pos[lc_train_rows]).mean()
            else:
                lift_loss = lift.sum() * 0.0

            easy_keep = (
                lift[easy_rows].pow(2).mean()
                if bool(easy_rows.any().item())
                else lift.sum() * 0.0
            )
            train_non_gt_mask = (
                lc_train_rows.view(-1, 1).expand_as(candidate_masks)
                & candidate_masks
            )
            train_non_gt_mask.scatter_(
                1, target_slots.clamp(min=0).view(-1, 1), False
            )
            non_gt_keep = (
                lift[train_non_gt_mask].pow(2).mean()
                if bool(train_non_gt_mask.any().item())
                else lift.sum() * 0.0
            )
            background_keep = (
                lift[row_is_background].pow(2).mean()
                if bool(row_is_background.any().item())
                else lift.sum() * 0.0
            )
            ooc_keep = (
                lift[row_is_ooc_keep].pow(2).mean()
                if bool(row_is_ooc_keep.any().item())
                else lift.sum() * 0.0
            )
            residual_reg = class_delta[candidate_masks].pow(2).mean()
            lift_reg = lift[candidate_masks].pow(2).mean()
            loss = (
                float(getattr(self, "self_facb_crm_ce_weight", 0.5)) * hard_ce
                + float(getattr(self, "self_facb_crm_pairwise_weight", 1.0))
                * hard_pairwise
                + float(getattr(self, "self_facb_lc_lift_weight", 2.0))
                * lift_loss
                + float(getattr(self, "self_facb_lc_non_gt_keep_weight", 0.02))
                * non_gt_keep
                + float(getattr(self, "self_facb_crm_easy_keep_weight", 0.02))
                * easy_keep
                + float(getattr(self, "self_facb_crm_background_keep_weight", 0.05))
                * background_keep
                + float(getattr(self, "self_facb_crm_ooc_keep_weight", 0.10))
                * ooc_keep
                + float(getattr(self, "self_facb_crm_residual_reg_weight", 0.001))
                * residual_reg
                + float(getattr(self, "self_facb_lc_lift_reg_weight", 0.001))
                * lift_reg
            )
            pred = fused_logits.argmax(dim=1)
            supervised_count = int(supervised_rows.sum().item())
            acc = (
                (pred[supervised_rows] == target_slots[supervised_rows])
                .float()
                .mean()
                .detach()
                if supervised_count > 0
                else zero
            )
            semantic_acc = (
                semantic_correct / max(semantic_total, 1)
                if semantic_total > 0
                else zero
            )
            gt_lift = lift.gather(1, target_slots.clamp(min=0).view(-1, 1)).squeeze(1)
            gt_fused_score = pos.sigmoid()
            top1_raw = raw_scores.gather(
                1,
                semantic_top1_slots.clamp(
                    min=0, max=raw_scores.shape[1] - 1
                ).view(-1, 1),
            ).squeeze(1)
            gt_raw = raw_scores.gather(1, target_slots.clamp(min=0).view(-1, 1)).squeeze(1)
            app_margin_gt = gt_raw - top1_raw
            valid_lift_rows = lc_train_rows & torch.isfinite(gt_fused_score)
            result = {
                "loss_self_facb": loss,
                "self_facb_acc": acc,
                "self_facb_pairwise_acc": hard_pairwise_acc,
                "self_facb_semantic_acc": semantic_acc.detach(),
                "self_facb_crm_hard_ce": hard_ce.detach(),
                "self_facb_crm_pairwise": hard_pairwise.detach(),
                "self_facb_crm_easy_keep": easy_keep.detach(),
                "self_facb_crm_background_keep": background_keep.detach(),
                "self_facb_crm_ooc_keep": ooc_keep.detach(),
                "self_facb_lc_non_gt_keep": non_gt_keep.detach(),
                "self_facb_crm_residual_reg": residual_reg.detach(),
                "self_facb_crm_hard_rows": fused_logits.new_tensor(float(top1_wrong.sum().item())).detach(),
                "self_facb_crm_ambiguous_rows": fused_logits.new_tensor(float(ambiguous_correct.sum().item())).detach(),
                "self_facb_crm_train_rows": fused_logits.new_tensor(float(lc_train_rows.sum().item())).detach(),
                "self_facb_crm_easy_rows": fused_logits.new_tensor(float(easy_rows.sum().item())).detach(),
                "self_facb_crm_background_rows": fused_logits.new_tensor(float(row_is_background.sum().item())).detach(),
                "self_facb_crm_ooc_keep_rows": fused_logits.new_tensor(float(row_is_ooc_keep.sum().item())).detach(),
                "self_facb_num_rows": fused_logits.new_tensor(float(fused_logits.shape[0])).detach(),
                "self_facb_lc_lift_loss": lift_loss.detach(),
                "self_facb_lc_lift_reg": lift_reg.detach(),
                "self_facb_lc_lift_abs_mean": lift[candidate_masks].abs().mean().detach(),
                "self_facb_lc_lift_abs_max": lift[candidate_masks].abs().max().detach(),
                "self_facb_lc_lift_gt_mean": (
                    gt_lift[valid_lift_rows].mean().detach()
                    if bool(valid_lift_rows.any().item())
                    else zero
                ),
                "self_facb_lc_fused_gt_score_q50": (
                    gt_fused_score[valid_lift_rows].quantile(0.50).detach()
                    if bool(valid_lift_rows.any().item())
                    else zero
                ),
                "self_facb_lc_fused_gt_score_mean": (
                    gt_fused_score[valid_lift_rows].mean().detach()
                    if bool(valid_lift_rows.any().item())
                    else zero
                ),
                "self_facb_lc_lift_confidence_gt_mean": (
                    lift_confidence.gather(1, target_slots.clamp(min=0).view(-1, 1)).squeeze(1)[valid_lift_rows].mean().detach()
                    if bool(valid_lift_rows.any().item())
                    else zero
                ),
                "self_facb_lc_app_margin_gt_q50": (
                    app_margin_gt[valid_lift_rows].quantile(0.50).detach()
                    if bool(valid_lift_rows.any().item())
                    else zero
                ),
                "self_facb_lc_background_lift_abs_mean": (
                    lift[row_is_background].abs().mean().detach()
                    if bool(row_is_background.any().item())
                    else zero
                ),
                "self_facb_lc_ooc_lift_abs_mean": (
                    lift[row_is_ooc_keep].abs().mean().detach()
                    if bool(row_is_ooc_keep.any().item())
                    else zero
                ),
                "self_facb_crm_residual_abs_mean": class_delta[candidate_masks].abs().mean().detach(),
                "self_facb_crm_residual_abs_max": class_delta[candidate_masks].abs().max().detach(),
            }
            if (
                facet_scores is not None
                and facet_scores.dim() == 3
                and facet_scores.shape[-1] > 0
                and supervised_count > 0
            ):
                for facet_idx in range(int(facet_scores.shape[-1])):
                    facet_logits = facet_scores[..., facet_idx].float().masked_fill(
                        ~candidate_masks, float("-inf")
                    )
                    result[f"self_facb_facet{facet_idx}_acc"] = (
                        facet_logits[supervised_rows].argmax(dim=1)
                        == target_slots[supervised_rows]
                    ).float().mean().detach()
            return result
        if crm_mode:
            residual, raw_scores, facet_scores = head.forward_residual(
                roi_tokens,
                candidate_prototypes,
                candidate_masks,
                residual_bound=float(
                    getattr(
                        self,
                        "self_facb_crm_train_residual_bound",
                        getattr(self, "self_facb_crm_residual_bound", 0.5),
                    )
                ),
            )
            residual = residual.float().masked_fill(~candidate_masks, 0.0)
            eps = 1e-5
            base_prob = semantic_candidate_scores.clamp(min=eps, max=1.0 - eps)
            base_logits = torch.logit(base_prob).masked_fill(
                ~candidate_masks, float("-inf")
            )
            scores = (base_logits + residual).masked_fill(
                ~candidate_masks, float("-inf")
            )
            supervised_rows = ~(row_is_background | row_is_ooc_keep)
            semantic_top1_valid = (
                supervised_rows
                & (semantic_top1_slots >= 0)
                & (semantic_top1_slots < scores.shape[1])
                & candidate_masks.gather(
                    1,
                    semantic_top1_slots.clamp(
                        min=0, max=scores.shape[1] - 1
                    ).view(-1, 1),
                ).squeeze(1)
            )
            top1_wrong = semantic_top1_valid & (semantic_top1_slots != target_slots)
            top1_correct = semantic_top1_valid & (semantic_top1_slots == target_slots)
            ambiguous_mode = bool(
                getattr(self, "self_facb_crm_ambiguous_supervision", False)
            )
            ambiguous_margin = float(
                getattr(self, "self_facb_crm_ambiguous_margin", 0.10)
            )
            top1_semantic_scores = semantic_candidate_scores.gather(
                1,
                semantic_top1_slots.clamp(
                    min=0, max=scores.shape[1] - 1
                ).view(-1, 1),
            ).squeeze(1)
            ambiguous_min_score = float(
                getattr(self, "self_facb_crm_ambiguous_min_score", -1.0)
            )
            ambiguous_max_score = float(
                getattr(self, "self_facb_crm_ambiguous_max_score", -1.0)
            )
            ambiguous_score_gate = torch.ones_like(top1_correct)
            if ambiguous_min_score >= 0.0:
                ambiguous_score_gate = ambiguous_score_gate & (
                    top1_semantic_scores >= ambiguous_min_score
                )
            if ambiguous_max_score >= 0.0:
                ambiguous_score_gate = ambiguous_score_gate & (
                    top1_semantic_scores <= ambiguous_max_score
                )
            ambiguous_correct = (
                top1_correct
                & (semantic_margins <= ambiguous_margin)
                & ambiguous_score_gate
                if ambiguous_mode
                else scores.new_zeros(scores.shape[0], dtype=torch.bool)
            )
            crm_train_rows = top1_wrong | ambiguous_correct
            easy_rows = supervised_rows & ~crm_train_rows

            pos = scores.gather(
                1, target_slots.clamp(min=0).view(-1, 1)
            ).squeeze(1)
            base_pos = base_logits.gather(
                1, target_slots.clamp(min=0).view(-1, 1)
            ).squeeze(1)
            neg_scores = scores.masked_fill(~candidate_masks, float("-inf")).clone()
            neg_scores.scatter_(1, target_slots.clamp(min=0).view(-1, 1), float("-inf"))
            hard_neg = neg_scores.max(dim=1).values
            competitor_scores = hard_neg.clone()
            base_competitor_scores = base_logits.masked_fill(
                ~candidate_masks, float("-inf")
            ).clone()
            base_competitor_scores.scatter_(
                1, target_slots.clamp(min=0).view(-1, 1), float("-inf")
            )
            base_competitor_scores = base_competitor_scores.max(dim=1).values
            if bool(top1_wrong.any().item()):
                top1_score = scores.gather(
                    1,
                    semantic_top1_slots.clamp(
                        min=0, max=scores.shape[1] - 1
                    ).view(-1, 1),
                ).squeeze(1)
                competitor_scores[top1_wrong] = top1_score[top1_wrong]
                base_top1_score = base_logits.gather(
                    1,
                    semantic_top1_slots.clamp(
                        min=0, max=scores.shape[1] - 1
                    ).view(-1, 1),
                ).squeeze(1)
                base_competitor_scores[top1_wrong] = base_top1_score[top1_wrong]
            valid_pairwise_rows = crm_train_rows & torch.isfinite(competitor_scores)
            hard_base_gap = base_pos[top1_wrong] - base_competitor_scores[top1_wrong]

            if bool(crm_train_rows.any().item()):
                ce_per_row = F.cross_entropy(
                    scores[crm_train_rows],
                    target_slots[crm_train_rows],
                    reduction="none",
                )
                train_weights = scores.new_full(
                    (int(crm_train_rows.sum().item()),),
                    float(getattr(self, "self_facb_crm_ambiguous_correct_weight", 0.25)),
                )
                train_wrong = top1_wrong[crm_train_rows]
                train_weights[train_wrong] = float(
                    getattr(self, "self_facb_crm_hard_wrong_weight", 1.0)
                )
                hard_ce = (ce_per_row * train_weights).sum() / train_weights.sum().clamp(min=1e-6)
            else:
                hard_ce = residual.sum() * 0.0

            if bool(valid_pairwise_rows.any().item()):
                pairwise_margin = pos[valid_pairwise_rows] - competitor_scores[valid_pairwise_rows]
                pairwise_per_row = F.softplus(
                    scores.new_tensor(
                        float(getattr(self, "self_facb_crm_pairwise_margin", 0.2))
                    )
                    - pairwise_margin
                )
                pairwise_weights = scores.new_full(
                    (int(valid_pairwise_rows.sum().item()),),
                    float(getattr(self, "self_facb_crm_ambiguous_correct_weight", 0.25)),
                )
                valid_wrong = top1_wrong[valid_pairwise_rows]
                pairwise_weights[valid_wrong] = float(
                    getattr(self, "self_facb_crm_hard_wrong_weight", 1.0)
                )
                hard_pairwise = (
                    pairwise_per_row * pairwise_weights
                ).sum() / pairwise_weights.sum().clamp(min=1e-6)
                hard_pairwise_acc = (pairwise_margin > 0).float().mean().detach()
            else:
                hard_pairwise = residual.sum() * 0.0
                hard_pairwise_acc = zero

            easy_keep = (
                residual[easy_rows].pow(2).mean()
                if bool(easy_rows.any().item())
                else residual.sum() * 0.0
            )
            background_keep = (
                residual[row_is_background].pow(2).mean()
                if bool(row_is_background.any().item())
                else residual.sum() * 0.0
            )
            ooc_keep = (
                residual[row_is_ooc_keep].pow(2).mean()
                if bool(row_is_ooc_keep.any().item())
                else residual.sum() * 0.0
            )
            residual_reg = residual[candidate_masks].pow(2).mean()
            loss = (
                float(getattr(self, "self_facb_crm_ce_weight", 1.0)) * hard_ce
                + float(getattr(self, "self_facb_crm_pairwise_weight", 1.0))
                * hard_pairwise
                + float(getattr(self, "self_facb_crm_easy_keep_weight", 0.5))
                * easy_keep
                + float(getattr(self, "self_facb_crm_background_keep_weight", 1.0))
                * background_keep
                + float(getattr(self, "self_facb_crm_ooc_keep_weight", 1.0))
                * ooc_keep
                + float(getattr(self, "self_facb_crm_residual_reg_weight", 0.01))
                * residual_reg
            )
            pred = scores.argmax(dim=1)
            supervised_count = int(supervised_rows.sum().item())
            acc = (
                (pred[supervised_rows] == target_slots[supervised_rows])
                .float()
                .mean()
                .detach()
                if supervised_count > 0
                else zero
            )
            semantic_acc = (
                semantic_correct / max(semantic_total, 1)
                if semantic_total > 0
                else zero
            )
            result = {
                "loss_self_facb": loss,
                "self_facb_acc": acc,
                "self_facb_pairwise_acc": hard_pairwise_acc,
                "self_facb_semantic_acc": semantic_acc.detach(),
                "self_facb_crm_hard_ce": hard_ce.detach(),
                "self_facb_crm_pairwise": hard_pairwise.detach(),
                "self_facb_crm_easy_keep": easy_keep.detach(),
                "self_facb_crm_background_keep": background_keep.detach(),
                "self_facb_crm_ooc_keep": ooc_keep.detach(),
                "self_facb_crm_residual_reg": residual_reg.detach(),
                "self_facb_crm_residual_abs_mean": residual[candidate_masks]
                .abs()
                .mean()
                .detach(),
                "self_facb_crm_residual_abs_max": residual[candidate_masks]
                .abs()
                .max()
                .detach(),
                "self_facb_crm_hard_rows": scores.new_tensor(
                    float(top1_wrong.sum().item())
                ).detach(),
                "self_facb_crm_ambiguous_rows": scores.new_tensor(
                    float(ambiguous_correct.sum().item())
                ).detach(),
                "self_facb_crm_train_rows": scores.new_tensor(
                    float(crm_train_rows.sum().item())
                ).detach(),
                "self_facb_crm_easy_rows": scores.new_tensor(
                    float(easy_rows.sum().item())
                ).detach(),
                "self_facb_crm_background_rows": scores.new_tensor(
                    float(row_is_background.sum().item())
                ).detach(),
                "self_facb_crm_ooc_keep_rows": scores.new_tensor(
                    float(row_is_ooc_keep.sum().item())
                ).detach(),
                "self_facb_num_rows": scores.new_tensor(
                    float(scores.shape[0])
                ).detach(),
            }
            if bool(hard_base_gap.numel() > 0):
                result.update(
                    {
                        "self_facb_crm_hard_base_gap_mean": hard_base_gap.mean().detach(),
                        "self_facb_crm_hard_base_gap_q25": hard_base_gap.quantile(0.25).detach(),
                        "self_facb_crm_hard_base_gap_q50": hard_base_gap.quantile(0.50).detach(),
                        "self_facb_crm_hard_base_gap_q75": hard_base_gap.quantile(0.75).detach(),
                    }
                )
            else:
                result.update(
                    {
                        "self_facb_crm_hard_base_gap_mean": zero,
                        "self_facb_crm_hard_base_gap_q25": zero,
                        "self_facb_crm_hard_base_gap_q50": zero,
                        "self_facb_crm_hard_base_gap_q75": zero,
                    }
                )
            if (
                facet_scores is not None
                and facet_scores.dim() == 3
                and facet_scores.shape[-1] > 0
                and supervised_count > 0
            ):
                for facet_idx in range(int(facet_scores.shape[-1])):
                    facet_logits = facet_scores[..., facet_idx].float().masked_fill(
                        ~candidate_masks, float("-inf")
                    )
                    result[f"self_facb_facet{facet_idx}_acc"] = (
                        facet_logits[supervised_rows].argmax(dim=1)
                        == target_slots[supervised_rows]
                    ).float().mean().detach()
            return result

        scores, facet_scores = head(
            roi_tokens,
            candidate_prototypes,
            candidate_masks,
            return_facet_scores=True,
        )
        scores = scores.float().masked_fill(~candidate_masks, float("-inf"))
        semantic_top1_valid = (
            (semantic_top1_slots >= 0)
            & (semantic_top1_slots < scores.shape[1])
            & candidate_masks.gather(
                1, semantic_top1_slots.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
            ).squeeze(1)
        )
        semantic_top1_wrong = semantic_top1_valid & (semantic_top1_slots != target_slots)
        low_semantic_margin = semantic_margins < float(
            getattr(self, "self_facb_hard_case_semantic_margin", 0.03)
        )
        hard_cases = semantic_top1_wrong | low_semantic_margin
        sample_weights = scores.new_ones(scores.shape[0])
        hard_case_weight = float(getattr(self, "self_facb_hard_case_weight", 0.0))
        if hard_case_weight > 0.0:
            sample_weights = sample_weights + hard_case_weight * hard_cases.to(sample_weights.dtype)
        same_group_hard_weight = float(getattr(self, "self_facb_same_group_hard_weight", 0.0))
        if same_group_hard_weight > 0.0:
            sample_weights = sample_weights + same_group_hard_weight * same_group_confusers.to(sample_weights.dtype)
        sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)

        ce_per_row = F.cross_entropy(scores, target_slots, reduction="none")
        ce = (ce_per_row * sample_weights).mean()

        pos_scores = scores.gather(1, target_slots.view(-1, 1)).squeeze(1)
        neg_scores = scores.masked_fill(~candidate_masks, float("-inf")).clone()
        neg_scores.scatter_(1, target_slots.view(-1, 1), float("-inf"))
        hard_neg = neg_scores.max(dim=1).values
        valid_neg = torch.isfinite(hard_neg)
        if bool(valid_neg.any().item()):
            margins = pos_scores[valid_neg] - hard_neg[valid_neg]
            pairwise_per_row = F.softplus(
                scores.new_tensor(float(getattr(self, "self_facb_pairwise_loss_margin", 0.0)))
                - margins
            )
            pairwise = (pairwise_per_row * sample_weights[valid_neg]).mean()
            pairwise_acc = (margins > 0).float().mean().detach()
        else:
            pairwise = ce.new_zeros(())
            pairwise_acc = zero

        semantic_top1_loss_weight = float(
            getattr(self, "self_facb_semantic_top1_loss_weight", 0.0)
        )
        semantic_top1_valid_count = int(semantic_top1_wrong.sum().item())
        if semantic_top1_loss_weight > 0.0 and bool(semantic_top1_wrong.any().item()):
            top1_scores = scores.gather(
                1, semantic_top1_slots.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
            ).squeeze(1)
            top1_margins = pos_scores[semantic_top1_wrong] - top1_scores[semantic_top1_wrong]
            top1_loss_per_row = F.softplus(
                scores.new_tensor(
                    float(getattr(self, "self_facb_semantic_top1_loss_margin", 0.2))
                )
                - top1_margins
            )
            semantic_top1_loss = (
                top1_loss_per_row * sample_weights[semantic_top1_wrong]
            ).mean()
            semantic_top1_acc = (top1_margins > 0).float().mean().detach()
        else:
            semantic_top1_loss = ce.new_zeros(())
            semantic_top1_acc = zero

        loss = (
            float(getattr(self, "self_facb_pairwise_loss_weight", 1.0)) * pairwise
            + float(getattr(self, "self_facb_ce_aux_weight", 0.5)) * ce
            + semantic_top1_loss_weight * semantic_top1_loss
        )
        pred = scores.argmax(dim=1)
        acc = (pred == target_slots).float().mean().detach()
        semantic_acc = (
            semantic_correct / max(semantic_total, 1)
            if semantic_total > 0
            else zero
        )
        result = {
            "loss_self_facb": loss,
            "self_facb_acc": acc,
            "self_facb_pairwise_acc": pairwise_acc,
            "self_facb_semantic_acc": semantic_acc.detach(),
            "self_facb_ce": ce.detach(),
            "self_facb_pairwise_loss": pairwise.detach(),
            "self_facb_semantic_top1_loss": semantic_top1_loss.detach(),
            "self_facb_semantic_top1_acc": semantic_top1_acc,
            "self_facb_semantic_top1_rows": scores.new_tensor(
                float(semantic_top1_valid_count)
            ).detach(),
            "self_facb_hard_case_frac": hard_cases.float().mean().detach(),
            "self_facb_sample_weight_mean": sample_weights.mean().detach(),
            "self_facb_num_rows": scores.new_tensor(float(scores.shape[0])).detach(),
        }
        if facet_scores is not None and facet_scores.dim() == 3 and facet_scores.shape[-1] > 0:
            for facet_idx in range(int(facet_scores.shape[-1])):
                facet_logits = facet_scores[..., facet_idx].float().masked_fill(
                    ~candidate_masks, float("-inf")
                )
                result[f"self_facb_facet{facet_idx}_acc"] = (
                    facet_logits.argmax(dim=1) == target_slots
                ).float().mean().detach()
        return result

    def loss_appearance_text_alignment(self, outputs, targets, indices, num_boxes):
        text_logits = outputs.get("appearance_text_logits", None)
        if text_logits is None or not self.use_appearance_text_alignment:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_text_alignment": zero}

        total_loss = outputs["pred_logits"].new_tensor(0.0)
        total_count = 0
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            batch_logits = text_logits[batch_idx, src_idx]
            if batch_logits.numel() == 0:
                continue
            target_labels = targets[batch_idx]["labels"][tgt_idx].to(device=batch_logits.device, dtype=torch.long)
            if target_labels.numel() == 0:
                continue
            valid_mask = (target_labels >= 0) & (target_labels < batch_logits.shape[-1])
            if not bool(valid_mask.any()):
                continue
            batch_logits = batch_logits[valid_mask]
            target_labels = target_labels[valid_mask]
            batch_logits = torch.nan_to_num(batch_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            total_loss = total_loss + F.cross_entropy(batch_logits, target_labels, reduction="sum")
            total_count += int(target_labels.numel())

        if total_count == 0:
            zero = outputs["pred_boxes"].new_tensor(0.0)
            return {"loss_appearance_text_alignment": zero}

        return {
            "loss_appearance_text_alignment": total_loss / max(total_count, 1),
        }

    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        pred_logits = pred_logits.clamp(min=-20.0, max=20.0)
        new_targets = outputs['one_hot'].to(pred_logits.device)
        text_mask = outputs.get(
            'query_token_ignore_mask',
            outputs.get('loss_text_mask', outputs['text_mask'])
        )

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to
        
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample may supervise a different subset of caption tokens.
            # Keep the [bs, num_queries, num_tokens] structure so we can normalize
            # per query instead of summing over all valid tokens, which makes the
            # loss explode on long captions or small-box batches.
            text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)
            text_mask = text_mask.unsqueeze(1).expand(-1, pred_logits.size(1), -1)
        else:
            text_mask = torch.ones_like(pred_logits, dtype=torch.bool)

        new_targets = new_targets.float()
        dgcl_loss_debug = (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_loss_debug_printed", False)
        )
        if dgcl_loss_debug:
            print(
                "[dgcl-flow][loss] pred_logits.shape={} new_targets.shape={} text_mask.shape={}".format(
                    tuple(pred_logits.shape),
                    tuple(new_targets.shape),
                    tuple(text_mask.shape),
                ),
                flush=True,
            )
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if dgcl_loss_debug:
            print(
                "[dgcl-flow][loss] p.shape={} p_t.shape={} token_loss.shape={}".format(
                    tuple(p.shape), tuple(p_t.shape), tuple(loss.shape)
                ),
                flush=True,
            )
            self._dgcl_flow_loss_debug_printed = True

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        loss = loss * text_mask.to(dtype=loss.dtype)
        # Normalize by num_boxes (same as bbox/giou) instead of averaging
        # per-query over all tokens — the latter dilutes positive-token
        # signal ~100x, making CE loss negligible vs. giou.
        loss = loss.sum() / num_boxes
        
        losses = {'loss_ce': loss}
        return losses

    def loss_facb_agf(self, outputs, targets, indices, num_boxes):
        """FACB-guided positive-token focal reweighting.

        FACB only selects appearance-resolvable low-confidence matched rows.
        The detector's native token logits receive the gradient; the FACB gate
        is detached so the detector classification loss cannot train the FACB
        verifier or create a new inference-time score path.
        """
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        if not bool(getattr(self, "use_facb_agf", False)):
            return {"loss_facb_agf": zero}
        head = getattr(self, "self_facb_head_ref", None)
        srcs = outputs.get("self_facb_srcs", None)
        proto_bank = outputs.get("self_facb_proto_bank", None)
        label_map_list = outputs.get("_criterion_label_map_list", None)
        pred_boxes = outputs.get("pred_boxes", None)
        if head is None or srcs is None or proto_bank is None or label_map_list is None:
            return {"loss_facb_agf": zero}

        pred_logits = outputs["pred_logits"]
        pred_logits_safe = torch.nan_to_num(
            pred_logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)
        one_hot = outputs["one_hot"].to(device=pred_logits.device, dtype=torch.float32)
        text_mask = outputs.get(
            "query_token_ignore_mask",
            outputs.get("loss_text_mask", outputs["text_mask"]),
        )
        if text_mask is not None:
            text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)
        device = pred_logits.device
        proto_bank = proto_bank.to(device=device, dtype=pred_logits.dtype)
        if proto_bank.dim() not in (2, 3) or int(proto_bank.shape[0]) <= 0:
            return {"loss_facb_agf": zero}
        num_global = int(proto_bank.shape[0])
        proto_mask = outputs.get("self_facb_proto_bank_mask", None)
        if proto_mask is not None:
            proto_mask = proto_mask.to(device=device, dtype=torch.bool)

        rows = []
        boxes = []
        batch_rows = []
        candidate_ids = []
        candidate_masks = []
        target_slots = []
        semantic_top1_slots = []
        semantic_scores_for_gt = []
        semantic_correct = pred_logits.new_tensor(0.0, dtype=torch.float32)
        semantic_total = 0
        topk = max(int(getattr(self, "facb_agf_topk", getattr(self, "self_facb_topk", 10))), 2)
        max_score = float(getattr(self, "facb_agf_max_score", 0.20))
        max_margin = float(getattr(self, "facb_agf_max_margin", 0.10))
        min_score = float(getattr(self, "facb_agf_min_score", 0.0))

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0 or batch_idx >= len(label_map_list):
                continue
            label_map = label_map_list[batch_idx]
            if label_map is None or label_map.numel() == 0:
                continue
            label_map = label_map.to(device=device, dtype=torch.float32)
            num_local = int(label_map.shape[0])
            if num_local <= 0:
                continue
            weights = label_map / label_map.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sem_scores = torch.nan_to_num(
                pred_logits_safe[batch_idx].detach().sigmoid().float() @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            local_to_global = self._local_to_global_label_ids(targets[batch_idx], num_local, device)
            valid_local = (local_to_global >= 0) & (local_to_global < num_global)
            if proto_mask is not None:
                valid_proto = proto_mask[local_to_global.clamp(min=0, max=max(num_global - 1, 0))]
                if valid_proto.dim() > 1:
                    valid_proto = valid_proto.all(dim=-1)
                valid_local = valid_local & valid_proto.to(device=device, dtype=torch.bool)

            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            gt_local_all = targets[batch_idx]["labels"].to(device=device, dtype=torch.long)
            gt_boxes_all = targets[batch_idx]["boxes"].to(device=device, dtype=pred_logits.dtype)
            for row_pos in range(src_idx.numel()):
                query_idx = int(src_idx[row_pos].item())
                target_idx = int(tgt_idx[row_pos].item())
                if target_idx < 0 or target_idx >= gt_local_all.numel():
                    continue
                gt_local = int(gt_local_all[target_idx].item())
                if gt_local < 0 or gt_local >= num_local or not bool(valid_local[gt_local].item()):
                    continue
                sem_row = sem_scores[query_idx].clone()
                sem_row[~valid_local] = float("-inf")
                valid_count = int(torch.isfinite(sem_row).sum().item())
                if valid_count < 2:
                    continue
                k = min(topk, valid_count)
                top_local = torch.topk(sem_row, k=k, dim=0).indices
                gt_in_top = bool((top_local == gt_local).any().item())
                if not gt_in_top:
                    continue
                top1_score = float(sem_row[int(top_local[0].item())].item())
                top2_score = float(sem_row[int(top_local[1].item())].item()) if k >= 2 else 0.0
                margin = top1_score - top2_score
                if top1_score < min_score:
                    continue
                if max_score >= 0.0 and top1_score > max_score:
                    continue
                if max_margin >= 0.0 and margin > max_margin:
                    continue

                cand_global = local_to_global[top_local]
                target_pos = torch.nonzero(
                    cand_global == int(local_to_global[gt_local].item()),
                    as_tuple=False,
                )
                if target_pos.numel() == 0:
                    continue
                target_slot = int(target_pos[0, 0].item())
                cand_mask = torch.ones(cand_global.numel(), device=device, dtype=torch.bool)
                if cand_global.numel() < topk:
                    pad = topk - cand_global.numel()
                    cand_global = torch.cat([cand_global, cand_global.new_zeros(pad)], dim=0)
                    cand_mask = torch.cat(
                        [cand_mask, torch.zeros(pad, device=device, dtype=torch.bool)],
                        dim=0,
                    )
                else:
                    cand_global = cand_global[:topk]
                    cand_mask = cand_mask[:topk]
                if target_slot >= topk or not bool(cand_mask[target_slot].item()):
                    continue
                top1_global = int(local_to_global[int(top_local[0].item())].item())
                gt_global = int(local_to_global[gt_local].item())
                semantic_correct = semantic_correct + pred_logits.new_tensor(
                    float(top1_global == gt_global), dtype=torch.float32
                )
                semantic_total += 1
                top1_match = torch.nonzero(cand_global == top1_global, as_tuple=False)
                top1_slot = int(top1_match[0, 0].item()) if top1_match.numel() > 0 else 0
                roi_box = (
                    pred_boxes[batch_idx, query_idx].detach()
                    if pred_boxes is not None and self.self_facb_train_box_source == "pred"
                    else gt_boxes_all[target_idx].detach()
                )
                rows.append((batch_idx, query_idx))
                boxes.append(roi_box)
                batch_rows.append(torch.tensor(batch_idx, device=device, dtype=torch.long))
                candidate_ids.append(cand_global)
                candidate_masks.append(cand_mask)
                target_slots.append(torch.tensor(target_slot, device=device, dtype=torch.long))
                semantic_top1_slots.append(torch.tensor(top1_slot, device=device, dtype=torch.long))
                semantic_scores_for_gt.append(pred_logits.new_tensor(float(sem_row[gt_local].item())))

        if not rows:
            return {
                "loss_facb_agf": zero,
                "facb_agf_rows": zero.detach(),
                "facb_agf_weight_mean": zero.detach(),
                "facb_agf_gt_sem_score_q50": zero.detach(),
            }

        boxes = torch.stack(boxes, dim=0).to(device=device, dtype=pred_logits.dtype)
        batch_rows_t = torch.stack(batch_rows, dim=0).to(device=device, dtype=torch.long)
        candidate_ids = torch.stack(candidate_ids, dim=0).to(device=device, dtype=torch.long)
        candidate_masks = torch.stack(candidate_masks, dim=0).to(device=device, dtype=torch.bool)
        target_slots = torch.stack(target_slots, dim=0).to(device=device, dtype=torch.long)
        semantic_top1_slots = torch.stack(semantic_top1_slots, dim=0).to(device=device, dtype=torch.long)
        semantic_scores_for_gt = torch.stack(semantic_scores_for_gt, dim=0).to(device=device, dtype=torch.float32)
        if proto_mask is not None:
            gathered_mask = proto_mask[candidate_ids.clamp(min=0, max=max(num_global - 1, 0))]
            if gathered_mask.dim() > 2:
                gathered_mask = gathered_mask.all(dim=-1)
            candidate_masks = candidate_masks & gathered_mask.to(device=device, dtype=torch.bool)
        target_valid = candidate_masks.gather(1, target_slots.view(-1, 1)).squeeze(1)
        top1_valid = candidate_masks.gather(
            1, semantic_top1_slots.clamp(min=0, max=candidate_masks.shape[1] - 1).view(-1, 1)
        ).squeeze(1)
        valid_rows = target_valid & top1_valid
        if not bool(valid_rows.any().item()):
            return {"loss_facb_agf": zero}
        boxes = boxes[valid_rows]
        batch_rows_t = batch_rows_t[valid_rows]
        candidate_ids = candidate_ids[valid_rows]
        candidate_masks = candidate_masks[valid_rows]
        target_slots = target_slots[valid_rows]
        semantic_top1_slots = semantic_top1_slots[valid_rows]
        semantic_scores_for_gt = semantic_scores_for_gt[valid_rows]
        rows = [row for row, keep in zip(rows, valid_rows.detach().cpu().tolist()) if keep]

        roi_tokens = self._extract_self_facb_roi_tokens_for_loss(
            srcs=srcs,
            boxes_cxcywh=boxes,
            batch_indices=batch_rows_t,
            levels=getattr(self, "self_facb_levels", outputs.get("self_facb_levels", [0])),
            roi_size=int(getattr(self, "self_facb_roi_size", outputs.get("self_facb_roi_size", 7))),
        )
        if roi_tokens is None or roi_tokens.numel() == 0:
            return {"loss_facb_agf": zero}
        candidate_prototypes = proto_bank[candidate_ids]
        with torch.no_grad():
            evidence, _facet_scores = head(
                roi_tokens,
                candidate_prototypes,
                candidate_masks,
                return_facet_scores=True,
            )
            evidence = evidence.float().masked_fill(~candidate_masks, float("-inf"))
            e_gt = evidence.gather(1, target_slots.view(-1, 1)).squeeze(1)
            e_top1 = evidence.gather(
                1,
                semantic_top1_slots.clamp(min=0, max=evidence.shape[1] - 1).view(-1, 1),
            ).squeeze(1)
            margin = e_gt - e_top1
            tau = max(float(getattr(self, "facb_agf_temperature", 0.10)), 1e-6)
            gate = torch.sigmoid(margin / tau)
            gate = torch.where(margin > 0, gate, gate.new_zeros(gate.shape))
            min_extra = max(float(getattr(self, "facb_agf_min_extra_weight", 0.0)), 0.0)
            row_weight = (
                1.0 + min_extra + float(getattr(self, "facb_agf_alpha", 3.0)) * gate
            ).detach()
            cap = float(getattr(self, "facb_agf_weight_cap", 5.0))
            if cap > 0:
                row_weight = row_weight.clamp(max=cap)

        alpha = self.focal_alpha
        gamma = self.focal_gamma
        terms = []
        active_weights = []
        selected_rows = int(row_weight.numel())
        for row_idx, (batch_idx, query_idx) in enumerate(rows):
            extra_weight = row_weight[row_idx] - 1.0
            if float(extra_weight.detach().item()) <= 1e-6:
                continue
            pos_mask = one_hot[batch_idx, query_idx].bool()
            if text_mask is not None:
                pos_mask = pos_mask & text_mask[batch_idx]
            if not bool(pos_mask.any().item()):
                continue
            logits = pred_logits_safe[batch_idx, query_idx][pos_mask]
            target = torch.ones_like(logits)
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p = torch.sigmoid(logits)
            focal = ce * ((1 - p) ** gamma)
            if alpha >= 0:
                focal = alpha * focal
            terms.append(extra_weight.to(dtype=focal.dtype) * focal.mean())
            active_weights.append(row_weight[row_idx])

        if not terms:
            return {
                "loss_facb_agf": zero,
                "facb_agf_rows": pred_logits.new_tensor(float(selected_rows)).detach(),
                "facb_agf_active_rows": zero.detach(),
                "facb_agf_weight_mean": row_weight.float().mean().detach(),
                "facb_agf_weight_max": row_weight.float().max().detach(),
                "facb_agf_gate_active_frac": zero.detach(),
                "facb_agf_gt_sem_score_q50": zero.detach(),
            }
        active_weights = torch.stack(active_weights).float()
        loss = torch.stack(terms).sum() / num_boxes
        gt_scores_sorted = semantic_scores_for_gt.detach().float().sort().values
        q50 = gt_scores_sorted[int(gt_scores_sorted.numel() // 2)] if gt_scores_sorted.numel() else zero
        return {
            "loss_facb_agf": loss,
            "facb_agf_rows": pred_logits.new_tensor(float(selected_rows)).detach(),
            "facb_agf_active_rows": pred_logits.new_tensor(float(len(terms))).detach(),
            "facb_agf_weight_mean": row_weight.float().mean().detach(),
            "facb_agf_weight_max": row_weight.float().max().detach(),
            "facb_agf_gate_active_frac": (row_weight.float() > 1.0001).float().mean().detach(),
            "facb_agf_gt_sem_score_q50": q50.detach(),
            "facb_agf_semantic_acc": (
                semantic_correct / max(semantic_total, 1)
                if semantic_total > 0
                else zero
            ).detach(),
        }

    @staticmethod
    def _facb_text_orth_signature(name):
        import re

        text = str(name or "").lower()
        text = re.sub(r"\([^)]*\)", "", text)
        text = text.replace("km / h", "km/h").replace("kmh", "km/h")
        text = re.sub(r"\s+", " ", text).strip()
        if not re.search(r"\d", text):
            return None
        text = re.sub(r"\d+(?:\.\d+)?", "<num>", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else None

    def _facb_text_orth_build_pairs(self):
        labels = list(getattr(self, "facb_text_orth_label_list", None) or [])
        allowed_ids = getattr(self, "facb_text_orth_allowed_global_ids", None)
        if allowed_ids is not None:
            allowed_ids = {int(x) for x in allowed_ids}
        allowed_values = getattr(self, "facb_text_orth_allowed_values", None)
        if allowed_values:
            allowed_values = {str(x).strip() for x in allowed_values}
        allowed_signatures = getattr(self, "facb_text_orth_allowed_signatures", None)
        if allowed_signatures:
            allowed_signatures = {str(x).strip().lower() for x in allowed_signatures}
        groups = {}
        for idx, name in enumerate(labels):
            if allowed_ids is not None and idx not in allowed_ids:
                continue
            sig = self._facb_text_orth_signature(name)
            if sig is None:
                continue
            if allowed_signatures and sig.lower() not in allowed_signatures:
                continue
            if allowed_values:
                value_match = re.search(r"\d+(?:\.\d+)?", str(name or ""))
                if value_match is None or value_match.group(0).strip() not in allowed_values:
                    continue
            groups.setdefault(sig, []).append(idx)

        pairs = []
        pair_names = []
        max_pairs_per_group = int(getattr(self, "facb_text_orth_max_pairs_per_group", 64))
        for sig, ids in sorted(groups.items()):
            if len(ids) < 2:
                continue
            count = 0
            for pos_i in range(len(ids)):
                for pos_j in range(pos_i + 1, len(ids)):
                    pairs.append((ids[pos_i], ids[pos_j]))
                    pair_names.append((sig, labels[ids[pos_i]], labels[ids[pos_j]]))
                    count += 1
                    if count >= max_pairs_per_group:
                        break
                if count >= max_pairs_per_group:
                    break
        self.facb_text_orth_pairs = pairs
        self.facb_text_orth_pair_names = pair_names
        return pairs

    def loss_facb_text_orth(self, outputs, targets, indices, num_boxes):
        """Separate numeric fine-grained text prototypes in detector text space."""
        pred_logits = outputs.get("pred_logits", None)
        if pred_logits is None:
            return {"loss_facb_text_orth": torch.tensor(0.0)}
        zero = pred_logits.sum() * 0.0
        if not bool(getattr(self, "use_facb_text_orth_loss", False)):
            return {"loss_facb_text_orth": zero}
        model_ref = getattr(self, "facb_text_orth_model_ref", None)
        if model_ref is None or not hasattr(model_ref, "_encode_captions"):
            return {"loss_facb_text_orth": zero}
        labels = list(getattr(self, "facb_text_orth_label_list", None) or [])
        if not labels:
            return {"loss_facb_text_orth": zero}
        pairs = self.facb_text_orth_pairs
        if pairs is None:
            pairs = self._facb_text_orth_build_pairs()
        if not pairs:
            return {"loss_facb_text_orth": zero}

        device = pred_logits.device
        text_dict, _, _ = model_ref._encode_captions([str(x) for x in labels], device)
        proto, valid = model_ref._pool_content_tokens(
            text_dict.get("encoded_text", None),
            text_dict.get("text_token_mask", None),
        )
        if proto is None or valid is None:
            return {"loss_facb_text_orth": zero}
        proto = F.normalize(torch.nan_to_num(proto.float()), dim=-1)
        valid = valid.to(device=proto.device, dtype=torch.bool)

        pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=proto.device)
        pair_valid = valid[pair_tensor[:, 0]] & valid[pair_tensor[:, 1]]
        if not bool(pair_valid.any().item()):
            return {"loss_facb_text_orth": zero}
        pair_tensor = pair_tensor[pair_valid]
        sim = (proto[pair_tensor[:, 0]] * proto[pair_tensor[:, 1]]).sum(dim=-1)
        margin = float(getattr(self, "facb_text_orth_margin", 0.85))
        orth_loss = F.relu(sim - margin).pow(2).mean()

        anchor_weight = float(getattr(self, "facb_text_orth_anchor_weight", 0.05))
        anchor_loss = zero
        selected_ids = torch.unique(pair_tensor.reshape(-1))
        if anchor_weight > 0.0:
            anchor = self.facb_text_orth_anchor_proto
            if anchor is None or anchor.shape != proto.shape or anchor.device != proto.device:
                self.facb_text_orth_anchor_proto = proto.detach().clone()
                anchor = self.facb_text_orth_anchor_proto
            anchor_loss = F.mse_loss(proto[selected_ids], anchor[selected_ids].detach())

        total = orth_loss + anchor_weight * anchor_loss
        if not torch.isfinite(total):
            total = zero
        with torch.no_grad():
            sim_sorted = sim.detach().sort().values
            if sim_sorted.numel() > 0:
                q50 = sim_sorted[sim_sorted.numel() // 2]
                q90 = sim_sorted[min(sim_sorted.numel() - 1, int(0.9 * (sim_sorted.numel() - 1)))]
            else:
                q50 = zero.detach()
                q90 = zero.detach()

        return {
            "loss_facb_text_orth": total,
            "facb_text_orth_sim_mean": sim.detach().mean(),
            "facb_text_orth_sim_q50": q50.detach(),
            "facb_text_orth_sim_q90": q90.detach(),
            "facb_text_orth_num_pairs": pred_logits.new_tensor(float(pair_tensor.shape[0])).detach(),
            "facb_text_orth_anchor_loss": anchor_loss.detach(),
        }

    def _facb_speedhr_make_rois(self, batch_indices, boxes_cxcywh, targets, scale):
        rois = []
        for row_idx in range(batch_indices.numel()):
            b = int(batch_indices[row_idx].item())
            box = boxes_cxcywh[row_idx].float().clamp(0.0, 1.0)
            cx, cy, bw, bh = box.unbind(dim=0)
            bw = (bw * float(scale)).clamp(min=1e-4, max=1.0)
            bh = (bh * float(scale)).clamp(min=1e-4, max=1.0)
            xyxy = torch.stack(
                [
                    (cx - 0.5 * bw).clamp(0.0, 1.0),
                    (cy - 0.5 * bh).clamp(0.0, 1.0),
                    (cx + 0.5 * bw).clamp(0.0, 1.0),
                    (cy + 0.5 * bh).clamp(0.0, 1.0),
                ],
                dim=0,
            )
            size = targets[b].get("size", targets[b].get("orig_size", None))
            if size is None:
                continue
            size_t = torch.as_tensor(size, device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)
            h = size_t[0].clamp(min=1.0)
            w = size_t[1].clamp(min=1.0)
            abs_xyxy = xyxy * torch.stack([w, h, w, h], dim=0)
            abs_xyxy[2] = torch.maximum(abs_xyxy[2], abs_xyxy[0] + 1.0)
            abs_xyxy[3] = torch.maximum(abs_xyxy[3], abs_xyxy[1] + 1.0)
            rois.append(
                torch.cat(
                    [
                        boxes_cxcywh.new_tensor([float(b)]),
                        abs_xyxy.to(dtype=boxes_cxcywh.dtype),
                    ],
                    dim=0,
                )
            )
        if not rois:
            return None
        return torch.stack(rois, dim=0)

    def _facb_speedhr_make_raw_crop_batch(
        self,
        batch_indices,
        boxes_cxcywh,
        targets,
        scale,
        crop_size,
        device,
        dtype,
    ):
        crops = []
        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        for row_idx in range(batch_indices.numel()):
            b = int(batch_indices[row_idx].item())
            if b < 0 or b >= len(targets):
                return None, None
            image_path = targets[b].get("image_path", None)
            if not image_path:
                return None, None
            box = boxes_cxcywh[row_idx].float().clamp(0.0, 1.0)
            cx, cy, bw, bh = box.unbind(dim=0)
            bw = (bw * float(scale)).clamp(min=1e-4, max=1.0)
            bh = (bh * float(scale)).clamp(min=1e-4, max=1.0)
            x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
            y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
            x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
            y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
            try:
                with Image.open(str(image_path)) as img_raw:
                    img = img_raw.convert("RGB")
                    img_w, img_h = img.size
                    abs_box = (
                        float(x1.item()) * float(img_w),
                        float(y1.item()) * float(img_h),
                        max(float(x2.item()) * float(img_w), float(x1.item()) * float(img_w) + 1.0),
                        max(float(y2.item()) * float(img_h), float(y1.item()) * float(img_h) + 1.0),
                    )
                    crop = img.crop(abs_box).resize(
                        (int(crop_size), int(crop_size)),
                        resampling,
                    )
                    arr = np.asarray(crop, dtype=np.float32) / 255.0
            except Exception:
                return None, None
            crop_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            mean = crop_t.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = crop_t.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            crops.append((crop_t - mean) / std)
        if not crops:
            return None, None
        crop_batch = torch.stack(crops, dim=0).to(device=device, dtype=dtype)
        crop_size_f = float(crop_size)
        rois = crop_batch.new_zeros((crop_batch.shape[0], 5), dtype=torch.float32)
        rois[:, 0] = torch.arange(
            crop_batch.shape[0], device=device, dtype=torch.float32
        )
        rois[:, 3] = crop_size_f
        rois[:, 4] = crop_size_f
        return crop_batch, rois

    def _loss_facb_smallshot_candidate(self, outputs, targets, indices, num_boxes):
        """Train a shared crop-to-top-K matcher on few-shot matched objects."""
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs.get("pred_boxes", None)
        zero = pred_logits.new_zeros((), dtype=torch.float32)
        diag_counts = {
            "input_gt_rows": 0,
            "input_smallshot_gt_rows": 0,
            "skip_missing_global": 0,
            "skip_invalid_local": 0,
            "skip_target_local_bounds": 0,
            "skip_global_not_in_caption": 0,
            "skip_text_invalid": 0,
            "local_global_mismatch": 0,
            "skip_out_of_scope": 0,
            "skip_few_candidates": 0,
            "selected_rows": 0,
            "selected_smallshot_rows": 0,
        }

        def _empty_result():
            result = {
                "loss_facb_speedhr": zero,
                "facb_speedhr_crop_rows": zero.detach(),
                "facb_speedhr_candidate_rows": zero.detach(),
                "facb_speedhr_candidate_smallshot_rows": zero.detach(),
                "facb_speedhr_candidate_anchor_rows": zero.detach(),
                "facb_speedhr_candidate_hard_rows": zero.detach(),
                "facb_speedhr_candidate_smallshot_hard_rows": zero.detach(),
                "facb_speedhr_candidate_smallshot_correctable_hard_rows": zero.detach(),
            }
            for name, value in diag_counts.items():
                result[f"facb_speedhr_diag_{name}"] = pred_logits.new_tensor(
                    float(value)
                ).detach()
            return result

        head = getattr(self, "facb_speedhr_head_ref", None)
        images = outputs.get("facb_speedhr_samples", None)
        label_map_list = outputs.get("_criterion_label_map_list", None)
        text_prototypes = outputs.get("facb_speedhr_text_prototypes", None)
        text_valid = outputs.get("facb_speedhr_text_valid", None)
        facet_prototypes = outputs.get("facb_speedhr_facet_prototypes", None)
        facet_valid_global = outputs.get("facb_speedhr_facet_valid", None)
        shared_facet_prototypes = outputs.get(
            "facb_speedhr_shared_facet_prototypes", None
        )
        shared_facet_valid = outputs.get(
            "facb_speedhr_shared_facet_valid", None
        )
        shared_class_targets = getattr(
            self, "facb_speedhr_shared_facet_class_targets", None
        )
        shared_class_target_mask = getattr(
            self, "facb_speedhr_shared_facet_class_target_mask", None
        )
        global_to_target = getattr(self, "facb_speedhr_global_to_digit", None)
        if (
            head is None
            or images is None
            or label_map_list is None
            or text_prototypes is None
            or global_to_target is None
            or pred_boxes is None
        ):
            return _empty_result()

        device = pred_logits.device
        images = images.to(device=device)
        global_to_target = global_to_target.to(device=device, dtype=torch.long)
        topk = max(int(getattr(self, "facb_speedhr_candidate_topk", 10)), 2)
        pred_logits_safe = torch.nan_to_num(
            pred_logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)
        crop_boxes = []
        crop_batches = []
        candidate_text_rows = []
        candidate_valid_rows = []
        candidate_shared_target_rows = []
        candidate_shared_target_mask_rows = []
        candidate_base_score_rows = []
        target_slots = []
        row_weights = []
        smallshot_flags = []
        hard_flags = []
        natural_hit_flags = []
        natural_hits = 0
        semantic_correct = 0
        hard_rows = 0
        base_gt_scores = []
        selected_ious = []
        allowed_global_ids = getattr(self, "facb_speedhr_allowed_global_ids", None)
        if allowed_global_ids is not None:
            allowed_global_ids = allowed_global_ids.to(
                device=device, dtype=torch.long
            ).view(-1)
        smallshot_weight = float(
            getattr(self, "facb_speedhr_candidate_smallshot_weight", 5.0)
        )
        anchor_weight = float(
            getattr(self, "facb_speedhr_candidate_anchor_weight", 0.05)
        )
        anchor_keep_prob = float(
            getattr(self, "facb_speedhr_candidate_anchor_keep_prob", 1.0)
        )
        smallshot_hard_weight = float(
            getattr(self, "facb_speedhr_candidate_smallshot_hard_weight", 10.0)
        )
        anchor_hard_weight = float(
            getattr(self, "facb_speedhr_candidate_anchor_hard_weight", 0.10)
        )
        use_multi_facet = bool(
            facet_prototypes is not None
            and facet_valid_global is not None
            and facet_prototypes.dim() == 3
        )
        if use_multi_facet:
            facet_prototypes = facet_prototypes.to(
                device=device, dtype=torch.float32
            )
            facet_valid_global = facet_valid_global.to(
                device=device, dtype=torch.bool
            )
        use_shared_facets = bool(
            shared_facet_prototypes is not None
            and shared_facet_valid is not None
            and shared_class_targets is not None
        )
        if use_shared_facets:
            shared_facet_prototypes = shared_facet_prototypes.to(
                device=device, dtype=torch.float32
            )
            shared_facet_valid = shared_facet_valid.to(
                device=device, dtype=torch.bool
            )
            shared_class_targets = shared_class_targets.to(
                device=device, dtype=torch.long
            )
            shared_class_target_mask = shared_class_target_mask.to(
                device=device, dtype=torch.bool
            )

        query_source = str(
            getattr(self, "facb_speedhr_candidate_query_source", "matcher")
            or "matcher"
        ).strip().lower()
        min_candidate_iou = float(
            getattr(self, "facb_speedhr_candidate_min_iou", 0.0)
        )
        crop_source = str(
            getattr(self, "facb_speedhr_candidate_crop_source", "pred") or "pred"
        ).strip().lower()
        crop_pixel_source = str(
            getattr(self, "facb_speedhr_crop_pixel_source", "tensor") or "tensor"
        ).strip().lower()

        for batch_idx, (matched_src_idx, matched_tgt_idx) in enumerate(indices):
            if batch_idx >= len(label_map_list):
                continue
            label_map = label_map_list[batch_idx]
            prototypes = text_prototypes[batch_idx]
            if label_map is None or label_map.numel() == 0 or prototypes.numel() == 0:
                continue
            label_map = label_map.to(device=device, dtype=torch.float32)
            prototypes = prototypes.to(device=device, dtype=torch.float32)
            num_local = min(int(label_map.shape[0]), int(prototypes.shape[0]))
            local_to_global = self._local_to_global_label_ids(
                targets[batch_idx], num_local, device
            )
            valid_local = (
                (local_to_global >= 0)
                & (local_to_global < global_to_target.numel())
            )
            if text_valid is not None and batch_idx < len(text_valid):
                valid_local = valid_local & text_valid[batch_idx][:num_local].to(
                    device=device, dtype=torch.bool
                )
            weights = label_map[:num_local]
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sem_scores = torch.nan_to_num(
                pred_logits_safe[batch_idx].detach().sigmoid().float() @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            gt_labels = targets[batch_idx]["labels"].to(device=device, dtype=torch.long)
            object_global_labels = targets[batch_idx].get("global_labels", None)
            if object_global_labels is not None:
                object_global_labels = object_global_labels.to(
                    device=device, dtype=torch.long
                ).view(-1)
            if query_source in {
                "best_iou",
                "geometry",
                "geometric",
                "hard_iou",
                "hard_geometry",
            }:
                gt_boxes = targets[batch_idx].get("boxes", None)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue
                pred_xyxy = box_ops.box_cxcywh_to_xyxy(
                    pred_boxes[batch_idx].detach().float()
                )
                gt_xyxy = box_ops.box_cxcywh_to_xyxy(
                    gt_boxes.to(device=device, dtype=torch.float32)
                )
                pairwise_iou = box_ops.box_iou(pred_xyxy, gt_xyxy)[0]
                best_iou, best_src_idx = pairwise_iou.max(dim=0)
                tgt_idx = torch.arange(
                    gt_boxes.shape[0], device=device, dtype=torch.long
                )
                if query_source in {"hard_iou", "hard_geometry"}:
                    query_sem_scores = sem_scores[:, :num_local].clone()
                    query_sem_scores[:, ~valid_local] = float("-inf")
                    query_top1 = query_sem_scores.argmax(dim=-1)
                    target_local = gt_labels[: gt_boxes.shape[0]]
                    wrong_query = query_top1[:, None] != target_local[None, :]
                    query_topk = torch.topk(
                        query_sem_scores,
                        k=min(topk, int(valid_local.sum().item())),
                        dim=-1,
                    ).indices
                    gt_in_query_topk = (
                        query_topk[:, :, None] == target_local[None, None, :]
                    ).any(dim=1)
                    localized_wrong = (
                        wrong_query
                        & gt_in_query_topk
                        & (pairwise_iou >= min_candidate_iou)
                    )
                    hard_iou = pairwise_iou.masked_fill(~localized_wrong, -1.0)
                    hard_best_iou, hard_src_idx = hard_iou.max(dim=0)
                    has_hard_query = hard_best_iou >= 0.0
                    src_idx = torch.where(
                        has_hard_query, hard_src_idx, best_src_idx
                    )
                    row_ious = torch.where(
                        has_hard_query, hard_best_iou, best_iou
                    )
                else:
                    src_idx = best_src_idx
                    row_ious = best_iou
                # Keep every GT. min_candidate_iou only defines a usable hard
                # query; when none exists we fall back to the best-localized one.
            else:
                src_idx = matched_src_idx.to(device=device, dtype=torch.long)
                tgt_idx = matched_tgt_idx.to(device=device, dtype=torch.long)
                row_ious = pred_logits.new_full(
                    (src_idx.numel(),), float("nan"), dtype=torch.float32
                )
            if src_idx.numel() == 0:
                continue
            for row_pos in range(src_idx.numel()):
                query_idx = int(src_idx[row_pos].item())
                target_idx = int(tgt_idx[row_pos].item())
                diag_counts["input_gt_rows"] += 1
                if target_idx < 0 or target_idx >= gt_labels.numel():
                    diag_counts["skip_invalid_local"] += 1
                    diag_counts["skip_target_local_bounds"] += 1
                    continue

                target_gt_local = int(gt_labels[target_idx].item())
                if (
                    object_global_labels is not None
                    and target_idx < object_global_labels.numel()
                ):
                    gt_global = int(object_global_labels[target_idx].item())
                elif 0 <= target_gt_local < num_local:
                    gt_global = int(local_to_global[target_gt_local].item())
                else:
                    diag_counts["skip_missing_global"] += 1
                    continue

                is_smallshot = bool(
                    allowed_global_ids is None
                    or allowed_global_ids.numel() == 0
                    or (allowed_global_ids == gt_global).any().item()
                )
                diag_counts["input_smallshot_gt_rows"] += int(is_smallshot)

                if gt_global < 0 or gt_global >= global_to_target.numel():
                    diag_counts["skip_missing_global"] += 1
                    continue
                matching_local = torch.nonzero(
                    local_to_global[:num_local] == gt_global, as_tuple=False
                ).flatten()
                if matching_local.numel() == 0:
                    diag_counts["skip_invalid_local"] += 1
                    diag_counts["skip_global_not_in_caption"] += 1
                    continue
                gt_local = int(matching_local[0].item())
                if target_gt_local != gt_local:
                    diag_counts["local_global_mismatch"] += 1
                if not bool(valid_local[gt_local].item()):
                    diag_counts["skip_invalid_local"] += 1
                    diag_counts["skip_text_invalid"] += 1
                    continue
                if int(global_to_target[gt_global].item()) < 0:
                    diag_counts["skip_out_of_scope"] += 1
                    continue
                if not is_smallshot and anchor_weight <= 0.0:
                    diag_counts["skip_out_of_scope"] += 1
                    continue
                if not is_smallshot and anchor_keep_prob < 1.0:
                    if bool(
                        torch.rand((), device=device).item()
                        >= max(anchor_keep_prob, 0.0)
                    ):
                        diag_counts["skip_out_of_scope"] += 1
                        continue
                sem_row = sem_scores[query_idx, :num_local].clone()
                sem_row[~valid_local] = float("-inf")
                valid_count = int(torch.isfinite(sem_row).sum().item())
                if valid_count < 2:
                    diag_counts["skip_few_candidates"] += 1
                    continue
                k = min(topk, valid_count)
                candidate_local = torch.topk(sem_row, k=k, dim=0).indices
                natural_hit = bool((candidate_local == gt_local).any().item())
                natural_hits += int(natural_hit)
                if not natural_hit:
                    candidate_local[-1] = gt_local
                target_slot = int(
                    torch.nonzero(candidate_local == gt_local, as_tuple=False)[0, 0].item()
                )
                semantic_top1 = int(candidate_local[0].item())
                is_hard = semantic_top1 != gt_local
                semantic_correct += int(not is_hard)
                hard_rows += int(is_hard)
                base_gt_scores.append(sem_row[gt_local].detach())
                candidate_base_scores = sem_row[candidate_local].detach().clone()
                if torch.isfinite(row_ious[row_pos]):
                    selected_ious.append(row_ious[row_pos].detach())
                if crop_source in {"gt", "target", "ground_truth"}:
                    crop_boxes.append(
                        targets[batch_idx]["boxes"][target_idx]
                        .to(device=device, dtype=pred_boxes.dtype)
                        .detach()
                    )
                else:
                    # Predicted-box crops match inference; GT crops are useful
                    # for the low-shot reader warm-up when detections are sparse.
                    crop_boxes.append(pred_boxes[batch_idx, query_idx].detach())
                crop_batches.append(
                    torch.tensor(batch_idx, device=device, dtype=torch.long)
                )
                candidate_global = local_to_global[candidate_local]
                if use_shared_facets:
                    candidate_text = prototypes[candidate_local].detach()
                    candidate_valid = torch.ones(
                        k, device=device, dtype=torch.bool
                    )
                    candidate_shared_targets = shared_class_targets[
                        candidate_global
                    ].detach().clone()
                    candidate_shared_target_mask = shared_class_target_mask[
                        candidate_global
                    ].detach().clone()
                elif use_multi_facet:
                    candidate_text = facet_prototypes[candidate_global].detach().clone()
                    candidate_valid = facet_valid_global[candidate_global].detach().clone()
                    # Keep the detector's exact class-name prototype as facet 0.
                    candidate_text[:, 0] = prototypes[candidate_local].detach()
                    candidate_valid[:, 0] = True
                else:
                    candidate_text = prototypes[candidate_local].detach()
                    candidate_valid = torch.ones(
                        k, device=device, dtype=torch.bool
                    )
                if k < topk:
                    candidate_text = torch.cat(
                        [
                            candidate_text,
                            candidate_text.new_zeros(
                                (topk - k,) + tuple(candidate_text.shape[1:])
                            ),
                        ],
                        dim=0,
                    )
                    candidate_valid = torch.cat(
                        [
                            candidate_valid,
                            torch.zeros(
                                (topk - k,) + tuple(candidate_valid.shape[1:]),
                                device=device,
                                dtype=torch.bool,
                            ),
                        ],
                        dim=0,
                    )
                    if use_shared_facets:
                        shared_pad_shape = (
                            (topk - k,)
                            + tuple(candidate_shared_targets.shape[1:])
                        )
                        candidate_shared_targets = torch.cat(
                            [
                                candidate_shared_targets,
                                torch.full(
                                    shared_pad_shape,
                                    -1,
                                    device=device,
                                    dtype=torch.long,
                                ),
                            ],
                            dim=0,
                        )
                        candidate_shared_target_mask = torch.cat(
                            [
                                candidate_shared_target_mask,
                                torch.zeros(
                                    shared_pad_shape,
                                    device=device,
                                    dtype=torch.bool,
                                ),
                            ],
                            dim=0,
                        )
                    candidate_base_scores = torch.cat(
                        [
                            candidate_base_scores,
                            candidate_base_scores.new_zeros(topk - k),
                        ],
                        dim=0,
                    )
                candidate_text_rows.append(candidate_text)
                candidate_valid_rows.append(candidate_valid)
                candidate_base_score_rows.append(candidate_base_scores)
                if use_shared_facets:
                    candidate_shared_target_rows.append(candidate_shared_targets)
                    candidate_shared_target_mask_rows.append(
                        candidate_shared_target_mask
                    )
                target_slots.append(
                    torch.tensor(target_slot, device=device, dtype=torch.long)
                )
                if is_smallshot:
                    row_weight = (
                        smallshot_hard_weight if is_hard else smallshot_weight
                    )
                else:
                    row_weight = anchor_hard_weight if is_hard else anchor_weight
                row_weights.append(
                    pred_logits.new_tensor(row_weight, dtype=torch.float32)
                )
                smallshot_flags.append(is_smallshot)
                hard_flags.append(is_hard)
                natural_hit_flags.append(natural_hit)
                diag_counts["selected_rows"] += 1
                diag_counts["selected_smallshot_rows"] += int(is_smallshot)

        if not crop_boxes:
            return _empty_result()
        crop_boxes_t = torch.stack(crop_boxes)
        crop_batches_t = torch.stack(crop_batches)
        candidate_text_t = torch.stack(candidate_text_rows)
        candidate_valid_t = torch.stack(candidate_valid_rows)
        candidate_base_scores_t = torch.stack(candidate_base_score_rows).float()
        candidate_shared_targets_t = (
            torch.stack(candidate_shared_target_rows)
            if use_shared_facets
            else None
        )
        candidate_shared_target_mask_t = (
            torch.stack(candidate_shared_target_mask_rows)
            if use_shared_facets
            else None
        )
        target_slots_t = torch.stack(target_slots)
        row_weights_t = torch.stack(row_weights).float()
        if crop_pixel_source in {"raw", "raw_image", "original", "pil"}:
            raw_images, rois = self._facb_speedhr_make_raw_crop_batch(
                crop_batches_t,
                crop_boxes_t,
                targets,
                float(getattr(self, "facb_speedhr_crop_scale", 1.2)),
                int(getattr(head, "crop_size", getattr(self, "facb_speedhr_crop_size", 224))),
                device,
                images.dtype,
            )
            if raw_images is None or rois is None:
                return _empty_result()
            head_images = raw_images
        else:
            rois = self._facb_speedhr_make_rois(
                crop_batches_t,
                crop_boxes_t,
                targets,
                float(getattr(self, "facb_speedhr_crop_scale", 1.2)),
            )
            if rois is None:
                return _empty_result()
            head_images = images
        candidate_output = head(
            head_images,
            rois.to(device=device),
            candidate_text=candidate_text_t,
            candidate_mask=candidate_valid_t,
            shared_facet_prototypes=shared_facet_prototypes,
            shared_facet_valid=shared_facet_valid,
            candidate_shared_targets=candidate_shared_targets_t,
            candidate_shared_target_mask=candidate_shared_target_mask_t,
        )
        if candidate_output is None:
            return _empty_result()
        facet_logits = None
        facet_weights = None
        shared_facet_logits = None
        shared_facet_fusion = None
        if isinstance(candidate_output, dict):
            candidate_logits = candidate_output["candidate_logits"]
            shared_facet_logits = candidate_output.get(
                "shared_facet_logits", None
            )
            shared_facet_fusion = candidate_output.get(
                "shared_facet_fusion", None
            )
        elif isinstance(candidate_output, tuple):
            candidate_logits, facet_logits, facet_weights = candidate_output
        else:
            candidate_logits = candidate_output
        candidate_logits = torch.nan_to_num(
            candidate_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0
        )
        candidate_ce = F.cross_entropy(
            candidate_logits, target_slots_t, reduction="none"
        )
        candidate_loss = (candidate_ce * row_weights_t).sum() / row_weights_t.sum().clamp(min=1e-6)
        shared_facet_loss = zero
        shared_facet_losses = []
        shared_facet_accs = []
        if shared_facet_logits is not None and candidate_shared_targets_t is not None:
            row_ids = torch.arange(
                target_slots_t.numel(), device=device, dtype=torch.long
            )
            gt_shared_targets = candidate_shared_targets_t[
                row_ids, target_slots_t
            ]
            gt_shared_mask = candidate_shared_target_mask_t[
                row_ids, target_slots_t
            ]
            configured_weights = list(
                getattr(
                    self,
                    "facb_speedhr_shared_facet_type_weights",
                    [0.5, 0.5, 1.0, 1.0],
                )
            )
            use_symbol_bce = bool(
                getattr(self, "facb_speedhr_shared_symbol_bce", False)
            )
            shared_facet_names = list(
                getattr(self, "facb_speedhr_shared_facet_names", []) or []
            )
            shared_facet_scope = str(
                getattr(
                    self,
                    "facb_speedhr_shared_facet_supervision_scope",
                    "all",
                )
                or "all"
            ).strip().lower()
            smallshot_row_mask = torch.as_tensor(
                smallshot_flags, device=device, dtype=torch.bool
            )
            weighted_losses = []
            weight_sum = 0.0
            for facet_idx in range(int(shared_facet_logits.shape[1])):
                logits_f = shared_facet_logits[:, facet_idx].float()
                targets_f = gt_shared_targets[:, facet_idx]
                mask_f = gt_shared_mask[:, facet_idx] & (targets_f >= 0)
                valid_rows = mask_f.any(dim=-1)
                if shared_facet_scope in {"smallshot", "smallshot_only", "rare", "rare_only"}:
                    valid_rows = valid_rows & smallshot_row_mask
                elif shared_facet_scope in {"head", "head_only", "non_smallshot"}:
                    valid_rows = valid_rows & (~smallshot_row_mask)
                elif shared_facet_scope in {"none", "off", "disabled"}:
                    valid_rows = valid_rows & False
                if not bool(valid_rows.any().item()):
                    shared_facet_losses.append(zero.detach())
                    shared_facet_accs.append(zero.detach())
                    continue
                safe_targets = targets_f[valid_rows].clamp(min=0)
                valid_targets = mask_f[valid_rows]
                is_symbol_facet = (
                    use_symbol_bce
                    and facet_idx < len(shared_facet_names)
                    and str(shared_facet_names[facet_idx]).strip().lower() == "symbol"
                )
                if is_symbol_facet:
                    logits_valid = logits_f[valid_rows]
                    target_multi = logits_valid.new_zeros(logits_valid.shape)
                    target_multi.scatter_(1, safe_targets, valid_targets.float())
                    loss_all = F.binary_cross_entropy_with_logits(
                        logits_valid,
                        target_multi,
                        reduction="none",
                    )
                    focal_gamma = float(
                        getattr(
                            self,
                            "facb_speedhr_shared_symbol_focal_gamma",
                            0.0,
                        )
                    )
                    if focal_gamma > 0.0:
                        prob_valid = torch.sigmoid(logits_valid)
                        pt = torch.where(
                            target_multi > 0.5,
                            prob_valid,
                            1.0 - prob_valid,
                        )
                        loss_all = loss_all * (1.0 - pt).clamp(min=1e-6).pow(
                            focal_gamma
                        )
                    pos_count = valid_targets.sum(dim=-1).clamp(min=1).float()
                    neg_count = (
                        target_multi.shape[1] - pos_count
                    ).clamp(min=1.0)
                    pos_loss = (loss_all * target_multi).sum(dim=-1) / pos_count
                    neg_loss = (
                        loss_all * (1.0 - target_multi)
                    ).sum(dim=-1) / neg_count
                    neg_weight = float(
                        getattr(
                            self,
                            "facb_speedhr_shared_symbol_bce_neg_weight",
                            0.05,
                        )
                    )
                    per_row = pos_loss + neg_weight * neg_loss
                    hard_neg_weight = float(
                        getattr(
                            self,
                            "facb_speedhr_shared_symbol_hard_neg_weight",
                            0.0,
                        )
                    )
                    if hard_neg_weight > 0.0 and bool(
                        getattr(
                            self,
                            "facb_speedhr_shared_symbol_hard_neg_from_candidates",
                            False,
                        )
                    ):
                        row_candidate_targets = candidate_shared_targets_t[
                            valid_rows, :, facet_idx
                        ].clamp(min=0)
                        row_candidate_mask = candidate_shared_target_mask_t[
                            valid_rows, :, facet_idx
                        ]
                        hard_neg_multi = logits_valid.new_zeros(
                            logits_valid.shape
                        )
                        hard_neg_multi.scatter_(
                            1,
                            row_candidate_targets.reshape(
                                row_candidate_targets.shape[0], -1
                            ),
                            row_candidate_mask.reshape(
                                row_candidate_mask.shape[0], -1
                            ).float(),
                        )
                        hard_neg_multi = hard_neg_multi * (1.0 - target_multi)
                        hard_neg_count = hard_neg_multi.sum(dim=-1).clamp(min=1.0)
                        hard_neg_loss = (
                            loss_all * hard_neg_multi
                        ).sum(dim=-1) / hard_neg_count
                        has_hard_neg = hard_neg_multi.sum(dim=-1) > 0
                        per_row = per_row + hard_neg_weight * hard_neg_loss * (
                            has_hard_neg.float()
                        )
                else:
                    log_prob = F.log_softmax(logits_f[valid_rows], dim=-1)
                    gathered = log_prob.gather(1, safe_targets)
                    per_row = -(
                        gathered * valid_targets.float()
                    ).sum(dim=-1) / valid_targets.sum(dim=-1).clamp(min=1)
                row_weight_f = row_weights_t[valid_rows]
                loss_f = (per_row * row_weight_f).sum() / row_weight_f.sum().clamp(
                    min=1e-6
                )
                pred_f = logits_f[valid_rows].argmax(dim=-1)
                acc_f = (
                    (safe_targets == pred_f[:, None]) & valid_targets
                ).any(dim=-1).float().mean()
                type_weight = float(
                    configured_weights[facet_idx]
                    if facet_idx < len(configured_weights)
                    else 1.0
                )
                weighted_losses.append(type_weight * loss_f)
                weight_sum += type_weight
                shared_facet_losses.append(loss_f.detach())
                shared_facet_accs.append(acc_f.detach())
            if weighted_losses:
                shared_facet_loss = sum(weighted_losses) / max(weight_sum, 1e-6)
        candidate_prob = F.softmax(candidate_logits, dim=-1)
        gt_prob = candidate_prob.gather(1, target_slots_t[:, None]).squeeze(1)
        candidate_acc = (candidate_logits.argmax(dim=-1) == target_slots_t).float().mean()
        smallshot_mask = torch.as_tensor(
            smallshot_flags, device=device, dtype=torch.bool
        )
        hard_mask = torch.as_tensor(hard_flags, device=device, dtype=torch.bool)
        natural_hit_mask = torch.as_tensor(
            natural_hit_flags, device=device, dtype=torch.bool
        )
        smallshot_hard_mask = smallshot_mask & hard_mask
        smallshot_correctable_hard_mask = (
            smallshot_mask & hard_mask & natural_hit_mask
        )
        smallshot_acc = (
            (candidate_logits[smallshot_mask].argmax(dim=-1) == target_slots_t[smallshot_mask])
            .float()
            .mean()
            if bool(smallshot_mask.any().item())
            else zero.detach()
        )
        smallshot_hard_acc = (
            (
                candidate_logits[smallshot_hard_mask].argmax(dim=-1)
                == target_slots_t[smallshot_hard_mask]
            )
            .float()
            .mean()
            if bool(smallshot_hard_mask.any().item())
            else zero.detach()
        )
        sorted_prob = candidate_prob.sort(dim=-1, descending=True).values
        candidate_margin = (sorted_prob[:, 0] - sorted_prob[:, 1]).mean()
        base_gt_row = torch.stack(base_gt_scores).float()
        base_gt = base_gt_row.sort().values
        score_residual_loss = zero
        score_residual_active = candidate_logits.new_zeros((), dtype=torch.float32)
        score_residual_alpha = candidate_logits.new_zeros((), dtype=torch.float32)
        score_residual_base_q50 = zero.detach()
        score_residual_final_q50 = zero.detach()
        score_residual_evidence_q50 = zero.detach()
        score_residual_rank_base_margin_q50 = zero.detach()
        score_residual_rank_final_margin_q50 = zero.detach()
        score_residual_evidence_gap_q50 = zero.detach()
        score_residual_pair_rows = candidate_logits.new_zeros((), dtype=torch.float32)
        score_residual_weight = float(
            getattr(self, "facb_speedhr_score_residual_loss_weight", 0.0)
        )
        if score_residual_weight > 0.0:
            current_epoch = int(getattr(self, "current_epoch", 0))
            start_epoch = int(
                getattr(self, "facb_speedhr_score_residual_start_epoch", 0)
            )
            ramp_epochs = max(
                int(getattr(self, "facb_speedhr_score_residual_ramp_epochs", 1)),
                1,
            )
            if current_epoch < start_epoch:
                score_residual_weight = 0.0
            else:
                progress = min(
                    1.0,
                    float(current_epoch - start_epoch + 1) / float(ramp_epochs),
                )
                score_residual_weight = score_residual_weight * progress
        if score_residual_weight > 0.0 and hasattr(head, "score_residual_alpha"):
            gt_candidate_logits = candidate_logits.gather(
                1, target_slots_t[:, None]
            ).squeeze(1)
            evidence_prob = torch.sigmoid(gt_candidate_logits)
            pred_slots = candidate_logits.detach().argmax(dim=-1)
            facb_correct = pred_slots == target_slots_t
            with torch.no_grad():
                conf_gate = gt_prob.detach() >= float(
                    getattr(self, "facb_speedhr_score_residual_min_conf", 0.80)
                )
                margin_gate = (
                    sorted_prob[:, 0].detach() - sorted_prob[:, 1].detach()
                ) >= float(
                    getattr(self, "facb_speedhr_score_residual_min_margin", 0.10)
                )
                min_sem = float(
                    getattr(
                        self,
                        "facb_speedhr_score_residual_min_semantic_score",
                        0.0,
                    )
                )
                max_sem = float(
                    getattr(
                        self,
                        "facb_speedhr_score_residual_max_semantic_score",
                        0.20,
                    )
                )
                score_gate = torch.ones_like(smallshot_hard_mask, dtype=torch.bool)
                if min_sem >= 0.0:
                    score_gate = score_gate & (base_gt_row.detach() >= min_sem)
                if max_sem >= 0.0:
                    score_gate = score_gate & (base_gt_row.detach() <= max_sem)
                active_mask = (
                    smallshot_hard_mask
                    & natural_hit_mask
                    & facb_correct.detach()
                    & conf_gate
                    & margin_gate
                    & score_gate
                )
            score_residual_active = active_mask.float().sum()
            alpha_raw = head.score_residual_alpha()
            alpha_cap = float(
                getattr(self, "facb_speedhr_score_residual_alpha_cap", 1.0)
            )
            alpha_eff = alpha_raw.clamp(max=alpha_cap) if alpha_cap > 0 else alpha_raw
            score_residual_alpha = alpha_eff
            objective = str(
                getattr(self, "facb_speedhr_score_residual_objective", "floor")
                or "floor"
            ).strip().lower()
            base_score = base_gt_row.detach().clamp(min=1e-6, max=1.0 - 1e-6)
            if objective in {"pairwise", "rank", "ranking"}:
                use_soft_gate = bool(
                    getattr(self, "facb_speedhr_score_residual_soft_gate", False)
                )
                valid_candidate_mask = candidate_valid_t
                if valid_candidate_mask.dim() > 2:
                    valid_candidate_mask = valid_candidate_mask.any(dim=-1)
                valid_candidate_mask = valid_candidate_mask.to(
                    device=device, dtype=torch.bool
                )
                row_ids = torch.arange(
                    target_slots_t.numel(), device=device, dtype=torch.long
                )
                wrong_mask = valid_candidate_mask.clone()
                wrong_mask[row_ids, target_slots_t] = False
                wrong_scores_for_select = candidate_base_scores_t.detach().clone()
                wrong_scores_for_select = wrong_scores_for_select.masked_fill(
                    ~wrong_mask, float("-inf")
                )
                has_wrong = torch.isfinite(wrong_scores_for_select).any(dim=1)
                wrong_slots = wrong_scores_for_select.argmax(dim=1)
                evidence_all = torch.sigmoid(candidate_logits)
                wrong_evidence = evidence_all[
                    torch.arange(target_slots_t.numel(), device=device),
                    wrong_slots,
                ]
                if use_soft_gate:
                    gate_type = str(
                        getattr(
                            self,
                            "facb_speedhr_score_residual_gate_type",
                            "prob",
                        )
                        or "prob"
                    ).strip().lower()
                    if gate_type in {"gap", "evidence_gap", "prob_gap"}:
                        residual_gate = (
                            evidence_prob.detach() - wrong_evidence.detach()
                        ).clamp(min=0.0, max=1.0)
                    else:
                        residual_gate = gt_prob.detach().clamp(min=0.0, max=1.0)
                    pair_mask = (
                        smallshot_hard_mask
                        & natural_hit_mask
                        & score_gate
                        & conf_gate
                        & has_wrong
                        & (residual_gate > 0.0)
                    )
                else:
                    pair_mask = active_mask & has_wrong
                    residual_gate = pair_mask.to(candidate_logits.dtype)
                score_residual_pair_rows = pair_mask.float().sum()
                base_all = candidate_base_scores_t.detach().clamp(
                    min=1e-6, max=1.0 - 1e-6
                )
                final_all = (
                    base_all
                    + (
                        residual_gate.to(evidence_all.dtype)
                        * pair_mask.to(evidence_all.dtype)
                    )[:, None]
                    * alpha_eff
                    * evidence_all
                ).clamp(min=1e-6, max=1.0 - 1e-6)
                gt_final = final_all[row_ids, target_slots_t]
                wrong_final = final_all[row_ids, wrong_slots]
                gt_base = base_all[row_ids, target_slots_t]
                wrong_base = base_all[row_ids, wrong_slots]
                rank_margin = float(
                    getattr(
                        self,
                        "facb_speedhr_score_residual_rank_margin",
                        0.02,
                    )
                )
                rank_temp = max(
                    float(
                        getattr(
                            self,
                            "facb_speedhr_score_residual_rank_temperature",
                            1.0,
                        )
                    ),
                    1e-6,
                )
                per_pair = F.softplus(
                    (rank_margin - (gt_final - wrong_final)) / rank_temp
                )
                if bool(pair_mask.any().item()):
                    pair_weights = row_weights_t[pair_mask]
                    if use_soft_gate:
                        pair_weights = pair_weights * residual_gate[pair_mask].clamp(
                            min=1e-3
                        )
                    score_residual_loss = (
                        per_pair[pair_mask] * pair_weights
                    ).sum() / pair_weights.sum().clamp(min=1e-6)
                    selected_base = gt_base[pair_mask].detach().sort().values
                    selected_final = gt_final[pair_mask].detach().sort().values
                    selected_evidence = evidence_prob[pair_mask].detach().sort().values
                    base_margin = (gt_base - wrong_base)[pair_mask].detach().sort().values
                    final_margin = (
                        gt_final - wrong_final
                    )[pair_mask].detach().sort().values
                    evidence_gap = (
                        evidence_prob - wrong_evidence
                    )[pair_mask].detach().sort().values
                    score_residual_base_q50 = selected_base[
                        selected_base.numel() // 2
                    ]
                    score_residual_final_q50 = selected_final[
                        selected_final.numel() // 2
                    ]
                    score_residual_evidence_q50 = selected_evidence[
                        selected_evidence.numel() // 2
                    ]
                    score_residual_rank_base_margin_q50 = base_margin[
                        base_margin.numel() // 2
                    ]
                    score_residual_rank_final_margin_q50 = final_margin[
                        final_margin.numel() // 2
                    ]
                    score_residual_evidence_gap_q50 = evidence_gap[
                        evidence_gap.numel() // 2
                    ]
            else:
                final_score = (
                    base_score
                    + active_mask.to(evidence_prob.dtype) * alpha_eff * evidence_prob
                ).clamp(min=1e-6, max=1.0 - 1e-6)
                floor = float(
                    getattr(self, "facb_speedhr_score_residual_floor", 0.20)
                )
                floor_t = candidate_logits.new_tensor(floor).clamp(
                    min=1e-6, max=1.0 - 1e-6
                )
                floor_logit = torch.logit(floor_t)
                per_row_score = F.softplus(floor_logit - torch.logit(final_score))
                if bool(active_mask.any().item()):
                    active_weights = row_weights_t[active_mask]
                    score_residual_loss = (
                        per_row_score[active_mask] * active_weights
                    ).sum() / active_weights.sum().clamp(min=1e-6)
                    selected_base = base_score[active_mask].detach().sort().values
                    selected_final = final_score[active_mask].detach().sort().values
                    selected_evidence = evidence_prob[active_mask].detach().sort().values
                    score_residual_base_q50 = selected_base[
                        selected_base.numel() // 2
                    ]
                    score_residual_final_q50 = selected_final[
                        selected_final.numel() // 2
                    ]
                    score_residual_evidence_q50 = selected_evidence[
                        selected_evidence.numel() // 2
                    ]
            alpha_reg = float(
                getattr(self, "facb_speedhr_score_residual_alpha_reg", 0.001)
            )
            if alpha_reg > 0.0:
                score_residual_loss = score_residual_loss + alpha_reg * alpha_eff.pow(2)
        total = (
            float(getattr(self, "facb_speedhr_digit_loss_weight", 1.0))
            * candidate_loss
            + float(
                getattr(
                    self, "facb_speedhr_shared_facet_loss_weight", 1.0
                )
            )
            * shared_facet_loss
            + score_residual_weight * score_residual_loss
        )
        result = {
            "loss_facb_speedhr": total,
            "facb_speedhr_digit_loss": candidate_loss.detach(),
            "facb_speedhr_shared_facet_loss": shared_facet_loss.detach(),
            "facb_speedhr_score_residual_loss": score_residual_loss.detach(),
            "facb_speedhr_score_residual_active_rows": score_residual_active.detach(),
            "facb_speedhr_score_residual_alpha": score_residual_alpha.detach(),
            "facb_speedhr_score_residual_base_q50": score_residual_base_q50.detach(),
            "facb_speedhr_score_residual_final_q50": score_residual_final_q50.detach(),
            "facb_speedhr_score_residual_evidence_q50": score_residual_evidence_q50.detach(),
            "facb_speedhr_score_residual_ranking_pair_rows": score_residual_pair_rows.detach(),
            "facb_speedhr_score_residual_rank_base_margin_q50": (
                score_residual_rank_base_margin_q50.detach()
            ),
            "facb_speedhr_score_residual_rank_final_margin_q50": (
                score_residual_rank_final_margin_q50.detach()
            ),
            "facb_speedhr_score_residual_evidence_gap_q50": (
                score_residual_evidence_gap_q50.detach()
            ),
            "facb_speedhr_crop_rows": pred_logits.new_tensor(float(len(crop_boxes))).detach(),
            "facb_speedhr_candidate_rows": pred_logits.new_tensor(float(len(crop_boxes))).detach(),
            "facb_speedhr_candidate_smallshot_rows": smallshot_mask.float().sum().detach(),
            "facb_speedhr_candidate_anchor_rows": (~smallshot_mask).float().sum().detach(),
            "facb_speedhr_candidate_hard_rows": pred_logits.new_tensor(float(hard_rows)).detach(),
            "facb_speedhr_candidate_smallshot_hard_rows": smallshot_hard_mask.float().sum().detach(),
            "facb_speedhr_candidate_smallshot_correctable_hard_rows": (
                smallshot_correctable_hard_mask.float().sum().detach()
            ),
            "facb_speedhr_candidate_natural_topk_frac": pred_logits.new_tensor(
                float(natural_hits) / max(len(crop_boxes), 1)
            ).detach(),
            "facb_speedhr_digit_acc": candidate_acc.detach(),
            "facb_speedhr_candidate_acc": candidate_acc.detach(),
            "facb_speedhr_candidate_smallshot_acc": smallshot_acc.detach(),
            "facb_speedhr_candidate_smallshot_hard_acc": smallshot_hard_acc.detach(),
            "facb_speedhr_candidate_smallshot_semantic_acc": (
                (~hard_mask[smallshot_mask]).float().mean().detach()
                if bool(smallshot_mask.any().item())
                else zero.detach()
            ),
            "facb_speedhr_candidate_smallshot_natural_topk_frac": (
                natural_hit_mask[smallshot_mask].float().mean().detach()
                if bool(smallshot_mask.any().item())
                else zero.detach()
            ),
            "facb_speedhr_candidate_margin": candidate_margin.detach(),
            "facb_speedhr_gt_prob_mean": gt_prob.mean().detach(),
            "facb_speedhr_gt_prob_q50": gt_prob.sort().values[gt_prob.numel() // 2].detach(),
            "facb_speedhr_gt_sem_score_q50": base_gt[base_gt.numel() // 2].detach(),
            "facb_speedhr_semantic_acc": pred_logits.new_tensor(
                float(semantic_correct) / max(len(crop_boxes), 1)
            ).detach(),
            "facb_speedhr_agf_rows": zero.detach(),
            "facb_speedhr_agf_active_rows": zero.detach(),
        }
        for facet_idx, loss_f in enumerate(shared_facet_losses):
            result[f"facb_speedhr_shared_facet_loss_{facet_idx}"] = loss_f
        for facet_idx, acc_f in enumerate(shared_facet_accs):
            result[f"facb_speedhr_shared_facet_acc_{facet_idx}"] = acc_f
        if shared_facet_fusion is not None:
            result["facb_speedhr_shared_facet_fusion"] = (
                shared_facet_fusion.detach()
            )
        for name, value in diag_counts.items():
            result[f"facb_speedhr_diag_{name}"] = pred_logits.new_tensor(
                float(value)
            ).detach()
        if selected_ious:
            selected_iou_t = torch.stack(selected_ious).float().sort().values
            result["facb_speedhr_candidate_iou_q50"] = selected_iou_t[
                selected_iou_t.numel() // 2
            ].detach()
        if facet_weights is not None:
            mean_facet_weights = facet_weights.detach().mean(dim=(0, 1))
            for facet_idx in range(int(mean_facet_weights.numel())):
                result[f"facb_speedhr_facet_weight_{facet_idx}"] = (
                    mean_facet_weights[facet_idx]
                )
        if facet_logits is not None and candidate_valid_t.dim() == 3:
            facet_target_valid = candidate_valid_t.gather(
                1,
                target_slots_t[:, None, None].expand(
                    -1, 1, candidate_valid_t.shape[-1]
                ),
            ).squeeze(1)
            facet_predictions = facet_logits.argmax(dim=1)
            for facet_idx in range(int(facet_logits.shape[-1])):
                valid_rows = facet_target_valid[:, facet_idx] & smallshot_mask
                result[f"facb_speedhr_smallshot_facet_acc_{facet_idx}"] = (
                    (
                        facet_predictions[valid_rows, facet_idx]
                        == target_slots_t[valid_rows]
                    )
                    .float()
                    .mean()
                    .detach()
                    if bool(valid_rows.any().item())
                    else zero.detach()
                )
        return result

    def loss_facb_speedhr(self, outputs, targets, indices, num_boxes):
        """Speed-limit high-resolution crop reader plus native-score focusing."""
        pred_logits = outputs.get("pred_logits", None)
        if pred_logits is None:
            return {"loss_facb_speedhr": torch.tensor(0.0)}
        zero = pred_logits.new_zeros((), dtype=torch.float32)
        if not bool(getattr(self, "use_facb_speedhr_loss", False)):
            return {"loss_facb_speedhr": zero}
        if bool(getattr(self, "facb_speedhr_candidate_matching", False)):
            return self._loss_facb_smallshot_candidate(
                outputs, targets, indices, num_boxes
            )
        head = getattr(self, "facb_speedhr_head_ref", None)
        images = outputs.get("facb_speedhr_samples", None)
        label_map_list = outputs.get("_criterion_label_map_list", None)
        global_to_digit = getattr(self, "facb_speedhr_global_to_digit", None)
        if head is None or images is None or label_map_list is None or global_to_digit is None:
            return {"loss_facb_speedhr": zero}

        device = pred_logits.device
        images = images.to(device=device)
        global_to_digit = global_to_digit.to(device=device, dtype=torch.long)
        allowed_global_ids = getattr(self, "facb_speedhr_allowed_global_ids", None)
        if allowed_global_ids is not None:
            allowed_global_ids = allowed_global_ids.to(device=device, dtype=torch.long).view(-1)
        allowed_digit_ids = getattr(self, "facb_speedhr_allowed_digit_ids", None)
        if allowed_digit_ids is not None:
            allowed_digit_ids = allowed_digit_ids.to(device=device, dtype=torch.long).view(-1)
        pred_logits_safe = torch.nan_to_num(
            pred_logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)
        one_hot = outputs["one_hot"].to(device=device, dtype=torch.float32)
        text_mask = outputs.get(
            "query_token_ignore_mask",
            outputs.get("loss_text_mask", outputs.get("text_mask", None)),
        )
        if text_mask is not None:
            text_mask = text_mask.to(device=device, dtype=torch.bool)

        crop_boxes = []
        crop_batches = []
        crop_targets = []
        agf_rows = []
        agf_crop_indices = []
        semantic_scores_for_gt = []
        semantic_correct = pred_logits.new_tensor(0.0)
        semantic_total = 0
        topk = max(int(getattr(self, "facb_speedhr_agf_topk", 10)), 2)
        min_score = float(getattr(self, "facb_speedhr_agf_min_score", 0.0))
        max_score = float(getattr(self, "facb_speedhr_agf_max_score", 0.20))
        max_margin = float(getattr(self, "facb_speedhr_agf_max_margin", 0.10))
        require_semantic_topk = bool(
            getattr(self, "facb_speedhr_agf_require_semantic_topk", True)
        )

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0 or batch_idx >= len(label_map_list):
                continue
            label_map = label_map_list[batch_idx]
            if label_map is None or label_map.numel() == 0:
                continue
            label_map = label_map.to(device=device, dtype=torch.float32)
            num_local = int(label_map.shape[0])
            local_to_global = self._local_to_global_label_ids(targets[batch_idx], num_local, device)
            valid_local = (local_to_global >= 0) & (local_to_global < global_to_digit.numel())
            weights = label_map / label_map.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sem_scores = torch.nan_to_num(
                pred_logits_safe[batch_idx].detach().sigmoid().float() @ weights.t(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            gt_labels = targets[batch_idx]["labels"].to(device=device, dtype=torch.long)
            gt_boxes = targets[batch_idx]["boxes"].to(device=device, dtype=pred_logits.dtype)
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            for row_pos in range(src_idx.numel()):
                query_idx = int(src_idx[row_pos].item())
                target_idx = int(tgt_idx[row_pos].item())
                if target_idx < 0 or target_idx >= gt_labels.numel():
                    continue
                gt_local = int(gt_labels[target_idx].item())
                if gt_local < 0 or gt_local >= num_local or not bool(valid_local[gt_local].item()):
                    continue
                gt_global = int(local_to_global[gt_local].item())
                digit_target = int(global_to_digit[gt_global].item()) if gt_global < global_to_digit.numel() else -1
                if digit_target < 0:
                    continue
                crop_index = len(crop_boxes)
                crop_boxes.append(gt_boxes[target_idx].detach())
                crop_batches.append(torch.tensor(batch_idx, device=device, dtype=torch.long))
                crop_targets.append(torch.tensor(digit_target, device=device, dtype=torch.long))

                if allowed_global_ids is not None and allowed_global_ids.numel() > 0:
                    if not bool((allowed_global_ids == gt_global).any().item()):
                        continue
                if allowed_digit_ids is not None and allowed_digit_ids.numel() > 0:
                    if not bool((allowed_digit_ids == digit_target).any().item()):
                        continue
                sem_row = sem_scores[query_idx].clone()
                sem_row[~valid_local] = float("-inf")
                valid_count = int(torch.isfinite(sem_row).sum().item())
                if valid_count < 2:
                    continue
                k = min(topk, valid_count)
                top_local = torch.topk(sem_row, k=k, dim=0).indices
                if require_semantic_topk and not bool((top_local == gt_local).any().item()):
                    continue
                top1_score = float(sem_row[int(top_local[0].item())].item())
                top2_score = float(sem_row[int(top_local[1].item())].item()) if k >= 2 else 0.0
                margin = top1_score - top2_score
                if top1_score < min_score:
                    continue
                if max_score >= 0.0 and top1_score > max_score:
                    continue
                if max_margin >= 0.0 and margin > max_margin:
                    continue
                top1_global = int(local_to_global[int(top_local[0].item())].item())
                semantic_correct = semantic_correct + pred_logits.new_tensor(float(top1_global == gt_global))
                semantic_total += 1
                agf_rows.append((batch_idx, query_idx))
                agf_crop_indices.append(crop_index)
                semantic_scores_for_gt.append(pred_logits.new_tensor(float(sem_row[gt_local].item())))

        if not crop_boxes:
            return {
                "loss_facb_speedhr": zero,
                "facb_speedhr_crop_rows": zero.detach(),
                "facb_speedhr_agf_rows": zero.detach(),
            }

        crop_boxes_t = torch.stack(crop_boxes, dim=0).to(device=device, dtype=pred_logits.dtype)
        crop_batches_t = torch.stack(crop_batches, dim=0).to(device=device, dtype=torch.long)
        crop_targets_t = torch.stack(crop_targets, dim=0).to(device=device, dtype=torch.long)
        rois = self._facb_speedhr_make_rois(
            crop_batches_t,
            crop_boxes_t,
            targets,
            float(getattr(self, "facb_speedhr_crop_scale", 1.2)),
        )
        if rois is None:
            return {"loss_facb_speedhr": zero}
        digit_logits = head(images, rois.to(device=device))
        if digit_logits is None:
            return {"loss_facb_speedhr": zero}
        digit_loss = F.cross_entropy(digit_logits.float(), crop_targets_t)
        digit_prob_train = F.softmax(digit_logits.float(), dim=-1)
        with torch.no_grad():
            gt_prob = digit_prob_train.gather(1, crop_targets_t.view(-1, 1)).squeeze(1)
            digit_acc = (digit_logits.argmax(dim=-1) == crop_targets_t).float().mean()

        agf_loss = zero
        active_weights = []
        if agf_rows and float(getattr(self, "facb_speedhr_agf_loss_weight", 1.0)) > 0.0:
            terms = []
            for row_idx, (batch_idx, query_idx) in enumerate(agf_rows):
                crop_index = agf_crop_indices[row_idx]
                target_digit = crop_targets_t[crop_index]
                p_gt = digit_prob_train[crop_index, target_digit].detach()
                crop_row = digit_prob_train[crop_index].detach()
                crop_pred = int(crop_row.argmax(dim=-1).item())
                crop_other = crop_row.clone()
                crop_other[target_digit] = -1.0
                crop_margin = p_gt - crop_other.max()
                if bool(getattr(self, "facb_speedhr_agf_require_crop_correct", False)):
                    if crop_pred != int(target_digit.item()):
                        continue
                min_crop_prob = float(getattr(self, "facb_speedhr_agf_min_crop_prob", 0.0))
                if min_crop_prob > 0.0 and float(p_gt.item()) < min_crop_prob:
                    continue
                min_crop_margin = float(getattr(self, "facb_speedhr_agf_min_crop_margin", -1.0))
                if min_crop_margin > -1.0 and float(crop_margin.item()) < min_crop_margin:
                    continue
                weight_mode = str(
                    getattr(self, "facb_speedhr_agf_weight_mode", "prob")
                ).strip().lower()
                if weight_mode == "margin":
                    evidence = torch.clamp(crop_margin, min=0.0)
                elif weight_mode in {"prob_margin", "prob*margin"}:
                    evidence = p_gt * torch.clamp(crop_margin, min=0.0)
                else:
                    evidence = p_gt
                extra = float(getattr(self, "facb_speedhr_agf_min_extra", 0.0)) + float(
                    getattr(self, "facb_speedhr_agf_alpha", 2.0)
                ) * evidence
                cap = float(getattr(self, "facb_speedhr_agf_weight_cap", 3.0))
                if cap > 0:
                    extra = min(float(extra.item()), max(cap - 1.0, 0.0))
                else:
                    extra = float(extra.item())
                if extra <= 1e-6:
                    continue
                pos_mask = one_hot[batch_idx, query_idx].bool()
                if text_mask is not None:
                    pos_mask = pos_mask & text_mask[batch_idx]
                if not bool(pos_mask.any().item()):
                    continue
                logits = pred_logits_safe[batch_idx, query_idx][pos_mask]
                target = torch.ones_like(logits)
                ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                p = torch.sigmoid(logits)
                focal = ce * ((1 - p) ** self.focal_gamma)
                if self.focal_alpha >= 0:
                    focal = self.focal_alpha * focal
                terms.append(logits.new_tensor(extra) * focal.mean())
                active_weights.append(logits.new_tensor(1.0 + extra))
            if terms:
                agf_loss = torch.stack(terms).sum() / num_boxes

        total = (
            float(getattr(self, "facb_speedhr_digit_loss_weight", 1.0)) * digit_loss
            + float(getattr(self, "facb_speedhr_agf_loss_weight", 1.0)) * agf_loss
        )
        if not torch.isfinite(total):
            total = zero
        if semantic_scores_for_gt:
            sem_gt = torch.stack(semantic_scores_for_gt).float().sort().values
            sem_q50 = sem_gt[sem_gt.numel() // 2]
        else:
            sem_q50 = zero.detach()
        active_weight_mean = (
            torch.stack(active_weights).float().mean().detach()
            if active_weights
            else zero.detach()
        )
        return {
            "loss_facb_speedhr": total,
            "facb_speedhr_digit_loss": digit_loss.detach(),
            "facb_speedhr_agf_loss": agf_loss.detach(),
            "facb_speedhr_crop_rows": pred_logits.new_tensor(float(crop_boxes_t.shape[0])).detach(),
            "facb_speedhr_agf_rows": pred_logits.new_tensor(float(len(agf_rows))).detach(),
            "facb_speedhr_agf_active_rows": pred_logits.new_tensor(float(len(active_weights))).detach(),
            "facb_speedhr_digit_acc": digit_acc.detach(),
            "facb_speedhr_gt_prob_mean": gt_prob.mean().detach(),
            "facb_speedhr_gt_prob_q50": gt_prob.sort().values[gt_prob.numel() // 2].detach(),
            "facb_speedhr_weight_mean": active_weight_mean,
            "facb_speedhr_gt_sem_score_q50": sem_q50.detach(),
            "facb_speedhr_semantic_acc": (
                semantic_correct / max(semantic_total, 1)
                if semantic_total > 0
                else zero
            ).detach(),
        }


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_semantic_group_margin(self, outputs, targets, indices, num_boxes):
        """Method B: same-group token-margin loss on the detector's OWN class scores.

        Conservative form (point-debuggable).  For each matched query (caption-local
        space) with >=1 same-group confuser, hinge: relu(margin - (gt - hardest_conf)).
        Gradient flows into pred_logits to sharpen query-text alignment on the
        discriminative token ('40' vs '60').

        NOTE on the prior nan: `pred_logits.sum() * 0.0` was the culprit — under AMP
        fp16 the 900x256 sum overflows to inf, and inf*0.0 = nan.  The zero baseline
        is now a plain device/dtype scalar with no such reduction.
        """
        debug = bool(int(os.environ.get("GROUNDINGDINO_GROUP_MARGIN_DEBUG", "0") or "0")) \
            and not getattr(self, "_group_margin_dbg_done", False)
        scores_list = outputs.get("semantic_group_margin_scores", None)
        # Safe zero: NOT pred_logits.sum()*0 (inf*0=nan under AMP).
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        if scores_list is None:
            return {"loss_semantic_group_margin": zero}

        margin = float(getattr(self, "semantic_group_margin", 0.2))
        topk = max(int(getattr(self, "semantic_group_margin_topk", 1)), 1)

        margin_terms = []          # list of scalar tensors (grad-carrying)
        dbg = {"matched": 0, "no_group": 0, "gt_oob": 0, "no_confuser": 0,
               "nonfinite_gt": 0, "nonfinite_conf": 0, "pairs": 0}

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            scores = scores_list[batch_idx] if batch_idx < len(scores_list) else None
            if scores is None or scores.shape[0] == 0 or scores.shape[1] == 0:
                continue
            n_cap = scores.shape[1]
            group_ids = targets[batch_idx].get("cap_group_ids", None)
            if group_ids is None:
                dbg["no_group"] += int(src_idx.numel())
                continue
            group_ids = group_ids.to(device=scores.device, dtype=torch.long)
            n_grp = min(n_cap, int(group_ids.numel()))
            gt_local = targets[batch_idx]["labels"][tgt_idx].to(
                device=scores.device, dtype=torch.long
            )
            for row, q in enumerate(src_idx.tolist()):
                dbg["matched"] += 1
                gt = int(gt_local[row].item())
                if gt < 0 or gt >= n_grp:
                    dbg["gt_oob"] += 1
                    continue
                row_scores = torch.nan_to_num(scores[q], nan=0.0, posinf=20.0, neginf=-20.0)
                gt_group = int(group_ids[gt].item())
                same = (group_ids[:n_grp] == gt_group).clone()
                same[gt] = False
                same_idx = torch.nonzero(same, as_tuple=False).squeeze(-1)
                if same_idx.numel() == 0:               # no valid confuser -> skip
                    dbg["no_confuser"] += 1
                    continue
                gt_score = row_scores[gt]
                if not torch.isfinite(gt_score):
                    dbg["nonfinite_gt"] += 1
                    continue
                conf_scores = row_scores[same_idx]
                conf_scores = conf_scores[torch.isfinite(conf_scores)]
                if conf_scores.numel() == 0:
                    dbg["nonfinite_conf"] += 1
                    continue
                k = min(topk, int(conf_scores.numel()))
                top_conf = torch.topk(conf_scores, k=k).values   # [k], all finite
                term = F.relu(margin - (gt_score - top_conf)).sum()
                if torch.isfinite(term):
                    margin_terms.append(term)
                    dbg["pairs"] += k

        if debug:
            self._group_margin_dbg_done = True
            print(f"[group-margin-dbg] {dbg} | n_terms={len(margin_terms)} margin={margin} topk={topk}")

        if not margin_terms or dbg["pairs"] == 0:
            return {"loss_semantic_group_margin": zero}
        result = torch.stack(margin_terms).sum() / float(dbg["pairs"])
        if not torch.isfinite(result):
            return {"loss_semantic_group_margin": zero}
        return {"loss_semantic_group_margin": result}

    def loss_dgcl_contrastive(self, outputs, targets, indices, num_boxes):
        """Category-level hard-negative supervision for the DGCL semantic head.

        The detector focal loss is dominated by unmatched query-token negatives.
        DGCL instead needs a direct signal that ranks the matched ground-truth
        category above the strongest prompted confusers. The loss operates on
        The logit source is configurable: v5 probes the raw head, while v6
        optimizes the residual-blended scores used by final detections.
        """
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        logit_source = str(
            getattr(self, "dgcl_contrastive_logit_source", "raw")
        ).strip().lower()
        label_map_list = outputs.get("_criterion_label_map_list")
        if label_map_list is None:
            return {"loss_dgcl_contrastive": zero}

        if logit_source in {"category", "category_raw", "cat", "cat_raw"}:
            category_logits = outputs.get("pred_logits_dgcl_category_raw")
            if category_logits is None:
                return {"loss_dgcl_contrastive": zero}
            result, stats = matched_category_logits_contrastive_loss(
                category_logits,
                outputs["pred_boxes"],
                targets,
                indices,
                label_map_list=label_map_list,
                baseline_token_logits=outputs.get("pred_logits_baseline"),
                margin=getattr(self, "dgcl_contrastive_margin", 0.2),
                topk=getattr(self, "dgcl_contrastive_topk", 5),
                iou_threshold=getattr(
                    self, "dgcl_contrastive_iou_threshold", 0.5
                ),
                ce_weight=getattr(self, "dgcl_contrastive_ce_weight", 1.0),
                rank_weight=getattr(
                    self, "dgcl_contrastive_rank_weight", 0.5
                ),
                hard_only=getattr(
                    self, "dgcl_contrastive_hard_only", False
                ),
                baseline_margin_threshold=getattr(
                    self, "dgcl_contrastive_baseline_margin_threshold", 0.5
                ),
                preserve_weight=getattr(
                    self, "dgcl_contrastive_preserve_weight", 0.0
                ),
                class_weights=getattr(
                    self, "dgcl_contrastive_class_weights", None
                ),
                zero_like=outputs["pred_logits"],
                use_inference_fused_logits=getattr(
                    self, "dgcl_contrastive_use_inference_fused_logits", False
                ),
                category_residual_weight=getattr(
                    self, "dgcl_category_residual_weight", 1.0
                ),
                category_residual_logit_space=getattr(
                    self, "dgcl_category_residual_logit_space", False
                ),
                category_residual_match_baseline_distribution=getattr(
                    self,
                    "dgcl_category_residual_match_baseline_distribution",
                    False,
                ),
                error_targeted=getattr(
                    self, "dgcl_contrastive_error_targeted", False
                ),
                preserve_correct=getattr(
                    self, "dgcl_contrastive_preserve_correct", False
                ),
            )
        else:
            if logit_source == "final":
                token_logits = outputs.get("pred_logits_dgcl")
            else:
                token_logits = outputs.get("pred_logits_dgcl_raw")
            if token_logits is None:
                return {"loss_dgcl_contrastive": zero}
            result, stats = matched_category_contrastive_loss(
                token_logits,
                outputs["pred_boxes"],
                label_map_list,
                targets,
                indices,
                baseline_token_logits=outputs.get("pred_logits_baseline"),
                margin=getattr(self, "dgcl_contrastive_margin", 0.2),
                topk=getattr(self, "dgcl_contrastive_topk", 5),
                iou_threshold=getattr(
                    self, "dgcl_contrastive_iou_threshold", 0.5
                ),
                ce_weight=getattr(self, "dgcl_contrastive_ce_weight", 1.0),
                rank_weight=getattr(
                    self, "dgcl_contrastive_rank_weight", 0.5
                ),
                hard_only=getattr(
                    self, "dgcl_contrastive_hard_only", False
                ),
                baseline_margin_threshold=getattr(
                    self, "dgcl_contrastive_baseline_margin_threshold", 0.5
                ),
                preserve_weight=getattr(
                    self, "dgcl_contrastive_preserve_weight", 0.0
                ),
                class_weights=getattr(
                    self, "dgcl_contrastive_class_weights", None
                ),
            )
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_contrastive_debug_printed", False)
            and stats["valid_rows"] > 0
        ):
            print(
                "[dgcl-flow][contrastive] source={} matched={} hard={} "
                "preserved={} weight_mean={:.4f} loss={:.6f}".format(
                    logit_source,
                    stats.get("matched_rows", stats["valid_rows"]),
                    stats["valid_rows"],
                    stats.get("preserved_rows", 0),
                    float(stats.get("weight_mean", zero).detach().cpu()),
                    float(result.detach().cpu()),
                ),
                flush=True,
            )
            self._dgcl_contrastive_debug_printed = True
        return {"loss_dgcl_contrastive": result}


    def loss_dgcl_unmatched_preserve(self, outputs, targets, indices, num_boxes):
        """Keep DGCL from promoting high-score unmatched baseline queries."""
        zero = outputs["pred_logits"].new_zeros((), dtype=torch.float32)
        category_logits_list = outputs.get("pred_logits_dgcl_category_raw")
        baseline_token_logits = outputs.get("pred_logits_baseline")
        label_map_list = outputs.get("_criterion_label_map_list")
        pred_boxes = outputs.get("pred_boxes")
        if (
            category_logits_list is None
            or baseline_token_logits is None
            or label_map_list is None
            or pred_boxes is None
        ):
            return {"loss_dgcl_unmatched_preserve": zero}

        iou_thr = float(getattr(self, "dgcl_unmatched_preserve_iou_thr", 0.3))
        query_topk = int(getattr(self, "dgcl_unmatched_preserve_query_topk", 300))
        class_topk = int(getattr(self, "dgcl_unmatched_preserve_topk", 10))
        terms = []
        for batch_idx, target in enumerate(targets):
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
            label_map = label_map_list[batch_idx].to(
                device=baseline_token_logits.device, dtype=torch.float32
            )
            if (
                label_map.numel() == 0
                or label_map.shape[0] != sample_category_logits.shape[-1]
            ):
                continue
            label_map = label_map / label_map.sum(dim=-1, keepdim=True).clamp_min(1.0)
            sample_baseline_tokens = torch.nan_to_num(
                baseline_token_logits[batch_idx].detach().float(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            baseline_cat_prob = (
                sample_baseline_tokens.sigmoid() @ label_map.transpose(0, 1)
            )
            baseline_detection_score = sample_baseline_tokens.sigmoid().max(dim=-1).values

            num_queries = min(
                int(pred_boxes.shape[1]),
                int(sample_category_logits.shape[0]),
                int(baseline_cat_prob.shape[0]),
            )
            if num_queries <= 0:
                continue
            unmatched = torch.ones(
                num_queries, device=pred_boxes.device, dtype=torch.bool
            )
            gt_boxes = target.get("boxes")
            if gt_boxes is not None and gt_boxes.numel() > 0:
                pred_xyxy = box_ops.box_cxcywh_to_xyxy(
                    pred_boxes[batch_idx, :num_queries].float()
                )
                gt_xyxy = box_ops.box_cxcywh_to_xyxy(
                    gt_boxes.to(device=pred_boxes.device, dtype=torch.float32)
                )
                max_iou = box_ops.box_iou(pred_xyxy, gt_xyxy)[0].max(dim=-1).values
                unmatched = max_iou < iou_thr

            if not unmatched.any():
                continue
            kq = max(1, min(query_topk, num_queries))
            threshold = baseline_detection_score[:num_queries].topk(k=kq, dim=0).values[-1]
            selected = unmatched & (baseline_detection_score[:num_queries] >= threshold)
            if not selected.any():
                continue

            base_rows = baseline_cat_prob[:num_queries][selected]
            dgcl_rows = sample_category_logits[:num_queries][selected].float().sigmoid()
            kc = max(1, min(class_topk, base_rows.shape[-1]))
            top_classes = base_rows.topk(k=kc, dim=-1).indices
            terms.append(
                F.smooth_l1_loss(
                    dgcl_rows.gather(1, top_classes),
                    base_rows.gather(1, top_classes).detach(),
                )
            )

        if not terms:
            return {"loss_dgcl_unmatched_preserve": zero}
        result = torch.stack(terms).mean()
        if not torch.isfinite(result):
            result = zero
        return {"loss_dgcl_unmatched_preserve": result}


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'quality': self.loss_quality,
            'lgpr_survival': self.loss_lgpr_survival,
            'lgpr_loc_preserve': self.loss_lgpr_loc_preserve,
            'lgpr_box_refine': self.loss_lgpr_box_refine,
            'visual_confidence': self.loss_visual_confidence,
            'appearance_gate': self.loss_appearance_gate,
            'query_contrastive': self.loss_query_contrastive,
            'global_image_text_sigmoid': self.loss_global_image_text_sigmoid,
            'query_selector': self.loss_query_selector,
            'aux_token_ce': self.loss_aux_token_ce,
            'proto_cls': self.loss_proto_cls,
            'appearance_cls': self.loss_appearance_cls,
            'appearance_contrastive': self.loss_appearance_contrastive,
            'appearance_text_alignment': self.loss_appearance_text_alignment,
            'semantic_group_margin': self.loss_semantic_group_margin,
            'dgcl_contrastive': self.loss_dgcl_contrastive,
            'dgcl_unmatched_preserve': self.loss_dgcl_unmatched_preserve,
            'facb_symbol_align': self.loss_facb_symbol_align,
            'self_facb': self.loss_self_facb,
            'facb_agf': self.loss_facb_agf,
            'facb_text_orth': self.loss_facb_text_orth,
            'facb_speedhr': self.loss_facb_speedhr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _build_loss_text_mask(self, outputs, targets, label_map_list):
        base_text_mask = outputs.get("text_mask", None)
        if base_text_mask is None:
            return None

        if not any("caption_label_loss_mask" in target for target in targets):
            return base_text_mask

        loss_text_mask = base_text_mask.clone()
        device = loss_text_mask.device
        for batch_idx, target in enumerate(targets):
            label_loss_mask = target.get("caption_label_loss_mask", None)
            if label_loss_mask is None:
                continue

            label_loss_mask = label_loss_mask.to(device=device, dtype=torch.bool).view(-1)
            if label_loss_mask.numel() != label_map_list[batch_idx].shape[0]:
                raise ValueError(
                    "caption_label_loss_mask has {} entries, but caption has {} labels".format(
                        label_loss_mask.numel(), label_map_list[batch_idx].shape[0]
                    )
                )

            ignored_label_mask = ~label_loss_mask.cpu()
            if ignored_label_mask.any():
                ignored_token_mask = label_map_list[batch_idx][ignored_label_mask].any(dim=0)
                loss_text_mask[batch_idx] = loss_text_mask[batch_idx] & ~ignored_token_mask.to(
                    device=device, dtype=torch.bool
                )

        return loss_text_mask

    def _build_target_token_map(self, label_map, target, target_indices):
        target_labels = target["labels"].cpu()[target_indices]
        if target_labels.numel() > 0:
            min_label = int(target_labels.min().item())
            max_label = int(target_labels.max().item())
            if min_label < 0 or max_label >= label_map.shape[0]:
                raise IndexError(
                    "Target label index out of bounds for current caption: "
                    f"min={min_label}, max={max_label}, "
                    f"num_caption_labels={label_map.shape[0]}"
                )
        positive_map = target.get("label_positive_map", None)
        if positive_map is None:
            return label_map[target_labels].to(torch.float32)

        rows = []
        for target_index, label_id in zip(target_indices.cpu().tolist(), target_labels.tolist()):
            label_indices = positive_map[target_index]
            if torch.is_tensor(label_indices):
                label_indices = label_indices.cpu().tolist()
            if not label_indices:
                label_indices = [label_id]
            label_indices = torch.as_tensor(label_indices, dtype=torch.long)
            if label_indices.numel() > 0:
                min_label_index = int(label_indices.min().item())
                max_label_index = int(label_indices.max().item())
                if min_label_index < 0 or max_label_index >= label_map.shape[0]:
                    raise IndexError(
                        "label_positive_map index out of bounds for current caption: "
                        f"min={min_label_index}, max={max_label_index}, "
                        f"num_caption_labels={label_map.shape[0]}, "
                        f"fallback_label={label_id}"
                    )
            rows.append(label_map[label_indices].amax(dim=0))

        if not rows:
            return label_map[target_labels].to(torch.float32)
        return torch.stack(rows, dim=0).to(torch.float32)

    def _build_query_token_ignore_mask(self, outputs, targets, label_map_list):
        base_text_mask = outputs.get("text_mask", None)
        if base_text_mask is None:
            return None
        if not any("token_ignore_map" in target for target in targets):
            return base_text_mask

        query_token_mask = base_text_mask.clone()
        device = query_token_mask.device
        for batch_idx, target in enumerate(targets):
            ignore_map = target.get("token_ignore_map", None)
            if ignore_map is None:
                continue
            ignore_map = ignore_map.to(device=device, dtype=torch.bool)
            if ignore_map.shape[0] != label_map_list[batch_idx].shape[0]:
                raise ValueError(
                    "token_ignore_map has {} entries, but caption has {} labels".format(
                        ignore_map.shape[0], label_map_list[batch_idx].shape[0]
                    )
                )
            ignored_token_mask = ignore_map.any(dim=0)
            query_token_mask[batch_idx] = query_token_mask[batch_idx] & ~ignored_token_mask
        return query_token_mask

    def forward(
        self,
        outputs,
        targets,
        cat_list,
        caption,
        return_indices=False,
        aux_cat_list=None,
        aux_caption=None,
    ):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device=next(iter(outputs.values())).device
        one_hot = torch.zeros(
            outputs['pred_logits'].size(),
            dtype=torch.float32,
            device=device,
        )  # torch.Size([bs, 900, 256])
        token = outputs.get('supervision_token', outputs['token']) 
        
        label_map_list = []
        indices = []
        for j in range(len(cat_list)): # bs
            if len(cat_list[j]) == 0:
                label_map = torch.zeros(
                    (0, outputs["pred_logits"].size(-1)),
                    dtype=torch.float32,
                )
            else:
                label_map = []
                for i in range(len(cat_list[j])):
                    label_id=torch.tensor([i])
                    per_label = create_positive_map(
                        token[j],
                        label_id,
                        cat_list[j],
                        caption[j],
                        max_text_len=outputs["pred_logits"].size(-1),
                    )
                    label_map.append(per_label)
                label_map=torch.stack(label_map,dim=0).squeeze(1)
            label_map_list.append(label_map)
        outputs["_criterion_label_map_list"] = label_map_list
        for j in range(len(cat_list)): # bs
            for_match = {
                "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
            }
            # Phase 2: IDF-weight the label_map for the MATCHER ONLY.  label_map_list[j]
            # (binary) is left untouched for the loss one_hot below.
            matcher_label_map = self._matcher_label_map(label_map_list[j], token[j])
            inds = self.matcher(for_match, [targets[j]], matcher_label_map)
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        quality_target = None
        if 'pred_quality' in outputs:
            quality_target = torch.zeros(outputs['pred_quality'].size(), dtype=torch.float32, device=device)

        for i in range(len(indices)):
            src_idx = indices[i][0]
            tgt_idx = indices[i][1]
            token_map = self._build_target_token_map(
                label_map_list[i],
                targets[i],
                tgt_idx,
            ).to(device=device, dtype=torch.float32)
            if len(src_idx) > 0:
                pred_box = outputs['pred_boxes'][i, src_idx]
                gt_box = targets[i]['boxes'][tgt_idx]
                pred_box = torch.nan_to_num(pred_box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                gt_box = torch.nan_to_num(gt_box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                pred_box = torch.cat([pred_box[:, :2], pred_box[:, 2:].clamp(min=1e-6)], dim=-1)
                gt_box = torch.cat([gt_box[:, :2], gt_box[:, 2:].clamp(min=1e-6)], dim=-1)
                iou = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(pred_box),
                    box_ops.box_cxcywh_to_xyxy(gt_box)
                )[0]
                iou_per_pair = torch.nan_to_num(
                    iou.diag().to(device=device, dtype=torch.float32),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                if quality_target is not None:
                    quality_target[i, src_idx, 0] = iou_per_pair
            one_hot[i, src_idx] = token_map
        outputs['one_hot'] = one_hot
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_target_debug_printed", False)
        ):
            print(
                "[dgcl-flow][target] one_hot.shape={}".format(tuple(one_hot.shape)),
                flush=True,
            )
            if len(label_map_list) > 0:
                label_map0 = label_map_list[0]
                print(
                    "[dgcl-flow][target] label_map[0].shape={}".format(
                        tuple(label_map0.shape)
                    ),
                    flush=True,
                )
                for row_idx in range(min(2, int(label_map0.shape[0]))):
                    row_nz = (label_map0[row_idx] > 0).nonzero(as_tuple=False).flatten()
                    print(
                        "[dgcl-flow][target] label_map[0][{}] positive_tokens={}".format(
                            row_idx, row_nz[:30].detach().cpu().tolist()
                        ),
                        flush=True,
                    )
            matched_rows = (one_hot[0].sum(dim=-1) > 0).nonzero(as_tuple=False).flatten() if one_hot.shape[0] > 0 else []
            if len(matched_rows) > 0:
                q_idx = int(matched_rows[0].item())
                token_nz = (one_hot[0, q_idx] > 0).nonzero(as_tuple=False).flatten()
                print(
                    "[dgcl-flow][target] first matched query batch=0 query={} positive_tokens={}".format(
                        q_idx, token_nz[:50].detach().cpu().tolist()
                    ),
                    flush=True,
                )
            self._dgcl_flow_target_debug_printed = True
        if quality_target is not None:
            outputs['quality_target'] = quality_target
        outputs['loss_text_mask'] = self._build_loss_text_mask(outputs, targets, label_map_list)
        outputs['query_token_ignore_mask'] = self._build_query_token_ignore_mask(
            outputs, targets, label_map_list
        )
        if "pred_aux_token_logits" in outputs:
            if aux_cat_list is None or aux_caption is None:
                raise ValueError("aux_token_ce requires aux_cat_list and aux_caption")
            aux_token = outputs.get('aux_token', outputs.get('supervision_token', outputs['token']))
            aux_label_map_list = []
            for j in range(len(aux_cat_list)):
                if len(aux_cat_list[j]) == 0:
                    aux_label_map = torch.zeros(
                        (0, outputs["pred_aux_token_logits"].size(-1)),
                        dtype=torch.float32,
                    )
                else:
                    aux_label_map = []
                    for i in range(len(aux_cat_list[j])):
                        label_id = torch.tensor([i])
                        per_label = create_positive_map(
                            aux_token[j],
                            label_id,
                            aux_cat_list[j],
                            aux_caption[j],
                            max_text_len=outputs["pred_aux_token_logits"].size(-1),
                        )
                        aux_label_map.append(per_label)
                    aux_label_map = torch.stack(aux_label_map, dim=0).squeeze(1)
                aux_label_map_list.append(aux_label_map)
            aux_one_hot = torch.zeros(
                outputs["pred_aux_token_logits"].size(),
                dtype=torch.float32,
                device=device,
            )
            for i in range(len(indices)):
                src_idx = indices[i][0]
                tgt_idx = indices[i][1]
                token_map = self._build_target_token_map(
                    aux_label_map_list[i],
                    targets[i],
                    tgt_idx,
                ).to(device=device, dtype=torch.float32)
                aux_one_hot[i, src_idx] = token_map
            outputs["aux_one_hot"] = aux_one_hot
            outputs["aux_label_map_list"] = aux_label_map_list
            outputs["aux_loss_text_mask"] = self._build_loss_text_mask(
                {"text_mask": outputs["aux_text_mask"]}, targets, aux_label_map_list
            )
            outputs["aux_query_token_ignore_mask"] = self._build_query_token_ignore_mask(
                {"text_mask": outputs["aux_text_mask"]}, targets, aux_label_map_list
            )
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if self.aggr_regularizer is not None:
            if "category_text_prototypes" not in outputs or "category_prototype_valid_mask" not in outputs:
                pass  # skip AGGR when prototypes are unavailable (e.g. eval on COCO-format data)
            elif any("cap_group_ids" not in target for target in targets):
                pass  # skip AGGR when group ids are unavailable (e.g. eval on COCO-format data)
            else:
                losses.update(
                    self.aggr_regularizer(
                        category_prototypes=outputs["category_text_prototypes"],
                        category_group_ids=[target["cap_group_ids"] for target in targets],
                        positive_label_indices=[target["labels"] for target in targets],
                        prototype_valid_masks=outputs["category_prototype_valid_mask"],
                    )
                )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)): # bs
                    aux_output_single = {
                        'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(aux_output_single, [targets[j]], self._matcher_label_map(label_map_list[j], token[j]))
                    indices.extend(inds)
                one_hot_aux = torch.zeros(
                    outputs['pred_logits'].size(),
                    dtype=torch.float32,
                    device=device,
                )
                quality_target_aux = None
                if 'pred_quality' in aux_outputs:
                    quality_target_aux = torch.zeros(aux_outputs['pred_quality'].size(), dtype=torch.float32, device=device)
                for i in range(len(indices)):
                    src_idx = indices[i][0]
                    tgt_idx = indices[i][1]
                    token_map = self._build_target_token_map(
                        label_map_list[i],
                        targets[i],
                        tgt_idx,
                    ).to(device=device, dtype=torch.float32)
                    if len(src_idx) > 0:
                        pred_box = aux_outputs['pred_boxes'][i, src_idx]
                        gt_box = targets[i]['boxes'][tgt_idx]
                        pred_box = torch.nan_to_num(pred_box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                        gt_box = torch.nan_to_num(gt_box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                        pred_box = torch.cat([pred_box[:, :2], pred_box[:, 2:].clamp(min=1e-6)], dim=-1)
                        gt_box = torch.cat([gt_box[:, :2], gt_box[:, 2:].clamp(min=1e-6)], dim=-1)
                        iou = box_ops.box_iou(
                            box_ops.box_cxcywh_to_xyxy(pred_box),
                            box_ops.box_cxcywh_to_xyxy(gt_box)
                        )[0]
                        iou_per_pair = torch.nan_to_num(
                            iou.diag().to(device=device, dtype=torch.float32),
                            nan=0.0,
                            posinf=1.0,
                            neginf=0.0,
                        ).clamp(0.0, 1.0)
                        if quality_target_aux is not None:
                            quality_target_aux[i, src_idx, 0] = iou_per_pair
                    one_hot_aux[i, src_idx] = token_map
                aux_outputs['one_hot'] = one_hot_aux
                if quality_target_aux is not None:
                    aux_outputs['quality_target'] = quality_target_aux
                aux_outputs['text_mask'] = outputs['text_mask']
                aux_outputs['loss_text_mask'] = outputs['loss_text_mask']
                aux_outputs['query_token_ignore_mask'] = outputs['query_token_ignore_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss in {'query_contrastive', 'global_image_text_sigmoid', 'aux_token_ce', 'proto_cls', 'appearance_cls', 'appearance_gate', 'appearance_contrastive', 'visual_confidence', 'semantic_group_margin', 'dgcl_contrastive', 'facb_symbol_align', 'self_facb', 'facb_text_orth', 'facb_speedhr', 'lgpr_survival', 'lgpr_loc_preserve', 'lgpr_box_refine'}:
                        continue
                    if loss == 'query_selector':
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = []
            for j in range(len(cat_list)): # bs
                interm_output_single = {
                    'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
                    'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                }
                inds = self.matcher(interm_output_single, [targets[j]], self._matcher_label_map(label_map_list[j], token[j]))
                indices.extend(inds)
            one_hot_aux = torch.zeros(
                outputs['pred_logits'].size(),
                dtype=torch.float32,
                device=device,
            )
            for i in range(len(indices)):
                src_idx = indices[i][0]
                tgt_idx = indices[i][1]
                token_map = self._build_target_token_map(
                    label_map_list[i],
                    targets[i],
                    tgt_idx,
                ).to(device=device, dtype=torch.float32)
                one_hot_aux[i, src_idx] = token_map
            interm_outputs['one_hot'] = one_hot_aux
            interm_outputs['text_mask'] = outputs['text_mask']
            interm_outputs['loss_text_mask'] = outputs['loss_text_mask']
            interm_outputs['query_token_ignore_mask'] = outputs['query_token_ignore_mask']
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'quality':
                    continue  # encoder has no quality head
                if loss in {'query_contrastive', 'global_image_text_sigmoid', 'aux_token_ce', 'proto_cls', 'appearance_cls', 'appearance_gate', 'appearance_contrastive', 'visual_confidence', 'semantic_group_margin', 'dgcl_contrastive', 'facb_symbol_align', 'self_facb', 'facb_text_orth', 'facb_speedhr', 'lgpr_survival', 'lgpr_loc_preserve', 'lgpr_box_refine'}:
                    continue  # keep query contrastive on final decoder only
                if loss == 'query_selector':
                    continue
                kwargs = {}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100,text_encoder_type='text_encoder_type', nms_iou_threshold=-1,use_coco_eval=False,args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.use_lgpr_survival_rescue = bool(
            getattr(args, "use_lgpr_survival_rescue", False)
        )
        configured_rescue_num = max(
            0, int(getattr(args, "lgpr_survival_rescue_num", 10))
        )
        self.lgpr_survival_rescue_num = min(configured_rescue_num, num_select)
        self.lgpr_survival_protected_num = int(
            getattr(
                args,
                "lgpr_survival_protected_num",
                num_select - self.lgpr_survival_rescue_num,
            )
        )
        self.lgpr_survival_protected_num = max(
            0,
            min(self.lgpr_survival_protected_num, num_select),
        )
        self.lgpr_survival_rescue_num = min(
            self.lgpr_survival_rescue_num,
            max(num_select - self.lgpr_survival_protected_num, 0),
        )
        self.lgpr_survival_score_weight = float(
            getattr(args, "lgpr_survival_score_weight", 0.2)
        )
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.max_text_len = getattr(args, "max_text_len", 256)
        self.use_appearance_class_fusion = bool(getattr(args, "use_appearance_class_fusion", False))
        self.appearance_class_fusion_weight = float(
            getattr(args, "appearance_class_fusion_weight", 1.0)
        )
        self.use_dgcl_category_residual = bool(
            getattr(args, "use_dgcl_category_residual", False)
        )
        self.dgcl_category_residual_weight = float(
            getattr(args, "dgcl_category_residual_weight", 0.2)
        )
        self.dgcl_category_residual_logit_space = bool(
            getattr(args, "dgcl_category_residual_logit_space", False)
        )
        self.dgcl_category_residual_match_baseline_distribution = bool(
            getattr(args, "dgcl_category_residual_match_baseline_distribution", False)
        )
        self.use_dgcl_category_residual_query_gate = bool(
            getattr(args, "use_dgcl_category_residual_query_gate", False)
        )
        self.dgcl_category_residual_gate_mode = str(
            getattr(args, "dgcl_category_residual_gate_mode", "topk_margin")
        ).strip().lower()
        self.dgcl_category_residual_gate_topk = int(
            getattr(args, "dgcl_category_residual_gate_topk", num_select)
        )
        self.dgcl_category_residual_gate_margin_quantile = float(
            getattr(args, "dgcl_category_residual_gate_margin_quantile", 0.5)
        )
        self.dgcl_category_residual_gate_margin_tau = float(
            getattr(args, "dgcl_category_residual_gate_margin_tau", -1.0)
        )
        self.use_dgcl_weighted_positive_map = bool(
            getattr(args, "use_dgcl_weighted_positive_map", False)
        )
        self.dgcl_weighted_positive_map_mix = float(
            getattr(args, "dgcl_weighted_positive_map_mix", 1.0)
        )
        self.use_appearance_proto_rerank = bool(getattr(args, "use_appearance_proto_rerank", False))
        self.appearance_proto_rerank_weight = float(
            getattr(args, "appearance_proto_rerank_weight", 0.5)
        )
        self.appearance_proto_rerank_mode = str(
            getattr(args, "appearance_proto_rerank_mode", "max")
        ).strip().lower()
        self.use_appearance_uncertainty_rerank = bool(
            getattr(args, "use_appearance_uncertainty_rerank", False)
        )
        self.use_appearance_local_evidence_rerank = bool(
            getattr(args, "use_appearance_local_evidence_rerank", False)
        )
        self.use_appearance_local_evidence_zscore_fusion = bool(
            getattr(args, "use_appearance_local_evidence_zscore_fusion", False)
        )
        self.appearance_local_evidence_zscore_lambda = float(
            getattr(args, "appearance_local_evidence_zscore_lambda", 2.0)
        )
        self.use_appearance_local_evidence_winner_take_top1 = bool(
            getattr(args, "use_appearance_local_evidence_winner_take_top1", False)
        )
        self.use_appearance_local_evidence_gated_replace = bool(
            getattr(args, "use_appearance_local_evidence_gated_replace", False)
        )
        self.appearance_local_evidence_gate_sem_margin = float(
            getattr(args, "appearance_local_evidence_gate_sem_margin", 1.3)
        )
        self.appearance_local_evidence_min_semantic_score = float(
            getattr(args, "appearance_local_evidence_min_semantic_score", 0.0)
        )
        self.appearance_local_evidence_max_semantic_score = float(
            getattr(args, "appearance_local_evidence_max_semantic_score", -1.0)
        )
        self.appearance_local_evidence_max_semantic_margin = float(
            getattr(args, "appearance_local_evidence_max_semantic_margin", -1.0)
        )
        self.appearance_rerank_weight = float(
            getattr(args, "appearance_rerank_weight", 0.05)
        )
        self.appearance_rerank_threshold = float(
            getattr(args, "appearance_rerank_threshold", 0.25)
        )
        self.appearance_rerank_margin_threshold = float(
            getattr(args, "appearance_rerank_margin_threshold", 0.08)
        )
        self.appearance_rerank_entropy_threshold = float(
            getattr(args, "appearance_rerank_entropy_threshold", 0.75)
        )
        self.appearance_rerank_topk = max(
            1, int(getattr(args, "appearance_rerank_topk", 5))
        )
        # ---- Neutral multiplicative fusion (replaces additive rerank) ----
        # When appearance has no information (uniform across top-k), multiplier≈1.0
        # → no change to baseline. When appearance supports/disagrees with a class,
        # it enhances/suppresses that class's score multiplicatively.
        self.use_appearance_neutral_fusion = bool(
            getattr(args, "use_appearance_neutral_fusion", False)
        )
        # Prototype-bank fusion: appearance_class_scores is already a global
        # [bs, nq, C] tensor (column i == global class id i), so it aligns directly
        # with prob_to_label and needs no local→global remap.  Reuses the neutral
        # multiplicative + uncertainty-gated logic, only the indexing differs.
        self.use_appearance_prototype_fusion = bool(
            getattr(args, "use_appearance_prototype_fusion", False)
        )
        self.use_appearance_same_group_fusion = bool(
            getattr(args, "use_appearance_same_group_fusion", False)
        )
        self.appearance_global_fine_group_ids = getattr(
            args, "appearance_global_fine_group_ids", None
        )
        if self.appearance_global_fine_group_ids is not None:
            self.appearance_global_fine_group_ids = torch.as_tensor(
                self.appearance_global_fine_group_ids, dtype=torch.long
            )
        # Alpha controls fusion strength. Start at 0.3; sweep [0.15, 0.3, 0.5, 0.7].
        self.appearance_fusion_alpha = float(
            getattr(args, "appearance_fusion_alpha", 0.3)
        )
        self.use_dynamic_appearance_fusion_alpha = bool(
            getattr(args, "use_dynamic_appearance_fusion_alpha", True)
        )
        self.appearance_fusion_clamp_min = float(
            getattr(args, "appearance_fusion_clamp_min", 0.2)
        )
        self.appearance_fusion_clamp_max = float(
            getattr(args, "appearance_fusion_clamp_max", 3.0)
        )
        self.use_appearance_seen_only_fusion = bool(
            getattr(args, "use_appearance_seen_only_fusion", False)
        )
        self.appearance_fusion_max_semantic_margin = float(
            getattr(args, "appearance_fusion_max_semantic_margin", -1.0)
        )
        self.appearance_seen_global_ids = self._resolve_appearance_seen_global_ids(args)
        if self.use_appearance_seen_only_fusion and not self.appearance_seen_global_ids:
            print(
                "Warning: use_appearance_seen_only_fusion=True but no seen category ids "
                "were resolved; appearance prototype fusion will be skipped."
            )
        # ---- Legacy additive rerank / rescoring (kept for compatibility) ----
        self.use_appearance_rescoring = bool(getattr(args, "use_appearance_rescoring", False))
        self.appearance_rescoring_temperature = float(
            getattr(args, "appearance_rescoring_temperature", 1.0)
        )
        self.appearance_rescoring_bias = float(
            getattr(args, "appearance_rescoring_bias", 0.0)
        )
        self.appearance_rescoring_mode = str(
            getattr(args, "appearance_rescoring_mode", "max")
        ).strip().lower()
        self.use_appearance_text_rerank = bool(getattr(args, "use_appearance_text_rerank", False))
        self.appearance_text_rerank_weight = float(
            getattr(args, "appearance_text_rerank_weight", 0.25)
        )
        self.use_visual_confidence = bool(getattr(args, "use_visual_confidence", False))
        self.visual_confidence_lambda = float(getattr(args, "visual_confidence_lambda", 1.0))
        self.use_appearance_conditioned_gate = bool(getattr(args, "use_appearance_conditioned_gate", False))
        self.appearance_gate_lambda = float(getattr(args, "appearance_gate_lambda", 0.3))
        self.appearance_gate_additive_weight = float(getattr(args, "appearance_gate_additive_weight", 0.0))
        self.use_postprocess_score_norm = bool(getattr(args, "use_postprocess_score_norm", False))
        self.postprocess_score_norm_type = str(
            getattr(args, "postprocess_score_norm_type", "zscore")
        ).strip().lower()
        self.postprocess_score_norm_eps = float(
            getattr(args, "postprocess_score_norm_eps", 1e-6)
        )
        # FACB high-resolution crop evidence is produced by the model at eval
        # time and consumed here before the final query-class top-k selection.
        self.enable_facb_speedhr_inference = bool(
            getattr(args, "enable_facb_speedhr_inference", False)
        )
        self.facb_speedhr_use_learned_quality = bool(getattr(args, "facb_speedhr_use_learned_quality", False))
        self.facb_local_selector_checkpoint = str(getattr(args, "facb_local_selector_checkpoint", ""))
        self.facb_local_selector_candidate_only = bool(getattr(args, "facb_local_selector_candidate_only", False))
        self._facb_local_selector = None
        self.facb_dual_facet_itm_checkpoint = str(
            getattr(args, "facb_dual_facet_itm_checkpoint", "") or ""
        )
        self._facb_dual_facet_itm_bundle = None
        self.facb_d1_score_enabled = bool(getattr(args, "facb_d1_score_enabled", False))
        self.facb_fixed_quality_checkpoint = str(getattr(args, "facb_fixed_quality_checkpoint", "") or "")
        self._facb_fixed_quality_saved = None
        if self.facb_fixed_quality_checkpoint and self.facb_d1_score_enabled:
            raise ValueError("Fixed quality and D1 cannot be enabled together")
        if self.facb_dual_facet_itm_checkpoint and self.facb_local_selector_checkpoint:
            raise ValueError("Dual-facet online bundle already contains frozen P0; do not also set facb_local_selector_checkpoint")
        self.facb_f1_mode = str(getattr(args, "facb_f1_mode", "off"))
        self.facb_f1_capture_dir = str(getattr(args, "facb_f1_capture_dir", ""))
        self.facb_f1_checkpoint = str(getattr(args, "facb_f1_checkpoint", ""))
        self.facb_f1_reader_id = str(getattr(args, "facb_f1_reader_id", ""))
        self._facb_f1_index = 0
        self._facb_f1_mapping = None
        self._facb_f1_gate_sha = None
        if self.facb_local_selector_checkpoint and self.facb_f1_mode != "off":
            raise ValueError("Recalibrate scoring for a new selector before enabling F1")
        if self.facb_f1_mode not in {"off", "f0", "f1"}:
            raise ValueError("facb_f1_mode must be off, f0 or f1")
        if self.facb_f1_capture_dir and self.facb_f1_mode != "f0":
            raise ValueError("Collect F1 mixed lists under F0 only")
        if self.facb_f1_mode != "off" and self.facb_speedhr_use_learned_quality:
            raise ValueError("F1/F0 cannot be combined with legacy direct-quality writes")
        self.facb_speedhr_infer_mode = str(
            getattr(args, "facb_speedhr_infer_mode", "label_only")
        ).strip().lower()
        self.facb_speedhr_infer_conf_threshold = float(
            getattr(args, "facb_speedhr_infer_conf_threshold", 0.7)
        )
        self.facb_speedhr_infer_margin_threshold = float(
            getattr(args, "facb_speedhr_infer_margin_threshold", 0.0)
        )
        self.facb_speedhr_infer_max_semantic_score = float(
            getattr(args, "facb_speedhr_infer_max_semantic_score", -1.0)
        )
        self.facb_speedhr_infer_min_semantic_score = float(
            getattr(args, "facb_speedhr_infer_min_semantic_score", -1.0)
        )
        self.facb_speedhr_infer_score_scale = float(
            getattr(args, "facb_speedhr_infer_score_scale", 1.0)
        )
        self.facb_speedhr_infer_score_cap = float(
            getattr(args, "facb_speedhr_infer_score_cap", 1.0)
        )
        # Evidence-calibrated score residual (paper-facing form).  When the
        # residual soft gate is off (default), ``residual`` mode reproduces the
        # exact prior behavior ``s += alpha * evidence``.  When on, the accepted
        # residual is scaled by a differentiable gate over FACB confidence and
        # margin, so the score correction is monotonically tied to evidence
        # strength instead of being a hard threshold plus a constant score.
        self.facb_speedhr_infer_residual_soft_gate = bool(
            getattr(args, "facb_speedhr_infer_residual_soft_gate", False)
        )
        self.facb_speedhr_infer_residual_gate_temperature = float(
            getattr(args, "facb_speedhr_infer_residual_gate_temperature", 0.1)
        )
        # Temperature applied to the three short-attribute similarity logits
        # (group/shape/symbol) before sigmoid in ``compositional`` mode.
        self.facb_speedhr_infer_facet_temperature = float(
            getattr(args, "facb_speedhr_infer_facet_temperature", 1.0)
        )
        self.facb_speedhr_infer_composition_lambda = float(
            getattr(args, "facb_speedhr_infer_composition_lambda", 1.0)
        )
        # Candidate selection: combine the detector's candidate prior with the
        # trained cross-attention candidate classifier.  This is distinct from
        # the hand-composed facet score used by ``composition_rerank``.
        self.facb_speedhr_infer_classifier_lambda = float(
            getattr(args, "facb_speedhr_infer_classifier_lambda", 1.0)
        )
        self.facb_speedhr_infer_group_gate_floor = float(
            getattr(args, "facb_speedhr_infer_group_gate_floor", 0.3)
        )
        self.facb_speedhr_accept_replay_rows = str(
            getattr(args, "facb_speedhr_accept_replay_rows", "") or ""
        )
        self.facb_speedhr_accept_replay_topn = int(
            getattr(args, "facb_speedhr_accept_replay_topn", 200) or 0
        )
        self.facb_speedhr_accept_replay_iou_thr = float(
            getattr(args, "facb_speedhr_accept_replay_iou_thr", 0.90)
        )
        self._facb_speedhr_accept_replay_index = None
        self.facb_speedhr_accept_gate_checkpoint = str(
            getattr(args, "facb_speedhr_accept_gate_checkpoint", "") or ""
        )
        self.facb_speedhr_accept_gate_threshold = float(
            getattr(args, "facb_speedhr_accept_gate_threshold", 0.5)
        )
        # Online selection: a deterministic, fully online query-level
        # abstention rule.  It uses only signals available in this forward pass
        # and runs before final query/class selection.  Unlike the legacy MLP
        # gate, it requires no checkpoint and can be audited condition by
        # condition.
        self.facb_speedhr_online_rule_gate = bool(
            getattr(args, "facb_speedhr_online_rule_gate", False)
        )
        self.facb_speedhr_rule_min_facb_margin = float(
            getattr(args, "facb_speedhr_rule_min_facb_margin", 0.95)
        )
        self.facb_speedhr_rule_min_facb_conf = float(
            getattr(args, "facb_speedhr_rule_min_facb_conf", 0.0)
        )
        self.facb_speedhr_rule_max_selected_slot = int(
            getattr(args, "facb_speedhr_rule_max_selected_slot", 3)
        )
        self.facb_speedhr_rule_min_det_winner_score = float(
            getattr(args, "facb_speedhr_rule_min_det_winner_score", 0.10)
        )
        self.facb_speedhr_rule_min_dup_winner_iou = float(
            getattr(args, "facb_speedhr_rule_min_dup_winner_iou", 0.50)
        )
        self.facb_speedhr_rule_gate_enabled = bool(
            getattr(args, "facb_speedhr_rule_gate_enabled", False)
        )
        self.facb_speedhr_rule_gate_min_margin = float(
            getattr(args, "facb_speedhr_rule_gate_min_margin", 0.95)
        )
        self.facb_speedhr_rule_gate_max_selected_slot = int(
            getattr(args, "facb_speedhr_rule_gate_max_selected_slot", 3)
        )
        self.facb_speedhr_rule_gate_min_det_winner_score = float(
            getattr(args, "facb_speedhr_rule_gate_min_det_winner_score", 0.10)
        )
        self.facb_speedhr_rule_gate_min_dup_iou_winner = float(
            getattr(args, "facb_speedhr_rule_gate_min_dup_iou_winner", 0.50)
        )
        self.facb_speedhr_accept_gate_audit_jsonl = str(
            getattr(args, "facb_speedhr_accept_gate_audit_jsonl", "")
            or os.environ.get("GROUNDINGDINO_FACB_SPEEDHR_ACCEPT_GATE_AUDIT_JSONL", "")
            or ""
        )
        self.facb_speedhr_accept_gate_audit_limit = int(
            getattr(args, "facb_speedhr_accept_gate_audit_limit", 0) or 0
        )
        self._facb_speedhr_accept_gate_audit_count = 0
        self._facb_speedhr_accept_gate_audit_initialized = False
        self._facb_speedhr_accept_gate = None
        self.facb_dual_facet_itm_audit_jsonl = str(
            getattr(args, "facb_dual_facet_itm_audit_jsonl", "") or ""
        )
        self._facb_dual_facet_itm_audit_initialized = False
        self._facb_dual_facet_itm_audit_count = 0
        # Independent train-side capture; disabled in the formal D0 config.
        self.facb_score_replay_capture_dir = str(getattr(args, "facb_score_replay_capture_dir", "") or "")
        self.facb_score_replay_provenance = str(getattr(args, "facb_score_replay_provenance", "") or "")
        self.facb_speedhr_infer_dump_jsonl = str(
            getattr(args, "facb_speedhr_infer_dump_jsonl", "")
            or os.environ.get("GROUNDINGDINO_FACB_SPEEDHR_INFER_DUMP_JSONL", "")
            or ""
        )
        self.facb_speedhr_infer_dump_limit = int(
            getattr(args, "facb_speedhr_infer_dump_limit", 0) or 0
        )
        self._facb_speedhr_infer_dump_count = 0
        self._facb_speedhr_infer_dump_initialized = False
        self.facb_speedhr_infer_raise_same_top = bool(
            getattr(args, "facb_speedhr_infer_raise_same_top", True)
        )
        self.facb_speedhr_infer_skip_high_iou_duplicates = bool(
            getattr(args, "facb_speedhr_infer_skip_high_iou_duplicates", False)
        )
        self.facb_speedhr_infer_duplicate_iou_thr = float(
            getattr(args, "facb_speedhr_infer_duplicate_iou_thr", 0.7)
        )
        self.facb_speedhr_infer_duplicate_score_thr = float(
            getattr(args, "facb_speedhr_infer_duplicate_score_thr", 0.3)
        )
        self._facb_speedhr_infer_batches = 0
        self._facb_speedhr_infer_rows = 0
        self._facb_speedhr_infer_passed = 0
        self._facb_speedhr_infer_changed = 0
        self._facb_speedhr_infer_dup_skipped = 0
        self._facb_speedhr_infer_conf_sum = 0.0
        self._facb_speedhr_infer_before_sum = 0.0
        self._facb_speedhr_infer_after_sum = 0.0
        self._facb_speedhr_infer_alpha_sum = 0.0
        self._facb_speedhr_infer_evidence_sum = 0.0
        self._facb_speedhr_infer_residual_sum = 0.0
        # ---- Method A (Phase 1): discriminative-token reweighting in class scoring ----
        # Replaces the uniform per-class token mean in _build_positive_map with
        #   w = (1 - lambda) * uniform + lambda * idf_norm
        # where idf is the inverse-document-frequency of each token across the FULL
        # class set (shared tokens like "speed"/"limit" -> low weight; discriminative
        # tokens like "80" -> high weight).  lambda=0 reproduces the baseline exactly.
        # DF is keyed by token id (not string) so BPE space-prefix variants are exact,
        # and is precomputed over all 232 classes via set_global_token_df() so chunked
        # eval never recomputes a chunk-local (biased) DF.
        self.postprocess_token_idf_lambda = float(
            getattr(args, "postprocess_token_idf_lambda", 0.0)
        )
        self.postprocess_token_idf_gamma = float(
            getattr(args, "postprocess_token_idf_gamma", 1.0)
        )
        # token_id -> document frequency (#classes whose span contains this token).
        # None until set_global_token_df() runs; while None, weighting stays uniform.
        self._global_token_df = None
        self._global_token_df_num_classes = 0
        if args.use_coco_eval:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            label_ids = [item['id'] for item in category_dict]
            arg_label_list = list(getattr(args, "label_list", []) or [])

            def _fallback_label(item):
                # TT100K COCO stores shorthand ids in `name` (e.g. pl40), so
                # semantic text is the safer fallback when no label_map exists.
                return str(item.get('semantic', '')).strip() or str(item['name']).strip()

            if arg_label_list:
                cat_list = []
                for cat_pos, item in enumerate(category_dict):
                    if cat_pos < len(arg_label_list):
                        cat_list.append(str(arg_label_list[cat_pos]).strip())
                    else:
                        cat_list.append(_fallback_label(item))
                print(
                    f"PostProcess prompt labels: using args.label_list for "
                    f"{len(cat_list)} COCO categories"
                )
            else:
                cat_list = [_fallback_label(item) for item in category_dict]
        else:
            cat_list=args.label_list
            label_ids = list(range(len(cat_list)))

        self.nms_iou_threshold=nms_iou_threshold
        self.cat_list = cat_list
        self._global_cat_list = list(cat_list)
        self._global_coco_label_ids = list(label_ids)
        self.label_ids = torch.tensor(label_ids, dtype=torch.long)
        self._label_text_to_coco_id = {}
        self._label_text_to_global_index = {}
        for _global_index, (_label_text, _label_id) in enumerate(
            zip(self.cat_list, label_ids)
        ):
            self._label_text_to_coco_id[str(_label_text).strip()] = int(_label_id)
            self._label_text_to_global_index[str(_label_text).strip()] = int(
                _global_index
            )
        self.current_global_label_ids = torch.arange(
            len(self.cat_list), dtype=torch.long
        )
        self.positive_map = self._build_positive_map(self.cat_list)

    def update_cat_list(self, cat_list):
        """Rebuild positive_map for a different cat_list (e.g. eval caption)."""
        self.cat_list = cat_list
        current_keys = [str(text).strip() for text in cat_list]
        global_keys = [str(text).strip() for text in self._global_cat_list]
        global_positions = None
        for start in range(max(len(global_keys) - len(current_keys) + 1, 0)):
            if global_keys[start : start + len(current_keys)] == current_keys:
                global_positions = list(range(start, start + len(current_keys)))
                break
        if global_positions is None:
            global_positions = [
                self._label_text_to_global_index.get(key, -1)
                for key in current_keys
            ]
        _label_ids = [
            self._global_coco_label_ids[position]
            if 0 <= position < len(self._global_coco_label_ids)
            else self._label_text_to_coco_id.get(current_keys[idx], idx)
            for idx, position in enumerate(global_positions)
        ]
        self.current_global_label_ids = torch.as_tensor(
            global_positions, dtype=torch.long
        )
        self.label_ids = torch.tensor(_label_ids, dtype=torch.long)
        self.positive_map = self._build_positive_map(self.cat_list)

    @torch.no_grad()
    def _load_facb_speedhr_accept_replay_index(self):
        if self._facb_speedhr_accept_replay_index is not None:
            return self._facb_speedhr_accept_replay_index
        path_value = str(getattr(self, "facb_speedhr_accept_replay_rows", "") or "")
        index = {}
        if not path_value:
            self._facb_speedhr_accept_replay_index = index
            return index
        path = Path(path_value)
        if not path.exists():
            print(
                f"[facb-speedhr][accept-replay] rows file not found: {path}",
                flush=True,
            )
            self._facb_speedhr_accept_replay_index = index
            return index
        rows = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows.sort(key=lambda item: float(item.get("accept_prob", 0.0) or 0.0), reverse=True)
        topn = int(getattr(self, "facb_speedhr_accept_replay_topn", 200) or 0)
        if topn > 0:
            rows = rows[: min(topn, len(rows))]
        for row in rows:
            try:
                image_id = int(row["image_id"])
                label = int(row["winner_label"])
                box = [float(x) for x in row["query_box_xyxy"]]
            except Exception:
                continue
            index.setdefault(image_id, []).append(
                {
                    "winner_label": label,
                    "query_box_xyxy": box,
                    "accept_prob": float(row.get("accept_prob", 0.0) or 0.0),
                }
            )
        self._facb_speedhr_accept_replay_index = index
        print(
            f"[facb-speedhr][accept-replay] loaded {sum(len(v) for v in index.values())} rows from {path}",
            flush=True,
        )
        return index

    def _facb_speedhr_accept_replay_match(
        self,
        image_id,
        winner_label,
        query_box_xyxy_abs,
    ):
        index = self._load_facb_speedhr_accept_replay_index()
        if not index:
            return True
        candidates = index.get(int(image_id), [])
        if not candidates:
            return False
        box = query_box_xyxy_abs.view(1, 4)
        best_iou = 0.0
        for row in candidates:
            if int(row["winner_label"]) != int(winner_label):
                continue
            ref = box.new_tensor(row["query_box_xyxy"]).view(1, 4)
            value = float(box_ops.box_iou(box, ref)[0].item())
            best_iou = max(best_iou, value)
            if value >= float(self.facb_speedhr_accept_replay_iou_thr):
                return True
        return False

    def _load_facb_dual_facet_itm(self, device):
        if self._facb_dual_facet_itm_bundle is not None:
            return self._facb_dual_facet_itm_bundle
        path_value = str(getattr(self, "facb_dual_facet_itm_checkpoint", "") or "")
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Requested FACB P0/ITM bundle not found: {path}")
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "facb_p0_dual_facet_itm_online_v1":
            raise ValueError("Unsupported FACB P0/dual-facet ITM bundle schema")
        p0_state = checkpoint["p0"]
        p0 = LocalSelector(p0_state["dim"], p0_state["hidden"])
        p0.load_state_dict(p0_state["state_dict"], strict=True)
        itm_state = checkpoint["itm"]
        itm = DualFacetITM(itm_state["state_dim"], itm_state["hidden"])
        itm.load_state_dict(itm_state["state_dict"], strict=True)
        p0.to(device=device).eval().requires_grad_(False)
        itm.to(device=device).eval().requires_grad_(False)
        self._facb_dual_facet_itm_bundle = {
            "p0": p0,
            "itm": itm,
            "facet_mean": itm_state["raw_facet_mean"].to(device=device, dtype=torch.float32),
            "facet_std": itm_state["raw_facet_std"].to(device=device, dtype=torch.float32).clamp_min(1e-6),
            "lambda_local": float(p0_state["lambda_local"]),
            "threshold": float(itm_state["frozen_accept_threshold"]),
            "target_global_ids": set(map(int, checkpoint["target_global_ids"])),
            "protocol": checkpoint["protocol"],
            "checkpoint_sha256": sha256(path),
            "d1": validate_facb_d1_score(checkpoint["d1"]) if self.facb_d1_score_enabled else None,
        }
        print(
            f"[facb-speedhr][dual-facet-itm] loaded {path} "
            f"threshold={self._facb_dual_facet_itm_bundle['threshold']:.9f}",
            flush=True,
        )
        return self._facb_dual_facet_itm_bundle

    def _load_facb_speedhr_accept_gate(self, device):
        if self._facb_speedhr_accept_gate is not None:
            return self._facb_speedhr_accept_gate
        path_value = str(getattr(self, "facb_speedhr_accept_gate_checkpoint", "") or "")
        if not path_value:
            self._facb_speedhr_accept_gate = {}
            return self._facb_speedhr_accept_gate
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Requested FACB gate checkpoint not found: {path}")
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        architecture = str(checkpoint.get("architecture", "mlp")).strip().lower()
        if architecture == "proposal_pair_v1":
            if checkpoint.get("schema") != SCHEMA:
                raise ValueError("Unsupported feature schema")
            model = ProposalGate(checkpoint["state_dim"], checkpoint["hidden_dim"], checkpoint.get("learn_quality", False))
            model.load_state_dict(checkpoint["model_state"], strict=True)
            self._facb_speedhr_accept_gate = {
                "architecture": architecture, "model": model.to(device).eval(),
                "mean": checkpoint["mean"].to(device),
                "std": checkpoint["std"].to(device).clamp_min(1e-6),
                "thresholds": checkpoint["thresholds"],
                "quality_calibration": checkpoint.get("quality_calibration", [1., 0.]),
                "protocol": checkpoint["protocol"],
            }
            return self._facb_speedhr_accept_gate
        if architecture == "two_stage_linear":
            stages = {}
            status_rows = []
            for stage_name in ("validity", "override"):
                stage_checkpoint = checkpoint.get(stage_name, {})
                feature_names = [
                    str(x) for x in stage_checkpoint.get("feature_names", [])
                ]
                in_dim = int(
                    stage_checkpoint.get("in_dim", len(feature_names))
                )
                if in_dim <= 0 or len(feature_names) != in_dim:
                    raise ValueError(
                        f"invalid {stage_name} stage in FACB gate checkpoint: "
                        f"in_dim={in_dim} features={len(feature_names)}"
                    )
                model = nn.Linear(in_dim, 1)
                state = stage_checkpoint.get(
                    "model_state", stage_checkpoint.get("state_dict", {})
                )
                if any(str(k).startswith("linear.") for k in state.keys()):
                    state = {str(k)[7:]: v for k, v in state.items()}
                status = model.load_state_dict(state, strict=False)
                model.to(device=device)
                model.eval()
                stages[stage_name] = {
                    "model": model,
                    "mean": stage_checkpoint.get(
                        "mean", torch.zeros(in_dim)
                    ).to(device=device, dtype=torch.float32),
                    "std": stage_checkpoint.get(
                        "std", torch.ones(in_dim)
                    ).to(device=device, dtype=torch.float32).clamp_min(1e-6),
                    "feature_names": feature_names,
                    "threshold": float(stage_checkpoint.get("threshold", 0.5)),
                }
                status_rows.append(f"{stage_name}={status}")
            self._facb_speedhr_accept_gate = {
                "architecture": architecture,
                "stages": stages,
            }
            print(
                f"[facb-speedhr][accept-gate] loaded {path} "
                f"architecture={architecture} {' '.join(status_rows)}",
                flush=True,
            )
            return self._facb_speedhr_accept_gate

        in_dim = int(checkpoint.get("in_dim", len(checkpoint.get("feature_names", []))))
        hidden_dim = int(checkpoint.get("hidden_dim", 32))
        if architecture == "linear":
            model = nn.Linear(in_dim, 1)
        else:
            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.0),
                nn.Linear(hidden_dim, 1),
            )
        state = checkpoint.get("model_state", checkpoint.get("state_dict", {}))
        # Saved GateMLP keys are "net.0.weight", ...; strip the wrapper prefix
        # for the local Sequential used at inference.
        if any(str(k).startswith("net.") for k in state.keys()):
            state = {str(k)[4:]: v for k, v in state.items()}
        if architecture == "linear" and any(
            str(k).startswith("linear.") for k in state.keys()
        ):
            state = {str(k)[7:]: v for k, v in state.items()}
        status = model.load_state_dict(state, strict=False)
        model.to(device=device)
        model.eval()
        mean = checkpoint.get("mean", torch.zeros(in_dim)).to(device=device, dtype=torch.float32)
        std = checkpoint.get("std", torch.ones(in_dim)).to(device=device, dtype=torch.float32).clamp_min(1e-6)
        feature_names = [str(x) for x in checkpoint.get("feature_names", [])]
        if not feature_names:
            feature_names = [
                "det_top1_score_repeat",
                "det_top1_margin_repeat",
                "facb_conf_repeat",
                "det_top1_score",
                "det_top1_margin",
                "det_winner_score",
                "facb_conf",
                "facb_margin",
                "score_gap_facb_minus_det",
                "max_dup_iou_any",
                "max_dup_score_any",
                "box_sqrt_area",
                "box_aspect",
            ][:in_dim]
        self._facb_speedhr_accept_gate = {
            "model": model,
            "mean": mean,
            "std": std,
            "feature_names": feature_names,
            "architecture": architecture,
        }
        print(
            f"[facb-speedhr][accept-gate] loaded {path} architecture={architecture} "
            f"in_dim={in_dim} status={status}",
            flush=True,
        )
        return self._facb_speedhr_accept_gate

    def _facb_speedhr_accept_gate_score(self, values, device):
        gate = self._load_facb_speedhr_accept_gate(device)
        if not gate:
            self._facb_speedhr_accept_gate_last_components = {}
            return 1.0
        if gate.get("architecture") == "proposal_pair_v1":
            if values.get("_protocol") != gate["protocol"]:
                raise ValueError("collection/deployment proposal protocol mismatch")
            x = values.get("_representation")
            if x is None:
                raise ValueError("checkpoint requires candidate/crop/query states")
            with torch.no_grad():
                validity, override = gate["model"](((x - gate["mean"]) / gate["std"]).unsqueeze(0))
                pv, pu = validity.sigmoid()[0], override.softmax(-1)[0, 0]
                score = stage_score(pv, pu, gate["thresholds"])
            self._facb_speedhr_accept_gate_last_components = {
                "validity": float(pv), "override": float(pu), "combined": float(score),
            }
            if gate["model"].learn_quality:
                with torch.no_grad():
                    q, _, weights = gate["model"].quality(
                        ((x-gate["mean"])/gate["std"]).unsqueeze(0),
                        values["_candidate_text_states"].unsqueeze(0),
                        values["_facet_logits"].unsqueeze(0),
                        torch.tensor([values["winner_rank"]],device=device,dtype=torch.long),
                    )
                    scale,bias = gate["quality_calibration"]
                    self._facb_speedhr_accept_gate_last_components["quality"] = float((scale*q[0]+bias).sigmoid())
                    for i,name in enumerate(("group", "shape", "symbol")):
                        self._facb_speedhr_accept_gate_last_components[f"quality_weight_{name}"] = float(weights[0,int(values["winner_rank"]),i])
            return float(score)
        if gate.get("architecture") == "two_stage_linear":
            components = {}
            centered_logits = []
            with torch.no_grad():
                for stage_name in ("validity", "override"):
                    stage = gate["stages"][stage_name]
                    vector = [
                        float(values.get(name, 0.0) or 0.0)
                        for name in stage["feature_names"]
                    ]
                    x = torch.as_tensor(
                        vector, device=device, dtype=torch.float32
                    )
                    x = (x - stage["mean"]) / stage["std"]
                    raw_logit = stage["model"](x.view(1, -1)).view(-1)[0]
                    raw_prob = torch.sigmoid(raw_logit)
                    threshold = min(
                        max(float(stage["threshold"]), 1e-6), 1.0 - 1e-6
                    )
                    threshold_logit = math.log(threshold / (1.0 - threshold))
                    centered_logits.append(raw_logit - threshold_logit)
                    components[stage_name] = float(raw_prob.item())
            # Exactly reproduces the two hard stage thresholds while exposing
            # one [0, 1] score to the existing gate interface: score >= 0.5 iff
            # both validity and override pass their train-calibrated cutoffs.
            score = torch.sigmoid(torch.stack(centered_logits).min())
            components["combined"] = float(score.item())
            self._facb_speedhr_accept_gate_last_components = components
            return float(score.item())
        names = gate["feature_names"]
        vector = [float(values.get(name, 0.0) or 0.0) for name in names]
        x = torch.as_tensor(vector, device=device, dtype=torch.float32)
        x = (x - gate["mean"]) / gate["std"]
        with torch.no_grad():
            prob = torch.sigmoid(gate["model"](x.view(1, -1)).view(-1)[0])
        self._facb_speedhr_accept_gate_last_components = {
            "combined": float(prob.item())
        }
        return float(prob.item())

    def _apply_facb_speedhr_inference(self, prob_to_label, outputs, target_sizes=None):
        outputs["_facb_f1_changes"] = []
        f1_mode = getattr(self, "facb_f1_mode", "off")
        if f1_mode != "off":
            if (self.facb_speedhr_infer_mode != "semantic_classifier_rerank"
                    or self.facb_speedhr_infer_score_scale != 1.0
                    or self.facb_speedhr_infer_score_cap != 1.0):
                raise ValueError("F0/F1 requires semantic_classifier_rerank with scale=cap=1")
            if not self.facb_f1_reader_id or not self.facb_speedhr_accept_gate_checkpoint:
                raise ValueError("F0/F1 requires frozen reader identity and quality gate checkpoint")
            if self._facb_f1_gate_sha is None:
                self._facb_f1_gate_sha = sha256(self.facb_speedhr_accept_gate_checkpoint)
            if f1_mode == "f1" and self._facb_f1_mapping is None:
                self._facb_f1_mapping = load_fusion(self.facb_f1_checkpoint,
                    self.facb_speedhr_accept_gate_checkpoint,self.facb_f1_reader_id)
            if f1_mode == "f1":
                actual = outputs.get("facb_speedhr_inference",{}).get("proposal_protocol")
                expected = self._facb_f1_mapping.get("proposal_protocol")
                if actual is not None and actual != expected:
                    raise ValueError("F1 proposal protocol changed since collection")
        infer = outputs.get("facb_speedhr_inference", None)
        if (
            not self.enable_facb_speedhr_inference
            or not isinstance(infer, dict)
            or not isinstance(infer.get("candidate_logits", None), torch.Tensor)
        ):
            return prob_to_label

        mode = self.facb_speedhr_infer_mode
        if mode not in {
            "label_only",
            "replace",
            "residual",
            "query_rerank",
            "compositional",
            "composition_rerank",
            "semantic_classifier_rerank",
            "composition_residual",
            "composition_direct",
            "composition_symbol_shape",
        }:
            raise ValueError(
                "facb_speedhr_infer_mode must be 'label_only', 'replace', "
                "'residual', 'query_rerank', 'compositional', or "
                "'composition_rerank'/'semantic_classifier_rerank'/"
                "'composition_residual'/'composition_direct'/"
                "'composition_symbol_shape', "
                f"got {mode!r}"
            )
        batch_ids = infer["batch_indices"].to(
            device=prob_to_label.device, dtype=torch.long
        )
        query_ids = infer["query_indices"].to(
            device=prob_to_label.device, dtype=torch.long
        )
        candidate_ids = infer["candidate_ids"].to(
            device=prob_to_label.device, dtype=torch.long
        )
        allowed_ids = infer.get("allowed_global_ids", None)
        if allowed_ids is None:
            allowed_ids = getattr(
                self, "facb_speedhr_infer_allowed_global_ids", None
            )
        if allowed_ids is not None:
            allowed_ids = torch.as_tensor(
                allowed_ids,
                device=prob_to_label.device,
                dtype=torch.long,
            ).view(-1)
        candidate_logits = infer["candidate_logits"].to(
            device=prob_to_label.device, dtype=prob_to_label.dtype
        )
        candidate_facet_scores = infer.get("candidate_facet_scores", None)
        if isinstance(candidate_facet_scores, torch.Tensor):
            candidate_facet_scores = candidate_facet_scores.to(
                device=prob_to_label.device, dtype=torch.float32
            )
        else:
            candidate_facet_scores = None
        candidate_probs = torch.nan_to_num(
            candidate_logits.float().softmax(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        top_count = min(2, int(candidate_probs.shape[-1]))
        top = candidate_probs.topk(k=top_count, dim=-1)
        winner_slot = top.indices[:, 0]
        confidence = top.values[:, 0]
        if top_count >= 2:
            confidence_margin = top.values[:, 0] - top.values[:, 1]
        else:
            confidence_margin = confidence

        current_global_ids = self.current_global_label_ids.to(
            device=prob_to_label.device, dtype=torch.long
        )
        global_to_position = {
            int(global_id): int(position)
            for position, global_id in enumerate(
                current_global_ids.detach().cpu().tolist()
            )
            if int(global_id) >= 0
        }
        local_probabilities = None
        p0_bundle_active = False
        itm_logits_all = None
        itm_probabilities_all = None
        itm_bundle = self._load_facb_dual_facet_itm(prob_to_label.device)
        if itm_bundle is not None:
            if mode != "semantic_classifier_rerank":
                raise ValueError("Dual-facet ITM requires semantic_classifier_rerank")
            actual = dict(
                infer.get("proposal_protocol", {}), mode=mode,
                classifier_lambda=float(self.facb_speedhr_infer_classifier_lambda),
            )
            if actual != itm_bundle["protocol"]:
                raise ValueError(
                    "P0/dual-facet ITM collection/deployment protocol mismatch: "
                    f"actual={actual} expected={itm_bundle['protocol']}"
                )
            if candidate_facet_scores is None:
                raise ValueError("Dual-facet ITM requires candidate facet scores")
            candidate_mask = infer.get("candidate_mask", None)
            if not isinstance(candidate_mask, torch.Tensor):
                candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
            else:
                candidate_mask = candidate_mask.to(
                    device=prob_to_label.device, dtype=torch.bool
                )
                while candidate_mask.ndim > 2:
                    candidate_mask = candidate_mask.all(dim=-1)
            position_rows = torch.as_tensor(
                [[global_to_position[int(value)] for value in row]
                 for row in candidate_ids.detach().cpu().tolist()],
                device=prob_to_label.device, dtype=torch.long,
            )
            detector_candidate_scores = prob_to_label[
                batch_ids, query_ids
            ].gather(1, position_rows).float()
            with torch.no_grad(), torch.autocast(
                device_type=prob_to_label.device.type, enabled=False
            ):
                p0_output = itm_bundle["p0"](
                    infer["crop_states"].float(),
                    infer["candidate_text_states"].float(),
                    candidate_facet_scores.float(),
                )
                p0_local_logits = p0_output[:, : candidate_ids.shape[1]]
                p0_fused_logits = (
                    detector_candidate_scores.clamp_min(1e-8).log()
                    + itm_bundle["lambda_local"] * p0_local_logits
                ).masked_fill(~candidate_mask, -1e9)
                candidate_probs = p0_fused_logits.softmax(dim=-1)
                itm_logits_all = dual_facet_logits(
                    itm_bundle["itm"], infer["candidate_states"],
                    infer["candidate_text_states"], candidate_facet_scores,
                    itm_bundle["facet_mean"], itm_bundle["facet_std"],
                )
                itm_probabilities_all = itm_logits_all.sigmoid()
            local_probabilities = torch.cat(
                [candidate_probs, candidate_probs.new_zeros((candidate_probs.shape[0], 1))],
                dim=-1,
            )
            top = candidate_probs.topk(k=min(2, candidate_probs.shape[-1]), dim=-1)
            winner_slot = top.indices[:, 0]
            confidence = top.values[:, 0]
            confidence_margin = (
                top.values[:, 0] - top.values[:, 1]
                if top.values.shape[-1] > 1 else top.values[:, 0]
            )
            p0_bundle_active = True
        if getattr(self, "facb_local_selector_checkpoint", ""):
            if mode != "semantic_classifier_rerank":
                raise ValueError("Local selector requires semantic_classifier_rerank")
            if self._facb_local_selector is None:
                local_model, local_state = load_selector(self.facb_local_selector_checkpoint,prob_to_label.device)
                self._facb_local_selector = (local_model,local_state,sha256(self.facb_local_selector_checkpoint))
            local_model,local_state,local_hash = self._facb_local_selector
            actual = dict(infer.get("proposal_protocol",{}),mode=mode,
                          classifier_lambda=float(self.facb_speedhr_infer_classifier_lambda))
            if actual != local_state["protocol"]:
                raise ValueError("Local selector collection/deployment protocol mismatch")
            local_probabilities = selector_probabilities(local_model,infer["crop_states"],
                infer["candidate_text_states"],candidate_facet_scores)
            if self.facb_local_selector_candidate_only:
                # Renormalize over actual classes, retaining reject for audit only.
                candidate_probs = local_probabilities[:, :-1]
                candidate_probs = candidate_probs / candidate_probs.sum(-1,keepdim=True).clamp_min(1e-30)
                top = candidate_probs.topk(k=2,dim=-1)
                winner_slot = top.indices[:,0]
                confidence = top.values[:,0]
                confidence_margin = top.values[:,0]-top.values[:,1]
            # A newly selected class changes gate inputs. A legacy gate must
            # fail protocol validation rather than silently reuse old calibration.
            infer = dict(infer,proposal_protocol=dict(infer["proposal_protocol"],
                         local_selector_sha256=local_hash,
                         local_selector_candidate_only=self.facb_local_selector_candidate_only))
        rows = int(candidate_logits.shape[0])
        passed = 0
        changed = 0
        dup_skipped = 0
        conf_sum = 0.0
        before_sum = 0.0
        after_sum = 0.0
        alpha_sum = 0.0
        evidence_sum = 0.0
        residual_sum = 0.0
        pred_boxes = outputs.get("pred_boxes", None)
        image_ids = outputs.get("image_ids", None)
        if isinstance(image_ids, torch.Tensor):
            image_ids = image_ids.detach().cpu().view(-1).tolist()
        elif image_ids is not None:
            image_ids = list(image_ids)
        duplicate_boxes_xyxy = None
        duplicate_high_masks = None
        dump_path = self.facb_speedhr_infer_dump_jsonl.strip()

        def _round_list(values):
            return [round(float(x), 6) for x in values]

        def _maybe_dump_facb_record(record):
            if not dump_path:
                return
            limit = int(self.facb_speedhr_infer_dump_limit)
            if limit > 0 and self._facb_speedhr_infer_dump_count >= limit:
                return
            path = Path(dump_path)
            if not self._facb_speedhr_infer_dump_initialized:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                self._facb_speedhr_infer_dump_initialized = True
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._facb_speedhr_infer_dump_count += 1

        def _maybe_dump_accept_gate_audit(record):
            audit_path = str(
                getattr(self, "facb_speedhr_accept_gate_audit_jsonl", "") or ""
            ).strip()
            if not audit_path:
                return
            limit = int(getattr(self, "facb_speedhr_accept_gate_audit_limit", 0) or 0)
            if (
                limit > 0
                and self._facb_speedhr_accept_gate_audit_count >= limit
            ):
                return
            path = Path(audit_path)
            if not self._facb_speedhr_accept_gate_audit_initialized:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                self._facb_speedhr_accept_gate_audit_initialized = True
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._facb_speedhr_accept_gate_audit_count += 1

        def _maybe_dump_dual_facet_itm_audit(record):
            if self.facb_fixed_quality_checkpoint:
                outputs.setdefault("_facb_fixed_quality_rows", []).append(record)
            audit_path = str(
                getattr(self, "facb_dual_facet_itm_audit_jsonl", "") or ""
            ).strip()
            if not audit_path:
                return
            path = Path(audit_path)
            if not self._facb_dual_facet_itm_audit_initialized:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                self._facb_dual_facet_itm_audit_initialized = True
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._facb_dual_facet_itm_audit_count += 1

        if (
            self.facb_speedhr_infer_skip_high_iou_duplicates
            and isinstance(pred_boxes, torch.Tensor)
            and pred_boxes.shape[:2] == prob_to_label.shape[:2]
        ):
            duplicate_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
                pred_boxes.to(device=prob_to_label.device).float()
            )
            duplicate_high_masks = (
                prob_to_label.max(dim=-1).values
                >= self.facb_speedhr_infer_duplicate_score_thr
            )
        baseline_scores = prob_to_label.clone()
        for row_idx in range(rows):
            batch_idx = int(batch_ids[row_idx].item())
            query_idx = int(query_ids[row_idx].item())
            if not (
                0 <= batch_idx < prob_to_label.shape[0]
                and 0 <= query_idx < prob_to_label.shape[1]
            ):
                continue
            classifier_winner_slot_i = int(winner_slot[row_idx].item())
            classifier_conf = float(confidence[row_idx].item())
            classifier_margin = float(confidence_margin[row_idx].item())
            # For non-compositional modes these remain the decision confidence.
            # Composition modes overwrite them below with confidence/margin for
            # the slot that is actually selected by the combined score.
            conf = classifier_conf
            margin = classifier_margin

            score_row = baseline_scores[batch_idx, query_idx]
            old_top_position = int(score_row.argmax().item())
            old_top_score = float(score_row[old_top_position].item())
            if (
                self.facb_speedhr_infer_min_semantic_score >= 0.0
                and old_top_score < self.facb_speedhr_infer_min_semantic_score
            ):
                continue
            if (
                self.facb_speedhr_infer_max_semantic_score >= 0.0
                and old_top_score > self.facb_speedhr_infer_max_semantic_score
            ):
                continue
            if duplicate_boxes_xyxy is not None and duplicate_high_masks is not None:
                high_mask = duplicate_high_masks[batch_idx].clone()
                high_mask[query_idx] = False
                high_idx = torch.nonzero(high_mask, as_tuple=False).flatten()
                if high_idx.numel() > 0:
                    current_box = duplicate_boxes_xyxy[
                        batch_idx, query_idx
                    ].view(1, 4)
                    high_boxes = duplicate_boxes_xyxy[batch_idx, high_idx]
                    ious = box_ops.box_iou(current_box, high_boxes)[0]
                    if bool(
                        (ious >= self.facb_speedhr_infer_duplicate_iou_thr)
                        .any()
                        .item()
                    ):
                        dup_skipped += 1
                        continue

            selected_slot_i = int(winner_slot[row_idx].item())
            composition_evidence = 0.0
            candidate_semantic_scores_dump = None
            candidate_composition_scores_dump = None
            candidate_combined_scores_dump = None
            candidate_combined_probs_dump = None
            candidate_facet_logits_dump = None
            candidate_facet_probs_dump = None
            selected_combined_score = None
            selected_combined_prob = None
            selected_combined_margin = None
            if mode in {
                "composition_rerank",
                "semantic_classifier_rerank",
                "composition_residual",
                "composition_direct",
                "composition_symbol_shape",
            }:
                selected_slot_i = -1
                if (
                    candidate_facet_scores is not None
                    and row_idx < candidate_facet_scores.shape[0]
                    and candidate_facet_scores[row_idx].ndim == 2
                ):
                    facet_logits_row = candidate_facet_scores[row_idx].float()
                    num_facets = int(facet_logits_row.shape[-1])
                    candidate_facet_logits_dump = [
                        _round_list(row)
                        for row in facet_logits_row.detach().cpu().tolist()
                    ]
                    valid_slots = []
                    semantic_terms = []
                    candidate_semantic_scores_dump = []
                    for slot_i, global_id in enumerate(
                        candidate_ids[row_idx].detach().cpu().tolist()
                    ):
                        position = global_to_position.get(int(global_id), None)
                        if position is None:
                            candidate_semantic_scores_dump.append(None)
                            continue
                        valid_slots.append(slot_i)
                        candidate_score = score_row[position].float()
                        candidate_semantic_scores_dump.append(
                            round(float(candidate_score.detach().item()), 6)
                        )
                        semantic_terms.append(candidate_score.clamp(min=1e-8).log())
                    if valid_slots and num_facets > 0:
                        head = getattr(self, "facb_speedhr_head", None)
                        if head is not None and hasattr(
                            head, "shared_facet_weight_logits"
                        ):
                            raw_weights = head.shared_facet_weight_logits.detach().float()
                            num_w = min(num_facets, int(raw_weights.numel()))
                            weights = F.softmax(raw_weights[:num_w], dim=0)
                        else:
                            num_w = num_facets
                            weights = facet_logits_row.new_ones(num_w) / max(
                                num_w, 1
                            )
                        if num_w > 0:
                            temp = max(
                                float(self.facb_speedhr_infer_facet_temperature),
                                1e-6,
                            )
                            slot_tensor = torch.as_tensor(
                                valid_slots,
                                device=facet_logits_row.device,
                                dtype=torch.long,
                            )
                            facet_probs = torch.sigmoid(
                                facet_logits_row[slot_tensor, :num_w] / temp
                            ).clamp(min=1e-6, max=1.0)
                            candidate_facet_probs_dump = [
                                None
                                for _ in range(int(candidate_ids.shape[1]))
                            ]
                            valid_probs = facet_probs.detach().cpu().tolist()
                            for dump_slot, prob_row in zip(valid_slots, valid_probs):
                                candidate_facet_probs_dump[dump_slot] = _round_list(
                                    prob_row
                                )
                            if mode == "composition_symbol_shape":
                                group_prob = facet_probs[:, 0]
                                shape_prob = (
                                    facet_probs[:, 1]
                                    if num_w > 1
                                    else torch.ones_like(group_prob)
                                )
                                symbol_prob = (
                                    facet_probs[:, 2]
                                    if num_w > 2
                                    else facet_probs[:, -1]
                                )
                                group_gate = group_prob.clamp(
                                    min=max(
                                        float(
                                            self.facb_speedhr_infer_group_gate_floor
                                        ),
                                        1e-6,
                                    ),
                                    max=1.0,
                                )
                                comp_scores_t = (
                                    symbol_prob
                                    * shape_prob.clamp(min=1e-6).sqrt()
                                    * group_gate
                                ).clamp(min=1e-6, max=1.0)
                                comp_log = comp_scores_t.log()
                            else:
                                comp_log = (
                                    facet_probs.log() * weights[:num_w].to(
                                        device=facet_probs.device
                                    )
                                ).sum(dim=-1)
                            candidate_composition_scores_dump = [
                                None
                                for _ in range(int(candidate_ids.shape[1]))
                            ]
                            comp_scores = comp_log.exp().detach().cpu().tolist()
                            for dump_slot, comp_value in zip(valid_slots, comp_scores):
                                candidate_composition_scores_dump[dump_slot] = round(
                                    float(comp_value), 6
                                )
                            semantic_log = torch.stack(semantic_terms).to(
                                device=comp_log.device
                            )
                            if mode == "composition_symbol_shape":
                                combined = comp_log
                            elif mode == "semantic_classifier_rerank":
                                classifier_log = candidate_probs[
                                    row_idx, slot_tensor
                                ].clamp(min=1e-8).log()
                                combined = (
                                    semantic_log
                                    + float(
                                        self.facb_speedhr_infer_classifier_lambda
                                    )
                                    * classifier_log
                                )
                            else:
                                combined = (
                                    semantic_log
                                    + float(self.facb_speedhr_infer_composition_lambda)
                                    * comp_log
                                )
                            combined_scores = combined.exp()
                            combined_probs = combined.softmax(dim=0)
                            candidate_combined_scores_dump = [
                                None
                                for _ in range(int(candidate_ids.shape[1]))
                            ]
                            candidate_combined_probs_dump = [
                                None
                                for _ in range(int(candidate_ids.shape[1]))
                            ]
                            for dump_slot, combined_score, combined_prob in zip(
                                valid_slots,
                                combined_scores.detach().cpu().tolist(),
                                combined_probs.detach().cpu().tolist(),
                            ):
                                candidate_combined_scores_dump[dump_slot] = round(
                                    float(combined_score), 6
                                )
                                candidate_combined_probs_dump[dump_slot] = round(
                                    float(combined_prob), 6
                                )
                            best_rel = int(combined.argmax().item())
                            selected_slot_i = int(valid_slots[best_rel])
                            selected_combined_score = float(
                                combined_scores[best_rel].detach().item()
                            )
                            selected_combined_prob = float(
                                combined_probs[best_rel].detach().item()
                            )
                            combined_top = combined_probs.topk(
                                k=min(2, int(combined_probs.numel())), dim=0
                            ).values
                            selected_combined_margin = float(
                                (
                                    combined_top[0] - combined_top[1]
                                    if combined_top.numel() > 1
                                    else combined_top[0]
                                )
                                .detach()
                                .item()
                            )
                            composition_evidence = float(
                                comp_log[best_rel].exp().detach().item()
                            )
                if selected_slot_i < 0:
                    continue

                # Gate/inference thresholds must describe the candidate that
                # composition actually selected, not an unrelated classifier
                # argmax.  The old fields are still dumped explicitly below as
                # classifier_conf/classifier_margin for diagnosis.
                if selected_combined_prob is None or selected_combined_margin is None:
                    continue
                conf = selected_combined_prob
                margin = selected_combined_margin

            selector_rejected = False
            if local_probabilities is not None:
                local_row = local_probabilities[row_idx]
                if self.facb_local_selector_candidate_only or p0_bundle_active:
                    local_row = torch.cat([candidate_probs[row_idx],candidate_probs.new_zeros(1)])
                selected_slot_i = int(local_row[:-1].argmax())
                selector_rejected = int(local_row.argmax()) == int(candidate_ids.shape[1])
                conf = float(local_row[selected_slot_i])
                alternatives = local_row.clone()
                alternatives[selected_slot_i] = -1.
                margin = conf-float(alternatives.max())
                # Keep rejected rows in audit; rejection is enforced at writer.
                if selector_rejected:
                    margin = max(margin,0.)
                selected_combined_score = selected_combined_prob = conf
                selected_combined_margin = margin
                candidate_combined_scores_dump = _round_list(local_row[:-1].tolist())
                candidate_combined_probs_dump = candidate_combined_scores_dump
            winner_global = int(candidate_ids[row_idx, selected_slot_i].item())
            if p0_bundle_active:
                old_top_global_for_itm = int(current_global_ids[old_top_position].item())
                selected_itm_logits = itm_logits_all[row_idx, selected_slot_i]
                selected_itm_probs = itm_probabilities_all[row_idx, selected_slot_i]
                itm_in_scope = winner_global in itm_bundle["target_global_ids"]
                itm_changed = winner_global != old_top_global_for_itm
                itm_accepted = bool(
                    itm_in_scope and itm_changed
                    and float(selected_itm_probs[0].item()) >= itm_bundle["threshold"]
                )
                image_id_value = None
                if image_ids is not None and 0 <= batch_idx < len(image_ids):
                    image_id_value = int(image_ids[batch_idx])
                d1_pair = None
                selected_position_for_itm = global_to_position[winner_global]
                old_slot_for_itm = int(torch.nonzero(
                    (candidate_ids[row_idx] == old_top_global_for_itm) & candidate_mask[row_idx],
                    as_tuple=False)[0].item())
                fused_order_for_itm = p0_fused_logits[row_idx].argsort(descending=True, stable=True)
                margin_old_for_itm = float(p0_fused_logits[row_idx, selected_slot_i] - p0_fused_logits[row_idx, old_slot_for_itm])
                margin_top_for_itm = float(p0_fused_logits[row_idx, selected_slot_i] - p0_fused_logits[row_idx, fused_order_for_itm[1]])
                if itm_accepted and itm_bundle["d1"] is not None:
                    d1_pair = facb_d1_score_pair(itm_bundle["d1"],
                        score_row[old_top_position].item(), score_row[selected_position_for_itm].item(),
                        selected_itm_probs.tolist(), margin_old_for_itm, margin_top_for_itm)
                _maybe_dump_dual_facet_itm_audit({
                    "image_id": image_id_value,
                    "batch_index": batch_idx,
                    "query_index": query_idx,
                    "query_box_cxcywh_norm": _round_list(
                        pred_boxes[batch_idx, query_idx].detach().cpu().tolist()
                    ) if isinstance(pred_boxes, torch.Tensor) else None,
                    "candidate_global_ids": [
                        int(x) for x in candidate_ids[row_idx].detach().cpu().tolist()
                    ],
                    "candidate_mask": [
                        bool(x) for x in candidate_mask[row_idx].detach().cpu().tolist()
                    ],
                    "p0_fused_probabilities": _round_list(
                        candidate_probs[row_idx].detach().cpu().tolist()
                    ),
                    "p0_selected_slot": selected_slot_i,
                    "p0_selected_global_id": winner_global,
                    "old_top_global_id": old_top_global_for_itm,
                    "itm_class_logit": float(selected_itm_logits[0].item()),
                    "itm_shape_logit": float(selected_itm_logits[1].item()),
                    "itm_symbol_logit": float(selected_itm_logits[2].item()),
                    "itm_class_probability": float(selected_itm_probs[0].item()),
                    "itm_shape_probability": float(selected_itm_probs[1].item()),
                    "itm_symbol_probability": float(selected_itm_probs[2].item()),
                    "itm_threshold": float(itm_bundle["threshold"]),
                    "itm_in_scope": itm_in_scope,
                    "itm_changed": itm_changed,
                    "itm_accepted": itm_accepted,
                    "old_score": float(score_row[old_top_position].item()),
                    "selected_original_score": float(score_row[selected_position_for_itm].item()),
                    "fused_margin_over_old": margin_old_for_itm,
                    "fused_top1_margin": margin_top_for_itm,
                    "d1_scores": d1_pair,
                })
                if not itm_accepted:
                    continue
            if allowed_ids is not None and allowed_ids.numel() > 0:
                if not bool((allowed_ids == winner_global).any().item()):
                    continue
            if conf < self.facb_speedhr_infer_conf_threshold:
                continue
            if margin < self.facb_speedhr_infer_margin_threshold:
                continue
            winner_position = global_to_position.get(winner_global, None)
            if winner_position is None:
                continue
            selected_classifier_prob = float(
                candidate_probs[row_idx, selected_slot_i].item()
            )
            if int(candidate_probs.shape[1]) > 1:
                other_classifier_probs = torch.cat(
                    [
                        candidate_probs[row_idx, :selected_slot_i],
                        candidate_probs[row_idx, selected_slot_i + 1 :],
                    ],
                    dim=0,
                )
                selected_classifier_margin = float(
                    selected_classifier_prob
                    - float(other_classifier_probs.max().item())
                )
            else:
                selected_classifier_margin = selected_classifier_prob
            classifier_winner_global = int(
                candidate_ids[row_idx, classifier_winner_slot_i].item()
            )
            classifier_composition_agree = (
                classifier_winner_slot_i == selected_slot_i
            )
            query_box_abs = None
            if target_sizes is not None and isinstance(pred_boxes, torch.Tensor):
                target_hw = target_sizes[batch_idx].to(
                    device=prob_to_label.device, dtype=torch.float32
                )
                height = float(target_hw[0].item())
                width = float(target_hw[1].item())
                query_box = box_ops.box_cxcywh_to_xyxy(
                    pred_boxes[batch_idx, query_idx].view(1, 4).to(
                        device=prob_to_label.device, dtype=torch.float32
                    )
                )[0]
                query_box_abs = query_box * query_box.new_tensor(
                    [width, height, width, height]
                )
            replay_index = self._load_facb_speedhr_accept_replay_index()
            if replay_index:
                if image_ids is None or query_box_abs is None:
                    continue
                if not (0 <= batch_idx < len(image_ids)):
                    continue
                image_id_value = int(image_ids[batch_idx])
                if not self._facb_speedhr_accept_replay_match(
                    image_id_value,
                    winner_global,
                    query_box_abs,
                ):
                    continue
            accept_gate_prob = 1.0
            use_mlp_accept_gate = bool(
                str(getattr(self, "facb_speedhr_accept_gate_checkpoint", "") or "")
            )
            use_online_rule_gate = bool(
                getattr(self, "facb_speedhr_online_rule_gate", False)
            )
            if use_mlp_accept_gate or use_online_rule_gate:
                det_top = score_row.float().topk(k=min(2, int(score_row.numel())), dim=0)
                det_top1_score = float(det_top.values[0].item()) if det_top.values.numel() else 0.0
                det_top2_score = float(det_top.values[1].item()) if det_top.values.numel() > 1 else 0.0
                det_top1_margin = det_top1_score - det_top2_score
                det_winner_score = float(score_row[winner_position].item())
                dup_any_iou = 0.0
                dup_any_score = 0.0
                dup_winner_iou = 0.0
                dup_winner_score = 0.0
                box_sqrt_area = 0.0
                box_aspect = 0.0
                if isinstance(pred_boxes, torch.Tensor):
                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
                        pred_boxes[batch_idx].to(
                            device=prob_to_label.device, dtype=torch.float32
                        )
                    )
                    current_box_norm = boxes_xyxy[query_idx].view(1, 4)
                    iou_matrix, _ = box_ops.box_iou(current_box_norm, boxes_xyxy)
                    ious_all = iou_matrix[0]
                    ious_all[query_idx] = -1.0
                    query_scores = baseline_scores[batch_idx].max(dim=-1).values.float()
                    if ious_all.numel() > 0:
                        best_any = int(ious_all.argmax().item())
                        if float(ious_all[best_any].item()) > 0.0:
                            dup_any_iou = float(ious_all[best_any].item())
                            dup_any_score = float(query_scores[best_any].item())
                    top_labels = baseline_scores[batch_idx].argmax(dim=-1)
                    same_winner = (top_labels == winner_position)
                    same_winner[query_idx] = False
                    same_idx = torch.nonzero(same_winner, as_tuple=False).flatten()
                    if same_idx.numel() > 0:
                        same_ious = ious_all[same_idx]
                        best_rel = int(same_ious.argmax().item())
                        best_idx = int(same_idx[best_rel].item())
                        if float(same_ious[best_rel].item()) > 0.0:
                            dup_winner_iou = float(same_ious[best_rel].item())
                            dup_winner_score = float(query_scores[best_idx].item())
                if query_box_abs is not None:
                    bw = max(float((query_box_abs[2] - query_box_abs[0]).item()), 0.0)
                    bh = max(float((query_box_abs[3] - query_box_abs[1]).item()), 0.0)
                    box_sqrt_area = math.sqrt(max(bw * bh, 0.0))
                    box_aspect = bw / max(bh, 1e-6)
                selected_semantic_score = det_winner_score
                selected_composition_score = float(composition_evidence)
                selected_group_prob = 0.0
                selected_shape_prob = 0.0
                selected_symbol_prob = 0.0
                if (
                    candidate_facet_probs_dump is not None
                    and 0 <= selected_slot_i < len(candidate_facet_probs_dump)
                    and candidate_facet_probs_dump[selected_slot_i] is not None
                ):
                    selected_facet_probs = candidate_facet_probs_dump[
                        selected_slot_i
                    ]
                    if len(selected_facet_probs) > 0:
                        selected_group_prob = float(selected_facet_probs[0])
                    if len(selected_facet_probs) > 1:
                        selected_shape_prob = float(selected_facet_probs[1])
                    if len(selected_facet_probs) > 2:
                        selected_symbol_prob = float(selected_facet_probs[2])
                old_top_global_id = (
                    int(current_global_ids[old_top_position].item())
                    if 0 <= old_top_position < int(current_global_ids.numel())
                    else None
                )
                old_candidate_slot = None
                if old_top_global_id is not None:
                    old_slot_matches = torch.nonzero(
                        candidate_ids[row_idx] == old_top_global_id,
                        as_tuple=False,
                    ).flatten()
                    if old_slot_matches.numel() > 0:
                        old_candidate_slot = int(old_slot_matches[0].item())
                old_candidate_classifier_prob = None
                selected_vs_old_classifier_delta = None
                old_combined_score = None
                selected_vs_old_combined_delta = None
                if old_candidate_slot is not None:
                    old_candidate_classifier_prob = float(
                        candidate_probs[row_idx, old_candidate_slot].item()
                    )
                    selected_vs_old_classifier_delta = (
                        selected_classifier_prob - old_candidate_classifier_prob
                    )
                    if (
                        candidate_combined_scores_dump is not None
                        and old_candidate_slot
                        < len(candidate_combined_scores_dump)
                    ):
                        old_combined_score = candidate_combined_scores_dump[
                            old_candidate_slot
                        ]
                        if (
                            old_combined_score is not None
                            and selected_combined_score is not None
                        ):
                            selected_vs_old_combined_delta = (
                                selected_combined_score
                                - float(old_combined_score)
                            )
                valid_k = max(int(candidate_ids.shape[1]) - 1, 1)
                gate_values = {
                    "det_top1_score_repeat": det_top1_score,
                    "det_top1_margin_repeat": det_top1_margin,
                    "facb_conf_repeat": conf,
                    "det_top1_score": det_top1_score,
                    "det_top1_margin": det_top1_margin,
                    "det_winner_score": det_winner_score,
                    "facb_conf": conf,
                    "facb_margin": margin,
                    "score_gap_facb_minus_det": conf - det_winner_score,
                    "classifier_conf": classifier_conf,
                    "classifier_margin": classifier_margin,
                    "selected_classifier_prob": selected_classifier_prob,
                    "selected_classifier_margin": selected_classifier_margin,
                    "selected_combined_score": (
                        selected_combined_score
                        if selected_combined_score is not None
                        else selected_classifier_prob
                    ),
                    "selected_combined_prob": (
                        selected_combined_prob
                        if selected_combined_prob is not None
                        else selected_classifier_prob
                    ),
                    "selected_combined_margin": (
                        selected_combined_margin
                        if selected_combined_margin is not None
                        else selected_classifier_margin
                    ),
                    "selected_vs_old_classifier_delta": (
                        selected_vs_old_classifier_delta
                        if selected_vs_old_classifier_delta is not None
                        else 0.0
                    ),
                    "selected_vs_old_combined_delta": (
                        selected_vs_old_combined_delta
                        if selected_vs_old_combined_delta is not None
                        else 0.0
                    ),
                    "classifier_composition_agree": (
                        1.0 if classifier_composition_agree else 0.0
                    ),
                    "selected_semantic_score": selected_semantic_score,
                    "selected_composition_score": selected_composition_score,
                    "selected_group_prob": selected_group_prob,
                    "selected_shape_prob": selected_shape_prob,
                    "selected_symbol_prob": selected_symbol_prob,
                    "max_dup_iou_any": dup_any_iou,
                    "max_dup_score_any": dup_any_score,
                    "max_dup_iou_winner": dup_winner_iou,
                    "max_dup_score_winner": dup_winner_score,
                    "winner_rank": float(selected_slot_i),
                    "winner_rank_norm": float(selected_slot_i) / float(valid_k),
                    "same_top1": 1.0 if selected_slot_i == 0 else 0.0,
                    "box_sqrt_area": box_sqrt_area,
                    "box_aspect": box_aspect,
                }
                representation = None
                proposal_protocol = dict(infer.get("proposal_protocol", {}), mode=mode,
                                         classifier_lambda=float(self.facb_speedhr_infer_classifier_lambda))
                if isinstance(infer.get("candidate_states"), torch.Tensor):
                    if old_candidate_slot is None:
                        raise ValueError("old detector class is absent from natural candidates")
                    states = infer["candidate_states"][row_idx]
                    representation = pack_features(
                        infer["crop_states"][row_idx], infer["query_states"][row_idx],
                        states[selected_slot_i], states[old_candidate_slot],
                        [det_top1_score, det_winner_score, selected_classifier_prob,
                         old_candidate_classifier_prob, box_sqrt_area / 1000.,
                         box_aspect, dup_any_iou, dup_any_score],
                    )
                    gate_values["_representation"] = representation
                    gate_values["_protocol"] = proposal_protocol
                    gate_values["_candidate_text_states"] = infer["candidate_text_states"][row_idx]
                    gate_values["_facet_logits"] = candidate_facet_scores[row_idx]
                if use_mlp_accept_gate:
                    accept_gate_prob = self._facb_speedhr_accept_gate_score(
                        gate_values,
                        prob_to_label.device,
                    )
                    mlp_gate_accepted = (
                        accept_gate_prob
                        >= float(self.facb_speedhr_accept_gate_threshold)
                    )
                else:
                    mlp_gate_accepted = True

                rule_margin_pass = (
                    margin >= float(self.facb_speedhr_rule_min_facb_margin)
                )
                rule_conf_pass = (
                    conf >= float(self.facb_speedhr_rule_min_facb_conf)
                )
                rule_slot_pass = (
                    selected_slot_i
                    <= int(self.facb_speedhr_rule_max_selected_slot)
                )
                rule_det_winner_pass = (
                    det_winner_score
                    >= float(self.facb_speedhr_rule_min_det_winner_score)
                )
                rule_dup_winner_pass = (
                    dup_winner_iou
                    >= float(self.facb_speedhr_rule_min_dup_winner_iou)
                )
                if use_online_rule_gate:
                    rule_gate_accepted = bool(
                        rule_margin_pass
                        and rule_conf_pass
                        and rule_slot_pass
                        and rule_det_winner_pass
                        and rule_dup_winner_pass
                    )
                else:
                    rule_gate_accepted = True
                gate_accepted = bool(
                    mlp_gate_accepted and rule_gate_accepted and not selector_rejected
                )
                image_id_value = None
                if image_ids is not None and 0 <= batch_idx < len(image_ids):
                    image_id_value = int(image_ids[batch_idx])
                query_box_abs_dump = None
                if query_box_abs is not None:
                    query_box_abs_dump = _round_list(
                        query_box_abs.detach().cpu().tolist()
                    )
                _maybe_dump_accept_gate_audit(
                    {
                        "mode": mode,
                        "representation": representation.detach().cpu().tolist() if representation is not None else None,
                        "proposal_protocol": proposal_protocol,
                        "local_selector_rejected": selector_rejected,
                        "local_selector_probabilities": local_probabilities[row_idx].detach().cpu().tolist() if local_probabilities is not None else None,
                        "candidate_text_states": infer["candidate_text_states"][row_idx].detach().cpu().tolist() if representation is not None else None,
                        "candidate_facet_logits": candidate_facet_scores[row_idx].detach().cpu().tolist() if representation is not None else None,
                        "image_id": image_id_value,
                        "batch_index": batch_idx,
                        "query_index": query_idx,
                        "selected_slot": int(selected_slot_i),
                        "selected_global_id": int(winner_global),
                        "winner_position": int(winner_position),
                        "old_top_position": int(old_top_position),
                        "old_top_global_id": old_top_global_id,
                        "old_top_score": round(float(old_top_score), 6),
                        "winner_score_before": round(
                            float(det_winner_score), 6
                        ),
                        "facb_conf": round(float(conf), 6),
                        "facb_margin": round(float(margin), 6),
                        "classifier_winner_slot": int(classifier_winner_slot_i),
                        "classifier_winner_global_id": int(
                            classifier_winner_global
                        ),
                        "classifier_conf": round(float(classifier_conf), 6),
                        "classifier_margin": round(float(classifier_margin), 6),
                        "classifier_composition_agree": bool(
                            classifier_composition_agree
                        ),
                        "selected_classifier_prob": round(
                            float(selected_classifier_prob), 6
                        ),
                        "selected_classifier_margin": round(
                            float(selected_classifier_margin), 6
                        ),
                        "selected_combined_score": (
                            round(float(selected_combined_score), 6)
                            if selected_combined_score is not None
                            else None
                        ),
                        "selected_combined_prob": (
                            round(float(selected_combined_prob), 6)
                            if selected_combined_prob is not None
                            else None
                        ),
                        "selected_combined_margin": (
                            round(float(selected_combined_margin), 6)
                            if selected_combined_margin is not None
                            else None
                        ),
                        "old_candidate_slot": old_candidate_slot,
                        "old_candidate_classifier_prob": (
                            round(float(old_candidate_classifier_prob), 6)
                            if old_candidate_classifier_prob is not None
                            else None
                        ),
                        "old_combined_score": (
                            round(float(old_combined_score), 6)
                            if old_combined_score is not None
                            else None
                        ),
                        "selected_vs_old_classifier_delta": (
                            round(float(selected_vs_old_classifier_delta), 6)
                            if selected_vs_old_classifier_delta is not None
                            else None
                        ),
                        "selected_vs_old_combined_delta": (
                            round(float(selected_vs_old_combined_delta), 6)
                            if selected_vs_old_combined_delta is not None
                            else None
                        ),
                        "selected_semantic_score": round(
                            float(selected_semantic_score), 6
                        ),
                        "selected_composition_score": round(
                            float(selected_composition_score), 6
                        ),
                        "selected_group_prob": round(
                            float(selected_group_prob), 6
                        ),
                        "selected_shape_prob": round(
                            float(selected_shape_prob), 6
                        ),
                        "selected_symbol_prob": round(
                            float(selected_symbol_prob), 6
                        ),
                        "candidate_global_ids": [
                            int(x)
                            for x in candidate_ids[row_idx]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "candidate_softmax_probs": _round_list(
                            candidate_probs[row_idx].detach().cpu().tolist()
                        ),
                        "candidate_semantic_scores": candidate_semantic_scores_dump,
                        "candidate_composition_scores": (
                            candidate_composition_scores_dump
                        ),
                        "candidate_combined_scores": (
                            candidate_combined_scores_dump
                        ),
                        "candidate_combined_probs": (
                            candidate_combined_probs_dump
                        ),
                        "candidate_facet_probs": candidate_facet_probs_dump,
                        "accept_gate_prob": round(float(accept_gate_prob), 6),
                        "accept_gate_components": {
                            key: round(float(value), 6)
                            for key, value in getattr(
                                self,
                                "_facb_speedhr_accept_gate_last_components",
                                {},
                            ).items()
                        },
                        "accept_gate_threshold": round(
                            float(self.facb_speedhr_accept_gate_threshold), 6
                        ),
                        "gate_accepted": bool(gate_accepted),
                        "accept_gate_type": (
                            "mlp+rule"
                            if use_mlp_accept_gate and use_online_rule_gate
                            else "mlp"
                            if use_mlp_accept_gate
                            else "rule"
                        ),
                        "mlp_gate_accepted": bool(mlp_gate_accepted),
                        "rule_gate_accepted": bool(rule_gate_accepted),
                        "rule_conditions": {
                            "facb_margin": bool(rule_margin_pass),
                            "facb_conf": bool(rule_conf_pass),
                            "selected_slot": bool(rule_slot_pass),
                            "det_winner_score": bool(rule_det_winner_pass),
                            "max_dup_iou_winner": bool(rule_dup_winner_pass),
                        },
                        "query_box_xyxy_abs": query_box_abs_dump,
                        "gate_values": {
                            key: round(float(value), 6)
                            for key, value in gate_values.items() if not key.startswith("_")
                        },
                    }
                )
                if not gate_accepted:
                    continue
            if selector_rejected:
                continue
            passed += 1
            conf_sum += conf
            score_row = prob_to_label[batch_idx, query_idx]
            winner_before = float(score_row[winner_position].item())
            before_sum += winner_before
            if p0_bundle_active and itm_bundle["d1"] is not None:
                if d1_pair is None:
                    raise ValueError("Missing accepted D1 score pair")
                score_row[winner_position] = d1_pair["selected"]
                score_row[old_top_position] = d1_pair["old"]
            elif f1_mode != "off":
                components = getattr(self, "_facb_speedhr_accept_gate_last_components", {})
                if "quality" not in components:
                    raise ValueError("F0/F1 requires a quality-trained gate")
                if old_top_position != winner_position:
                    old_score = baseline_scores[batch_idx,query_idx,old_top_position]
                    previous_selected = baseline_scores[batch_idx,query_idx,winner_position]
                    quality = float(components["quality"])
                    if not (0 <= float(old_score) <= 1 and 0 <= quality <= 1):
                        raise ValueError("F1 expects probability-valued detector/quality scores")
                    abc = [0.,1.,0.] if f1_mode == "f0" else self._facb_f1_mapping["abc"]
                    score_row[winner_position] = fused_score(old_score.new_tensor(quality),old_score,abc)
                    score_row[old_top_position] = previous_selected
                    outputs["_facb_f1_changes"].append(dict(batch=batch_idx,query=query_idx,
                        selected=winner_position,old=old_top_position,quality=quality,
                        old_score=float(old_score),previous_selected=float(previous_selected)))
            elif self.facb_speedhr_use_learned_quality:
                components = getattr(self, "_facb_speedhr_accept_gate_last_components", {})
                if "quality" not in components:
                    raise ValueError("Learned quality writes require a quality-trained gate")
                if old_top_position != winner_position:
                    score_row[winner_position] = float(components["quality"])
                    # Remove the superseded class hypothesis. This is a quality
                    # assignment, not max(old score, quality) or a fixed floor.
                    score_row[old_top_position] = 0.0
            elif mode == "label_only":
                if old_top_position != winner_position:
                    winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = score_row[old_top_position]
                    score_row[old_top_position] = winner_score
            elif mode == "query_rerank":
                if old_top_position != winner_position:
                    target_score = min(
                        max(old_top_score * self.facb_speedhr_infer_score_scale, 0.0),
                        self.facb_speedhr_infer_score_cap,
                    )
                    old_winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = torch.maximum(
                        score_row[winner_position],
                        score_row.new_tensor(target_score),
                    )
                    score_row[old_top_position] = torch.minimum(
                        score_row[old_top_position],
                        old_winner_score,
                    )
            elif mode == "replace":
                replacement = min(
                    max(conf * self.facb_speedhr_infer_score_scale, 0.0),
                    self.facb_speedhr_infer_score_cap,
                )
                score_row[winner_position] = torch.maximum(
                    score_row[winner_position],
                    score_row.new_tensor(replacement),
                )
            elif mode in {
                "composition_rerank",
                "semantic_classifier_rerank",
            }:
                if old_top_position != winner_position:
                    target_score = min(
                        max(old_top_score * self.facb_speedhr_infer_score_scale, 0.0),
                        self.facb_speedhr_infer_score_cap,
                    )
                    old_winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = torch.maximum(
                        score_row[winner_position],
                        score_row.new_tensor(target_score),
                    )
                    score_row[old_top_position] = torch.minimum(
                        score_row[old_top_position],
                        old_winner_score,
                    )
                evidence_sum += composition_evidence
                residual_sum += max(
                    float(score_row[winner_position].item()) - winner_before,
                    0.0,
                )
            elif mode == "composition_residual":
                # Attribute-derived score update.  The localized
                # group/shape/symbol evidence decides both the candidate class
                # and the residual size, while the detector score keeps the
                # query-level reliability.  This avoids fixed-score writes:
                # confident compositional evidence raises low detector scores
                # more, but the update remains bounded by (1 - s).
                if (
                    old_top_position == winner_position
                    and not self.facb_speedhr_infer_raise_same_top
                ):
                    evidence_sum += composition_evidence
                    residual_sum += 0.0
                else:
                    beta = max(float(self.facb_speedhr_infer_score_scale), 0.0)
                    residual = (
                        beta
                        * max(float(composition_evidence), 0.0)
                        * max(1.0 - old_top_score, 0.0)
                    )
                    target_score = min(
                        max(old_top_score + residual, 0.0),
                        self.facb_speedhr_infer_score_cap,
                    )
                    old_winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = torch.maximum(
                        score_row[winner_position],
                        score_row.new_tensor(target_score),
                    )
                    if old_top_position != winner_position:
                        score_row[old_top_position] = torch.minimum(
                            score_row[old_top_position],
                            old_winner_score,
                        )
                    evidence_sum += composition_evidence
                    residual_sum += max(
                        float(score_row[winner_position].item()) - winner_before,
                        0.0,
                    )
            elif mode == "composition_direct":
                # Direct compositional score.  The final score is no longer
                # controlled by a hand-set global beta.  Instead, the localized
                # group/shape/symbol match itself becomes the target confidence:
                #   s_target = geometric_mean_f sigmoid(g_f / T) ^ w_f
                # A low discriminative facet therefore lowers the whole score.
                if (
                    old_top_position == winner_position
                    and not self.facb_speedhr_infer_raise_same_top
                ):
                    evidence_sum += composition_evidence
                    residual_sum += 0.0
                else:
                    target_score = min(
                        max(float(composition_evidence), 0.0),
                        self.facb_speedhr_infer_score_cap,
                    )
                    old_winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = torch.maximum(
                        score_row[winner_position],
                        score_row.new_tensor(target_score),
                    )
                    if old_top_position != winner_position:
                        score_row[old_top_position] = torch.minimum(
                            score_row[old_top_position],
                            old_winner_score,
                        )
                    evidence_sum += composition_evidence
                    residual_sum += max(
                        float(score_row[winner_position].item()) - winner_before,
                        0.0,
                    )
            elif mode == "composition_symbol_shape":
                # Symbol-dominant dictionary-constrained score.  Candidate
                # classes remain constrained to detector top-K, but their score
                # is driven by the most discriminative short attribute:
                #   symbol * sqrt(shape) * max(group, floor).
                # This keeps group as a weak family gate instead of letting its
                # frequent saturation dominate the final confidence.
                if (
                    old_top_position == winner_position
                    and not self.facb_speedhr_infer_raise_same_top
                ):
                    evidence_sum += composition_evidence
                    residual_sum += 0.0
                else:
                    target_score = min(
                        max(float(composition_evidence), 0.0),
                        self.facb_speedhr_infer_score_cap,
                    )
                    old_winner_score = score_row[winner_position].clone()
                    score_row[winner_position] = torch.maximum(
                        score_row[winner_position],
                        score_row.new_tensor(target_score),
                    )
                    if old_top_position != winner_position:
                        score_row[old_top_position] = torch.minimum(
                            score_row[old_top_position],
                            old_winner_score,
                        )
                    evidence_sum += composition_evidence
                    residual_sum += max(
                        float(score_row[winner_position].item()) - winner_before,
                        0.0,
                    )
            elif mode == "compositional":
                # Compositional similarity score: convert the winner's three
                # short-attribute similarity logits (group/shape/symbol) into a
                # calibrated confidence weighted by the learned facet prior, and
                # fuse with the semantic score.  This is the paper-facing
                # replacement for both the constant-score hack and the learned
                # scalar residual: every term has physical meaning and can be
                # ablated per attribute.
                head = getattr(self, "facb_speedhr_head", None)
                comp_score = 0.0
                winner_slot_i = int(winner_slot[row_idx].item())
                if (
                    candidate_facet_scores is not None
                    and row_idx < candidate_facet_scores.shape[0]
                    and winner_slot_i < candidate_facet_scores.shape[1]
                ):
                    facet_logits = candidate_facet_scores[row_idx, winner_slot_i]
                    num_facets = int(facet_logits.numel())
                    if num_facets <= 0:
                        weights = None
                    elif head is not None and hasattr(
                        head, "shared_facet_weight_logits"
                    ):
                        raw_weights = head.shared_facet_weight_logits.detach().float()
                        num_w = min(num_facets, int(raw_weights.numel()))
                        if num_w > 0:
                            weights = F.softmax(raw_weights[:num_w], dim=0)
                        else:
                            weights = facet_logits.new_ones(num_facets) / max(
                                num_facets, 1
                            )
                    else:
                        weights = facet_logits.new_ones(num_facets) / max(
                            num_facets, 1
                        )
                    if weights is not None and weights.numel() > 0:
                        temp = max(
                            float(self.facb_speedhr_infer_facet_temperature), 1e-6
                        )
                        num_w = min(num_facets, int(weights.numel()))
                        probs = torch.sigmoid(facet_logits[:num_w] / temp)
                        comp_score = float(
                            (probs * weights[:num_w]).sum().item()
                        )
                target = min(
                    max(comp_score * self.facb_speedhr_infer_score_scale, 0.0),
                    self.facb_speedhr_infer_score_cap,
                )
                score_row[winner_position] = torch.maximum(
                    score_row[winner_position],
                    score_row.new_tensor(target),
                )
                evidence_sum += comp_score
            else:
                head = getattr(self, "facb_speedhr_head", None)
                alpha_value = 0.0
                if head is not None and hasattr(head, "score_residual_alpha"):
                    alpha_value = float(head.score_residual_alpha().detach().item())
                alpha_cap = float(
                    getattr(self, "facb_speedhr_score_residual_alpha_cap", -1.0)
                )
                if alpha_cap > 0.0:
                    alpha_value = min(alpha_value, alpha_cap)
                alpha_value = max(alpha_value, 0.0)
                evidence = float(
                    torch.sigmoid(candidate_logits[row_idx, winner_slot[row_idx]])
                    .detach()
                    .item()
                )
                gate_weight = 1.0
                if self.facb_speedhr_infer_residual_soft_gate:
                    # Differentiable acceptance gate: how far the winner clears
                    # the confidence/margin thresholds, mapped through sigmoids.
                    # gate_weight -> 1 when evidence is clearly above threshold,
                    # -> 0 when below, and is smooth in between, so the residual
                    # ``alpha * gate_weight * evidence`` is evidence-calibrated
                    # rather than an all-or-nothing constant bump.
                    temp = max(
                        float(self.facb_speedhr_infer_residual_gate_temperature),
                        1e-6,
                    )
                    conf_gate = 1.0 / (
                        1.0
                        + math.exp(
                            -(conf - self.facb_speedhr_infer_conf_threshold) / temp
                        )
                    )
                    margin_gate = 1.0 / (
                        1.0
                        + math.exp(
                            -(margin - self.facb_speedhr_infer_margin_threshold) / temp
                        )
                    )
                    gate_weight = conf_gate * margin_gate
                residual = max(
                    alpha_value
                    * gate_weight
                    * evidence
                    * self.facb_speedhr_infer_score_scale,
                    0.0,
                )
                alpha_sum += alpha_value
                evidence_sum += evidence
                residual_sum += residual
                score_row[winner_position] = torch.minimum(
                    score_row[winner_position] + score_row.new_tensor(residual),
                    score_row.new_tensor(self.facb_speedhr_infer_score_cap),
                )
            after_score = float(score_row[winner_position].item())
            after_sum += after_score
            changed += int(
                old_top_position != int(score_row.argmax().item())
                or abs(after_score - winner_before) > 1e-12
            )
            if dump_path and mode in {
                "compositional",
                "composition_rerank",
                "semantic_classifier_rerank",
                "composition_residual",
                "composition_direct",
                "composition_symbol_shape",
            }:
                old_top_global = None
                current_global_cpu = current_global_ids.detach().cpu().tolist()
                if 0 <= old_top_position < len(current_global_cpu):
                    old_top_global = int(current_global_cpu[old_top_position])
                after_top_position = int(score_row.argmax().item())
                after_top_global = None
                if 0 <= after_top_position < len(current_global_cpu):
                    after_top_global = int(current_global_cpu[after_top_position])
                image_id = None
                if image_ids is not None and 0 <= batch_idx < len(image_ids):
                    image_id = int(image_ids[batch_idx])
                pred_box_cxcywh = None
                pred_box_xyxy = None
                if (
                    isinstance(pred_boxes, torch.Tensor)
                    and pred_boxes.ndim == 3
                    and 0 <= batch_idx < pred_boxes.shape[0]
                    and 0 <= query_idx < pred_boxes.shape[1]
                ):
                    pred_box = (
                        pred_boxes[batch_idx, query_idx]
                        .detach()
                        .float()
                        .cpu()
                    )
                    pred_box_cxcywh = _round_list(pred_box.tolist())
                    pred_box_xyxy = _round_list(
                        box_ops.box_cxcywh_to_xyxy(pred_box.view(1, 4))[0]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                _maybe_dump_facb_record(
                    {
                        "mode": mode,
                        "image_id": image_id,
                        "batch_index": batch_idx,
                        "query_index": query_idx,
                        "pred_box_cxcywh_norm": pred_box_cxcywh,
                        "pred_box_xyxy_norm": pred_box_xyxy,
                        "candidate_global_ids": [
                            int(x)
                            for x in candidate_ids[row_idx]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "selected_slot": int(selected_slot_i),
                        "selected_global_id": int(winner_global),
                        "old_top_position": int(old_top_position),
                        "old_top_global_id": old_top_global,
                        "after_top_position": int(after_top_position),
                        "after_top_global_id": after_top_global,
                        "old_top_score": round(float(old_top_score), 6),
                        "winner_score_before": round(float(winner_before), 6),
                        "winner_score_after": round(float(after_score), 6),
                        "score_delta": round(float(after_score - winner_before), 6),
                        "facb_conf": round(float(conf), 6),
                        "facb_margin": round(float(margin), 6),
                        "classifier_winner_slot": int(classifier_winner_slot_i),
                        "classifier_winner_global_id": int(
                            classifier_winner_global
                        ),
                        "classifier_conf": round(float(classifier_conf), 6),
                        "classifier_margin": round(float(classifier_margin), 6),
                        "classifier_composition_agree": bool(
                            classifier_composition_agree
                        ),
                        "selected_classifier_prob": round(
                            float(selected_classifier_prob), 6
                        ),
                        "selected_classifier_margin": round(
                            float(selected_classifier_margin), 6
                        ),
                        "selected_combined_score": (
                            round(float(selected_combined_score), 6)
                            if selected_combined_score is not None
                            else None
                        ),
                        "selected_combined_prob": (
                            round(float(selected_combined_prob), 6)
                            if selected_combined_prob is not None
                            else None
                        ),
                        "selected_combined_margin": (
                            round(float(selected_combined_margin), 6)
                            if selected_combined_margin is not None
                            else None
                        ),
                        "accept_gate_prob": round(float(accept_gate_prob), 6),
                        "accept_gate_threshold": round(
                            float(self.facb_speedhr_accept_gate_threshold), 6
                        ),
                        "candidate_softmax_probs": _round_list(
                            candidate_probs[row_idx].detach().cpu().tolist()
                        ),
                        "candidate_semantic_scores": candidate_semantic_scores_dump,
                        "candidate_composition_scores": (
                            candidate_composition_scores_dump
                        ),
                        "candidate_combined_scores": (
                            candidate_combined_scores_dump
                        ),
                        "candidate_combined_probs": (
                            candidate_combined_probs_dump
                        ),
                        "candidate_facet_logits": candidate_facet_logits_dump,
                        "candidate_facet_probs": candidate_facet_probs_dump,
                        "s_comp": round(float(composition_evidence), 6),
                        "score_scale_beta": round(
                            float(self.facb_speedhr_infer_score_scale), 6
                        ),
                        "score_cap": round(
                            float(self.facb_speedhr_infer_score_cap), 6
                        ),
                    }
                )

        self._facb_speedhr_infer_batches += 1
        self._facb_speedhr_infer_rows += rows
        self._facb_speedhr_infer_passed += passed
        self._facb_speedhr_infer_changed += changed
        self._facb_speedhr_infer_dup_skipped += dup_skipped
        self._facb_speedhr_infer_conf_sum += conf_sum
        self._facb_speedhr_infer_before_sum += before_sum
        self._facb_speedhr_infer_after_sum += after_sum
        self._facb_speedhr_infer_alpha_sum += alpha_sum
        self._facb_speedhr_infer_evidence_sum += evidence_sum
        self._facb_speedhr_infer_residual_sum += residual_sum
        debug_enabled = (
            os.environ.get("GROUNDINGDINO_FACB_SPEEDHR_INFER_DEBUG", "0") == "1"
        )
        if debug_enabled and (
            self._facb_speedhr_infer_batches == 1
            or self._facb_speedhr_infer_batches % 100 == 0
        ):
            total_passed = max(self._facb_speedhr_infer_passed, 1)
            print(
                "[facb-speedhr][infer] "
                f"mode={mode} batches={self._facb_speedhr_infer_batches} "
                f"rows={self._facb_speedhr_infer_rows} "
                f"passed={self._facb_speedhr_infer_passed} "
                f"changed={self._facb_speedhr_infer_changed} "
                f"dup_skipped={self._facb_speedhr_infer_dup_skipped} "
                f"conf_mean={self._facb_speedhr_infer_conf_sum / total_passed:.4f} "
                f"winner_score_before_mean="
                f"{self._facb_speedhr_infer_before_sum / total_passed:.6f} "
                f"winner_score_after_mean="
                f"{self._facb_speedhr_infer_after_sum / total_passed:.6f} "
                f"alpha_mean="
                f"{self._facb_speedhr_infer_alpha_sum / total_passed:.6f} "
                f"evidence_mean="
                f"{self._facb_speedhr_infer_evidence_sum / total_passed:.6f} "
                f"residual_mean="
                f"{self._facb_speedhr_infer_residual_sum / total_passed:.6f}",
                flush=True,
            )
        return prob_to_label

    def _build_positive_map(self, cat_list):
        caption, normalized_cat_list = build_caption_from_labels(cat_list)
        tokenized = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        label_list = torch.arange(len(normalized_cat_list))
        binary_map = create_positive_map(
            tokenized,
            label_list,
            normalized_cat_list,
            caption,
            max_text_len=self.max_text_len,
        )
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_positive_map_debug_printed", False)
        ):
            print(
                "[dgcl-flow][PostProcess] positive_map(binary).shape={}".format(
                    tuple(binary_map.shape)
                ),
                flush=True,
            )
            for row_idx in range(min(2, int(binary_map.shape[0]))):
                row_nz = (binary_map[row_idx] > 0).nonzero(as_tuple=False).flatten()
                print(
                    "[dgcl-flow][PostProcess] positive_map(binary)[{}] positive_tokens={}".format(
                        row_idx, row_nz[:30].detach().cpu().tolist()
                    ),
                    flush=True,
                )
            self._dgcl_flow_positive_map_debug_printed = True
        # Method A: λ=0 OR no DF table -> return the binary membership map; the
        # caller (forward) normalizes by row-sum, giving the exact uniform-mean
        # baseline.  This branch guarantees λ=0 reproduces baseline bit-for-bit.
        lam = float(self.postprocess_token_idf_lambda)
        if lam == 0.0 or self._global_token_df is None:
            return binary_map
        # Shared with the matcher cost (idf_weight_label_map) so eval scoring and
        # training matcher use identical weighting math.
        return idf_weight_label_map(
            binary_map,
            tokenized,
            self._global_token_df,
            self._global_token_df_num_classes,
            lam,
            self.postprocess_token_idf_gamma,
        )

    def set_global_token_df(self, full_cat_list):
        """Precompute token-id document frequency over the FULL class set.

        Delegates to the shared module-level compute_token_df_from_cat_list so eval
        (this) and training (SetCriterion) use the SAME DF code path.  Keyed by token
        id; computed in the same chunked tokenization used at eval.
        """
        df, N = compute_token_df_from_cat_list(
            self.tokenizer, list(full_cat_list), self.max_text_len
        )
        self._global_token_df = df
        self._global_token_df_num_classes = N
        # Rebuild the current positive_map under the new weighting.
        self.positive_map = self._build_positive_map(self.cat_list)
        return df

    def _resolve_appearance_seen_global_ids(self, args):
        if args is None:
            return None

        for attr_name in ("appearance_seen_global_ids", "seen_category_ids"):
            seen_ids = getattr(args, attr_name, None)
            if seen_ids:
                return {int(cat_id) for cat_id in seen_ids}

        seen_meta_path = getattr(args, "seen_category_meta_path", None)
        if seen_meta_path:
            seen_ids = self._extract_seen_ids_from_split_meta(seen_meta_path)
            if seen_ids:
                return seen_ids

        datasets_path = getattr(args, "datasets", None)
        if not datasets_path:
            return None
        manifest_path = Path(datasets_path)
        if not manifest_path.is_absolute():
            manifest_path = (Path.cwd() / manifest_path).resolve()
        if not manifest_path.exists():
            return None

        try:
            with manifest_path.open("r", encoding="utf-8-sig") as handle:
                dataset_meta = json.load(handle)
        except Exception:
            return None

        seen_ids = set()
        for entry in dataset_meta.get("train", []) or []:
            ann_path = entry.get("anno")
            if not ann_path:
                continue
            ann_path = Path(ann_path)
            if not ann_path.is_absolute():
                ann_path = (manifest_path.parent / ann_path).resolve()
            seen_ids.update(self._extract_seen_ids_from_annotation(ann_path))
        return seen_ids or None

    @staticmethod
    def _extract_seen_ids_from_split_meta(split_meta_path):
        split_meta_path = Path(split_meta_path)
        if not split_meta_path.exists():
            return set()
        try:
            with split_meta_path.open("r", encoding="utf-8-sig") as handle:
                split_meta = json.load(handle)
        except Exception:
            return set()
        seen_ids = split_meta.get("seen_orig_ids")
        if seen_ids is not None:
            return {int(cat_id) for cat_id in seen_ids}
        extracted = set()
        for row in split_meta.get("class_table", []) or []:
            if row.get("split") != "seen":
                continue
            if row.get("orig_id") is not None:
                extracted.add(int(row["orig_id"]))
            elif row.get("new_id") is not None:
                extracted.add(int(row["new_id"]))
        return extracted

    @staticmethod
    def _extract_seen_ids_from_annotation(ann_path):
        ann_path = Path(ann_path)
        if not ann_path.exists():
            return set()
        try:
            if ann_path.suffix.lower() == ".jsonl":
                seen_ids = set()
                with ann_path.open("r", encoding="utf-8-sig") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        for instance in sample.get("detection", {}).get("instances", []):
                            label = instance.get("label")
                            if label is not None:
                                seen_ids.add(int(label))
                return seen_ids

            if ann_path.suffix.lower() == ".json":
                with ann_path.open("r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                return {
                    int(annotation["category_id"])
                    for annotation in data.get("annotations", []) or []
                    if annotation.get("category_id") is not None
                }
        except Exception:
            return set()
        return set()

    def _compute_uncertainty_strength(self, semantic_row):
        if semantic_row.numel() == 0:
            return 0.0, 0.0, 1.0, 0.0
        topk = min(2, semantic_row.shape[0])
        top_vals = semantic_row.topk(k=topk, dim=0).values
        semantic_max = float(top_vals[0].item())
        semantic_margin = float((top_vals[0] - top_vals[1]).item()) if topk >= 2 else semantic_max
        semantic_prob = semantic_row / semantic_row.sum().clamp(min=1e-6)
        semantic_entropy = float(
            (-(semantic_prob * semantic_prob.clamp(min=1e-6).log()).sum() / math.log(max(int(semantic_prob.numel()), 2)))
            .item()
        )

        low_conf = max(0.0, self.appearance_rerank_threshold - semantic_max) / max(
            self.appearance_rerank_threshold, 1e-6
        )
        low_margin = max(0.0, self.appearance_rerank_margin_threshold - semantic_margin) / max(
            self.appearance_rerank_margin_threshold, 1e-6
        )
        high_entropy = max(0.0, semantic_entropy - self.appearance_rerank_entropy_threshold) / max(
            1.0 - self.appearance_rerank_entropy_threshold, 1e-6
        )
        uncertainty_strength = max(low_conf, low_margin, high_entropy)
        return semantic_max, semantic_margin, semantic_entropy, float(max(0.0, min(1.0, uncertainty_strength)))

    def _compute_querywise_fusion_alpha(self, semantic_row, app_probs):
        _, _, _, uncertainty_strength = self._compute_uncertainty_strength(semantic_row)
        if uncertainty_strength <= 0.0:
            return 0.0
        if not self.use_dynamic_appearance_fusion_alpha:
            return self.appearance_fusion_alpha
        if app_probs.numel() == 0:
            return 0.0
        app_topk = min(2, app_probs.shape[0])
        app_top_vals = app_probs.topk(k=app_topk, dim=0).values
        app_margin = float((app_top_vals[0] - app_top_vals[1]).item()) if app_topk >= 2 else float(app_top_vals[0].item())
        app_confidence = max(0.0, min(1.0, app_margin))
        return float(self.appearance_fusion_alpha * uncertainty_strength * app_confidence)

    def _aggregate_with_dgcl_token_gate(self, prob_to_token, pos_maps, outputs):
        token_gate_list = outputs.get("pred_logits_dgcl_token_gate", None)
        if not isinstance(token_gate_list, list):
            return None
        device = prob_to_token.device
        dtype = prob_to_token.dtype
        batch_size, num_queries, num_tokens = prob_to_token.shape
        num_classes = pos_maps.shape[0]
        weighted = prob_to_token.new_zeros((batch_size, num_queries, num_classes))
        class_token_indices = [
            (pos_maps[class_idx, :num_tokens] > 0).nonzero(as_tuple=False).flatten()
            for class_idx in range(num_classes)
        ]
        mapped_batches = 0
        for batch_idx in range(min(batch_size, len(token_gate_list))):
            token_gate = token_gate_list[batch_idx]
            if not isinstance(token_gate, torch.Tensor):
                continue
            token_gate = token_gate.to(device=device, dtype=dtype)
            gate_queries = min(num_queries, token_gate.shape[0])
            gate_tokens = min(num_tokens, token_gate.shape[-1])
            if gate_queries <= 0 or gate_tokens <= 0:
                continue
            sample_prob = prob_to_token[batch_idx, :gate_queries, :gate_tokens]
            sample_gate = token_gate[:gate_queries, :gate_tokens].clamp_min(0.0)
            for class_idx, token_indices in enumerate(class_token_indices):
                if token_indices.numel() == 0:
                    continue
                valid_indices = token_indices[token_indices < gate_tokens]
                if valid_indices.numel() == 0:
                    continue
                phrase_gate = sample_gate[:, valid_indices]
                denom = phrase_gate.sum(dim=-1, keepdim=True)
                uniform = pos_maps[class_idx, valid_indices].to(
                    device=device, dtype=dtype
                )
                uniform = uniform / uniform.sum().clamp_min(1e-6)
                phrase_weights = torch.where(
                    denom > 1e-6,
                    phrase_gate / denom.clamp_min(1e-6),
                    uniform.unsqueeze(0).expand_as(phrase_gate),
                )
                weighted[
                    batch_idx, :gate_queries, class_idx
                ] = (
                    sample_prob[:, valid_indices] * phrase_weights
                ).sum(dim=-1)
            mapped_batches += 1
        if mapped_batches == 0:
            return None
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_weighted_posmap_debug_printed", False)
        ):
            print(
                "[dgcl-flow][WeightedPositiveMap] mapped={} mix={:.4f} "
                "score_mean/std={:.6f}/{:.6f}".format(
                    mapped_batches,
                    float(self.dgcl_weighted_positive_map_mix),
                    float(weighted.detach().float().mean().cpu()),
                    float(weighted.detach().float().std(unbiased=False).cpu()),
                ),
                flush=True,
            )
            self._dgcl_weighted_posmap_debug_printed = True
        return weighted

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']


        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        pos_maps = pos_maps.clone()
        pos_map_denominator = pos_maps.sum(dim=1, keepdim=True)
        valid_pos_rows = pos_map_denominator.squeeze(-1) > 0
        pos_maps[valid_pos_rows] = pos_maps[valid_pos_rows] / pos_map_denominator[valid_pos_rows]

        uniform_prob_to_label = prob_to_token @ pos_maps.T
        prob_to_label = uniform_prob_to_label
        if self.use_dgcl_weighted_positive_map:
            weighted_prob_to_label = self._aggregate_with_dgcl_token_gate(
                prob_to_token, pos_maps, outputs
            )
            if weighted_prob_to_label is not None:
                mix = max(0.0, min(1.0, float(self.dgcl_weighted_positive_map_mix)))
                if (
                    os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
                    and not getattr(self, "_dgcl_weighted_posmap_delta_debug_printed", False)
                ):
                    weighted_delta = (weighted_prob_to_label - uniform_prob_to_label).abs()
                    weighted_changed = (
                        weighted_prob_to_label.argmax(dim=-1)
                        != uniform_prob_to_label.argmax(dim=-1)
                    ).float()
                    print(
                        "[dgcl-flow][WeightedPositiveMapDelta] "
                        "delta_mean/max={:.6f}/{:.6f} category_top1_changed={:.4f}".format(
                            float(weighted_delta.detach().float().mean().cpu()),
                            float(weighted_delta.detach().float().max().cpu()),
                            float(weighted_changed.detach().float().mean().cpu()),
                        ),
                        flush=True,
                    )
                    self._dgcl_weighted_posmap_delta_debug_printed = True
                prob_to_label = (
                    (1.0 - mix) * uniform_prob_to_label
                    + mix * weighted_prob_to_label
                )
        if (
            self.use_dgcl_category_residual
            and isinstance(outputs.get("pred_logits_dgcl_category_raw", None), list)
            and isinstance(outputs.get("pred_logits_baseline", None), torch.Tensor)
        ):
            baseline_prob_to_label = (
                outputs["pred_logits_baseline"]
                .to(device=out_logits.device, dtype=out_logits.dtype)
                .sigmoid()
                @ pos_maps.T
            )
            dgcl_category_scores = prob_to_label.new_zeros(prob_to_label.shape)
            dgcl_category_logits = prob_to_label.new_zeros(prob_to_label.shape)
            dgcl_category_valid = torch.zeros(
                prob_to_label.shape,
                device=prob_to_label.device,
                dtype=torch.bool,
            )
            category_logits_list = outputs["pred_logits_dgcl_category_raw"]
            label_ids = self.label_ids.to(
                device=prob_to_label.device, dtype=torch.long
            )
            mapped_batches = 0
            skipped_batches = 0
            for batch_idx, category_logits in enumerate(category_logits_list):
                if batch_idx >= dgcl_category_scores.shape[0]:
                    break
                if not isinstance(category_logits, torch.Tensor):
                    skipped_batches += 1
                    continue
                category_logits = category_logits.to(
                    device=prob_to_label.device, dtype=prob_to_label.dtype
                )
                if category_logits.shape[-1] != label_ids.numel():
                    skipped_batches += 1
                    continue
                num_local = int(category_logits.shape[-1])
                if num_local <= 0:
                    skipped_batches += 1
                    continue
                global_ids = label_ids[:num_local]
                valid_global = (
                    (global_ids >= 0)
                    & (global_ids < dgcl_category_scores.shape[-1])
                )
                if not valid_global.any():
                    skipped_batches += 1
                    continue
                local_positions = torch.arange(
                    num_local, device=prob_to_label.device
                )[valid_global]
                global_positions = global_ids[valid_global]
                num_queries = min(
                    category_logits.shape[0], dgcl_category_scores.shape[1]
                )
                dgcl_category_scores[
                    batch_idx, :num_queries, global_positions
                ] = category_logits[:num_queries, local_positions].sigmoid()
                dgcl_category_logits[
                    batch_idx, :num_queries, global_positions
                ] = category_logits[:num_queries, local_positions]
                dgcl_category_valid[
                    batch_idx, :num_queries, global_positions
                ] = True
                mapped_batches += 1
            residual_weight = float(self.dgcl_category_residual_weight)
            query_gate = None
            if mapped_batches > 0:
                if self.use_dgcl_category_residual_query_gate:
                    base_detection_score = (
                        outputs["pred_logits_baseline"]
                        .to(device=out_logits.device, dtype=out_logits.dtype)
                        .sigmoid()
                        .max(dim=-1)
                        .values
                    )
                    gate_topk = max(
                        1,
                        min(
                            int(self.dgcl_category_residual_gate_topk),
                            int(base_detection_score.shape[-1]),
                        ),
                    )
                    topk_threshold = base_detection_score.topk(
                        k=gate_topk, dim=-1
                    ).values[..., -1:]
                    in_topk = base_detection_score >= topk_threshold
                    top2 = baseline_prob_to_label.topk(
                        k=min(2, baseline_prob_to_label.shape[-1]), dim=-1
                    ).values
                    if top2.shape[-1] >= 2:
                        semantic_margin = top2[..., 0] - top2[..., 1]
                    else:
                        semantic_margin = top2[..., 0].new_zeros(top2.shape[:-1])
                    if self.dgcl_category_residual_gate_margin_tau >= 0.0:
                        margin_tau = semantic_margin.new_tensor(
                            float(self.dgcl_category_residual_gate_margin_tau)
                        )
                    else:
                        margin_quantile = min(
                            max(
                                float(
                                    self.dgcl_category_residual_gate_margin_quantile
                                ),
                                0.0,
                            ),
                            1.0,
                        )
                        selected_margins = semantic_margin[in_topk]
                        if selected_margins.numel() > 0:
                            margin_tau = selected_margins.float().quantile(
                                margin_quantile
                            ).to(device=semantic_margin.device, dtype=semantic_margin.dtype)
                        else:
                            margin_tau = semantic_margin.float().quantile(
                                margin_quantile
                            ).to(device=semantic_margin.device, dtype=semantic_margin.dtype)
                    query_gate = (in_topk & (semantic_margin <= margin_tau)).to(
                        device=prob_to_label.device, dtype=prob_to_label.dtype
                    )
                if self.dgcl_category_residual_logit_space:
                    eps = 1e-6
                    base_prob = baseline_prob_to_label.clamp(min=eps, max=1.0 - eps)
                    base_logits = torch.logit(base_prob)
                    calibrated_logits = dgcl_category_logits
                    if self.dgcl_category_residual_match_baseline_distribution:
                        calibrated_logits = dgcl_category_logits.clone()
                        for batch_idx in range(calibrated_logits.shape[0]):
                            valid = dgcl_category_valid[batch_idx]
                            if not valid.any():
                                continue
                            src = dgcl_category_logits[batch_idx][valid]
                            ref = base_logits[batch_idx][valid]
                            src_std = src.float().std(unbiased=False).clamp_min(1e-6)
                            ref_std = ref.float().std(unbiased=False).clamp_min(1e-6)
                            calibrated_logits[batch_idx][valid] = (
                                (src - src.float().mean().to(src.dtype))
                                / src_std.to(src.dtype)
                                * ref_std.to(src.dtype)
                                + ref.float().mean().to(src.dtype)
                            )
                    final_logits = base_logits.clone()
                    residual = calibrated_logits - base_logits.detach()
                    if query_gate is not None:
                        residual = residual * query_gate.unsqueeze(-1)
                    final_logits[dgcl_category_valid] = (
                        base_logits[dgcl_category_valid]
                        + residual_weight * residual[dgcl_category_valid]
                    )
                    prob_to_label = final_logits.sigmoid().clamp(min=0.0, max=1.0)
                else:
                    residual = dgcl_category_scores - baseline_prob_to_label
                    if query_gate is not None:
                        residual = residual * query_gate.unsqueeze(-1)
                    prob_to_label = (
                        prob_to_label
                        + residual_weight * residual
                    ).clamp(min=0.0, max=1.0)
            if (
                os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
                and not getattr(self, "_dgcl_category_residual_debug_printed", False)
            ):
                changed = (
                    prob_to_label.argmax(dim=-1)
                    != baseline_prob_to_label.argmax(dim=-1)
                ).float()
                delta = (prob_to_label - baseline_prob_to_label).abs()
                print(
                    "[dgcl-flow][CategoryResidual] weight={:.4f} logit_space={} "
                    "match_dist={} gate={} mapped={} skipped={} "
                    "delta_mean/max={:.6f}/{:.6f} category_top1_changed={:.4f}".format(
                        residual_weight,
                        self.dgcl_category_residual_logit_space,
                        self.dgcl_category_residual_match_baseline_distribution,
                        (
                            "off"
                            if query_gate is None
                            else "mean={:.4f}".format(
                                float(query_gate.detach().float().mean().cpu())
                            )
                        ),
                        mapped_batches,
                        skipped_batches,
                        float(delta.mean().detach().float().cpu()),
                        float(delta.max().detach().float().cpu()),
                        float(changed.mean().detach().cpu()),
                    ),
                    flush=True,
                )
                self._dgcl_category_residual_debug_printed = True
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_postprocess_debug_printed", False)
        ):
            print(
                "[dgcl-flow][PostProcess] out_logits.shape={} prob_to_token.shape={}".format(
                    tuple(out_logits.shape), tuple(prob_to_token.shape)
                ),
                flush=True,
            )
            print(
                "[dgcl-flow][PostProcess] positive_map(normalized).shape={} prob_to_label.shape={}".format(
                    tuple(pos_maps.shape), tuple(prob_to_label.shape)
                ),
                flush=True,
            )
            for row_idx in range(min(2, int(pos_maps.shape[0]))):
                row_nz = (pos_maps[row_idx] > 0).nonzero(as_tuple=False).flatten()
                weights = pos_maps[row_idx, row_nz[:30]].detach().cpu().tolist()
                print(
                    "[dgcl-flow][PostProcess] positive_map(normalized)[{}] tokens={} weights={}".format(
                        row_idx,
                        row_nz[:30].detach().cpu().tolist(),
                        [round(float(w), 6) for w in weights],
                    ),
                    flush=True,
                )
            self._dgcl_flow_postprocess_debug_printed = True
        crm_category_residual = outputs.get(
            "self_facb_crm_category_residual", None
        )
        if isinstance(crm_category_residual, torch.Tensor):
            crm_category_residual = crm_category_residual.to(
                device=prob_to_label.device, dtype=prob_to_label.dtype
            )
            if crm_category_residual.shape == prob_to_label.shape:
                crm_eps = 1e-5
                crm_base = prob_to_label.clamp(
                    min=crm_eps, max=1.0 - crm_eps
                )
                prob_to_label = torch.sigmoid(
                    torch.logit(crm_base) + crm_category_residual
                )
        semantic_prob_to_label = prob_to_label.clone()
        if (
            self.use_appearance_prototype_fusion
            and "appearance_class_scores" in outputs
            and isinstance(outputs["appearance_class_scores"], torch.Tensor)
        ):
            # ---- Prototype-bank fusion (global-aligned) ----
            # appearance_class_scores: [bs, nq, C], column i == global class id i,
            # directly aligned with prob_to_label's class axis.  For each query that
            # the semantic head is uncertain about, take its semantic top-k classes,
            # softmax the appearance scores over exactly those classes, and apply the
            # neutral multiplicative update.  Uniform appearance → multiplier ≈ 1 →
            # baseline untouched.  Never writes into pred_logits_semantic.
            app_scores = outputs["appearance_class_scores"].to(
                device=prob_to_label.device, dtype=prob_to_label.dtype
            )
            app_scores = torch.nan_to_num(
                app_scores, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
            bs_a, nq_a, c_a = app_scores.shape
            nq = min(nq_a, prob_to_label.shape[1])
            label_ids = self.label_ids.to(device=prob_to_label.device, dtype=torch.long)
            c_common = min(label_ids.numel(), prob_to_label.shape[2])
            if c_common <= 0:
                pass
            label_ids = label_ids[:c_common]
            valid_bank_cols = (label_ids >= 0) & (label_ids < c_a)
            if self.appearance_global_fine_group_ids is not None:
                group_ids = self.appearance_global_fine_group_ids.to(
                    device=prob_to_label.device, dtype=torch.long
                )
            else:
                group_ids = None
            for batch_idx in range(min(bs_a, prob_to_label.shape[0])):
                semantic_query_scores = semantic_prob_to_label[batch_idx, :nq, :c_common]
                for query_idx in range(nq):
                    base_row = semantic_query_scores[query_idx]
                    topk = min(self.appearance_rerank_topk, base_row.shape[0])
                    if topk <= 1:
                        continue
                    semantic_top = base_row.topk(k=topk, dim=0)
                    top_positions = semantic_top.indices
                    if top_positions.numel() < 2:
                        continue
                    semantic_top1_local = int(top_positions[0].item())
                    semantic_top1_global = int(label_ids[semantic_top1_local].item())
                    if self.use_appearance_seen_only_fusion:
                        if (
                            not self.appearance_seen_global_ids
                            or semantic_top1_global not in self.appearance_seen_global_ids
                        ):
                            continue
                    if self.appearance_fusion_max_semantic_margin >= 0.0:
                        semantic_margin = float(
                            (semantic_top.values[0] - semantic_top.values[1]).item()
                        )
                        if semantic_margin > self.appearance_fusion_max_semantic_margin:
                            continue
                    top_positions = top_positions[valid_bank_cols[top_positions]]
                    if top_positions.numel() < 2:
                        continue
                    if self.use_appearance_same_group_fusion and group_ids is not None:
                        top1_bank_idx = int(label_ids[top_positions[0]].item())
                        if 0 <= top1_bank_idx < group_ids.numel():
                            top1_group = group_ids[top1_bank_idx]
                            cand_bank_ids = label_ids[top_positions].clamp(
                                min=0, max=max(c_a - 1, 0)
                            )
                            same_group_mask = group_ids[cand_bank_ids] == top1_group
                            top_positions = top_positions[same_group_mask]
                            if top_positions.numel() < 2:
                                continue
                    app_bank_positions = label_ids[top_positions].clamp(
                        min=0, max=max(c_a - 1, 0)
                    )
                    app_logits = app_scores[batch_idx, query_idx, app_bank_positions]
                    app_probs = app_logits.softmax(dim=-1)
                    semantic_local_row = base_row[top_positions]
                    query_alpha = self._compute_querywise_fusion_alpha(
                        semantic_local_row, app_probs
                    )
                    if query_alpha <= 0.0:
                        continue
                    if self.use_appearance_neutral_fusion:
                        K = int(top_positions.numel())
                        multiplier = (app_probs * K).clamp(
                            min=self.appearance_fusion_clamp_min,
                            max=self.appearance_fusion_clamp_max,
                        ) ** query_alpha
                        prob_to_label[batch_idx, query_idx, top_positions] *= multiplier
                    else:
                        prob_to_label[batch_idx, query_idx, top_positions] = (
                            prob_to_label[batch_idx, query_idx, top_positions]
                            + (
                                self.appearance_rerank_weight
                                * query_alpha
                                / max(self.appearance_fusion_alpha, 1e-6)
                            )
                            * app_probs
                        )
        if self.use_appearance_rescoring and "appearance_class_scores" in outputs:
            appearance_class_scores = outputs["appearance_class_scores"]
            if isinstance(appearance_class_scores, list):
                num_queries = prob_to_label.shape[1]
                for batch_idx, score in enumerate(appearance_class_scores):
                    if not isinstance(score, torch.Tensor) or score.numel() == 0:
                        continue
                    score = score.to(device=prob_to_label.device, dtype=prob_to_label.dtype)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    nq = min(score.shape[0], num_queries)
                    if nq == 0:
                        continue
                    if self.appearance_rescoring_mode == "margin" and score.shape[-1] >= 2:
                        top2 = score[:nq].topk(k=2, dim=-1).values
                        quality_logit = top2[:, 0] - top2[:, 1]
                    else:
                        quality_logit = score[:nq].max(dim=-1).values
                    quality = (
                        quality_logit / self.appearance_rescoring_temperature
                        + self.appearance_rescoring_bias
                    ).sigmoid()
                    quality = torch.nan_to_num(quality, nan=0.5)
                    prob_to_label[batch_idx, :nq] = prob_to_label[batch_idx, :nq] * quality.unsqueeze(-1)
        if self.use_appearance_proto_rerank and "appearance_class_scores" in outputs:
            appearance_class_scores = outputs["appearance_class_scores"]
            appearance_caption_label_ids = outputs.get("appearance_caption_label_ids", None)
            if isinstance(appearance_class_scores, list) and appearance_caption_label_ids is not None:
                num_queries = prob_to_label.shape[1]
                num_global_classes = prob_to_label.shape[2]
                for batch_idx, score in enumerate(appearance_class_scores):
                    if (
                        not isinstance(score, torch.Tensor)
                        or score.numel() == 0
                        or batch_idx >= len(appearance_caption_label_ids)
                    ):
                        continue
                    local_label_ids = appearance_caption_label_ids[batch_idx]
                    if local_label_ids is None:
                        continue
                    score = score.to(device=prob_to_label.device, dtype=prob_to_label.dtype)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    nq = min(score.shape[0], num_queries)
                    if nq == 0:
                        continue
                    global_score = prob_to_label.new_zeros((nq, num_global_classes))
                    for local_idx, label_id in enumerate(local_label_ids):
                        if local_idx >= score.shape[1]:
                            break
                        try:
                            global_idx = int(label_id)
                        except (TypeError, ValueError):
                            continue
                        if global_idx < 0 or global_idx >= num_global_classes:
                            continue
                        global_score[:, global_idx] = score[:nq, local_idx]
                    if self.appearance_proto_rerank_mode == "sigmoid":
                        global_score = global_score.sigmoid()
                    else:
                        topk = min(8, global_score.shape[-1])
                        if topk > 0:
                            topk_vals, topk_inds = global_score.topk(k=topk, dim=-1)
                            sparse_score = global_score.new_zeros(global_score.shape)
                            sparse_score.scatter_(1, topk_inds, topk_vals)
                            global_score = sparse_score.softmax(dim=-1)
                    prob_to_label[batch_idx, :nq] = (
                        prob_to_label[batch_idx, :nq]
                        + self.appearance_proto_rerank_weight * global_score
                    )
        if self.use_appearance_uncertainty_rerank and "appearance_class_scores" in outputs:
            appearance_class_scores = outputs["appearance_class_scores"]
            appearance_caption_label_ids = outputs.get("appearance_caption_label_ids", None)
            if isinstance(appearance_class_scores, list) and appearance_caption_label_ids is not None:
                num_queries = prob_to_label.shape[1]
                num_global_classes = prob_to_label.shape[2]
                for batch_idx, score in enumerate(appearance_class_scores):
                    if (
                        not isinstance(score, torch.Tensor)
                        or score.numel() == 0
                        or batch_idx >= len(appearance_caption_label_ids)
                    ):
                        continue
                    local_label_ids = appearance_caption_label_ids[batch_idx]
                    if local_label_ids is None:
                        continue
                    score = score.to(device=prob_to_label.device, dtype=prob_to_label.dtype)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    nq = min(score.shape[0], num_queries)
                    if nq == 0:
                        continue

                    global_to_local = {}
                    for local_idx, label_id in enumerate(local_label_ids):
                        if local_idx >= score.shape[1]:
                            break
                        try:
                            global_idx = int(label_id)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= global_idx < num_global_classes:
                            global_to_local[global_idx] = local_idx
                    if not global_to_local:
                        continue

                    semantic_query_scores = semantic_prob_to_label[batch_idx, :nq]

                    for query_idx in range(nq):
                        base_row = semantic_query_scores[query_idx]
                        topk = min(self.appearance_rerank_topk, base_row.shape[0])
                        if topk <= 0:
                            continue
                        top_positions = base_row.topk(k=topk, dim=0).indices.tolist()
                        valid_global_positions = []
                        valid_local_positions = []
                        for global_pos in top_positions:
                            local_pos = global_to_local.get(int(global_pos), None)
                            if local_pos is None or local_pos >= score.shape[1]:
                                continue
                            valid_global_positions.append(int(global_pos))
                            valid_local_positions.append(int(local_pos))
                        if len(valid_global_positions) < 2:
                            continue

                        app_logits = score[
                            query_idx,
                            torch.as_tensor(
                                valid_local_positions,
                                device=score.device,
                                dtype=torch.long,
                            ),
                        ]
                        app_probs = app_logits.softmax(dim=-1)
                        semantic_local_row = base_row[
                            torch.as_tensor(
                                valid_global_positions,
                                device=base_row.device,
                                dtype=torch.long,
                            )
                        ]
                        query_alpha = self._compute_querywise_fusion_alpha(
                            semantic_local_row, app_probs
                        )
                        if query_alpha <= 0.0:
                            continue

                        if self.use_appearance_neutral_fusion:
                            # Neutral multiplicative fusion:
                            #   multiplier = ((app_probs * K).clamp(0.2, 3.0)) ** alpha
                            # When app is uniform (1/K): multiplier ≈ 1.0 → neutral.
                            # When app supports a class: multiplier > 1 → enhanced.
                            # When app disagrees: multiplier < 1 → suppressed.
                            K = len(valid_global_positions)
                            multiplier = (app_probs * K).clamp(
                                min=self.appearance_fusion_clamp_min,
                                max=self.appearance_fusion_clamp_max,
                            ) ** query_alpha
                            global_pos_tensor = torch.as_tensor(
                                valid_global_positions,
                                device=prob_to_label.device,
                                dtype=torch.long,
                            )
                            prob_to_label[batch_idx, query_idx, global_pos_tensor] *= multiplier
                        else:
                            # Legacy additive rerank (kept for backward compatibility).
                            prob_to_label[
                                batch_idx, query_idx,
                                torch.as_tensor(valid_global_positions, device=prob_to_label.device, dtype=torch.long),
                            ] = (
                                prob_to_label[
                                    batch_idx, query_idx,
                                    torch.as_tensor(valid_global_positions, device=prob_to_label.device, dtype=torch.long),
                                ]
                                + (self.appearance_rerank_weight * query_alpha / max(self.appearance_fusion_alpha, 1e-6)) * app_probs
                            )
        if (
            self.use_appearance_local_evidence_rerank
            and outputs.get("appearance_is_local_evidence", False)
            and "appearance_local_evidence_logits" in outputs
            and "appearance_local_candidate_ids" in outputs
        ):
            debug_enabled = os.environ.get("GROUNDINGDINO_FACB_DEBUG", "0") == "1"
            trace_limit = int(os.environ.get("GROUNDINGDINO_FACB_TRACE_LIMIT", "0") or 0)
            trace_records = []
            debug_valid_queries = 0
            debug_reranked_queries = 0
            debug_total_abs_delta = 0.0
            debug_max_abs_delta = 0.0
            local_logits = outputs["appearance_local_evidence_logits"]
            local_candidate_ids = outputs["appearance_local_candidate_ids"]
            local_candidate_mask = outputs.get("appearance_local_candidate_mask", None)
            label_ids = self.label_ids.to(device=prob_to_label.device, dtype=torch.long)
            global_to_position = {
                int(global_id.item()): int(pos)
                for pos, global_id in enumerate(label_ids)
            }

            for batch_idx in range(min(prob_to_label.shape[0], local_logits.shape[0])):
                nq = min(prob_to_label.shape[1], local_logits.shape[1])
                semantic_query_scores = semantic_prob_to_label[batch_idx, :nq]
                for query_idx in range(nq):
                    ev_row = local_logits[batch_idx, query_idx].to(
                        device=prob_to_label.device, dtype=prob_to_label.dtype
                    )
                    cid_row = local_candidate_ids[batch_idx, query_idx].to(
                        device=prob_to_label.device, dtype=torch.long
                    )
                    if local_candidate_mask is not None:
                        mask_row = local_candidate_mask[batch_idx, query_idx].to(
                            device=prob_to_label.device, dtype=torch.bool
                        )
                    else:
                        mask_row = cid_row >= 0
                    valid = mask_row & torch.isfinite(ev_row) & (cid_row >= 0)
                    if valid.sum().item() < 2:
                        continue
                    debug_valid_queries += 1

                    valid_global_ids = [int(x) for x in cid_row[valid].detach().cpu().tolist()]
                    valid_positions = [
                        global_to_position[gid]
                        for gid in valid_global_ids
                        if gid in global_to_position
                    ]
                    if len(valid_positions) < 2:
                        continue

                    position_tensor = torch.as_tensor(
                        valid_positions,
                        device=prob_to_label.device,
                        dtype=torch.long,
                    )
                    score_values = []
                    for gid, score in zip(valid_global_ids, ev_row[valid]):
                        if gid in global_to_position:
                            score_values.append(score)
                    if len(score_values) != len(valid_positions):
                        continue
                    app_logits = torch.stack(score_values).to(dtype=prob_to_label.dtype)
                    app_probs = app_logits.softmax(dim=-1)
                    semantic_local_row = semantic_query_scores[query_idx, position_tensor]
                    if semantic_local_row.numel() >= 2:
                        sem_top2 = semantic_local_row.topk(k=2, dim=0).values
                        sem_top1_score = float(sem_top2[0].detach().item())
                        sem_margin = float((sem_top2[0] - sem_top2[1]).detach().item())
                    elif semantic_local_row.numel() == 1:
                        sem_top1_score = float(semantic_local_row[0].detach().item())
                        sem_margin = float("inf")
                    else:
                        continue
                    if sem_top1_score < self.appearance_local_evidence_min_semantic_score:
                        continue
                    if (
                        self.appearance_local_evidence_max_semantic_score >= 0.0
                        and sem_top1_score > self.appearance_local_evidence_max_semantic_score
                    ):
                        continue
                    if (
                        self.appearance_local_evidence_max_semantic_margin >= 0.0
                        and sem_margin > self.appearance_local_evidence_max_semantic_margin
                    ):
                        continue

                    before = prob_to_label[batch_idx, query_idx, position_tensor].clone()
                    query_alpha = 1.0
                    if self.use_appearance_local_evidence_gated_replace:
                        # Gated replace: when semantic is uncertain (top-1 not
                        # significantly ahead of top-2), replace top-1 with FACB's
                        # best pick.  Uses ratio so it works for both high- and
                        # low-score queries.
                        sem_sorted = before.sort(descending=True)
                        sem_ratio = float(sem_sorted.values[0] / sem_sorted.values[1].clamp(min=1e-8)) if before.numel() >= 2 else 1.0
                        # gate_sem_margin is interpreted as ratio threshold, e.g. 1.2
                        gate_ratio = max(1.0, float(self.appearance_local_evidence_gate_sem_margin))
                        facb_best_local = int(app_probs.argmax().item())
                        if sem_ratio < gate_ratio:
                            # Swap: FACB best goes to top, old top-1 goes to FACB's slot
                            old_top1_pos = int(before.argmax().item())
                            if facb_best_local != old_top1_pos:
                                old_top1_val = before[old_top1_pos].clone()
                                old_facb_val = before[facb_best_local].clone()
                                prob_to_label[batch_idx, query_idx, position_tensor[old_top1_pos]] = old_facb_val
                                prob_to_label[batch_idx, query_idx, position_tensor[facb_best_local]] = old_top1_val
                    elif self.use_appearance_local_evidence_zscore_fusion:
                        eps = 1e-6
                        sem = semantic_local_row.to(dtype=prob_to_label.dtype)
                        app = app_logits.to(dtype=prob_to_label.dtype)
                        sem_z = (sem - sem.mean()) / (sem.std(unbiased=False) + eps)
                        app_z = (app - app.mean()) / (app.std(unbiased=False) + eps)
                        fused = sem_z + self.appearance_local_evidence_zscore_lambda * app_z

                        if self.use_appearance_local_evidence_winner_take_top1:
                            old_top1_pos = int(before.argmax().item())
                            fused_top1_pos = int(fused.argmax().item())
                            reranked_values = before.clone()
                            if fused_top1_pos != old_top1_pos:
                                old_top1_val = before[old_top1_pos].clone()
                                old_fused_val = before[fused_top1_pos].clone()
                                reranked_values[old_top1_pos] = old_fused_val
                                reranked_values[fused_top1_pos] = old_top1_val
                            prob_to_label[batch_idx, query_idx, position_tensor] = reranked_values
                        else:
                            sem_sorted_values = before.sort(descending=True).values
                            fused_order = fused.argsort(descending=True)
                            reranked_values = before.clone()
                            reranked_values[fused_order] = sem_sorted_values
                            prob_to_label[batch_idx, query_idx, position_tensor] = reranked_values
                    else:
                        query_alpha = self._compute_querywise_fusion_alpha(
                            semantic_local_row, app_probs
                        )
                        if query_alpha <= 0.0:
                            continue
                    if (not self.use_appearance_local_evidence_zscore_fusion) and (not self.use_appearance_local_evidence_gated_replace) and self.use_appearance_neutral_fusion:
                        K = int(position_tensor.numel())
                        multiplier = (app_probs * K).clamp(
                            min=self.appearance_fusion_clamp_min,
                            max=self.appearance_fusion_clamp_max,
                        ) ** query_alpha
                        prob_to_label[batch_idx, query_idx, position_tensor] *= multiplier
                    elif (not self.use_appearance_local_evidence_zscore_fusion) and (not self.use_appearance_local_evidence_gated_replace):
                        prob_to_label[batch_idx, query_idx, position_tensor] = (
                            prob_to_label[batch_idx, query_idx, position_tensor]
                            + (
                                self.appearance_rerank_weight
                                * query_alpha
                                / max(self.appearance_fusion_alpha, 1e-6)
                            )
                            * app_probs
                        )
                    after = prob_to_label[batch_idx, query_idx, position_tensor]
                    delta = (after - before).abs()
                    debug_reranked_queries += 1
                    debug_total_abs_delta += float(delta.sum().item())
                    debug_max_abs_delta = max(debug_max_abs_delta, float(delta.max().item()))
                    if trace_limit > 0 and len(trace_records) < trace_limit:
                        trace_records.append(
                            {
                                "batch": int(batch_idx),
                                "query": int(query_idx),
                                "candidate_global_ids": [int(x) for x in valid_global_ids],
                                "class_positions": [int(x) for x in valid_positions],
                                "semantic_before": [
                                    round(float(x), 6) for x in before.detach().float().cpu().tolist()
                                ],
                                "appearance_logits": [
                                    round(float(x), 6) for x in app_logits.detach().float().cpu().tolist()
                                ],
                                "appearance_probs": [
                                    round(float(x), 6) for x in app_probs.detach().float().cpu().tolist()
                                ],
                                "semantic_after": [
                                    round(float(x), 6) for x in after.detach().float().cpu().tolist()
                                ],
                                "abs_delta": [
                                    round(float(x), 6) for x in delta.detach().float().cpu().tolist()
                                ],
                                "query_alpha": round(float(query_alpha), 6),
                                "semantic_top1_score": round(float(sem_top1_score), 6),
                                "semantic_margin": round(float(sem_margin), 6),
                                "selected_before_global_id": int(valid_global_ids[int(before.argmax().item())]),
                                "selected_after_global_id": int(valid_global_ids[int(after.argmax().item())]),
                                "appearance_best_global_id": int(valid_global_ids[int(app_probs.argmax().item())]),
                            }
                        )
            if debug_enabled and not getattr(self, "_facb_rerank_debug_printed", False):
                print(
                    "[facb-debug][postprocess] "
                    f"enabled=True valid_queries={debug_valid_queries} "
                    f"reranked_queries={debug_reranked_queries} "
                    f"total_abs_delta={debug_total_abs_delta:.6f} "
                    f"max_abs_delta={debug_max_abs_delta:.6f} "
                    f"neutral_fusion={self.use_appearance_neutral_fusion}",
                    flush=True,
                )
                self._facb_rerank_debug_printed = True
            if trace_limit > 0 and not getattr(self, "_facb_rerank_trace_printed", False):
                for record in trace_records:
                    print(
                        "[facb-trace][postprocess] " + json.dumps(record, ensure_ascii=False),
                        flush=True,
                    )
                self._facb_rerank_trace_printed = True
        if self.use_appearance_text_rerank and "appearance_text_logits" in outputs:
            appearance_text_logits = outputs["appearance_text_logits"]
            if isinstance(appearance_text_logits, torch.Tensor):
                num_queries = prob_to_label.shape[1]
                num_tokens = prob_to_label.shape[2]
                for batch_idx, score in enumerate(appearance_text_logits):
                    if not isinstance(score, torch.Tensor) or score.numel() == 0:
                        continue
                    score = score.to(device=prob_to_label.device, dtype=prob_to_label.dtype)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    nq = min(score.shape[0], num_queries)
                    nt = min(score.shape[1], num_tokens)
                    if nq == 0 or nt == 0:
                        continue
                    text_score = score[:nq, :nt].sigmoid()
                    text_score = F.pad(text_score, (0, num_tokens - nt)) if nt < num_tokens else text_score
                    prob_to_label[batch_idx, :nq] = (
                        prob_to_label[batch_idx, :nq]
                        + self.appearance_text_rerank_weight * text_score
                    )
        if self.use_appearance_class_fusion and "appearance_class_scores" in outputs:
            appearance_class_scores = outputs["appearance_class_scores"]
            if isinstance(appearance_class_scores, list):
                fused_appearance = []
                for batch_idx, score in enumerate(appearance_class_scores):
                    if not isinstance(score, torch.Tensor) or score.numel() == 0:
                        fused_appearance.append(prob_to_label.new_zeros(prob_to_label.shape[1], prob_to_label.shape[2]))
                        continue
                    score = score.to(device=prob_to_label.device, dtype=prob_to_label.dtype)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    score_prob = score.sigmoid()
                    if score_prob.shape[-1] < prob_to_label.shape[-1]:
                        padded = prob_to_label.new_zeros(prob_to_label.shape[1], prob_to_label.shape[2])
                        padded[:, :score_prob.shape[-1]] = score_prob
                        score_prob = padded
                    elif score_prob.shape[-1] > prob_to_label.shape[-1]:
                        score_prob = score_prob[:, :prob_to_label.shape[-1]]
                    fused_appearance.append(score_prob)
                appearance_prob = torch.stack(fused_appearance, dim=0)
                prob_to_label = prob_to_label + self.appearance_class_fusion_weight * appearance_prob

        if 'pred_quality' in outputs:
            quality = outputs['pred_quality'].sigmoid().squeeze(-1)  # [bs, num_queries]
            prob_to_label = prob_to_label * quality.unsqueeze(-1)

        if self.use_visual_confidence and 'pred_visual_confidence' in outputs:
            visual_confidence = outputs['pred_visual_confidence'].sigmoid().squeeze(-1)
            semantic_score = prob_to_label.max(dim=-1).values
            ambiguity = (1.0 - semantic_score).clamp(0.0, 1.0)
            query_gate = 1.0 + self.visual_confidence_lambda * visual_confidence * ambiguity
            prob_to_label = prob_to_label * query_gate.unsqueeze(-1)

        if self.use_appearance_conditioned_gate and 'pred_appearance_gate_scores' in outputs:
            app_scores = outputs['pred_appearance_gate_scores']  # [bs, nq, num_c]
            app_compat = app_scores.sigmoid()  # [0, 1] per-class compatibility
            if app_compat.shape[-1] > prob_to_label.shape[-1]:
                app_compat = app_compat[..., :prob_to_label.shape[-1]]
            elif app_compat.shape[-1] < prob_to_label.shape[-1]:
                padded = app_compat.new_zeros(*app_compat.shape[:-1], prob_to_label.shape[-1])
                padded[..., :app_compat.shape[-1]] = app_compat
                app_compat = padded
            semantic_score = prob_to_label.max(dim=-1).values  # [bs, nq]
            ambiguity = (1.0 - semantic_score).clamp(0.0, 1.0)  # [bs, nq]
            class_gate = 1.0 + self.appearance_gate_lambda * app_compat * ambiguity.unsqueeze(-1)
            prob_to_label = prob_to_label * class_gate

        if self.use_appearance_conditioned_gate and 'pred_appearance_gate_scores' in outputs and self.appearance_gate_additive_weight > 0.0:
            app_scores = outputs['pred_appearance_gate_scores']
            app_compat = app_scores.sigmoid()
            if app_compat.shape[-1] > prob_to_label.shape[-1]:
                app_compat = app_compat[..., :prob_to_label.shape[-1]]
            elif app_compat.shape[-1] < prob_to_label.shape[-1]:
                padded = app_compat.new_zeros(*app_compat.shape[:-1], prob_to_label.shape[-1])
                padded[..., :app_compat.shape[-1]] = app_compat
                app_compat = padded
            prob_to_label = prob_to_label + self.appearance_gate_additive_weight * app_compat

        if self.use_postprocess_score_norm:
            if self.postprocess_score_norm_type == "zscore":
                mean = prob_to_label.mean(dim=(1, 2), keepdim=True)
                std = prob_to_label.std(dim=(1, 2), keepdim=True, unbiased=False)
                prob_to_label = (prob_to_label - mean) / (std + self.postprocess_score_norm_eps)
            elif self.postprocess_score_norm_type == "query_zscore":
                mean = prob_to_label.mean(dim=-1, keepdim=True)
                std = prob_to_label.std(dim=-1, keepdim=True, unbiased=False)
                prob_to_label = (prob_to_label - mean) / (std + self.postprocess_score_norm_eps)
            elif self.postprocess_score_norm_type == "image_minmax":
                minv = prob_to_label.amin(dim=(1, 2), keepdim=True)
                maxv = prob_to_label.amax(dim=(1, 2), keepdim=True)
                prob_to_label = (prob_to_label - minv) / (maxv - minv + self.postprocess_score_norm_eps)
            elif self.postprocess_score_norm_type == "query_score_zscore":
                query_score = prob_to_label.max(dim=-1, keepdim=True).values
                mean = query_score.mean(dim=1, keepdim=True)
                std = query_score.std(dim=1, keepdim=True, unbiased=False)
                query_weight = ((query_score - mean) / (std + self.postprocess_score_norm_eps)).sigmoid()
                prob_to_label = prob_to_label * query_weight

        deferred_facb_reader = outputs.pop("_facb_speedhr_deferred_reader", None)
        if deferred_facb_reader is not None:
            if not callable(deferred_facb_reader):
                raise TypeError("Deferred FACB reader must be callable")
            if outputs.get("facb_speedhr_inference", None) is not None:
                raise RuntimeError("FACB inference was produced both before and after postprocess scoring")
            facb_infer = deferred_facb_reader(prob_to_label.detach())
            if facb_infer is not None:
                outputs["facb_speedhr_inference"] = facb_infer

        if self.facb_score_replay_capture_dir:
            if self.use_lgpr_survival_rescue or not_to_xyxy:
                raise ValueError("Complete FACB replay supports standard top-k/xyxy postprocessing only")
            from util.facb_detection_replay import capture_pre_topk
            capture_pre_topk(
                self.facb_score_replay_capture_dir, prob_to_label, out_bbox,
                self.label_ids, self.current_global_label_ids, target_sizes, outputs,
                self.num_select, self.nms_iou_threshold, self.facb_score_replay_provenance,
            )

        fixed_records = None
        if self.facb_fixed_quality_checkpoint:
            if self.use_lgpr_survival_rescue or not_to_xyxy:
                raise ValueError("Fixed quality requires standard top-k/xyxy")
            from util.facb_detection_replay import SCHEMA
            fixed_records = {int(outputs["image_ids"][b]): dict(
                schema=SCHEMA, image_id=int(outputs["image_ids"][b]),
                scores=prob_to_label[b].detach().cpu().clone(), boxes=out_bbox[b].detach().cpu().clone(),
                labels=self.label_ids.detach().cpu(), global_ids=self.current_global_label_ids.detach().cpu(),
                size=target_sizes[b].detach().cpu(), num_select=self.num_select,
                nms=self.nms_iou_threshold, score_device=str(prob_to_label.device), provenance="live")
                for b in range(len(prob_to_label))}
            outputs["_facb_fixed_quality_rows"] = []
        prob_to_label = self._apply_facb_speedhr_inference(
            prob_to_label, outputs, target_sizes=target_sizes
        )
        if fixed_records is not None:
            from util.facb_fixed_quality_online import apply as apply_fixed_quality, validate as validate_fixed_quality
            if self._facb_fixed_quality_saved is None:
                self._facb_fixed_quality_saved = validate_fixed_quality(torch.load(
                    self.facb_fixed_quality_checkpoint, map_location="cpu", weights_only=False))
            bundle = self._load_facb_dual_facet_itm(prob_to_label.device)
            if self._facb_fixed_quality_saved["threshold"] != bundle["threshold"]:
                raise ValueError("Fixed quality threshold differs from P0/ITM bundle")
            prob_to_label = apply_fixed_quality(self._facb_fixed_quality_saved, fixed_records,
                outputs.pop("_facb_fixed_quality_rows"), bundle["target_global_ids"], prob_to_label)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        if getattr(self, "facb_f1_capture_dir", ""):
            if self.use_lgpr_survival_rescue or not_to_xyxy:
                raise ValueError("F1 capture supports standard top-k/xyxy postprocessing only")
            capture_chunk(self.facb_f1_capture_dir,self._facb_f1_index,
                prob, out_bbox, self.label_ids, target_sizes, outputs,
                outputs.get("_facb_f1_changes",[]),self.num_select,self.nms_iou_threshold,
                self._facb_f1_gate_sha,self.facb_f1_reader_id)
            self._facb_f1_index += 1
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_postprocess_compare_printed", False)
            and isinstance(outputs.get("pred_logits_baseline", None), torch.Tensor)
        ):
            baseline_logits = outputs["pred_logits_baseline"].to(
                device=out_logits.device, dtype=out_logits.dtype
            )
            if baseline_logits.shape[-1] > out_logits.shape[-1]:
                baseline_logits = baseline_logits[..., : out_logits.shape[-1]]
            elif baseline_logits.shape[-1] < out_logits.shape[-1]:
                baseline_logits = F.pad(
                    baseline_logits,
                    (0, out_logits.shape[-1] - baseline_logits.shape[-1]),
                    value=float("-inf"),
                )
            baseline_prob_to_label = baseline_logits.sigmoid() @ pos_maps.T
            common_classes = min(
                baseline_prob_to_label.shape[-1], prob_to_label.shape[-1]
            )
            common_queries = min(
                baseline_prob_to_label.shape[1], prob_to_label.shape[1]
            )
            if common_classes > 0 and common_queries > 0:
                baseline_cmp = baseline_prob_to_label[
                    :, :common_queries, :common_classes
                ]
                dgcl_cmp = prob_to_label[:, :common_queries, :common_classes]
                category_delta = (dgcl_cmp - baseline_cmp).abs()
                category_top1_changed = (
                    dgcl_cmp.argmax(dim=-1) != baseline_cmp.argmax(dim=-1)
                ).float()
                flat_count = min(
                    int(self.num_select),
                    int(common_queries * common_classes),
                )
                baseline_top = torch.topk(
                    baseline_cmp.reshape(baseline_cmp.shape[0], -1),
                    flat_count,
                    dim=1,
                ).indices
                dgcl_top = torch.topk(
                    dgcl_cmp.reshape(dgcl_cmp.shape[0], -1),
                    flat_count,
                    dim=1,
                ).indices
                topk_overlap = []
                for batch_idx in range(baseline_top.shape[0]):
                    baseline_set = set(
                        int(v) for v in baseline_top[batch_idx].detach().cpu().tolist()
                    )
                    dgcl_set = set(
                        int(v) for v in dgcl_top[batch_idx].detach().cpu().tolist()
                    )
                    topk_overlap.append(
                        len(baseline_set & dgcl_set) / max(len(baseline_set), 1)
                    )
                print(
                    "[dgcl-flow][PostProcessCompare] category_delta_mean/max="
                    "{:.6f}/{:.6f} category_top1_changed={:.4f} "
                    "top{}_overlap={:.4f}".format(
                        float(category_delta.mean().detach().float().cpu()),
                        float(category_delta.max().detach().float().cpu()),
                        float(category_top1_changed.mean().detach().cpu()),
                        flat_count,
                        float(sum(topk_overlap) / max(len(topk_overlap), 1)),
                    ),
                    flush=True,
                )
                self._dgcl_flow_postprocess_compare_printed = True
        use_survival_rescue = (
            self.use_lgpr_survival_rescue
            and self.lgpr_survival_rescue_num > 0
            and isinstance(outputs.get("pred_lgpr_survival", None), torch.Tensor)
        )
        if use_survival_rescue:
            protected_num = self.lgpr_survival_protected_num
            rescue_num = self.lgpr_survival_rescue_num
            flat_prob = prob.view(prob.shape[0], -1)
            survival_logits = outputs["pred_lgpr_survival"].squeeze(-1).to(
                device=prob.device, dtype=prob.dtype
            )
            baseline_values, baseline_indexes = torch.topk(
                flat_prob, num_select, dim=1
            )
            protected_values = baseline_values[:, :protected_num]
            protected_indexes = baseline_indexes[:, :protected_num]
            query_class_values, query_class_positions = prob.max(dim=-1)
            query_survival_values = query_class_values * torch.exp(
                self.lgpr_survival_score_weight * torch.tanh(survival_logits)
            )
            combined_values = []
            combined_indexes = []
            num_classes = prob.shape[-1]
            for batch_idx in range(prob.shape[0]):
                baseline_query_ids = torch.div(
                    baseline_indexes[batch_idx],
                    num_classes,
                    rounding_mode="trunc",
                )
                protected_query_ids = torch.div(
                    protected_indexes[batch_idx],
                    num_classes,
                    rounding_mode="trunc",
                )
                rescue_scores = query_survival_values[batch_idx].clone()
                # Exclude only the protected top queries. Candidates can come
                # from baseline ranks 91..num_select or from outside baseline
                # top-num_select, so the rescue can affect COCO top-100 without
                # discarding the rest of the baseline top-300 list.
                rescue_scores[torch.unique(protected_query_ids)] = -1e8
                current_rescue_num = min(rescue_num, int(rescue_scores.numel()))
                rescue_values, rescue_query_ids = torch.topk(
                    rescue_scores, current_rescue_num, dim=0
                )
                rescue_class_positions = query_class_positions[
                    batch_idx, rescue_query_ids
                ]
                rescue_indexes = (
                    rescue_query_ids * num_classes + rescue_class_positions
                )
                if protected_num > 0 and current_rescue_num > 0:
                    # COCO sorts detections by score. Keep rescue entries just
                    # above the protected boundary so they are actually evaluated
                    # in maxDets=100, then refill the remaining output budget
                    # with baseline order.
                    front_floor = protected_values[batch_idx, -1].detach()
                    eps = front_floor.abs() * 1e-6 + front_floor.new_tensor(1e-8)
                    rescue_rank_offsets = torch.arange(
                        current_rescue_num,
                        device=prob.device,
                        dtype=prob.dtype,
                    )
                    calibrated_rescue_values = (
                        front_floor
                        + eps * (float(current_rescue_num) - rescue_rank_offsets)
                    ).clamp(max=1.0)
                else:
                    calibrated_rescue_values = rescue_values
                front_values = torch.cat(
                    [protected_values[batch_idx], calibrated_rescue_values], dim=0
                )
                front_indexes = torch.cat(
                    [protected_indexes[batch_idx], rescue_indexes], dim=0
                )
                used_flat = {int(value) for value in front_indexes.detach().cpu().tolist()}
                filler_values = []
                filler_indexes = []
                for value, index in zip(
                    baseline_values[batch_idx].detach().cpu().tolist(),
                    baseline_indexes[batch_idx].detach().cpu().tolist(),
                ):
                    index = int(index)
                    if index in used_flat:
                        continue
                    filler_values.append(float(value))
                    filler_indexes.append(index)
                    if len(filler_indexes) + int(front_indexes.numel()) >= num_select:
                        break
                if filler_indexes:
                    filler_values_tensor = torch.as_tensor(
                        filler_values, device=prob.device, dtype=prob.dtype
                    )
                    filler_indexes_tensor = torch.as_tensor(
                        filler_indexes, device=prob.device, dtype=torch.long
                    )
                    batch_values = torch.cat([front_values, filler_values_tensor], dim=0)
                    batch_indexes = torch.cat([front_indexes, filler_indexes_tensor], dim=0)
                else:
                    batch_values = front_values
                    batch_indexes = front_indexes
                combined_values.append(batch_values[:num_select])
                combined_indexes.append(batch_indexes[:num_select])
            topk_values = torch.stack(combined_values, dim=0)
            topk_indexes = torch.stack(combined_indexes, dim=0)
        else:
            topk_values, topk_indexes = torch.topk(
                prob.view(prob.shape[0], -1), num_select, dim=1
            )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        class_positions = topk_indexes % prob.shape[2]
        labels = class_positions
        label_ids = self.label_ids.to(prob.device)
        labels = label_ids[labels]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                batched_nms(b, s, l, iou_threshold=self.nms_iou_threshold)
                for b, s, l in zip(boxes, scores, labels)
            ]

            results = [
                {
                    'scores': s[i],
                    'labels': l[i],
                    'boxes': b[i],
                    'query_ids': q[i],
                    'class_positions': c[i],
                }
                for s, l, b, q, c, i in zip(scores, labels, boxes, topk_boxes, class_positions, item_indices)
            ]
        else:
            results = [
                {
                    'scores': s,
                    'labels': l,
                    'boxes': b,
                    'query_ids': q,
                    'class_positions': c,
                }
                for s, l, b, q, c in zip(scores, labels, boxes, topk_boxes, class_positions)
            ]
        return results


def _as_bool_flag(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_optional_path(path_value):
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _build_dgcl_contrastive_class_weights(args):
    if not _as_bool_flag(getattr(args, "dgcl_contrastive_tail_boost", False)):
        return None

    meta_path = (
        getattr(args, "dgcl_contrastive_class_freq_meta_path", None)
        or getattr(args, "seen_category_meta_path", None)
    )
    meta_path = _resolve_optional_path(meta_path)
    if meta_path is None or not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)
    except Exception:
        return None

    class_table = meta.get("class_table", []) or []
    if not class_table:
        return None

    rows = []
    for row in class_table:
        label_id = row.get("new_id", row.get("orig_id"))
        if label_id is None:
            continue
        count = None
        for count_key in (
            "train_other_instances",
            "train_instances",
            "train_count",
            "instances",
        ):
            if row.get(count_key) is not None:
                count = row[count_key]
                break
        if count is None:
            continue
        rows.append((int(label_id), float(count)))
    if not rows:
        return None

    max_label_id = max(label_id for label_id, _ in rows)
    weights = torch.ones(max_label_id + 1, dtype=torch.float32)
    tail_max_count = float(
        getattr(args, "dgcl_contrastive_tail_boost_max_count", 10.0)
    )
    tail_weight = float(
        getattr(args, "dgcl_contrastive_tail_boost_weight", 2.0)
    )
    mid_max_count = float(
        getattr(args, "dgcl_contrastive_mid_boost_max_count", 50.0)
    )
    mid_weight = float(
        getattr(args, "dgcl_contrastive_mid_boost_weight", 1.25)
    )
    for label_id, count in rows:
        if count <= tail_max_count:
            weights[label_id] = max(1.0, tail_weight)
        elif count <= mid_max_count:
            weights[label_id] = max(1.0, mid_weight)
    return weights


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        use_aggr=getattr(args, "use_aggr", False),
        freeze_text_encoder=getattr(args, "freeze_text_encoder", False),
        text_encoder_unfreeze_last_n_layers=getattr(
            args, "text_encoder_unfreeze_last_n_layers", 0
        ),
        text_encoder_unfreeze_embeddings=getattr(
            args, "text_encoder_unfreeze_embeddings", False
        ),
        use_text_adapter_ffn=getattr(args, "use_text_adapter_ffn", False),
        text_adapter_hidden_dim=getattr(args, "text_adapter_hidden_dim", args.hidden_dim),
        text_adapter_dropout=getattr(args, "text_adapter_dropout", 0.0),
        use_similarity_calibration=getattr(args, "use_similarity_calibration", False),
        similarity_calibration_type=getattr(args, "similarity_calibration_type", "star"),
        similarity_calibration_hidden_dim=getattr(
            args, "similarity_calibration_hidden_dim", None
        ),
        similarity_calibration_dropout=getattr(args, "similarity_calibration_dropout", 0.0),
        similarity_calibration_residual=getattr(
            args, "similarity_calibration_residual", True
        ),
        similarity_calibration_mode=getattr(args, "similarity_calibration_mode", "additive"),
        class_logit_norm=getattr(args, "class_logit_norm", False),
        class_logit_scale=getattr(args, "class_logit_scale", 10.0),
        class_logit_bias=getattr(args, "class_logit_bias", 0.0),
        use_modality_projection=getattr(args, "use_modality_projection", False),
        modality_projection_type=getattr(args, "modality_projection_type", "star"),
        modality_projection_hidden_dim=getattr(args, "modality_projection_hidden_dim", None),
        modality_projection_dropout=getattr(args, "modality_projection_dropout", 0.0),
        modality_projection_width_factor=getattr(args, "modality_projection_width_factor", 2),
        modality_projection_residual=getattr(args, "modality_projection_residual", True),
        dual_prompt_class_logit_norm=getattr(args, "dual_prompt_class_logit_norm", False),
        dual_prompt_class_logit_scale=getattr(args, "dual_prompt_class_logit_scale", 10.0),
        dual_prompt_class_logit_bias=getattr(args, "dual_prompt_class_logit_bias", 0.0),
        use_quality_head=getattr(args, "use_quality_head", False),
        use_lgpr_survival=getattr(args, "use_lgpr_survival", False),
        lgpr_survival_hidden_dim=getattr(
            args, "lgpr_survival_hidden_dim", args.hidden_dim
        ),
        use_lgpr_box_refine=getattr(args, "use_lgpr_box_refine", False),
        lgpr_box_refine_hidden_dim=getattr(
            args, "lgpr_box_refine_hidden_dim", args.hidden_dim
        ),
        lgpr_box_refine_scale=getattr(args, "lgpr_box_refine_scale", 0.05),
        use_lgpr_box_refine_inference=getattr(
            args, "use_lgpr_box_refine_inference", True
        ),
        use_query_contrastive=getattr(args, "use_query_contrastive", False),
        use_global_image_text_sigmoid_loss=getattr(
            args, "use_global_image_text_sigmoid_loss", False
        ),
        use_sail_alignment_head=getattr(args, "use_sail_alignment_head", False),
        sail_alignment_target_dim=getattr(args, "sail_alignment_target_dim", None),
        sail_alignment_width_factor=getattr(args, "sail_alignment_width_factor", 8),
        sail_alignment_logit_scale=getattr(args, "sail_alignment_logit_scale", 20.0),
        sail_alignment_logit_bias=getattr(args, "sail_alignment_logit_bias", -10.0),
        use_sail_last_class_embed=getattr(args, "use_sail_last_class_embed", False),
        sail_last_class_embed_width_factor=getattr(
            args, "sail_last_class_embed_width_factor", 8
        ),
        sail_last_class_embed_alpha_init=getattr(
            args, "sail_last_class_embed_alpha_init", 0.2
        ),
        sail_last_class_embed_use_similarity_gate=getattr(
            args, "sail_last_class_embed_use_similarity_gate", True
        ),
        sail_last_class_embed_similarity_gate_bottleneck_dim=getattr(
            args, "sail_last_class_embed_similarity_gate_bottleneck_dim", 64
        ),
        sail_last_class_embed_similarity_gate_initial_bias=getattr(
            args, "sail_last_class_embed_similarity_gate_initial_bias", 0.0
        ),
        sail_last_class_embed_use_text_gate=getattr(
            args, "sail_last_class_embed_use_text_gate", False
        ),
        sail_last_class_embed_text_gate_hidden_dim=getattr(
            args, "sail_last_class_embed_text_gate_hidden_dim", None
        ),
        sail_last_class_embed_text_gate_strength_init=getattr(
            args, "sail_last_class_embed_text_gate_strength_init", 0.05
        ),
        sail_last_class_embed_gate_hidden_dim=getattr(
            args, "sail_last_class_embed_gate_hidden_dim", 128
        ),
        sail_last_class_embed_adapter_scale_init=getattr(
            args, "sail_last_class_embed_adapter_scale_init", 0.1
        ),
        sail_last_class_embed_temperature_init=getattr(
            args, "sail_last_class_embed_temperature_init", 0.2
        ),
        sail_last_class_embed_use_frequency_prior=getattr(
            args, "sail_last_class_embed_use_frequency_prior", True
        ),
        sail_last_class_embed_non_category_logit=getattr(
            args, "sail_last_class_embed_non_category_logit", -10.0
        ),
        sail_last_class_embed_use_baseline_residual=getattr(
            args, "sail_last_class_embed_use_baseline_residual", True
        ),
        sail_last_class_embed_use_main_token_blend=getattr(
            args, "sail_last_class_embed_use_main_token_blend", True
        ),
        sail_last_class_embed_output_blend_init=getattr(
            args, "sail_last_class_embed_output_blend_init", 0.1
        ),
        sail_last_class_embed_match_baseline_distribution=getattr(
            args, "sail_last_class_embed_match_baseline_distribution", True
        ),
        sail_last_class_embed_use_margin_gate=getattr(
            args, "sail_last_class_embed_use_margin_gate", False
        ),
        sail_last_class_embed_margin_gate_tau=getattr(
            args, "sail_last_class_embed_margin_gate_tau", 1.0
        ),
        sail_last_class_embed_margin_gate_gamma=getattr(
            args, "sail_last_class_embed_margin_gate_gamma", 0.2
        ),
        sail_last_class_embed_use_word_cocontribution=getattr(
            args, "sail_last_class_embed_use_word_cocontribution", False
        ),
        sail_last_class_embed_word_cocontribution_similarity_scale_init=getattr(
            args,
            "sail_last_class_embed_word_cocontribution_similarity_scale_init",
            0.0,
        ),
        sail_last_class_embed_word_cocontribution_completeness_init=getattr(
            args,
            "sail_last_class_embed_word_cocontribution_completeness_init",
            0.95,
        ),
        sail_last_class_embed_word_cocontribution_discriminative_init=getattr(
            args,
            "sail_last_class_embed_word_cocontribution_discriminative_init",
            0.0,
        ),
        use_aux_token_ce=getattr(args, "use_aux_token_ce", False),
        use_local_roi_query_fusion=getattr(args, "use_local_roi_query_fusion", False),
        local_roi_query_fusion_output_size=getattr(
            args, "local_roi_query_fusion_output_size", 7
        ),
        use_appearance_roi_fusion=getattr(args, "use_appearance_roi_fusion", False),
        use_feature_level_fusion=getattr(args, "use_feature_level_fusion", False),
        appearance_roi_output_size=getattr(args, "appearance_roi_output_size", 7),
        appearance_roi_temperature=getattr(args, "appearance_roi_temperature", 0.07),
        appearance_roi_alpha=getattr(args, "appearance_roi_alpha", 0.5),
        appearance_roi_semantic_topk=getattr(args, "appearance_roi_semantic_topk", 0),
        use_learnable_appearance_proto=getattr(args, "use_learnable_appearance_proto", False),
        use_appearance_query_bias=getattr(args, "use_appearance_query_bias", False),
        appearance_num_classes=getattr(args, "num_classes", 0),
        use_proto_cls_head=getattr(args, "use_proto_cls_head", False),
        proto_cls_temperature=getattr(args, "proto_cls_temperature", 0.07),
        use_appearance_rescoring=getattr(args, "use_appearance_rescoring", False),
        appearance_rescoring_temperature=getattr(args, "appearance_rescoring_temperature", 1.0),
        appearance_rescoring_bias=getattr(args, "appearance_rescoring_bias", 0.0),
        appearance_rescoring_mode=getattr(args, "appearance_rescoring_mode", "max"),
        use_appearance_text_alignment=getattr(args, "use_appearance_text_alignment", False),
        appearance_text_loss_coef=getattr(args, "appearance_text_loss_coef", 0.5),
        use_appearance_text_rerank=getattr(args, "use_appearance_text_rerank", False),
        appearance_text_rerank_weight=getattr(args, "appearance_text_rerank_weight", 0.25),
        use_visual_confidence=getattr(args, "use_visual_confidence", False),
        visual_confidence_roi_output_size=getattr(args, "visual_confidence_roi_output_size", 7),
        use_appearance_conditioned_gate=getattr(args, "use_appearance_conditioned_gate", False),
        appearance_gate_temperature=getattr(args, "appearance_gate_temperature", 0.07),
        appearance_gate_feature_source=getattr(args, "appearance_gate_feature_source", "backbone"),
        appearance_logit_bias_weight=getattr(args, "appearance_logit_bias_weight", 0.0),
        appearance_logit_bias_detach=getattr(args, "appearance_logit_bias_detach", False),
        appearance_logit_bias_norm=getattr(args, "appearance_logit_bias_norm", "tanh"),
        use_appearance_token_branch=getattr(args, "use_appearance_token_branch", False),
        appearance_token_branch_weight=getattr(args, "appearance_token_branch_weight", 0.2),
        appearance_token_branch_temperature=getattr(args, "appearance_token_branch_temperature", 0.07),
        appearance_token_branch_scale_init=getattr(args, "appearance_token_branch_scale_init", -2.0),
        use_roi_appearance_branch=getattr(args, "use_roi_appearance_branch", False),
        roi_appearance_output_size=getattr(args, "roi_appearance_output_size", 7),
        roi_appearance_temperature=getattr(args, "roi_appearance_temperature", 0.07),
        roi_appearance_scale_init=getattr(args, "roi_appearance_scale_init", -2.0),
        use_appearance_contrastive_head=getattr(args, "use_appearance_contrastive_head", False),
        appearance_contrastive_head_class_logit_norm=getattr(
            args, "appearance_contrastive_head_class_logit_norm", None
        ),
        appearance_contrastive_head_class_logit_scale=getattr(
            args, "appearance_contrastive_head_class_logit_scale", None
        ),
        appearance_contrastive_head_class_logit_bias=getattr(
            args, "appearance_contrastive_head_class_logit_bias", None
        ),
        appearance_contrastive_head_use_similarity_calibration=getattr(
            args, "appearance_contrastive_head_use_similarity_calibration", None
        ),
        appearance_contrastive_head_similarity_calibration_type=getattr(
            args, "appearance_contrastive_head_similarity_calibration_type", None
        ),
        appearance_contrastive_head_similarity_calibration_hidden_dim=getattr(
            args, "appearance_contrastive_head_similarity_calibration_hidden_dim", None
        ),
        appearance_contrastive_head_similarity_calibration_dropout=getattr(
            args, "appearance_contrastive_head_similarity_calibration_dropout", None
        ),
        appearance_contrastive_head_similarity_calibration_residual=getattr(
            args, "appearance_contrastive_head_similarity_calibration_residual", None
        ),
        appearance_contrastive_head_similarity_calibration_mode=getattr(
            args, "appearance_contrastive_head_similarity_calibration_mode", None
        ),
        appearance_contrastive_head_use_modality_projection=getattr(
            args, "appearance_contrastive_head_use_modality_projection", None
        ),
        appearance_contrastive_head_modality_projection_type=getattr(
            args, "appearance_contrastive_head_modality_projection_type", None
        ),
        appearance_contrastive_head_modality_projection_hidden_dim=getattr(
            args, "appearance_contrastive_head_modality_projection_hidden_dim", None
        ),
        appearance_contrastive_head_modality_projection_dropout=getattr(
            args, "appearance_contrastive_head_modality_projection_dropout", None
        ),
        appearance_contrastive_head_modality_projection_width_factor=getattr(
            args, "appearance_contrastive_head_modality_projection_width_factor", None
        ),
        appearance_contrastive_head_modality_projection_residual=getattr(
            args, "appearance_contrastive_head_modality_projection_residual", None
        ),
        use_appearance_prototype_bank=getattr(
            args, "use_appearance_prototype_bank", False
        ),
        appearance_prototype_texts=(
            getattr(args, "appearance_multi_facet_label_list", None)
            if getattr(args, "use_self_facb_multi_facet", False)
            else getattr(args, "appearance_eval_label_list", None)
        ),
        appearance_shared_prototype_texts=getattr(
            args, "appearance_shared_label_list", None
        ),
        use_appearance_discriminative_residual=getattr(
            args, "use_appearance_discriminative_residual", False
        ),
        appearance_discriminative_mode=getattr(
            args, "appearance_discriminative_mode", "residual"
        ),
        appearance_contrastive_head_feature_source=getattr(
            args, "appearance_contrastive_head_feature_source", "decoder_query"
        ),
    )

    model.query_score_pos_iou_thr = getattr(args, "query_score_pos_iou_thr", 0.5)
    model.query_score_neg_iou_thr = getattr(args, "query_score_neg_iou_thr", 0.1)
    model.query_score_num_pos = getattr(args, "query_score_num_pos", 16)
    model.query_score_num_neg = getattr(args, "query_score_num_neg", 32)
    model.query_score_rank_margin = getattr(args, "query_score_rank_margin", 0.2)
    # Expose sem_confuser mode on the model so forward() can conditionally
    # compute sem_logits_bank without importing args.
    model.appearance_contrastive_sem_confuser_mode_flag = bool(
        getattr(args, "appearance_contrastive_sem_confuser_mode", False)
    )
    # ---- Candidate-conditioned local evidence config ----
    model.use_appearance_local_evidence = bool(
        getattr(args, "use_appearance_local_evidence", False)
    )
    model.appearance_local_evidence_topm = int(
        getattr(args, "appearance_local_evidence_topm", 8)
    )
    _le_levels = getattr(args, "appearance_local_evidence_levels", None)
    if _le_levels is None:
        _le_levels = [0]
    elif isinstance(_le_levels, int):
        _le_levels = [_le_levels]
    model.appearance_local_evidence_levels = list(_le_levels)
    model.appearance_local_evidence_roi_size = int(
        getattr(args, "appearance_local_evidence_roi_size", 7)
    )
    _self_facb_levels_cfg = getattr(args, "self_facb_levels", _le_levels)
    if isinstance(_self_facb_levels_cfg, int):
        _self_facb_levels_cfg = [_self_facb_levels_cfg]
    model.self_facb_levels = list(_self_facb_levels_cfg or _le_levels or [0])
    model.self_facb_roi_size = int(
        getattr(args, "self_facb_roi_size", model.appearance_local_evidence_roi_size)
    )
    # Semantic-text bank for candidate ranking (true semantic full-space top-M).
    # Aligned to global class id order, same as the appearance bank.
    _sem_texts = getattr(args, "semantic_label_list", None)
    model.appearance_semantic_prototype_texts = (
        list(_sem_texts) if _sem_texts else None
    )

    # ---- Detector-native Self-FACB branch ----
    model.use_self_facb = bool(getattr(args, "use_self_facb", False))
    model.return_self_facb_features = bool(
        getattr(args, "return_self_facb_features", model.use_self_facb)
        or getattr(args, "use_self_facb_loss", False)
        or getattr(args, "use_facb_symbol_align_loss", False)
        or getattr(args, "use_facb_agf", False)
    )
    _self_facb_levels_for_head = getattr(args, "self_facb_levels", [0]) or [0]
    if isinstance(_self_facb_levels_for_head, int):
        _self_facb_levels_for_head = [_self_facb_levels_for_head]
    _self_facb_num_levels_for_head = max(len(_self_facb_levels_for_head), 1)
    _self_facb_context_mult = (
        2
        if bool(getattr(args, "use_self_facb_lc", False))
        and float(getattr(args, "self_facb_lc_context_scale", 1.0)) > 1.0
        else 1
    )
    model.self_facb_head = SelfFACBHead(
        embed_dim=model.hidden_dim,
        logit_scale=float(getattr(args, "self_facb_logit_scale", 10.0)),
        text_proj_hidden_dim=int(getattr(args, "self_facb_text_proj_hidden_dim", 512)),
        num_facets=int(getattr(args, "self_facb_num_facets", 3 if getattr(args, "use_self_facb_multi_facet", False) else 1)),
        facet_temperature=float(getattr(args, "self_facb_facet_temperature", 2.0)),
        facet_aggregation=getattr(args, "self_facb_facet_aggregation", "softmin"),
        scorer_type=getattr(args, "self_facb_scorer_type", "local_evidence"),
        decoder_layers=int(getattr(args, "self_facb_decoder_layers", 1)),
        decoder_nheads=int(getattr(args, "self_facb_decoder_nheads", 4)),
        decoder_ffn_dim=int(getattr(args, "self_facb_decoder_ffn_dim", 1024)),
        decoder_dropout=float(getattr(args, "self_facb_decoder_dropout", 0.0)),
        facet_weight_init=getattr(args, "self_facb_facet_weight_init", None),
        roi_size=int(getattr(args, "self_facb_roi_size", 7)),
        max_levels=int(
            getattr(
                args,
                "self_facb_max_levels",
                _self_facb_num_levels_for_head * _self_facb_context_mult,
            )
        ),
        residual_bound=float(getattr(args, "self_facb_crm_residual_bound", 0.5)),
        use_roi_position_level_embeddings=bool(
            getattr(args, "self_facb_crm_use_roi_position_level_embeddings", False)
        ),
    )
    _self_facb_path = str(getattr(args, "self_facb_head_path", "") or "")
    if _self_facb_path:
        _ckpt = torch.load(_self_facb_path, map_location="cpu")
        _state = _ckpt.get("model", _ckpt.get("state_dict", _ckpt))
        _missing, _unexpected = model.self_facb_head.load_state_dict(_state, strict=False)
        _allowed_missing = set()
        _allowed_unexpected = set()
        if getattr(model.self_facb_head, "facet_weight_logits", None) is not None:
            _allowed_missing.add("facet_weight_logits")
        else:
            _allowed_unexpected.add("facet_weight_logits")
        _allowed_missing.update(key for key in _missing if key.startswith("decoder."))
        _crm_optional_keys = {
            "roi_position_embedding",
            "roi_level_embedding",
            "residual_calibration_head.weight",
            "residual_calibration_head.bias",
            "lc_class_delta_head.weight",
            "lc_class_delta_head.bias",
            "lc_lift_head.weight",
            "lc_lift_head.bias",
            "lc_lift_confidence_head.weight",
            "lc_lift_confidence_head.bias",
        }
        _allowed_missing.update(key for key in _missing if key in _crm_optional_keys)
        _allowed_unexpected.update(key for key in _unexpected if key.startswith("decoder."))
        _missing = [key for key in _missing if key not in _allowed_missing]
        _unexpected = [key for key in _unexpected if key not in _allowed_unexpected]
        if _unexpected:
            raise RuntimeError(f"Unexpected Self-FACB checkpoint keys: {_unexpected}")
        if _missing:
            raise RuntimeError(f"Missing Self-FACB checkpoint keys: {_missing}")

    model.use_facb_symbol_align = bool(getattr(args, "use_facb_symbol_align_loss", False))
    model.facb_symbol_align_texts = (
        getattr(args, "appearance_multi_facet_label_list", None)
        if getattr(args, "use_self_facb_multi_facet", False)
        else getattr(args, "appearance_eval_label_list", None)
    )
    model.facb_symbol_align_fallback_texts = getattr(
        args, "appearance_eval_label_list", None
    )
    model.facb_symbol_align_semantic_texts = getattr(
        args, "semantic_label_list", None
    )
    model.facb_symbol_align_extract_from_semantic = bool(
        getattr(args, "facb_symbol_align_extract_from_semantic", True)
    )
    model.facb_symbol_align_facet_index = int(
        getattr(args, "facb_symbol_align_facet_index", 1)
    )
    model.facb_symbol_align_online_chunk = int(
        getattr(args, "facb_symbol_align_online_chunk", 64)
    )

    # ---- Method B: same-group token-margin ----
    model.use_semantic_group_margin = bool(getattr(args, "use_semantic_group_margin", False))

    # ---- CLIP crop evidence branch ----
    model.use_appearance_clip_evidence = bool(getattr(args, "use_appearance_clip_evidence", False))
    if model.use_appearance_clip_evidence:
        model.appearance_clip_model_path = getattr(args, "appearance_clip_model_path", None)
        model.appearance_clip_text_model_path = getattr(
            args, "appearance_clip_text_model_path", None
        ) or model.appearance_clip_model_path
        model.appearance_clip_crop_size = int(getattr(args, "appearance_clip_crop_size", 224))
        model.appearance_clip_topm = int(getattr(args, "appearance_clip_topm", 8))
        model.appearance_clip_eval_select_mode = str(
            getattr(args, "appearance_clip_eval_select_mode", "detections")
        )
        model.appearance_clip_min_box_area = float(
            getattr(args, "appearance_clip_min_box_area", 0.0)
        )
        model.appearance_clip_min_det_score = float(
            getattr(args, "appearance_clip_min_det_score", 0.0)
        )
        model.appearance_clip_max_eval_queries = int(
            getattr(args, "appearance_clip_max_eval_queries", 64)
        )
        # Candidate texts for the CLIP TEXT bank (global class id order).
        # For FACB we support both CLIP-friendly category-name prompts and
        # discriminative visual_cue_text prompts. The latter is useful for probing
        # whether short appearance cues are more helpful than long descriptions.
        _clip_template = getattr(args, "appearance_clip_text_template", "a photo of a {label} traffic sign")
        _clip_label_src = getattr(args, "appearance_clip_text_label_source", "label_list")
        if _clip_label_src in {"appearance_eval_label_list", "visual_cue_text"}:
            _clip_base = getattr(args, "appearance_eval_label_list", None)
        elif _clip_label_src == "semantic_label_list":
            _clip_base = getattr(args, "semantic_label_list", None)
        else:
            _clip_base = getattr(args, "label_list", None)
        if _clip_base:
            if _clip_template and "{label}" in _clip_template:
                model.appearance_clip_text_prototype_texts = [
                    _clip_template.format(label=str(x)) for x in _clip_base
                ]
            else:
                model.appearance_clip_text_prototype_texts = [str(x) for x in _clip_base]
        else:
            model.appearance_clip_text_prototype_texts = None
        model.appearance_clip_evidence_head = CLIPCropEvidence(
            clip_model_path=model.appearance_clip_model_path,
            crop_size=model.appearance_clip_crop_size,
            logit_scale=float(getattr(args, "appearance_clip_logit_scale", 10.0)),
            use_visual_projection=bool(
                getattr(args, "appearance_clip_use_visual_projection", False)
            ),
            pairwise_head_path=getattr(args, "appearance_clip_pairwise_head_path", None),
            pairwise_hidden_dim=int(getattr(args, "appearance_clip_pairwise_hidden_dim", 512)),
            pairwise_dropout=float(getattr(args, "appearance_clip_pairwise_dropout", 0.1)),
        )



    def _facb_select_ids_from_split_meta(meta_path, min_count=1, max_count=50, seen_only=True):
        selected = []
        if not meta_path:
            return selected
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            count_by_id = {}
            for row in meta.get("class_table", []):
                if not isinstance(row, dict):
                    continue
                count = None
                for key in (
                    "train_other_instances",
                    "train_instances",
                    "train_count",
                    "num_train",
                    "count",
                ):
                    if key in row:
                        count = row.get(key)
                        break
                if count is None:
                    continue
                count = int(count)
                for key in ("new_id", "id", "orig_id"):
                    if key in row and row.get(key) is not None:
                        count_by_id[int(row.get(key))] = count
                        break
            for key in (
                "train_instances_by_class_original",
                "train_instances_by_class",
            ):
                for class_id, count in (meta.get(key, {}) or {}).items():
                    count_by_id[int(class_id)] = int(count)
            train_records = meta.get("train_records", {}) or {}
            for class_id, count in (train_records.get("instances_by_class", {}) or {}).items():
                count_by_id[int(class_id)] = int(count)
            fewshot_count_by_id = {}
            for group in (meta.get("fewshot_groups", {}) or {}).values():
                if not isinstance(group, dict):
                    continue
                shot = int(group.get("shot", 0) or 0)
                for class_id in group.get("classes", []) or []:
                    fewshot_count_by_id[int(class_id)] = shot
            if (
                meta.get("protocol") == "tt100k_freqtier_combined_fewshot"
                and fewshot_count_by_id
            ):
                count_by_id = fewshot_count_by_id
            else:
                count_by_id.update(fewshot_count_by_id)

            seen_ids = {int(x) for x in (meta.get("seen_orig_ids", []) or [])}
            for class_id, count in count_by_id.items():
                if count < int(min_count) or count > int(max_count):
                    continue
                if seen_only and seen_ids and class_id not in seen_ids:
                    continue
                selected.append(int(class_id))
        except Exception as exc:
            print(
                "[facb-smallhr] failed to parse split meta {}: {}".format(
                    meta_path, exc
                ),
                flush=True,
            )
        return sorted(set(selected))

    # ---- FACB-SpeedHR/SmallHR: high-resolution crop reader for selected classes ----
    model.use_facb_speedhr_loss = bool(getattr(args, "use_facb_speedhr_loss", False))
    model.facb_speedhr_head = None
    model.facb_speedhr_global_to_digit = None
    model.facb_speedhr_digit_values = []
    model.facb_speedhr_candidate_multi_facet = bool(
        getattr(args, "facb_speedhr_candidate_multi_facet", False)
    )
    model.facb_speedhr_multi_facet_texts = getattr(
        args, "facb_speedhr_multi_facet_label_list", None
    )
    model.facb_speedhr_facet_names = getattr(
        args, "facb_speedhr_facet_names", None
    )
    model.facb_speedhr_shared_compositional_facets = bool(
        getattr(args, "facb_speedhr_shared_compositional_facets", False)
    )
    model.facb_speedhr_shared_facet_names = list(
        getattr(args, "facb_speedhr_shared_facet_names", None) or []
    )
    model.facb_speedhr_shared_facet_vocab_texts = list(
        getattr(args, "facb_speedhr_shared_facet_vocab_texts", None) or []
    )
    _shared_class_targets = list(
        getattr(args, "facb_speedhr_shared_facet_class_targets", None) or []
    )
    if _shared_class_targets:
        _shared_facet_count = max(len(row) for row in _shared_class_targets)
        _shared_max_targets = max(
            (
                len(ids)
                for row in _shared_class_targets
                for ids in row
            ),
            default=1,
        )
        _shared_target_tensor = torch.full(
            (
                len(_shared_class_targets),
                _shared_facet_count,
                max(_shared_max_targets, 1),
            ),
            -1,
            dtype=torch.long,
        )
        for class_id, row in enumerate(_shared_class_targets):
            for facet_idx, target_ids in enumerate(row):
                if target_ids:
                    _shared_target_tensor[
                        class_id, facet_idx, : len(target_ids)
                    ] = torch.as_tensor(target_ids, dtype=torch.long)
        model.facb_speedhr_shared_facet_class_targets = _shared_target_tensor
        model.facb_speedhr_shared_facet_class_target_mask = (
            _shared_target_tensor >= 0
        )
    model.facb_speedhr_fixed_candidate_ids = list(getattr(args, "facb_speedhr_fixed_candidate_ids", []) or [])
    model.facb_speedhr_export_states = bool(getattr(args, "facb_speedhr_export_states", False))
    model.facb_speedhr_infer_pixel_source = str(getattr(args, "facb_speedhr_infer_pixel_source", "tensor"))
    if model.facb_speedhr_infer_pixel_source not in {"tensor", "raw_image"}:
        raise ValueError("FACB infer pixel source must be tensor or raw_image")
    model.facb_speedhr_defer_to_postprocess = bool(
        getattr(args, "facb_speedhr_defer_to_postprocess", False)
    )
    model.enable_facb_speedhr_inference = bool(
        getattr(args, "enable_facb_speedhr_inference", False)
    )
    model.facb_speedhr_candidate_matching = bool(
        getattr(args, "facb_speedhr_candidate_matching", False)
    )
    model.facb_speedhr_crop_scale = float(
        getattr(args, "facb_speedhr_crop_scale", 1.2)
    )
    model.facb_speedhr_infer_candidate_topk = int(
        getattr(args, "facb_speedhr_infer_candidate_topk", 10)
    )
    model.facb_speedhr_candidate_policy = str(
        getattr(args, "facb_speedhr_candidate_policy", "fixed_bank_plus_outside_topk_v1")
    ).strip().lower()
    model.facb_speedhr_infer_max_rows_per_image = int(
        getattr(args, "facb_speedhr_infer_max_rows_per_image", 32)
    )
    model.facb_speedhr_infer_chunk_size = int(
        getattr(args, "facb_speedhr_infer_chunk_size", 16)
    )
    model.facb_speedhr_infer_min_semantic_score = float(
        getattr(args, "facb_speedhr_infer_min_semantic_score", -1.0)
    )
    model.facb_speedhr_infer_max_semantic_score = float(
        getattr(args, "facb_speedhr_infer_max_semantic_score", -1.0)
    )
    model.facb_speedhr_infer_allowed_global_ids = None
    model.facb_speedhr_label_texts = list(
        getattr(args, "label_list", None)
        or getattr(args, "semantic_label_list", None)
        or []
    )
    if model.use_facb_speedhr_loss:
        _speed_label_texts = list(
            getattr(args, "label_list", None)
            or getattr(args, "semantic_label_list", None)
            or []
        )
        _scope = str(getattr(args, "facb_speedhr_scope", "speed_limit")).strip().lower()
        _speed_entries = []
        _manual_allowed_ids = getattr(args, "facb_speedhr_allowed_global_ids", None) or []
        _manual_allowed_ids = [int(x) for x in _manual_allowed_ids] if _manual_allowed_ids else []
        if _scope in {"all", "all_seen", "all_train", "candidate_all"}:
            _speed_entries = [
                (_gid, "{}:{}".format(_gid, str(_name)))
                for _gid, _name in enumerate(_speed_label_texts)
            ]
        elif _scope in {"smallshot", "fewshot", "tail"}:
            _meta_path = (
                getattr(args, "facb_speedhr_split_meta_path", None)
                or getattr(args, "split_meta_path", None)
                or getattr(args, "category_meta_path", None)
            )
            _max_count = int(getattr(args, "facb_speedhr_max_train_count", 10))
            _min_count = int(getattr(args, "facb_speedhr_min_train_count", 1))
            _allowed = _facb_select_ids_from_split_meta(
                _meta_path,
                min_count=_min_count,
                max_count=_max_count,
                seen_only=False,
            )
            if _manual_allowed_ids:
                _allowed = _manual_allowed_ids
            _allowed = sorted(set(_allowed))
            _speed_entries = [
                (_gid, "{}:{}".format(_gid, str(_speed_label_texts[_gid])))
                for _gid in _allowed
                if 0 <= _gid < len(_speed_label_texts)
            ]
        elif _scope in {"allowed", "manual"}:
            _speed_entries = [
                (_gid, "{}:{}".format(_gid, str(_speed_label_texts[_gid])))
                for _gid in _manual_allowed_ids
                if 0 <= _gid < len(_speed_label_texts)
            ]
        else:
            _speed_pattern = re.compile(
                r"^\s*speed\s+limit\s+(\d+(?:\.\d+)?)\s*km/?h\s*$",
                flags=re.IGNORECASE,
            )
            for _gid, _name in enumerate(_speed_label_texts):
                _match = _speed_pattern.match(str(_name))
                if _match is None:
                    continue
                _speed_entries.append((_gid, _match.group(1)))
        _values = [value for _, value in _speed_entries]
        _value_to_digit = {v: i for i, v in enumerate(_values)}
        _global_to_digit = torch.full((len(_speed_label_texts),), -1, dtype=torch.long)
        for _gid, _value in _speed_entries:
            _global_to_digit[_gid] = int(_value_to_digit[_value])
        model.facb_speedhr_digit_values = list(_values)
        model.facb_speedhr_global_to_digit = _global_to_digit
        _speedhr_encoder = str(
            getattr(args, "facb_speedhr_encoder", "cnn") or "cnn"
        ).strip().lower()
        if _speedhr_encoder in {"gdino", "groundingdino", "backbone"}:
            _crop_backbone = model.backbone[0] if hasattr(model.backbone, "__getitem__") else model.backbone
            _feature_dim = int(getattr(args, "facb_speedhr_feature_dim", 0) or model.backbone.num_channels[-1])
            model.facb_speedhr_head = FACBGDinoHRCropHead(
                crop_backbone=_crop_backbone,
                feature_dim=_feature_dim,
                num_classes=max(len(_values), 1),
                crop_size=int(getattr(args, "facb_speedhr_crop_size", 224)),
                hidden_dim=int(getattr(args, "facb_speedhr_hidden_dim", 256)),
                adapter_rank=int(getattr(args, "facb_speedhr_adapter_rank", 16)),
                adapter_scale=float(getattr(args, "facb_speedhr_adapter_scale", 1.0)),
                lora_rank=int(getattr(args, "facb_speedhr_lora_rank", 8)),
                lora_scale=float(getattr(args, "facb_speedhr_lora_scale", 1.0)),
                lora_last_qkv=int(getattr(args, "facb_speedhr_lora_last_qkv", 2)),
                candidate_matching=bool(
                    getattr(args, "facb_speedhr_candidate_matching", False)
                ),
                candidate_decoder_layers=int(
                    getattr(args, "facb_speedhr_candidate_decoder_layers", 2)
                ),
                candidate_nheads=int(
                    getattr(args, "facb_speedhr_candidate_nheads", 8)
                ),
                candidate_max_facets=int(
                    getattr(args, "facb_speedhr_candidate_max_facets", 5)
                ),
                shared_facet_count=len(
                    model.facb_speedhr_shared_facet_names
                )
                if model.facb_speedhr_shared_compositional_facets
                else 0,
                shared_candidate_fusion_init=float(
                    getattr(
                        args,
                        "facb_speedhr_shared_candidate_fusion_init",
                        -2.0,
                    )
                ),
                dropout=float(getattr(args, "facb_speedhr_dropout", 0.1)),
            )
            if hasattr(model.facb_speedhr_head, "use_shared_candidate_space"):
                model.facb_speedhr_head.use_shared_candidate_space = bool(
                    getattr(args, "facb_speedhr_use_shared_candidate_space", False)
                )
        else:
            model.facb_speedhr_head = FACBSpeedHRCropHead(
                num_classes=max(len(_values), 1),
                crop_size=int(getattr(args, "facb_speedhr_crop_size", 64)),
                hidden_dim=int(getattr(args, "facb_speedhr_hidden_dim", 128)),
                dropout=float(getattr(args, "facb_speedhr_dropout", 0.1)),
            )
        print(
            "[facb-speedhr] enabled encoder={} scope={} classes={} "
            "candidate_matching={} multi_facet={} query_source={} crop_source={} "
            "shared_facets={} shared_space={} crop_lora_qkv={} values={}".format(
                _speedhr_encoder,
                _scope,
                len(_speed_entries),
                bool(getattr(args, "facb_speedhr_candidate_matching", False)),
                model.facb_speedhr_candidate_multi_facet,
                str(getattr(args, "facb_speedhr_candidate_query_source", "matcher")),
                str(getattr(args, "facb_speedhr_candidate_crop_source", "pred")),
                model.facb_speedhr_shared_facet_names,
                bool(getattr(model.facb_speedhr_head, "use_shared_candidate_space", False)),
                len(getattr(model.facb_speedhr_head, "_crop_lora_targets", [])),
                model.facb_speedhr_digit_values,
            ),
            flush=True,
        )
        if hasattr(model.facb_speedhr_head, "shared_symbol_bce"):
            model.facb_speedhr_head.shared_symbol_bce = bool(
                getattr(args, "facb_speedhr_shared_symbol_bce", False)
            )
            model.facb_speedhr_head.shared_facet_names = list(
                getattr(args, "facb_speedhr_shared_facet_names", None) or []
            )
            model.facb_speedhr_head.shared_facet_alignment_mode = str(
                getattr(args, "facb_speedhr_shared_facet_alignment_mode", "decoder")
                or "decoder"
            )
            model.facb_speedhr_head.shared_facet_token_topk = int(
                getattr(args, "facb_speedhr_shared_facet_token_topk", 3)
            )
            model.facb_speedhr_head.shared_facet_token_temperature = float(
                getattr(args, "facb_speedhr_shared_facet_token_temperature", 0.07)
            )

    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if getattr(args, "use_quality_head", False):
        weight_dict['loss_quality'] = getattr(args, 'quality_loss_coef', 1.0)
    if getattr(args, "use_lgpr_survival", False):
        weight_dict['loss_lgpr_survival'] = getattr(
            args, 'lgpr_survival_loss_coef', 0.5
        )
    if getattr(args, "use_lgpr_loc_preserve", False):
        weight_dict['loss_lgpr_loc_preserve'] = getattr(
            args, 'lgpr_loc_loss_coef', 0.2
        )
    if getattr(args, "use_lgpr_box_refine", False):
        weight_dict['loss_lgpr_box_refine'] = getattr(
            args, 'lgpr_box_refine_loss_coef', 1.0
        )
    if getattr(args, "use_query_contrastive", False):
        weight_dict['loss_query_contrastive'] = getattr(args, 'query_contrastive_loss_coef', 1.0)
    if getattr(args, "use_global_image_text_sigmoid_loss", False):
        weight_dict['loss_global_image_text_sigmoid'] = getattr(
            args, 'global_image_text_sigmoid_loss_coef', 1.0
        )
    if getattr(args, "use_query_score_selector", False):
        weight_dict['loss_query_selector'] = getattr(args, 'query_score_loss_coef', 1.0)
    if getattr(args, "use_aux_token_ce", False):
        weight_dict['loss_aux_token_ce'] = getattr(args, 'aux_token_ce_loss_coef', 0.5)
    if getattr(args, "use_proto_cls_head", False):
        weight_dict['loss_proto_cls'] = getattr(args, 'proto_cls_loss_coef', 0.5)
    if getattr(args, "use_appearance_cls_loss", False):
        weight_dict['loss_appearance_cls'] = getattr(args, 'appearance_cls_loss_coef', 0.5)
    if getattr(args, "use_appearance_contrastive_loss", False):
        weight_dict['loss_appearance_contrastive'] = getattr(
            args, 'appearance_contrastive_loss_coef', 0.5
        )
    if getattr(args, "use_appearance_text_alignment", False):
        weight_dict['loss_appearance_text_alignment'] = getattr(args, 'appearance_text_loss_coef', 0.5)
    if getattr(args, "use_semantic_group_margin", False):
        # Method B: conservative coef — gradient flows into the main detection head,
        # so keep it small to sharpen same-group margins without destabilizing boxes.
        weight_dict['loss_semantic_group_margin'] = getattr(args, 'semantic_group_margin_loss_coef', 0.2)
    if getattr(args, "use_dgcl_contrastive_loss", False):
        weight_dict["loss_dgcl_contrastive"] = getattr(
            args, "dgcl_contrastive_loss_coef", 0.2
        )
    if getattr(args, "use_dgcl_unmatched_preserve", False):
        weight_dict["loss_dgcl_unmatched_preserve"] = getattr(
            args, "dgcl_unmatched_preserve_coef", 0.5
        )
    if getattr(args, "use_visual_confidence", False):
        weight_dict['loss_visual_confidence'] = getattr(args, 'visual_confidence_loss_coef', 1.0)
    if getattr(args, "use_appearance_conditioned_gate", False):
        weight_dict['loss_appearance_gate'] = getattr(args, 'appearance_gate_loss_coef', 1.0)
    if getattr(args, "use_facb_symbol_align_loss", False):
        weight_dict['loss_facb_symbol_align'] = getattr(
            args, 'facb_symbol_align_loss_coef', 1.0
        )
    if getattr(args, "use_self_facb_loss", False):
        weight_dict['loss_self_facb'] = getattr(args, 'self_facb_loss_coef', 1.0)
    if getattr(args, "use_facb_agf", False):
        weight_dict['loss_facb_agf'] = getattr(args, 'facb_agf_loss_coef', 1.0)
    if getattr(args, "use_facb_text_orth_loss", False):
        weight_dict['loss_facb_text_orth'] = getattr(
            args, 'facb_text_orth_loss_coef', 1.0
        )
    if getattr(args, "use_facb_speedhr_loss", False):
        weight_dict['loss_facb_speedhr'] = getattr(
            args, 'facb_speedhr_loss_coef', 1.0
        )
    roi_appearance_branch_only = bool(
        getattr(args, "use_roi_appearance_branch", False)
        and getattr(args, "roi_appearance_branch_only", False)
    )
    self_facb_branch_only = bool(
        getattr(args, "use_self_facb_loss", False)
        and getattr(args, "self_facb_branch_only", False)
    )
    if roi_appearance_branch_only:
        weight_dict = {
            'loss_appearance_contrastive': getattr(
                args, 'appearance_contrastive_loss_coef', 1.0
            )
        }
    if self_facb_branch_only:
        weight_dict = {
            'loss_self_facb': getattr(args, 'self_facb_loss_coef', 1.0)
        }
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {
                k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
                if k in _coeff_weight_dict
            }
        )
        weight_dict.update(interm_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality']
    if roi_appearance_branch_only:
        losses = ['appearance_contrastive']
    elif self_facb_branch_only:
        losses = ['self_facb']
    else:
        losses = ['labels', 'boxes']
        if getattr(args, "use_quality_head", False):
            losses.append('quality')
        if getattr(args, "use_lgpr_survival", False):
            losses.append('lgpr_survival')
        if getattr(args, "use_lgpr_loc_preserve", False):
            losses.append('lgpr_loc_preserve')
        if getattr(args, "use_lgpr_box_refine", False):
            losses.append('lgpr_box_refine')
        if getattr(args, "use_query_contrastive", False):
            losses.append('query_contrastive')
        if getattr(args, "use_global_image_text_sigmoid_loss", False):
            losses.append('global_image_text_sigmoid')
        if getattr(args, "use_query_score_selector", False):
            losses.append('query_selector')
        if getattr(args, "use_aux_token_ce", False):
            losses.append('aux_token_ce')
        if getattr(args, "use_proto_cls_head", False):
            losses.append('proto_cls')
        if getattr(args, "use_appearance_cls_loss", False):
            losses.append('appearance_cls')
        if getattr(args, "use_appearance_contrastive_loss", False):
            losses.append('appearance_contrastive')
        if getattr(args, "use_appearance_text_alignment", False):
            losses.append('appearance_text_alignment')
        if getattr(args, "use_semantic_group_margin", False):
            losses.append('semantic_group_margin')
        if getattr(args, "use_dgcl_contrastive_loss", False):
            losses.append("dgcl_contrastive")
        if getattr(args, "use_dgcl_unmatched_preserve", False):
            losses.append("dgcl_unmatched_preserve")
        if getattr(args, "use_visual_confidence", False):
            losses.append('visual_confidence')
        if getattr(args, "use_appearance_conditioned_gate", False):
            losses.append('appearance_gate')
        if getattr(args, "use_facb_symbol_align_loss", False):
            losses.append('facb_symbol_align')
    if getattr(args, "use_self_facb_loss", False):
        losses.append('self_facb')
    if getattr(args, "use_facb_agf", False):
        losses.append('facb_agf')
    if getattr(args, "use_facb_text_orth_loss", False):
        losses.append('facb_text_orth')
    if getattr(args, "use_facb_speedhr_loss", False):
        losses.append('facb_speedhr')

    aggr_regularizer = None
    if getattr(args, "use_aggr", False):
        aggr_regularizer = AppearanceGuidedGroupRegularizer(
            hidden_dim=args.hidden_dim,
            num_groups=getattr(args, "aggr_num_groups", 3),
            ema_momentum=getattr(args, "aggr_ema_momentum", 0.9),
            use_ema=getattr(args, "aggr_use_ema", True),
            consistency_loss=getattr(args, "aggr_use_ema_consistency", False),
            positive_only=getattr(args, "aggr_positive_only", True),
        )
        weight_dict["loss_aggr_orth"] = getattr(args, "aggr_loss_coef", 0.1)
        if getattr(args, "aggr_use_ema_consistency", False):
            weight_dict["loss_aggr_consistency"] = getattr(
                args, "aggr_ema_consistency_loss_coef", 0.05
            )

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses,
                             aggr_regularizer=aggr_regularizer,
                             query_contrastive_temperature=getattr(
                                 args, "query_contrastive_temperature", 0.07
                             ),
                             query_contrastive_mode=getattr(
                                 args, "query_contrastive_mode", "softmax_ce"
                             ),
                             query_contrastive_bias=getattr(
                                 args, "query_contrastive_bias", 0.0
                             ),
                             query_contrastive_topk_neg=getattr(
                                 args, "query_contrastive_topk_neg", 5
                             ),
                             query_contrastive_same_group_only=getattr(
                                 args, "query_contrastive_same_group_only", False
                             ),
                             query_contrastive_fallback_all=getattr(
                                 args, "query_contrastive_fallback_all", False
                             ),
                             use_global_image_text_sigmoid_loss=getattr(
                                 args, "use_global_image_text_sigmoid_loss", False
                             ),
                             use_aux_token_hard_neg=getattr(
                                 args, "use_aux_token_hard_neg", False
                             ),
                             aux_token_hard_neg_pos_weight=getattr(
                                 args, "aux_token_hard_neg_pos_weight", 1.0
                             ),
                             aux_token_hard_neg_same_group_weight=getattr(
                                 args, "aux_token_hard_neg_same_group_weight", 4.0
                             ),
                             aux_token_hard_neg_diff_group_weight=getattr(
                                 args, "aux_token_hard_neg_diff_group_weight", 0.1
                             ),
                             use_proto_cls_head=getattr(args, "use_proto_cls_head", False),
                             use_appearance_cls_loss=getattr(args, "use_appearance_cls_loss", False),
                             use_appearance_contrastive_loss=getattr(
                                 args, "use_appearance_contrastive_loss", False
                             ),
                             use_appearance_text_alignment=getattr(args, "use_appearance_text_alignment", False),
                             appearance_contrastive_topk_neg=getattr(
                                 args, "appearance_contrastive_topk_neg", 5
                             ),
                             appearance_contrastive_same_group_only=getattr(
                                 args, "appearance_contrastive_same_group_only", True
                             ),
                             appearance_contrastive_fallback_all=getattr(
                                 args, "appearance_contrastive_fallback_all", True
                             ),
                             appearance_global_fine_group_ids=getattr(
                                 args, "appearance_global_fine_group_ids", None
                             ),
                             appearance_contrastive_sem_confuser_mode=getattr(
                                 args, "appearance_contrastive_sem_confuser_mode", False
                             ),
                             appearance_full_ce_loss_coef=getattr(
                                 args, "appearance_full_ce_loss_coef", 0.0
                             ),
                             appearance_contrastive_loss_mode=getattr(
                                 args, "appearance_contrastive_loss_mode", "candidate_ce"
                             ),
                             appearance_pairwise_loss_margin=getattr(
                                 args, "appearance_pairwise_loss_margin", 0.0
                             ),
                             use_self_facb_loss=getattr(args, "use_self_facb_loss", False),
                             self_facb_topk=getattr(
                                 args,
                                 "self_facb_train_topk",
                                 getattr(args, "self_facb_topm", 10),
                             ),
                             self_facb_ce_aux_weight=getattr(
                                 args, "self_facb_ce_aux_weight", 0.5
                             ),
                             self_facb_pairwise_loss_weight=getattr(
                                 args, "self_facb_pairwise_loss_weight", 1.0
                             ),
                             self_facb_pairwise_loss_margin=getattr(
                                 args, "self_facb_pairwise_loss_margin", 0.0
                             ),
                             self_facb_force_gt_candidate=getattr(
                                 args, "self_facb_force_gt_candidate", True
                             ),
                             self_facb_train_box_source=getattr(
                                 args, "self_facb_train_box_source", "gt"
                             ),
                             self_facb_train_same_group_only=getattr(
                                 args, "self_facb_train_same_group_only", False
                             ),
                             self_facb_semantic_top1_loss_weight=getattr(
                                 args, "self_facb_semantic_top1_loss_weight", 0.0
                             ),
                             self_facb_semantic_top1_loss_margin=getattr(
                                 args, "self_facb_semantic_top1_loss_margin", 0.2
                             ),
                             self_facb_hard_case_weight=getattr(
                                 args, "self_facb_hard_case_weight", 0.0
                             ),
                             self_facb_hard_case_semantic_margin=getattr(
                                 args, "self_facb_hard_case_semantic_margin", 0.03
                             ),
                             self_facb_same_group_hard_weight=getattr(
                                 args, "self_facb_same_group_hard_weight", 0.0
                             ),
                             facb_symbol_align_topk=getattr(
                                 args, "facb_symbol_align_topk", 5
                             ),
                             facb_symbol_align_temperature=getattr(
                                 args, "facb_symbol_align_temperature", 0.07
                             ),
                             facb_symbol_align_pairwise_margin=getattr(
                                 args, "facb_symbol_align_pairwise_margin", 0.1
                             ),
                             facb_symbol_align_pairwise_weight=getattr(
                                 args, "facb_symbol_align_pairwise_weight", 0.5
                             ),
                             facb_symbol_align_inbatch_weight=getattr(
                                 args, "facb_symbol_align_inbatch_weight", 0.5
                             ),
                             facb_symbol_align_hard_weight=getattr(
                                 args, "facb_symbol_align_hard_weight", 1.0
                             ),
                             facb_symbol_align_easy_weight=getattr(
                                 args, "facb_symbol_align_easy_weight", 0.1
                             ),
                             facb_symbol_align_error_only=getattr(
                                 args, "facb_symbol_align_error_only", False
                             ),
                             use_facb_text_orth_loss=getattr(
                                 args, "use_facb_text_orth_loss", False
                             ),
                             facb_text_orth_margin=getattr(
                                 args, "facb_text_orth_margin", 0.85
                             ),
                             facb_text_orth_anchor_weight=getattr(
                                 args, "facb_text_orth_anchor_weight", 0.05
                             ),
                             )
    criterion.query_score_pos_iou_thr = float(
        getattr(args, "query_score_pos_iou_thr", 0.5)
    )
    criterion.query_score_neg_iou_thr = float(
        getattr(args, "query_score_neg_iou_thr", 0.1)
    )
    criterion.query_score_num_pos = int(
        getattr(args, "query_score_num_pos", 16)
    )
    criterion.query_score_num_neg = int(
        getattr(args, "query_score_num_neg", 32)
    )
    criterion.query_score_rank_margin = float(
        getattr(args, "query_score_rank_margin", 0.2)
    )
    criterion.lgpr_survival_aux_last_n = int(
        getattr(args, "lgpr_survival_aux_last_n", 2)
    )
    criterion.lgpr_survival_pos_iou_thr = float(
        getattr(args, "lgpr_survival_pos_iou_thr", 0.5)
    )
    criterion.lgpr_survival_small_pos_iou_thr = float(
        getattr(args, "lgpr_survival_small_pos_iou_thr", 0.3)
    )
    criterion.lgpr_survival_neg_iou_thr = float(
        getattr(args, "lgpr_survival_neg_iou_thr", 0.1)
    )
    criterion.lgpr_survival_num_hard_neg = int(
        getattr(args, "lgpr_survival_num_hard_neg", 64)
    )
    criterion.lgpr_survival_selection_topk = int(
        getattr(args, "lgpr_survival_selection_topk", args.num_select)
    )
    criterion.lgpr_survival_rank_margin = float(
        getattr(args, "lgpr_survival_rank_margin", 0.1)
    )
    criterion.lgpr_survival_temperature = float(
        getattr(args, "lgpr_survival_temperature", 0.2)
    )
    criterion.lgpr_survival_score_weight = float(
        getattr(args, "lgpr_survival_score_weight", 0.2)
    )
    criterion.lgpr_survival_objectness_coef = float(
        getattr(args, "lgpr_survival_objectness_coef", 0.25)
    )
    criterion.lgpr_survival_coverage_coef = float(
        getattr(args, "lgpr_survival_coverage_coef", 1.0)
    )
    criterion.lgpr_survival_small_gt_weight = float(
        getattr(args, "lgpr_survival_small_gt_weight", 2.0)
    )
    criterion.lgpr_survival_aux_decay = float(
        getattr(args, "lgpr_survival_aux_decay", 0.5)
    )
    criterion.lgpr_loc_aux_last_n = int(
        getattr(args, "lgpr_loc_aux_last_n", 2)
    )
    criterion.lgpr_loc_aux_decay = float(
        getattr(args, "lgpr_loc_aux_decay", 0.5)
    )
    criterion.lgpr_loc_pos_iou_thr = float(
        getattr(args, "lgpr_loc_pos_iou_thr", 0.5)
    )
    criterion.lgpr_loc_target_iou = float(
        getattr(args, "lgpr_loc_target_iou", 0.75)
    )
    criterion.lgpr_loc_max_pos_per_gt = int(
        getattr(args, "lgpr_loc_max_pos_per_gt", 3)
    )
    criterion.lgpr_loc_small_gt_weight = float(
        getattr(args, "lgpr_loc_small_gt_weight", 2.0)
    )
    criterion.lgpr_loc_lowfreq_gt_weight = float(
        getattr(args, "lgpr_loc_lowfreq_gt_weight", 1.0)
    )
    criterion.lgpr_loc_l1_coef = float(
        getattr(args, "lgpr_loc_l1_coef", 1.0)
    )
    criterion.lgpr_loc_giou_coef = float(
        getattr(args, "lgpr_loc_giou_coef", 1.0)
    )
    criterion.lgpr_loc_iou75_coef = float(
        getattr(args, "lgpr_loc_iou75_coef", 1.0)
    )
    criterion.lgpr_loc_lowfreq_label_ids = getattr(
        args, "lgpr_loc_lowfreq_label_ids", None
    )
    criterion.lgpr_box_refine_pos_iou_low = float(
        getattr(args, "lgpr_box_refine_pos_iou_low", 0.5)
    )
    criterion.lgpr_box_refine_pos_iou_high = float(
        getattr(args, "lgpr_box_refine_pos_iou_high", 0.85)
    )
    criterion.lgpr_box_refine_target_iou = float(
        getattr(args, "lgpr_box_refine_target_iou", 0.75)
    )
    criterion.lgpr_box_refine_max_pos_per_gt = int(
        getattr(args, "lgpr_box_refine_max_pos_per_gt", 2)
    )
    criterion.lgpr_box_refine_small_gt_weight = float(
        getattr(args, "lgpr_box_refine_small_gt_weight", 2.0)
    )
    criterion.lgpr_box_refine_lowfreq_gt_weight = float(
        getattr(args, "lgpr_box_refine_lowfreq_gt_weight", 1.0)
    )
    criterion.lgpr_box_refine_l1_coef = float(
        getattr(args, "lgpr_box_refine_l1_coef", 1.0)
    )
    criterion.lgpr_box_refine_giou_coef = float(
        getattr(args, "lgpr_box_refine_giou_coef", 1.0)
    )
    criterion.lgpr_box_refine_iou75_coef = float(
        getattr(args, "lgpr_box_refine_iou75_coef", 1.0)
    )
    criterion.lgpr_box_refine_lowfreq_label_ids = getattr(
        args, "lgpr_box_refine_lowfreq_label_ids", None
    )
    object.__setattr__(criterion, "self_facb_head_ref", model.self_facb_head)
    object.__setattr__(criterion, "facb_symbol_align_model_ref", model)
    criterion.self_facb_crm_mode = bool(
        getattr(args, "use_self_facb_crm", False)
    )
    criterion.self_facb_crm_min_score = float(
        getattr(args, "self_facb_crm_min_score", 0.02)
    )
    criterion.self_facb_crm_max_score = float(
        getattr(args, "self_facb_crm_max_score", 0.20)
    )
    criterion.self_facb_crm_max_margin = float(
        getattr(args, "self_facb_crm_max_margin", 0.10)
    )
    criterion.self_facb_crm_background_rows_per_image = int(
        getattr(args, "self_facb_crm_background_rows_per_image", 8)
    )
    criterion.self_facb_crm_background_max_iou = float(
        getattr(args, "self_facb_crm_background_max_iou", 0.3)
    )
    criterion.self_facb_crm_ce_weight = float(
        getattr(args, "self_facb_crm_ce_weight", 1.0)
    )
    criterion.self_facb_crm_pairwise_weight = float(
        getattr(args, "self_facb_crm_pairwise_weight", 1.0)
    )
    criterion.self_facb_crm_pairwise_margin = float(
        getattr(args, "self_facb_crm_pairwise_margin", 0.2)
    )
    criterion.self_facb_crm_residual_bound = float(
        getattr(args, "self_facb_crm_residual_bound", 0.5)
    )
    criterion.self_facb_crm_train_residual_bound = float(
        getattr(
            args,
            "self_facb_crm_train_residual_bound",
            getattr(args, "self_facb_crm_residual_bound", 0.5),
        )
    )
    criterion.self_facb_crm_easy_keep_weight = float(
        getattr(args, "self_facb_crm_easy_keep_weight", 0.5)
    )
    criterion.self_facb_crm_background_keep_weight = float(
        getattr(args, "self_facb_crm_background_keep_weight", 1.0)
    )
    criterion.self_facb_crm_ooc_keep_weight = float(
        getattr(args, "self_facb_crm_ooc_keep_weight", 1.0)
    )
    criterion.self_facb_crm_residual_reg_weight = float(
        getattr(args, "self_facb_crm_residual_reg_weight", 0.01)
    )
    criterion.self_facb_crm_ambiguous_supervision = bool(
        getattr(args, "self_facb_crm_ambiguous_supervision", False)
    )
    criterion.self_facb_crm_ambiguous_margin = float(
        getattr(args, "self_facb_crm_ambiguous_margin", 0.10)
    )
    criterion.self_facb_crm_ambiguous_min_score = float(
        getattr(args, "self_facb_crm_ambiguous_min_score", -1.0)
    )
    criterion.self_facb_crm_ambiguous_max_score = float(
        getattr(args, "self_facb_crm_ambiguous_max_score", -1.0)
    )
    criterion.self_facb_crm_hard_wrong_weight = float(
        getattr(args, "self_facb_crm_hard_wrong_weight", 1.0)
    )
    criterion.self_facb_crm_ambiguous_correct_weight = float(
        getattr(args, "self_facb_crm_ambiguous_correct_weight", 0.25)
    )
    criterion.self_facb_crm_keep_out_of_candidate = bool(
        getattr(args, "self_facb_crm_keep_out_of_candidate", False)
    )
    criterion.self_facb_lc_mode = bool(getattr(args, "use_self_facb_lc", False))
    _criterion_self_facb_levels = getattr(args, "self_facb_levels", [0])
    if isinstance(_criterion_self_facb_levels, int):
        _criterion_self_facb_levels = [_criterion_self_facb_levels]
    criterion.self_facb_levels = list(_criterion_self_facb_levels or [0])
    criterion.self_facb_roi_size = int(getattr(args, "self_facb_roi_size", 7))
    criterion.self_facb_lc_context_scale = float(
        getattr(args, "self_facb_lc_context_scale", 1.0)
    )
    criterion.self_facb_lc_train_class_delta_bound = float(
        getattr(
            args,
            "self_facb_lc_train_class_delta_bound",
            getattr(args, "self_facb_crm_train_residual_bound", 2.0),
        )
    )
    criterion.self_facb_lc_train_lift_bound = float(
        getattr(args, "self_facb_lc_train_lift_bound", 4.0)
    )
    criterion.self_facb_lc_score_floor = float(
        getattr(args, "self_facb_lc_score_floor", 0.10)
    )
    criterion.self_facb_lc_lift_weight = float(
        getattr(args, "self_facb_lc_lift_weight", 2.0)
    )
    criterion.self_facb_lc_lift_reg_weight = float(
        getattr(args, "self_facb_lc_lift_reg_weight", 0.001)
    )
    criterion.self_facb_lc_non_gt_keep_weight = float(
        getattr(args, "self_facb_lc_non_gt_keep_weight", 0.02)
    )
    criterion.use_facb_agf = bool(getattr(args, "use_facb_agf", False))
    criterion.facb_agf_alpha = float(getattr(args, "facb_agf_alpha", 3.0))
    criterion.facb_agf_min_extra_weight = float(
        getattr(args, "facb_agf_min_extra_weight", 0.0)
    )
    criterion.facb_agf_weight_cap = float(getattr(args, "facb_agf_weight_cap", 5.0))
    criterion.facb_agf_temperature = float(getattr(args, "facb_agf_temperature", 0.10))
    criterion.facb_agf_topk = int(getattr(args, "facb_agf_topk", getattr(args, "self_facb_train_topk", 10)))
    criterion.facb_agf_min_score = float(getattr(args, "facb_agf_min_score", 0.0))
    criterion.facb_agf_max_score = float(getattr(args, "facb_agf_max_score", 0.20))
    criterion.facb_agf_max_margin = float(getattr(args, "facb_agf_max_margin", 0.10))
    criterion.use_facb_text_orth_loss = bool(
        getattr(args, "use_facb_text_orth_loss", False)
    )
    criterion.facb_text_orth_model_ref = model
    criterion.facb_text_orth_label_list = list(
        getattr(args, "label_list", None)
        or getattr(args, "semantic_label_list", None)
        or []
    )
    criterion.facb_text_orth_margin = float(
        getattr(args, "facb_text_orth_margin", 0.85)
    )
    criterion.facb_text_orth_anchor_weight = float(
        getattr(args, "facb_text_orth_anchor_weight", 0.05)
    )
    criterion.facb_text_orth_max_pairs_per_group = int(
        getattr(args, "facb_text_orth_max_pairs_per_group", 64)
    )
    criterion.facb_text_orth_allowed_global_ids = getattr(
        args, "facb_text_orth_allowed_global_ids", None
    )
    criterion.facb_text_orth_allowed_values = getattr(
        args, "facb_text_orth_allowed_values", None
    )
    criterion.facb_text_orth_allowed_signatures = getattr(
        args, "facb_text_orth_allowed_signatures", None
    )
    criterion.use_facb_speedhr_loss = bool(
        getattr(args, "use_facb_speedhr_loss", False)
    )
    object.__setattr__(
        criterion,
        "facb_speedhr_head_ref",
        getattr(model, "facb_speedhr_head", None),
    )
    criterion.facb_speedhr_global_to_digit = getattr(
        model, "facb_speedhr_global_to_digit", None
    )
    criterion.facb_speedhr_shared_facet_class_targets = getattr(
        model, "facb_speedhr_shared_facet_class_targets", None
    )
    criterion.facb_speedhr_shared_facet_class_target_mask = getattr(
        model, "facb_speedhr_shared_facet_class_target_mask", None
    )
    criterion.facb_speedhr_crop_scale = float(
        getattr(args, "facb_speedhr_crop_scale", 1.2)
    )
    criterion.facb_speedhr_digit_loss_weight = float(
        getattr(args, "facb_speedhr_digit_loss_weight", 1.0)
    )
    criterion.facb_speedhr_shared_facet_loss_weight = float(
        getattr(args, "facb_speedhr_shared_facet_loss_weight", 1.0)
    )
    criterion.facb_speedhr_shared_facet_type_weights = list(
        getattr(
            args,
            "facb_speedhr_shared_facet_type_weights",
            [0.5, 0.5, 1.0, 1.0],
        )
    )
    criterion.facb_speedhr_shared_symbol_focal_gamma = float(
        getattr(args, "facb_speedhr_shared_symbol_focal_gamma", 0.0)
    )
    criterion.facb_speedhr_shared_symbol_bce = bool(
        getattr(args, "facb_speedhr_shared_symbol_bce", False)
    )
    criterion.facb_speedhr_shared_symbol_bce_neg_weight = float(
        getattr(args, "facb_speedhr_shared_symbol_bce_neg_weight", 0.05)
    )
    criterion.facb_speedhr_shared_symbol_hard_neg_weight = float(
        getattr(args, "facb_speedhr_shared_symbol_hard_neg_weight", 0.0)
    )
    criterion.facb_speedhr_shared_symbol_hard_neg_from_candidates = bool(
        getattr(args, "facb_speedhr_shared_symbol_hard_neg_from_candidates", False)
    )
    criterion.facb_speedhr_shared_facet_names = list(
        getattr(args, "facb_speedhr_shared_facet_names", None) or []
    )
    criterion.facb_speedhr_shared_facet_supervision_scope = str(
        getattr(args, "facb_speedhr_shared_facet_supervision_scope", "all")
        or "all"
    )
    criterion.facb_speedhr_agf_loss_weight = float(
        getattr(args, "facb_speedhr_agf_loss_weight", 1.0)
    )
    criterion.facb_speedhr_agf_alpha = float(
        getattr(args, "facb_speedhr_agf_alpha", 2.0)
    )
    criterion.facb_speedhr_agf_min_extra = float(
        getattr(args, "facb_speedhr_agf_min_extra", 0.0)
    )
    criterion.facb_speedhr_agf_weight_cap = float(
        getattr(args, "facb_speedhr_agf_weight_cap", 3.0)
    )
    criterion.facb_speedhr_agf_topk = int(
        getattr(args, "facb_speedhr_agf_topk", 10)
    )
    criterion.facb_speedhr_agf_min_score = float(
        getattr(args, "facb_speedhr_agf_min_score", 0.0)
    )
    criterion.facb_speedhr_agf_max_score = float(
        getattr(args, "facb_speedhr_agf_max_score", 0.20)
    )
    criterion.facb_speedhr_agf_max_margin = float(
        getattr(args, "facb_speedhr_agf_max_margin", 0.10)
    )
    criterion.facb_speedhr_agf_require_semantic_topk = bool(
        getattr(args, "facb_speedhr_agf_require_semantic_topk", True)
    )
    criterion.facb_speedhr_agf_require_crop_correct = bool(
        getattr(args, "facb_speedhr_agf_require_crop_correct", False)
    )
    criterion.facb_speedhr_agf_min_crop_prob = float(
        getattr(args, "facb_speedhr_agf_min_crop_prob", 0.0)
    )
    criterion.facb_speedhr_agf_min_crop_margin = float(
        getattr(args, "facb_speedhr_agf_min_crop_margin", -1.0)
    )
    criterion.facb_speedhr_agf_weight_mode = str(
        getattr(args, "facb_speedhr_agf_weight_mode", "prob")
    )
    criterion.facb_speedhr_candidate_matching = bool(
        getattr(args, "facb_speedhr_candidate_matching", False)
    )
    criterion.facb_speedhr_candidate_topk = int(
        getattr(args, "facb_speedhr_candidate_topk", 10)
    )
    criterion.facb_speedhr_candidate_smallshot_weight = float(
        getattr(args, "facb_speedhr_candidate_smallshot_weight", 5.0)
    )
    criterion.facb_speedhr_candidate_anchor_weight = float(
        getattr(args, "facb_speedhr_candidate_anchor_weight", 0.05)
    )
    criterion.facb_speedhr_candidate_smallshot_hard_weight = float(
        getattr(args, "facb_speedhr_candidate_smallshot_hard_weight", 10.0)
    )
    criterion.facb_speedhr_candidate_anchor_hard_weight = float(
        getattr(args, "facb_speedhr_candidate_anchor_hard_weight", 0.10)
    )
    criterion.facb_speedhr_candidate_query_source = str(
        getattr(args, "facb_speedhr_candidate_query_source", "matcher")
        or "matcher"
    )
    criterion.facb_speedhr_candidate_min_iou = float(
        getattr(args, "facb_speedhr_candidate_min_iou", 0.0)
    )
    criterion.facb_speedhr_candidate_crop_source = str(
        getattr(args, "facb_speedhr_candidate_crop_source", "pred") or "pred"
    )
    criterion.facb_speedhr_crop_pixel_source = str(
        getattr(args, "facb_speedhr_crop_pixel_source", "tensor") or "tensor"
    )
    criterion.facb_speedhr_score_residual_loss_weight = float(
        getattr(args, "facb_speedhr_score_residual_loss_weight", 0.0)
    )
    criterion.facb_speedhr_score_residual_floor = float(
        getattr(args, "facb_speedhr_score_residual_floor", 0.20)
    )
    criterion.facb_speedhr_score_residual_min_conf = float(
        getattr(args, "facb_speedhr_score_residual_min_conf", 0.80)
    )
    criterion.facb_speedhr_score_residual_min_margin = float(
        getattr(args, "facb_speedhr_score_residual_min_margin", 0.10)
    )
    criterion.facb_speedhr_score_residual_min_semantic_score = float(
        getattr(args, "facb_speedhr_score_residual_min_semantic_score", 0.0)
    )
    criterion.facb_speedhr_score_residual_max_semantic_score = float(
        getattr(args, "facb_speedhr_score_residual_max_semantic_score", 0.20)
    )
    criterion.facb_speedhr_score_residual_alpha_reg = float(
        getattr(args, "facb_speedhr_score_residual_alpha_reg", 0.001)
    )
    criterion.facb_speedhr_score_residual_alpha_cap = float(
        getattr(args, "facb_speedhr_score_residual_alpha_cap", 1.0)
    )
    criterion.facb_speedhr_score_residual_objective = str(
        getattr(args, "facb_speedhr_score_residual_objective", "floor")
        or "floor"
    )
    criterion.facb_speedhr_score_residual_rank_margin = float(
        getattr(args, "facb_speedhr_score_residual_rank_margin", 0.02)
    )
    criterion.facb_speedhr_score_residual_rank_temperature = float(
        getattr(args, "facb_speedhr_score_residual_rank_temperature", 1.0)
    )
    criterion.facb_speedhr_candidate_anchor_keep_prob = float(
        getattr(args, "facb_speedhr_candidate_anchor_keep_prob", 1.0)
    )
    criterion.facb_speedhr_score_residual_soft_gate = bool(
        getattr(args, "facb_speedhr_score_residual_soft_gate", False)
    )
    criterion.facb_speedhr_score_residual_gate_type = str(
        getattr(args, "facb_speedhr_score_residual_gate_type", "prob")
        or "prob"
    )
    criterion.facb_speedhr_score_residual_start_epoch = int(
        getattr(args, "facb_speedhr_score_residual_start_epoch", 0)
    )
    criterion.facb_speedhr_score_residual_ramp_epochs = int(
        getattr(args, "facb_speedhr_score_residual_ramp_epochs", 1)
    )
    _speedhr_allowed_ids = (
        getattr(args, "facb_speedhr_agf_allowed_global_ids", None)
        or getattr(args, "facb_speedhr_allowed_global_ids", None)
    )
    _speedhr_agf_scope = str(
        getattr(args, "facb_speedhr_agf_scope", "crop")
    ).strip().lower()
    if not _speedhr_allowed_ids and _speedhr_agf_scope in {"smallshot", "fewshot", "tail"}:
        _speedhr_allowed_ids = _facb_select_ids_from_split_meta(
            getattr(args, "facb_speedhr_split_meta_path", None)
            or getattr(args, "split_meta_path", None)
            or getattr(args, "category_meta_path", None),
            min_count=int(getattr(args, "facb_speedhr_agf_min_train_count", 1)),
            max_count=int(getattr(args, "facb_speedhr_agf_max_train_count", 50)),
            seen_only=False,
        )
    if _speedhr_allowed_ids:
        criterion.facb_speedhr_allowed_global_ids = torch.as_tensor(
            [int(x) for x in _speedhr_allowed_ids],
            dtype=torch.long,
        )
        if bool(getattr(args, "facb_speedhr_infer_unscoped_proposals", False)):
            # Pool-v2 proposal admission is detector-only.  Reusing the AGF
            # training scope here silently filters deployment queries before
            # P0/ITM can inspect them.
            model.facb_speedhr_infer_allowed_global_ids = None
            args.facb_speedhr_infer_allowed_global_ids = None
        else:
            model.facb_speedhr_infer_allowed_global_ids = [
                int(x) for x in _speedhr_allowed_ids
            ]
            args.facb_speedhr_infer_allowed_global_ids = list(
                model.facb_speedhr_infer_allowed_global_ids
            )
    else:
        criterion.facb_speedhr_allowed_global_ids = None
    _speedhr_allowed_values = getattr(args, "facb_speedhr_allowed_values", None)
    if _speedhr_allowed_values and getattr(model, "facb_speedhr_digit_values", None):
        _allowed_value_set = {str(x).strip() for x in _speedhr_allowed_values}
        _allowed_digit_ids = [
            i
            for i, value in enumerate(model.facb_speedhr_digit_values)
            if str(value).strip() in _allowed_value_set
        ]
        criterion.facb_speedhr_allowed_digit_ids = torch.as_tensor(
            _allowed_digit_ids,
            dtype=torch.long,
        )
    else:
        criterion.facb_speedhr_allowed_digit_ids = None
    print(
        "[facb-speedhr] agf_scope={} allowed_global_ids={}".format(
            _speedhr_agf_scope,
            0 if criterion.facb_speedhr_allowed_global_ids is None else int(criterion.facb_speedhr_allowed_global_ids.numel()),
        ),
        flush=True,
    )
    # Confuser construction mode for the pairwise bank loss:
    #   "sem_topk"    (default) — semantic top-k only (legacy behavior)
    #   "same_group"  — same-fine-group hard negatives only
    #   "union"       — union(sem_topk, same_group) deduped
    #   "same_group_topk" — same_group ∩ semantic_topk: the few same-group classes
    #                  the semantic head ranks highest.  Mirrors the eval metric
    #                  (GT vs semantic runner-up, same group) most closely.
    #   "same_group_then_sem_topk" — same-group negatives first, then semantic
    #                  top-k fills the remaining slots to keep enough negatives.
    criterion.appearance_confuser_mode = getattr(
        args, "appearance_confuser_mode", "sem_topk"
    )
    # Only train on queries the semantic head is NOT already confident about
    # (margin = sem_top1 - sem_top2 below this threshold).  1e9 = no filtering.
    # This concentrates the loss on the low-margin same-group queries the eval
    # metric actually scores, instead of being dominated by easy high-margin ones.
    criterion.appearance_pairwise_margin_filter_thr = float(
        getattr(args, "appearance_pairwise_margin_filter_thr", 1e9)
    )
    criterion.appearance_margin_weight_loss = bool(
        getattr(args, "appearance_margin_weight_loss", False)
    )
    criterion.appearance_margin_weight_temperature = float(
        getattr(args, "appearance_margin_weight_temperature", 0.2)
    )
    criterion.appearance_margin_weight_min = float(
        getattr(args, "appearance_margin_weight_min", 0.25)
    )
    criterion.appearance_force_semantic_runnerup = bool(
        getattr(args, "appearance_force_semantic_runnerup", False)
    )
    criterion.appearance_contrastive_loss_mode = str(
        getattr(args, "appearance_contrastive_loss_mode", "candidate_ce")
    ).strip().lower()
    criterion.appearance_pairwise_loss_margin = float(
        getattr(args, "appearance_pairwise_loss_margin", 0.0)
    )
    # Method B: same-group token-margin config.
    criterion.semantic_group_margin = float(getattr(args, "semantic_group_margin", 0.2))
    criterion.semantic_group_margin_topk = int(getattr(args, "semantic_group_margin_topk", 1))
    criterion.dgcl_contrastive_margin = float(
        getattr(args, "dgcl_contrastive_margin", 0.2)
    )
    criterion.dgcl_contrastive_topk = int(
        getattr(args, "dgcl_contrastive_topk", 5)
    )
    criterion.dgcl_contrastive_iou_threshold = float(
        getattr(args, "dgcl_contrastive_iou_threshold", 0.5)
    )
    criterion.dgcl_contrastive_ce_weight = float(
        getattr(args, "dgcl_contrastive_ce_weight", 1.0)
    )
    criterion.dgcl_contrastive_rank_weight = float(
        getattr(args, "dgcl_contrastive_rank_weight", 0.5)
    )
    criterion.dgcl_contrastive_logit_source = str(
        getattr(args, "dgcl_contrastive_logit_source", "raw")
    )
    criterion.dgcl_contrastive_hard_only = bool(
        getattr(args, "dgcl_contrastive_hard_only", False)
    )
    criterion.dgcl_contrastive_baseline_margin_threshold = float(
        getattr(args, "dgcl_contrastive_baseline_margin_threshold", 0.5)
    )
    criterion.dgcl_contrastive_preserve_weight = float(
        getattr(args, "dgcl_contrastive_preserve_weight", 0.0)
    )
    criterion.dgcl_contrastive_error_targeted = bool(
        getattr(args, "dgcl_contrastive_error_targeted", False)
    )
    criterion.dgcl_contrastive_preserve_correct = bool(
        getattr(args, "dgcl_contrastive_preserve_correct", False)
    )
    criterion.dgcl_contrastive_use_inference_fused_logits = bool(
        getattr(args, "dgcl_contrastive_use_inference_fused_logits", False)
    )
    criterion.dgcl_category_residual_weight = float(
        getattr(args, "dgcl_category_residual_weight", 1.0)
    )
    criterion.dgcl_category_residual_logit_space = bool(
        getattr(args, "dgcl_category_residual_logit_space", False)
    )
    criterion.dgcl_category_residual_match_baseline_distribution = bool(
        getattr(
            args,
            "dgcl_category_residual_match_baseline_distribution",
            False,
        )
    )
    criterion.dgcl_unmatched_preserve_iou_thr = float(
        getattr(args, "dgcl_unmatched_preserve_iou_thr", 0.3)
    )
    criterion.dgcl_unmatched_preserve_query_topk = int(
        getattr(args, "dgcl_unmatched_preserve_query_topk", 300)
    )
    criterion.dgcl_unmatched_preserve_topk = int(
        getattr(args, "dgcl_unmatched_preserve_topk", 10)
    )
    criterion.dgcl_contrastive_class_weights = (
        _build_dgcl_contrastive_class_weights(args)
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select  , text_encoder_type=args.text_encoder_type,nms_iou_threshold=args.nms_iou_threshold,args=args)}
    object.__setattr__(
        postprocessors["bbox"],
        "facb_speedhr_head",
        getattr(model, "facb_speedhr_head", None),
    )

    return model, criterion, postprocessors

def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """Construct a positive map from the deterministic caption layout."""

    rebuilt_caption, token_spans = build_caption_and_token_spans(cat_list)
    if caption is not None and caption != rebuilt_caption:
        caption = rebuilt_caption

    selected_token_spans = [token_spans[int(label)] for label in tokens_positive]
    positive_map = create_positive_map_from_span(
        tokenized,
        selected_token_spans,
        max_text_len=max_text_len,
    )
    return (positive_map > 0).to(dtype=torch.float)


def _greedy_chunk_cat_list(cat_list, tokenizer, max_text_len):
    """Greedy left-to-right packing so each chunk caption fits max_text_len.

    Mirrors engine._split_cat_list_into_chunks and PostProcess._chunk_cat_list_for_df
    so DF token spans match eval-time tokenization exactly.  Whole classes only.
    """
    if not cat_list:
        return [list(cat_list)]
    chunks = []
    i = 0
    n = len(cat_list)
    while i < n:
        chunk = [cat_list[i]]
        i += 1
        while i < n:
            candidate = chunk + [cat_list[i]]
            caption, _ = build_caption_from_labels(candidate)
            if len(tokenizer(caption)["input_ids"]) > max_text_len:
                break
            chunk = candidate
            i += 1
        chunks.append(chunk)
    return chunks


def compute_token_df_from_cat_list(tokenizer, full_cat_list, max_text_len):
    """Global token-id document frequency over the FULL class set.

    df[token_id] = number of classes whose caption span contains that token id.
    SHARED by PostProcess (eval scoring) and SetCriterion (matcher cost) so training
    and eval use the SAME DF table.  Keyed by token id (not surface string) so BPE
    space-prefix variants are exact.  Returns (df_dict, num_classes).
    """
    full_cat_list = list(full_cat_list)
    chunks = _greedy_chunk_cat_list(full_cat_list, tokenizer, max_text_len)
    df = {}
    N = len(full_cat_list)
    for chunk in chunks:
        caption, normalized = build_caption_from_labels(chunk)
        tokenized = tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        labels = torch.arange(len(normalized))
        binary = create_positive_map(
            tokenized, labels, normalized, caption, max_text_len=max_text_len
        )
        input_ids = tokenized["input_ids"][0].to(torch.long)
        L = min(binary.shape[1], input_ids.shape[0])
        for ci in range(binary.shape[0]):
            positions = (binary[ci, :L] > 0).nonzero(as_tuple=True)[0].tolist()
            seen_ids = {int(input_ids[p]) for p in positions}
            for tid in seen_ids:
                df[tid] = df.get(tid, 0) + 1
    return df, N


def idf_weight_label_map(
    binary_label_map, tokenized, token_df, num_classes, lam, gamma=1.0
):
    """Mix uniform and IDF token weights per class row, row-normalized.

        w = (1 - λ) * uniform_norm + λ * idf_norm

    binary_label_map: [C, max_text_len] 0/1 token membership per class.
    Returns a row-normalized weight map; classes with all-shared tokens (idf≡0)
    fall back to uniform so they are never zeroed (preserves cross-group ranking).
    Used by BOTH PostProcess scoring and the matcher class-cost so the two stay
    on the same weighting.
    """
    if lam == 0.0 or token_df is None:
        return binary_label_map
    device = binary_label_map.device
    # `tokenized` may be either a HF BatchEncoding (PostProcess: ["input_ids"] is a
    # [1, L] tensor) or a single tokenizers.Encoding (criterion's token[j]: has .ids).
    if hasattr(tokenized, "ids") and not hasattr(tokenized, "keys"):
        ids_list = list(tokenized.ids)
    else:
        _ids = tokenized["input_ids"]
        if _ids.dim() == 2:
            _ids = _ids[0]
        ids_list = _ids.to(torch.long).tolist()
    L = min(binary_label_map.shape[1], len(ids_list))
    N = max(int(num_classes), 1)
    gamma = max(float(gamma), 1e-6)
    idf_per_pos = torch.zeros(binary_label_map.shape[1], dtype=torch.float, device=device)
    for pos in range(L):
        df = max(int(token_df.get(int(ids_list[pos]), 1)), 1)
        idf_per_pos[pos] = max(math.log(N / df), 0.0) ** gamma
    idf_per_pos.clamp_(min=0.0)

    binary = (binary_label_map > 0).to(torch.float)
    uni_sum = binary.sum(dim=1, keepdim=True)
    uni_norm = torch.zeros_like(binary)
    uni_valid = uni_sum.squeeze(1) > 0
    uni_norm[uni_valid] = binary[uni_valid] / uni_sum[uni_valid]
    idf_map = binary * idf_per_pos.unsqueeze(0)
    idf_sum = idf_map.sum(dim=1, keepdim=True)
    idf_norm = torch.zeros_like(binary)
    idf_valid = idf_sum.squeeze(1) > 1e-6
    idf_norm[idf_valid] = idf_map[idf_valid] / idf_sum[idf_valid]
    idf_norm[~idf_valid] = uni_norm[~idf_valid]
    return (1.0 - lam) * uni_norm + lam * idf_norm


def build_caption_and_token_spans(cat_list):
    _, cat_list = build_caption_from_labels(cat_list)
    caption_parts = []
    token_spans = []
    cursor = 0

    for idx, category_text in enumerate(cat_list):
        if idx > 0:
            separator = " . "
            caption_parts.append(separator)
            cursor += len(separator)

        start = cursor
        caption_parts.append(category_text)
        cursor += len(category_text)
        token_spans.append([[start, cursor]])

    caption_parts.append(" .")
    caption = "".join(caption_parts)
    return caption, token_spans
