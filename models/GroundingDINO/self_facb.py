"""Detector-native FACB head.

Self-FACB scores semantic top-M candidates with detector backbone ROI tokens
and detector text prototypes.  It contains no per-class parameters and no
external vision-language tower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import CandidateConditionedLocalEvidence


class _ZeroInitializedLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class FACBVerificationDecoder(nn.Module):
    """Lightweight decoder-style verifier for structured FACB anchors.

    Each candidate owns F text-conditioned anchor queries.  The anchors first
    exchange information with each other, then cross-attend to ROI spatial tokens.
    Scoring remains class-agnostic: the same decoder weights are shared by all
    classes, and logits are cosine evidence between decoded anchors and their
    corresponding text prototypes.
    """

    def __init__(
        self,
        embed_dim,
        num_facets=2,
        logit_scale=10.0,
        num_layers=1,
        nheads=4,
        ffn_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_facets = max(int(num_facets), 1)
        self.num_layers = max(int(num_layers), 1)
        nheads = max(int(nheads), 1)
        ffn_dim = max(int(ffn_dim), self.embed_dim)
        self.facet_query = nn.Parameter(torch.zeros(self.num_facets, self.embed_dim))
        self.text_query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_score_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.embed_dim,
                    nheads,
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.embed_dim,
                    nheads,
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_self = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                    nn.Linear(ffn_dim, self.embed_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        logit_scale = max(float(logit_scale), 1e-6)
        self.log_logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale, dtype=torch.float32)))
        self.logit_clip = 20.0
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.facet_query, mean=0.0, std=0.02)
        for module in (self.text_query_proj, self.text_score_proj):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        for block in self.ffn:
            for module in block:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)

    def _facet_query_bias(self, num_facets, device, dtype):
        base = self.facet_query
        if base.shape[0] < num_facets:
            pad = base.new_zeros(num_facets - base.shape[0], base.shape[1])
            base = torch.cat([base, pad], dim=0)
        return base[:num_facets].to(device=device, dtype=dtype)

    def forward(self, roi_tokens, candidate_prototypes, candidate_mask=None):
        if roi_tokens.dim() != 3 or candidate_prototypes.dim() != 4:
            raise ValueError(
                "FACBVerificationDecoder expects roi_tokens [N,L,D] and "
                f"candidate_prototypes [N,K,F,D], got {tuple(roi_tokens.shape)} "
                f"and {tuple(candidate_prototypes.shape)}"
            )
        N, L, D = roi_tokens.shape
        K, Fct = candidate_prototypes.shape[1], candidate_prototypes.shape[2]
        if D != self.embed_dim or candidate_prototypes.shape[-1] != self.embed_dim:
            raise ValueError(
                f"embed dim mismatch: decoder={self.embed_dim}, "
                f"roi={D}, proto={candidate_prototypes.shape[-1]}"
            )

        proto = candidate_prototypes.reshape(N * K, Fct, D)
        query_bias = self._facet_query_bias(Fct, proto.device, proto.dtype).view(1, Fct, D)
        query = self.text_query_proj(proto) + query_bias
        memory = roi_tokens[:, None, :, :].expand(N, K, L, D).reshape(N * K, L, D)

        for layer_idx in range(self.num_layers):
            self_out, _ = self.self_attn[layer_idx](
                query=query,
                key=query,
                value=query,
                need_weights=False,
            )
            query = self.norm_self[layer_idx](query + self_out)
            cross_out, _ = self.cross_attn[layer_idx](
                query=query,
                key=memory,
                value=memory,
                need_weights=False,
            )
            query = self.norm_cross[layer_idx](query + cross_out)
            query = self.norm_ffn[layer_idx](query + self.ffn[layer_idx](query))

        score_proto = self.text_score_proj(proto)
        query_n = F.normalize(query, p=2, dim=-1, eps=1e-6)
        proto_n = F.normalize(score_proto, p=2, dim=-1, eps=1e-6)
        scores = (query_n * proto_n).sum(dim=-1)
        logit_scale = self.log_logit_scale.exp().clamp(max=self.logit_clip)
        scores = (scores * logit_scale).clamp(min=-self.logit_clip, max=self.logit_clip)
        scores = scores.view(N, K, Fct)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool().unsqueeze(-1), float("-inf"))
        return scores


class SelfFACBHead(nn.Module):
    """Candidate-conditioned verifier over ROI spatial tokens.

    Args:
        embed_dim: detector hidden dimension.
        logit_scale: initial cosine logit scale.

    Inputs:
        roi_tokens: [N, L, D] multi-scale ROI spatial tokens.
        candidate_prototypes: [N, K, D] text prototypes for semantic top-K.
        candidate_mask: optional [N, K] valid mask.

    Returns:
        [N, K] candidate evidence logits.
    """

    def __init__(
        self,
        embed_dim,
        logit_scale=10.0,
        text_proj_hidden_dim=512,
        num_facets=1,
        facet_temperature=2.0,
        facet_aggregation="softmin",
        scorer_type="local_evidence",
        decoder_layers=1,
        decoder_nheads=4,
        decoder_ffn_dim=1024,
        decoder_dropout=0.0,
        facet_weight_init=None,
        roi_size=7,
        max_levels=3,
        residual_bound=0.5,
        use_roi_position_level_embeddings=False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_facets = max(int(num_facets), 1)
        self.facet_temperature = float(facet_temperature)
        self.facet_aggregation = self._normalize_facet_aggregation(facet_aggregation)
        self.scorer_type = str(scorer_type or "local_evidence").strip().lower().replace("-", "_")
        if isinstance(roi_size, int):
            roi_size = (roi_size, roi_size)
        if not isinstance(roi_size, (tuple, list)) or len(roi_size) != 2:
            raise ValueError(f"roi_size must be an int or (height, width), got {roi_size!r}")
        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        if self.roi_size[0] <= 0 or self.roi_size[1] <= 0:
            raise ValueError(f"roi_size must be positive, got {self.roi_size}")
        self.max_levels = max(int(max_levels), 1)
        self.residual_bound = float(residual_bound)
        if self.residual_bound < 0.0:
            raise ValueError(f"residual_bound must be non-negative, got {self.residual_bound}")
        self.use_roi_position_level_embeddings = bool(use_roi_position_level_embeddings)
        hidden_dim = max(int(text_proj_hidden_dim), self.embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.text_proj_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.scorer = CandidateConditionedLocalEvidence(
            embed_dim=self.embed_dim,
            logit_scale=float(logit_scale),
        )
        self.decoder = FACBVerificationDecoder(
            embed_dim=self.embed_dim,
            num_facets=self.num_facets,
            logit_scale=float(logit_scale),
            num_layers=int(decoder_layers),
            nheads=int(decoder_nheads),
            ffn_dim=int(decoder_ffn_dim),
            dropout=float(decoder_dropout),
        )
        if self.facet_aggregation == "weighted":
            self.facet_weight_logits = nn.Parameter(torch.zeros(self.num_facets))
            if facet_weight_init is not None:
                init = torch.as_tensor(facet_weight_init, dtype=torch.float32).flatten()
                if init.numel() > 0:
                    if init.numel() < self.num_facets:
                        pad = init.new_full((self.num_facets - init.numel(),), 1.0)
                        init = torch.cat([init, pad], dim=0)
                    init = init[: self.num_facets].clamp(min=1e-6)
                    init = init / init.sum().clamp(min=1e-6)
                    with torch.no_grad():
                        self.facet_weight_logits.copy_(init.log())
        else:
            self.register_parameter("facet_weight_logits", None)
        roi_area = self.roi_size[0] * self.roi_size[1]
        self.roi_position_embedding = nn.Parameter(torch.zeros(roi_area, self.embed_dim))
        self.roi_level_embedding = nn.Parameter(torch.zeros(self.max_levels, self.embed_dim))
        self.residual_calibration_head = _ZeroInitializedLinear(self.num_facets + 1, 1)
        lc_feature_dim = self.num_facets + 5
        self.lc_class_delta_head = _ZeroInitializedLinear(self.num_facets + 1, 1)
        self.lc_lift_head = _ZeroInitializedLinear(lc_feature_dim, 1)
        self.lc_lift_confidence_head = _ZeroInitializedLinear(lc_feature_dim, 1)
        self._reset_text_proj()
        self._reset_residual_parameters()

    @staticmethod
    def _normalize_facet_aggregation(value):
        value = str(value or "softmin").strip().lower().replace("_", "-")
        aliases = {
            "soft-min": "softmin",
            "avg": "mean",
            "average": "mean",
            "learned": "weighted",
            "learnable": "weighted",
            "learnable-weight": "weighted",
            "learnable-weights": "weighted",
        }
        return aliases.get(value, value)

    def _reset_text_proj(self):
        for module in self.text_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def _reset_residual_parameters(self):
        nn.init.constant_(self.roi_position_embedding, 0.0)
        nn.init.constant_(self.roi_level_embedding, 0.0)
        nn.init.constant_(self.residual_calibration_head.weight, 0.0)
        nn.init.constant_(self.residual_calibration_head.bias, 0.0)
        for module in (
            self.lc_class_delta_head,
            self.lc_lift_head,
            self.lc_lift_confidence_head,
        ):
            nn.init.constant_(module.weight, 0.0)
            nn.init.constant_(module.bias, 0.0)

    def _add_roi_position_level_embeddings(self, roi_tokens):
        if not self.use_roi_position_level_embeddings:
            return roi_tokens
        if roi_tokens.dim() != 3:
            raise ValueError(f"roi_tokens must have shape [N,L,D], got {tuple(roi_tokens.shape)}")
        if roi_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"ROI token dim mismatch: head={self.embed_dim}, roi={roi_tokens.shape[-1]}"
            )

        tokens_per_level = self.roi_size[0] * self.roi_size[1]
        num_tokens = int(roi_tokens.shape[1])
        if num_tokens % tokens_per_level != 0:
            raise ValueError(
                f"ROI token count {num_tokens} is not divisible by roi_size area "
                f"{tokens_per_level} ({self.roi_size[0]}x{self.roi_size[1]})"
            )
        num_levels = num_tokens // tokens_per_level
        if num_levels > self.max_levels:
            raise ValueError(
                f"ROI tokens contain {num_levels} levels, exceeding max_levels={self.max_levels}"
            )

        position = self.roi_position_embedding.to(device=roi_tokens.device, dtype=roi_tokens.dtype)
        position = position.repeat(num_levels, 1)
        level = self.roi_level_embedding[:num_levels].to(
            device=roi_tokens.device,
            dtype=roi_tokens.dtype,
        )
        level = level.repeat_interleave(tokens_per_level, dim=0)
        return roi_tokens + position.unsqueeze(0) + level.unsqueeze(0)

    def _residual_features(self, raw_scores, facet_scores, candidate_mask):
        num_facets = int(facet_scores.shape[-1])
        if num_facets < self.num_facets:
            padding_shape = tuple(facet_scores.shape[:-1]) + (self.num_facets - num_facets,)
            padding = facet_scores.new_zeros(padding_shape)
            facet_scores = torch.cat([facet_scores, padding], dim=-1)
        else:
            facet_scores = facet_scores[..., : self.num_facets]

        if candidate_mask is None:
            candidate_mask = torch.ones_like(raw_scores, dtype=torch.bool)
        else:
            candidate_mask = candidate_mask.to(device=raw_scores.device, dtype=torch.bool)
            if candidate_mask.shape != raw_scores.shape:
                raise ValueError(
                    f"candidate_mask must have shape {tuple(raw_scores.shape)}, "
                    f"got {tuple(candidate_mask.shape)}"
                )
        safe_raw = raw_scores.masked_fill(~candidate_mask, 0.0)
        safe_facets = facet_scores.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        return torch.cat([safe_raw.unsqueeze(-1), safe_facets], dim=-1), candidate_mask

    def _aggregate_facet_scores(self, scores):
        if scores.shape[-1] == 1:
            return scores.squeeze(-1)

        mode = self.facet_aggregation
        if mode == "softmin":
            tau = max(float(self.facet_temperature), 1e-6)
            num_facets = max(int(scores.shape[-1]), 1)
            return -tau * torch.logsumexp(-scores / tau, dim=-1) + tau * math.log(num_facets)
        if mode == "mean":
            return scores.mean(dim=-1)
        if mode == "max":
            return scores.max(dim=-1).values
        if mode == "min":
            return scores.min(dim=-1).values
        if mode == "weighted":
            logits = self.facet_weight_logits
            if logits is None:
                logits = scores.new_zeros(scores.shape[-1])
            logits = logits.to(device=scores.device, dtype=scores.dtype)
            if logits.numel() < scores.shape[-1]:
                logits = torch.cat(
                    [logits, scores.new_zeros(scores.shape[-1] - logits.numel())],
                    dim=0,
                )
            logits = logits[: scores.shape[-1]]
            weights = torch.softmax(logits, dim=0)
            return (scores * weights.view(*([1] * (scores.dim() - 1)), -1)).sum(dim=-1)
        if mode.startswith("facet"):
            index_text = mode[len("facet"):].lstrip(":-")
            if not index_text:
                raise ValueError("facet aggregation mode must include an index, e.g. facet1")
            facet_idx = int(index_text)
            if facet_idx < 0 or facet_idx >= scores.shape[-1]:
                raise ValueError(
                    f"facet aggregation index {facet_idx} out of range for {scores.shape[-1]} facets"
                )
            return scores[..., facet_idx]
        raise ValueError(
            "Unsupported Self-FACB facet aggregation mode: "
            f"{self.facet_aggregation}. Expected softmin, mean, max, min, weighted, or facet<N>."
        )

    def get_facet_weights(self, num_facets=None):
        if self.facet_weight_logits is None:
            return None
        if num_facets is None:
            num_facets = self.num_facets
        num_facets = int(num_facets)
        logits = self.facet_weight_logits
        if logits.numel() < num_facets:
            logits = torch.cat(
                [logits, logits.new_zeros(num_facets - logits.numel())],
                dim=0,
            )
        logits = logits[:num_facets]
        return torch.softmax(logits, dim=0)

    def forward(self, roi_tokens, candidate_prototypes, candidate_mask=None, return_facet_scores=False):
        if candidate_prototypes.dim() == 4:
            residual = self.text_proj(candidate_prototypes)
            candidate_prototypes = candidate_prototypes + torch.tanh(self.text_proj_alpha) * residual
            if self.scorer_type == "decoder":
                scores = self.decoder(roi_tokens, candidate_prototypes, candidate_mask)
            else:
                facet_scores = []
                for facet_idx in range(candidate_prototypes.shape[2]):
                    proto_f = candidate_prototypes[:, :, facet_idx, :]
                    facet_scores.append(self.scorer(roi_tokens, proto_f, candidate_mask))
                scores = torch.stack(facet_scores, dim=-1)
            aggregated = self._aggregate_facet_scores(scores)
            if return_facet_scores:
                return aggregated, scores
            return aggregated

        residual = self.text_proj(candidate_prototypes)
        candidate_prototypes = candidate_prototypes + torch.tanh(self.text_proj_alpha) * residual
        scores = self.scorer(roi_tokens, candidate_prototypes, candidate_mask)
        if return_facet_scores:
            return scores, scores.unsqueeze(-1)
        return scores

    def forward_residual(
        self,
        roi_tokens,
        candidate_prototypes,
        candidate_mask=None,
        residual_bound=None,
    ):
        """Return identity-initialized, bounded residuals for candidate logits.

        This opt-in path may add learned ROI 2D position and feature-level
        embeddings, depending on ``use_roi_position_level_embeddings``.  The
        calibration head is shared by all candidates and initialized to zero,
        so the initial residual is exactly zero.  Centering is computed over
        valid candidates only and prevents a common score shift.

        Returns:
            residuals: [N, K] centered residuals bounded by ``residual_bound``.
                If the argument is omitted, the head's inference-time bound is
                used. Training may pass a larger bound while keeping inference
                conservative.
            raw_scores: [N, K] aggregated FACB evidence scores.
            facet_scores: [N, K, F] unaggregated FACB evidence scores.
        """
        roi_tokens = self._add_roi_position_level_embeddings(roi_tokens)
        raw_scores, facet_scores = self.forward(
            roi_tokens,
            candidate_prototypes,
            candidate_mask=candidate_mask,
            return_facet_scores=True,
        )
        calibration_features, valid_mask = self._residual_features(
            raw_scores,
            facet_scores,
            candidate_mask,
        )
        calibration_logits = self.residual_calibration_head(calibration_features).squeeze(-1)
        bounded = torch.tanh(calibration_logits).masked_fill(~valid_mask, 0.0)
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=bounded.dtype)
        centered = bounded - bounded.sum(dim=-1, keepdim=True) / valid_count
        bound = self.residual_bound if residual_bound is None else float(residual_bound)
        if bound < 0.0:
            raise ValueError(f"residual_bound must be non-negative, got {bound}")
        residuals = 0.5 * bound * centered
        residuals = residuals.masked_fill(~valid_mask, 0.0)
        return residuals, raw_scores, facet_scores

    def forward_lc(
        self,
        roi_tokens,
        candidate_prototypes,
        candidate_mask=None,
        score_features=None,
        class_delta_bound=None,
        lift_bound=4.0,
    ):
        """Return FACB-LC class deltas and non-negative candidate score lifts.

        ``class_delta`` is centered across valid candidates and starts at zero.
        ``lift`` is a separate non-negative head with zero output at
        initialization but non-zero gradient, so low-confidence score raising can
        be learned without forcing a global class-score shift.
        """
        roi_tokens = self._add_roi_position_level_embeddings(roi_tokens)
        raw_scores, facet_scores = self.forward(
            roi_tokens,
            candidate_prototypes,
            candidate_mask=candidate_mask,
            return_facet_scores=True,
        )
        class_features, valid_mask = self._residual_features(
            raw_scores,
            facet_scores,
            candidate_mask,
        )
        class_logits = self.lc_class_delta_head(class_features).squeeze(-1)
        class_bounded = torch.tanh(class_logits).masked_fill(~valid_mask, 0.0)
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(
            dtype=class_bounded.dtype
        )
        class_centered = class_bounded - class_bounded.sum(dim=-1, keepdim=True) / valid_count
        delta_bound = self.residual_bound if class_delta_bound is None else float(class_delta_bound)
        if delta_bound < 0.0:
            raise ValueError(f"class_delta_bound must be non-negative, got {delta_bound}")
        class_delta = (0.5 * delta_bound * class_centered).masked_fill(~valid_mask, 0.0)

        if score_features is None:
            score_features = class_features.new_zeros(*class_features.shape[:2], 4)
        else:
            score_features = score_features.to(device=class_features.device, dtype=class_features.dtype)
            if score_features.dim() != 3 or score_features.shape[:2] != class_features.shape[:2]:
                raise ValueError(
                    "score_features must have shape [N,K,S] matching candidates, "
                    f"got {tuple(score_features.shape)} for {tuple(class_features.shape[:2])}"
                )
            if score_features.shape[-1] < 4:
                pad = score_features.new_zeros(*score_features.shape[:2], 4 - score_features.shape[-1])
                score_features = torch.cat([score_features, pad], dim=-1)
            elif score_features.shape[-1] > 4:
                score_features = score_features[..., :4]

        lift_features = torch.cat([class_features, score_features.detach()], dim=-1)
        lift_logits = self.lc_lift_head(lift_features).squeeze(-1)
        lift_raw = F.softplus(lift_logits) - math.log(2.0)
        lift_scale = float(lift_bound)
        if lift_scale < 0.0:
            raise ValueError(f"lift_bound must be non-negative, got {lift_scale}")
        lift = (lift_scale * lift_raw).clamp(min=0.0, max=lift_scale)
        lift = lift.masked_fill(~valid_mask, 0.0)
        lift_confidence = torch.sigmoid(
            self.lc_lift_confidence_head(lift_features.detach()).squeeze(-1)
        ).masked_fill(~valid_mask, 0.0)
        return {
            "class_delta": class_delta,
            "lift": lift,
            "lift_confidence": lift_confidence,
            "raw_scores": raw_scores,
            "facet_scores": facet_scores,
            "candidate_mask": valid_mask,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path,
        map_location="cpu",
        embed_dim=None,
        logit_scale=10.0,
        text_proj_hidden_dim=512,
        num_facets=1,
        facet_temperature=2.0,
        facet_aggregation="softmin",
        scorer_type="local_evidence",
        decoder_layers=1,
        decoder_nheads=4,
        decoder_ffn_dim=1024,
        decoder_dropout=0.0,
        roi_size=7,
        max_levels=3,
        residual_bound=0.5,
        use_roi_position_level_embeddings=False,
    ):
        checkpoint = torch.load(path, map_location=map_location)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if embed_dim is None:
            for key, value in state.items():
                if key.endswith("scorer.q_proj.weight") and value.dim() == 2:
                    embed_dim = int(value.shape[0])
                    break
                if key.endswith("q_proj.weight") and value.dim() == 2:
                    embed_dim = int(value.shape[0])
                    break
        if embed_dim is None:
            raise ValueError(f"Cannot infer SelfFACBHead embed_dim from checkpoint: {path}")
        head = cls(
            embed_dim=embed_dim,
            logit_scale=logit_scale,
            text_proj_hidden_dim=text_proj_hidden_dim,
            num_facets=num_facets,
            facet_temperature=facet_temperature,
            facet_aggregation=facet_aggregation,
            scorer_type=scorer_type,
            decoder_layers=decoder_layers,
            decoder_nheads=decoder_nheads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_dropout=decoder_dropout,
            roi_size=roi_size,
            max_levels=max_levels,
            residual_bound=residual_bound,
            use_roi_position_level_embeddings=use_roi_position_level_embeddings,
        )
        missing, unexpected = head.load_state_dict(state, strict=False)
        allowed_missing = set()
        allowed_unexpected = set()
        if head.facet_weight_logits is not None:
            allowed_missing.add("facet_weight_logits")
        else:
            allowed_unexpected.add("facet_weight_logits")
        allowed_missing.update(key for key in missing if key.startswith("decoder."))
        residual_keys = {
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
        allowed_missing.update(key for key in missing if key in residual_keys)
        allowed_unexpected.update(key for key in unexpected if key.startswith("decoder."))
        missing = [key for key in missing if key not in allowed_missing]
        unexpected = [key for key in unexpected if key not in allowed_unexpected]
        if unexpected:
            raise RuntimeError(f"Unexpected SelfFACB checkpoint keys: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing SelfFACB checkpoint keys: {missing}")
        return head
