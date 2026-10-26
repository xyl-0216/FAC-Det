import math

import torch
from torch import nn
import torch.nn.functional as F


class QueryScoreSelector(nn.Module):
    """Text-conditioned proposal scorer for two-stage query initialization."""

    def __init__(
        self,
        d_model=256,
        num_queries=900,
        candidate_topk=0,
        hidden_dim=256,
        dropout=0.0,
        use_prior_score=True,
        blend_with_prior=True,
        alpha_init=0.5,
        selection_temperature=1.0,
        use_scale_adaptive=False,
        use_level_standardization=False,
        direct_temperature=1.0,
        context_temperature=1.0,
        scale_reference_area=0.0025,
        scale_beta=0.75,
        score_chunk_size=256,
        scale_lambda=4.0,
        scale_eps=1e-6,
        selection_gate_strength=1.0,
        use_prior_residual_fusion=False,
        prior_residual_weight=0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.candidate_topk = max(int(candidate_topk), 0)
        self.use_prior_score = use_prior_score
        self.blend_with_prior = blend_with_prior
        self.selection_temperature = max(float(selection_temperature), 1e-4)
        self.use_scale_adaptive = bool(use_scale_adaptive)
        self.use_level_standardization = bool(use_level_standardization)
        self.direct_temperature = max(float(direct_temperature), 1e-4)
        self.context_temperature = max(float(context_temperature), 1e-4)
        self.scale_reference_area = min(
            max(float(scale_reference_area), scale_eps), 1.0
        )
        self.scale_beta = max(float(scale_beta), 1e-4)
        self.score_chunk_size = max(int(score_chunk_size), 0)
        self.scale_lambda = max(float(scale_lambda), 1e-6)
        self.scale_eps = max(float(scale_eps), 1e-12)
        self.selection_gate_strength = max(float(selection_gate_strength), 0.0)
        self.use_prior_residual_fusion = bool(use_prior_residual_fusion)
        self.prior_residual_weight = min(max(float(prior_residual_weight), 0.0), 1.0)
        self.runtime_alpha_override = None
        self.runtime_route_with_selector = False

        self.image_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.context_proj = nn.Linear(d_model, d_model)

        score_input_dim = d_model * 4 + (1 if use_prior_score else 0)
        self.score_head = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Start from the pretrained joint embedding geometry. The projections can
        # adapt during LGPR training without replacing a strong baseline by a
        # random proposal ordering on the first step.
        self.reset_alignment_projections()

        if self.blend_with_prior:
            alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
            alpha_logit = math.log(alpha_init / (1.0 - alpha_init))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        else:
            self.register_parameter("alpha_logit", None)

    def reset_alignment_projections(self):
        for projection in (
            self.image_proj,
            self.text_proj,
            self.value_proj,
            self.context_proj,
        ):
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _masked_softmax(self, logits, mask):
        if mask is None:
            return torch.softmax(logits, dim=-1)

        masked_logits = logits.masked_fill(~mask[:, None, :], -1e4)
        return torch.softmax(masked_logits, dim=-1)

    def _build_text_context(self, image_features, text_features, text_token_mask):
        query = self.image_proj(self.image_norm(image_features))
        key = self.text_proj(self.text_norm(text_features))
        value = self.value_proj(text_features)
        scale = math.sqrt(query.shape[-1])
        attn_logits = torch.bmm(query, key.transpose(1, 2)) / scale
        attn = self._masked_softmax(attn_logits, text_token_mask)
        return torch.bmm(attn, value)

    def _for_each_candidate_chunk(self, image_features, function):
        if self.score_chunk_size <= 0 or image_features.shape[1] <= self.score_chunk_size:
            return function(image_features)
        chunks = []
        for start in range(0, image_features.shape[1], self.score_chunk_size):
            chunks.append(function(image_features[:, start : start + self.score_chunk_size]))
        return torch.cat(chunks, dim=1)

    def _compute_direct_scores(self, image_features, text_features, text_token_mask):
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-6)

        def score_chunk(image_chunk):
            image_chunk = F.normalize(image_chunk, p=2, dim=-1, eps=1e-6)
            logits = torch.bmm(image_chunk, text_features.transpose(1, 2))
            logits = logits / self.direct_temperature
            logits = logits.masked_fill(~text_token_mask[:, None, :], -1e4)
            return logits.max(dim=-1).values

        return self._for_each_candidate_chunk(image_features, score_chunk)

    def _compute_context(self, image_features, text_features, text_token_mask):
        return self._for_each_candidate_chunk(
            image_features,
            lambda image_chunk: self._build_text_context(
                image_chunk, text_features, text_token_mask
            ),
        )

    def _gather_candidates(self, tensor, candidate_indices):
        return torch.gather(
            tensor,
            1,
            candidate_indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]),
        )

    def _compute_contextual_scores(self, image_features, text_context):
        image_features = self.context_proj(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
        text_context = F.normalize(text_context, p=2, dim=-1, eps=1e-6)
        return (image_features * text_context).sum(dim=-1) / self.context_temperature

    def _compute_scale_weights(self, proposal_boxes):
        # Paper equation: alpha=sigmoid(beta*(log(rho_0)-log(rho))).
        wh = proposal_boxes[..., 2:].clamp(min=self.scale_eps, max=1.0)
        area = (wh[..., 0] * wh[..., 1]).clamp(min=self.scale_eps, max=1.0)
        reference = area.new_tensor(self.scale_reference_area)
        logits = self.scale_beta * (
            torch.log(reference) - torch.log(area + self.scale_eps)
        )
        return torch.sigmoid(logits)

    def _standardize_scores_by_level(self, scores, level_ids, valid_mask):
        normalized = scores.clone()
        if level_ids is None:
            for batch_idx in range(scores.shape[0]):
                mask = valid_mask[batch_idx]
                values = scores[batch_idx][mask]
                if values.numel() < 2:
                    continue
                normalized[batch_idx][mask] = (
                    values - values.mean()
                ) / values.std(unbiased=False).clamp_min(1e-6)
            return normalized

        for batch_idx in range(scores.shape[0]):
            batch_levels = level_ids[batch_idx]
            batch_valid = valid_mask[batch_idx]
            for level in torch.unique(batch_levels[batch_valid]):
                mask = batch_valid & (batch_levels == level)
                values = scores[batch_idx][mask]
                if values.numel() < 2:
                    continue
                normalized[batch_idx][mask] = (
                    values - values.mean()
                ) / values.std(unbiased=False).clamp_min(1e-6)
        return normalized

    def get_alpha(self):
        if self.runtime_alpha_override is not None:
            return torch.tensor(
                float(self.runtime_alpha_override),
                dtype=torch.float32,
                device=self.image_proj.weight.device,
            )
        if self.alpha_logit is None:
            return None
        return torch.sigmoid(self.alpha_logit)

    def set_runtime_alpha(self, alpha):
        self.runtime_alpha_override = None if alpha is None else float(alpha)

    def set_route_with_selector(self, enabled):
        self.runtime_route_with_selector = bool(enabled)

    def _build_selection_metadata(self, ranking_scores, selected_candidate_idx, valid_mask=None):
        if valid_mask is not None:
            safe_scores = ranking_scores.masked_fill(~valid_mask, -1e4)
        else:
            safe_scores = ranking_scores
        candidate_probs = torch.softmax(safe_scores / self.selection_temperature, dim=1)
        selected_soft_probs = torch.gather(candidate_probs, 1, selected_candidate_idx)
        selected_soft_weights = torch.gather(candidate_probs, 1, selected_candidate_idx)
        selected_soft_weights = selected_soft_weights / selected_soft_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        selected_soft_weights = selected_soft_weights * selected_soft_weights.shape[1]
        selection_gates = None
        if self.selection_gate_strength > 0.0:
            # Forward values stay exactly one. Only the surrogate gradient is
            # scaled; setting the strength to zero uses the paper's explicit
            # proposal-ranking loss without an additional straight-through path.
            selection_gates = 1.0 + self.selection_gate_strength * (
                selected_soft_weights - selected_soft_weights.detach()
            )
        return {
            "candidate_probs": candidate_probs,
            "selected_soft_probs": selected_soft_probs,
            "selected_soft_weights": selected_soft_weights,
            "selection_gates": selection_gates,
        }

    def forward(
        self,
        image_features,
        text_features,
        text_token_mask,
        heuristic_scores,
        valid_mask=None,
        proposal_boxes=None,
        proposal_level_ids=None,
    ):
        if valid_mask is None:
            valid_mask = torch.ones_like(heuristic_scores, dtype=torch.bool)

        candidate_indices = None
        candidate_valid_mask = valid_mask
        masked_scores = heuristic_scores.masked_fill(~valid_mask, -1e8)
        candidate_prior_scores = masked_scores
        candidate_features = image_features
        candidate_boxes = proposal_boxes
        candidate_level_ids = proposal_level_ids

        if self.candidate_topk > 0 and self.candidate_topk < masked_scores.shape[1]:
            candidate_prior_scores, candidate_indices = torch.topk(
                masked_scores, self.candidate_topk, dim=1
            )
            candidate_features = self._gather_candidates(image_features, candidate_indices)
            candidate_valid_mask = torch.gather(valid_mask, 1, candidate_indices)
            if proposal_boxes is not None:
                candidate_boxes = self._gather_candidates(proposal_boxes, candidate_indices)
            if proposal_level_ids is not None:
                candidate_level_ids = torch.gather(proposal_level_ids, 1, candidate_indices)

        direct_scores = self._compute_direct_scores(
            candidate_features, text_features, text_token_mask
        )
        text_context = self._compute_context(
            candidate_features, text_features, text_token_mask
        )
        contextual_scores = self._compute_contextual_scores(candidate_features, text_context)
        learned_scores = None
        scale_weights = None
        if self.use_scale_adaptive and candidate_boxes is not None:
            scale_weights = self._compute_scale_weights(candidate_boxes).to(
                device=direct_scores.device, dtype=direct_scores.dtype
            )
            context_scores = contextual_scores
            if self.use_level_standardization:
                direct_scores = self._standardize_scores_by_level(
                    direct_scores, candidate_level_ids, candidate_valid_mask
                )
                context_scores = self._standardize_scores_by_level(
                    context_scores, candidate_level_ids, candidate_valid_mask
                )
            lgpr_scores = (
                (1.0 - scale_weights) * direct_scores
                + scale_weights * context_scores
            )
            fusion_strength = self.get_alpha()
            if fusion_strength is None:
                fusion_strength = direct_scores.new_tensor(1.0)
            ranking_scores = (
                (1.0 - fusion_strength) * direct_scores
                + fusion_strength * lgpr_scores
            )
        else:
            feature_product = candidate_features * text_context
            feature_delta = candidate_features - text_context
            score_inputs = [
                self.image_norm(candidate_features),
                self.text_norm(text_context),
                feature_product,
                feature_delta,
            ]
            if self.use_prior_score:
                score_inputs.append(candidate_prior_scores.unsqueeze(-1))
            learned_scores = self.score_head(torch.cat(score_inputs, dim=-1)).squeeze(-1)

        if not self.use_scale_adaptive and self.blend_with_prior:
            alpha = self.get_alpha()
            ranking_scores = (
                (1.0 - alpha) * candidate_prior_scores + alpha * learned_scores
            )
        elif not self.use_scale_adaptive:
            ranking_scores = learned_scores
        prior_residual_prior_scores = None
        prior_residual_lgpr_scores = None
        if self.use_prior_residual_fusion:
            # Put both signals on the same per-image scale. At weight zero this
            # is exactly the baseline heuristic ordering; at one it is exactly
            # the learned LGPR ordering.
            prior_residual_prior_scores = self._standardize_scores_by_level(
                candidate_prior_scores, None, candidate_valid_mask
            )
            prior_residual_lgpr_scores = self._standardize_scores_by_level(
                ranking_scores, None, candidate_valid_mask
            )
            residual_weight = ranking_scores.new_tensor(self.prior_residual_weight)
            ranking_scores = (
                (1.0 - residual_weight) * prior_residual_prior_scores
                + residual_weight * prior_residual_lgpr_scores
            )

        ranking_scores = ranking_scores.masked_fill(~candidate_valid_mask, -1e8)

        topk = min(self.num_queries, ranking_scores.shape[1])
        selected_scores, selected_candidate_idx = torch.topk(ranking_scores, topk, dim=1)
        selection_metadata = self._build_selection_metadata(
            ranking_scores, selected_candidate_idx, valid_mask=candidate_valid_mask
        )

        if candidate_indices is not None:
            selected_indices = torch.gather(candidate_indices, 1, selected_candidate_idx)
        else:
            selected_indices = selected_candidate_idx

        selection_metadata["ranking_scores"] = ranking_scores
        selection_metadata["learned_scores"] = learned_scores
        selection_metadata["direct_scores"] = direct_scores
        selection_metadata["contextual_scores"] = contextual_scores
        selection_metadata["scale_weights"] = scale_weights
        selection_metadata["candidate_level_ids"] = candidate_level_ids
        selection_metadata["selected_scores"] = selected_scores
        selection_metadata["selected_candidate_indices"] = selected_candidate_idx
        selection_metadata["selected_indices"] = selected_indices
        selection_metadata["candidate_indices"] = candidate_indices
        selection_metadata["candidate_scores"] = direct_scores
        selection_metadata["candidate_prior_scores"] = candidate_prior_scores
        selection_metadata["prior_residual_prior_scores"] = prior_residual_prior_scores
        selection_metadata["prior_residual_lgpr_scores"] = prior_residual_lgpr_scores
        selection_metadata["prior_residual_weight"] = self.prior_residual_weight
        selection_metadata["candidate_valid_mask"] = candidate_valid_mask
        return selected_indices, selected_scores, selection_metadata
