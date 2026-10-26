# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def gen_encoder_output_proposals(
    memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
        -1, keepdim=True
    )
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class RandomBoxPerturber:
    def __init__(
        self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2
    ) -> None:
        self.noise_scale = torch.Tensor(
            [x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale]
        )

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SimilarityStarMLP(nn.Module):
    """GLU-based calibration on similarity matrix rows.

    Each query produces a similarity vector over text tokens.
    This module applies gated processing (GLU) to recalibrate the
    similarity profile, amplifying discriminative token matches
    and suppressing noise.
    """
    def __init__(self, input_dim, output_dim, width_factor=2, activation=None):
        super().__init__()
        hidden_dim = max(width_factor * input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation if activation is not None else nn.ReLU6()
        self.g = nn.Linear(hidden_dim, output_dim)
        # Zero-init output projection: with residual connection,
        # this ensures the calibration starts as identity (S + 0 = S).
        nn.init.zeros_(self.g.weight)
        nn.init.zeros_(self.g.bias)

    def forward(self, x):
        x = self.norm(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.activation(x1) * x2
        return self.g(x)


class SimilaritySiglipMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class SimilarityShareLockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        x = self.layers(x)
        return x.reshape(*original_shape[:-1], -1)


class ModalityStarMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width_factor=2, activation=None):
        super().__init__()
        hidden_dim = max(width_factor * input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation if activation is not None else nn.ReLU6()
        self.g = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.g.weight)
        nn.init.zeros_(self.g.bias)

    def forward(self, x):
        x = self.norm(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.activation(x1) * x2
        return self.g(x)


class SimilaritySigmoidGate(nn.Module):
    """Row-wise signed residual gating on the similarity matrix (DGCL Stage 2).

    For each query row, the gate observes the full token/class competition
    distribution and produces an element-wise gate G = σ(S · W_g + b_g).
    This makes the decision relative to the row context rather than to an
    absolute similarity threshold.  The final similarity uses signed residual
    refinement:

        S' = S + gamma * (2G - 1) * S

    Thus G > 0.5 increases the magnitude of a row-wise winner, G < 0.5
    attenuates runner-up noise, and G ≈ 0.5 keeps the baseline unchanged.
    This is a contrast-enhancement residual, not a pure suppressive gate.

    To keep the parameter count feasible with large max_text_len, the
    cross-category interaction is implemented as a low-rank bottleneck:
    S → down_proj → ReLU → up_proj → sigmoid.

    The up_proj bias is initialised at 0 so G starts at 0.5 and the residual
    gate is initially close to identity.  gate_strength controls the maximum
    residual magnitude.
    """

    def __init__(self, text_dim, bottleneck_dim=64, initial_bias=0.0, strength_init=0.1):
        super().__init__()
        self.down = nn.Linear(text_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, text_dim)
        strength_init = min(max(float(strength_init), 1e-4), 1.0 - 1e-4)
        self.gate_strength_logit = nn.Parameter(
            torch.logit(torch.tensor(strength_init, dtype=torch.float32))
        )
        # Zero-init up weight: the gate logit starts near 0 + bias.
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.up.bias, float(initial_bias))

    def forward(self, S):
        # S: [bs, nq, N_t]  row-wise similarity for each query over all text tokens
        gate = torch.sigmoid(self.up(F.relu(self.down(S))))
        signed_gate = 2.0 * gate - 1.0
        strength = torch.sigmoid(self.gate_strength_logit)
        return S + strength * signed_gate * S


class SAILStarMLP(nn.Module):
    """Official SAIL-style single-layer GLU/Star mapping."""

    def __init__(self, input_dim, output_dim, width_factor=8, activation=None):
        super().__init__()
        hidden_dim = max(int(width_factor) * input_dim, output_dim)
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation if activation is not None else nn.ReLU6()
        self.g = nn.Linear(hidden_dim, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x1 = self.f1(x).clamp(min=-1e3, max=1e3)
        x2 = self.f2(x).clamp(min=-1e3, max=1e3)
        x = self.activation(x1) * x2
        return self.g(x)


class SAILAlignmentHead(nn.Module):
    """Standalone SAIL alignment layer, separate from detector class_embed."""

    def __init__(
        self,
        embed_dim,
        target_dim=None,
        width_factor=8,
        logit_scale=20.0,
        logit_bias=-10.0,
    ):
        super().__init__()
        target_dim = int(target_dim or embed_dim)
        self.vision_layer_norm = nn.LayerNorm(embed_dim)
        self.text_layer_norm = nn.LayerNorm(embed_dim)
        self.vision_mapping_network = SAILStarMLP(
            input_dim=embed_dim,
            output_dim=target_dim,
            width_factor=width_factor,
        )
        self.text_mapping_network = SAILStarMLP(
            input_dim=embed_dim,
            output_dim=target_dim,
            width_factor=width_factor,
        )
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(float(logit_scale), dtype=torch.float32))
        )
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias), dtype=torch.float32))

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()

    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(self, image_features, text_features):
        image_features = self.vision_layer_norm(image_features)
        text_features = self.text_layer_norm(text_features)
        return {
            "image_features": self.vision_mapping_network(image_features),
            "text_features": self.text_mapping_network(text_features),
            "logit_scale": self.get_logit_scale,
            "logit_bias": self.get_logit_bias,
        }


class LegacySAILLastClassEmbed(nn.Module):
    """DGCL (Dual-Stage Gated Contrastive Learning) for the last decoder layer.

    Pipeline:
      Stage 1 — Modality-specific GLU refinement with residual (DGCL Eq. 1-2):
        Z_I = Q + A_I(LN(Q)),   Z_T = T + A_T(LN(T))
        where each A(·) is a GLU ↓8 with zero-init output projection W_o.
        At init W_o=0, so Z ≈ X (identity-like, no baseline degradation).

      Similarity computation (DGCL Eq. 3):
        S = normalize(Z_I) · normalize(Z_T)^T

      Stage 2 — Similarity-matrix signed residual gate (DGCL Eq. 4-5):
        G = σ(S · W_g + b_g),   S_gated = S + gamma · (2G - 1) · S
        W_g captures row-wise cross-category interactions; the gate performs
        contrast enhancement by strengthening row-wise winners and attenuating
        runner-up noise.

      Optional text prior gate:
        G_T = σ(MLP(Z_T)),   S_out = S_gated + eta · (2G_T - 1) · S_gated
        This provides a query-independent token/category reliability prior and
        is intended as a complement to the query-dependent row-wise gate.

    The result is blended into the main pred_logits via a learnable
    residual alpha.
    """

    def __init__(
        self,
        embed_dim=256,
        max_text_len=256,
        width_factor=8,
        logit_scale=20.0,
        logit_bias=-10.0,
        residual_alpha_init=0.2,
        use_similarity_gate=True,
        similarity_gate_bottleneck_dim=64,
        similarity_gate_initial_bias=-0.5,
        use_text_gate=False,
        text_gate_hidden_dim=None,
        text_gate_strength_init=0.05,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_text_len = int(max_text_len)
        self.use_similarity_gate = bool(use_similarity_gate)
        self.use_text_gate = bool(use_text_gate)

        # ---- Stage 1: GLU alignment with zero-init output + residual ----
        # ModalityStarMLP: GLU ↓8, ReLU6, W_o zero-initialised.
        # Residual: Z = X + A(LN(X)), at init A≈0 → Z≈X (identity-like start).
        self.query_layer_norm = nn.LayerNorm(self.embed_dim)
        self.text_layer_norm = nn.LayerNorm(self.embed_dim)
        self.query_projection = ModalityStarMLP(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            width_factor=width_factor,
        )
        self.text_projection = ModalityStarMLP(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            width_factor=width_factor,
        )

        # Kept in the signature for config/checkpoint compatibility.  DGCL uses
        # pure normalized dot-product similarity, matching the paper equation.
        self._unused_logit_scale_init = float(logit_scale)
        self._unused_logit_bias_init = float(logit_bias)

        # ---- Stage 2: similarity-matrix sigmoid gate (DGCL Eq. 4-5) ----
        if self.use_similarity_gate:
            self.similarity_gate = SimilaritySigmoidGate(
                text_dim=self.max_text_len,
                bottleneck_dim=similarity_gate_bottleneck_dim,
                initial_bias=similarity_gate_initial_bias,
            )
        else:
            self.similarity_gate = None

        # ---- Optional text-side prior gate ----
        # Query-independent: one signed residual prior per text token/category.
        # Final projection is zero-initialised, so G_T starts at 0.5 and the
        # branch is initially identity even when enabled.
        if self.use_text_gate:
            text_gate_hidden_dim = int(text_gate_hidden_dim or max(self.embed_dim // 2, 1))
            self.text_gate = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, text_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(text_gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.text_gate[-1].weight)
            nn.init.zeros_(self.text_gate[-1].bias)
            text_gate_strength_init = min(
                max(float(text_gate_strength_init), 1e-4), 1.0 - 1e-4
            )
            self.text_gate_strength_logit = nn.Parameter(
                torch.logit(torch.tensor(text_gate_strength_init, dtype=torch.float32))
            )
        else:
            self.text_gate = None
            self.text_gate_strength_logit = None

        # ---- Residual alpha (small positive → no zero-gradient deadlock) ----
        self.residual_alpha = nn.Parameter(
            torch.tensor(float(residual_alpha_init), dtype=torch.float32)
        )
        self.latest_gate_cache = {}

    @property
    def get_logit_scale(self):
        return 1.0

    def forward(self, x, text_dict):
        dgcl_flow_debug = (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_debug_printed", False)
        )
        cache_enabled = os.environ.get("GROUNDINGDINO_DGCL_GATE_CACHE", "0") == "1"
        self.latest_gate_cache = {}
        y = text_dict["encoded_text"]

        # ---- Stage 1: GLU alignment with zero-init residual (DGCL Eq. 1-2) ----
        query = x + self.query_projection(self.query_layer_norm(x))
        text = y + self.text_projection(self.text_layer_norm(y))
        if dgcl_flow_debug:
            print(
                "[dgcl-flow][SAIL] stage1 query.shape={} text.shape={}".format(
                    tuple(query.shape), tuple(text.shape)
                ),
                flush=True,
            )
        text_gate = None
        if self.text_gate is not None:
            text_gate = torch.sigmoid(self.text_gate(text).squeeze(-1))
            if dgcl_flow_debug:
                print(
                    "[dgcl-flow][SAIL] raw text_gate.shape={}".format(
                        tuple(text_gate.shape)
                    ),
                    flush=True,
                )

        # ---- Similarity computation (DGCL Eq. 3) ----
        query = F.normalize(query, p=2, dim=-1, eps=1e-6)
        text = F.normalize(text, p=2, dim=-1, eps=1e-6)
        logits = query @ text.transpose(-1, -2)
        raw_logits = logits
        if dgcl_flow_debug:
            print(
                "[dgcl-flow][SAIL] similarity logits.shape={}".format(tuple(logits.shape)),
                flush=True,
            )

        # ---- Stage 2: row-wise signed residual gate (Eq. 4-5) ----
        # Pad to max_text_len BEFORE gating so W_g sees the full text dim.
        pad_len = self.max_text_len - logits.shape[-1]
        if pad_len > 0:
            logits = F.pad(logits, (0, pad_len), value=float("-inf"))
        elif pad_len < 0:
            logits = logits[..., : self.max_text_len]

        text_token_mask = text_dict.get("text_token_mask", None)
        if text_token_mask is not None:
            mask = text_token_mask.to(device=logits.device, dtype=torch.bool)
        else:
            mask = torch.ones(logits.shape[0], logits.shape[-1], device=logits.device, dtype=torch.bool)

        # Pad mask to max_text_len so it matches the already-padded logits.
        if mask.shape[-1] < self.max_text_len:
            pad_mask = torch.zeros(
                mask.shape[0], self.max_text_len, device=mask.device, dtype=torch.bool
            )
            pad_mask[:, : mask.shape[-1]] = mask
            mask = pad_mask
        elif mask.shape[-1] > self.max_text_len:
            mask = mask[:, : self.max_text_len]

        if dgcl_flow_debug:
            print(
                "[dgcl-flow][SAIL] padded logits.shape={} mask.shape={}".format(
                    tuple(logits.shape), tuple(mask.shape)
                ),
                flush=True,
            )

        if self.similarity_gate is not None:
            # Feed valid positions as 0 (not -inf) so the gate MLP sees clean inputs.
            gate_input = logits.masked_fill(~mask[:, None, :], 0.0)
            gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=0.0, neginf=0.0)
            gate = torch.sigmoid(
                self.similarity_gate.up(F.relu(self.similarity_gate.down(gate_input)))
            )
            signed_gate = 2.0 * gate - 1.0
            gate_strength = torch.sigmoid(self.similarity_gate.gate_strength_logit)
            logits = self.similarity_gate(gate_input)
            if cache_enabled:
                raw_cache = raw_logits
                if raw_cache.shape[-1] < self.max_text_len:
                    raw_cache = F.pad(raw_cache, (0, self.max_text_len - raw_cache.shape[-1]), value=float("nan"))
                elif raw_cache.shape[-1] > self.max_text_len:
                    raw_cache = raw_cache[..., : self.max_text_len]
                self.latest_gate_cache.update(
                    {
                        "raw_similarity": raw_cache.detach().float().cpu(),
                        "similarity_gate": gate.detach().float().cpu(),
                        "signed_similarity_gate": signed_gate.detach().float().cpu(),
                        "similarity_gate_strength": float(gate_strength.detach().float().cpu()),
                    }
                )
            if dgcl_flow_debug:
                print(
                    "[dgcl-flow][SAIL] after self_gate logits.shape={}".format(
                        tuple(logits.shape)
                    ),
                    flush=True,
                )

        if text_gate is not None:
            if text_gate.shape[-1] < self.max_text_len:
                text_gate = F.pad(text_gate, (0, self.max_text_len - text_gate.shape[-1]), value=0.5)
            elif text_gate.shape[-1] > self.max_text_len:
                text_gate = text_gate[:, : self.max_text_len]
            signed_text_gate = 2.0 * text_gate - 1.0
            text_gate_strength = torch.sigmoid(self.text_gate_strength_logit)
            gate_base = logits.masked_fill(~mask[:, None, :], 0.0)
            gate_base = torch.nan_to_num(gate_base, nan=0.0, posinf=0.0, neginf=0.0)
            logits = gate_base + text_gate_strength * signed_text_gate[:, None, :] * gate_base
            if cache_enabled:
                self.latest_gate_cache.update(
                    {
                        "text_gate": text_gate.detach().float().cpu(),
                        "signed_text_gate": signed_text_gate.detach().float().cpu(),
                        "text_gate_strength": float(text_gate_strength.detach().float().cpu()),
                    }
                )
            if dgcl_flow_debug:
                print(
                    "[dgcl-flow][SAIL] text_gate.shape={} signed_text_gate[:,None,:].shape={} after text_gate logits.shape={}".format(
                        tuple(text_gate.shape),
                        tuple(signed_text_gate[:, None, :].shape),
                        tuple(logits.shape),
                    ),
                    flush=True,
                )

        # Restore -inf masking AFTER gating.
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        if cache_enabled:
            self.latest_gate_cache.update(
                {
                    "text_token_mask": mask.detach().bool().cpu(),
                    "final_logits": logits.detach().float().cpu(),
                }
            )
        if dgcl_flow_debug:
            finite_logits = logits[torch.isfinite(logits)]
            if finite_logits.numel() > 0:
                stats = (
                    float(finite_logits.mean().detach().cpu()),
                    float(finite_logits.std(unbiased=False).detach().cpu()),
                    float(finite_logits.min().detach().cpu()),
                    float(finite_logits.max().detach().cpu()),
                )
                print(
                    "[dgcl-flow][SAIL] final logits finite mean/std/min/max={:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                        *stats
                    ),
                    flush=True,
                )
            self._dgcl_flow_debug_printed = True

        return logits


class DGCLResidualAdapter(nn.Module):
    """Gated residual modality adapter from the DGCL formulation."""

    def __init__(self, embed_dim, hidden_dim, scale_init=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(scale_init), dtype=torch.float32)
        )

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        # Near-identity initialization without a zero-gradient deadlock.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        normalized = self.norm(x)
        hidden = F.gelu(self.gate_proj(normalized)) * self.value_proj(normalized)
        return x + self.residual_scale * self.out_proj(hidden)


class SAILLastClassEmbed(nn.Module):
    """Paper-aligned Dynamic Gated Contrastive Learning classification head.

    The historical class name is retained for config compatibility. The head
    implements residual modality adapters, a query-conditioned position-shared
    token gate, an inverse prompt-frequency prior, and category-wise normalized
    token aggregation. Category logits are broadcast to their phrase tokens so
    the existing matcher, focal loss, and positive-map postprocessor consume the
    same DGCL scores.
    """

    def __init__(
        self,
        embed_dim=256,
        max_text_len=256,
        width_factor=2,
        gate_hidden_dim=128,
        adapter_scale_init=0.1,
        temperature_init=0.2,
        use_frequency_prior=True,
        non_category_logit=-10.0,
        use_baseline_residual=True,
        output_blend_init=0.1,
        match_baseline_distribution=True,
        use_margin_gate=False,
        margin_gate_tau=1.0,
        margin_gate_gamma=0.2,
        use_word_cocontribution=False,
        word_cocontribution_similarity_scale_init=0.0,
        word_cocontribution_completeness_init=0.95,
        word_cocontribution_discriminative_init=0.0,
        **_legacy_kwargs,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_text_len = int(max_text_len)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.use_frequency_prior = bool(use_frequency_prior)
        self.non_category_logit = float(non_category_logit)
        self.use_baseline_residual = bool(use_baseline_residual)
        self.match_baseline_distribution = bool(match_baseline_distribution)
        self.use_margin_gate = bool(use_margin_gate)
        self.margin_gate_tau = float(margin_gate_tau)
        self.margin_gate_gamma = max(float(margin_gate_gamma), 1e-4)
        self.use_word_cocontribution = bool(use_word_cocontribution)

        adapter_hidden_dim = max(int(width_factor) * self.embed_dim, 1)
        self.query_adapter = DGCLResidualAdapter(
            self.embed_dim, adapter_hidden_dim, scale_init=adapter_scale_init
        )
        self.text_adapter = DGCLResidualAdapter(
            self.embed_dim, adapter_hidden_dim, scale_init=adapter_scale_init
        )

        self.query_gate_proj = nn.Linear(self.embed_dim, self.gate_hidden_dim, bias=False)
        self.text_gate_proj = nn.Linear(self.embed_dim, self.gate_hidden_dim, bias=False)
        self.frequency_vector = nn.Parameter(torch.zeros(self.gate_hidden_dim))
        self.gate_bias = nn.Parameter(torch.zeros(self.gate_hidden_dim))
        self.gate_score = nn.Linear(self.gate_hidden_dim, 1, bias=False)
        self.word_cocontribution_similarity_scale = nn.Parameter(
            torch.tensor(float(word_cocontribution_similarity_scale_init))
        )
        self.word_cocontribution_discriminative_scale = nn.Parameter(
            torch.tensor(float(word_cocontribution_discriminative_init))
        )
        word_cocontribution_completeness_init = min(
            max(float(word_cocontribution_completeness_init), 1e-4),
            1.0 - 1e-4,
        )
        self.word_cocontribution_completeness_init = (
            word_cocontribution_completeness_init
        )
        self.word_cocontribution_completeness = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 1),
        )
        nn.init.xavier_uniform_(self.query_gate_proj.weight)
        nn.init.xavier_uniform_(self.text_gate_proj.weight)
        nn.init.normal_(self.frequency_vector, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.gate_score.weight)
        nn.init.constant_(self.word_cocontribution_completeness[-1].weight, 0.0)
        nn.init.constant_(
            self.word_cocontribution_completeness[-1].bias,
            float(torch.logit(torch.tensor(word_cocontribution_completeness_init))),
        )

        temperature_init = max(float(temperature_init), 1e-4)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(temperature_init, dtype=torch.float32))
        )
        if self.use_baseline_residual:
            output_blend_init = min(max(float(output_blend_init), 1e-4), 1.0 - 1e-4)
            self.output_blend_logit = nn.Parameter(
                torch.logit(torch.tensor(output_blend_init, dtype=torch.float32))
            )
            self.output_query_gate = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, 1),
            )
            nn.init.constant_(self.output_query_gate[-1].weight, 0.0)
            nn.init.constant_(self.output_query_gate[-1].bias, 0.0)
            self.output_residual_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("output_blend_logit", None)
            self.output_query_gate = None
            self.register_parameter("output_residual_log_scale", None)
        self.latest_gate_cache = {}

    def get_output_blend_alpha(self):
        if self.output_blend_logit is None:
            return None
        return torch.sigmoid(self.output_blend_logit)

    @staticmethod
    def _match_valid_score_distribution(scores, reference, valid_token_mask):
        valid = valid_token_mask[:, None, :].expand_as(scores)
        score_values = torch.nan_to_num(
            scores, nan=0.0, posinf=0.0, neginf=0.0
        ).masked_fill(~valid, 0.0)
        reference_values = torch.nan_to_num(
            reference, nan=0.0, posinf=0.0, neginf=0.0
        ).masked_fill(~valid, 0.0)
        count = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1)

        score_mean = score_values.sum(dim=(1, 2), keepdim=True) / count
        reference_mean = reference_values.sum(dim=(1, 2), keepdim=True) / count
        score_var = (
            ((score_values - score_mean) ** 2).masked_fill(~valid, 0.0)
        ).sum(dim=(1, 2), keepdim=True) / count
        reference_var = (
            ((reference_values - reference_mean) ** 2).masked_fill(~valid, 0.0)
        ).sum(dim=(1, 2), keepdim=True) / count

        calibrated = (
            (score_values - score_mean)
            / score_var.sqrt().clamp_min(1e-4)
            * reference_var.sqrt().clamp_min(1e-4)
            + reference_mean
        )
        return calibrated.masked_fill(~valid, float("-inf"))

    def blend_with_baseline(
        self,
        dgcl_logits,
        baseline_logits,
        text_token_mask,
        query_features=None,
        category_token_masks=None,
    ):
        if not self.use_baseline_residual:
            return dgcl_logits
        valid_token_mask = text_token_mask.to(
            device=dgcl_logits.device, dtype=torch.bool
        )
        target_len = dgcl_logits.shape[-1]
        if valid_token_mask.shape[-1] < target_len:
            valid_token_mask = F.pad(
                valid_token_mask,
                (0, target_len - valid_token_mask.shape[-1]),
                value=False,
            )
        elif valid_token_mask.shape[-1] > target_len:
            valid_token_mask = valid_token_mask[:, :target_len]
        if self.match_baseline_distribution:
            calibrated_dgcl = self._match_valid_score_distribution(
                dgcl_logits, baseline_logits, valid_token_mask
            )
        else:
            calibrated_dgcl = dgcl_logits.masked_fill(
                ~valid_token_mask[:, None, :], float("-inf")
            )
        valid = valid_token_mask[:, None, :].expand_as(dgcl_logits)
        baseline_values = torch.nan_to_num(
            baseline_logits, nan=0.0, posinf=0.0, neginf=0.0
        )
        dgcl_values = torch.nan_to_num(
            calibrated_dgcl, nan=0.0, posinf=0.0, neginf=0.0
        )
        base_alpha_logit = self.output_blend_logit.to(
            device=dgcl_logits.device, dtype=dgcl_logits.dtype
        )
        if query_features is not None and self.output_query_gate is not None:
            query_gate_delta = self.output_query_gate(
                query_features.to(device=dgcl_logits.device, dtype=dgcl_logits.dtype)
            )
            alpha = torch.sigmoid(base_alpha_logit + query_gate_delta)
        else:
            alpha = torch.sigmoid(base_alpha_logit).view(1, 1, 1)
        margin_gate = None
        if self.use_margin_gate:
            margin_gate = self._compute_baseline_margin_gate(
                baseline_logits=baseline_logits,
                valid_token_mask=valid_token_mask,
                category_token_masks=category_token_masks,
                target_len=target_len,
            ).to(device=dgcl_logits.device, dtype=dgcl_logits.dtype)
            alpha = alpha * margin_gate
        residual_scale = self.output_residual_log_scale.exp().clamp(
            min=0.25, max=4.0
        ).to(device=dgcl_logits.device, dtype=dgcl_logits.dtype)
        blended = baseline_values + alpha * residual_scale * (
            dgcl_values - baseline_values
        )
        if (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_blend_debug_printed", False)
        ):
            valid_baseline = baseline_values[valid]
            valid_dgcl = dgcl_values[valid]
            valid_blended = blended[valid]
            valid_delta = (valid_blended - valid_baseline).abs()
            token_top1_changed = (
                blended.masked_fill(~valid, float("-inf")).argmax(dim=-1)
                != baseline_values.masked_fill(~valid, float("-inf")).argmax(dim=-1)
            )
            alpha_values = alpha.detach().float()
            if margin_gate is None:
                margin_values = torch.ones_like(alpha_values)
            else:
                margin_values = margin_gate.detach().float()
            print(
                "[dgcl-flow][blend] alpha_mean/std/min/max={:.4f}/{:.4f}/{:.4f}/{:.4f} "
                "margin_gate_mean/std/min/max={:.4f}/{:.4f}/{:.4f}/{:.4f} "
                "residual_scale={:.4f} baseline_mean/std={:.4f}/{:.4f} "
                "calibrated_dgcl_mean/std={:.4f}/{:.4f} "
                "blend_delta_mean/max={:.6f}/{:.6f} token_top1_changed={:.4f}".format(
                    float(alpha_values.mean().cpu()),
                    float(alpha_values.std(unbiased=False).cpu()),
                    float(alpha_values.min().cpu()),
                    float(alpha_values.max().cpu()),
                    float(margin_values.mean().cpu()),
                    float(margin_values.std(unbiased=False).cpu()),
                    float(margin_values.min().cpu()),
                    float(margin_values.max().cpu()),
                    float(residual_scale.detach().float().cpu()),
                    float(valid_baseline.mean().detach().float().cpu()),
                    float(valid_baseline.std(unbiased=False).detach().float().cpu()),
                    float(valid_dgcl.mean().detach().float().cpu()),
                    float(valid_dgcl.std(unbiased=False).detach().float().cpu()),
                    float(valid_delta.mean().detach().float().cpu()),
                    float(valid_delta.max().detach().float().cpu()),
                    float(token_top1_changed.float().mean().detach().cpu()),
                ),
                flush=True,
            )
            self._dgcl_blend_debug_printed = True
        if os.environ.get("GROUNDINGDINO_DGCL_GATE_CACHE", "0") == "1":
            alpha_detached = alpha.detach().float().cpu()
            self.latest_gate_cache["output_blend_alpha_mean"] = float(
                alpha_detached.mean()
            )
            self.latest_gate_cache["output_blend_alpha_std"] = float(
                alpha_detached.std(unbiased=False)
            )
            self.latest_gate_cache["output_residual_scale"] = float(
                residual_scale.detach().float().cpu()
            )
            if margin_gate is not None:
                margin_detached = margin_gate.detach().float().cpu()
                self.latest_gate_cache["output_margin_gate_mean"] = float(
                    margin_detached.mean()
                )
                self.latest_gate_cache["output_margin_gate_std"] = float(
                    margin_detached.std(unbiased=False)
                )
        return blended.masked_fill(~valid, float("-inf"))

    def _compute_baseline_margin_gate(
        self,
        baseline_logits,
        valid_token_mask,
        category_token_masks=None,
        target_len=None,
    ):
        batch_size, num_queries, logit_len = baseline_logits.shape
        text_len = int(target_len or logit_len)
        gates = []
        with torch.no_grad():
            baseline_values = torch.nan_to_num(
                baseline_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            for batch_idx in range(batch_size):
                valid_tokens = valid_token_mask[batch_idx, :text_len].to(
                    device=baseline_logits.device, dtype=torch.bool
                )
                category_mask = None
                if category_token_masks is not None and batch_idx < len(
                    category_token_masks
                ):
                    category_mask = category_token_masks[batch_idx]
                category_mask = self._prepare_category_mask(
                    category_mask, valid_tokens, text_len
                )
                if category_mask is not None and category_mask.shape[0] >= 2:
                    membership = category_mask.to(
                        device=baseline_logits.device, dtype=baseline_logits.dtype
                    )
                    class_scores = (
                        baseline_values[batch_idx, :, :text_len]
                        @ membership.transpose(0, 1)
                    ) / membership.sum(dim=-1).clamp_min(1.0).unsqueeze(0)
                    top2 = torch.topk(class_scores.float(), k=2, dim=-1).values
                    margin = top2[:, 0] - top2[:, 1]
                else:
                    token_scores = baseline_values[batch_idx, :, :text_len].masked_fill(
                        ~valid_tokens.unsqueeze(0), float("-inf")
                    )
                    if valid_tokens.sum() >= 2:
                        top2 = torch.topk(token_scores.float(), k=2, dim=-1).values
                        margin = top2[:, 0] - top2[:, 1]
                    else:
                        margin = token_scores.new_full((num_queries,), 1e4)
                gate = torch.sigmoid(
                    (self.margin_gate_tau - margin) / self.margin_gate_gamma
                )
                gates.append(gate.view(num_queries, 1))
        return torch.stack(gates, dim=0)

    @staticmethod
    def _prepare_category_mask(category_mask, valid_token_mask, text_len):
        if category_mask is None or category_mask.numel() == 0:
            return None
        category_mask = category_mask.to(
            device=valid_token_mask.device, dtype=torch.bool
        )
        if category_mask.dim() != 2:
            raise ValueError(
                "DGCL category masks must be [num_categories, num_tokens], got "
                f"{tuple(category_mask.shape)}"
            )
        category_mask = category_mask[:, :text_len]
        if category_mask.shape[-1] < text_len:
            category_mask = F.pad(
                category_mask, (0, text_len - category_mask.shape[-1]), value=False
            )
        category_mask = category_mask & valid_token_mask[:text_len].unsqueeze(0)
        valid_categories = category_mask.any(dim=-1)
        return category_mask[valid_categories]

    def forward(self, x, text_dict, category_token_masks=None):
        encoded_text = text_dict["encoded_text"]
        text_token_mask = text_dict.get("text_token_mask")
        if text_token_mask is None:
            text_token_mask = torch.ones(
                encoded_text.shape[:2], device=encoded_text.device, dtype=torch.bool
            )
        else:
            text_token_mask = text_token_mask.to(
                device=encoded_text.device, dtype=torch.bool
            )

        query = F.normalize(self.query_adapter(x), p=2, dim=-1, eps=1e-6)
        text = F.normalize(self.text_adapter(encoded_text), p=2, dim=-1, eps=1e-6)
        temperature = self.log_temperature.exp().clamp(min=0.05, max=10.0)
        similarity = (query @ text.transpose(-1, -2)) / temperature

        batch_size, num_queries = x.shape[:2]
        text_len = min(encoded_text.shape[1], self.max_text_len)
        token_logits = x.new_full(
            (batch_size, num_queries, self.max_text_len), float("-inf")
        )
        cache_enabled = os.environ.get("GROUNDINGDINO_DGCL_GATE_CACHE", "0") == "1"
        debug_enabled = (
            os.environ.get("GROUNDINGDINO_DGCL_FLOW_DEBUG", "0") == "1"
            and not getattr(self, "_dgcl_flow_debug_printed", False)
        )
        self.latest_gate_cache = {}
        cached_gate = []
        cached_prior = []
        cached_class_logits = []
        category_logits_for_output = []
        category_token_gate_for_output = []
        category_counts = []

        for batch_idx in range(batch_size):
            valid_tokens = text_token_mask[batch_idx, :text_len]
            if category_token_masks is None or batch_idx >= len(category_token_masks):
                category_mask = None
            else:
                category_mask = category_token_masks[batch_idx]
            category_mask = self._prepare_category_mask(
                category_mask, valid_tokens, text_len
            )

            if category_mask is None or category_mask.shape[0] == 0:
                raise ValueError(
                    "DGCL requires category-to-token masks for every prompt; "
                    f"sample {batch_idx} has none"
                )

            num_categories = category_mask.shape[0]
            category_counts.append(int(num_categories))
            input_ids = text_dict.get("input_ids")
            if input_ids is None:
                raise ValueError("DGCL frequency prior requires text_dict['input_ids']")
            sample_token_ids = input_ids[batch_idx, :text_len].to(
                device=category_mask.device
            )
            # df_t counts category phrases containing the same lexical token,
            # irrespective of where that token occurs in the concatenated prompt.
            document_frequency = category_mask.new_zeros(
                text_len, dtype=torch.float32
            )
            for category_idx in range(num_categories):
                category_token_ids = torch.unique(
                    sample_token_ids[category_mask[category_idx]]
                )
                same_lexical_token = (
                    sample_token_ids[:, None] == category_token_ids[None, :]
                ).any(dim=-1)
                document_frequency = document_frequency + same_lexical_token.float()
            if self.use_frequency_prior:
                frequency_prior = torch.log(
                    (document_frequency.new_tensor(float(num_categories)) + 1.0)
                    / (document_frequency + 1.0)
                )
                frequency_prior = frequency_prior / torch.log(
                    document_frequency.new_tensor(float(num_categories)) + 1.0
                ).clamp_min(1e-6)
            else:
                frequency_prior = torch.zeros_like(document_frequency)

            query_gate = self.query_gate_proj(query[batch_idx])[:, None, :]
            text_gate = self.text_gate_proj(text[batch_idx, :text_len])[None, :, :]
            prior_gate = frequency_prior[None, :, None] * self.frequency_vector
            gate_hidden = torch.tanh(
                query_gate + text_gate + prior_gate + self.gate_bias
            )
            importance_logits = self.gate_score(gate_hidden).squeeze(-1)

            class_logit_columns = []
            normalized_token_gate = importance_logits.new_zeros(
                importance_logits.shape
            )
            if self.use_word_cocontribution:
                phrase_token_indices = []
                phrase_token_valid = []
                max_phrase_tokens = 1
                for category_idx in range(num_categories):
                    token_indices = category_mask[category_idx].nonzero(
                        as_tuple=False
                    ).flatten()
                    phrase_token_indices.append(token_indices)
                    max_phrase_tokens = max(
                        max_phrase_tokens, int(token_indices.numel())
                    )
                padded_phrase_token_indices = []
                for token_indices in phrase_token_indices:
                    valid_count = int(token_indices.numel())
                    if valid_count < max_phrase_tokens:
                        token_indices = F.pad(
                            token_indices,
                            (0, max_phrase_tokens - valid_count),
                            value=0,
                        )
                    padded_phrase_token_indices.append(token_indices)
                    phrase_token_valid.append(
                        torch.arange(
                            max_phrase_tokens,
                            device=category_mask.device,
                        )
                        < valid_count
                    )
                phrase_indices = torch.stack(padded_phrase_token_indices, dim=0)
                phrase_valid = torch.stack(phrase_token_valid, dim=0)
                phrase_similarity = similarity[
                    batch_idx, :, phrase_indices
                ]  # [Q, C, K]
                phrase_gate_logits = importance_logits[
                    :, phrase_indices
                ].float()
                phrase_gate_logits = phrase_gate_logits.masked_fill(
                    ~phrase_valid.unsqueeze(0), -1e4
                )
                similarity_scale = (
                    self.word_cocontribution_similarity_scale.clamp(
                        min=-2.0, max=2.0
                    )
                    .to(device=phrase_similarity.device, dtype=torch.float32)
                )
                phrase_gate_logits = (
                    phrase_gate_logits
                    + similarity_scale * phrase_similarity.float()
                ).masked_fill(~phrase_valid.unsqueeze(0), -1e4)
                phrase_gate = torch.softmax(
                    phrase_gate_logits, dim=-1
                ).to(dtype=similarity.dtype)
                phrase_logit = (phrase_gate * phrase_similarity).sum(dim=-1)

                phrase_similarity_float = phrase_similarity.float()
                phrase_valid_float = phrase_valid.to(
                    device=phrase_similarity.device, dtype=torch.float32
                ).unsqueeze(0)
                phrase_lengths = phrase_valid_float.sum(dim=-1).clamp_min(1.0)
                phrase_mean = (
                    phrase_similarity_float * phrase_valid_float
                ).sum(dim=-1) / phrase_lengths
                phrase_max = phrase_similarity_float.masked_fill(
                    ~phrase_valid.unsqueeze(0), -1e4
                ).max(dim=-1).values
                phrase_std = (
                    (
                        (
                            phrase_similarity_float
                            - phrase_mean.unsqueeze(-1)
                        )
                        ** 2
                        * phrase_valid_float
                    ).sum(dim=-1)
                    / phrase_lengths
                ).sqrt()
                entropy = -(
                    phrase_gate.float()
                    * phrase_gate.float().clamp_min(1e-6).log()
                    * phrase_valid_float
                ).sum(dim=-1)
                entropy_norm = phrase_lengths.squeeze(0).log().clamp_min(1.0)
                entropy = entropy / entropy_norm.unsqueeze(0)
                entropy = entropy.masked_fill(phrase_lengths <= 1.0, 0.0)
                completeness_features = torch.stack(
                    [phrase_mean, phrase_max, phrase_std, entropy], dim=-1
                )
                completeness = torch.sigmoid(
                    self.word_cocontribution_completeness(
                        completeness_features.to(
                            device=x.device,
                            dtype=self.word_cocontribution_completeness[
                                -1
                            ].weight.dtype,
                        )
                    ).squeeze(-1)
                )
                completeness = (
                    completeness / self.word_cocontribution_completeness_init
                ).clamp(min=0.5, max=1.5)
                discriminative_scale = (
                    self.word_cocontribution_discriminative_scale.clamp(
                        min=0.0, max=2.0
                    )
                    .to(device=phrase_logit.device, dtype=phrase_logit.dtype)
                )
                discriminative_delta = (
                    phrase_max - phrase_mean
                ).to(device=phrase_logit.device, dtype=phrase_logit.dtype)
                class_logits = phrase_logit * completeness.to(
                    device=phrase_logit.device, dtype=phrase_logit.dtype
                ) + discriminative_scale * discriminative_delta
                normalized_token_gate.scatter_add_(
                    1,
                    phrase_indices.reshape(1, -1).expand(
                        importance_logits.shape[0], -1
                    ),
                    (
                        phrase_gate
                        * phrase_valid.to(
                            device=phrase_gate.device, dtype=phrase_gate.dtype
                        ).unsqueeze(0)
                    ).reshape(importance_logits.shape[0], -1),
                )
            else:
                # Phrase spans are short and disjoint. Computing [Q,C,T] would waste
                # memory on almost entirely masked entries, so normalize each phrase
                # over only its member tokens while preserving the exact equation.
                for category_idx in range(num_categories):
                    token_indices = category_mask[category_idx].nonzero(
                        as_tuple=False
                    ).flatten()
                    phrase_similarity = similarity[batch_idx, :, token_indices]
                    phrase_gate_logits = importance_logits[:, token_indices].float()
                    phrase_gate = torch.softmax(
                        phrase_gate_logits, dim=-1
                    ).to(dtype=similarity.dtype)
                    phrase_logit = (
                        phrase_gate
                        * phrase_similarity
                    ).sum(dim=-1)
                    class_logit_columns.append(phrase_logit)
                    normalized_token_gate[:, token_indices] = phrase_gate
                class_logits = torch.stack(class_logit_columns, dim=-1)

            # Caption phrase spans are disjoint. Division keeps custom prompts
            # with intentionally overlapping spans mathematically well-defined.
            membership = category_mask.to(dtype=class_logits.dtype)
            token_scores = class_logits @ membership
            overlap_count = membership.sum(dim=0).clamp(min=1.0)
            token_scores = token_scores / overlap_count.unsqueeze(0)
            covered_tokens = category_mask.any(dim=0)
            token_scores = token_scores.masked_fill(
                ~covered_tokens.unsqueeze(0), self.non_category_logit
            )
            token_logits[batch_idx, :, :text_len] = token_scores
            token_logits[batch_idx, :, :text_len] = token_logits[
                batch_idx, :, :text_len
            ].masked_fill(~valid_tokens.unsqueeze(0), float("-inf"))
            category_logits_for_output.append(class_logits)
            if not self.training:
                gate_for_output = x.new_zeros(
                    (normalized_token_gate.shape[0], self.max_text_len)
                )
                gate_len = min(int(normalized_token_gate.shape[-1]), self.max_text_len)
                if gate_len > 0:
                    gate_for_output[:, :gate_len] = normalized_token_gate[
                        :, :gate_len
                    ].to(device=x.device, dtype=x.dtype)
                category_token_gate_for_output.append(gate_for_output.detach())

            if cache_enabled:
                cached_gate.append(normalized_token_gate.detach().float().cpu())
                cached_prior.append(frequency_prior.detach().float().cpu())
                cached_class_logits.append(class_logits.detach().float().cpu())

        self.latest_gate_cache["category_logits_tensor"] = category_logits_for_output
        if category_token_gate_for_output:
            self.latest_gate_cache[
                "category_token_gate_tensor"
            ] = category_token_gate_for_output

        if cache_enabled:
            self.latest_gate_cache.update(
                {
                    "category_token_gate": cached_gate,
                    "token_frequency_prior": cached_prior,
                    "category_logits": cached_class_logits,
                    "text_token_mask": text_token_mask.detach().bool().cpu(),
                    "final_logits": token_logits.detach().float().cpu(),
                    "temperature": float(temperature.detach().float().cpu()),
                }
            )

        if debug_enabled:
            finite_logits = token_logits[torch.isfinite(token_logits)]
            print(
                "[dgcl-flow] query={} text={} categories={} temperature={:.4f} "
                "finite_mean={:.4f} finite_std={:.4f}".format(
                    tuple(query.shape),
                    tuple(text.shape),
                    category_counts,
                    float(temperature.detach().cpu()),
                    float(finite_logits.mean().detach().cpu()),
                    float(finite_logits.std(unbiased=False).detach().cpu()),
                ),
                flush=True,
            )
            self._dgcl_flow_debug_printed = True

        return token_logits


class ModalitySiglipMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        return self.proj(self.norm(x))


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class ContrastiveEmbed(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        max_text_len=256,
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
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_text_len = max_text_len
        self.use_similarity_calibration = use_similarity_calibration
        self.similarity_calibration_type = similarity_calibration_type
        self.similarity_calibration_residual = similarity_calibration_residual
        self.similarity_calibration_mode = str(similarity_calibration_mode).strip().lower()
        self.class_logit_norm = bool(class_logit_norm)
        self.class_logit_bias = float(class_logit_bias)
        self.class_logit_bias_param = nn.Parameter(
            torch.tensor(float(class_logit_bias), dtype=torch.float32)
        )
        self.use_modality_projection = bool(use_modality_projection)
        self.modality_projection_type = modality_projection_type
        self.modality_projection_hidden_dim = modality_projection_hidden_dim
        self.modality_projection_dropout = float(modality_projection_dropout)
        self.modality_projection_width_factor = max(int(modality_projection_width_factor), 1)
        self.modality_projection_residual = bool(modality_projection_residual)
        class_logit_scale = max(float(class_logit_scale), 1e-6)
        self.log_class_logit_scale = nn.Parameter(
            torch.log(torch.tensor(class_logit_scale, dtype=torch.float32))
        )
        # Keep contrastive logits in a numerically safe range during early training.
        self.logit_clip = 20.0
        self.calibration_delta_clip = 10.0
        self.latest_stats = {}
        self.gate_scale = None
        self.gate_logit_scale = None
        self.gate_logit_bias = None

        if self.use_similarity_calibration:
            if self.similarity_calibration_mode == "residual_gate":
                self.similarity_head = None
                self.gate_scale = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32))
                self.gate_logit_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
                self.gate_logit_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            else:
                hidden_dim = (
                    similarity_calibration_hidden_dim
                    if similarity_calibration_hidden_dim is not None
                    else 2 * max_text_len
                )
                self.similarity_head = self._build_similarity_head(
                    similarity_calibration_type,
                    input_dim=max_text_len,
                    hidden_dim=hidden_dim,
                    dropout=similarity_calibration_dropout,
                )
        else:
            self.similarity_head = None
        if self.use_modality_projection:
            self.query_projection = self._build_modality_projection(self.embed_dim)
            self.text_projection = self._build_modality_projection(self.embed_dim)
        else:
            self.query_projection = None
            self.text_projection = None

    def _build_modality_projection(self, input_dim):
        proj_type = str(self.modality_projection_type).strip().lower()
        if proj_type == "star":
            return ModalityStarMLP(
                input_dim=input_dim,
                output_dim=input_dim,
                width_factor=self.modality_projection_width_factor,
            )
        if proj_type == "siglip":
            return ModalitySiglipMLP(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_dim=self.modality_projection_hidden_dim,
                dropout=self.modality_projection_dropout,
            )
        if proj_type == "linear":
            return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim))
        raise ValueError(f"Unsupported modality_projection_type: {proj_type}")

    def _build_similarity_head(self, head_type, input_dim, hidden_dim, dropout):
        head_type = head_type.lower()
        if head_type == "linear":
            return nn.Linear(input_dim, input_dim)
        if head_type == "star":
            width_factor = max(2, math.ceil(hidden_dim / input_dim))
            return SimilarityStarMLP(
                input_dim=input_dim,
                output_dim=input_dim,
                width_factor=width_factor,
            )
        if head_type == "siglip":
            return SimilaritySiglipMLP(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        if head_type == "sharelock":
            return SimilarityShareLockMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                dropout=dropout,
            )
        raise ValueError(f"Unsupported similarity_calibration_type: {head_type}")

    def _update_stats(self, raw_logits, final_logits, delta_logits, valid_mask):
        with torch.no_grad():
            if not valid_mask.any():
                self.latest_stats = {}
                return
            def _masked_mean_std(values):
                expanded_mask = valid_mask.expand_as(values)
                masked_values = values.masked_select(expanded_mask)
                if masked_values.numel() == 0:
                    return 0.0, 0.0
                mean = masked_values.mean()
                std = masked_values.std(unbiased=False)
                return mean.item(), std.item()

            raw_mean, raw_std = _masked_mean_std(raw_logits)
            final_mean, final_std = _masked_mean_std(final_logits)
            stats = {
                "similarity_raw_mean": raw_mean,
                "similarity_raw_std": raw_std,
                "similarity_logit_mean": final_mean,
                "similarity_logit_std": final_std,
            }
            if delta_logits is not None:
                delta_mean, delta_std = _masked_mean_std(delta_logits)
                stats["similarity_delta_mean"] = delta_mean
                stats["similarity_delta_std"] = delta_std
            self.latest_stats = stats

    def forward(self, x, text_dict):
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]
        valid_mask = text_token_mask[:, None, :]

        if self.use_modality_projection:
            x_proj = self.query_projection(x)
            y_proj = self.text_projection(y)
            if self.modality_projection_residual:
                x = x + x_proj
                y = y + y_proj
            else:
                x = x_proj
                y = y_proj

        if self.class_logit_norm:
            x_for_logits = F.normalize(x, p=2, dim=-1, eps=1e-6)
            y_for_logits = F.normalize(y, p=2, dim=-1, eps=1e-6)
            class_logit_scale = self.log_class_logit_scale.exp().clamp(max=self.logit_clip)
            raw_res = (
                class_logit_scale
                * (x_for_logits @ y_for_logits.transpose(-1, -2))
                + self.class_logit_bias_param
            )
        else:
            raw_res = x @ y.transpose(-1, -2)
        raw_res = torch.nan_to_num(raw_res, nan=0.0, posinf=0.0, neginf=0.0)
        masked_res = raw_res.masked_fill(~valid_mask, float("-inf"))
        final_res = masked_res

        delta_logits = None
        if self.similarity_head is not None:
            if self.similarity_calibration_residual:
                calibration_input = raw_res.masked_fill(~valid_mask, 0.0)
                calibrated_logits = self.similarity_head(calibration_input)
                calibrated_logits = torch.nan_to_num(
                    calibrated_logits, nan=0.0, posinf=0.0, neginf=0.0
                )
                delta_logits = calibrated_logits.clamp(
                    min=-self.calibration_delta_clip, max=self.calibration_delta_clip
                )
                final_res = calibration_input + delta_logits
            else:
                calibration_input = raw_res.masked_fill(~valid_mask, 0.0)
                calibrated_logits = self.similarity_head(calibration_input)
                calibrated_logits = torch.nan_to_num(
                    calibrated_logits, nan=0.0, posinf=0.0, neginf=0.0
                )
                delta_logits = calibrated_logits - calibration_input
                final_res = calibrated_logits
            final_res = final_res.masked_fill(~valid_mask, float("-inf"))
        elif self.use_similarity_calibration and self.similarity_calibration_mode == "residual_gate":
            calibration_input = raw_res.masked_fill(~valid_mask, 0.0)
            gate_logits = calibration_input
            if self.gate_logit_scale is not None:
                gate_logits = gate_logits * self.gate_logit_scale
            if self.gate_logit_bias is not None:
                gate_logits = gate_logits + self.gate_logit_bias
            gate = 2.0 * torch.sigmoid(gate_logits) - 1.0
            gate_scale = (
                torch.sigmoid(self.gate_scale) if self.gate_scale is not None else 1.0
            )
            delta_logits = (
                gate_scale * calibration_input * gate
            ).clamp(min=-self.calibration_delta_clip, max=self.calibration_delta_clip)
            final_res = (raw_res + delta_logits).masked_fill(~valid_mask, float("-inf"))

        final_res = torch.nan_to_num(
            final_res,
            nan=0.0,
            posinf=self.logit_clip,
            neginf=-self.logit_clip,
        )
        final_res = final_res.clamp(min=-self.logit_clip, max=self.logit_clip)
        final_res = final_res.masked_fill(~valid_mask, float("-inf"))

        padded_logits = torch.full(
            (*final_res.shape[:-1], self.max_text_len),
            float("-inf"),
            device=final_res.device,
        )
        padded_logits[..., : final_res.shape[-1]] = final_res

        if self.use_similarity_calibration:
            self._update_stats(
                raw_logits=raw_res,
                final_logits=final_res.masked_fill(~valid_mask, 0.0),
                delta_logits=delta_logits,
                valid_mask=valid_mask,
            )
        else:
            self.latest_stats = {}

        return padded_logits

    def forward_against_bank(self, x, bank, bank_mask=None):
        """Score queries against a precomputed [C, D] class-prototype bank.

        Mirrors the norm / scale / bias / modality-projection logic of forward(),
        but contrasts each query against a fixed set of C global class prototypes
        instead of the per-image padded text tokens.  This is the prototype-bank
        path for the appearance head: every matched query is scored against ALL
        classes (or a fine-group subset, handled by the caller), so hard negatives
        come from the real confusers (pl40 vs pl50/pl60) rather than whatever
        classes happen to share the image.

        Args:
            x:    [bs, nq, D] visual queries (already detached by the caller).
            bank: [C, D] class prototypes (text features through feat_map, pooled).
                  The text_projection still trains, so it is applied here each step
                  even though `bank` itself is a frozen-BERT buffer.
            bank_mask: optional [C] bool, False entries get -inf (e.g. empty texts).

        Returns:
            [bs, nq, C] class scores.  No -inf padding to max_text_len, no
            positive-map aggregation needed downstream.
        """
        if bank.dim() != 2:
            raise ValueError(f"bank must be [C, D], got shape {tuple(bank.shape)}")

        # text_projection is a trainable D->D transform; apply it to the pooled
        # prototypes so the bank path shares exactly the parameters trained on
        # the token path.  query_projection mirrors the semantic-query side.
        if self.use_modality_projection:
            x_proj = self.query_projection(x)
            bank_proj = self.text_projection(bank)
            if self.modality_projection_residual:
                x = x + x_proj
                bank = bank + bank_proj
            else:
                x = x_proj
                bank = bank_proj

        if self.class_logit_norm:
            x_for_logits = F.normalize(x, p=2, dim=-1, eps=1e-6)
            bank_for_logits = F.normalize(bank, p=2, dim=-1, eps=1e-6)
            class_logit_scale = self.log_class_logit_scale.exp().clamp(max=self.logit_clip)
            raw_res = (
                class_logit_scale * (x_for_logits @ bank_for_logits.transpose(-1, -2))
                + self.class_logit_bias_param
            )
        else:
            raw_res = x @ bank.transpose(-1, -2)

        raw_res = torch.nan_to_num(raw_res, nan=0.0, posinf=0.0, neginf=0.0)
        # NOTE: the similarity_calibration head operates on the max_text_len token
        # profile and is not meaningful over C class prototypes, so it is skipped
        # in bank mode.  The mirror appearance head runs with calibration off.
        final_res = raw_res.clamp(min=-self.logit_clip, max=self.logit_clip)
        if bank_mask is not None:
            invalid = (~bank_mask.bool()).view(*([1] * (final_res.dim() - 1)), -1)
            final_res = final_res.masked_fill(invalid, float("-inf"))
        return final_res


class CandidateConditionedLocalEvidence(nn.Module):
    """Candidate-conditioned local evidence head.

    Replaces the failed `ROI -> single vector -> prototype matching` paradigm.
    Instead of pooling the ROI into one candidate-agnostic vector (which averages
    away the discriminative digit/marker region), each candidate prototype reads
    the ROI's spatial tokens conditioned on itself:

        for each candidate c:
            q_c        = Wq · proto_c                       [D]
            k, v       = Wk · roi_tokens, Wv · roi_tokens   [L, D]
            attn_c     = softmax(q_c · k^T / sqrt(D))       [L]   <- candidate-specific
            read_c     = attn_c · v                         [D]   region evidence for c
            logit_c    = scale · cosine(read_c, q_c)              "does ROI look like c
                                                                  in the region c cares about"

    The attention map is DIFFERENT per candidate, so the region selection is
    conditioned on which class we are scoring — exactly what a fixed pooling could
    not do.  Scoring is restricted to the candidate set (semantic top-M), so this is
    conditional evidence, NOT a standalone 232-way classifier.  No per-class
    parameters (the reverse-of-generalization residual is retired): the only weights
    are Wq/Wk/Wv shared across all classes, so the mechanism transfers to new
    appearance styles by re-reading local evidence rather than re-memorizing a vector.

    Inputs:
        roi_tokens : [N, L, D]  flattened multi-scale ROI spatial tokens per query
        cand_proto : [N, K, D]  K candidate appearance prototypes per query
        cand_mask  : [N, K] bool, False = padding candidate (gets -inf logit)
    Output:
        evidence_logits : [N, K]
    """

    def __init__(self, embed_dim, logit_scale=10.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        logit_scale = max(float(logit_scale), 1e-6)
        self.log_logit_scale = nn.Parameter(
            torch.log(torch.tensor(logit_scale, dtype=torch.float32))
        )
        self.logit_clip = 20.0
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, roi_tokens, cand_proto, cand_mask=None):
        # roi_tokens [N, L, D], cand_proto [N, K, D]
        if roi_tokens.dim() != 3 or cand_proto.dim() != 3:
            raise ValueError(
                f"expected [N,L,D] and [N,K,D], got {tuple(roi_tokens.shape)} "
                f"and {tuple(cand_proto.shape)}"
            )
        N, L, D = roi_tokens.shape
        K = cand_proto.shape[1]
        scale_attn = 1.0 / math.sqrt(max(D, 1))

        q = self.q_proj(cand_proto)                 # [N, K, D]
        k = self.k_proj(roi_tokens)                 # [N, L, D]
        v = self.v_proj(roi_tokens)                 # [N, L, D]

        # Per-candidate attention over spatial tokens: [N, K, L]
        attn = torch.matmul(q, k.transpose(1, 2)) * scale_attn
        attn = torch.softmax(attn, dim=-1)
        read = torch.matmul(attn, v)                # [N, K, D] region evidence per cand

        # cosine(read_c, q_c) · scale  — no extra scoring params, mirrors the
        # class_logit_norm path so logit magnitude stays comparable/diagnosable.
        read_n = F.normalize(read, p=2, dim=-1, eps=1e-6)
        q_n = F.normalize(q, p=2, dim=-1, eps=1e-6)
        cos = (read_n * q_n).sum(dim=-1)            # [N, K]
        logit_scale = self.log_logit_scale.exp().clamp(max=self.logit_clip)
        evidence_logits = (logit_scale * cos).clamp(min=-self.logit_clip, max=self.logit_clip)

        if cand_mask is not None:
            evidence_logits = evidence_logits.masked_fill(~cand_mask.bool(), float("-inf"))
        return evidence_logits


class CLIPPairwiseCompatibilityHead(nn.Module):
    """Shared FACB compatibility head used by the offline pairwise scorer."""

    def __init__(self, embed_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, image_embed, text_embed):
        image = F.normalize(image_embed, p=2, dim=-1, eps=1e-6)
        text = F.normalize(text_embed, p=2, dim=-1, eps=1e-6)
        image_expanded = image[:, None, :].expand_as(text)
        features = torch.cat(
            [
                image_expanded,
                text,
                (image_expanded - text).abs(),
                image_expanded * text,
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class CLIPCropEvidence(nn.Module):
    """Candidate-conditioned evidence from original-pixel crops via a frozen CLIP.

    Replaces the failed detector-ROI-token local evidence (which stayed at / below
    random even with GT in the candidate set, because the detector's downsampled
    feature map does not carry the digit/marker pixels that separate same-group
    signs).  Here each matched query's box is cropped from the INPUT image tensor,
    de-normalized back to pixels, fed to a frozen CLIP vision encoder, and scored
    by cosine against frozen CLIP text embeddings of the candidate appearance
    descriptions.  Vision and text share CLIP's joint space -> no modality gap.

    Both CLIP towers are frozen; the only trainable parameter is a logit scale
    (optionally a small visual projection), so this is a conditional evidence
    branch, not a re-trained classifier, and it transfers across appearance styles.

    CLIP is lazily constructed on first forward so importing this module costs
    nothing when the branch is disabled.
    """

    # ImageNet stats used by the detector input pipeline (datasets/odvg.py).
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    # CLIP preprocessing stats.
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        clip_model_path,
        crop_size=224,
        logit_scale=10.0,
        use_visual_projection=False,
        embed_dim=None,
        pairwise_head_path=None,
        pairwise_hidden_dim=512,
        pairwise_dropout=0.1,
    ):
        super().__init__()
        self.clip_model_path = str(clip_model_path)
        self.crop_size = int(crop_size)
        logit_scale = max(float(logit_scale), 1e-6)
        self.log_logit_scale = nn.Parameter(
            torch.log(torch.tensor(logit_scale, dtype=torch.float32))
        )
        self.logit_clip = 20.0
        self._clip_vision = None          # lazily built (frozen)
        self._clip_dim = None
        # De-norm / re-norm buffers (detector ImageNet space -> [0,1] -> CLIP space).
        self.register_buffer("_in_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_in_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_clip_mean", torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_clip_std", torch.tensor(self.CLIP_STD).view(1, 3, 1, 1), persistent=False)
        # Optional trainable visual projection.
        self.use_visual_projection = bool(use_visual_projection)
        self.visual_projection = None
        self._embed_dim_hint = embed_dim
        self.pairwise_head_path = str(pairwise_head_path) if pairwise_head_path else ""
        self.pairwise_hidden_dim = int(pairwise_hidden_dim)
        self.pairwise_dropout = float(pairwise_dropout)
        self.pairwise_head = None

    def _ensure_clip(self, device):
        if self._clip_vision is not None:
            return
        from transformers import CLIPVisionModelWithProjection
        model = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._clip_vision = model.to(device)
        self._clip_dim = int(model.config.projection_dim)
        if self.use_visual_projection and self.visual_projection is None:
            self.visual_projection = nn.Linear(self._clip_dim, self._clip_dim).to(device)
            nn.init.eye_(self.visual_projection.weight)
            nn.init.zeros_(self.visual_projection.bias)
        if self.pairwise_head_path and self.pairwise_head is None:
            head = CLIPPairwiseCompatibilityHead(
                embed_dim=self._clip_dim,
                hidden_dim=self.pairwise_hidden_dim,
                dropout=self.pairwise_dropout,
            )
            checkpoint = torch.load(self.pairwise_head_path, map_location="cpu")
            state = checkpoint.get("head", checkpoint)
            head.load_state_dict(state, strict=True)
            head.eval()
            for p in head.parameters():
                p.requires_grad_(False)
            self.pairwise_head = head.to(device)

    @torch.no_grad()
    def _renormalize(self, crops):
        # crops: [N, 3, h, w] in detector ImageNet-normalized space.
        x = crops * self._in_std.to(crops.dtype) + self._in_mean.to(crops.dtype)  # -> [0,1]
        x = x.clamp(0.0, 1.0)
        x = (x - self._clip_mean.to(x.dtype)) / self._clip_std.to(x.dtype)        # -> CLIP space
        return x

    def encode_crops(self, crops):
        """crops [N,3,h,w] (detector-normalized) -> L2-normalized image embeds [N, d]."""
        device = crops.device
        self._ensure_clip(device)
        x = self._renormalize(crops)
        if x.shape[-1] != self.crop_size or x.shape[-2] != self.crop_size:
            x = F.interpolate(x, size=(self.crop_size, self.crop_size),
                              mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self._clip_vision(pixel_values=x.to(next(self._clip_vision.parameters()).dtype))
            img = out.image_embeds                       # [N, d]
        img = img.float()
        if self.visual_projection is not None:
            img = self.visual_projection(img)
        return F.normalize(img, p=2, dim=-1, eps=1e-6)

    def score(self, image_embeds, cand_text_embeds, cand_mask=None):
        """image_embeds [N, d] (L2), cand_text_embeds [N, M, d] (L2) -> logits [N, M]."""
        if self.pairwise_head is not None:
            logits = self.pairwise_head(image_embeds, cand_text_embeds).float()
        else:
            cos = (image_embeds.unsqueeze(1) * cand_text_embeds).sum(dim=-1)   # [N, M]
            scale = self.log_logit_scale.exp().clamp(max=self.logit_clip)
            logits = (scale * cos).clamp(min=-self.logit_clip, max=self.logit_clip)
        if cand_mask is not None:
            logits = logits.masked_fill(~cand_mask.bool(), float("-inf"))
        return logits
