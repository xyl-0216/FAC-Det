# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    @staticmethod
    def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        valid_mask = torch.isfinite(logits)
        masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, -1e4))
        has_valid = valid_mask.any(dim=dim, keepdim=True)

        max_logits = masked_logits.max(dim=dim, keepdim=True).values
        shifted = masked_logits - max_logits
        exp_logits = torch.exp(shifted) * valid_mask.to(masked_logits.dtype)
        denom = exp_logits.sum(dim=dim, keepdim=True)
        probs = exp_logits / denom.clamp_min(1e-12)
        probs = probs * has_valid.to(probs.dtype)
        return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """Memory-efficient bidirectional cross-attention using scaled_dot_product_attention.

        V→L: vision queries attend to language keys/values (masked by attention_mask_l)
        L→V: language queries attend to vision keys/values (masked by attention_mask_v)

        Uses FlashAttention/MemEfficientAttention backend when available (PyTorch 2.0+),
        avoiding O(n^2) materialization of the full attention matrix.
        """
        bsz, tgt_len, _ = v.size()
        src_len = l.size(1)

        # Project QKV
        query_v = self.v_proj(v) * self.scale          # V→L query:  scaled vision
        key_l_proj = self.l_proj(l)                     # V→L key:    raw language
        value_v = self.values_v_proj(v)
        value_l = self.values_l_proj(l)

        # Reshape to [bs, nheads, seq, head_dim] for sdpa
        q_v = self._shape(query_v, tgt_len, bsz)       # [bs, nheads, vision_tok, head_dim]
        k_l = self._shape(key_l_proj, -1, bsz)          # [bs, nheads, text_tok,  head_dim]
        v_v = self._shape(value_v, -1, bsz)
        v_l = self._shape(value_l, -1, bsz)

        dropout_p = self.dropout if self.training else 0.0

        # Build float attention masks: 0.0 for valid, -inf for padding.
        # Using float (not bool) to avoid backend-dependent boolean semantics.
        def _make_mask(padding_mask):
            # padding_mask: [bs, seq], True = padding
            if padding_mask is None:
                return None
            return padding_mask[:, None, None, :].float().masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )

        mask_v = _make_mask(attention_mask_v)
        mask_l = _make_mask(attention_mask_l)

        # Vision → Language: sdpa(Q_v, K_l, V_l), mask text padding
        out_v = F.scaled_dot_product_attention(
            q_v, k_l, v_l, attn_mask=mask_l, dropout_p=dropout_p, scale=1.0,
        )
        out_v = out_v.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        out_v = self.out_v_proj(out_v)

        # Language → Vision: Q=K_l, K=Q_v, V=V_v
        # Original computes softmax((Q_v @ K_l^T)^T, dim=vision) @ V_v
        # sdpa(K_l, Q_v, V_v) computes softmax(K_l @ Q_v^T) @ V_v — identical result.
        out_l = F.scaled_dot_product_attention(
            k_l, q_v, v_v, attn_mask=mask_v, dropout_p=dropout_p, scale=1.0,
        )
        out_l = out_l.transpose(1, 2).contiguous().view(bsz, src_len, self.embed_dim)
        out_l = self.out_l_proj(out_l)

        return out_v, out_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)
