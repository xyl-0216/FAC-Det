"""Frozen dual-facet image-text matching heads used by FACB inference."""
import torch
from torch import nn


SCHEMA = "facb_p0_dual_facet_itm_online_v1"


class DualFacetITM(nn.Module):
    def __init__(self, state_dim=256, hidden=48):
        super().__init__()
        self.decoder = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.shape = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.symbol = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.class_match = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, decoder_state, text_state, raw_facets, facet_mean, facet_std):
        decoded = self.decoder(decoder_state)
        text = self.text(text_state)
        pair = self.pair(torch.cat([decoded, text, torch.abs(decoded - text), decoded * text], dim=-1))
        standardized = (raw_facets - facet_mean) / facet_std
        shape_logit = self.shape(torch.cat([pair, standardized[:, 0:1]], dim=-1)).squeeze(-1)
        symbol_logit = self.symbol(torch.cat([pair, standardized[:, 1:2]], dim=-1)).squeeze(-1)
        class_logit = self.class_match(
            torch.cat([pair, shape_logit[:, None], symbol_logit[:, None], standardized], dim=-1)
        ).squeeze(-1)
        return class_logit, shape_logit, symbol_logit


def dual_facet_logits(model, candidate_states, candidate_text_states, candidate_facet_scores, mean, std):
    """Return [rows,candidates,3] class/shape/symbol logits in FP32."""
    if candidate_facet_scores is None or candidate_facet_scores.shape[-1] < 3:
        raise ValueError("Dual-facet ITM requires raw group/shape/symbol candidate scores")
    if candidate_states is None or candidate_text_states is None:
        raise ValueError("Dual-facet ITM requires exported candidate/text states")
    shape_symbol = candidate_facet_scores[..., [1, 2]].float()
    shape = candidate_states.shape
    flat = model(
        candidate_states.float().reshape(-1, shape[-1]),
        candidate_text_states.float().reshape(-1, shape[-1]),
        shape_symbol.reshape(-1, 2), mean, std,
    )
    return torch.stack(flat, dim=-1).reshape(shape[0], shape[1], 3)
