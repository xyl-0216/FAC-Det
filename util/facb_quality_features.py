"""Shared method utilities."""
from util.facb_quality_targets import logit


FEATURES = [
    "old_score_logit", "selected_original_logit", "itm_class_logit",
    "itm_shape_logit", "itm_symbol_logit", "p0_margin_over_old",
    "p0_top1_margin",
]


def row_feature(row):
    return [
        logit(row["old_score"]), logit(row["selected_original_score"]),
        logit(row["itm_class_probability"]), logit(row["itm_shape_probability"]),
        logit(row["itm_symbol_probability"]), float(row["fused_margin_over_old"]),
        float(row["fused_top1_margin"]),
    ]


