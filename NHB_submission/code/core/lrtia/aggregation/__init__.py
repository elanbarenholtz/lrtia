"""Aggregation and summary statistics."""

from lrtia.aggregation.curves import build_memory_curve, build_curves_by_group, MemoryCurve
from lrtia.aggregation.summary import (
    compute_auc,
    compute_half_life,
    compute_tail_mass,
    compute_summary_metrics,
    fit_decay_models,
    get_best_decay_fit,
    DecayFit,
    SummaryMetrics,
)

__all__ = [
    "build_memory_curve",
    "build_curves_by_group",
    "MemoryCurve",
    "compute_auc",
    "compute_half_life",
    "compute_tail_mass",
    "compute_summary_metrics",
    "fit_decay_models",
    "get_best_decay_fit",
    "DecayFit",
    "SummaryMetrics",
]
