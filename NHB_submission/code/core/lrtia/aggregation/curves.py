"""Memory curve building and aggregation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CurvePoint:
    """A single point on a memory curve."""

    distance: int
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    count: int
    values: list[float] = field(default_factory=list)


@dataclass
class MemoryCurve:
    """A memory curve showing effect size vs distance."""

    metric: str  # "delta_nll", "kl_divergence", "topk_overlap", "rank_change"
    group_by: str  # "population", "author", "document", "all"
    group_id: str  # Identifier for the group
    points: list[CurvePoint] = field(default_factory=list)

    @property
    def distances(self) -> list[int]:
        """Get list of distances."""
        return [p.distance for p in self.points]

    @property
    def means(self) -> list[float]:
        """Get list of mean values."""
        return [p.mean for p in self.points]

    @property
    def stds(self) -> list[float]:
        """Get list of standard deviations."""
        return [p.std for p in self.points]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(
            {
                "distance": self.distances,
                "mean": self.means,
                "std": self.stds,
                "ci_lower": [p.ci_lower for p in self.points],
                "ci_upper": [p.ci_upper for p in self.points],
                "count": [p.count for p in self.points],
                "metric": self.metric,
                "group_by": self.group_by,
                "group_id": self.group_id,
            }
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "group_by": self.group_by,
            "group_id": self.group_id,
            "distances": self.distances,
            "means": self.means,
            "stds": self.stds,
            "ci_lower": [p.ci_lower for p in self.points],
            "ci_upper": [p.ci_upper for p in self.points],
            "counts": [p.count for p in self.points],
        }


def build_memory_curve(
    df: pd.DataFrame,
    metric: str = "delta_nll",
    group_by: str | None = None,
    group_id: str | None = None,
    distance_col: str = "distance",
    confidence_level: float = 0.95,
    min_samples: int = 2,
) -> MemoryCurve:
    """Build a memory curve from evaluation results.

    Args:
        df: DataFrame with evaluation results. Must have distance and metric columns.
        metric: Which metric to use for the curve.
        group_by: Grouping level ("population", "author", "document", or None for all).
        group_id: Identifier for the group (if group_by is specified).
        distance_col: Name of the distance column.
        confidence_level: Confidence level for confidence intervals.
        min_samples: Minimum samples required for a point.

    Returns:
        MemoryCurve object.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")
    if distance_col not in df.columns:
        raise ValueError(f"Distance column '{distance_col}' not found in DataFrame")

    # Filter by group if specified
    if group_by and group_id:
        if group_by not in df.columns:
            raise ValueError(f"Group column '{group_by}' not found in DataFrame")
        df = df[df[group_by] == group_id]

    if len(df) == 0:
        return MemoryCurve(
            metric=metric,
            group_by=group_by or "all",
            group_id=group_id or "all",
            points=[],
        )

    # Group by distance and aggregate
    points = []
    for distance, group in df.groupby(distance_col):
        values = group[metric].dropna().values

        if len(values) < min_samples:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        # Compute confidence interval
        if len(values) > 1:
            ci = stats.t.interval(
                confidence_level,
                df=len(values) - 1,
                loc=mean,
                scale=stats.sem(values),
            )
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        point = CurvePoint(
            distance=int(distance),
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            count=len(values),
            values=list(values),
        )
        points.append(point)

    # Sort by distance
    points.sort(key=lambda p: p.distance)

    return MemoryCurve(
        metric=metric,
        group_by=group_by or "all",
        group_id=group_id or "all",
        points=points,
    )


def build_curves_by_group(
    df: pd.DataFrame,
    metric: str = "delta_nll",
    group_by: str = "population",
    distance_col: str = "distance",
    confidence_level: float = 0.95,
    min_samples: int = 2,
) -> dict[str, MemoryCurve]:
    """Build memory curves for each group.

    Args:
        df: DataFrame with evaluation results.
        metric: Which metric to use.
        group_by: Column to group by.
        distance_col: Name of the distance column.
        confidence_level: Confidence level for CIs.
        min_samples: Minimum samples per point.

    Returns:
        Dictionary mapping group_id to MemoryCurve.
    """
    if group_by not in df.columns:
        raise ValueError(f"Group column '{group_by}' not found in DataFrame")

    curves = {}
    for group_id in df[group_by].unique():
        curve = build_memory_curve(
            df,
            metric=metric,
            group_by=group_by,
            group_id=str(group_id),
            distance_col=distance_col,
            confidence_level=confidence_level,
            min_samples=min_samples,
        )
        curves[str(group_id)] = curve

    return curves


def aggregate_curves(
    curves: list[MemoryCurve],
    weights: list[float] | None = None,
) -> MemoryCurve:
    """Aggregate multiple memory curves into one.

    Args:
        curves: List of curves to aggregate.
        weights: Optional weights for each curve (default: equal weights).

    Returns:
        Aggregated MemoryCurve.
    """
    if not curves:
        raise ValueError("Cannot aggregate empty list of curves")

    if weights is None:
        weights = [1.0] * len(curves)

    if len(weights) != len(curves):
        raise ValueError("Number of weights must match number of curves")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Get all unique distances
    all_distances = set()
    for curve in curves:
        all_distances.update(curve.distances)
    all_distances = sorted(all_distances)

    # Aggregate at each distance
    points = []
    for distance in all_distances:
        all_values = []
        for curve, weight in zip(curves, weights):
            for point in curve.points:
                if point.distance == distance:
                    # Add values weighted by curve weight
                    all_values.extend(point.values)
                    break

        if not all_values:
            continue

        mean = float(np.mean(all_values))
        std = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else 0.0

        if len(all_values) > 1:
            ci = stats.t.interval(
                0.95,
                df=len(all_values) - 1,
                loc=mean,
                scale=stats.sem(all_values),
            )
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        point = CurvePoint(
            distance=distance,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            count=len(all_values),
            values=all_values,
        )
        points.append(point)

    return MemoryCurve(
        metric=curves[0].metric,
        group_by="aggregated",
        group_id="all",
        points=points,
    )
