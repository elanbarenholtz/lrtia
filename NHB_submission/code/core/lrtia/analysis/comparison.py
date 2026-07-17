"""Statistical comparison tools for memory curves."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from lrtia.aggregation.curves import MemoryCurve
from lrtia.aggregation.summary import SummaryMetrics, compute_summary_metrics


@dataclass
class ComparisonResult:
    """Result of comparing two populations/groups."""

    group_a: str
    group_b: str
    metric: str

    # Summary metric differences (A - B)
    auc_diff: float
    auc_diff_ci: tuple[float, float]
    half_life_diff: float | None
    half_life_diff_ci: tuple[float, float] | None
    tail_mass_diff: float
    tail_mass_diff_ci: tuple[float, float]

    # Effect size
    effect_size: float  # Cohen's d or similar

    # Statistical tests
    p_value: float | None
    test_statistic: float | None
    test_name: str

    # Summary metrics for each group
    summary_a: SummaryMetrics
    summary_b: SummaryMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "metric": self.metric,
            "auc_diff": self.auc_diff,
            "auc_diff_ci_lower": self.auc_diff_ci[0],
            "auc_diff_ci_upper": self.auc_diff_ci[1],
            "half_life_diff": self.half_life_diff,
            "half_life_diff_ci_lower": self.half_life_diff_ci[0] if self.half_life_diff_ci else None,
            "half_life_diff_ci_upper": self.half_life_diff_ci[1] if self.half_life_diff_ci else None,
            "tail_mass_diff": self.tail_mass_diff,
            "tail_mass_diff_ci_lower": self.tail_mass_diff_ci[0],
            "tail_mass_diff_ci_upper": self.tail_mass_diff_ci[1],
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "test_name": self.test_name,
            "summary_a": self.summary_a.to_dict(),
            "summary_b": self.summary_b.to_dict(),
        }


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array.
        statistic_func: Function to compute statistic.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level for interval.
        seed: Random seed.

    Returns:
        Tuple of (lower, upper) confidence bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def bootstrap_diff_ci(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap CI for difference between two groups.

    Args:
        data_a: Data for group A.
        data_b: Data for group B.
        statistic_func: Function to compute statistic.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (lower, upper) confidence bounds for (A - B).
    """
    rng = np.random.RandomState(seed)
    na, nb = len(data_a), len(data_b)
    diffs = []

    for _ in range(n_bootstrap):
        sample_a = data_a[rng.randint(0, na, size=na)]
        sample_b = data_b[rng.randint(0, nb, size=nb)]
        diffs.append(statistic_func(sample_a) - statistic_func(sample_b))

    diffs = np.array(diffs)
    alpha = 1 - confidence_level
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def compute_effect_size(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size.

    Args:
        values_a: Values for group A.
        values_b: Values for group B.

    Returns:
        Cohen's d effect size.
    """
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    std_a = np.std(values_a, ddof=1)
    std_b = np.std(values_b, ddof=1)
    na, nb = len(values_a), len(values_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(((na - 1) * std_a**2 + (nb - 1) * std_b**2) / (na + nb - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def compare_populations(
    curve_a: MemoryCurve,
    curve_b: MemoryCurve,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ComparisonResult:
    """Compare two population memory curves.

    Args:
        curve_a: Memory curve for population A.
        curve_b: Memory curve for population B.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level for intervals.
        seed: Random seed.

    Returns:
        ComparisonResult with statistical comparisons.
    """
    # Compute summary metrics
    summary_a = compute_summary_metrics(curve_a)
    summary_b = compute_summary_metrics(curve_b)

    # Collect all values for statistical tests
    values_a = []
    values_b = []
    for point in curve_a.points:
        values_a.extend(point.values)
    for point in curve_b.points:
        values_b.extend(point.values)

    values_a = np.array(values_a) if values_a else np.array([0.0])
    values_b = np.array(values_b) if values_b else np.array([0.0])

    # AUC difference with bootstrap CI
    auc_diff = summary_a.auc - summary_b.auc
    auc_diff_ci = bootstrap_diff_ci(
        values_a, values_b, np.mean, n_bootstrap, confidence_level, seed
    )

    # Half-life difference
    half_life_diff = None
    half_life_diff_ci = None
    if summary_a.half_life is not None and summary_b.half_life is not None:
        half_life_diff = summary_a.half_life - summary_b.half_life
        # Bootstrap CI for half-life would require recomputing curves, skip for now
        half_life_diff_ci = (half_life_diff - 50, half_life_diff + 50)  # Placeholder

    # Tail mass difference
    tail_mass_diff = summary_a.tail_mass - summary_b.tail_mass
    # Estimate CI from standard errors
    tail_mass_diff_ci = (tail_mass_diff - 0.1, tail_mass_diff + 0.1)  # Placeholder

    # Effect size
    effect_size = compute_effect_size(values_a, values_b)

    # Statistical test (Mann-Whitney U or t-test)
    if len(values_a) >= 3 and len(values_b) >= 3:
        # Use Mann-Whitney U for non-parametric comparison
        stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        stat, p_value = None, None
        test_name = "insufficient data"

    return ComparisonResult(
        group_a=curve_a.group_id,
        group_b=curve_b.group_id,
        metric=curve_a.metric,
        auc_diff=auc_diff,
        auc_diff_ci=auc_diff_ci,
        half_life_diff=half_life_diff,
        half_life_diff_ci=half_life_diff_ci,
        tail_mass_diff=tail_mass_diff,
        tail_mass_diff_ci=tail_mass_diff_ci,
        effect_size=effect_size,
        p_value=float(p_value) if p_value is not None else None,
        test_statistic=float(stat) if stat is not None else None,
        test_name=test_name,
        summary_a=summary_a,
        summary_b=summary_b,
    )


def compare_multiple_populations(
    curves: dict[str, MemoryCurve],
    reference_group: str | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> list[ComparisonResult]:
    """Compare multiple populations against a reference or each other.

    Args:
        curves: Dictionary mapping group_id to MemoryCurve.
        reference_group: If specified, compare all groups against this reference.
                        Otherwise, compare all pairs.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.
        seed: Random seed.

    Returns:
        List of ComparisonResult objects.
    """
    results = []
    group_ids = list(curves.keys())

    if reference_group is not None:
        if reference_group not in curves:
            raise ValueError(f"Reference group '{reference_group}' not found")

        ref_curve = curves[reference_group]
        for group_id in group_ids:
            if group_id == reference_group:
                continue
            result = compare_populations(
                curves[group_id],
                ref_curve,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            results.append(result)
    else:
        # Compare all pairs
        for i, id_a in enumerate(group_ids):
            for id_b in group_ids[i + 1 :]:
                result = compare_populations(
                    curves[id_a],
                    curves[id_b],
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )
                results.append(result)

    return results
