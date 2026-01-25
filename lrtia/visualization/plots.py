"""Visualization tools for memory curves and comparisons."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lrtia.aggregation.curves import MemoryCurve
from lrtia.analysis.comparison import ComparisonResult


# Set default style
sns.set_theme(style="whitegrid")


def plot_memory_curve(
    curve: MemoryCurve,
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str | None = None,
    show_ci: bool = True,
    log_x: bool = True,
    title: str | None = None,
    xlabel: str = "Distance (tokens)",
    ylabel: str | None = None,
) -> plt.Axes:
    """Plot a single memory curve.

    Args:
        curve: Memory curve to plot.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        color: Line color.
        label: Legend label.
        show_ci: Whether to show confidence intervals.
        log_x: Whether to use log scale for x-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label (default: based on metric).

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if not curve.points:
        return ax

    distances = np.array(curve.distances)
    means = np.array(curve.means)

    # Default ylabel based on metric
    if ylabel is None:
        ylabel_map = {
            "delta_nll": "ΔNLL",
            "kl_divergence": "KL Divergence",
            "topk_overlap": "Top-k Overlap",
            "rank_change": "Rank Change",
        }
        ylabel = ylabel_map.get(curve.metric, curve.metric)

    # Default label
    if label is None:
        label = curve.group_id

    # Plot main line
    line = ax.plot(distances, means, "-o", color=color, label=label, linewidth=2, markersize=6)
    line_color = line[0].get_color()

    # Plot confidence interval
    if show_ci:
        ci_lower = np.array([p.ci_lower for p in curve.points])
        ci_upper = np.array([p.ci_upper for p in curve.points])
        ax.fill_between(distances, ci_lower, ci_upper, alpha=0.2, color=line_color)

    # Styling
    if log_x:
        ax.set_xscale("log", base=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_comparison(
    curves: dict[str, MemoryCurve] | list[MemoryCurve],
    ax: plt.Axes | None = None,
    colors: dict[str, str] | list[str] | None = None,
    show_ci: bool = True,
    log_x: bool = True,
    title: str = "Memory Curves Comparison",
    xlabel: str = "Distance (tokens)",
    ylabel: str | None = None,
) -> plt.Axes:
    """Plot multiple memory curves for comparison.

    Args:
        curves: Dictionary or list of memory curves.
        ax: Matplotlib axes.
        colors: Colors for each curve (dict or list).
        show_ci: Whether to show confidence intervals.
        log_x: Whether to use log scale for x-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))

    # Convert list to dict if needed
    if isinstance(curves, list):
        curves = {c.group_id: c for c in curves}

    # Default color palette
    palette = sns.color_palette("husl", len(curves))
    if colors is None:
        colors = {group_id: palette[i] for i, group_id in enumerate(curves.keys())}
    elif isinstance(colors, list):
        colors = {group_id: colors[i] for i, group_id in enumerate(curves.keys())}

    # Plot each curve
    for group_id, curve in curves.items():
        color = colors.get(group_id)
        plot_memory_curve(
            curve,
            ax=ax,
            color=color,
            label=group_id,
            show_ci=show_ci,
            log_x=log_x,
            title=None,  # Set title at the end
            xlabel=xlabel,
            ylabel=ylabel,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(title="Group", loc="upper right")

    return ax


def plot_intervention_breakdown(
    curves_by_intervention: dict[str, MemoryCurve],
    ax: plt.Axes | None = None,
    colors: dict[str, str] | None = None,
    show_ci: bool = True,
    log_x: bool = True,
    title: str = "Effect by Intervention Type",
    xlabel: str = "Distance (tokens)",
    ylabel: str | None = None,
) -> plt.Axes:
    """Plot memory curves broken down by intervention type.

    Args:
        curves_by_intervention: Dict mapping intervention type to curve.
        ax: Matplotlib axes.
        colors: Colors for each intervention type.
        show_ci: Whether to show confidence intervals.
        log_x: Whether to use log scale for x-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Default colors for intervention types
    default_colors = {
        "mask": "#1f77b4",
        "shuffle": "#ff7f0e",
        "delete": "#d62728",
    }
    if colors is None:
        colors = default_colors

    # Plot each intervention type
    for intervention_type, curve in curves_by_intervention.items():
        color = colors.get(intervention_type, None)
        plot_memory_curve(
            curve,
            ax=ax,
            color=color,
            label=intervention_type.capitalize(),
            show_ci=show_ci,
            log_x=log_x,
            title=None,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(title="Intervention", loc="upper right")

    return ax


def plot_summary_comparison(
    results: Sequence[ComparisonResult],
    metric: str = "auc",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot bar chart comparing summary metrics across groups.

    Args:
        results: List of ComparisonResult objects.
        metric: Which summary metric to plot ("auc", "half_life", "tail_mass").
        ax: Matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Collect data
    groups = []
    values = []
    errors = []

    for result in results:
        # Get values based on metric
        if metric == "auc":
            groups.append(result.group_a)
            values.append(result.summary_a.auc)
            errors.append(0)  # Could compute from CI if available

            groups.append(result.group_b)
            values.append(result.summary_b.auc)
            errors.append(0)
        elif metric == "half_life":
            if result.summary_a.half_life is not None:
                groups.append(result.group_a)
                values.append(result.summary_a.half_life)
                errors.append(0)
            if result.summary_b.half_life is not None:
                groups.append(result.group_b)
                values.append(result.summary_b.half_life)
                errors.append(0)
        elif metric == "tail_mass":
            groups.append(result.group_a)
            values.append(result.summary_a.tail_mass)
            errors.append(0)

            groups.append(result.group_b)
            values.append(result.summary_b.tail_mass)
            errors.append(0)

    # Remove duplicates while preserving order
    seen = set()
    unique_groups = []
    unique_values = []
    for g, v in zip(groups, values):
        if g not in seen:
            seen.add(g)
            unique_groups.append(g)
            unique_values.append(v)

    # Create bar plot
    x = np.arange(len(unique_groups))
    bars = ax.bar(x, unique_values, color=sns.color_palette("husl", len(unique_groups)))

    ax.set_xticks(x)
    ax.set_xticklabels(unique_groups, rotation=45, ha="right")

    ylabel_map = {
        "auc": "Area Under Curve (AUC)",
        "half_life": "Half-Life (tokens)",
        "tail_mass": "Tail Mass (fraction)",
    }
    ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=12)

    if title is None:
        title = f"Summary Metric Comparison: {metric.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return ax


def plot_effect_size_bars(
    results: Sequence[ComparisonResult],
    ax: plt.Axes | None = None,
    title: str = "Effect Sizes (Cohen's d)",
) -> plt.Axes:
    """Plot effect sizes from pairwise comparisons.

    Args:
        results: List of ComparisonResult objects.
        ax: Matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    labels = []
    effect_sizes = []

    for result in results:
        labels.append(f"{result.group_a} vs {result.group_b}")
        effect_sizes.append(result.effect_size)

    # Color by effect size magnitude
    colors = ["#d62728" if abs(es) > 0.8 else "#ff7f0e" if abs(es) > 0.5 else "#2ca02c" for es in effect_sizes]

    x = np.arange(len(labels))
    bars = ax.barh(x, effect_sizes, color=colors)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add reference lines for effect size interpretation
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5, label="Small (0.2)")
    ax.axvline(x=-0.2, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Medium (0.5)")
    ax.axvline(x=-0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0.8, color="gray", linestyle="-.", alpha=0.5, label="Large (0.8)")
    ax.axvline(x=-0.8, color="gray", linestyle="-.", alpha=0.5)

    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")

    plt.tight_layout()
    return ax


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    formats: list[str] | None = None,
) -> None:
    """Save a figure to file(s).

    Args:
        fig: Matplotlib figure to save.
        path: Output path (without extension if multiple formats).
        dpi: Resolution for raster formats.
        formats: List of formats to save (default: ["png"]).
    """
    if formats is None:
        formats = ["png"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = path.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=fmt)


def create_summary_figure(
    curves: dict[str, MemoryCurve],
    comparison_results: list[ComparisonResult] | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Create a comprehensive summary figure with multiple panels.

    Args:
        curves: Dictionary of memory curves by group.
        comparison_results: Optional list of comparison results.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    if comparison_results:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Top row spans both columns
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
    else:
        ax1 = fig.add_subplot(111)

    # Main comparison plot
    plot_comparison(
        curves,
        ax=ax1,
        title="Memory Curves by Population",
        show_ci=True,
        log_x=True,
    )

    # Summary bar chart
    if comparison_results:
        plot_summary_comparison(
            comparison_results,
            metric="auc",
            ax=ax2,
            title="AUC Comparison",
        )

        # Effect sizes
        plot_effect_size_bars(
            comparison_results,
            ax=ax3,
            title="Effect Sizes",
        )

    plt.suptitle("LRTIA Analysis Summary", fontsize=16, y=1.02)
    return fig
