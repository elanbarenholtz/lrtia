"""CLI entry point for LRTIA using Typer."""

import json
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from lrtia.config import LRTIAConfig

app = typer.Typer(
    name="lrtia",
    help="Long-Range Token Influence Analyzer - Analyze memory in language models",
    add_completion=False,
)

console = Console()


def load_config(config_path: Path | None) -> LRTIAConfig:
    """Load configuration from file or use defaults."""
    if config_path and config_path.exists():
        return LRTIAConfig.from_yaml(config_path)
    return LRTIAConfig()


@app.command()
def run(
    data: Annotated[Path, typer.Argument(help="Path to corpus file (JSONL/CSV/Parquet)")],
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Configuration file")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model name")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Limit number of documents")] = None,
    device: Annotated[str, typer.Option("--device", "-d", help="Device (auto/cuda/cpu/mps)")] = "auto",
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
) -> None:
    """Run LRTIA analysis on a corpus.

    Performs causal interventions and computes memory curves for each document.
    """
    from lrtia.data import load_corpus, create_windows
    from lrtia.intervention import SpanSelector, apply_intervention
    from lrtia.model import HuggingFaceBackend
    from lrtia.metrics import compute_all_metrics
    from lrtia.data.windowing import get_target_positions

    # Load configuration
    cfg = load_config(config)
    if model:
        cfg.model.name = model
    if output:
        cfg.output.results_dir = output
    cfg.model.device = device
    cfg.computation.seed = seed

    console.print(f"[bold blue]LRTIA Analysis[/bold blue]")
    console.print(f"  Model: {cfg.model.name}")
    console.print(f"  Device: {cfg.model.device}")
    console.print(f"  Data: {data}")

    # Create output directory
    results_dir = cfg.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    with console.status("[bold green]Loading corpus..."):
        documents = load_corpus(data)
        if limit:
            documents = documents[:limit]
        console.print(f"  Loaded {len(documents)} documents")

    # Load model
    with console.status(f"[bold green]Loading model {cfg.model.name}..."):
        backend = HuggingFaceBackend(
            model_name=cfg.model.name,
            device=cfg.model.device,
            dtype=cfg.model.dtype,
        )
        backend.load()

    # Initialize span selector
    span_selector = SpanSelector(
        distances=cfg.spans.distances,
        widths=cfg.spans.widths,
        spans_per_distance=cfg.spans.spans_per_distance,
        seed=cfg.computation.seed,
    )

    # Get mask token ID
    mask_token_id = backend.get_mask_token_id()

    # Process documents
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        doc_task = progress.add_task("[cyan]Processing documents...", total=len(documents))

        for doc in documents:
            # Create windows
            windows = create_windows(
                doc,
                backend.tokenizer,
                window_tokens=cfg.windowing.window_tokens,
                stride_tokens=cfg.windowing.stride_tokens,
            )

            for window in windows:
                # Get target positions
                targets = get_target_positions(
                    window,
                    method=cfg.targets.method,
                    interval=cfg.targets.interval,
                    min_context=cfg.targets.min_context,
                    region_length=cfg.targets.region_length,
                )

                for target_start, target_end in targets:
                    # Get spans for this target
                    spans = span_selector.get_all_spans_for_target(
                        target_start=target_start,
                        target_end=target_end,
                        context_length=window.num_tokens,
                        doc_id=doc.doc_id,
                        window_id=window.window_id,
                    )

                    # Skip if no valid spans
                    if not spans:
                        continue

                    # Get original predictions
                    target_positions = list(range(target_start, min(target_end, window.num_tokens - 1)))
                    if not target_positions:
                        continue

                    original_output = backend.forward(
                        window.token_ids,
                        positions=target_positions,
                        return_full_distribution=True,
                    )

                    target_tokens = [window.token_ids[p + 1] for p in target_positions]

                    # Apply each intervention
                    for span in spans:
                        for intervention_type in cfg.interventions.types:
                            # Apply intervention
                            result = apply_intervention(
                                window.token_ids,
                                span,
                                intervention_type,
                                mask_token_id=mask_token_id,
                                seed=cfg.computation.seed,
                            )

                            # Adjust positions for length-changing interventions
                            if result.length_changed:
                                adjusted_positions = [
                                    p - len(result.original_tokens) if p >= span.end else p
                                    for p in target_positions
                                    if p < len(result.token_ids) - 1
                                ]
                                if not adjusted_positions:
                                    continue
                            else:
                                adjusted_positions = target_positions

                            # Get intervened predictions
                            intervened_output = backend.forward(
                                result.token_ids,
                                positions=adjusted_positions,
                                return_full_distribution=True,
                            )

                            # Compute metrics
                            metrics = compute_all_metrics(
                                original_output.log_probs[0].cpu().numpy(),
                                intervened_output.log_probs[0].cpu().numpy(),
                                target_tokens[: len(adjusted_positions)],
                                top_k_for_kl=cfg.computation.top_k_for_kl,
                                top_k_for_stability=cfg.computation.top_k_for_stability,
                            )

                            # Store result
                            all_results.append({
                                "doc_id": doc.doc_id,
                                "author_id": doc.author_id,
                                "population": doc.population,
                                "window_id": window.window_id,
                                "target_start": target_start,
                                "target_end": target_end,
                                "span_start": span.start,
                                "span_end": span.end,
                                "distance": span.distance,
                                "actual_distance": span.actual_distance,
                                "span_width": span.width,
                                "intervention_type": intervention_type,
                                "delta_nll": metrics.delta_nll,
                                "kl_divergence": metrics.kl_divergence,
                                "topk_overlap": metrics.topk_overlap,
                                "rank_change": metrics.rank_change,
                                "length_changed": result.length_changed,
                                "seed": cfg.computation.seed,
                            })

            progress.advance(doc_task)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_results)

    if cfg.output.format == "parquet":
        output_path = results_dir / "evaluations.parquet"
        df.to_parquet(output_path, index=False)
    elif cfg.output.format == "csv":
        output_path = results_dir / "evaluations.csv"
        df.to_csv(output_path, index=False)
    else:
        output_path = results_dir / "evaluations.json"
        df.to_json(output_path, orient="records", indent=2)

    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")
    console.print(f"  Total evaluations: {len(all_results)}")

    # Cleanup
    backend.unload()


@app.command()
def analyze(
    results: Annotated[Path, typer.Argument(help="Path to results file (Parquet/CSV/JSON)")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    group_by: Annotated[str, typer.Option("--group-by", "-g", help="Group by field")] = "population",
    metric: Annotated[str, typer.Option("--metric", "-m", help="Metric to analyze")] = "delta_nll",
) -> None:
    """Analyze results and build memory curves.

    Aggregates evaluation results into memory curves and computes summary metrics.
    """
    from lrtia.aggregation import build_curves_by_group, compute_summary_metrics

    console.print(f"[bold blue]Analyzing results[/bold blue]")
    console.print(f"  Input: {results}")
    console.print(f"  Group by: {group_by}")
    console.print(f"  Metric: {metric}")

    # Load results
    if results.suffix == ".parquet":
        df = pd.read_parquet(results)
    elif results.suffix == ".csv":
        df = pd.read_csv(results)
    else:
        df = pd.read_json(results)

    console.print(f"  Loaded {len(df)} evaluations")

    # Build curves
    curves = build_curves_by_group(df, metric=metric, group_by=group_by)

    # Compute summary metrics
    summaries = {}
    for group_id, curve in curves.items():
        summaries[group_id] = compute_summary_metrics(curve)

    # Display results
    table = Table(title="Summary Metrics by Group")
    table.add_column("Group", style="cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Half-Life", justify="right")
    table.add_column("Tail Mass", justify="right")
    table.add_column("Peak Effect", justify="right")

    for group_id, summary in summaries.items():
        table.add_row(
            group_id,
            f"{summary.auc:.4f}",
            f"{summary.half_life:.0f}" if summary.half_life else "N/A",
            f"{summary.tail_mass:.3f}",
            f"{summary.peak_effect:.4f}",
        )

    console.print(table)

    # Save outputs
    if output:
        output.mkdir(parents=True, exist_ok=True)

        # Save curves
        curves_data = {gid: curve.to_dict() for gid, curve in curves.items()}
        with open(output / "curves.json", "w") as f:
            json.dump(curves_data, f, indent=2)

        # Save summaries
        summaries_data = {gid: s.to_dict() for gid, s in summaries.items()}
        with open(output / "summaries.json", "w") as f:
            json.dump(summaries_data, f, indent=2)

        console.print(f"\n[bold green]Analysis saved to {output}[/bold green]")


@app.command()
def plot(
    results: Annotated[Path, typer.Argument(help="Path to results file")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./figures"),
    group_by: Annotated[str, typer.Option("--group-by", "-g", help="Group by field")] = "population",
    metric: Annotated[str, typer.Option("--metric", "-m", help="Metric to plot")] = "delta_nll",
    format: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "png",
) -> None:
    """Generate plots from results.

    Creates memory curve plots and comparison visualizations.
    """
    from lrtia.aggregation import build_curves_by_group
    from lrtia.visualization import plot_comparison, save_figure
    import matplotlib.pyplot as plt

    console.print(f"[bold blue]Generating plots[/bold blue]")

    # Load results
    if results.suffix == ".parquet":
        df = pd.read_parquet(results)
    elif results.suffix == ".csv":
        df = pd.read_csv(results)
    else:
        df = pd.read_json(results)

    # Build curves
    curves = build_curves_by_group(df, metric=metric, group_by=group_by)

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Main comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_comparison(curves, ax=ax, title=f"Memory Curves ({metric})")
    save_figure(fig, output / f"memory_curves_{metric}", formats=[format])
    plt.close(fig)

    console.print(f"[bold green]Plots saved to {output}[/bold green]")


@app.command()
def compare(
    results: Annotated[Path, typer.Argument(help="Path to results file")],
    groups: Annotated[str, typer.Option("--groups", "-g", help="Groups to compare (comma-separated)")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Metric to compare")] = "delta_nll",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file for comparison")] = None,
) -> None:
    """Compare populations statistically.

    Performs statistical comparisons between specified groups.
    """
    from lrtia.aggregation import build_curves_by_group
    from lrtia.analysis import compare_populations

    console.print(f"[bold blue]Comparing groups[/bold blue]")

    # Parse groups
    group_list = [g.strip() for g in groups.split(",")]
    if len(group_list) != 2:
        console.print("[bold red]Error: Please specify exactly 2 groups to compare[/bold red]")
        raise typer.Exit(1)

    # Load results
    if results.suffix == ".parquet":
        df = pd.read_parquet(results)
    elif results.suffix == ".csv":
        df = pd.read_csv(results)
    else:
        df = pd.read_json(results)

    # Build curves
    curves = build_curves_by_group(df, metric=metric, group_by="population")

    # Check groups exist
    for g in group_list:
        if g not in curves:
            console.print(f"[bold red]Error: Group '{g}' not found in data[/bold red]")
            raise typer.Exit(1)

    # Compare
    result = compare_populations(curves[group_list[0]], curves[group_list[1]])

    # Display results
    table = Table(title=f"Comparison: {result.group_a} vs {result.group_b}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("95% CI", justify="right")

    table.add_row(
        "AUC Difference",
        f"{result.auc_diff:.4f}",
        f"[{result.auc_diff_ci[0]:.4f}, {result.auc_diff_ci[1]:.4f}]",
    )
    table.add_row(
        "Effect Size (Cohen's d)",
        f"{result.effect_size:.3f}",
        "-",
    )
    if result.p_value is not None:
        table.add_row(
            f"p-value ({result.test_name})",
            f"{result.p_value:.4f}",
            "-",
        )

    console.print(table)

    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[bold green]Comparison saved to {output}[/bold green]")


@app.command()
def gui() -> None:
    """Launch the interactive GUI (requires streamlit)."""
    import subprocess
    import sys
    from pathlib import Path

    gui_path = Path(__file__).parent / "gui.py"

    if not gui_path.exists():
        console.print("[bold red]Error: GUI module not found[/bold red]")
        raise typer.Exit(1)

    try:
        import streamlit
    except ImportError:
        console.print("[bold red]Error: Streamlit not installed[/bold red]")
        console.print("Install with: pip install streamlit plotly")
        raise typer.Exit(1)

    console.print("[bold blue]Launching LRTIA GUI...[/bold blue]")
    console.print("Open your browser to the URL shown below")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])


@app.command()
def info() -> None:
    """Display information about LRTIA."""
    from lrtia import __version__

    console.print(f"[bold blue]LRTIA - Long-Range Token Influence Analyzer[/bold blue]")
    console.print(f"  Version: {__version__}")
    console.print()
    console.print("Commands:")
    console.print("  run      - Run analysis on a corpus")
    console.print("  analyze  - Analyze results and build curves")
    console.print("  plot     - Generate visualizations")
    console.print("  compare  - Statistical comparison between groups")
    console.print("  gui      - Launch interactive GUI")
    console.print()
    console.print("For help on a specific command, run: lrtia <command> --help")


if __name__ == "__main__":
    app()
