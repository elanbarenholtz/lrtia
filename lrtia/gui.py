"""Streamlit GUI for LRTIA - Long-Range Token Influence Analyzer."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="LRTIA - Long-Range Token Influence Analyzer",
    page_icon="🧠",
    layout="wide",
)


def load_corpus_from_file(uploaded_file) -> pd.DataFrame:
    """Load corpus from uploaded file."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".jsonl":
        lines = uploaded_file.read().decode("utf-8").strip().split("\n")
        data = [json.loads(line) for line in lines if line.strip()]
        return pd.DataFrame(data)
    elif suffix == ".json":
        data = json.loads(uploaded_file.read().decode("utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    elif suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix == ".parquet":
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def run_analysis(corpus_df: pd.DataFrame, config: dict, progress_bar, status_text):
    """Run LRTIA analysis on corpus."""
    from lrtia.data.ingestion import Document
    from lrtia.data.windowing import create_windows, get_target_positions
    from lrtia.model import HuggingFaceBackend
    from lrtia.intervention import SpanSelector, apply_intervention
    from lrtia.metrics import compute_all_metrics

    # Convert DataFrame to Documents
    documents = []
    for _, row in corpus_df.iterrows():
        doc = Document(
            doc_id=str(row.get("doc_id", row.get("id", f"doc_{len(documents)}"))),
            author_id=str(row.get("author_id", row.get("author", "unknown"))),
            population=str(row.get("population", row.get("group", "default"))),
            text=str(row.get("text", row.get("content", ""))),
        )
        documents.append(doc)

    # Limit documents if specified
    if config["max_docs"] > 0:
        documents = documents[:config["max_docs"]]

    status_text.text("Loading model...")
    backend = HuggingFaceBackend(
        model_name=config["model"],
        device="auto",
        dtype="auto",
    )
    backend.load()

    # Initialize span selector
    span_selector = SpanSelector(
        distances=config["distances"],
        widths=[config["span_width"]],
        spans_per_distance=1,
        seed=42,
    )

    mask_token_id = backend.get_mask_token_id()
    all_results = []
    total_docs = len(documents)

    for doc_idx, doc in enumerate(documents):
        progress_bar.progress((doc_idx + 1) / total_docs)
        status_text.text(f"Processing document {doc_idx + 1}/{total_docs}: {doc.doc_id}")

        windows = create_windows(
            doc,
            backend.tokenizer,
            window_tokens=config["window_tokens"],
            stride_tokens=config["stride_tokens"],
        )

        for window in windows[:config["max_windows_per_doc"]]:
            targets = get_target_positions(
                window,
                method="interval",
                interval=config["target_interval"],
                min_context=64,
                region_length=config["region_length"],
            )

            for target_start, target_end in targets[:config["max_targets_per_window"]]:
                spans = span_selector.get_all_spans_for_target(
                    target_start=target_start,
                    target_end=target_end,
                    context_length=window.num_tokens,
                    doc_id=doc.doc_id,
                    window_id=window.window_id,
                )

                if not spans:
                    continue

                target_positions = list(range(target_start, min(target_end, window.num_tokens - 1)))
                if not target_positions:
                    continue

                original_output = backend.forward(
                    window.token_ids,
                    positions=target_positions,
                    return_full_distribution=True,
                )

                target_tokens = [window.token_ids[p + 1] for p in target_positions]

                for span in spans:
                    for intervention_type in config["interventions"]:
                        result = apply_intervention(
                            window.token_ids,
                            span,
                            intervention_type,
                            mask_token_id=mask_token_id,
                            seed=42,
                        )

                        if result.length_changed:
                            continue  # Skip length-changing for simplicity

                        intervened_output = backend.forward(
                            result.token_ids,
                            positions=target_positions,
                            return_full_distribution=True,
                        )

                        metrics = compute_all_metrics(
                            original_output.log_probs[0].cpu().numpy(),
                            intervened_output.log_probs[0].cpu().numpy(),
                            target_tokens,
                        )

                        all_results.append({
                            "doc_id": doc.doc_id,
                            "author_id": doc.author_id,
                            "population": doc.population,
                            "distance": span.distance,
                            "intervention_type": intervention_type,
                            "delta_nll": metrics.delta_nll,
                            "kl_divergence": metrics.kl_divergence,
                            "topk_overlap": metrics.topk_overlap,
                            "rank_change": metrics.rank_change,
                        })

    backend.unload()
    status_text.text("Analysis complete!")
    return pd.DataFrame(all_results)


def plot_memory_curves(df: pd.DataFrame, metric: str = "delta_nll", group_by: str = None):
    """Create interactive memory curve plot."""
    if group_by and group_by in df.columns:
        # Group by population
        agg_df = df.groupby(["distance", group_by])[metric].agg(["mean", "std", "count"]).reset_index()
        agg_df.columns = ["distance", group_by, "mean", "std", "count"]

        fig = px.line(
            agg_df,
            x="distance",
            y="mean",
            color=group_by,
            markers=True,
            title=f"Memory Curves by {group_by.title()}",
            labels={"distance": "Distance (tokens)", "mean": metric.replace("_", " ").title()},
        )

        # Add error bands
        for group in agg_df[group_by].unique():
            group_data = agg_df[agg_df[group_by] == group]
            fig.add_traces([
                go.Scatter(
                    x=group_data["distance"],
                    y=group_data["mean"] + group_data["std"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=group_data["distance"],
                    y=group_data["mean"] - group_data["std"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(128,128,128,0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ])
    else:
        # Overall curve
        agg_df = df.groupby("distance")[metric].agg(["mean", "std", "count"]).reset_index()

        fig = px.line(
            agg_df,
            x="distance",
            y="mean",
            markers=True,
            title="Memory Curve (Overall)",
            labels={"distance": "Distance (tokens)", "mean": metric.replace("_", " ").title()},
        )

        # Add error band
        fig.add_traces([
            go.Scatter(
                x=agg_df["distance"],
                y=agg_df["mean"] + agg_df["std"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=agg_df["distance"],
                y=agg_df["mean"] - agg_df["std"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(68,68,68,0.2)",
                showlegend=False,
            ),
        ])

    fig.update_xaxes(type="log", dtick="D1")
    fig.update_layout(height=500)
    return fig


def plot_summary_bars(df: pd.DataFrame, metric: str = "delta_nll"):
    """Create bar chart of summary metrics by population."""
    if "population" not in df.columns:
        return None

    summary = df.groupby("population")[metric].mean().reset_index()
    summary.columns = ["population", "mean"]

    fig = px.bar(
        summary,
        x="population",
        y="mean",
        title=f"Mean {metric.replace('_', ' ').title()} by Population",
        labels={"population": "Population", "mean": f"Mean {metric.replace('_', ' ').title()}"},
        color="population",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_intervention_comparison(df: pd.DataFrame, metric: str = "delta_nll"):
    """Compare effects by intervention type."""
    if "intervention_type" not in df.columns:
        return None

    agg_df = df.groupby(["distance", "intervention_type"])[metric].mean().reset_index()

    fig = px.line(
        agg_df,
        x="distance",
        y=metric,
        color="intervention_type",
        markers=True,
        title="Effect by Intervention Type",
        labels={"distance": "Distance (tokens)", metric: metric.replace("_", " ").title()},
    )
    fig.update_xaxes(type="log", dtick="D1")
    fig.update_layout(height=400)
    return fig


def compute_summary_stats(df: pd.DataFrame, group_col: str = None):
    """Compute summary statistics."""
    from lrtia.aggregation import build_memory_curve, compute_summary_metrics, fit_decay_models

    if group_col and group_col in df.columns:
        summaries = []
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            curve = build_memory_curve(group_df, metric="delta_nll")
            summary = compute_summary_metrics(curve)
            fits = fit_decay_models(curve)

            row = {
                "Group": group,
                "AUC": f"{summary.auc:.4f}",
                "Empirical Half-Life": f"{summary.half_life:.0f}" if summary.half_life else "N/A",
                "Peak Effect": f"{summary.peak_effect:.4f}",
            }

            # Add best fit info
            if fits:
                best_fit = max(fits.values(), key=lambda f: f.r_squared or 0)
                row["Best Model"] = best_fit.model
                row["R²"] = f"{best_fit.r_squared:.3f}" if best_fit.r_squared else "N/A"
                row["Fitted Half-Life"] = f"{best_fit.half_life:.1f}" if best_fit.half_life else "N/A"

            summaries.append(row)
        return pd.DataFrame(summaries)
    else:
        curve = build_memory_curve(df, metric="delta_nll")
        summary = compute_summary_metrics(curve)
        fits = fit_decay_models(curve)

        row = {
            "AUC": f"{summary.auc:.4f}",
            "Empirical Half-Life": f"{summary.half_life:.0f}" if summary.half_life else "N/A",
            "Peak Effect": f"{summary.peak_effect:.4f}",
        }

        if fits:
            best_fit = max(fits.values(), key=lambda f: f.r_squared or 0)
            row["Best Model"] = best_fit.model
            row["R²"] = f"{best_fit.r_squared:.3f}" if best_fit.r_squared else "N/A"
            row["Fitted Half-Life"] = f"{best_fit.half_life:.1f}" if best_fit.half_life else "N/A"

        return pd.DataFrame([row])


def compute_decay_fits(df: pd.DataFrame, metric: str = "delta_nll", group_col: str = None):
    """Compute decay fits for all groups."""
    from lrtia.aggregation import build_memory_curve, fit_decay_models

    results = {}

    if group_col and group_col in df.columns:
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            curve = build_memory_curve(group_df, metric=metric)
            fits = fit_decay_models(curve)
            results[group] = {
                "curve": curve,
                "fits": fits,
            }
    else:
        curve = build_memory_curve(df, metric=metric)
        fits = fit_decay_models(curve)
        results["all"] = {
            "curve": curve,
            "fits": fits,
        }

    return results


def plot_decay_fits(df: pd.DataFrame, metric: str = "delta_nll", group_col: str = None, selected_model: str = "best"):
    """Plot memory curves with fitted decay functions."""
    fit_results = compute_decay_fits(df, metric=metric, group_col=group_col)

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for group_name, data in fit_results.items():
        curve = data["curve"]
        fits = data["fits"]
        color = colors[color_idx % len(colors)]
        color_idx += 1

        if not curve.points:
            continue

        distances = np.array(curve.distances)
        means = np.array(curve.means)
        stds = np.array(curve.stds)

        # Plot actual data points
        fig.add_trace(go.Scatter(
            x=distances,
            y=means,
            mode="markers",
            name=f"{group_name} (data)",
            marker=dict(size=10, color=color),
            error_y=dict(type="data", array=stds, visible=True),
        ))

        # Plot fitted curve
        if fits:
            if selected_model == "best":
                fit = max(fits.values(), key=lambda f: f.r_squared or 0)
            elif selected_model in fits:
                fit = fits[selected_model]
            else:
                continue

            if fit.predicted is not None:
                # Create smooth curve for plotting
                d_smooth = np.linspace(distances.min(), distances.max(), 100)

                if fit.model == "exponential":
                    y_smooth = fit.amplitude * np.exp(-d_smooth / fit.decay_param)
                    param_label = f"τ={fit.decay_param:.1f}"
                elif fit.model == "power_law":
                    y_smooth = fit.amplitude * np.power(np.maximum(d_smooth, 1), -fit.decay_param)
                    param_label = f"α={fit.decay_param:.3f}"
                elif fit.model == "linear":
                    y_smooth = np.maximum(fit.amplitude - fit.decay_param * d_smooth, 0)
                    param_label = f"slope={fit.decay_param:.4f}"
                else:
                    continue

                fig.add_trace(go.Scatter(
                    x=d_smooth,
                    y=y_smooth,
                    mode="lines",
                    name=f"{group_name} ({fit.model}, R²={fit.r_squared:.3f}, {param_label})",
                    line=dict(color=color, dash="dash"),
                ))

    fig.update_layout(
        title=f"Memory Curves with Decay Fit ({selected_model})",
        xaxis_title="Distance (tokens)",
        yaxis_title=metric.replace("_", " ").title(),
        xaxis_type="log",
        height=500,
    )

    return fig


def create_decay_comparison_table(df: pd.DataFrame, metric: str = "delta_nll", group_col: str = None):
    """Create a table comparing all decay models."""
    from lrtia.aggregation import build_memory_curve, compute_half_life

    fit_results = compute_decay_fits(df, metric=metric, group_col=group_col)

    rows = []
    for group_name, data in fit_results.items():
        curve = data["curve"]
        fits = data["fits"]

        # Get empirical half-life
        empirical_hl = compute_half_life(curve)

        for model_name, fit in fits.items():
            rows.append({
                "Group": group_name,
                "Model": model_name,
                "R²": f"{fit.r_squared:.4f}" if fit.r_squared else "N/A",
                "Empirical HL": f"{empirical_hl:.1f}" if empirical_hl else "N/A",
                "Fitted HL": f"{fit.half_life:.1f}" if fit.half_life else "N/A",
                "Decay Param": f"{fit.decay_param:.4f}" if fit.decay_param else "N/A",
            })

    return pd.DataFrame(rows)


def main():
    st.title("🧠 LRTIA - Long-Range Token Influence Analyzer")
    st.markdown("Analyze how earlier text influences language model predictions")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Model")
        model = st.selectbox(
            "Model",
            ["gpt2", "gpt2-medium", "gpt2-large"],
            index=0,
            help="Language model to analyze",
        )

        st.subheader("Analysis Settings")
        max_docs = st.number_input("Max documents", min_value=1, max_value=1000, value=10)
        max_windows = st.number_input("Max windows per doc", min_value=1, max_value=10, value=2)
        max_targets = st.number_input("Max targets per window", min_value=1, max_value=10, value=3)

        st.subheader("Distances")

        # Preset distance configurations
        distance_presets = {
            "Default": [32, 64, 128, 256],
            "Fine-grained (short)": [8, 16, 32, 48, 64, 96, 128],
            "Long-range": [64, 128, 256, 512, 1024],
            "Full spectrum": [16, 32, 64, 128, 256, 512, 1024],
            "Dense sampling": [16, 24, 32, 48, 64, 96, 128, 192, 256],
            "Custom": [],
        }

        preset = st.selectbox(
            "Distance preset",
            list(distance_presets.keys()),
            index=0,
            help="Choose a preset or select 'Custom' to enter your own",
        )

        if preset == "Custom":
            custom_distances = st.text_input(
                "Custom distances (comma-separated)",
                value="32, 64, 128, 256",
                help="Enter distance values separated by commas, e.g., '20, 40, 80, 160'",
            )
            try:
                distances = [int(d.strip()) for d in custom_distances.split(",") if d.strip()]
                distances = sorted(set(distances))  # Remove duplicates and sort
                if distances:
                    st.caption(f"Using distances: {distances}")
                else:
                    st.warning("Please enter at least one distance value")
                    distances = [32, 64, 128, 256]
            except ValueError:
                st.error("Invalid input. Please enter comma-separated numbers.")
                distances = [32, 64, 128, 256]
        else:
            distances = distance_presets[preset]
            # Also allow modifying presets
            distances = st.multiselect(
                "Distance buckets (modify if needed)",
                [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
                default=distances,
            )

        st.subheader("Interventions")
        interventions = st.multiselect(
            "Intervention types",
            ["mask", "shuffle"],
            default=["mask"],
        )

        st.subheader("Advanced")
        window_tokens = st.number_input("Window size (tokens)", min_value=128, max_value=1024, value=512)
        span_width = st.number_input("Span width", min_value=8, max_value=64, value=16)
        target_interval = st.number_input("Target interval", min_value=50, max_value=500, value=150)
        region_length = st.number_input("Region length", min_value=5, max_value=50, value=20)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📂 Load Data", "🔬 Run Analysis", "📊 Results"])

    # Initialize session state
    if "corpus_df" not in st.session_state:
        st.session_state.corpus_df = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None

    with tab1:
        st.header("Load Corpus")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload File")
            uploaded_file = st.file_uploader(
                "Choose a corpus file",
                type=["jsonl", "json", "csv", "parquet"],
                help="File should have columns: doc_id, text, and optionally author_id, population",
            )

            if uploaded_file is not None:
                try:
                    st.session_state.corpus_df = load_corpus_from_file(uploaded_file)
                    st.success(f"Loaded {len(st.session_state.corpus_df)} documents")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with col2:
            st.subheader("Or Use Demo Data")
            if st.button("Generate Demo Corpus"):
                # Generate synthetic data
                import random
                random.seed(42)

                docs = []
                words = ["the", "a", "is", "are", "was", "were", "be", "been", "have", "has",
                         "do", "does", "did", "will", "would", "could", "should", "may", "might",
                         "this", "that", "these", "those", "it", "they", "we", "you", "he", "she"]

                for i in range(20):
                    pop = "group_a" if i < 10 else "group_b"
                    text = " ".join(random.choices(words, k=random.randint(200, 400)))
                    docs.append({
                        "doc_id": f"doc_{i:03d}",
                        "author_id": f"author_{i % 5}",
                        "population": pop,
                        "text": text,
                    })

                st.session_state.corpus_df = pd.DataFrame(docs)
                st.success("Generated 20 demo documents")

        if st.session_state.corpus_df is not None:
            st.subheader("Corpus Preview")
            st.dataframe(st.session_state.corpus_df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", len(st.session_state.corpus_df))
            with col2:
                if "population" in st.session_state.corpus_df.columns:
                    st.metric("Populations", st.session_state.corpus_df["population"].nunique())
            with col3:
                if "author_id" in st.session_state.corpus_df.columns:
                    st.metric("Authors", st.session_state.corpus_df["author_id"].nunique())

    with tab2:
        st.header("Run Analysis")

        if st.session_state.corpus_df is None:
            st.warning("Please load a corpus first")
        else:
            st.info(f"Ready to analyze {min(max_docs, len(st.session_state.corpus_df))} documents with model: {model}")

            config = {
                "model": model,
                "max_docs": max_docs,
                "max_windows_per_doc": max_windows,
                "max_targets_per_window": max_targets,
                "distances": sorted(distances),
                "interventions": interventions,
                "window_tokens": window_tokens,
                "stride_tokens": window_tokens // 2,
                "span_width": span_width,
                "target_interval": target_interval,
                "region_length": region_length,
            }

            if st.button("🚀 Run Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    st.session_state.results_df = run_analysis(
                        st.session_state.corpus_df,
                        config,
                        progress_bar,
                        status_text,
                    )
                    st.success(f"Analysis complete! {len(st.session_state.results_df)} evaluations")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab3:
        st.header("Results & Visualization")

        if st.session_state.results_df is None:
            st.warning("No results yet. Run an analysis first.")
        else:
            df = st.session_state.results_df

            # Metric selector
            metric = st.selectbox(
                "Metric",
                ["delta_nll", "kl_divergence", "topk_overlap", "rank_change"],
                index=0,
            )

            # Summary statistics
            st.subheader("Summary Statistics")
            has_populations = "population" in df.columns and df["population"].nunique() > 1
            summary_df = compute_summary_stats(df, "population" if has_populations else None)
            st.dataframe(summary_df, use_container_width=True)

            # Main memory curve plot
            st.subheader("Memory Curves")
            group_by = "population" if has_populations else None
            fig = plot_memory_curves(df, metric=metric, group_by=group_by)
            st.plotly_chart(fig, use_container_width=True)

            # Decay Fit Analysis
            st.subheader("Decay Model Fitting")
            st.markdown("""
            Fit mathematical decay functions to quantify how influence decreases with distance:
            - **Exponential**: y = A·e^(-d/τ) — τ is the decay constant
            - **Power Law**: y = A·d^(-α) — α is the decay exponent
            - **Linear**: y = A - slope·d — simple linear decrease
            """)

            col_model, col_info = st.columns([1, 2])
            with col_model:
                selected_model = st.selectbox(
                    "Decay model to display",
                    ["best", "exponential", "power_law", "linear"],
                    index=0,
                    help="'best' selects the model with highest R²",
                )

            # Plot with decay fits
            decay_fig = plot_decay_fits(df, metric=metric, group_col=group_by, selected_model=selected_model)
            st.plotly_chart(decay_fig, use_container_width=True)

            # Decay comparison table
            st.markdown("**All Decay Models Comparison:**")
            decay_table = create_decay_comparison_table(df, metric=metric, group_col=group_by)
            if not decay_table.empty:
                st.dataframe(decay_table, use_container_width=True)

                # Highlight key metrics
                st.markdown("**Key Decay Scores:**")
                cols = st.columns(len(decay_table["Group"].unique()))
                for i, group in enumerate(decay_table["Group"].unique()):
                    group_fits = decay_table[decay_table["Group"] == group]
                    best_row = group_fits.loc[group_fits["R²"].apply(lambda x: float(x) if x != "N/A" else 0).idxmax()]
                    with cols[i]:
                        st.markdown(f"**{group}** ({best_row['Model']})")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Empirical HL", f"{best_row['Empirical HL']} tok")
                        with col_b:
                            st.metric("Fitted HL", f"{best_row['Fitted HL']} tok")
                        st.caption(f"R² = {best_row['R²']}")

            st.divider()

            # Additional plots
            col1, col2 = st.columns(2)

            with col1:
                if has_populations:
                    st.subheader("Population Comparison")
                    bar_fig = plot_summary_bars(df, metric=metric)
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)

            with col2:
                if "intervention_type" in df.columns and df["intervention_type"].nunique() > 1:
                    st.subheader("Intervention Comparison")
                    int_fig = plot_intervention_comparison(df, metric=metric)
                    if int_fig:
                        st.plotly_chart(int_fig, use_container_width=True)

            # Raw data
            st.subheader("Raw Data")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "lrtia_results.csv",
                "text/csv",
            )


if __name__ == "__main__":
    main()
