"""
DIF v1 Pipeline — Decay Influence Factor analysis.

Canonical definition (frozen):
  - Kernel bins = incremental influence shares
      p_{a→b} = (gain_b - gain_a) / gain_128
  - DIF scalars per doc: L50, L80, L90, mean-lag, Tail_33-128, Tail_65-128
  - Compute per document, bootstrap documents for CI.

Usage:
    from dif_pipeline import run_dif_pipeline
    run_dif_pipeline(df, corpus_name="ecsc", out_dir="results/dif_analysis_v1/ecsc")

Or as CLI:
    python scripts/dif_pipeline.py --corpus ecsc --input results/.../transcript_level_metrics.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS
# =====================================================================
K_GRID = np.array([0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
N_BINS = len(K_GRID) - 1
N_BOOT = 2000
SEED = 42


def bin_label(i):
    lo = K_GRID[i] + 1 if K_GRID[i] > 0 else 1
    hi = K_GRID[i + 1]
    return f"lag {lo}-{hi}"


BIN_LABELS = [bin_label(i) for i in range(N_BINS)]
BIN_RIGHTS = K_GRID[1:]


# =====================================================================
# CORE FUNCTIONS
# =====================================================================

def compute_kernel(gains_by_k: dict[int, float]) -> np.ndarray:
    """Compute normalised kernel shares from gain values.

    Parameters
    ----------
    gains_by_k : dict mapping k -> gain(k) for k in K_GRID

    Returns
    -------
    shares : ndarray of shape (N_BINS,), sums to ~1.
             Returns None if gain_128 <= 0.
    """
    g128 = gains_by_k[128]
    if g128 <= 0:
        return None

    shares = np.empty(N_BINS)
    for i in range(N_BINS):
        delta = gains_by_k[K_GRID[i + 1]] - gains_by_k[K_GRID[i]]
        shares[i] = delta / g128

    return shares


def check_monotonicity(gains_by_k: dict[int, float]) -> tuple[bool, list[str]]:
    """Check that gains are monotonically non-decreasing with k."""
    warnings = []
    prev_k, prev_g = 0, 0.0
    for k in K_GRID[1:]:
        g = gains_by_k[k]
        if g < prev_g - 1e-9:
            warnings.append(f"gain({k})={g:.4f} < gain({prev_k})={prev_g:.4f}")
        prev_k, prev_g = k, g
    return len(warnings) == 0, warnings


def check_nonnegativity(shares: np.ndarray) -> tuple[bool, list[str]]:
    """Check that all bin shares are non-negative."""
    warnings = []
    for i in range(N_BINS):
        if shares[i] < -1e-9:
            warnings.append(f"bin {BIN_LABELS[i]}: share={shares[i]:.4f} < 0")
    return len(warnings) == 0, warnings


def check_normalization(shares: np.ndarray, tol: float = 1e-6) -> bool:
    """Check that shares sum to 1 within tolerance."""
    return abs(shares.sum() - 1.0) < tol


def compute_Lq(shares: np.ndarray, q: float) -> float:
    """Smallest lag where cumulative share >= q, with linear interpolation."""
    cum = 0.0
    for i in range(N_BINS):
        prev_cum = cum
        cum += shares[i]
        if cum >= q:
            frac = (q - prev_cum) / (shares[i] + 1e-15)
            lo = K_GRID[i] + 1 if K_GRID[i] > 0 else 1
            hi = K_GRID[i + 1]
            return lo + frac * (hi - lo)
    return float(K_GRID[-1])


def compute_mean_lag(shares: np.ndarray) -> float:
    """Expected lag under the kernel."""
    total = 0.0
    for i in range(N_BINS):
        lo = K_GRID[i] + 1 if K_GRID[i] > 0 else 1
        hi = K_GRID[i + 1]
        mid = (lo + hi) / 2.0
        total += shares[i] * mid
    return total


def compute_dif_scalars(shares: np.ndarray) -> dict:
    """Compute all DIF scalars from a kernel share vector."""
    return {
        "L50": compute_Lq(shares, 0.50),
        "L80": compute_Lq(shares, 0.80),
        "L90": compute_Lq(shares, 0.90),
        "tail_33_128": sum(shares[i] for i in range(N_BINS) if K_GRID[i] >= 32),
        "tail_65_128": sum(shares[i] for i in range(N_BINS) if K_GRID[i] >= 64),
        "mean_lag": compute_mean_lag(shares),
    }


def boot_ci(vals: np.ndarray, n: int = N_BOOT, alpha: float = 0.05,
            rng: np.random.Generator = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    vals = np.asarray(vals)
    means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                       for _ in range(n)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(vals.mean()), float(lo), float(hi)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_dif_pipeline(
    df: pd.DataFrame,
    corpus_name: str,
    out_dir: str | Path,
    doc_id_col: str = "doc_id",
    gain_col_template: str = "mean_gain_{k}",
    metadata_cols: list[str] | None = None,
    closure_kernel: np.ndarray | None = None,
    closure_label: str = "early+mid",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Run the full DIF v1 pipeline on a corpus.

    Parameters
    ----------
    df : DataFrame with one row per document.
    corpus_name : short name for file prefixes.
    out_dir : output directory.
    doc_id_col : column name for document IDs.
    gain_col_template : template for gain columns, with {k} placeholder.
    metadata_cols : extra columns to carry through to doc-level output.
    closure_kernel : optional N_BINS-length array of early+mid kernel shares
                     for overlay comparison.
    closure_label : label for the closure control condition.
    n_boot : bootstrap iterations.
    seed : random seed.

    Returns
    -------
    dict with keys: 'doc_df', 'kernel_summary', 'dif_summary', 'diagnostics'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if metadata_cols is None:
        metadata_cols = []

    # ------------------------------------------------------------------
    # Step 1: Compute kernel per doc
    # ------------------------------------------------------------------
    gain_cols = [gain_col_template.format(k=k) for k in K_GRID]
    for c in gain_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    rows = []
    diagnostics = {"total": len(df), "valid": 0, "gain128_le0": 0,
                   "non_monotone": 0, "negative_shares": 0}

    for _, row in df.iterrows():
        gains = {k: row[gain_col_template.format(k=k)] for k in K_GRID}

        # Skip if gain_128 <= 0
        if gains[128] <= 0:
            diagnostics["gain128_le0"] += 1
            continue

        # Monotonicity check
        mono_ok, mono_warns = check_monotonicity(gains)
        if not mono_ok:
            diagnostics["non_monotone"] += 1

        # Compute kernel
        shares = compute_kernel(gains)
        if shares is None:
            continue

        # Non-negativity check (allow small violations from non-monotonicity)
        nn_ok, nn_warns = check_nonnegativity(shares)
        if not nn_ok:
            diagnostics["negative_shares"] += 1
            # Clamp negatives to zero and renormalize
            shares = np.maximum(shares, 0)
            s = shares.sum()
            if s > 0:
                shares /= s

        # Normalization check
        assert check_normalization(shares), f"Shares sum to {shares.sum()}"

        # DIF scalars
        scalars = compute_dif_scalars(shares)

        rec = {
            doc_id_col: row[doc_id_col],
            "gain128": gains[128],
            "baseline_nll": row.get("baseline_nll", row.get("mean_nll_128", np.nan)),
            "token_count": row.get("token_count", np.nan),
        }
        for mc in metadata_cols:
            rec[mc] = row.get(mc, np.nan)
        rec.update(scalars)
        for i in range(N_BINS):
            rec[f"p_{i}"] = shares[i]

        rows.append(rec)
        diagnostics["valid"] += 1

    doc_df = pd.DataFrame(rows)

    print(f"\n[{corpus_name}] Diagnostics:")
    print(f"  Total docs: {diagnostics['total']}")
    print(f"  Valid (gain_128 > 0): {diagnostics['valid']}")
    print(f"  Excluded (gain_128 <= 0): {diagnostics['gain128_le0']}")
    print(f"  Non-monotone gains: {diagnostics['non_monotone']}")
    print(f"  Negative bin shares (clamped): {diagnostics['negative_shares']}")

    # Save doc-level
    doc_df.to_csv(out_dir / f"{corpus_name}_doc_level_kernel.csv", index=False)

    # ------------------------------------------------------------------
    # Step 2: Bootstrap summary
    # ------------------------------------------------------------------
    share_cols = [f"p_{i}" for i in range(N_BINS)]
    scalar_names = ["L50", "L80", "L90", "tail_33_128", "tail_65_128",
                    "mean_lag", "gain128"]

    # Kernel summary
    kernel_rows = []
    for i in range(N_BINS):
        mn, lo, hi = boot_ci(doc_df[f"p_{i}"].values, n=n_boot, rng=rng)
        kernel_rows.append({
            "bin": BIN_LABELS[i], "share_mean": mn, "ci_lo": lo, "ci_hi": hi
        })
    kernel_df = pd.DataFrame(kernel_rows)
    kernel_df.to_csv(out_dir / f"{corpus_name}_kernel_summary.csv", index=False)

    # Scalar summary
    dif_rows = []
    print(f"\n[{corpus_name}] DIF scalars (N={len(doc_df)}):")
    for m in scalar_names:
        mn, lo, hi = boot_ci(doc_df[m].values, n=n_boot, rng=rng)
        dif_rows.append({"metric": m, "mean": mn, "ci_lo": lo, "ci_hi": hi,
                         "condition": "full"})
        fmt = f"{mn:.1%}" if "tail" in m else f"{mn:.2f}"
        print(f"  {m:14s}: {fmt}  [{lo:.2f}, {hi:.2f}]")

    dif_df = pd.DataFrame(dif_rows)
    dif_df.to_csv(out_dir / f"{corpus_name}_dif_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Step 3: Figures
    # ------------------------------------------------------------------
    _plot_kernel_overlay(doc_df, kernel_df, corpus_name, out_dir,
                         closure_kernel=closure_kernel,
                         closure_label=closure_label)
    _plot_L50_distribution(doc_df, corpus_name, out_dir,
                           group_col=metadata_cols[0] if metadata_cols else None)

    return {
        "doc_df": doc_df,
        "kernel_summary": kernel_df,
        "dif_summary": dif_df,
        "diagnostics": diagnostics,
    }


# =====================================================================
# PLOTTING
# =====================================================================

def _plot_kernel_overlay(doc_df, kernel_df, corpus_name, out_dir,
                         closure_kernel=None, closure_label="early+mid"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(N_BINS)
    full_means = kernel_df["share_mean"].values
    full_errs = kernel_df["ci_hi"].values - full_means

    # Bar chart
    ax = axes[0]
    if closure_kernel is not None:
        width = 0.35
        ax.bar(x - width / 2, full_means, width, yerr=full_errs, capsize=2,
               label="Full sample", color="#3498db", alpha=0.8, edgecolor="#2980b9")
        ax.bar(x + width / 2, closure_kernel, width,
               label=closure_label, color="#e67e22", alpha=0.8, edgecolor="#d35400")
    else:
        ax.bar(x, full_means, 0.6, yerr=full_errs, capsize=2,
               color="#3498db", alpha=0.8, edgecolor="#2980b9", label="Full sample")

    # Tail annotation
    tail_x = [i for i in range(N_BINS) if K_GRID[i] >= 32]
    ax.axvspan(tail_x[0] - 0.5, N_BINS - 0.5, alpha=0.08, color="red")
    tail33 = sum(full_means[i] for i in tail_x)
    ax.annotate(f"Tail 33-128\n= {tail33:.1%}",
                xy=(tail_x[1], max(full_means) * 0.3),
                fontsize=9, color="#c0392b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of gain(128)")
    ax.set_title(f"{corpus_name.upper()}: Influence Kernel")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative curve
    ax2 = axes[1]
    cum_full = np.cumsum(full_means)
    ax2.plot(BIN_RIGHTS, cum_full, "o-", color="#3498db",
             label="Full sample", linewidth=2)
    if closure_kernel is not None:
        cum_em = np.cumsum(closure_kernel)
        ax2.plot(BIN_RIGHTS, cum_em, "s--", color="#e67e22",
                 label=closure_label, linewidth=2)

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax2.annotate(lbl, xy=(1.5, q + 0.01), fontsize=9, color="gray")

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title(f"{corpus_name.upper()}: Cumulative Kernel")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 150)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{corpus_name}_kernel_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_L50_distribution(doc_df, corpus_name, out_dir, group_col=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram
    ax = axes[0]
    ax.hist(doc_df["L50"], bins=30, color="#3498db", alpha=0.7, edgecolor="#2980b9")
    mn = doc_df["L50"].mean()
    md = doc_df["L50"].median()
    ax.axvline(mn, color="#e74c3c", linewidth=2, label=f"Mean = {mn:.1f}")
    ax.axvline(md, color="#f39c12", linewidth=2, linestyle="--",
               label=f"Median = {md:.1f}")
    ax.set_xlabel("L50 (tokens)")
    ax.set_ylabel("Number of documents")
    ax.set_title(f"{corpus_name.upper()}: L50 Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Boxplot by group
    ax2 = axes[1]
    if group_col and group_col in doc_df.columns:
        groups = sorted(doc_df[group_col].dropna().unique())
        data = [doc_df[doc_df[group_col] == g]["L50"].values for g in groups]
        data = [d for d, g in zip(data, groups) if len(d) >= 3]
        labels = [g for g in groups if len(doc_df[doc_df[group_col] == g]) >= 3]
        if data:
            bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True,
                             medianprops=dict(color="#e74c3c", linewidth=2))
            colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22",
                      "#e74c3c", "#9b59b6"]
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[j % len(colors)])
                patch.set_alpha(0.6)
            ax2.set_xlabel(group_col)
    else:
        ax2.text(0.5, 0.5, "No grouping variable", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")

    ax2.set_ylabel("L50 (tokens)")
    ax2.set_title(f"{corpus_name.upper()}: L50 by Group")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / f"{corpus_name}_L50_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIF v1 Pipeline")
    parser.add_argument("--corpus", required=True, help="Corpus name")
    parser.add_argument("--input", required=True, help="Path to gain-curve CSV")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--doc-id", default="doc_id", help="Doc ID column name")
    parser.add_argument("--metadata", nargs="*", default=[], help="Metadata columns")
    args = parser.parse_args()

    out = args.output or f"results/dif_analysis_v1/{args.corpus}"
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} documents from {args.input}")

    run_dif_pipeline(
        df, corpus_name=args.corpus, out_dir=out,
        doc_id_col=args.doc_id, metadata_cols=args.metadata,
    )
    print(f"\nDone. Outputs in {out}/")
