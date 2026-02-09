"""
Cross-corpus DIF comparison + PERSUADE closure control.

Produces:
  results/dif_analysis_v1/persuade/persuade_dif_closure.csv
  results/dif_cross_corpus_v1/dif_cross_corpus_table.csv
  results/dif_cross_corpus_v1/kernel_overlay_all_corpora.png
  results/dif_cross_corpus_v1/kernel_distance_matrix.csv
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from dif_pipeline import (
    K_GRID, N_BINS, BIN_LABELS, BIN_RIGHTS, N_BOOT, SEED,
    compute_kernel, compute_dif_scalars, boot_ci,
)


# =====================================================================
# PERSUADE closure control (early+mid only)
# =====================================================================

def persuade_closure_control(
    window_csv: str | Path,
    essay_csv: str | Path,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Compute early+mid kernel for PERSUADE from window-level data.

    Assigns position quartiles within each essay, aggregates only Q1+Q2
    windows, then computes DIF on those aggregated gain curves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== PERSUADE Closure Control ===")
    win = pd.read_csv(window_csv)
    essays = pd.read_csv(essay_csv)

    # Pivot windows to wide format: one row per (essay_id, window_idx)
    win_wide = win.pivot_table(
        index=["essay_id", "window_idx", "target_start"],
        columns="k", values="gain_k"
    ).reset_index()
    win_wide.columns = [
        f"gain_{int(c)}" if isinstance(c, (int, float, np.integer, np.floating))
        else c
        for c in win_wide.columns
    ]

    # Assign quartiles within each essay
    def assign_quartile(grp):
        n = len(grp)
        grp = grp.sort_values("target_start")
        grp["pos_quartile"] = pd.qcut(
            range(n), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        return grp

    win_wide = win_wide.groupby("essay_id", group_keys=False).apply(
        assign_quartile, include_groups=True
    )
    print(f"  Windows: {len(win_wide)}")
    print(f"  Quartile distribution:\n{win_wide['pos_quartile'].value_counts().sort_index().to_string()}")

    # Filter to early+mid (Q1+Q2) and aggregate per essay
    em = win_wide[win_wide["pos_quartile"].isin(["Q1", "Q2"])]
    gain_cols = [f"gain_{k}" for k in K_GRID]
    em_agg = em.groupby("essay_id")[gain_cols].mean().reset_index()
    em_agg.rename(columns={f"gain_{k}": f"mean_gain_{k}" for k in K_GRID},
                  inplace=True)

    # Compute per-doc kernel on early+mid data
    rows = []
    for _, row in em_agg.iterrows():
        gains = {k: row[f"mean_gain_{k}"] for k in K_GRID}
        if gains[128] <= 0:
            continue
        shares = compute_kernel(gains)
        if shares is None:
            continue
        shares = np.maximum(shares, 0)
        s = shares.sum()
        if s > 0:
            shares /= s
        scalars = compute_dif_scalars(shares)
        rec = {"essay_id": row["essay_id"]}
        rec.update(scalars)
        for i in range(N_BINS):
            rec[f"p_{i}"] = shares[i]
        rows.append(rec)

    em_doc = pd.DataFrame(rows)
    print(f"  Early+mid valid docs: {len(em_doc)}")

    # Bootstrap CIs for early+mid
    rng = np.random.default_rng(SEED)
    closure_rows = []
    for m in ["L50", "L80", "L90", "tail_33_128", "tail_65_128", "mean_lag"]:
        mn, lo, hi = boot_ci(em_doc[m].values, n=N_BOOT, rng=rng)
        closure_rows.append({"metric": m, "mean": mn, "ci_lo": lo, "ci_hi": hi,
                             "condition": "early+mid"})

    # Also add gain_128 for early+mid from the aggregated data
    em_g128 = em_agg.merge(em_doc[["essay_id"]], on="essay_id")["mean_gain_128"]
    mn, lo, hi = boot_ci(em_g128.values, n=N_BOOT, rng=rng)
    closure_rows.append({"metric": "gain128", "mean": mn, "ci_lo": lo, "ci_hi": hi,
                         "condition": "early+mid"})

    closure_df = pd.DataFrame(closure_rows)

    # Load full DIF summary and append early+mid
    full_dif = pd.read_csv(out_dir / "persuade_dif_summary.csv")
    combined = pd.concat([full_dif, closure_df], ignore_index=True)
    combined.to_csv(out_dir / "persuade_dif_closure.csv", index=False)

    # Bin-level comparison
    em_kernel = np.array([em_doc[f"p_{i}"].mean() for i in range(N_BINS)])
    full_kernel_df = pd.read_csv(out_dir / "persuade_kernel_summary.csv")
    full_kernel = full_kernel_df["share_mean"].values

    print("\n  Kernel comparison (full vs early+mid):")
    max_diff = 0
    for i in range(N_BINS):
        diff = em_kernel[i] - full_kernel[i]
        max_diff = max(max_diff, abs(diff))
        print(f"    {BIN_LABELS[i]:12s}: full={full_kernel[i]:.4f}  "
              f"em={em_kernel[i]:.4f}  diff={diff:+.4f}")
    print(f"  Max absolute bin difference: {max_diff:.4f}")

    # Print closure DIF scalars
    print("\n  Early+mid DIF scalars:")
    for _, r in closure_df.iterrows():
        if r["metric"] == "gain128":
            print(f"    {r['metric']:14s}: {r['mean']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        elif "tail" in r["metric"]:
            print(f"    {r['metric']:14s}: {r['mean']:.1%}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        else:
            print(f"    {r['metric']:14s}: {r['mean']:.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    return em_kernel


# =====================================================================
# Cross-corpus comparison
# =====================================================================

def load_corpus_results(corpus_name: str, base_dir: str | Path) -> dict:
    """Load kernel summary and DIF summary for a corpus."""
    d = Path(base_dir) / corpus_name
    kernel = pd.read_csv(d / f"{corpus_name}_kernel_summary.csv")
    dif = pd.read_csv(d / f"{corpus_name}_dif_summary.csv")
    doc = pd.read_csv(d / f"{corpus_name}_doc_level_kernel.csv")
    return {
        "name": corpus_name,
        "kernel": kernel["share_mean"].values,
        "kernel_ci_lo": kernel["ci_lo"].values,
        "kernel_ci_hi": kernel["ci_hi"].values,
        "dif": dif,
        "n_docs": len(doc),
        "median_tokens": doc["token_count"].median() if "token_count" in doc.columns else np.nan,
        "mean_baseline_nll": doc["baseline_nll"].mean() if "baseline_nll" in doc.columns else np.nan,
    }


def cross_corpus_table(corpora: list[dict], out_dir: Path) -> pd.DataFrame:
    """Build comparison table: rows=corpora, columns=metrics."""
    rows = []
    for c in corpora:
        dif = c["dif"]
        rec = {"corpus": c["name"], "n_docs": c["n_docs"],
               "median_tokens": c["median_tokens"],
               "baseline_nll": c["mean_baseline_nll"]}
        for _, r in dif.iterrows():
            m = r["metric"]
            rec[f"{m}_mean"] = r["mean"]
            rec[f"{m}_ci_lo"] = r["ci_lo"]
            rec[f"{m}_ci_hi"] = r["ci_hi"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dif_cross_corpus_table.csv", index=False)

    print("\n=== Cross-Corpus DIF Comparison ===")
    print(df.to_string(index=False))
    return df


def kernel_distances(corpora: list[dict], out_dir: Path) -> pd.DataFrame:
    """Compute L2 and Jensen-Shannon distances between corpus kernels."""
    names = [c["name"] for c in corpora]
    n = len(names)
    l2_mat = np.zeros((n, n))
    js_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ki = corpora[i]["kernel"]
            kj = corpora[j]["kernel"]
            l2_mat[i, j] = np.linalg.norm(ki - kj)
            # Jensen-Shannon requires non-negative, normalized distributions
            ki_safe = np.maximum(ki, 0)
            kj_safe = np.maximum(kj, 0)
            ki_safe /= ki_safe.sum()
            kj_safe /= kj_safe.sum()
            js_mat[i, j] = jensenshannon(ki_safe, kj_safe)

    # Save combined
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "corpus_a": names[i], "corpus_b": names[j],
                "L2": l2_mat[i, j], "JensenShannon": js_mat[i, j],
            })
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "kernel_distance_matrix.csv", index=False)

    print("\n=== Kernel Distance Matrix ===")
    print("L2 distances:")
    l2_df = pd.DataFrame(l2_mat, index=names, columns=names)
    print(l2_df.to_string())
    print("\nJensen-Shannon distances:")
    js_df = pd.DataFrame(js_mat, index=names, columns=names)
    print(js_df.to_string())

    return dist_df


def plot_kernel_overlay_all(corpora: list[dict], out_dir: Path):
    """Multi-corpus kernel overlay: bar chart + cumulative curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(N_BINS)
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]

    # Bar chart
    ax = axes[0]
    n_corpora = len(corpora)
    width = 0.7 / n_corpora
    for idx, c in enumerate(corpora):
        offset = (idx - n_corpora / 2 + 0.5) * width
        ax.bar(x + offset, c["kernel"], width, color=colors[idx % len(colors)],
               alpha=0.8, label=c["name"].upper(), edgecolor="white")

    # Tail region
    tail_x = [i for i in range(N_BINS) if K_GRID[i] >= 32]
    ax.axvspan(tail_x[0] - 0.5, N_BINS - 0.5, alpha=0.06, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of gain(128)")
    ax.set_title("Influence Kernel: Cross-Corpus Comparison")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative curve
    ax2 = axes[1]
    for idx, c in enumerate(corpora):
        cum = np.cumsum(c["kernel"])
        ax2.plot(BIN_RIGHTS, cum, "o-", color=colors[idx % len(colors)],
                 label=c["name"].upper(), linewidth=2, markersize=5)

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax2.annotate(lbl, xy=(1.5, q + 0.01), fontsize=9, color="gray")

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("Cumulative Kernel: Cross-Corpus")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 150)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "kernel_overlay_all_corpora.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir / 'kernel_overlay_all_corpora.png'}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    base = Path("results/dif_analysis_v1")
    cross_dir = Path("results/dif_cross_corpus_v1")
    cross_dir.mkdir(parents=True, exist_ok=True)

    # --- PERSUADE closure control ---
    em_kernel = persuade_closure_control(
        window_csv="results/local_lag_curve_sliding_v1/window_level_results_local_lag.csv",
        essay_csv="results/local_lag_curve_sliding_v1/essay_level_results_local_lag.csv",
        out_dir=base / "persuade",
    )

    # --- Load corpus results ---
    ecsc = load_corpus_results("ecsc", base)
    persuade = load_corpus_results("persuade", base)
    corpora = [ecsc, persuade]

    # --- Cross-corpus table ---
    cross_corpus_table(corpora, cross_dir)

    # --- Kernel distances ---
    kernel_distances(corpora, cross_dir)

    # --- Overlay figure ---
    plot_kernel_overlay_all(corpora, cross_dir)

    print("\nDone. Cross-corpus outputs in results/dif_cross_corpus_v1/")
