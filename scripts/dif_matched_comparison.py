"""
Length- and difficulty-matched cross-corpus DIF comparison.

Matches ECSC and PERSUADE documents on:
  - token_count (document length)
  - baseline_nll (no-context perplexity / difficulty)

Uses nearest-neighbor matching (1:1 without replacement) to create
balanced subsets, then recomputes DIF + kernel distances.

Produces:
  results/dif_cross_corpus_v1/matched_ecsc_docs.csv
  results/dif_cross_corpus_v1/matched_persuade_docs.csv
  results/dif_cross_corpus_v1/matched_comparison_table.csv
  results/dif_cross_corpus_v1/matched_kernel_overlay.png
  results/dif_cross_corpus_v1/matched_kernel_distances.csv
  results/dif_cross_corpus_v1/matching_diagnostics.png
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon

from dif_pipeline import (
    K_GRID, N_BINS, BIN_LABELS, BIN_RIGHTS, N_BOOT, SEED,
    compute_dif_scalars, boot_ci,
)


def nearest_neighbor_match(
    target_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    match_cols: list[str],
    caliper: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match target docs to pool docs using nearest-neighbor on standardized features.

    Parameters
    ----------
    target_df : The smaller corpus (match from this)
    pool_df : The larger corpus (match into this)
    match_cols : Columns to match on
    caliper : Max allowed Mahalanobis distance (in SD units per variable)

    Returns
    -------
    matched_target, matched_pool, diagnostics_df
    """
    # Standardize matching variables using pooled stats
    combined = pd.concat([
        target_df[match_cols].assign(source="target"),
        pool_df[match_cols].assign(source="pool"),
    ], ignore_index=True)

    means = combined[match_cols].mean()
    stds = combined[match_cols].std()

    target_z = ((target_df[match_cols] - means) / stds).values
    pool_z = ((pool_df[match_cols] - means) / stds).values

    # Euclidean distance in standardized space
    dists = cdist(target_z, pool_z, metric="euclidean")

    # Greedy 1:1 matching without replacement
    matched_target_idx = []
    matched_pool_idx = []
    used_pool = set()

    # Sort target docs by their best available match (hardest to match first)
    best_dists = dists.min(axis=1)
    order = np.argsort(-best_dists)  # hardest first

    for t_idx in order:
        row_dists = dists[t_idx].copy()
        row_dists[list(used_pool)] = np.inf
        best_p = np.argmin(row_dists)
        best_d = row_dists[best_p]

        if best_d <= caliper * np.sqrt(len(match_cols)):
            matched_target_idx.append(t_idx)
            matched_pool_idx.append(best_p)
            used_pool.add(best_p)

    m_target = target_df.iloc[matched_target_idx].copy().reset_index(drop=True)
    m_pool = pool_df.iloc[matched_pool_idx].copy().reset_index(drop=True)

    # Diagnostics
    diag_rows = []
    for col in match_cols:
        diag_rows.append({
            "variable": col,
            "target_mean_pre": target_df[col].mean(),
            "pool_mean_pre": pool_df[col].mean(),
            "smd_pre": (target_df[col].mean() - pool_df[col].mean()) / combined[col].std(),
            "target_mean_post": m_target[col].mean(),
            "pool_mean_post": m_pool[col].mean(),
            "smd_post": (m_target[col].mean() - m_pool[col].mean()) / combined[col].std(),
        })
    diag = pd.DataFrame(diag_rows)

    return m_target, m_pool, diag


def compute_subset_dif(doc_df: pd.DataFrame, corpus_name: str,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Compute DIF summary from pre-computed doc-level kernel shares."""
    share_cols = [f"p_{i}" for i in range(N_BINS)]
    rng = np.random.default_rng(seed)

    kernel = doc_df[share_cols].mean().values
    scalars = {}
    for m in ["L50", "L80", "L90", "tail_33_128", "tail_65_128", "mean_lag"]:
        mn, lo, hi = boot_ci(doc_df[m].values, n=n_boot, rng=rng)
        scalars[m] = {"mean": mn, "ci_lo": lo, "ci_hi": hi}

    # gain128
    mn, lo, hi = boot_ci(doc_df["gain128"].values, n=n_boot, rng=rng)
    scalars["gain128"] = {"mean": mn, "ci_lo": lo, "ci_hi": hi}

    return {"kernel": kernel, "scalars": scalars, "n": len(doc_df),
            "name": corpus_name}


def plot_matching_diagnostics(
    ecsc_full, persuade_full, ecsc_matched, persuade_matched,
    match_cols, out_dir,
):
    """Plot distributions before/after matching."""
    n_vars = len(match_cols)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(match_cols):
        # Pre-matching
        ax = axes[i, 0]
        bins = np.linspace(
            min(ecsc_full[col].min(), persuade_full[col].min()),
            max(ecsc_full[col].max(), persuade_full[col].max()),
            30,
        )
        ax.hist(ecsc_full[col], bins=bins, alpha=0.5, color="#3498db",
                label=f"ECSC (n={len(ecsc_full)})", density=True)
        ax.hist(persuade_full[col], bins=bins, alpha=0.5, color="#e67e22",
                label=f"PERSUADE (n={len(persuade_full)})", density=True)
        ax.set_title(f"{col} — Before Matching")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Post-matching
        ax = axes[i, 1]
        ax.hist(ecsc_matched[col], bins=bins, alpha=0.5, color="#3498db",
                label=f"ECSC (n={len(ecsc_matched)})", density=True)
        ax.hist(persuade_matched[col], bins=bins, alpha=0.5, color="#e67e22",
                label=f"PERSUADE (n={len(persuade_matched)})", density=True)
        ax.set_title(f"{col} — After Matching")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "matching_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_matched_kernel_overlay(ecsc_res, persuade_res, out_dir):
    """Kernel overlay for matched subsets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(N_BINS)

    # Bar chart
    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, ecsc_res["kernel"], width, color="#3498db",
           alpha=0.8, edgecolor="#2980b9",
           label=f"ECSC matched (n={ecsc_res['n']})")
    ax.bar(x + width / 2, persuade_res["kernel"], width, color="#e67e22",
           alpha=0.8, edgecolor="#d35400",
           label=f"PERSUADE matched (n={persuade_res['n']})")

    tail_x = [i for i in range(N_BINS) if K_GRID[i] >= 32]
    ax.axvspan(tail_x[0] - 0.5, N_BINS - 0.5, alpha=0.06, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of gain(128)")
    ax.set_title("Matched Subsets: Influence Kernel")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative
    ax2 = axes[1]
    cum_e = np.cumsum(ecsc_res["kernel"])
    cum_p = np.cumsum(persuade_res["kernel"])
    ax2.plot(BIN_RIGHTS, cum_e, "o-", color="#3498db",
             label=f"ECSC matched (n={ecsc_res['n']})", linewidth=2)
    ax2.plot(BIN_RIGHTS, cum_p, "s-", color="#e67e22",
             label=f"PERSUADE matched (n={persuade_res['n']})", linewidth=2)

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax2.annotate(lbl, xy=(1.5, q + 0.01), fontsize=9, color="gray")

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("Matched Subsets: Cumulative Kernel")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 150)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "matched_kernel_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    out_dir = Path("results/dif_cross_corpus_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load doc-level DIF data
    ecsc = pd.read_csv("results/dif_analysis_v1/ecsc/ecsc_doc_level_kernel.csv")
    persuade = pd.read_csv("results/dif_analysis_v1/persuade/persuade_doc_level_kernel.csv")

    print(f"ECSC: {len(ecsc)} docs, tokens [{ecsc['token_count'].min():.0f}, {ecsc['token_count'].max():.0f}], "
          f"median={ecsc['token_count'].median():.0f}")
    print(f"PERSUADE: {len(persuade)} docs, tokens [{persuade['token_count'].min():.0f}, {persuade['token_count'].max():.0f}], "
          f"median={persuade['token_count'].median():.0f}")

    print(f"\nECSC baseline_nll: [{ecsc['baseline_nll'].min():.3f}, {ecsc['baseline_nll'].max():.3f}], "
          f"mean={ecsc['baseline_nll'].mean():.3f}")
    print(f"PERSUADE baseline_nll: [{persuade['baseline_nll'].min():.3f}, {persuade['baseline_nll'].max():.3f}], "
          f"mean={persuade['baseline_nll'].mean():.3f}")

    # --- Matching ---
    # Token distributions barely overlap (ECSC median=393, PERSUADE median=915).
    # Strategy: match on baseline_nll only within the token-count overlap band.
    # Use the raw min/max overlap, not percentiles.

    tc_lo = persuade["token_count"].min()  # PERSUADE lower bound
    tc_hi = ecsc["token_count"].max()       # ECSC upper bound

    print(f"\nToken-count overlap range: [{tc_lo:.0f}, {tc_hi:.0f}]")

    ecsc_eligible = ecsc[ecsc["token_count"] >= tc_lo].copy()
    persuade_eligible = persuade[persuade["token_count"] <= tc_hi].copy()

    print(f"ECSC eligible (>= {tc_lo:.0f} tok): {len(ecsc_eligible)}")
    print(f"PERSUADE eligible (<= {tc_hi:.0f} tok): {len(persuade_eligible)}")

    # Match on both token_count and baseline_nll with generous caliper
    match_cols = ["token_count", "baseline_nll"]

    # ECSC eligible is smaller — match each ECSC doc to a PERSUADE doc
    if len(ecsc_eligible) <= len(persuade_eligible):
        target, pool = ecsc_eligible, persuade_eligible
        target_name, pool_name = "ecsc", "persuade"
    else:
        target, pool = persuade_eligible, ecsc_eligible
        target_name, pool_name = "persuade", "ecsc"

    m_target, m_pool, diag = nearest_neighbor_match(
        target, pool, match_cols, caliper=1.5,
    )

    print(f"\nMatched {len(m_target)} pairs")
    print(f"\nMatching diagnostics (SMD):")
    print(diag.to_string(index=False))

    # Assign back to corpus names
    if target_name == "ecsc":
        ecsc_matched, persuade_matched = m_target, m_pool
    else:
        persuade_matched, ecsc_matched = m_target, m_pool

    # Save matched subsets
    ecsc_matched.to_csv(out_dir / "matched_ecsc_docs.csv", index=False)
    persuade_matched.to_csv(out_dir / "matched_persuade_docs.csv", index=False)

    # --- Compute DIF on matched subsets ---
    ecsc_res = compute_subset_dif(ecsc_matched, "ecsc_matched")
    persuade_res = compute_subset_dif(persuade_matched, "persuade_matched")

    # Comparison table
    rows = []
    for res in [ecsc_res, persuade_res]:
        rec = {"corpus": res["name"], "n_docs": res["n"],
               "median_tokens": (ecsc_matched if "ecsc" in res["name"] else persuade_matched)["token_count"].median(),
               "mean_baseline_nll": (ecsc_matched if "ecsc" in res["name"] else persuade_matched)["baseline_nll"].mean()}
        for m, s in res["scalars"].items():
            rec[f"{m}_mean"] = s["mean"]
            rec[f"{m}_ci_lo"] = s["ci_lo"]
            rec[f"{m}_ci_hi"] = s["ci_hi"]
        rows.append(rec)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "matched_comparison_table.csv", index=False)

    print("\n=== Matched Cross-Corpus DIF Comparison ===")
    for _, r in comp_df.iterrows():
        print(f"\n{r['corpus']} (n={r['n_docs']:.0f}, median_tok={r['median_tokens']:.0f}, "
              f"mean_nll={r['mean_baseline_nll']:.3f}):")
        for m in ["L50", "L80", "L90", "tail_33_128", "tail_65_128", "mean_lag", "gain128"]:
            v = r[f"{m}_mean"]
            lo, hi = r[f"{m}_ci_lo"], r[f"{m}_ci_hi"]
            if "tail" in m:
                print(f"  {m:14s}: {v:.1%}  [{lo:.4f}, {hi:.4f}]")
            else:
                print(f"  {m:14s}: {v:.2f}  [{lo:.2f}, {hi:.2f}]")

    # --- Kernel distances (matched) ---
    ki = ecsc_res["kernel"]
    kj = persuade_res["kernel"]
    l2 = np.linalg.norm(ki - kj)
    ki_safe = np.maximum(ki, 0); ki_safe /= ki_safe.sum()
    kj_safe = np.maximum(kj, 0); kj_safe /= kj_safe.sum()
    js = jensenshannon(ki_safe, kj_safe)

    dist_rows = [
        {"pair": "ecsc_matched vs persuade_matched", "L2": l2, "JensenShannon": js},
    ]
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "matched_kernel_distances.csv", index=False)

    print(f"\n=== Matched Kernel Distances ===")
    print(f"  L2:              {l2:.4f}")
    print(f"  Jensen-Shannon:  {js:.4f}")

    # Load unmatched distances for comparison
    try:
        unmatched = pd.read_csv(out_dir / "kernel_distance_matrix.csv")
        um = unmatched[(unmatched["corpus_a"] == "ecsc") & (unmatched["corpus_b"] == "persuade")]
        print(f"\n  (Unmatched L2: {um['L2'].values[0]:.4f}, JS: {um['JensenShannon'].values[0]:.4f})")
        print(f"  Matching {'reduced' if l2 < um['L2'].values[0] else 'increased'} L2 distance "
              f"by {abs(l2 - um['L2'].values[0]) / um['L2'].values[0]:.0%}")
    except Exception:
        pass

    # --- Figures ---
    plot_matching_diagnostics(ecsc, persuade, ecsc_matched, persuade_matched,
                              match_cols, out_dir)
    plot_matched_kernel_overlay(ecsc_res, persuade_res, out_dir)

    print(f"\nDone. Outputs in {out_dir}/")
