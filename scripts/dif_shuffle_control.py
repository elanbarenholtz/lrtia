"""
DIF v1 Shuffle Control — permutation-based null hypothesis tests.

Two controls:
  (A) Increment permutation: within each document, permute the per-bin
      gain increments across bins, then re-integrate to monotone gain curves.
      This destroys the kernel shape while preserving total gain.

  (C) Quartile label shuffle (PERSUADE only): within each essay, shuffle
      window position quartile labels (destroying position structure while
      keeping the same windows).

Produces:
  results/dif_shuffle_control_v1/{corpus}_shuffle_results.csv
  results/dif_shuffle_control_v1/{corpus}_shuffle_kernel_comparison.png
  results/dif_shuffle_control_v1/persuade_quartile_shuffle.csv
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dif_pipeline import (
    K_GRID, N_BINS, BIN_LABELS, BIN_RIGHTS,
    compute_dif_scalars, boot_ci,
)


# =====================================================================
# (A) Increment permutation
# =====================================================================

def permute_increments(doc_shares: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute bin shares (increments/gain_128) across bins for one doc.

    Since kernel shares ARE the normalized increments, shuffling shares
    across bins is equivalent to permuting increments and re-integrating.
    The result is still a valid probability distribution (sums to 1,
    all non-negative after clamping).
    """
    perm = rng.permutation(doc_shares.copy())
    return perm


def run_increment_shuffle(
    corpus_name: str,
    doc_kernel_csv: str | Path,
    n_perms: int = 1000,
    seed: int = 42,
    out_dir: str | Path = "results/dif_shuffle_control_v1",
) -> pd.DataFrame:
    """Run increment-permutation null for a corpus.

    For each of n_perms iterations:
      1. For every doc, permute its 11 bin shares across bins.
      2. Compute mean kernel across docs.
      3. Compute DIF scalars from the mean kernel.

    Compare real DIF to the null distribution.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    doc_df = pd.read_csv(doc_kernel_csv)
    share_cols = [f"p_{i}" for i in range(N_BINS)]
    real_shares = doc_df[share_cols].values  # (n_docs, N_BINS)
    n_docs = len(doc_df)

    print(f"\n=== Increment Shuffle Control: {corpus_name.upper()} ===")
    print(f"  Docs: {n_docs}, Permutations: {n_perms}")

    # Real mean kernel + DIF
    real_kernel = real_shares.mean(axis=0)
    real_scalars = compute_dif_scalars(real_kernel)

    # Permutation null
    perm_kernels = np.zeros((n_perms, N_BINS))
    perm_scalars = {m: [] for m in real_scalars}

    for p in range(n_perms):
        perm_shares = np.zeros_like(real_shares)
        for d in range(n_docs):
            perm_shares[d] = permute_increments(real_shares[d], rng)
        perm_kernel = perm_shares.mean(axis=0)
        perm_kernels[p] = perm_kernel
        sc = compute_dif_scalars(perm_kernel)
        for m in sc:
            perm_scalars[m].append(sc[m])

    # Compute p-values (fraction of permutations more extreme than real)
    results = []
    print("\n  Results (real vs permuted null):")
    for m in real_scalars:
        real_val = real_scalars[m]
        perm_vals = np.array(perm_scalars[m])
        perm_mean = perm_vals.mean()
        perm_std = perm_vals.std()

        # Two-sided p-value
        p_val = np.mean(np.abs(perm_vals - perm_mean) >= np.abs(real_val - perm_mean))
        z_score = (real_val - perm_mean) / (perm_std + 1e-15)

        results.append({
            "metric": m, "real": real_val,
            "perm_mean": perm_mean, "perm_std": perm_std,
            "perm_2.5": np.percentile(perm_vals, 2.5),
            "perm_97.5": np.percentile(perm_vals, 97.5),
            "z_score": z_score, "p_value": p_val,
        })

        fmt_real = f"{real_val:.1%}" if "tail" in m else f"{real_val:.2f}"
        fmt_perm = f"{perm_mean:.1%}" if "tail" in m else f"{perm_mean:.2f}"
        print(f"    {m:14s}: real={fmt_real}  perm={fmt_perm}±{perm_std:.3f}  z={z_score:+.1f}  p={p_val:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / f"{corpus_name}_shuffle_results.csv", index=False)

    # Plot: real kernel vs permuted null
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    x = np.arange(N_BINS)
    perm_mean_kernel = perm_kernels.mean(axis=0)
    perm_std_kernel = perm_kernels.std(axis=0)

    ax.bar(x - 0.18, real_kernel, 0.35, label="Real", color="#3498db",
           alpha=0.8, edgecolor="#2980b9")
    ax.bar(x + 0.18, perm_mean_kernel, 0.35, yerr=perm_std_kernel * 2,
           capsize=2, label="Permuted (±2σ)", color="#95a5a6",
           alpha=0.7, edgecolor="#7f8c8d")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of gain(128)")
    ax.set_title(f"{corpus_name.upper()}: Real vs Increment-Permuted Kernel")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative
    ax2 = axes[1]
    cum_real = np.cumsum(real_kernel)
    cum_perm = np.cumsum(perm_mean_kernel)
    ax2.plot(BIN_RIGHTS, cum_real, "o-", color="#3498db",
             label="Real", linewidth=2)
    ax2.plot(BIN_RIGHTS, cum_perm, "s--", color="#95a5a6",
             label="Permuted mean", linewidth=2)

    # Permutation envelope
    cum_perms = np.cumsum(perm_kernels, axis=1)
    ax2.fill_between(BIN_RIGHTS,
                      np.percentile(cum_perms, 2.5, axis=0),
                      np.percentile(cum_perms, 97.5, axis=0),
                      color="#95a5a6", alpha=0.2, label="Permuted 95% CI")

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax2.annotate(lbl, xy=(1.5, q + 0.01), fontsize=9, color="gray")

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title(f"{corpus_name.upper()}: Cumulative Real vs Permuted")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 150)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{corpus_name}_shuffle_kernel_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_dir / f'{corpus_name}_shuffle_kernel_comparison.png'}")

    return results_df


# =====================================================================
# (C) Quartile label shuffle (PERSUADE only)
# =====================================================================

def run_quartile_shuffle(
    window_csv: str | Path,
    n_perms: int = 1000,
    seed: int = 42,
    out_dir: str | Path = "results/dif_shuffle_control_v1",
) -> pd.DataFrame:
    """Shuffle window quartile labels within each essay.

    For each permutation:
      1. Within each essay, randomly reassign quartile labels to windows.
      2. Aggregate Q1+Q2 windows (now random subset) to get early+mid gain curve.
      3. Compute DIF scalars on the shuffled early+mid data.

    Compare real early+mid DIF to the null where position doesn't matter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("\n=== Quartile Label Shuffle Control: PERSUADE ===")
    win = pd.read_csv(window_csv)

    # Pivot to wide
    win_wide = win.pivot_table(
        index=["essay_id", "window_idx", "target_start"],
        columns="k", values="gain_k"
    ).reset_index()
    win_wide.columns = [
        f"gain_{int(c)}" if isinstance(c, (int, float, np.integer, np.floating))
        else c
        for c in win_wide.columns
    ]

    gain_cols = [f"gain_{k}" for k in K_GRID]

    # Assign real quartiles
    def assign_quartile(grp):
        n = len(grp)
        grp = grp.sort_values("target_start")
        grp["pos_quartile"] = pd.qcut(range(n), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        return grp

    win_wide = win_wide.groupby("essay_id", group_keys=False).apply(
        assign_quartile, include_groups=True
    )

    essay_ids = win_wide["essay_id"].unique()
    n_essays = len(essay_ids)
    print(f"  Essays: {n_essays}, Total windows: {len(win_wide)}")

    # Real early+mid DIF
    def compute_em_dif(df_windows):
        """Compute mean DIF scalars from early+mid windows."""
        em = df_windows[df_windows["pos_quartile"].isin(["Q1", "Q2"])]
        em_agg = em.groupby("essay_id")[gain_cols].mean()
        em_agg.columns = [f"mean_{c}" for c in gain_cols]

        doc_scalars = []
        for eid, row in em_agg.iterrows():
            gains = {k: row[f"mean_gain_{k}"] for k in K_GRID}
            if gains[128] <= 0:
                continue
            g128 = gains[128]
            shares = np.array([
                max(0, (gains[K_GRID[i + 1]] - gains[K_GRID[i]]) / g128)
                for i in range(N_BINS)
            ])
            s = shares.sum()
            if s > 0:
                shares /= s
            doc_scalars.append(compute_dif_scalars(shares))

        if not doc_scalars:
            return {}
        means = {}
        for m in doc_scalars[0]:
            means[m] = np.mean([d[m] for d in doc_scalars])
        return means

    real_dif = compute_em_dif(win_wide)
    print(f"\n  Real early+mid DIF:")
    for m, v in real_dif.items():
        fmt = f"{v:.1%}" if "tail" in m else f"{v:.2f}"
        print(f"    {m:14s}: {fmt}")

    # Permutation null
    perm_results = {m: [] for m in real_dif}
    for p in range(n_perms):
        # Shuffle quartile labels within each essay
        shuffled = win_wide.copy()
        for eid in essay_ids:
            mask = shuffled["essay_id"] == eid
            labels = np.array(shuffled.loc[mask, "pos_quartile"].values)
            rng.shuffle(labels)
            shuffled.loc[mask, "pos_quartile"] = labels

        perm_dif = compute_em_dif(shuffled)
        for m in perm_results:
            perm_results[m].append(perm_dif.get(m, np.nan))

    # Results
    results = []
    print(f"\n  Results (real early+mid vs quartile-shuffled null, {n_perms} perms):")
    for m in real_dif:
        real_val = real_dif[m]
        perm_vals = np.array(perm_results[m])
        perm_mean = np.nanmean(perm_vals)
        perm_std = np.nanstd(perm_vals)
        p_val = np.nanmean(np.abs(perm_vals - perm_mean) >= np.abs(real_val - perm_mean))
        z_score = (real_val - perm_mean) / (perm_std + 1e-15)

        results.append({
            "metric": m, "real_em": real_val,
            "perm_mean": perm_mean, "perm_std": perm_std,
            "z_score": z_score, "p_value": p_val,
        })

        fmt_real = f"{real_val:.1%}" if "tail" in m else f"{real_val:.2f}"
        fmt_perm = f"{perm_mean:.1%}" if "tail" in m else f"{perm_mean:.2f}"
        print(f"    {m:14s}: real_em={fmt_real}  perm={fmt_perm}±{perm_std:.4f}  z={z_score:+.2f}  p={p_val:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "persuade_quartile_shuffle.csv", index=False)
    print(f"\n  Saved: {out_dir / 'persuade_quartile_shuffle.csv'}")

    return results_df


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    out = Path("results/dif_shuffle_control_v1")

    # (A) Increment permutation — both corpora
    for corpus in ["ecsc", "persuade"]:
        run_increment_shuffle(
            corpus_name=corpus,
            doc_kernel_csv=f"results/dif_analysis_v1/{corpus}/{corpus}_doc_level_kernel.csv",
            n_perms=1000,
            out_dir=out,
        )

    # (C) Quartile label shuffle — PERSUADE only
    run_quartile_shuffle(
        window_csv="results/local_lag_curve_sliding_v1/window_level_results_local_lag.csv",
        n_perms=1000,
        out_dir=out,
    )

    print("\nDone. Shuffle control outputs in results/dif_shuffle_control_v1/")
