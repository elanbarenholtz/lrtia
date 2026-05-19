"""Compute old-style essay-level metrics (log_slope, half_life, delta_max, auc_full,
auc_early, auc_late, early_fraction) alongside the new alpha on corpus_expansion
cache files, so the new run is directly comparable to RAID / HumanVsAI / ECSC.

Each cache file is BASE/<corpus_id>.json: a list of per-target dicts with keys
'ordered_ppl' (length mc+1), 'shuffled_ppl', 'delta_ppl' (length mc, the
ordered-minus-shuffled marginal), plus metadata. We treat the dense ordered_ppl
curve as the equivalent of the older essay pipeline's ppl_by_W with W = 1..mc.

Usage:
    python analysis/reconcile_metrics_corpus_expansion.py \
        --cache-dir /content/drive/MyDrive/LRTIA/Results/corpus_expansion/cache \
        --out-dir   /content/drive/MyDrive/LRTIA/Results/corpus_expansion
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Match the older essay pipelines.
W_BASELINE = 4
EARLY_LATE_SPLIT = 32

# Same bins as fit_pl in Corpus_Expansion_Llama.ipynb so refit alpha is identical.
BIN_EDGES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def fit_alpha(delta_marginal):
    """Log-log fit of marginal Δppl vs distance, binned. Mirrors fit_pl()."""
    arr = np.asarray(delta_marginal, dtype=float)
    bm, bc = [], []
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:]):
        if lo - 1 >= len(arr):
            break
        vals = arr[lo - 1:min(hi - 1, len(arr))]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0 and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((lo + hi) / 2)
    if len(bm) >= 4:
        slope, _, r, _, _ = stats.linregress(np.log(bc), np.log(bm))
        return slope, r
    return np.nan, np.nan


def half_life(windows, ppls):
    """Window at which 50% of the total ppl benefit (relative to W_BASELINE) is hit.

    Linear interpolation. Returns nan if total benefit is non-positive."""
    base_idx = next((i for i, w in enumerate(windows) if w >= W_BASELINE), 0)
    base_ppl = ppls[base_idx]
    total = base_ppl - ppls[-1]
    if total <= 0:
        return np.nan
    target = base_ppl - 0.5 * total
    for i in range(base_idx, len(ppls) - 1):
        if ppls[i] >= target >= ppls[i + 1]:
            num = ppls[i] - target
            den = ppls[i] - ppls[i + 1]
            frac = num / den if den > 0 else 0.0
            return windows[i] + frac * (windows[i + 1] - windows[i])
    return float(windows[-1])


def old_style_metrics(ordered_ppl):
    """Replicate the HumanVsAI / RAID essay-level metric pack on a dense ppl curve.

    ordered_ppl[i] is perplexity given i tokens of preceding context (i = 0..mc).
    """
    ppls = np.asarray(ordered_ppl, dtype=float)
    mc = len(ppls) - 1
    windows = np.arange(0, mc + 1)

    # Anchor at W_BASELINE to match the old pipeline (smallest window).
    keep = windows >= W_BASELINE
    W = windows[keep]
    P = ppls[keep]
    if len(W) < 3 or not np.all(np.isfinite(P)):
        return {}

    delta_ppl = P[0] - P  # cumulative improvement from W_BASELINE
    out = {
        "delta_max": float(delta_ppl[-1]),
        "auc_full": float(np.trapz(delta_ppl, W)),
        "half_life": float(half_life(W.tolist(), P.tolist())),
    }

    slope, _, _, _, _ = stats.linregress(np.log(W), delta_ppl)
    out["log_slope"] = float(slope)

    early_mask = W <= EARLY_LATE_SPLIT
    late_mask = W >= EARLY_LATE_SPLIT
    if early_mask.sum() >= 2:
        out["auc_early"] = float(np.trapz(P[0] - P[early_mask], W[early_mask]))
    if late_mask.sum() >= 2:
        L = P[late_mask]
        out["auc_late"] = float(np.trapz(L[0] - L, W[late_mask]))
    if out.get("auc_full", 0) > 0 and "auc_early" in out:
        out["early_fraction"] = out["auc_early"] / out["auc_full"]

    # Also a "from zero" version, since corpus_expansion has the no-context point.
    delta0 = ppls[0] - ppls
    out["delta_from_zero_max"] = float(delta0[-1])
    out["auc_full_from_zero"] = float(np.trapz(delta0, windows))

    return out


def per_target_row(rec):
    row = {
        "corpus_id": rec.get("corpus_id"),
        "document_id": rec.get("document_id"),
        "target_id": rec.get("target_id"),
        "target_frac": rec.get("target_frac"),
        "language": rec.get("language"),
        "genre": rec.get("genre"),
        "modality": rec.get("modality"),
        "context_len": len(rec.get("ordered_ppl", [])) - 1,
    }
    row.update(old_style_metrics(rec["ordered_ppl"]))

    # Per-target alpha on the corrected marginal — same metric as the run's print.
    alpha, r = fit_alpha(rec["delta_ppl"])
    row["alpha_corrected"] = alpha
    row["alpha_corrected_r"] = r

    # And alpha on raw ordered marginal, for an apples-to-apples shape diagnostic
    # that doesn't depend on the shuffled baseline.
    ord_ppl = np.asarray(rec["ordered_ppl"], dtype=float)
    ord_marg = ord_ppl[:-1] - ord_ppl[1:]  # length mc, marginal at distance 1..mc
    alpha_o, r_o = fit_alpha(ord_marg)
    row["alpha_ordered"] = alpha_o
    row["alpha_ordered_r"] = r_o

    # Shuffled curve as the chain-vs-decay diagnostic: under the chained-influence
    # story the shuffled marginal should be ~flat (alpha_shuffled near 0), since
    # shuffling preserves total information but destroys the chain. If it has a
    # similar heavy-tailed shape to the ordered curve, the heavy tail isn't
    # order-specific and the memory-decay alternative is alive.
    shuf_ppl = np.asarray(rec["shuffled_ppl"], dtype=float)
    shuf_marg = shuf_ppl[:-1] - shuf_ppl[1:]
    alpha_s, r_s = fit_alpha(shuf_marg)
    row["alpha_shuffled"] = alpha_s
    row["alpha_shuffled_r"] = r_s
    # Shuffled-only old-style metrics so log_slope and half_life are also comparable.
    shuf_old = old_style_metrics(shuf_ppl)
    for k, v in shuf_old.items():
        row[f"{k}_shuffled"] = v

    # Direct chain-strength diagnostics:
    # - alpha_gap: how much steeper ordered is than shuffled. Larger negative gap
    #   => shuffled is much flatter than ordered => strong chain signal.
    # - mean_delta / mean_ordered_marginal: fraction of marginal benefit that is
    #   order-specific. Near 1 => almost all benefit comes from the chain;
    #   near 0 => most benefit is order-agnostic (memory-decay-flavored).
    if np.isfinite(alpha_o) and np.isfinite(alpha_s):
        row["alpha_gap_ord_minus_shuf"] = float(alpha_o - alpha_s)
    mean_ord_marg = float(np.nanmean(ord_marg))
    row["mean_ordered_marginal"] = mean_ord_marg
    row["mean_shuffled_marginal"] = float(np.nanmean(shuf_marg))
    if mean_ord_marg > 0:
        row["chain_share"] = float(np.nanmean(rec["delta_ppl"])) / mean_ord_marg

    row["mean_delta_corrected"] = float(np.nanmean(rec["delta_ppl"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted(args.cache_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No cache JSONs in {args.cache_dir}")

    for fp in files:
        with open(fp) as f:
            recs = json.load(f)
        for rec in recs:
            if "ordered_ppl" not in rec or "delta_ppl" not in rec:
                continue
            rows.append(per_target_row(rec))
        print(f"  {fp.name}: {len(recs)} targets")

    df = pd.DataFrame(rows)
    target_path = args.out_dir / "per_target_metrics.csv"
    df.to_csv(target_path, index=False)
    print(f"\nWrote {target_path}  ({len(df)} rows)")

    # Per-corpus aggregate: mean + median + n + sd for the metrics that travel.
    metric_cols = [
        "half_life", "log_slope", "delta_max", "auc_full",
        "auc_early", "auc_late", "early_fraction",
        "alpha_corrected", "alpha_corrected_r",
        "alpha_ordered", "alpha_ordered_r",
        "alpha_shuffled", "alpha_shuffled_r",
        "half_life_shuffled", "log_slope_shuffled",
        "auc_full_shuffled", "delta_max_shuffled",
        "alpha_gap_ord_minus_shuf",
        "mean_ordered_marginal", "mean_shuffled_marginal",
        "chain_share",
        "mean_delta_corrected",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    agg = (
        df.groupby(["corpus_id", "language", "genre", "modality"])[metric_cols]
        .agg(["mean", "median", "std", "count"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    corpus_path = args.out_dir / "per_corpus_metrics.csv"
    agg.to_csv(corpus_path, index=False)
    print(f"Wrote {corpus_path}  ({len(agg)} corpora)")

    print("\nQuick read (chain-vs-decay diagnostic columns first):")
    show = [
        "corpus_id",
        "alpha_ordered_mean", "alpha_shuffled_mean", "alpha_gap_ord_minus_shuf_mean",
        "chain_share_mean",
        "alpha_corrected_mean", "log_slope_mean",
        "half_life_mean", "auc_full_mean",
    ]
    show = [c for c in show if c in agg.columns]
    print(agg[show].to_string(index=False))


if __name__ == "__main__":
    main()
