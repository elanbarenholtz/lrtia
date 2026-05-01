#!/usr/bin/env python3
"""
refit_persuade_intercept.py

Refit per-essay power-law on the cached PERSUADE corrected marginals to recover
the intercept β (which the original notebook discarded), plus a few magnitude
measures we never saved.

Hypothesis: α (slope) is universal — confirmed null in our v1 run. β (intercept)
is the orthogonal axis: same shape, different vertical position. If individual
variation lives anywhere, this is the most likely place.

Outputs: persuade_per_essay_magnitude.csv with columns
  doc_id, score, grade, ell, token_count, n_curves,
  alpha, intercept, r, p,
  benefit_at_1, benefit_total, benefit_short, benefit_long,
  ppl_drop_intact
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results/PERSUADE_finegrain"
INTACT_PATH = RESULTS / "persuade_intact_v1.json"
SHUF_PATH = RESULTS / "persuade_shuffled_v1.json"
META_PATH = RESULTS / "persuade_per_essay_alpha.csv"
OUT_PATH = RESULTS / "persuade_per_essay_magnitude.csv"

MAX_CONTEXT = 64
COMMON_X = np.arange(1, MAX_CONTEXT + 1)
BIN_EDGES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, MAX_CONTEXT]


def compute_raw_ppl_curve(curves):
    arr = []
    for c in curves:
        interp = np.interp(
            COMMON_X, np.array(c["ctx_lengths"]), np.array(c["ppls"]),
            left=np.nan, right=np.nan,
        )
        arr.append(interp)
    return np.nanmean(np.array(arr), axis=0)


def fit_power_law(marg):
    bm, bc = [], []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        vals = marg[lo - 1:hi - 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((lo + hi) / 2)
    if len(bm) < 4:
        return None
    slope, intercept, r, p, _ = stats.linregress(np.log(bc), np.log(bm))
    return slope, intercept, r, p


def main():
    print(f"Loading {INTACT_PATH}")
    with open(INTACT_PATH) as f:
        all_intact = json.load(f)
    with open(SHUF_PATH) as f:
        all_shuf = json.load(f)
    print(f"  {len(all_intact)} intact, {len(all_shuf)} shuffled curves")

    by_doc_intact, by_doc_shuf = {}, {}
    for c in all_intact:
        by_doc_intact.setdefault(c["doc_id"], []).append(c)
    for c in all_shuf:
        by_doc_shuf.setdefault(c["doc_id"], []).append(c)

    rows = []
    for doc_id in sorted(by_doc_intact):
        ci = by_doc_intact[doc_id]
        cs = by_doc_shuf.get(doc_id, [])
        if len(ci) < 2 or len(cs) < 2:
            continue
        ip = compute_raw_ppl_curve(ci)
        sp = compute_raw_ppl_curve(cs)
        intact_marg = -np.diff(ip)
        shuf_marg = -np.diff(sp)
        corrected = intact_marg - shuf_marg

        fit = fit_power_law(corrected)
        if fit is None:
            continue
        alpha, intercept, r, p = fit

        meta = ci[0]
        # Magnitude family — all in the same units (per-token NLL drop)
        valid = ~np.isnan(corrected)
        cm = corrected[valid]
        # Predicted benefit at c=1 from the fit (closes form: exp(β))
        benefit_at_1 = float(np.exp(intercept))
        benefit_total = float(np.nansum(corrected))  # sum of marginals = ppl drop achieved
        # Short-range (c=1..3) vs long-range (c=10..MAX) summed contribution
        benefit_short = float(np.nansum(corrected[0:3]))
        benefit_long = float(np.nansum(corrected[9:]))
        # Raw intact ppl drop (no correction) — gross measure of how much context helped
        valid_ip = ~np.isnan(ip)
        if valid_ip.any():
            ppl_drop_intact = float(ip[valid_ip][0] - ip[valid_ip][-1])
        else:
            ppl_drop_intact = float("nan")

        rows.append({
            "doc_id": doc_id,
            "score": meta["score"],
            "grade": meta.get("grade"),
            "ell": meta.get("ell"),
            "token_count": meta["token_count"],
            "n_curves": len(ci),
            "alpha": alpha,
            "intercept": intercept,
            "r": r,
            "p": p,
            "benefit_at_1": benefit_at_1,
            "benefit_total": benefit_total,
            "benefit_short": benefit_short,
            "benefit_long": benefit_long,
            "ppl_drop_intact": ppl_drop_intact,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} per-essay rows to {OUT_PATH}")

    print(f"\n=== Per-essay summary ===")
    print(df[["alpha", "intercept", "benefit_at_1", "benefit_total",
             "benefit_short", "benefit_long", "ppl_drop_intact"]]
          .describe().round(3))

    print(f"\n=== Per-score means ===")
    print(df.groupby("score").agg(
        n=("alpha", "count"),
        alpha=("alpha", "mean"),
        intercept=("intercept", "mean"),
        b_at_1=("benefit_at_1", "mean"),
        b_total=("benefit_total", "mean"),
        b_short=("benefit_short", "mean"),
        b_long=("benefit_long", "mean"),
        ppl_drop=("ppl_drop_intact", "mean"),
    ).round(3))

    print(f"\n=== Spearman(score, X) and Spearman(log_tokens, X) ===")
    measures = ["alpha", "intercept", "benefit_at_1", "benefit_total",
                "benefit_short", "benefit_long", "ppl_drop_intact"]
    log_tc = np.log(df["token_count"])
    for m in measures:
        vals = df[m]
        r_s, p_s = stats.spearmanr(df["score"], vals)
        r_t, p_t = stats.spearmanr(log_tc, vals)
        print(f"  {m:<18}  vs score: r={r_s:+.3f} p={p_s:.3g}   "
              f"vs log_tc: r={r_t:+.3f} p={p_t:.3g}")

    # OLS for the most promising covariates: control for length
    try:
        import statsmodels.api as sm
        df["log_tc"] = log_tc
        print(f"\n=== Length-controlled OLS: X ~ score + log(token_count) ===")
        for m in measures:
            X = sm.add_constant(df[["score", "log_tc"]])
            res = sm.OLS(df[m], X).fit()
            b = res.params["score"]
            p = res.pvalues["score"]
            r2 = res.rsquared
            print(f"  {m:<18}  β(score)={b:+.4f}  p={p:.3g}  R²={r2:.3f}")
    except ImportError:
        print("\n(install statsmodels for length-controlled OLS)")


if __name__ == "__main__":
    main()
