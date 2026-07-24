#!/usr/bin/env python3
"""Exp 2 analysis: span-level order contribution Q(d), its exponent beta, and model comparison.

Reads the per-target caches written by ``experiments/exp2_increment_shuffle/run_exp2.py``
(``results/exp2_increment_shuffle/<probe>/<corpus>.json``) and, per corpus:

    Q(d_i) = ( mean_k ppl_B[i] - ppl_A[i] ) / (c_i - c_{i-1}),  d_i = sqrt(c_{i-1} c_i)

then fits Q(d) ~ d^{-beta} with document-bootstrap CIs and runs the checks that
decide H1 vs H2:

  * beta with 95% CI (full range) plus restricted-range fits (d>=8, d>=16) that
    drop the smallest token bands where a token is not a comparable linguistic
    unit -- the only place tokenization can bias the *slope*;
  * model comparison by AIC: power-law vs exponential vs curved (log-normal-ish)
    on log Q -- what licenses the word "power law";
  * beta in nats (robustness), the null-control beta (near+far shuffle), and the
    corpus's alpha from P(d) for comparison.

Note: a log-log slope is invariant to constant rescaling of either axis, so
token->char/word normalization shifts intercepts, not beta. Cross-lingual beta
comparison is therefore already tokenization-robust; no distance normalization
is applied.

Runs locally, no GPU. Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SMALL_N = 20  # flag corpora with fewer docs


# --- Q(d) aggregation -----------------------------------------------------------
def q_bins(records, ppl_key="B_ppl_mean", a_key="A_ppl"):
    """Per-interval (distance, mean Q, n_targets) aggregated across targets."""
    bi: dict[int, dict] = {}
    for r in records:
        for p in r["pairs"]:
            b = p.get(ppl_key)
            a = p.get(a_key)
            if b is None or a is None or (isinstance(b, float) and math.isnan(b)):
                continue
            if p["width"] < 2:
                # Degenerate single-token band (the (1,2) pair): increment-shuffle is a
                # no-op, so Q is 0 +/- fp16 GPU noise. Drop structurally, else a tiny
                # positive noise value becomes an extreme low-Q leverage point at small d
                # and flips the log-log slope. See PR discussion.
                continue
            bi.setdefault(p["i"], {"d": p["distance"], "qs": []})["qs"].append((b - a) / p["width"])
    return [(v["d"], sum(v["qs"]) / len(v["qs"]), len(v["qs"])) for i, v in sorted(bi.items())]


# --- fitting --------------------------------------------------------------------
def _ols(X, y):
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return c, float(((y - X @ c) ** 2).sum())


def fit_powerlaw(bins, dmin=0.0):
    """Log-log OLS on positive bins with d >= dmin. Returns beta, r2, n, arrays."""
    pts = [(d, q) for (d, q, _) in bins if q > 0 and d >= dmin]
    if len(pts) < 4:
        return None
    d = np.array([p[0] for p in pts])
    q = np.array([p[1] for p in pts])
    x = np.log(d)
    y = np.log(q)
    n = len(x)
    c_pl, rss = _ols(np.vstack([np.ones(n), x]).T, y)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return {"beta": -c_pl[1], "n": n, "r2": r2, "rss": rss, "x": x, "y": y, "d": d}


def model_compare(bins):
    """AIC over log Q: power-law (~log d) vs exponential (~d) vs curved (~log d + log d^2)."""
    f = fit_powerlaw(bins)
    if f is None:
        return None
    x, y, d, n = f["x"], f["y"], f["d"], f["n"]

    def aic(rss, k):
        return n * math.log(rss / n) + 2 * k if rss > 0 else -math.inf

    _, rss_pl = _ols(np.vstack([np.ones(n), x]).T, y)
    _, rss_ex = _ols(np.vstack([np.ones(n), d]).T, y)
    _, rss_cu = _ols(np.vstack([np.ones(n), x, x ** 2]).T, y)
    A = {"power": aic(rss_pl, 2), "exp": aic(rss_ex, 2), "curved": aic(rss_cu, 3)}
    # Parsimony: prefer the power law unless a richer model beats it by dAIC > 2
    # (differences below 2 are not meaningful).
    best = "power" if A["power"] - min(A.values()) <= 2 else min(A, key=A.get)
    return {
        "best": best,
        "dAIC_exp_minus_pl": A["exp"] - A["power"],
        "dAIC_curved_minus_pl": A["curved"] - A["power"],
    }


def bootstrap_beta(records, n_boot=1000, seed=12345, dmin=0.0):
    """Document-resampled bootstrap CI for beta. One target per doc => doc-level."""
    if len(records) < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    bs = []
    for _ in range(n_boot):
        samp = [records[j] for j in rng.choice(idx, size=len(records), replace=True)]
        f = fit_powerlaw(q_bins(samp), dmin=dmin)
        if f:
            bs.append(f["beta"])
    if len(bs) < n_boot * 0.5:
        return None
    a = np.array(bs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def alpha_from_main(cache: Path, dmin: float = 2.0):
    """Recompute the corpus's P(d) exponent alpha from the main long-range cache."""
    if not cache.exists():
        return None
    recs = json.load(open(cache))
    bi: dict[int, dict] = {}
    for r in recs:
        cs, op, sp = r["context_lengths"], r["ordered_ppl"], r["shuffled_ppl"]
        for i in range(1, len(cs)):
            if cs[i - 1] == 0:
                continue
            w = cs[i] - cs[i - 1]
            gr = ((op[i - 1] - op[i]) - (sp[i - 1] - sp[i])) / w
            bi.setdefault(i, {"d": math.sqrt(cs[i - 1] * cs[i]), "g": []})["g"].append(gr)
    bins = [(v["d"], sum(v["g"]) / len(v["g"]), 0) for v in bi.values()]
    f = fit_powerlaw(bins, dmin=dmin)   # dmin=2 drops the degenerate near interval to match Q's range
    return f["beta"] if f else None


def _f(x, p=3):
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return f"{x:.{p}f}"
    return "--"


def analyze(res_dir: Path, main_dir: Path | None, n_boot: int):
    rows = []
    for cache in sorted(res_dir.glob("*.json")):
        recs = json.load(open(cache))
        if not recs:
            continue
        bins = q_bins(recs)
        f = fit_powerlaw(bins)
        f8 = fit_powerlaw(bins, 8)
        f16 = fit_powerlaw(bins, 16)
        ci = bootstrap_beta(recs, n_boot) if n_boot else None
        mc = model_compare(bins)
        acache = main_dir / cache.name if main_dir else None
        a = alpha_from_main(acache, dmin=2.0) if acache else None      # matched to full-range beta
        a8 = alpha_from_main(acache, dmin=8.0) if acache else None
        a16 = alpha_from_main(acache, dmin=16.0) if acache else None
        has_ctrl = any("null_full_ppl_mean" in p for r in recs for p in r["pairs"])
        nb = fit_powerlaw(q_bins(recs, ppl_key="null_full_ppl_mean")) if has_ctrl else None
        fn = fit_powerlaw(q_bins(recs, ppl_key="B_nll_mean", a_key="A_nll"))
        rows.append(dict(
            corpus=cache.stem, n=len(recs), npos=(f["n"] if f else 0),
            beta=(f["beta"] if f else None), r2=(f["r2"] if f else None), ci=ci,
            best=(mc["best"] if mc else None),
            dexp=(mc["dAIC_exp_minus_pl"] if mc else None),
            b8=(f8["beta"] if f8 else None), b16=(f16["beta"] if f16 else None),
            bn=(fn["beta"] if fn else None), nbeta=(nb["beta"] if nb else None),
            alpha=a, a8=a8, a16=a16,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/exp2_increment_shuffle/llama")
    ap.add_argument("--main-results", default="results/corpus_expansion_longrange/llama")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    res_dir = Path(args.results)
    main_dir = Path(args.main_results)
    rows = analyze(res_dir, main_dir if main_dir.exists() else None, args.boot)
    if not rows:
        print(f"No caches in {res_dir}")
        return

    print("=== HEADLINE: power-law fit, full range ===")
    print(f"{'corpus':<24}{'n':>4}{'npos':>5}{'beta':>8}{'95% CI':>16}{'r2':>7}{'best':>8}{'alpha(P)':>10}")
    print("-" * 82)
    for r in rows:
        flag = " *" if r["n"] < SMALL_N else ""
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]" if r["ci"] else "--"
        print(f"{r['corpus'] + flag:<24}{r['n']:>4}{r['npos']:>5}{_f(r['beta']):>8}"
              f"{ci:>16}{_f(r['r2']):>7}{str(r['best']):>8}{_f(r['alpha']):>10}")

    print("\n=== BETA vs ALPHA, range-matched  (b/a on same distance floor; a = P(d) exponent) ===")
    print(f"{'corpus':<22}{'b(full)':>8}{'a(full)':>8}{'b(d>=8)':>9}{'a(d>=8)':>9}{'b(d>=16)':>10}{'a(d>=16)':>10}")
    print("-" * 76)
    for r in rows:
        print(f"{r['corpus']:<22}{_f(r['beta']):>8}{_f(r['alpha']):>8}{_f(r['b8']):>9}{_f(r['a8']):>9}"
              f"{_f(r['b16']):>10}{_f(r['a16']):>10}")

    print("\n=== ROBUSTNESS  (beta_nats, null-control beta, dAIC = exp - power) ===")
    print(f"{'corpus':<24}{'beta':>8}{'beta_nats':>10}{'null_beta':>10}{'dAIC':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['corpus']:<24}{_f(r['beta']):>8}{_f(r['bn']):>10}{_f(r['nbeta']):>10}{_f(r['dexp'], 1):>8}")
    print(f"\n* = n < {SMALL_N} docs (underpowered).  beta ~ alpha & best=power => H1.")

    if args.json_out:
        clean = [{k: v for k, v in r.items()} for r in rows]
        json.dump(clean, open(args.json_out, "w"), indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
