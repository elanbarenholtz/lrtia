#!/usr/bin/env python3
"""Exp 2 (increment-shuffle) SI figure + derived summary — local, no GPU.

Given the per-target Q(d) caches from ``run_exp2.py`` and the main P(d) caches,
this recomputes, per corpus:

    Q(d_i) = ( mean_k ppl_B[i] - ppl_A[i] ) / (c_i - c_{i-1}),   d_i = sqrt(c_{i-1} c_i)   [span-internal order]
    P(d)   = m_intact(d) - m_shuffled(d)                                                    [marginal / integrated]

fits both by log--log OLS over the *matched* distance range (d >= DMIN), and writes:

  * figures/exp2_qd_loglog.pdf   — title-free SI figure (Q(d) vs range-matched P(d))
  * results/exp2_increment_shuffle/qd_summary.csv — per-corpus beta, alpha, r^2, N_bands

It also prints N = the number of corpora that have BOTH a Q(d) cache and a
range-matched P(d) cache — the integer for the Results paragraph.

NOTE on the exponent: the reported per-corpus beta depends on the fit range.
This script uses a single log--log OLS over d >= DMIN (default 10), matching how
alpha is fit here (the alpha column reproduces Table S1). The manuscript headline
"beta ~ -1.3" comes from the restricted-range / curved estimator in
analysis/exp2_increment_shuffle_qd.py; set --dmin to match that range if you want
the figure's per-corpus fits to coincide with the headline value.

Inputs (edit paths / env if needed):
  QDIR  = results/exp2_increment_shuffle/llama         # drop the Drive 'llama/' folder here
  PDIR  = NHB_submission/data/derived/corpus_expansion_longrange/llama
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
QDIR = Path(os.environ.get("EXP2_QDIR", ROOT / "results/exp2_increment_shuffle/llama"))
PDIR = Path(os.environ.get("CPF_PDIR", ROOT / "NHB_submission/data/derived/corpus_expansion_longrange/llama"))
FIG = ROOT / "paper/arxiv/figures/exp2_qd_loglog.pdf"
OUTCSV = ROOT / "results/exp2_increment_shuffle/qd_summary.csv"


def ols_exp(d, y, dmin):
    m = (np.asarray(y) > 0) & (np.asarray(d) >= dmin)
    if m.sum() < 4:
        return None, None, int(m.sum())
    x = np.log(np.asarray(d)[m]); yy = np.log(np.asarray(y)[m])
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    rss = float(((yy - A @ c) ** 2).sum()); tss = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return -c[1], r2, int(m.sum())          # exponent = -slope


def q_curve(path):
    recs = json.load(open(path)); bi = {}
    for r in recs:
        for p in r.get("pairs", []):
            if p.get("width", 0) < 2:
                continue          # drop degenerate (1,2) band; matches exp2_increment_shuffle_qd.py
            b, a = p.get("B_ppl_mean"), p.get("A_ppl")
            if b is None or a is None or (isinstance(b, float) and math.isnan(b)):
                continue
            bi.setdefault(p["i"], {"d": p["distance"], "q": []})["q"].append((b - a) / p["width"])
    items = sorted(bi.items())
    return (np.array([v["d"] for _, v in items]),
            np.array([sum(v["q"]) / len(v["q"]) for _, v in items]))


def p_curve(path):
    recs = json.load(open(path)); cs = recs[0]["context_lengths"]; acc = {}
    for r in recs:
        op, sp = r["ordered_ppl"], r["shuffled_ppl"]
        for i in range(1, len(cs)):
            if cs[i - 1] == 0:
                continue
            w = cs[i] - cs[i - 1]; d = math.sqrt(cs[i - 1] * cs[i])
            acc.setdefault(i, {"d": d, "o": [], "s": []})
            acc[i]["o"].append((op[i - 1] - op[i]) / w)
            acc[i]["s"].append((sp[i - 1] - sp[i]) / w)
    items = sorted(acc.items())
    return (np.array([v["d"] for _, v in items]),
            np.array([sum(v["o"]) / len(v["o"]) - sum(v["s"]) / len(v["s"]) for _, v in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmin", type=float, default=10.0, help="min distance for the log-log fit")
    args = ap.parse_args()
    dmin = args.dmin
    if not QDIR.exists():
        raise SystemExit(f"Q(d) caches not found at {QDIR}\n"
                         f"  Drop the Drive folder Results/exp2_increment_shuffle/llama/ there "
                         f"(or set $EXP2_QDIR).")
    rows, panels = [], []
    for qp in sorted(QDIR.glob("*.json")):
        corpus = qp.stem; pp = PDIR / f"{corpus}.json"
        dQ, Q = q_curve(qp); beta, r2q, nq = ols_exp(dQ, Q, dmin)
        alpha = r2p = dP = P = None
        if pp.exists():
            dP, P = p_curve(pp)
            lo = dQ[(Q > 0)].min() if (Q > 0).any() else dmin
            mask = (dP >= max(lo, dmin)) & (dP <= dQ.max())
            alpha, r2p, _ = ols_exp(dP[mask], P[mask], dmin)
        rows.append({"corpus": corpus, "beta": beta, "r2_Q": r2q,
                     "alpha_matched": alpha, "r2_P": r2p, "matched": pp.exists(), "n_bands": nq})
        if pp.exists():
            panels.append((corpus, dQ, Q, dP, P, beta, alpha))

    N = sum(1 for r in rows if r["matched"] and r["beta"] is not None and r["alpha_matched"] is not None)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(panels); ncol = 4; nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for ax, (corpus, dQ, Q, dP, P, beta, alpha) in zip(axes.flat, panels):
        ax.plot(dP[P > 0], P[P > 0], "o-", ms=4, color="C0", label=fr"$P(d)$  $\alpha$={alpha:.2f}")
        ax.plot(dQ[Q > 0], Q[Q > 0], "s--", ms=4, color="#7B3FA0", label=fr"$Q(d)$  $\beta$={beta:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(corpus, fontsize=9); ax.legend(fontsize=7, framealpha=0.9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("per-token order-specific influence")
    fig.supxlabel(fr"distance $d$ (tokens); fits over $d\geq{dmin:.0f}$")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight"); fig.savefig(FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")

    print(f"N (corpora with both Q and range-matched P) = {N}   [fit d>={dmin:.0f}]")
    print(f"{'corpus':28s} {'beta':>6s} {'alpha':>6s} {'beta>alpha':>10s}")
    for r in rows:
        if r["beta"] is None:
            continue
        a = r["alpha_matched"]
        flag = "yes" if (a is not None and r["beta"] > a) else ("--" if a is None else "NO")
        print(f"{r['corpus']:28s} {r['beta']:6.2f} {('' if a is None else f'{a:6.2f}'):>6s} {flag:>10s}")
    print(f"\nwrote {FIG}\nwrote {OUTCSV}")


if __name__ == "__main__":
    main()
