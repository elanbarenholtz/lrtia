#!/usr/bin/env python3
"""Sensitivity sweeps for power law exponent.

Tests whether the exponent is robust to:
1. Distance range (1-50, 5-100, 10-100, 1-100)
2. Binning (unbinned log-log, linear bins, log bins)
3. Target position (25%, 50%, 75% separately)
4. PPL vs NLL marginals (where available)

Runs locally on cached results. No GPU needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# === Bin definitions ===
LOG_BINS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]
LINEAR_BINS = list(range(1, 101, 10))  # [1, 11, 21, ..., 91, 100]
COMMON_X = np.arange(1, 101)


def compute_corrected_marginal_from_paired(intact_path, shuffled_path):
    """Load paired files, return per-target corrected marginals with metadata."""
    with open(intact_path) as f:
        intact = json.load(f)
    with open(shuffled_path) as f:
        shuffled = json.load(f)

    imap = {(c.get('doc_id', ''), c.get('target_frac', 0)): c for c in intact}
    smap = {(c.get('doc_id', ''), c.get('target_frac', 0)): c for c in shuffled}

    results = []
    for k, ic in imap.items():
        sc = smap.get(k)
        if sc is None:
            continue
        ip = np.array(ic['ppls'])
        sp = np.array(sc['ppls'])
        im = np.diff(-ip)
        sm = np.diff(-sp)
        ml = min(len(im), len(sm))
        results.append({
            'doc_id': ic.get('doc_id', ''),
            'target_frac': ic.get('target_frac', 0),
            'delta': (im[:ml] - sm[:ml]).tolist(),
        })
    return results


def fit_power_law_binned(marg, bin_edges, dist_range=(1, 100)):
    """Fit power law on binned marginals within distance range."""
    lo_d, hi_d = dist_range
    bm, bc = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo_d or lo >= hi_d:
            continue
        # Clamp to range
        elo = max(lo, lo_d)
        ehi = min(hi, hi_d)
        if elo >= ehi:
            continue
        vals = marg[elo - 1:ehi - 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0 and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((elo + ehi) / 2)
    if len(bm) >= 3:
        slope, intercept, r, p, _ = stats.linregress(np.log(bc), np.log(bm))
        return slope, r, p, len(bm)
    return None, None, None, 0


def fit_power_law_unbinned(marg, dist_range=(1, 100)):
    """Fit power law on unbinned positive marginals within range."""
    lo_d, hi_d = dist_range
    dists = []
    vals = []
    for d in range(lo_d, min(hi_d + 1, len(marg) + 1)):
        v = marg[d - 1]
        if not np.isnan(v) and v > 0:
            dists.append(d)
            vals.append(v)
    if len(dists) >= 5:
        slope, intercept, r, p, _ = stats.linregress(np.log(dists), np.log(vals))
        return slope, r, p, len(dists)
    return None, None, None, 0


def run_sweeps(results, label):
    """Run all sensitivity sweeps on a set of per-target results."""
    all_delta = np.array([r['delta'] for r in results])
    mean_curve = np.mean(all_delta, axis=0)

    # Split by target position
    by_frac = {}
    for r in results:
        frac = r.get('target_frac', 0)
        by_frac.setdefault(frac, []).append(r)

    print(f'\n{"="*80}')
    print(f'{label} (n={len(results)} targets)')
    print(f'{"="*80}')

    # === 1. Distance range ===
    print(f'\n  Distance range sweep (log bins):')
    print(f'  {"Range":<15} {"α":>8} {"r":>8} {"p":>10} {"n_bins":>8}')
    print(f'  {"-"*52}')
    for lo, hi in [(1, 100), (1, 50), (5, 100), (10, 100), (1, 30), (30, 100)]:
        a, r, p, nb = fit_power_law_binned(mean_curve, LOG_BINS, (lo, hi))
        if a is not None:
            print(f'  {f"d={lo}–{hi}":<15} {a:>8.3f} {r:>8.3f} {p:>10.4f} {nb:>8}')
        else:
            print(f'  {f"d={lo}–{hi}":<15} {"—":>8}')

    # === 2. Binning method ===
    print(f'\n  Binning method sweep (d=1–100):')
    print(f'  {"Method":<15} {"α":>8} {"r":>8} {"n_pts":>8}')
    print(f'  {"-"*42}')

    # Log bins (standard)
    a, r, p, nb = fit_power_law_binned(mean_curve, LOG_BINS)
    if a: print(f'  {"Log bins":<15} {a:>8.3f} {r:>8.3f} {nb:>8}')

    # Linear bins
    a, r, p, nb = fit_power_law_binned(mean_curve, LINEAR_BINS)
    if a: print(f'  {"Linear bins":<15} {a:>8.3f} {r:>8.3f} {nb:>8}')

    # Unbinned
    a, r, p, nb = fit_power_law_unbinned(mean_curve)
    if a: print(f'  {"Unbinned":<15} {a:>8.3f} {r:>8.3f} {nb:>8}')

    # === 3. Target position ===
    print(f'\n  Target position sweep (log bins, d=1–100):')
    print(f'  {"Position":<15} {"α":>8} {"r":>8} {"n_targets":>10}')
    print(f'  {"-"*45}')
    for frac in sorted(by_frac.keys()):
        fc = by_frac[frac]
        if len(fc) < 5:
            continue
        fc_mean = np.mean([r['delta'] for r in fc], axis=0)
        a, r, p, nb = fit_power_law_binned(fc_mean, LOG_BINS)
        if a:
            print(f'  {f"{frac:.0%}":<15} {a:>8.3f} {r:>8.3f} {len(fc):>10}')

    # === Summary ===
    # Baseline exponent for reference
    a_base, r_base, _, _ = fit_power_law_binned(mean_curve, LOG_BINS)
    print(f'\n  Baseline: α = {a_base:.3f} (r = {r_base:.3f})')

    return a_base


def main():
    results_base = Path('results')
    if not results_base.exists():
        print('Results directory not found')
        sys.exit(1)

    # Datasets to sweep
    datasets = [
        ('Llama Chinese Wiki',
         results_base / 'Llama_crosslingual/wiki_zh_intact.json',
         results_base / 'Llama_crosslingual/wiki_zh_shuffled.json'),
        ('Llama Japanese Wiki',
         results_base / 'Llama_crosslingual/wiki_ja_intact.json',
         results_base / 'Llama_crosslingual/wiki_ja_shuffled.json'),
        ('Llama Buckeye',
         results_base / 'Llama_crosslingual/buckeye_intact.json',
         results_base / 'Llama_crosslingual/buckeye_shuffled.json'),
        ('Llama French Oral',
         results_base / 'Llama_crosslingual/french_intact.json',
         results_base / 'Llama_crosslingual/french_shuffled.json'),
        ('Mistral Chinese Wiki',
         results_base / 'Wiki_multilingual_finegrain/zh_intact_v1.json',
         results_base / 'Wiki_multilingual_finegrain/zh_shuffled_v1.json'),
        ('Mistral Buckeye',
         results_base / 'Buckeye_finegrain/buckeye_intact_v1.json',
         results_base / 'Buckeye_finegrain/buckeye_shuffled_v1.json'),
    ]

    baselines = {}
    for name, ip, sp in datasets:
        if not (ip.exists() and sp.exists()):
            print(f'\n{name}: files not found, skipping')
            continue
        results = compute_corrected_marginal_from_paired(ip, sp)
        if not results:
            print(f'\n{name}: no matched pairs')
            continue
        baselines[name] = run_sweeps(results, name)

    # === Cross-dataset summary ===
    print(f'\n\n{"="*80}')
    print('SENSITIVITY SUMMARY')
    print(f'{"="*80}')
    print(f'\nBaseline exponents (log bins, d=1–100):')
    for name, a in baselines.items():
        print(f'  {name:<30} α = {a:.3f}')


if __name__ == '__main__':
    main()
