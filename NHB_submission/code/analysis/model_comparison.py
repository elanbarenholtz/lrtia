#!/usr/bin/env python3
"""Formal model comparison: power law vs alternatives.

Fits 6 functional forms to corrected marginals across all datasets.
Reports AIC, BIC, R², and held-out prediction error.

Held-out test: fit on odd-indexed bins, predict even-indexed bins.
This goes beyond AIC/BIC by testing actual generalization.

Runs locally on cached results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats, optimize


# === Bin definitions ===
LOG_BINS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def get_binned(marg, bin_edges=LOG_BINS):
    """Bin marginals."""
    bm, bc = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        vals = marg[lo - 1:hi - 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0 and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((lo + hi) / 2)
    return np.array(bc), np.array(bm)


# === Candidate models ===

def power_law(x, a, b):
    return a * np.power(x, -b)

def exponential(x, a, b):
    return a * np.exp(-b * x)

def stretched_exp(x, a, b, c):
    return a * np.exp(-b * np.power(x, c))

def logarithmic(x, a, b):
    return a - b * np.log(x)

def linear_fn(x, a, b):
    return a - b * x

def broken_power_law(x, a, b1, b2, xb):
    """Two-segment power law with breakpoint at xb."""
    result = np.where(
        x <= xb,
        a * np.power(x, -b1),
        a * np.power(xb, -b1 + b2) * np.power(x, -b2)
    )
    return result


MODELS = {
    'Power law': {
        'func': power_law,
        'p0': [1.0, 0.75],
        'bounds': ((0, 0), (np.inf, 5)),
        'k': 2,
    },
    'Exponential': {
        'func': exponential,
        'p0': [1.0, 0.05],
        'bounds': ((0, 0), (np.inf, 1)),
        'k': 2,
    },
    'Stretched exp': {
        'func': stretched_exp,
        'p0': [1.0, 0.05, 0.5],
        'bounds': ((0, 0, 0.01), (np.inf, 10, 2)),
        'k': 3,
    },
    'Logarithmic': {
        'func': logarithmic,
        'p0': [1.0, 0.1],
        'bounds': ((-np.inf, 0), (np.inf, np.inf)),
        'k': 2,
    },
    'Linear': {
        'func': linear_fn,
        'p0': [1.0, 0.01],
        'bounds': ((0, 0), (np.inf, 1)),
        'k': 2,
    },
}


def compute_aic(n, rss, k):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def compute_bic(n, rss, k):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def fit_model(func, x, y, p0, bounds, k):
    """Fit one model, return dict of results."""
    try:
        popt, _ = optimize.curve_fit(func, x, y, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = func(x, *popt)
        rss = np.sum((y - y_pred) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / tss if tss > 0 else 0
        n = len(x)
        return {
            'params': popt,
            'r2': r2,
            'rss': rss,
            'aic': compute_aic(n, rss, k),
            'bic': compute_bic(n, rss, k),
            'k': k,
            'y_pred': y_pred,
            'success': True,
        }
    except (RuntimeError, ValueError):
        return {'success': False}


def held_out_test(func, x, y, p0, bounds):
    """Fit on odd-indexed points, predict even-indexed. Return RMSE."""
    if len(x) < 6:
        return None
    odd_idx = list(range(0, len(x), 2))
    even_idx = list(range(1, len(x), 2))
    x_train, y_train = x[odd_idx], y[odd_idx]
    x_test, y_test = x[even_idx], y[even_idx]
    try:
        popt, _ = optimize.curve_fit(func, x_train, y_train, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = func(x_test, *popt)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        return rmse
    except (RuntimeError, ValueError):
        return None


def analyze_dataset(marg, label):
    """Full model comparison for one dataset."""
    bc, bm = get_binned(marg)
    if len(bc) < 4:
        print(f'  {label}: too few bins')
        return None

    results = {}
    for name, spec in MODELS.items():
        fit = fit_model(spec['func'], bc, bm, spec['p0'], spec['bounds'], spec['k'])
        if fit['success']:
            ho_rmse = held_out_test(spec['func'], bc, bm, spec['p0'], spec['bounds'])
            fit['ho_rmse'] = ho_rmse
            results[name] = fit

    if not results:
        return None

    # Delta AIC/BIC
    best_aic = min(r['aic'] for r in results.values())
    best_bic = min(r['bic'] for r in results.values())
    for name in results:
        results[name]['delta_aic'] = results[name]['aic'] - best_aic
        results[name]['delta_bic'] = results[name]['bic'] - best_bic

    return results


def load_corrected_marginal(intact_path, shuffled_path):
    """Load paired files, return mean corrected marginal."""
    with open(intact_path) as f:
        intact = json.load(f)
    with open(shuffled_path) as f:
        shuffled = json.load(f)

    COMMON_X = np.arange(1, 101)

    def mean_ppl(curves):
        all_ppl = []
        for c in curves:
            ppl = np.array(c['ppls'])
            if len(ppl) == 101:
                all_ppl.append(ppl)
            else:
                ctx = np.array(c.get('ctx_lengths', list(range(1, 101))))
                interp = np.interp(COMMON_X, ctx, ppl, left=np.nan, right=np.nan)
                all_ppl.append(interp)
        return np.nanmean(np.array(all_ppl), axis=0)

    ip = mean_ppl(intact)
    sp = mean_ppl(shuffled)
    return -np.diff(ip) - (-np.diff(sp))


def main():
    R = Path('results')

    datasets = [
        ('Llama zh Wiki', R / 'Llama_crosslingual/wiki_zh_intact.json',
         R / 'Llama_crosslingual/wiki_zh_shuffled.json'),
        ('Llama ja Wiki', R / 'Llama_crosslingual/wiki_ja_intact.json',
         R / 'Llama_crosslingual/wiki_ja_shuffled.json'),
        ('Llama ko Wiki', R / 'Llama_crosslingual/wiki_ko_intact.json',
         R / 'Llama_crosslingual/wiki_ko_shuffled.json'),
        ('Llama tr Wiki', R / 'Llama_crosslingual/wiki_tr_intact.json',
         R / 'Llama_crosslingual/wiki_tr_shuffled.json'),
        ('Llama ar Wiki', R / 'Llama_crosslingual/wiki_ar_intact.json',
         R / 'Llama_crosslingual/wiki_ar_shuffled.json'),
        ('Llama fi Wiki', R / 'Llama_crosslingual/wiki_fi_intact.json',
         R / 'Llama_crosslingual/wiki_fi_shuffled.json'),
        ('Llama Buckeye', R / 'Llama_crosslingual/buckeye_intact.json',
         R / 'Llama_crosslingual/buckeye_shuffled.json'),
        ('Llama French', R / 'Llama_crosslingual/french_intact.json',
         R / 'Llama_crosslingual/french_shuffled.json'),
        ('Mistral zh Wiki', R / 'Wiki_multilingual_finegrain/zh_intact_v1.json',
         R / 'Wiki_multilingual_finegrain/zh_shuffled_v1.json'),
        ('Mistral Buckeye', R / 'Buckeye_finegrain/buckeye_intact_v1.json',
         R / 'Buckeye_finegrain/buckeye_shuffled_v1.json'),
        ('Mistral French', R / 'French_oral_finegrain/french_oral_intact_v1.json',
         R / 'French_oral_finegrain/french_oral_shuffled_v1.json'),
    ]

    # Tally wins
    aic_wins = {}
    bic_wins = {}
    ho_wins = {}

    print(f'{"="*90}')
    print('MODEL COMPARISON: Power Law vs Alternatives')
    print(f'{"="*90}')

    for name, ip, sp in datasets:
        if not (ip.exists() and sp.exists()):
            continue

        marg = load_corrected_marginal(ip, sp)
        results = analyze_dataset(marg, name)
        if results is None:
            continue

        print(f'\n--- {name} ---')
        print(f'  {"Model":<16} {"R²":>7} {"AIC":>9} {"ΔAIC":>7} {"BIC":>9} {"ΔBIC":>7} {"HO RMSE":>9}')
        print(f'  {"-"*66}')

        sorted_models = sorted(results.items(), key=lambda x: x[1]['aic'])
        best_aic_name = sorted_models[0][0]
        best_bic_name = min(results, key=lambda k: results[k]['bic'])

        # Held-out best
        ho_results = {k: v['ho_rmse'] for k, v in results.items() if v.get('ho_rmse') is not None}
        best_ho_name = min(ho_results, key=ho_results.get) if ho_results else None

        for mname, fit in sorted_models:
            marker = ''
            if mname == best_aic_name:
                marker += ' ←AIC'
            if mname == best_bic_name:
                marker += ' ←BIC'
            if mname == best_ho_name:
                marker += ' ←HO'
            ho_str = f'{fit["ho_rmse"]:.5f}' if fit.get('ho_rmse') is not None else '—'
            print(f'  {mname:<16} {fit["r2"]:>7.4f} {fit["aic"]:>9.2f} {fit["delta_aic"]:>7.2f} '
                  f'{fit["bic"]:>9.2f} {fit["delta_bic"]:>7.2f} {ho_str:>9}{marker}')

        aic_wins[best_aic_name] = aic_wins.get(best_aic_name, 0) + 1
        bic_wins[best_bic_name] = bic_wins.get(best_bic_name, 0) + 1
        if best_ho_name:
            ho_wins[best_ho_name] = ho_wins.get(best_ho_name, 0) + 1

    # Summary
    n_datasets = sum(aic_wins.values())
    print(f'\n{"="*90}')
    print(f'SUMMARY ({n_datasets} datasets)')
    print(f'{"="*90}')

    print(f'\n  {"Model":<16} {"AIC wins":>10} {"BIC wins":>10} {"HO wins":>10}')
    print(f'  {"-"*48}')
    all_models = set(list(aic_wins.keys()) + list(bic_wins.keys()) + list(ho_wins.keys()))
    for m in ['Power law', 'Exponential', 'Stretched exp', 'Logarithmic', 'Linear']:
        if m in all_models or True:
            aw = aic_wins.get(m, 0)
            bw = bic_wins.get(m, 0)
            hw = ho_wins.get(m, 0)
            print(f'  {m:<16} {aw:>10} {bw:>10} {hw:>10}')


if __name__ == '__main__':
    main()
