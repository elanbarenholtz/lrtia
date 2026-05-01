#!/usr/bin/env python3
"""Bootstrap confidence intervals for power law exponents.

Resamples by document (written) or speaker/storyteller (spoken),
refits power law on each resample, reports 95% CIs.

Runs locally — no GPU needed. Loads cached finegrain results from Drive
or local results directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# === Config ===
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
SEED = 12345
BIN_EDGES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]
COMMON_X = np.arange(1, 101)


def compute_raw_ppl_curve(curves, key='ppls'):
    """Mean raw PPL at each context length."""
    all_ppl = []
    for curve in curves:
        ctx = np.array(curve.get('ctx_lengths', list(range(1, 101))))
        ppl = np.array(curve[key])
        if len(ppl) == 101:  # c=0..100
            all_ppl.append(ppl)
        else:
            interp = np.interp(COMMON_X, ctx, ppl, left=np.nan, right=np.nan)
            all_ppl.append(interp)
    return np.nanmean(np.array(all_ppl), axis=0)


def get_corrected_marginal(intact_curves, shuffled_curves):
    """Compute corrected marginal from intact and shuffled curves."""
    ip = compute_raw_ppl_curve(intact_curves)
    sp = compute_raw_ppl_curve(shuffled_curves)
    return -np.diff(ip) - (-np.diff(sp))


def get_marginal_from_delta(curves, key='delta_ppl'):
    """Get pre-computed corrected marginals."""
    all_delta = []
    for c in curves:
        all_delta.append(np.array(c[key]))
    return np.mean(all_delta, axis=0)


def fit_power_law(marg):
    """Fit power law to binned marginals. Returns (slope, r) or (None, None)."""
    bm, bc = [], []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        vals = marg[lo - 1:hi - 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0 and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((lo + hi) / 2)
    if len(bm) >= 4:
        slope, intercept, r, p, _ = stats.linregress(np.log(bc), np.log(bm))
        return slope, r
    return None, None


def bootstrap_exponent(
    curves,
    marginal_key='delta_ppl',
    group_key='doc_id',
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):
    """Bootstrap power law exponent by resampling groups (documents/speakers).

    Args:
        curves: list of per-target result dicts
        marginal_key: key for corrected marginal in each result
        group_key: key to group by (doc_id, author_id, etc.)
        n_bootstrap: number of resamples
        seed: random seed

    Returns:
        dict with point estimate, CI, bootstrap distribution
    """
    # Group curves by document/speaker
    groups = {}
    for c in curves:
        gid = c.get(group_key, c.get('doc_id', 'unknown'))
        groups.setdefault(gid, []).append(c)

    group_ids = list(groups.keys())
    n_groups = len(group_ids)

    if n_groups < 3:
        return {'alpha': None, 'ci_lo': None, 'ci_hi': None,
                'r': None, 'n_groups': n_groups, 'error': 'too few groups'}

    # Point estimate
    all_marg = np.mean([np.array(c[marginal_key]) for c in curves], axis=0)
    alpha_point, r_point = fit_power_law(all_marg)

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot_alphas = []

    for _ in range(n_bootstrap):
        # Resample groups with replacement
        sampled_ids = rng.choice(group_ids, size=n_groups, replace=True)
        sampled_curves = []
        for gid in sampled_ids:
            sampled_curves.extend(groups[gid])

        boot_marg = np.mean([np.array(c[marginal_key]) for c in sampled_curves], axis=0)
        alpha, r = fit_power_law(boot_marg)
        if alpha is not None:
            boot_alphas.append(alpha)

    boot_alphas = np.array(boot_alphas)

    if len(boot_alphas) < n_bootstrap * 0.5:
        return {'alpha': alpha_point, 'ci_lo': None, 'ci_hi': None,
                'r': r_point, 'n_groups': n_groups,
                'error': f'too many failed fits ({len(boot_alphas)}/{n_bootstrap})'}

    lo = np.percentile(boot_alphas, (1 - CI_LEVEL) / 2 * 100)
    hi = np.percentile(boot_alphas, (1 + CI_LEVEL) / 2 * 100)

    return {
        'alpha': alpha_point,
        'ci_lo': lo,
        'ci_hi': hi,
        'r': r_point,
        'n_groups': n_groups,
        'n_boot_valid': len(boot_alphas),
        'boot_mean': float(np.mean(boot_alphas)),
        'boot_sd': float(np.std(boot_alphas)),
        'boot_distribution': boot_alphas.tolist(),
    }


def bootstrap_from_paired_files(intact_path, shuffled_path, group_key='doc_id'):
    """Bootstrap from separate intact/shuffled JSON files (old format)."""
    with open(intact_path) as f:
        intact = json.load(f)
    with open(shuffled_path) as f:
        shuffled = json.load(f)

    # Build corrected marginals per target
    # Group by doc_id + target_frac to match intact/shuffled pairs
    intact_by_key = {}
    for c in intact:
        k = (c.get('doc_id', ''), c.get('target_frac', 0))
        intact_by_key[k] = c
    shuffled_by_key = {}
    for c in shuffled:
        k = (c.get('doc_id', ''), c.get('target_frac', 0))
        shuffled_by_key[k] = c

    merged = []
    for k, ic in intact_by_key.items():
        sc = shuffled_by_key.get(k)
        if sc is None:
            continue
        ip = np.array(ic['ppls'] if 'ppls' in ic else ic.get('ctx_lengths', []))
        sp = np.array(sc['ppls'] if 'ppls' in sc else sc.get('ctx_lengths', []))

        if 'ppls' in ic and 'ppls' in sc:
            i_ppl = np.array(ic['ppls'])
            s_ppl = np.array(sc['ppls'])
            # Compute corrected marginal
            i_marg = np.array([i_ppl[d - 1] - i_ppl[d] for d in range(1, len(i_ppl))])
            s_marg = np.array([s_ppl[d - 1] - s_ppl[d] for d in range(1, len(s_ppl))])
            min_len = min(len(i_marg), len(s_marg))
            delta = (i_marg[:min_len] - s_marg[:min_len]).tolist()
        else:
            continue

        merged.append({
            'doc_id': ic.get('doc_id', ''),
            'target_frac': ic.get('target_frac', 0),
            'delta_ppl': delta,
        })

    if not merged:
        return {'error': 'no matched pairs'}

    return bootstrap_exponent(merged, marginal_key='delta_ppl', group_key=group_key)


# === Dataset definitions ===

DATASETS = {
    # Llama crosslingual (use paired intact/shuffled files)
    'Llama Chinese Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_zh_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_zh_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Japanese Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_ja_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_ja_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Korean Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_ko_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_ko_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Turkish Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_tr_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_tr_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Arabic Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_ar_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_ar_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Finnish Wiki': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/wiki_fi_intact.json',
        'shuffled': 'Llama_crosslingual/wiki_fi_shuffled.json',
        'group': 'doc_id',
    },
    'Llama Buckeye': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/buckeye_intact.json',
        'shuffled': 'Llama_crosslingual/buckeye_shuffled.json',
        'group': 'doc_id',  # speaker = doc_id for Buckeye
    },
    'Llama French Oral': {
        'type': 'paired',
        'intact': 'Llama_crosslingual/french_intact.json',
        'shuffled': 'Llama_crosslingual/french_shuffled.json',
        'group': 'doc_id',
    },
    # Exp1A formal (corrected marginals already computed)
    'Llama zh intact (corrected)': {
        'type': 'corrected',
        'path': 'Exp1A_formal/llama_wiki_zh_intact.json',
        'group': 'doc_id',
    },
    'Llama zh D4 (corrected)': {
        'type': 'corrected',
        'path': 'Exp1A_formal/llama_wiki_zh_D4.json',
        'group': 'doc_id',
    },
}


def main():
    # Try local results first, then Drive
    results_base = Path('results')
    if not results_base.exists():
        results_base = Path('/content/drive/MyDrive/LRTIA/Results')
    if not results_base.exists():
        print(f'Results directory not found')
        sys.exit(1)

    print(f'Results base: {results_base}')
    print(f'Bootstrap: {N_BOOTSTRAP} resamples, {CI_LEVEL*100:.0f}% CI')
    print(f'{"="*80}')
    print(f'{"Dataset":<30} {"α":>8} {"95% CI":>18} {"r":>8} {"N grp":>6} {"Boot SD":>8}')
    print(f'{"-"*80}')

    for name, spec in DATASETS.items():
        if spec['type'] == 'paired':
            ip = results_base / spec['intact']
            sp = results_base / spec['shuffled']
            if not (ip.exists() and sp.exists()):
                print(f'{name:<30} {"— files not found":>8}')
                continue
            result = bootstrap_from_paired_files(ip, sp, spec['group'])
        elif spec['type'] == 'corrected':
            p = results_base / spec['path']
            if not p.exists():
                print(f'{name:<30} {"— file not found":>8}')
                continue
            with open(p) as f:
                curves = json.load(f)
            result = bootstrap_exponent(curves, 'delta_ppl', spec['group'])
        else:
            continue

        if 'error' in result:
            print(f'{name:<30} {"ERROR: " + result["error"]:>8}')
            continue

        a = result['alpha']
        lo = result['ci_lo']
        hi = result['ci_hi']
        r = result['r']
        ng = result['n_groups']
        sd = result.get('boot_sd', 0)

        ci_str = f'[{lo:.3f}, {hi:.3f}]' if lo is not None else '—'
        print(f'{name:<30} {a:>8.3f} {ci_str:>18} {r:>8.3f} {ng:>6} {sd:>8.3f}')


if __name__ == '__main__':
    main()
