"""Compare functional forms for the per-token order-specific gap rate.

For each natural-language long-range cell, fit alternative models to the
mean per-distance gap rate and compare residuals (r²) and AIC.

Models compared:
  1. Power law:           log(g) = a + slope * log(d)                (k=2)
  2. Exponential:         log(g) = a + b * d                          (k=2)
  3. Quadratic log-log:   log(g) = a + b * log(d) + c * (log d)^2     (k=3)
                          (captures lognormal-tail-like curvature)
  4. Stretched exp:       log(g) = a - (d/tau)^beta                   (k=3, NLLS via grid)

Reports per-cell r² and ΔAIC vs the power law.
"""
import json
import math
from pathlib import Path

DRIVE = Path('/Users/elansmini/Library/CloudStorage/'
             'GoogleDrive-mpcrlab@gmail.com/My Drive/LRTIA')
BASE = DRIVE / 'Results/corpus_expansion_longrange/llama'

CELLS = [
    'gutenberg_fiction_en', 'ted_transcripts_en', 'ted_transcripts_de',
    'ted_transcripts_fr', 'ted_transcripts_tr', 'literary_ja',
    'literary_fi', 'news_en', 'buckeye',
]


def per_target_gap_rates(rec):
    cs = rec['context_lengths']
    op = rec['ordered_ppl']
    sp = rec['shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            continue
        width = cs[i] - cs[i - 1]
        gap_rate = ((op[i - 1] - op[i]) - (sp[i - 1] - sp[i])) / width
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        out.append((d_mid, gap_rate))
    return out


def cell_mean_curve(cell):
    fp = BASE / f'{cell}.json'
    with open(fp) as f:
        recs = json.load(f)
    by_i = {}
    for r in recs:
        for i, (d, g) in enumerate(per_target_gap_rates(r)):
            by_i.setdefault(i, []).append((d, g))
    pts = []
    for i in sorted(by_i):
        d = by_i[i][0][0]
        m = sum(g for _, g in by_i[i]) / len(by_i[i])
        pts.append((d, m))
    return pts


def lstsq(rows, ys):
    """Solve normal equations for least squares. rows = list of feature vectors.
    Returns (coefs, predictions, ss_res)."""
    n = len(rows)
    k = len(rows[0])
    # X^T X (k x k)
    xtx = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(rows[i][a] * ys[i] for i in range(n)) for a in range(k)]
    # Solve via Gauss-Jordan.
    aug = [xtx[r] + [xty[r]] for r in range(k)]
    for c in range(k):
        # Pivot.
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        aug[c], aug[piv] = aug[piv], aug[c]
        if abs(aug[c][c]) < 1e-12:
            return None, None, None
        for r in range(k):
            if r == c:
                continue
            f = aug[r][c] / aug[c][c]
            for cc in range(c, k + 1):
                aug[r][cc] -= f * aug[c][cc]
    coefs = [aug[r][k] / aug[r][r] for r in range(k)]
    preds = [sum(rows[i][a] * coefs[a] for a in range(k)) for i in range(n)]
    ss_res = sum((ys[i] - preds[i]) ** 2 for i in range(n))
    return coefs, preds, ss_res


def aic_from_ssr(n, k, ssr):
    if ssr <= 0:
        return float('-inf')
    return n * math.log(ssr / n) + 2 * k


def fit_alternatives(pts):
    pos = [(d, g) for d, g in pts if g > 0]
    n = len(pos)
    if n < 4:
        return None
    ds = [d for d, _ in pos]
    gs = [g for _, g in pos]
    log_d = [math.log(d) for d in ds]
    log_g = [math.log(g) for g in gs]

    out = {'n_pts': n}

    # Power law: log(g) = a + b*log(d). k=2.
    coefs, preds, ssr = lstsq([[1, ld] for ld in log_d], log_g)
    out['power_law'] = {
        'coefs': coefs, 'ssr': ssr, 'k': 2,
        'aic': aic_from_ssr(n, 2, ssr),
    }

    # Exponential in d (linear semi-log): log(g) = a + b*d. k=2.
    coefs, preds, ssr = lstsq([[1, d] for d in ds], log_g)
    out['exponential'] = {
        'coefs': coefs, 'ssr': ssr, 'k': 2,
        'aic': aic_from_ssr(n, 2, ssr),
    }

    # Quadratic in log d: log(g) = a + b*log(d) + c*(log d)^2. k=3.
    coefs, preds, ssr = lstsq([[1, ld, ld * ld] for ld in log_d], log_g)
    out['log_quadratic'] = {
        'coefs': coefs, 'ssr': ssr, 'k': 3,
        'aic': aic_from_ssr(n, 3, ssr),
    }

    # Truncated power law: g = A * d^(-alpha) * exp(-d/d_cut).
    # log(g) = a + b*log(d) + c*d, with c = -1/d_cut (so c < 0 means real cutoff). k=3.
    coefs, preds, ssr = lstsq([[1, ld, d] for ld, d in zip(log_d, ds)], log_g)
    out['truncated_pl'] = {
        'coefs': coefs, 'ssr': ssr, 'k': 3,
        'aic': aic_from_ssr(n, 3, ssr),
        'd_cut': (-1.0 / coefs[2]) if (coefs and coefs[2] < 0) else float('inf'),
    }

    # Stretched exponential: log(g) = a - (d/tau)^beta. k=3.
    # Grid search beta in (0, 2], for each beta solve linear: log(g) = a - c*d^beta.
    best = None
    for beta_idx in range(1, 41):  # beta = 0.05 .. 2.0
        beta = beta_idx * 0.05
        rows = [[1, -(d ** beta)] for d in ds]
        try:
            coefs_se, _, ssr_se = lstsq(rows, log_g)
        except Exception:
            continue
        if coefs_se is None or coefs_se[1] < 0:
            continue
        if best is None or ssr_se < best['ssr']:
            best = {
                'coefs': coefs_se + [beta], 'ssr': ssr_se, 'k': 3,
                'aic': aic_from_ssr(n, 3, ssr_se), 'beta': beta,
            }
    out['stretched_exp'] = best

    return out


def main():
    print(f'{"cell":<24}  {"PL slope":>9} {"PL r²":>7} '
          f'{"ΔAIC exp":>9} {"ΔAIC quad":>10} {"ΔAIC strE":>10} {"strE β":>8} '
          f'{"ΔAIC trPL":>10} {"d_cut":>9}')
    print('-' * 110)
    summary = {'pl_wins_aic': 0, 'pl_wins_r2': 0, 'total': 0}
    for cell in CELLS:
        pts = cell_mean_curve(cell)
        f = fit_alternatives(pts)
        if f is None:
            print(f'{cell:<24}  -- no fit --')
            continue
        pl = f['power_law']
        ex = f['exponential']
        qd = f['log_quadratic']
        se = f['stretched_exp']
        tp = f['truncated_pl']

        # r² in log-log space: 1 - ssr / ss_tot(log g).
        log_g = [math.log(g) for d, g in pts if g > 0]
        mean_lg = sum(log_g) / len(log_g)
        ss_tot = sum((y - mean_lg) ** 2 for y in log_g)

        def r2(ssr):
            return 1 - ssr / ss_tot if ss_tot > 0 else float('nan')

        d_aic_exp = ex['aic'] - pl['aic']
        d_aic_quad = qd['aic'] - pl['aic']
        d_aic_se = se['aic'] - pl['aic'] if se else float('nan')
        d_aic_tp = tp['aic'] - pl['aic']
        beta = se['beta'] if se else float('nan')
        d_cut = tp['d_cut']

        # Track wins.
        summary['total'] += 1
        aics = {'pl': pl['aic'], 'exp': ex['aic'], 'quad': qd['aic'],
                'trPL': tp['aic']}
        if se: aics['strE'] = se['aic']
        if min(aics.values()) == pl['aic']:
            summary['pl_wins_aic'] += 1
        r2s = {'pl': r2(pl['ssr']), 'exp': r2(ex['ssr']),
               'quad': r2(qd['ssr']), 'trPL': r2(tp['ssr'])}
        if se: r2s['strE'] = r2(se['ssr'])
        if max(r2s.values()) == r2s['pl']:
            summary['pl_wins_r2'] += 1

        d_cut_str = f'{d_cut:>9.0f}' if math.isfinite(d_cut) else '       inf'
        print(f'{cell:<24}  {pl["coefs"][1]:>+9.3f} {r2(pl["ssr"]):>7.3f} '
              f'{d_aic_exp:>+9.2f} {d_aic_quad:>+10.2f} {d_aic_se:>+10.2f} '
              f'{beta:>8.2f} {d_aic_tp:>+10.2f} {d_cut_str}')

    print('-' * 110)
    print(f'\nPower law has the BEST AIC of all 4 forms in '
          f'{summary["pl_wins_aic"]}/{summary["total"]} cells.')
    print(f'Power law has the BEST r² of all 4 forms in '
          f'{summary["pl_wins_r2"]}/{summary["total"]} cells.')
    print('\nΔAIC > 0 means model is WORSE than power law. Rule of thumb: '
          'ΔAIC < 2 = indistinguishable, > 10 = strong evidence for the '
          'better one.')
    print('strE β: shape parameter (β=1 is plain exponential, β<1 is '
          'stretched exp / heavier tail, but for it to be a power-law-like '
          'tail you need β→0).')


if __name__ == '__main__':
    main()
