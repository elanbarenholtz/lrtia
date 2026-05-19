"""Per-distance decomposition of long-range influence into discourse-chaining
and content-access components, from the sentence-shuffle cache.

For each target record we have three perplexity arrays at log-spaced ctx
[0, 1, 2, 4, ..., 1024]: ordered, sentence_shuffled, token_shuffled.

Per-interval per-token marginal benefit rates (normalized by interval width):
    ord_rate        = (ord_ppl[i-1]   - ord_ppl[i])   / width
    sent_shuf_rate  = (ss_ppl[i-1]    - ss_ppl[i])    / width
    tok_shuf_rate   = (ts_ppl[i-1]    - ts_ppl[i])    / width

Three components per interval:
    total order-specific  =  ord_rate - tok_shuf_rate     (matches the original paper)
    content-access        =  sent_shuf_rate - tok_shuf_rate
    discourse-chaining    =  ord_rate - sent_shuf_rate

Per cell, fit log-log power law on the positive bins of each component and
report slopes + r². Also report chain_share = chaining / total_order_specific
at each distance bin (how much of the long-range influence is order-of-
sentences specific, separate from content-access).

Pure stdlib.
"""
import json
import math
from pathlib import Path

DRIVE = Path('/Users/elansmini/Library/CloudStorage/'
             'GoogleDrive-mpcrlab@gmail.com/My Drive/LRTIA')
BASE = DRIVE / 'Results/corpus_expansion_longrange_sentshuf/llama'

CELLS = [
    'gutenberg_fiction_en', 'ted_transcripts_en', 'ted_transcripts_de',
    'ted_transcripts_fr', 'ted_transcripts_tr', 'literary_ja',
    'literary_fi', 'news_en',
    # buckeye excluded — spontaneous-speech transcripts have no sentence
    # boundaries so the sent_shuf condition is undefined for it.
]


def linreg(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None, None, None
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy)
    return slope, intercept, r2


def per_target_rates(rec):
    """Returns list of (d_mid, ord_rate, ss_rate, ts_rate) per interval."""
    cs = rec['context_lengths']
    op = rec['ordered_ppl']
    sp = rec['sentence_shuffled_ppl']
    tp = rec['token_shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            continue
        width = cs[i] - cs[i - 1]
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        ord_r = (op[i - 1] - op[i]) / width
        ss_r = (sp[i - 1] - sp[i]) / width
        ts_r = (tp[i - 1] - tp[i]) / width
        out.append((d_mid, ord_r, ss_r, ts_r))
    return out


def cell_decomposition(cell):
    fp = BASE / f'{cell}.json'
    if not fp.exists():
        return None
    with open(fp) as f:
        records = json.load(f)
    if not records:
        return None

    by_i = {}  # interval idx -> list of (d, ord_r, ss_r, ts_r)
    for r in records:
        for i, tup in enumerate(per_target_rates(r)):
            by_i.setdefault(i, []).append(tup)

    bins = []  # per-bin cell-level mean rates
    for i in sorted(by_i):
        rows = by_i[i]
        d = rows[0][0]
        n = len(rows)
        ord_m = sum(t[1] for t in rows) / n
        ss_m = sum(t[2] for t in rows) / n
        ts_m = sum(t[3] for t in rows) / n
        total = ord_m - ts_m
        content = ss_m - ts_m
        chain = ord_m - ss_m
        bins.append({
            'd': d, 'n': n,
            'ord_rate': ord_m, 'ss_rate': ss_m, 'ts_rate': ts_m,
            'total': total, 'content': content, 'chain': chain,
            'chain_share': chain / total if total > 0 else None,
        })

    def fit(component_key):
        pos = [b for b in bins if b[component_key] > 0]
        if len(pos) < 4:
            return None, None, len(pos)
        s, _, r2 = linreg([math.log(b['d']) for b in pos],
                          [math.log(b[component_key]) for b in pos])
        return s, r2, len(pos)

    s_total, r_total, n_total = fit('total')
    s_content, r_content, n_content = fit('content')
    s_chain, r_chain, n_chain = fit('chain')

    return {
        'cell': cell,
        'n_targets': len(records),
        'bins': bins,
        'fit_total': (s_total, r_total, n_total),
        'fit_content': (s_content, r_content, n_content),
        'fit_chain': (s_chain, r_chain, n_chain),
    }


def fmt(x, w=8, p=3):
    if x is None:
        return ' ' * (w - 4) + ' n/a'
    return f'{x:>{w}.{p}f}'


def fmt_pct(x, w=8):
    if x is None:
        return ' ' * (w - 4) + ' n/a'
    return f'{x*100:>{w-1}.1f}%'


def main():
    rows = [cell_decomposition(c) for c in CELLS]
    rows = [r for r in rows if r is not None]

    print('=' * 100)
    print('Per-cell power-law fits on each component (log-log slope, r², n_pos_bins)')
    print('=' * 100)
    print(f'{"cell":<24}  {"n":>4} | {"total slope":>12} {"r²":>6} {"npos":>5} | '
          f'{"content slope":>14} {"r²":>6} {"npos":>5} | '
          f'{"chain slope":>12} {"r²":>6} {"npos":>5}')
    print('-' * 100)

    slopes = {'total': [], 'content': [], 'chain': []}
    for r in rows:
        st, rt, nt = r['fit_total']
        sc, rc, nc = r['fit_content']
        sh, rh, nh = r['fit_chain']
        if st is not None: slopes['total'].append(st)
        if sc is not None: slopes['content'].append(sc)
        if sh is not None: slopes['chain'].append(sh)
        print(f'{r["cell"]:<24}  {r["n_targets"]:>4} | '
              f'{fmt(st, 12)} {fmt(rt, 6, 2)} {nt:>5} | '
              f'{fmt(sc, 14)} {fmt(rc, 6, 2)} {nc:>5} | '
              f'{fmt(sh, 12)} {fmt(rh, 6, 2)} {nh:>5}')
    print('-' * 100)

    def stats(vals, name):
        if not vals:
            return f'{name}: no fits'
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1))
        return f'{name:>16}: mean = {m:+.3f}  ±  {sd:.3f}  (n = {len(vals)})'
    print(stats(slopes['total'],   'total slope'))
    print(stats(slopes['content'], 'content slope'))
    print(stats(slopes['chain'],   'chain slope'))

    print()
    print('=' * 100)
    print('Per-distance chain_share (= chain / total_order_specific) per cell')
    print('=' * 100)
    if rows:
        ds = [b['d'] for b in rows[0]['bins']]
        header = f'{"cell":<24} ' + ' '.join(f'{d:>7.1f}' for d in ds)
        print(header)
        print('-' * len(header))
        for r in rows:
            line = f'{r["cell"]:<24} '
            for b in r['bins']:
                if b['chain_share'] is None:
                    line += f'{"   --":>7} '
                else:
                    line += f'{b["chain_share"]*100:>6.1f}% '
            print(line.rstrip())
        # Mean across cells per distance.
        print('-' * len(header))
        line = f'{"MEAN (8 cells)":<24} '
        for j in range(len(ds)):
            vals = [r['bins'][j]['chain_share'] for r in rows
                    if r['bins'][j]['chain_share'] is not None]
            if vals:
                line += f'{sum(vals)/len(vals)*100:>6.1f}% '
            else:
                line += f'{"--":>7} '
        print(line.rstrip())

    print()
    print('=' * 100)
    print('Per-distance per-cell rates: discourse-chaining component (positive = ordered helps beyond sent-shuffle)')
    print('=' * 100)
    if rows:
        ds = [b['d'] for b in rows[0]['bins']]
        header = f'{"cell":<24} ' + ' '.join(f'{d:>9.1f}' for d in ds)
        print(header)
        print('-' * len(header))
        for r in rows:
            line = f'{r["cell"]:<24} '
            for b in r['bins']:
                line += f'{b["chain"]:>+9.4f} '
            print(line.rstrip())


if __name__ == '__main__':
    main()
