"""Per-token order-specific gap slope fit on long-range cells.

For each cell (corpus_id), reads the long-range cache JSON containing
per-target {context_lengths, ordered_ppl, shuffled_ppl} arrays, then:

  1. Computes per-target marginal context benefit at each interval,
     ord_marg[i]  = ord_ppl[i-1] - ord_ppl[i]
     shuf_marg[i] = shuf_ppl[i-1] - shuf_ppl[i]
  2. Normalizes by interval width (log-spaced grid),
     ord_rate  = ord_marg  / (c[i] - c[i-1])
     shuf_rate = shuf_marg / (c[i] - c[i-1])
  3. Order-specific gap rate = ord_rate - shuf_rate.
  4. Averages across targets, fits log(gap_rate) vs log(distance) on
     intervals where the cell-mean gap is positive. Distance is the
     geometric midpoint of the interval.

Includes random_vocab nulls for sanity (slope ≈ 0 / no fit).

Pure stdlib: no numpy required.
"""
import json
import math
from pathlib import Path

DRIVE = Path('/Users/elansmini/Library/CloudStorage/'
             'GoogleDrive-mpcrlab@gmail.com/My Drive/LRTIA')
BASE = DRIVE / 'Results/corpus_expansion_longrange/llama'

CELLS_NATURAL = [
    'gutenberg_fiction_en',
    'ted_transcripts_en',
    'ted_transcripts_de',
    'ted_transcripts_fr',
    'ted_transcripts_tr',
    'literary_ja',
    'literary_fi',
    'news_en',
    'buckeye',
]
CELLS_NULL = ['random_vocab_uniform', 'random_vocab_freq']


def linreg(xs, ys):
    """Stdlib linear regression. Returns slope, intercept, r_squared."""
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


def per_target_gap_rates(rec):
    """Returns list of (midpoint_distance, gap_rate) per interval for one target."""
    cs = rec['context_lengths']
    op = rec['ordered_ppl']
    sp = rec['shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            # Skip the 0->1 interval since shuf is undefined at c=0 (set to ord by pipeline)
            # which would zero out the gap by construction.
            continue
        width = cs[i] - cs[i - 1]
        ord_marg = op[i - 1] - op[i]
        shuf_marg = sp[i - 1] - sp[i]
        gap_rate = (ord_marg - shuf_marg) / width
        # Geometric midpoint for log-spaced grid.
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        out.append((d_mid, gap_rate))
    return out


def fit_cell(cell):
    fp = BASE / f'{cell}.json'
    if not fp.exists():
        return None
    with open(fp) as f:
        records = json.load(f)
    if not records:
        return None

    # Bin by interval index; average across targets.
    by_interval = {}  # interval_idx -> list of (d, gap_rate)
    for r in records:
        for i, (d, g) in enumerate(per_target_gap_rates(r)):
            by_interval.setdefault(i, []).append((d, g))

    if not by_interval:
        return None

    bins = []
    for i in sorted(by_interval):
        vals = by_interval[i]
        d = vals[0][0]  # interval midpoint is fixed across targets
        mean_gap = sum(g for _, g in vals) / len(vals)
        bins.append((d, mean_gap, len(vals)))

    # Fit on positive bins only.
    pos = [(d, g) for (d, g, _) in bins if g > 0]
    fit = linreg([math.log(d) for d, _ in pos],
                 [math.log(g) for _, g in pos])

    return {
        'cell': cell,
        'n_targets': len(records),
        'bins': bins,
        'n_pos_bins': len(pos),
        'slope': fit[0],
        'r2': fit[2],
    }


def summarize(results):
    slopes = [r['slope'] for r in results if r and r['slope'] is not None]
    if not slopes:
        return None, None
    m = sum(slopes) / len(slopes)
    var = sum((s - m) ** 2 for s in slopes) / max(1, len(slopes) - 1)
    return m, math.sqrt(var)


def fmt(x, w=7, p=3):
    if x is None:
        return ' ' * (w - 4) + ' n/a'
    return f'{x:>{w}.{p}f}'


def report(label, cells):
    print(f'\n=== {label} ===')
    print(f'{"cell":<26} {"n":>4} {"pos_bins":>9} {"slope":>8} {"r²":>7} '
          f'{"gap@~362":>10} {"gap@~724":>10}')
    print('-' * 80)
    rows = []
    for c in cells:
        r = fit_cell(c)
        rows.append(r)
        if r is None:
            print(f'{c:<26} {"--":>4} {"--":>9} {"--":>8} {"--":>7}')
            continue
        gap_362 = next((g for d, g, _ in r['bins'] if 300 < d < 400), None)
        gap_724 = next((g for d, g, _ in r['bins'] if 600 < d < 800), None)
        print(f'{c:<26} {r["n_targets"]:>4} {r["n_pos_bins"]:>9} '
              f'{fmt(r["slope"], 8, 3)} {fmt(r["r2"], 7, 3)} '
              f'{fmt(gap_362, 10, 5)} {fmt(gap_724, 10, 5)}')
    m, sd = summarize(rows)
    if m is not None:
        print('-' * 80)
        print(f'mean slope = {m:+.3f}  ±  {sd:.3f} SD across {len(cells)} cells')
    return rows


def main():
    nat_rows = report('NATURAL LANGUAGE (long-range, log-spaced 1..1024)', CELLS_NATURAL)
    null_rows = report('RANDOM-VOCAB NULL (no order structure)', CELLS_NULL)

    # Per-distance summary across natural cells.
    print('\n=== Per-distance mean gap rate (natural cells) ===')
    print(f'{"d_mid":>8}  {"mean_gap_rate":>14}  {"n_cells_pos":>13}')
    print('-' * 40)
    bin_idx_to_d = {}
    bin_collect = {}  # i -> list of cell-mean gaps
    for r in nat_rows:
        if r is None:
            continue
        for i, (d, g, _) in enumerate(r['bins']):
            bin_idx_to_d[i] = d
            bin_collect.setdefault(i, []).append(g)
    for i in sorted(bin_collect):
        gs = bin_collect[i]
        m = sum(gs) / len(gs)
        n_pos = sum(1 for g in gs if g > 0)
        print(f'{bin_idx_to_d[i]:>8.1f}  {m:>+14.5f}  {n_pos:>4} / {len(gs)}')


if __name__ == '__main__':
    main()
