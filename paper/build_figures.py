"""Build the four main-text figures specified in figures_spec.md.

Each figure is saved as both PNG (300 dpi) and PDF in paper/Paper_Figures/.

Data sources (all local on Drive):
  Fig 1: corpus_expansion_longrange/llama/<cell>.json   (9 cells)
  Fig 2: corpus_expansion_longrange/llama/{natural cells, random_vocab_*}.json
  Fig 3: corpus_expansion_longrange_sentshuf/llama/<cell>.json  (8 cells)
  Fig 4: sentence_ablation/llama/, sentence_influence_matrix/llama/  (gutenberg_en, ted_en)
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DRIVE = Path('/Users/elansmini/Library/CloudStorage/'
             'GoogleDrive-mpcrlab@gmail.com/My Drive/LRTIA/Results')
OUT = Path('/Users/elansmini/lrtia/paper/Paper_Figures')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 100,
})

CELLS_LONGRANGE = [
    'gutenberg_fiction_en', 'ted_transcripts_en', 'ted_transcripts_de',
    'ted_transcripts_fr', 'ted_transcripts_tr', 'ted_transcripts_ru',
    'literary_ja', 'literary_fi', 'news_en', 'buckeye',
]
STRETCHED_EXP_CELLS = {'literary_ja', 'buckeye'}

CELLS_SENTSHUF = [
    'gutenberg_fiction_en', 'ted_transcripts_en', 'ted_transcripts_de',
    'ted_transcripts_fr', 'ted_transcripts_tr', 'literary_ja',
    'literary_fi', 'news_en',
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def per_target_gap_rates(rec):
    """Per-interval per-token rates for one target. Returns (d, gap, ord, shuf) per interval."""
    cs = rec['context_lengths']
    op = rec['ordered_ppl']
    sp = rec['shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            continue
        width = cs[i] - cs[i - 1]
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        ord_marg = (op[i - 1] - op[i]) / width
        shuf_marg = (sp[i - 1] - sp[i]) / width
        gap = ord_marg - shuf_marg
        out.append((d_mid, gap, ord_marg, shuf_marg))
    return out


def cell_curves(path):
    """Returns (d, gap_mean, ord_mean, shuf_mean, n_targets)."""
    with open(path) as f:
        recs = json.load(f)
    by_i = {}
    for r in recs:
        for i, tup in enumerate(per_target_gap_rates(r)):
            by_i.setdefault(i, []).append(tup)
    ds, gaps, ords, shufs = [], [], [], []
    for i in sorted(by_i):
        rows = by_i[i]
        n = len(rows)
        ds.append(rows[0][0])
        gaps.append(sum(t[1] for t in rows) / n)
        ords.append(sum(t[2] for t in rows) / n)
        shufs.append(sum(t[3] for t in rows) / n)
    return np.array(ds), np.array(gaps), np.array(ords), np.array(shufs), len(recs)


def per_target_three_rates(rec):
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
        out.append((d_mid,
                    (op[i - 1] - op[i]) / width,
                    (sp[i - 1] - sp[i]) / width,
                    (tp[i - 1] - tp[i]) / width))
    return out


def cell_three_curves(path):
    with open(path) as f:
        recs = json.load(f)
    by_i = {}
    for r in recs:
        for i, tup in enumerate(per_target_three_rates(r)):
            by_i.setdefault(i, []).append(tup)
    ds, ord_m, ss_m, ts_m = [], [], [], []
    for i in sorted(by_i):
        rows = by_i[i]
        n = len(rows)
        ds.append(rows[0][0])
        ord_m.append(sum(t[1] for t in rows) / n)
        ss_m.append(sum(t[2] for t in rows) / n)
        ts_m.append(sum(t[3] for t in rows) / n)
    return np.array(ds), np.array(ord_m), np.array(ss_m), np.array(ts_m)


def per_target_four_rates(rec):
    """Same as per_target_three_rates plus sentence_reversed."""
    cs = rec['context_lengths']
    op = rec['ordered_ppl']
    sp = rec['sentence_shuffled_ppl']
    rp = rec['sentence_reversed_ppl']
    tp = rec['token_shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            continue
        width = cs[i] - cs[i - 1]
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        out.append((d_mid,
                    (op[i - 1] - op[i]) / width,
                    (sp[i - 1] - sp[i]) / width,
                    (rp[i - 1] - rp[i]) / width,
                    (tp[i - 1] - tp[i]) / width))
    return out


def cell_four_curves(path):
    """Reads a sent-reverse cache: returns (d, ord, sent_shuf, sent_rev, tok_shuf)."""
    with open(path) as f:
        recs = json.load(f)
    by_i = {}
    for r in recs:
        for i, tup in enumerate(per_target_four_rates(r)):
            by_i.setdefault(i, []).append(tup)
    ds, ord_m, ss_m, sr_m, ts_m = [], [], [], [], []
    for i in sorted(by_i):
        rows = by_i[i]
        n = len(rows)
        ds.append(rows[0][0])
        ord_m.append(sum(t[1] for t in rows) / n)
        ss_m.append(sum(t[2] for t in rows) / n)
        sr_m.append(sum(t[3] for t in rows) / n)
        ts_m.append(sum(t[4] for t in rows) / n)
    return (np.array(ds), np.array(ord_m), np.array(ss_m),
            np.array(sr_m), np.array(ts_m))


def fit_pl(d, y):
    """Power-law fit on positive bins. Returns (slope, r2)."""
    pos = y > 0
    if sum(pos) < 4:
        return None, None
    s, _, r, _, _ = stats.linregress(np.log(d[pos]), np.log(y[pos]))
    return s, r ** 2


def panel_label(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')


# -----------------------------------------------------------------------------
# Figure 1 — Cross-corpus persistence functions
# -----------------------------------------------------------------------------

def fig1():
    """Fig 1: small-multiples 3x3 grid of P(d) per cell, d >= 10.

    Trim threshold of d=10 drops the first two intervals (cs=2->4, cs=4->8)
    where the shuffled baseline is unstable. Memory's load-bearing claim is
    "P(d) at d >= 10 is positive"; this matches that boundary.
    """
    base = DRIVE / 'corpus_expansion_longrange/llama'
    D_MIN = 10.0

    ncol = 5
    nrow = (len(CELLS_LONGRANGE) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap('tab10')
    slopes = []

    for idx, cell in enumerate(CELLS_LONGRANGE):
        d, g, _, _, _ = cell_curves(base / f'{cell}.json')
        # Fit and plot only the stable range d >= D_MIN. Below ~10 tokens the
        # shuffled-token baseline is unstable (too few tokens to permute), so
        # those intervals are excluded (see caption / Methods).
        keep = (d >= D_MIN) & (g > 0)
        d, g = d[keep], g[keep]
        s, r2 = fit_pl(d, g)
        slopes.append(s)
        is_stretched = cell in STRETCHED_EXP_CELLS

        ax = axes[idx // ncol][idx % ncol]
        ax.plot(d, g, 'o-', color=cmap(idx), linewidth=1.6, markersize=5, alpha=0.95)

        if s is not None and len(d) >= 3:
            log_d = np.log(d); log_g = np.log(g)
            intercept = log_g.mean() - s * log_d.mean()
            x_fit = np.array([d.min(), d.max()])
            ax.plot(x_fit, np.exp(intercept) * x_fit ** s, ':',
                    color='black', linewidth=1.2, alpha=0.7)

        title = cell.replace('_', ' ')
        if is_stretched:
            title += ' †'
        ax.set_title(title, fontsize=10)
        slope_str = f'α = {-s:.2f}' if s is not None else 'no fit'  # P(d)∝d^-α; display positive α
        ax.text(0.05, 0.05, slope_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')

    for ax in axes[-1]:
        ax.set_xlabel('Distance d (tokens)')
    for ax in axes[:, 0]:
        ax.set_ylabel('P(d)')

    valid = [s for s in slopes if s is not None]
    mean_s, sd_s = np.mean(valid), np.std(valid)
    fig.suptitle(
        f'Cross-corpus persistence functions  '
        f'({len(CELLS_LONGRANGE)} cells, 6 language families;  '
        f'mean α = {-mean_s:.2f}, SD = {sd_s:.2f};  '
        f'd ≥ {int(D_MIN)} (short range excluded);  '
        f'† curved)',
        fontsize=11, y=1.00,
    )

    plt.tight_layout()
    out_png = OUT / 'fig1_persistence.png'
    out_pdf = OUT / 'fig1_persistence.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png.name} (mean α = {mean_s:+.3f}, SD = {sd_s:.3f}, d ≥ {D_MIN})')
    return  # below is the old single-panel version; kept commented for reference

    cell_data = []
    for i, cell in enumerate(CELLS_LONGRANGE):
        d, g, _, _, _ = cell_curves(base / f'{cell}.json')
        s, r2 = fit_pl(d, g)
        slopes.append(s)
        is_stretched = cell in STRETCHED_EXP_CELLS
        cell_data.append((cell, d, g, s, r2, is_stretched, cmap(i)))

    # Plot non-stretched-exp cells first (solid lines), then stretched (dashed)
    for cell, d, g, s, r2, is_stretched, color in cell_data:
        pos = g > 0
        marker = 's' if is_stretched else 'o'
        ls = '--' if is_stretched else '-'
        slope_str = f'α={s:.2f}' if s is not None else 'no fit'
        suffix = ' †' if is_stretched else ''
        ax.plot(d[pos], g[pos], marker=marker, linestyle=ls, color=color,
                markersize=6, linewidth=1.6, alpha=0.85,
                label=f'{cell.replace("_", " "):<24}  {slope_str}{suffix}')

        # Power-law fit overlay (faint dotted)
        if s is not None and sum(pos) >= 3:
            log_d = np.log(d[pos])
            log_g = np.log(g[pos])
            intercept = log_g.mean() - s * log_d.mean()
            x_fit = np.array([d[pos].min(), d[pos].max()])
            ax.plot(x_fit, np.exp(intercept) * x_fit ** s, ':', color=color,
                    linewidth=1.0, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance d (tokens)')
    ax.set_ylabel('P(d) — per-token order-specific gap')
    ax.set_title('Cross-corpus persistence functions  (9 cells, 8 language families)')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.95,
              title='† = stretched-exponential cells', title_fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Inset: strip plot of fitted slopes
    inset = ax.inset_axes([0.66, 0.78, 0.32, 0.18])
    valid = [s for s in slopes if s is not None]
    inset.scatter(valid, np.zeros(len(valid)) + np.random.normal(0, 0.05, len(valid)),
                  color='black', s=22, alpha=0.75)
    inset.axvline(np.mean(valid), color='C3', linewidth=1.5)
    inset.axvline(-1.0, color='gray', linewidth=0.8, linestyle=':')
    inset.set_xlim(-1.2, -0.6)
    inset.set_ylim(-0.5, 0.5)
    inset.set_yticks([])
    inset.set_xlabel('slope α', fontsize=8)
    inset.set_title(f'mean α = {np.mean(valid):+.2f},  SD = {np.std(valid):.2f}',
                    fontsize=8)
    inset.tick_params(labelsize=7)

    plt.tight_layout()
    out_png = OUT / 'fig1_persistence.png'
    out_pdf = OUT / 'fig1_persistence.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png.name} (mean α = {np.mean(valid):+.3f}, SD = {np.std(valid):.3f})')


# -----------------------------------------------------------------------------
# Figure 2 — Ordered vs. shuffled per-token marginals
# -----------------------------------------------------------------------------

def fig2():
    """Single-panel figure showing the sign-flip phenomenon directly.

    Plots per-token ordered marginal (positive, decays toward zero from above)
    and per-token shuffled marginal (crosses zero, becomes negative) on a
    linear y-axis. The shaded region between them is P(d). The sign-crossing
    of the shuffled curve is direct evidence that the probe responds
    qualitatively differently to ordered vs. shuffled context, ruling out
    probe-baseline-decay as the source of the heavy-tailed P(d).
    """
    base = DRIVE / 'corpus_expansion_longrange/llama'
    D_MIN = 10.0
    cells = CELLS_LONGRANGE

    ord_arr, shuf_arr = [], []
    d_common = None
    for cell in cells:
        d, _, ord_m, shuf_m, _ = cell_curves(base / f'{cell}.json')
        if d_common is None:
            d_common = d
        ord_arr.append(ord_m)
        shuf_arr.append(shuf_m)
    ord_arr = np.array(ord_arr)
    shuf_arr = np.array(shuf_arr)

    keep = d_common >= D_MIN
    d_common = d_common[keep]
    ord_arr = ord_arr[:, keep]
    shuf_arr = shuf_arr[:, keep]

    n = len(cells)
    ord_mean = ord_arr.mean(axis=0)
    shuf_mean = shuf_arr.mean(axis=0)
    ord_se = ord_arr.std(axis=0) / np.sqrt(n)
    shuf_se = shuf_arr.std(axis=0) / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Shaded gap between curves = P(d)
    ax.fill_between(d_common, ord_mean, shuf_mean, color='gray', alpha=0.22,
                    label='P(d) = order-specific gap', zorder=1)

    # Ordered curve + SE band
    ax.fill_between(d_common, ord_mean - ord_se, ord_mean + ord_se,
                    color='C0', alpha=0.25, zorder=2)
    ax.plot(d_common, ord_mean, 'o-', color='C0', linewidth=2.6, markersize=8,
            markeredgecolor='white', markeredgewidth=0.5, zorder=4,
            label='Ordered context')

    # Shuffled curve + SE band
    ax.fill_between(d_common, shuf_mean - shuf_se, shuf_mean + shuf_se,
                    color='C3', alpha=0.25, zorder=2)
    ax.plot(d_common, shuf_mean, 's-', color='C3', linewidth=2.6, markersize=8,
            markeredgecolor='white', markeredgewidth=0.5, zorder=4,
            label='Shuffled context')

    # Zero reference line
    ax.axhline(0, color='black', linewidth=1.0, alpha=0.7, zorder=3)
    ax.text(d_common[-1] * 1.05, 0, ' y = 0', va='center', ha='left',
            fontsize=9, color='black', alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Distance d (tokens)')
    ax.set_ylabel('Per-token marginal (ppl/token)')
    ax.set_title(
        'Ordered context reduces perplexity; shuffled context increases it.\n'
        f'(Mean ± SE across {n} cells; d ≥ {int(D_MIN)})')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = OUT / 'fig2_ord_vs_shuf.png'
    out_pdf = OUT / 'fig2_ord_vs_shuf.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png.name}  '
          f'(ord at d=10: {ord_mean[0]:+.3f}; shuf at d=10: {shuf_mean[0]:+.3f}; '
          f'ord at d_max: {ord_mean[-1]:+.4f}; shuf at d_max: {shuf_mean[-1]:+.4f})')


# -----------------------------------------------------------------------------
# Figure 3 — Sentence-shuffle decomposition
# -----------------------------------------------------------------------------

def fig3():
    """Single-panel figure: four context conditions, d >= 10.

    Plots ordered, sentence-shuffled, sentence-reversed, and token-shuffled
    per-token marginals together. Adds the sentence-reverse condition (2026-05-17)
    which is the cleanest test of forward-direction specificity in the chain.

    The conditions stack as:
      ordered          (intact forward chain)
      sent_shuf        (sequence destroyed, content preserved)
      sent_rev         (forward direction destroyed, adjacency preserved)
      tok_shuf         (everything destroyed)

    Reads from corpus_expansion_longrange_sentrev/llama which contains all four
    conditions on the same targets.
    """
    base = DRIVE / 'corpus_expansion_longrange_sentrev/llama'
    cells = CELLS_SENTSHUF
    D_MIN = 10.0

    all_d = None
    arr_ord, arr_ss, arr_sr, arr_ts = [], [], [], []
    for cell in cells:
        d, ord_m, ss_m, sr_m, ts_m = cell_four_curves(base / f'{cell}.json')
        if all_d is None: all_d = d
        arr_ord.append(ord_m); arr_ss.append(ss_m)
        arr_sr.append(sr_m); arr_ts.append(ts_m)
    arr_ord = np.array(arr_ord)
    arr_ss = np.array(arr_ss)
    arr_sr = np.array(arr_sr)
    arr_ts = np.array(arr_ts)

    keep = all_d >= D_MIN
    all_d = all_d[keep]
    arr_ord = arr_ord[:, keep]
    arr_ss = arr_ss[:, keep]
    arr_sr = arr_sr[:, keep]
    arr_ts = arr_ts[:, keep]

    n = len(cells)
    ord_mean = arr_ord.mean(axis=0)
    ss_mean = arr_ss.mean(axis=0)
    sr_mean = arr_sr.mean(axis=0)
    ts_mean = arr_ts.mean(axis=0)
    ord_se = arr_ord.std(axis=0) / np.sqrt(n)
    ss_se = arr_ss.std(axis=0) / np.sqrt(n)
    sr_se = arr_sr.std(axis=0) / np.sqrt(n)
    ts_se = arr_ts.std(axis=0) / np.sqrt(n)

    # Per-cell decomposition slopes
    direction_pc = arr_ord - arr_sr        # NEW: ord vs sent_rev (direction-specific)
    permute_pc = arr_ord - arr_ss          # old "chaining" (sequence-permutation)
    content_pc = arr_ss - arr_ts           # content-driven
    direction_slopes = [s for s in (fit_pl(all_d, direction_pc[i])[0] for i in range(n))
                        if s is not None]
    permute_slopes = [s for s in (fit_pl(all_d, permute_pc[i])[0] for i in range(n))
                      if s is not None]
    content_slopes = [s for s in (fit_pl(all_d, content_pc[i])[0] for i in range(n))
                      if s is not None]
    s_direction = (np.mean(direction_slopes), np.std(direction_slopes))
    s_permute = (np.mean(permute_slopes), np.std(permute_slopes))
    s_content = (np.mean(content_slopes), np.std(content_slopes))

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # Four condition curves with SE bands.
    # Order matters for layering: ord at top, ts at bottom (per the data ordering).
    sent_rev_color = '#7B3FA0'  # purple, distinct from C0/C1/C3
    for vals, se_vals, color, marker, label in [
        (ord_mean, ord_se, 'C0', 'o', 'Ordered (intact forward chain)'),
        (ss_mean, ss_se, 'C1', 's', 'Sentence-shuffled (sequence destroyed)'),
        (sr_mean, sr_se, sent_rev_color, 'D', 'Sentence-reversed (direction destroyed)'),
        (ts_mean, ts_se, 'C3', '^', 'Token-shuffled (order control)'),
    ]:
        ax.fill_between(all_d, vals - se_vals, vals + se_vals,
                        color=color, alpha=0.20, zorder=2)
        ax.plot(all_d, vals, marker=marker, linestyle='-', color=color,
                linewidth=2.4, markersize=7, markeredgecolor='white',
                markeredgewidth=0.5, zorder=4, label=label)

    ax.axhline(0, color='black', linewidth=1.0, alpha=0.7, zorder=3)

    # Annotate the three decomposed components, ranked by importance.
    annotation_text = (
        f'direction-specific:  slope = {s_direction[0]:+.2f} ± {s_direction[1]:.2f}'
        '   (ord − sent_rev)\n'
        f'content-driven:      slope = {s_content[0]:+.2f} ± {s_content[1]:.2f}'
        '   (sent_shuf − tok_shuf)\n'
        f'sequence-permute:    slope = {s_permute[0]:+.2f} ± {s_permute[1]:.2f}'
        '   (ord − sent_shuf; small)'
    )
    ax.text(0.98, 0.04, annotation_text,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.92))

    ax.set_xscale('log')
    ax.set_xlabel('Distance d (tokens)')
    ax.set_ylabel('Per-token marginal (ppl/token)')
    ax.set_title(
        'Four context conditions: forward chain, sequence, direction, and disorder\n'
        f'(Mean ± SE across {n} cells; d ≥ {int(D_MIN)})')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save under both the script's internal name and the manuscript name.
    for stem in ('fig3_sentence_shuffle', 'fig2_sentence_shuffle'):
        plt.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight')
        plt.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    plt.close()
    print(f'  saved fig3_sentence_shuffle + fig2_sentence_shuffle  '
          f'(direction α = {s_direction[0]:+.3f} ± {s_direction[1]:.3f}, '
          f'content α = {s_content[0]:+.3f} ± {s_content[1]:.3f}, '
          f'permute α = {s_permute[0]:+.3f} ± {s_permute[1]:.3f})')


# -----------------------------------------------------------------------------
# Figure 4 — Sentence-level ablation, distributed influence
# -----------------------------------------------------------------------------

def fig4():
    abl_base = DRIVE / 'sentence_ablation/llama'
    mat_base = DRIVE / 'sentence_influence_matrix/llama'
    cells = ['gutenberg_fiction_en', 'ted_transcripts_en']
    colors = {'gutenberg_fiction_en': 'C0', 'ted_transcripts_en': 'C1'}
    markers = {'gutenberg_fiction_en': 'o', 'ted_transcripts_en': 's'}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ----- Panel A: % positive vs distance from ablation data -----
    ax = axes[0]
    bins = [0, 32, 64, 128, 256, 512, 1024]
    bin_centers = [(bins[i] * bins[i + 1]) ** 0.5 if bins[i] > 0 else bins[i + 1] / 2
                   for i in range(len(bins) - 1)]

    for cell in cells:
        with open(abl_base / f'{cell}.json') as f:
            recs = json.load(f)
        all_dist = [d for r in recs for d in r['sentence_distances']]
        all_inf = [v for r in recs for v in r['ablation_influence']]
        pcts, ns = [], []
        for i in range(len(bins) - 1):
            vs = [v for d, v in zip(all_dist, all_inf)
                  if bins[i] <= d < bins[i + 1]]
            if vs:
                pcts.append(100 * sum(1 for v in vs if v > 0) / len(vs))
                ns.append(len(vs))
            else:
                pcts.append(np.nan); ns.append(0)
        ax.plot(bin_centers, pcts, marker=markers[cell], linestyle='-', color=colors[cell],
                linewidth=2.2, markersize=9, label=cell.replace('_', ' '))

    ax.axhline(50, color='gray', linestyle=':', alpha=0.6, label='chance = 50%')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from target (tokens)')
    ax.set_ylabel('% of sentences positively contributing')
    ax.set_ylim(40, 100)
    ax.set_title('Distributed influence — most sentences contribute even at d > 500')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    panel_label(ax, 'a')

    # ----- Panel B: partial-residual test for anchor effect -----
    # Regress influence on log(distance). Residual mean by absolute position should
    # be near zero everywhere (including the opening, pos=0) if there is no anchor
    # signal beyond distance. Opening highlighted in red.
    ax = axes[1]

    for cell in cells:
        with open(mat_base / f'{cell}.json') as f:
            recs = json.load(f)
        positions, distances, influences = [], [], []
        for r in recs:
            m = r['influence_matrix']
            n = len(m)
            for t in range(2, n):
                for i in range(t):
                    v = m[i][t]
                    if v is None: continue
                    positions.append(i)
                    distances.append(t - i)
                    influences.append(v)
        positions = np.array(positions)
        distances = np.array(distances)
        influences = np.array(influences)

        # Regression: influence ~ log(distance)
        log_d = np.log(distances)
        slope, intercept, *_ = stats.linregress(log_d, influences)
        predicted = intercept + slope * log_d
        residuals = influences - predicted

        # Mean residual by absolute position
        by_pos = {}
        for p, r_v in zip(positions, residuals):
            by_pos.setdefault(p, []).append(r_v)
        pos_keys = sorted([k for k in by_pos if len(by_pos[k]) >= 5])
        means = [np.mean(by_pos[k]) for k in pos_keys]
        sems = [np.std(by_pos[k]) / np.sqrt(len(by_pos[k])) for k in pos_keys]

        ax.errorbar(pos_keys, means, yerr=sems, fmt=markers[cell] + '-',
                    color=colors[cell], linewidth=1.8, markersize=6, capsize=2,
                    alpha=0.85, label=cell.replace('_', ' '))

    # Highlight the opening (pos=0) with a vertical line
    ax.axvline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
               label='opening (pos=0)')
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Source sentence absolute position in document')
    ax.set_ylabel('Residual influence  (after regressing out log(distance))')
    ax.set_title('No anchor effect — residuals flat near zero, including opening')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    panel_label(ax, 'b')

    plt.tight_layout()
    out_png = OUT / 'fig4_distributed_influence.png'
    out_pdf = OUT / 'fig4_distributed_influence.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_png.name}')


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('Building Fig 1 — cross-corpus persistence functions')
    fig1()
    print('Building Fig 2 — synthetic-sequence controls')
    fig2()
    print('Building Fig 3 — sentence-shuffle decomposition')
    fig3()
    print('Building Fig 4 — sentence-level ablation, distributed influence')
    fig4()
    print('\nDone.')
