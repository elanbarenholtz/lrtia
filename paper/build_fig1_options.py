"""Generate two clean alternatives for Fig 1.

Option A — aggregate ribbon style (single panel, mean ± SE)
Option B — 3x3 small-multiples grid (one panel per cell)
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DRIVE = Path('/Users/elansmini/Library/CloudStorage/'
             'GoogleDrive-mpcrlab@gmail.com/My Drive/LRTIA/Results')
OUT = Path('/Users/elansmini/lrtia/paper/Paper_Figures')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'figure.dpi': 100,
})

CELLS = [
    'gutenberg_fiction_en', 'ted_transcripts_en', 'ted_transcripts_de',
    'ted_transcripts_fr', 'ted_transcripts_tr', 'literary_ja',
    'literary_fi', 'news_en', 'buckeye',
]
STRETCHED = {'literary_ja', 'buckeye'}


def per_target(rec):
    cs = rec['context_lengths']
    op = rec['ordered_ppl']; sp = rec['shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i-1] == 0: continue
        w = cs[i] - cs[i-1]
        d = math.sqrt(cs[i-1] * cs[i])
        gap = (op[i-1]-op[i])/w - (sp[i-1]-sp[i])/w
        out.append((d, gap))
    return out


def cell_curve(path):
    recs = json.load(open(path))
    by_i = {}
    for r in recs:
        for i, t in enumerate(per_target(r)):
            by_i.setdefault(i, []).append(t)
    ds, gs = [], []
    for i in sorted(by_i):
        rows = by_i[i]
        ds.append(rows[0][0])
        gs.append(np.mean([t[1] for t in rows]))
    return np.array(ds), np.array(gs)


def fit_pl(d, y):
    pos = y > 0
    if sum(pos) < 4: return None
    s, _, _, _, _ = stats.linregress(np.log(d[pos]), np.log(y[pos]))
    return s


# Load all 9 cells
base = DRIVE / 'corpus_expansion_longrange/llama'
data = []
for cell in CELLS:
    d, g = cell_curve(base / f'{cell}.json')
    s = fit_pl(d, g)
    data.append((cell, d, g, s))

# Common distance grid (all cells should have same log-spaced contexts)
d_common = data[0][1]


# =============================================================================
# Option A — aggregate ribbon
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Stack cells into matrix for mean/SE — only positive-everywhere subset
all_g = np.array([g for _, _, g, _ in data])

cmap = plt.get_cmap('tab10')
for i, (cell, d, g, s) in enumerate(data):
    pos = g > 0
    ls = '--' if cell in STRETCHED else '-'
    ax.plot(d[pos], g[pos], ls, color=cmap(i), linewidth=1.0, alpha=0.4,
            label=cell.replace('_', ' '))

# Aggregate: geometric mean of positive values per distance bin
log_g = np.log(np.where(all_g > 0, all_g, np.nan))
g_geo = np.exp(np.nanmean(log_g, axis=0))
g_se_log = np.nanstd(log_g, axis=0) / np.sqrt(np.sum(~np.isnan(log_g), axis=0))
g_lo = np.exp(np.log(g_geo) - g_se_log)
g_hi = np.exp(np.log(g_geo) + g_se_log)

valid = np.isfinite(g_geo) & (g_geo > 0)
ax.fill_between(d_common[valid], g_lo[valid], g_hi[valid],
                color='black', alpha=0.18, label='mean ± SE')
ax.plot(d_common[valid], g_geo[valid], '-', color='black', linewidth=2.5,
        label='aggregate mean')

# Single aggregate power-law fit
s_agg = fit_pl(d_common, g_geo)
slopes = [s for _, _, _, s in data if s is not None]
mean_s, sd_s = np.mean(slopes), np.std(slopes)

if s_agg is not None:
    log_d = np.log(d_common[valid])
    log_y = np.log(g_geo[valid])
    intercept = log_y.mean() - s_agg * log_d.mean()
    x_fit = np.array([d_common[valid].min(), d_common[valid].max()])
    ax.plot(x_fit, np.exp(intercept) * x_fit ** s_agg, ':',
            color='C3', linewidth=2.0, alpha=0.9,
            label=f'power-law fit: α = {s_agg:.2f}')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Distance d (tokens)')
ax.set_ylabel('P(d) — per-token order-specific gap')
ax.set_title('Cross-corpus persistence functions  (9 cells, 5 language families)')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=8, loc='lower left', framealpha=0.95, ncol=2)

# Inset: slope strip plot
inset = ax.inset_axes([0.62, 0.78, 0.34, 0.18])
inset.scatter(slopes, np.zeros(len(slopes)) + np.random.normal(0, 0.05, len(slopes)),
              color='black', s=22, alpha=0.75)
inset.axvline(mean_s, color='C3', linewidth=1.5)
inset.set_xlim(-1.2, -0.6); inset.set_ylim(-0.5, 0.5); inset.set_yticks([])
inset.set_xlabel('per-cell α', fontsize=8)
inset.set_title(f'mean α = {mean_s:+.2f}, SD = {sd_s:.2f}', fontsize=8)
inset.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT / 'fig1_optA_aggregate.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'fig1_optA_aggregate.pdf', bbox_inches='tight')
plt.close()
print(f'Option A saved: agg α = {s_agg:.3f}, mean per-cell α = {mean_s:+.3f}, SD = {sd_s:.3f}')


# =============================================================================
# Option B — 3x3 small multiples
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)

cmap = plt.get_cmap('tab10')
for idx, (cell, d, g, s) in enumerate(data):
    ax = axes[idx // 3][idx % 3]
    pos = g > 0
    is_stretch = cell in STRETCHED
    color = cmap(idx)

    ax.plot(d[pos], g[pos], 'o-', color=color, linewidth=1.6, markersize=5,
            alpha=0.9)

    # Per-cell power-law fit
    if s is not None and sum(pos) >= 3:
        log_d = np.log(d[pos]); log_g = np.log(g[pos])
        intercept = log_g.mean() - s * log_d.mean()
        x_fit = np.array([d[pos].min(), d[pos].max()])
        ax.plot(x_fit, np.exp(intercept) * x_fit ** s, ':', color='black',
                linewidth=1.2, alpha=0.7)

    title = cell.replace('_', ' ')
    if is_stretch: title += ' †'
    ax.set_title(title, fontsize=10)

    slope_str = f'α = {s:.2f}' if s is not None else 'no fit'
    ax.text(0.05, 0.05, slope_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontweight='bold')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

# Shared axis labels
for ax in axes[-1]:
    ax.set_xlabel('Distance d (tokens)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(d)')

slopes = [s for _, _, _, s in data if s is not None]
mean_s, sd_s = np.mean(slopes), np.std(slopes)
fig.suptitle(f'Cross-corpus persistence functions  (9 cells, 5 language families;  '
             f'mean α = {mean_s:+.2f}, SD = {sd_s:.2f};  † stretched-exp)',
             fontsize=11, y=1.00)

plt.tight_layout()
plt.savefig(OUT / 'fig1_optB_smallmultiples.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'fig1_optB_smallmultiples.pdf', bbox_inches='tight')
plt.close()
print(f'Option B saved: mean per-cell α = {mean_s:+.3f}, SD = {sd_s:.3f}')
