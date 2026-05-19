"""Fig 1 small-multiples (Option B) trimmed to d >= 5 to drop the
shuffled-baseline collapse point at d ~ 2.8.
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
D_MIN = 10.0  # trim threshold — first interval where both ordered AND shuffled bases ≥ 8


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


# Load all 9 cells, trim to d >= D_MIN
data = []
for cell in CELLS:
    d, g = cell_curve(DRIVE / 'corpus_expansion_longrange/llama' / f'{cell}.json')
    keep = d >= D_MIN
    d, g = d[keep], g[keep]
    s = fit_pl(d, g)
    data.append((cell, d, g, s))

slopes = [s for _, _, _, s in data if s is not None]
mean_s, sd_s = np.mean(slopes), np.std(slopes)

# 3x3 small multiples, trimmed
fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
cmap = plt.get_cmap('tab10')

for idx, (cell, d, g, s) in enumerate(data):
    ax = axes[idx // 3][idx % 3]
    pos = g > 0
    is_stretch = cell in STRETCHED
    color = cmap(idx)

    ax.plot(d[pos], g[pos], 'o-', color=color, linewidth=1.6, markersize=5,
            alpha=0.9)

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

for ax in axes[-1]:
    ax.set_xlabel('Distance d (tokens)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(d)')

fig.suptitle(
    f'Cross-corpus persistence functions  (9 cells, 5 language families;  '
    f'mean α = {mean_s:+.2f}, SD = {sd_s:.2f};  d ≥ {D_MIN:.0f};  † stretched-exp)',
    fontsize=11, y=1.00)

plt.tight_layout()
plt.savefig(OUT / 'fig1_optB_trimmed.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT / 'fig1_optB_trimmed.pdf', bbox_inches='tight')
plt.close()
print(f'saved fig1_optB_trimmed: mean α = {mean_s:+.3f}, SD = {sd_s:.3f}, d ≥ {D_MIN}')
