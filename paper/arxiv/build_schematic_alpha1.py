#!/usr/bin/env python3
"""Conceptual three-regime schematic for the alpha ~ 1 boundary (NON-data figure).

Three columns: alpha>1, alpha=1, alpha<1.
 Top row:    power-law decay P(d) ~ d^-alpha on log-log axes (slopes ~ -1.4, -1, -0.6),
             dashed decade guides at 10 and 100.
 Bottom row: three equal-width logarithmic distance bands (1-10, 10-100, 100-1000)
             whose heights = aggregate influence per decade ~ d P(d) ~ d^(1-alpha):
             decreasing / flat / increasing. Bands are conceptual, not empirical.
Middle panel is visually emphasized.
Saved as schematic_alpha1.pdf for inclusion in main_nhb.tex.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif",
    "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
    "mathtext.fontset": "cm",
})

GREY   = "#6b6b6b"      # side-panel ink
ACC    = "#08519c"      # emphasized middle-panel ink
TINT   = "#eef4fb"      # faint middle-column background
BARG   = "#d9d9d9"      # side-panel bar fill
BARB   = "#bdd7ec"      # middle-panel bar fill

regimes = [
    dict(alpha=1.4, title=r"$\alpha>1$",  ink=GREY, barfill=BARG, tint="white",
         label="Recent context\ndominates"),
    dict(alpha=1.0, title=r"$\alpha=1$",  ink=ACC,  barfill=BARB, tint=TINT,
         label="Equal aggregate influence\nper logarithmic decade"),
    dict(alpha=0.6, title=r"$\alpha<1$",  ink=GREY, barfill=BARG, tint="white",
         label="Distant context\ndominates"),
]

fig = plt.figure(figsize=(7.4, 4.1))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0],
                      hspace=0.55, wspace=0.32,
                      left=0.15, right=0.985, top=0.80, bottom=0.13)

d = np.logspace(0, 3, 200)
band_labels = ["1–10", "10–100", "100–1000"]
band_d = np.array([np.sqrt(1*10), np.sqrt(10*100), np.sqrt(100*1000)])  # geo-mean per decade

for j, r in enumerate(regimes):
    ink, a = r["ink"], r["alpha"]

    # ---- top: power-law curve on log-log
    axT = fig.add_subplot(gs[0, j])
    axT.set_facecolor(r["tint"])
    P = d ** (-a)
    axT.loglog(d, P, color=ink, lw=2.0 if j == 1 else 1.5, solid_capstyle="round")
    for x in (10, 100):
        axT.axvline(x, color="#c7c7c7", lw=0.6, ls=(0, (3, 3)), zorder=0)
    axT.set_xlim(1, 1000); axT.set_ylim(4e-5, 1.6)
    axT.set_xticks([1, 10, 100, 1000]); axT.set_xticklabels(["1", "10", "100", "1000"])
    axT.set_yticks([])
    axT.set_title(r["title"], fontsize=12, color=ink,
                  fontweight="bold" if j == 1 else "normal", pad=5)
    axT.tick_params(labelsize=7)
    axT.set_xlabel(r"distance $d$", fontsize=8, labelpad=1)
    axT.text(0.94, 0.90, rf"$\alpha={a:.1f}$", transform=axT.transAxes,
             ha="right", va="top", fontsize=8.5, color=ink)
    for s in ("top", "right"):
        axT.spines[s].set_visible(False)

    # ---- bottom: aggregate influence per decade  ~  d^(1-alpha)
    axB = fig.add_subplot(gs[1, j])
    axB.set_facecolor(r["tint"])
    h = band_d ** (1 - a)
    h = h / h.max()                          # max-normalize per panel (conceptual)
    for i, hi in enumerate(h):
        axB.add_patch(Rectangle((i - 0.4, 0), 0.8, hi,
                                facecolor=r["barfill"], edgecolor=ink,
                                lw=1.2 if j == 1 else 0.9))
    axB.set_xlim(-0.65, 2.65); axB.set_ylim(0, 1.18)
    axB.set_xticks(range(3)); axB.set_xticklabels(band_labels, fontsize=7)
    axB.set_yticks([])
    axB.set_xlabel("distance decade (tokens)", fontsize=8, labelpad=2)
    axB.text(0.5, -0.52, r["label"], transform=axB.transAxes, ha="center", va="top",
             fontsize=8.2, color=ink, fontweight="bold" if j == 1 else "normal")
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)

# ---- left-hand row labels: observation -> interpretation
LBL = "#2a2a2a"
fig.text(0.030, 0.655, "Power-law\ndecay", rotation=90, ha="center", va="center",
         fontsize=9, color=LBL, fontweight="bold", linespacing=0.95)
fig.text(0.066, 0.655, "observation", rotation=90, ha="center", va="center",
         fontsize=7, color="#9a9a9a", style="italic")
fig.text(0.030, 0.250, "Influence\nper decade", rotation=90, ha="center", va="center",
         fontsize=9, color=LBL, fontweight="bold", linespacing=0.95)
fig.text(0.066, 0.250, "interpretation", rotation=90, ha="center", va="center",
         fontsize=7, color="#9a9a9a", style="italic")
# downward arrow linking the two rows (exponent -> organization of influence)
fig.patches.append(matplotlib.patches.FancyArrowPatch(
    (0.048, 0.475), (0.048, 0.405), transform=fig.transFigure,
    arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#888888"))

# ---- equation (figure-level headline removed; belongs in the caption)
fig.text(0.5, 0.885,
         r"aggregate influence within a logarithmic decade $\;\propto\; d\,P(d)\propto d^{\,1-\alpha}$",
         ha="center", fontsize=9, color="#222222")
fig.savefig("schematic_alpha1.pdf", bbox_inches="tight", dpi=300)
print("wrote schematic_alpha1.pdf")
