#!/usr/bin/env python3
"""Conceptual 'ladder' schematic: the hierarchy of statistical structure in language.
Three rungs of increasing structural specificity, each obtained by removing the contribution
of the level below. Bottom: frequency (Zipf). Middle: co-occurrence (mutual information).
Top: order-specific dependence (the contextual persistence law, this work)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif"})
ACC = "#08519c"; GREY = "#4d4d4d"; GBOX = "#f0f0f0"; BBOX = "#e2edf7"

fig, ax = plt.subplots(figsize=(7.6, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")

X, W, H = 0.4, 8.2, 2.45          # box geometry (spans x = 0.4 .. 8.6, inside xlim)
rungs = [
    (0.5, "1.  Frequency", "which units occur, and how often",
     r"Zipf's law:  $f\propto r^{-1}$", False),
    (3.6, "2.  Co-occurrence", "which units co-occur across distance",
     r"mutual information:  $I(d)\propto d^{-\beta}$", False),
    (6.7, "3.  Order", "in what arrangement (order-specific)",
     r"persistence law:  $P(d)\propto d^{-\alpha},\ \alpha\approx1$", True),
]
for y, level, q, ex, cpf in rungs:
    fc, ec = (BBOX, ACC) if cpf else (GBOX, "#bdbdbd")
    ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.8 if cpf else 1.0, zorder=2))
    tcol = ACC if cpf else GREY
    ax.text(X + 0.35, y + H - 0.55, level, fontsize=12.5, fontweight="bold", color=tcol, zorder=3)
    ax.text(X + 0.4, y + H - 1.18, q, fontsize=9, color="#333333", style="italic", zorder=3)
    ax.text(X + 0.4, y + 0.42, ex, fontsize=9.5, color=tcol, zorder=3)
    if cpf:
        ax.text(X + W - 0.3, y + H - 0.5, "this work", fontsize=8.5, color=ACC,
                ha="right", fontweight="bold", zorder=3)

# upward arrows between rungs, labelling the operator that peels a level off
for y0, lab in [(3.05, "remove frequency"),
                (6.15, "remove co-occurrence  (shuffle isolates order)")]:
    ax.add_patch(FancyArrowPatch((2.2, y0), (2.2, y0 + 0.55), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.5, color="#8a8a8a", zorder=1))
    ax.text(2.5, y0 + 0.27, lab, fontsize=7.8, color="#777777", va="center", zorder=3)

# vertical "increasing structural specificity" guide on the right
ax.add_patch(FancyArrowPatch((9.15, 0.6), (9.15, 9.15), arrowstyle="-|>", mutation_scale=13,
                             lw=1.1, color="#aaaaaa"))
ax.text(9.5, 4.9, "increasing structural specificity", rotation=90, va="center", ha="center",
        fontsize=8, color="#999999")

ax.text(X, 10.5, "A hierarchy of statistical structure in language",
        fontsize=12, fontweight="bold", color="#1a1a1a")
ax.text(X, 9.95, "each rung is scale-free and removes the contribution of the rung below",
        fontsize=8.8, color="#666666")

fig.savefig("figures/hierarchy_ladder.pdf", bbox_inches="tight", dpi=300)
print("wrote figures/hierarchy_ladder.pdf")
