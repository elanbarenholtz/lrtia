#!/usr/bin/env python3
"""'Language vs biology' contrast, built from the REAL cached CPF outputs.
Three log-log panels of the order-specific persistence P(d): human language (aggregate of
ten corpora), DNA (HyenaDNA), protein (ProGen2 at two scales). The discriminator is
scale-freeness and exponent stability, not presence vs absence of long-range structure:
protein is shown honestly as positive-but-scattered with sign-flips, and its exponent
instability across model scale is annotated. Non-positive P(d) are marked at the floor.

Reads data/derived/ (relative to the capsule root). Saves figures/lang_vs_bio.pdf.
"""
import json, math, os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent
DER = Path(os.environ.get("CPF_DATA",
      ROOT.parents[1] / "NHB_submission" / "data" / "derived"))
if not DER.exists():   # fallback: derived synced next to this script's project
    DER = ROOT.parents[1] / "lrtia" / "NHB_submission" / "data" / "derived"

def marg(recs):
    by = {}
    for r in recs:
        cs, op, sp = r["context_lengths"], r["ordered_ppl"], r["shuffled_ppl"]
        for i in range(1, len(cs)):
            if cs[i-1] == 0: continue
            w = cs[i]-cs[i-1]; d = math.sqrt(cs[i-1]*cs[i])
            by.setdefault(i, (d, []))[1].append((op[i-1]-op[i])/w - (sp[i-1]-sp[i])/w)
    ds = np.array([by[i][0] for i in sorted(by)])
    P  = np.array([np.mean(by[i][1]) for i in sorted(by)])
    return ds, P

def fit(d, P):
    m = (d >= 10) & (P > 0)
    if m.sum() < 4: return None, None
    s, _, r, _, _ = stats.linregress(np.log(d[m]), np.log(P[m]))
    return -s, r**2

def load(*names):
    for n in names:
        p = DER / n
        if p.exists(): return json.load(open(p))
    raise FileNotFoundError(names)

CELLS = ['gutenberg_fiction_en','news_en','buckeye','ted_transcripts_en','ted_transcripts_de',
         'ted_transcripts_fr','ted_transcripts_tr','ted_transcripts_ru','literary_ja','literary_fi']
lang = []
for c in CELLS:
    lang += json.load(open(DER / "corpus_expansion_longrange" / "llama" / f"{c}.json"))
dL, PL = marg(lang);                aL, rL = fit(dL, PL)
dD, PD = marg(load("domain_controls/dna_hyenadna.json")); aD, rD = fit(dD, PD)
dS, PS = marg(load("domain_controls/protein_progen2-small_dense.json")); aS, rS = fit(dS, PS)
dB, PB = marg(load("domain_controls/protein_progen2-large_dense.json")); aB, rB = fit(dB, PB)

plt.rcParams.update({"font.size": 8.5, "font.family": "serif",
                     "axes.linewidth": 0.7, "axes.edgecolor": "#444"})
LANG = "#08519c"; G1 = "#525252"; G2 = "#969696"
FLOOR = 1.5e-4
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.30})

def amp(d, P, alpha):
    m = (d >= 10) & (P > 0)
    return np.exp(np.mean(np.log(P[m]) + alpha*np.log(d[m])))

def plot_pts(a, d, P, color, marker, label=None, line=None, lstyle="-", lw=1.4):
    m = (d >= 10)
    pos = m & (P > 0); neg = m & (P <= 0)
    a.loglog(d[pos], P[pos], marker, ms=4.5, color=color, mec="white", mew=0.5, label=label, ls="none")
    # non-positive bins pinned at the floor to show sign-flips honestly
    a.loglog(d[neg], np.full(neg.sum(), FLOOR), "x", ms=4, color=color, mew=0.9, alpha=0.7, ls="none")
    if line is not None:
        xf = np.array([10, 1000]); a.loglog(xf, line[0]*xf**(-line[1]), lstyle, color=color, lw=lw)

# Language (solid fit line = clean scale-free law)
plot_pts(ax[0], dL, PL, LANG, "o", line=(amp(dL, PL, aL), aL), lstyle="-", lw=1.6)
ax[0].set_title("Human language", color=LANG, fontsize=10, fontweight="bold")
ax[0].text(0.05, 0.05, r"$\alpha=1.04$" "\n" r"median $r^2=0.96$", transform=ax[0].transAxes,
           fontsize=8, va="bottom")

# DNA (dashed fit = poor fit; note tiny magnitude + non-monotonic)
plot_pts(ax[1], dD, PD, G1, "s", line=(amp(dD, PD, aD), aD), lstyle="--", lw=1.1)
ax[1].set_title("DNA (HyenaDNA)", color=G1, fontsize=10, fontweight="bold")
ax[1].text(0.05, 0.05, r"$\sim\!100\times$ weaker," "\n" "non-monotonic\n" r"($r^2=0.64$)",
           transform=ax[1].transAxes, fontsize=8, va="bottom")

# Protein, two scales (dashed fits diverge = unstable exponent)
plot_pts(ax[2], dS, PS, G1, "^", label="151M", line=(amp(dS, PS, aS), aS), lstyle="--", lw=1.0)
plot_pts(ax[2], dB, PB, G2, "v", label="2.7B", line=(amp(dB, PB, aB), aB), lstyle="--", lw=1.0)
ax[2].set_title("Protein (ProGen2)", color=G1, fontsize=10, fontweight="bold")
ax[2].text(0.05, 0.05, r"$\alpha:1.07\!\to\!0.74$" "\n" r"$r^2:0.62\!\to\!0.36$" "\n" "not scale-free",
           transform=ax[2].transAxes, fontsize=8, va="bottom")
ax[2].legend(fontsize=6.5, loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

for a in ax:
    a.set_xlim(9, 1100); a.set_ylim(1e-4, 3)
    a.set_xlabel(r"distance $d$ (tokens)")
    a.set_xticks([10, 100, 1000]); a.set_xticklabels(["10", "100", "1000"])
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel(r"persistence $P(d)$")
# Figure-level superheading/subtitle removed (journal style: content belongs in the caption).
fig.savefig(ROOT / "figures" / "lang_vs_bio.pdf", bbox_inches="tight", dpi=300)
print(f"LANG a={aL:.3f} r2={rL:.3f} | DNA a={aD:.3f} r2={rD:.3f} | "
      f"protS a={aS:.3f} r2={rS:.3f} | protL a={aB:.3f} r2={rB:.3f}")
print("wrote figures/lang_vs_bio.pdf")
