# Figure spec — LRTIA Nature submission (v2)

Four main-text figures. All log-log axes unless noted. Distance d in tokens. P(d) = order-specific persistence function (per-token: ordered marginal − shuffled marginal at distance d).

---

## Figure 1 — Cross-corpus persistence functions

**Data source:** `Corpus_Expansion_LongRange_Llama`
**Cells (9):** gutenberg_en, ted_en, ted_de, ted_fr, ted_tr, literary_ja, literary_fi, news_en, buckeye
**Contexts:** log-spaced [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
**Y-quantity:** per-token order-specific gap (P(d) per added token)

**Layout:** single main panel. Optional small inset top-right.

**Main panel:**
- 9 lines, one per cell, distinct colors
- Power-law fits overlaid as dashed lines (one per cell)
- Mark literary_ja and buckeye with a different line/marker style (these are the two stretched-exponential cells)
- X: distance d (log scale), range covering ~1 to ~1000
- Y: P(d) per token (log scale)
- Legend listing the 9 cells

**Inset (top-right, optional):**
- Strip plot of the 9 fitted power-law slopes
- Annotate mean = −0.94, SD = 0.12

**Headline result the figure must show:** all 9 lines visibly heavy-tailed across more than two decades, with similar slopes clustering near −1.

---

## Figure 2 — Synthetic-sequence controls

**Data source:** `random_vocab_uniform` runs on both probes (Llama-3.1-8B, Mistral-7B), plus matched natural-language cells
**Y-quantities:** raw perplexity reduction (Panel A); P(d) (Panel B)

**Layout:** two panels side by side (a, b).

**Panel A — Raw perplexity reduction:**
- Two lines (or two pairs, one per probe): synthetic random-vocab vs. natural language
- Both should show heavy-tailed decay with comparable slopes (synthetic Llama ≈ −1.27; synthetic Mistral ≈ −1.34)
- Annotate the slopes inline
- X: distance d (log); Y: raw per-token perplexity reduction (log)

**Panel B — Order-specific P(d):**
- Two lines: synthetic random-vocab P(d) (should sit near zero across all distances) and natural-language P(d) (heavy-tailed)
- X: distance d (log); Y: P(d) per token (log; or linear with zero baseline if zero-line readability matters)

**Headline result the figure must show:** Panel A — probes give heavy-tailed perplexity reduction even on sequences with no real structure. Panel B — the shuffled-token subtraction absorbs that probe baseline, leaving genuine sequential dependence visible only on natural language.

---

## Figure 3 — Sentence-shuffle decomposition

**Data source:** `Corpus_Expansion_LongRange_SentShuffle_Llama`
**Cells:** 8 cells from the long-range run, excluding buckeye (no clean sentence boundaries in spontaneous speech)
**Conditions per target:** ordered, sentence-shuffled, token-shuffled

**Layout:** two panels side by side (a, b).

**Panel A — Three context conditions:**
- Three lines: P(d) under ordered context, sentence-shuffled context, token-shuffled context (token-shuffled should sit near zero)
- Either one representative cell (suggest gutenberg_en) or aggregate across the 8 cells with shaded ±SD bands
- X: distance d (log); Y: P(d) (log; ordered + sentence-shuffled positive, token-shuffled near zero)

**Panel B — Decomposition:**
- Two lines:
  - Chaining-specific component: ordered − sentence-shuffled (slope ≈ −1.32)
  - Content-driven component: sentence-shuffled − token-shuffled (slope ≈ −0.82)
- Annotate each slope inline
- Mark d > 100 region where chaining-specific approaches zero
- X: distance d (log); Y: component magnitude (log)

**Headline result the figure must show:** chaining-specific and content-driven components are distinguishable. Chaining decays faster and dies by d > 100; content carries the long tail.

---

## Figure 4 — Sentence-level ablation, distributed influence

**Data source:**
- Panel A: `Sentence_Ablation_Density_Llama` on gutenberg_en + ted_en (20 docs/cell, 1024-token prior context)
- Panel B: `Sentence_Influence_Matrix_Adult_Llama` (15 docs/cell, full leave-one-sentence-out matrix per doc)

**Layout:** two panels side by side (a, b).

**Panel A — Proportion of contributing sentences vs distance:**
- Distance-binned proportion of prior sentences whose removal increases target perplexity
- Two lines (gutenberg_en, ted_en) or aggregated with shaded band
- Should decline smoothly from ~90% at d < 32 to ~55–60% at d > 512
- Annotate "no sharp transition" or similar in caption
- X: distance bin (log or linear), range from ~1 to ~1024; Y: proportion (linear, 0–1)

**Panel B — Position vs distance:**
- Two lines:
  - Mean per-sentence influence by **absolute position** in the prior context
  - Mean per-sentence influence by **distance to target**
- Both should track each other closely (position effect fully explained by distance — no residual anchor signal)
- Alternative implementation: paired-bar or partial-residual plot showing position effect ≈ 0 after distance is regressed out
- X: position index / distance bin; Y: mean influence

**Headline result the figure must show:** Panel A — contribution density declines smoothly with distance, no cliff. Panel B — there is no privileged structural position; what looks like a position effect is fully explained by recency.

---

## Style notes (apply to all four)

- All P(d) plots in log-log unless a zero-baseline reading is essential
- Use a consistent color palette across the four figures where possible
- Slope annotations should be visible inside the panel, not buried in legends
- Each panel should have a one-letter label (a, b) in the upper-left corner
- Captions are drafted in `manuscript_nature_v2.md` (look under "Figure plan") — dev does not need to write these

---

## Optional supplementary figures (lower priority)

- **SI Fig — functional-form comparison.** Power law vs. exponential vs. stretched-exp vs. quadratic-in-log fits per cell, with ΔAIC values. Existing `fig2_functional_form.png` is the right format but uses old 1–100 token range data; needs regenerating on the long-range run.
- **SI Fig — cross-language sign-flip replication.** Bar chart or curve set showing the long-range sign of P(d) (positive at d ≥ 10) across the broader 8-family cross-language sample including Slavic, Afro-Asiatic, Koreanic. Existing `fig3_crosslingual_convergence.png` is a bar-chart format reference; data needs regenerating around the sign-flip metric, not the alpha-clustering metric.
- **SI Fig — per-speaker variability in Buckeye.** Existing `fig4_per_speaker.png` is usable as-is.
