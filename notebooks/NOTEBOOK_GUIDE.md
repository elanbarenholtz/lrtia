# LRTIA Notebook Guide: Measuring Long-Range Coherence in Text

## Overview

This series of notebooks develops and validates a method for measuring how far back in a text sequence prior content influences current production. The journey goes through several iterations, each addressing limitations discovered in the previous version.

**Core question:** Does human text contain long-range coherence structure that reflects memory constraints on language production?

**Dataset:** RAID benchmark — 8 genres (abstracts, books, news, poetry, recipes, reddit, reviews, wiki) × 6 sources (human + 5 AI models), ~1400 essays.

**Measuring model:** Mistral-7B-v0.1 (4-bit quantized), used as a probe to detect structure in the text — not the object of study itself.

---

## The Notebook Progression

### v1: `RAID_Multi_Genre_Analysis_v1.ipynb`
**Purpose:** Original analysis using perplexity-based memory curves.

**Method:** Context ablation — progressively truncate context before a target region (last 64 tokens). Measure perplexity at each context window size (4, 8, 12, 16, 24, 32, 48, 64, 96, 128 tokens). The "memory curve" shows how perplexity improves as context increases.

**Key metric:** Half-life — context length where 50% of total perplexity benefit is achieved.

**Findings:**
- Memory curve shape appeared universal across 7 genres (ANOVA p=0.53, excluding recipes)
- Half-life ~22-27 tokens across all genres
- Recipes were an outlier (~36 tokens) — attributed to external scaffolding (numbered steps)
- Human vs AI showed similar shape but small systematic difference

**Limitation discovered later:** The curve shape was largely driven by instrument bias (see v3).

**How to run:** Colab with T4 GPU, ~20 min runtime. Requires `raid_corpus.jsonl` on Google Drive.

---

### v2: `RAID_Multi_Genre_Analysis_v2_cross_model.ipynb`
**Purpose:** Test whether results are robust to the choice of measuring model, and whether RLHF affects measurement.

**Method:** Same as v1 but runs with multiple measuring models:
- Mistral-7B (base)
- Mistral-7B-Instruct (RLHF'd)
- Llama-3-8B or Gemma-2-9B (different architecture)
- Qwen2-7B

Also includes shuffled-token and uniform-random controls.

**Key findings:**
- Base and instruct models recover the same curve shape (per-essay r=0.73)
- RLHF doesn't distort the measurement, just adds noise
- Base model is the better measuring instrument

**Critical discovery:** Uniform random text showed a 37% perplexity drop with context — revealing that the "memory curve" is heavily contaminated by the model calibrating its output distribution, not by detecting coherence structure.

---

### v3: `RAID_Multi_Genre_Analysis_v3_topk.ipynb`
**Purpose:** Replace perplexity with top-k accuracy to eliminate the instrument bias.

**Method:** Instead of perplexity, measure whether the correct next token is in the model's top-k predictions (k=1, 5, 10, 50, 100). Also compute mean rank of the correct token.

**Controls:**
- Shuffled tokens (same vocabulary, random order)
- Uniform random tokens (from full vocabulary)

**Key findings:**
- **Uniform random: top-k accuracy = 0 at all context windows.** Clean null baseline — the instrument bias is eliminated.
- Shuffled text: ~16-22% accuracy (vocabulary/frequency estimation)
- Human text: ~78-85% accuracy
- **After subtracting shuffled baseline:** Sequential coherence contribution is FLAT at ~62% across all windows. Delta from W4 to W128 = 0.004 (essentially zero).

**Critical insight:** Token-level prediction is the wrong grain for measuring long-range coherence. Long-range context doesn't help predict specific tokens — it constrains semantic regions. Like knowing which city you're in doesn't tell you which restaurant you'll visit.

---

### v4: `RAID_Multi_Genre_Analysis_v4_semantic.ipynb`
**Purpose:** Test semantic-level metrics that might capture long-range coherence invisible to token-level accuracy.

**Method:** At each target position, compute:
- **Expected embedding similarity:** Probability-weighted average embedding of the model's output distribution compared to actual token embedding (cosine similarity)
- **Top-k semantic similarity:** Average cosine similarity between top-50 predicted token embeddings and actual token
- **Content-word accuracy:** Top-k accuracy scored only on content words (nouns, verbs, adjectives)

**Key findings:**
- Expected embedding similarity shows a genuine coherence signal after shuffled subtraction: corrected value increases from 0.412 (W4) to 0.449 (W128)
- This is small (+3.7% over 32x context increase) but monotonic and real
- Other metrics (top-k similarity, content-word accuracy) are flat after correction
- Long-range context shifts probability mass toward the right semantic neighborhood without changing the top candidates

---

### v4b: `RAID_v4b_dense_semantic.ipynb`
**Purpose:** Dense sampling of the expected embedding similarity curve to find inflection points or regime transitions.

**Method:** Same as v4 but with 24 context windows (every 2 tokens from 2-32, then sparser to 128) instead of 10.

**Key findings:**
- Two-regime structure: rapid decay W2-W8 (~4x falloff in marginal gain per token), then slow steady decay W8-W128
- The transition around W6-W10 may correspond to the boundary between syntactic (local) and discourse-level (extended) influence
- Marginal gain per token follows roughly 1/distance — logarithmic scaling

---

### v5: `RAID_v5_sentence_contrast.ipynb`
**Purpose:** Move to sentence-level measurement where long-range coherence actually operates.

**Method:** For each sentence in the target region of a document:
1. Provide W tokens of preceding context
2. Compute perplexity on the **real** next sentence
3. Compute perplexity on **impostor** sentences (from other documents in the same genre)
4. Contrast score = log(ppl_impostor / ppl_real)

Higher contrast = the model better distinguishes the real sentence from an impostor given the context.

**Controls:** Sentence-shuffled documents (sentence order randomized within document).

**Key findings:**
- Contrast increases monotonically from 0.32 (W4) to 1.34 (W128) — a 4x gain
- Marginal gain per token decays from 0.044/token (W4-W8) to 0.001/token (W96-W128) — continuous decay with no floor
- Shuffled text also increases (topic identification) but 25% less than intact text — the sequential coherence component
- Genre universality: all 8 genres show similar contrast scaling

**Important design note:** Impostors must come from **other documents** in the same genre, not from the same document. Same-document impostors produce paradoxical results because they become more plausible with more context.

---

### v6: `RAID_v6_sentence_influence.ipynb`
**Purpose:** Identify which specific sentences carry long-range influence — the "connective tissue" of narrative structure.

**Method:** Leave-one-sentence-out design:
1. For each target sentence, compute perplexity with **all** prior sentences as context
2. Then compute perplexity with each prior sentence **dropped** one at a time
3. Influence(sentence_i → target_t) = log(ppl_without_i / ppl_with_i)

Produces a sentence × sentence **influence matrix** per document.

**Key findings:**
- Aggregate influence decays with sentence distance (~0.25 at d=1, ~0.02 at d=5, ~0.01 at d=10+)
- Influence is significantly positive out to at least 16 sentences (~250-300 tokens)
- **Individual sentences do NOT decay smoothly** — monotonicity score = 0.54, barely above random. The smooth aggregate is an average of spiky, content-driven individual curves.
- **Long-range hotspots:** 3.5% of sentence pairs at distance 15+ show substantial influence (>0.1). These are specific "load-bearing" dependencies — character introductions, ingredient lists, thesis statements.
- **Sentence lifespans:** Mean 8.5 sentences, 30% extend beyond 10, 7% beyond 20.

**Human vs AI comparison:**
- AI text has ~2x the long-range hotspot density of human text
- At distance 10+: 14.4% of AI sentence pairs show substantial influence vs 7.9% for human
- Same two-component structure (local decay + sparse hotspots) but AI has denser long-range connections
- Interpretation: AI has no memory bottleneck, so it embeds more long-range dependencies

**What the most influential sentences look like:**
- Recipe ingredient lists (influence cooking steps 10-26 sentences later)
- Character introductions (influence every subsequent reference)
- Setting/premise establishment (first sentences have longest lifespans)
- Thesis statements and emotional turning points

**What least influential sentences look like:**
- Generic commentary ("Of course there are different ways of looking at this")
- Self-contained evaluations ("The ending feels rushed")
- Terminal actions ("Cool for 10 minutes")

---

### ECSC Developmental: `ECSC_Sentence_Contrast_v1.ipynb`
**Purpose:** Test whether long-range coherence in spontaneous child speech increases with age.

**Data:** ECSC (Eugene Children's Story Corpus) — 330 frog story narrations from children aged 5-11 years. Spontaneous speech, no pre-planning.

**Method:** Same sentence contrast approach as v5, with 5 context windows (4, 16, 32, 64, 128 tokens). Impostors from other children in the same age bin.

**Key findings:**
- Age correlation with contrast increases with context distance:
  - W4: r=0.001 (zero)
  - W32: r=0.030 (zero)
  - W64: r=0.084 (trending)
  - W128: r=0.111, p=0.045 (significant)
- Local coherence (short context) is age-invariant — even 5-year-olds produce locally coherent sentences
- Long-range coherence specifically increases with age
- The developmental jump is largest between 5-6y and 7-8y, then plateaus — consistent with WM development literature

---

## Key Theoretical Insights

### Three Components of "Contextual Benefit"
1. **Distributional calibration** — model adjusts output distribution to input statistics. Present even for random text. Dominates perplexity. NOT a text structure signal.
2. **Vocabulary/frequency estimation** — model learns document-specific token frequencies. Present for shuffled text. Drives the "memory curve shape" in perplexity.
3. **Sequential coherence** — long-range context improves sentence-level prediction. Only present in coherent text. Measured by sentence contrast and influence matrices.

### Two Components of Long-Range Influence
1. **Local production (distance 1-3 sentences):** Smooth, strong decay. Syntax, sentence-to-sentence coherence, pronoun resolution. Likely working memory / short-term activation.
2. **Discourse framework (variable distance, spike-based):** Specific sentences exerting influence at specific later moments when their content becomes relevant. Not decay-based — more like selective retrieval from a discourse representation.

### The Argument for Memory
1. Long-range predictive structure exists (sentence contrast increases to W128)
2. Local-only production cannot explain it (benefit continues beyond local range)
3. Topic persistence alone cannot explain it (genre-matched impostors control for this; benefit continues after topic is established)
4. Pre-planning is weakened by similar effects in spontaneous speech and less-planned genres
5. The structure develops with age in children (ECSC results)
6. AI text shows denser long-range structure (no memory bottleneck)
7. The argument from necessity: the structure couldn't exist unless the writer maintained information from distant prior content during production

---

## Practical Notes

### Running the Notebooks
- All notebooks are designed for Google Colab
- Minimum: T4 GPU (16GB). L4 or H100 recommended for v5/v6
- Model: Mistral-7B-v0.1, 4-bit quantized via bitsandbytes
- Data: `raid_corpus.jsonl` and `transcripts.jsonl` (ECSC) needed on Google Drive or uploaded directly
- Resume support: most notebooks save intermediate results and skip completed steps

### Key Gotcha
Always run controls (shuffled + uniform random) on any new metric before interpreting results. The v1-v2 perplexity results looked beautiful but were largely artifact.
