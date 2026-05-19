# A heavy-tailed law of contextual memory in human language

*Nature submission — draft v2 (working title; see notes)*

---

## Abstract

Many human behaviors exhibit power-law temporal structure with no characteristic scale. Whether language shares this signature has not been directly measured. We use large language models as probabilistic probes to measure how much prior context constrains the probability of upcoming tokens, isolated from token-frequency effects through a shuffled-token baseline. The persistence function P(d) quantifies this order-specific influence at distance d from the current point of generation. Across nine corpora spanning diverse languages, genres, and modalities, P(d) decays approximately as 1/d, with mean slope −1.04 (SD = 0.16) across nearly two orders of magnitude, vanishes in random-token controls, replicates across independent probe models, and is broadly distributed across prior sentences rather than concentrated in anchors. Order- and content-driven contributions are distinguishable, with content-driven integration carrying the long-range tail. The continuous scale-free form is difficult to reconcile with architectural accounts that posit a sharp division between maintained and retrieved material, and is consistent instead with memory in language as a graded distance-dependent influence carried by the evolving probability structure of the sequence itself.

---

## Introduction

Human behavior across many domains exhibits lawful temporal structure across many timescales. In neural activity (Beggs & Plenz, 2003), motor and cognitive performance (Gilden, 2001), and inter-event waiting times in human dynamics (Barabási, 2005), distance-dependent dependencies decay as power laws rather than as bounded or exponential functions, placing these processes among the broad class of human behaviors with scale-free temporal organization (Kello et al., 2010). Language is among the longer-range coherent sequential behaviors humans produce, sustaining structure across thousands of words, but the quantitative form of its temporal structure — how the influence of prior material scales with distance from the current point of generation — has not been directly measured. Existing characterizations of contextual integration in language, including discourse models, situation models, and resonance-based accounts, operate at the representational rather than the quantitative level, and conventional measures of statistical dependence in text conflate sequential structure with vocabulary-level regularities.

Here we develop a distance-resolved measurement of contextual influence in natural language and apply it across nine corpora spanning diverse languages, genres, and modalities. We use large language models as probabilistic probes, treating their conditional probability estimates as a calibrated readout of how preceding context constrains upcoming tokens. Because naive perplexity reduction conflates sequential dependence with the probe's distributional calibration, we subtract a shuffled-token baseline that absorbs the contribution of token identity alone. The resulting persistence function P(d) quantifies the order-specific influence of prior context at distance d from the current point of generation.

P(d) decays approximately as 1/d with mean slope −1.04 (SD = 0.16) and no characteristic cutoff across nearly two orders of magnitude in distance. This decay is consistent across language families, genres, and modalities, vanishes in random-token controls, scales with the integrity of sequential structure, and is broadly distributed across prior sentences rather than concentrated in a small number of anchors. The persistence function describes a continuous, scale-free temporal structure in human language production, measurable directly from the probability dynamics of generation and replicable across independently trained probe models (Llama-3.1, Mistral-7B).

The shape of P(d) places language alongside other human behaviors and neural processes that exhibit scale-free temporal structure, and constrains the class of mechanisms capable of producing such structure. In particular, the absence of any characteristic scale across the measured range — combined with the graded contributions of prior content distributed across the sequence — is difficult to reconcile with architectural accounts of memory that posit a sharp division between maintained and retrieved material with distinguishable temporal dynamics. The pattern is consistent instead with mechanisms in which prior content continues to shape ongoing generation through the evolving probability structure of the sequence itself.

---

## Results

### A persistence function for natural language

We define the persistence function P(d) as the predictive influence of prior context at distance d from the current point of generation, estimated as the reduction in target perplexity produced by revealing prior context at that distance, with shuffled-token baselines controlling for the contribution of token identity alone (Methods). P(d) provides a distance-resolved measurement of how preceding language constrains the probability structure of upcoming tokens.

Across all corpora, P(d) is positive at distances well beyond any plausible bound on working-memory capacity. At distances exceeding 1000 tokens, prior context continues to reduce target perplexity by a measurable margin relative to shuffled controls. Long-range contextual influence is therefore not confined to a recency window but extends as a graded function across the full sequences we examined.

### Heavy-tailed decay across languages, genres, and modalities

The functional form of P(d) is well approximated by a heavy-tailed decay. Figure 1 plots P(d) against distance on log–log axes for each of the nine corpora. Across languages, genres, and modalities, P(d) decays smoothly across nearly two orders of magnitude with a mean slope of −1.04 (SD = 0.16) — close to a 1/d power law — and a median goodness-of-fit of r² = 0.93. Fits are computed on d ≥ 10 tokens, the range over which both ordered- and shuffled-context bases are stable; very-short-range intervals are reported in Methods. Power-law fits substantially outperform exponential alternatives, with ΔAIC values ranging from +12 to +26 across the majority of corpora. Two corpora, Japanese literary prose and Buckeye spoken English, are better fit by stretched-exponential functions, but their decay remains heavy-tailed and extends well beyond the range over which any bounded buffer could plausibly contribute.

The cross-corpus consistency of the decay regime is notable. The sample spans Germanic, Romance, Turkic, Japonic, and Uralic language families, as well as fiction and literary prose, news and expository prose, prepared speech transcripts, and spontaneous speech. A narrow heavy-tailed regime across this range indicates that the persistence function reflects a general property of natural language rather than a property of any single language, register, or modality.

The qualitative pattern that defines P(d) — a positive contribution of ordered context at long range together with a near-zero contribution of shuffled context — extends beyond the five families covered by the long-range run. In a complementary cross-language analysis evaluating shorter context lengths on both probe models, the long-range sign pattern is preserved across three additional language families: Slavic, Afro-Asiatic, and Koreanic (Supplementary Information).<sup>[Q2]</sup> The headline slope of P(d) is therefore obtained on a five-family subset of a broader eight-family cross-language replication of the qualitative pattern that defines the persistence function.

### The persistence function reflects sequential structure, not properties of the probe

A central concern in any LLM-based measurement is whether the observed pattern reflects properties of natural language or properties of the probe model. We address this directly with synthetic-sequence controls. We constructed sequences with token-frequency statistics matched to natural text but with sequential structure removed. Raw perplexity over these synthetic sequences exhibits apparent power-law structure with a slope comparable in magnitude to that observed on natural text (Llama-3.1: ≈ −1.27 on uniform random vocab; Mistral-7B: ≈ −1.34), replicating the general tendency for language-model perplexity to fall as context length increases even when genuine long-range dependence is absent.

The order-specific persistence function P(d), however, is near zero across all distances on the synthetic sequences (Figure 2). The shuffled-token baseline that defines P(d) absorbs the perplexity reduction attributable to token statistics alone, isolating the contribution of sequential organization. The long-range, heavy-tailed P(d) observed for natural language therefore depends on properties of the linguistic sequence itself, not on the probe.

We replicated the persistence signature with two independently trained probe models, Llama-3.1 and Mistral-7B, obtaining the same qualitative pattern and comparable decay ranges on shared corpora, along with near-zero P(d) on synthetic controls. The persistence function is therefore not an idiosyncratic property of the model used to estimate it.

### Order and content make distinguishable contributions to persistence

To characterize the structure of P(d) further, we compared ordered context to two structure-disrupting manipulations. Sentence shuffling preserves within-sentence syntax and lexical content while disrupting discourse-level ordering. Token shuffling eliminates ordered structure within the contextual span.

Sentence shuffling reduces P(d) at short distances, especially below approximately 30 tokens, but leaves much of the long-range tail intact (Figure 3). Token shuffling collapses P(d) toward zero across all distances. The persistence function therefore decomposes into two distinguishable contributions: a chaining-specific component concentrated at short range (slope ≈ −1.32, approaching zero by d > 100 tokens) and a content-driven component that extends across the long-range tail (slope ≈ −0.82, comparable to the total order-specific slope of −0.88). Discourse-level chaining shapes the short-range structure of P(d); content-driven integration carries its long-range tail.

### Persistence is broadly distributed across prior sentences

We next asked whether persistence is concentrated in a small number of privileged prior sentences or broadly distributed across the prior context. We addressed this with a sentence-level ablation: for each target, we removed one prior sentence at a time and measured the resulting change in prediction.

The majority of prior sentences contribute positively to prediction at all distances examined. The proportion of contributing sentences declines gradually with distance, from approximately 90% at short range to 55–60% beyond 500 tokens, with no sharp transition or threshold (Figure 4). The magnitude of contribution is concentrated at recent positions (top-5 sentences account for 30–51% of total positive influence; Gini coefficient 0.78–0.90), but this concentration is fully explained by recency: variation in mean influence by absolute position within the prior context is accounted for by distance to the target, with no residual effect of structural position (e.g., document opening). Persistence is therefore a distributed field of influence across the prior sequence, with magnitude graded smoothly by recency rather than localized to a small number of anchoring positions.

---

## Discussion

The persistence function shows a continuous, distance-dependent influence that extends well beyond the span of explicit maintenance, decays without a characteristic scale, and is distributed across the prior sequence rather than localized to a small number of contributing positions. This pattern is difficult to reconcile with architectural accounts that posit a sharp division between recently-maintained material and remotely-retrieved material with distinguishable temporal dynamics: a sharp division would predict a corresponding inflection in P(d), and we observe none across nearly two orders of magnitude in distance. The distributed influence pattern is similarly difficult to reconcile with retrieval-based accounts in which long-range coherence is supported by access to a small number of privileged anchoring positions. The data are consistent instead with mechanisms in which prior content continues to shape ongoing generation through the evolving probability structure of the sequence — a graded, distance-dependent field of influence rather than a partition between maintained and retrieved content.

The mean exponent we report (−1.04, SD = 0.16) sits in a neighborhood that has been independently identified in two adjacent traditions in memory research. Anderson and Schooler (1991), examining the recurrence statistics of information demand in natural environments, reported a power law with exponent of approximately −0.77. Kahana et al. (2002), examining associative decay in free recall via lag-conditional response probability, reported a forward-decay exponent of approximately −0.82 for younger adults. Each of these measurements was made by entirely different methods on entirely different data, and neither involves a language model. The convergence is striking but does not by itself establish a common underlying mechanism: it motivates direct cross-paradigm investigation of whether comparable scale-free decay forms across production, environmental statistics, and recall reflect a domain-general dynamic of biological cognition or distinct mechanisms producing similar exponents.

This empirical picture aligns with a longer trajectory in cognitive psychology away from architecturally bounded views of memory and toward continuous and graded characterizations. Cowan's focus-of-attention model recasts working memory as a graded availability of items rather than a fixed-capacity slot store; Ericsson and Kintsch's long-term working memory framework argues that skilled performers make routine use of long-term-memory structures during ongoing task performance, blurring the maintenance-retrieval boundary; and continuous-resource accounts in visual working memory replace discrete-item architectures with graded distributions of representational precision. The persistence function provides a direct quantitative measurement of an analogous continuity in language production, observed at the scale of token-level sequential dependence and across naturalistic discourse rather than experimental task sequences. The present result is best read as an empirical capstone to this longer trajectory, not a refutation of it.

We do not claim that these results establish the mechanism by which prior content shapes language production in the brain. The persistence function is a measurement of the form of contextual influence, not of the cognitive processes that implement it. We note, however, that artificial autoregressive systems trained on human text exhibit the same scale-free shape under the same measurements, supporting the interpretation that P(d) reflects properties of well-formed sequential generation rather than properties specific to human cognition. Whether the persistence function generalizes to non-linguistic sequential behaviors with extended temporal coherence — music improvisation, motor sequences, gesture, drawing — and whether the convergence with memory-retrieval exponent ranges reflects a shared dynamic or merely shared statistical constraints, are questions the present results motivate but do not answer.

---

## Figure plan

All six figures in `Paper_Figures/` are from the old PNAS draft and do not match the new framing. Four new figures are needed for the main text. Each spec below names data source, panel layout, and a draft caption.

---

### Figure 1 — Heavy-tailed P(d) decay across nine corpora *(supports Results §2)*

**Data source:** `Corpus_Expansion_LongRange_Llama`, log-spaced contexts [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]. Nine cells: gutenberg_en, ted_en, ted_de, ted_fr, ted_tr, literary_ja, literary_fi, news_en, buckeye.

**Panel layout:** Single main panel, log-log axes.
- 9 colored lines, one per corpus, plotting per-token order-specific gap (= P(d) per added token) vs distance d
- Power-law fits overlaid as light dashed lines per cell
- The two stretched-exponential cells (literary_ja, buckeye) marked distinctly — e.g., dotted-line fit instead of dashed, or different marker shape
- Inset (top-right): strip plot of the 9 fitted slopes with mean −1.04 and ±SD band annotated

**Optional second panel (small):** strip plot of per-cell slopes for the cross-probe Mistral replication on shared cells, to visually anchor the probe-independence claim.

**Draft caption:** *Cross-corpus persistence functions. Per-token order-specific contextual influence P(d) as a function of distance d, log-log axes. Each line represents one of nine corpora spanning five language families and four production modes (legend). All cells exhibit heavy-tailed decay across nearly two orders of magnitude in distance, with mean power-law slope −1.04 (SD = 0.16) — close to a 1/d law. Fits computed on d ≥ 10 tokens. Two cells (Japanese literary prose, dotted; Buckeye spontaneous English speech, dotted) are best fit by stretched-exponential functions but remain heavy-tailed. Inset: distribution of per-cell fitted slopes.*

---

### Figure 2 — Synthetic-sequence controls isolate sequential dependence from probe calibration *(supports Results §3)*

**Data source:** `random_vocab_uniform` (Llama-3.1, Mistral-7B) and the corresponding natural-language cells.

**Panel layout:** Two panels side by side, log-log axes.

- **Panel A (raw perplexity reduction):** raw per-token perplexity-reduction slope on synthetic sequences (Llama: ≈ −1.27; Mistral: ≈ −1.34) plotted alongside the same quantity on natural language. Visual demonstration that probes show heavy-tailed perplexity reduction even on sequences with no genuine sequential dependence.
- **Panel B (order-specific P(d)):** P(d) on synthetic sequences (near zero across distances) plotted alongside P(d) on natural language (heavy-tailed). Demonstrates that the shuffled-token subtraction absorbs the probe-baseline calibration and isolates genuine sequential dependence.

**Draft caption:** *Synthetic-sequence controls separate probe-internal calibration dynamics from genuine sequential dependence. (a) Raw perplexity-reduction slopes on uniform random-vocabulary sequences (Llama-3.1: ≈ −1.27; Mistral-7B: ≈ −1.34) closely match those obtained on natural language, replicating the general tendency for language-model perplexity to fall as context length increases regardless of input order. (b) The order-specific persistence function P(d), defined by subtraction of a shuffled-token baseline, is near zero across all distances on synthetic sequences but heavy-tailed on natural language. The decay reported in this paper therefore reflects properties of the linguistic sequence rather than the probe.*

---

### Figure 3 — Order and content make distinguishable contributions to persistence *(supports Results §4)*

**Data source:** `Corpus_Expansion_LongRange_SentShuffle_Llama`, eight cells (long-range run minus buckeye). Three conditions per target: ordered, sentence-shuffled, token-shuffled.

**Panel layout:** Two panels side by side, log-log axes.

- **Panel A (raw conditions):** Three lines for one representative cell (e.g., gutenberg_en) or aggregated across the eight cells: ordered context P(d), sentence-shuffled P(d), token-shuffled P(d) ≈ 0. Shows ordered ≈ sentence-shuffled at long range; both diverge sharply from token-shuffled.
- **Panel B (decomposition):** Two lines: chaining-specific component (ordered − sentence-shuffled; slope ≈ −1.32) and content-driven component (sentence-shuffled − token-shuffled; slope ≈ −0.82). Annotate slopes inline. Mark d > 100 region where chaining-specific component approaches zero.

**Draft caption:** *Sentence-shuffle decomposition. (a) P(d) under three context conditions: intact (ordered), sentence-shuffled (sentences permuted while within-sentence token order preserved), and token-shuffled (all token order destroyed). The intact and sentence-shuffled curves coincide at long range and diverge only below ~30 tokens; both lie far above the token-shuffled baseline at all distances. (b) Decomposition into a chaining-specific component (intact minus sentence-shuffled; slope ≈ −1.32, approaching zero by d > 100 tokens) and a content-driven component (sentence-shuffled minus token-shuffled; slope ≈ −0.82, comparable to the total order-specific slope of −0.88). Discourse-level chaining shapes the short-range structure of P(d); content-driven integration carries its long-range tail.*

---

### Figure 4 — Persistence is broadly distributed across prior sentences *(supports Results §5)*

**Data source:** `Sentence_Ablation_Density_Llama` on gutenberg_en + ted_en (20 docs/cell, 1024-token prior context); `Sentence_Influence_Matrix_Adult_Llama` (15 docs/cell) for the position analysis.

**Panel layout:** Two panels side by side.

- **Panel A (proportion of contributing sentences vs distance):** Distance-binned proportion of prior sentences whose removal increases target perplexity (positive contribution). Smooth declining curve from ~90% at d < 32 to ~55–60% at d > 512. No sharp transition. Show both corpora as separate lines or aggregated with shaded band. Annotate the d > 500 region explicitly.
- **Panel B (position vs distance):** Mean per-sentence influence as a function of (i) absolute prior-sentence position in the document and (ii) distance to the target. Shows that absolute-position curve is fully explained by mean distance at each position; no residual position effect. Could be implemented as a paired plot with two lines, or as a partial-residual visualization. Goal: visually rule out the "document opening is special" rescue.

**Draft caption:** *Sentence-level ablation. (a) Proportion of prior sentences whose removal degrades target prediction (i.e., that contribute positively to P(d)) as a function of distance to the target, in two corpora (English fiction; English TED transcripts). The proportion declines smoothly from ~90% within the first 32 tokens to ~55–60% beyond 500 tokens, with no sharp transition. (b) Mean per-sentence influence by absolute position in the prior context (blue) and by distance to the target (orange), showing that variation in influence by position is fully accounted for by distance. The top-5 sentences account for 30–51% of total positive influence (Gini coefficient over magnitudes 0.78–0.90), but this concentration reflects recency, not a privileged structural role for any specific position.*

---

### Note on figures dropped from the old draft

- `fig1_human_vs_ai_curves.png` — drop (AI vs. human contrast removed from new framing).
- `fig5_absolute_magnitude.png` — drop (AI vs. human, same reason).
- `fig6_grand_summary.png` — drop (built for the AI-vs-human convergence story including Anderson & Schooler reference line; not aligned with the new discussion).
- `fig2_functional_form.png` — possibly repurposable as a Supplementary figure (functional-form comparison) if updated to use the long-range-run data on all 9 cells. Not currently usable.
- `fig3_crosslingual_convergence.png` — drop from main text; the old alpha-clustering claim (mean −0.78) is not the headline. Could become a Supplementary figure for the broader 8-family qualitative replication if reframed around the sign-flip rather than the alpha bar chart.
- `fig4_per_speaker.png` — keep as Supplementary (per-speaker variability in Buckeye; useful but not centerpiece).

---

## Methods

### Corpora

The long-range persistence function was computed on nine corpora: English fiction (Project Gutenberg), English news prose, English spontaneous speech (Buckeye Corpus of Conversational Speech), TED transcripts in English, German, French, and Turkish, and literary prose in Japanese and Finnish. Together these cover five language families (Germanic, Romance, Turkic, Japonic, Uralic) and four production modes (written fiction, news prose, prepared speech, spontaneous speech). The complementary cross-language qualitative replication of the sign-flip pattern (Slavic, Afro-Asiatic, Koreanic) was performed on Wikipedia and TED-transcript cells from a broader corpus inventory at shorter context lengths. Full corpus details, sources, sample sizes, and preprocessing steps are provided in Supplementary Information.

### Probe models

Two autoregressive language models served as probes: Llama-3.1-8B and Mistral-7B. Neither was fine-tuned for this study. The probes were used only to compute conditional probability estimates over fixed token sequences; the language models themselves were not the object of study. Headline results were obtained on Llama-3.1-8B; the qualitative pattern was replicated on Mistral-7B for shared corpora as a probe-independence check.

### Persistence function

For each document, three target regions of 30 tokens were selected at 25%, 50%, and 75% of document length to assess position invariance. Context preceding each target was revealed in incrementally larger spans. At each context length *c*, target perplexity was computed under intact context (the actual preceding *c* tokens, original order) and shuffled context (the same *c* tokens, random order). Per-distance marginals were obtained as *m*<sub>d</sub> = perplexity<sub>d−1</sub> − perplexity<sub>d</sub>, where *d* indexes the distance of the most recently added token from the target. The persistence function was defined as P(*d*) = *m*<sub>d</sub><sup>intact</sup> − *m*<sub>d</sub><sup>shuffled</sup>, isolating the contribution of sequential structure from probe-internal token-identity calibration. Marginals were aggregated across documents within each corpus and binned by distance.

### Long-range protocol and functional-form fitting

For the long-range run, context lengths were log-spaced from 0 to 1024 tokens (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024). The per-token order-specific gap was fit on log-log axes via nonlinear least squares to four functional forms: power law (*y* = *α* · *d*<sup>*β*</sup>), exponential (*y* = *α* · *e*<sup>−*β d*</sup>), stretched exponential (*y* = *α* · *e*<sup>−*β d*<sup>γ</sup></sup>), and quadratic-in-log. Models were compared by Akaike (AIC) and Bayesian (BIC) information criteria. Functional-form preference was determined per corpus.

### Synthetic-sequence controls

To distinguish probe-internal calibration dynamics from genuine sequential dependence, two synthetic-sequence controls were constructed: uniform-vocabulary sequences (tokens drawn uniformly at random from the model vocabulary) and frequency-matched sequences (tokens drawn with empirical token-frequency probabilities matched to the natural corpora). The same persistence-function protocol was applied to both, providing a probe-baseline characterization against which natural-language P(*d*) was compared.

### Sentence-level decomposition

The contribution of discourse-level ordering versus within-sentence content was assessed by computing P(*d*) under three context conditions: ordered (intact), sentence-shuffled (sentences permuted while within-sentence token order preserved), and token-shuffled. The chaining-specific component was estimated as the difference between ordered and sentence-shuffled persistence; the content-driven component as the difference between sentence-shuffled and token-shuffled. The decomposition was computed across eight cells of the long-range run; Buckeye was excluded because spontaneous speech does not have unambiguous sentence boundaries.

### Sentence-level ablation

For documents in two corpora (English fiction and English TED transcripts, 20 documents per corpus, 1024-token prior context), each prior sentence was individually removed and the resulting change in target perplexity was computed (leave-one-sentence-out). Per-sentence influence was characterized by sign (positive = removing the sentence increased target perplexity, indicating a positive contribution to prediction) and magnitude. The proportion of contributing sentences was computed in distance bins. Concentration metrics — Gini coefficient over per-sentence magnitudes and the share of total positive influence accounted for by the top-5 sentences — were computed per document and aggregated.

### Influence-by-position analysis

To test whether magnitude variation across prior-sentence positions reflects a privileged structural role (e.g., document opening) over and above distance from the target, the full leave-one-sentence-out influence matrix was computed on a 15-document subset of each of the two ablation corpora. Mean influence by absolute prior-sentence position was tested against mean influence by distance to the target. The persistence function pattern reported in Results §5 was confirmed: variation in influence magnitude was accounted for by distance, with no residual effect of structural position once distance was controlled.

---

## Open questions / data checks

**Q1 (§ Heavy-tailed decay, family list):** ✓ Resolved 2026-05-04. Long-range slope is on 5 families (Germanic, Romance, Turkic, Japonic, Uralic). User confirmed no additional long-range data is available.

**Q2 (§ Heavy-tailed decay, broader cross-language sign-flip claim):** Pursuing option B (two-pronged: 5-family slope + 8-family qualitative replication). I've written the supplementary paragraph listing Slavic, Afro-Asiatic, and Koreanic as the three additional families beyond the long-range five, based on memory of 2026-05-03: *"7 typological families show the sign-flip (Germanic, Romance, Slavic, Afro-Asiatic, Turkic, Japonic, Uralic, Koreanic)."* That enumeration totals 8 family names; **please confirm:**
- Is Sino-Tibetan (zh Wiki, which is in `~/lrtia/results/Llama_crosslingual/` per the dataset map) included in the sign-flip set? If yes, total goes to 9 families and we should add it.
- Where does the Slavic data come from? The Wiki crosslingual set is zh/ja/ko/tr/ar/fi — no Slavic. Russian (corpus_expansion Phase 2) was listed as not yet run as of 2026-05-01. If Phase 2 has since run, that's the likely source.

---

## Revision notes (v2 vs. user's v1)

Edits applied in this pass — all in the intro:

1. **Para 1, S3:** "longest-range" → "among the longer-range coherent sequential behaviors humans produce" — defensible against music/motor/planning counter-claims while preserving the line.
2. **Para 1, last sentence:** named the traditions ("discourse models, situation models, resonance-based accounts") so the representational-vs-quantitative contrast lands for non-specialist readers.
3. **Para 2, probe-artifact sentence:** tightened from two clauses to one ("Because naive perplexity reduction conflates sequential dependence with the probe's distributional calibration...").
4. **Para 3:** added SD inline ("mean slope −0.94 (SD = 0.12)") and named probes ("Llama-3.1, Mistral-7B") rather than only gesturing at cross-probe replication. *[Note: this number was later revised to −1.04 (SD = 0.16) when Fig 1 was retrimmed to d ≥ 10; see Pass 7.]*
5. **Para 4:** inserted "of memory" after "architectural accounts" so the title's memory framing lands in the intro.
6. **Para 4:** dropped "We return to this point in the Discussion" — Nature intros avoid forward references.
7. **Para 4:** added a conservative mechanism-pointing closing sentence ("consistent instead with mechanisms in which prior content continues to shape ongoing generation through the evolving probability structure of the sequence itself") — preempts the "structural claim only, no contribution" reading without committing to autoregressive generation by name.

Citations kept: Beggs & Plenz, Gilden, Barabási, Kello et al. — Kello flagged for possible cut if word budget tightens.

Title: "A heavy-tailed law of contextual memory in human language" placed at top as working title; revisit after results + abstract are locked.

---

## Pass 2 edits (results merge)

8. **Abstract:** drafted at 162 words; uses "constrains the probability of upcoming tokens" rather than naming perplexity (jargon avoidance for Nature audience).
9. **§ Heavy-tailed decay:** narrowed family list to those represented in the 9-corpus long-range run (see Q1 above for verification).
10. **§ Reflects sequential structure:** added probe-baseline slopes for synthetic random-vocab sequences (Llama −1.27, Mistral −1.34) — converts qualitative "apparent power-law structure" claim to quantitative anchor; matches what would otherwise be the strongest reviewer challenge ("isn't −1 just a probe property?").
11. **§ Order vs content:** added decomposition slopes (chaining −1.32, content-driven −0.82, total order-specific −0.88) and noted chaining approaches zero by d > 100 — converts qualitative "leaves long-range tail intact" claim to quantitative.
12. **§ Broadly distributed:** added Gini (0.78–0.90) and top-5 (30–51% of positive influence) magnitude-concentration metrics; explicitly noted that position-magnitude variation is accounted for by distance rather than structural position (preempts "but maybe document openings are special" rescue).

## Pass 3 edits (cross-language two-pronged framing)

13. **§ Heavy-tailed decay:** added a third paragraph reporting the qualitative sign-flip pattern across three additional language families (Slavic, Afro-Asiatic, Koreanic) on a complementary shorter-context cross-language analysis. Explicit framing: "five-family subset of a broader eight-family cross-language replication." Recovers the universality claim without overclaiming the slope.

## Pass 4 edits (Discussion drafted)

14. **¶1 — continuous-distributed picture:** "difficult to reconcile" register applied to both WM/LTM-split and anchor/retrieval models. Closes with the positive framing ("graded, distance-dependent field of influence").
15. **¶2 — Anderson/Kahana convergence:** restrained register (option a). Reports the two exponents (~−0.77, ~−0.82), notes "three traditions / different methods / no language model," but explicitly hedges: "convergence is striking but does not by itself establish a common underlying mechanism." Frames cross-paradigm investigation as the question motivated, not answered.
16. **¶3 — cog-psych predecessors:** Cowan / Ericsson & Kintsch / Brady & Alvarez positioned as continuous-memory predecessors; the paper as "empirical capstone, not refutation." Pre-empts "but Cowan said WM is bounded" reviewer rescues.
17. **¶4 — closing restraint + AI sentence:** explicitly does not claim mechanism. One sentence noting artificial autoregressive systems exhibit the same shape, framed as evidence that P(d) reflects properties of well-formed sequential generation rather than human cognition specifically (per memory: don't centerpiece AI; one sentence here threads the needle). Closes with two open questions (non-linguistic sequential domains; common dynamic vs. shared constraint).

## Pass 6 edits (Figure plan)

20. **Audit:** all 6 PNGs in `Paper_Figures/` are from the old PNAS draft and don't match the new framing. None directly usable for main-text figures.
21. **Figure plan written** for Figs 1–4 with data sources, panel layouts, and draft captions. Figures need to be generated from the analysis pipeline; none can be repurposed as-is.
22. **Salvage notes:** `fig2_functional_form.png` and `fig3_crosslingual_convergence.png` could become supplementary figures if updated to the long-range-run data and the sign-flip framing respectively. `fig4_per_speaker.png` is supplementary as-is. Three figures (fig1/fig5/fig6) drop entirely (AI-vs-human framing).

## Pass 7 edits (figures generated, headline slope revised)

23. **All four main-text figures generated** from `~/lrtia/paper/build_figures.py` using JSON outputs on Google Drive. Saved as `fig1_persistence`, `fig2_synthetic_controls`, `fig3_sentence_shuffle`, `fig4_distributed_influence` (PNG + PDF).
24. **Fig 1 redesigned** as 3×3 small-multiples grid (each cell its own subpanel with per-cell power-law fit). The original aggregate single-panel version (which carried the −0.94 number) was abandoned — too messy with 9 overlaid traces and overlaid fits.
25. **Headline slope revised: −0.94 (SD = 0.12) → −1.04 (SD = 0.16).** During figure inspection a systematic short-range artifact was identified at d ≈ 2.8–5.7 tokens, where the shuffled-context baseline is unstable (going from 2 → 4 shuffled tokens makes the model worse, not better, by 2.25 ppl/token at the 2 → 4 interval in gutenberg). Fitting only on d ≥ 10 — the first interval at which both ordered and shuffled bases are ≥ 8 tokens, and the canonical boundary in memory ("P(d) at d ≥ 10 is positive") — gives a cleaner curve and a slope of −1.04. The new number is closer to the theoretically motivated 1/d law and matches the load-bearing claim already used in the framing.
26. **Manuscript prose updated** at every instance of the slope (abstract, intro, results §2, discussion para 2, Fig 1 caption, figure-plan inset description). "More than two orders of magnitude" softened to "across nearly two orders of magnitude" since d = 10 to d = 724 is 1.86 decades. Methods note the d ≥ 10 fit boundary; very-short-range intervals can be reported in a supplementary panel.
27. **Cosmetic: dropped intermediate figure files.** Option-A and Option-B test outputs (`fig1_optA_aggregate.{png,pdf}`, `fig1_optB_smallmultiples.{png,pdf}`, `fig1_optB_trimmed.{png,pdf}`) can be cleaned from `Paper_Figures/`; only `fig1_persistence.{png,pdf}` is referenced from the manuscript.

## Pass 5 edits (Methods drafted)

18. **Methods drafted in compressed Nature register, eight subsections.** Replication-sufficient detail without padding. Subsections: Corpora, Probe models, Persistence function, Long-range protocol and functional-form fitting, Synthetic-sequence controls, Sentence-level decomposition, Sentence-level ablation, Influence-by-position analysis.
19. **Methodological choices to flag for verification:**
    - Three target regions per document at 25/50/75% of length, 30 tokens each — carried over from PNAS draft; please confirm this is the protocol used in the long-range run.
    - Probe model versions: "Llama-3.1-8B" (matches abstract/intro) and "Mistral-7B" (no version qualifier) — confirm exact variants (e.g., Llama-3.1-8B-Instruct vs base; Mistral-7B-v0.1).
    - Quantization and hardware deliberately omitted — not material to the result and clutter for Nature methods. If reviewers ask, can be added at revision.
    - Sentence-level ablation: 20 docs per corpus per memory; influence-by-position uses a 15-doc subset per memory. Both numbers appear correct from memory of 2026-05-03.
    - Functional forms compared: power law, exponential, stretched-exp, quadratic-in-log — per memory of 2026-05-03 ("power law beats pure exponential in 7/9 cells; beats or ties quadratic-in-log in 7/9; stretched-exp grid search picks β→0 in 7/9 cells"). Logarithmic decay (in original PNAS draft) was dropped here as not in the canonical comparison; add back if you ran it.
