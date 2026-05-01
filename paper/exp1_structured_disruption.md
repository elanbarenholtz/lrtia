# Experiment 1: Naturalistic disruption validation of the context-influence measure

**Status.** Pre-experiment design (v3, second-review polish; pre-registration-ready pending §10 implementation). To be locked in (timestamped pre-registration) before any disrupted analysis is run.

**One-line goal.** Test whether the corrected-marginal measure is locally sensitive to known order disruptions introduced into natural-language context, in the operating regime that matters for the published claim.

**Positioning.** This is Experiment 1A — a naturalistic positive-control test of the measurement instrument. It does not replace synthetic validation entirely; synthetic validation (Experiment 1B) remains a smaller pipeline-bias check that can be run later or moved to SI. The naturalistic disruption test is positioned first because it operates on real text, with the actual probe, in the actual regime relevant to the human paper's claims.

**Changes from v1.** This revision incorporates ten methodological corrections from external review. The most consequential are: condition-specific shuffled baselines (real technical bug); replacement of pointwise-equality predictions with directional/correlation-based criteria; expansion of D5 into a four-variant family for topic vs. sequence decomposition; addition of D0 no-op control; addition of random cut points (M ∈ {25, 50, 75}) for localization testing; restructured decision rules separating core order-sensitivity from localization, near-context interaction, and topic decomposition; aggregate pass/fail rules across corpora and probes pre-committed in advance.

---

## 1. Rationale

### What the original pipeline measures

The pipeline measures, for a fixed natural-language target region, how much an autoregressive language-model probe's perplexity on that region declines as additional tokens of preceding context are revealed. After subtraction of a token-shuffled baseline, the result is interpreted as the marginal contribution of *ordered* prior context to predicting the target. Across human corpora, this corrected marginal $\Delta_d$ falls as a power law in token distance $d$.

### What this experiment tests

Two reviewer concerns motivate the experiment: pipeline bias (the analysis chain might produce power-law-shaped outputs on inputs that lack power-law structure), and instrument bias (the LLM probe might generate power-law-shaped corrected marginals on any natural-looking text regardless of its underlying structural properties). **`[v3]`** A test that operates on natural-language text with the actual LLM probe directly addresses the instrument-bias concern and provides a naturalistic check against one form of pipeline bias — the form in which the pipeline is invariant to changes in the structural properties of natural-language input. It does not constitute a complete test of pipeline bias more generally, since the curve-fitting machinery's preference for power laws among candidate functional forms is not exercised by these manipulations. Synthetic validation (Experiment 1B) remains necessary for that more general pipeline-bias check.

A *structured disruption* is a manipulation of the prior context that preserves natural-language local structure (so the LLM behaves as it normally would) but introduces a discontinuity in temporal/sequential structure at a *known position*. Each disruption generates a falsifiable predicted feature in the influence curve. If the pipeline produces the predicted features at the predicted positions, it is detecting the structural property that was manipulated. If not, it is not measuring what the published paper claims.

This is a *positive control*: we know what the right answer should look like because we put the structure there. Conventional baselines (token shuffling, sentence shuffling) destroy structure to see what survives. Structured disruptions create structure where we know it should appear.

### What this experiment does not establish

The experiment cannot prove that the human curve reflects human production mechanisms specifically; it cannot establish that autoregressive generation is the unique theoretical interpretation of the empirical pattern; it cannot rule out hierarchical contributions to language structure; it cannot fully prove that the original power law is not partly produced by topic persistence — D5 below estimates topic and document-specific contributions, but does not eliminate them. The experiment establishes that the *measurement instrument* is sensitive to forward order, localized to the manipulated positions, and not a generic artifact of the LLM probe over natural-looking text. Those are the load-bearing methodological claims for the paper.

---

## 2. Disruption set

The protocol uses seven disruption conditions plus the intact reference. The shorthand notation: $C$ = maximum context length in tokens (we use $C = 100$); position 1 = closest to target, position $C$ = most distant; $M$ = cut point dividing the context into a near half (positions 1 to $M$) and a far half (positions $M+1$ to $C$). Primary cut point $M = 50$; secondary cut points $M \in \{25, 75\}$ for localization checks (D1, D3).

Original context: $[t_C, t_{C-1}, \ldots, t_2, t_1, \text{TARGET}]$, where $t_i$ is the token at distance $i$ from the target. The pipeline reveals tokens one at a time starting from position 1 (closest) and growing the prefix outward.

### D0 — No-op control

**Manipulation.** Cut the context at $M$ and immediately rejoin without rearrangement.

This is a sanity check that the disruption pipeline's slicing, joining, and baseline-recomputation code does not introduce artifacts. The disrupted context is byte-identical to the intact context, but it has been routed through the same code path that performs all other disruptions. D0 should produce a curve indistinguishable from intact.

### D1 — Reverse far half

**Manipulation.** Reverse the far half (positions $M+1$ to $C$); leave the near half unchanged.

Disrupted ordering (in temporal sequence preceding the target):
$$[t_{M+1}, t_{M+2}, \ldots, t_C, t_M, \ldots, t_1, \text{TARGET}]$$

When the pipeline grows context past length $M$, the tokens added beyond the cut are the original far-half tokens but in reversed temporal order — the originally most-distant token appears first past the cut.

### D2 — Reverse near half

**Manipulation.** Reverse the near half (positions 1 to $M$); leave the far half unchanged.

Disrupted ordering:
$$[t_C, t_{C-1}, \ldots, t_{M+1}, t_1, t_2, \ldots, t_M, \text{TARGET}]$$

The originally-immediate-predecessor token now sits at distance $M$ from the target. **Caveat:** this disruption creates highly unnatural local syntax immediately before the target. Failure to detect the predicted signature in D2 could reflect either the absence of forward-order sensitivity at close distances *or* breakdown of the LLM's prediction machinery on malformed local input. D2 is therefore designated as exploratory/secondary in the decision logic (§4).

### D3 — Swap halves

**Manipulation.** Place the originally-far-half tokens close to the target; place the originally-near-half tokens far from the target. Within-half order is preserved.

Disrupted ordering:
$$[t_M, t_{M-1}, \ldots, t_1, t_C, t_{C-1}, \ldots, t_{M+1}, \text{TARGET}]$$

This produces the largest predicted feature: the originally-closest token (a strong predictor of the target) now sits at distance $M+1$ rather than distance $1$, generating a large upward jump in the marginal at $d = M+1$.

### D4 — Full reverse

**Manipulation.** Reverse the entire context.

Disrupted ordering:
$$[t_1, t_2, \ldots, t_C, \text{TARGET}]$$

The originally-closest token is now most distant, and vice versa. D4 is the simplest and most decisive test of forward-direction sensitivity: if the pipeline is genuinely measuring forward-order-specific structure, the disrupted curve should look qualitatively unlike the intact power law.

### D5a–d — Foreign-block insertion family

**Manipulation.** Cut the context at $M$ and insert $K = 20$ foreign tokens between the two halves. Total disrupted context length is $C + K$. Four variants of the inserted block:

- **D5a** — same-topic foreign block (same-language donor document with high topic similarity).
- **D5b** — different-topic foreign block (same-language donor document with low topic similarity).
- **D5c** — same-document distant block (a distant passage from the same document, far enough from the target that it is not the actual sequential antecedent).
- **D5d** — shuffled / nonsense block (the same $K$ tokens drawn at random from the same document, then randomly shuffled — preserves token-level statistics but destroys all local coherence).

The graded comparison across D5a–d separates the topical, document-specific, sequential, and disruption-cost components of the long-range signal. Single-condition D5 is replaced by the family.

**`[v2]`** D1–D4 preserve the full multiset of context tokens. D5 variants do not preserve the original token multiset because they insert additional tokens; total disrupted context length is $C + K = 120$. They are *insertion* controls, not *permutation* controls. They test different things and are evaluated separately in §3 and §4. **`[v3]`** For D5 conditions only, context length in the analysis is evaluated from $c = 1$ to $c = C + K = 120$, so the post-insertion recovery segment ($d > M + K$) is fully sampled.

### Cut-point variation

D1 and D3 are run at three cut points: $M \in \{25, 50, 75\}$. The localization criterion (§4) requires that the detected discontinuity in the disrupted curve appear within $\pm 5$ tokens of the manipulated cut point in each case. This converts a single-position test into a localization test: wherever the disruption is placed, the pipeline should detect it there.

---

## 3. Pre-registered predictions and acceptance criteria

`[v2]` Predictions are stated as *directional and correlation-based*, not as pointwise-equality recoveries. Because the contribution of a token to LLM prediction depends on the surrounding context (not just its identity), exact remapping of intact-condition values to disrupted-condition positions overpredicts. The criteria below test whether the disrupted curve has the predicted *shape* and *similarity relationships*, with bootstrap CIs.

For all disruptions, "pass" requires that the qualitative prediction be visible *and* the quantitative criterion hold under bootstrap resampling. For all correlation-based criteria, scale-normalize curves before comparison (subtract mean, divide by SD).

| Disruption | Qualitative prediction | Quantitative criterion |
|---|---|---|
| **D0** No-op | Statistically indistinguishable from intact at the curve level; any pointwise deviations consistent with shuffle Monte Carlo noise | Mean absolute (D0 − intact) difference across $d \in [1, C]$ is less than 5% of the intact curve's mean magnitude, AND fewer than 5% of distances show a bootstrap 95% CI on (D0 − intact) excluding zero |
| **D1** Reverse far half | Near half tracks intact; discontinuity at $d = M+1$; far half tracks reversed intact far half | (a) Mean *signed* (D1 − intact) over $d \leq M$ has bootstrap 95% CI including zero, AND mean *absolute* (D1 − intact) over $d \leq M$ is less than 10% of intact near-half mean magnitude; (b) breakpoint detected at $\hat M_1 \in [M-3, M+3]$; (c) Spearman(D1 far half, reversed intact far half) − Spearman(D1 far half, intact far half) $> 0.3$ with bootstrap CI excluding zero |
| **D2** Reverse near half | Increasing curve $d = 1$ to $M$; far half tracks intact | (a) D2 near half slope significantly less negative (or positive) than intact near half; (b) Spearman(D2 near half, reversed intact near half) − Spearman(D2 near half, intact near half) $> 0.3$; (c) D2 far half tracks intact far half within bootstrap CI |
| **D3** Swap halves | Small/flat $d \leq M$; large upward jump at $d = M+1$; post-jump segment tracks intact near half | (a) Signed jump $\Delta_{M+1}^{D3} - \Delta_M^{D3} > 0$ with bootstrap 95% CI excluding zero, AND standardized jump $(\Delta_{M+1}^{D3} - \Delta_M^{D3}) / \mathrm{SD}(\Delta_d^{D3} : d \leq M) > 0.5$; (b) Spearman(D3 post-jump, intact near half) − Spearman(D3 post-jump, intact far half) $> 0.3$; (c) breakpoint detected at $\hat M_3 \in [M-3, M+3]$ |
| **D4** Full reverse | Disrupted curve mirrors reversed intact curve | (a) Spearman(D4, reversed intact) − Spearman(D4, intact) $> 0.3$ with bootstrap CI excluding zero; (b) Spearman(D4, reversed intact) $\geq 0.6$ |
| **D5a** Same-topic foreign | Plateau in $[M+1, M+K]$ above floor, below intact | Slope of $\Delta_d^{D5a}$ over $d \in [M+1, M+K]$ overlaps zero (95% CI); plateau level is a measurement, not a pass/fail (used in graded comparison below) |
| **D5b** Different-topic foreign | Plateau in $[M+1, M+K]$ at or near floor | Plateau level lower than D5a's plateau (one-sided bootstrap test) |
| **D5c** Same-document distant | Plateau in $[M+1, M+K]$ higher than D5a | Plateau level higher than D5a's plateau (one-sided bootstrap test) |
| **D5d** Shuffled/nonsense | Plateau intermediate; uses same tokens as D5a but shuffled within block | **Diagnostic only**, not pass/fail. The contrast D5a − D5d estimates the contribution of local coherence within the inserted block beyond topic-words alone |

**`[v3]` D5 family graded prediction (revised).** The pre-committed *primary* ordering of plateau levels is:
$$\text{D5c} \; > \; \text{D5a} \; > \; \text{D5b}$$

This is the test. **D5d is reported as a diagnostic estimate of the bag-of-words / topic-token contribution rather than as part of the strict pass/fail ordering.** Because D5d uses the same tokens as D5a (just shuffled), it can plausibly fall above D5b — different-topic foreign tokens may be *more* damaging than shuffled same-topic tokens. The contrast D5a − D5d specifically estimates the contribution of *local coherence within the inserted block* over and above the topic-words contribution; this difference is a measurement, not a pass/fail.

D5 passes the topic/document decomposition test if D5c $>$ D5a $>$ D5b in $\geq 5$ of 6 Wikipedia corpora (see §4.5 for the D5-specific aggregate rule).

**Post-insertion recovery (D5a–d).** For $d > M + K$ in each D5 condition, compare $\Delta_d^{D5}$ to $\Delta_{d - K}^{\text{intact}}$. The expectation is that the original far-half signal resumes but at shifted distances. Reported as a measurement; not pass/fail.

**`[v3]` Floor-limited / inconclusive handling.** Spearman correlations on flat or near-floor segments are unstable. For any segment-level correlation criterion in this table (D1 far half, D2 near half, D3 post-jump, D4 full curve), a segment is classified as *floor-limited* if its mean $\Delta_d$ across the segment is less than two bootstrap standard errors above zero. Floor-limited segments are not assigned pass/fail; they are reported as **inconclusive**. Inconclusive outcomes are reported transparently in the corpus-level table and do not count toward either the pass or fail tally in the aggregate rules in §4.5. This matters most for Finnish and Arabic, where the original Mistral fits showed weak calibration and far-half signal is closest to floor.

---

## 4. Decision rules

`[v2]` Pass/fail outcomes are organized into four functional groups, each addressing a distinct claim. This replaces v1's flat "all five must pass" structure.

### 4.1 Core order-sensitivity test (D3, D4 primary)

D3 and D4 are the strongest disruptions, and their predictions are the most decisive about whether the pipeline measures forward-directional structure.

**Pass:** Both D3 and D4 pass at the aggregate level (see §4.5 for aggregate rules).

**Interpretation if pass:** "The corrected-marginal measure is sensitive to forward order. The intact power-law decay is at least partly driven by ordered, direction-sensitive structure in the natural-language context, not a position-invariant property of the LLM probe."

**Interpretation if D4 fails:** The pipeline is not measuring forward-direction-specific structure. The original paper's claim about sequential influence cannot be sustained without substantial reframing. Stop and reconsider before further analyses.

**Interpretation if D3 fails but D4 passes:** Forward direction matters at the aggregate level, but the specific position-of-token effect predicted by halves-swap is not detected. Possible explanations include scale-of-effect issues or interaction effects in the LLM that prevent the predicted upward jump from appearing cleanly. Acknowledge in the paper.

### 4.2 Localization test (D1 + cut-point variation primary)

D1 at $M = 50$, plus D1 and D3 at $M \in \{25, 75\}$, test whether disruptions are detected at the manipulated position.

**Pass:** The detected breakpoint in the disrupted curve falls within $\pm 5$ tokens of the manipulated cut point in $\geq 75\%$ of corpus-cut-point combinations. The discontinuity moves with the manipulation.

**Interpretation if pass:** "The measure is positionally localized: disruptions introduced at a known distance are detected at that distance, not smeared across the curve."

**Interpretation if fail:** The pipeline detects that something is wrong with the disrupted context but cannot localize it. The position-specific interpretation of the intact curve weakens. Acknowledge.

### 4.3 Near-context interaction (D2 secondary)

D2 reverses the near half, creating unnatural local syntax. Failure of D2 is not interpreted as failure of forward-order sensitivity at close distances, because the unnatural-syntax confound is unresolvable in this design.

**If D2 passes:** Bonus — the near-half signal is forward-order-specific even when local syntax is preserved-but-reversed.

**If D2 fails:** Report neutrally. State that "near-context reversal creates strong local incoherence; failure of the predicted remapping under D2 does not distinguish between bag-of-words processing of near tokens and disruption of the LLM's local prediction machinery on malformed input." Do not infer bag-of-words.

### 4.4 Topic / sequence decomposition (D5 family)

The graded comparison D5c $\geq$ D5a > D5b $\geq$ D5d, plus the absolute plateau levels, decomposes the long-range signal into topical, document-specific, and disruption components.

**Pass on graded ordering:** The predicted ordering is observed in $\geq 6$ of 8 corpora.

**Interpretation when ordering passes:**

- If D5a plateau is small relative to intact $\Delta_d$ at the same distance: topic alone is a small contributor. Long-range signal is largely sequence-specific.
- If D5a plateau is comparable to intact $\Delta_d$: topic carries substantial signal; the autoregressive interpretation should be qualified.
- If D5c – D5a gap is large: there is document-specific structure beyond topic that the pipeline detects.
- If D5d is well above floor: even nonsense insertions interact with the LLM's local processing in ways that weakly remap to predictability — flag as a possible artifact.

**Interpretation when ordering fails:** Topic vs. sequence is not cleanly separable in the data. The paper's interpretation should soften to "the signal includes ordered-context structure but topic and sequence cannot be cleanly distinguished by this method."

### 4.5 Aggregate pass/fail rules across corpora and probes

`[v2]` Pre-committed aggregate rules across corpora and probes. **`[v3]`** D5 has a separate rule because it runs only on Wikipedia corpora, and the D1/D3 cut-point denominator is now made explicit.

**General rule (D0, D1, D2, D3, D4 at primary $M = 50$):**

- **Primary validation uses Llama-3-8B**, since it was the better-calibrated probe in the original analysis.
- **Mistral-7B serves as a robustness check.**
- A disruption *passes globally* if it passes in $\geq 6$ of 8 corpora under Llama-3-8B *and* in $\geq 5$ of 8 corpora under Mistral-7B.
- A meta-analytic random-effects model (corpus as random effect) is also fit for each disruption's primary criterion; the meta-analytic effect must be significant at $p < 0.01$.
- Conditions where Mistral fits in the original analysis were poor ($|r| < 0.80$) — Arabic, Finnish — are flagged in the report and downweighted in the Mistral aggregate.
- Inconclusive (floor-limited) outcomes are excluded from both numerator and denominator in these aggregate counts.

**`[v3]` D1/D3 cut-point variation aggregate rule.** The localization test combines D1 and D3 across $M \in \{25, 50, 75\}$ — six condition-cut combinations total per corpus per probe (D1 at three $M$, D3 at three $M$), giving $6 \times 8 = 48$ corpus × condition × cut combinations per probe. Localization passes if the detected breakpoint falls within $\pm 5$ tokens of the manipulated $M$ in $\geq 75\%$ of these combinations under Llama-3-8B.

**`[v3]` D5 family aggregate rule (six Wikipedia corpora only).** D5 passes the topic/document decomposition test if the primary ordering D5c $>$ D5a $>$ D5b holds in $\geq 5$ of 6 Wikipedia corpora under Llama-3-8B *and* $\geq 4$ of 6 under Mistral-7B, with meta-analytic effect $p < 0.01$. D5d's contribution to the contrast D5a − D5d is reported as a diagnostic estimate of within-block coherence and is not subject to the aggregate pass/fail rule.

---

## 5. Materials

The eight existing human corpora used in the original analysis: Wikipedia in 6 languages (Chinese, Japanese, Korean, Turkish, Arabic, Finnish; 60 articles per language), the Buckeye Corpus of Conversational Speech (English, 26 speakers), and the French Oral Narrative Corpus (87 stories, 17 storytellers).

Within each document, target regions are sampled at 25%, 50%, and 75% of document length. Target length 30 tokens. Maximum context length $C = 100$. Primary cut point $M = 50$; secondary cut points $M \in \{25, 75\}$ for D1 and D3 only.

### `[v2]` Topic-matching for D5

D5 variants for non-Wikipedia spoken corpora are deferred to a follow-up; the primary D5 analysis is run on the six Wikipedia corpora only, with results aggregated across languages.

For D5a (same-topic foreign) and D5b (different-topic foreign):

- **Embedding model:** `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers; supports the six target languages with reasonable quality). LaBSE is a fallback.
- **Within-language matching:** donors are drawn from the same language as the target document. No cross-lingual matching.
- **Similarity computation:** cosine similarity between mean-pooled paragraph embeddings.
- **D5a threshold:** donor paragraph similarity to target document $\geq 0.6$, donor document not the same as target document.
- **D5b threshold:** donor paragraph similarity to target document $\leq 0.2$.
- **No-donor handling:** if no paragraph in the corpus exceeds D5a threshold, the target is dropped from D5a/D5b analysis. Pre-committed: this should affect $< 5\%$ of targets per corpus; if it affects more, the threshold is documented and the corpus is reported with a footnote.

**`[v3]`** For D5c (document-specific nonlocal): donor block drawn from the same document as the target, with all of the following constraints to ensure the donor is *not* part of the local sequential antecedent:

- the donor block must not overlap any token in the 100-token original context preceding the target;
- the donor block must be at least 200 tokens from the target region (in either direction);
- the donor block must come from the *opposite half* of the document (if the target is in the first half, the donor is from the second half, and vice versa); and
- the donor block must not be immediately adjacent to the target paragraph.

This is labeled "document-specific nonlocal" rather than "not the actual sequential antecedent" because it remains within the same discourse and may share thematic material with the target.

For D5d (shuffled/nonsense): the same 20 tokens that would be used in D5a, randomly shuffled within the inserted block.

---

## 6. Probe and corrected-marginal computation

Two probe models: Meta-Llama-3.1-8B base (`unsloth/Meta-Llama-3.1-8B`) and Mistral-7B-v0.1 (`mistralai/Mistral-7B-v0.1`), both in 4-bit quantization (BitsAndBytes nf4) on a single H100 GPU. No fine-tuning. Identical to the original analysis. (Note: an earlier draft of this document and the manuscript Methods section incorrectly named the Llama variant as "Llama-3-8B-Instruct." The model actually used throughout the published analyses is the base model, version 3.1, via the unsloth mirror.)

### `[v2]` Condition-specific shuffled baselines (real technical fix)

For each condition $X \in \{\text{intact}, D0, D1, D2, D3, D4, D5a, D5b, D5c, D5d\}$ and each context length $c$, the corrected marginal at distance $d$ is:

$$\Delta_d^X = \left( m_d^{X, \text{ordered}} \right) - \left( m_d^{X, \text{shuffled}} \right)$$

where:

- $m_d^{X, \text{ordered}} = \text{ppl}_{c=d-1}^{X} - \text{ppl}_{c=d}^{X}$ on the disrupted context (or intact for $X = \text{intact}$).
- $m_d^{X, \text{shuffled}}$ uses a uniform random shuffle of the *exact* $c$-token prefix that condition $X$ reveals at context length $c$, **not** the intact prefix.

This is a substantive fix from v1. At small $c$, different disruptions reveal different prefixes — for example, at $c = 1$ in D3 the revealed token is the originally-far-half token at position $M$, whereas in intact it is position 1. The shuffled baseline must shuffle the same multiset that is being revealed in the ordered condition at that $d$, so that the subtraction isolates the order-effect within that condition's token revelation.

For D5 conditions, the inserted foreign tokens are part of the prefix being shuffled at distances after the insertion point. Pre-committed: shuffling pools the inserted tokens with the original tokens revealed up to that $c$. (Alternative — exclude inserted tokens from the shuffle pool — was considered and rejected; pooling preserves the principle that the shuffled baseline uses the same multiset as the ordered condition.)

For each condition and each $c$, 50 random shuffles are averaged to estimate $m_d^{X, \text{shuffled}}$ (matching the original protocol's shuffle-count).

---

## 7. Statistical procedure

For each disruption × corpus × probe combination:

1. **Bootstrap.** 1000 resamples at the document level (or speaker / storyteller level for spoken corpora). For each resample, recompute condition-specific shuffled baselines and disrupted/intact curves.
2. **Per-disruption similarity contrasts.** As specified per disruption in §3:
   - D4: Spearman(disrupted, reversed intact) vs. Spearman(disrupted, intact).
   - D1: Spearman over the far-half segment, against intact-far-half and reversed-intact-far-half.
   - D3: Spearman over post-jump segment, against intact near-half and intact far-half.
   - D2: Spearman over near-half segment, against intact near-half and reversed intact near-half.
3. **Breakpoint detection.** For D1 and D3 (and their cut-point variants), fit a piecewise model with a free breakpoint location and report $\hat M$ with bootstrap CI. Test whether $|\hat M - M| \leq 5$.
4. **D5 plateau extraction.** Estimate plateau level for $d \in [M+1, M+K]$ as the mean of $\Delta_d$ over that interval, with bootstrap CI. Test the predicted ordering across the four D5 variants.
5. **Aggregate meta-analysis.** Random-effects model with corpus as random effect, computing the meta-analytic estimate of each disruption's primary criterion across corpora.
6. **Aggregate pass/fail.** Apply the rules in §4.5.

Per-disruption, the primary contrast is what determines pass/fail. Other diagnostic statistics (point-by-point comparisons, residual plots, slope estimates per segment) are reported for transparency but are not part of the pass/fail logic.

---

## 8. Reporting plan

`[v2]` Both linear-axis and log-log plots are produced for each disrupted curve, since the predicted features (discontinuities, jumps, plateaus, increasing segments) are obscured on log-log axes.

Per disruption, per corpus, per probe:

- Linear-axis plot of $\Delta_d$ vs. $d$ for intact and disrupted, overlaid, with bootstrap 95% CI bands. Predicted-signature features annotated.
- Log-log plot for direct comparison to the original power-law claim.
- Quantitative-criterion table: predicted relation, observed value, 95% CI, pass/fail.

Aggregate (main figure):

- Multi-panel small-multiples grid: 7 disruption conditions (D0 + D1–D5d primary) × 8 corpora × 2 probes, on linear axes, intact and disrupted overlaid.
- Summary pass/fail table with aggregate rules applied.

For the cut-point variation (D1 and D3 at $M \in \{25, 50, 75\}$), an additional figure shows breakpoint location vs. manipulated cut point, with diagonal as the prediction. Localization passes if the points fall on the diagonal within the $\pm 5$ tolerance.

---

## 9. Pre-registration

A timestamped commit on the analysis repository (or OSF entry) before any disrupted analysis is run. Contents:

1. The seven disruption conditions (D0, D1, D2, D3, D4, D5a–d), exactly as specified in §2.
2. Cut points $M = 50$ primary, $M \in \{25, 75\}$ secondary (for D1, D3).
3. Foreign-block size $K = 20$.
4. Topic-matching specification (model, threshold, within-language matching, no-donor handling).
5. Condition-specific shuffled baseline procedure.
6. All quantitative criteria from §3.
7. Decision rules from §4 including aggregate rules.
8. Statistical procedure from §7.
9. A statement that no parameter or threshold will be changed after data are generated.

---

## 10. Effort and timeline

`[v2]` Updated for the expanded protocol.

| Task | Effort |
|---|---|
| Implement D0–D4 disruption functions | 1 day |
| Implement D5a–d with topic-matching infrastructure (multilingual sentence-transformer) | 3 days |
| Refactor pipeline for condition-specific shuffled baselines | 2 days |
| Add piecewise breakpoint-detection to model-comparison | 1 day |
| Run all conditions on all 8 corpora with both probes (10 conditions × 3 cut points for D1/D3 × 8 × 2) | 5 days (compute) |
| Bootstrap, similarity contrasts, meta-analysis | 2 days |
| Linear and log-log plots, tables, writeup | 3 days |
| Pre-registration document | 1 day |
| **Total** | **~3 weeks** |

The condition-specific shuffled baseline refactor is the largest unforeseen cost. If existing code assumed a shared baseline, this requires rewriting the inner loop. **`[v3]`** Pilot plots may be generated during implementation to verify code behavior, but no inferential pass/fail analyses will be run until the condition-specific shuffled baseline is implemented. The shared-baseline approximation is not pre-registered as an acceptable interim primary analysis; the condition-specific baseline is the correct procedure regardless of implementation difficulty.

---

## 11. What this experiment buys you

If the core order-sensitivity test (D3 + D4) passes globally, with D1 + cut-point localization passing, and D5 family showing the predicted graded ordering:

> "We tested whether the corrected-marginal measure is locally sensitive to known order disruptions in natural-language context. Five primary disruptions were applied to the prior context: reversal of the far half, reversal of the near half, swapping the two halves, full reversal, and insertion of foreign blocks of four kinds. A no-op control verified the absence of pipeline artifacts. Disruptions at three cut points ($M \in \{25, 50, 75\}$) tested positional localization. Across eight corpora and two probe models, the swap-halves and full-reverse disruptions produced the predicted directional features in the influence curve, with the disrupted curves more similar to reversed-intact than to intact under bootstrap test. Reversal of the far half produced a localized discontinuity at the manipulated cut point ($\pm 5$ tokens) in $X\%$ of corpus-cut-point combinations. The graded plateau across foreign-block variants — same-document distant $\geq$ same-topic foreign $>$ different-topic foreign $\geq$ shuffled — supports a topic-plus-sequence decomposition of the long-range signal in which sequential structure contributes a substantial component beyond topic. The corrected-marginal measure responds to the structural properties it is claimed to measure."

This is a defensible response to the central reviewer concerns about pipeline bias and instrument bias. It does not establish the theoretical claim about autoregressive generation, but it establishes the *measurement validity* on which any theoretical claim must rest.

If specific tests fail, the paper's claims sharpen rather than collapsing. Most likely partial outcomes — D2 ambiguous, D5a showing substantial topic contribution — are scientifically informative and lead to refined claims that survive review. If D4 fails, the experiment has saved the paper from submission with a fatal flaw. The remaining experiments in the master plan (multi-AI comparison, topic persistence controls, structural-boundary analysis) further constrain the interpretation.

---

## Appendix B — `[v3]` Summary of changes from v2

The v3 revision implements ten polish edits before pre-registration lock, all responsive to a second review pass that approved the v2 conceptual restructuring:

1. D5 language clarified: insertion (not replacement) of $K = 20$ tokens; analysis runs to $c = C + K = 120$ for D5 conditions.
2. Removed the interim shared-baseline fallback from §10. The condition-specific shuffled baseline is the correct procedure regardless of implementation cost.
3. D3 jump threshold made explicit: signed jump > 0 with bootstrap CI excluding zero AND standardized jump > 0.5 (replaces the underspecified "corpus-specific bootstrap-derived threshold").
4. Floor-limited / inconclusive handling added for low-SNR correlation tests. Segments with mean $\Delta_d$ < 2 bootstrap SE above zero are reported as inconclusive, not failed.
5. D0 criterion revised to curve-level equivalence (mean abs. diff < 5% of intact magnitude; < 5% of distances showing CI excluding zero), replacing the brittle "every distance" criterion.
6. D1 near-half criterion uses signed difference (CI including zero) plus practical-equivalence (mean abs. diff < 10%), replacing the conceptually awkward "abs. diff CI of zero."
7. D5 ordering loosened: primary test is D5c > D5a > D5b. D5d is reported as a diagnostic estimate of within-block coherence (D5a − D5d), not as part of the strict pass/fail ordering.
8. Aggregate rules separated: the general rule covers D0–D4 across 8 corpora; D1/D3 cut-point variation has its own rule (≥75% of 48 combinations under Llama); D5 has a separate Wikipedia-only rule (≥5/6 corpora under Llama, ≥4/6 under Mistral).
9. D5c donor restrictions made concrete: no overlap with original 100-token context, ≥200 tokens from target, opposite half of document, not adjacent to target paragraph. Renamed "document-specific nonlocal."
10. "Addresses both concerns" softened to "directly addresses the instrument-bias concern and provides a naturalistic check against one form of pipeline bias; synthetic validation remains necessary for a full pipeline-bias test." Strongest-claim language softened to "at least partly driven by ordered, direction-sensitive structure."

---

## Appendix A — `[v2]` Summary of changes from v1

1. Renamed to "Naturalistic disruption validation of the context-influence measure"; explicit positioning as Experiment 1A, not a replacement for synthetic validation.
2. Added D0 no-op control.
3. Expanded D5 into family of four variants (D5a–d): same-topic, different-topic, same-document distant, shuffled/nonsense. Graded ordering as primary test.
4. Added cut-point variation: D1 and D3 run at $M \in \{25, 50, 75\}$ for localization testing.
5. Replaced pointwise-equality predictions with directional/correlation-based criteria (Spearman contrasts; bootstrap CIs).
6. **Critical fix:** condition-specific shuffled baselines, replacing the v1 assumption that shuffling destroys order so the same baseline applies across conditions.
7. Restructured decision rules into four functional groups: core order-sensitivity (D3, D4 primary), localization (D1 + cut points), near-context interaction (D2 secondary, with unnatural-syntax caveat), topic/sequence decomposition (D5 family).
8. Added aggregate pass/fail rules across corpora and probes ($\geq 6/8$ for Llama, $\geq 5/8$ for Mistral, with meta-analytic significance).
9. Added linear-axis plots alongside log-log to make discontinuities and increasing segments visible.
10. Specified multilingual topic-matching: `paraphrase-multilingual-mpnet-base-v2`, within-language matching only, D5 limited to Wikipedia for first run.
11. Updated effort estimate (~3 weeks) reflecting expanded suite, condition-specific baselines, and localization runs.
12. Reframed §1 to acknowledge what the experiment does *not* establish: human production mechanisms, theoretical interpretation, complete topic exclusion. Establishes measurement validity, not theory.
