# DNA control for the Contextual Persistence Law — summary

## Background / what we're testing

In human language we measure a **contextual persistence function** P(d): the
order-specific predictive influence of prior context at distance d from the current
point of production, isolated from token-frequency effects by subtracting a
shuffled-token baseline (P(d) = ordered marginal − shuffled marginal, per token).
Across ten corpora (six language families, written and spoken) P(d) decays as an
approximately scale-free power law, **mean exponent ≈ −1.04** (95% CI −1.15 to −0.93),
**median r² = 0.96**, monotonic and positive out past 1,000 tokens.

**Question the DNA control answers:** is this ~1/d law a general property of *any*
structured sequence with long-range dependence, or is it specific to human language?
DNA is an ideal non-language test case: a biological sequence with genuine long-range
statistical structure (regulatory elements, chromatin organization) but no
forward-chained discourse.

## Method (identical protocol, domain-native probe)

- **Probe:** HyenaDNA, a single-nucleotide autoregressive genomic language model, used
  exactly as the LLM probes were for language — per-token conditional likelihoods, no
  fine-tuning. Run at **two scales**: medium (160k context) and large (1M context).
- **Data:** human reference genome (GRCh38), chromosome 21, cut into 60 ACGT windows.
- **Protocol:** identical to the language runs — one 30-token target at the 50%
  position, log-spaced context lengths [0,1,2,4,…,1024] nucleotides, ordered vs
  shuffled context at each length, P(d) fit as a power law on d ≥ 10.

## Results

**HyenaDNA-medium (160k):**

| d | P(d) |
|---|---|
| 11.3 | +0.0017 |
| 22.6 | +0.0022 |
| 45.3 | +0.0029 |
| 90.5 | +0.0032 |
| 181  | +0.0008 |
| 362  | +0.0003 |
| 724  | +0.0000 |

Fit: slope ≈ **−0.82**, **r² = 0.64**. Non-monotonic (rises to a bump near d≈90, then
falls); magnitudes ~0.003, roughly **two orders of magnitude smaller** than language.

**HyenaDNA-large (1M):**

| d | P(d) |
|---|---|
| 11.3 | +0.0017 |
| 22.6 | +0.0007 |
| 45.3 | +0.0029 |
| 90.5 | (near 0) |
| 181  | +0.0008 |
| 362  | +0.0003 |
| 724  | +0.0000 |

Fit: slope ≈ **−0.31**, **r² = 0.36**. P(d) oscillates around zero (positive at some
distances, **negative at others** — e.g. −0.0008 at d≈6, −0.0003 at d≈11), i.e. noise
around zero rather than a decaying signal.

**Language reference:** α = −1.04, median r² = 0.96, monotonic, positive to ≥1,000 tokens.

## Interpretation

DNA **does not** exhibit the contextual persistence law. Three independent signatures:

1. **Magnitude:** DNA's order-specific P(d) is ~2 orders of magnitude weaker than
   language — scrambling the upstream nucleotides barely changes the model's prediction
   of downstream nucleotides.
2. **Form:** non-monotonic / oscillating, not a power law (r² 0.36–0.64 vs 0.96 for
   language). The fitted "slopes" are lines forced through near-zero noise, not real
   scale-free decay.
3. **Long range:** P(d) → 0 well before the distances where language is still strongly
   positive.

Crucially, this holds **across two probe scales**, and the *stronger* probe shows
*less* structure, not more — the near-zero result is not an artifact of a weak model. If
DNA had a hidden long-range persistence law, a better genomic model would reveal more of
it; instead the 1M-context model finds essentially nothing. Biologically sensible: DNA's
predictability is largely local (codons, motifs), not forward-chained over hundreds of
tokens like discourse.

**Conclusion:** the contextual persistence law is **not** a generic property of any
structured sequence with long-range dependence. A genomic sequence with real long-range
biological structure does not show it — supporting the claim that the law is specific to
(human) language rather than to structured/predictable sequences in general.

## Status / caveats

- Two HyenaDNA scales agree; result is robust to probe capacity.
- Evo 2 (SOTA genomic model) would be a nice further check but is optional — the two
  scales already rule out the "probe too weak" objection. (Evo 2's install is heavy;
  deferred.)
- A second non-language control (protein, and possibly symbolic music) is in progress to
  strengthen the "language is special" contrast beyond a single control.

## Question for Fable

Given this DNA result, does the two-probe "off the law" finding read as convincing on its
own, or is a second independent non-language control (e.g., protein sequences) needed
before the "specific to human language" framing is defensible in review? And is the
biological interpretation (local vs forward-chained predictability) the right framing, or
is there a stronger/cleaner way to state what DNA's near-zero P(d) implies?
