# Exp 2: Increment-shuffle persistence (span-level order contribution at distance)

Status: PREREGISTERED, not yet run. Handoff spec — written 2026-07-20, before any data collection.

## Question

The headline CPF, P(d), is a **marginal** measure: the order-specific gain in target
prediction as the revealed context block is extended past distance d, with the shuffled
baseline permuting the **entire** revealed span. It therefore measures how block-level
ordered context accrues influence with extension. It does not measure whether the
**internal order of a specific span at distance d** influences prediction when everything
nearer than d is left intact — the span-level analogue, and the more direct reading of
"order-specific influence at distance d."

This experiment measures that quantity, Q(d), directly.

## Design

Reuse the CPF pipeline (Methods, "Contextual persistence function") with one new condition.

Targets: identical selection to the main experiment — single 30-token target region at the
50% point of each document; documents require ≥ 1024 + 30 + 50 tokens. Reuse the existing
target manifest where available (`results/corpus_expansion/targets_llama.jsonl` for
expansion cells) or regenerate with the same procedure and seed.

Context lengths: same log-spaced ladder c ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}.

Conditions, per target and per adjacent pair (c_{i-1}, c_i):

- **A (intact):** the true preceding c_i tokens, original order. (Identical to the intact
  condition of the main experiment; recompute rather than reuse caches so A and B share
  tokenization and code path.)
- **B (increment-shuffled):** tokens at distances 1 … c_{i-1} intact and in place; tokens at
  distances c_{i-1}+1 … c_i permuted uniformly at random **within that span only**, in place.
  Target never permuted. No special/boundary tokens introduced.

Permutations: K = 20 independent permutations per (target, i), seeds fixed and logged
(seed = hash(document_id, target_id, c_i, k) for reproducibility), same permutations across
probe models.

## Measure

Per-token span order contribution at distance band i:

    Q(d_i) = ( mean_k ppl_B(k) − ppl_A ) / (c_i − c_{i-1}),   d_i = sqrt(c_{i-1} · c_i)

Positive Q(d) ⇒ destroying the internal order of the span at distance ~d, with intact
nearer context, degrades prediction of the target: span-level order matters at that
distance. Report both perplexity units (main) and log-probability/nats (robustness), as in
the main paper.

## Corpora and probes

Priority subset (cleaned text committed in this repo under
`data/corpus_expansion/clean/`): `ted_transcripts_en`, `gutenberg_fiction_en`,
`ted_transcripts_ru`. Extend to the remaining Fig-1 cells if compute allows:
`news_en`, `ted_transcripts_de`, `ted_transcripts_fr`, `ted_transcripts_tr` (in repo);
`buckeye` (LICENSE-RESTRICTED — not in repo, obtain locally; never commit),
`ja_literary` (Aozora), `fi_literary` (not on this machine — rebuild via
`NHB_submission/code/ingest/` or copy from GPU box).

Probe: Llama-3.1-8B (base), as in main experiment. Replicate one corpus
(ted_transcripts_en) on Mistral-7B-v0.1 as a probe-independence check.

## Analysis

1. Aggregate Q(d) across documents within corpus; bin by distance (same binning as P(d)).
2. Log–log fit Q(d) ∝ d^(−β) over the same distance range used for α; bootstrap CIs over
   documents (reuse `analysis/bootstrap_exponents.py` machinery).
3. Compare β to the corpus's α from P(d).
4. Null: repeat with the near context (1 … c_{i-1}) ALSO fully shuffled — Q should
   collapse toward the P(d)-style marginal; and a random-token increment control.

## Predictions (stated in advance)

- H1: Q(d) > 0 across the measured range, decaying heavy-tailed with no characteristic
  cutoff. Strong form: β ≈ α (near 1) — span-internal order carries the persistence law.
- H2 (dissociation): Q(d) → 0 at large d while P(d) remains positive — distant spans
  contribute through content-in-ordered-block interaction, not span-internal order. This
  would require reframing the headline claim as marginal/block-level; it is a meaningful
  outcome, not a failure.

## Compute estimate

Per target: 11 context lengths × (1 A + 20 B) ≈ 231 forward passes, ≤ 1054 tokens each.
Priority subset ≈ 300 documents ⇒ ~7 × 10^4 passes; single A100/H100, well under a day.
Full ten corpora (449 docs): scale accordingly.

## Repo notes for the runner

- Pipeline code: `lrtia/` package (`intervention/span_selection.py`,
  `intervention/transforms.py` are the natural home for the increment-shuffle transform).
- Write outputs to `results/exp2_increment_shuffle/<probe>/` as per-target JSON, mirroring
  `results/corpus_expansion/` layout, so figure/stat code can be reused.
- Do not modify the main CPF code paths; add the new condition alongside.
