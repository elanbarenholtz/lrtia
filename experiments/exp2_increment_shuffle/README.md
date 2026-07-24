# Exp 2 — Increment-shuffle persistence, Q(d)

Measures the **span-level order contribution at distance**: does destroying the
internal order of the band at distance ~d, with everything nearer than d left
intact, degrade prediction of the target? This is the direct span-level analogue
of the headline (marginal) persistence function P(d).

## Definition

For each adjacent pair on the log-spaced ladder `(c_prev, c_cur)`:

- **A (intact):** true preceding `c_cur` tokens, original order.
- **B (increment-shuffled):** distances `1..c_prev` intact and in place;
  distances `c_prev+1..c_cur` permuted uniformly within that band only.

```
Q(d_i) = ( mean_k ppl_B(k) - ppl_A ) / (c_i - c_{i-1}),   d_i = sqrt(c_prev * c_cur)
```

Positive Q(d) ⇒ span-internal order at distance ~d matters for the target.

## Files

- `lrtia/intervention/increment_shuffle.py` — canonical, unit-tested transform
  (`increment_shuffle_prefix`, null controls, `stable_seed`, `ladder_pairs`).
  Kept separate from `transforms.py`; the main CPF paths are untouched.
- `tests/test_increment_shuffle.py` — 16 tests (seed determinism, near-band
  invariance, far-band permutation, single-token identity, controls, geometry).
- `experiments/exp2_increment_shuffle/run_exp2.py` — GPU runner. Model-agnostic
  scoring core `compute_exp2_curves(...)` (unit-testable with a stub scorer)
  plus HF probe + document resolver mirroring the main long-range notebook.
- `NHB_submission/code/probes/Corpus_Expansion_Exp2_IncrementShuffle_Llama.ipynb`
  — self-contained Colab runner mirroring the sibling probes.
- `analysis/exp2_increment_shuffle_qd.py` — aggregates Q(d), bins by ladder
  interval (same binning as P(d)), log-log fit β with document-bootstrap CIs,
  compares β to the corpus's α from P(d), reports both null controls.

## Running

Colab: open the notebook, `WITH_CONTROLS=True`, `K_SHUFFLES=20`.

Local / GPU box (priority subset, controls on):
```
python experiments/exp2_increment_shuffle/run_exp2.py --drive /path/to/LRTIA --k 20
```
Headline compute budget only (no controls, ~231 passes/target):
```
python experiments/exp2_increment_shuffle/run_exp2.py --drive /path/to/LRTIA --k 20 --no-controls
```
Full ten-corpus set:  add `--full`.
Mistral probe-independence check (one corpus):
```
python experiments/exp2_increment_shuffle/run_exp2.py --drive /path/to/LRTIA \
  --model mistralai/Mistral-7B-v0.1 --corpora ted_transcripts_en --k 20
```
Smoke test on a few docs:  add `--limit 5`.

Analysis (no GPU):
```
python analysis/exp2_increment_shuffle_qd.py \
  --results results/exp2_increment_shuffle/llama \
  --main-results results/corpus_expansion_longrange/llama --boot 1000
```

## Predictions (pre-registered)

- **H1:** Q(d) > 0 across the range, heavy-tailed, no cutoff. Strong form β ≈ α.
- **H2 (dissociation):** Q(d) → 0 at large d while P(d) stays positive — distant
  spans contribute through content-in-ordered-block interaction, not span-internal
  order. Meaningful outcome, not a failure; would reframe the headline claim as
  marginal/block-level.

Null controls (built into the runner): near+far shuffle should collapse Q toward
the P(d)-style marginal; random-token increment isolates ordered content from
mere content presence.

## Output layout

Per-target JSON at `results/exp2_increment_shuffle/<probe>/<corpus>.json`,
mirroring `results/corpus_expansion_longrange/` so figure/stat code is reusable.
Each record carries the intact `ordered_ppl`/`ordered_nll` curve and a `pairs`
list with per-band `A_ppl`, `B_ppl` (K values) and `B_ppl_mean`, plus control
means and the per-(target, c, k) `seeds`.
