# Claude Code implementation instructions: Experiment 1 (naturalistic disruption validation)

You are implementing the experimental pipeline for a methodological validation study of an LLM-probe-based measurement of sequential influence in human language. The full protocol — including all generators, criteria, decision rules, and pre-registration requirements — is specified in:

```
/Users/elanbarenholtz/Projects/lrtia/paper/exp1_structured_disruption.md
```

(If that path is wrong, search the user's project for `exp1_structured_disruption.md` and confirm before proceeding.)

**Read that protocol document end-to-end before writing any code.** It is the authoritative specification. These instructions tell you how to implement it; the protocol tells you *what* to implement. Where the protocol and these instructions disagree, the protocol wins. If something is genuinely ambiguous, stop and ask the user.

## Critical rules

1. **Pre-registration gate.** No inferential pass/fail analysis runs until the pre-registration commit is tagged. You can write code, generate plots from synthetic data, and verify the pipeline produces well-formed outputs at any time, but the actual disrupted-vs-intact comparison on real corpora does not run until the user explicitly approves locking the pre-registration document. This is non-negotiable.
2. **Condition-specific shuffled baselines are the correct procedure.** Do not implement a shared shuffled baseline across conditions, even temporarily, even if the existing pipeline is structured around one. If the refactor is hard, the answer is "do the refactor," not "use the wrong baseline as an interim approximation."
3. **No silent parameter changes.** Every parameter, threshold, and hyperparameter named in the protocol must match exactly. If you discover during implementation that a value should be different, raise it with the user before changing — it may need a pre-registration update.
4. **Track progress with TodoWrite.** This is a multi-phase project. Use the todo list aggressively; mark items in_progress when starting and completed when finishing.

## Phase 0: Environment and inventory

Goal: understand what already exists before writing anything new.

1. Locate the existing analysis pipeline. Look for:
   - the corpora (Wikipedia in 6 languages, Buckeye, French Oral)
   - the existing probe code (Llama-3-8B-Instruct, Mistral-7B-v0.1)
   - the existing corrected-marginal computation
   - the existing curve-fitting / model-comparison code
   - any existing plotting or table-generation code
2. Document what you found in a short file `IMPLEMENTATION_NOTES.md` at the project root. List file paths, the public API of each module, and any departures from the protocol you observe in the existing code.
3. Confirm the Python environment has, or install:
   - `transformers`, `torch`, `accelerate`, `bitsandbytes` (for 4-bit probes)
   - `numpy`, `scipy`, `pandas`, `scikit-learn`
   - `sentence-transformers` (for D5 topic matching)
   - `statsmodels` (for piecewise / breakpoint detection and meta-analysis)
   - `matplotlib`, `seaborn` (for plots)
   - `pytest` (for sanity tests)
4. Verify GPU availability (`torch.cuda.is_available()`). All probe runs require an H100 or equivalent. If no GPU is available, stop and tell the user.
5. Do **not** modify the existing pipeline yet. The first principle of refactoring is to know what you are starting from.

Verification gate: do not proceed until `IMPLEMENTATION_NOTES.md` exists, the user has confirmed it accurately describes the existing code, and the environment imports cleanly.

## Phase 1: Project structure

Create the following directory under the project root (path: ask the user; assume `experiments/exp1_disruption/` if not specified):

```
exp1_disruption/
├── PREREG.md                      # pre-registration document (Phase 7)
├── IMPLEMENTATION_NOTES.md        # from Phase 0
├── README.md                      # how to reproduce
├── disruptions/
│   ├── __init__.py
│   ├── core.py                    # D0-D4 permutation disruptions
│   ├── insertion.py               # D5a-d insertion disruptions
│   └── topic_matching.py          # multilingual sentence-transformer infra
├── pipeline/
│   ├── __init__.py
│   ├── probe.py                   # Llama / Mistral probe wrapper
│   ├── marginals.py               # corrected-marginal with condition-specific baselines
│   └── runner.py                  # orchestrates conditions × corpora × probes
├── stats/
│   ├── __init__.py
│   ├── bootstrap.py               # 1000 resamples at document level
│   ├── breakpoint.py              # piecewise breakpoint detection
│   ├── correlations.py            # Spearman contrasts with bootstrap CIs
│   └── meta.py                    # random-effects meta-analysis
├── plots/
│   ├── __init__.py
│   ├── per_condition.py           # linear + log-log per disruption per corpus
│   ├── localization.py            # cut-point variation: detected vs manipulated M
│   └── aggregate.py               # multi-panel small-multiples grid
├── tests/
│   ├── test_disruptions.py        # unit tests for the disruption transforms
│   ├── test_baselines.py          # unit tests for condition-specific baselines
│   └── test_d0_no_op.py           # D0 must produce intact-equivalent output
├── config/
│   └── exp1.yaml                  # all hyperparameters in one place
└── scripts/
    ├── 01_run_pipeline.py         # full experiment runner
    ├── 02_run_stats.py            # statistical analysis
    └── 03_make_figures.py         # all plots and tables
```

Add an `__init__.py` everywhere so it imports as a package.

## Phase 2: Disruption transforms

Implement the seven disruption conditions in `disruptions/core.py` (D0–D4) and `disruptions/insertion.py` (D5a–d). Each disruption is a function taking a token sequence (the 100-token original context, ordered from most distant to immediately-before-target) and returning the disrupted token sequence.

Use this convention throughout: positions are indexed by *temporal distance from target*. Position 1 = immediately before target, position 100 = most distant. The full context as a list `ctx` has `ctx[0]` = most distant token, `ctx[-1]` = closest. The cut point `M = 50` divides the context into a near half (`ctx[-M:]`) and a far half (`ctx[:-M]`).

Function signatures:

```python
def d0_no_op(ctx: List[int], M: int = 50) -> List[int]:
    """Cut at M, rejoin unchanged. Sanity check on the code path."""
    near = ctx[-M:]
    far = ctx[:-M]
    return far + near  # identical to ctx

def d1_reverse_far(ctx: List[int], M: int = 50) -> List[int]:
    """Reverse far half; keep near half intact."""
    near = ctx[-M:]
    far = ctx[:-M]
    return list(reversed(far)) + near

def d2_reverse_near(ctx: List[int], M: int = 50) -> List[int]:
    """Reverse near half; keep far half intact."""
    near = ctx[-M:]
    far = ctx[:-M]
    return far + list(reversed(near))

def d3_swap_halves(ctx: List[int], M: int = 50) -> List[int]:
    """Swap halves: near goes far, far comes close."""
    near = ctx[-M:]
    far = ctx[:-M]
    return near + far  # near now at far positions, far now at near positions

def d4_full_reverse(ctx: List[int]) -> List[int]:
    """Reverse the entire context."""
    return list(reversed(ctx))

def d5_insert_block(
    ctx: List[int], block: List[int], M: int = 50
) -> List[int]:
    """Insert a foreign block of K tokens between near and far halves.
    Resulting length is len(ctx) + len(block)."""
    near = ctx[-M:]
    far = ctx[:-M]
    return far + block + near
```

Write `tests/test_disruptions.py` covering:
- D0 produces a token sequence identical to the input.
- D1 leaves the last `M` tokens unchanged and reverses the first `len(ctx) - M` tokens.
- D2 leaves the first `len(ctx) - M` tokens unchanged and reverses the last `M`.
- D3 swaps halves cleanly: `d3(ctx)[:M] == ctx[-M:]` and `d3(ctx)[M:] == ctx[:-M]`.
- D4 produces `list(reversed(ctx))`.
- D5 produces a context of length `len(ctx) + len(block)` with the block inserted at the cut.
- All disruptions preserve the multiset of input tokens for D0–D4; D5 adds the block tokens.

Run the tests. They must all pass before continuing.

## Phase 3: Topic-matching infrastructure (for D5)

In `disruptions/topic_matching.py`:

1. Use `sentence-transformers` with model `paraphrase-multilingual-mpnet-base-v2`. Cache it locally.
2. For each Wikipedia corpus (six languages), compute paragraph-level mean-pooled embeddings for every paragraph in every document.
3. Build per-corpus indices to support:
   - **D5a (same-topic foreign):** given a target document and target region, return a paragraph from a *different* document in the same language with cosine similarity ≥ 0.6 to the target document's mean embedding.
   - **D5b (different-topic foreign):** same, but similarity ≤ 0.2.
   - **D5c (document-specific nonlocal):** given a target document and target region, return a paragraph from the *same* document, with all of: not overlapping the 100-token original context; ≥200 tokens from the target region; from the opposite half of the document; not adjacent to the target paragraph. Implement these constraints as filters.
   - **D5d (shuffled/nonsense):** given a D5a block, return its tokens in random order.
4. Donor blocks must be exactly K=20 tokens. If a candidate paragraph is longer, take a contiguous 20-token slice; if shorter, concatenate adjacent paragraphs from the donor and take a 20-token slice.
5. **No-donor handling:** if no candidate paragraph satisfies D5a's constraints, drop the target from D5 analysis. Pre-committed: this should affect <5% of targets per corpus. Log the exclusion rate per corpus. If any corpus exceeds 5%, raise to the user.
6. **D5 runs only on the six Wikipedia corpora.** Buckeye and French Oral are excluded from D5 in this experiment.

Write a smoke test that, for one Wikipedia corpus, builds the index and confirms that:
- ≥95% of targets find a valid D5a donor
- 100% of targets find a valid D5c donor (it should always exist given a 2000-token document)
- D5b candidates are findable

Run the test before continuing.

## Phase 4: Pipeline refactor — condition-specific shuffled baselines

This is the most consequential refactor. Read carefully.

In `pipeline/marginals.py`, implement the corrected-marginal computation with **condition-specific** shuffled baselines.

For each combination of (target, condition, context_length c):

1. Get the disrupted full context for this condition (apply the disruption function from Phase 2 to the original 100 tokens, plus the inserted block for D5 conditions).
2. The "ordered" context at length c is the *last c tokens* of the disrupted full context. Compute target perplexity given this ordered prefix.
3. The "shuffled" baseline at length c is constructed by taking *the same c tokens* (the multiset that the ordered prefix contains at length c), randomly permuting them, and computing target perplexity on that permuted prefix. Repeat 50 times and average the perplexity across permutations.
4. The marginal at distance d is: `m_d = ppl[c=d-1] - ppl[c=d]`, computed separately for ordered and shuffled.
5. The corrected marginal at distance d for this condition is: `Delta_d = m_d_ordered - m_d_shuffled`.

The key correctness condition: **at every (condition, c), the shuffled baseline shuffles the c tokens that the ordered prefix at that c contains for that condition.** Different conditions reveal different prefixes at the same c; their shuffled baselines must use the appropriate token multiset for each.

For D5 conditions only, c ranges from 1 to C+K = 120 (since the disrupted context is 120 tokens long). For D0–D4, c ranges from 1 to 100.

Implementation hints:
- A clean way to structure this: a function `compute_corrected_marginal(target, ordered_full_ctx, probe, n_shuffles=50, seed=...)` where `ordered_full_ctx` is whatever the disrupted full context is for the condition. This function does not need to know about the disruption identity — it just needs the disrupted context. The shuffle is performed on the prefix at each c.
- Pre-compute the 50 shuffled permutations once per (condition, document, target) and reuse across c-values to save compute, but make sure each shuffled prefix at length c is a permutation of the ordered prefix at length c.
- Make the random seed for shuffles deterministic and document the seed in the run config.

Write `tests/test_baselines.py` verifying:
- For D0 (no-op), the corrected marginal must match the existing pipeline's output for intact context, within numerical tolerance.
- For an intact context, computing the corrected marginal twice with the same seed gives bit-identical output.
- For different conditions, the shuffled baseline at c=1 reflects the different first-revealed token (smoke test: shuffle a single token returns that token, so the marginal at c=1 should depend only on which single token is revealed).
- The corrected marginal at c=0 is zero (no context, nothing to subtract).

This phase is done when D0 produces a corrected-marginal curve that matches the existing pipeline's intact curve.

## Phase 5: Experiment runner

`scripts/01_run_pipeline.py` orchestrates the full experiment.

For each (corpus, probe, condition, target_position):
- Apply the disruption to the original 100-token context.
- Compute the corrected marginal at every distance from 1 to C (or C+K for D5).
- Store the result in a structured output (Parquet recommended).

Conditions to run:
- intact (reference)
- D0 (sanity)
- D1 at M ∈ {25, 50, 75}
- D2 at M = 50
- D3 at M ∈ {25, 50, 75}
- D4 (full reverse)
- D5a, D5b, D5c, D5d (each at M = 50, K = 20) — Wikipedia corpora only

For each corpus, sample target regions at 25%, 50%, 75% of document length, target length 30 tokens, matching the original protocol.

Probes:
- Llama-3-8B-Instruct (4-bit)
- Mistral-7B-v0.1 (4-bit)

Output schema (one row per `(corpus, probe, condition, document_id, target_position, distance)`):

```
corpus           str
probe            str
condition        str            # 'intact', 'D0', 'D1_M25', 'D1_M50', ..., 'D5d'
document_id      str
target_position  str            # '25', '50', '75'
distance         int
delta_d          float          # corrected marginal
m_ordered        float          # raw ordered marginal
m_shuffled       float          # raw shuffled marginal (averaged across permutations)
ppl_intact       float          # raw target perplexity at this c
ppl_shuffled     float          # raw target perplexity at this c (averaged)
seed             int
```

Save to `exp1_disruption/results/raw_marginals.parquet`. Compute time will be substantial (estimate: ~5 days on a single H100 for the full grid). Implement checkpointing so the run can resume from the last completed (corpus, probe, condition).

Before launching the full run, do a smoke test on one corpus, one probe, one condition, three documents. Verify the output schema is correct and the values are in the expected range. Show the user a sample of the output. Get explicit approval before launching the full grid.

## Phase 6: Statistical analysis

`scripts/02_run_stats.py` consumes `raw_marginals.parquet` and produces:

1. **Bootstrap.** 1000 document-level resamples per (corpus, probe, condition). Aggregate `delta_d` across documents and target positions within each resample.
2. **Per-disruption similarity contrasts.** Implement in `stats/correlations.py`:
   - D4: Spearman(D4 curve, reversed intact curve) − Spearman(D4 curve, intact curve), with bootstrap 95% CI.
   - D1: same logic on the far-half segment (d > M) — D1 far half vs reversed intact far half vs intact far half.
   - D3: Spearman(D3 post-jump segment, intact near half) − Spearman(D3 post-jump, intact far half).
   - D2: Spearman(D2 near half, reversed intact near half) − Spearman(D2 near half, intact near half).
3. **Floor-limited handling.** For each segment-level correlation test, classify the segment as floor-limited if its mean `delta_d` < 2 × bootstrap SE above zero. Floor-limited segments are reported as `inconclusive`, not pass/fail.
4. **Breakpoint detection.** For D1 and D3 (and their cut-point variants), fit a piecewise-with-free-breakpoint model and report the estimated breakpoint location with bootstrap CI. Implement in `stats/breakpoint.py`. A simple approach: grid-search the breakpoint location, fit power-law segments before and after, choose the location minimizing residual SS. Report whether |estimated_M − manipulated_M| ≤ 5.
5. **D5 plateau extraction.** For each D5 variant, compute the mean `delta_d` over `d ∈ [M+1, M+K]` with bootstrap 95% CI. Test the predicted ordering D5c > D5a > D5b across corpora.
6. **Aggregate meta-analysis.** Random-effects model with corpus as random effect, computing the meta-analytic estimate of each disruption's primary criterion. Use `statsmodels` or equivalent. Implement in `stats/meta.py`.
7. **Pass/fail tabulation.** Apply the rules in §4.5 of the protocol document. Produce a per-corpus pass/fail table and an aggregate pass/fail summary. **Inconclusive outcomes are excluded from both numerator and denominator** in the aggregate counts (per the v3 floor-limited handling rule).

Output: `exp1_disruption/results/stats_summary.parquet` and a per-disruption JSON with all the contrast statistics.

## Phase 7: Plots and tables

`scripts/03_make_figures.py`. All plots must save as both PNG (high-DPI) and PDF.

Per (disruption, corpus, probe):
- Linear-axis plot of `delta_d` vs `d`, intact and disrupted overlaid, with bootstrap 95% CI bands. Annotate the predicted-signature features (cut location, predicted plateau, etc.).
- Log-log plot of the same data.

Aggregate (main figure for the paper, eventually):
- Multi-panel small-multiples: rows = disruptions, columns = corpora, with both probes. Shared y-axis where possible.
- Linear axes are primary because the disruption signatures (jumps, plateaus, increasing segments) are obscured on log-log.

Localization figure:
- Detected breakpoint vs manipulated M, scatter plot, one point per (corpus, probe, M ∈ {25, 50, 75}, condition ∈ {D1, D3}). Diagonal as the prediction. ±5-token tolerance bands.

Quantitative-criterion table:
- Per disruption, per corpus, per probe: predicted relation, observed value, 95% CI, pass/fail/inconclusive.

D5 family figure (Wikipedia only):
- For each of the six Wikipedia corpora, the four D5 plateau levels with 95% CI, ordered as D5c, D5a, D5b, D5d. The predicted ordering D5c > D5a > D5b should be visually clear.

## Phase 8: Pre-registration document

Generate `PREREG.md` programmatically from the protocol document and the config. Contents:

1. Title, author, date.
2. The seven disruption conditions, exactly as specified in §2 of the protocol.
3. Cut points M = 50 primary, M ∈ {25, 75} secondary for D1 and D3.
4. Foreign-block size K = 20.
5. Topic-matching specification (model, threshold, within-language matching, no-donor handling).
6. Condition-specific shuffled-baseline procedure.
7. All quantitative criteria from §3 of the protocol.
8. Decision rules from §4 including aggregate rules.
9. Statistical procedure from §7.
10. A statement that no parameter or threshold will be changed after data are generated.

After generating, **stop and ask the user to review and lock the pre-registration before any inferential analysis runs.** This means:

- The user reviews PREREG.md.
- If approved, the user commits PREREG.md, all of `disruptions/`, `pipeline/`, `stats/`, and `config/exp1.yaml` to git.
- The user tags the commit `prereg-exp1-v1` and pushes to a public remote.
- Only after the tag is pushed do we run `scripts/01_run_pipeline.py` on real data.

Do not skip this. Pre-registration is the entire methodological argument.

## Phase 9: Run, analyze, report

After pre-registration is locked:

1. Run `scripts/01_run_pipeline.py`. Estimated wall-time ~5 days on a single H100. Use checkpointing.
2. Run `scripts/02_run_stats.py`. Should complete in ~1 hour.
3. Run `scripts/03_make_figures.py`. Should complete in ~10 minutes.
4. Generate a summary report `RESULTS.md` containing:
   - The aggregate pass/fail table.
   - Each disruption's per-corpus outcome.
   - The localization figure.
   - The D5 family figure.
   - A short discussion of which decision-rule branches were triggered (per §4 of the protocol) and what they imply for the paper's framing.

Show the user the report. Do not draft revisions to the human paper; that is a separate task to be initiated by the user.

## Verification gates

Do not advance to the next phase until the prior phase passes its verification:

| Phase | Gate |
|---|---|
| 0 → 1 | `IMPLEMENTATION_NOTES.md` exists; user has confirmed it; environment imports |
| 1 → 2 | Directory structure exists; `pytest` runs (zero tests at this point is fine) |
| 2 → 3 | All disruption unit tests pass |
| 3 → 4 | Topic-matching smoke test passes; <5% no-donor exclusion per corpus |
| 4 → 5 | D0 produces intact-equivalent output on existing test data |
| 5 → 6 | Smoke run on one corpus × one probe × three conditions × three documents produces well-formed output; user approves the schema and sample values |
| 6 → 7 | `stats_summary.parquet` exists; floor-limited handling correctly identifies low-SNR segments |
| 7 → 8 | All plots render; localization figure shows expected scatter pattern (or not) |
| 8 → 9 | User has reviewed `PREREG.md`, committed it, tagged the commit, and pushed to remote |

## Common pitfalls to avoid

1. **Sharing the shuffled baseline across conditions.** Don't. Each condition needs its own. This is the whole point of the v3 refactor.
2. **Pointwise-equality criteria.** The protocol uses correlation-based contrasts (Spearman) for D1, D2, D3, D4 segment comparisons. Do not implement absolute-value tolerance criteria; they overpredict because LLM token contributions are context-dependent.
3. **Truncating D5 analysis at c=100.** D5 conditions add 20 tokens; the analysis runs to c = 120. The post-insertion recovery segment is at d > M + K = 70.
4. **Skipping the no-op control.** D0 is the canary in the coal mine for the disruption infrastructure. It must produce output indistinguishable from the existing pipeline's intact run.
5. **Floor-limited segments forced into pass/fail.** Some segments will be too noisy to evaluate (mean `delta_d` < 2 SE above zero). These must be reported as `inconclusive`, not `fail`. They are excluded from the aggregate count.
6. **Treating D5d as part of the D5 ordering.** D5d is diagnostic-only. The pass/fail ordering is D5c > D5a > D5b.
7. **Running anything inferential before pre-registration is locked.** The whole methodological argument depends on the timestamps. No exceptions.
8. **Modifying parameters mid-run.** Every parameter is in `config/exp1.yaml`. If a parameter looks wrong, stop and raise to the user. Do not silently adjust.

## When in doubt

- Read the protocol document.
- If the protocol is unclear, ask the user.
- If you discover that something the protocol assumes about the existing code is wrong, document it in `IMPLEMENTATION_NOTES.md` and ask the user.
- If you have to choose between the protocol and the existing code's conventions, the protocol wins.

Begin with Phase 0. Use TodoWrite to track each phase. Report back after each verification gate.
