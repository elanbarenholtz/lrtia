# Experiment 1 — Implementation Notes

## Protocol document
`paper/exp1_structured_disruption.md` (v3, pre-registration-ready)

## Existing codebase inventory

### Corpora (processed JSONL, ready for use)

All paths verified to exist with correct document counts.

| Corpus | Path | Docs | Used in Exp1? |
|--------|------|------|---------------|
| Chinese Wiki | `data/wiki_multilingual/zh_articles.jsonl` | 60 | Yes |
| Japanese Wiki | `data/wiki_multilingual/ja_articles.jsonl` | 60 | Yes |
| Korean Wiki | `data/wiki_multilingual/ko_articles.jsonl` | 60 | Yes |
| Turkish Wiki | `data/wiki_multilingual/tr_articles.jsonl` | 60 | Yes |
| Arabic Wiki | `data/wiki_multilingual/ar_articles.jsonl` | 60 | Yes |
| Finnish Wiki | `data/wiki_multilingual/fi_articles.jsonl` | 60 | Yes |
| Buckeye spoken | `data/buckeye_processed/speaker_concatenated.jsonl` | 26 | Yes |
| French Oral | `data/french_oral_processed/per_story.jsonl` | 87 | Yes |

D5 variants run on the 6 Wikipedia corpora only (§5 of protocol).

### Probe models (confirmed from code)

| Probe | HuggingFace ID | Quantization | Confirmed in |
|-------|----------------|-------------|-------------|
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | 4-bit (BnB nf4) | All 30+ notebooks consistently |
| Llama-3.1-8B | `unsloth/Meta-Llama-3.1-8B` | 4-bit (BnB nf4) | `Llama_Multilingual_Finegrain_v1.ipynb` |

**Protocol correction applied:** §6 originally said "Llama-3-8B-Instruct" — updated to `unsloth/Meta-Llama-3.1-8B` (base, not instruct, version 3.1) to match the actual analyses. This is also a manuscript bug: the PNAS draft Methods section says "Llama-3-8B-Instruct" and must be corrected to `Meta-Llama-3.1-8B` (base).

**Mistral confirmed:** `mistralai/Mistral-7B-v0.1` is used everywhere. No version mismatch.

### Probe code — perplexity computation

**No standalone probe module.** `compute_ppl` and `compute_token_reveal_curve` are defined inline in each notebook. The `lrtia/model/hf_backend.py` exists but is not used by the finegrain pipeline.

**Exact `compute_ppl` implementation** (identical across all finegrain notebooks):
```python
@torch.no_grad()
def compute_ppl(token_ids, target_start, target_end):
    if target_start >= target_end - 1:
        return float('inf')
    input_ids = torch.tensor([token_ids], device=model.device)
    outputs = model(input_ids)
    logits = outputs.logits[0]
    total_loss = 0.0
    count = 0
    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        total_loss += -log_probs[token_ids[i + 1]].item()
        count += 1
    del outputs, logits
    torch.cuda.empty_cache()
    return math.exp(total_loss / count) if count > 0 else float('inf')
```

**No KV caching.** Every call to `compute_ppl` does a full forward pass from scratch. No `past_key_values`, no `use_cache` parameter anywhere in the codebase. This means:
- Each context length c requires a separate full forward pass
- For 100 context lengths × 2 conditions (intact + shuffled) = 200 forward passes per target per condition
- With 50 shuffles (protocol requirement), this becomes 100 (intact) + 100×50 (shuffled) = 5,100 forward passes per target per condition
- **This is the dominant compute cost.** Timing estimates must account for this.

**Tokenizer usage:**
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # if None
full_ids = tokenizer.encode(doc['text'], add_special_tokens=False)
```
Standard HuggingFace tokenizer, no special tokens prepended.

**Context reveal direction:**
```python
context_pool = list(full_ids[:target_start])  # all tokens before target
ctx_tokens = context_pool[-ctx_len:]          # last ctx_len tokens (closest to target)
chunk = ctx_tokens + target_ids               # context + target
ppl = compute_ppl(chunk, len(ctx_tokens), len(chunk))
```
Context grows backward from target: at ctx_len=1, only the immediately-preceding token is revealed. At ctx_len=100, the 100 tokens immediately before the target are revealed. This maps to the protocol's convention: position 1 = closest to target = `context_pool[-1]`, position C = most distant = `context_pool[-C]`.

### Corrected marginal computation

**Current implementation:** Single-pass shuffled baseline, **not** the condition-specific procedure required by the protocol.

The existing pattern:
1. `context_pool = list(full_ids[:target_start])` — all tokens before target
2. For intact: `ctx_tokens = context_pool[-ctx_len:]` at each ctx_len
3. For shuffled: `rng_shuf.shuffle(context_pool)` **once**, then `ctx_tokens = context_pool[-ctx_len:]`
4. Corrected marginal computed post-hoc on aggregated mean PPL curves across documents

**Critical departures from protocol:**
- **1 shuffle per document** (protocol requires 50 shuffles averaged per c)
- **Shuffled baseline uses the same shuffled pool across all c** — at c=1, the shuffled prefix is one token drawn from the shuffled version of the full 100-token context; this is NOT a random permutation of the 1-token prefix. Different from what the protocol specifies (shuffle the specific c-token multiset at each c).
- **No condition-specific baselines** — only intact and one global shuffled condition exist
- **Marginals computed on aggregated curves** — mean PPL across documents first, then take marginals, rather than per-document marginals then aggregate

**This is the major refactor (Phase 4).** However, note: for the intact condition, the existing "shuffle the full pool then take the last c tokens" is actually equivalent to "take a random c-token subset from the full pool" — which is close to but not identical to "shuffle the specific c-token prefix." The difference: the existing approach draws a random subset from all tokens before target, not specifically from the c closest tokens. For disrupted conditions this distinction matters enormously.

### Curve fitting

- `lrtia/aggregation/summary.py`: `fit_decay_models()` fitting power law, exponential, linear with scipy.optimize.curve_fit
- Inline `fit_power_law()` in notebooks: `scipy.stats.linregress` on log-log binned data
- `notebooks/Functional_Form_Final.ipynb`: AIC/BIC comparison across 5 functional forms
- **No breakpoint detection code exists.** Must be written for Phase 6.

### Plotting

- `lrtia/visualization/plots.py`: basic `plot_memory_curve()` and comparison functions
- `notebooks/Paper_Figures.ipynb`: publication figures from cached Drive data
- All experiment-specific plotting will be new

## Environment

**Local venv** at `.venv/`:
- ✅ torch 2.10.0 (no CUDA — local development only; GPU runs on Colab/H100)
- ✅ transformers 4.57.6
- ✅ scipy 1.17.0, numpy 2.4.1, pandas 2.3.3
- ✅ statsmodels 0.14.6
- ✅ accelerate 1.12.0
- ✅ pytest 9.0.2
- ❌ sentence-transformers (needed for D5 topic matching)
- ❌ scikit-learn (needed for some stats)
- ❌ bitsandbytes (GPU-only, not needed locally)

**GPU availability:** No local CUDA. All GPU-intensive work executes on Colab Pro via notebooks:
- Phase 3: Sentence-transformer embedding computation (can run on CPU locally but slow)
- Phase 5: Full pipeline runner (probe forward passes) — must be Colab H100/A100
- Local machine handles: transforms, unit tests, statistical analysis, plotting

## Compute estimate (revised)

The no-KV-caching forward pass is the bottleneck. Per (document, target_position, condition):
- Intact: 100 forward passes (c = 1..100), or 120 for D5
- Shuffled baseline (50 permutations): 100 × 50 = 5,000 forward passes, or 120 × 50 = 6,000 for D5
- Total per target: ~5,100 forward passes (D0-D4) or ~6,120 (D5)

Per corpus (60 docs × 3 targets = 180 targets):
- Per condition: 180 × 5,100 = 918,000 forward passes
- D0-D4 at M=50: 5 conditions × 918K = 4.59M forward passes
- D1+D3 at M=25,75: 4 conditions × 918K = 3.67M
- D5a-d: 4 conditions × 180 × 6,120 = 4.41M
- Total per corpus per probe: ~12.7M forward passes

At ~3ms per forward pass on H100 (130 tokens, 4-bit 8B model): ~10.5 hours per corpus per probe.
8 corpora × 2 probes: **~168 hours = 7 days** on a single H100.

This matches the protocol's 5-day estimate (which assumed some parallelism or faster passes).

## Resolved decisions

1. **Llama model:** `unsloth/Meta-Llama-3.1-8B` (base, not instruct). Protocol updated.
2. **Mistral model:** `mistralai/Mistral-7B-v0.1` — confirmed, no mismatch.
3. **Context direction:** Existing code reveals backward from target (position 1 = closest). Maps correctly to protocol convention.
4. **Experiment directory:** `experiments/exp1_disruption/`
