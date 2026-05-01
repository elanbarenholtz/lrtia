# Music Coherence Decay: Implementation Spec v1

A faithful port of the LRTIA v7 / Buckeye Finegrain pipeline to music. The primary scientific claim this spec can support is: **α is directly comparable between language and music if and only if the methodology is identical.**

## Hypothesis

Improvised music (WM-constrained sequential generation) will show α ≈ −0.72 ± 0.05, matching the universal human language band (RAID written −0.75, Buckeye spoken −0.73, French oral −0.69). Composed music and AI-generated music will show steeper α (planned non-sequential generation resembles the RAID AI result of −1.97).

## Core methodology — DO NOT DEVIATE

This is an exact port of `notebooks/RAID_v7_finegrain.ipynb` and `notebooks/Buckeye_Finegrain_v1.ipynb`. The measured quantity is the **marginal corrected benefit of each additional context token**, not the absolute surprisal at distance d.

For each music document:

1. Tokenize to a flat autoregressive token stream.
2. For each of three target positions (50%, 70%, 85% of the stream), select a 30-token target span.
3. For each `ctx_len` in a log-spaced grid (e.g. `[1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64]`), compute the mean per-token negative log-likelihood over the target span conditioned on the preceding `ctx_len` tokens.
4. Repeat with a **token-shuffled** context (same tokens, random order) as the distributional-calibration control.
5. Intact marginal `m_intact[c] = -diff(ppl_intact)`; shuffled marginal `m_shuf[c] = -diff(ppl_shuf)`.
6. **Corrected marginal** `m_corr[c] = m_intact[c] - m_shuf[c]`.
7. Fit a power law on log-binned `m_corr` using bin edges `[1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, MAX_CONTEXT]`. The slope is α.

The key difference from the previous spec: **α is fit on the differenced (marginal) series, not on the raw surprisal-vs-distance series.** Different formula → different exponent. Do not substitute.

## Probe model — strong recommendation: symbolic MIDI

Use a **symbolic autoregressive music LM operating on MIDI-event tokens**, not an audio token model. Why:

- MusicGen's RVQ delay pattern (4 codebooks interleaved with per-stream offsets) makes "the first codebook" ambiguous in the AR stream. Correct extraction is fiddly and a single off-by-one here invalidates every α.
- EnCodec tokens at 32 kHz encode acoustic detail — pitch, timbre, room reverb — that dominates local predictability and swamps the discourse-level signal we want.
- Symbolic MIDI = (time, pitch, duration, instrument) events ≈ the musical analog of words. Clean flat stream, well-defined likelihood.
- Training corpora for symbolic LMs are larger and better-matched to musical structure than audio LMs.

**Primary probe**: Stanford CRFM's **Anticipatory Music Transformer** (Thickstun et al. 2024). Open weights on HuggingFace (`stanford-crfm/music-small-800k` or `music-medium-800k`), clean autoregressive interface via the `anticipation` pip package, tokenizer ships with the model. Runs on a single consumer GPU.

**Fallback** if AMT proves insufficient: **MusicGen-medium** (audio) with *explicit codebook-0 extraction after undoing the delay pattern*. See `transformers.models.musicgen.modeling_musicgen.MusicgenForCausalLM` for the delay pattern implementation. Use `musicgen.build_delay_pattern_mask` and its inverse. If the coder isn't comfortable with this, stay symbolic.

## Corpora

### Improvised (primary human condition)

- **Weimar Jazz Database** (https://jazzomat.hfm-weimar.de) — ~456 jazz solo transcriptions as MIDI. CC BY-NC-SA. The MIDI transcriptions are distributable; only the *source audio* is copyright-encumbered. Since we're using MIDI, this is fine.
- **Filosax** (https://www.filosax.com) — 48 hours of jazz saxophone improvisation with aligned MIDI transcriptions and audio. CC-BY-NC.
- **Backup**: Jazz Trio Database (piano/bass/drums improvisations, MIDI available).

Target: 200+ solos, minimum 512 tokens each after tokenization.

### Composed (planned-generation human condition)

- **MAESTRO** (MIDI) — 200+ hours piano performances of classical compositions. Primary choice.
- **Lakh MIDI Dataset (LMD-matched)** — multi-instrument, ~45K pieces. Restrict to classical subset.
- **JSB Chorales** — small but pure composed-music corpus, useful as a sanity-check subset.

Match sample size to improvised corpus (~200 pieces, 512+ tokens each).

### AI-generated (control)

Critical: **use a different AI model to generate than the probe model**. If you probe with AMT and generate with AMT, you're measuring AMT's likelihood on AMT's samples, which is biased.

- Generate with **Magenta Music Transformer** (Huang et al. 2018, open weights) or **MuseNet-style transformers**, **then** probe with AMT.
- Alternatively: probe with a small AMT checkpoint and generate with a large one. Not ideal but mitigates the worst of the self-bias.
- Target: 200+ unconditional generations, 2048+ tokens each.

## Configuration (adapted from PERSUADE_Finegrain_v1.ipynb)

```python
MAX_CONTEXT = 64                       # tokens of context to reveal
TARGET_LEN = 30                        # tokens in target region
TARGET_FRACTIONS = [0.50, 0.70, 0.85]  # push targets late — need context before them
MIN_CONTEXT_BEFORE_TARGET = MAX_CONTEXT + 10
RANDOM_SEED = 42

# Sparse log-spaced grid — power-law fit bins in log space anyway
CTX_LENGTHS = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, MAX_CONTEXT]))
```

Minimum document length to be included: `n_tokens >= int(MIN_CONTEXT_BEFORE_TARGET / min(TARGET_FRACTIONS))` = 148 tokens. Most music documents will far exceed this.

## Pipeline skeleton (Python, to be filled in by coder)

```python
import torch, math, json
import numpy as np
from scipy import stats
from tqdm import tqdm

# --- Load probe ---
from anticipation import ops
from anticipation.sample import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "stanford-crfm/music-medium-800k"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                              device_map="auto").eval()

@torch.no_grad()
def compute_ppl(token_ids, target_start, target_end):
    """Mean per-token NLL over [target_start, target_end)."""
    if target_start >= target_end - 1:
        return float('inf')
    input_ids = torch.tensor([token_ids], device=model.device)
    logits = model(input_ids).logits[0]
    total, count = 0.0, 0
    for i in range(target_start, target_end - 1):
        lp = torch.log_softmax(logits[i], dim=-1)
        total += -lp[token_ids[i + 1]].item()
        count += 1
    return math.exp(total / count) if count > 0 else float('inf')


def compute_token_reveal_curve(full_ids, target_start, target_end,
                                shuffled=False, rng_shuf=None):
    target_ids = full_ids[target_start:target_end]
    context_pool = list(full_ids[:target_start])
    if shuffled:
        rng_shuf.shuffle(context_pool)
    max_ctx = min(MAX_CONTEXT, len(context_pool))
    if max_ctx < 10:
        return None
    ppls, ctx_lengths = [], []
    for ctx_len in CTX_LENGTHS:
        if ctx_len > max_ctx:
            break
        chunk = context_pool[-ctx_len:] + target_ids
        ppl = compute_ppl(chunk, ctx_len, len(chunk))
        if not math.isinf(ppl):
            ppls.append(ppl)
            ctx_lengths.append(ctx_len)
    if len(ppls) < 4:
        return None
    return {'ctx_lengths': ctx_lengths, 'ppls': ppls}


def process_document(doc, rng_shuf):
    """doc = {'doc_id': str, 'tokens': List[int], 'corpus': str}"""
    full_ids = doc['tokens']
    n = len(full_ids)
    intact, shuffled = [], []
    for frac in TARGET_FRACTIONS:
        target_start = int(n * frac)
        target_end = min(target_start + TARGET_LEN, n)
        if target_start < MIN_CONTEXT_BEFORE_TARGET or target_end - target_start < 5:
            continue
        r_i = compute_token_reveal_curve(full_ids, target_start, target_end, False, rng_shuf)
        r_s = compute_token_reveal_curve(full_ids, target_start, target_end, True, rng_shuf)
        for r, bucket in [(r_i, intact), (r_s, shuffled)]:
            if r is not None:
                r.update({'doc_id': doc['doc_id'], 'corpus': doc['corpus'],
                          'target_frac': frac, 'token_count': n})
                bucket.append(r)
    return intact, shuffled
```

## Analysis (identical to Buckeye/PERSUADE)

```python
common_x = np.arange(1, MAX_CONTEXT + 1)
bin_edges = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, MAX_CONTEXT]

def compute_raw_ppl_curve(curves):
    arr = []
    for c in curves:
        interp = np.interp(common_x, np.array(c['ctx_lengths']),
                           np.array(c['ppls']), left=np.nan, right=np.nan)
        arr.append(interp)
    return np.nanmean(np.array(arr), axis=0)

def fit_power_law(marg):
    bm, bc = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        vals = marg[lo-1:hi-1]
        vals = vals[~np.isnan(vals)]
        if len(vals) and np.mean(vals) > 0:
            bm.append(np.mean(vals)); bc.append((lo + hi) / 2)
    if len(bm) < 4:
        return None
    slope, intercept, r, p, _ = stats.linregress(np.log(bc), np.log(bm))
    return slope, r, p, bc, bm, intercept

# Per corpus
for corpus_name in ['improvised', 'composed', 'ai_generated']:
    ci = [c for c in all_intact if c['corpus'] == corpus_name]
    cs = [c for c in all_shuffled if c['corpus'] == corpus_name]
    ip, sp = compute_raw_ppl_curve(ci), compute_raw_ppl_curve(cs)
    corr = -np.diff(ip) - (-np.diff(sp))
    fit = fit_power_law(corr)
    print(f"{corpus_name}: α = {fit[0]:+.3f}  (r = {fit[1]:+.2f})")
```

Also produce: per-document α distribution (like the Buckeye per-speaker panel) and a cross-corpus overlay figure matching `fig2_cross_modality.png` in the Buckeye notebook.

## Sanity checks (REQUIRED before interpreting any α)

These all come from hard-learned LRTIA lessons. Skip them at your peril:

1. **Uniform-random token baseline**: generate a sequence of uniformly-random-over-vocabulary tokens, run the pipeline. `intact_marg` should equal `shuf_marg` (both zero after correction). If not, the probe is leaking signal from outside the measured pathway.
2. **Shuffled-control effect size**: `shuf_marg` should show a meaningful positive bump at short distances (distributional calibration) before the corrected marginal starts to look power-law. If shuffled is flat, the probe isn't doing distributional calibration and correction does nothing.
3. **Intact-raw curve before correction**: plot it. It should drop monotonically. If it plateaus or oscillates, something's off with tokenization or the probe isn't handling this domain.
4. **Per-target-position stability**: fit α separately at the 50%/70%/85% targets. They should agree within ~0.15.
5. **Hold-out a small calibration set** before looking at any score comparisons. Use it only for the sanity checks above, so that the main test is confirmatory.

## Compute

Per-essay runtime scales as: `N_docs × 3 targets × 2 (intact+shuf) × len(CTX_LENGTHS) forward_passes`. With AMT-medium (400M params) on short sequences (~90 tokens each), one A100 should cover 200 docs/corpus × 3 corpora = 600 docs in ~30-60 minutes.

Do **not** call `torch.cuda.empty_cache()` inside the inner loop — it syncs the device on every forward pass and catastrophically slows short-sequence workloads.

## Deliverables

1. Per-corpus aggregate α with r, p (the headline table).
2. Per-document α distribution per corpus, mean ± SD (the per-speaker-equivalent plot).
3. Cross-corpus overlay: intact perplexity curves, corrected marginals, power-law fits — one figure with all three corpora and the language anchors (RAID, Buckeye, French oral, A&S, RAID-AI).
4. All raw curves (intact and shuffled) saved as JSON per corpus so the analysis can be re-run without re-probing.
5. A short notebook or markdown report with the five sanity-check plots from the required list.

## What would constitute a successful result

- **Primary prediction**: improvised corpus α ∈ [−0.82, −0.62] (within 0.1 of the −0.72 language anchor).
- **Secondary**: composed corpus α steeper than improvised by >0.3.
- **Tertiary**: AI-generated α more negative than −1.5.

If improvised α matches language, this is a substantial extension — "α is a universal of biological WM-constrained sequential generation, not a property of language specifically." That's a separate paper, not a footnote to the current one.

If improvised α differs from language but is consistent within music, that's still publishable but the theoretical frame shifts to domain-specific WM signatures.
