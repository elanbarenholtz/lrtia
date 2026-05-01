#!/usr/bin/env python3
"""
_build_persuade_finegrain_notebook.py

One-shot notebook builder for PERSUADE_Finegrain_v1.ipynb. Constructs the
notebook JSON from source strings so we don't have to hand-write the ipynb
envelope. Not a reusable tool — delete after running if desired.
"""

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks/PERSUADE_Finegrain_v1.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = []

CELLS.append(md("""# PERSUADE: Token-Level Coherence Decay vs Essay Score

Replicates the RAID v7 / Buckeye fine-grained token-reveal analysis on the **PERSUADE** argumentative-essay corpus. Primary research question: **does the coherence-decay exponent correlate with holistic essay score?**

**Sample**: 500 essays, stratified 100/bin across scores 2–6 (score 1 dropped — too sparse and short for reliable per-essay fits). Source: `data/persuade_clean/cohorts/persuade_finegrain_sample.jsonl`.

**Method**: Identical v7/v4 corrected power-law pipeline used on RAID, Buckeye, and French Oral:
1. For each essay, sample 3 target regions (30-token spans) at 50%/70%/85% of the token stream.
2. Reveal context token-by-token back from the target (1 to MAX_CONTEXT tokens).
3. Compute perplexity on the target for each context length.
4. Repeat with token-shuffled context (distributional-calibration control).
5. Corrected marginal = intact marginal − shuffled marginal.
6. Fit a power law on binned corrected marginals → α.

**PERSUADE-specific tuning**: Essays are much shorter than Buckeye monologues (median ≈ 400 words ≈ 500 tokens), so:
- `MAX_CONTEXT = 64` (not 100) to keep the context window feasible for short essays.
- Target fractions pushed late (0.50, 0.70, 0.85) to maximise context before the target.

**Length confound to watch**: score in PERSUADE is tightly correlated with essay length (score 2 median 286 tokens → score 6 median 988 tokens). Any α-vs-score relationship must be checked against a length control (regression with log(token_count), length-matched subsample).

**Predictions**:
- Higher-scoring essays → flatter decay (α closer to 0, more long-range structure)?
- Or: all human essays cluster near −0.75 regardless of score, and only shape features (L50, tail) move with score?
"""))

CELLS.append(code("""!pip install -q -U bitsandbytes>=0.46.1 accelerate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import json, math, time, gc, os, re, torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print('Imports OK')
"""))

CELLS.append(code("""# === Configuration ===
IN_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content')

SAMPLE_FILE = "persuade_finegrain_sample.jsonl"

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DATA = Path("/content/drive/MyDrive/LRTIA/Data/persuade_clean/cohorts")
    DRIVE_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/PERSUADE_finegrain")
    DRIVE_RAID_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/RAID_finegrain")
    DRIVE_BUCKEYE_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/Buckeye_finegrain")
    if (DRIVE_DATA / SAMPLE_FILE).exists():
        DATA_DIR = DRIVE_DATA
    else:
        LOCAL_DATA = Path("/content/data/persuade_clean/cohorts")
        if not (LOCAL_DATA / SAMPLE_FILE).exists():
            LOCAL_DATA.mkdir(parents=True, exist_ok=True)
            from google.colab import files
            print(f"Upload {SAMPLE_FILE}:")
            uploaded = files.upload()
            for fname in uploaded:
                with open(LOCAL_DATA / fname, 'wb') as f:
                    f.write(uploaded[fname])
        DATA_DIR = LOCAL_DATA
    BASE_DIR = DRIVE_RESULTS
    BASE_DIR.mkdir(parents=True, exist_ok=True)
else:
    BASE_DIR = Path("../results/persuade_finegrain")
    DATA_DIR = Path("../data/persuade_clean/cohorts")
    DRIVE_RAID_RESULTS = Path("../results/raid_finegrain")
    DRIVE_BUCKEYE_RESULTS = Path("../results/buckeye_finegrain")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
USE_4BIT = True

# --- PERSUADE-specific tuning (short essays, ~200-1000 tokens) ---
MAX_CONTEXT = 64
TARGET_LEN = 30
TARGET_FRACTIONS = [0.50, 0.70, 0.85]
MIN_CONTEXT_BEFORE_TARGET = MAX_CONTEXT + 10  # 74 tokens minimum before target
RANDOM_SEED = 42

print(f"Max context: {MAX_CONTEXT} tokens")
print(f"Target length: {TARGET_LEN} tokens")
print(f"Target positions: {TARGET_FRACTIONS}")
print(f"Results dir: {BASE_DIR}")
"""))

CELLS.append(code("""# === Load PERSUADE sample ===
corpus = []
with open(DATA_DIR / SAMPLE_FILE) as f:
    for line in f:
        r = json.loads(line)
        corpus.append({
            'doc_id': r['essay_id'],
            'text': r['text'],
            'score': r['score'],
            'grade': r.get('grade'),
            'ell': r.get('ell'),
            'prompt_id': r.get('prompt_id'),
            'word_count': r.get('word_count'),
            'token_count': r.get('token_count'),
        })

print(f"Loaded {len(corpus)} essays")
meta = pd.DataFrame([{k: d[k] for k in ['doc_id','score','grade','ell','word_count','token_count']} for d in corpus])
print("\\nPer-score breakdown:")
print(meta.groupby('score').agg(n=('doc_id','count'),
                                 wc_med=('word_count','median'),
                                 tc_med=('token_count','median')))
print("\\nELL breakdown:")
print(meta['ell'].value_counts())
"""))

CELLS.append(code("""# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
model.eval()
print("Model loaded")
"""))

CELLS.append(code("""# === Core functions (identical to Buckeye_Finegrain_v1 / RAID v7) ===

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


def compute_token_reveal_curve(full_ids, target_start, target_end,
                                shuffled=False, rng_shuf=None):
    target_ids = full_ids[target_start:target_end]
    context_pool = list(full_ids[:target_start])
    if shuffled and rng_shuf is not None:
        context_pool = list(context_pool)
        rng_shuf.shuffle(context_pool)
    max_ctx = min(MAX_CONTEXT, len(context_pool))
    if max_ctx < 10:
        return None
    ppls, ctx_lengths = [], []
    for ctx_len in range(1, max_ctx + 1):
        ctx_tokens = context_pool[-ctx_len:]
        chunk = ctx_tokens + target_ids
        ppl = compute_ppl(chunk, len(ctx_tokens), len(chunk))
        if not math.isinf(ppl):
            ppls.append(ppl)
            ctx_lengths.append(ctx_len)
    if len(ppls) < 10:
        return None
    return {'ctx_lengths': ctx_lengths, 'ppls': ppls}


def process_document(doc, rng_shuf):
    full_ids = tokenizer.encode(doc['text'], add_special_tokens=False)
    n = len(full_ids)
    intact_curves, shuffled_curves = [], []
    for frac in TARGET_FRACTIONS:
        target_start = int(n * frac)
        target_end = min(target_start + TARGET_LEN, n)
        if target_start < MIN_CONTEXT_BEFORE_TARGET or target_end - target_start < 5:
            continue
        result = compute_token_reveal_curve(full_ids, target_start, target_end, shuffled=False)
        if result is not None:
            result.update({'doc_id': doc['doc_id'], 'target_frac': frac,
                           'score': doc['score'], 'grade': doc.get('grade'),
                           'ell': doc.get('ell'), 'token_count': n})
            intact_curves.append(result)
        result_s = compute_token_reveal_curve(full_ids, target_start, target_end,
                                               shuffled=True, rng_shuf=rng_shuf)
        if result_s is not None:
            result_s.update({'doc_id': doc['doc_id'], 'target_frac': frac,
                             'score': doc['score'], 'grade': doc.get('grade'),
                             'ell': doc.get('ell'), 'token_count': n})
            shuffled_curves.append(result_s)
    return intact_curves, shuffled_curves

print("Functions defined")
"""))

CELLS.append(code("""# === Run computation (or load cached results) ===
results_path = BASE_DIR / "persuade_intact_v1.json"
shuffled_path = BASE_DIR / "persuade_shuffled_v1.json"

if results_path.exists() and shuffled_path.exists():
    with open(results_path) as f:
        all_intact = json.load(f)
    with open(shuffled_path) as f:
        all_shuffled = json.load(f)
    print(f"Loaded {len(all_intact)} intact + {len(all_shuffled)} shuffled curves from cache")
else:
    all_intact, all_shuffled = [], []
    rng_shuf = np.random.RandomState(RANDOM_SEED + 99)
    for doc in tqdm(corpus, desc="Processing essays"):
        intact, shuffled = process_document(doc, rng_shuf)
        all_intact.extend(intact)
        all_shuffled.extend(shuffled)
    with open(results_path, 'w') as f:
        json.dump(all_intact, f)
    with open(shuffled_path, 'w') as f:
        json.dump(all_shuffled, f)
    print(f"Computed {len(all_intact)} intact + {len(all_shuffled)} shuffled curves")

unique_docs = set(c['doc_id'] for c in all_intact)
print(f"\\nUnique essays with curves: {len(unique_docs)}")
print(f"Curves per target position:")
for frac in TARGET_FRACTIONS:
    n = sum(1 for c in all_intact if c['target_frac'] == frac)
    print(f"  {frac:.0%}: {n} curves")

print(f"\\nPer-score curve counts:")
by_score = {}
for c in all_intact:
    by_score.setdefault(c['score'], set()).add(c['doc_id'])
for s in sorted(by_score):
    n_essays = len(by_score[s])
    n_curves = sum(1 for c in all_intact if c['score'] == s)
    print(f"  score={s}: {n_essays} essays, {n_curves} curves")
"""))

CELLS.append(md("""## Aggregate Corrected Power Law (baseline)

First, fit the overall corrected α on all 500 essays. This is the cross-corpus anchor: we expect it to land near −0.75 (the universal human value across RAID, Buckeye, French Oral) if PERSUADE behaves like the other human corpora.
"""))

CELLS.append(code("""# === Analysis helpers (identical to RAID v7 / Buckeye) ===
common_x = np.arange(1, MAX_CONTEXT + 1)
bin_edges = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, MAX_CONTEXT]

def compute_raw_ppl_curve(curves):
    all_ppl = []
    for curve in curves:
        ctx = np.array(curve['ctx_lengths'])
        ppl = np.array(curve['ppls'])
        interp = np.interp(common_x, ctx, ppl, left=np.nan, right=np.nan)
        all_ppl.append(interp)
    all_ppl = np.array(all_ppl)
    return np.nanmean(all_ppl, axis=0)

def fit_power_law(marg):
    bm, bc = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        vals = marg[lo-1:hi-1]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0 and np.mean(vals) > 0:
            bm.append(np.mean(vals))
            bc.append((lo + hi) / 2)
    if len(bm) >= 4:
        slope, intercept, r, p, _ = stats.linregress(np.log(bc), np.log(bm))
        return slope, r, p, bc, bm, intercept
    return None

# Overall corrected fit
intact_ppl = compute_raw_ppl_curve(all_intact)
shuf_ppl = compute_raw_ppl_curve(all_shuffled)
intact_marg = -np.diff(intact_ppl)
shuf_marg = -np.diff(shuf_ppl)
corrected_marg = intact_marg - shuf_marg

fit = fit_power_law(corrected_marg)
if fit:
    s, r, p, bc, bm, inter = fit
    print(f"PERSUADE overall (corrected): alpha = {s:.3f}  (r = {r:.3f}, p = {p:.4g})")
print(f"Uncorrected intact fit: ", end="")
fit_u = fit_power_law(intact_marg)
if fit_u:
    print(f"alpha = {fit_u[0]:.3f}  (r = {fit_u[1]:.3f})")
"""))

CELLS.append(md("""## α by Score Bin

Split essays by score (2–6), compute corrected marginals per bin, fit α per bin. This is the headline result: **does α vary monotonically with score?**
"""))

CELLS.append(code("""# === Per-score-bin corrected fits ===
SCORE_COLORS = {2: '#d73027', 3: '#fc8d59', 4: '#fee08b', 5: '#91cf60', 6: '#1a9850'}

per_score_fits = {}
per_score_corrected = {}
for s in sorted(set(c['score'] for c in all_intact)):
    sc_intact = [c for c in all_intact if c['score'] == s]
    sc_shuf = [c for c in all_shuffled if c['score'] == s]
    if len(sc_intact) < 10:
        continue
    sc_intact_ppl = compute_raw_ppl_curve(sc_intact)
    sc_shuf_ppl = compute_raw_ppl_curve(sc_shuf)
    sc_corr = -np.diff(sc_intact_ppl) - (-np.diff(sc_shuf_ppl))
    per_score_corrected[s] = sc_corr
    fit = fit_power_law(sc_corr)
    if fit:
        per_score_fits[s] = fit
        slope, r, p, bc, bm, inter = fit
        n_essays = len(set(c['doc_id'] for c in sc_intact))
        print(f"  score={s} (n={n_essays:>3}): alpha = {slope:+.3f}  (r = {r:+.3f}, p = {p:.3g})")

# Score trend
scores_sorted = sorted(per_score_fits.keys())
exponents_sorted = [per_score_fits[s][0] for s in scores_sorted]
if len(scores_sorted) >= 3:
    trend_r, trend_p = stats.spearmanr(scores_sorted, exponents_sorted)
    print(f"\\nSpearman(score, alpha_by_bin) = {trend_r:+.3f}, p = {trend_p:.3g}")
"""))

CELLS.append(code("""# === Figure 1: score-bin power-law overlay + alpha bar chart + kernels ===
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# A: Power-law fits overlaid
ax = axes[0]
for s in scores_sorted:
    slope, r, p, bc, bm, inter = per_score_fits[s]
    color = SCORE_COLORS.get(s, 'gray')
    ax.plot(bc, bm, 'o-', color=color, linewidth=2, markersize=6,
            label=f'score {s}: α={slope:+.2f} (r={r:+.2f})')
    fit_x = np.linspace(min(bc), max(bc), 100)
    ax.plot(fit_x, np.exp(inter) * fit_x**slope, '--', color=color, alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel('Context distance (tokens, log)')
ax.set_ylabel('Corrected marginal benefit')
ax.set_title('A. Power-law fit by score bin', fontweight='bold')
ax.legend(fontsize=9, title='Score bin')
ax.grid(True, alpha=0.2)

# B: alpha by score (bar + trend)
ax = axes[1]
ax.bar([str(s) for s in scores_sorted], exponents_sorted,
       color=[SCORE_COLORS.get(s,'gray') for s in scores_sorted],
       alpha=0.8, edgecolor='black')
for i, exp in enumerate(exponents_sorted):
    ax.text(i, exp - 0.03, f'{exp:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(-0.75, color='blue', linestyle=':', alpha=0.6, label='RAID human: −0.75')
ax.axhline(-0.77, color='gray', linestyle=':', alpha=0.6, label='Anderson & Schooler: −0.77')
ax.set_xlabel('Essay score')
ax.set_ylabel('Decay exponent α')
ax.set_title(f'B. α by score bin  (Spearman r={trend_r:+.2f}, p={trend_p:.2g})',
             fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')

# C: Corrected marginals smoothed overlay
ax = axes[2]
for s in scores_sorted:
    color = SCORE_COLORS.get(s,'gray')
    ax.plot(common_x[1:], uniform_filter1d(per_score_corrected[s], 5),
            color=color, linewidth=2, label=f'score {s}')
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Corrected marginal (smoothed)')
ax.set_title('C. Coherence signal by score', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle('PERSUADE: Coherence Decay by Essay Score',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(BASE_DIR / 'fig1_persuade_score_alpha.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

CELLS.append(md("""## Per-Essay α Fits

Individual essays are short (~200–1000 tokens), so per-essay α fits will be noisy. Report the distribution, then check the Spearman(α, score) correlation at the essay level.
"""))

CELLS.append(code("""# === Per-essay α distribution ===
essay_rows = []
for doc_id in sorted(set(c['doc_id'] for c in all_intact)):
    doc_intact = [c for c in all_intact if c['doc_id'] == doc_id]
    doc_shuf = [c for c in all_shuffled if c['doc_id'] == doc_id]
    if len(doc_intact) < 2 or len(doc_shuf) < 2:
        continue
    di_ppl = compute_raw_ppl_curve(doc_intact)
    ds_ppl = compute_raw_ppl_curve(doc_shuf)
    di_corr = -np.diff(di_ppl) - (-np.diff(ds_ppl))
    fit = fit_power_law(di_corr)
    if fit is None:
        continue
    slope, r, p, bc, bm, inter = fit
    meta = doc_intact[0]
    essay_rows.append({
        'doc_id': doc_id,
        'score': meta['score'],
        'grade': meta.get('grade'),
        'ell': meta.get('ell'),
        'token_count': meta['token_count'],
        'alpha': slope,
        'r': r,
        'p': p,
        'n_curves': len(doc_intact),
    })

df_essay = pd.DataFrame(essay_rows)
print(f"Per-essay fits: {len(df_essay)} / {len(set(c['doc_id'] for c in all_intact))} essays")
print(df_essay.describe()[['alpha','r','p','token_count']].round(3))
df_essay.to_csv(BASE_DIR / 'persuade_per_essay_alpha.csv', index=False)
"""))

CELLS.append(code("""# === Figure 2: per-essay α distribution + score relationship ===
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# A: Histogram
ax = axes[0]
for s in sorted(df_essay['score'].unique()):
    vals = df_essay.loc[df_essay['score'] == s, 'alpha']
    ax.hist(vals, bins=25, alpha=0.5, color=SCORE_COLORS.get(s,'gray'),
            label=f'score {s} (n={len(vals)})')
ax.axvline(-0.75, color='blue', linestyle=':', label='RAID human: −0.75')
ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Per-essay α')
ax.set_ylabel('Count')
ax.set_title(f'A. Per-essay α distribution  (median {df_essay["alpha"].median():.2f})',
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# B: Boxplot by score
ax = axes[1]
box_data = [df_essay.loc[df_essay['score'] == s, 'alpha'].values
            for s in sorted(df_essay['score'].unique())]
bp = ax.boxplot(box_data, labels=sorted(df_essay['score'].unique()),
                patch_artist=True, showmeans=True)
for patch, s in zip(bp['boxes'], sorted(df_essay['score'].unique())):
    patch.set_facecolor(SCORE_COLORS.get(s,'gray'))
    patch.set_alpha(0.6)
ax.axhline(-0.75, color='blue', linestyle=':', alpha=0.6)
# Per-essay Spearman
r_spear, p_spear = stats.spearmanr(df_essay['score'], df_essay['alpha'])
r_pear, p_pear = stats.pearsonr(df_essay['score'], df_essay['alpha'])
ax.set_xlabel('Essay score')
ax.set_ylabel('Per-essay α')
ax.set_title(f'B. Per-essay α by score\\nSpearman r={r_spear:+.3f}, p={p_spear:.2g}  |  '
             f'Pearson r={r_pear:+.3f}',
             fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# C: Scatter α vs log(tokens), colored by score
ax = axes[2]
for s in sorted(df_essay['score'].unique()):
    sub = df_essay[df_essay['score'] == s]
    ax.scatter(np.log(sub['token_count']), sub['alpha'],
               color=SCORE_COLORS.get(s,'gray'), alpha=0.7, label=f'score {s}')
ax.set_xlabel('log(token_count)')
ax.set_ylabel('Per-essay α')
ax.set_title('C. α vs essay length (length-confound diagnostic)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle('PERSUADE: Per-Essay α × Score',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(BASE_DIR / 'fig2_persuade_per_essay.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nPer-essay Spearman(score, α): r={r_spear:+.3f}, p={p_spear:.4g}")
print(f"Per-essay Pearson(score, α):  r={r_pear:+.3f}, p={p_pear:.4g}")
print(f"Per-essay Spearman(log token_count, α): "
      f"r={stats.spearmanr(np.log(df_essay['token_count']), df_essay['alpha'])[0]:+.3f}")
"""))

CELLS.append(md("""## Length-Controlled Robustness

PERSUADE score is tightly confounded with essay length. A partial regression on log(token_count) + length-matched subsample disentangle the two.
"""))

CELLS.append(code("""# === OLS: alpha ~ score + log(token_count) (+ ELL) ===
import statsmodels.api as sm

df_reg = df_essay.dropna(subset=['alpha','score','token_count']).copy()
df_reg['log_tc'] = np.log(df_reg['token_count'])
df_reg['is_ell'] = (df_reg['ell'] == 'ELL').astype(int)

X_simple = sm.add_constant(df_reg[['score']])
m_simple = sm.OLS(df_reg['alpha'], X_simple).fit()

X_len = sm.add_constant(df_reg[['score','log_tc']])
m_len = sm.OLS(df_reg['alpha'], X_len).fit()

X_full = sm.add_constant(df_reg[['score','log_tc','is_ell']])
m_full = sm.OLS(df_reg['alpha'], X_full).fit()

print("=== Model 1: alpha ~ score ===")
print(m_simple.summary().tables[1])
print("\\n=== Model 2: alpha ~ score + log(token_count) ===")
print(m_len.summary().tables[1])
print("\\n=== Model 3: alpha ~ score + log(token_count) + is_ell ===")
print(m_full.summary().tables[1])

print(f"\\nscore coefficient:")
print(f"  naive:            beta = {m_simple.params['score']:+.4f}, p = {m_simple.pvalues['score']:.4g}")
print(f"  length-controlled: beta = {m_len.params['score']:+.4f}, p = {m_len.pvalues['score']:.4g}")
print(f"  + ELL control:     beta = {m_full.params['score']:+.4f}, p = {m_full.pvalues['score']:.4g}")
"""))

CELLS.append(code("""# === Length-matched subsample robustness ===
# For each essay at score s, find nearest-length match at score s' within log_tc tolerance
# Use a coarser version: restrict to essays in overlapping length band across scores.
df_reg = df_essay.dropna(subset=['alpha','score','token_count']).copy()
df_reg['log_tc'] = np.log(df_reg['token_count'])

# Overlap band: take 10th-90th percentile of score=4 (the middle score) as reference
ref = df_reg[df_reg['score'] == 4]['log_tc']
lo, hi = np.percentile(ref, [10, 90])
df_match = df_reg[(df_reg['log_tc'] >= lo) & (df_reg['log_tc'] <= hi)].copy()
print(f"Length-matched band log(tc) in [{lo:.2f}, {hi:.2f}]  ({np.exp(lo):.0f}-{np.exp(hi):.0f} tokens)")
print(df_match.groupby('score').size().rename('n').to_frame().T)

if df_match['score'].nunique() >= 3 and len(df_match) >= 50:
    r_m, p_m = stats.spearmanr(df_match['score'], df_match['alpha'])
    print(f"\\nLength-matched Spearman(score, α): r={r_m:+.3f}, p={p_m:.4g}  (n={len(df_match)})")
    # Per-score alpha on length-matched subset (pooled fit)
    print("\\nPooled per-score α (length-matched):")
    for s in sorted(df_match['score'].unique()):
        ids = set(df_match[df_match['score'] == s]['doc_id'])
        sc_intact = [c for c in all_intact if c['doc_id'] in ids]
        sc_shuf = [c for c in all_shuffled if c['doc_id'] in ids]
        if len(sc_intact) < 5:
            continue
        ip = compute_raw_ppl_curve(sc_intact)
        sp = compute_raw_ppl_curve(sc_shuf)
        corr = -np.diff(ip) - (-np.diff(sp))
        fit = fit_power_law(corr)
        if fit:
            print(f"  score={s} (n_essays={len(ids)}): α = {fit[0]:+.3f}  (r={fit[1]:+.2f})")
else:
    print("Not enough essays in length-matched band for reliable subsample fit.")
"""))

CELLS.append(md("""## Secondary Covariates: Grade, ELL

Quick look at whether grade-level or ELL status predicts α independently of score.
"""))

CELLS.append(code("""# === ELL and grade ===
if 'ell' in df_essay.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ELL
    ax = axes[0]
    for i, label in enumerate(['non_ELL','ELL']):
        vals = df_essay.loc[df_essay['ell'] == label, 'alpha']
        if len(vals) == 0: continue
        ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=['steelblue','tomato'][i], alpha=0.6))
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.1, 0.1, len(vals)),
                   vals, alpha=0.4, s=10, color='black')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['non_ELL','ELL'])
    ax.set_ylabel('Per-essay α')
    ax.set_title('α by ELL status')
    ax.grid(True, alpha=0.2, axis='y')
    ell_sub = df_essay[df_essay['ell'].isin(['ELL','non_ELL'])]
    if ell_sub['ell'].nunique() == 2:
        a = ell_sub[ell_sub['ell']=='non_ELL']['alpha']
        b = ell_sub[ell_sub['ell']=='ELL']['alpha']
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"ELL: non_ELL mean α = {a.mean():.3f} (n={len(a)}), "
              f"ELL mean α = {b.mean():.3f} (n={len(b)}),  t={t:.2f}, p={p:.3g}")

    # Grade
    ax = axes[1]
    grade_sub = df_essay.dropna(subset=['grade'])
    if len(grade_sub) > 10:
        grades = sorted(grade_sub['grade'].unique())
        data = [grade_sub.loc[grade_sub['grade']==g, 'alpha'].values for g in grades]
        ax.boxplot(data, labels=[str(int(g)) for g in grades], patch_artist=True)
        ax.set_xlabel('Grade')
        ax.set_ylabel('Per-essay α')
        ax.set_title('α by grade')
        ax.grid(True, alpha=0.2, axis='y')
        r_g, p_g = stats.spearmanr(grade_sub['grade'], grade_sub['alpha'])
        print(f"Grade Spearman(grade, α): r={r_g:+.3f}, p={p_g:.3g}")

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'fig3_persuade_ell_grade.png', dpi=150, bbox_inches='tight')
    plt.show()
"""))

CELLS.append(md("""## Cross-Corpus Context

Anchor the PERSUADE α against the other human corpora (RAID written, Buckeye spoken, French oral, Anderson & Schooler). This confirms whether the PERSUADE overall fit lands in the same ~0.72±0.03 universality band.
"""))

CELLS.append(code("""# === Cross-corpus comparison ===
persuade_alpha = fit_power_law(corrected_marg)
print("Cross-corpus anchors (human):")
print(f"  RAID (written):      α = -0.75")
print(f"  Buckeye (spoken):    α = -0.73")
print(f"  French oral:         α = -0.69")
print(f"  Anderson & Schooler: α = -0.77")
print(f"  AI (RAID):           α = -1.97")
if persuade_alpha:
    print(f"\\n  PERSUADE (this run): α = {persuade_alpha[0]:+.3f}  (r = {persuade_alpha[1]:+.2f})")
"""))

CELLS.append(md("""## Summary

Key numbers to report:
- **Overall PERSUADE corrected α** — does it land in the ~−0.72 ± 0.03 human band?
- **Spearman(score, α_by_bin)** — does α move monotonically with essay score?
- **Per-essay Spearman(score, α)** — is the relationship detectable at the essay level (likely noisy)?
- **Length-controlled β(score)** — does the score effect survive when log(token_count) is partialled out?
- **ELL effect** — do ELL writers show different α from non-ELL writers?
"""))


NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print(f"Wrote {NB_PATH}  ({len(CELLS)} cells)")
