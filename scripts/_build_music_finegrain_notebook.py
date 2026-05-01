#!/usr/bin/env python3
"""
_build_music_finegrain_notebook.py

One-shot builder for Music_Finegrain_v1.ipynb. Produces a clean Colab-ready
notebook following the PERSUADE_Finegrain_v1 / Buckeye_Finegrain_v1 template,
adapted for symbolic MIDI via the Anticipatory Music Transformer.
"""

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks/Music_Finegrain_v1.ipynb"


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# Music: Token-Level Coherence Decay (Improvised vs Composed)

Direct port of the LRTIA v7 / Buckeye Finegrain pipeline to symbolic MIDI, using the **Anticipatory Music Transformer** (Stanford CRFM, Thickstun et al. 2024) as the probe.

**Hypothesis**: improvised music (working-memory-constrained sequential generation) shows α ≈ −0.72 ± 0.05, matching the universal human language band (RAID written −0.75, Buckeye spoken −0.73, French oral −0.69). Composed music shows a steeper α because global planning decouples from on-line memory constraints.

**Method — identical formula to the language analyses**:
1. MIDI → flat list of integer event tokens via `anticipation.convert.midi_to_events` (3 tokens per event: time, duration, note).
2. For each recording, select 3 target regions (30-token spans = 10 events) at 50%/70%/85% of the token stream.
3. For each `ctx_len` in a log-spaced grid of whole-event counts (`CTX_LENGTHS`, multiples of 3), compute mean per-token NLL over the target region given the preceding `ctx_len` tokens.
4. Repeat with token-shuffled context (distributional-calibration control).
5. Corrected marginal = `-diff(ppl_intact) − -diff(ppl_shuf)`.
6. Fit power law on log-binned corrected marginals.

**Music-specific tuning (differs from language/speech)**:
- `MAX_CONTEXT = 255` (85 events, ~15-30 s of music — phrase level), not 64. Music structure lives on longer timescales than words.
- `CTX_LENGTHS` and bin edges are all multiples of 3 so we always reveal whole musical events.
- Target fractions pushed late (0.50/0.70/0.85), same as PERSUADE.

**Corpora**:
- `improvised.jsonl` — Weimar Jazz Database MIDI transcriptions (CC-licensed, ~450 jazz solos)
- `composed.jsonl` — MAESTRO MIDI (classical piano performances of composed works)
- `ai_generated.jsonl` — optional; generate with a DIFFERENT AMT size than the probe to avoid self-likelihood bias

**Key sanity checks** (see Section 6 — DO NOT skip):
- Uniform-random token control: `intact == shuffled` (corrected ≈ 0)
- Shuffled raw marginals show distributional-calibration bump at short distances
- Per-target-position stability (50%/70%/85% should give α within ~0.15)
"""))

CELLS.append(code("""!pip install -q -U bitsandbytes>=0.46.1 accelerate
!pip install -q git+https://github.com/jthickstun/anticipation.git

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import json, math, time, gc, os, re, torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

print('Imports OK')
"""))

CELLS.append(code("""# === Configuration ===
IN_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content')

CORPUS_FILES = {
    'improvised': 'improvised.jsonl',
    'composed':   'composed.jsonl',
    # 'ai_generated': 'ai_generated.jsonl',  # uncomment once you have AI samples
}

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DATA = Path("/content/drive/MyDrive/LRTIA/Data/music_processed")
    DRIVE_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/Music_finegrain")
    DRIVE_RAID_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/RAID_finegrain")
    DRIVE_BUCKEYE_RESULTS = Path("/content/drive/MyDrive/LRTIA/Results/Buckeye_finegrain")
    DATA_DIR = DRIVE_DATA if all((DRIVE_DATA / v).exists() for v in CORPUS_FILES.values()) else Path("/content/data/music_processed")
    if DATA_DIR != DRIVE_DATA:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        from google.colab import files
        for name, fname in CORPUS_FILES.items():
            if not (DATA_DIR / fname).exists():
                print(f"Upload {fname} (for {name}):")
                for fn, data in files.upload().items():
                    with open(DATA_DIR / fn, 'wb') as f:
                        f.write(data)
    BASE_DIR = DRIVE_RESULTS
    BASE_DIR.mkdir(parents=True, exist_ok=True)
else:
    BASE_DIR = Path("../results/music_finegrain")
    DATA_DIR = Path("../data/music_processed")
    DRIVE_RAID_RESULTS = Path("../results/raid_finegrain")
    DRIVE_BUCKEYE_RESULTS = Path("../results/buckeye_finegrain")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "stanford-crfm/music-medium-800k"

# --- Music-specific tuning (events are 3 tokens each: time, dur, note) ---
EVENT_SIZE = 3
N_EVENTS_CONTEXT = 85              # ~phrase-level context
MAX_CONTEXT = N_EVENTS_CONTEXT * EVENT_SIZE  # 255 tokens
TARGET_LEN = 10 * EVENT_SIZE        # 30 tokens = 10 events
TARGET_FRACTIONS = [0.50, 0.70, 0.85]
MIN_CONTEXT_BEFORE_TARGET = MAX_CONTEXT + EVENT_SIZE  # 258

# Log-spaced ctx grid in EVENTS, converted to tokens (all multiples of 3)
_EVENT_GRID = [1, 2, 3, 5, 8, 12, 18, 27, 40, 55, 70, N_EVENTS_CONTEXT]
CTX_LENGTHS = sorted(set(e * EVENT_SIZE for e in _EVENT_GRID))

RANDOM_SEED = 42

print(f"Model: {MODEL_NAME}")
print(f"Max context: {MAX_CONTEXT} tokens ({N_EVENTS_CONTEXT} events)")
print(f"Target length: {TARGET_LEN} tokens ({TARGET_LEN // EVENT_SIZE} events)")
print(f"Target positions: {TARGET_FRACTIONS}")
print(f"Ctx grid ({len(CTX_LENGTHS)} pts, token counts): {CTX_LENGTHS}")
print(f"Ctx grid (event counts): {[c // EVENT_SIZE for c in CTX_LENGTHS]}")
print(f"Results dir: {BASE_DIR}")
"""))

CELLS.append(code("""# === Load music corpora ===
corpus = []
for corpus_name, fname in CORPUS_FILES.items():
    path = DATA_DIR / fname
    if not path.exists():
        print(f"WARNING: missing {path} — skipping {corpus_name}")
        continue
    n = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            corpus.append({
                'doc_id': r['doc_id'],
                'corpus': r['corpus'],
                'tokens': r['tokens'],
                'n_tokens': r['n_tokens'],
            })
            n += 1
    print(f"  {corpus_name}: {n} recordings from {path.name}")

print(f"\\nTotal recordings: {len(corpus)}")
meta = pd.DataFrame([{'doc_id': d['doc_id'], 'corpus': d['corpus'],
                      'n_tokens': d['n_tokens']} for d in corpus])
print("\\nPer-corpus length stats (token counts):")
print(meta.groupby('corpus')['n_tokens'].describe()[['count','min','50%','mean','max']].round(0))
"""))

CELLS.append(code("""# === Load probe ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
"""))

CELLS.append(code("""# === Core functions ===
# Differences from PERSUADE/Buckeye: none in the shape of the pipeline, only in
# the underlying vocabulary (AMT event tokens vs Mistral SentencePiece).

@torch.no_grad()
def compute_ppl(token_ids, target_start, target_end):
    if target_start >= target_end - 1:
        return float('inf')
    input_ids = torch.tensor([token_ids], device=model.device)
    logits = model(input_ids).logits[0]
    total, count = 0.0, 0
    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        total += -log_probs[token_ids[i + 1]].item()
        count += 1
    return math.exp(total / count) if count > 0 else float('inf')


def compute_token_reveal_curve(full_ids, target_start, target_end,
                                shuffled=False, rng_shuf=None):
    target_ids = full_ids[target_start:target_end]
    context_pool = list(full_ids[:target_start])
    if shuffled and rng_shuf is not None:
        # Shuffle in event-aligned triples to preserve token-role distribution
        # (time tokens stay time tokens, etc) — otherwise shuffled has a very
        # different vocabulary distribution and swamps the correction.
        triples = [tuple(context_pool[i:i+EVENT_SIZE])
                   for i in range(0, len(context_pool), EVENT_SIZE)
                   if i + EVENT_SIZE <= len(context_pool)]
        rng_shuf.shuffle(triples)
        context_pool = [t for tr in triples for t in tr]
    max_ctx = min(MAX_CONTEXT, len(context_pool))
    if max_ctx < EVENT_SIZE * 2:
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
    full_ids = doc['tokens']
    n = len(full_ids)
    intact, shuffled = [], []
    for frac in TARGET_FRACTIONS:
        target_start = int(n * frac)
        # Snap target_start to event boundary
        target_start = (target_start // EVENT_SIZE) * EVENT_SIZE
        target_end = min(target_start + TARGET_LEN, n)
        # Snap target_end to event boundary too
        target_end = (target_end // EVENT_SIZE) * EVENT_SIZE
        if target_start < MIN_CONTEXT_BEFORE_TARGET or target_end - target_start < EVENT_SIZE * 2:
            continue
        r_i = compute_token_reveal_curve(full_ids, target_start, target_end,
                                          shuffled=False)
        if r_i is not None:
            r_i.update({'doc_id': doc['doc_id'], 'corpus': doc['corpus'],
                        'target_frac': frac, 'n_tokens': n})
            intact.append(r_i)
        r_s = compute_token_reveal_curve(full_ids, target_start, target_end,
                                          shuffled=True, rng_shuf=rng_shuf)
        if r_s is not None:
            r_s.update({'doc_id': doc['doc_id'], 'corpus': doc['corpus'],
                        'target_frac': frac, 'n_tokens': n})
            shuffled.append(r_s)
    return intact, shuffled

print("Functions defined")
"""))

CELLS.append(code("""# === Run computation (or load cache) ===
results_path = BASE_DIR / "music_intact_v1.json"
shuffled_path = BASE_DIR / "music_shuffled_v1.json"

if results_path.exists() and shuffled_path.exists():
    with open(results_path) as f: all_intact = json.load(f)
    with open(shuffled_path) as f: all_shuffled = json.load(f)
    print(f"Loaded {len(all_intact)} intact + {len(all_shuffled)} shuffled curves from cache")
else:
    all_intact, all_shuffled = [], []
    rng_shuf = np.random.RandomState(RANDOM_SEED + 99)
    for doc in tqdm(corpus, desc="Processing recordings"):
        intact, shuffled = process_document(doc, rng_shuf)
        all_intact.extend(intact)
        all_shuffled.extend(shuffled)
    with open(results_path, 'w') as f: json.dump(all_intact, f)
    with open(shuffled_path, 'w') as f: json.dump(all_shuffled, f)
    print(f"Computed {len(all_intact)} intact + {len(all_shuffled)} shuffled curves")

unique_docs = set(c['doc_id'] for c in all_intact)
print(f"\\nUnique recordings with curves: {len(unique_docs)}")
for cname in CORPUS_FILES:
    ids = set(c['doc_id'] for c in all_intact if c['corpus'] == cname)
    n_curves = sum(1 for c in all_intact if c['corpus'] == cname)
    print(f"  {cname}: {len(ids)} recordings, {n_curves} curves")
"""))

CELLS.append(md("""## 1. Aggregate Corrected Power Law, by Corpus

The headline comparison. For each corpus, fit α on the mean corrected marginal across all its curves.
"""))

CELLS.append(code("""# === Analysis helpers (identical to PERSUADE/Buckeye) ===
common_x = np.arange(1, MAX_CONTEXT + 1)
bin_edges = [3, 6, 9, 15, 24, 36, 54, 81, 120, 180, MAX_CONTEXT]  # all multiples of 3

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

per_corpus_fits = {}
per_corpus_curves = {}
for cname in CORPUS_FILES:
    ci = [c for c in all_intact if c['corpus'] == cname]
    cs = [c for c in all_shuffled if c['corpus'] == cname]
    if len(ci) < 10:
        continue
    ip = compute_raw_ppl_curve(ci)
    sp = compute_raw_ppl_curve(cs)
    im = -np.diff(ip)
    sm = -np.diff(sp)
    corr = im - sm
    per_corpus_curves[cname] = {'intact_ppl': ip, 'shuf_ppl': sp,
                                  'intact_marg': im, 'shuf_marg': sm,
                                  'corrected': corr}
    fit = fit_power_law(corr)
    if fit:
        per_corpus_fits[cname] = fit
        slope, r, p, *_ = fit
        n_docs = len(set(c['doc_id'] for c in ci))
        print(f"  {cname:<14} (n_docs={n_docs:>3}, n_curves={len(ci):>4}): α = {slope:+.3f}  (r = {r:+.3f}, p = {p:.3g})")

print("\\nLanguage anchors for comparison:")
print("  RAID written (human)  α = -0.75")
print("  Buckeye spoken        α = -0.73")
print("  French oral           α = -0.69")
print("  Anderson & Schooler   α = -0.77")
print("  RAID written (AI)     α = -1.97")
"""))

CELLS.append(code("""# === Figure 1: per-corpus power-law fit + cross-corpus overlay ===
CORPUS_COLORS = {'improvised': '#1f78b4', 'composed': '#e31a1c', 'ai_generated': '#6a3d9a'}

fig, axes = plt.subplots(2, 3, figsize=(22, 11))

# A: Raw ppl intact vs shuffled, per corpus
ax = axes[0, 0]
for cname, cv in per_corpus_curves.items():
    c = CORPUS_COLORS.get(cname, 'gray')
    ax.plot(common_x, cv['intact_ppl'], '-', color=c, linewidth=2, label=f'{cname} intact')
    ax.plot(common_x, cv['shuf_ppl'], ':', color=c, linewidth=2, label=f'{cname} shuffled')
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Perplexity')
ax.set_title('A. Raw perplexity: intact vs shuffled', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# B: Marginal intact vs shuffled
ax = axes[0, 1]
for cname, cv in per_corpus_curves.items():
    c = CORPUS_COLORS.get(cname, 'gray')
    ax.plot(common_x[1:], uniform_filter1d(cv['intact_marg'], 5), '-', color=c, linewidth=2, label=f'{cname} intact')
    ax.plot(common_x[1:], uniform_filter1d(cv['shuf_marg'], 5), ':', color=c, linewidth=2, label=f'{cname} shuffled')
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Marginal PPL drop per token')
ax.set_title('B. Marginal gain: intact vs shuffled', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# C: Corrected marginals overlay
ax = axes[0, 2]
for cname, cv in per_corpus_curves.items():
    c = CORPUS_COLORS.get(cname, 'gray')
    ax.plot(common_x[1:], uniform_filter1d(cv['corrected'], 5),
            color=c, linewidth=2, label=cname)
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Corrected marginal (intact - shuffled)')
ax.set_title('C. Pure coherence signal', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# D: Power-law fit per corpus (log-log)
ax = axes[1, 0]
for cname, fit in per_corpus_fits.items():
    slope, r, p, bc, bm, inter = fit
    c = CORPUS_COLORS.get(cname, 'gray')
    ax.plot(bc, bm, 'o-', color=c, linewidth=2, markersize=7,
            label=f'{cname}: α={slope:+.2f} (r={r:+.2f})')
    fit_x = np.linspace(min(bc), max(bc), 100)
    ax.plot(fit_x, np.exp(inter) * fit_x ** slope, '--', color=c, alpha=0.4)
ax.set_xscale('log')
ax.set_xlabel('Context distance (tokens, log)')
ax.set_ylabel('Corrected marginal benefit')
ax.set_title('D. Power-law fit (corrected)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# E: α bar chart vs language anchors
ax = axes[1, 1]
bars_labels, bars_vals, bars_colors = [], [], []
for cname, fit in per_corpus_fits.items():
    bars_labels.append(f'Music\\n({cname})')
    bars_vals.append(fit[0])
    bars_colors.append(CORPUS_COLORS.get(cname, 'gray'))
# Language anchors
for lbl, val, color in [
    ('RAID written\\n(human)', -0.75, '#33a02c'),
    ('Buckeye\\n(spoken)', -0.73, '#b2df8a'),
    ('French oral', -0.69, '#a6cee3'),
    ('Anderson &\\nSchooler', -0.77, 'gray'),
    ('RAID\\n(AI)', -1.97, '#fb9a99'),
]:
    bars_labels.append(lbl)
    bars_vals.append(val)
    bars_colors.append(color)
x = range(len(bars_labels))
ax.bar(x, bars_vals, color=bars_colors, alpha=0.8, edgecolor='black')
for i, v in enumerate(bars_vals):
    ax.text(i, v - 0.08, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(-0.72, color='black', linestyle=':', alpha=0.4, label='Human language band −0.72')
ax.set_xticks(x); ax.set_xticklabels(bars_labels, fontsize=9)
ax.set_ylabel('Decay exponent α')
ax.set_title('E. Music α vs language anchors', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')

# F: Cumulative corrected benefit
ax = axes[1, 2]
for cname, cv in per_corpus_curves.items():
    c = CORPUS_COLORS.get(cname, 'gray')
    cum = np.cumsum(cv['corrected'])
    if cum[-1] > 0:
        ax.plot(common_x[1:], cum / cum[-1], color=c, linewidth=2, label=cname)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Fraction of total corrected benefit')
ax.set_title('F. Cumulative coherence benefit', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle('Music Coherence Decay: Improvised vs Composed',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'fig1_music_corpus_alpha.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

CELLS.append(md("""## 2. Per-Recording α Distribution

Like the per-speaker Buckeye plot. Fit α on each recording individually, look at the distribution within each corpus, and compare distributions.
"""))

CELLS.append(code("""# === Per-recording α fits ===
rec_rows = []
for doc_id in sorted(set(c['doc_id'] for c in all_intact)):
    doc_intact = [c for c in all_intact if c['doc_id'] == doc_id]
    doc_shuf = [c for c in all_shuffled if c['doc_id'] == doc_id]
    if len(doc_intact) < 2 or len(doc_shuf) < 2:
        continue
    ip = compute_raw_ppl_curve(doc_intact)
    sp = compute_raw_ppl_curve(doc_shuf)
    corr = -np.diff(ip) - (-np.diff(sp))
    fit = fit_power_law(corr)
    if fit is None:
        continue
    slope, r, p, *_ = fit
    m = doc_intact[0]
    rec_rows.append({
        'doc_id': doc_id, 'corpus': m['corpus'],
        'n_tokens': m['n_tokens'], 'alpha': slope, 'r': r, 'p': p,
        'n_curves': len(doc_intact),
    })
df_rec = pd.DataFrame(rec_rows)
print(f"Per-recording fits: {len(df_rec)} / {len(set(c['doc_id'] for c in all_intact))} recordings")
df_rec.to_csv(BASE_DIR / 'music_per_recording_alpha.csv', index=False)

print("\\nPer-corpus α summary (per-recording fits):")
print(df_rec.groupby('corpus').agg(
    n=('alpha','count'),
    mean=('alpha','mean'),
    median=('alpha','median'),
    sd=('alpha','std'),
    mean_r=('r','mean'),
).round(3))

# Between-corpus test if we have improvised AND composed
if {'improvised','composed'}.issubset(set(df_rec['corpus'].unique())):
    imp = df_rec[df_rec['corpus']=='improvised']['alpha']
    com = df_rec[df_rec['corpus']=='composed']['alpha']
    t, p = stats.ttest_ind(imp, com, equal_var=False)
    u, pu = stats.mannwhitneyu(imp, com, alternative='two-sided')
    print(f"\\nImprovised vs Composed per-recording α:")
    print(f"  Welch t = {t:+.2f}, p = {p:.3g}")
    print(f"  Mann-Whitney U = {u:.0f}, p = {pu:.3g}")
    print(f"  Δ mean α (improvised − composed) = {imp.mean() - com.mean():+.3f}")
"""))

CELLS.append(code("""# === Figure 2: per-recording α distributions ===
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# A: overlapping histograms
ax = axes[0]
for cname in df_rec['corpus'].unique():
    vals = df_rec.loc[df_rec['corpus']==cname, 'alpha']
    ax.hist(vals, bins=25, alpha=0.5, color=CORPUS_COLORS.get(cname, 'gray'),
            label=f'{cname} (n={len(vals)}, μ={vals.mean():.2f})')
ax.axvline(-0.72, color='black', linestyle=':', alpha=0.6, label='Human language band −0.72')
ax.axvline(-1.97, color='red', linestyle=':', alpha=0.4, label='RAID AI −1.97')
ax.set_xlabel('Per-recording α')
ax.set_ylabel('Count')
ax.set_title('A. Per-recording α distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# B: boxplot by corpus
ax = axes[1]
corpora_order = [c for c in ['improvised','composed','ai_generated'] if c in df_rec['corpus'].unique()]
data = [df_rec.loc[df_rec['corpus']==c, 'alpha'].values for c in corpora_order]
bp = ax.boxplot(data, labels=corpora_order, patch_artist=True, showmeans=True)
for patch, c in zip(bp['boxes'], corpora_order):
    patch.set_facecolor(CORPUS_COLORS.get(c, 'gray')); patch.set_alpha(0.6)
ax.axhline(-0.72, color='black', linestyle=':', alpha=0.6)
ax.set_ylabel('Per-recording α')
ax.set_title('B. Per-recording α by corpus', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(BASE_DIR / 'fig2_music_per_recording.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

CELLS.append(md("""## 3. Sanity Checks (REQUIRED)

Any null or positive result on α must first pass these controls. Adapted from hard lessons in LRTIA v1/v2.

1. Uniform-random token baseline: corrected marginal ≈ 0 at all distances.
2. Shuffled raw marginals show a distributional-calibration bump at short distance.
3. Per-target-position stability: α at 50%/70%/85% within ~0.15.
4. Intact raw curve drops monotonically.
"""))

CELLS.append(code("""# === 1. Uniform-random token baseline ===
# Generate synthetic docs with uniformly-random tokens over the model vocab,
# then run the same pipeline. We expect corrected marginal to sit at 0.

VOCAB_SIZE = model.config.vocab_size
print(f"Model vocab size: {VOCAB_SIZE}")

N_RANDOM_DOCS = 10
random_intact, random_shuffled = [], []
rng_syn = np.random.RandomState(RANDOM_SEED + 7)
rng_shuf_syn = np.random.RandomState(RANDOM_SEED + 17)

for i in tqdm(range(N_RANDOM_DOCS), desc="Random baseline"):
    n = 2000  # well above MIN_CONTEXT_BEFORE_TARGET
    fake = rng_syn.randint(0, VOCAB_SIZE, size=n).tolist()
    doc = {'doc_id': f'random_{i}', 'corpus': 'random', 'tokens': fake, 'n_tokens': n}
    intact, shuffled = process_document(doc, rng_shuf_syn)
    random_intact.extend(intact)
    random_shuffled.extend(shuffled)

if random_intact:
    rip = compute_raw_ppl_curve(random_intact)
    rsp = compute_raw_ppl_curve(random_shuffled)
    r_corr = -np.diff(rip) - (-np.diff(rsp))
    print(f"Random baseline corrected marginal: mean = {np.nanmean(r_corr):+.4f}, |max| = {np.nanmax(np.abs(r_corr)):.4f}")
    r_fit = fit_power_law(r_corr)
    if r_fit:
        print(f"Random baseline fit (should be unstable / near-zero): α = {r_fit[0]:+.3f}, r = {r_fit[1]:+.2f}")
    print("PASS" if np.nanmax(np.abs(r_corr)) < 0.05 else "WARN: non-trivial signal on random tokens — check probe.")
"""))

CELLS.append(code("""# === 2. Shuffled calibration-bump check ===
# Raw shuffled marginals should be nonzero at short distances (model adjusting
# to the vocab distribution) before flattening. If shuffled is flat everywhere,
# distributional calibration isn't happening and the correction does nothing.

fig, ax = plt.subplots(figsize=(10, 5))
for cname, cv in per_corpus_curves.items():
    c = CORPUS_COLORS.get(cname, 'gray')
    ax.plot(common_x[1:], uniform_filter1d(cv['shuf_marg'], 5), '-', color=c,
            linewidth=2, label=f'{cname} (shuffled)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Shuffled marginal')
ax.set_title('Sanity 2: shuffled raw marginals (should show short-distance bump)',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(BASE_DIR / 'sanity_2_shuffled_bump.png', dpi=150, bbox_inches='tight')
plt.show()

for cname, cv in per_corpus_curves.items():
    early = np.nanmean(cv['shuf_marg'][:10])
    late = np.nanmean(cv['shuf_marg'][100:])
    print(f"  {cname:<14} shuffled marg: early (1-10) = {early:+.4f}, late (100+) = {late:+.4f}")
"""))

CELLS.append(code("""# === 3. Per-target-position stability ===
fig, axes = plt.subplots(1, len(per_corpus_fits), figsize=(7*len(per_corpus_fits), 5),
                         squeeze=False)
for ax_i, cname in enumerate(per_corpus_fits):
    ax = axes[0, ax_i]
    for frac in TARGET_FRACTIONS:
        ci = [c for c in all_intact if c['corpus']==cname and c['target_frac']==frac]
        cs = [c for c in all_shuffled if c['corpus']==cname and c['target_frac']==frac]
        if len(ci) < 5:
            continue
        corr = -np.diff(compute_raw_ppl_curve(ci)) - (-np.diff(compute_raw_ppl_curve(cs)))
        fit = fit_power_law(corr)
        if fit:
            slope, r, p, bc, bm, inter = fit
            ax.plot(bc, bm, 'o-', markersize=6,
                    label=f'frac={frac:.0%}: α={slope:+.2f} (r={r:+.2f})')
    ax.set_xscale('log')
    ax.set_xlabel('Context distance (tokens)')
    ax.set_ylabel('Corrected marginal')
    ax.set_title(f'{cname}: α stability by target position', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(BASE_DIR / 'sanity_3_target_stability.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

CELLS.append(code("""# === 4. Intact raw curve monotonicity check ===
for cname, cv in per_corpus_curves.items():
    ip = cv['intact_ppl']
    # Is ppl decreasing monotonically? Compute Spearman of (ctx_len, ppl)
    valid = ~np.isnan(ip)
    r_mono, p_mono = stats.spearmanr(common_x[valid], ip[valid])
    print(f"  {cname:<14} intact ppl vs ctx_len: Spearman r = {r_mono:+.3f}, "
          f"drop = {ip[valid][0]:.2f} → {ip[valid][-1]:.2f} "
          f"({(1 - ip[valid][-1]/ip[valid][0])*100:.1f}% reduction)")
"""))

CELLS.append(md("""## 4. Cross-Domain Anchor Table

Assemble the final comparison table across language corpora (RAID, Buckeye, French Oral) and music corpora. This is the number that goes in the paper.
"""))

CELLS.append(code("""anchors = [
    ('RAID written (human)', 'English', 'Written text', -0.75, -0.87, None),
    ('Buckeye',              'English', 'Spoken text',  -0.73, -0.93, None),
    ('French oral narrative','French',  'Spoken text',  -0.69, -0.85, None),
    ('Anderson & Schooler',  '—',       'Memory retrieval', -0.77, None, None),
    ('RAID written (AI)',    'English', 'Written text', -1.97, -0.95, None),
]
for cname, fit in per_corpus_fits.items():
    slope, r, p, *_ = fit
    label = f'Music ({cname})'
    modality = 'Symbolic MIDI'
    anchors.append((label, '—', modality, slope, r, p))

df_anchor = pd.DataFrame(anchors, columns=['Corpus','Language','Modality','α','r','p'])
print(df_anchor.to_string(index=False))
df_anchor.to_csv(BASE_DIR / 'cross_domain_alpha_table.csv', index=False)
"""))

CELLS.append(md("""## Summary

Key deliverables of this run:
- **Per-corpus α** — the headline comparison (is improvised α within 0.1 of −0.72?).
- **Per-recording α distribution** — individual-level stability within each corpus and between-corpus test.
- **Cross-domain table** — music α side-by-side with all language anchors.
- **Sanity pass** — uniform-random baseline, shuffled calibration bump, target-position stability, intact monotonicity.

**Positive result interpretation**: if improvised α lands near −0.72, this is a genuinely new claim — α is a universal of biological WM-constrained sequential generation across language AND music. That's a standalone paper, not a footnote to the current one.
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
