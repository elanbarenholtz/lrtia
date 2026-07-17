#!/usr/bin/env python3
"""
Contextual Persistence Function on symbolic MUSIC — human non-language control.

Probe: Anticipatory Music Transformer (AMT), GPT-2 over arrival-time MIDI tokens
(Stanford CRFM, trained on Lakh MIDI). Data: MAESTRO piano MIDI.
Output JSON matches the language-cell format.

This version has the fixes baked in so a clean Run-all works:
  * arrival-time REBASE per window (midi_to_events emits absolute times that exceed
    MAX_TIME=10000 for long pieces; we rebase each window's TIME tokens to start ~0),
  * vectorized per-token NLL (fast),
  * progress prints.

RUN ON GPU (A100). Writes music_amt.json + prints the P(d) table and slope.
"""

import json, math, random, time, glob, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------- config
MODEL_NAME   = "stanford-crfm/music-medium-800k"
OUT_PATH     = Path("music_amt.json")
MIDI_DIR     = "midi"
TARGET_LEN   = 30
TARGET_FRAC  = 0.5
N_DOCS       = 60
SEED         = 20260715
MAX_TIME_V   = 10000     # TIME_OFFSET=0 -> valid time tokens in [0, MAX_TIME_V)
VOCAB        = 55028     # AMT arrival-time vocab (matches checkpoint)
random.seed(SEED); np.random.seed(SEED)

from anticipation.convert import midi_to_events

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
print(f"probe loaded: {MODEL_NAME}")

# The AMT is a GPT-2 with a fixed maximum context (n_positions). Cap the context
# grid so context + target never exceeds it (else the position embedding asserts).
MAXPOS = getattr(model.config, "n_positions", None) or getattr(model.config, "n_ctx", 1024)
_base  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CTX_LENGTHS = [c for c in _base if c + TARGET_LEN <= MAXPOS]
if CTX_LENGTHS[-1] < MAXPOS - TARGET_LEN:
    CTX_LENGTHS.append(MAXPOS - TARGET_LEN)
MAX_CTX = max(CTX_LENGTHS)
MIN_LEN = MAX_CTX + TARGET_LEN + 50
print(f"model max positions: {MAXPOS}  ->  context grid: {CTX_LENGTHS}")


@torch.no_grad()
def ppl_of_target(ctx_ids, tgt_ids):
    """Vectorized: mean per-token NLL of tgt given ctx, then exp."""
    if len(tgt_ids) < 2:
        return float("inf")
    full = list(ctx_ids) + list(tgt_ids)
    ids = torch.tensor([full], device=model.device)
    logits = model(ids).logits[0]                     # [L, V]
    ts, L = len(ctx_ids), len(full)
    lp = torch.log_softmax(logits[ts:L-1].float(), dim=-1)
    nxt = torch.tensor(full[ts+1:L], device=model.device)
    nll = -lp[torch.arange(lp.shape[0], device=model.device), nxt].mean().item()
    del logits; torch.cuda.empty_cache()
    return float(np.exp(nll))


def longrange_curves(W, ts, te):
    tgt = W[ts:te]
    o, s = [], []
    for c in CTX_LENGTHS:
        pfx = [] if c == 0 else W[ts-c:ts]
        o.append(ppl_of_target(pfx, tgt))
        if c == 0:
            s.append(o[-1])
        else:
            sh = list(pfx); random.Random(SEED+c).shuffle(sh)
            s.append(ppl_of_target(sh, tgt))
    return {"context_lengths": list(CTX_LENGTHS), "ordered_ppl": o, "shuffled_ppl": s}


def rebased_window(ev, start, end):
    """Slice ev[start:end]; rebase TIME tokens (global idx %3==0) to start near 0."""
    W = list(ev[start:end])
    times = [W[i] for i in range(len(W)) if (start + i) % 3 == 0]
    base = min(times) if times else 0
    for i in range(len(W)):
        if (start + i) % 3 == 0:
            W[i] = min(max(W[i] - base, 0), MAX_TIME_V - 1)
        W[i] = min(W[i], VOCAB - 1)
    return W


# ---------------------------------------------------------------- data
def download_maestro_if_needed():
    if glob.glob(os.path.join(MIDI_DIR, "**", "*.mid*"), recursive=True):
        return
    import urllib.request, zipfile, io
    os.makedirs(MIDI_DIR, exist_ok=True)
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    print("downloading MAESTRO midi:", url)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
    z.extractall(MIDI_DIR)


download_maestro_if_needed()
paths = sorted(glob.glob(os.path.join(MIDI_DIR, "**", "*.mid*"), recursive=True))
random.shuffle(paths)

proc = []
for p in paths:
    try:
        ev = list(midi_to_events(p))         # flat triples (time, dur, note), phase 0
    except Exception:
        continue
    n = len(ev)
    if n < MIN_LEN:
        continue
    ts = int(MAX_CTX + TARGET_FRAC * ((n - TARGET_LEN) - MAX_CTX))
    ts -= ts % 3
    te = ts + TARGET_LEN
    if ts - MAX_CTX < 0 or te > n:
        continue
    W = rebased_window(ev, ts - MAX_CTX, te)  # length MAX_CTX + TARGET_LEN
    if max(W) >= VOCAB:
        continue
    proc.append((os.path.basename(p), W))
    if len(proc) >= N_DOCS:
        break
print(f"{len(proc)} valid rebased windows")

# ---------------------------------------------------------------- run
results = []
t0 = time.time()
for k, (doc_id, W) in enumerate(proc):
    r = longrange_curves(W, MAX_CTX, MAX_CTX + TARGET_LEN)
    r.update(corpus_id="music_amt", document_id=doc_id,
             target_id=f"{doc_id}__pos50", target_frac=TARGET_FRAC)
    results.append(r)
    if (k + 1) % 5 == 0:
        print(f"  {k+1}/{len(proc)}  ({(time.time()-t0)/60:.1f} min)")

json.dump(results, open(OUT_PATH, "w"))
print(f"wrote {len(results)} records -> {OUT_PATH}")

# ---------------------------------------------------------------- analysis
from scipy import stats
by_i = {}
for e in results:
    cs, op, sp = e["context_lengths"], e["ordered_ppl"], e["shuffled_ppl"]
    for i in range(1, len(cs)):
        if cs[i-1] == 0:
            continue
        w = cs[i]-cs[i-1]; d = (cs[i-1]*cs[i])**0.5
        by_i.setdefault(i, (d, []))[1].append((op[i-1]-op[i])/w - (sp[i-1]-sp[i])/w)
d = np.array([by_i[i][0] for i in sorted(by_i)])
y = np.array([np.mean(by_i[i][1]) for i in sorted(by_i)])
print("\nd      P(d)")
for a, b in zip(d, y):
    print(f"{a:7.2f}  {b:+.5f}")
m = (d >= 10) & (y > 0)
if m.sum() >= 4:
    sl, _, rr, _, _ = stats.linregress(np.log(d[m]), np.log(y[m]))
    print(f"\nalpha = {sl:.3f}, r^2 = {rr**2:.3f}, positive bins d>=10: {m.sum()}/{(d>=10).sum()}")
else:
    print(f"\nonly {m.sum()} positive bins at d>=10")
