#!/usr/bin/env python3
"""
Contextual Persistence Function on DNA — non-language structured-sequence control.

Runs the EXACT long-range persistence protocol used for the language corpora, but
with a domain-native autoregressive probe (HyenaDNA, single-nucleotide next-token
model) on genomic sequence. Output JSON matches the language-cell format, so
build_figures.py / the analysis pipeline pick it up directly.

Protocol (identical to Corpus_Expansion_LongRange):
  - one 30-token target per sequence at the 50% position
  - log-spaced context lengths [0,1,2,4,8,16,32,64,128,256,512,1024]
  - ordered perplexity vs shuffled-context perplexity at each length
  - P(d) = ordered_marginal - shuffled_marginal  (computed downstream)

RUN ON GPU (Colab A100 or similar). ~single-nucleotide model, fast.
This is a TEMPLATE: verify the two spots marked [VERIFY] against the model card,
then Run all. Hand back dna_hyenadna.json.

Probe:  LongSafari/hyenadna-medium-160k-seqlen-hf   (autoregressive, char-level A/C/G/T/N)
        (swap to Evo 2 `arcinstitute/evo2_7b` for a stronger, heavier probe.)
Data:   human GRCh38 chromosome 21, sampled into non-overlapping windows.
"""

import json, math, random, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config
MODEL_NAME   = "LongSafari/hyenadna-medium-160k-seqlen-hf"
OUT_PATH     = Path("dna_hyenadna.json")
CTX_LENGTHS  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MAX_CTX      = max(CTX_LENGTHS)
TARGET_LEN   = 30
TARGET_FRAC  = 0.5
MIN_LEN      = MAX_CTX + TARGET_LEN + 50          # 1104 nucleotides
N_DOCS       = 60                                 # match language cells
WINDOW_LEN   = 4000                               # nucleotides per sampled window
SEED         = 20260715
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------- probe
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto"
).eval()
print(f"probe loaded: {MODEL_NAME}")


@torch.no_grad()
def ppl_of_target(ctx_ids, tgt_ids):
    """Perplexity of tgt_ids conditioned on ctx_ids (mean per-token NLL, then exp)."""
    if len(tgt_ids) < 2:
        return float("inf")
    ids = torch.tensor([list(ctx_ids) + list(tgt_ids)], device=model.device)
    logits = model(ids).logits[0]
    ts = len(ctx_ids)
    nll, cnt = 0.0, 0
    for i in range(ts, len(ids[0]) - 1):
        lp = torch.log_softmax(logits[i].float(), dim=-1)
        nll += -lp[ids[0][i + 1]].item(); cnt += 1
    del logits; torch.cuda.empty_cache()
    return math.exp(nll / cnt) if cnt else float("inf")


def longrange_curves(full_ids, ts, te):
    tgt = full_ids[ts:te]
    o_ppl, s_ppl = [], []
    for c in CTX_LENGTHS:
        pfx = [] if c == 0 else full_ids[ts - c:ts]
        o_ppl.append(ppl_of_target(pfx, tgt))
        if c == 0:
            s_ppl.append(o_ppl[-1])                # nothing to shuffle at c=0
        else:
            rng = random.Random(SEED + c)
            sh = list(pfx); rng.shuffle(sh)
            s_ppl.append(ppl_of_target(sh, tgt))
    return {"context_lengths": list(CTX_LENGTHS),
            "ordered_ppl": o_ppl, "shuffled_ppl": s_ppl}


# ---------------------------------------------------------------- data: human GRCh38 chr21 windows
def load_dna_windows():
    """Download one human chromosome and cut non-overlapping windows of ACGT-only sequence."""
    import urllib.request, gzip, io
    # [VERIFY] URL — Ensembl GRCh38 chromosome 21 (soft-masked repeats are lowercased; we upper-case).
    url = ("https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
           "Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz")
    print("downloading", url)
    raw = urllib.request.urlopen(url).read()
    text = gzip.decompress(raw).decode()
    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">")).upper()
    # keep only ACGT stretches; drop long N runs (telomeres/centromere)
    windows = []
    step = WINDOW_LEN
    for i in range(0, len(seq) - WINDOW_LEN, step):
        w = seq[i:i + WINDOW_LEN]
        if w.count("N") == 0 and len(w) >= MIN_LEN:
            windows.append((f"chr21_win_{i}", w))
        if len(windows) >= N_DOCS * 3:             # oversample, we filter by token length below
            break
    random.shuffle(windows)
    return windows


docs = load_dna_windows()
print(f"{len(docs)} candidate windows")

# ---------------------------------------------------------------- main loop
results, skipped = [], 0
t0 = time.time()
for doc_id, seq in docs:
    if len(results) >= N_DOCS:
        break
    ids = tokenizer(seq, add_special_tokens=False)["input_ids"]   # [VERIFY] key name
    n = len(ids)
    if n < MIN_LEN:
        skipped += 1; continue
    rem_start, rem_end = MAX_CTX, n - TARGET_LEN
    ts = int(rem_start + TARGET_FRAC * (rem_end - rem_start))
    te = ts + TARGET_LEN
    if ts - MAX_CTX < 0 or te > n:
        skipped += 1; continue
    r = longrange_curves(ids, ts, te)
    r.update(corpus_id="dna_hyenadna", document_id=doc_id,
             target_id=f"{doc_id}__pos50", target_frac=TARGET_FRAC)
    results.append(r)
    if len(results) % 10 == 0:
        print(f"  {len(results)}/{N_DOCS}  ({(time.time()-t0)/60:.1f} min)")

json.dump(results, open(OUT_PATH, "w"))
print(f"wrote {len(results)} records -> {OUT_PATH}  ({skipped} skipped)")

# quick sanity: order-specific gap at the longest interval should be >= ~0
def gap_last(recs):
    vals = []
    for e in recs:
        op, sp = e["ordered_ppl"], e["shuffled_ppl"]
        vals.append((op[-2] - op[-1]) - (sp[-2] - sp[-1]))
    return float(np.mean(vals))
print("mean long-range order-specific gap:", gap_last(results))
