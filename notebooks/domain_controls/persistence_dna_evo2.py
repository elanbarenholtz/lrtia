#!/usr/bin/env python3
"""
Contextual Persistence Function on DNA — STRONGER PROBE (Evo 2-7B).

Same protocol as persistence_dna_hyenadna.py, but with a state-of-the-art genomic
autoregressive model (Evo 2-7B, Arc Institute) as the probe. Purpose: preempt the
referee objection "your DNA model was too weak to see the structure." If the near-zero,
non-power-law result holds with Evo 2, the "DNA is off the law" conclusion is bulletproof.

Output JSON matches the language-cell format.

RUN ON GPU (A100 40GB fine; Evo 2 prefers H100 FP8 but runs in bf16 on A100).

--- INSTALL NOTE ---
Evo 2 has a heavier install than HyenaDNA (vortex / transformer_engine / flash-attn).
On Colab A100:  !pip install evo2
If FP8 errors appear (A100 is Ampere, not Hopper), Evo 2 falls back to bf16 automatically
in recent versions; if not, use the HyenaDNA-large FALLBACK below — a guaranteed-to-run
one-line switch that is still a much stronger probe than the medium model.
"""

import json, math, random, time
from pathlib import Path
import numpy as np
import torch

# ---------------------------------------------------------------- config
OUT_PATH     = Path("dna_evo2.json")
CTX_LENGTHS  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MAX_CTX      = max(CTX_LENGTHS)
TARGET_LEN   = 30
TARGET_FRAC  = 0.5
MIN_LEN      = MAX_CTX + TARGET_LEN + 50
N_DOCS       = 60
WINDOW_LEN   = 4000
SEED         = 20260715
random.seed(SEED); np.random.seed(SEED)

USE_EVO2 = True   # set False to use the HyenaDNA-large-1M fallback (clean HF interface)

# ---------------------------------------------------------------- probe
if USE_EVO2:
    # pip install evo2
    from evo2 import Evo2
    evo2 = Evo2("evo2_7b")
    DEVICE = "cuda:0"
    print("probe loaded: Evo 2-7B")

    def tokenize(seq):
        return list(evo2.tokenizer.tokenize(seq))

    @torch.no_grad()
    def ppl_of_target(ctx_ids, tgt_ids):
        if len(tgt_ids) < 2:
            return float("inf")
        full = list(ctx_ids) + list(tgt_ids)
        ids = torch.tensor([full], dtype=torch.int, device=DEVICE)
        outputs, _ = evo2(ids)                 # outputs: [1, seqlen, vocab]
        logits = outputs[0]
        ts = len(ctx_ids); nll, cnt = 0.0, 0
        for i in range(ts, len(full) - 1):
            lp = torch.log_softmax(logits[i].float(), dim=-1)
            nll += -lp[full[i + 1]].item(); cnt += 1
        del outputs, logits; torch.cuda.empty_cache()
        return math.exp(nll / cnt) if cnt else float("inf")
else:
    # ---- FALLBACK: HyenaDNA-large-1M (clean HF causal LM, guaranteed to run on A100) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    NAME = "LongSafari/hyenadna-large-1m-seqlen-hf"
    tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
    hmodel = AutoModelForCausalLM.from_pretrained(
        NAME, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto").eval()
    print("probe loaded: HyenaDNA-large-1M (fallback)")

    def tokenize(seq):
        return tok(seq, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def ppl_of_target(ctx_ids, tgt_ids):
        if len(tgt_ids) < 2:
            return float("inf")
        ids = torch.tensor([list(ctx_ids) + list(tgt_ids)], device=hmodel.device)
        logits = hmodel(ids).logits[0]
        ts = len(ctx_ids); nll, cnt = 0.0, 0
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
            s_ppl.append(o_ppl[-1])
        else:
            rng = random.Random(SEED + c)
            sh = list(pfx); rng.shuffle(sh)
            s_ppl.append(ppl_of_target(sh, tgt))
    return {"context_lengths": list(CTX_LENGTHS),
            "ordered_ppl": o_ppl, "shuffled_ppl": s_ppl}


# ---------------------------------------------------------------- data: human GRCh38 chr21 windows
def load_dna_windows():
    import urllib.request, gzip
    url = ("https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
           "Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz")
    print("downloading", url)
    raw = urllib.request.urlopen(url).read()
    text = gzip.decompress(raw).decode()
    seq = "".join(l.strip() for l in text.splitlines() if not l.startswith(">")).upper()
    windows, step = [], WINDOW_LEN
    for i in range(0, len(seq) - WINDOW_LEN, step):
        w = seq[i:i + WINDOW_LEN]
        if w.count("N") == 0 and len(w) >= MIN_LEN:
            windows.append((f"chr21_win_{i}", w))
        if len(windows) >= N_DOCS * 3:
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
    ids = tokenize(seq)
    n = len(ids)
    if n < MIN_LEN:
        skipped += 1; continue
    ts = int(MAX_CTX + TARGET_FRAC * ((n - TARGET_LEN) - MAX_CTX))
    te = ts + TARGET_LEN
    if ts - MAX_CTX < 0 or te > n:
        skipped += 1; continue
    r = longrange_curves(ids, ts, te)
    r.update(corpus_id="dna_evo2", document_id=doc_id,
             target_id=f"{doc_id}__pos50", target_frac=TARGET_FRAC)
    results.append(r)
    if len(results) % 5 == 0:
        print(f"  {len(results)}/{N_DOCS}  ({(time.time()-t0)/60:.1f} min)")

json.dump(results, open(OUT_PATH, "w"))
print(f"wrote {len(results)} records -> {OUT_PATH}  ({skipped} skipped)")

# quick in-notebook analysis (same as the language cells)
from scipy import stats
by_i = {}
for e in results:
    cs, op, sp = e["context_lengths"], e["ordered_ppl"], e["shuffled_ppl"]
    for i in range(1, len(cs)):
        if cs[i-1] == 0: continue
        w = cs[i]-cs[i-1]; d = (cs[i-1]*cs[i])**0.5
        g = (op[i-1]-op[i])/w - (sp[i-1]-sp[i])/w
        by_i.setdefault(i, (d, []))[1].append(g)
d = np.array([by_i[i][0] for i in sorted(by_i)])
y = np.array([np.mean(by_i[i][1]) for i in sorted(by_i)])
print("\nd      P(d)")
for a, b in zip(d, y): print(f"{a:7.2f}  {b:+.5f}")
m = (d >= 10) & (y > 0)
if m.sum() >= 4:
    s, _, r, _, _ = stats.linregress(np.log(d[m]), np.log(y[m]))
    print(f"\nslope α = {s:.3f}, r² = {r**2:.3f}, positive bins d≥10: {m.sum()}/{(d>=10).sum()}")
else:
    print(f"\nno clean power law — {m.sum()} positive bins at d≥10")
