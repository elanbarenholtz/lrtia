#!/usr/bin/env python3
"""
Contextual Persistence Function on PROTEIN — preregistered non-language control.

Probe: ProGen2 (autoregressive decoder transformer) at >=2 scales — matches the
language/DNA probes on objective and (vs language) architecture family, so a null
cannot be blamed on objective/architecture mismatch. See protein_control_prereg.md.

Run per the preregistration:
  * >=2 ProGen2 scales (small 151M, large 2.7B); capability = lower per-residue loss.
  * Probe validation: (1) per-residue perplexity on held-out natural protein,
    (2) short-range positive-control floor (ordered must beat shuffled at small d).
  * Identical CPF protocol: 30-token target at 50%, log-spaced context (capped to the
    model's 1024 context), shuffled-residue baseline, P(d) fit on d >= 10.
  * Scope: max probed distance ~1000 residues (stated, range-matched to language/DNA).

RUN ON GPU (A100). Prints, per scale: held-out per-residue perplexity, the P(d) table,
the fitted slope/r^2, and the short-range floor. Writes protein_progen2_<scale>.json.

Expect short-range non-monotonicity (alpha-helix ~3.6, beta-strand alternation) — that is
periodic structural order, the protein analog of DNA codon/nucleosome periodicity, NOT noise.
"""

import json, math, random, time, urllib.request, gzip
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

# ---------------------------------------------------------------- config (recorded before analysis)
SCALES = [("progen2-small", "hugohrban/progen2-small"),   # 151M
          ("progen2-large", "hugohrban/progen2-large")]   # 2.7B  (wide span; add medium/xlarge if desired)
OUT_DIR      = Path(".")
TARGET_LEN   = 30
TARGET_FRAC  = 0.5
N_DOCS       = 60
N_HELDOUT    = 200          # held-out proteins for the per-residue perplexity check
BASE_GRID    = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEED         = 20260715
AA = set("ACDEFGHIKLMNPQRSTVWY")
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------- data: Swiss-Prot
def load_swissprot():
    import os
    fp = "/content/uniprot_sprot.fasta.gz"
    url = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
           "knowledgebase/complete/uniprot_sprot.fasta.gz")
    if os.path.exists(fp):
        raw = open(fp, "rb").read()
    else:
        print("downloading Swiss-Prot:", url)
        raw = urllib.request.urlopen(url).read()
        open(fp, "wb").write(raw)
    text = gzip.decompress(raw).decode()
    seqs, cur_id, cur = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_id and cur:
                s = "".join(cur)
                if set(s) <= AA:
                    seqs.append((cur_id, s))
            cur_id = line[1:].split()[0]; cur = []
        else:
            cur.append(line.strip())
    if cur_id and cur and set("".join(cur)) <= AA:
        seqs.append((cur_id, "".join(cur)))
    return seqs

allseq = load_swissprot()
print(f"{len(allseq)} standard-AA Swiss-Prot proteins")

def ppl_vec(model, ids_full, ts):
    """Vectorized mean per-token NLL of tokens ts+1..end, given prefix (same convention as
    the language/DNA runs), then exp."""
    L = len(ids_full)
    if L - ts < 2:
        return float("inf")
    ids = torch.tensor([ids_full], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]
        lp = torch.log_softmax(logits[ts:L-1].float(), dim=-1)
        nxt = torch.tensor(ids_full[ts+1:L], device=model.device)
        nll = -lp[torch.arange(lp.shape[0], device=model.device), nxt].mean().item()
    del logits; torch.cuda.empty_cache()
    return float(np.exp(nll))


for scale_name, hf_id in SCALES:
    print("\n" + "="*70 + f"\nPROBE: {scale_name}  ({hf_id})\n" + "="*70)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # Dispatch the ENTIRE model onto GPU 0 via accelerate (device_map={"": 0}). This
    # materializes every weight from the checkpoint directly onto cuda:0 with no meta
    # tensors and no offload — the reliable cure for this custom port. Then re-materialize
    # any stray non-persistent buffers (causal mask etc.) that could still be on meta.
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": 0}).eval()
    # GPT-J (ProGen2) caches sinusoidal position embeddings as a PLAIN attribute created on
    # meta during load; forward then does embed_positions.to(cuda) which fails. Recreate any
    # meta plain-tensor attributes on cuda (2D -> sinusoidal position table, else zeros).
    def _make_sinusoidal(num_pos, dim, dtype):
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device="cuda") / dim))
        s = torch.einsum("i,j->ij", torch.arange(num_pos, dtype=torch.float, device="cuda"), inv)
        return torch.cat([torch.sin(s), torch.cos(s)], dim=1).to(dtype)
    for _mod in model.modules():
        for _bn, _b in list(getattr(_mod, "_buffers", {}).items()):
            if _b is not None and _b.is_meta:
                if _b.dim() >= 2 and _b.shape[-1] == _b.shape[-2]:
                    n = _b.shape[-1]
                    _mod._buffers[_bn] = torch.tril(torch.ones(n, n, dtype=_b.dtype, device="cuda")).view(_b.shape)
                else:
                    _mod._buffers[_bn] = torch.zeros(_b.shape, dtype=_b.dtype, device="cuda")
        for _an, _av in list(vars(_mod).items()):
            if torch.is_tensor(_av) and _av.is_meta:
                if _av.dim() == 2:
                    setattr(_mod, _an, _make_sinusoidal(_av.shape[0], _av.shape[-1], _av.dtype))
                else:
                    setattr(_mod, _an, torch.zeros(_av.shape, dtype=_av.dtype, device="cuda"))
    # The ProGen2 custom port calls self.get_head_mask(...), removed from base in recent
    # transformers. Restore it as a no-op so the forward pass runs unchanged.
    for _m in model.modules():
        if type(_m).__name__ == "ProGenModel" and not hasattr(_m, "get_head_mask"):
            type(_m).get_head_mask = (lambda self, head_mask, num_layers, *a, **k: [None]*num_layers)
    BOS = tok.bos_token_id                       # [VERIFY] ProGen2 start token
    def encode(seq):
        return tok(seq, add_special_tokens=False)["input_ids"]   # [VERIFY] 1 token / residue

    MAXPOS = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    GRID = [c for c in BASE_GRID if c + TARGET_LEN + (1 if BOS is not None else 0) <= MAXPOS]
    if GRID[-1] < MAXPOS - TARGET_LEN - 1:
        GRID.append(MAXPOS - TARGET_LEN - (1 if BOS is not None else 0))
    MAX_CTX = max(GRID); MIN_LEN = MAX_CTX + TARGET_LEN + 50
    print(f"max positions {MAXPOS}; context grid {GRID}")

    # ---- probe validation 1: per-residue perplexity on held-out natural protein ----
    held = [s for _, s in random.Random(SEED).sample(allseq, min(N_HELDOUT*3, len(allseq)))]
    held = [s[:MAXPOS-1] for s in held if len(s) >= 50][:N_HELDOUT]
    nlls = []
    for s in held:
        ids = ([BOS] if BOS is not None else []) + encode(s)
        L = len(ids)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=model.device)).logits[0]
            lp = torch.log_softmax(logits[:L-1].float(), dim=-1)
            nxt = torch.tensor(ids[1:L], device=model.device)
            nlls.append((-lp[torch.arange(L-1, device=model.device), nxt]).mean().item())
        del logits; torch.cuda.empty_cache()
    per_res_ppl = float(np.exp(np.mean(nlls)))
    print(f"[validation 1] held-out per-residue perplexity = {per_res_ppl:.3f}  "
          f"(compare to published ProGen2 {scale_name})")

    # ---- CPF on long proteins ----
    cpf = [(i, s) for i, s in allseq if len(s) >= MIN_LEN]
    random.Random(SEED).shuffle(cpf)
    cpf = cpf[:N_DOCS]
    print(f"{len(cpf)} proteins >= {MIN_LEN} residues for CPF")

    def curves(ids):
        ts0 = int(MAX_CTX + TARGET_FRAC * ((len(ids) - TARGET_LEN) - MAX_CTX))
        te0 = ts0 + TARGET_LEN
        tgt = ids[ts0:te0]
        o, s = [], []
        for c in GRID:
            pfx = [] if c == 0 else ids[ts0-c:ts0]
            full = ([BOS] if BOS is not None else []) + list(pfx) + list(tgt)
            ts = (1 if BOS is not None else 0) + len(pfx)
            o.append(ppl_vec(model, full, ts))
            if c == 0:
                s.append(o[-1])
            else:
                sh = list(pfx); random.Random(SEED+c).shuffle(sh)
                fs = ([BOS] if BOS is not None else []) + sh + list(tgt)
                s.append(ppl_vec(model, fs, ts))
        return {"context_lengths": list(GRID), "ordered_ppl": o, "shuffled_ppl": s}

    results = []; t0 = time.time()
    for k, (pid, s) in enumerate(cpf):
        ids = encode(s)
        if len(ids) < MIN_LEN:
            continue
        r = curves(ids)
        r.update(corpus_id=f"protein_{scale_name}", document_id=pid, target_frac=TARGET_FRAC)
        results.append(r)
        if (k+1) % 10 == 0:
            print(f"  {k+1}/{len(cpf)}  ({(time.time()-t0)/60:.1f} min)")
    outp = OUT_DIR / f"protein_{scale_name}.json"
    json.dump(results, open(outp, "w"))
    print(f"wrote {len(results)} -> {outp}")

    # ---- aggregate + report (median across proteins; robust to fp16 outliers) ----
    clean = [e for e in results if max(max(e["ordered_ppl"]), max(e["shuffled_ppl"])) < 1e4]
    by = {}
    for e in clean:
        cs, op, sp = e["context_lengths"], e["ordered_ppl"], e["shuffled_ppl"]
        for i in range(1, len(cs)):
            if cs[i-1] == 0: continue
            w = cs[i]-cs[i-1]; d = (cs[i-1]*cs[i])**0.5
            by.setdefault(i, (d, []))[1].append((op[i-1]-op[i])/w - (sp[i-1]-sp[i])/w)
    d = np.array([by[i][0] for i in sorted(by)])
    ymed = np.array([np.median(by[i][1]) for i in sorted(by)])
    print(f"\n[{scale_name}] {len(clean)}/{len(results)} proteins with sane ppl (<1e4)")
    print("d        median P(d)")
    for a, b in zip(d, ymed): print(f"{a:8.2f}  {b:+.5f}")
    # probe validation 2: short-range positive-control floor
    short = ymed[d < 10]
    print(f"[validation 2] short-range (d<10) median P(d): {short.tolist()}  "
          f"-> ordered>shuffled floor {'PRESENT' if np.nansum(short) > 0 else 'ABSENT (pipeline suspect)'}")
    m = (d >= 10) & (ymed > 0)
    if m.sum() >= 4:
        sl, _, rr, _, _ = stats.linregress(np.log(d[m]), np.log(ymed[m]))
        print(f"[{scale_name}] alpha = {sl:.3f}, r^2 = {rr**2:.3f}, positive bins d>=10: {m.sum()}/{(d>=10).sum()}")
    else:
        print(f"[{scale_name}] no clean power law — {m.sum()} positive bins at d>=10 (null-consistent)")

    del model; torch.cuda.empty_cache()

print("\nDONE. Capability-scaling check: compare per-residue perplexity vs the fitted "
      "structure across scales — if structure does NOT grow as perplexity falls, the null "
      "is not a capability ceiling.")
