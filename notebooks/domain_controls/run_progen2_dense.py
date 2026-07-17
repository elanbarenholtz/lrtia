#!/usr/bin/env python3
"""
Contextual Persistence Function (CPF) — PROTEIN control, ProGen2 (both scales).
DENSE-GRID version to pin the power-law exponent with a reviewer-grade CI.
Supersedes run_progen2_fixed.py for the exponent question.

What changed vs run_progen2_fixed.py
------------------------------------
1. DENSE log-spaced context grid: ~28 bins between d=10 and 993 (was 7), plus the
   short-range points [0,1,2,4,8] for the validation floor. The manuscript's P(d)
   is a *differenced* marginal, so more context lengths => more usable fit bins.
2. More documents (N_DOCS=300; bump to 500 if you want tighter CIs).
3. Exponent estimated with the MANUSCRIPT'S EXACT estimator (per_target_gap_rates +
   fit_pl below, copied verbatim from paper/build_figures.py) and a 1000x
   document-bootstrap CI — identical method to the language Table S1. This is the
   number that goes head-to-head with language's alpha = -1.04.

Loader (correct_load) is UNCHANGED from run_progen2_fixed.py: it restores the three
persistent=False attention tensors (bias/masked_bias/scale_attn) that device_map
leaves zeroed/on-meta. ProGen2 uses inline rotary embeddings -> no embed_positions.

    pip install transformers accelerate sentencepiece scipy      # A100 / fp16
    python run_progen2_dense.py
"""
import json, random, time, gzip, os, urllib.request, math
from pathlib import Path
import numpy as np, torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT     = Path("/content/drive/MyDrive/LRTIA/Results/domain_controls")
SP_PATH = "/content/uniprot_sprot.fasta.gz"
SP_URL  = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
           "knowledgebase/complete/uniprot_sprot.fasta.gz")
SCALES  = [("progen2-small", "hugohrban/progen2-small"),
           ("progen2-large", "hugohrban/progen2-large")]
TARGET_LEN, TARGET_FRAC, N_DOCS, N_HELDOUT = 30, 0.5, 300, 200
N_BOOT   = 1000
FIT_DMIN = 10.0                      # manuscript fit floor: d >= 10
SEED = 20260715
AA = set("ACDEFGHIKLMNPQRSTVWY")
random.seed(SEED); np.random.seed(SEED)

# Dense grid: short-range floor points + ~28 log-spaced bins out to the 1024 cap.
_dense = np.unique(np.round(np.geomspace(10, 993, 28)).astype(int)).tolist()
BASE_GRID = sorted(set([0, 1, 2, 4, 8] + _dense))


# ============================ manuscript estimator (verbatim) ================
# copied from paper/build_figures.py so the exponent is computed identically.
def per_target_gap_rates(rec):
    """Per-interval per-token rates for one target. Returns (d, gap, ord, shuf)."""
    cs = rec['context_lengths']; op = rec['ordered_ppl']; sp = rec['shuffled_ppl']
    out = []
    for i in range(1, len(cs)):
        if cs[i - 1] == 0:
            continue
        width = cs[i] - cs[i - 1]
        d_mid = math.sqrt(cs[i - 1] * cs[i])
        ord_marg = (op[i - 1] - op[i]) / width
        shuf_marg = (sp[i - 1] - sp[i]) / width
        out.append((d_mid, ord_marg - shuf_marg, ord_marg, shuf_marg))
    return out


def fit_pl(d, y):
    """Power-law fit on positive bins with d >= FIT_DMIN. Returns (slope, r2, npos)."""
    d = np.asarray(d); y = np.asarray(y)
    m = (y > 0) & (d >= FIT_DMIN)
    if m.sum() < 4:
        return None, None, int(m.sum())
    s, _, r, _, _ = stats.linregress(np.log(d[m]), np.log(y[m]))
    return s, r ** 2, int(m.sum())


def mean_gap_curve(recs):
    """Aggregate per-interval gap across a set of records -> (d, mean_gap)."""
    by_i = {}
    for r in recs:
        for i, tup in enumerate(per_target_gap_rates(r)):
            by_i.setdefault(i, []).append(tup)
    ds, gaps = [], []
    for i in sorted(by_i):
        rows = by_i[i]
        ds.append(rows[0][0])
        gaps.append(sum(t[1] for t in rows) / len(rows))
    return np.array(ds), np.array(gaps)


def bootstrap_alpha(recs, n_boot=N_BOOT, seed=SEED):
    """Document bootstrap on the mean gap curve -> alpha point, r2, 95% CI."""
    d0, g0 = mean_gap_curve(recs)
    a0, r0, npos0 = fit_pl(d0, g0)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(recs))
    alphas, r2s = [], []
    for _ in range(n_boot):
        samp = [recs[j] for j in rng.choice(idx, len(idx), replace=True)]
        d, g = mean_gap_curve(samp)
        a, r, _ = fit_pl(d, g)
        if a is not None:
            alphas.append(a); r2s.append(r)
    alphas = np.array(alphas)
    ci = (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))) if len(alphas) else (None, None)
    return dict(alpha=a0, r2=r0, npos=npos0,
                alpha_ci=ci, median_boot_r2=float(np.median(r2s)) if r2s else None,
                n_boot_ok=len(alphas), d=d0.tolist(), gap=g0.tolist())


# ============================ data ==========================================
def load_swissprot():
    if os.path.exists(SP_PATH):
        raw = open(SP_PATH, "rb").read()
    else:
        raw = urllib.request.urlopen(SP_URL).read(); open(SP_PATH, "wb").write(raw)
    text = gzip.decompress(raw).decode()
    seqs, cid, cur = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if cid and cur:
                s = "".join(cur)
                if set(s) <= AA: seqs.append((cid, s))
            cid = line[1:].split()[0]; cur = []
        else:
            cur.append(line.strip())
    if cid and cur and set("".join(cur)) <= AA:
        seqs.append((cid, "".join(cur)))
    return seqs


# ============================ corrected loader (verbatim) ===================
def correct_load(hf_id, dtype=torch.float16):
    """Restore the three persistent=False attention tensors device_map leaves broken."""
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=dtype, device_map={"": 0}).eval()
    cfg = model.config
    scale = float(np.sqrt(cfg.embed_dim // cfg.n_head))
    for mod in model.modules():
        if type(mod).__name__ == "ProGenAttention":
            N = mod.bias.shape[-1]
            mod.bias = torch.tril(torch.ones(N, N, dtype=torch.bool, device="cuda")).view(1, 1, N, N)
            mod.masked_bias = torch.tensor(-1e9, device="cuda")
            mod.scale_attn = torch.tensor(scale, dtype=dtype, device="cuda")
    for m in model.modules():
        if type(m).__name__ == "ProGenModel" and not hasattr(type(m), "get_head_mask"):
            type(m).get_head_mask = (
                lambda self, hm, n=None, *a, **k: [None] * (n if n is not None else self.config.n_layer))
    return tok, model


# ============================ main ==========================================
def main():
    allseq = load_swissprot()
    print(f"{len(allseq)} standard-AA Swiss-Prot proteins")
    print(f"dense grid ({len(BASE_GRID)} ctx lengths): {BASE_GRID}")
    summary = {}
    for scale, hf in SCALES:
        print("\n" + "=" * 72 + f"\nPROBE {scale}\n" + "=" * 72)
        t0 = time.time()
        tok, model = correct_load(hf)
        BOS = tok.bos_token_id
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        MAXPOS = getattr(model.config, "n_positions", 1024)
        bos_n = 1 if BOS is not None else 0
        GRID = [c for c in BASE_GRID if c + TARGET_LEN + bos_n <= MAXPOS]
        if GRID[-1] < MAXPOS - TARGET_LEN - bos_n:
            GRID.append(MAXPOS - TARGET_LEN - bos_n)
        MAX_CTX = max(GRID); MIN_LEN = MAX_CTX + TARGET_LEN + 50

        def ppl_vec(idsf, ts):
            L = len(idsf)
            if L - ts < 2: return float("inf")
            with torch.no_grad():
                lg = model(torch.tensor([idsf], device="cuda")).logits[0]
                lp = torch.log_softmax(lg[ts:L - 1].float(), -1)
                nxt = torch.tensor(idsf[ts + 1:L], device="cuda")
                nll = -lp[torch.arange(lp.shape[0], device="cuda"), nxt].mean().item()
            return float(np.exp(nll))

        # ---- gate #1: held-out per-residue perplexity on natural protein
        held = [s for _, s in random.Random(SEED).sample(allseq, N_HELDOUT * 3)
                if 50 <= len(s) <= MAXPOS - 1][:N_HELDOUT]
        nlls = []
        for s in held:
            ids = ([BOS] if BOS is not None else []) + enc(s); L = len(ids)
            with torch.no_grad():
                lg = model(torch.tensor([ids], device="cuda")).logits[0]
                lp = torch.log_softmax(lg[:L - 1].float(), -1)
                nxt = torch.tensor(ids[1:L], device="cuda")
                nlls.append((-lp[torch.arange(L - 1, device="cuda"), nxt]).mean().item())
        ppl1 = float(np.exp(np.mean(nlls)))
        print(f"[gate1] held-out per-residue ppl (n={len(held)}) = {ppl1:.3f}")

        # ---- CPF over dense grid
        def curves(ids):
            ts0 = int(MAX_CTX + TARGET_FRAC * ((len(ids) - TARGET_LEN) - MAX_CTX)); te0 = ts0 + TARGET_LEN
            tgt = ids[ts0:te0]; o, sh = [], []
            for c in GRID:
                pfx = [] if c == 0 else ids[ts0 - c:ts0]
                full = ([BOS] if BOS is not None else []) + list(pfx) + list(tgt)
                ts = bos_n + len(pfx)
                o.append(ppl_vec(full, ts))
                if c == 0:
                    sh.append(o[-1])
                else:
                    shuf = list(pfx); random.Random(SEED + c).shuffle(shuf)
                    fs = ([BOS] if BOS is not None else []) + shuf + list(tgt)
                    sh.append(ppl_vec(fs, ts))
            return {"context_lengths": list(GRID), "ordered_ppl": o, "shuffled_ppl": sh}

        cpf = [(i, s) for i, s in allseq if len(s) >= MIN_LEN]
        random.Random(SEED).shuffle(cpf); cpf = cpf[:N_DOCS]
        results = []
        for k, (pid, s) in enumerate(cpf):
            ids = enc(s)
            if len(ids) < MIN_LEN: continue
            r = curves(ids)
            r.update(corpus_id=f"protein_{scale}", document_id=pid, target_frac=TARGET_FRAC)
            results.append(r)
            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{len(cpf)} docs  ({(time.time()-t0)/60:.1f} min)")

        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(OUT / f"protein_{scale}_dense.json", "w"))

        # ---- gate #2 level floor (median shuffled - ordered, raw ppl)
        O = np.array([e["ordered_ppl"] for e in results])
        S = np.array([e["shuffled_ppl"] for e in results])
        level = np.median(S - O, 0)
        print(f"[gate2] level Delta(d) median: {[round(x,3) for x in level]}")

        # ---- exponent with manuscript estimator + document bootstrap
        boot = bootstrap_alpha(results)
        summary[scale] = dict(per_residue_ppl=ppl1, n_docs=len(results),
                              grid=GRID, level_delta=level.tolist(), **boot)
        ci = boot["alpha_ci"]
        print(f"[alpha] slope alpha = {boot['alpha']:.3f}  (magnitude {abs(boot['alpha']):.3f})"
              f"   r2 = {boot['r2']:.3f}   pos-bins = {boot['npos']}")
        print(f"        95% CI (doc-bootstrap) = [{ci[0]:.3f}, {ci[1]:.3f}]"
              f"   median boot r2 = {boot['median_boot_r2']:.3f}")
        print(f"        NaN check = {int(np.isnan(np.r_[O,S]).sum())}   ({(time.time()-t0)/60:.1f} min)")
        del model; torch.cuda.empty_cache()

    json.dump(summary, open(OUT / "protein_dense_summary.json", "w"), indent=2)
    print("\n" + "=" * 72)
    print("SUMMARY (compare to language alpha = -1.04, median r2 = 0.96)")
    for scale, d in summary.items():
        ci = d["alpha_ci"]
        print(f"  {scale:16s} ppl={d['per_residue_ppl']:.1f}  alpha={d['alpha']:+.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}]  r2={d['r2']:.2f}  n={d['n_docs']}")
    print("wrote protein_{small,large}_dense.json + protein_dense_summary.json")


if __name__ == "__main__":
    main()
