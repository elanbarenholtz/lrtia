#!/usr/bin/env python3
"""Exp 2 runner: increment-shuffle persistence, Q(d).

Measures the span-level order contribution at distance d: does destroying the
internal order of the band at distance ~d, while everything nearer than d is
left intact, degrade prediction of the target?

Mirrors the CPF long-range pipeline (see
``NHB_submission/code/probes/Corpus_Expansion_LongRange_Llama.ipynb``): same
target manifest, same log-spaced ladder, same ``ppl_nll`` scoring and code path
for the intact (condition A) curve. Adds condition B (increment-shuffle) and two
preregistered null controls, alongside the main CPF code without modifying it.

The scoring core (:func:`compute_exp2_curves`) is model-agnostic — it takes a
``score_fn(ctx_ids, tgt_ids) -> (ppl, nll)`` — so it is unit-testable with a
stub model. ``main()`` wires up the HF probe and the document resolver.

Outputs: per-target JSON list to ``results/exp2_increment_shuffle/<probe>/<corpus>.json``,
mirroring ``results/corpus_expansion_longrange/`` so figure/stat code is reusable.

Compute (priority subset, ~300 docs): per target, ordered curve = 12 passes;
condition B = sum over 10 pairs of K passes = 10*K. With K=20 and 2 controls
enabled it is ~12 + 10*20*3 ~= 612 passes/target, <=1054 tokens each. Disable
controls (--no-controls) for the ~231 passes/target headline budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from lrtia.intervention.increment_shuffle import (
    stable_seed,
    increment_shuffle_prefix,
    near_and_far_shuffle_prefix,
    random_token_increment_prefix,
    ladder_pairs,
)

# --- Protocol (identical to Corpus_Expansion_LongRange_Llama) -------------------
CTX_LENGTHS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MAX_CTX = max(CTX_LENGTHS)
TARGET_LEN = 30
TARGET_FRACS = [0.5]
MIN_DOC_TOK = MAX_CTX + TARGET_LEN + 50
DEFAULT_K = 20

RUN_CORPORA_PRIORITY = [
    "ted_transcripts_en",
    "gutenberg_fiction_en",
    "ted_transcripts_ru",
]
RUN_CORPORA_FULL = [
    "gutenberg_fiction_en",
    "ted_transcripts_en",
    "ted_transcripts_de",
    "ted_transcripts_fr",
    "ted_transcripts_tr",
    "ted_transcripts_ru",
    "literary_ja",
    "literary_fi",
    "news_en",
    "buckeye",
]


# --- Model-agnostic scoring core ------------------------------------------------
def compute_exp2_curves(
    full_ids,
    target_start,
    target_end,
    score_fn,
    *,
    document_id,
    target_id,
    k_shuffles=DEFAULT_K,
    vocab_size=None,
    with_controls=True,
    context_lengths=CTX_LENGTHS,
):
    """Compute the intact (A) curve and per-pair increment-shuffle (B) stats.

    Args:
        full_ids: full document token ids.
        target_start, target_end: target region [start, end).
        score_fn: callable (ctx_ids: list[int], tgt_ids: list[int]) -> (ppl, nll).
        document_id, target_id: identifiers, used for reproducible per-(target, c, k)
            seeds via stable_seed(document_id, target_id, c_cur, k).
        k_shuffles: K independent permutations per (target, pair).
        vocab_size: needed only when with_controls (random-token control).
        with_controls: also run the two preregistered null controls.
        context_lengths: the ladder (includes 0).

    Returns:
        dict record with 'ordered_ppl'/'ordered_nll' across the ladder and a
        'pairs' list carrying per-band A and B (and control) perplexities.
    """
    tgt = list(full_ids[target_start:target_end])

    # Condition A: intact ordered curve across the full ladder (shared code path).
    ordered_ppl, ordered_nll = [], []
    for c in context_lengths:
        pfx = [] if c == 0 else list(full_ids[target_start - c: target_start])
        ppl, nll = score_fn(pfx, tgt)
        ordered_ppl.append(ppl)
        ordered_nll.append(nll)

    ppl_at = dict(zip(context_lengths, ordered_ppl))
    nll_at = dict(zip(context_lengths, ordered_nll))

    pairs_out = []
    for lp in ladder_pairs(context_lengths):
        c_prev, c_cur = lp.c_prev, lp.c_cur
        base_pfx = list(full_ids[target_start - c_cur: target_start])

        b_ppls, b_nlls, seeds = [], [], []
        null_ppls, rand_ppls = [], []
        for k in range(k_shuffles):
            seed = stable_seed(document_id, target_id, c_cur, k)
            seeds.append(seed)

            sh = increment_shuffle_prefix(base_pfx, c_prev, c_cur, seed)
            ppl_b, nll_b = score_fn(sh, tgt)
            if math.isfinite(ppl_b):
                b_ppls.append(ppl_b)
                b_nlls.append(nll_b)

            if with_controls:
                # Null 1: near band ALSO shuffled -> should collapse to marginal.
                nf = near_and_far_shuffle_prefix(base_pfx, c_prev, c_cur, seed)
                ppl_n, _ = score_fn(nf, tgt)
                if math.isfinite(ppl_n):
                    null_ppls.append(ppl_n)
                # Null 2: far band replaced with random tokens (content, not order).
                if vocab_size is not None:
                    rt = random_token_increment_prefix(
                        base_pfx, c_prev, c_cur, seed, vocab_size
                    )
                    ppl_r, _ = score_fn(rt, tgt)
                    if math.isfinite(ppl_r):
                        rand_ppls.append(ppl_r)

        rec = {
            "i": lp.index,
            "c_prev": c_prev,
            "c_cur": c_cur,
            "width": lp.width,
            "distance": lp.distance,
            "A_ppl": ppl_at[c_cur],
            "A_nll": nll_at[c_cur],
            "B_ppl_mean": _mean(b_ppls),
            "B_nll_mean": _mean(b_nlls),
            "B_ppl": b_ppls,
            "B_nll": b_nlls,
            "seeds": seeds,
        }
        if with_controls:
            rec["null_full_ppl_mean"] = _mean(null_ppls)
            rec["rand_tok_ppl_mean"] = _mean(rand_ppls) if rand_ppls else None
        pairs_out.append(rec)

    return {
        "document_id": document_id,
        "target_id": target_id,
        "context_lengths": list(context_lengths),
        "ordered_ppl": ordered_ppl,
        "ordered_nll": ordered_nll,
        "pairs": pairs_out,
        "K": k_shuffles,
    }


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


# --- HF probe scoring (mirrors ppl_nll in the notebook) -------------------------
def make_hf_score_fn(model, torch):
    @torch.no_grad()
    def score_fn(ctx_toks, tgt_toks):
        if len(tgt_toks) < 2:
            return float("inf"), float("inf")
        full = list(ctx_toks) + list(tgt_toks)
        ts = len(ctx_toks)
        ids = torch.tensor([full], device=model.device)
        out = model(ids)
        logits = out.logits[0]
        nll = 0.0
        cnt = 0
        for i in range(ts, len(full) - 1):
            lp = torch.log_softmax(logits[i], dim=-1)
            nll += -lp[full[i + 1]].item()
            cnt += 1
        del out, logits
        torch.cuda.empty_cache()
        if cnt == 0:
            return float("inf"), float("inf")
        mn = nll / cnt
        return math.exp(mn), mn

    return score_fn


# --- Document resolver (mirrors the notebook loaders) ---------------------------
def build_resolver(drive: Path):
    targets_path = drive / "Results/corpus_expansion/targets_llama.jsonl"
    tok_manifest_path = drive / "Results/corpus_expansion/tokenized_manifest_llama.jsonl"

    tok_manifest = {}
    if tok_manifest_path.exists():
        for line in open(tok_manifest_path):
            d = json.loads(line)
            tok_manifest[d["document_id"]] = d["file_path"]

    ce_corpora: dict[str, set] = {}
    if targets_path.exists():
        for line in open(targets_path):
            t = json.loads(line)
            ce_corpora.setdefault(t["corpus_id"], set()).add(t["document_id"])

    ce_data = drive / "Data/corpus_expansion"
    ml_data = drive / "Data/multilingual_literary"
    buckeye_jsonl = drive / "Data/buckeye_processed/speaker_concatenated.jsonl"

    def fix_ce_path(p):
        return str(p).replace("data/corpus_expansion/clean/", str(ce_data) + "/")

    def ce_docs(corpus_id):
        for doc_id in ce_corpora.get(corpus_id, set()):
            fp = tok_manifest.get(doc_id)
            if fp is None:
                continue
            abs_fp = Path(fix_ce_path(fp))
            if not abs_fp.exists():
                abs_fp = ce_data / Path(fp).name
            try:
                yield doc_id, abs_fp.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue

    def literary_docs(lang):
        m = json.loads((ml_data / "manifests" / f"{lang}.json").read_text(encoding="utf-8"))
        for author in m["authors"]:
            for t in author["texts"]:
                fp = ml_data / t["text_path"]
                try:
                    yield t["text_id"], fp.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue

    def buckeye_docs():
        if not buckeye_jsonl.exists():
            return
        for line in open(buckeye_jsonl):
            d = json.loads(line)
            txt = d.get("text", "").strip()
            if txt:
                yield d["doc_id"], txt

    def get_doc_texts(corpus_id):
        if corpus_id == "buckeye":
            yield from buckeye_docs()
        elif corpus_id.startswith("literary_"):
            yield from literary_docs(corpus_id.split("_", 1)[1])
        else:
            yield from ce_docs(corpus_id)

    return get_doc_texts


# --- Main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default=os.environ.get("LRTIA_DRIVE", "."),
                    help="Root holding Results/ and Data/ (Colab: /content/drive/MyDrive/LRTIA)")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <drive>/Results/exp2_increment_shuffle/<probe>)")
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--probe", default=None, help="Probe label for output path (default: derived)")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="Corpora to run (default: priority subset)")
    ap.add_argument("--full", action="store_true", help="Run the full ten-corpus set")
    ap.add_argument("--no-controls", action="store_true", help="Skip null controls (headline budget)")
    ap.add_argument("--limit", type=int, default=None, help="Cap docs per corpus (smoke test)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    drive = Path(args.drive)
    probe = args.probe or args.model.rstrip("/").split("/")[-1].lower()
    out_dir = Path(args.out) if args.out else drive / f"Results/exp2_increment_shuffle/{probe}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = args.corpora or (RUN_CORPORA_FULL if args.full else RUN_CORPORA_PRIORITY)

    print(f"Probe: {args.model}  ->  {out_dir}")
    print(f"K={args.k}  controls={not args.no_controls}  corpora={corpora}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    score_fn = make_hf_score_fn(model, torch)
    get_doc_texts = build_resolver(drive)

    for corpus_id in corpora:
        cache = out_dir / f"{corpus_id}.json"
        if cache.exists():
            print(f"{corpus_id}: cached"); continue
        docs = list(get_doc_texts(corpus_id))
        if args.limit:
            docs = docs[: args.limit]
        if not docs:
            print(f"{corpus_id}: no docs"); continue

        print(f"\n{'='*60}\n{corpus_id} ({len(docs)} candidate docs)\n{'='*60}")
        t0 = time.time()
        results, skipped = [], 0
        for doc_id, text in docs:
            full_ids = tokenizer.encode(text, add_special_tokens=False)
            n_tok = len(full_ids)
            if n_tok < MIN_DOC_TOK:
                skipped += 1; continue
            rem_start, rem_end = MAX_CTX, n_tok - TARGET_LEN
            for frac in TARGET_FRACS:
                ts = int(rem_start + frac * (rem_end - rem_start))
                te = ts + TARGET_LEN
                if ts - MAX_CTX < 0 or te > n_tok:
                    continue
                target_id = f"{doc_id}__pos{int(frac*100):02d}"
                r = compute_exp2_curves(
                    full_ids, ts, te, score_fn,
                    document_id=doc_id, target_id=target_id,
                    k_shuffles=args.k, vocab_size=vocab_size,
                    with_controls=not args.no_controls,
                )
                r["corpus_id"] = corpus_id
                r["target_frac"] = frac
                r["probe"] = args.model
                results.append(r)
        json.dump(results, open(cache, "w"))
        print(f"  {len(results)} targets in {(time.time()-t0)/60:.1f} min ({skipped} short-skipped)")

    print("\nDone.")


if __name__ == "__main__":
    main()
