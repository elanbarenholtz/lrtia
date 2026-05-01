#!/usr/bin/env python3
"""
sample_persuade_finegrain.py

Stratified sample for PERSUADE_Finegrain_v1 analysis: 100 essays per score bin,
scores 2-6 (drops score 1 which is sparse and short). Output consumed by the
fine-grained coherence-decay pipeline.

Usage:
    python scripts/sample_persuade_finegrain.py
    python scripts/sample_persuade_finegrain.py --n-per-score 50 --seed 7
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/persuade_clean/persuade_clean.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/persuade_clean/cohorts/persuade_finegrain_sample.jsonl"
SCORES = [2, 3, 4, 5, 6]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-per-score", type=int, default=100)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    print(f"Loaded {len(records):,} essays from {args.input}")

    by_score = defaultdict(list)
    for r in records:
        by_score[r["score"]].append(r)

    sampled = []
    for s in SCORES:
        pool = by_score[s]
        n = min(args.n_per_score, len(pool))
        picked = rng.sample(pool, n)
        sampled.extend(picked)
        wc = [r["word_count"] for r in picked]
        tc = [r["token_count"] for r in picked]
        print(
            f"  score={s}: pool={len(pool):>5}  picked={n:>3}  "
            f"word_count median={sorted(wc)[len(wc)//2]}  "
            f"token_count median={sorted(tc)[len(tc)//2]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled):,} essays to {args.output}")

    ell_counts = defaultdict(int)
    grade_counts = defaultdict(int)
    for r in sampled:
        ell_counts[r.get("ell", "UNK")] += 1
        grade_counts[r.get("grade", "UNK")] += 1
    print(f"  ell: {dict(ell_counts)}")
    print(f"  grade: {dict(sorted(grade_counts.items(), key=lambda x: (x[0] is None, x[0])))}")


if __name__ == "__main__":
    main()
