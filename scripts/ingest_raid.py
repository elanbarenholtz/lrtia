#!/usr/bin/env python3
"""
ingest_raid.py

Sample from the RAID dataset for multi-genre, multi-model memory curve analysis.

Since most RAID texts are short (150-330 words), we:
1. Filter to texts with 200+ words (~270+ tokens) per domain
2. Sample up to N human + N per AI model per domain
3. Use a shorter burn-in / window config in the analysis notebook

Usage:
    python scripts/ingest_raid.py
    python scripts/ingest_raid.py --n-per-group 30 --min-words 250
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


SEED = 99
OUTPUT_DIR = Path("data/raid_sampled")

# Focus on models that represent different architectures
TARGET_MODELS = [
    "human",
    "chatgpt",    # GPT-3.5
    "gpt4",       # GPT-4
    "llama-chat",  # Meta
    "mistral-chat",  # Mistral
    "cohere-chat",   # Cohere
]

# All domains
TARGET_DOMAINS = [
    "abstracts", "books", "news", "poetry",
    "recipes", "reddit", "reviews", "wiki",
]


def main():
    parser = argparse.ArgumentParser(description="Sample RAID for LRTIA analysis")
    parser.add_argument("--n-per-group", type=int, default=30,
                        help="Max docs per model per domain (default: 30)")
    parser.add_argument("--min-words", type=int, default=200,
                        help="Minimum word count (default: 200)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load RAID
    print("Loading RAID dataset (this downloads ~800MB on first run)...")
    from raid.utils import load_data
    df = load_data(split="train", include_adversarial=False)
    print(f"Loaded {len(df)} records")

    # Filter to target models
    df = df[df["model"].isin(TARGET_MODELS)]
    print(f"After model filter: {len(df)} records")

    # Add word count
    df["word_count"] = df["generation"].str.split().str.len()

    # Filter by min words
    df = df[df["word_count"] >= args.min_words]
    print(f"After min words ({args.min_words}): {len(df)} records")

    # Sample
    corpus = []
    print(f"\nSampling up to {args.n_per_group} per model per domain:")
    print(f"{'Domain':<15} ", end="")
    for m in TARGET_MODELS:
        print(f"{m:<14}", end="")
    print()
    print("-" * (15 + 14 * len(TARGET_MODELS)))

    for domain in TARGET_DOMAINS:
        print(f"{domain:<15} ", end="")
        for model in TARGET_MODELS:
            pool = df[(df["domain"] == domain) & (df["model"] == model)]
            n_available = len(pool)
            n_sample = min(args.n_per_group, n_available)

            if n_sample > 0:
                sampled = pool.sample(n=n_sample, random_state=args.seed)
                for _, row in sampled.iterrows():
                    population = "human" if model == "human" else "ai"
                    corpus.append({
                        "doc_id": f"{domain}_{model}_{row['id'][:8]}",
                        "domain": domain,
                        "model": model,
                        "population": population,
                        "text": row["generation"],
                        "word_count": row["word_count"],
                    })

            print(f"{n_sample:>4}/{n_available:<8}", end="")
        print()

    # Summary
    print(f"\nTotal corpus: {len(corpus)} documents")

    by_pop = defaultdict(int)
    by_domain = defaultdict(int)
    for doc in corpus:
        by_pop[doc["population"]] += 1
        by_domain[doc["domain"]] += 1

    print(f"  Human: {by_pop['human']}, AI: {by_pop['ai']}")
    print(f"\n  Per domain:")
    for d in TARGET_DOMAINS:
        print(f"    {d:<15} {by_domain[d]}")

    # Word count stats
    wcs = [d["word_count"] for d in corpus]
    print(f"\n  Word counts: median={int(np.median(wcs))}, "
          f"range=[{min(wcs)}, {max(wcs)}]")

    # Save
    out_path = OUTPUT_DIR / "raid_corpus.jsonl"
    with open(out_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
