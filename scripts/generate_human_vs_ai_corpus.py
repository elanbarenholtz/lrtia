#!/usr/bin/env python3
"""
generate_human_vs_ai_corpus.py

Generate a matched corpus of human-written vs AI-generated essays for
memory curve comparison.

Human essays: sampled from PERSUADE 2.0 (real student argumentative essays)
AI essays: generated via Claude API on the same prompts, targeting similar lengths

Usage:
    # Step 1: Sample human essays and create prompts file
    python scripts/generate_human_vs_ai_corpus.py sample

    # Step 2: Generate AI essays (requires ANTHROPIC_API_KEY)
    python scripts/generate_human_vs_ai_corpus.py generate

    # Step 3: Combine into final corpus
    python scripts/generate_human_vs_ai_corpus.py combine
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

SEED = 77
INPUT_PATH = Path("data/persuade_clean/persuade_clean.jsonl")
OUTPUT_DIR = Path("data/human_vs_ai")

# Prompts with enough long essays and clear argumentative framing
PROMPTS = {
    "electoral_college": {
        "prompt_id": "Does the electoral college work?",
        "instruction": (
            "Write a persuasive essay arguing whether the Electoral College "
            "should be kept or abolished in United States presidential elections. "
            "Take a clear position and support it with reasoning and evidence."
        ),
    },
    "seeking_opinions": {
        "prompt_id": "Seeking multiple opinions",
        "instruction": (
            "Write a persuasive essay about whether seeking multiple opinions "
            "helps people make better decisions. Take a clear position and "
            "support it with reasoning and examples."
        ),
    },
    "distance_learning": {
        "prompt_id": "Distance learning",
        "instruction": (
            "Write a persuasive essay about the benefits or drawbacks of "
            "students attending classes from home via online or video "
            "conferencing. Take a clear position and support it with "
            "reasoning and examples."
        ),
    },
    "driverless_cars": {
        "prompt_id": "Driverless cars",
        "instruction": (
            "Write a persuasive essay about whether driverless cars are a "
            "positive or negative development for society. Take a clear "
            "position and support it with reasoning and examples."
        ),
    },
    "car_free_cities": {
        "prompt_id": "Car-free cities",
        "instruction": (
            "Write a persuasive essay about the advantages or disadvantages "
            "of limiting car usage in cities. Take a clear position and "
            "support it with reasoning and examples."
        ),
    },
}

N_HUMAN_PER_PROMPT = 30
N_AI_PER_PROMPT = 30
MIN_TOKENS = 400
MAX_TOKENS = 700


# ── Step 1: Sample human essays ────────────────────────────────────────────

def sample_human_essays():
    """Sample human essays from PERSUADE, length-filtered."""
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PERSUADE corpus...")
    records = []
    with open(INPUT_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records):,} total essays")

    human_essays = []
    prompt_stats = {}

    for slug, info in PROMPTS.items():
        pid = info["prompt_id"]
        pool = [
            r for r in records
            if r["prompt_id"] == pid
            and MIN_TOKENS <= r["token_count"] <= MAX_TOKENS
        ]
        rng.shuffle(pool)
        selected = pool[:N_HUMAN_PER_PROMPT]

        lengths = [r["token_count"] for r in selected]
        prompt_stats[slug] = {
            "pool_size": len(pool),
            "sampled": len(selected),
            "median_tokens": int(np.median(lengths)),
            "range": (min(lengths), max(lengths)),
        }

        for r in selected:
            human_essays.append({
                "doc_id": f"human_{slug}_{r['essay_id']}",
                "essay_id": r["essay_id"],
                "prompt_slug": slug,
                "prompt_id": pid,
                "population": "human",
                "text": r["text"],
                "token_count": r["token_count"],
                "score": r["score"],
            })

    # Save human essays
    out_path = OUTPUT_DIR / "human_essays.jsonl"
    with open(out_path, "w") as f:
        for doc in human_essays:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSampled {len(human_essays)} human essays -> {out_path}")
    print(f"\n{'Prompt':<25} {'Pool':>6} {'Sampled':>8} {'Med Tok':>8} {'Range':>12}")
    print("-" * 65)
    for slug, s in prompt_stats.items():
        print(f"{slug:<25} {s['pool_size']:>6} {s['sampled']:>8} {s['median_tokens']:>8} {s['range'][0]}-{s['range'][1]:>4}")

    # Save length targets for AI generation (median per prompt)
    targets = {
        slug: prompt_stats[slug]["median_tokens"]
        for slug in PROMPTS
    }
    targets_path = OUTPUT_DIR / "length_targets.json"
    with open(targets_path, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"\nLength targets saved to {targets_path}")


# ── Step 2: Generate AI essays ─────────────────────────────────────────────

def generate_ai_essays():
    """Generate AI essays via Claude API."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        print("  Then set ANTHROPIC_API_KEY environment variable")
        return

    client = anthropic.Anthropic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load length targets
    targets_path = OUTPUT_DIR / "length_targets.json"
    if not targets_path.exists():
        print("ERROR: Run 'sample' step first to generate length targets")
        return
    with open(targets_path) as f:
        length_targets = json.load(f)

    # Check for existing progress
    out_path = OUTPUT_DIR / "ai_essays.jsonl"
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                existing.add(d["doc_id"])
        print(f"Resuming: {len(existing)} AI essays already generated")

    total_generated = len(existing)
    total_needed = N_AI_PER_PROMPT * len(PROMPTS)

    for slug, info in PROMPTS.items():
        target_words = int(length_targets[slug] * 0.75)  # tokens -> approx words
        word_range = f"{target_words - 50}-{target_words + 50}"

        for i in range(N_AI_PER_PROMPT):
            doc_id = f"ai_{slug}_{i:03d}"
            if doc_id in existing:
                continue

            system_prompt = (
                "You are a student writing a persuasive essay for a school assignment. "
                "Write naturally and authentically. Do not use markdown formatting, "
                "bullet points, or headers. Write in plain paragraph form only."
            )

            user_prompt = (
                f"{info['instruction']}\n\n"
                f"Write approximately {word_range} words. "
                f"Write the essay directly with no preamble or title."
            )

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text.strip()

                doc = {
                    "doc_id": doc_id,
                    "prompt_slug": slug,
                    "prompt_id": info["prompt_id"],
                    "population": "ai",
                    "text": text,
                    "model": "claude-sonnet-4-20250514",
                }

                # Append (allows resume)
                with open(out_path, "a") as f:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                total_generated += 1
                print(f"  [{total_generated}/{total_needed}] {doc_id} ({len(text.split())} words)")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"  ERROR generating {doc_id}: {e}")
                time.sleep(2)

    print(f"\nGenerated {total_generated} AI essays -> {out_path}")


# ── Step 3: Combine into final corpus ──────────────────────────────────────

def combine_corpus():
    """Combine human and AI essays into a single corpus JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    human_path = OUTPUT_DIR / "human_essays.jsonl"
    ai_path = OUTPUT_DIR / "ai_essays.jsonl"

    if not human_path.exists():
        print("ERROR: Run 'sample' step first")
        return
    if not ai_path.exists():
        print("ERROR: Run 'generate' step first")
        return

    corpus = []

    # Load human
    with open(human_path) as f:
        for line in f:
            doc = json.loads(line)
            corpus.append(doc)

    # Load AI (add token counts)
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        can_tokenize = True
    except ImportError:
        print("WARNING: transformers not installed, skipping token counts for AI essays")
        can_tokenize = False

    with open(ai_path) as f:
        for line in f:
            doc = json.loads(line)
            if can_tokenize:
                doc["token_count"] = len(tokenizer.encode(doc["text"]))
            corpus.append(doc)

    # Summary stats
    human_docs = [d for d in corpus if d["population"] == "human"]
    ai_docs = [d for d in corpus if d["population"] == "ai"]

    print(f"Corpus summary:")
    print(f"  Human essays: {len(human_docs)}")
    print(f"  AI essays:    {len(ai_docs)}")

    if can_tokenize:
        h_lengths = [d["token_count"] for d in human_docs]
        a_lengths = [d["token_count"] for d in ai_docs]
        print(f"\n  Human tokens: median={int(np.median(h_lengths))}, range=[{min(h_lengths)}, {max(h_lengths)}]")
        print(f"  AI tokens:    median={int(np.median(a_lengths))}, range=[{min(a_lengths)}, {max(a_lengths)}]")

        # Per-prompt breakdown
        print(f"\n  {'Prompt':<25} {'Human med':>10} {'AI med':>10} {'Ratio':>8}")
        print("  " + "-" * 58)
        for slug in PROMPTS:
            h = [d["token_count"] for d in human_docs if d["prompt_slug"] == slug]
            a = [d["token_count"] for d in ai_docs if d["prompt_slug"] == slug]
            if h and a:
                hm, am = int(np.median(h)), int(np.median(a))
                print(f"  {slug:<25} {hm:>10} {am:>10} {am/hm:>8.2f}")

    # Save
    out_path = OUTPUT_DIR / "human_vs_ai_corpus.jsonl"
    with open(out_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(corpus)} docs -> {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate human vs AI essay corpus for LRTIA analysis"
    )
    parser.add_argument(
        "step",
        choices=["sample", "generate", "combine", "all"],
        help="Which step to run",
    )
    args = parser.parse_args()

    if args.step in ("sample", "all"):
        print("=" * 60)
        print("STEP 1: Sample human essays from PERSUADE")
        print("=" * 60)
        sample_human_essays()
        print()

    if args.step in ("generate", "all"):
        print("=" * 60)
        print("STEP 2: Generate AI essays via Claude API")
        print("=" * 60)
        generate_ai_essays()
        print()

    if args.step in ("combine", "all"):
        print("=" * 60)
        print("STEP 3: Combine into final corpus")
        print("=" * 60)
        combine_corpus()


if __name__ == "__main__":
    main()
