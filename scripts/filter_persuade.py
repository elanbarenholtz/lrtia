#!/usr/bin/env python3
"""
filter_persuade.py

Step 2: Create the analysis cohort by filtering persuade_clean.jsonl.

Filter criteria:
- word_count >= 250
- token_count >= 350
- score in {1..6} (non-missing)
- grade known (for grade analysis)
- prompt_id known

Usage:
    python scripts/filter_persuade.py
    python scripts/filter_persuade.py --min-words 200 --min-tokens 300
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_records(
    records: list[dict],
    min_words: int = 250,
    min_tokens: int = 350,
) -> tuple[list[dict], dict]:
    """Apply filtering criteria.

    Returns: (filtered_records, drop_stats)
    """
    kept = []
    drop_stats = Counter()

    for r in records:
        # Check word count
        if r['word_count'] < min_words:
            drop_stats['below_min_words'] += 1
            continue

        # Check token count
        if r['token_count'] < min_tokens:
            drop_stats['below_min_tokens'] += 1
            continue

        # Check score is present and valid
        if r['score'] is None or r['score'] not in {1, 2, 3, 4, 5, 6}:
            drop_stats['missing_score'] += 1
            continue

        # Check grade is present (for grade analysis)
        if r['grade'] is None:
            drop_stats['missing_grade'] += 1
            continue

        # Check prompt_id is present
        if r['prompt_id'] is None:
            drop_stats['missing_prompt'] += 1
            continue

        kept.append(r)

    return kept, dict(drop_stats)


def print_summary(records: list[dict], drop_stats: dict) -> None:
    """Print summary of filtered cohort."""

    print("\n" + "=" * 60)
    print("FILTERED COHORT SUMMARY")
    print("=" * 60)

    print(f"\nTotal essays retained: {len(records):,}")

    if drop_stats:
        print("\nDropped:")
        for reason, count in sorted(drop_stats.items()):
            print(f"  {reason}: {count:,}")

    # Score distribution
    print("\nBy score:")
    scores = Counter(r['score'] for r in records)
    for score in sorted(scores.keys()):
        print(f"  {score}: {scores[score]:,}")

    # Score bins
    print("\nBy score bin:")
    bins = {'low (1-2)': 0, 'mid (3-4)': 0, 'high (5-6)': 0}
    for r in records:
        if r['score'] <= 2:
            bins['low (1-2)'] += 1
        elif r['score'] <= 4:
            bins['mid (3-4)'] += 1
        else:
            bins['high (5-6)'] += 1
    for bin_name, count in bins.items():
        print(f"  {bin_name}: {count:,}")

    # Grade distribution
    print("\nBy grade:")
    grades = Counter(r['grade'] for r in records)
    for grade in sorted(grades.keys()):
        print(f"  {grade}: {grades[grade]:,}")

    # ELL distribution (note: UNK kept in filtered cohort, excluded in ELL analysis)
    print("\nBy ELL status:")
    ell_counts = Counter(r['ell'] for r in records)
    for ell in ['ELL', 'non_ELL', 'UNK']:
        if ell in ell_counts:
            print(f"  {ell}: {ell_counts[ell]:,}")

    # Top prompts
    print("\nTop prompts:")
    prompts = Counter(r['prompt_id'] for r in records)
    for prompt, count in prompts.most_common(6):
        print(f"  {prompt}: {count:,}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Filter PERSUADE cohort')
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/persuade_clean/persuade_clean.jsonl'),
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/persuade_clean/persuade_250w.jsonl'),
        help='Output JSONL file'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=250,
        help='Minimum word count (default: 250)'
    )
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=350,
        help='Minimum token count (default: 350)'
    )

    args = parser.parse_args()

    # Load
    print(f"Loading {args.input}...")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records):,} records")

    # Filter
    print(f"\nFiltering (min_words={args.min_words}, min_tokens={args.min_tokens})...")
    filtered, drop_stats = filter_records(
        records,
        min_words=args.min_words,
        min_tokens=args.min_tokens,
    )

    # Summary
    print_summary(filtered, drop_stats)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(filtered):,} records to {args.output}")


if __name__ == '__main__':
    main()
