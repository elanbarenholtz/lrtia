#!/usr/bin/env python3
"""
sample_persuade_cohorts.py

Step 3: Create stratified samples for the four analysis experiments.

Experiments:
1. Score comparison: low (1-2) vs mid (3-4) vs high (5-6), n=150 each
2. Grade trajectory: grades 6, 9, 12, n=150 each
3. ELL comparison: ELL vs non_ELL, n=150 each
4. Prompt effects: top K prompts, n=150 each

Usage:
    python scripts/sample_persuade_cohorts.py
    python scripts/sample_persuade_cohorts.py --n-per-group 100
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 13


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_cohort(records: list[dict], output_path: Path, essay_ids_path: Path) -> None:
    """Write cohort JSONL and essay IDs sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(essay_ids_path, 'w') as f:
        for r in records:
            f.write(r['essay_id'] + '\n')

    print(f"  Wrote {len(records):,} records to {output_path}")


def length_match_sample(
    pool: list[dict],
    n: int,
    target_distribution: list[int] = None,
    rng: random.Random = None,
) -> list[dict]:
    """Sample n records, optionally matching token_count distribution.

    If target_distribution is provided, stratify by token_count bins.
    Otherwise, just random sample.
    """
    if rng is None:
        rng = random.Random(SEED)

    if len(pool) <= n:
        return pool.copy()

    if target_distribution is None:
        return rng.sample(pool, n)

    # For length matching, bin by token_count and sample proportionally
    # This is a simplified version - could be more sophisticated
    return rng.sample(pool, n)


def sample_by_score(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for score comparison experiment.

    Groups: low (1-2), mid (3-4), high (5-6)
    """
    groups = {
        'low': [r for r in records if r['score'] in {1, 2}],
        'mid': [r for r in records if r['score'] in {3, 4}],
        'high': [r for r in records if r['score'] in {5, 6}],
    }

    print(f"\n  Score experiment pools:")
    for name, pool in groups.items():
        print(f"    {name}: {len(pool):,} available")

    sampled = []
    for name, pool in groups.items():
        if len(pool) < n_per_group:
            print(f"    Warning: {name} has only {len(pool)} (need {n_per_group})")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n_per_group))

    # Add score_bin field
    for r in sampled:
        if r['score'] <= 2:
            r['score_bin'] = 'low'
        elif r['score'] <= 4:
            r['score_bin'] = 'mid'
        else:
            r['score_bin'] = 'high'

    return sampled


def sample_by_grade(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for grade trajectory experiment.

    Groups: grades 6, 9, 12 (anchor grades for clean developmental signal)
    """
    target_grades = [6, 9, 12]
    groups = {g: [r for r in records if r['grade'] == g] for g in target_grades}

    print(f"\n  Grade experiment pools:")
    for grade, pool in groups.items():
        print(f"    Grade {grade}: {len(pool):,} available")

    sampled = []
    for grade, pool in groups.items():
        if len(pool) < n_per_group:
            print(f"    Warning: grade {grade} has only {len(pool)} (need {n_per_group})")
            sampled.extend(pool)
        else:
            # Cap any single prompt to avoid dominance (30-40%)
            by_prompt = defaultdict(list)
            for r in pool:
                by_prompt[r['prompt_id']].append(r)

            # If any prompt is >40% of pool, cap it
            max_per_prompt = int(n_per_group * 0.4)
            capped_pool = []
            for prompt, essays in by_prompt.items():
                capped_pool.extend(essays[:max_per_prompt * 3])  # Allow some oversampling pre-selection

            if len(capped_pool) >= n_per_group:
                sampled.extend(rng.sample(capped_pool, n_per_group))
            else:
                sampled.extend(rng.sample(pool, min(n_per_group, len(pool))))

    return sampled


def sample_by_ell(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for ELL comparison experiment.

    Groups: ELL vs non_ELL (exclude UNK)
    """
    groups = {
        'ELL': [r for r in records if r['ell'] == 'ELL'],
        'non_ELL': [r for r in records if r['ell'] == 'non_ELL'],
    }

    print(f"\n  ELL experiment pools:")
    for name, pool in groups.items():
        print(f"    {name}: {len(pool):,} available")

    # For ELL comparison, strongly recommend length-matching
    # Get token_count distribution from the smaller group
    smaller_group = min(groups.values(), key=len)
    target_tokens = [r['token_count'] for r in smaller_group]

    sampled = []
    for name, pool in groups.items():
        if len(pool) < n_per_group:
            print(f"    Warning: {name} has only {len(pool)} (need {n_per_group})")
            sampled.extend(pool)
        else:
            # Simple random sample for now
            # TODO: implement proper length matching
            sampled.extend(rng.sample(pool, n_per_group))

    return sampled


def sample_by_prompt(records: list[dict], n_per_group: int, top_k: int, rng: random.Random) -> list[dict]:
    """Sample for prompt effects experiment.

    Select top K prompts by count, n_per_group from each.
    """
    # Count essays per prompt
    by_prompt = defaultdict(list)
    for r in records:
        by_prompt[r['prompt_id']].append(r)

    # Sort by count
    sorted_prompts = sorted(by_prompt.items(), key=lambda x: -len(x[1]))

    print(f"\n  Prompt experiment (top {top_k}):")
    sampled = []
    selected_prompts = []

    for prompt, pool in sorted_prompts[:top_k]:
        print(f"    {prompt}: {len(pool):,} available")
        selected_prompts.append(prompt)

        if len(pool) < n_per_group:
            print(f"      Warning: only {len(pool)} (need {n_per_group})")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n_per_group))

    return sampled


def main():
    parser = argparse.ArgumentParser(description='Sample PERSUADE cohorts')
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/persuade_clean/persuade_250w.jsonl'),
        help='Input filtered JSONL file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/persuade_clean/cohorts'),
        help='Output directory for cohort files'
    )
    parser.add_argument(
        '--n-per-group',
        type=int,
        default=150,
        help='Number of essays per group (default: 150)'
    )
    parser.add_argument(
        '--top-prompts',
        type=int,
        default=6,
        help='Number of prompts for prompt experiment (default: 6)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help=f'Random seed (default: {SEED})'
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Load
    print(f"Loading {args.input}...")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records):,} records")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Score
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: SCORE COMPARISON")
    print("=" * 50)
    score_cohort = sample_by_score(records, args.n_per_group, rng)
    write_cohort(
        score_cohort,
        args.output_dir / 'persuade_score_cohort.jsonl',
        args.output_dir / 'persuade_score_essay_ids.txt',
    )

    # Experiment 2: Grade
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: GRADE TRAJECTORY")
    print("=" * 50)
    grade_cohort = sample_by_grade(records, args.n_per_group, rng)
    write_cohort(
        grade_cohort,
        args.output_dir / 'persuade_grade_cohort.jsonl',
        args.output_dir / 'persuade_grade_essay_ids.txt',
    )

    # Experiment 3: ELL
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: ELL COMPARISON")
    print("=" * 50)
    ell_cohort = sample_by_ell(records, args.n_per_group, rng)
    write_cohort(
        ell_cohort,
        args.output_dir / 'persuade_ell_cohort.jsonl',
        args.output_dir / 'persuade_ell_essay_ids.txt',
    )

    # Experiment 4: Prompt
    print("\n" + "=" * 50)
    print("EXPERIMENT 4: PROMPT EFFECTS")
    print("=" * 50)
    prompt_cohort = sample_by_prompt(records, args.n_per_group, args.top_prompts, rng)
    write_cohort(
        prompt_cohort,
        args.output_dir / 'persuade_prompt_cohort.jsonl',
        args.output_dir / 'persuade_prompt_essay_ids.txt',
    )

    # Summary
    print("\n" + "=" * 50)
    print("SAMPLING COMPLETE")
    print("=" * 50)
    print(f"\nCohorts written to {args.output_dir}/")
    print(f"  Score cohort:  {len(score_cohort):,} essays")
    print(f"  Grade cohort:  {len(grade_cohort):,} essays")
    print(f"  ELL cohort:    {len(ell_cohort):,} essays")
    print(f"  Prompt cohort: {len(prompt_cohort):,} essays")
    print(f"\nRandom seed: {args.seed}")


if __name__ == '__main__':
    main()
