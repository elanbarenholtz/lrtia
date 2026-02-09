#!/usr/bin/env python3
"""
sample_persuade_cohorts_v2.py

Step 3 (v2): Create LENGTH-MATCHED stratified samples for the four experiments.

Key improvement: Length-matching across groups to avoid confounding.

Experiments:
1. Score comparison: low (1-2) vs mid (3-4) vs high (5-6), n=150 each, LENGTH-MATCHED
2. Grade trajectory: grades 6, 9, 12, n=150 each
3. ELL comparison: ELL vs non_ELL, n=150 each, LENGTH-MATCHED
4. Prompt effects: top K prompts, n=150 each

Usage:
    python scripts/sample_persuade_cohorts_v2.py
    python scripts/sample_persuade_cohorts_v2.py --n-per-group 100
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


SEED = 42  # Changed from 13 for v2


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


def length_matched_sample(
    groups: dict[str, list[dict]],
    n_per_group: int,
    rng: random.Random,
    length_field: str = 'token_count',
    n_bins: int = 10,
) -> dict[str, list[dict]]:
    """
    Sample n_per_group from each group with matched length distributions.

    Strategy:
    1. Find the group with the most constrained length distribution (usually smallest group)
    2. Bin that group's lengths into n_bins quantile bins
    3. Sample proportionally from each bin in other groups

    This ensures all groups have similar length distributions.
    """
    # Find reference group (smallest, or most constrained)
    ref_group_name = min(groups.keys(), key=lambda g: len(groups[g]))
    ref_group = groups[ref_group_name]

    # Get length distribution of reference group
    ref_lengths = [r[length_field] for r in ref_group]

    # Create bins based on reference group quantiles
    bin_edges = [np.percentile(ref_lengths, p) for p in np.linspace(0, 100, n_bins + 1)]
    bin_edges[0] = 0  # Ensure we capture everything
    bin_edges[-1] = float('inf')

    def get_bin(length):
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= length < bin_edges[i + 1]:
                return i
        return len(bin_edges) - 2

    # Bin the reference group
    ref_binned = defaultdict(list)
    for r in ref_group:
        bin_idx = get_bin(r[length_field])
        ref_binned[bin_idx].append(r)

    # Calculate target counts per bin (proportional to reference)
    total_ref = len(ref_group)
    target_per_bin = {
        bin_idx: max(1, int(n_per_group * len(items) / total_ref))
        for bin_idx, items in ref_binned.items()
    }

    # Adjust to hit exact n_per_group
    total_target = sum(target_per_bin.values())
    if total_target < n_per_group:
        # Add to largest bins
        sorted_bins = sorted(target_per_bin.keys(), key=lambda b: -len(ref_binned[b]))
        for bin_idx in sorted_bins:
            if total_target >= n_per_group:
                break
            target_per_bin[bin_idx] += 1
            total_target += 1
    elif total_target > n_per_group:
        # Remove from smallest bins
        sorted_bins = sorted(target_per_bin.keys(), key=lambda b: len(ref_binned[b]))
        for bin_idx in sorted_bins:
            if total_target <= n_per_group:
                break
            if target_per_bin[bin_idx] > 1:
                target_per_bin[bin_idx] -= 1
                total_target -= 1

    # Sample from each group using the same bin structure
    sampled = {}

    for group_name, group_records in groups.items():
        # Bin this group
        binned = defaultdict(list)
        for r in group_records:
            bin_idx = get_bin(r[length_field])
            binned[bin_idx].append(r)

        # Sample from each bin
        group_sample = []
        for bin_idx, target_n in target_per_bin.items():
            pool = binned[bin_idx]
            if len(pool) >= target_n:
                group_sample.extend(rng.sample(pool, target_n))
            else:
                # Take all available and note the shortfall
                group_sample.extend(pool)

        sampled[group_name] = group_sample

    return sampled


def sample_by_score_matched(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for score comparison experiment with LENGTH MATCHING.

    Uses band-restricted sampling: find a token count range where all groups
    have enough essays, then sample from within that band.
    """

    groups = {
        'low': [r for r in records if r['score'] in {1, 2}],
        'mid': [r for r in records if r['score'] in {3, 4}],
        'high': [r for r in records if r['score'] in {5, 6}],
    }

    print(f"\n  Score experiment pools (before matching):")
    for name, pool in groups.items():
        tokens = [r['token_count'] for r in pool]
        print(f"    {name}: n={len(pool):,}, p10={int(np.percentile(tokens,10))}, median={int(np.median(tokens))}, p90={int(np.percentile(tokens,90))}")

    # Find length band where all groups have >= n_per_group essays
    # Score groups have very different length distributions (high essays are longer)
    # We need to find the overlapping range
    best_band = None
    best_min_count = 0

    for min_tok in range(400, 650, 50):
        for max_tok in range(700, 1100, 100):
            counts = {name: sum(1 for r in pool if min_tok <= r['token_count'] <= max_tok)
                      for name, pool in groups.items()}
            min_count = min(counts.values())
            if min_count >= n_per_group and min_count > best_min_count:
                best_band = (min_tok, max_tok)
                best_min_count = min_count

    if best_band is None:
        # Fallback: use widest range
        best_band = (400, 1000)
        print(f"  WARNING: Could not find band with {n_per_group} per group, using {best_band}")

    min_tok, max_tok = best_band
    print(f"\n  Using length band: [{min_tok}, {max_tok}] tokens")

    # Filter to band and sample
    sampled = []
    for name, pool in groups.items():
        band_pool = [r for r in pool if min_tok <= r['token_count'] <= max_tok]
        print(f"    {name} in band: n={len(band_pool)}")

        if len(band_pool) >= n_per_group:
            selected = rng.sample(band_pool, n_per_group)
        else:
            selected = band_pool
            print(f"      WARNING: only {len(band_pool)} available")

        for r in selected:
            r['score_bin'] = name
        sampled.extend(selected)

    return sampled


def sample_by_grade(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for grade trajectory experiment."""

    target_grades = [6, 9, 12]
    groups = {g: [r for r in records if r['grade'] == g] for g in target_grades}

    print(f"\n  Grade experiment pools:")
    for grade, pool in groups.items():
        tokens = [r['token_count'] for r in pool]
        print(f"    Grade {grade}: n={len(pool):,}, median_tokens={int(np.median(tokens))}")

    # Check if length matching is needed (ratio > 1.3)
    medians = {g: np.median([r['token_count'] for r in pool]) for g, pool in groups.items()}
    ratio = max(medians.values()) / min(medians.values())

    if ratio > 1.3:
        print(f"  Length imbalance detected ({ratio:.2f}x), applying length matching...")
        matched = length_matched_sample(groups, n_per_group, rng)

        print(f"\n  After length matching:")
        for grade, pool in matched.items():
            tokens = [r['token_count'] for r in pool]
            print(f"    Grade {grade}: n={len(pool):,}, median_tokens={int(np.median(tokens))}")

        sampled = []
        for pool in matched.values():
            sampled.extend(pool)
    else:
        # Simple random sample
        sampled = []
        for grade, pool in groups.items():
            # Cap any single prompt to 40%
            by_prompt = defaultdict(list)
            for r in pool:
                by_prompt[r['prompt_id']].append(r)

            max_per_prompt = int(n_per_group * 0.4)
            capped_pool = []
            for prompt, essays in by_prompt.items():
                capped_pool.extend(essays[:max_per_prompt * 2])

            if len(capped_pool) >= n_per_group:
                sampled.extend(rng.sample(capped_pool, n_per_group))
            else:
                sampled.extend(rng.sample(pool, min(n_per_group, len(pool))))

    return sampled


def sample_by_ell_matched(records: list[dict], n_per_group: int, rng: random.Random) -> list[dict]:
    """Sample for ELL comparison experiment with LENGTH MATCHING."""

    groups = {
        'ELL': [r for r in records if r['ell'] == 'ELL'],
        'non_ELL': [r for r in records if r['ell'] == 'non_ELL'],
    }

    print(f"\n  ELL experiment pools (before matching):")
    for name, pool in groups.items():
        tokens = [r['token_count'] for r in pool]
        print(f"    {name}: n={len(pool):,}, median_tokens={int(np.median(tokens))}")

    # Check imbalance
    medians = {g: np.median([r['token_count'] for r in pool]) for g, pool in groups.items()}
    ratio = max(medians.values()) / min(medians.values())

    if ratio > 1.15:
        print(f"  Length imbalance detected ({ratio:.2f}x), applying length matching...")
        matched = length_matched_sample(groups, n_per_group, rng)

        print(f"\n  After length matching:")
        for name, pool in matched.items():
            tokens = [r['token_count'] for r in pool]
            print(f"    {name}: n={len(pool):,}, median_tokens={int(np.median(tokens))}")

        sampled = []
        for pool in matched.values():
            sampled.extend(pool)
    else:
        sampled = []
        for name, pool in groups.items():
            sampled.extend(rng.sample(pool, min(n_per_group, len(pool))))

    return sampled


def sample_by_prompt(records: list[dict], n_per_group: int, top_k: int, rng: random.Random) -> list[dict]:
    """Sample for prompt effects experiment."""

    by_prompt = defaultdict(list)
    for r in records:
        by_prompt[r['prompt_id']].append(r)

    sorted_prompts = sorted(by_prompt.items(), key=lambda x: -len(x[1]))

    print(f"\n  Prompt experiment (top {top_k}):")
    sampled = []

    for prompt, pool in sorted_prompts[:top_k]:
        tokens = [r['token_count'] for r in pool]
        print(f"    {prompt}: n={len(pool):,}, median_tokens={int(np.median(tokens))}")

        if len(pool) < n_per_group:
            print(f"      Warning: only {len(pool)} (need {n_per_group})")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n_per_group))

    return sampled


def verify_length_balance(records: list[dict], group_field: str) -> None:
    """Verify length balance after sampling."""

    by_group = defaultdict(list)
    for r in records:
        by_group[r.get(group_field, 'unknown')].append(r['token_count'])

    print(f"\n  Final length verification ({group_field}):")
    medians = {}
    for group, lengths in sorted(by_group.items()):
        median = int(np.median(lengths))
        medians[group] = median
        print(f"    {group}: n={len(lengths)}, median_tokens={median}, range=[{min(lengths)}, {max(lengths)}]")

    if medians:
        ratio = max(medians.values()) / min(medians.values())
        status = "OK" if ratio <= 1.3 else "WARNING"
        print(f"  Imbalance ratio: {ratio:.2f}x [{status}]")


def main():
    parser = argparse.ArgumentParser(description='Sample PERSUADE cohorts (v2 with length matching)')
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

    # Experiment 1: Score (LENGTH MATCHED)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: SCORE COMPARISON (LENGTH-MATCHED)")
    print("=" * 60)
    score_cohort = sample_by_score_matched(records, args.n_per_group, rng)
    verify_length_balance(score_cohort, 'score_bin')
    write_cohort(
        score_cohort,
        args.output_dir / 'persuade_score_cohort.jsonl',
        args.output_dir / 'persuade_score_essay_ids.txt',
    )

    # Experiment 2: Grade
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: GRADE TRAJECTORY")
    print("=" * 60)
    grade_cohort = sample_by_grade(records, args.n_per_group, rng)
    verify_length_balance(grade_cohort, 'grade')
    write_cohort(
        grade_cohort,
        args.output_dir / 'persuade_grade_cohort.jsonl',
        args.output_dir / 'persuade_grade_essay_ids.txt',
    )

    # Experiment 3: ELL (LENGTH MATCHED)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: ELL COMPARISON (LENGTH-MATCHED)")
    print("=" * 60)
    ell_cohort = sample_by_ell_matched(records, args.n_per_group, rng)
    verify_length_balance(ell_cohort, 'ell')
    write_cohort(
        ell_cohort,
        args.output_dir / 'persuade_ell_cohort.jsonl',
        args.output_dir / 'persuade_ell_essay_ids.txt',
    )

    # Experiment 4: Prompt
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: PROMPT EFFECTS")
    print("=" * 60)
    prompt_cohort = sample_by_prompt(records, args.n_per_group, args.top_prompts, rng)
    verify_length_balance(prompt_cohort, 'prompt_id')
    write_cohort(
        prompt_cohort,
        args.output_dir / 'persuade_prompt_cohort.jsonl',
        args.output_dir / 'persuade_prompt_essay_ids.txt',
    )

    # Summary
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE (v2 with length matching)")
    print("=" * 60)
    print(f"\nCohorts written to {args.output_dir}/")
    print(f"  Score cohort:  {len(score_cohort):,} essays (length-matched)")
    print(f"  Grade cohort:  {len(grade_cohort):,} essays")
    print(f"  ELL cohort:    {len(ell_cohort):,} essays (length-matched)")
    print(f"  Prompt cohort: {len(prompt_cohort):,} essays")
    print(f"\nRandom seed: {args.seed}")


if __name__ == '__main__':
    main()
