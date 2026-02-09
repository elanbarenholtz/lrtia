#!/usr/bin/env python3
"""
sanity_check_persuade.py

Pre-flight sanity checks before running GPU memory curve analysis.

Checks:
A. Schema validation
B. Text cleanliness (artifacts, weird chars)
C. Language check (non-ASCII fraction)
D. Length distribution by group (critical for confound detection)

Usage:
    python scripts/sanity_check_persuade.py
    python scripts/sanity_check_persuade.py --cohort score
"""

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# LOADING
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ============================================================
# A. SCHEMA CHECK
# ============================================================

def check_schema(records: list[dict], cohort_type: str) -> dict:
    """Validate schema for cohort type."""

    required_fields = ['essay_id', 'text', 'word_count', 'token_count']

    # Add cohort-specific required field
    if cohort_type == 'score':
        required_fields.extend(['score', 'score_bin'])
    elif cohort_type == 'grade':
        required_fields.append('grade')
    elif cohort_type == 'ell':
        required_fields.append('ell')
    elif cohort_type == 'prompt':
        required_fields.append('prompt_id')

    issues = []
    field_missing_counts = Counter()

    for i, r in enumerate(records):
        for field in required_fields:
            if field not in r or r[field] is None:
                field_missing_counts[field] += 1

    # Check for unexpected values
    if cohort_type == 'score':
        score_bins = set(r.get('score_bin') for r in records)
        expected = {'low', 'mid', 'high'}
        if score_bins - expected:
            issues.append(f"Unexpected score_bin values: {score_bins - expected}")

    if cohort_type == 'ell':
        ell_values = set(r.get('ell') for r in records)
        expected = {'ELL', 'non_ELL'}
        unexpected = ell_values - expected - {None}
        if unexpected:
            issues.append(f"Unexpected ell values: {unexpected}")

    return {
        'required_fields': required_fields,
        'missing_counts': dict(field_missing_counts),
        'issues': issues,
        'pass': len(field_missing_counts) == 0 and len(issues) == 0,
    }


# ============================================================
# B. TEXT CLEANLINESS CHECK
# ============================================================

def check_text_cleanliness(records: list[dict]) -> dict:
    """Scan for artifacts that could mess with perplexity."""

    patterns = {
        'brackets': r'\[[^\]]*\]',
        'angle_brackets': r'<[^>]*>',
        'ampersand_codes': r'&[=+\-]\w+',
        'chat_tags': r'[@%]\w+:',
        'excessive_punctuation': r'[.!?]{4,}',
        'control_chars': r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        'repeated_spaces': r' {3,}',
    }

    flagged = defaultdict(list)

    for r in records:
        text = r.get('text', '')
        essay_id = r.get('essay_id', 'unknown')

        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                flagged[pattern_name].append({
                    'essay_id': essay_id,
                    'matches': matches[:5],  # First 5 matches
                    'count': len(matches),
                })

    # Digit analysis
    digit_stats = []
    for r in records:
        text = r.get('text', '')
        words = text.split()
        digit_words = [w for w in words if re.search(r'\d', w)]
        digit_stats.append(len(digit_words) / len(words) if words else 0)

    return {
        'flagged_patterns': {k: len(v) for k, v in flagged.items()},
        'flagged_examples': {k: v[:3] for k, v in flagged.items()},  # First 3 examples
        'digit_pct': {
            'mean': np.mean(digit_stats) * 100,
            'max': np.max(digit_stats) * 100,
            'p90': np.percentile(digit_stats, 90) * 100,
        },
        'pass': all(len(v) == 0 for v in flagged.values()),
    }


# ============================================================
# C. LANGUAGE CHECK
# ============================================================

def check_language(records: list[dict], sample_size: int = 25) -> dict:
    """Check for non-English or corrupted text."""

    non_ascii_fracs = []
    flagged = []

    for r in records:
        text = r.get('text', '')
        essay_id = r.get('essay_id', 'unknown')

        if not text:
            continue

        # Calculate non-ASCII fraction
        non_ascii = sum(1 for c in text if ord(c) > 127)
        frac = non_ascii / len(text)
        non_ascii_fracs.append(frac)

        # Flag if high non-ASCII
        if frac > 0.05:  # >5% non-ASCII
            flagged.append({
                'essay_id': essay_id,
                'non_ascii_pct': frac * 100,
                'sample': text[:200],
            })

    # Random sample for manual inspection
    random.seed(42)
    sample_indices = random.sample(range(len(records)), min(sample_size, len(records)))
    samples = [
        {
            'essay_id': records[i].get('essay_id'),
            'first_100_chars': records[i].get('text', '')[:100],
        }
        for i in sample_indices
    ]

    return {
        'non_ascii_pct': {
            'mean': np.mean(non_ascii_fracs) * 100,
            'max': np.max(non_ascii_fracs) * 100,
            'p99': np.percentile(non_ascii_fracs, 99) * 100,
        },
        'high_non_ascii_count': len(flagged),
        'high_non_ascii_examples': flagged[:5],
        'random_samples': samples[:10],
        'pass': len(flagged) == 0,
    }


# ============================================================
# D. LENGTH DISTRIBUTION CHECK
# ============================================================

def check_length_distribution(records: list[dict], group_field: str) -> dict:
    """Check length distribution by group - critical for confound detection."""

    by_group = defaultdict(list)

    for r in records:
        group = r.get(group_field)
        if group is None:
            continue
        by_group[group].append({
            'word_count': r.get('word_count', 0),
            'token_count': r.get('token_count', 0),
        })

    stats = {}
    for group, items in by_group.items():
        word_counts = [x['word_count'] for x in items]
        token_counts = [x['token_count'] for x in items]

        stats[group] = {
            'n': len(items),
            'word_count': {
                'min': min(word_counts),
                'p10': int(np.percentile(word_counts, 10)),
                'p25': int(np.percentile(word_counts, 25)),
                'median': int(np.median(word_counts)),
                'p75': int(np.percentile(word_counts, 75)),
                'p90': int(np.percentile(word_counts, 90)),
                'max': max(word_counts),
                'mean': np.mean(word_counts),
            },
            'token_count': {
                'min': min(token_counts),
                'p10': int(np.percentile(token_counts, 10)),
                'p25': int(np.percentile(token_counts, 25)),
                'median': int(np.median(token_counts)),
                'p75': int(np.percentile(token_counts, 75)),
                'p90': int(np.percentile(token_counts, 90)),
                'max': max(token_counts),
                'mean': np.mean(token_counts),
            },
        }

    # Check for imbalance
    medians = {g: s['token_count']['median'] for g, s in stats.items()}
    if medians:
        max_median = max(medians.values())
        min_median = min(medians.values())
        imbalance_ratio = max_median / min_median if min_median > 0 else float('inf')
    else:
        imbalance_ratio = 1.0

    return {
        'group_stats': stats,
        'imbalance_ratio': imbalance_ratio,
        'imbalance_warning': imbalance_ratio > 1.3,
        'pass': imbalance_ratio <= 1.3,
    }


# ============================================================
# MAIN
# ============================================================

def run_checks(cohort_path: Path, cohort_type: str) -> dict:
    """Run all sanity checks on a cohort."""

    print(f"\n{'='*60}")
    print(f"SANITY CHECKS: {cohort_path.name}")
    print(f"{'='*60}")

    records = load_jsonl(cohort_path)
    print(f"Loaded {len(records)} records")

    results = {}

    # A. Schema check
    print("\n[A] Schema Check...")
    schema_result = check_schema(records, cohort_type)
    results['schema'] = schema_result
    if schema_result['pass']:
        print("    PASS")
    else:
        print(f"    FAIL: {schema_result['issues']}")
        print(f"    Missing: {schema_result['missing_counts']}")

    # B. Text cleanliness
    print("\n[B] Text Cleanliness Check...")
    clean_result = check_text_cleanliness(records)
    results['cleanliness'] = clean_result
    if clean_result['pass']:
        print("    PASS (no artifacts found)")
    else:
        print(f"    WARNING: Found artifacts")
        for pattern, count in clean_result['flagged_patterns'].items():
            if count > 0:
                print(f"      {pattern}: {count} essays")
    print(f"    Digit content: mean={clean_result['digit_pct']['mean']:.1f}%, max={clean_result['digit_pct']['max']:.1f}%")

    # C. Language check
    print("\n[C] Language Check...")
    lang_result = check_language(records)
    results['language'] = lang_result
    if lang_result['pass']:
        print("    PASS (all essays appear to be English)")
    else:
        print(f"    WARNING: {lang_result['high_non_ascii_count']} essays with high non-ASCII content")
    print(f"    Non-ASCII: mean={lang_result['non_ascii_pct']['mean']:.2f}%, p99={lang_result['non_ascii_pct']['p99']:.2f}%")

    # D. Length distribution
    print("\n[D] Length Distribution Check...")

    # Determine group field
    if cohort_type == 'score':
        group_field = 'score_bin'
    elif cohort_type == 'grade':
        group_field = 'grade'
    elif cohort_type == 'ell':
        group_field = 'ell'
    elif cohort_type == 'prompt':
        group_field = 'prompt_id'
    else:
        group_field = 'score_bin'

    length_result = check_length_distribution(records, group_field)
    results['length'] = length_result

    print(f"\n    Length stats by {group_field}:")
    print(f"    {'Group':<20} {'N':>6} {'Median Words':>14} {'Median Tokens':>14}")
    print(f"    {'-'*20} {'-'*6} {'-'*14} {'-'*14}")
    for group, stats in length_result['group_stats'].items():
        print(f"    {str(group):<20} {stats['n']:>6} {stats['word_count']['median']:>14} {stats['token_count']['median']:>14}")

    print(f"\n    Imbalance ratio: {length_result['imbalance_ratio']:.2f}x")
    if length_result['imbalance_warning']:
        print("    *** WARNING: Groups have imbalanced lengths! Consider length-matching. ***")
    else:
        print("    PASS (groups have similar length distributions)")

    # Overall
    all_pass = all([
        schema_result['pass'],
        clean_result['pass'],
        lang_result['pass'],
        length_result['pass'],
    ])

    results['overall_pass'] = all_pass

    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS' if all_pass else 'NEEDS ATTENTION'}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Sanity checks for PERSUADE cohorts')
    parser.add_argument(
        '--cohort',
        type=str,
        default='all',
        choices=['all', 'score', 'grade', 'ell', 'prompt'],
        help='Which cohort to check (default: all)'
    )
    parser.add_argument(
        '--cohort-dir',
        type=Path,
        default=Path('data/persuade_clean/cohorts'),
        help='Directory containing cohort files'
    )

    args = parser.parse_args()

    cohorts = {
        'score': 'persuade_score_cohort.jsonl',
        'grade': 'persuade_grade_cohort.jsonl',
        'ell': 'persuade_ell_cohort.jsonl',
        'prompt': 'persuade_prompt_cohort.jsonl',
    }

    if args.cohort == 'all':
        to_check = list(cohorts.items())
    else:
        to_check = [(args.cohort, cohorts[args.cohort])]

    all_results = {}
    for cohort_type, filename in to_check:
        cohort_path = args.cohort_dir / filename
        if cohort_path.exists():
            all_results[cohort_type] = run_checks(cohort_path, cohort_type)
        else:
            print(f"\nSkipping {cohort_type}: {cohort_path} not found")

    # Final summary
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for cohort_type, result in all_results.items():
        status = "PASS" if result['overall_pass'] else "NEEDS ATTENTION"
        length_warn = " (LENGTH IMBALANCE)" if result['length']['imbalance_warning'] else ""
        print(f"  {cohort_type}: {status}{length_warn}")


if __name__ == '__main__':
    main()
