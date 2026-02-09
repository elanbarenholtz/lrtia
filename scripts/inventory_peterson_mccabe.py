#!/usr/bin/env python3
"""
Inventory the Peterson & McCabe corpus for developmental analysis.

Step 1: Inventory ages and file structure
Step 2: Extract narrative samples with metadata
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/PetersonMcCabe")


def parse_age(age_str: str) -> int:
    """Convert age string like '4;00.' to months."""
    # Strip trailing punctuation
    age_str = age_str.rstrip('.')
    match = re.match(r'(\d+);(\d+)', age_str)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return None


def parse_cha_file(filepath: Path) -> dict:
    """Parse a single .cha file and extract metadata + narratives."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    result = {
        'filename': filepath.name,
        'age_months': None,
        'age_str': None,
        'sex': None,
        'types': [],
        'narratives': []
    }

    # Extract CHI @ID line
    chi_id_match = re.search(r'@ID:\s*eng\|PetersonMcCabe\|CHI\|([^|]+)\|([^|]*)\|', content)
    if chi_id_match:
        age_str = chi_id_match.group(1)
        sex = chi_id_match.group(2) if chi_id_match.group(2) else None
        result['age_str'] = age_str.rstrip('.')
        result['age_months'] = parse_age(age_str)
        result['sex'] = sex

    # Extract @Types
    types_matches = re.findall(r'@Types:\s*(.+)', content)
    all_types = set()
    for types_str in types_matches:
        for t in types_str.split(','):
            all_types.add(t.strip())
    result['types'] = list(all_types)

    # Split by @G: Narrative markers
    lines = content.split('\n')
    current_narrative = None
    current_narrative_idx = 0
    current_types = []
    current_chi_utterances = []

    for line in lines:
        # Check for narrative marker
        narrative_match = re.match(r'@G:\s*Narrative\s*(\d+)', line)
        if narrative_match:
            # Save previous narrative if exists
            if current_narrative is not None and current_chi_utterances:
                result['narratives'].append({
                    'narrative_idx': current_narrative_idx,
                    'types': current_types,
                    'chi_utterances': current_chi_utterances,
                    'chi_text': ' '.join(current_chi_utterances)
                })
            # Start new narrative
            current_narrative_idx = int(narrative_match.group(1))
            current_narrative = current_narrative_idx
            current_chi_utterances = []
            current_types = []
            continue

        # Check for @Types within narrative
        types_match = re.match(r'@Types:\s*(.+)', line)
        if types_match and current_narrative is not None:
            for t in types_match.group(1).split(','):
                current_types.append(t.strip())
            continue

        # Check for CHI utterance
        chi_match = re.match(r'\*CHI:\s*(.+)', line)
        if chi_match and current_narrative is not None:
            # Clean the utterance - remove CHAT annotations
            utterance = chi_match.group(1)
            # Remove [/], [//], [*...], xxx, etc. but keep the words
            utterance = re.sub(r'\[/+\]', '', utterance)
            utterance = re.sub(r'\[\*[^\]]*\]', '', utterance)
            utterance = re.sub(r'\[=![^\]]*\]', '', utterance)
            utterance = re.sub(r'\[=[^\]]*\]', '', utterance)
            utterance = re.sub(r'<[^>]+>', '', utterance)  # Remove <...>
            utterance = re.sub(r'\bxxx\b', '', utterance)
            utterance = re.sub(r'\byyy\b', '', utterance)
            utterance = re.sub(r'\s+', ' ', utterance).strip()
            if utterance:
                current_chi_utterances.append(utterance)

    # Save last narrative
    if current_narrative is not None and current_chi_utterances:
        result['narratives'].append({
            'narrative_idx': current_narrative_idx,
            'types': current_types,
            'chi_utterances': current_chi_utterances,
            'chi_text': ' '.join(current_chi_utterances)
        })

    return result


def main():
    print("=" * 70)
    print("PETERSON & MCCABE CORPUS INVENTORY")
    print("=" * 70)

    # Parse all .cha files
    cha_files = sorted(DATA_DIR.glob("*.cha"))
    print(f"\nFound {len(cha_files)} .cha files\n")

    all_data = []
    for filepath in cha_files:
        data = parse_cha_file(filepath)
        all_data.append(data)

    # =========================================================================
    # STEP 1: Age Inventory
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: AGE INVENTORY")
    print("=" * 70)

    # Counts by age in months
    age_months_counts = defaultdict(int)
    for d in all_data:
        if d['age_months']:
            age_months_counts[d['age_months']] += 1

    print("\nCounts by age (months):")
    print("-" * 40)
    for age in sorted(age_months_counts.keys()):
        years = age // 12
        months = age % 12
        print(f"  {years};{months:02d} ({age} mo): {age_months_counts[age]} files")

    # Counts by age year bins
    age_year_counts = defaultdict(int)
    for d in all_data:
        if d['age_months']:
            year_bin = d['age_months'] // 12
            age_year_counts[year_bin] += 1

    print("\nCounts by age-year bin:")
    print("-" * 40)
    for year in sorted(age_year_counts.keys()):
        print(f"  {year}-year-olds: {age_year_counts[year]} files")

    # Sex distribution
    sex_counts = defaultdict(int)
    for d in all_data:
        sex_counts[d['sex']] += 1

    print("\nSex distribution:")
    print("-" * 40)
    for sex, count in sex_counts.items():
        print(f"  {sex}: {count}")

    # Types distribution
    types_counts = defaultdict(int)
    for d in all_data:
        for t in d['types']:
            types_counts[t] += 1

    print("\n@Types distribution (file level):")
    print("-" * 40)
    for t, count in sorted(types_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # =========================================================================
    # STEP 2: Narrative Sample Inventory
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: NARRATIVE SAMPLE INVENTORY")
    print("=" * 70)

    # Count narratives per file
    narr_per_file = [len(d['narratives']) for d in all_data]
    print(f"\nNarratives per file: min={min(narr_per_file)}, max={max(narr_per_file)}, "
          f"mean={sum(narr_per_file)/len(narr_per_file):.1f}")
    print(f"Total narratives: {sum(narr_per_file)}")

    # Build flat sample list
    samples = []
    for d in all_data:
        for narr in d['narratives']:
            word_count = len(narr['chi_text'].split())
            samples.append({
                'filename': d['filename'],
                'narrative_idx': narr['narrative_idx'],
                'age_months': d['age_months'],
                'age_years': d['age_months'] // 12 if d['age_months'] else None,
                'sex': d['sex'],
                'types': ', '.join(narr['types']),
                'word_count': word_count,
                'chi_text': narr['chi_text']
            })

    df = pd.DataFrame(samples)

    print(f"\nTotal narrative samples: {len(df)}")

    # Word count statistics
    print("\nWord count per narrative:")
    print("-" * 40)
    print(df['word_count'].describe())

    # Filter to reasonable-length narratives (e.g., >= 20 words)
    MIN_WORDS = 20
    df_filtered = df[df['word_count'] >= MIN_WORDS]
    print(f"\nNarratives with >= {MIN_WORDS} words: {len(df_filtered)}")

    # Age distribution of filtered samples
    print("\nFiltered samples by age-year bin:")
    print("-" * 40)
    age_sample_counts = df_filtered.groupby('age_years').size()
    for age, count in age_sample_counts.items():
        print(f"  {age}-year-olds: {count} samples")

    # =========================================================================
    # STEP 3: Proposed Cohort Structure
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PROPOSED COHORT STRUCTURE")
    print("=" * 70)

    # Create age bins
    def age_bin(months):
        if months is None:
            return None
        year = months // 12
        if year <= 4:
            return "4yo"
        elif year <= 5:
            return "5yo"
        elif year <= 6:
            return "6yo"
        elif year <= 7:
            return "7yo"
        else:
            return "8-9yo"

    df_filtered['age_bin'] = df_filtered['age_months'].apply(age_bin)

    print("\nSamples by age bin (>= 20 words):")
    print("-" * 40)
    bin_counts = df_filtered.groupby('age_bin').size()
    for bin_name in ["4yo", "5yo", "6yo", "7yo", "8-9yo"]:
        if bin_name in bin_counts.index:
            print(f"  {bin_name}: {bin_counts[bin_name]} samples")

    # Word count by age bin
    print("\nMean word count by age bin:")
    print("-" * 40)
    word_by_age = df_filtered.groupby('age_bin')['word_count'].mean()
    for bin_name in ["4yo", "5yo", "6yo", "7yo", "8-9yo"]:
        if bin_name in word_by_age.index:
            print(f"  {bin_name}: {word_by_age[bin_name]:.1f} words")

    # =========================================================================
    # DECISION
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    n_bins_with_data = sum(1 for b in ["4yo", "5yo", "6yo", "7yo", "8-9yo"]
                          if b in bin_counts.index and bin_counts[b] >= 10)

    if n_bins_with_data >= 3:
        print("\n✓ STRONG CANDIDATE for developmental analysis!")
        print(f"  - {n_bins_with_data} age bins with >= 10 samples each")
        print("  - Good age range for replicating Frog Story effect")
    else:
        print("\n✗ Limited for developmental analysis")
        print("  - May need to combine with other corpora")

    # Save sample dataframe for later use
    output_path = DATA_DIR / "inventory_samples.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered samples to: {output_path}")

    # Also show some example texts
    print("\n" + "=" * 70)
    print("EXAMPLE NARRATIVES")
    print("=" * 70)

    for age_bin in ["4yo", "6yo", "8-9yo"]:
        subset = df_filtered[df_filtered['age_bin'] == age_bin]
        if len(subset) > 0:
            example = subset.iloc[0]
            print(f"\n{age_bin} example ({example['word_count']} words):")
            print("-" * 40)
            text = example['chi_text'][:300] + "..." if len(example['chi_text']) > 300 else example['chi_text']
            print(text)


if __name__ == "__main__":
    main()
