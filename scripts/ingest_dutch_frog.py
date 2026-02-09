#!/usr/bin/env python3
"""
ingest_dutch_frog.py

Ingest Dutch-AarssenBos CHAT files into JSONL format for LRTIA analysis.
Extracts only the frog story portion (marked @G: frog story).

Usage:
    python ingest_dutch_frog.py --input-dir data/Dutch-AarssenBos --output data/dutch_frog_transcripts.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_age_from_id(id_line: str) -> Optional[int]:
    """Parse age in months from @ID line.

    Format: nld|Dutch-AarssenBos|CHI|6;02.07||||Target_Child|||
    Age part: 6;02.07 means 6 years, 2 months, 7 days
    """
    match = re.search(r'\|(\d+);(\d+)\.?\d*\|', id_line)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return None


def clean_utterance(text: str) -> str:
    """Clean CHAT utterance codes from text."""
    # Remove speaker code (*CHI:, *PET:)
    text = re.sub(r'^\*\w+:\s*', '', text)

    # Remove CHAT codes
    text = re.sub(r'\[\^[^\]]*\]', '', text)
    text = re.sub(r'\[\*[^\]]*\]', '', text)
    text = re.sub(r'\[/\]', '', text)
    text = re.sub(r'\[//\]', '', text)
    text = re.sub(r'\[///\]', '', text)
    text = re.sub(r'\[\?\]', '', text)
    text = re.sub(r'\[!\]', '', text)
    text = re.sub(r'\[%[^\]]*\]', '', text)
    text = re.sub(r'<[^>]*>\s*\[[^\]]*\]', '', text)

    # Remove quoted speech markers
    text = re.sub(r'\+"/\.', '', text)
    text = re.sub(r'\+"', '', text)
    text = re.sub(r'\+/\.', '', text)

    # Remove unintelligible markers
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'\byyy\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)  # Remove bare numbers

    # Remove onomatopoeia markers (@o, @i)
    text = re.sub(r'@[oig]', '', text)

    # Remove non-standard forms (@g)
    text = re.sub(r'\w+@g', lambda m: m.group(0).replace('@g', ''), text)

    # Remove &= actions and &~ fragments
    text = re.sub(r'&=[^\s]+', '', text)
    text = re.sub(r'&~[^\s]+', '', text)
    text = re.sub(r'&[^\s]+', '', text)

    # Remove lengthening markers
    text = re.sub(r':+', '', text)

    # Remove pauses
    text = re.sub(r'\(\.\.\.\)', '', text)
    text = re.sub(r'\(\.\)', '', text)

    # Remove 0word (omitted words)
    text = re.sub(r'\b0\w*', '', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove trailing punctuation markers
    text = re.sub(r'\s*[+/]+\s*$', '', text)

    return text


def parse_cha_file(filepath: Path) -> Optional[Dict]:
    """Parse a single .cha file and extract frog story portion."""

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Parse metadata
    age_months = None
    sex = None
    doc_id = filepath.stem

    for line in lines:
        line = line.strip()
        if line.startswith('@ID:') and 'CHI' in line:
            age_months = parse_age_from_id(line)
            if '|male|' in line.lower():
                sex = 'male'
            elif '|female|' in line.lower():
                sex = 'female'

    # Find frog story section
    in_frog_story = False
    frog_start = None
    frog_end = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if '@G:' in line_stripped and 'frog' in line_stripped.lower():
            in_frog_story = True
            frog_start = i + 1
        elif in_frog_story and line_stripped.startswith('@G:'):
            # New story section - frog story ended
            frog_end = i
            break
        elif in_frog_story and line_stripped == '@End':
            frog_end = i
            break

    if frog_start is None:
        return None  # No frog story in this file

    if frog_end is None:
        frog_end = len(lines)

    # Extract utterances from frog story section
    child_utterances = []
    adult_utterances = []
    current_speaker = None
    current_text = ""

    for line in lines[frog_start:frog_end]:
        line = line.rstrip()

        # New speaker turn
        if line.startswith('*'):
            # Save previous turn
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned and len(cleaned) > 1:
                    if current_speaker == 'CHI':
                        child_utterances.append(cleaned)
                    else:
                        adult_utterances.append(cleaned)

            # Parse new speaker
            match = re.match(r'\*(\w+):\s*(.*)', line)
            if match:
                current_speaker = match.group(1)
                current_text = match.group(2)
            else:
                current_speaker = None
                current_text = ""

        # Continuation line (starts with tab, not annotation tier)
        elif current_speaker and line.startswith('\t') and not line.startswith('\t%'):
            current_text += ' ' + line.strip()

        # Skip annotation tiers and metadata
        elif line.startswith('%') or line.startswith('@'):
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned and len(cleaned) > 1:
                    if current_speaker == 'CHI':
                        child_utterances.append(cleaned)
                    else:
                        adult_utterances.append(cleaned)
                current_speaker = None
                current_text = ""

    # Don't forget last turn
    if current_speaker and current_text:
        cleaned = clean_utterance(current_text)
        if cleaned and len(cleaned) > 1:
            if current_speaker == 'CHI':
                child_utterances.append(cleaned)
            else:
                adult_utterances.append(cleaned)

    # Concatenate child text
    child_text = ' '.join(child_utterances)

    if not child_text or len(child_text.split()) < 20:
        return None  # Too short

    # Compute scaffolding metrics
    child_tokens = len(child_text.split())
    adult_text = ' '.join(adult_utterances)
    adult_tokens = len(adult_text.split()) if adult_text else 0
    total_tokens = child_tokens + adult_tokens

    return {
        'doc_id': doc_id,
        'text': child_text,
        'age_months': age_months,
        'sex': sex,
        'n_utterances_child': len(child_utterances),
        'n_utterances_adult': len(adult_utterances),
        'child_tokens': child_tokens,
        'adult_tokens': adult_tokens,
        'adult_token_frac': adult_tokens / total_tokens if total_tokens > 0 else 0,
        'adult_turn_frac': len(adult_utterances) / (len(child_utterances) + len(adult_utterances)) if (len(child_utterances) + len(adult_utterances)) > 0 else 0,
        'source_file': str(filepath.name),
    }


def ingest_dutch_frog(input_dir: Path, output_path: Path) -> None:
    """Ingest all Dutch-AarssenBos .cha files (frog story only)."""

    records = []

    # Age directories: 04, 05, 06, 07, 08, 09, 10
    for age_dir in sorted(input_dir.iterdir()):
        if not age_dir.is_dir() or age_dir.name.startswith('.') or age_dir.name.startswith('0m'):
            continue

        try:
            age_years = int(age_dir.name)
        except ValueError:
            continue

        cha_files = list(age_dir.glob('*.cha'))
        print(f"\nProcessing age {age_years}: {len(cha_files)} files")

        for filepath in sorted(cha_files):
            try:
                record = parse_cha_file(filepath)

                if record is None:
                    print(f"  Skipping {filepath.name}: no frog story or too short")
                    continue

                # Add group metadata
                record['dataset'] = 'dutch_frog'
                record['age_years'] = age_years
                record['group'] = f'age{age_years}'

                # Use parsed age or fall back to directory-based
                if record['age_months'] is None:
                    record['age_months'] = age_years * 12 + 6  # Default to mid-year

                # Create age bins (similar to ECSC for comparison)
                age_m = record['age_months']
                if age_m < 60:
                    record['age_bin'] = '4-5yr'
                elif age_m < 72:
                    record['age_bin'] = '5-6yr'
                elif age_m < 84:
                    record['age_bin'] = '6-7yr'
                elif age_m < 96:
                    record['age_bin'] = '7-8yr'
                elif age_m < 108:
                    record['age_bin'] = '8-9yr'
                elif age_m < 120:
                    record['age_bin'] = '9-10yr'
                else:
                    record['age_bin'] = '10+yr'

                # Population dict for compatibility
                record['population'] = json.dumps({
                    'age_months': record['age_months'],
                    'age_years': age_years,
                    'sex': record['sex'],
                    'group': record['group'],
                })

                record['word_count'] = len(record['text'].split())
                records.append(record)

            except Exception as e:
                print(f"  Error processing {filepath}: {e}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    # Summary
    print(f"\n{'='*60}")
    print("DUTCH FROG STORY INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")
    print(f"Output: {output_path}")

    # Stats by age
    from collections import defaultdict
    by_age = defaultdict(list)
    for r in records:
        by_age[r['age_years']].append(r)

    print("\nBy age:")
    for age in sorted(by_age.keys()):
        recs = by_age[age]
        words = [r['word_count'] for r in recs]
        adult_fracs = [r['adult_token_frac'] for r in recs]
        print(f"  {age}yr: n={len(recs)}, words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]}), adult_frac={sum(adult_fracs)/len(adult_fracs):.2%}")


def main():
    parser = argparse.ArgumentParser(description='Ingest Dutch-AarssenBos frog story CHAT files')
    parser.add_argument('--input-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/Dutch-AarssenBos'),
                        help='Input directory')
    parser.add_argument('--output', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/dutch_frog_transcripts.jsonl'),
                        help='Output JSONL file')

    args = parser.parse_args()
    ingest_dutch_frog(args.input_dir, args.output)


if __name__ == '__main__':
    main()
