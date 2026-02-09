#!/usr/bin/env python3
"""
ingest_miami_mono.py

Ingest Miami Mono CHAT files into JSONL format for LRTIA analysis.

Usage:
    python ingest_miami_mono.py --input-dir /path/to/English-MiamiMono --output miami_mono_transcripts.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_age_from_id(id_line: str) -> Optional[int]:
    """Parse age in months from @ID line.

    Format: eng|English-MiamiMono|CHI|8;00.|male|||Target_Child|||
    Age part: 8;00. means 8 years, 0 months
    """
    match = re.search(r'\|(\d+);(\d+)\.?\|', id_line)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return None


def parse_grade_from_comment(lines: List[str]) -> Optional[int]:
    """Parse grade from @Comment lines."""
    for line in lines:
        if line.startswith('@Comment:'):
            # Look for "2nd Grade" or "5th Grade"
            match = re.search(r'(\d+)(?:st|nd|rd|th)\s+[Gg]rade', line)
            if match:
                return int(match.group(1))
    return None


def clean_utterance(text: str) -> str:
    """Clean CHAT utterance codes from text."""
    # Remove speaker code (*CHI:)
    text = re.sub(r'^\*\w+:\s*', '', text)

    # Remove CHAT codes
    text = re.sub(r'\[\^[^\]]*\]', '', text)  # [^c], [^ the end]
    text = re.sub(r'\[\*[^\]]*\]', '', text)  # [* word]
    text = re.sub(r'\[/\]', '', text)          # [/] retracing
    text = re.sub(r'\[//\]', '', text)         # [//] reformulation
    text = re.sub(r'\[///\]', '', text)        # [///] false start
    text = re.sub(r'\[\?\]', '', text)         # [?] uncertain
    text = re.sub(r'\[!\]', '', text)          # [!] emphatic
    text = re.sub(r'\[%[^\]]*\]', '', text)    # [% comment]
    text = re.sub(r'<[^>]*>\s*\[[^\]]*\]', '', text)  # <word> [/] repetition markers

    # Remove unintelligible markers
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'\byyy\b', '', text)
    text = re.sub(r'\bwww\b', '', text)

    # Remove lengthening markers (co:lon)
    text = re.sub(r':+', '', text)

    # Remove pauses
    text = re.sub(r'\(\.\.\.\)', '', text)
    text = re.sub(r'\(\.\)', '', text)

    # Remove 0word (omitted words)
    text = re.sub(r'\b0\w+', '', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def parse_cha_file(filepath: Path) -> Dict:
    """Parse a single .cha file and extract child utterances."""

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Parse metadata
    age_months = None
    grade = None
    sex = None
    doc_id = filepath.stem

    for line in lines:
        line = line.strip()

        # Parse @ID line
        if line.startswith('@ID:') and 'CHI' in line:
            age_months = parse_age_from_id(line)
            # Parse sex
            if '|male|' in line.lower():
                sex = 'male'
            elif '|female|' in line.lower():
                sex = 'female'

    # Parse grade from comments
    grade = parse_grade_from_comment(lines)

    # Extract child utterances
    utterances = []
    current_utterance = None

    for line in lines:
        line = line.rstrip()

        # New utterance from child
        if line.startswith('*CHI:'):
            if current_utterance:
                utterances.append(current_utterance)
            current_utterance = line

        # Continuation line (starts with tab)
        elif current_utterance and line.startswith('\t') and not line.startswith('\t%'):
            current_utterance += ' ' + line.strip()

        # Annotation tier or new speaker - save current utterance
        elif line.startswith('%') or line.startswith('*'):
            if current_utterance:
                utterances.append(current_utterance)
                current_utterance = None

    # Don't forget last utterance
    if current_utterance:
        utterances.append(current_utterance)

    # Clean and concatenate
    cleaned = [clean_utterance(u) for u in utterances]
    cleaned = [u for u in cleaned if u and len(u) > 1]  # Filter empty

    text = ' '.join(cleaned)

    return {
        'doc_id': doc_id,
        'text': text,
        'age_months': age_months,
        'grade': grade,
        'sex': sex,
        'n_utterances': len(cleaned),
        'source_file': str(filepath.name),
    }


def ingest_miami_mono(input_dir: Path, output_path: Path) -> None:
    """Ingest all Miami Mono .cha files."""

    records = []

    for group_dir in ['mleng2', 'mleng5']:
        group_path = input_dir / group_dir
        if not group_path.exists():
            print(f"Warning: {group_path} not found")
            continue

        grade = 2 if group_dir == 'mleng2' else 5
        group_label = group_dir

        cha_files = list(group_path.glob('*.cha'))
        print(f"\nProcessing {group_dir}: {len(cha_files)} files")

        for filepath in sorted(cha_files):
            try:
                record = parse_cha_file(filepath)

                # Add group metadata
                record['dataset'] = 'miami_mono'
                record['group'] = group_label

                # Use parsed grade or fall back to directory-based
                if record['grade'] is None:
                    record['grade'] = grade

                # Create age_bin for master table
                if record['grade'] == 2:
                    record['age_bin'] = 'grade2'
                    # Typical age for grade 2: 7-8 years
                    if record['age_months'] is None:
                        record['age_months'] = 7 * 12 + 6  # Default 7;6
                else:
                    record['age_bin'] = 'grade5'
                    # Typical age for grade 5: 10-11 years
                    if record['age_months'] is None:
                        record['age_months'] = 10 * 12 + 6  # Default 10;6

                # Create population dict (for compatibility with ECSC format)
                record['population'] = json.dumps({
                    'age_months': record['age_months'],
                    'grade': record['grade'],
                    'sex': record['sex'],
                    'group': record['group'],
                })

                # Skip if text is too short
                word_count = len(record['text'].split())
                record['word_count'] = word_count

                if word_count < 50:
                    print(f"  Skipping {filepath.name}: only {word_count} words")
                    continue

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
    print("MIAMI MONO INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")
    print(f"Output: {output_path}")

    # Stats by group
    for group in ['mleng2', 'mleng5']:
        group_records = [r for r in records if r['group'] == group]
        if group_records:
            words = [r['word_count'] for r in group_records]
            ages = [r['age_months'] for r in group_records if r['age_months']]
            print(f"\n{group}:")
            print(f"  n = {len(group_records)}")
            print(f"  words: {min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")
            if ages:
                print(f"  age_months: {min(ages)}-{max(ages)} (median: {sorted(ages)[len(ages)//2]})")


def main():
    parser = argparse.ArgumentParser(description='Ingest Miami Mono CHAT files')
    parser.add_argument('--input-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/English-MiamiMono'),
                        help='Input directory containing mleng2/ and mleng5/')
    parser.add_argument('--output', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/miami_mono_transcripts.jsonl'),
                        help='Output JSONL file')

    args = parser.parse_args()
    ingest_miami_mono(args.input_dir, args.output)


if __name__ == '__main__':
    main()
