#!/usr/bin/env python3
"""
ingest_enni.py

ENNI dataset ingestion following v4.2 pipeline spec.

Each document = one child story episode (A1...B3) with:
- group (TD vs LI)
- age_years
- story_id (A1...B3)
- child-only text

Handles:
- Multiple stories per child (multiple @G: blocks)
- Pre-@G child speech (excluded from main dataset)
- Adult scaffolding metrics

Usage:
    python ingest_enni.py --input-dir data/ENNI --output-dir data/enni_processed
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

# Group mapping
GROUP_MAP = {
    'TD': 'TD', 'TYP': 'TD', 'CONTROL': 'TD',
    'LI': 'LI', 'SLI': 'LI', 'LANGIMP': 'LI', 'IMPAIRED': 'LI',
}

# Valid story IDs
VALID_STORY_IDS = {'A1', 'A2', 'A3', 'B1', 'B2', 'B3'}


# ============================================================
# PARSING FUNCTIONS
# ============================================================

def parse_age(age_str: str) -> Optional[float]:
    """Parse age string like '4;10.02' to float years.

    Format: Y;M where M may be decimal (months + days/30)
    """
    if not age_str:
        return None

    # Try Y;M format
    match = re.match(r'(\d+);(\d+\.?\d*)', age_str)
    if match:
        years = int(match.group(1))
        months = float(match.group(2))
        return years + months / 12

    # Try just years
    match = re.match(r'(\d+)', age_str)
    if match:
        return float(match.group(1))

    return None


def parse_header(lines: List[str]) -> Dict:
    """Parse CHAT header for metadata."""

    metadata = {
        'child_code': 'CHI',
        'age_years': None,
        'age_str': None,
        'sex': None,
        'group': 'UNK',
        'group_raw': None,
        'age_source': None,
        'group_source': None,
    }

    # Find child participant code from @Participants
    for line in lines:
        if line.startswith('@Participants:'):
            # Default is CHI, but check for explicit child marker
            if 'CHI' in line:
                metadata['child_code'] = 'CHI'
            break

    # Find child @ID line
    child_code = metadata['child_code']
    for line in lines:
        if line.startswith('@ID:') and f'|{child_code}|' in line:
            # Parse @ID line
            # Format: lang|corpus|code|age|sex|group|...|role|...
            parts = line.replace('@ID:', '').strip().split('|')

            if len(parts) >= 6:
                # Age is field 3 (0-indexed)
                metadata['age_str'] = parts[3] if len(parts) > 3 else None
                metadata['age_years'] = parse_age(parts[3]) if len(parts) > 3 else None
                metadata['age_source'] = 'ID' if metadata['age_years'] else None

                # Sex is field 4
                metadata['sex'] = parts[4] if len(parts) > 4 and parts[4] else None

                # Group is field 5
                if len(parts) > 5 and parts[5]:
                    metadata['group_raw'] = parts[5]
                    metadata['group'] = GROUP_MAP.get(parts[5].upper(), 'UNK')
                    metadata['group_source'] = 'ID'

                    if metadata['group'] == 'UNK' and parts[5]:
                        print(f"    Warning: Unknown group '{parts[5]}'")

            break

    return metadata


def clean_utterance(text: str) -> str:
    """Clean CHAT utterance aggressively for LM analysis.

    Goal: Remove ALL transcription artifacts so we model narrative only.
    """

    # === TIMESTAMPS ===
    # Remove timestamps like 123_456 or •123_456•
    text = re.sub(r'•?\d+_\d+•?', '', text)

    # === BRACKETED CODES (all types) ===
    text = re.sub(r'\[\+[^\]]*\]', '', text)   # [+ bch], [+ exc]
    text = re.sub(r'\[/+\]', '', text)          # [/], [//], [///]
    text = re.sub(r'\[\.\.\.\]', '', text)      # [...]
    text = re.sub(r'\[\*[^\]]*\]', '', text)    # [* error]
    text = re.sub(r'\[\?[^\]]*\]', '', text)    # [?]
    text = re.sub(r'\[%[^\]]*\]', '', text)     # [% comment]
    text = re.sub(r'\[>[^\]]*\]', '', text)     # [>]
    text = re.sub(r'\[<[^\]]*\]', '', text)     # [<]
    text = re.sub(r'\[=[^\]]*\]', '', text)     # [= explanation]
    text = re.sub(r'\[:+[^\]]*\]', '', text)    # [: replacement], [:: replacement]
    text = re.sub(r'\[\^[^\]]*\]', '', text)    # [^c]
    text = re.sub(r'\[[^\]]*\]', '', text)      # Catch-all remaining brackets

    # === ANGLE BRACKET MARKUP ===
    text = re.sub(r'<[^>]*>\s*\[[^\]]*\]', '', text)  # <word> [code]
    text = re.sub(r'<[^>]*>', '', text)               # standalone <word>

    # === DISFLUENCY & ACTION MARKERS ===
    text = re.sub(r'&=[^\s]+', '', text)       # &=whispers, &=laughs (actions) - REMOVE
    text = re.sub(r'&~[^\s]+', '', text)       # &~word (phonological fragment) - REMOVE
    text = re.sub(r'&-(\w+)', r'\1', text)     # &-um → um (filled pause)
    text = re.sub(r'&\+(\w+)', r'\1', text)    # &+s → s
    text = re.sub(r'&\*\w+:\w+', '', text)     # &*EXA:mhm (interlocutor vocalization)
    text = re.sub(r'&[a-z]+', '', text)        # &word → remove remaining

    # === OMITTED LETTERS - EXPAND ===
    # (u)s → us, n(o)t → not, (i)s → is, (wi)ll → will, etc.
    text = re.sub(r'\((\w+)\)', r'\1', text)

    # === CHAT SPECIAL MARKERS ===
    text = re.sub(r'\(\.\)', '', text)         # (.) pause
    text = re.sub(r'\(\.\.\)', '', text)       # (..) pause
    text = re.sub(r'\(\.\.\.\)', '', text)     # (...) pause
    text = re.sub(r'\bxxx\b', '', text)        # unintelligible
    text = re.sub(r'\byyy\b', '', text)        # phonological
    text = re.sub(r'\bwww\b', '', text)        # untranscribed
    text = re.sub(r'\b0\w*\b', '', text)       # 0word (omitted word)

    # === QUOTATION/CONTINUATION MARKERS ===
    text = re.sub(r'\+"/\.?', '', text)
    text = re.sub(r'\+"/?', '', text)
    text = re.sub(r'\+/\.?', '', text)
    text = re.sub(r'\+,', '', text)
    text = re.sub(r'\+\.\.\.', '', text)
    text = re.sub(r'\+\.\.\?', '', text)
    text = re.sub(r'\+!\.?', '', text)
    text = re.sub(r'\+\^', '', text)

    # === @MARKERS ===
    text = re.sub(r'@[a-z:]+', '', text)       # @o, @g, @si:eng, etc.

    # === LENGTHENING ===
    text = re.sub(r':+', '', text)

    # === SPECIAL CHARACTERS ===
    text = re.sub(r'[‡„]', '', text)           # CHAT special chars
    text = re.sub(r'↑|↓', '', text)            # Intonation markers
    text = re.sub(r'#', ' ', text)             # Pause marker

    # === CLEANUP ===
    text = re.sub(r'\s+', ' ', text)           # Collapse whitespace
    text = text.strip()
    text = re.sub(r'^[.!?,;\s]+', '', text)    # Leading punctuation
    text = re.sub(r'\s+([.!?,;])', r'\1', text)  # Space before punctuation

    # Ensure ends with punctuation
    if text and text[-1] not in '.!?':
        text = text + '.'

    return text


def compute_contamination_metrics(text: str) -> Dict:
    """Compute format-contamination audit metrics for a cleaned text.

    Focus on TRUE contamination indicators (markup artifacts), not natural speech patterns.
    """

    if not text:
        return {'pct_markup': 1.0, 'pct_digits': 1.0, 'n_tokens': 0}

    tokens = text.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {'pct_markup': 1.0, 'pct_digits': 1.0, 'n_tokens': 0}

    # Count TRUE contamination (not natural speech)
    n_with_digits = 0
    n_with_underscore = 0
    n_with_brackets = 0
    n_with_ampersand = 0
    n_with_at = 0

    for token in tokens:
        if re.search(r'\d', token):
            n_with_digits += 1
        if '_' in token:
            n_with_underscore += 1
        if re.search(r'[\[\]<>{}]', token):
            n_with_brackets += 1
        if '&' in token:
            n_with_ampersand += 1
        if '@' in token:
            n_with_at += 1

    n_markup = n_with_digits + n_with_underscore + n_with_brackets + n_with_ampersand + n_with_at

    return {
        'n_tokens': n_tokens,
        'pct_markup': n_markup / n_tokens,
        'pct_digits': n_with_digits / n_tokens,
    }


def extract_speaker_text(line: str, speaker_code: str) -> Optional[str]:
    """Extract text from a speaker line like *CHI: text here."""
    pattern = rf'^\*{speaker_code}:\s*(.*)$'
    match = re.match(pattern, line)
    if match:
        return match.group(1)
    return None


def parse_story_id(line: str) -> Optional[str]:
    """Extract story ID from @G: line."""
    if not line.startswith('@G:'):
        return None

    # Look for A1, A2, A3, B1, B2, B3
    match = re.search(r'\b([AB][123])\b', line, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def segment_file(lines: List[str], child_code: str = 'CHI') -> Tuple[Dict[str, List[str]], List[str], Dict[str, List[str]]]:
    """Segment file into story documents.

    Returns:
        story_buffers: dict of story_id -> list of child utterances
        pretask_buffer: list of child utterances before first @G:
        adult_buffers: dict of story_id -> list of adult utterances
    """

    story_buffers = defaultdict(list)
    adult_buffers = defaultdict(list)
    pretask_buffer = []
    pretask_adult = []

    current_story_id = None
    first_g_seen = False

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.rstrip()

        # Check for @G: marker
        if line.startswith('@G:'):
            # Save any pending utterance
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    if not first_g_seen:
                        if current_speaker == child_code:
                            pretask_buffer.append(cleaned)
                        else:
                            pretask_adult.append(cleaned)
                    elif current_story_id:
                        if current_speaker == child_code:
                            story_buffers[current_story_id].append(cleaned)
                        else:
                            adult_buffers[current_story_id].append(cleaned)

            current_speaker = None
            current_text = ""

            story_id = parse_story_id(line)
            if story_id and story_id in VALID_STORY_IDS:
                current_story_id = story_id
                first_g_seen = True
            continue

        # Check for speaker line
        if line.startswith('*'):
            # Save previous utterance
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    if not first_g_seen:
                        if current_speaker == child_code:
                            pretask_buffer.append(cleaned)
                        else:
                            pretask_adult.append(cleaned)
                    elif current_story_id:
                        if current_speaker == child_code:
                            story_buffers[current_story_id].append(cleaned)
                        else:
                            adult_buffers[current_story_id].append(cleaned)

            # Parse new speaker
            match = re.match(r'^\*(\w+):\s*(.*)', line)
            if match:
                current_speaker = match.group(1)
                current_text = match.group(2)
            else:
                current_speaker = None
                current_text = ""
            continue

        # Continuation line (not annotation tier)
        if line.startswith('\t') and not line.startswith('\t%'):
            if current_speaker:
                current_text += ' ' + line.strip()
            continue

        # Annotation tier or other - ignore
        if line.startswith('%') or line.startswith('@'):
            continue

    # Don't forget last utterance
    if current_speaker and current_text:
        cleaned = clean_utterance(current_text)
        if cleaned:
            if not first_g_seen:
                if current_speaker == child_code:
                    pretask_buffer.append(cleaned)
                else:
                    pretask_adult.append(cleaned)
            elif current_story_id:
                if current_speaker == child_code:
                    story_buffers[current_story_id].append(cleaned)
                else:
                    adult_buffers[current_story_id].append(cleaned)

    return dict(story_buffers), pretask_buffer, dict(adult_buffers)


def compute_adult_frac(child_text: str, adult_utterances: List[str]) -> float:
    """Compute adult token fraction."""
    child_tokens = len(child_text.split())
    adult_text = ' '.join(adult_utterances)
    adult_tokens = len(adult_text.split())

    total = child_tokens + adult_tokens
    if total == 0:
        return 0.0
    return adult_tokens / total


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest_enni(input_dir: Path, output_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Ingest all ENNI .cha files."""

    story_docs = []
    pretask_docs = []

    # Find all .cha files
    cha_files = list(input_dir.rglob('*.cha'))
    print(f"Found {len(cha_files)} .cha files")

    for filepath in sorted(cha_files):
        file_id = filepath.stem
        path_rel = str(filepath.relative_to(input_dir))

        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            continue

        # Parse header
        metadata = parse_header(lines)

        if metadata['group'] == 'UNK':
            print(f"  Warning: {file_id} has unknown group")

        if metadata['age_years'] is None:
            print(f"  Warning: {file_id} has no age")

        # Segment into stories
        story_buffers, pretask_buffer, adult_buffers = segment_file(
            lines, metadata['child_code']
        )

        # Create story documents
        for story_id in sorted(story_buffers.keys()):
            utterances = story_buffers[story_id]
            if not utterances:
                continue

            text = ' '.join(utterances)
            word_count = len(text.split())

            if word_count < 10:
                continue

            adult_utts = adult_buffers.get(story_id, [])
            adult_frac = compute_adult_frac(text, adult_utts)

            # Compute contamination metrics
            contam = compute_contamination_metrics(text)

            doc = {
                'doc_id': f'{file_id}_{story_id}',
                'file_id': file_id,
                'story_id': story_id,
                'group': metadata['group'],
                'age_years': metadata['age_years'],
                'sex': metadata['sex'],
                'adult_token_frac': adult_frac,
                'text': text,
                'word_count': word_count,
                'pct_markup': contam['pct_markup'],
                'pct_digits': contam['pct_digits'],
                'path_rel': path_rel,
            }

            story_docs.append(doc)

        # Create pretask document if substantial
        if pretask_buffer:
            text = ' '.join(pretask_buffer)
            word_count = len(text.split())

            if word_count >= 20:
                contam = compute_contamination_metrics(text)
                doc = {
                    'doc_id': f'{file_id}_PRE',
                    'file_id': file_id,
                    'story_id': 'PRE',
                    'group': metadata['group'],
                    'age_years': metadata['age_years'],
                    'sex': metadata['sex'],
                    'adult_token_frac': 0.0,  # Not tracked for pretask
                    'text': text,
                    'word_count': word_count,
                    'pct_markup': contam['pct_markup'],
                    'pct_digits': contam['pct_digits'],
                    'path_rel': path_rel,
                }
                pretask_docs.append(doc)

    return story_docs, pretask_docs


def write_csv(docs: List[Dict], output_path: Path) -> None:
    """Write documents to CSV."""
    if not docs:
        print(f"  No documents to write to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['doc_id', 'file_id', 'story_id', 'group', 'age_years',
                  'sex', 'adult_token_frac', 'word_count', 'pct_markup',
                  'pct_digits', 'text', 'path_rel']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(docs)

    print(f"  Wrote {len(docs)} documents to {output_path}")


def write_jsonl(docs: List[Dict], output_path: Path) -> None:
    """Write documents to JSONL (for LM pipeline compatibility)."""
    if not docs:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            # Add population field for compatibility
            doc_out = doc.copy()
            doc_out['population'] = json.dumps({
                'age_years': doc['age_years'],
                'age_months': int(doc['age_years'] * 12) if doc['age_years'] else None,
                'group': doc['group'],
                'sex': doc['sex'],
                'story_id': doc['story_id'],
            })
            doc_out['age_months'] = int(doc['age_years'] * 12) if doc['age_years'] else None
            f.write(json.dumps(doc_out) + '\n')

    print(f"  Wrote {len(docs)} documents to {output_path}")


def print_summary(docs: List[Dict], name: str) -> None:
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"{name}: {len(docs)} documents")
    print('='*60)

    if not docs:
        return

    # By group
    by_group = defaultdict(list)
    for d in docs:
        by_group[d['group']].append(d)

    print("\nBy group:")
    for group in sorted(by_group.keys()):
        group_docs = by_group[group]
        ages = [d['age_years'] for d in group_docs if d['age_years']]
        words = [d['word_count'] for d in group_docs]
        if ages:
            print(f"  {group}: n={len(group_docs)}, age={min(ages):.1f}-{max(ages):.1f} (median: {sorted(ages)[len(ages)//2]:.1f}), words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")
        else:
            print(f"  {group}: n={len(group_docs)}, age=N/A, words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")

    # By story_id
    by_story = defaultdict(list)
    for d in docs:
        by_story[d['story_id']].append(d)

    print("\nBy story_id:")
    for story_id in sorted(by_story.keys()):
        story_docs = by_story[story_id]
        words = [d['word_count'] for d in story_docs]
        print(f"  {story_id}: n={len(story_docs)}, words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")

    # Adult scaffolding by age
    td_docs = [d for d in docs if d['group'] == 'TD' and d['age_years']]
    if td_docs:
        print("\nAdult scaffolding by age (TD only):")
        by_age = defaultdict(list)
        for d in td_docs:
            age_bin = int(d['age_years'])
            by_age[age_bin].append(d['adult_token_frac'])

        for age in sorted(by_age.keys()):
            fracs = by_age[age]
            mean_frac = sum(fracs) / len(fracs)
            print(f"  {age}yr: n={len(fracs)}, adult_frac={mean_frac:.1%}")


def sanity_check(docs: List[Dict], n: int = 10) -> None:
    """Print random sample for verification."""
    import random

    print(f"\n{'='*60}")
    print(f"SANITY CHECK: {n} random documents")
    print('='*60)

    sample = random.sample(docs, min(n, len(docs)))

    for doc in sample:
        print(f"\n{doc['doc_id']} | {doc['group']} | age={doc['age_years']:.1f} | {doc['story_id']} | words={doc['word_count']} | adult={doc['adult_token_frac']:.1%}")
        print(f"  {doc['text'][:120]}...")


def main():
    parser = argparse.ArgumentParser(description='Ingest ENNI CHAT files')
    parser.add_argument('--input-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/ENNI'),
                        help='Input directory')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/data/enni_processed'),
                        help='Output directory')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Print sanity check sample')

    args = parser.parse_args()

    print(f"Ingesting ENNI from {args.input_dir}")

    story_docs, pretask_docs = ingest_enni(args.input_dir, args.output_dir)

    # Write outputs
    print("\nWriting outputs...")
    write_csv(story_docs, args.output_dir / 'enni_story_docs.csv')
    write_csv(pretask_docs, args.output_dir / 'enni_pretask_docs.csv')
    write_jsonl(story_docs, args.output_dir / 'enni_story_transcripts.jsonl')

    # Summary
    print_summary(story_docs, "ENNI STORY DOCUMENTS")
    print_summary(pretask_docs, "ENNI PRETASK DOCUMENTS")

    # Sanity check
    if args.sanity_check and story_docs:
        sanity_check(story_docs)


if __name__ == '__main__':
    main()
