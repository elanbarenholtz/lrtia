#!/usr/bin/env python3
"""
ingest_enni_v2.py

ENNI CHAT/CLAN ingestion following the detailed specification.

Output: One cleaned narrative per (file_id, story_id) containing CHI-only utterances,
plus demographic metadata and QA diagnostics.

Usage:
    python ingest_enni_v2.py --input-dir data/ENNI --output-dir data/enni_clean
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'min_word_count': 30,
    'max_pct_markup': 0.01,
    'max_pct_digits': 0.05,
    'lowercase': False,  # Match ECSC/Miami if they were lowercased
}

# Group normalization
GROUP_MAP = {
    'TD': 'TD', 'TYP': 'TD', 'CONTROL': 'TD', 'TYPICAL': 'TD',
    'LI': 'LI', 'SLI': 'LI', 'LANGIMP': 'LI', 'IMPAIRED': 'LI',
}


# ============================================================
# 3. METADATA EXTRACTION
# ============================================================

def parse_age_string(age_str: str) -> Tuple[Optional[float], Optional[int]]:
    """Parse CHAT age string like '4;10.01' to (age_years, age_months).

    Format: Y;MM.xx where MM.xx can include fractional months.
    Returns: (age_years as float, age_months as int)
    """
    if not age_str or not age_str.strip():
        return None, None

    # Try Y;MM.xx format
    match = re.match(r'(\d+);(\d+\.?\d*)', age_str.strip())
    if match:
        years = int(match.group(1))
        months_frac = float(match.group(2))
        total_months = 12 * years + months_frac
        age_months = int(math.floor(total_months))
        age_years = total_months / 12.0
        return age_years, age_months

    # Try just years
    match = re.match(r'^(\d+)$', age_str.strip())
    if match:
        years = int(match.group(1))
        return float(years), years * 12

    return None, None


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string like '26-AUG-1995'."""
    if not date_str:
        return None

    # Try DD-MMM-YYYY format
    try:
        return datetime.strptime(date_str.strip(), '%d-%b-%Y')
    except ValueError:
        pass

    # Try other formats
    for fmt in ['%d-%B-%Y', '%Y-%m-%d', '%m/%d/%Y']:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass

    return None


def compute_age_from_dates(dob: datetime, rec_date: datetime) -> Tuple[float, int]:
    """Compute age from date of birth and recording date."""
    days_diff = (rec_date - dob).days
    age_years = days_diff / 365.25
    age_months = int(math.floor(days_diff / 30.4375))  # Average days per month
    return age_years, age_months


def parse_chi_id_line(line: str) -> Dict:
    """Parse CHI @ID line for metadata.

    Format: @ID: lang|corpus|CHI|age|sex|group|...|role|...
    Example: @ID: eng|ENNI|CHI|4;10.01|female|TD||Target_Child|||
    """
    result = {
        'age_str': None,
        'age_years': None,
        'age_months': None,
        'sex': None,
        'group': None,
        'group_raw': None,
    }

    # Extract the part after @ID:
    content = line.replace('@ID:', '').strip()
    parts = content.split('|')

    if len(parts) < 6:
        return result

    # Parts: lang, corpus, code, age, sex, group, ...
    # Index:   0      1      2     3    4     5

    if parts[2].upper() != 'CHI':
        return result  # Not the CHI line

    # Age (index 3)
    if len(parts) > 3 and parts[3]:
        result['age_str'] = parts[3]
        result['age_years'], result['age_months'] = parse_age_string(parts[3])

    # Sex (index 4)
    if len(parts) > 4 and parts[4]:
        result['sex'] = parts[4].lower()

    # Group (index 5)
    if len(parts) > 5 and parts[5]:
        result['group_raw'] = parts[5]
        result['group'] = GROUP_MAP.get(parts[5].upper(), parts[5].upper())

    return result


def extract_metadata(lines: List[str]) -> Dict:
    """Extract all metadata from file header."""

    metadata = {
        'age_years': None,
        'age_months': None,
        'age_str': None,
        'sex': None,
        'group': None,
        'group_raw': None,
        'dob': None,
        'rec_date': None,
        'age_source': None,
    }
    qa_flags = []

    for line in lines:
        line = line.strip()

        # Parse CHI @ID line
        if line.startswith('@ID:') and '|CHI|' in line:
            id_data = parse_chi_id_line(line)
            if id_data['age_years'] is not None:
                metadata['age_years'] = id_data['age_years']
                metadata['age_months'] = id_data['age_months']
                metadata['age_str'] = id_data['age_str']
                metadata['age_source'] = 'ID'
            if id_data['sex']:
                metadata['sex'] = id_data['sex']
            if id_data['group']:
                metadata['group'] = id_data['group']
                metadata['group_raw'] = id_data['group_raw']

        # Parse DOB from @Comment
        if line.startswith('@Comment:') and 'Birth of CHI' in line:
            match = re.search(r'Birth of CHI is\s*(\d{1,2}-[A-Z]{3}-\d{4})', line, re.IGNORECASE)
            if match:
                metadata['dob'] = parse_date(match.group(1))

        # Parse recording date
        if line.startswith('@Date:'):
            date_str = line.replace('@Date:', '').strip()
            metadata['rec_date'] = parse_date(date_str)

    # Fallback: compute age from DOB + recording date
    if metadata['age_years'] is None and metadata['dob'] and metadata['rec_date']:
        metadata['age_years'], metadata['age_months'] = compute_age_from_dates(
            metadata['dob'], metadata['rec_date']
        )
        metadata['age_source'] = 'dates'
        qa_flags.append('age_from_dates')

    # Default group if not found
    if not metadata['group']:
        metadata['group'] = 'UNK'

    return metadata, qa_flags


# ============================================================
# 2. SEGMENTATION
# ============================================================

def parse_story_id_from_g_line(line: str) -> Optional[str]:
    """Extract story ID from @G: line.

    Regex pattern: ^@G:\s*([A-Za-z]\d+)\s*$
    """
    match = re.match(r'^@G:\s*([A-Za-z]\d+)\s*$', line.strip())
    if match:
        return match.group(1).upper()
    return None


def is_adult_prompt(line: str) -> bool:
    """Check if line is an adult prompt to start a story.

    Regex (case-insensitive): ^\*(EXA|INV|ADU):.*\btell me\b.*\bstory\b
    """
    if not re.match(r'^\*(EXA|INV|ADU):', line, re.IGNORECASE):
        return False
    return bool(re.search(r'\btell me\b.*\bstory\b', line, re.IGNORECASE))


def segment_into_story_blocks(lines: List[str]) -> List[Dict]:
    """Segment file into story blocks.

    Returns list of blocks, each with:
        story_id, start_line, end_line, lines, segmentation_method
    """
    blocks = []
    current_block = None
    synthetic_id_counter = 0
    has_g_markers = any(line.strip().startswith('@G:') for line in lines)

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Check for @G: marker (primary segmentation)
        if line_stripped.startswith('@G:'):
            story_id = parse_story_id_from_g_line(line_stripped)
            if story_id:
                # Close previous block
                if current_block:
                    current_block['end_line'] = i
                    blocks.append(current_block)

                # Start new block
                current_block = {
                    'story_id': story_id,
                    'start_line': i + 1,
                    'end_line': None,
                    'lines': [],
                    'segmentation_method': 'G_marker',
                }
                continue

        # Fallback: adult prompt (only if no @G: markers in file)
        if not has_g_markers and is_adult_prompt(line_stripped):
            synthetic_id_counter += 1

            if current_block:
                current_block['end_line'] = i
                blocks.append(current_block)

            current_block = {
                'story_id': f'S{synthetic_id_counter}',
                'start_line': i + 1,
                'end_line': None,
                'lines': [],
                'segmentation_method': 'prompt_fallback',
            }
            continue

        # Add line to current block
        if current_block:
            current_block['lines'].append(line)

    # Close final block
    if current_block:
        current_block['end_line'] = len(lines)
        blocks.append(current_block)

    return blocks


# ============================================================
# 4. TEXT EXTRACTION
# ============================================================

def extract_main_tier(line: str) -> Optional[Tuple[str, str]]:
    """Extract speaker code and text from main tier line.

    Regex pattern: ^\*([A-Z]{3}):\s*(.*)$
    Returns: (speaker_code, utterance_text) or None
    """
    match = re.match(r'^\*([A-Z]{3}):\s*(.*)$', line)
    if match:
        return match.group(1), match.group(2)
    return None


def extract_utterances_from_block(block_lines: List[str]) -> Dict:
    """Extract CHI and adult utterances from a story block.

    Returns:
        chi_utterances: list of CHI utterance strings
        adult_utterances: list of adult utterance strings
        n_chi_utts: count of CHI utterances
        n_total_utts: count of all main tier utterances
    """
    chi_utterances = []
    adult_utterances = []
    n_chi_utts = 0
    n_total_utts = 0

    current_speaker = None
    current_text = ""

    for line in block_lines:
        line = line.rstrip()

        # Check for main tier
        tier_match = extract_main_tier(line)
        if tier_match:
            # Save previous utterance
            if current_speaker and current_text.strip():
                if current_speaker == 'CHI':
                    chi_utterances.append(current_text.strip())
                    n_chi_utts += 1
                else:
                    adult_utterances.append(current_text.strip())
                n_total_utts += 1

            current_speaker, current_text = tier_match
            continue

        # Continuation line (starts with tab, not dependent tier)
        if current_speaker and line.startswith('\t') and not line.startswith('\t%'):
            current_text += ' ' + line.strip()
            continue

        # Dependent tier or other - ignore but close current utterance
        if line.startswith('%') or line.startswith('@'):
            if current_speaker and current_text.strip():
                if current_speaker == 'CHI':
                    chi_utterances.append(current_text.strip())
                    n_chi_utts += 1
                else:
                    adult_utterances.append(current_text.strip())
                n_total_utts += 1
            current_speaker = None
            current_text = ""

    # Don't forget final utterance
    if current_speaker and current_text.strip():
        if current_speaker == 'CHI':
            chi_utterances.append(current_text.strip())
            n_chi_utts += 1
        else:
            adult_utterances.append(current_text.strip())
        n_total_utts += 1

    return {
        'chi_utterances': chi_utterances,
        'adult_utterances': adult_utterances,
        'n_chi_utts': n_chi_utts,
        'n_total_utts': n_total_utts,
    }


def compute_adult_token_frac_pre(chi_utts: List[str], adult_utts: List[str]) -> float:
    """Compute adult token fraction from raw utterances (speaker-based).

    Uses naive whitespace tokenization after basic control char removal.
    """
    def tokenize_raw(text: str) -> List[str]:
        # Remove control chars for tokenization
        text = re.sub(r'[\x00-\x1F\x7F\u0080-\u009F]+', ' ', text)
        return text.split()

    chi_tokens = sum(len(tokenize_raw(u)) for u in chi_utts)
    adult_tokens = sum(len(tokenize_raw(u)) for u in adult_utts)
    total_tokens = chi_tokens + adult_tokens

    if total_tokens == 0:
        return 0.0

    return adult_tokens / total_tokens


# ============================================================
# 5. CLEANING PIPELINE
# ============================================================

def clean_text(text: str, lowercase: bool = False) -> Tuple[str, List[str]]:
    """Apply full cleaning pipeline to text.

    Returns: (cleaned_text, qa_flags)
    """
    qa_flags = []

    # 5.1 Remove control characters (MUST)
    # ASCII control: [\x00-\x1F\x7F]
    # C1 controls: [\u0080-\u009F]
    control_chars_found = len(re.findall(r'[\x00-\x1F\x7F\u0080-\u009F]', text))
    if control_chars_found > 0:
        qa_flags.append(f'control_chars_removed:{control_chars_found}')
    text = re.sub(r'[\x00-\x1F\x7F\u0080-\u009F]+', '', text)

    # 5.2 Remove CHAT timecodes
    # Pattern: \d+_\d+ (any length - some are 7+ digits)
    timecodes_found = len(re.findall(r'\d+_\d+', text))
    if timecodes_found > 0:
        qa_flags.append(f'timecodes_removed:{timecodes_found}')
    text = re.sub(r'\d+_\d+', '', text)

    # 5.3.1 Remove square brackets and contents
    # Pattern: \[[^\]]*\]
    brackets_found = len(re.findall(r'\[[^\]]*\]', text))
    if brackets_found > 0:
        qa_flags.append(f'brackets_removed:{brackets_found}')
    text = re.sub(r'\[[^\]]*\]', '', text)

    # 5.3.2 Remove parenthetical pause markers
    # Pattern: \(\s*\.?\s*\d*\.?\d*\s*\)
    # Also handles (.), (..), (...), (2.0), etc.
    pauses_found = len(re.findall(r'\(\s*\.+\s*\)|\(\d+\.?\d*\)', text))
    text = re.sub(r'\(\s*\.+\s*\)', '', text)  # (.), (..), (...)
    text = re.sub(r'\(\d+\.?\d*\)', '', text)  # (2.0), (3)

    # 5.3.3 Remove angle brackets but keep content
    text = re.sub(r'<', '', text)
    text = re.sub(r'>', '', text)

    # 5.4 Remove CHAT special tokens (xxx, yyy, www)
    noise_tokens = len(re.findall(r'(?i)\b(xxx|yyy|www)\b', text))
    if noise_tokens > 0:
        qa_flags.append(f'noise_tokens_removed:{noise_tokens}')
    text = re.sub(r'(?i)\b(xxx|yyy|www)\b', '', text)

    # Additional CHAT cleanup
    # Remove &=action markers
    text = re.sub(r'&=[^\s]+', '', text)
    # Remove &-um, &+s type markers (keep the word)
    text = re.sub(r'&[-+](\w+)', r'\1', text)
    # Remove &*speaker:word (interlocutor vocalizations)
    text = re.sub(r'&\*\w+:\w+', '', text)
    # Remove remaining & markers
    text = re.sub(r'&\w*', '', text)
    # Remove @ markers
    text = re.sub(r'@[a-z:]+', '', text)
    # Expand omitted letters: (u)s → us, n(o)t → not
    text = re.sub(r'\((\w+)\)', r'\1', text)
    # Remove CHAT quotation markers
    text = re.sub(r'\+["\'/?!,\.]+', '', text)
    text = re.sub(r'\+\^', '', text)
    # Remove lengthening colons
    text = re.sub(r':+', '', text)
    # Remove 0word (omitted words)
    text = re.sub(r'\b0\w*', '', text)

    # 5.5 Normalize punctuation and whitespace
    # Collapse whitespace first
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove isolated/repeated punctuation artifacts (from pause removal)
    # ". ." → "." and ". . ." → "."
    text = re.sub(r'(\.\s*){2,}', '. ', text)  # Multiple periods → single
    text = re.sub(r'(\?\s*){2,}', '? ', text)
    text = re.sub(r'(!\s*){2,}', '! ', text)

    # Standardize sentence punctuation spacing
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*\?\s*', '? ', text)
    text = re.sub(r'\s*!\s*', '! ', text)
    text = re.sub(r'\s*,\s*', ', ', text)

    # Collapse spaces again
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove trailing space before final punctuation
    text = re.sub(r'\s+([.?!])$', r'\1', text)

    # Ensure ends with punctuation
    if text and text[-1] not in '.?!':
        text = text + '.'

    # Lowercase if configured
    if lowercase:
        text = text.lower()

    return text, qa_flags


# ============================================================
# 6. DERIVED FIELDS
# ============================================================

def compute_derived_fields(text: str) -> Dict:
    """Compute word_count, pct_digits, pct_markup."""

    tokens = text.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'word_count': 0,
            'pct_digits': 0.0,
            'pct_markup': 0.0,
        }

    # pct_digits: tokens that are purely digits
    digit_tokens = sum(1 for t in tokens if re.match(r'^\d+$', t))

    # pct_markup: tokens with markup-like characters
    # [\[\]\(\)<>]|^\*|^%|^@|_\d+
    markup_tokens = sum(1 for t in tokens if (
        re.search(r'[\[\]\(\)<>]', t) or
        t.startswith('*') or
        t.startswith('%') or
        t.startswith('@') or
        re.search(r'_\d+', t)
    ))

    return {
        'word_count': n_tokens,
        'pct_digits': digit_tokens / n_tokens,
        'pct_markup': markup_tokens / n_tokens,
    }


# ============================================================
# 9. QA CHECKS
# ============================================================

def run_qa_checks(text: str, qa_flags: List[str]) -> List[str]:
    """Run QA assertions and add flags."""

    # 9.1 No control characters remain
    if re.search(r'[\x00-\x1F\x7F\u0080-\u009F]', text):
        qa_flags.append('FAIL:control_chars_remain')

    # 9.2 No non-CHI leakage (tier prefixes)
    if re.search(r'(^|\s)[@%*][A-Za-z]{1,10}:', text):
        qa_flags.append('FAIL:tier_prefix_leakage')

    return qa_flags


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_file(filepath: Path, input_dir: Path) -> List[Dict]:
    """Process a single .cha file into story records."""

    records = []
    file_id = filepath.stem
    path_rel = str(filepath.relative_to(input_dir))

    # Read file
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return []

    # Extract metadata
    metadata, meta_qa_flags = extract_metadata(lines)

    # Segment into story blocks
    blocks = segment_into_story_blocks(lines)

    # QA: story count
    file_qa_flags = list(meta_qa_flags)
    if len(blocks) < 3:
        file_qa_flags.append('few_stories')
    elif len(blocks) > 8:
        file_qa_flags.append('many_stories')

    # Check for no @G: markers
    if blocks and blocks[0]['segmentation_method'] == 'prompt_fallback':
        file_qa_flags.append('no_G_marker')

    # Process each story block
    for block in blocks:
        story_id = block['story_id']

        # Extract utterances
        utt_data = extract_utterances_from_block(block['lines'])

        if not utt_data['chi_utterances']:
            continue  # No CHI utterances in this block

        # Build raw CHI text
        text_raw_chi = ' '.join(utt_data['chi_utterances'])

        # Compute adult token fraction (speaker-based, pre-cleaning)
        adult_token_frac_pre = compute_adult_token_frac_pre(
            utt_data['chi_utterances'],
            utt_data['adult_utterances']
        )

        # Clean text
        text, clean_qa_flags = clean_text(text_raw_chi, lowercase=CONFIG['lowercase'])

        # Compute derived fields
        derived = compute_derived_fields(text)

        # Combine QA flags
        qa_flags = list(file_qa_flags) + clean_qa_flags
        qa_flags = run_qa_checks(text, qa_flags)

        # Build record
        record = {
            'doc_id': f'{file_id}_{story_id}',
            'file_id': file_id,
            'story_id': story_id,
            'group': metadata['group'],
            'age_years': round(metadata['age_years'], 4) if metadata['age_years'] else None,
            'age_months': metadata['age_months'],
            'sex': metadata['sex'],
            'adult_token_frac_pre': round(adult_token_frac_pre, 4),
            'text_raw_chi': text_raw_chi,
            'text': text,
            'word_count': derived['word_count'],
            'pct_markup': round(derived['pct_markup'], 6),
            'pct_digits': round(derived['pct_digits'], 6),
            'n_chi_utts': utt_data['n_chi_utts'],
            'n_total_utts': utt_data['n_total_utts'],
            'qa_flags': qa_flags,
            'path_rel': path_rel,
            'population': json.dumps({
                'age_years': round(metadata['age_years'], 4) if metadata['age_years'] else None,
                'age_months': metadata['age_months'],
                'group': metadata['group'],
                'sex': metadata['sex'],
                'story_id': story_id,
            }),
        }

        records.append(record)

    return records


def apply_filters(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Apply filtering rules and return kept records + drop stats."""

    kept = []
    drop_stats = defaultdict(int)

    for r in records:
        # Empty text
        if not r['text'].strip():
            drop_stats['empty_text'] += 1
            continue

        # Min word count
        if r['word_count'] < CONFIG['min_word_count']:
            drop_stats['below_min_words'] += 1
            continue

        # pct_markup threshold
        if r['pct_markup'] > CONFIG['max_pct_markup']:
            drop_stats['high_pct_markup'] += 1
            continue

        # pct_digits threshold (optional)
        if r['pct_digits'] > CONFIG['max_pct_digits']:
            drop_stats['high_pct_digits'] += 1
            continue

        kept.append(r)

    return kept, dict(drop_stats)


def write_outputs(records: List[Dict], output_dir: Path) -> None:
    """Write JSONL and CSV outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = output_dir / 'enni_clean.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            # Convert qa_flags list to JSON-friendly format
            r_out = dict(r)
            f.write(json.dumps(r_out) + '\n')
    print(f"Wrote {len(records)} records to {jsonl_path}")

    # CSV
    csv_path = output_dir / 'enni_clean.csv'
    fieldnames = [
        'doc_id', 'file_id', 'story_id', 'group', 'age_years', 'age_months',
        'sex', 'adult_token_frac_pre', 'text', 'word_count', 'pct_markup',
        'pct_digits', 'n_chi_utts', 'n_total_utts', 'qa_flags', 'path_rel'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            r_out = dict(r)
            r_out['qa_flags'] = ';'.join(r['qa_flags'])  # Join for CSV
            writer.writerow(r_out)
    print(f"Wrote {len(records)} records to {csv_path}")


def print_summary(records: List[Dict], drop_stats: Dict) -> None:
    """Print summary statistics."""

    print("\n" + "=" * 70)
    print("ENNI INGESTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal records: {len(records)}")

    # Drop stats
    if drop_stats:
        print("\nDrop statistics:")
        for reason, count in sorted(drop_stats.items()):
            print(f"  {reason}: {count}")

    # By group
    print("\nBy group:")
    groups = defaultdict(list)
    for r in records:
        groups[r['group']].append(r)

    for group in sorted(groups.keys()):
        recs = groups[group]
        ages = [r['age_years'] for r in recs if r['age_years']]
        words = [r['word_count'] for r in recs]
        print(f"  {group}: n={len(recs)}", end='')
        if ages:
            print(f", age={min(ages):.1f}-{max(ages):.1f}", end='')
        print(f", words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")

    # By story
    print("\nBy story_id:")
    stories = defaultdict(list)
    for r in records:
        stories[r['story_id']].append(r)

    for story_id in sorted(stories.keys()):
        recs = stories[story_id]
        words = [r['word_count'] for r in recs]
        print(f"  {story_id}: n={len(recs)}, words={min(words)}-{max(words)} (median: {sorted(words)[len(words)//2]})")

    # QA flag summary
    print("\nQA flags:")
    flag_counts = defaultdict(int)
    for r in records:
        for flag in r['qa_flags']:
            flag_counts[flag.split(':')[0]] += 1

    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {count}")

    # Contamination stats
    print("\nContamination metrics:")
    pct_markups = [r['pct_markup'] for r in records]
    pct_digits = [r['pct_digits'] for r in records]
    print(f"  pct_markup: mean={sum(pct_markups)/len(pct_markups):.6f}, max={max(pct_markups):.6f}")
    print(f"  pct_digits: mean={sum(pct_digits)/len(pct_digits):.6f}, max={max(pct_digits):.6f}")
    print(f"  Records with pct_markup=0: {sum(1 for x in pct_markups if x == 0)} ({100*sum(1 for x in pct_markups if x == 0)/len(pct_markups):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='ENNI CHAT ingestion v2')
    parser.add_argument('--input-dir', type=Path, default=Path('data/ENNI'),
                        help='Input directory with .cha files')
    parser.add_argument('--output-dir', type=Path, default=Path('data/enni_clean'),
                        help='Output directory')
    parser.add_argument('--min-words', type=int, default=30,
                        help='Minimum word count (default: 30)')

    args = parser.parse_args()

    CONFIG['min_word_count'] = args.min_words

    # Find all .cha files
    cha_files = list(args.input_dir.rglob('*.cha'))
    print(f"Found {len(cha_files)} .cha files in {args.input_dir}")

    if not cha_files:
        print("No .cha files found!")
        return

    # Process all files
    all_records = []
    for filepath in sorted(cha_files):
        records = process_file(filepath, args.input_dir)
        all_records.extend(records)

    print(f"\nExtracted {len(all_records)} raw records")

    # Apply filters
    kept_records, drop_stats = apply_filters(all_records)

    # Write outputs
    write_outputs(kept_records, args.output_dir)

    # Print summary
    print_summary(kept_records, drop_stats)


if __name__ == '__main__':
    main()
