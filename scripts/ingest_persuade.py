#!/usr/bin/env python3
"""
ingest_persuade.py

PERSUADE 2.0 Corpus ingestion for LRTIA analysis.

Produces a cleaned JSONL file at the essay level with standardized fields
and Mistral tokenizer counts.

Usage:
    python scripts/ingest_persuade.py --input /path/to/persuade_corpus_2.0_train.csv
    python scripts/ingest_persuade.py --input /path/to/persuade.csv --output-dir data/persuade_clean
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

# Try to import transformers for tokenization
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: transformers not installed. Token counts will be estimated.")


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'model_name': 'mistralai/Mistral-7B-v0.1',
    'min_word_count': 50,  # Very loose filter for cleaning stage
}


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text: str) -> str:
    """Apply minimal normalization to essay text.

    Rules:
    - Normalize whitespace (collapse repeated spaces, unify newlines)
    - Remove control characters
    - Keep punctuation intact
    - Do NOT fix grammar/spelling
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove control characters (except newlines and tabs initially)
    # ASCII control: [\x00-\x08\x0B\x0C\x0E-\x1F\x7F]
    # C1 controls: [\u0080-\u009F]
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u0080-\u009F]', '', text)

    # Normalize various unicode spaces to regular space
    text = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]', ' ', text)

    # Collapse multiple newlines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final trim
    text = text.strip()

    return text


def compute_word_count(text: str) -> int:
    """Compute word count using simple whitespace split."""
    if not text:
        return 0
    return len(text.split())


# ============================================================
# ELL NORMALIZATION
# ============================================================

def normalize_ell(ell_value) -> str:
    """Normalize ELL status to standard values.

    Returns: 'ELL', 'non_ELL', or 'UNK'
    """
    if pd.isna(ell_value) or ell_value is None:
        return 'UNK'

    ell_str = str(ell_value).strip().lower()

    if ell_str in ('yes', 'y', '1', 'true', 'ell'):
        return 'ELL'
    elif ell_str in ('no', 'n', '0', 'false', 'non-ell', 'non_ell'):
        return 'non_ELL'
    elif ell_str in ('', ' ', 'na', 'n/a', 'nan', 'none', 'unknown', 'unk'):
        return 'UNK'
    else:
        return 'UNK'


# ============================================================
# MAIN PROCESSING
# ============================================================

def load_and_dedupe_essays(csv_path: Path) -> pd.DataFrame:
    """Load CSV and dedupe to essay level.

    The PERSUADE corpus has multiple rows per essay (discourse-level).
    We want one row per essay with the full_text.
    """
    print(f"Loading {csv_path}...")

    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows (discourse level)")

    # Check required columns
    required = ['essay_id_comp', 'full_text', 'holistic_essay_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Dedupe to essay level - take first occurrence of each essay
    # (all rows for same essay should have same full_text, score, etc.)
    essays = df.drop_duplicates(subset=['essay_id_comp'], keep='first').copy()
    print(f"  Deduped to {len(essays):,} unique essays")

    return essays


def process_essays(df: pd.DataFrame, tokenizer=None) -> list[dict]:
    """Process essays into standardized records."""

    records = []

    for idx, row in df.iterrows():
        # Extract fields
        essay_id = str(row.get('essay_id_comp', row.get('essay_id', f'essay_{idx}')))
        raw_text = row.get('full_text', '')

        # Skip if no text
        if pd.isna(raw_text) or not str(raw_text).strip():
            continue

        # Normalize text
        text = normalize_text(str(raw_text))

        # Skip empty after normalization
        if not text:
            continue

        # Extract score
        score = row.get('holistic_essay_score')
        if pd.notna(score):
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = None
        else:
            score = None

        # Extract grade
        grade = row.get('grade_level')
        if pd.notna(grade):
            try:
                grade = int(float(grade))
            except (ValueError, TypeError):
                grade = None
        else:
            grade = None

        # Normalize ELL status
        ell = normalize_ell(row.get('ell_status'))

        # Extract prompt
        prompt_id = row.get('prompt_name')
        if pd.isna(prompt_id):
            prompt_id = row.get('assignment', None)
        if pd.isna(prompt_id):
            prompt_id = None
        else:
            prompt_id = str(prompt_id).strip()

        # Compute word count
        word_count = compute_word_count(text)

        # Compute token count
        if tokenizer is not None:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(token_ids)
        else:
            # Estimate: ~1.3 tokens per word for English
            token_count = int(word_count * 1.3)

        # Build record
        record = {
            'essay_id': essay_id,
            'text': text,
            'score': score,
            'grade': grade,
            'ell': ell,
            'prompt_id': prompt_id,
            'word_count': word_count,
            'token_count': token_count,
        }

        # Optional: include source metadata
        if 'competition_set' in row and pd.notna(row['competition_set']):
            record['source_split'] = str(row['competition_set'])

        records.append(record)

    return records


def generate_report(records: list[dict], output_path: Path) -> str:
    """Generate validation report."""

    lines = []
    lines.append("=" * 70)
    lines.append("PERSUADE CORPUS INGESTION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # 1. Total records
    lines.append(f"Total essays: {len(records):,}")
    lines.append("")

    # 2. Missingness table
    lines.append("MISSINGNESS")
    lines.append("-" * 40)

    fields = ['score', 'grade', 'ell', 'prompt_id']
    for field in fields:
        if field == 'ell':
            missing = sum(1 for r in records if r[field] == 'UNK')
        else:
            missing = sum(1 for r in records if r[field] is None)
        pct = 100 * missing / len(records) if records else 0
        lines.append(f"  {field}: {missing:,} missing ({pct:.1f}%)")
    lines.append("")

    # 3. Distribution tables
    lines.append("SCORE DISTRIBUTION")
    lines.append("-" * 40)
    scores = Counter(r['score'] for r in records if r['score'] is not None)
    for score in sorted(scores.keys()):
        lines.append(f"  Score {score}: {scores[score]:,}")
    lines.append("")

    lines.append("GRADE DISTRIBUTION")
    lines.append("-" * 40)
    grades = Counter(r['grade'] for r in records if r['grade'] is not None)
    for grade in sorted(grades.keys()):
        lines.append(f"  Grade {grade}: {grades[grade]:,}")
    lines.append("")

    lines.append("ELL DISTRIBUTION")
    lines.append("-" * 40)
    ell_counts = Counter(r['ell'] for r in records)
    for ell in ['ELL', 'non_ELL', 'UNK']:
        if ell in ell_counts:
            lines.append(f"  {ell}: {ell_counts[ell]:,}")
    lines.append("")

    lines.append("PROMPT DISTRIBUTION (top 10)")
    lines.append("-" * 40)
    prompts = Counter(r['prompt_id'] for r in records if r['prompt_id'] is not None)
    for prompt, count in prompts.most_common(10):
        lines.append(f"  {prompt}: {count:,}")
    lines.append(f"  (Total unique prompts: {len(prompts)})")
    lines.append("")

    # 4. Length statistics
    lines.append("LENGTH STATISTICS")
    lines.append("-" * 40)

    word_counts = [r['word_count'] for r in records]
    token_counts = [r['token_count'] for r in records]

    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    lines.append("  Word counts:")
    lines.append(f"    Min:    {min(word_counts):,}")
    lines.append(f"    P10:    {percentile(word_counts, 10):,}")
    lines.append(f"    Median: {percentile(word_counts, 50):,}")
    lines.append(f"    Mean:   {sum(word_counts) / len(word_counts):,.1f}")
    lines.append(f"    P90:    {percentile(word_counts, 90):,}")
    lines.append(f"    Max:    {max(word_counts):,}")
    lines.append("")

    lines.append("  Token counts (Mistral):")
    lines.append(f"    Min:    {min(token_counts):,}")
    lines.append(f"    P10:    {percentile(token_counts, 10):,}")
    lines.append(f"    Median: {percentile(token_counts, 50):,}")
    lines.append(f"    Mean:   {sum(token_counts) / len(token_counts):,.1f}")
    lines.append(f"    P90:    {percentile(token_counts, 90):,}")
    lines.append(f"    Max:    {max(token_counts):,}")
    lines.append("")

    # 5. Filter threshold feasibility
    lines.append("FILTER THRESHOLD FEASIBILITY")
    lines.append("-" * 40)

    thresholds = [150, 200, 250, 300, 350]
    for thresh in thresholds:
        n_above = sum(1 for r in records if r['word_count'] >= thresh)
        pct = 100 * n_above / len(records) if records else 0
        lines.append(f"  word_count >= {thresh}: {n_above:,} ({pct:.1f}%)")
    lines.append("")

    token_thresholds = [250, 300, 350, 400, 450]
    for thresh in token_thresholds:
        n_above = sum(1 for r in records if r['token_count'] >= thresh)
        pct = 100 * n_above / len(records) if records else 0
        lines.append(f"  token_count >= {thresh}: {n_above:,} ({pct:.1f}%)")

    lines.append("")
    lines.append("=" * 70)

    report = '\n'.join(lines)

    # Print to console
    print(report)

    # Write to file
    with open(output_path, 'w') as f:
        f.write(report)

    return report


def write_outputs(records: list[dict], output_dir: Path) -> None:
    """Write JSONL and CSV outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL (primary output)
    jsonl_path = output_dir / 'persuade_clean.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nWrote {len(records):,} records to {jsonl_path}")

    # CSV (for convenience)
    csv_path = output_dir / 'persuade_clean.csv'
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(records):,} records to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='PERSUADE 2.0 Corpus Ingestion for LRTIA'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to persuade_corpus_2.0_train.csv'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/persuade_clean'),
        help='Output directory (default: data/persuade_clean)'
    )
    parser.add_argument(
        '--no-tokenize',
        action='store_true',
        help='Skip tokenization (use estimated token counts)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=CONFIG['model_name'],
        help=f'Tokenizer model name (default: {CONFIG["model_name"]})'
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load tokenizer
    tokenizer = None
    if not args.no_tokenize and HAS_TOKENIZER:
        print(f"Loading tokenizer: {args.model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")
            print("  Using estimated token counts.")
    elif not HAS_TOKENIZER:
        print("Tokenizer not available. Using estimated token counts.")

    # Load and process
    essays_df = load_and_dedupe_essays(args.input)

    print("Processing essays...")
    records = process_essays(essays_df, tokenizer)
    print(f"  Processed {len(records):,} valid essays")

    # Write outputs
    write_outputs(records, args.output_dir)

    # Generate report
    report_path = args.output_dir / 'persuade_ingest_report.txt'
    generate_report(records, report_path)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
