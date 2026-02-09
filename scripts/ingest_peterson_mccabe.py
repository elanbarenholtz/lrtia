#!/usr/bin/env python3
"""
Peterson & McCabe Corpus Ingestion Pipeline.

Parses the Peterson & McCabe narrative corpus from CHILDES for developmental
analysis using LRTIA memory curve / local-lag machinery.

The corpus contains personal narratives from children ages 4-9, elicited through
conversational interaction. Each file contains multiple @G: Narrative blocks.

Output format matches ECSC for consistency with existing analysis pipelines.

Usage:
    python scripts/ingest_peterson_mccabe.py
    python scripts/ingest_peterson_mccabe.py --min-words 30
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/PetersonMcCabe")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/petmcc_processed")


def parse_age(age_str: str) -> int:
    """Convert age string like '4;00.' to months."""
    age_str = age_str.rstrip('.')
    match = re.match(r'(\d+);(\d+)', age_str)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return None


def clean_utterance(text: str) -> str:
    """Clean CHAT transcription conventions from utterance text.

    Removes:
    - [/], [//] - retracing markers
    - [*...] - error markers
    - [=!...] - paralinguistic markers
    - [=...] - explanations
    - <...> - overlap/repetition markers
    - xxx, yyy - unintelligible
    - &~xxx - phonological fragments
    - &-xxx - filler syllables (keep the word though)
    - +... - trailing off
    - (.) (..) - pauses
    """
    # Remove bracket annotations
    text = re.sub(r'\[/+\]', '', text)
    text = re.sub(r'\[\*[^\]]*\]', '', text)
    text = re.sub(r'\[=![^\]]*\]', '', text)
    text = re.sub(r'\[=[^\]]*\]', '', text)
    text = re.sub(r'\[\?[^\]]*\]', '', text)
    text = re.sub(r'\[%[^\]]*\]', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)  # catch remaining brackets

    # Remove angle brackets (overlaps, repetitions)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove unintelligible markers
    text = re.sub(r'\bxxx\b', '', text)
    text = re.sub(r'\byyy\b', '', text)
    text = re.sub(r'\bwww\b', '', text)

    # Remove phonological fragments (&~xxx) but keep filler syllables (&-um -> um)
    text = re.sub(r'&~\w+', '', text)
    text = re.sub(r'&-(\w+)', r'\1', text)
    text = re.sub(r'&\w+', '', text)  # catch remaining

    # Remove trailing off marker
    text = re.sub(r'\+\.\.\.?', '', text)
    text = re.sub(r'\+/\.?', '', text)
    text = re.sub(r'\+\"\.?', '', text)
    text = re.sub(r'\+,', '', text)

    # Remove pause markers
    text = re.sub(r'\(\.\.\.\)', '', text)
    text = re.sub(r'\(\.\.\)', '', text)
    text = re.sub(r'\(\.\)', '', text)

    # Remove special punctuation
    text = re.sub(r'[‡„]', '', text)

    # Normalize possessives with ~ (e.g., mom~s -> mom's)
    text = re.sub(r'~s\b', "'s", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove isolated punctuation at start
    text = re.sub(r'^[.,;:!?]+\s*', '', text)

    return text


def parse_cha_file(filepath: Path) -> dict:
    """Parse a single .cha file and extract metadata + narratives."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    result = {
        'filename': filepath.name,
        'file_stem': filepath.stem,
        'age_months': None,
        'age_str': None,
        'sex': None,
        'file_types': [],
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

    # Extract file-level @Types
    types_matches = re.findall(r'@Types:\s*(.+)', content)
    all_types = set()
    for types_str in types_matches:
        for t in types_str.split(','):
            all_types.add(t.strip())
    result['file_types'] = list(all_types)

    # Split by @G: Narrative markers
    lines = content.split('\n')
    current_narrative = None
    current_narrative_idx = 0
    current_types = []
    current_chi_utterances = []
    current_raw_utterances = []

    for line in lines:
        # Check for narrative marker
        narrative_match = re.match(r'@G:\s*Narrative\s*(\d+)', line)
        if narrative_match:
            # Save previous narrative if exists
            if current_narrative is not None and current_chi_utterances:
                clean_text = ' '.join(current_chi_utterances)
                result['narratives'].append({
                    'narrative_idx': current_narrative_idx,
                    'types': current_types.copy(),
                    'chi_utterances': current_chi_utterances.copy(),
                    'raw_utterances': current_raw_utterances.copy(),
                    'clean_text': clean_text,
                    'word_count': len(clean_text.split())
                })
            # Start new narrative
            current_narrative_idx = int(narrative_match.group(1))
            current_narrative = current_narrative_idx
            current_chi_utterances = []
            current_raw_utterances = []
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
            raw_utt = chi_match.group(1)
            current_raw_utterances.append(raw_utt)

            # Clean the utterance
            clean_utt = clean_utterance(raw_utt)
            if clean_utt:
                current_chi_utterances.append(clean_utt)

    # Save last narrative
    if current_narrative is not None and current_chi_utterances:
        clean_text = ' '.join(current_chi_utterances)
        result['narratives'].append({
            'narrative_idx': current_narrative_idx,
            'types': current_types.copy(),
            'chi_utterances': current_chi_utterances.copy(),
            'raw_utterances': current_raw_utterances.copy(),
            'clean_text': clean_text,
            'word_count': len(clean_text.split())
        })

    return result


def create_document(file_data: dict, narrative: dict) -> dict:
    """Create an LRTIA-compatible document from a narrative."""
    # Build population metadata as JSON string (matching ECSC format)
    population = {
        "dataset": "PetersonMcCabe",
        "subject_id": file_data['file_stem'],
        "age_months": file_data['age_months'],
        "sex": file_data['sex'],
        "narrative_idx": narrative['narrative_idx'],
        "types": narrative['types'],
        "other_meta": {}
    }

    doc_id = f"PetMcc_{file_data['file_stem']}_N{narrative['narrative_idx']}"

    return {
        "doc_id": doc_id,
        "author_id": file_data['file_stem'],
        "population": json.dumps(population),
        "text": narrative['clean_text']
    }


def main():
    parser = argparse.ArgumentParser(description="Peterson & McCabe Ingestion")
    parser.add_argument("--min-words", type=int, default=20,
                       help="Minimum words per narrative (default: 20)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                       help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PETERSON & MCCABE INGESTION PIPELINE")
    logger.info("=" * 70)

    # Parse all .cha files
    cha_files = sorted(DATA_DIR.glob("*.cha"))
    logger.info(f"Found {len(cha_files)} .cha files")

    all_file_data = []
    for filepath in cha_files:
        data = parse_cha_file(filepath)
        all_file_data.append(data)

    # Create documents
    documents = []
    skipped_short = 0

    for file_data in all_file_data:
        for narrative in file_data['narratives']:
            if narrative['word_count'] >= args.min_words:
                doc = create_document(file_data, narrative)
                documents.append(doc)
            else:
                skipped_short += 1

    logger.info(f"Created {len(documents)} documents (skipped {skipped_short} as too short)")

    # Write transcripts.jsonl (one narrative per document)
    transcripts_path = output_dir / "transcripts.jsonl"
    with open(transcripts_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {transcripts_path}")

    # Also create concatenated docs (all narratives per child concatenated)
    by_author = defaultdict(list)
    for doc in documents:
        by_author[doc['author_id']].append(doc)

    concat_documents = []
    for author_id, author_docs in by_author.items():
        # Sort by narrative index
        author_docs_sorted = sorted(author_docs,
            key=lambda d: json.loads(d['population'])['narrative_idx'])

        # Get metadata from first doc
        first_pop = json.loads(author_docs_sorted[0]['population'])

        concat_text = ' '.join(d['text'] for d in author_docs_sorted)

        concat_pop = {
            "dataset": "PetersonMcCabe",
            "subject_id": author_id,
            "age_months": first_pop['age_months'],
            "sex": first_pop['sex'],
            "n_narratives": len(author_docs_sorted),
            "other_meta": {}
        }

        concat_documents.append({
            "doc_id": f"PetMcc_{author_id}_concatenated",
            "author_id": author_id,
            "population": json.dumps(concat_pop),
            "text": concat_text
        })

    concat_path = output_dir / "concatenated_docs.jsonl"
    with open(concat_path, 'w', encoding='utf-8') as f:
        for doc in concat_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {concat_path}")

    # Create summary statistics
    df = pd.DataFrame([
        {
            'doc_id': d['doc_id'],
            'author_id': d['author_id'],
            'age_months': json.loads(d['population'])['age_months'],
            'word_count': len(d['text'].split())
        }
        for d in documents
    ])

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)

    logger.info(f"\nTotal narratives: {len(df)}")
    logger.info(f"Unique children: {df['author_id'].nunique()}")

    # Age distribution
    df['age_years'] = df['age_months'] // 12
    age_counts = df.groupby('age_years').size()
    logger.info("\nNarratives by age year:")
    for age, count in age_counts.items():
        logger.info(f"  {age}-year-olds: {count}")

    # Word count stats
    logger.info(f"\nWord count: mean={df['word_count'].mean():.1f}, "
                f"median={df['word_count'].median():.1f}, "
                f"min={df['word_count'].min()}, max={df['word_count'].max()}")

    # Word count by age
    wc_by_age = df.groupby('age_years')['word_count'].mean()
    logger.info("\nMean word count by age:")
    for age, wc in wc_by_age.items():
        logger.info(f"  {age}-year-olds: {wc:.1f} words")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "Peterson & McCabe Narrative Corpus",
            "source": "TalkBank CHILDES",
            "doi": "doi:10.21415/T5G886",
            "citation": "Peterson & McCabe - see CHILDES documentation"
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_words": args.min_words,
        },
        "outputs": {
            "transcripts_jsonl": str(transcripts_path),
            "concatenated_jsonl": str(concat_path),
        },
        "counts": {
            "files_processed": len(all_file_data),
            "narratives_total": sum(len(f['narratives']) for f in all_file_data),
            "narratives_kept": len(documents),
            "narratives_skipped_short": skipped_short,
            "unique_children": df['author_id'].nunique(),
        },
        "age_distribution": {
            f"{int(age)}yo": int(count) for age, count in age_counts.items()
        }
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"\nWrote {manifest_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
