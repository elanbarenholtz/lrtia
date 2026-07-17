#!/usr/bin/env python3
"""
Buckeye Speech Corpus Ingestion Pipeline.

Parses the Buckeye .txt transcripts for analysis using LRTIA token-reveal
machinery. Extracts interviewee speech only (strips interviewer turns marked
with <IVER>).

The corpus contains conversational interviews with 40 speakers from Columbus, OH.
Each speaker has multiple recording segments (typically 5-11), which we concatenate
into one document per speaker.

Buckeye .txt format:
    - Running text with inline annotations
    - <IVER> marks interviewer turns (removed)
    - <SIL> silence
    - <VOCNOISE> vocal noise (throat clear, lip smack, etc.)
    - <NOISE> background noise
    - <LAUGH>, <LAUGH-word> laughter / laughing speech
    - <HES-word> hesitation (uh, um, etc.)
    - <EXT-word> extended/unclear pronunciation
    - <CUTOFF-word> truncated word
    - <EXCLUDE-name> anonymized personal info
    - Paragraphs are separated by newlines

Output: JSONL matching the format used by ECSC, Peterson-McCabe, and SBCSAE
ingestion scripts.

Usage:
    python scripts/ingest_buckeye.py
    python scripts/ingest_buckeye.py --min-words 200
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

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/Buckeye/extracted_txt")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/buckeye_processed")


def clean_utterance(text: str) -> str:
    """Clean Buckeye transcription conventions from a text chunk.

    Strips interviewer turns, paralinguistic annotations, and noise markers
    while preserving the interviewee's spoken words.
    """
    # Split on <IVER> to get interviewee chunks only.
    # Text between two <IVER> tags is interviewer speech — drop it.
    # The pattern is: interviewee text <IVER> interviewer text <IVER> interviewee text ...
    # But sometimes <IVER> just marks the boundary without clear alternation.
    # Safest approach: split on <IVER> and keep odd-indexed chunks (0-indexed),
    # since text starts with interviewee.
    # Actually, looking at the data more carefully: <IVER> appears at the END of
    # each interviewee chunk (marking where the interviewer speaks). The interviewer's
    # words aren't transcribed — <IVER> is just a boundary marker. So we just
    # strip <IVER> and keep everything else.

    # Remove <IVER> markers
    text = text.replace('<IVER>', ' ')

    # Remove silence/noise markers
    text = re.sub(r'<SIL>', ' ', text)
    text = re.sub(r'<VOCNOISE>', '', text)
    text = re.sub(r'<NOISE>', '', text)

    # Handle laughter: <LAUGH-word> -> keep word, <LAUGH> -> remove
    text = re.sub(r'<LAUGH-(\w+)>', r'\1', text)
    text = re.sub(r'<LAUGH>', '', text)

    # Handle hesitations: <HES-uh> -> uh (keep as natural speech)
    text = re.sub(r'<HES-(\w+)>', r'\1', text)

    # Handle extended forms: <EXT-word> -> word
    text = re.sub(r'<EXT-([^>]+)>', r'\1', text)

    # Handle cutoffs: <CUTOFF-word> -> remove (incomplete word)
    text = re.sub(r'<CUTOFF-[^>]*>', '', text)

    # Remove excluded/anonymized content
    text = re.sub(r'<EXCLUDE-[^>]*>', '', text)

    # Remove any remaining angle-bracket annotations
    text = re.sub(r'<[^>]+>', '', text)

    # Remove {B_TRANS} and {E_TRANS} markers if present
    text = re.sub(r'\{[^}]+\}', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_speaker_files(speaker_id: str, filepaths: list[Path]) -> dict:
    """Parse all segment files for one speaker and concatenate.

    Returns dict with speaker_id, segment texts, and combined text.
    """
    segments = []

    for filepath in sorted(filepaths):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw_text = f.read()

        clean_text = clean_utterance(raw_text)

        if clean_text:
            segments.append({
                'segment_id': filepath.stem,
                'raw_length': len(raw_text),
                'clean_text': clean_text,
                'word_count': len(clean_text.split()),
            })

    # Concatenate all segments in order
    full_text = ' '.join(seg['clean_text'] for seg in segments)

    return {
        'speaker_id': speaker_id,
        'n_segments': len(segments),
        'segments': segments,
        'full_text': full_text,
        'total_words': len(full_text.split()),
    }


def create_document(speaker_data: dict) -> dict:
    """Create an LRTIA-compatible document from a speaker's data."""
    population = {
        "dataset": "Buckeye",
        "speaker_id": speaker_data['speaker_id'],
        "n_segments": speaker_data['n_segments'],
        "mode": "interviewee_concatenated",
    }

    return {
        "doc_id": f"Buckeye_{speaker_data['speaker_id']}",
        "author_id": speaker_data['speaker_id'],
        "domain": "spoken_interview",
        "population": "spoken",
        "text": speaker_data['full_text'],
        "metadata": json.dumps(population),
    }


def main():
    parser = argparse.ArgumentParser(description="Buckeye Corpus Ingestion Pipeline")
    parser.add_argument("--min-words", type=int, default=200,
                        help="Minimum words per speaker document (default: 200)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BUCKEYE CORPUS INGESTION PIPELINE")
    logger.info("=" * 70)

    # Find all txt files and group by speaker
    txt_files = sorted(DATA_DIR.glob("s*.txt"))
    logger.info(f"Found {len(txt_files)} text files")

    # Group by speaker (first 3 chars: s01, s02, etc.)
    by_speaker = defaultdict(list)
    for f in txt_files:
        speaker_id = f.stem[:3]
        by_speaker[speaker_id].append(f)

    logger.info(f"Found {len(by_speaker)} speakers")

    # Parse each speaker
    all_speaker_data = []
    for speaker_id in sorted(by_speaker.keys()):
        data = parse_speaker_files(speaker_id, by_speaker[speaker_id])
        all_speaker_data.append(data)
        logger.debug(f"  {speaker_id}: {data['n_segments']} segments, "
                      f"{data['total_words']} words")

    # Create documents
    documents = []
    skipped = 0
    for speaker_data in all_speaker_data:
        if speaker_data['total_words'] >= args.min_words:
            doc = create_document(speaker_data)
            documents.append(doc)
        else:
            skipped += 1
            logger.warning(f"  Skipping {speaker_data['speaker_id']} "
                           f"({speaker_data['total_words']} words < {args.min_words})")

    # Also write per-segment documents (for within-speaker analyses)
    segment_docs = []
    for speaker_data in all_speaker_data:
        for seg in speaker_data['segments']:
            if seg['word_count'] >= 50:  # lower threshold for segments
                seg_pop = {
                    "dataset": "Buckeye",
                    "speaker_id": speaker_data['speaker_id'],
                    "segment_id": seg['segment_id'],
                    "mode": "per_segment",
                }
                segment_docs.append({
                    "doc_id": f"Buckeye_{seg['segment_id']}",
                    "author_id": speaker_data['speaker_id'],
                    "domain": "spoken_interview",
                    "population": "spoken",
                    "text": seg['clean_text'],
                    "metadata": json.dumps(seg_pop),
                })

    # Write outputs
    concat_path = output_dir / "speaker_concatenated.jsonl"
    with open(concat_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(documents)} speaker documents to {concat_path}")

    segment_path = output_dir / "per_segment.jsonl"
    with open(segment_path, 'w', encoding='utf-8') as f:
        for doc in segment_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(segment_docs)} segment documents to {segment_path}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nSpeakers processed: {len(all_speaker_data)}")
    logger.info(f"Speaker documents created: {len(documents)} (skipped {skipped})")
    logger.info(f"Segment documents created: {len(segment_docs)}")

    if documents:
        word_counts = [len(d['text'].split()) for d in documents]
        logger.info(f"\nPer-speaker word counts:")
        logger.info(f"  min={min(word_counts)}, max={max(word_counts)}, "
                     f"mean={sum(word_counts)/len(word_counts):.0f}, "
                     f"total={sum(word_counts)}")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "Buckeye Speech Corpus",
            "source": "Ohio State University",
            "url": "https://buckeyecorpus.osu.edu",
            "license": "Free for noncommercial use",
            "description": "Conversational interviews with speakers from Columbus, OH",
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_words_speaker": args.min_words,
            "min_words_segment": 50,
            "notes": "Interviewer turns (<IVER>) removed; all segments concatenated per speaker",
        },
        "counts": {
            "txt_files": len(txt_files),
            "speakers": len(by_speaker),
            "speaker_docs": len(documents),
            "segment_docs": len(segment_docs),
        },
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
