#!/usr/bin/env python3
"""
Santa Barbara Corpus of Spoken American English (SBCSAE) Ingestion Pipeline.

Parses the 60 TRN-format transcripts from the SBCSAE corpus for analysis
using LRTIA token-reveal / sentence-contrast machinery.

The corpus contains naturally occurring spoken American English across diverse
interaction types: face-to-face conversations, telephone calls, lectures,
sermons, medical consultations, classroom interactions, etc.

TRN format:
    start_time end_time [TAB] SPEAKER: [TAB] utterance text
    (continuation lines have blank speaker field)

Transcription conventions stripped:
    (H), (Hx)     - breaths/exhales
    @, <@ ... @>   - laughter / laughing speech
    ...            - pauses
    =              - lengthening
    --             - truncation
    %              - false start / retracing
    [text]         - overlapping speech
    <X text X>     - uncertain hearing
    XX, XXX        - unintelligible
    ~Name          - proper noun marker (keep the name)
    (TSK)          - tongue click
    <YWN ... YWN>  - yawning speech
    <MRC ... MRC>  - marcato (emphatic) speech
    >ENV:          - environmental sounds (removed entirely)

Output: JSONL with one document per recording, matching the format used by
ECSC and Peterson-McCabe ingestion scripts.

Two output modes:
    1. full_conversation: all speakers concatenated in temporal order
    2. per_speaker: one document per speaker per recording (for single-speaker analyses)

Usage:
    python scripts/ingest_sbcsae.py
    python scripts/ingest_sbcsae.py --min-words 100 --mode per_speaker
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/sbcsae/transcripts/TRN")
METADATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/sbcsae/metadata")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/sbcsae_processed")

# Recording titles from the SBCSAE website
RECORDING_TITLES = {
    "SBC001": "Actual Blacksmithing",
    "SBC002": "Lambada",
    "SBC003": "Conceptual Pesticides",
    "SBC004": "Raging Bureaucracy",
    "SBC005": "A Book about Death",
    "SBC006": "Cuz",
    "SBC007": "A Tree's Life",
    "SBC008": "Tell the Jury That",
    "SBC009": "Zero Equals Zero",
    "SBC010": "Letter of Concerns",
    "SBC011": "This Retirement Bit",
    "SBC012": "American Democracy is Dying",
    "SBC013": "Appease the Monster",
    "SBC014": "Bank Products",
    "SBC015": "Deadly Diseases",
    "SBC016": "Tape Deck",
    "SBC017": "Wonderful Abstract Notions",
    "SBC018": "Vet Morning",
    "SBC019": "Doesn't Work in this Household",
    "SBC020": "God's Love",
    "SBC021": "Fear",
    "SBC022": "Runway Heading",
    "SBC023": "Howard's End",
    "SBC024": "Risk",
    "SBC025": "The Egg which Luther Hatched",
    "SBC026": "Hundred Million Dollars",
    "SBC027": "Atoms Hanging Out",
    "SBC028": "Hey Cutie Pie",
    "SBC029": "Ancient Furnace",
    "SBC030": "Vision",
    "SBC031": "Tastes Very Special",
    "SBC032": "Handshakes All Around",
    "SBC033": "Guilt",
    "SBC034": "What Time is it Now?",
    "SBC035": "Hold my Breath",
    "SBC036": "Judgmental on People",
    "SBC037": "Very Good Tamales",
    "SBC038": "Good Strong Dam",
    "SBC039": "Pretty Busy Bird",
    "SBC040": "Beaten on a Regular Basis",
    "SBC041": "X Units of Insulin",
    "SBC042": "Stay out of It",
    "SBC043": "Try a Couple Spoonfuls",
    "SBC044": "He Knows",
    "SBC045": "The Classic Hooker",
    "SBC046": "Flumpity-Bump Down the Hill",
    "SBC047": "On the Lot",
    "SBC048": "Mickey Mouse Watch",
    "SBC049": "Noise Pollution",
    "SBC050": "Just Wanna Hang",
    "SBC051": "New Yorkers Anonymous",
    "SBC052": "Oh You Need a Breadbox",
    "SBC053": "I Will Appeal",
    "SBC054": "'That's Good', Said Tiger",
    "SBC055": "The Mama of Dada",
    "SBC056": "What is a Brand Inspection?",
    "SBC057": "Throw Me",
    "SBC058": "Swingin' Kid",
    "SBC059": "You Baked",
    "SBC060": "Shaggy Dog Story",
}


def clean_utterance(text: str) -> str:
    """Clean SBCSAE transcription conventions from utterance text.

    Strips paralinguistic annotations while preserving the spoken words.
    """
    # Remove environmental sound lines entirely (handled at line level, but safety check)
    if text.strip().startswith("((") and text.strip().endswith("))"):
        return ""

    # Remove breath/vocalization markers
    text = re.sub(r'\(H[x]?\)', '', text)
    text = re.sub(r'\(Hx?\)', '', text)
    text = re.sub(r'\(TSK\)', '', text)

    # Remove yawn markers but keep the words inside
    text = re.sub(r'<YWN\s*', '', text)
    text = re.sub(r'\s*YWN>', '', text)

    # Remove marcato markers but keep words
    text = re.sub(r'<MRC\s*', '', text)
    text = re.sub(r'\s*MRC>', '', text)

    # Remove laughing-speech markers but keep words
    text = re.sub(r'<@\s*', '', text)
    text = re.sub(r'\s*@>', '', text)

    # Remove standalone laughter
    text = re.sub(r'\b@+\b', '', text)
    text = re.sub(r'^@+', '', text)

    # Remove uncertain-hearing markers but keep words
    text = re.sub(r'<X\s*', '', text)
    text = re.sub(r'\s*X>', '', text)

    # Remove overlap brackets but keep words inside
    # Numbered overlaps like [2text2], [3text3]
    text = re.sub(r'\[(\d+)(.*?)\1\]', r'\2', text)
    # Simple overlaps [text]
    text = re.sub(r'\[(.*?)\]', r'\1', text)

    # Remove unintelligible markers
    text = re.sub(r'\bXX+\b', '', text)

    # Remove proper noun marker ~ but keep the name
    text = re.sub(r'~(\w)', r'\1', text)

    # Remove lengthening marker
    text = text.replace('=', '')

    # Remove truncation marker
    text = re.sub(r'\s*--\s*', ' ', text)

    # Remove false start / retracing marker
    text = text.replace('%', '')

    # Remove pause markers
    text = re.sub(r'\.\.\.+', '', text)
    text = re.sub(r'\.\.', '', text)

    # Remove parenthetical annotations like ((COUGH)), ((LAUGHTER))
    text = re.sub(r'\(\([^)]*\)\)', '', text)

    # Remove remaining isolated parenthetical markers
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove 0-prefixed annotations (0m, 0you, etc. — false starts with specific targets)
    text = re.sub(r'\b0\w+', '', text)

    # Remove asterisks (used for special annotations)
    text = text.replace('*', '')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove isolated punctuation at start/end
    text = re.sub(r'^[,;:\s]+', '', text)
    text = re.sub(r'[,;:\s]+$', '', text)

    return text


def parse_trn_file(filepath: Path) -> dict:
    """Parse a single .trn file into structured utterances.

    Returns dict with:
        - file_id: e.g. "SBC001"
        - utterances: list of {speaker, start, end, text, raw_text}
        - speakers: set of unique speakers
    """
    file_id = filepath.stem
    utterances = []
    current_speaker = None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line.strip():
                continue

            # Parse TRN format: start_time end_time [TAB] SPEAKER: [TAB] text
            # or continuation: start_time end_time [TAB] (whitespace) [TAB] text
            # Some files use tab separators, some use spaces
            match = re.match(
                r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\S+:)?\s*(.*)',
                line
            )
            if not match:
                continue

            start = float(match.group(1))
            end = float(match.group(2))
            speaker_field = match.group(3)
            raw_text = match.group(4).strip()

            # Skip environmental sound lines
            if speaker_field and speaker_field.startswith('>ENV'):
                continue
            if not speaker_field and raw_text.startswith('(('):
                continue

            # Update speaker if present
            if speaker_field:
                current_speaker = speaker_field.rstrip(':').strip()

            if not current_speaker:
                continue

            clean_text = clean_utterance(raw_text)
            if not clean_text:
                continue

            utterances.append({
                'speaker': current_speaker,
                'start': start,
                'end': end,
                'text': clean_text,
                'raw_text': raw_text,
            })

    speakers = sorted(set(u['speaker'] for u in utterances))

    return {
        'file_id': file_id,
        'title': RECORDING_TITLES.get(file_id, "Unknown"),
        'utterances': utterances,
        'speakers': speakers,
        'n_speakers': len(speakers),
    }


def classify_interaction_type(parsed: dict) -> str:
    """Classify the interaction type based on speaker count and patterns."""
    n = parsed['n_speakers']
    if n == 1:
        return "monologue"
    elif n == 2:
        return "dyad"
    elif n <= 4:
        return "small_group"
    else:
        return "large_group"


def build_full_conversation_doc(parsed: dict) -> dict:
    """Build a single document from all speakers in temporal order."""
    # Utterances are already in temporal order from the file
    text_parts = [u['text'] for u in parsed['utterances']]
    full_text = ' '.join(text_parts)

    interaction_type = classify_interaction_type(parsed)

    population = {
        "dataset": "SBCSAE",
        "file_id": parsed['file_id'],
        "title": parsed['title'],
        "interaction_type": interaction_type,
        "n_speakers": parsed['n_speakers'],
        "speakers": parsed['speakers'],
        "mode": "full_conversation",
    }

    return {
        "doc_id": f"SBCSAE_{parsed['file_id']}",
        "author_id": parsed['file_id'],
        "domain": interaction_type,
        "population": "spoken",
        "text": full_text,
        "metadata": json.dumps(population),
    }


def build_per_speaker_docs(parsed: dict, min_words: int) -> list[dict]:
    """Build one document per speaker (only speakers with enough text)."""
    docs = []
    interaction_type = classify_interaction_type(parsed)

    for speaker in parsed['speakers']:
        speaker_utts = [u['text'] for u in parsed['utterances'] if u['speaker'] == speaker]
        full_text = ' '.join(speaker_utts)
        word_count = len(full_text.split())

        if word_count < min_words:
            continue

        population = {
            "dataset": "SBCSAE",
            "file_id": parsed['file_id'],
            "title": parsed['title'],
            "interaction_type": interaction_type,
            "n_speakers": parsed['n_speakers'],
            "speaker": speaker,
            "mode": "per_speaker",
        }

        docs.append({
            "doc_id": f"SBCSAE_{parsed['file_id']}_{speaker}",
            "author_id": speaker,
            "domain": interaction_type,
            "population": "spoken",
            "text": full_text,
            "metadata": json.dumps(population),
        })

    return docs


def main():
    parser = argparse.ArgumentParser(description="SBCSAE Ingestion Pipeline")
    parser.add_argument("--min-words", type=int, default=100,
                        help="Minimum words per document (default: 100)")
    parser.add_argument("--mode", choices=["full_conversation", "per_speaker", "both"],
                        default="both", help="Output mode (default: both)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SBCSAE INGESTION PIPELINE")
    logger.info("=" * 70)

    # Parse all TRN files
    trn_files = sorted(DATA_DIR.glob("SBC*.trn"))
    logger.info(f"Found {len(trn_files)} TRN files")

    all_parsed = []
    for filepath in trn_files:
        parsed = parse_trn_file(filepath)
        all_parsed.append(parsed)
        logger.debug(f"  {parsed['file_id']}: {len(parsed['utterances'])} utterances, "
                      f"{parsed['n_speakers']} speakers ({parsed['title']})")

    # Build documents
    full_docs = []
    speaker_docs = []

    for parsed in all_parsed:
        if args.mode in ("full_conversation", "both"):
            doc = build_full_conversation_doc(parsed)
            word_count = len(doc['text'].split())
            if word_count >= args.min_words:
                full_docs.append(doc)
            else:
                logger.warning(f"  Skipping {parsed['file_id']} full conversation "
                               f"({word_count} words < {args.min_words})")

        if args.mode in ("per_speaker", "both"):
            docs = build_per_speaker_docs(parsed, args.min_words)
            speaker_docs.extend(docs)

    # Write outputs
    if full_docs:
        path = output_dir / "full_conversation.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for doc in full_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(full_docs)} full-conversation documents to {path}")

    if speaker_docs:
        path = output_dir / "per_speaker.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for doc in speaker_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(speaker_docs)} per-speaker documents to {path}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nFiles parsed: {len(all_parsed)}")

    # Interaction type breakdown
    type_counts = {}
    for p in all_parsed:
        t = classify_interaction_type(p)
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("\nInteraction types:")
    for t, c in sorted(type_counts.items()):
        logger.info(f"  {t}: {c}")

    # Speaker count distribution
    speaker_counts = [p['n_speakers'] for p in all_parsed]
    logger.info(f"\nSpeakers per recording: min={min(speaker_counts)}, "
                f"max={max(speaker_counts)}, mean={sum(speaker_counts)/len(speaker_counts):.1f}")

    if full_docs:
        word_counts = [len(d['text'].split()) for d in full_docs]
        logger.info(f"\nFull conversation docs: {len(full_docs)}")
        logger.info(f"  Word count: min={min(word_counts)}, max={max(word_counts)}, "
                     f"mean={sum(word_counts)/len(word_counts):.0f}")

    if speaker_docs:
        word_counts = [len(d['text'].split()) for d in speaker_docs]
        logger.info(f"\nPer-speaker docs: {len(speaker_docs)}")
        logger.info(f"  Word count: min={min(word_counts)}, max={max(word_counts)}, "
                     f"mean={sum(word_counts)/len(word_counts):.0f}")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "Santa Barbara Corpus of Spoken American English",
            "source": "UC Santa Barbara Linguistics",
            "url": "https://linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english",
            "license": "CC BY-ND 3.0 US",
            "n_recordings": 60,
            "description": "Naturally occurring spoken American English across diverse interaction types",
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_words": args.min_words,
            "mode": args.mode,
        },
        "counts": {
            "files_parsed": len(all_parsed),
            "full_conversation_docs": len(full_docs),
            "per_speaker_docs": len(speaker_docs),
            "interaction_types": type_counts,
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
