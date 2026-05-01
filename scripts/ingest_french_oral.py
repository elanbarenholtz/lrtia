#!/usr/bin/env python3
"""
French Oral Narrative Corpus Ingestion Pipeline.

Parses TEI P5 XML transcripts of French oral storytelling for analysis
using LRTIA token-reveal machinery.

87 stories from 18 storytellers, recorded live. True monologues (oral narrative).
This is the cross-linguistic test: does the power law decay exponent hold
for French as well as English?

TEI structure:
    <body> contains <div type="performance"> sections
    <u> elements contain utterance text with inline annotations:
        <seg ana="..."> — linguistic annotations (keep text, strip tags)
        <vocal><desc>...</desc></vocal> — paralinguistic events (remove)
        <incident><desc>...</desc></incident> — non-verbal events (remove)
        <pause/> — pauses (remove)

Output: JSONL matching the format used by Buckeye, ECSC, etc.

Usage:
    python scripts/ingest_french_oral.py
    python scripts/ingest_french_oral.py --min-words 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/french_oral/TEIP5")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/french_oral_processed")

# TEI namespace
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


def extract_text_recursive(element):
    """Extract all text content from an element, stripping tags but keeping text.

    Skips <vocal> and <incident> elements entirely.
    """
    parts = []

    # Skip vocal/incident/pause elements
    tag = element.tag.replace(f'{{{NS["tei"]}}}', '')
    if tag in ('vocal', 'incident', 'pause', 'kinesic'):
        return ''

    # Get this element's direct text
    if element.text:
        parts.append(element.text)

    # Recurse into children
    for child in element:
        parts.append(extract_text_recursive(child))
        # Get tail text (text after the child's closing tag)
        if child.tail:
            parts.append(child.tail)

    return ''.join(parts)


def clean_french_text(text: str) -> str:
    """Clean extracted French text."""
    # Remove parenthetical stage directions like (il)
    text = re.sub(r'\([^)]*\)', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_tei_file(filepath: Path) -> dict:
    """Parse a single TEI P5 XML file and extract story metadata + text."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    file_id = filepath.stem

    # Extract metadata from header
    title_el = root.find('.//tei:titleStmt/tei:title', NS)
    title = title_el.text.strip() if title_el is not None and title_el.text else file_id

    author_el = root.find('.//tei:titleStmt/tei:author', NS)
    storyteller = author_el.text.strip() if author_el is not None and author_el.text else "Unknown"

    # Extract recording date
    date_el = root.find('.//tei:recording/tei:date', NS)
    date = date_el.get('when', '') if date_el is not None else ''

    # Extract story type from textClass
    story_types = []
    for catref in root.findall('.//tei:textClass//tei:catRef', NS):
        target = catref.get('target', '')
        if target:
            story_types.append(target.lstrip('#'))

    # Extract all utterance text from body
    body = root.find('.//tei:body', NS)
    if body is None:
        return None

    utterances = []
    for u in body.iter(f'{{{NS["tei"]}}}u'):
        text = extract_text_recursive(u)
        clean = clean_french_text(text)
        if clean:
            utterances.append(clean)

    full_text = ' '.join(utterances)

    # Derive storyteller ID from filename (first part before underscore)
    storyteller_id = file_id.split('_')[0]

    return {
        'file_id': file_id,
        'title': title,
        'storyteller': storyteller,
        'storyteller_id': storyteller_id,
        'date': date,
        'story_types': story_types,
        'text': full_text,
        'word_count': len(full_text.split()),
        'n_utterances': len(utterances),
    }


def main():
    parser = argparse.ArgumentParser(description="French Oral Narrative Ingestion Pipeline")
    parser.add_argument("--min-words", type=int, default=100,
                        help="Minimum words per story (default: 100)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FRENCH ORAL NARRATIVE INGESTION PIPELINE")
    logger.info("=" * 70)

    # Parse all TEI files
    xml_files = sorted(DATA_DIR.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} TEI P5 XML files")

    all_stories = []
    for filepath in xml_files:
        try:
            story = parse_tei_file(filepath)
            if story is not None:
                all_stories.append(story)
        except ET.ParseError as e:
            logger.warning(f"  Failed to parse {filepath.name}: {e}")

    logger.info(f"Parsed {len(all_stories)} stories")

    # Create per-story documents
    documents = []
    skipped = 0
    for story in all_stories:
        if story['word_count'] >= args.min_words:
            metadata = {
                "dataset": "FrenchOralNarrative",
                "file_id": story['file_id'],
                "title": story['title'],
                "storyteller": story['storyteller'],
                "storyteller_id": story['storyteller_id'],
                "date": story['date'],
                "story_types": story['story_types'],
                "language": "fr",
                "mode": "per_story",
            }
            documents.append({
                "doc_id": f"FrOral_{story['file_id']}",
                "author_id": story['storyteller_id'],
                "domain": "oral_narrative",
                "population": "spoken_french",
                "text": story['text'],
                "metadata": json.dumps(metadata, ensure_ascii=False),
            })
        else:
            skipped += 1

    # Also create per-storyteller concatenated documents
    by_teller = defaultdict(list)
    for doc in documents:
        by_teller[doc['author_id']].append(doc)

    concat_docs = []
    for teller_id, docs in sorted(by_teller.items()):
        concat_text = ' '.join(d['text'] for d in docs)
        first_meta = json.loads(docs[0]['metadata'])
        concat_docs.append({
            "doc_id": f"FrOral_{teller_id}_concat",
            "author_id": teller_id,
            "domain": "oral_narrative",
            "population": "spoken_french",
            "text": concat_text,
            "metadata": json.dumps({
                "dataset": "FrenchOralNarrative",
                "storyteller_id": teller_id,
                "storyteller": first_meta['storyteller'],
                "n_stories": len(docs),
                "language": "fr",
                "mode": "per_storyteller_concat",
            }, ensure_ascii=False),
        })

    # Write outputs
    story_path = output_dir / "per_story.jsonl"
    with open(story_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(documents)} story documents to {story_path}")

    concat_path = output_dir / "per_storyteller.jsonl"
    with open(concat_path, 'w', encoding='utf-8') as f:
        for doc in concat_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(concat_docs)} storyteller documents to {concat_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nStories parsed: {len(all_stories)}")
    logger.info(f"Stories kept (>={args.min_words} words): {len(documents)}")
    logger.info(f"Stories skipped (too short): {skipped}")
    logger.info(f"Unique storytellers: {len(by_teller)}")

    if documents:
        word_counts = [len(d['text'].split()) for d in documents]
        logger.info(f"\nPer-story word counts:")
        logger.info(f"  min={min(word_counts)}, max={max(word_counts)}, "
                     f"mean={sum(word_counts)/len(word_counts):.0f}, "
                     f"total={sum(word_counts):,}")

    if concat_docs:
        word_counts = [len(d['text'].split()) for d in concat_docs]
        logger.info(f"\nPer-storyteller word counts:")
        logger.info(f"  min={min(word_counts)}, max={max(word_counts)}, "
                     f"mean={sum(word_counts)/len(word_counts):.0f}")

    # Stories per storyteller
    logger.info("\nStories per storyteller:")
    for teller_id, docs in sorted(by_teller.items()):
        total_words = sum(len(d['text'].split()) for d in docs)
        logger.info(f"  {teller_id}: {len(docs)} stories, {total_words:,} words")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "French Oral Narrative Corpus",
            "source": "Queen's University Belfast / CLIO",
            "url": "https://frenchoralnarrative.qub.ac.uk",
            "archive": "https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2555",
            "license": "CC BY-NC-SA 3.0",
            "language": "French",
            "description": "87 oral stories from 18 French storytellers, recorded live",
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_words": args.min_words,
            "format_parsed": "TEI P5 XML",
        },
        "counts": {
            "xml_files": len(xml_files),
            "stories_parsed": len(all_stories),
            "stories_kept": len(documents),
            "stories_skipped": skipped,
            "storytellers": len(by_teller),
            "total_words": sum(len(d['text'].split()) for d in documents),
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
