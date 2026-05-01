#!/usr/bin/env python3
"""
MLSUM News Articles Ingestion Pipeline.

Pulls news articles from MLSUM (HuggingFace) for cross-genre comparison.
Available languages: French, Turkish, German, Spanish, Russian.

We focus on French and Turkish (where we have Wiki + spoken data)
plus German as a third language.

Usage:
    python scripts/ingest_mlsum_news.py
    python scripts/ingest_mlsum_news.py --languages fr tr de
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/mlsum_news")

# XL-Sum language codes
LANGUAGES = {
    'en': {'mlsum_code': 'english', 'name': 'English'},
    'fr': {'mlsum_code': 'french', 'name': 'French'},
    'tr': {'mlsum_code': 'turkish', 'name': 'Turkish'},
    'zh': {'mlsum_code': 'chinese_simplified', 'name': 'Chinese'},
    'ja': {'mlsum_code': 'japanese', 'name': 'Japanese'},
    'ko': {'mlsum_code': 'korean', 'name': 'Korean'},
    'ar': {'mlsum_code': 'arabic', 'name': 'Arabic'},
}

MIN_CHARS = 3000
ARTICLES_PER_LANG = 60


def main():
    parser = argparse.ArgumentParser(description="MLSUM News Ingestion")
    parser.add_argument("--languages", nargs="+", default=['fr', 'tr', 'de'])
    parser.add_argument("--articles-per-lang", type=int, default=ARTICLES_PER_LANG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MLSUM NEWS INGESTION")
    logger.info("=" * 70)

    for lang_code in args.languages:
        if lang_code not in LANGUAGES:
            logger.warning(f"Unknown language: {lang_code}")
            continue

        lang_info = LANGUAGES[lang_code]
        logger.info(f"\n--- {lang_info['name']} ({lang_code}) ---")

        # Load from HuggingFace — XL-Sum (BBC news, good multilingual coverage)
        logger.info(f"  Loading XL-Sum {lang_info['mlsum_code']}...")
        ds = load_dataset("csebuetnlp/xlsum", lang_info['mlsum_code'], split="train")
        logger.info(f"  Total articles in train: {len(ds)}")

        # Filter by length and sample
        documents = []
        for article in ds:
            text = article['text'].strip()
            if len(text) < MIN_CHARS:
                continue
            documents.append({
                "doc_id": f"News_{lang_code}_{len(documents):04d}",
                "author_id": f"mlsum_{lang_code}",
                "domain": "news",
                "population": f"written_{lang_code}",
                "text": text,
                "metadata": json.dumps({
                    "dataset": "MLSUM",
                    "language": lang_code,
                    "language_name": lang_info['name'],
                    "title": article.get('title', ''),
                    "topic": article.get('topic', ''),
                    "char_count": len(text),
                }, ensure_ascii=False),
            })
            if len(documents) >= args.articles_per_lang:
                break

        # Write
        out_path = output_dir / f"{lang_code}_news.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        char_counts = [len(d['text']) for d in documents]
        logger.info(f"  Wrote {len(documents)} articles to {out_path}")
        logger.info(f"  Chars: min={min(char_counts)}, max={max(char_counts)}, "
                     f"mean={sum(char_counts)//len(char_counts)}")

    # Also pull English news — use CC-News or similar
    # For now, we can use the English Wikipedia as baseline and add news later

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
