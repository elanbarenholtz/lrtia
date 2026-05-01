#!/usr/bin/env python3
"""
Generate AI Wikipedia articles matched to human articles.

For each human Wikipedia article in the multilingual corpus, generates
an AI version on the same topic in the same language using Claude.
This creates a perfectly matched human vs AI comparison for coherence
decay analysis.

Usage:
    python scripts/generate_ai_wiki_articles.py
    python scripts/generate_ai_wiki_articles.py --languages zh ja ko
    python scripts/generate_ai_wiki_articles.py --max-per-lang 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/wiki_multilingual")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/wiki_multilingual_ai")

LANG_NAMES = {
    'en': 'English',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'fi': 'Finnish',
}

LANG_INSTRUCTIONS = {
    'en': 'Write in English',
    'zh': '请用中文撰写',
    'ja': '日本語で書いてください',
    'ko': '한국어로 작성해 주세요',
    'tr': 'Türkçe yazınız',
    'ar': 'اكتب باللغة العربية',
    'fi': 'Kirjoita suomeksi',
}


def generate_article(client: anthropic.Anthropic, title: str, lang_code: str,
                     target_chars: int) -> str:
    """Generate a Wikipedia-style article using Claude."""
    lang_name = LANG_NAMES[lang_code]
    lang_instruction = LANG_INSTRUCTIONS[lang_code]

    # Target word count (rough: chars / 2 for CJK, chars / 5 for others)
    if lang_code in ('zh', 'ja', 'ko'):
        target_words = target_chars // 2
    else:
        target_words = target_chars // 5

    # Cap at reasonable length
    target_words = min(target_words, 3000)
    target_words = max(target_words, 500)

    prompt = f"""{lang_instruction}

Write a Wikipedia-style encyclopedia article about: {title}

Requirements:
- Write entirely in {lang_name}
- Use a neutral, encyclopedic tone typical of Wikipedia
- Include an introduction, multiple sections with factual content
- Aim for approximately {target_words} words
- Do not include any metadata, references section, or markup
- Write continuous prose, not bullet points
- Do not include the title as a heading at the start"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(description="Generate AI Wikipedia articles")
    parser.add_argument("--languages", nargs="+", default=list(LANG_NAMES.keys()))
    parser.add_argument("--max-per-lang", type=int, default=60,
                        help="Max articles per language (default: 60)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    logger.info("=" * 70)
    logger.info("AI WIKIPEDIA ARTICLE GENERATION")
    logger.info("=" * 70)

    for lang_code in args.languages:
        if lang_code not in LANG_NAMES:
            continue

        logger.info(f"\n--- {LANG_NAMES[lang_code]} ({lang_code}) ---")

        # Load human articles to get titles and target lengths
        human_path = DATA_DIR / f"{lang_code}_articles.jsonl"
        if not human_path.exists():
            logger.warning(f"  No human articles found at {human_path}")
            continue

        human_articles = []
        with open(human_path) as f:
            for line in f:
                human_articles.append(json.loads(line))

        # Check for existing output (resume support)
        output_path = output_dir / f"{lang_code}_ai_articles.jsonl"
        existing = set()
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    d = json.loads(line)
                    existing.add(d.get('doc_id', ''))
            logger.info(f"  Found {len(existing)} existing articles, resuming")

        generated = 0
        errors = 0

        with open(output_path, 'a', encoding='utf-8') as out_f:
            for article in human_articles[:args.max_per_lang]:
                meta = json.loads(article['metadata'])
                title = meta['title']
                doc_id = f"AI_{article['doc_id']}"

                if doc_id in existing:
                    continue

                try:
                    ai_text = generate_article(
                        client, title, lang_code,
                        target_chars=len(article['text'])
                    )

                    doc = {
                        "doc_id": doc_id,
                        "author_id": "claude-sonnet",
                        "domain": "encyclopedia",
                        "population": f"ai_{lang_code}",
                        "text": ai_text,
                        "metadata": json.dumps({
                            "dataset": "Wikipedia_AI",
                            "language": lang_code,
                            "language_name": LANG_NAMES[lang_code],
                            "title": title,
                            "generator": "claude-sonnet-4-20250514",
                            "matched_human_doc": article['doc_id'],
                            "human_char_count": len(article['text']),
                            "ai_char_count": len(ai_text),
                        }, ensure_ascii=False),
                    }

                    out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    out_f.flush()
                    generated += 1

                    if generated % 10 == 0:
                        logger.info(f"  Generated {generated} articles...")

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"  Error on '{title}': {e}")
                    errors += 1
                    time.sleep(2)

        logger.info(f"  Done: {generated} generated, {errors} errors")
        logger.info(f"  Total in {output_path.name}: {len(existing) + generated}")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "AI-Generated Wikipedia Articles (Multilingual)",
            "generator": "claude-sonnet-4-20250514",
            "purpose": "Matched AI comparison for coherence decay analysis",
            "method": "Each article generated to match a human Wikipedia article by topic and approximate length",
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
