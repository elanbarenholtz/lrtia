#!/usr/bin/env python3
"""
Multilingual Wikipedia Ingestion Pipeline.

Downloads Wikipedia articles in multiple non-Indo-European languages for
cross-linguistic coherence decay analysis.

Target languages (non-Indo-European, with good Mistral-7B coverage):
    zh  - Chinese (Sino-Tibetan, isolating, SVO)
    ja  - Japanese (Japonic, agglutinative, SOV)
    ko  - Korean (Koreanic, agglutinative, SOV)
    tr  - Turkish (Turkic, agglutinative, SOV)
    ar  - Arabic (Afro-Asiatic, root-pattern morphology, VSO/SVO)
    fi  - Finnish (Uralic, agglutinative, rich case system)

Strategy: Pull long articles from each language's Wikipedia. We want articles
with 1000+ words to ensure enough context for the 100-token reveal window.
Use a mix of topic categories to avoid domain bias.

Output: JSONL per language, matching the format used by other ingestion scripts.

Usage:
    python scripts/ingest_wikipedia_multilingual.py
    python scripts/ingest_wikipedia_multilingual.py --languages zh ja ko
    python scripts/ingest_wikipedia_multilingual.py --articles-per-lang 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import wikipediaapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/data/wiki_multilingual")

# Languages and seed categories
# We use a mix of topics to avoid domain bias
LANGUAGES = {
    'zh': {
        'name': 'Chinese',
        'family': 'Sino-Tibetan',
        'typology': 'isolating, SVO',
        'seeds': [
            '历史', '物理学', '生物学', '哲学', '经济学',
            '数学', '化学', '地理学', '天文学', '心理学',
            '文学', '音乐', '医学', '政治学', '社会学',
        ],
    },
    'ja': {
        'name': 'Japanese',
        'family': 'Japonic',
        'typology': 'agglutinative, SOV',
        'seeds': [
            '歴史', '物理学', '生物学', '哲学', '経済学',
            '数学', '化学', '地理学', '天文学', '心理学',
            '文学', '音楽', '医学', '政治学', '社会学',
        ],
    },
    'ko': {
        'name': 'Korean',
        'family': 'Koreanic',
        'typology': 'agglutinative, SOV',
        'seeds': [
            '역사', '물리학', '생물학', '철학', '경제학',
            '수학', '화학', '지리학', '천문학', '심리학',
            '문학', '음악', '의학', '정치학', '사회학',
        ],
    },
    'tr': {
        'name': 'Turkish',
        'family': 'Turkic',
        'typology': 'agglutinative, SOV, vowel harmony',
        'seeds': [
            'Tarih', 'Fizik', 'Biyoloji', 'Felsefe', 'Ekonomi',
            'Matematik', 'Kimya', 'Coğrafya', 'Astronomi', 'Psikoloji',
            'Edebiyat', 'Müzik', 'Tıp', 'Siyaset bilimi', 'Sosyoloji',
        ],
    },
    'ar': {
        'name': 'Arabic',
        'family': 'Afro-Asiatic',
        'typology': 'root-pattern morphology, VSO/SVO',
        'seeds': [
            'تاريخ', 'فيزياء', 'أحياء', 'فلسفة', 'اقتصاد',
            'رياضيات', 'كيمياء', 'جغرافيا', 'علم الفلك', 'علم النفس',
            'أدب', 'موسيقى', 'طب', 'علوم سياسية', 'علم الاجتماع',
        ],
    },
    'fi': {
        'name': 'Finnish',
        'family': 'Uralic',
        'typology': 'agglutinative, rich case system',
        'seeds': [
            'Historia', 'Fysiikka', 'Biologia', 'Filosofia', 'Taloustiede',
            'Matematiikka', 'Kemia', 'Maantiede', 'Tähtitiede', 'Psykologia',
            'Kirjallisuus', 'Musiikki', 'Lääketiede', 'Politiikka', 'Sosiologia',
        ],
    },
}

# Minimum character count (not words — word counting varies across languages)
MIN_CHARS = 3000


def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia article text."""
    # Remove section headers (== Header ==)
    text = re.sub(r'={2,}[^=]+={2,}', '', text)

    # Remove reference markers like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove parenthetical empty references
    text = re.sub(r'\(\s*\)', '', text)

    # Normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fetch_articles_for_language(lang_code: str, lang_info: dict,
                                 target_count: int) -> list[dict]:
    """Fetch Wikipedia articles for a single language."""
    wiki = wikipediaapi.Wikipedia(
        user_agent='LRTIA-Research/1.0 (elanbarenholtz@gmail.com)',
        language=lang_code,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    articles = []
    seen_titles = set()

    for seed in lang_info['seeds']:
        if len(articles) >= target_count:
            break

        # Try the seed article itself
        page = wiki.page(seed)
        if page.exists() and page.title not in seen_titles:
            text = clean_wiki_text(page.text)
            if len(text) >= MIN_CHARS:
                articles.append({
                    'title': page.title,
                    'text': text,
                    'char_count': len(text),
                    'source': 'seed',
                })
                seen_titles.add(page.title)

        # Follow links from the seed article to get more articles
        if page.exists():
            for link_title, link_page in list(page.links.items())[:30]:
                if len(articles) >= target_count:
                    break
                if link_title in seen_titles:
                    continue

                try:
                    linked = wiki.page(link_title)
                    if not linked.exists():
                        continue
                    text = clean_wiki_text(linked.text)
                    if len(text) >= MIN_CHARS:
                        articles.append({
                            'title': linked.title,
                            'text': text,
                            'char_count': len(text),
                            'source': f'link_from_{seed}',
                        })
                        seen_titles.add(link_title)
                except Exception:
                    continue

            # Small delay to be nice to the API
            time.sleep(0.5)

    return articles


def main():
    parser = argparse.ArgumentParser(description="Multilingual Wikipedia Ingestion")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()),
                        help="Language codes to process")
    parser.add_argument("--articles-per-lang", type=int, default=60,
                        help="Target number of articles per language (default: 60)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MULTILINGUAL WIKIPEDIA INGESTION")
    logger.info("=" * 70)

    all_stats = {}

    for lang_code in args.languages:
        if lang_code not in LANGUAGES:
            logger.warning(f"Unknown language code: {lang_code}, skipping")
            continue

        lang_info = LANGUAGES[lang_code]
        logger.info(f"\n--- {lang_info['name']} ({lang_code}) ---")
        logger.info(f"  Family: {lang_info['family']}, Typology: {lang_info['typology']}")

        articles = fetch_articles_for_language(lang_code, lang_info, args.articles_per_lang)
        logger.info(f"  Fetched {len(articles)} articles (>={MIN_CHARS} chars)")

        if not articles:
            logger.warning(f"  No articles found for {lang_code}, skipping")
            continue

        # Create documents
        documents = []
        for art in articles:
            documents.append({
                "doc_id": f"Wiki_{lang_code}_{art['title'][:50]}",
                "author_id": f"wikipedia_{lang_code}",
                "domain": "encyclopedia",
                "population": f"written_{lang_code}",
                "text": art['text'],
                "metadata": json.dumps({
                    "dataset": "Wikipedia",
                    "language": lang_code,
                    "language_name": lang_info['name'],
                    "language_family": lang_info['family'],
                    "typology": lang_info['typology'],
                    "title": art['title'],
                    "char_count": art['char_count'],
                    "source": art['source'],
                }, ensure_ascii=False),
            })

        # Write output
        lang_path = output_dir / f"{lang_code}_articles.jsonl"
        with open(lang_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        char_counts = [len(d['text']) for d in documents]
        stats = {
            'articles': len(documents),
            'total_chars': sum(char_counts),
            'mean_chars': sum(char_counts) // len(char_counts),
            'min_chars': min(char_counts),
            'max_chars': max(char_counts),
        }
        all_stats[lang_code] = stats

        logger.info(f"  Wrote {len(documents)} articles to {lang_path}")
        logger.info(f"  Chars: min={stats['min_chars']}, max={stats['max_chars']}, "
                     f"mean={stats['mean_chars']}, total={stats['total_chars']:,}")

    # Write manifest
    manifest = {
        "corpus": {
            "name": "Multilingual Wikipedia Articles",
            "source": "Wikipedia API",
            "license": "CC BY-SA 3.0",
            "purpose": "Cross-linguistic coherence decay analysis",
        },
        "processing": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_chars": MIN_CHARS,
            "target_per_lang": args.articles_per_lang,
        },
        "languages": {
            code: {
                **LANGUAGES[code],
                'seeds': LANGUAGES[code]['seeds'],  # keep for reproducibility
                **all_stats.get(code, {}),
            }
            for code in args.languages if code in LANGUAGES
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for code, stats in all_stats.items():
        name = LANGUAGES[code]['name']
        family = LANGUAGES[code]['family']
        logger.info(f"  {name} ({code}, {family}): {stats['articles']} articles, "
                     f"{stats['total_chars']:,} chars")
    logger.info(f"\nWrote manifest to {manifest_path}")
    logger.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
