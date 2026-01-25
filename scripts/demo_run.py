#!/usr/bin/env python3
"""Demo script for LRTIA with synthetic corpus generation.

This script generates a synthetic corpus with varied document structures
and runs a small-scale LRTIA analysis for demonstration purposes.
"""

import json
import random
import string
from pathlib import Path

import numpy as np


def generate_synthetic_corpus(
    n_documents: int = 50,
    min_words: int = 500,
    max_words: int = 2000,
    n_populations: int = 2,
    n_authors_per_pop: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate a synthetic corpus with varied document structures.

    Args:
        n_documents: Number of documents to generate.
        min_words: Minimum words per document.
        max_words: Maximum words per document.
        n_populations: Number of distinct populations.
        n_authors_per_pop: Number of authors per population.
        seed: Random seed.

    Returns:
        List of document dictionaries.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Common word pools for generating text
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    ]

    # Topic-specific words for populations
    population_topics = {
        "tech": ["computer", "software", "data", "system", "network", "algorithm",
                 "program", "code", "digital", "interface", "database", "server",
                 "cloud", "machine", "learning", "artificial", "intelligence", "robot",
                 "automation", "platform", "application", "developer", "engineering"],
        "nature": ["forest", "river", "mountain", "ocean", "tree", "animal",
                   "wildlife", "ecosystem", "climate", "weather", "species", "habitat",
                   "conservation", "environment", "natural", "biodiversity", "landscape",
                   "wilderness", "earth", "planet", "sustainable", "organic", "ecological"],
        "social": ["community", "society", "culture", "family", "relationship", "social",
                   "group", "interaction", "communication", "tradition", "values", "belief",
                   "identity", "belonging", "connection", "network", "support", "trust",
                   "collaboration", "cooperation", "institution", "organization", "movement"],
    }

    population_names = list(population_topics.keys())[:n_populations]

    # Generate documents
    documents = []

    for doc_idx in range(n_documents):
        # Assign population and author
        pop_idx = doc_idx % n_populations
        population = population_names[pop_idx]
        author_idx = (doc_idx // n_populations) % n_authors_per_pop
        author_id = f"{population}_author_{author_idx:02d}"

        # Determine document length
        num_words = rng.randint(min_words, max_words)

        # Generate text with population-specific flavor
        topic_words = population_topics[population]
        topic_prob = 0.15  # Probability of using topic-specific word

        words = []
        for _ in range(num_words):
            if rng.random() < topic_prob:
                words.append(rng.choice(topic_words))
            else:
                words.append(rng.choice(common_words))

        # Add some structure (paragraphs)
        structured_text = []
        paragraph_lengths = np_rng.geometric(0.02, size=100)  # Average ~50 words per paragraph
        word_idx = 0

        for para_len in paragraph_lengths:
            para_len = max(10, int(para_len))  # Minimum 10 words
            if word_idx >= num_words:
                break

            end_idx = min(word_idx + para_len, num_words)
            paragraph = " ".join(words[word_idx:end_idx])

            # Capitalize first word and add period
            paragraph = paragraph[0].upper() + paragraph[1:] + "."

            structured_text.append(paragraph)
            word_idx = end_idx

        text = "\n\n".join(structured_text)

        documents.append({
            "doc_id": f"doc_{doc_idx:04d}",
            "author_id": author_id,
            "population": population,
            "text": text,
        })

    return documents


def save_corpus(documents: list[dict], output_path: Path) -> None:
    """Save corpus to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


def run_demo():
    """Run the LRTIA demo."""
    print("=" * 60)
    print("LRTIA Demo - Long-Range Token Influence Analyzer")
    print("=" * 60)
    print()

    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "demo_data"
    results_dir = project_dir / "demo_results"
    figures_dir = project_dir / "demo_figures"

    # Generate synthetic corpus
    print("1. Generating synthetic corpus...")
    documents = generate_synthetic_corpus(
        n_documents=50,
        min_words=500,
        max_words=1500,
        n_populations=2,
        seed=42,
    )

    corpus_path = data_dir / "synthetic_corpus.jsonl"
    save_corpus(documents, corpus_path)
    print(f"   Generated {len(documents)} documents")
    print(f"   Saved to: {corpus_path}")

    # Display corpus statistics
    print()
    print("2. Corpus statistics:")
    populations = {}
    authors = {}
    for doc in documents:
        pop = doc["population"]
        auth = doc["author_id"]
        populations[pop] = populations.get(pop, 0) + 1
        authors[auth] = authors.get(auth, 0) + 1

    for pop, count in populations.items():
        print(f"   Population '{pop}': {count} documents")

    print(f"   Total authors: {len(authors)}")
    print()

    # Check if we can run the full analysis
    print("3. Running LRTIA analysis...")
    print("   (This requires PyTorch and transformers to be installed)")
    print()

    try:
        import torch
        from transformers import AutoTokenizer

        # Quick check - just verify we can load a tokenizer
        print("   Loading GPT-2 tokenizer for verification...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("   Tokenizer loaded successfully!")
        print()

        # Show sample tokenization
        sample_text = documents[0]["text"][:200]
        tokens = tokenizer.encode(sample_text)
        print(f"   Sample text ({len(sample_text)} chars, {len(tokens)} tokens):")
        print(f"   \"{sample_text[:100]}...\"")
        print()

        # Check GPU availability
        if torch.cuda.is_available():
            print(f"   GPU available: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("   Apple Silicon MPS available")
            device = "mps"
        else:
            print("   Running on CPU (analysis will be slower)")
            device = "cpu"
        print()

        # Run minimal analysis
        print("4. Running minimal analysis (first 5 documents)...")
        print("   This may take a few minutes on CPU...")
        print()

        from lrtia import LRTIAConfig
        from lrtia.data import load_corpus, create_windows
        from lrtia.model import HuggingFaceBackend
        from lrtia.intervention import SpanSelector, apply_intervention
        from lrtia.metrics import compute_all_metrics
        from lrtia.data.windowing import get_target_positions
        from lrtia.aggregation import build_memory_curve, compute_summary_metrics

        # Load configuration
        cfg = LRTIAConfig()
        cfg.model.device = device
        cfg.windowing.window_tokens = 512  # Smaller for demo
        cfg.targets.interval = 200  # Fewer targets

        # Load model
        print("   Loading model...")
        backend = HuggingFaceBackend(
            model_name="gpt2",
            device=device,
            dtype="float32" if device == "cpu" else "auto",
        )
        backend.load()

        # Process subset of documents
        demo_docs = load_corpus(corpus_path)[:5]
        span_selector = SpanSelector(
            distances=[32, 64, 128, 256],  # Fewer distances for demo
            widths=[16],  # Smaller spans
            spans_per_distance=1,
            seed=42,
        )

        mask_token_id = backend.get_mask_token_id()
        all_results = []

        for i, doc in enumerate(demo_docs):
            print(f"   Processing document {i+1}/{len(demo_docs)}...")

            windows = create_windows(
                doc,
                backend.tokenizer,
                window_tokens=cfg.windowing.window_tokens,
                stride_tokens=cfg.windowing.stride_tokens,
            )

            for window in windows[:1]:  # Just first window per doc
                targets = get_target_positions(
                    window,
                    method="interval",
                    interval=cfg.targets.interval,
                    min_context=64,
                    region_length=20,
                )

                for target_start, target_end in targets[:2]:  # Just 2 targets
                    spans = span_selector.get_all_spans_for_target(
                        target_start=target_start,
                        target_end=target_end,
                        context_length=window.num_tokens,
                        doc_id=doc.doc_id,
                        window_id=window.window_id,
                    )

                    if not spans:
                        continue

                    target_positions = list(range(target_start, min(target_end, window.num_tokens - 1)))
                    if not target_positions:
                        continue

                    original_output = backend.forward(
                        window.token_ids,
                        positions=target_positions,
                        return_full_distribution=True,
                    )

                    target_tokens = [window.token_ids[p + 1] for p in target_positions]

                    for span in spans:
                        for intervention_type in ["mask"]:  # Just mask for demo
                            result = apply_intervention(
                                window.token_ids,
                                span,
                                intervention_type,
                                mask_token_id=mask_token_id,
                                seed=42,
                            )

                            intervened_output = backend.forward(
                                result.token_ids,
                                positions=target_positions,
                                return_full_distribution=True,
                            )

                            metrics = compute_all_metrics(
                                original_output.log_probs[0].cpu().numpy(),
                                intervened_output.log_probs[0].cpu().numpy(),
                                target_tokens,
                            )

                            all_results.append({
                                "doc_id": doc.doc_id,
                                "population": doc.population,
                                "distance": span.distance,
                                "delta_nll": metrics.delta_nll,
                                "kl_divergence": metrics.kl_divergence,
                            })

        # Cleanup
        backend.unload()

        if all_results:
            import pandas as pd

            print()
            print("5. Analysis complete!")
            print(f"   Collected {len(all_results)} evaluations")
            print()

            # Quick summary
            df = pd.DataFrame(all_results)
            print("   Results by distance:")
            print(df.groupby("distance")["delta_nll"].agg(["mean", "std", "count"]).round(4))
            print()

            # Save results
            results_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_dir / "demo_results.csv", index=False)
            print(f"   Results saved to: {results_dir / 'demo_results.csv'}")

    except ImportError as e:
        print(f"   Cannot run full analysis: {e}")
        print("   Install dependencies with: pip install -e '.[dev]'")
        print()
        print("   Skipping model-based analysis.")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install the package: pip install -e .")
    print("  2. Run full analysis: lrtia run demo_data/synthetic_corpus.jsonl")
    print("  3. Analyze results: lrtia analyze results/evaluations.parquet")
    print("  4. Generate plots: lrtia plot results/evaluations.parquet")
    print()


if __name__ == "__main__":
    run_demo()
