#!/usr/bin/env python3
"""
constraint_audit.py

Compute constraint/canonicalization metrics to characterize task structure.
Helps explain why developmental effects may differ across datasets.

Features:
1. Adult scaffolding (from raw CHAT files if available)
2. Canonicalization / template-likeness (TF-IDF similarity within group)
3. Narrative efficiency (tokens, sentences, TTR, MTLD)

Usage:
    python constraint_audit.py --dataset miami_mono --transcripts data/miami_mono_transcripts.jsonl
    python constraint_audit.py --dataset ecsc --transcripts data/ecsc_processed/transcripts.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# NARRATIVE EFFICIENCY METRICS
# ============================================================

def count_sentences(text: str) -> int:
    """Approximate sentence count via punctuation."""
    # Count sentence-ending punctuation
    return len(re.findall(r'[.!?]+', text))


def compute_ttr(tokens: List[str]) -> float:
    """Type-Token Ratio."""
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (MTLD).

    Forward and backward pass, averaged.
    """
    if len(tokens) < 10:
        return compute_ttr(tokens)  # Fall back to TTR for short texts

    def mtld_pass(tokens):
        factors = 0
        current_tokens = []

        for token in tokens:
            current_tokens.append(token)
            ttr = len(set(current_tokens)) / len(current_tokens)

            if ttr <= threshold:
                factors += 1
                current_tokens = []

        # Handle remainder
        if current_tokens:
            remaining_ttr = len(set(current_tokens)) / len(current_tokens)
            if remaining_ttr < 1.0:
                factors += (1 - remaining_ttr) / (1 - threshold)

        return len(tokens) / factors if factors > 0 else len(tokens)

    forward = mtld_pass(tokens)
    backward = mtld_pass(tokens[::-1])

    return (forward + backward) / 2


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization, lowercased."""
    # Remove punctuation, lowercase, split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if t]


def compute_efficiency_metrics(text: str) -> Dict:
    """Compute narrative efficiency metrics for a document."""
    tokens = tokenize_simple(text)
    n_tokens = len(tokens)
    n_sentences = max(1, count_sentences(text))

    return {
        'n_tokens': n_tokens,
        'n_sentences': n_sentences,
        'tokens_per_sentence': n_tokens / n_sentences,
        'ttr': compute_ttr(tokens),
        'mtld': compute_mtld(tokens),
    }


# ============================================================
# ADULT SCAFFOLDING (from raw CHAT files)
# ============================================================

def parse_cha_for_scaffolding(filepath: Path) -> Optional[Dict]:
    """Parse CHAT file for adult/child turn counts and tokens."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except:
        return None

    child_turns = 0
    adult_turns = 0
    child_tokens = 0
    adult_tokens = 0

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.rstrip()

        # New speaker turn
        if line.startswith('*'):
            # Save previous turn
            if current_speaker and current_text:
                tokens = len(tokenize_simple(current_text))
                if current_speaker == 'CHI':
                    child_turns += 1
                    child_tokens += tokens
                else:
                    adult_turns += 1
                    adult_tokens += tokens

            # Parse new speaker
            match = re.match(r'\*(\w+):\s*(.*)', line)
            if match:
                current_speaker = match.group(1)
                current_text = match.group(2)
            else:
                current_speaker = None
                current_text = ""

        # Continuation line
        elif line.startswith('\t') and not line.startswith('\t%'):
            current_text += ' ' + line.strip()

    # Don't forget last turn
    if current_speaker and current_text:
        tokens = len(tokenize_simple(current_text))
        if current_speaker == 'CHI':
            child_turns += 1
            child_tokens += tokens
        else:
            adult_turns += 1
            adult_tokens += tokens

    total_turns = child_turns + adult_turns
    total_tokens = child_tokens + adult_tokens

    return {
        'n_turns_total': total_turns,
        'n_turns_child': child_turns,
        'n_turns_adult': adult_turns,
        'adult_turn_frac': adult_turns / total_turns if total_turns > 0 else 0,
        'adult_token_frac': adult_tokens / total_tokens if total_tokens > 0 else 0,
        'child_tokens': child_tokens,
        'adult_tokens': adult_tokens,
    }


def find_cha_file(doc_id: str, source_file: str, raw_data_dir: Optional[Path]) -> Optional[Path]:
    """Find the raw CHAT file for a document."""
    if raw_data_dir is None:
        return None

    # Try source_file directly
    if source_file:
        candidates = list(raw_data_dir.rglob(source_file))
        if candidates:
            return candidates[0]

    # Try doc_id.cha
    candidates = list(raw_data_dir.rglob(f"{doc_id}.cha"))
    if candidates:
        return candidates[0]

    return None


# ============================================================
# CANONICALIZATION (TF-IDF similarity)
# ============================================================

def compute_tfidf_vectors(docs: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute TF-IDF vectors for documents."""
    # Build vocabulary
    vocab = {}
    doc_freqs = Counter()

    tokenized_docs = [tokenize_simple(doc) for doc in docs]

    for tokens in tokenized_docs:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freqs[token] += 1
            if token not in vocab:
                vocab[token] = len(vocab)

    n_docs = len(docs)
    n_vocab = len(vocab)

    # Compute TF-IDF
    tfidf = np.zeros((n_docs, n_vocab))

    for i, tokens in enumerate(tokenized_docs):
        tf = Counter(tokens)
        for token, count in tf.items():
            if token in vocab:
                j = vocab[token]
                idf = np.log(n_docs / (1 + doc_freqs[token]))
                tfidf[i, j] = count * idf

    # L2 normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf = tfidf / norms

    return tfidf, vocab


def compute_canonical_similarity(docs: List[str]) -> Tuple[float, List[float]]:
    """Compute mean pairwise cosine similarity within a group.

    Returns:
        mean_sim: mean pairwise similarity
        doc_sims: per-doc mean similarity to others
    """
    if len(docs) < 2:
        return 0.0, [0.0] * len(docs)

    tfidf, _ = compute_tfidf_vectors(docs)

    # Compute cosine similarity matrix
    sim_matrix = tfidf @ tfidf.T

    # Per-doc mean similarity (excluding self)
    n = len(docs)
    doc_sims = []
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        doc_sims.append(np.mean(others) if others else 0.0)

    # Mean pairwise (upper triangle)
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(sim_matrix[i, j])

    mean_sim = np.mean(pairwise) if pairwise else 0.0

    return mean_sim, doc_sims


# ============================================================
# MAIN AUDIT
# ============================================================

def run_constraint_audit(
    transcripts_path: Path,
    dataset: str,
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    group_field: str = 'age_group',
) -> Tuple[List[Dict], Dict]:
    """Run constraint audit on a dataset.

    Returns:
        doc_metrics: per-document metrics
        group_summaries: per-group summary statistics
    """

    # Load transcripts
    records = []
    with open(transcripts_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} documents from {transcripts_path}")

    # Compute per-doc metrics
    doc_metrics = []

    for record in records:
        doc_id = record.get('doc_id', '')
        text = record.get('text', '')
        group = record.get(group_field, record.get('age_bin', 'unknown'))

        metrics = {
            'doc_id': doc_id,
            'dataset': dataset,
            'group': group,
        }

        # Efficiency metrics
        metrics.update(compute_efficiency_metrics(text))

        # Adult scaffolding (if raw files available)
        source_file = record.get('source_file', '')
        cha_path = find_cha_file(doc_id, source_file, raw_data_dir)

        if cha_path:
            scaffolding = parse_cha_for_scaffolding(cha_path)
            if scaffolding:
                metrics.update(scaffolding)
        else:
            # Mark as unavailable
            metrics['n_turns_total'] = None
            metrics['adult_turn_frac'] = None
            metrics['adult_token_frac'] = None

        doc_metrics.append(metrics)

    # Compute canonicalization within groups
    groups = set(m['group'] for m in doc_metrics)

    for group in groups:
        group_docs = [r['text'] for r, m in zip(records, doc_metrics) if m['group'] == group]
        group_indices = [i for i, m in enumerate(doc_metrics) if m['group'] == group]

        if len(group_docs) >= 2:
            mean_sim, doc_sims = compute_canonical_similarity(group_docs)

            for idx, sim in zip(group_indices, doc_sims):
                doc_metrics[idx]['canonical_sim'] = sim
                doc_metrics[idx]['group_canonical_mean'] = mean_sim
        else:
            for idx in group_indices:
                doc_metrics[idx]['canonical_sim'] = None
                doc_metrics[idx]['group_canonical_mean'] = None

    # Compute group summaries with bootstrap CIs
    group_summaries = compute_group_summaries(doc_metrics, groups)

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-doc CSV
        import csv
        doc_path = output_dir / f'{dataset}_constraint_by_doc.csv'
        with open(doc_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=doc_metrics[0].keys())
            writer.writeheader()
            writer.writerows(doc_metrics)
        print(f"Saved: {doc_path}")

        # Per-group CSV
        group_path = output_dir / f'{dataset}_constraint_by_group.csv'
        with open(group_path, 'w', newline='') as f:
            if group_summaries:
                writer = csv.DictWriter(f, fieldnames=group_summaries[0].keys())
                writer.writeheader()
                writer.writerows(group_summaries)
        print(f"Saved: {group_path}")

    return doc_metrics, group_summaries


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    values = [v for v in values if v is not None and not np.isnan(v)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    values = np.array(values)
    mean = np.mean(values)

    if len(values) < 3:
        return mean, mean, mean

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)

    return mean, ci_low, ci_high


def compute_group_summaries(doc_metrics: List[Dict], groups: set) -> List[Dict]:
    """Compute group-level summary statistics with bootstrap CIs."""

    summaries = []

    for group in sorted(groups):
        group_docs = [m for m in doc_metrics if m['group'] == group]
        n = len(group_docs)

        summary = {
            'group': group,
            'n': n,
        }

        # Metrics to summarize
        metrics_to_summarize = [
            ('n_tokens', 'median_tokens'),
            ('ttr', 'ttr'),
            ('mtld', 'mtld'),
            ('tokens_per_sentence', 'tokens_per_sentence'),
            ('canonical_sim', 'canonical_sim'),
            ('adult_token_frac', 'adult_token_frac'),
            ('adult_turn_frac', 'adult_turn_frac'),
        ]

        for field, out_name in metrics_to_summarize:
            values = [m.get(field) for m in group_docs]
            values = [v for v in values if v is not None]

            if values:
                if field == 'n_tokens':
                    # Use median for tokens
                    summary[f'{out_name}_mean'] = np.median(values)
                    summary[f'{out_name}_ci_low'] = np.percentile(values, 25)
                    summary[f'{out_name}_ci_high'] = np.percentile(values, 75)
                else:
                    mean, ci_low, ci_high = bootstrap_ci(values)
                    summary[f'{out_name}_mean'] = mean
                    summary[f'{out_name}_ci_low'] = ci_low
                    summary[f'{out_name}_ci_high'] = ci_high
            else:
                summary[f'{out_name}_mean'] = None
                summary[f'{out_name}_ci_low'] = None
                summary[f'{out_name}_ci_high'] = None

        summaries.append(summary)

    return summaries


def run_enni_stratified_audit(
    csv_path: Path,
    output_dir: Path,
) -> None:
    """Run ENNI-specific stratified audit by story_id × age_bin × group.

    Computes canonical_sim within each story (A1-B3) separately,
    since different stories have different content.
    """
    import csv

    # Load ENNI story docs
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Loaded {len(records)} ENNI story documents")

    # Add age_bin to records
    def get_age_bin(age_years):
        if not age_years:
            return 'UNK'
        try:
            a = float(age_years)
        except (ValueError, TypeError):
            return 'UNK'
        if a < 5: return '4-5yr'
        elif a < 6: return '5-6yr'
        elif a < 7: return '6-7yr'
        elif a < 8: return '7-8yr'
        elif a < 9: return '8-9yr'
        else: return '9-10yr'

    # Filter out UNK group for main analysis
    records = [r for r in records if r['group'] in ('TD', 'LI')]
    print(f"After filtering UNK: {len(records)} documents")

    for r in records:
        r['age_bin'] = get_age_bin(r['age_years'])
        r['adult_token_frac'] = float(r['adult_token_frac']) if r['adult_token_frac'] else 0.0

    # Get unique strata
    stories = sorted(set(r['story_id'] for r in records))
    age_bins = ['4-5yr', '5-6yr', '6-7yr', '7-8yr', '8-9yr', '9-10yr']
    groups = ['TD', 'LI']

    # Compute per-document metrics
    doc_metrics = []
    for r in records:
        text = r['text']
        eff = compute_efficiency_metrics(text)

        doc_metrics.append({
            'doc_id': r['doc_id'],
            'file_id': r['file_id'],
            'story_id': r['story_id'],
            'group': r['group'],
            'age_bin': r['age_bin'],
            'age_years': float(r['age_years']),
            'word_count': int(r['word_count']),
            'n_tokens': eff['n_tokens'],
            'n_sentences': eff['n_sentences'],
            'tokens_per_sentence': eff['tokens_per_sentence'],
            'ttr': eff['ttr'],
            'mtld': eff['mtld'],
            'adult_token_frac': r['adult_token_frac'],
            'canonical_sim': None,  # Will fill in below
        })

    # Map doc_id -> index for quick lookup
    doc_id_to_idx = {m['doc_id']: i for i, m in enumerate(doc_metrics)}

    # Compute canonical_sim WITHIN each story_id (not across stories)
    for story in stories:
        story_docs = [m for m in doc_metrics if m['story_id'] == story]
        story_texts = [r['text'] for r in records if r['story_id'] == story]
        story_doc_ids = [m['doc_id'] for m in story_docs]

        if len(story_texts) >= 2:
            _, doc_sims = compute_canonical_similarity(story_texts)
            for doc_id, sim in zip(story_doc_ids, doc_sims):
                idx = doc_id_to_idx[doc_id]
                doc_metrics[idx]['canonical_sim'] = sim

    # Compute strata summaries: (story_id, age_bin, group)
    strata_summaries = []

    for story in stories:
        for age_bin in age_bins:
            for group in groups:
                stratum_docs = [m for m in doc_metrics
                                if m['story_id'] == story
                                and m['age_bin'] == age_bin
                                and m['group'] == group]

                if not stratum_docs:
                    continue

                n = len(stratum_docs)

                # Compute means with CIs
                summary = {
                    'story_id': story,
                    'age_bin': age_bin,
                    'group': group,
                    'n': n,
                }

                for field in ['n_tokens', 'ttr', 'mtld', 'tokens_per_sentence',
                              'canonical_sim', 'adult_token_frac']:
                    values = [m[field] for m in stratum_docs if m[field] is not None]

                    if values:
                        if field == 'n_tokens':
                            summary[f'{field}_median'] = np.median(values)
                            summary[f'{field}_q25'] = np.percentile(values, 25)
                            summary[f'{field}_q75'] = np.percentile(values, 75)
                        else:
                            mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=500)
                            summary[f'{field}_mean'] = mean
                            summary[f'{field}_ci_low'] = ci_low
                            summary[f'{field}_ci_high'] = ci_high
                    else:
                        if field == 'n_tokens':
                            summary[f'{field}_median'] = None
                            summary[f'{field}_q25'] = None
                            summary[f'{field}_q75'] = None
                        else:
                            summary[f'{field}_mean'] = None
                            summary[f'{field}_ci_low'] = None
                            summary[f'{field}_ci_high'] = None

                strata_summaries.append(summary)

    # Compute weighted means by group (across story × age_bin)
    weighted_summaries = []

    for group in groups:
        group_strata = [s for s in strata_summaries if s['group'] == group]
        total_n = sum(s['n'] for s in group_strata)

        weighted = {
            'story_id': 'ALL',
            'age_bin': 'ALL',
            'group': group,
            'n': total_n,
        }

        for field in ['ttr', 'mtld', 'tokens_per_sentence', 'canonical_sim', 'adult_token_frac']:
            # Weighted mean
            values_n = [(s[f'{field}_mean'], s['n']) for s in group_strata
                        if s.get(f'{field}_mean') is not None]
            if values_n:
                weighted_mean = sum(v * n for v, n in values_n) / sum(n for _, n in values_n)
                weighted[f'{field}_mean'] = weighted_mean
                weighted[f'{field}_ci_low'] = None  # CIs don't combine simply
                weighted[f'{field}_ci_high'] = None
            else:
                weighted[f'{field}_mean'] = None
                weighted[f'{field}_ci_low'] = None
                weighted[f'{field}_ci_high'] = None

        # Median tokens (use median of medians)
        medians = [s['n_tokens_median'] for s in group_strata if s.get('n_tokens_median')]
        if medians:
            weighted['n_tokens_median'] = np.median(medians)
            weighted['n_tokens_q25'] = np.percentile(medians, 25)
            weighted['n_tokens_q75'] = np.percentile(medians, 75)
        else:
            weighted['n_tokens_median'] = None
            weighted['n_tokens_q25'] = None
            weighted['n_tokens_q75'] = None

        weighted_summaries.append(weighted)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-doc CSV
    doc_path = output_dir / 'enni_constraint_by_doc_stratified.csv'
    with open(doc_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=doc_metrics[0].keys())
        writer.writeheader()
        writer.writerows(doc_metrics)
    print(f"Saved: {doc_path}")

    # Per-stratum CSV (story × age_bin × group)
    strata_path = output_dir / 'enni_constraint_by_stratum.csv'
    with open(strata_path, 'w', newline='') as f:
        if strata_summaries:
            writer = csv.DictWriter(f, fieldnames=strata_summaries[0].keys())
            writer.writeheader()
            writer.writerows(strata_summaries)
    print(f"Saved: {strata_path}")

    # Weighted summary CSV
    weighted_path = output_dir / 'enni_constraint_weighted.csv'
    with open(weighted_path, 'w', newline='') as f:
        if weighted_summaries:
            writer = csv.DictWriter(f, fieldnames=weighted_summaries[0].keys())
            writer.writeheader()
            writer.writerows(weighted_summaries)
    print(f"Saved: {weighted_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("ENNI STRATIFIED CONSTRAINT AUDIT")
    print(f"{'='*70}")

    print("\nCanonical similarity by story (TD only):")
    print(f"{'Story':>8}  {'n':>6}  {'Mean Sim':>10}  {'MTLD':>8}")
    for story in stories:
        story_td = [s for s in strata_summaries if s['story_id'] == story and s['group'] == 'TD']
        if story_td:
            total_n = sum(s['n'] for s in story_td)
            mean_sim = np.mean([s['canonical_sim_mean'] for s in story_td if s['canonical_sim_mean']])
            mean_mtld = np.mean([s['mtld_mean'] for s in story_td if s['mtld_mean']])
            print(f"{story:>8}  {total_n:>6}  {mean_sim:>10.4f}  {mean_mtld:>8.2f}")

    print("\nWeighted means by group:")
    for ws in weighted_summaries:
        print(f"  {ws['group']:>4}: n={ws['n']:>4}, canonical_sim={ws['canonical_sim_mean']:.4f}, "
              f"mtld={ws['mtld_mean']:.2f}, adult_frac={ws['adult_token_frac_mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Constraint audit for narrative datasets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (miami_mono, ecsc, enni)')
    parser.add_argument('--transcripts', type=Path, default=None, help='Path to transcripts JSONL')
    parser.add_argument('--enni-csv', type=Path, default=None,
                        help='Path to ENNI story docs CSV (for stratified analysis)')
    parser.add_argument('--raw-data-dir', type=Path, default=None,
                        help='Path to raw CHAT files (for adult scaffolding)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/results/constraint_audit'),
                        help='Output directory')
    parser.add_argument('--group-field', type=str, default='age_group',
                        help='Field to use for grouping')

    args = parser.parse_args()

    if args.dataset == 'enni' and args.enni_csv:
        run_enni_stratified_audit(args.enni_csv, args.output_dir)
    elif args.transcripts:
        run_constraint_audit(
            args.transcripts,
            args.dataset,
            args.raw_data_dir,
            args.output_dir,
            args.group_field,
        )
    else:
        parser.error("Must provide --transcripts or --enni-csv")


if __name__ == '__main__':
    main()
