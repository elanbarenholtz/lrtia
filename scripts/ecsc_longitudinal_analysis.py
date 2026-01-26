#!/usr/bin/env python3
"""
ECSC Longitudinal Analysis: Compare context dependency across study years (Y1 vs Y2 vs Y3)

Uses context ablation to measure how perplexity changes with varying context lengths,
then compares half-life metrics across populations.
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm


@dataclass
class AnalysisConfig:
    """Configuration for the analysis."""
    # Data
    input_file: Path = Path("data/ecsc_processed/transcripts.jsonl")
    output_dir: Path = Path("results/ecsc_longitudinal")
    min_words: int = 400

    # Model
    # Options:
    #   - "mistralai/Mistral-7B-v0.1" (best quality, requires ~14GB VRAM or CUDA + quantization)
    #   - "microsoft/phi-2" (2.7B params, good quality, ~6GB VRAM)
    #   - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params, faster, ~3GB VRAM)
    #   - "gpt2-large" (774M params, fastest, ~2GB VRAM)
    model_name: str = "microsoft/phi-2"  # Good balance of quality and memory
    use_quantization: bool = True  # 4-bit quantization (requires CUDA + bitsandbytes)

    # Context ablation parameters
    target_size: int = 40  # Last N tokens to measure perplexity on
    context_lengths: list = field(default_factory=lambda: [
        4, 8, 12, 16, 20, 24, 28, 32,
        40, 48, 56, 64,
        80, 96, 112, 128,
        160, 192, 224, 256,
        320, 384
    ])

    # Analysis
    population_field: str = "study_year"  # Field to compare
    bootstrap_samples: int = 1000
    random_seed: int = 42


def load_and_filter_data(config: AnalysisConfig) -> pd.DataFrame:
    """Load ECSC data and filter to documents meeting criteria."""
    records = []
    with open(config.input_file) as f:
        for line in f:
            record = json.loads(line)
            record['pop'] = json.loads(record['population'])
            record['word_count'] = len(record['text'].split())
            records.append(record)

    df = pd.DataFrame(records)

    # Filter by word count
    df = df[df['word_count'] > config.min_words].copy()

    # Extract population field
    df['group'] = df['pop'].apply(lambda x: x.get(config.population_field))

    print(f"Loaded {len(df)} documents with >{config.min_words} words")
    print(f"Population distribution ({config.population_field}):")
    print(df['group'].value_counts().sort_index())

    return df


def load_model(config: AnalysisConfig):
    """Load the language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    if config.use_quantization and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError:
            print("Warning: bitsandbytes not available, loading without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    elif device == "mps":
        # MPS-specific loading
        print("Using MPS (Apple Silicon) - loading in float32 for stability")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # MPS works better with float32
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    model.eval()
    print(f"Model loaded: {config.model_name}")

    return model, tokenizer, device


@torch.no_grad()
def compute_perplexity_on_region(model, token_ids: list, target_start: int, target_end: int, device: str) -> tuple:
    """
    Compute perplexity only on tokens in [target_start, target_end].
    Context before target_start is provided but not scored.
    """
    if target_end > len(token_ids):
        target_end = len(token_ids)

    input_ids = torch.tensor([token_ids], device=device)

    outputs = model(input_ids)
    logits = outputs.logits[0]

    total_loss = 0.0
    count = 0

    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        target_token = token_ids[i + 1]
        token_loss = -log_probs[target_token].item()
        total_loss += token_loss
        count += 1

    if count == 0:
        return float('inf'), 0

    avg_loss = total_loss / count
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


def analyze_document(model, tokenizer, text: str, config: AnalysisConfig, device: str) -> list:
    """
    Run context ablation on a single document.
    Returns list of {context_length, perplexity, loss} dicts.
    """
    full_tokens = tokenizer.encode(text)
    n_tokens = len(full_tokens)

    # Ensure we have enough tokens
    if n_tokens < config.target_size + min(config.context_lengths):
        return []

    # Filter context lengths that are achievable
    max_possible_context = n_tokens - config.target_size
    valid_context_lengths = [c for c in config.context_lengths if c <= max_possible_context]

    results = []

    for ctx_len in valid_context_lengths:
        # Truncate to keep only context + target
        doc_start = n_tokens - config.target_size - ctx_len
        truncated_tokens = full_tokens[doc_start:]

        target_start = len(truncated_tokens) - config.target_size
        target_end = len(truncated_tokens)

        ppl, loss = compute_perplexity_on_region(
            model, truncated_tokens, target_start, target_end, device
        )

        results.append({
            'context_length': ctx_len,
            'perplexity': ppl,
            'loss': loss,
        })

    return results


def compute_half_life(contexts: np.ndarray, perplexities: np.ndarray, percentile: float = 0.5) -> float:
    """
    Compute the context length at which `percentile` of total benefit is achieved.
    """
    order = np.argsort(contexts)
    contexts = contexts[order]
    perplexities = perplexities[order]

    total_benefit = perplexities[0] - perplexities[-1]

    if total_benefit <= 0:
        return np.nan

    target_ppl = perplexities[0] - percentile * total_benefit

    for i in range(len(perplexities) - 1):
        if perplexities[i] >= target_ppl >= perplexities[i+1]:
            frac = (perplexities[i] - target_ppl) / (perplexities[i] - perplexities[i+1])
            return contexts[i] + frac * (contexts[i+1] - contexts[i])

    return contexts[-1]


def compute_document_half_life(doc_results: list) -> dict:
    """Compute half-life and other metrics for a single document."""
    if not doc_results:
        return {'half_life': np.nan, 'total_benefit': np.nan}

    contexts = np.array([r['context_length'] for r in doc_results])
    perplexities = np.array([r['perplexity'] for r in doc_results])

    half_life = compute_half_life(contexts, perplexities, 0.5)
    total_benefit = perplexities[0] - perplexities[-1] if len(perplexities) > 0 else np.nan

    # Also compute quartile lives
    hl_25 = compute_half_life(contexts, perplexities, 0.25)
    hl_75 = compute_half_life(contexts, perplexities, 0.75)

    return {
        'half_life': half_life,
        'quarter_life': hl_25,
        'three_quarter_life': hl_75,
        'total_benefit': total_benefit,
        'min_perplexity': perplexities[-1] if len(perplexities) > 0 else np.nan,
        'max_perplexity': perplexities[0] if len(perplexities) > 0 else np.nan,
    }


def bootstrap_ci(values: np.ndarray, n_samples: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return lower, upper


def compare_populations(df: pd.DataFrame, config: AnalysisConfig) -> dict:
    """Statistical comparison of half-lives across populations."""
    groups = sorted(df['group'].unique())

    results = {
        'group_stats': {},
        'pairwise_comparisons': [],
    }

    # Per-group statistics
    for group in groups:
        group_df = df[df['group'] == group]
        half_lives = group_df['half_life'].dropna().values

        if len(half_lives) < 2:
            continue

        mean_hl = np.mean(half_lives)
        std_hl = np.std(half_lives)
        ci_lower, ci_upper = bootstrap_ci(half_lives, config.bootstrap_samples)

        results['group_stats'][group] = {
            'n': len(half_lives),
            'mean': mean_hl,
            'std': std_hl,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    # Pairwise comparisons
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            vals1 = df[df['group'] == g1]['half_life'].dropna().values
            vals2 = df[df['group'] == g2]['half_life'].dropna().values

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2)
            cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0

            results['pairwise_comparisons'].append({
                'group1': g1,
                'group2': g2,
                'diff': np.mean(vals1) - np.mean(vals2),
                'cohens_d': cohens_d,
                'mann_whitney_u': stat,
                'p_value': p_value,
            })

    return results


def run_analysis(config: AnalysisConfig):
    """Main analysis pipeline."""
    # Setup output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_filter_data(config)

    # Load model
    model, tokenizer, device = load_model(config)

    # Run context ablation on each document
    all_results = []
    doc_metrics = []

    print(f"\nRunning context ablation on {len(df)} documents...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        doc_results = analyze_document(model, tokenizer, row['text'], config, device)

        # Store raw results
        for r in doc_results:
            r['doc_id'] = row['doc_id']
            r['author_id'] = row['author_id']
            r['group'] = row['group']
            all_results.append(r)

        # Compute document-level metrics
        metrics = compute_document_half_life(doc_results)
        metrics['doc_id'] = row['doc_id']
        metrics['author_id'] = row['author_id']
        metrics['group'] = row['group']
        metrics['word_count'] = row['word_count']
        doc_metrics.append(metrics)

    # Create DataFrames
    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(doc_metrics)

    # Population comparison
    print("\n" + "=" * 70)
    print(f"POPULATION COMPARISON: {config.population_field}")
    print("=" * 70)

    comparison = compare_populations(metrics_df, config)

    print("\nPer-group statistics (Half-Life in tokens):")
    print("-" * 50)
    for group, stats_dict in sorted(comparison['group_stats'].items()):
        print(f"  Year {group}: mean={stats_dict['mean']:.1f} ± {stats_dict['std']:.1f} "
              f"[{stats_dict['ci_lower']:.1f}, {stats_dict['ci_upper']:.1f}] (n={stats_dict['n']})")

    print("\nPairwise comparisons:")
    print("-" * 50)
    for comp in comparison['pairwise_comparisons']:
        sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
        print(f"  Year {comp['group1']} vs Year {comp['group2']}: "
              f"diff={comp['diff']:+.1f}, d={comp['cohens_d']:.2f}, p={comp['p_value']:.4f} {sig}")

    # Save results
    results_df.to_csv(config.output_dir / "context_ablation_results.csv", index=False)
    metrics_df.to_csv(config.output_dir / "document_metrics.csv", index=False)

    with open(config.output_dir / "population_comparison.json", 'w') as f:
        # Convert numpy types for JSON serialization
        comparison_json = json.loads(json.dumps(comparison, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
        json.dump(comparison_json, f, indent=2)

    print(f"\nResults saved to {config.output_dir}/")

    # Generate summary plot
    try:
        create_summary_plot(results_df, metrics_df, comparison, config)
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")

    return results_df, metrics_df, comparison


def create_summary_plot(results_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        comparison: dict, config: AnalysisConfig):
    """Create visualization of results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    groups = sorted(results_df['group'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))

    # Top Left: Perplexity curves by group
    ax = axes[0, 0]
    for i, group in enumerate(groups):
        group_df = results_df[results_df['group'] == group]
        means = group_df.groupby('context_length')['perplexity'].mean()
        sems = group_df.groupby('context_length')['perplexity'].sem()

        ax.errorbar(means.index, means.values, yerr=sems.values,
                    marker='o', capsize=3, label=f'Year {group}',
                    color=colors[i], linewidth=2, markersize=4)

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs Context by Study Year')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top Right: Half-life distributions
    ax = axes[0, 1]
    half_life_data = [metrics_df[metrics_df['group'] == g]['half_life'].dropna().values
                      for g in groups]
    bp = ax.boxplot(half_life_data, labels=[f'Year {g}' for g in groups], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Half-Life (tokens)')
    ax.set_title('Half-Life Distribution by Study Year')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom Left: Mean half-life with CI
    ax = axes[1, 0]
    group_labels = [f'Year {g}' for g in groups]
    means = [comparison['group_stats'][g]['mean'] for g in groups]
    ci_lower = [comparison['group_stats'][g]['ci_lower'] for g in groups]
    ci_upper = [comparison['group_stats'][g]['ci_upper'] for g in groups]
    errors = [[m - l for m, l in zip(means, ci_lower)],
              [u - m for m, u in zip(means, ci_upper)]]

    bars = ax.bar(group_labels, means, yerr=errors, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Half-Life (tokens)')
    ax.set_title('Mean Half-Life with 95% CI')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom Right: Perplexity reduction (memory curve)
    ax = axes[1, 1]
    for i, group in enumerate(groups):
        group_df = results_df[results_df['group'] == group]
        means = group_df.groupby('context_length')['perplexity'].mean().sort_index()

        baseline = means.iloc[0]
        reduction = baseline - means

        ax.plot(reduction.index, reduction.values,
                marker='o', label=f'Year {group}',
                color=colors[i], linewidth=2, markersize=4)

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Perplexity Reduction')
    ax.set_title('Cumulative Benefit of Context')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.output_dir / 'longitudinal_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {config.output_dir / 'longitudinal_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="ECSC Longitudinal Analysis")
    parser.add_argument("--input", type=Path, default=Path("data/ecsc_processed/transcripts.jsonl"),
                        help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=Path("results/ecsc_longitudinal"),
                        help="Output directory")
    parser.add_argument("--min-words", type=int, default=400,
                        help="Minimum word count threshold")
    parser.add_argument("--model", type=str, default="microsoft/phi-2",
                        help="Model to use (phi-2, Mistral-7B, TinyLlama, gpt2-large)")
    parser.add_argument("--no-quantization", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--target-size", type=int, default=40,
                        help="Target region size in tokens")

    args = parser.parse_args()

    config = AnalysisConfig(
        input_file=args.input,
        output_dir=args.output,
        min_words=args.min_words,
        model_name=args.model,
        use_quantization=not args.no_quantization,
        target_size=args.target_size,
    )

    run_analysis(config)


if __name__ == "__main__":
    main()
