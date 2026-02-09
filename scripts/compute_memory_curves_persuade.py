#!/usr/bin/env python3
"""
compute_memory_curves_persuade.py

Step 4: Compute memory curves for PERSUADE cohorts.

For each essay, computes perplexity at multiple context windows on a fixed
scoring region. This produces the data needed for half-life analysis.

CRITICAL: The same target region is scored across all context windows.
This is what makes the half-life metric meaningful.

Usage (Colab):
    !python compute_memory_curves_persuade.py \
        --input cohorts/persuade_score_cohort.jsonl \
        --output results/persuade_score/ \
        --experiment score

Usage (local, CPU-only for testing):
    python scripts/compute_memory_curves_persuade.py \
        --input data/persuade_clean/cohorts/persuade_score_cohort.jsonl \
        --output results/test/ \
        --max-essays 5 \
        --no-quantize
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    'model_name': 'mistralai/Mistral-7B-v0.1',
    'windows': [32, 64, 128, 256, 384, 512, 768, 1024],
    'burn_in': 128,
    'max_score_tokens': 512,  # Cap on tokens to score (for speed)
    'batch_size': 1,  # Essays processed at once (keep at 1 for simplicity)
    'use_4bit': True,
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name: str, use_4bit: bool = True, device: str = 'auto'):
    """Load model and tokenizer.

    Args:
        model_name: HuggingFace model name
        use_4bit: Whether to use 4-bit quantization
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        model, tokenizer, device_str
    """
    print(f"Loading model: {model_name}")
    print(f"  4-bit quantization: {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        device_str = "cuda"
    else:
        # CPU or non-quantized
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            device_str = "cuda"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            device_str = "cpu"
            model = model.to(device_str)

    model.eval()

    print(f"  Device: {device_str}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return model, tokenizer, device_str


# ============================================================
# PERPLEXITY COMPUTATION
# ============================================================

@torch.no_grad()
def compute_perplexity_on_region(
    model,
    token_ids: list[int],
    target_start: int,
    target_end: int,
    device: str,
) -> tuple[float, float, int]:
    """Compute perplexity on a specific region of tokens.

    Context before target_start is provided but not scored.
    Only tokens in [target_start, target_end) are scored.

    Args:
        model: The language model
        token_ids: Full sequence of token IDs
        target_start: Start index of scoring region
        target_end: End index of scoring region
        device: Device string

    Returns:
        (perplexity, avg_loss, num_scored_tokens)
    """
    if target_end > len(token_ids):
        target_end = len(token_ids)

    if target_start >= target_end - 1:
        return float('inf'), float('inf'), 0

    # Create input tensor
    input_ids = torch.tensor([token_ids], device=device)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

    # Compute loss on target region
    total_loss = 0.0
    count = 0

    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        target_token = token_ids[i + 1]
        token_loss = -log_probs[target_token].item()
        total_loss += token_loss
        count += 1

    if count == 0:
        return float('inf'), float('inf'), 0

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss, count


# ============================================================
# MEMORY CURVE COMPUTATION
# ============================================================

def compute_memory_curve(
    model,
    tokenizer,
    text: str,
    windows: list[int],
    burn_in: int,
    max_score_tokens: int,
    device: str,
) -> dict:
    """Compute memory curve for a single essay.

    Scoring protocol:
    - Tokenize the full essay
    - Define target region as tokens after burn_in (up to max_score_tokens)
    - For each window size W, provide context [burn_in-W : burn_in] before target
    - Compute perplexity on the SAME target region for all windows

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Essay text
        windows: List of context window sizes to test
        burn_in: Number of tokens before target region starts
        max_score_tokens: Maximum tokens to score in target region
        device: Device string

    Returns:
        Dictionary with:
        - ppl_by_W: {window: perplexity}
        - computed_windows: List of windows actually computed
        - token_count: Total tokens in essay
        - target_tokens: Number of tokens in target region
        - errors: Any error messages
    """
    result = {
        'ppl_by_W': {},
        'computed_windows': [],
        'token_count': 0,
        'target_tokens': 0,
        'errors': [],
    }

    # Tokenize
    try:
        full_tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        result['errors'].append(f"Tokenization error: {e}")
        return result

    n_tokens = len(full_tokens)
    result['token_count'] = n_tokens

    # Check if essay is long enough
    if n_tokens <= burn_in:
        result['errors'].append(f"Essay too short: {n_tokens} tokens <= burn_in {burn_in}")
        return result

    # Define target region
    target_start_in_full = burn_in
    target_end_in_full = min(n_tokens, burn_in + max_score_tokens)
    target_tokens = target_end_in_full - target_start_in_full
    result['target_tokens'] = target_tokens

    if target_tokens < 10:
        result['errors'].append(f"Target region too small: {target_tokens} tokens")
        return result

    # Compute perplexity for each window
    for W in windows:
        # Calculate the range of tokens to include
        # We want: context of W tokens + target region
        # Context ends at burn_in, so context starts at max(0, burn_in - W)
        context_start = max(0, burn_in - W)
        actual_context = burn_in - context_start

        # Build the truncated sequence
        truncated_tokens = full_tokens[context_start:target_end_in_full]

        # Adjust target indices for truncated sequence
        target_start = actual_context  # Target starts after context
        target_end = len(truncated_tokens)

        # Skip if we can't even fit minimal context
        if actual_context < 4:
            continue

        # Compute perplexity
        try:
            ppl, loss, n_scored = compute_perplexity_on_region(
                model, truncated_tokens, target_start, target_end, device
            )

            if not math.isinf(ppl):
                result['ppl_by_W'][W] = ppl
                result['computed_windows'].append(W)

        except Exception as e:
            result['errors'].append(f"Error at W={W}: {e}")

    return result


def compute_half_life(ppl_by_W: dict[int, float], percentile: float = 0.5) -> float:
    """Compute half-life from perplexity-by-window dictionary.

    Args:
        ppl_by_W: Dictionary mapping window size to perplexity
        percentile: Benefit percentile (0.5 = half-life)

    Returns:
        Window size at which percentile of benefit is achieved (interpolated)
    """
    if len(ppl_by_W) < 2:
        return float('nan')

    # Sort by window size
    items = sorted(ppl_by_W.items())
    windows = np.array([x[0] for x in items])
    ppls = np.array([x[1] for x in items])

    # Perplexity should decrease as window increases
    min_ppl = ppls[0]   # Smallest window = highest perplexity
    max_ppl = ppls[-1]  # Largest window = lowest perplexity

    total_benefit = min_ppl - max_ppl
    if total_benefit <= 0:
        return float('nan')

    target_ppl = min_ppl - percentile * total_benefit

    # Find crossing point
    for i in range(len(ppls) - 1):
        if ppls[i] >= target_ppl >= ppls[i + 1]:
            # Linear interpolation
            frac = (ppls[i] - target_ppl) / (ppls[i] - ppls[i + 1])
            return windows[i] + frac * (windows[i + 1] - windows[i])

    return windows[-1]


def compute_curve_features(ppl_by_W: dict[int, float]) -> dict:
    """Compute summary features from a memory curve.

    Args:
        ppl_by_W: Dictionary mapping window size to perplexity

    Returns:
        Dictionary with:
        - half_life: Window size at 50% benefit
        - delta_max: Total perplexity reduction (largest_W - smallest_W)
        - ppl_min_W: Perplexity at smallest window
        - ppl_max_W: Perplexity at largest window
        - slope_small: Slope from 32→128 (if available)
        - slope_large: Slope from 256→512 (if available)
    """
    features = {
        'half_life': float('nan'),
        'delta_max': float('nan'),
        'ppl_min_W': float('nan'),
        'ppl_max_W': float('nan'),
        'slope_small': float('nan'),
        'slope_large': float('nan'),
    }

    if len(ppl_by_W) < 2:
        return features

    # Sort by window
    items = sorted(ppl_by_W.items())
    windows = [x[0] for x in items]
    ppls = [x[1] for x in items]

    # Basic metrics
    features['ppl_min_W'] = ppls[0]
    features['ppl_max_W'] = ppls[-1]
    features['delta_max'] = ppls[0] - ppls[-1]

    # Half-life
    features['half_life'] = compute_half_life(ppl_by_W, 0.5)

    # Slopes at specific ranges
    ppl_dict = dict(items)

    # Slope small: 32→128
    if 32 in ppl_dict and 128 in ppl_dict:
        features['slope_small'] = (ppl_dict[32] - ppl_dict[128]) / (128 - 32)

    # Slope large: 256→512
    if 256 in ppl_dict and 512 in ppl_dict:
        features['slope_large'] = (ppl_dict[256] - ppl_dict[512]) / (512 - 256)

    return features


# ============================================================
# MAIN PROCESSING
# ============================================================

def load_cohort(path: Path) -> list[dict]:
    """Load cohort JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def process_cohort(
    cohort: list[dict],
    model,
    tokenizer,
    config: dict,
    device: str,
    max_essays: Optional[int] = None,
    progress_callback=None,
) -> list[dict]:
    """Process a cohort of essays.

    Args:
        cohort: List of essay records
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dictionary
        device: Device string
        max_essays: Maximum essays to process (for testing)
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of result records
    """
    windows = config['windows']
    burn_in = config['burn_in']
    max_score_tokens = config['max_score_tokens']

    if max_essays is not None:
        cohort = cohort[:max_essays]

    results = []
    n_total = len(cohort)

    for i, essay in enumerate(cohort):
        essay_id = essay.get('essay_id', f'essay_{i}')

        # Compute memory curve
        curve_result = compute_memory_curve(
            model=model,
            tokenizer=tokenizer,
            text=essay['text'],
            windows=windows,
            burn_in=burn_in,
            max_score_tokens=max_score_tokens,
            device=device,
        )

        # Compute features
        features = compute_curve_features(curve_result['ppl_by_W'])

        # Build result record
        result = {
            'essay_id': essay_id,
            'score': essay.get('score'),
            'grade': essay.get('grade'),
            'ell': essay.get('ell'),
            'prompt_id': essay.get('prompt_id'),
            'word_count': essay.get('word_count'),
            'token_count': curve_result['token_count'],
            'target_tokens': curve_result['target_tokens'],
            'computed_windows': curve_result['computed_windows'],
            'errors': curve_result['errors'],
            **features,
        }

        # Add individual window perplexities
        for W in windows:
            col = f'ppl_W{W}'
            result[col] = curve_result['ppl_by_W'].get(W, float('nan'))

        results.append(result)

        # Progress update
        if progress_callback:
            progress_callback(i + 1, n_total)
        elif (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i + 1}/{n_total} essays...")

    return results


def compute_group_summary(results: list[dict], group_field: str) -> pd.DataFrame:
    """Compute summary statistics by group.

    Args:
        results: List of result records
        group_field: Field to group by (e.g., 'score_bin', 'grade', 'ell')

    Returns:
        DataFrame with group-level statistics
    """
    df = pd.DataFrame(results)

    # Create score_bin if needed
    if group_field == 'score_bin' and 'score_bin' not in df.columns:
        df['score_bin'] = df['score'].apply(
            lambda x: 'low' if x <= 2 else ('mid' if x <= 4 else 'high')
            if pd.notna(x) else 'unk'
        )

    summary_rows = []

    for group_val in df[group_field].dropna().unique():
        group_df = df[df[group_field] == group_val]

        row = {
            'group': group_val,
            'n': len(group_df),
            'half_life_mean': group_df['half_life'].mean(),
            'half_life_std': group_df['half_life'].std(),
            'half_life_sem': group_df['half_life'].sem(),
            'delta_max_mean': group_df['delta_max'].mean(),
            'delta_max_std': group_df['delta_max'].std(),
            'ppl_min_W_mean': group_df['ppl_min_W'].mean(),
            'ppl_max_W_mean': group_df['ppl_max_W'].mean(),
        }

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description='Compute memory curves for PERSUADE cohorts'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input cohort JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='unknown',
        help='Experiment name (score, grade, ell, prompt)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_CONFIG['model_name'],
        help=f'Model name (default: {DEFAULT_CONFIG["model_name"]})'
    )
    parser.add_argument(
        '--windows',
        type=str,
        default=None,
        help='Comma-separated window sizes (default: 32,64,128,256,384,512,768,1024)'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        default=DEFAULT_CONFIG['burn_in'],
        help=f'Burn-in tokens (default: {DEFAULT_CONFIG["burn_in"]})'
    )
    parser.add_argument(
        '--max-score-tokens',
        type=int,
        default=DEFAULT_CONFIG['max_score_tokens'],
        help=f'Max tokens to score (default: {DEFAULT_CONFIG["max_score_tokens"]})'
    )
    parser.add_argument(
        '--max-essays',
        type=int,
        default=None,
        help='Maximum essays to process (for testing)'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable 4-bit quantization'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        default=None,
        help='Field to group summary by (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Build config
    config = dict(DEFAULT_CONFIG)
    config['model_name'] = args.model
    config['burn_in'] = args.burn_in
    config['max_score_tokens'] = args.max_score_tokens
    config['use_4bit'] = not args.no_quantize

    if args.windows:
        config['windows'] = [int(x) for x in args.windows.split(',')]

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load cohort
    print(f"Loading cohort: {args.input}")
    cohort = load_cohort(args.input)
    print(f"  Loaded {len(cohort)} essays")

    # Load model
    model, tokenizer, device = load_model(
        config['model_name'],
        use_4bit=config['use_4bit'],
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save config
    run_config = {
        'input_file': str(args.input),
        'experiment': args.experiment,
        'n_essays': len(cohort) if args.max_essays is None else min(len(cohort), args.max_essays),
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'device': device,
    }

    with open(args.output / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # Process
    print(f"\nProcessing {run_config['n_essays']} essays...")
    start_time = time.time()

    results = process_cohort(
        cohort=cohort,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        max_essays=args.max_essays,
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / len(results):.2f}s per essay)")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(args.output / 'essay_level_results.parquet', index=False)
    results_df.to_csv(args.output / 'essay_level_results.csv', index=False)
    print(f"Saved essay-level results to {args.output}")

    # Compute and save group summary
    group_field = args.group_by
    if group_field is None:
        # Auto-detect based on experiment name
        if args.experiment == 'score':
            group_field = 'score_bin'
        elif args.experiment == 'grade':
            group_field = 'grade'
        elif args.experiment == 'ell':
            group_field = 'ell'
        elif args.experiment == 'prompt':
            group_field = 'prompt_id'
        else:
            group_field = 'score_bin'  # Default

    summary_df = compute_group_summary(results, group_field)
    summary_df.to_csv(args.output / 'group_summary.csv', index=False)
    print(f"Saved group summary to {args.output / 'group_summary.csv'}")

    # Print summary
    print("\n" + "=" * 60)
    print("GROUP SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Quick stats
    valid_results = [r for r in results if not math.isnan(r['half_life'])]
    if valid_results:
        half_lives = [r['half_life'] for r in valid_results]
        print(f"\nOverall half-life: {np.mean(half_lives):.1f} ± {np.std(half_lives):.1f} tokens")
        print(f"  Range: {np.min(half_lives):.0f} - {np.max(half_lives):.0f}")

    # Errors
    n_errors = sum(1 for r in results if r['errors'])
    if n_errors > 0:
        print(f"\nWarning: {n_errors} essays had errors")


if __name__ == '__main__':
    main()
