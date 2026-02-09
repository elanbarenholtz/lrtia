#!/usr/bin/env python3
"""
update_master_results.py

Maintains master CSV files tracking all LRTIA runs across datasets and models.

Outputs:
  - master_runs.csv: One row per run (metadata, paths, sample sizes)
  - master_agebin.csv: One row per run × age_bin (all metrics for plotting)

Usage:
  python update_master_results.py --results-dir /path/to/results
  python update_master_results.py --run-dirs ecsc_v4.2_mistral7b ecsc_v4.2_qwen7b
  python update_master_results.py --scan  # Auto-scan results directory
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Standard age bins (normalize all inputs to these)
STANDARD_AGE_BINS = ['4-6', '6-8', '8-10', '10+']
AGE_BIN_ORDER = {b: i for i, b in enumerate(STANDARD_AGE_BINS)}

# Age bin normalization map
AGE_BIN_NORMALIZE = {
    '4-6yr': '4-6',
    '6-8yr': '6-8',
    '8-10yr': '8-10',
    '10+yr': '10+',
    '4-6': '4-6',
    '6-8': '6-8',
    '8-10': '8-10',
    '10+': '10+',
}

# Expected config values for validation
EXPECTED_CONFIG = {
    'K_TARGETS': 20,
    'TARGET_REGION_TOKENS': 40,
    'SHORT_CONTEXT': 32,
    'LONG_CONTEXTS': [512, 256],
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_prefix(prefix: str) -> Dict[str, str]:
    """Parse run prefix into components.

    Expected format: {dataset}_{version}_{modeltag}
    Examples: ecsc_v4.2_mistral7b, frog_v1.0_qwen7b
    """
    parts = prefix.split('_')
    if len(parts) >= 3:
        dataset = parts[0]
        version = parts[1]
        modeltag = '_'.join(parts[2:])  # Handle tags like "llama3_8b"
    else:
        dataset = parts[0] if len(parts) > 0 else 'unknown'
        version = parts[1] if len(parts) > 1 else 'unknown'
        modeltag = parts[2] if len(parts) > 2 else 'unknown'

    return {
        'dataset': dataset.upper(),
        'version': version,
        'modeltag': modeltag,
    }


def parse_model_info(model_name: str, modeltag: str) -> Dict[str, str]:
    """Extract model family and size from model name."""
    model_name_lower = model_name.lower()

    # Determine family
    if 'mistral' in model_name_lower:
        family = 'mistral'
    elif 'qwen' in model_name_lower:
        family = 'qwen'
    elif 'llama' in model_name_lower:
        family = 'llama'
    elif 'phi' in model_name_lower:
        family = 'phi'
    else:
        family = 'other'

    # Extract size (e.g., 7B, 8B, 13B)
    size_match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    size = size_match.group(0).upper() if size_match else 'unknown'

    return {
        'model_family': family,
        'model_size': size,
    }


def normalize_age_bin(age_bin: str) -> str:
    """Normalize age bin label to standard format."""
    return AGE_BIN_NORMALIZE.get(age_bin, age_bin)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Safely read CSV, returning None if not found."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  Warning: Could not read {path}: {e}")
    return None


def safe_read_json(path: Path) -> Optional[Dict]:
    """Safely read JSON, returning None if not found."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not read {path}: {e}")
    return None


# ============================================================
# RUN PROCESSING
# ============================================================

def find_run_files(run_dir: Path, prefix: str) -> Dict[str, Path]:
    """Find all output files for a run."""
    files = {}

    # Config
    for pattern in [f'{prefix}_config.json', 'config.json']:
        p = run_dir / pattern
        if p.exists():
            files['config'] = p
            break

    # Summary files
    file_patterns = {
        'endofdoc_summary': f'{prefix}_endofdoc_summary.csv',
        'burst_summary_256': f'{prefix}_burst_summary_256.csv',
        'burst_summary_512': f'{prefix}_burst_summary_512.csv',
        'anchor_summary': f'{prefix}_anchor_summary.csv',
        'anchor': f'{prefix}_anchor.csv',
        'metrics': f'{prefix}_metrics.csv',
    }

    for key, pattern in file_patterns.items():
        p = run_dir / pattern
        if p.exists():
            files[key] = p

    return files


def process_run(run_dir: Path, prefix: str) -> Tuple[Dict, List[Dict]]:
    """Process a single run, returning (run_row, agebin_rows)."""

    print(f"\nProcessing: {prefix}")
    parsed = parse_prefix(prefix)
    files = find_run_files(run_dir, prefix)

    # Load config
    config = safe_read_json(files.get('config')) or {}

    # Parse model info
    model_name = config.get('MODEL_NAME', config.get('model_name', 'unknown'))
    model_info = parse_model_info(model_name, parsed['modeltag'])

    # Build run_id
    timestamp = datetime.now().strftime('%Y%m%d')
    run_id = f"{prefix}_{timestamp}"

    # Load per-doc files for counts
    metrics_df = safe_read_csv(files.get('metrics'))
    anchor_df = safe_read_csv(files.get('anchor'))

    # Compute sample sizes
    n_docs_total = len(metrics_df) if metrics_df is not None else 0
    n_docs_endofdoc = n_docs_total  # Assume all docs get endofdoc analysis
    n_docs_burst256_valid = 0
    n_docs_burst512_valid = 0
    n_docs_anchor512_valid = 0

    if metrics_df is not None:
        if 'burst_valid_256' in metrics_df.columns:
            n_docs_burst256_valid = metrics_df['burst_valid_256'].sum()
        if 'burst_valid_512' in metrics_df.columns:
            n_docs_burst512_valid = metrics_df['burst_valid_512'].sum()

    if anchor_df is not None:
        n_docs_anchor512_valid = len(anchor_df)

    # Build run row
    run_row = {
        'run_id': run_id,
        'dataset': parsed['dataset'],
        'task': config.get('task', 'narrative'),
        'model_name': model_name,
        'model_family': model_info['model_family'],
        'model_size': model_info['model_size'],
        'precision': config.get('QUANTIZATION', config.get('precision', 'unknown')),
        'context_fixed': config.get('MAX_CONTEXT_FIXED', 512),
        'short_context': config.get('SHORT_CONTEXT', 32),
        'long_contexts': str(config.get('LONG_CONTEXTS', [512, 256])),
        'target_region_tokens': config.get('TARGET_REGION_TOKENS', 40),
        'k_targets': config.get('K_TARGETS', 20),
        'min_words': config.get('MIN_WORDS', 200),
        'bootstrap_n': config.get('N_BOOTSTRAP', 1000),
        'git_commit': get_git_commit(),
        'date_run': datetime.now().isoformat(),
        'endofdoc_summary_path': str(files.get('endofdoc_summary', '')),
        'burst256_path': str(files.get('burst_summary_256', '')),
        'burst512_path': str(files.get('burst_summary_512', '')),
        'anchor_summary_path': str(files.get('anchor_summary', '')),
        'anchor_path': str(files.get('anchor', '')),
        'metrics_path': str(files.get('metrics', '')),
        'n_docs_total': n_docs_total,
        'n_docs_endofdoc': n_docs_endofdoc,
        'n_docs_burst256_valid': n_docs_burst256_valid,
        'n_docs_burst512_valid': n_docs_burst512_valid,
        'n_docs_anchor512_valid': n_docs_anchor512_valid,
    }

    # Build agebin rows
    agebin_rows = build_agebin_rows(run_id, parsed, model_name,
                                     config.get('QUANTIZATION', 'unknown'),
                                     files)

    # Validation
    validate_config(prefix, config)

    return run_row, agebin_rows


def build_agebin_rows(run_id: str, parsed: Dict, model_name: str,
                      precision: str, files: Dict[str, Path]) -> List[Dict]:
    """Build age-bin level rows from summary files."""

    rows = []

    # Load summary files
    endofdoc_df = safe_read_csv(files.get('endofdoc_summary'))
    burst256_df = safe_read_csv(files.get('burst_summary_256'))
    burst512_df = safe_read_csv(files.get('burst_summary_512'))
    anchor_df = safe_read_csv(files.get('anchor_summary'))

    # Get all age bins from available data
    age_bins = set()
    for df in [endofdoc_df, burst256_df, burst512_df, anchor_df]:
        if df is not None and 'age_group' in df.columns:
            age_bins.update(df['age_group'].unique())

    # Normalize and ensure we have all standard bins
    age_bins = {normalize_age_bin(b) for b in age_bins}
    age_bins.update(STANDARD_AGE_BINS)

    for age_bin in sorted(age_bins, key=lambda x: AGE_BIN_ORDER.get(x, 99)):
        row = {
            'run_id': run_id,
            'dataset': parsed['dataset'],
            'model_name': model_name,
            'precision': precision,
            'age_bin': age_bin,
            'age_bin_order': AGE_BIN_ORDER.get(age_bin, 99),
        }

        # Add end-of-doc metrics
        row.update(extract_endofdoc_metrics(endofdoc_df, age_bin))

        # Add burstiness metrics (256 and 512)
        row.update(extract_burst_metrics(burst256_df, age_bin, 256))
        row.update(extract_burst_metrics(burst512_df, age_bin, 512))

        # Add anchor metrics
        row.update(extract_anchor_metrics(anchor_df, age_bin))

        rows.append(row)

    return rows


def extract_endofdoc_metrics(df: Optional[pd.DataFrame], age_bin: str) -> Dict:
    """Extract end-of-doc metrics for an age bin."""
    prefix = ''
    metrics = {
        f'half_life_512_mean': np.nan,
        f'half_life_512_ci_low': np.nan,
        f'half_life_512_ci_high': np.nan,
        f'n_endofdoc': 0,
        f'early_drop_pct_512_mean': np.nan,
        f'early_drop_pct_512_ci_low': np.nan,
        f'early_drop_pct_512_ci_high': np.nan,
        f'early_slope_mean': np.nan,
        f'early_slope_ci_low': np.nan,
        f'early_slope_ci_high': np.nan,
        f'ppl_min_ctx_mean': np.nan,
        f'ppl_min_ctx_ci_low': np.nan,
        f'ppl_min_ctx_ci_high': np.nan,
    }

    if df is None:
        return metrics

    # Find matching row (handle both normalized and original labels)
    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

    # Map columns (handle different naming conventions)
    col_map = {
        'half_life_512_mean': ['half_life_fixed_mean', 'half_life_mean'],
        'half_life_512_ci_low': ['half_life_fixed_ci_low', 'half_life_ci_low'],
        'half_life_512_ci_high': ['half_life_fixed_ci_high', 'half_life_ci_high'],
        'n_endofdoc': ['n'],
        'early_drop_pct_512_mean': ['early_drop_pct_fixed_mean', 'early_drop_pct_mean'],
        'early_drop_pct_512_ci_low': ['early_drop_pct_fixed_ci_low', 'early_drop_pct_ci_low'],
        'early_drop_pct_512_ci_high': ['early_drop_pct_fixed_ci_high', 'early_drop_pct_ci_high'],
        'early_slope_mean': ['early_slope_mean'],
        'early_slope_ci_low': ['early_slope_ci_low'],
        'early_slope_ci_high': ['early_slope_ci_high'],
        'ppl_min_ctx_mean': ['ppl_min_ctx_mean'],
        'ppl_min_ctx_ci_low': ['ppl_min_ctx_ci_low'],
        'ppl_min_ctx_ci_high': ['ppl_min_ctx_ci_high'],
    }

    for out_col, in_cols in col_map.items():
        for in_col in in_cols:
            if in_col in row.index:
                metrics[out_col] = row[in_col]
                break

    return metrics


def extract_burst_metrics(df: Optional[pd.DataFrame], age_bin: str, L: int) -> Dict:
    """Extract burstiness metrics for an age bin and context length."""
    prefix = f'b{L}_'
    metrics = {
        f'{prefix}baseline_mean': np.nan,
        f'{prefix}baseline_ci_low': np.nan,
        f'{prefix}baseline_ci_high': np.nan,
        f'{prefix}n_valid': 0,
        f'{prefix}spike_mass_c_mean': np.nan,
        f'{prefix}spike_mass_c_ci_low': np.nan,
        f'{prefix}spike_mass_c_ci_high': np.nan,
        f'{prefix}cv_c_mean': np.nan,
        f'{prefix}cv_c_ci_low': np.nan,
        f'{prefix}cv_c_ci_high': np.nan,
        f'{prefix}kurtosis_c_mean': np.nan,
        f'{prefix}kurtosis_c_ci_low': np.nan,
        f'{prefix}kurtosis_c_ci_high': np.nan,
        f'{prefix}n_targets_median': np.nan,
    }

    if df is None:
        return metrics

    # Find matching row
    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

    # Map columns
    col_map = {
        f'{prefix}baseline_mean': ['baseline_mean'],
        f'{prefix}baseline_ci_low': ['baseline_ci_low'],
        f'{prefix}baseline_ci_high': ['baseline_ci_high'],
        f'{prefix}n_valid': ['n'],
        f'{prefix}spike_mass_c_mean': ['spike_mass_c_mean'],
        f'{prefix}spike_mass_c_ci_low': ['spike_mass_c_ci_low'],
        f'{prefix}spike_mass_c_ci_high': ['spike_mass_c_ci_high'],
        f'{prefix}cv_c_mean': ['cv_c_mean'],
        f'{prefix}cv_c_ci_low': ['cv_c_ci_low'],
        f'{prefix}cv_c_ci_high': ['cv_c_ci_high'],
        f'{prefix}kurtosis_c_mean': ['kurtosis_c_mean'],
        f'{prefix}kurtosis_c_ci_low': ['kurtosis_c_ci_low'],
        f'{prefix}kurtosis_c_ci_high': ['kurtosis_c_ci_high'],
        f'{prefix}n_targets_median': ['n_targets_median'],
    }

    for out_col, in_cols in col_map.items():
        for in_col in in_cols:
            if in_col in row.index:
                metrics[out_col] = row[in_col]
                break

    return metrics


def extract_anchor_metrics(df: Optional[pd.DataFrame], age_bin: str) -> Dict:
    """Extract anchor specificity metrics for an age bin."""
    metrics = {}

    # Initialize for both L values
    for L in [256, 512]:
        prefix = f'a{L}_'
        metrics.update({
            f'{prefix}top_share_mean': np.nan,
            f'{prefix}top_share_ci_low': np.nan,
            f'{prefix}top_share_ci_high': np.nan,
            f'{prefix}n_valid': 0,
            f'{prefix}n_eff_mean': np.nan,
            f'{prefix}n_eff_ci_low': np.nan,
            f'{prefix}n_eff_ci_high': np.nan,
        })

    if df is None:
        return metrics

    # Find matching row
    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

    # Map columns for 512
    col_map_512 = {
        'a512_top_share_mean': ['top_share_512_mean', 'top_share_mean'],
        'a512_top_share_ci_low': ['top_share_512_ci_low', 'top_share_ci_low'],
        'a512_top_share_ci_high': ['top_share_512_ci_high', 'top_share_ci_high'],
        'a512_n_valid': ['n'],
        'a512_n_eff_mean': ['n_eff_512_mean', 'n_eff_mean'],
        'a512_n_eff_ci_low': ['n_eff_512_ci_low', 'n_eff_ci_low'],
        'a512_n_eff_ci_high': ['n_eff_512_ci_high', 'n_eff_ci_high'],
    }

    # Map columns for 256
    col_map_256 = {
        'a256_top_share_mean': ['top_share_256_mean'],
        'a256_n_valid': ['n_256'],
        'a256_n_eff_mean': ['n_eff_256_mean'],
    }

    for out_col, in_cols in {**col_map_512, **col_map_256}.items():
        for in_col in in_cols:
            if in_col in row.index:
                metrics[out_col] = row[in_col]
                break

    return metrics


def validate_config(prefix: str, config: Dict) -> None:
    """Validate config against expected values."""
    warnings = []

    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual is not None and actual != expected:
            warnings.append(f"  {key}: expected {expected}, got {actual}")

    if warnings:
        print(f"  Config warnings for {prefix}:")
        for w in warnings:
            print(w)


# ============================================================
# MAIN
# ============================================================

def scan_results_dir(results_dir: Path) -> List[Tuple[Path, str]]:
    """Scan results directory for run folders."""
    runs = []

    # Look for directories containing config files
    for item in results_dir.rglob('*_config.json'):
        run_dir = item.parent
        # Extract prefix from config filename
        prefix = item.stem.replace('_config', '')
        runs.append((run_dir, prefix))

    return runs


def update_master_results(runs: List[Tuple[Path, str]], output_dir: Path) -> None:
    """Update master result files with data from runs."""

    all_run_rows = []
    all_agebin_rows = []

    for run_dir, prefix in runs:
        try:
            run_row, agebin_rows = process_run(run_dir, prefix)
            all_run_rows.append(run_row)
            all_agebin_rows.extend(agebin_rows)
        except Exception as e:
            print(f"  Error processing {prefix}: {e}")
            continue

    # Create DataFrames
    runs_df = pd.DataFrame(all_run_rows)
    agebin_df = pd.DataFrame(all_agebin_rows)

    # Load existing master files and merge (update existing run_ids)
    master_runs_path = output_dir / 'master_runs.csv'
    master_agebin_path = output_dir / 'master_agebin.csv'

    if master_runs_path.exists():
        existing_runs = pd.read_csv(master_runs_path)
        # Remove runs that are being updated
        new_run_ids = set(runs_df['run_id'].str.rsplit('_', n=1).str[0])
        existing_runs = existing_runs[
            ~existing_runs['run_id'].str.rsplit('_', n=1).str[0].isin(new_run_ids)
        ]
        runs_df = pd.concat([existing_runs, runs_df], ignore_index=True)

    if master_agebin_path.exists():
        existing_agebin = pd.read_csv(master_agebin_path)
        # Remove age bins for runs being updated
        new_run_ids = set(agebin_df['run_id'].str.rsplit('_', n=1).str[0])
        existing_agebin = existing_agebin[
            ~existing_agebin['run_id'].str.rsplit('_', n=1).str[0].isin(new_run_ids)
        ]
        agebin_df = pd.concat([existing_agebin, agebin_df], ignore_index=True)

    # Sort
    runs_df = runs_df.sort_values(['dataset', 'model_family', 'date_run'])
    agebin_df = agebin_df.sort_values(['dataset', 'model_name', 'age_bin_order'])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(master_runs_path, index=False)
    agebin_df.to_csv(master_agebin_path, index=False)

    # Also save parquet versions
    runs_df.to_parquet(output_dir / 'master_runs.parquet', index=False)
    agebin_df.to_parquet(output_dir / 'master_agebin.parquet', index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("MASTER RESULTS UPDATED")
    print("=" * 60)
    print(f"\nRuns processed: {len(all_run_rows)}")
    print(f"Total runs in master: {len(runs_df)}")
    print(f"Total age-bin rows: {len(agebin_df)}")
    print(f"\nSaved to:")
    print(f"  {master_runs_path}")
    print(f"  {master_agebin_path}")

    # Validation report
    print("\n" + "-" * 40)
    print("VALIDATION SUMMARY")
    print("-" * 40)
    for _, row in runs_df.iterrows():
        print(f"\n{row['run_id']}:")
        print(f"  n_docs_total: {row['n_docs_total']}")
        print(f"  n_docs_burst512_valid: {row['n_docs_burst512_valid']}")
        print(f"  n_docs_anchor512_valid: {row['n_docs_anchor512_valid']}")


def main():
    parser = argparse.ArgumentParser(description='Update master LRTIA results')
    parser.add_argument('--results-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/results'),
                        help='Base results directory')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/Users/elanbarenholtz/Projects/lrtia/results'),
                        help='Output directory for master files')
    parser.add_argument('--run-dirs', nargs='+', type=str,
                        help='Specific run prefixes to process')
    parser.add_argument('--scan', action='store_true',
                        help='Auto-scan results directory for runs')

    args = parser.parse_args()

    if args.run_dirs:
        # Process specific runs
        runs = []
        for prefix in args.run_dirs:
            # Try to find the run directory
            for candidate in [
                args.results_dir / prefix,
                args.results_dir / f"{prefix.split('_')[0]}" / prefix,  # e.g., results/ECSC/ecsc_v4.2_...
            ]:
                if candidate.exists():
                    runs.append((candidate, prefix))
                    break
            else:
                # Search for it
                found = list(args.results_dir.rglob(f'{prefix}_config.json'))
                if found:
                    runs.append((found[0].parent, prefix))
                else:
                    print(f"Warning: Could not find run {prefix}")
    else:
        # Scan for all runs
        runs = scan_results_dir(args.results_dir)

    if not runs:
        print("No runs found to process")
        return

    print(f"Found {len(runs)} runs to process:")
    for run_dir, prefix in runs:
        print(f"  {prefix} ({run_dir})")

    update_master_results(runs, args.output_dir)


if __name__ == '__main__':
    main()
