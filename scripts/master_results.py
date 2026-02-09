"""
master_results.py

Auto-update master results in Google Drive after each Colab run.

Usage (add to end of any analysis notebook):

    from master_results import update_master_from_run

    update_master_from_run(
        run_prefix='ecsc_v4.2_qwen7b',
        config=config_dict,
        endofdoc_summary=endofdoc_summary_df,
        burst_summary_256=burst_summaries[256],
        burst_summary_512=burst_summaries[512],
        anchor_summary=anchor_summary_df,
        metrics_df=metrics_df,
        drive_root='/content/drive/MyDrive/LRTIA'
    )
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# ECSC age bins
ECSC_AGE_BINS = ['4-6', '6-8', '8-10', '10+']

# Miami Mono age bins (grade-based)
MIAMI_AGE_BINS = ['grade2', 'grade5']

# Combined for ordering
ALL_AGE_BINS = ECSC_AGE_BINS + MIAMI_AGE_BINS
AGE_BIN_ORDER = {b: i for i, b in enumerate(ALL_AGE_BINS)}

AGE_BIN_NORMALIZE = {
    # ECSC format
    '4-6yr': '4-6', '6-8yr': '6-8', '8-10yr': '8-10', '10+yr': '10+',
    '4-6': '4-6', '6-8': '6-8', '8-10': '8-10', '10+': '10+',
    # Miami Mono format
    'grade2': 'grade2', 'grade5': 'grade5',
    'mleng2': 'grade2', 'mleng5': 'grade5',
}

EXPECTED_CONFIG = {
    'K_TARGETS': 20,
    'TARGET_REGION_TOKENS': 40,
    'SHORT_CONTEXT': 32,
}


def normalize_age_bin(age_bin: str) -> str:
    return AGE_BIN_NORMALIZE.get(age_bin, age_bin)


def parse_prefix(prefix: str) -> Dict[str, str]:
    parts = prefix.split('_')
    return {
        'dataset': parts[0].upper() if len(parts) > 0 else 'UNKNOWN',
        'version': parts[1] if len(parts) > 1 else 'v0',
        'modeltag': '_'.join(parts[2:]) if len(parts) > 2 else 'unknown',
    }


def parse_model_info(model_name: str) -> Dict[str, str]:
    model_lower = model_name.lower()

    if 'mistral' in model_lower:
        family = 'mistral'
    elif 'qwen' in model_lower:
        family = 'qwen'
    elif 'llama' in model_lower:
        family = 'llama'
    else:
        family = 'other'

    size_match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    size = size_match.group(0).upper() if size_match else 'unknown'

    return {'model_family': family, 'model_size': size}


# ============================================================
# EXTRACT METRICS FROM DATAFRAMES
# ============================================================

def extract_endofdoc_metrics(df: Optional[pd.DataFrame], age_bin: str) -> Dict:
    metrics = {
        'half_life_512_mean': np.nan, 'half_life_512_ci_low': np.nan, 'half_life_512_ci_high': np.nan,
        'n_endofdoc': 0,
        'early_drop_pct_512_mean': np.nan, 'early_drop_pct_512_ci_low': np.nan, 'early_drop_pct_512_ci_high': np.nan,
        'early_slope_mean': np.nan, 'early_slope_ci_low': np.nan, 'early_slope_ci_high': np.nan,
        'ppl_min_ctx_mean': np.nan, 'ppl_min_ctx_ci_low': np.nan, 'ppl_min_ctx_ci_high': np.nan,
    }

    if df is None or len(df) == 0:
        return metrics

    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

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
    prefix = f'b{L}_'
    metrics = {
        f'{prefix}baseline_mean': np.nan, f'{prefix}baseline_ci_low': np.nan, f'{prefix}baseline_ci_high': np.nan,
        f'{prefix}n_valid': 0,
        f'{prefix}spike_mass_c_mean': np.nan, f'{prefix}spike_mass_c_ci_low': np.nan, f'{prefix}spike_mass_c_ci_high': np.nan,
        f'{prefix}cv_c_mean': np.nan, f'{prefix}cv_c_ci_low': np.nan, f'{prefix}cv_c_ci_high': np.nan,
        f'{prefix}kurtosis_c_mean': np.nan, f'{prefix}kurtosis_c_ci_low': np.nan, f'{prefix}kurtosis_c_ci_high': np.nan,
        f'{prefix}n_targets_median': np.nan,
    }

    if df is None or len(df) == 0:
        return metrics

    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

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
    metrics = {}
    for L in [256, 512]:
        prefix = f'a{L}_'
        metrics.update({
            f'{prefix}top_share_mean': np.nan, f'{prefix}top_share_ci_low': np.nan, f'{prefix}top_share_ci_high': np.nan,
            f'{prefix}n_valid': 0,
            f'{prefix}n_eff_mean': np.nan, f'{prefix}n_eff_ci_low': np.nan, f'{prefix}n_eff_ci_high': np.nan,
        })

    if df is None or len(df) == 0:
        return metrics

    mask = df['age_group'].apply(normalize_age_bin) == age_bin
    if not mask.any():
        return metrics

    row = df[mask].iloc[0]

    col_map = {
        'a512_top_share_mean': ['top_share_512_mean', 'top_share_mean'],
        'a512_top_share_ci_low': ['top_share_512_ci_low', 'top_share_ci_low'],
        'a512_top_share_ci_high': ['top_share_512_ci_high', 'top_share_ci_high'],
        'a512_n_valid': ['n'],
        'a512_n_eff_mean': ['n_eff_512_mean', 'n_eff_mean'],
        'a512_n_eff_ci_low': ['n_eff_512_ci_low', 'n_eff_ci_low'],
        'a512_n_eff_ci_high': ['n_eff_512_ci_high', 'n_eff_ci_high'],
        'a256_top_share_mean': ['top_share_256_mean'],
        'a256_n_valid': ['n_256'],
        'a256_n_eff_mean': ['n_eff_256_mean'],
    }

    for out_col, in_cols in col_map.items():
        for in_col in in_cols:
            if in_col in row.index:
                metrics[out_col] = row[in_col]
                break

    return metrics


# ============================================================
# MAIN UPDATE FUNCTION
# ============================================================

def update_master_from_run(
    run_prefix: str,
    config: Dict,
    endofdoc_summary: Optional[pd.DataFrame] = None,
    burst_summary_256: Optional[pd.DataFrame] = None,
    burst_summary_512: Optional[pd.DataFrame] = None,
    anchor_summary: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    drive_root: str = '/content/drive/MyDrive/LRTIA',
) -> None:
    """
    Update master results files in Google Drive after a run.

    Call this at the end of each analysis notebook.
    """

    print("\n" + "=" * 60)
    print("UPDATING MASTER RESULTS")
    print("=" * 60)

    master_dir = f"{drive_root}/results"
    os.makedirs(master_dir, exist_ok=True)

    master_runs_path = f"{master_dir}/master_runs.csv"
    master_agebin_path = f"{master_dir}/master_agebin.csv"

    # Parse run info
    parsed = parse_prefix(run_prefix)
    model_name = config.get('MODEL_NAME', config.get('model_name', 'unknown'))
    model_info = parse_model_info(model_name)

    # Build run_id (prefix + date)
    date_str = datetime.now().strftime('%Y%m%d')
    run_id = f"{run_prefix}_{date_str}"

    # Compute sample sizes from metrics_df
    n_docs_total = len(metrics_df) if metrics_df is not None else 0
    n_docs_burst256_valid = 0
    n_docs_burst512_valid = 0

    if metrics_df is not None:
        if 'burst_valid_256' in metrics_df.columns:
            n_docs_burst256_valid = int(metrics_df['burst_valid_256'].sum())
        if 'burst_valid_512' in metrics_df.columns:
            n_docs_burst512_valid = int(metrics_df['burst_valid_512'].sum())

    n_docs_anchor512_valid = len(anchor_summary) if anchor_summary is not None else 0

    # Build run row
    run_row = {
        'run_id': run_id,
        'dataset': parsed['dataset'],
        'version': parsed['version'],
        'model_name': model_name,
        'model_tag': parsed['modeltag'],
        'model_family': model_info['model_family'],
        'model_size': model_info['model_size'],
        'precision': config.get('QUANTIZATION', config.get('precision', 'unknown')),
        'short_context': config.get('SHORT_CONTEXT', 32),
        'long_contexts': str(config.get('LONG_CONTEXTS', [512, 256])),
        'k_targets': config.get('K_TARGETS', 20),
        'min_words': config.get('MIN_WORDS', 200),
        'bootstrap_n': config.get('N_BOOTSTRAP', 1000),
        'date_run': datetime.now().isoformat(),
        'n_docs_total': n_docs_total,
        'n_docs_burst256_valid': n_docs_burst256_valid,
        'n_docs_burst512_valid': n_docs_burst512_valid,
        'n_docs_anchor512_valid': n_docs_anchor512_valid,
    }

    # Build agebin rows
    age_bins = set()
    for df in [endofdoc_summary, burst_summary_256, burst_summary_512, anchor_summary]:
        if df is not None and 'age_group' in df.columns:
            age_bins.update(df['age_group'].apply(normalize_age_bin).unique())
    age_bins.update(STANDARD_AGE_BINS)

    agebin_rows = []
    for age_bin in sorted(age_bins, key=lambda x: AGE_BIN_ORDER.get(x, 99)):
        row = {
            'run_id': run_id,
            'dataset': parsed['dataset'],
            'version': parsed['version'],
            'model_name': model_name,
            'model_tag': parsed['modeltag'],
            'precision': config.get('QUANTIZATION', 'unknown'),
            'age_bin': age_bin,
            'age_bin_order': AGE_BIN_ORDER.get(age_bin, 99),
        }

        row.update(extract_endofdoc_metrics(endofdoc_summary, age_bin))
        row.update(extract_burst_metrics(burst_summary_256, age_bin, 256))
        row.update(extract_burst_metrics(burst_summary_512, age_bin, 512))
        row.update(extract_anchor_metrics(anchor_summary, age_bin))

        agebin_rows.append(row)

    # Load existing master files
    if os.path.exists(master_runs_path):
        existing_runs = pd.read_csv(master_runs_path)
        # Remove rows with same prefix (keep date suffix for uniqueness check)
        existing_runs = existing_runs[
            ~existing_runs['run_id'].str.startswith(run_prefix + '_')
        ]
        runs_df = pd.concat([existing_runs, pd.DataFrame([run_row])], ignore_index=True)
    else:
        runs_df = pd.DataFrame([run_row])

    if os.path.exists(master_agebin_path):
        existing_agebin = pd.read_csv(master_agebin_path)
        existing_agebin = existing_agebin[
            ~existing_agebin['run_id'].str.startswith(run_prefix + '_')
        ]
        agebin_df = pd.concat([existing_agebin, pd.DataFrame(agebin_rows)], ignore_index=True)
    else:
        agebin_df = pd.DataFrame(agebin_rows)

    # Sort and save
    runs_df = runs_df.sort_values(['dataset', 'model_family', 'date_run'])
    agebin_df = agebin_df.sort_values(['dataset', 'model_name', 'age_bin_order'])

    runs_df.to_csv(master_runs_path, index=False)
    agebin_df.to_csv(master_agebin_path, index=False)

    # Also save parquet
    try:
        runs_df.to_parquet(f"{master_dir}/master_runs.parquet", index=False)
        agebin_df.to_parquet(f"{master_dir}/master_agebin.parquet", index=False)
    except:
        pass  # Parquet optional

    # Print summary
    print(f"\nRun: {run_id}")
    print(f"  n_docs_total: {n_docs_total}")
    print(f"  n_docs_burst512_valid: {n_docs_burst512_valid}")
    print(f"  n_docs_anchor512_valid: {n_docs_anchor512_valid}")
    print(f"\nMaster files updated:")
    print(f"  {master_runs_path} ({len(runs_df)} runs)")
    print(f"  {master_agebin_path} ({len(agebin_df)} rows)")

    # Config validation
    warnings = []
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual is not None and actual != expected:
            warnings.append(f"  {key}: expected {expected}, got {actual}")

    if warnings:
        print("\nConfig warnings:")
        for w in warnings:
            print(w)

    print("\n" + "=" * 60)
