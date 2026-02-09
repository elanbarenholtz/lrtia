#!/usr/bin/env python3
"""
Effect Size Analysis for Universal Kernel Metrics.

Computes Cohen's d and bootstrap CIs for age-bin contrasts
on log_slope, auc_log_k, and gain_128.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/universal_kernel_v1")

# Config
METRICS = ['log_slope_local', 'auc_log_k', 'mean_gain_128']
METRIC_LABELS = {'log_slope_local': 'log_slope', 'auc_log_k': 'auc_log_k', 'mean_gain_128': 'gain_128'}
AGE_GROUPS = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
N_BOOTSTRAP = 10000
np.random.seed(42)

print("=" * 70)
print("EFFECT SIZE ANALYSIS: AGE-BIN CONTRASTS")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")

def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD version)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def bootstrap_cohens_d(group1, group2, n_bootstrap=10000):
    """Bootstrap CI for Cohen's d."""
    d_samples = []
    n1, n2 = len(group1), len(group2)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx1 = np.random.randint(0, n1, n1)
        idx2 = np.random.randint(0, n2, n2)
        boot1 = group1[idx1]
        boot2 = group2[idx2]
        d_samples.append(cohens_d(boot1, boot2))

    d_samples = np.array(d_samples)
    ci_low = np.percentile(d_samples, 2.5)
    ci_high = np.percentile(d_samples, 97.5)
    return ci_low, ci_high

# Compute effect sizes for each metric
results = []

for metric in METRICS:
    label = METRIC_LABELS[metric]
    print(f"\n{'='*60}")
    print(f"METRIC: {label}")
    print(f"{'='*60}")

    # Youngest vs oldest comparison
    young = df[df['age_group'] == '5yr'][metric].values
    old = df[df['age_group'] == '10+yr'][metric].values

    d = cohens_d(old, young)  # Positive = older higher
    ci_low, ci_high = bootstrap_cohens_d(old, young, N_BOOTSTRAP)

    print(f"\n5yr vs 10+yr (developmental span):")
    print(f"  5yr:   mean={np.mean(young):.4f}, SD={np.std(young):.4f}, n={len(young)}")
    print(f"  10+yr: mean={np.mean(old):.4f}, SD={np.std(old):.4f}, n={len(old)}")
    print(f"  Cohen's d = {d:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # Interpret
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    print(f"  Interpretation: {interp} effect")

    results.append({
        'metric': label,
        'contrast': '5yr_vs_10+yr',
        'group1': '10+yr',
        'group2': '5yr',
        'n1': len(old),
        'n2': len(young),
        'mean1': np.mean(old),
        'mean2': np.mean(young),
        'cohens_d': d,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'interpretation': interp
    })

    # Adjacent age comparisons
    print(f"\nAdjacent age-bin contrasts:")
    print(f"{'Contrast':<20} {'d':>8} {'95% CI':>20} {'Interp':<12}")
    print("-" * 65)

    for i in range(len(AGE_GROUPS) - 1):
        ag1, ag2 = AGE_GROUPS[i], AGE_GROUPS[i+1]
        g1 = df[df['age_group'] == ag1][metric].values
        g2 = df[df['age_group'] == ag2][metric].values

        d_adj = cohens_d(g2, g1)  # Positive = older higher
        ci_l, ci_h = bootstrap_cohens_d(g2, g1, N_BOOTSTRAP)

        if abs(d_adj) < 0.2:
            interp_adj = "negligible"
        elif abs(d_adj) < 0.5:
            interp_adj = "small"
        elif abs(d_adj) < 0.8:
            interp_adj = "medium"
        else:
            interp_adj = "large"

        print(f"{ag1} vs {ag2:<12} {d_adj:>8.3f} [{ci_l:>7.3f}, {ci_h:>7.3f}] {interp_adj:<12}")

        results.append({
            'metric': label,
            'contrast': f'{ag1}_vs_{ag2}',
            'group1': ag2,
            'group2': ag1,
            'n1': len(g2),
            'n2': len(g1),
            'mean1': np.mean(g2),
            'mean2': np.mean(g1),
            'cohens_d': d_adj,
            'ci_low': ci_l,
            'ci_high': ci_h,
            'interpretation': interp_adj
        })

# Save results
results_df = pd.DataFrame(results)
output_path = OUTPUT_DIR / 'effect_sizes_age_contrasts.csv'
results_df.to_csv(output_path, index=False)
print(f"\n\nSaved: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: IS DEVELOPMENTAL DRIFT PRACTICALLY MEANINGFUL?")
print("=" * 70)

for metric in ['log_slope', 'auc_log_k', 'gain_128']:
    span = results_df[(results_df['metric'] == metric) & (results_df['contrast'] == '5yr_vs_10+yr')].iloc[0]
    print(f"\n{metric}:")
    print(f"  5yr→10+yr effect: d = {span['cohens_d']:.3f} [{span['ci_low']:.3f}, {span['ci_high']:.3f}]")
    print(f"  Practical significance: {span['interpretation'].upper()}")

    # Check if CI crosses zero
    if span['ci_low'] <= 0 <= span['ci_high']:
        print(f"  Note: CI includes zero - effect direction uncertain")
    else:
        direction = "increases" if span['cohens_d'] > 0 else "decreases"
        print(f"  Direction: {direction} with age")
