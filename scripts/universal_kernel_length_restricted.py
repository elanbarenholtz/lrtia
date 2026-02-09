#!/usr/bin/env python3
"""
Universal Kernel Analysis - Length-Restricted Subset.

Repeats kernel-collapse analysis on transcripts that support the full lag grid (k=128).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/universal_kernel_v1")

# Lag values
K_VALUES = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
GAIN_COLS = [f'mean_gain_{k}' for k in K_VALUES]

# Minimum token count to support k=128 lag
MIN_TOKENS = 128

print("=" * 70)
print("UNIVERSAL KERNEL ANALYSIS - LENGTH-RESTRICTED SUBSET")
print("=" * 70)

# Load and filter data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")

# Filter for transcripts supporting full lag grid
df_long = df[df['token_count'] >= MIN_TOKENS].copy()
print(f"After length filter (>= {MIN_TOKENS} tokens): {len(df_long)} transcripts")
print(f"Filtered out: {len(df) - len(df_long)} transcripts")

# Show per-age counts
print("\nPer-age counts after filter:")
age_groups = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
for ag in age_groups:
    n = (df_long['age_group'] == ag).sum()
    print(f"  {ag}: {n}")

# Extract curves
curve_matrix = df_long[GAIN_COLS].values
valid_mask = ~np.isnan(curve_matrix).any(axis=1)
df_valid = df_long[valid_mask].copy()
curves_valid = curve_matrix[valid_mask]
print(f"\nComplete curves: {len(df_valid)}")

# Normalize curves
max_gains = curves_valid[:, -1]  # gain_128
max_gains_safe = np.where(max_gains > 0.01, max_gains, 0.01)
curves_normalized = curves_valid / max_gains_safe[:, np.newaxis]

for i, k in enumerate(K_VALUES):
    df_valid[f'gain_norm_{k}'] = curves_normalized[:, i]

# ============================================================
# PLOT CURVE COLLAPSE
# ============================================================
print("\n" + "=" * 60)
print("PLOTTING CURVE COLLAPSE (LENGTH-RESTRICTED)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
k_array = np.array(K_VALUES[1:])  # Skip k=0

# Left: Raw curves
ax1 = axes[0]
for i, ag in enumerate(age_groups):
    ag_mask = df_valid['age_group'] == ag
    if ag_mask.sum() == 0:
        continue
    ag_curves = curves_valid[ag_mask, 1:]
    mean_curve = np.mean(ag_curves, axis=0)
    sem_curve = np.std(ag_curves, axis=0) / np.sqrt(np.sum(ag_mask))
    ax1.plot(k_array, mean_curve, color=colors[i], label=f'{ag} (n={np.sum(ag_mask)})', linewidth=2)
    ax1.fill_between(k_array, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve,
                     color=colors[i], alpha=0.2)

ax1.set_xscale('log')
ax1.set_xlabel('Lag k (tokens)', fontsize=12)
ax1.set_ylabel('Gain (NLL reduction)', fontsize=12)
ax1.set_title(f'Raw Gain Curves (tokens >= {MIN_TOKENS})', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Normalized curves
ax2 = axes[1]
for i, ag in enumerate(age_groups):
    ag_mask = df_valid['age_group'] == ag
    if ag_mask.sum() == 0:
        continue
    ag_curves_norm = curves_normalized[ag_mask, 1:]
    mean_curve = np.mean(ag_curves_norm, axis=0)
    sem_curve = np.std(ag_curves_norm, axis=0) / np.sqrt(np.sum(ag_mask))
    ax2.plot(k_array, mean_curve, color=colors[i], label=f'{ag}', linewidth=2)
    ax2.fill_between(k_array, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve,
                     color=colors[i], alpha=0.2)

ax2.set_xscale('log')
ax2.set_xlabel('Lag k (tokens)', fontsize=12)
ax2.set_ylabel('Normalized Gain (fraction of max)', fontsize=12)
ax2.set_title(f'Normalized Curves (tokens >= {MIN_TOKENS})', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kernel_collapse_length_restricted.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'kernel_collapse_length_restricted.png'}")

# ============================================================
# QUANTIFY SHAPE VARIATION
# ============================================================
print("\n" + "=" * 60)
print("CV REDUCTION ANALYSIS")
print("=" * 60)

collapse_stats = []
for i, k in enumerate(K_VALUES[1:], 1):
    raw_vals = curves_valid[:, i]
    norm_vals = curves_normalized[:, i]
    collapse_stats.append({
        'k': k,
        'raw_mean': np.mean(raw_vals),
        'raw_std': np.std(raw_vals),
        'raw_cv': np.std(raw_vals) / np.mean(raw_vals) if np.mean(raw_vals) > 0 else np.nan,
        'norm_mean': np.mean(norm_vals),
        'norm_std': np.std(norm_vals),
        'norm_cv': np.std(norm_vals) / np.mean(norm_vals) if np.mean(norm_vals) > 0 else np.nan,
    })

collapse_df = pd.DataFrame(collapse_stats)

raw_cv_mean = collapse_df['raw_cv'].mean()
norm_cv_mean = collapse_df['norm_cv'].mean()
cv_reduction = (raw_cv_mean - norm_cv_mean) / raw_cv_mean * 100

print(f"\nMean CV (raw):        {raw_cv_mean:.3f}")
print(f"Mean CV (normalized): {norm_cv_mean:.3f}")
print(f"CV reduction:         {cv_reduction:.1f}%")

# Save stats
collapse_df.to_csv(OUTPUT_DIR / 'kernel_collapse_stats_length_restricted.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'kernel_collapse_stats_length_restricted.csv'}")

# ============================================================
# COMPARISON WITH FULL DATASET
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: FULL vs LENGTH-RESTRICTED")
print("=" * 60)

print(f"""
                        Full Dataset    Length-Restricted
N transcripts:          341             {len(df_valid)}
CV reduction:           28.6%           {cv_reduction:.1f}%

Note: Filter removes only {len(df) - len(df_valid)} documents (min={df_long['token_count'].min()} tokens).
The full dataset already has strong support (min token_count = {df['token_count'].min()}).
""")

# Summary
if cv_reduction > 30:
    conclusion = "STRONG support for universal kernel"
elif cv_reduction > 20:
    conclusion = "MODERATE support for universal kernel"
else:
    conclusion = "WEAK support for universal kernel"

print(f"Conclusion (length-restricted): {conclusion}")
