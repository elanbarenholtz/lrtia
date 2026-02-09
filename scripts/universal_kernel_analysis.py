#!/usr/bin/env python3
"""
Universal Kernel Hypothesis Analysis for ECSC Data.

Tests whether memory curves have a shared shape across age groups,
with differences primarily in magnitude (height) rather than shape.

Uses pre-computed ECSC v4 outputs - no LLM runs needed.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
METRICS_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/ECSC/ecsc_age_v4_metrics (3).csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/universal_kernel")
OUTPUT_DIR.mkdir(exist_ok=True)

# Lag values in the data
K_VALUES = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
GAIN_COLS = [f'mean_gain_{k}' for k in K_VALUES]

print("=" * 70)
print("UNIVERSAL KERNEL HYPOTHESIS ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. LOADING DATA")
print("-" * 50)

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")
print(f"Columns: {list(df.columns)[:10]}...")

# Check age groups
print(f"\nAge groups: {df['age_group'].unique()}")
print(df.groupby('age_group').size())

# ============================================================
# 2. EXTRACT GAIN CURVES
# ============================================================
print("\n2. EXTRACTING GAIN CURVES")
print("-" * 50)

# Build curve matrix (docs x k)
curve_matrix = df[GAIN_COLS].values
print(f"Curve matrix shape: {curve_matrix.shape}")

# Check for NaNs
n_complete = np.sum(~np.isnan(curve_matrix).any(axis=1))
print(f"Complete curves: {n_complete}/{len(df)}")

# Use only complete curves
valid_mask = ~np.isnan(curve_matrix).any(axis=1)
df_valid = df[valid_mask].copy()
curves_valid = curve_matrix[valid_mask]
print(f"Using {len(df_valid)} complete curves")

# ============================================================
# 3. NORMALIZE CURVES (FRACTION OF MAX)
# ============================================================
print("\n3. NORMALIZING CURVES")
print("-" * 50)

# Get max gain (at k=128) for each doc
max_gains = curves_valid[:, -1]  # gain_128
print(f"Max gain (k=128): mean={np.mean(max_gains):.3f}, std={np.std(max_gains):.3f}")

# Avoid division by zero
max_gains_safe = np.where(max_gains > 0.01, max_gains, 0.01)

# Normalize: gain_norm(k) = gain(k) / gain(128)
curves_normalized = curves_valid / max_gains_safe[:, np.newaxis]

# Add normalized curves to dataframe
for i, k in enumerate(K_VALUES):
    df_valid[f'gain_norm_{k}'] = curves_normalized[:, i]

print(f"Normalized curve at k=128 should be ~1.0: mean={np.mean(curves_normalized[:, -1]):.3f}")

# ============================================================
# 4. PLOT CURVE COLLAPSE
# ============================================================
print("\n4. PLOTTING CURVE COLLAPSE")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use actual age groups from data (single-year bins)
age_groups = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
k_array = np.array(K_VALUES[1:])  # Skip k=0 (always 0)

# Left: Raw curves by age group
ax1 = axes[0]
for i, ag in enumerate(age_groups):
    ag_mask = df_valid['age_group'] == ag
    ag_curves = curves_valid[ag_mask, 1:]  # Skip k=0

    mean_curve = np.mean(ag_curves, axis=0)
    sem_curve = np.std(ag_curves, axis=0) / np.sqrt(np.sum(ag_mask))

    ax1.plot(k_array, mean_curve, color=colors[i], label=f'{ag} (n={np.sum(ag_mask)})', linewidth=2)
    ax1.fill_between(k_array, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve,
                     color=colors[i], alpha=0.2)

ax1.set_xscale('log')
ax1.set_xlabel('Lag k (tokens)', fontsize=12)
ax1.set_ylabel('Gain (NLL reduction)', fontsize=12)
ax1.set_title('Raw Gain Curves by Age', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Normalized curves (should collapse)
ax2 = axes[1]
for i, ag in enumerate(age_groups):
    ag_mask = df_valid['age_group'] == ag
    ag_curves_norm = curves_normalized[ag_mask, 1:]  # Skip k=0

    mean_curve = np.mean(ag_curves_norm, axis=0)
    sem_curve = np.std(ag_curves_norm, axis=0) / np.sqrt(np.sum(ag_mask))

    ax2.plot(k_array, mean_curve, color=colors[i], label=f'{ag}', linewidth=2)
    ax2.fill_between(k_array, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve,
                     color=colors[i], alpha=0.2)

ax2.set_xscale('log')
ax2.set_xlabel('Lag k (tokens)', fontsize=12)
ax2.set_ylabel('Normalized Gain (fraction of max)', fontsize=12)
ax2.set_title('Normalized Curves (Shape Only)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kernel_collapse_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'kernel_collapse_plot.png'}")

# ============================================================
# 5. QUANTIFY SHAPE VARIATION
# ============================================================
print("\n5. QUANTIFYING SHAPE VARIATION")
print("-" * 50)

# Per-k statistics
collapse_stats = []
for i, k in enumerate(K_VALUES[1:], 1):  # Skip k=0
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
print("\nVariation by lag (CV = coefficient of variation):")
print(collapse_df.to_string(index=False))

# Summary: does SD shrink after normalization?
raw_cv_mean = collapse_df['raw_cv'].mean()
norm_cv_mean = collapse_df['norm_cv'].mean()
cv_reduction = (raw_cv_mean - norm_cv_mean) / raw_cv_mean * 100

print(f"\nMean CV (raw):        {raw_cv_mean:.3f}")
print(f"Mean CV (normalized): {norm_cv_mean:.3f}")
print(f"CV reduction:         {cv_reduction:.1f}%")

# Per age group stats
print("\n\nPer-k normalized mean by age group:")
age_stats = []
for k in K_VALUES[1:]:
    row = {'k': k}
    for ag in age_groups:
        ag_mask = df_valid['age_group'] == ag
        row[f'{ag}_mean'] = df_valid.loc[ag_mask, f'gain_norm_{k}'].mean()
        row[f'{ag}_std'] = df_valid.loc[ag_mask, f'gain_norm_{k}'].std()
    age_stats.append(row)

age_stats_df = pd.DataFrame(age_stats)
age_stats_df.to_csv(OUTPUT_DIR / 'kernel_collapse_stats.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'kernel_collapse_stats.csv'}")

# ============================================================
# 6. VARIANCE DECOMPOSITION: HEIGHT VS SHAPE
# ============================================================
print("\n6. VARIANCE DECOMPOSITION")
print("-" * 50)

# Total variance in raw curves
total_var_raw = np.var(curves_valid[:, 1:])  # Skip k=0

# Variance due to height (scaling): var of max_gain
height_var = np.var(max_gains)

# Variance in shape (normalized curves)
total_var_norm = np.var(curves_normalized[:, 1:])

# The "height" contribution is proportional to how much variance shrinks
# when we normalize
height_contribution = (total_var_raw - total_var_norm * np.mean(max_gains)**2) / total_var_raw * 100
shape_contribution = 100 - height_contribution

decomposition_text = f"""
VARIANCE DECOMPOSITION: HEIGHT VS SHAPE
========================================

Raw curve variance:        {total_var_raw:.4f}
Normalized curve variance: {total_var_norm:.4f}
Max gain (height) variance: {height_var:.4f}

Interpretation:
- If curves differ mainly in HEIGHT, normalizing should collapse them
- CV reduction after normalization: {cv_reduction:.1f}%

Approximate decomposition:
- Height (magnitude) contribution: ~{height_contribution:.1f}%
- Shape contribution: ~{shape_contribution:.1f}%

CONCLUSION:
"""

if cv_reduction > 30:
    decomposition_text += "Strong support for UNIVERSAL KERNEL hypothesis.\n"
    decomposition_text += "Curves collapse well after height normalization.\n"
    decomposition_text += "Age differences are primarily in MAGNITUDE, not shape."
elif cv_reduction > 15:
    decomposition_text += "Moderate support for universal kernel.\n"
    decomposition_text += "Some shape differences remain after normalization."
else:
    decomposition_text += "Weak support for universal kernel.\n"
    decomposition_text += "Significant shape differences across ages."

print(decomposition_text)

with open(OUTPUT_DIR / 'shape_variance_decomposition.txt', 'w') as f:
    f.write(decomposition_text)
print(f"Saved: {OUTPUT_DIR / 'shape_variance_decomposition.txt'}")

# ============================================================
# 7. CONNECT TO METRICS: MAGNITUDE VS SHAPE BY AGE
# ============================================================
print("\n7. MAGNITUDE VS SHAPE BY AGE")
print("-" * 50)

# Load metrics if available
if METRICS_FILE.exists():
    metrics_df = pd.read_csv(METRICS_FILE)
    print(f"Loaded metrics: {len(metrics_df)} docs")

    # Merge with our valid data
    merged = df_valid.merge(metrics_df[['doc_id', 'total_benefit_fixed', 'half_life_fixed']],
                            on='doc_id', how='left')

    # Compare magnitude vs shape metrics by age
    print("\nBy age group:")
    print(f"{'Age':<10} {'N':>5} {'Gain_128 (mag)':>15} {'log_slope (shape)':>18} {'early_ratio (shape)':>20}")
    print("-" * 75)

    for ag in age_groups:
        ag_data = merged[merged['age_group'] == ag]
        if len(ag_data) > 0:
            mag = ag_data['mean_gain_128'].mean()
            slope = ag_data['log_slope_local'].mean()
            early = ag_data['early_ratio'].mean()
            print(f"{ag:<10} {len(ag_data):>5} {mag:>15.3f} {slope:>18.3f} {early:>20.3f}")

    # Test: does magnitude vary more with age than shape?
    from scipy.stats import spearmanr

    rho_mag, p_mag = spearmanr(merged['age_months'], merged['mean_gain_128'])
    rho_slope, p_slope = spearmanr(merged['age_months'], merged['log_slope_local'])
    rho_early, p_early = spearmanr(merged['age_months'], merged['early_ratio'])

    print(f"\nCorrelation with age:")
    print(f"  Magnitude (gain_128): rho={rho_mag:.3f}, p={p_mag:.4f}")
    print(f"  Shape (log_slope):    rho={rho_slope:.3f}, p={p_slope:.4f}")
    print(f"  Shape (early_ratio):  rho={rho_early:.3f}, p={p_early:.4f}")

    if abs(rho_mag) > abs(rho_slope) and abs(rho_mag) > abs(rho_early):
        print("\n  --> Magnitude varies MORE with age than shape metrics")
        print("  --> Supports UNIVERSAL KERNEL hypothesis")
    else:
        print("\n  --> Shape metrics vary as much or more than magnitude")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Files produced:
  - {OUTPUT_DIR / 'kernel_collapse_plot.png'}
  - {OUTPUT_DIR / 'kernel_collapse_stats.csv'}
  - {OUTPUT_DIR / 'shape_variance_decomposition.txt'}

Key findings:
  - CV reduction after normalization: {cv_reduction:.1f}%
  - {'SUPPORTS' if cv_reduction > 20 else 'DOES NOT STRONGLY SUPPORT'} universal kernel hypothesis
""")
