#!/usr/bin/env python3
"""
ECSC Position-Evolution Analysis of Influence Profile.

Tests whether local context influence is stationary across a narrative,
or drifts from early → middle → late in the same transcript.

Uses available ECSC traces data (per-target position + I_512/I_256).

NOTE: Full shape metric analysis (log_slope_local, early_ratio, etc.)
requires window-level gain curves which need to be generated separately.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Paths
TRACES_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/ECSC/ecsc_age_v4_traces (3).csv")
METRICS_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/position_evolution_v1")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("ECSC POSITION-EVOLUTION ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. LOADING DATA")
print("-" * 50)

# Load traces (per-target position data)
traces_df = pd.read_csv(TRACES_FILE)
print(f"Loaded traces: {len(traces_df)} target observations")
print(f"Columns: {list(traces_df.columns)}")

# Load transcript-level metrics for additional metadata
metrics_df = pd.read_csv(METRICS_FILE)
print(f"Loaded transcript metrics: {len(metrics_df)} transcripts")

# Merge to get task/story info
# First, extract doc_id base to match
traces_df['doc_base'] = traces_df['doc_id'].str.replace(r'_YR\d+$', '', regex=True)
metrics_df['doc_base'] = metrics_df['doc_id']

# Merge task info
traces_df = traces_df.merge(
    metrics_df[['doc_id', 'task', 'subject_id', 'token_count', 'baseline_nll', 'sex', 'study_year']],
    on='doc_id', how='left'
)

# Fill missing task with 'unknown'
traces_df['task'] = traces_df['task'].fillna('unknown')

print(f"\nAfter merge: {len(traces_df)} observations")
print(f"Unique docs: {traces_df['doc_id'].nunique()}")
print(f"Tasks: {traces_df['task'].value_counts().to_dict()}")

# ============================================================
# 2. ADD POSITION VARIABLES
# ============================================================
print("\n2. ADDING POSITION VARIABLES")
print("-" * 50)

# pos_frac is already in the data as target_pos_frac
traces_df['pos_frac'] = traces_df['target_pos_frac']

# Check actual position range
pos_min = traces_df['pos_frac'].min()
pos_max = traces_df['pos_frac'].max()
print(f"Actual position range: [{pos_min:.3f}, {pos_max:.3f}]")

# Create position bins RELATIVE TO ACTUAL DATA RANGE
# (because burn-in means we can't score the very beginning)
def pos_bin_relative(frac, data_min, data_max):
    """Bin positions into terciles of the AVAILABLE range."""
    range_span = data_max - data_min
    relative_pos = (frac - data_min) / range_span
    if relative_pos < 0.33:
        return 'early'
    elif relative_pos < 0.67:
        return 'mid'
    else:
        return 'late'

traces_df['pos_bin'] = traces_df['pos_frac'].apply(
    lambda x: pos_bin_relative(x, pos_min, pos_max)
)

# Also keep absolute bins for reference
def pos_bin_absolute(frac):
    if frac < 0.5:
        return 'first_half'
    else:
        return 'second_half'

traces_df['pos_bin_abs'] = traces_df['pos_frac'].apply(pos_bin_absolute)

# Order for plotting
pos_order = ['early', 'mid', 'late']

print("\nPosition bin distribution (relative to data range):")
print(traces_df['pos_bin'].value_counts())
print("\nPosition bin distribution (absolute halves):")
print(traces_df['pos_bin_abs'].value_counts())

# ============================================================
# 3. COVERAGE SANITY CHECKS
# ============================================================
print("\n3. COVERAGE SANITY CHECKS")
print("-" * 50)

# Windows per transcript
windows_per_doc = traces_df.groupby('doc_id').size()
print(f"\nWindows per transcript:")
print(f"  Mean: {windows_per_doc.mean():.1f}")
print(f"  Min: {windows_per_doc.min()}, Max: {windows_per_doc.max()}")

# Counts per age_group × position_bin
print("\n\nCounts by Age Group × Position Bin:")
age_pos_counts = pd.crosstab(traces_df['age_group'], traces_df['pos_bin'])
age_pos_counts = age_pos_counts.reindex(columns=pos_order)
print(age_pos_counts)

# Counts per task × position_bin
print("\n\nCounts by Task × Position Bin:")
task_pos_counts = pd.crosstab(traces_df['task'], traces_df['pos_bin'])
task_pos_counts = task_pos_counts.reindex(columns=pos_order)
print(task_pos_counts)

# Check for selection effects (chi-square test)
from scipy.stats import chi2_contingency

chi2, p_age, dof, expected = chi2_contingency(age_pos_counts)
print(f"\nAge × Position independence test: χ²={chi2:.2f}, p={p_age:.4f}")

chi2, p_task, dof, expected = chi2_contingency(task_pos_counts)
print(f"Task × Position independence test: χ²={chi2:.2f}, p={p_task:.4f}")

if p_age < 0.05 or p_task < 0.05:
    print("WARNING: Position coverage varies by age and/or task - potential selection effect")
else:
    print("OK: Position coverage is balanced across age and task")

# Save coverage tables
coverage_df = pd.DataFrame({
    'check': ['age_pos_chi2', 'age_pos_p', 'task_pos_chi2', 'task_pos_p'],
    'value': [chi2, p_age, chi2, p_task]
})
age_pos_counts.to_csv(OUTPUT_DIR / 'age_position_counts.csv')
task_pos_counts.to_csv(OUTPUT_DIR / 'task_position_counts.csv')
print(f"\nSaved coverage tables to {OUTPUT_DIR}")

# ============================================================
# 4. PLOTS: INFLUENCE vs POSITION
# ============================================================
print("\n4. GENERATING PLOTS")
print("-" * 50)

# Use I_256 as main metric (more complete than I_512)
# Remove NaN values
traces_valid = traces_df.dropna(subset=['I_256'])
print(f"Valid observations with I_256: {len(traces_valid)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean I_256 by position bin, overall
ax1 = axes[0, 0]
pos_means = traces_valid.groupby('pos_bin')['I_256'].agg(['mean', 'std', 'count'])
pos_means = pos_means.reindex(pos_order)
pos_means['sem'] = pos_means['std'] / np.sqrt(pos_means['count'])

bars = ax1.bar(pos_order, pos_means['mean'], yerr=1.96*pos_means['sem'],
               capsize=5, color=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.7)
ax1.set_xlabel('Position in Narrative', fontsize=12)
ax1.set_ylabel('Mean I_256 (influence magnitude)', fontsize=12)
ax1.set_title('Influence by Narrative Position (Overall)', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Mean I_256 by position bin, by age group
ax2 = axes[0, 1]
age_groups = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
colors = plt.cm.viridis(np.linspace(0, 0.8, len(age_groups)))

x = np.arange(len(pos_order))
width = 0.12
for i, ag in enumerate(age_groups):
    ag_data = traces_valid[traces_valid['age_group'] == ag]
    if len(ag_data) > 0:
        ag_means = ag_data.groupby('pos_bin')['I_256'].mean().reindex(pos_order)
        ax2.bar(x + i*width - 0.3, ag_means.values, width, label=ag, color=colors[i], alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(pos_order)
ax2.set_xlabel('Position in Narrative', fontsize=12)
ax2.set_ylabel('Mean I_256', fontsize=12)
ax2.set_title('Influence by Position × Age Group', fontsize=14)
ax2.legend(title='Age', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: I_256 vs pos_frac (continuous) with smooth trend
ax3 = axes[1, 0]

# Bin pos_frac for smoother plot
traces_valid['pos_frac_bin'] = (traces_valid['pos_frac'] * 20).astype(int) / 20
smooth_profile = traces_valid.groupby('pos_frac_bin')['I_256'].agg(['mean', 'std', 'count'])
smooth_profile['sem'] = smooth_profile['std'] / np.sqrt(smooth_profile['count'])

ax3.fill_between(smooth_profile.index,
                 smooth_profile['mean'] - 1.96*smooth_profile['sem'],
                 smooth_profile['mean'] + 1.96*smooth_profile['sem'],
                 alpha=0.3, color='blue')
ax3.plot(smooth_profile.index, smooth_profile['mean'], 'b-', linewidth=2)
ax3.axvline(0.33, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(0.67, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Position Fraction (0=start, 1=end)', fontsize=12)
ax3.set_ylabel('Mean I_256 (± 95% CI)', fontsize=12)
ax3.set_title('Continuous Influence Profile Across Narrative', fontsize=14)
ax3.grid(True, alpha=0.3)

# Plot 4: I_256 vs pos_frac by age (smooth lines)
ax4 = axes[1, 1]

for i, ag in enumerate(age_groups):
    ag_data = traces_valid[traces_valid['age_group'] == ag]
    if len(ag_data) > 50:  # Need enough data
        ag_data['pos_frac_bin'] = (ag_data['pos_frac'] * 10).astype(int) / 10
        ag_profile = ag_data.groupby('pos_frac_bin')['I_256'].mean()
        ax4.plot(ag_profile.index, ag_profile.values, 'o-', color=colors[i],
                 label=ag, linewidth=2, markersize=4, alpha=0.8)

ax4.axvline(0.33, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(0.67, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Position Fraction', fontsize=12)
ax4.set_ylabel('Mean I_256', fontsize=12)
ax4.set_title('Influence Profile by Age Group', fontsize=14)
ax4.legend(title='Age', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'influence_vs_position.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'influence_vs_position.png'}")

# ============================================================
# 5. STATISTICAL TESTS: POSITION EFFECT
# ============================================================
print("\n5. STATISTICAL TESTS")
print("-" * 50)

try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not available, skipping mixed models")

if HAS_STATSMODELS:
    # Prepare data
    model_data = traces_valid.copy()
    model_data['log_token_count'] = np.log(model_data['token_count'].fillna(model_data['token_count'].median()))

    # Model 1: Simple OLS - position effect
    print("\nModel 1: I_256 ~ pos_frac (OLS)")
    model1 = smf.ols('I_256 ~ pos_frac', data=model_data).fit()
    print(f"  pos_frac coef: {model1.params['pos_frac']:.4f} (p={model1.pvalues['pos_frac']:.4f})")
    print(f"  R² = {model1.rsquared:.4f}")

    # Model 2: With controls
    print("\nModel 2: I_256 ~ pos_frac + log_token_count + age_months (OLS)")
    model2 = smf.ols('I_256 ~ pos_frac + log_token_count + age_months', data=model_data).fit()
    print(f"  pos_frac coef: {model2.params['pos_frac']:.4f} (p={model2.pvalues['pos_frac']:.4f})")
    print(f"  R² = {model2.rsquared:.4f}")

    # Model 3: Mixed effects with random intercept for doc
    print("\nModel 3: I_256 ~ pos_frac + log_token_count | (1|doc_id)")
    try:
        # Need to handle the formula differently for MixedLM
        model3 = smf.mixedlm('I_256 ~ pos_frac + log_token_count',
                             data=model_data,
                             groups=model_data['doc_id']).fit()
        print(f"  pos_frac coef: {model3.fe_params['pos_frac']:.4f} (p={model3.pvalues['pos_frac']:.4f})")
        print(f"  Random effect variance: {model3.cov_re.iloc[0,0]:.4f}")
    except Exception as e:
        print(f"  Mixed model failed: {e}")

    # Model 4: With task as fixed effect
    print("\nModel 4: I_256 ~ pos_frac + log_token_count + C(task) (OLS)")
    model4 = smf.ols('I_256 ~ pos_frac + log_token_count + C(task)', data=model_data).fit()
    print(f"  pos_frac coef: {model4.params['pos_frac']:.4f} (p={model4.pvalues['pos_frac']:.4f})")
    print(f"  R² = {model4.rsquared:.4f}")

    # Save model summaries
    with open(OUTPUT_DIR / 'position_effect_models.txt', 'w') as f:
        f.write("POSITION EFFECT MODELS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model 1: I_256 ~ pos_frac\n")
        f.write(model1.summary().as_text() + "\n\n")
        f.write("Model 2: I_256 ~ pos_frac + log_token_count + age_months\n")
        f.write(model2.summary().as_text() + "\n\n")
        f.write("Model 4: I_256 ~ pos_frac + log_token_count + C(task)\n")
        f.write(model4.summary().as_text() + "\n")
    print(f"Saved: {OUTPUT_DIR / 'position_effect_models.txt'}")

# ============================================================
# 6. WITHIN-TRANSCRIPT DRIFT METRIC
# ============================================================
print("\n6. WITHIN-TRANSCRIPT DRIFT ANALYSIS")
print("-" * 50)

# Compute per-transcript drift = mean(late) - mean(early)
# Using RELATIVE position bins (terciles of available range)
drift_data = []

# Get global position range for consistent binning
global_pos_min = traces_valid['pos_frac'].min()
global_pos_max = traces_valid['pos_frac'].max()

for doc_id, doc_df in traces_valid.groupby('doc_id'):
    # Recompute bins for this doc using global range
    doc_df = doc_df.copy()
    doc_df['pos_bin'] = doc_df['pos_frac'].apply(
        lambda x: pos_bin_relative(x, global_pos_min, global_pos_max)
    )

    early_mean = doc_df[doc_df['pos_bin'] == 'early']['I_256'].mean()
    mid_mean = doc_df[doc_df['pos_bin'] == 'mid']['I_256'].mean()
    late_mean = doc_df[doc_df['pos_bin'] == 'late']['I_256'].mean()

    # Get doc metadata
    doc_meta = doc_df.iloc[0]

    drift_data.append({
        'doc_id': doc_id,
        'age_months': doc_meta['age_months'],
        'age_group': doc_meta['age_group'],
        'task': doc_meta['task'],
        'early_mean': early_mean,
        'mid_mean': mid_mean,
        'late_mean': late_mean,
        'drift_late_early': late_mean - early_mean if pd.notna(late_mean) and pd.notna(early_mean) else np.nan,
        'drift_mid_early': mid_mean - early_mean if pd.notna(mid_mean) and pd.notna(early_mean) else np.nan,
        'drift_late_mid': late_mean - mid_mean if pd.notna(late_mean) and pd.notna(mid_mean) else np.nan,
    })

drift_df = pd.DataFrame(drift_data)
drift_df = drift_df.dropna(subset=['drift_late_early'])

print(f"Transcripts with valid drift: {len(drift_df)}")

# Overall drift statistics
print(f"\nOverall Drift (late - early):")
mean_drift = drift_df['drift_late_early'].mean()
std_drift = drift_df['drift_late_early'].std()
sem_drift = drift_df['drift_late_early'].sem()
print(f"  Mean: {mean_drift:.4f}")
print(f"  Std: {std_drift:.4f}")
print(f"  95% CI: [{mean_drift - 1.96*sem_drift:.4f}, {mean_drift + 1.96*sem_drift:.4f}]")

# Test if drift differs from 0
from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(drift_df['drift_late_early'].dropna(), 0)
print(f"  t-test vs 0: t={t_stat:.3f}, p={p_value:.4f}")

# Drift by age group
print("\n\nDrift by Age Group:")
print(f"{'Age':<10} {'N':>5} {'Mean Drift':>12} {'95% CI':>25}")
print("-" * 55)

for ag in ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']:
    ag_drift = drift_df[drift_df['age_group'] == ag]['drift_late_early']
    if len(ag_drift) > 2:
        mean = ag_drift.mean()
        sem = ag_drift.sem()
        print(f"{ag:<10} {len(ag_drift):>5} {mean:>12.4f}   [{mean-1.96*sem:.4f}, {mean+1.96*sem:.4f}]")

# Save drift data
drift_df.to_csv(OUTPUT_DIR / 'drift_per_transcript.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'drift_per_transcript.csv'}")

# Plot drift distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall drift distribution
ax1 = axes[0]
ax1.hist(drift_df['drift_late_early'].dropna(), bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No drift')
ax1.axvline(mean_drift, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean_drift:.3f}')
ax1.set_xlabel('Drift (late - early)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Within-Transcript Drift', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Drift by age group
ax2 = axes[1]
age_drifts = []
age_labels = []
for ag in ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']:
    ag_drift = drift_df[drift_df['age_group'] == ag]['drift_late_early'].dropna()
    if len(ag_drift) > 0:
        age_drifts.append(ag_drift.values)
        age_labels.append(ag)

bp = ax2.boxplot(age_drifts, labels=age_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 0.8, len(age_labels)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Age Group', fontsize=12)
ax2.set_ylabel('Drift (late - early)', fontsize=12)
ax2.set_title('Drift by Age Group', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'drift_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'drift_plots.png'}")

# ============================================================
# 7. SAVE WINDOW-LEVEL DATA WITH POSITION
# ============================================================
print("\n7. SAVING DATA")
print("-" * 50)

# Save window-level data with position variables
output_cols = ['doc_id', 'target_idx', 'target_pos', 'target_pos_frac', 'pos_frac', 'pos_bin',
               'I_512', 'I_256', 'age_months', 'age_group', 'task', 'token_count']
traces_df[output_cols].to_csv(OUTPUT_DIR / 'window_level_with_position.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'window_level_with_position.csv'}")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
DATA:
  - {len(traces_valid)} target observations across {traces_valid['doc_id'].nunique()} transcripts
  - Position coverage balanced across age and task

POSITION EFFECT ON INFLUENCE (I_256):
  - Overall drift (late - early): {mean_drift:.4f} (p={p_value:.4f})
""")

if p_value < 0.05:
    if mean_drift > 0:
        print("  → SIGNIFICANT INCREASE in influence late in narratives")
        print("  → Suggests 'closure effect' - more context-dependent toward ending")
    else:
        print("  → SIGNIFICANT DECREASE in influence late in narratives")
        print("  → Suggests narratives become more predictable/formulaic toward ending")
else:
    print("  → NO SIGNIFICANT position effect")
    print("  → Influence profile is STATIONARY across narrative")

print(f"""
NOTE: This analysis uses I_256 (influence magnitude at 256-token span).
      For full shape metric analysis (log_slope, early_ratio, etc.),
      window-level gain curves need to be computed separately.

OUTPUT FILES:
  - window_level_with_position.csv
  - age_position_counts.csv / task_position_counts.csv
  - position_effect_models.txt
  - drift_per_transcript.csv
  - influence_vs_position.png
  - drift_plots.png
""")
