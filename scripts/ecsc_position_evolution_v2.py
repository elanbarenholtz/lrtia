#!/usr/bin/env python3
"""
ECSC Position-Evolution Analysis v2.

Fixes from v1:
1. pos_eligible: position normalized to [0,1] within each transcript's eligible range
2. Balanced sampling: K windows per position quantile per transcript
3. Within-transcript tests: doc fixed effects / within-doc demeaning
4. Quantile splits: bottom/top 25% instead of hard thresholds

NOTE: This version uses I_256 (influence magnitude).
      For shape metrics (log_slope, early_ratio), need window-level gain curves.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
TRACES_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/ECSC/ecsc_age_v4_traces (3).csv")
METRICS_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/position_evolution_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("ECSC POSITION-EVOLUTION ANALYSIS v2")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. LOADING DATA")
print("-" * 50)

traces_df = pd.read_csv(TRACES_FILE)
print(f"Loaded traces: {len(traces_df)} observations")

metrics_df = pd.read_csv(METRICS_FILE)

# Merge metadata
traces_df = traces_df.merge(
    metrics_df[['doc_id', 'task', 'subject_id', 'token_count', 'baseline_nll']],
    on='doc_id', how='left'
)
traces_df['task'] = traces_df['task'].fillna('unknown')

# Remove NaN I_256
traces_df = traces_df.dropna(subset=['I_256'])
print(f"After removing NaN I_256: {len(traces_df)} observations")
print(f"Unique transcripts: {traces_df['doc_id'].nunique()}")

# ============================================================
# 2. COMPUTE pos_eligible (NORMALIZED WITHIN TRANSCRIPT)
# ============================================================
print("\n2. COMPUTING pos_eligible")
print("-" * 50)

def compute_pos_eligible(group):
    """Normalize position to [0,1] within each transcript's eligible range."""
    pos = group['target_pos_frac'].values
    pos_min = pos.min()
    pos_max = pos.max()

    if pos_max > pos_min:
        group['pos_eligible'] = (pos - pos_min) / (pos_max - pos_min)
    else:
        group['pos_eligible'] = 0.5  # Single window

    group['eligible_start'] = pos_min
    group['eligible_end'] = pos_max
    group['eligible_range'] = pos_max - pos_min

    return group

traces_df = traces_df.groupby('doc_id', group_keys=False).apply(compute_pos_eligible)

print(f"pos_eligible range: [{traces_df['pos_eligible'].min():.3f}, {traces_df['pos_eligible'].max():.3f}]")
print(f"Mean eligible_range per doc: {traces_df.groupby('doc_id')['eligible_range'].first().mean():.3f}")

# ============================================================
# 3. CREATE QUANTILE-BASED POSITION BINS
# ============================================================
print("\n3. CREATING QUANTILE-BASED POSITION BINS")
print("-" * 50)

def assign_quantile_bins(group):
    """Assign position quartiles within each transcript."""
    pos = group['pos_eligible'].values

    # Use quartiles: Q1 = early, Q2-Q3 = mid, Q4 = late
    q25 = np.percentile(pos, 25)
    q75 = np.percentile(pos, 75)

    def bin_pos(p):
        if p <= q25:
            return 'early'
        elif p >= q75:
            return 'late'
        else:
            return 'mid'

    group['pos_quartile'] = group['pos_eligible'].apply(bin_pos)
    return group

traces_df = traces_df.groupby('doc_id', group_keys=False).apply(assign_quantile_bins)

print("Position quartile distribution (per-doc normalized):")
print(traces_df['pos_quartile'].value_counts())

# Verify balance within transcripts
quartile_counts = traces_df.groupby(['doc_id', 'pos_quartile']).size().unstack(fill_value=0)
print(f"\nWindows per quartile per transcript:")
print(f"  Early: mean={quartile_counts['early'].mean():.1f}, min={quartile_counts['early'].min()}")
print(f"  Mid:   mean={quartile_counts['mid'].mean():.1f}")
print(f"  Late:  mean={quartile_counts['late'].mean():.1f}, min={quartile_counts['late'].min()}")

# ============================================================
# 4. BALANCED SAMPLING (K windows per position bin per transcript)
# ============================================================
print("\n4. CREATING BALANCED SAMPLE")
print("-" * 50)

K_PER_BIN = 3  # Windows per quartile per transcript

def balanced_sample(group):
    """Sample K windows from each position bin."""
    early = group[group['pos_quartile'] == 'early']
    mid = group[group['pos_quartile'] == 'mid']
    late = group[group['pos_quartile'] == 'late']

    samples = []

    # Sample from each bin (or take all if fewer than K)
    for bin_df, bin_name in [(early, 'early'), (mid, 'mid'), (late, 'late')]:
        if len(bin_df) >= K_PER_BIN:
            samples.append(bin_df.sample(n=K_PER_BIN, random_state=42))
        elif len(bin_df) > 0:
            samples.append(bin_df)

    if samples:
        return pd.concat(samples)
    return pd.DataFrame()

balanced_df = traces_df.groupby('doc_id', group_keys=False).apply(balanced_sample)
balanced_df = balanced_df.reset_index(drop=True)

print(f"Balanced sample: {len(balanced_df)} observations from {balanced_df['doc_id'].nunique()} transcripts")
print(f"Position quartile distribution (balanced):")
print(balanced_df['pos_quartile'].value_counts())

# ============================================================
# 5. COVERAGE TABLES
# ============================================================
print("\n5. COVERAGE AFTER BALANCING")
print("-" * 50)

# Windows per transcript
windows_per_doc = balanced_df.groupby('doc_id').size()
print(f"Windows per transcript: mean={windows_per_doc.mean():.1f}, min={windows_per_doc.min()}, max={windows_per_doc.max()}")

# By age group
age_pos_counts = pd.crosstab(balanced_df['age_group'], balanced_df['pos_quartile'])
age_pos_counts = age_pos_counts.reindex(columns=['early', 'mid', 'late'])
print("\nCounts by Age × Position (balanced):")
print(age_pos_counts)

# By task
task_pos_counts = pd.crosstab(balanced_df['task'], balanced_df['pos_quartile'])
task_pos_counts = task_pos_counts.reindex(columns=['early', 'mid', 'late'])
print("\nCounts by Task × Position (balanced):")
print(task_pos_counts)

# Save coverage
age_pos_counts.to_csv(OUTPUT_DIR / 'coverage_age_position.csv')
task_pos_counts.to_csv(OUTPUT_DIR / 'coverage_task_position.csv')

# ============================================================
# 6. WITHIN-TRANSCRIPT ANALYSIS
# ============================================================
print("\n6. WITHIN-TRANSCRIPT ANALYSIS")
print("-" * 50)

# 6a. Within-doc demeaning
print("\n6a. Within-Document Demeaning:")
print("-" * 40)

# Compute doc means
doc_means = balanced_df.groupby('doc_id').agg({
    'I_256': 'mean',
    'pos_eligible': 'mean'
}).rename(columns={'I_256': 'I_256_doc_mean', 'pos_eligible': 'pos_doc_mean'})

balanced_df = balanced_df.merge(doc_means, on='doc_id')
balanced_df['I_256_demeaned'] = balanced_df['I_256'] - balanced_df['I_256_doc_mean']
balanced_df['pos_demeaned'] = balanced_df['pos_eligible'] - balanced_df['pos_doc_mean']

# Regress demeaned Y on demeaned pos
from scipy.stats import pearsonr, spearmanr

r_demeaned, p_demeaned = pearsonr(balanced_df['pos_demeaned'], balanced_df['I_256_demeaned'])
print(f"Demeaned correlation: r={r_demeaned:.4f}, p={p_demeaned:.4f}")

# Simple OLS on demeaned data
try:
    import statsmodels.formula.api as smf

    model_demeaned = smf.ols('I_256_demeaned ~ pos_demeaned', data=balanced_df).fit()
    print(f"Demeaned regression: β={model_demeaned.params['pos_demeaned']:.4f}, p={model_demeaned.pvalues['pos_demeaned']:.4f}")

except ImportError:
    print("statsmodels not available")

# 6b. Doc fixed effects model
print("\n6b. Document Fixed Effects Model:")
print("-" * 40)

try:
    # Y ~ pos_eligible + C(doc_id)
    # This is equivalent to within-doc variation
    model_fe = smf.ols('I_256 ~ pos_eligible + C(doc_id)', data=balanced_df).fit()
    print(f"Fixed effects: pos_eligible β={model_fe.params['pos_eligible']:.4f}, p={model_fe.pvalues['pos_eligible']:.4f}")

except Exception as e:
    print(f"Fixed effects model failed: {e}")

# ============================================================
# 7. WITHIN-TRANSCRIPT DRIFT
# ============================================================
print("\n7. WITHIN-TRANSCRIPT DRIFT ANALYSIS")
print("-" * 50)

# Compute drift per transcript: mean(late quartile) - mean(early quartile)
drift_results = []

for doc_id, doc_df in balanced_df.groupby('doc_id'):
    early_vals = doc_df[doc_df['pos_quartile'] == 'early']['I_256']
    late_vals = doc_df[doc_df['pos_quartile'] == 'late']['I_256']

    if len(early_vals) > 0 and len(late_vals) > 0:
        drift = late_vals.mean() - early_vals.mean()

        # Get metadata
        meta = doc_df.iloc[0]

        drift_results.append({
            'doc_id': doc_id,
            'age_months': meta['age_months'],
            'age_group': meta['age_group'],
            'task': meta['task'],
            'subject_id': meta['subject_id'],
            'n_early': len(early_vals),
            'n_late': len(late_vals),
            'early_mean': early_vals.mean(),
            'late_mean': late_vals.mean(),
            'drift': drift,
        })

drift_df = pd.DataFrame(drift_results)
print(f"Transcripts with valid drift: {len(drift_df)}")

# Overall drift statistics
mean_drift = drift_df['drift'].mean()
std_drift = drift_df['drift'].std()
sem_drift = drift_df['drift'].sem()
t_stat, p_value = stats.ttest_1samp(drift_df['drift'], 0)

print(f"\nOverall Drift (late Q4 - early Q1):")
print(f"  Mean: {mean_drift:.4f}")
print(f"  Std:  {std_drift:.4f}")
print(f"  95% CI: [{mean_drift - 1.96*sem_drift:.4f}, {mean_drift + 1.96*sem_drift:.4f}]")
print(f"  t-test vs 0: t={t_stat:.3f}, p={p_value:.4f}")

# Drift by age group
print("\n\nDrift by Age Group:")
print(f"{'Age':<10} {'N':>5} {'Mean Drift':>12} {'SE':>10} {'t':>8} {'p':>8}")
print("-" * 60)

age_drift_results = []
for ag in ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']:
    ag_drift = drift_df[drift_df['age_group'] == ag]['drift']
    if len(ag_drift) >= 3:
        mean = ag_drift.mean()
        se = ag_drift.sem()
        t, p = stats.ttest_1samp(ag_drift, 0)
        print(f"{ag:<10} {len(ag_drift):>5} {mean:>12.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f}")
        age_drift_results.append({'age_group': ag, 'n': len(ag_drift), 'mean_drift': mean, 'se': se, 't': t, 'p': p})

# Save drift results
drift_df.to_csv(OUTPUT_DIR / 'drift_per_transcript.csv', index=False)
pd.DataFrame(age_drift_results).to_csv(OUTPUT_DIR / 'drift_by_age_group.csv', index=False)

# ============================================================
# 8. PLOTS
# ============================================================
print("\n8. GENERATING PLOTS")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean I_256 by position quartile (balanced)
ax1 = axes[0, 0]
pos_order = ['early', 'mid', 'late']
pos_means = balanced_df.groupby('pos_quartile')['I_256'].agg(['mean', 'std', 'count'])
pos_means = pos_means.reindex(pos_order)
pos_means['sem'] = pos_means['std'] / np.sqrt(pos_means['count'])

colors = ['#2ecc71', '#f1c40f', '#e74c3c']
bars = ax1.bar(pos_order, pos_means['mean'], yerr=1.96*pos_means['sem'],
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Position Quartile', fontsize=12)
ax1.set_ylabel('Mean I_256 (± 95% CI)', fontsize=12)
ax1.set_title('Influence by Position (Balanced Sample)', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# Add counts
for i, (pos, row) in enumerate(pos_means.iterrows()):
    ax1.text(i, row['mean'] + 1.96*row['sem'] + 0.5, f'n={int(row["count"])}',
             ha='center', fontsize=10)

# Plot 2: Continuous profile (demeaned)
ax2 = axes[0, 1]

# Bin pos_eligible for smooth plot
balanced_df['pos_bin'] = (balanced_df['pos_eligible'] * 10).astype(int) / 10
profile = balanced_df.groupby('pos_bin')['I_256_demeaned'].agg(['mean', 'std', 'count'])
profile['sem'] = profile['std'] / np.sqrt(profile['count'])

ax2.fill_between(profile.index,
                 profile['mean'] - 1.96*profile['sem'],
                 profile['mean'] + 1.96*profile['sem'],
                 alpha=0.3, color='blue')
ax2.plot(profile.index, profile['mean'], 'b-', linewidth=2)
ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('pos_eligible (normalized position)', fontsize=12)
ax2.set_ylabel('I_256 (demeaned within doc)', fontsize=12)
ax2.set_title('Within-Transcript Position Effect', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: Drift distribution
ax3 = axes[1, 0]
ax3.hist(drift_df['drift'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No drift')
ax3.axvline(mean_drift, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean_drift:.2f}')
ax3.set_xlabel('Drift (late Q4 - early Q1)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title(f'Within-Transcript Drift Distribution (p={p_value:.3f})', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Drift by age group
ax4 = axes[1, 1]
age_groups = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
age_drifts = []
age_labels = []
for ag in age_groups:
    ag_data = drift_df[drift_df['age_group'] == ag]['drift']
    if len(ag_data) >= 3:
        age_drifts.append(ag_data.values)
        age_labels.append(ag)

if age_drifts:
    bp = ax4.boxplot(age_drifts, tick_labels=age_labels, patch_artist=True)
    colors_box = plt.cm.viridis(np.linspace(0, 0.8, len(age_labels)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Age Group', fontsize=12)
    ax4.set_ylabel('Drift (late - early)', fontsize=12)
    ax4.set_title('Within-Transcript Drift by Age', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'position_analysis_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'position_analysis_v2.png'}")

# ============================================================
# 9. SAVE BALANCED DATASET
# ============================================================
print("\n9. SAVING DATA")
print("-" * 50)

output_cols = ['doc_id', 'target_idx', 'target_pos', 'target_pos_frac',
               'pos_eligible', 'pos_quartile', 'eligible_start', 'eligible_end',
               'I_256', 'I_256_demeaned', 'age_months', 'age_group', 'task', 'subject_id']
balanced_df[output_cols].to_csv(OUTPUT_DIR / 'balanced_window_dataset.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'balanced_window_dataset.csv'}")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
DATA (v2 balanced sample):
  - {len(balanced_df)} windows from {balanced_df['doc_id'].nunique()} transcripts
  - {K_PER_BIN} windows per position quartile per transcript (when available)
  - pos_eligible normalized to [0,1] within each transcript

WITHIN-TRANSCRIPT POSITION EFFECT (I_256):

  Demeaned regression: β = {model_demeaned.params['pos_demeaned']:.4f}, p = {model_demeaned.pvalues['pos_demeaned']:.4f}

  Drift (Q4 - Q1): {mean_drift:.4f} [{mean_drift - 1.96*sem_drift:.4f}, {mean_drift + 1.96*sem_drift:.4f}]
  t-test vs 0: t = {t_stat:.3f}, p = {p_value:.4f}
""")

if p_value > 0.05:
    print("""CONCLUSION:
  → NO SIGNIFICANT within-transcript position effect
  → Influence profile is STATIONARY across narrative progression
  → v1 pooled effects were selection artifacts (between-transcript confounds)
""")
else:
    direction = "INCREASES" if mean_drift > 0 else "DECREASES"
    print(f"""CONCLUSION:
  → SIGNIFICANT within-transcript position effect detected
  → Influence {direction} from early to late in narratives
""")

print("""
LIMITATION:
  This analysis uses I_256 (influence magnitude at 256-token span).
  For shape metrics (log_slope_local, early_ratio, auc_log_k):
  → Need window-level gain curves from new inference run
  → Current ECSC data only has I_256/I_512 per target position

OUTPUT FILES:
  - balanced_window_dataset.csv (with pos_eligible)
  - coverage_age_position.csv / coverage_task_position.csv
  - drift_per_transcript.csv
  - drift_by_age_group.csv
  - position_analysis_v2.png
""")
