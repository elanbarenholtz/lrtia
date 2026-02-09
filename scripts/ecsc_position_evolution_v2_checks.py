#!/usr/bin/env python3
"""
ECSC Position-Evolution v2 - Additional Checks.

1. Add baseline_nll covariate (is position effect just "later is easier"?)
2. Add distance-to-end (is it an "ending/closure" effect?)
3. Sensitivity: exclude final 2-3 windows
4. Stratify by task/story
5. Age×position interaction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

# Paths
BALANCED_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/position_evolution_v2/balanced_window_dataset.csv")
TRACES_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/ECSC/ecsc_age_v4_traces (3).csv")
METRICS_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/position_evolution_v2")

np.random.seed(42)

print("=" * 70)
print("POSITION-EVOLUTION v2 - ADDITIONAL CHECKS")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\nLOADING DATA")
print("-" * 50)

# Load balanced dataset from v2
df = pd.read_csv(BALANCED_FILE)
print(f"Loaded balanced dataset: {len(df)} observations")

# Load full traces to get window-level baseline info if needed
traces_df = pd.read_csv(TRACES_FILE)
metrics_df = pd.read_csv(METRICS_FILE)

# We need window-level baseline_nll - check if available
# The traces file doesn't have this, so we'll use transcript-level baseline_nll as proxy
df = df.merge(metrics_df[['doc_id', 'baseline_nll']], on='doc_id', how='left', suffixes=('', '_doc'))

# Compute distance-to-end: 1 - pos_eligible (or tokens remaining)
df['dist_to_end'] = 1 - df['pos_eligible']

# Also compute actual tokens remaining (using eligible_end)
df['tokens_to_end'] = df['eligible_end'] - df['target_pos_frac']

# Compute doc-level means for demeaning
doc_means = df.groupby('doc_id').agg({
    'I_256': 'mean',
    'pos_eligible': 'mean',
    'baseline_nll': 'mean',
    'dist_to_end': 'mean'
}).rename(columns={
    'I_256': 'I_256_mean',
    'pos_eligible': 'pos_mean',
    'baseline_nll': 'baseline_mean',
    'dist_to_end': 'dist_to_end_mean'
})

df = df.merge(doc_means, on='doc_id', suffixes=('', '_docmean'))

# Demean variables
df['I_256_dm'] = df['I_256'] - df['I_256_mean']
df['pos_dm'] = df['pos_eligible'] - df['pos_mean']
df['baseline_dm'] = df['baseline_nll'] - df['baseline_mean']
df['dist_to_end_dm'] = df['dist_to_end'] - df['dist_to_end_mean']

print(f"Variables available: {list(df.columns)}")

# ============================================================
# MODEL RESULTS STORAGE
# ============================================================
model_results = []

# ============================================================
# CHECK 1: ADD baseline_nll COVARIATE
# ============================================================
print("\n" + "=" * 70)
print("CHECK 1: ADDING baseline_nll COVARIATE")
print("=" * 70)

print("\nQuestion: Is position effect just 'later is easier'?")

# Model 1a: Position only (baseline)
m1a = smf.ols('I_256_dm ~ pos_dm', data=df).fit()
print(f"\nModel 1a: I_256_dm ~ pos_dm")
print(f"  pos_dm: β={m1a.params['pos_dm']:.4f}, p={m1a.pvalues['pos_dm']:.4f}")

model_results.append({
    'check': '1a_baseline',
    'model': 'I_256_dm ~ pos_dm',
    'pos_coef': m1a.params['pos_dm'],
    'pos_se': m1a.bse['pos_dm'],
    'pos_p': m1a.pvalues['pos_dm'],
    'r2': m1a.rsquared,
    'n': len(df)
})

# Model 1b: Position + baseline_nll (demeaned)
m1b = smf.ols('I_256_dm ~ pos_dm + baseline_dm', data=df).fit()
print(f"\nModel 1b: I_256_dm ~ pos_dm + baseline_dm")
print(f"  pos_dm: β={m1b.params['pos_dm']:.4f}, p={m1b.pvalues['pos_dm']:.4f}")
print(f"  baseline_dm: β={m1b.params['baseline_dm']:.4f}, p={m1b.pvalues['baseline_dm']:.4f}")

model_results.append({
    'check': '1b_with_baseline',
    'model': 'I_256_dm ~ pos_dm + baseline_dm',
    'pos_coef': m1b.params['pos_dm'],
    'pos_se': m1b.bse['pos_dm'],
    'pos_p': m1b.pvalues['pos_dm'],
    'baseline_coef': m1b.params['baseline_dm'],
    'baseline_p': m1b.pvalues['baseline_dm'],
    'r2': m1b.rsquared,
    'n': len(df)
})

# Interpretation
pct_change = (m1b.params['pos_dm'] - m1a.params['pos_dm']) / abs(m1a.params['pos_dm']) * 100
print(f"\n  Position coefficient change: {pct_change:+.1f}%")
if m1b.pvalues['pos_dm'] < 0.05:
    print("  → Position effect SURVIVES after controlling for baseline_nll")
else:
    print("  → Position effect DISAPPEARS after controlling for baseline_nll")

# ============================================================
# CHECK 2: DISTANCE-TO-END vs POSITION
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: DISTANCE-TO-END (CLOSURE EFFECT?)")
print("=" * 70)

print("\nQuestion: Is the effect driven by 'ending' specifically?")

# Note: dist_to_end = 1 - pos_eligible, so they're collinear if both included
# We test them separately and compare

# Model 2a: distance-to-end only
m2a = smf.ols('I_256_dm ~ dist_to_end_dm', data=df).fit()
print(f"\nModel 2a: I_256_dm ~ dist_to_end_dm")
print(f"  dist_to_end_dm: β={m2a.params['dist_to_end_dm']:.4f}, p={m2a.pvalues['dist_to_end_dm']:.4f}")

model_results.append({
    'check': '2a_dist_to_end',
    'model': 'I_256_dm ~ dist_to_end_dm',
    'pos_coef': m2a.params['dist_to_end_dm'],  # Using pos_coef for main effect
    'pos_se': m2a.bse['dist_to_end_dm'],
    'pos_p': m2a.pvalues['dist_to_end_dm'],
    'r2': m2a.rsquared,
    'n': len(df)
})

# The coefficient should be opposite sign to pos_eligible (since dist_to_end = 1 - pos_eligible)
print(f"\n  Note: dist_to_end = 1 - pos_eligible")
print(f"  pos_dm β = {m1a.params['pos_dm']:.4f}")
print(f"  dist_to_end_dm β = {m2a.params['dist_to_end_dm']:.4f}")
print(f"  → Signs are {'opposite (expected)' if m1a.params['pos_dm'] * m2a.params['dist_to_end_dm'] < 0 else 'SAME (unexpected)'}")

# ============================================================
# CHECK 3: SENSITIVITY - EXCLUDE FINAL 2-3 WINDOWS
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: SENSITIVITY - EXCLUDE FINAL WINDOWS")
print("=" * 70)

print("\nQuestion: Is effect driven by very last windows only?")

# Identify final windows within each doc
df['window_rank_desc'] = df.groupby('doc_id')['pos_eligible'].rank(ascending=False)

# Exclude final 2 windows
df_excl2 = df[df['window_rank_desc'] > 2].copy()
print(f"\nExcluding final 2 windows: {len(df_excl2)} obs remaining")

m3a = smf.ols('I_256_dm ~ pos_dm', data=df_excl2).fit()
print(f"Model 3a (excl final 2): pos_dm β={m3a.params['pos_dm']:.4f}, p={m3a.pvalues['pos_dm']:.4f}")

model_results.append({
    'check': '3a_excl_final2',
    'model': 'I_256_dm ~ pos_dm (excl final 2)',
    'pos_coef': m3a.params['pos_dm'],
    'pos_se': m3a.bse['pos_dm'],
    'pos_p': m3a.pvalues['pos_dm'],
    'r2': m3a.rsquared,
    'n': len(df_excl2)
})

# Exclude final 3 windows
df_excl3 = df[df['window_rank_desc'] > 3].copy()
print(f"\nExcluding final 3 windows: {len(df_excl3)} obs remaining")

m3b = smf.ols('I_256_dm ~ pos_dm', data=df_excl3).fit()
print(f"Model 3b (excl final 3): pos_dm β={m3b.params['pos_dm']:.4f}, p={m3b.pvalues['pos_dm']:.4f}")

model_results.append({
    'check': '3b_excl_final3',
    'model': 'I_256_dm ~ pos_dm (excl final 3)',
    'pos_coef': m3b.params['pos_dm'],
    'pos_se': m3b.bse['pos_dm'],
    'pos_p': m3b.pvalues['pos_dm'],
    'r2': m3b.rsquared,
    'n': len(df_excl3)
})

# Interpretation
print(f"\n  Original β: {m1a.params['pos_dm']:.4f}")
print(f"  Excl final 2 β: {m3a.params['pos_dm']:.4f}")
print(f"  Excl final 3 β: {m3b.params['pos_dm']:.4f}")

if m3b.pvalues['pos_dm'] < 0.05:
    print("  → Effect PERSISTS even without final windows (not just closure)")
else:
    print("  → Effect DISAPPEARS without final windows (may be closure-specific)")

# ============================================================
# CHECK 4: STRATIFY BY TASK/STORY
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: STRATIFY BY TASK/STORY")
print("=" * 70)

print("\nQuestion: Is position effect consistent across tasks?")

tasks = df['task'].unique()
task_results = []

print(f"\n{'Task':<25} {'N':>6} {'pos_dm β':>12} {'SE':>10} {'p':>10}")
print("-" * 70)

for task in sorted(tasks):
    task_df = df[df['task'] == task]
    if len(task_df) >= 50:  # Need enough data
        try:
            m_task = smf.ols('I_256_dm ~ pos_dm', data=task_df).fit()
            print(f"{task:<25} {len(task_df):>6} {m_task.params['pos_dm']:>12.4f} {m_task.bse['pos_dm']:>10.4f} {m_task.pvalues['pos_dm']:>10.4f}")

            task_results.append({
                'task': task,
                'n': len(task_df),
                'pos_coef': m_task.params['pos_dm'],
                'pos_se': m_task.bse['pos_dm'],
                'pos_p': m_task.pvalues['pos_dm'],
            })

            model_results.append({
                'check': f'4_task_{task}',
                'model': f'I_256_dm ~ pos_dm (task={task})',
                'pos_coef': m_task.params['pos_dm'],
                'pos_se': m_task.bse['pos_dm'],
                'pos_p': m_task.pvalues['pos_dm'],
                'r2': m_task.rsquared,
                'n': len(task_df)
            })
        except Exception as e:
            print(f"{task:<25} {len(task_df):>6} ERROR: {e}")

# Task × position interaction
print("\n\nModel with task×position interaction:")
m4_int = smf.ols('I_256_dm ~ pos_dm * C(task)', data=df).fit()
print(f"  Main effect pos_dm: β={m4_int.params['pos_dm']:.4f}, p={m4_int.pvalues['pos_dm']:.4f}")

# Check if any interaction terms are significant
int_terms = [c for c in m4_int.params.index if 'pos_dm:' in c]
sig_int = [c for c in int_terms if m4_int.pvalues[c] < 0.05]
print(f"  Significant task×position interactions: {len(sig_int)}/{len(int_terms)}")

# ============================================================
# CHECK 5: AGE × POSITION INTERACTION
# ============================================================
print("\n" + "=" * 70)
print("CHECK 5: AGE × POSITION INTERACTION")
print("=" * 70)

print("\nQuestion: Does position effect vary by age?")

# Continuous age
m5a = smf.ols('I_256_dm ~ pos_dm * age_months + baseline_dm', data=df).fit()
print(f"\nModel 5a: I_256_dm ~ pos_dm * age_months + baseline_dm")
print(f"  pos_dm: β={m5a.params['pos_dm']:.4f}, p={m5a.pvalues['pos_dm']:.4f}")
print(f"  age_months: β={m5a.params['age_months']:.6f}, p={m5a.pvalues['age_months']:.4f}")
print(f"  pos_dm:age_months: β={m5a.params['pos_dm:age_months']:.6f}, p={m5a.pvalues['pos_dm:age_months']:.4f}")

model_results.append({
    'check': '5a_age_interaction',
    'model': 'I_256_dm ~ pos_dm * age_months + baseline_dm',
    'pos_coef': m5a.params['pos_dm'],
    'pos_se': m5a.bse['pos_dm'],
    'pos_p': m5a.pvalues['pos_dm'],
    'interaction_coef': m5a.params['pos_dm:age_months'],
    'interaction_p': m5a.pvalues['pos_dm:age_months'],
    'r2': m5a.rsquared,
    'n': len(df)
})

if m5a.pvalues['pos_dm:age_months'] < 0.05:
    print("  → SIGNIFICANT age×position interaction: effect varies by age")
else:
    print("  → No significant age×position interaction: effect consistent across ages")

# ============================================================
# COMPUTE DRIFT BY TASK WITH CIs
# ============================================================
print("\n" + "=" * 70)
print("DRIFT (Q4-Q1) BY TASK")
print("=" * 70)

drift_by_task = []

for task in sorted(tasks):
    task_df = df[df['task'] == task]

    task_drift = []
    for doc_id, doc_df in task_df.groupby('doc_id'):
        early = doc_df[doc_df['pos_quartile'] == 'early']['I_256']
        late = doc_df[doc_df['pos_quartile'] == 'late']['I_256']

        if len(early) > 0 and len(late) > 0:
            task_drift.append(late.mean() - early.mean())

    if len(task_drift) >= 3:
        mean_drift = np.mean(task_drift)
        se_drift = stats.sem(task_drift)
        t, p = stats.ttest_1samp(task_drift, 0)

        drift_by_task.append({
            'task': task,
            'n_docs': len(task_drift),
            'mean_drift': mean_drift,
            'se': se_drift,
            'ci_low': mean_drift - 1.96 * se_drift,
            'ci_high': mean_drift + 1.96 * se_drift,
            't': t,
            'p': p
        })

drift_task_df = pd.DataFrame(drift_by_task)

print(f"\n{'Task':<25} {'N docs':>8} {'Drift':>10} {'95% CI':>25} {'p':>10}")
print("-" * 85)
for _, row in drift_task_df.iterrows():
    print(f"{row['task']:<25} {row['n_docs']:>8} {row['mean_drift']:>10.3f} [{row['ci_low']:>9.3f}, {row['ci_high']:>9.3f}] {row['p']:>10.4f}")

# ============================================================
# CREATE SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("COEFFICIENT SUMMARY TABLE")
print("=" * 70)

results_df = pd.DataFrame(model_results)
print("\n")
print(results_df[['check', 'model', 'pos_coef', 'pos_se', 'pos_p', 'n']].to_string(index=False))

# Save
results_df.to_csv(OUTPUT_DIR / 'position_checks_coefficients.csv', index=False)
drift_task_df.to_csv(OUTPUT_DIR / 'drift_by_task.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'position_checks_coefficients.csv'}")
print(f"Saved: {OUTPUT_DIR / 'drift_by_task.csv'}")

# ============================================================
# PLOT: DRIFT BY TASK WITH CIs
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by drift magnitude
drift_task_df_sorted = drift_task_df.sort_values('mean_drift')

y_pos = np.arange(len(drift_task_df_sorted))
colors = ['#e74c3c' if p < 0.05 else '#3498db' for p in drift_task_df_sorted['p']]

ax.barh(y_pos, drift_task_df_sorted['mean_drift'], xerr=[
    drift_task_df_sorted['mean_drift'] - drift_task_df_sorted['ci_low'],
    drift_task_df_sorted['ci_high'] - drift_task_df_sorted['mean_drift']
], color=colors, alpha=0.7, capsize=5, edgecolor='black')

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{t}\n(n={n})" for t, n in zip(drift_task_df_sorted['task'], drift_task_df_sorted['n_docs'])])
ax.set_xlabel('Drift (late Q4 - early Q1)', fontsize=12)
ax.set_title('Within-Transcript Drift by Task (± 95% CI)\nRed = p<0.05', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'drift_by_task.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'drift_by_task.png'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF CHECKS")
print("=" * 70)

print(f"""
CHECK 1: Baseline NLL Control
  - Original pos_dm β: {m1a.params['pos_dm']:.4f} (p={m1a.pvalues['pos_dm']:.4f})
  - With baseline_dm β: {m1b.params['pos_dm']:.4f} (p={m1b.pvalues['pos_dm']:.4f})
  - Change: {pct_change:+.1f}%
  → Position effect {'SURVIVES' if m1b.pvalues['pos_dm'] < 0.05 else 'disappears'} controlling for baseline

CHECK 2: Distance-to-End
  - dist_to_end β: {m2a.params['dist_to_end_dm']:.4f} (p={m2a.pvalues['dist_to_end_dm']:.4f})
  → {'Confirms' if m2a.pvalues['dist_to_end_dm'] < 0.05 else 'Does not confirm'} closure/ending effect

CHECK 3: Sensitivity (Exclude Final Windows)
  - Original β: {m1a.params['pos_dm']:.4f}
  - Excl final 2 β: {m3a.params['pos_dm']:.4f} (p={m3a.pvalues['pos_dm']:.4f})
  - Excl final 3 β: {m3b.params['pos_dm']:.4f} (p={m3b.pvalues['pos_dm']:.4f})
  → Effect {'PERSISTS' if m3b.pvalues['pos_dm'] < 0.05 else 'disappears'} without final windows

CHECK 4: Task Stratification
  - {sum(1 for r in task_results if r['pos_p'] < 0.05)}/{len(task_results)} tasks show significant position effect
  → Effect is {'CONSISTENT' if sum(1 for r in task_results if r['pos_p'] < 0.05) >= len(task_results)/2 else 'NOT consistent'} across tasks

CHECK 5: Age × Position Interaction
  - Interaction p: {m5a.pvalues['pos_dm:age_months']:.4f}
  → {'SIGNIFICANT' if m5a.pvalues['pos_dm:age_months'] < 0.05 else 'No significant'} age-dependent position effect
""")
