#!/usr/bin/env python3
"""
Controlled Regression for Shape Metrics.

Regresses shape metric ~ age + story/task + length + baseline_nll (fluency).
Reports age coefficient with and without covariates.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/universal_kernel_v1")

# Metrics to analyze
SHAPE_METRICS = ['log_slope_local', 'early_ratio']
MAGNITUDE_METRIC = 'mean_gain_128'

print("=" * 70)
print("CONTROLLED REGRESSION ANALYSIS")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")

# Check task availability
print(f"\nTask values: {df['task'].unique()}")
print(f"Task counts:\n{df['task'].value_counts()}")
print(f"Missing task: {df['task'].isna().sum()}")

# Prepare variables
df['log_length'] = np.log(df['token_count'])

# For task, create dummy coding (drop NaN rows for task-controlled models)
df_with_task = df.dropna(subset=['task']).copy()
print(f"\nDocs with task info: {len(df_with_task)}")

results = []

for metric in SHAPE_METRICS + [MAGNITUDE_METRIC]:
    print(f"\n{'='*60}")
    print(f"METRIC: {metric}")
    print(f"{'='*60}")

    # Model 1: Age only
    model1 = smf.ols(f'{metric} ~ age_months', data=df).fit()
    age_coef1 = model1.params['age_months']
    age_se1 = model1.bse['age_months']
    age_p1 = model1.pvalues['age_months']
    r2_1 = model1.rsquared

    print(f"\nModel 1: {metric} ~ age_months")
    print(f"  N = {len(df)}")
    print(f"  Age coef: {age_coef1:.5f} (SE={age_se1:.5f}, p={age_p1:.4f})")
    print(f"  R² = {r2_1:.4f}")

    # Model 2: Age + length + baseline_nll
    model2 = smf.ols(f'{metric} ~ age_months + log_length + baseline_nll', data=df).fit()
    age_coef2 = model2.params['age_months']
    age_se2 = model2.bse['age_months']
    age_p2 = model2.pvalues['age_months']
    r2_2 = model2.rsquared

    print(f"\nModel 2: {metric} ~ age_months + log_length + baseline_nll")
    print(f"  N = {len(df)}")
    print(f"  Age coef: {age_coef2:.5f} (SE={age_se2:.5f}, p={age_p2:.4f})")
    print(f"  log_length coef: {model2.params['log_length']:.5f} (p={model2.pvalues['log_length']:.4f})")
    print(f"  baseline_nll coef: {model2.params['baseline_nll']:.5f} (p={model2.pvalues['baseline_nll']:.4f})")
    print(f"  R² = {r2_2:.4f}")

    # Model 3: Full model with task (on subset)
    model3 = smf.ols(f'{metric} ~ age_months + log_length + baseline_nll + C(task)', data=df_with_task).fit()
    age_coef3 = model3.params['age_months']
    age_se3 = model3.bse['age_months']
    age_p3 = model3.pvalues['age_months']
    r2_3 = model3.rsquared

    print(f"\nModel 3: {metric} ~ age_months + log_length + baseline_nll + task")
    print(f"  N = {len(df_with_task)} (only docs with task info)")
    print(f"  Age coef: {age_coef3:.5f} (SE={age_se3:.5f}, p={age_p3:.4f})")
    print(f"  R² = {r2_3:.4f}")

    # Task coefficients
    task_coefs = {k: v for k, v in model3.params.items() if 'task' in k}
    if task_coefs:
        print(f"  Task effects (vs reference):")
        for k, v in task_coefs.items():
            p = model3.pvalues[k]
            print(f"    {k}: {v:.4f} (p={p:.4f})")

    # Change in age coefficient
    change_with_controls = (age_coef2 - age_coef1) / abs(age_coef1) * 100 if age_coef1 != 0 else 0
    change_with_task = (age_coef3 - age_coef2) / abs(age_coef2) * 100 if age_coef2 != 0 else 0

    print(f"\n  Age coefficient changes:")
    print(f"    After length+fluency controls: {change_with_controls:+.1f}%")
    print(f"    After adding task: {change_with_task:+.1f}% (on task-available subset)")

    results.append({
        'metric': metric,
        'n_full': len(df),
        'n_task_subset': len(df_with_task),
        'age_coef_basic': age_coef1,
        'age_se_basic': age_se1,
        'age_p_basic': age_p1,
        'r2_basic': r2_1,
        'age_coef_controlled': age_coef2,
        'age_se_controlled': age_se2,
        'age_p_controlled': age_p2,
        'r2_controlled': r2_2,
        'age_coef_with_task': age_coef3,
        'age_se_with_task': age_se3,
        'age_p_with_task': age_p3,
        'r2_with_task': r2_3,
        'pct_change_controls': change_with_controls,
        'pct_change_task': change_with_task,
    })

# Save results
results_df = pd.DataFrame(results)
output_path = OUTPUT_DIR / 'controlled_regression_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\n\nSaved: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: DO CONTROLS CHANGE THE AGE EFFECT?")
print("=" * 70)

print(f"\n{'Metric':<20} {'Age coef (basic)':>18} {'Age coef (full)':>18} {'Change':>10}")
print("-" * 70)
for r in results:
    print(f"{r['metric']:<20} {r['age_coef_basic']:>18.5f} {r['age_coef_with_task']:>18.5f} {r['pct_change_controls']:>+10.1f}%")

print(f"""

Key interpretation:
- If age coefficient stays significant after controls → robust developmental effect
- If age coefficient shrinks substantially → confounded by length/fluency/task
- If task explains variance → different tasks elicit different curve shapes
""")
