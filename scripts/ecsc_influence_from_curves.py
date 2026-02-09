#!/usr/bin/env python3
"""
Analyze Influence Distribution from Existing ECSC Gain Curves.

Uses the gain curves (gain_k = NLL_reduction at context k) to derive:
1. tail_share: influence from distant context vs total
2. burstiness: concentration of influence across distance bins
3. kernel collapse: L2 distances between age-group influence profiles

Key insight:
  gain(k) = total benefit from k tokens of context
  influence[a,b] = gain(b) - gain(a) = incremental benefit from bin [a,b]
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/influence_analysis_v1")
OUTPUT_DIR.mkdir(exist_ok=True)

# Lag values
K_VALUES = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
GAIN_COLS = [f'mean_gain_{k}' for k in K_VALUES]
AGE_GROUPS = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']

# Tail thresholds: defines where "distant" context begins
# tail_share_lagX_128 = influence from tokens at lag X to 128 / total influence (gain_128)
# E.g., threshold=64 means tail = tokens at lag 65-128
TAIL_THRESHOLDS = [32, 48, 64]  # Context length thresholds (tail starts at lag threshold+1)

np.random.seed(42)

print("=" * 70)
print("INFLUENCE DISTRIBUTION ANALYSIS FROM EXISTING ECSC CURVES")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. LOADING DATA")
print("-" * 50)

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")

# Extract gain curves
curve_matrix = df[GAIN_COLS].values
print(f"Gain columns: {GAIN_COLS}")

# Filter complete curves
valid_mask = ~np.isnan(curve_matrix).any(axis=1)
df_valid = df[valid_mask].copy()
gains = curve_matrix[valid_mask]
print(f"Complete curves: {len(df_valid)}")

# ============================================================
# 2. COMPUTE INCREMENTAL INFLUENCE PER BIN
# ============================================================
print("\n2. COMPUTING INCREMENTAL INFLUENCE PER BIN")
print("-" * 50)

# Influence in bin [k_i, k_{i+1}] = gain(k_{i+1}) - gain(k_i)
# This gives us influence from context at distance approximately k_i to k_{i+1}

# Define bins by their endpoints (context lengths)
bin_edges = K_VALUES  # [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
n_bins = len(bin_edges) - 1  # 11 bins

# Compute incremental influence for each bin
# inf_bin[i] = gain[i+1] - gain[i]
# This represents influence from tokens at LAG (bin_edges[i]+1) to bin_edges[i+1]
inf_matrix = np.diff(gains, axis=1)  # Shape: (n_docs, n_bins)

# Bin labels using ACTUAL LAG RANGES (not context length endpoints)
# gain(k) includes tokens at lag 1 to k, so:
# gain(b) - gain(a) = influence from tokens at lag (a+1) to b
bin_lag_ranges = []
for i in range(n_bins):
    lag_start = bin_edges[i] + 1 if bin_edges[i] > 0 else 1
    lag_end = bin_edges[i + 1]
    bin_lag_ranges.append((lag_start, lag_end))

bin_labels = [f"lag {r[0]}-{r[1]}" for r in bin_lag_ranges]
bin_centers = [(r[0] + r[1]) / 2 for r in bin_lag_ranges]

print(f"Number of influence bins: {n_bins}")
print(f"Bin labels (actual lag ranges): {bin_labels}")

# Add to dataframe
for i, label in enumerate(bin_labels):
    df_valid[f'inf_{label}'] = inf_matrix[:, i]

# ============================================================
# 3. TAIL SHARE ANALYSIS
# ============================================================
print("\n3. TAIL SHARE ANALYSIS")
print("-" * 50)

def compute_tail_share(gains_row, threshold_k):
    """
    Compute tail_share = influence beyond threshold_k / total influence.

    tail_influence = gain_128 - gain_threshold_k
    total_influence = gain_128
    """
    # Find the gain at threshold
    k_idx = K_VALUES.index(threshold_k) if threshold_k in K_VALUES else None
    if k_idx is None:
        # Interpolate
        for i, k in enumerate(K_VALUES):
            if k > threshold_k:
                k_idx = i - 1
                break

    gain_at_threshold = gains_row[k_idx]
    gain_max = gains_row[-1]  # gain_128

    if gain_max <= 0:
        return np.nan

    tail_influence = gain_max - gain_at_threshold
    tail_share = tail_influence / gain_max
    return tail_share

# Compute tail_share for each threshold
# tail_share_lagX_128 = (gain_128 - gain_threshold) / gain_128
#                     = fraction of total benefit from tokens at lag (threshold+1) to 128
for threshold in TAIL_THRESHOLDS:
    lag_start = threshold + 1
    col_name = f'tail_share_lag{lag_start}_128'
    # Find index of threshold in K_VALUES
    if threshold in K_VALUES:
        k_idx = K_VALUES.index(threshold)
    else:
        k_idx = np.searchsorted(K_VALUES, threshold) - 1

    df_valid[col_name] = (gains[:, -1] - gains[:, k_idx]) / np.where(gains[:, -1] > 0, gains[:, -1], np.nan)

print("""
DEFINITION: tail_share_lagX_128 = (gain_128 - gain_{X-1}) / gain_128
          = fraction of total gain(128) contributed by tokens at lag X to 128
""")

print("Tail Share Statistics:")
print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'95% CI':>25}")
print("-" * 75)

for threshold in TAIL_THRESHOLDS:
    lag_start = threshold + 1
    col = f'tail_share_lag{lag_start}_128'
    mean = df_valid[col].mean()
    std = df_valid[col].std()
    sem = df_valid[col].sem()
    ci_low = mean - 1.96 * sem
    ci_high = mean + 1.96 * sem
    print(f"tail_share_lag{lag_start}_128      {mean:>10.3f} {std:>10.3f}   [{ci_low:.3f}, {ci_high:.3f}]")

# By age group
print("\n\nTail Share by Age Group (tail_share_lag65_128 = influence from lag 65-128):")
print(f"{'Age':<10} {'N':>5} {'Mean':>10} {'Std':>10} {'95% CI':>25}")
print("-" * 65)

for ag in AGE_GROUPS:
    ag_data = df_valid[df_valid['age_group'] == ag]['tail_share_lag65_128']
    if len(ag_data) > 0:
        mean = ag_data.mean()
        std = ag_data.std()
        sem = ag_data.sem()
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
        print(f"{ag:<10} {len(ag_data):>5} {mean:>10.3f} {std:>10.3f}   [{ci_low:.3f}, {ci_high:.3f}]")

# ============================================================
# 4. BURSTINESS METRICS
# ============================================================
print("\n4. BURSTINESS METRICS")
print("-" * 50)

def gini_coefficient(values):
    """Compute Gini coefficient (0 = equal, 1 = concentrated)."""
    values = np.array(values)
    values = np.maximum(values, 0)  # Treat negative as 0
    if values.sum() == 0:
        return np.nan
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * cumsum[-1])) - (n + 1) / n

def effective_support_size(values):
    """
    Effective number of bins (entropy-based).
    Higher = more spread out. Max = n_bins.
    """
    values = np.array(values)
    values = np.maximum(values, 0)
    total = values.sum()
    if total == 0:
        return 0
    probs = values / total
    probs = probs[probs > 0]  # Avoid log(0)
    entropy = -np.sum(probs * np.log(probs))
    return np.exp(entropy)  # Perplexity = effective support

# Compute for each document
burstiness_results = []

for idx, row in enumerate(inf_matrix):
    positive_inf = np.maximum(row, 0)
    total_inf = positive_inf.sum()

    if total_inf > 0:
        gini = gini_coefficient(positive_inf)
        top1_share = positive_inf.max() / total_inf

        # Top-3 share (since we have 11 bins, ~27%)
        sorted_inf = np.sort(positive_inf)[::-1]
        top3_share = sorted_inf[:3].sum() / total_inf

        eff_support = effective_support_size(positive_inf)

        burstiness_results.append({
            'gini': gini,
            'top1_share': top1_share,
            'top3_share': top3_share,
            'effective_support': eff_support,
            'n_positive_bins': (positive_inf > 0).sum(),
        })
    else:
        burstiness_results.append({
            'gini': np.nan,
            'top1_share': np.nan,
            'top3_share': np.nan,
            'effective_support': np.nan,
            'n_positive_bins': 0,
        })

burst_df = pd.DataFrame(burstiness_results)
for col in burst_df.columns:
    df_valid[col] = burst_df[col].values

print("\nBurstiness Statistics:")
print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'95% CI':>25} {'Interp':<20}")
print("-" * 90)

metrics_info = [
    ('gini', 'Gini coefficient', '<0.4 = spread out'),
    ('top1_share', 'Top-1 bin share', '<0.3 = not dominated'),
    ('top3_share', 'Top-3 bins share', '<0.6 = distributed'),
    ('effective_support', 'Effective # bins', f'>5 = diverse (max={n_bins})'),
]

for col, label, interp in metrics_info:
    mean = df_valid[col].mean()
    std = df_valid[col].std()
    sem = df_valid[col].sem()
    ci_low = mean - 1.96 * sem
    ci_high = mean + 1.96 * sem
    print(f"{label:<20} {mean:>10.3f} {std:>10.3f}   [{ci_low:.3f}, {ci_high:.3f}] {interp:<20}")

# By age group
print("\n\nBurstiness by Age Group (DOCUMENT-LEVEL, then averaged):")
print(f"{'Age':<10} {'Gini':>10} {'Top1':>10} {'EffSupp':>10}")
print("-" * 45)

for ag in AGE_GROUPS:
    ag_mask = df_valid['age_group'] == ag
    if ag_mask.sum() > 0:
        gini = df_valid.loc[ag_mask, 'gini'].mean()
        top1 = df_valid.loc[ag_mask, 'top1_share'].mean()
        eff = df_valid.loc[ag_mask, 'effective_support'].mean()
        print(f"{ag:<10} {gini:>10.3f} {top1:>10.3f} {eff:>10.2f}")

# ============================================================
# 4b. KERNEL SHAPE BURSTINESS (pooled mean - DIAGNOSTIC)
# ============================================================
print("\n" + "-" * 50)
print("DIAGNOSTIC: KERNEL SHAPE vs DOCUMENT-LEVEL BURSTINESS")
print("-" * 50)
print("""
NOTE: Document-level metrics above are computed PER DOCUMENT then averaged.
      Kernel shape metrics below are computed on the POOLED MEAN profile.
      If documents have spikes at DIFFERENT positions, pooling hides them.
      Compare these to check for this artifact.
""")

# Compute mean influence profile across all documents
mean_inf_profile = inf_matrix.mean(axis=0)
mean_inf_profile_positive = np.maximum(mean_inf_profile, 0)

# Burstiness on the pooled kernel
kernel_gini = gini_coefficient(mean_inf_profile_positive)
kernel_total = mean_inf_profile_positive.sum()
kernel_top1 = mean_inf_profile_positive.max() / kernel_total if kernel_total > 0 else 0
kernel_top3 = np.sort(mean_inf_profile_positive)[-3:].sum() / kernel_total if kernel_total > 0 else 0
kernel_eff_support = effective_support_size(mean_inf_profile_positive)

print(f"{'Metric':<25} {'Doc-Level (avg)':>18} {'Kernel Shape':>18} {'Difference':>12}")
print("-" * 75)
print(f"{'Gini coefficient':<25} {df_valid['gini'].mean():>18.3f} {kernel_gini:>18.3f} {kernel_gini - df_valid['gini'].mean():>+12.3f}")
print(f"{'Top-1 bin share':<25} {df_valid['top1_share'].mean():>18.3f} {kernel_top1:>18.3f} {kernel_top1 - df_valid['top1_share'].mean():>+12.3f}")
print(f"{'Top-3 bins share':<25} {df_valid['top3_share'].mean():>18.3f} {kernel_top3:>18.3f} {kernel_top3 - df_valid['top3_share'].mean():>+12.3f}")
print(f"{'Effective support':<25} {df_valid['effective_support'].mean():>18.2f} {kernel_eff_support:>18.2f} {kernel_eff_support - df_valid['effective_support'].mean():>+12.2f}")

# Interpretation
diff_gini = kernel_gini - df_valid['gini'].mean()
if abs(diff_gini) < 0.05:
    print(f"\n→ Kernel shape ≈ document-level: no evidence of position-varying spikes")
elif diff_gini < -0.05:
    print(f"\n→ Kernel smoother than docs: documents have spikes at VARYING positions")
    print(f"   (Pooling averages out position-specific bursts)")
else:
    print(f"\n→ Kernel MORE concentrated than docs: unusual pattern")

# Also show the actual mean profile
print(f"\nMean Influence Profile (kernel shape):")
print(f"{'Bin':<12} {'Mean Inf':>12} {'Share':>12} {'Cumul':>12}")
print("-" * 50)
cumul = 0
for i, (label, inf) in enumerate(zip(bin_labels, mean_inf_profile)):
    share = inf / mean_inf_profile.sum() if mean_inf_profile.sum() > 0 else 0
    cumul += share
    print(f"{label:<12} {inf:>12.4f} {share:>12.1%} {cumul:>12.1%}")

# ============================================================
# 5. INFLUENCE-NORMALIZED CURVE DISTANCES
# ============================================================
print("\n5. INFLUENCE-NORMALIZED CURVE DISTANCES")
print("-" * 50)

# Normalize influence profiles by total influence
total_inf_per_doc = np.maximum(inf_matrix.sum(axis=1), 1e-6)
inf_normalized = inf_matrix / total_inf_per_doc[:, np.newaxis]

# Compute mean curves per age group
age_mean_inf = {}
age_indices = {}

for ag in AGE_GROUPS:
    mask = (df_valid['age_group'] == ag).values
    if mask.sum() > 0:
        age_indices[ag] = np.where(mask)[0]
        age_mean_inf[ag] = np.mean(inf_normalized[mask], axis=0)

# Bootstrap L2 distance
def bootstrap_l2_distance(idx1, idx2, data, n_bootstrap=10000):
    """Bootstrap CI for L2 distance between group means."""
    distances = []
    n1, n2 = len(idx1), len(idx2)

    for _ in range(n_bootstrap):
        boot_idx1 = idx1[np.random.randint(0, n1, n1)]
        boot_idx2 = idx2[np.random.randint(0, n2, n2)]

        mean1 = np.mean(data[boot_idx1], axis=0)
        mean2 = np.mean(data[boot_idx2], axis=0)

        l2 = np.sqrt(np.mean((mean1 - mean2) ** 2))
        distances.append(l2)

    return np.percentile(distances, 2.5), np.percentile(distances, 97.5)

# Key comparison: youngest vs oldest
print("\nL2 Distance Between Age Groups (Influence-Normalized Curves):")
print(f"{'Comparison':<20} {'L2 Dist':>10} {'95% CI':>25}")
print("-" * 60)

# 5yr vs 10+yr
l2_span = np.sqrt(np.mean((age_mean_inf['5yr'] - age_mean_inf['10+yr']) ** 2))
ci_low, ci_high = bootstrap_l2_distance(age_indices['5yr'], age_indices['10+yr'], inf_normalized)
print(f"{'5yr vs 10+yr':<20} {l2_span:>10.4f}   [{ci_low:.4f}, {ci_high:.4f}]")

# Adjacent age comparisons
for i in range(len(AGE_GROUPS) - 1):
    ag1, ag2 = AGE_GROUPS[i], AGE_GROUPS[i+1]
    if ag1 in age_mean_inf and ag2 in age_mean_inf:
        l2 = np.sqrt(np.mean((age_mean_inf[ag1] - age_mean_inf[ag2]) ** 2))
        ci_l, ci_h = bootstrap_l2_distance(age_indices[ag1], age_indices[ag2], inf_normalized, n_bootstrap=5000)
        print(f"{ag1} vs {ag2:<12} {l2:>10.4f}   [{ci_l:.4f}, {ci_h:.4f}]")

# ============================================================
# 6. PLOTS
# ============================================================
print("\n6. GENERATING PLOTS")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Influence Profile by Age Group
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, ag in enumerate(AGE_GROUPS):
    if ag in age_mean_inf:
        ax1.plot(bin_centers, age_mean_inf[ag], 'o-', color=colors[i],
                 label=f'{ag} (n={len(age_indices[ag])})', linewidth=2, markersize=6)

ax1.set_xlabel('Token Lag (distance before target)', fontsize=12)
ax1.set_ylabel('Normalized Influence Share', fontsize=12)
ax1.set_title('Influence Profile by Age Group', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative Influence (shows tail contribution)
ax2 = axes[0, 1]

for i, ag in enumerate(AGE_GROUPS):
    if ag in age_mean_inf:
        cumsum = np.cumsum(age_mean_inf[ag])
        ax2.plot(K_VALUES[1:], cumsum, 'o-', color=colors[i], label=ag, linewidth=2)

ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax2.axvline(64, color='red', linestyle=':', alpha=0.5, label='64-token mark')
ax2.set_xlabel('Context Length (tokens)', fontsize=12)
ax2.set_ylabel('Cumulative Influence Share', fontsize=12)
ax2.set_title('Cumulative Influence (tail = 1 - this value)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of Tail Share
ax3 = axes[1, 0]
ax3.hist(df_valid['tail_share_lag65_128'].dropna(), bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(df_valid['tail_share_lag65_128'].mean(), color='red', linestyle='--',
            label=f'Mean = {df_valid["tail_share_lag65_128"].mean():.3f}')
ax3.set_xlabel('tail_share_lag65_128 (influence from lag 65-128)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Tail Influence Share', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Burstiness - Lorenz Curve
ax4 = axes[1, 1]

# Average Lorenz curve across documents
mean_inf_profile = inf_matrix.mean(axis=0)
mean_inf_profile = np.maximum(mean_inf_profile, 0)
sorted_inf = np.sort(mean_inf_profile)
cumsum = np.cumsum(sorted_inf) / sorted_inf.sum()
x_lorenz = np.linspace(0, 1, len(cumsum))

ax4.plot(x_lorenz, cumsum, 'b-', linewidth=2, label='Observed (mean profile)')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect equality')
ax4.fill_between(x_lorenz, x_lorenz, cumsum, alpha=0.2)
ax4.set_xlabel('Cumulative Share of Bins', fontsize=12)
ax4.set_ylabel('Cumulative Share of Influence', fontsize=12)
ax4.set_title(f'Lorenz Curve (Mean Gini = {df_valid["gini"].mean():.3f})', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'influence_analysis_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'influence_analysis_plots.png'}")

# ============================================================
# 7. CONTROLLED REGRESSION
# ============================================================
print("\n7. CONTROLLED REGRESSION: TAIL_SHARE ~ AGE + COVARIATES")
print("-" * 50)

try:
    import statsmodels.formula.api as smf

    df_valid['log_length'] = np.log(df_valid['token_count'])

    # Model 1: Age only
    model1 = smf.ols('tail_share_lag65_128 ~ age_months', data=df_valid).fit()

    # Model 2: With controls
    model2 = smf.ols('tail_share_lag65_128 ~ age_months + log_length + baseline_nll', data=df_valid).fit()

    print("\nModel 1: tail_share_lag65_128 ~ age_months")
    print(f"  Age coef: {model1.params['age_months']:.5f} (p={model1.pvalues['age_months']:.4f})")
    print(f"  R² = {model1.rsquared:.4f}")

    print("\nModel 2: tail_share_lag65_128 ~ age_months + log_length + baseline_nll")
    print(f"  Age coef: {model2.params['age_months']:.5f} (p={model2.pvalues['age_months']:.4f})")
    print(f"  log_length coef: {model2.params['log_length']:.5f} (p={model2.pvalues['log_length']:.4f})")
    print(f"  baseline_nll coef: {model2.params['baseline_nll']:.5f} (p={model2.pvalues['baseline_nll']:.4f})")
    print(f"  R² = {model2.rsquared:.4f}")

except ImportError:
    print("statsmodels not available, skipping regression")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n8. SAVING RESULTS")
print("-" * 50)

# Save document-level metrics
output_cols = ['doc_id', 'age_months', 'age_group', 'token_count', 'baseline_nll',
               'tail_share_lag33_128', 'tail_share_lag49_128', 'tail_share_lag65_128',
               'gini', 'top1_share', 'top3_share', 'effective_support', 'n_positive_bins']
df_valid[output_cols].to_csv(OUTPUT_DIR / 'doc_level_influence_metrics.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'doc_level_influence_metrics.csv'}")

# Save influence profiles per age group
age_profiles = pd.DataFrame({
    'bin_center': bin_centers,
    'bin_label': bin_labels,
    **{f'{ag}_mean': age_mean_inf[ag] for ag in AGE_GROUPS if ag in age_mean_inf}
})
age_profiles.to_csv(OUTPUT_DIR / 'age_group_influence_profiles.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'age_group_influence_profiles.csv'}")

# ============================================================
# 9. INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETATION: RESIDUAL INFLUENCE vs SPARSE RETRIEVAL")
print("=" * 70)

tail_share_mean = df_valid['tail_share_lag65_128'].mean()
tail_share_33 = df_valid['tail_share_lag33_128'].mean()
gini_mean = df_valid['gini'].mean()
top1_mean = df_valid['top1_share'].mean()
eff_support_mean = df_valid['effective_support'].mean()

print(f"""
KEY FINDINGS:

1. TAIL INFLUENCE (tokens at distant lags):
   tail_share_lag65_128: {tail_share_mean:.1%} of gain(128) from tokens at lag 65-128
   tail_share_lag33_128: {tail_share_33:.1%} of gain(128) from tokens at lag 33-128

   Definition: tail_share = (gain_128 - gain_threshold) / gain_128
             = fraction of total predictive benefit from the specified lag band
""")

if tail_share_mean > 0.15:
    print("   → SUPPORTS continuous influence: substantial benefit from distant context")
elif tail_share_mean > 0.05:
    print("   → MODERATE: some benefit from distant context")
else:
    print("   → WEAK: most benefit from recent context only")

print(f"""
2. BURSTINESS (is influence concentrated?):
   Gini coefficient: {gini_mean:.3f} (0=equal, 1=concentrated)
   Top-1 bin share: {top1_mean:.1%}
   Effective support: {eff_support_mean:.1f} bins (max = {n_bins})
""")

if gini_mean < 0.4 and top1_mean < 0.3:
    print("   → SUPPORTS distributed influence: spread across many distance bins")
    print("   → NOT consistent with sparse retrieval bursts")
elif gini_mean > 0.6 or top1_mean > 0.5:
    print("   → SUGGESTS concentrated influence: few bins dominate")
else:
    print("   → MODERATE concentration: some bins contribute more than others")

print(f"""
3. KERNEL COLLAPSE (do age groups have similar influence profiles?):
   5yr vs 10+yr L2 distance: {l2_span:.4f}
""")

if l2_span < 0.02:
    print("   → STRONG collapse: influence profiles nearly identical across ages")
elif l2_span < 0.05:
    print("   → GOOD collapse: minor differences in influence profiles")
else:
    print("   → MODERATE differences: influence profiles vary with age")

print(f"""
OVERALL CONCLUSION:
""")

if tail_share_mean > 0.05 and gini_mean < 0.5 and top1_mean < 0.4:
    print(f"""The data SUPPORT "memory as continuous residual influence":
   - Nonzero predictive support from distant context:
     * {tail_share_mean*100:.1f}% of gain(128) from tokens at lag 65-128
     * {tail_share_33*100:.1f}% of gain(128) from tokens at lag 33-128
   - Influence is distributed across lag bins (Gini={gini_mean:.2f}, top1={top1_mean:.1%})
   - This is consistent with gradient decay, not sparse retrieval bursts""")
else:
    print("The pattern is mixed - further analysis recommended.")
