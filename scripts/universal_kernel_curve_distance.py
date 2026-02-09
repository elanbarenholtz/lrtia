#!/usr/bin/env python3
"""
Curve Distance Analysis for Universal Kernel.

Computes L2 distance between age-group mean normalized curves
with bootstrap confidence intervals.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

# Paths
DATA_FILE = Path("/Users/elanbarenholtz/Projects/lrtia/results/within_child_signature_v1/transcript_level_metrics.csv")
OUTPUT_DIR = Path("/Users/elanbarenholtz/Projects/lrtia/results/universal_kernel_v1")

# Config
K_VALUES = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
GAIN_COLS = [f'mean_gain_{k}' for k in K_VALUES]
AGE_GROUPS = ['5yr', '6yr', '7yr', '8yr', '9yr', '10+yr']
N_BOOTSTRAP = 10000
np.random.seed(42)

print("=" * 70)
print("CURVE DISTANCE ANALYSIS")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transcripts")

# Extract and normalize curves
curve_matrix = df[GAIN_COLS].values
valid_mask = ~np.isnan(curve_matrix).any(axis=1)
df_valid = df[valid_mask].copy()
curves_valid = curve_matrix[valid_mask]

# Normalize
max_gains = curves_valid[:, -1]
max_gains_safe = np.where(max_gains > 0.01, max_gains, 0.01)
curves_normalized = curves_valid / max_gains_safe[:, np.newaxis]

# Skip k=0 for distance calculations (always 0)
curves_for_dist = curves_normalized[:, 1:]  # Shape: (n_docs, 11)
k_array = np.array(K_VALUES[1:])
n_k = len(k_array)

print(f"Using {len(df_valid)} complete curves, {n_k} lag points")

# Compute L2 distance between two curves (normalized by n_k for interpretability)
def curve_l2_distance(curve1, curve2):
    """RMSE between two curves."""
    return np.sqrt(np.mean((curve1 - curve2) ** 2))

# Compute mean curves per age group
age_mean_curves = {}
age_indices = {}

for ag in AGE_GROUPS:
    mask = (df_valid['age_group'] == ag).values
    if mask.sum() > 0:
        age_indices[ag] = np.where(mask)[0]
        age_mean_curves[ag] = np.mean(curves_for_dist[mask], axis=0)

print("\nAge group mean curves computed:")
for ag, curve in age_mean_curves.items():
    print(f"  {ag}: n={len(age_indices[ag])}, mean_norm_at_k64={curve[8]:.3f}")

# ============================================================
# PAIRWISE DISTANCES BETWEEN AGE GROUPS
# ============================================================
print("\n" + "=" * 60)
print("PAIRWISE L2 DISTANCES BETWEEN AGE GROUPS")
print("=" * 60)

def bootstrap_distance(group1_indices, group2_indices, n_bootstrap=10000):
    """Bootstrap CI for L2 distance between group means."""
    distances = []
    n1, n2 = len(group1_indices), len(group2_indices)

    for _ in range(n_bootstrap):
        # Resample indices
        boot_idx1 = group1_indices[np.random.randint(0, n1, n1)]
        boot_idx2 = group2_indices[np.random.randint(0, n2, n2)]

        # Compute mean curves
        mean1 = np.mean(curves_for_dist[boot_idx1], axis=0)
        mean2 = np.mean(curves_for_dist[boot_idx2], axis=0)

        distances.append(curve_l2_distance(mean1, mean2))

    distances = np.array(distances)
    return np.percentile(distances, 2.5), np.percentile(distances, 97.5)

pairwise_results = []

# All pairwise comparisons
print(f"\n{'Pair':<20} {'L2 Distance':>12} {'95% CI':>20}")
print("-" * 55)

for ag1, ag2 in combinations(AGE_GROUPS, 2):
    if ag1 not in age_mean_curves or ag2 not in age_mean_curves:
        continue

    dist = curve_l2_distance(age_mean_curves[ag1], age_mean_curves[ag2])
    ci_low, ci_high = bootstrap_distance(age_indices[ag1], age_indices[ag2], N_BOOTSTRAP)

    print(f"{ag1} vs {ag2:<12} {dist:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")

    pairwise_results.append({
        'group1': ag1,
        'group2': ag2,
        'l2_distance': dist,
        'ci_low': ci_low,
        'ci_high': ci_high,
    })

# ============================================================
# KEY COMPARISON: YOUNGEST VS OLDEST
# ============================================================
print("\n" + "=" * 60)
print("KEY COMPARISON: 5yr vs 10+yr (DEVELOPMENTAL SPAN)")
print("=" * 60)

span_dist = curve_l2_distance(age_mean_curves['5yr'], age_mean_curves['10+yr'])
span_ci_low, span_ci_high = bootstrap_distance(age_indices['5yr'], age_indices['10+yr'], N_BOOTSTRAP)

print(f"\nL2 Distance (5yr vs 10+yr): {span_dist:.4f} [{span_ci_low:.4f}, {span_ci_high:.4f}]")

# ============================================================
# GLOBAL SUMMARY: MAX DISTANCE ACROSS ALL PAIRS
# ============================================================
print("\n" + "=" * 60)
print("GLOBAL SUMMARY")
print("=" * 60)

pairwise_df = pd.DataFrame(pairwise_results)
max_dist = pairwise_df['l2_distance'].max()
max_pair = pairwise_df.loc[pairwise_df['l2_distance'].idxmax()]
min_dist = pairwise_df['l2_distance'].min()
mean_dist = pairwise_df['l2_distance'].mean()

print(f"\nDistance statistics across all age pairs:")
print(f"  Min distance:  {min_dist:.4f}")
print(f"  Mean distance: {mean_dist:.4f}")
print(f"  Max distance:  {max_dist:.4f} ({max_pair['group1']} vs {max_pair['group2']})")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# The normalized curves range from ~0.1 at k=2 to 1.0 at k=128
# So max possible L2 distance is sqrt(mean((1-0)^2)) ≈ 1.0 if curves were opposite
# A distance of 0.02 means ~2% difference on average

print(f"""
Context for interpretation:
- Normalized curves range from ~0.1 (at k=2) to 1.0 (at k=128)
- Maximum possible L2 distance ≈ 1.0 (if curves were maximally different)
- Observed L2 distance: {span_dist:.4f}

As percentage of scale:
- 5yr vs 10+yr difference: ~{span_dist*100:.1f}% of possible range

Conclusion:
""")

if span_dist < 0.02:
    print("  STRONG collapse: <2% shape difference across development")
    print("  Curves are essentially identical in shape")
elif span_dist < 0.05:
    print("  GOOD collapse: <5% shape difference across development")
    print("  Curves have minor shape differences")
else:
    print("  MODERATE collapse: >{span_dist*100:.0f}% shape difference")
    print("  Non-trivial shape changes with age")

# Save results
output_path = OUTPUT_DIR / 'curve_distances.csv'
pairwise_df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Also save summary
summary = {
    '5yr_vs_10yr_l2': span_dist,
    '5yr_vs_10yr_ci_low': span_ci_low,
    '5yr_vs_10yr_ci_high': span_ci_high,
    'mean_pairwise_l2': mean_dist,
    'max_pairwise_l2': max_dist,
    'max_pair': f"{max_pair['group1']}_vs_{max_pair['group2']}",
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_DIR / 'curve_distance_summary.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'curve_distance_summary.csv'}")
