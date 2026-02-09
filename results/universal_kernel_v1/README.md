# Universal Kernel Hypothesis Analysis - ECSC v1

**Date:** 2026-02-02
**Dataset:** ECSC (Edinburgh Child Spoken Corpus)
**Source:** `results/within_child_signature_v1/transcript_level_metrics.csv`

## Hypothesis

Tests whether memory curves have a **shared shape** across age groups, with differences primarily in **magnitude (height)** rather than shape. If true, normalizing curves by their maximum gain should cause them to collapse onto a single universal kernel.

## Data Filters & Thresholds

| Parameter | Value |
|-----------|-------|
| Input file | `transcript_level_metrics.csv` |
| N transcripts | 341 |
| Complete curves required | Yes (no NaN in any gain column) |
| Min gain for normalization | 0.01 (avoid division by zero) |

## Age Groups (6 bins)

| Age Group | N |
|-----------|---|
| 5yr | 86 |
| 6yr | 84 |
| 7yr | 85 |
| 8yr | 45 |
| 9yr | 17 |
| 10+yr | 24 |

## Lag Grid

```
K_VALUES = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
```

Gain columns: `mean_gain_0`, `mean_gain_2`, ..., `mean_gain_128`

## Normalization Method

For each transcript:
```
gain_norm(k) = gain(k) / gain(128)
```

This normalizes all curves to end at 1.0 at k=128, preserving shape while removing magnitude differences.

## Key Results

### CV Reduction
- Mean CV (raw): 0.149
- Mean CV (normalized): 0.106
- **CV reduction: 28.6%** (moderate collapse)

### Variance Decomposition
- Raw curve variance: 0.2608
- Normalized curve variance: 0.0823
- Height (magnitude) contribution: ~5.6%
- Shape contribution: ~94.4%

### Age Correlations (Spearman)
| Metric | rho | p-value |
|--------|-----|---------|
| Magnitude (gain_128) | 0.183 | 0.0007 |
| Shape (log_slope) | 0.207 | 0.0001 |
| Shape (early_ratio) | -0.092 | 0.0881 |

### Per-Age Shape Metrics
| Age | N | Gain_128 | log_slope | early_ratio |
|-----|---|----------|-----------|-------------|
| 5yr | 86 | 1.691 | 0.405 | 0.670 |
| 6yr | 84 | 1.679 | 0.401 | 0.668 |
| 7yr | 85 | 1.772 | 0.428 | 0.661 |
| 8yr | 45 | 1.793 | 0.436 | 0.650 |
| 9yr | 17 | 1.677 | 0.406 | 0.671 |
| 10+yr | 24 | 1.807 | 0.440 | 0.649 |

## Conclusion

**Moderate support** for universal kernel hypothesis:
- Curves visually collapse after height normalization (see plot)
- CV reduces 28.6% after normalizing
- However, shape metric (log_slope) correlates slightly more with age than magnitude
- Shape contributes 94% of remaining variance after normalization

The kernel is approximately universal, but subtle shape differences track development.

## Output Files

| File | Description |
|------|-------------|
| `kernel_collapse_plot.png` | Side-by-side raw vs normalized curves by age |
| `kernel_collapse_stats.csv` | Per-k normalized mean/std by age group |
| `shape_variance_decomposition.txt` | Variance decomposition summary |
| `README.md` | This file |

## Script

`scripts/universal_kernel_analysis.py`
