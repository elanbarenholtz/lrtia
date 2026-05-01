# Experiment 1: Naturalistic Disruption Validation

Tests whether the corrected-marginal measure is locally sensitive to known
order disruptions introduced into natural-language context.

## Protocol
See `../../paper/exp1_structured_disruption.md` for the full specification.

## Pre-registration
`PREREG.md` — must be committed, tagged, and pushed before any inferential
analysis runs. See Phase 8 of the implementation instructions.

## Structure
```
disruptions/     — D0-D4 permutation + D5a-d insertion transforms
pipeline/        — probe wrapper, condition-specific marginals, runner
stats/           — bootstrap, breakpoint, correlations, meta-analysis
plots/           — per-condition, localization, aggregate figures
tests/           — unit tests for transforms and baselines
config/          — exp1.yaml (all hyperparameters)
scripts/         — 01_run_pipeline.py, 02_run_stats.py, 03_make_figures.py
results/         — output data (Parquet)
```

## Reproduction
```bash
# 1. Pre-registration must be locked first
# 2. Run pipeline (requires H100 GPU)
python scripts/01_run_pipeline.py --config config/exp1.yaml
# 3. Run statistical analysis
python scripts/02_run_stats.py
# 4. Generate figures
python scripts/03_make_figures.py
```

## Probes
- `unsloth/Meta-Llama-3.1-8B` (base, 4-bit BnB)
- `mistralai/Mistral-7B-v0.1` (4-bit BnB)
