# LRTIA - Long-Range Token Influence Analyzer

A causal intervention framework that quantifies how earlier text influences current predictions in language models, producing "memory curves" (effect size vs. distance) with summary metrics.

## Overview

LRTIA performs causal interventions on language model inputs to measure how predictions at a target position depend on tokens at various distances. By systematically masking, shuffling, or deleting spans of text and measuring the resulting changes in model predictions, LRTIA produces "memory curves" that characterize the effective context window of a model.

## Features

- **Multiple intervention types**: Mask replacement, token shuffling, span deletion
- **Flexible distance sampling**: Log-spaced distance buckets for efficient coverage
- **Rich metrics**: ΔNLL, KL divergence, top-k stability, rank change
- **Aggregation**: Per-document, per-author, and per-population curves
- **Summary statistics**: AUC, half-life, tail mass
- **Statistical comparisons**: Bootstrap confidence intervals for group comparisons
- **Visualization**: Memory curves, comparison plots, intervention breakdowns

## Installation

```bash
# Clone the repository
git clone https://github.com/example/lrtia.git
cd lrtia

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run analysis on a corpus
lrtia run --config configs/default.yaml --data corpus.jsonl

# Generate plots
lrtia plot --results results/evaluations.parquet --output figures/

# Compare populations
lrtia compare --results results/evaluations.parquet --groups "population_a,population_b"
```

## Configuration

See `configs/default.yaml` for the full configuration schema. Key settings:

```yaml
model:
  name: "gpt2"
  backend: "huggingface"
  device: "cuda"

windowing:
  window_tokens: 2048
  stride_tokens: 512

spans:
  distances: [32, 64, 128, 256, 512, 1024, 2048]
  widths: [32]
  spans_per_distance: 1

interventions:
  types: ["mask", "shuffle"]
```

## Data Format

Input corpora should be JSONL, CSV, or Parquet with the following schema:

```json
{"doc_id": "doc_001", "author_id": "author_01", "population": "group_a", "text": "..."}
```

## Output

LRTIA produces:

- **Per-evaluation results**: Parquet files with full metrics for each intervention
- **Aggregated curves**: JSON/CSV with mean, std, CI by distance
- **Summary metrics**: JSON with AUC, half-life, tail mass per group

## License

MIT License
