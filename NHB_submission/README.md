# A scaling law of contextual persistence in human language — reproducibility capsule

This capsule reproduces the analyses and figures in the manuscript *"A scaling law of
contextual persistence in human language"* (Nature Human Behaviour submission).

It measures the **Contextual Persistence Function** `P(d)` — the order-specific predictive
influence of prior context at distance `d`, isolated from token-frequency effects with a
shuffled-token baseline — and shows that across ten human-language corpora `P(d)` decays as
an approximately scale-free power law with exponent near `-1` (mean `-1.04`, median
`r² = 0.96`). The capsule also runs the synthetic-sequence controls, the sentence-level
decomposition and ablation, the two-probe replication (Llama-3.1-8B, Mistral-7B), and the
non-language domain controls (DNA: HyenaDNA; protein: ProGen2).

## Layout

```
NHB_submission/
├── code/
│   ├── core/lrtia/        core CPF pipeline (data → intervention → metrics → aggregation)
│   ├── probes/            probe notebooks that produce the per-corpus CPF caches (GPU)
│   ├── domain_controls/   DNA (HyenaDNA) and protein (ProGen2) control scripts (GPU)
│   ├── analysis/          exponent bootstrap, functional-form/AICc, decomposition stats
│   ├── figures/           build_figures.py (Figs 1–3) and build_schematic_alpha1.py (Fig 4→2)
│   ├── ingest/            build the corpora from their public sources
│   └── run                master entrypoint (see REPRODUCING.md)
├── data/
│   ├── raw/               corpora (see data/DATA_MANIFEST.md for what is bundled vs downloaded)
│   └── derived/           cached per-corpus CPF outputs the figures/stats read
├── results/               figures and fitted-exponent tables are written here
├── environment/           Dockerfile + pinned requirements
├── REPRODUCING.md         step-by-step, with the figure → code → data map
└── AVAILABILITY_STATEMENTS.md   Code Availability and Data Availability text for the paper
```

## Quick start

Full instructions — including which steps require a GPU and which run on CPU in minutes —
are in **REPRODUCING.md**. The short version:

```bash
# 1. environment
pip install -r environment/requirements.txt
pip install -e code/core            # installs the `lrtia` package

# 2. reproduce figures + fitted exponents from the cached CPF outputs (CPU, ~minutes)
bash code/run figures

# 3. (optional, GPU) regenerate the cached CPF outputs from raw text end-to-end
bash code/run all
```

See `REPRODUCING.md` for the GPU requirements of step 3 (large-model probing).
