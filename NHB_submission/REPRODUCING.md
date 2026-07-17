# Reproducing the manuscript

This document maps every figure and headline number to the code and data that produce it,
and states what runs on CPU versus what needs a GPU.

## Two reproduction paths

**A. Fast path (CPU, minutes).** Regenerate all figures and fitted exponents from the cached
per-corpus CPF outputs in `data/derived/`. This is what a reviewer should run first; it
reproduces every figure and statistic in the paper without a GPU.

```bash
pip install -r environment/requirements.txt
pip install -e code/core
bash code/run figures
```

**B. Full path (GPU, hours).** Rebuild the cached CPF outputs from raw text by re-running the
large-model probes, then rebuild the figures. This reproduces the pipeline end to end.

```bash
bash code/run all
```

## GPU / environment requirements for the full path

| Stage | Probe | Hardware |
|---|---|---|
| Language CPF (Figs 1–3) | Llama-3.1-8B (base), Mistral-7B-v0.1 | 1× A100-40GB (or ≥24 GB) |
| DNA control | HyenaDNA medium-160k, large-1M | 1× A100 (fits smaller too) |
| Protein control | ProGen2 small (151M), large (2.7B) | 1× A100-40GB |

The probe notebooks in `code/probes/` were run on Google Colab A100 instances; the domain
controls in `code/domain_controls/` are plain `python` scripts. Model weights are pulled from
the Hugging Face Hub at runtime (`Meta-Llama-3.1-8B`, `mistralai/Mistral-7B-v0.1`,
`LongSafari/hyenadna-*`, `hugohrban/progen2-*`); a Hugging Face token with Llama access is
required for the language probes.

## Figure → code → data map

| Item | Code | Reads | Produces |
|---|---|---|---|
| **Fig 1** cross-corpus P(d) | `figures/build_figures.py` `fig1()` | `data/derived/corpus_expansion_longrange/llama/<cell>.json` (10 corpora) | `results/fig1_persistence.pdf` |
| **Fig 2** (schematic α) | `figures/build_schematic_alpha1.py` | — (conceptual) | `results/schematic_alpha1.pdf` |
| **Fig 3** order/content decomposition | `figures/build_figures.py` `fig2()`/`fig3()` | `.../llama/<cell>.json`, `random_vocab_*`, sent-shuffle & sent-reverse caches | `results/fig2_sentence_shuffle.pdf` |
| **Fig 4** distributed ablation | `figures/build_figures.py` `fig4()` | leave-one-sentence-out caches | `results/fig3_distributed_influence.pdf` |
| Mean exponent, CI, r² | `analysis/bootstrap_exponents.py` | `.../llama/<cell>.json` | printed table + `results/exponents.csv` |
| Functional-form / AICc (SI) | `analysis/longrange_functional_form.py` | same | `results/functional_form_aicc.csv` |
| Chaining vs content slopes | `analysis/sentshuffle_decomposition.py` | sent-shuffle/reverse caches | printed slopes |
| Probe independence | `analysis/model_comparison.py` | llama vs mistral caches | printed comparison |
| **DNA control** | `domain_controls/persistence_dna_hyenadna.py` (+ `_evo2.py`) | downloads GRCh38 chr21 | `data/derived/domain_controls/dna_*.json` |
| **Protein control** | `domain_controls/run_progen2_dense.py` | downloads Swiss-Prot | `data/derived/domain_controls/protein_*_dense.json` |

> Note on figure numbering: `build_figures.py` was written before the α schematic was added.
> In the manuscript the schematic is Figure 2 and the sentence-shuffle / distributed-influence
> panels are Figures 3 and 4; the builder's internal function names (`fig2`, `fig3`) refer to
> the older ordering. Output filenames are stable.

## Which cells are which corpus (Fig 1, 10 corpora)

`en_fiction`, `en_news`, `buckeye` (spoken), `ted_en`, `ted_de`, `ted_fr`, `ted_tr`,
`ted_ru`, `ja_literary`, `fi_literary`. Document counts (≥1 full 1024-token window): 60, 60,
26, 60, 60, 31, 12, 60, 42, 38 = **449**.

## Building corpora from source (full path only)

`code/ingest/` builds the corpora from their public sources; see `data/DATA_MANIFEST.md` for
per-corpus source, license, and whether the cleaned text is bundled here or must be downloaded
(Buckeye requires accepting its license at the source and cannot be redistributed in this
capsule).

## Exact versions

The protein control was generated with `transformers==5.12.1`, `torch==2.11.0+cu128` on an
A100-40GB. The language and DNA probes are tolerant to `transformers>=4.44`. Pinned versions
are in `environment/requirements.txt`; `environment/Dockerfile` builds the capsule image.
Random seed throughout: `20260715`.
