# Availability statements (for the manuscript)

Draft text for the manuscript's end-matter. Fill the bracketed placeholders once the Code
Ocean capsule and any data archive have DOIs.

## Code Availability

All analysis code — the Contextual Persistence Function pipeline, the large-model probing
scripts, the synthetic and non-language (DNA, protein) controls, and the figure- and
statistic-generating scripts — is available as a Code Ocean capsule at
[https://doi.org/10.24433/CO.XXXXXXX.vN] and is archived at [Zenodo DOI]. The capsule
reproduces every figure and reported statistic; a CPU "fast path" rebuilds all figures and
fitted exponents from cached per-corpus outputs, and a GPU "full path" reruns the model
probes end to end. Code is released under the [MIT / BSD-3-Clause] licence.

## Data Availability

The contextual persistence functions analysed in this study were computed from ten
publicly available corpora: English fiction (Project Gutenberg), an English news corpus
(XL-Sum), spontaneous spoken English (Buckeye Corpus of Conversational Speech), TED-talk
transcripts in English, German, French, Turkish and Russian, and literary prose in Japanese
(Aozora Bunko) and Finnish (Project Gutenberg / Wikisource). The non-language controls used
the human reference genome GRCh38 chromosome 21 (Ensembl) and UniProtKB/Swiss-Prot. Per-corpus
sources, licences and preprocessing are detailed in Supplementary Information and in the
capsule's DATA_MANIFEST. Derived contextual-persistence outputs (numerical per-distance
perplexity summaries, containing no source text) are included in the capsule. Corpora that
cannot be redistributed under their source licences (notably the Buckeye Corpus) must be
obtained from the original providers; scripts to reconstruct the cleaned text from each source
are provided. Model weights are publicly available from the Hugging Face Hub
(Meta-Llama-3.1-8B, Mistral-7B-v0.1, LongSafari/hyenadna-*, hugohrban/progen2-*).
