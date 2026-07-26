# Cover letter — Nature Human Behaviour

Dear Editors,

Please consider our manuscript, "A scaling law of contextual persistence in human language," for publication as an Article in *Nature Human Behaviour*.

The paper reports the discovery of a quantitative law governing how the influence of prior context on language decays with distance. Human language has long been known to exhibit scaling laws — Zipf's law for word frequency, Heaps' law for vocabulary growth — but these describe words in aggregate. No law has described the dimension of language in which grammar and meaning actually live: the arrangement of words in sequence. Using large language models purely as measuring instruments, we isolated the predictive influence of sequential arrangement from that of word content, and found that it decays as an approximate 1/d power law — the contextual persistence function P(d), with mean exponent α = 1.04 (SD = 0.15) across ten corpora spanning six language families and both written and spoken modalities.

Two features of the result give it, we believe, broad significance for the behavioural sciences. First, the near-unity exponent is not a generic heavy tail: α ≈ 1 is the unique boundary at which contextual influence is distributed approximately equally across logarithmic timescales — a specific organization of influence across scales, not merely the absence of a horizon. Second, the law is a property of integrated ordered context rather than a sum of separable span contributions: a localized order-disruption experiment shows that the internal order of isolated distant spans decays too steeply to carry the law, which emerges only at the level of the accumulating ordered sequence.

The measurement completes a program with a long history. Shannon's guessing experiments asked how uncertainty about upcoming text falls as more preceding context is given, and Hilberg's conjecture holds that the answer follows a power law; but in that tradition order itself was never a variable. Our design makes it one, and locates the lawful structure specifically in the arrangement component that prior measurements could not separate.

The result is extensively controlled: the decay vanishes in scrambled and frequency-matched synthetic sequences, replicates across independently trained probe models (Llama-3.1-8B, Mistral-7B), survives permutation-robustness and unit-robustness analyses, and does not appear in genomic or protein sequences probed with domain-native models — indicating a regularity of human language rather than of sequence statistics or of the instrument. We deliberately limit mechanistic claims: the law characterizes the linguistic product, not the producing process. It nonetheless yields specific, testable behavioural and neural predictions — reading-time, interference, and N400 costs of context disruption that should fall off with distance at the near-unity rate — and its convergence with model-free power-law measurements in memory research (environmental information demand, free-recall dynamics, the form of forgetting) opens a direct bridge between language statistics and the psychology of memory.

The manuscript is approximately 4,200 words of main text with five figures and includes Supplementary Information covering functional-form comparisons, robustness analyses, and corpus provenance. All analysis code and derived data will be deposited with a DOI upon acceptance.

This work is not under consideration elsewhere, has not been published previously, and the author declares no competing interests. [If a preprint has been or will be posted, note it here.]

Suggested reviewers: [names/emails — consider researchers spanning quantitative linguistics / language statistics, computational psycholinguistics, and memory.]

Thank you for your consideration.

Sincerely,

Elan Barenholtz
Department of Psychology and Center for Complex Systems
Florida Atlantic University
elanbarenholtz@gmail.com
