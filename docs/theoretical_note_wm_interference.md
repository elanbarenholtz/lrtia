# Contextual Influence Decay in Language and the Working Memory Literature: A Theoretical Note

## Background

Recent empirical work measuring long-range contextual influence in natural language (Barenholtz et al., in prep) found that the marginal benefit of each additional token of prior context follows a power law: influence ~ distance^(−α), with α = 1.21 (95% CI: 1.13–1.29) for human-authored text across 8 genres. The original write-up noted that this exponent falls within the range of the Ebbinghaus forgetting curve and suggested a potential connection to memory processes. On reflection, this comparison is probably the wrong one, and a better theoretical anchor exists.

## Why Ebbinghaus Is the Wrong Comparison

The Ebbinghaus forgetting curve was derived from a very specific paradigm: memorizing lists of nonsense syllables and measuring retention via the savings method at intervals ranging from 20 minutes to 31 days. It is fundamentally a measure of long-term memory operating over real clock time, with exponents typically in the range of 0.1 to 0.5.

Contextual influence in language production operates over a completely different scale. Within a single document or spoken narrative, the relevant "distances" span seconds to a few minutes at most. More importantly, the natural unit of measurement is not time at all but intervening linguistic items: tokens, words, or sentences. Invoking Ebbinghaus therefore conflates two very different memory regimes and introduces a units mismatch that weakens rather than strengthens the theoretical connection.

The comparison to Ebbinghaus should be dropped or mentioned only to dismiss it.

## The Correct Comparison: Item-Based Working Memory Forgetting

The working memory literature provides a much more appropriate empirical anchor, for two reasons.

First, the relevant timescale matches. Working memory research addresses forgetting over seconds to minutes, which corresponds to the temporal span of language production within a document or utterance.

Second, and more importantly, the dominant view in contemporary WM research is that forgetting is better characterized as a function of intervening items than elapsed time. A substantial body of evidence, developed principally by Lewandowsky, Oberauer, and Nairne, argues that what degrades WM representations is not passive temporal decay but interference from subsequent items. Experiments showing that filling a delay with articulatory suppression causes forgetting while simply extending the delay does not are particularly compelling. This item-based framing maps directly onto the token/sentence distance measure used in the contextual influence analysis.

## Convergence of Exponents

When the WM forgetting literature is examined specifically for item-based power law exponents, the numerical convergence with α = 1.21 is striking:

- Smith, Corbett, Lilburn & Kyllingsbæk (2018) reported power law exponents of 1.0–1.5 for VWM precision as a function of set size (in variance units)
- Emrich et al. (2017) reported k = 1.2 and k = 1.43 in the same metric
- Donkin & Nosofsky (2012) demonstrated that recognition memory strength follows a near-perfect power function of lag (intervening items), with individual exponents in a similar range
- Kahana & Adler (2002) proved mathematically that power law forgetting emerges inevitably from any system with heterogeneous interference rates across items, with simulated exponents ranging from 0.64 to 1.03

By contrast, time-based empirical exponents cluster between 0.1 and 0.5, far below 1.21. The value of 1.21 is simply inconsistent with time-based decay and consistent with item-based interference dynamics.

## Anderson & Schooler as a Bridge

Anderson & Schooler (1991) provide a particularly important bridge between the memory and language literatures. They showed that the probability of needing previously encountered information declines as a power function of recency, and critically, when measured in intervening utterances rather than elapsed time (using the CHILDES corpus of child-directed speech), the power law held with an exponent of approximately 0.77. Their argument was that human memory is adapted to match the statistical structure of environmental information demand, which itself follows power laws.

This is essentially the inverse of the contextual influence measurement: Anderson & Schooler measured how often prior content is needed again as a function of intervening items; the RAID analysis measures how much prior content still influences current production as a function of intervening items. The gap between their exponent (0.77) and the RAID exponent (1.21) is theoretically informative: generative influence decays somewhat faster than environmental information need, which could reflect the additional bottleneck imposed by working memory capacity on top of the statistical structure inherent in language.

## The Theoretical Interpretation: Memory as Generative Influence

The deeper theoretical claim motivating this comparison is that memory just is generative influence. On this view, "remembering" something means that prior content continues to shape current generation. Forgetting is the decay of that generative influence over intervening items. There is no separate storage-and-retrieval process; there is only ongoing generation, and the persistence of prior content in that process is what memory consists of.

This reframing has several consequences. First, it dissolves the units problem: if memory is generative influence and generative influence is measured in item distance, then the natural unit of memory is items, not time. This is precisely what the WM interference literature has been converging on empirically. Second, it makes the equivalence of exponents a theoretical prediction rather than an empirical coincidence. If the retrieval-side WM literature and the production-side text analysis are measuring the same underlying process from different angles, they should yield the same functional form and similar exponents. They do.

Third, it explains the human/AI divergence in the RAID data. AI-generated text shows a steeper decay exponent (α = 1.43) than human text (α = 1.21). AI systems have no working memory constraint: they have full access to all prior context. Yet their text shows a similar, if steeper, power law decay signature. The natural explanation is that AI has absorbed the generative statistics of human language production through training, including the decay structure that biological working memory imposes. The divergence in exponent reflects the fact that AI reproduces the human statistical signature without being subject to the underlying constraint, and does so imperfectly.

## Summary of the Argument

1. The Ebbinghaus comparison in the original write-up should be replaced with a comparison to the item-based WM forgetting literature, which operates on the right timescale and uses the right units.
2. The power law exponent of α = 1.21 from the RAID contextual influence analysis is numerically consistent with item-based WM forgetting exponents (1.0–1.5) and inconsistent with time-based forgetting exponents (0.1–0.5).
3. Anderson & Schooler (1991) establish that both memory and environmental information structure follow power laws as a function of intervening items, providing a principled theoretical bridge.
4. The convergence of exponents is a prediction of the hypothesis that memory is generative influence: production-side and retrieval-side measures should reflect the same underlying process.
5. The human/AI divergence in exponent is consistent with the interpretation that AI mimics human generative statistics, including the WM-imposed decay signature, without being subject to the underlying biological constraint.
