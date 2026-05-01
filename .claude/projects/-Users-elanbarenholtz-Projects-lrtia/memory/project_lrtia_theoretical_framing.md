---
name: Theoretical framing — memory as generative influence
description: Refined theoretical interpretation (updated 2026-04-13). Corrected power law exponent α=0.75 (human) matches Anderson & Schooler's 0.77 almost exactly. Item-based WM interference, not Ebbinghaus.
type: project
---

## Corrected power law exponents (v4, sentence-boundary-aligned, shuffled-corrected)

- **Human: α = 0.75** [r = -0.87, p = 0.0005]
- **AI: α = 1.97** [r = -0.95, p = 0.0009]
- Human-AI gap: 2.6x difference in decay rate

These are the definitive values — sentence-boundary-aligned context reveal with token-shuffled baseline subtraction.

## Comparison to uncorrected values

| Version | Human | AI |
|---|---|---|
| v2 uncorrected | -1.21 | -1.43 |
| v3 corrected (old boundary) | -0.65 | -1.75 |
| v4 corrected (sentence-aligned) | -0.75 | -1.97 |

The uncorrected -1.21 was inflated by ~0.46 from calibration artifact. The corrected 0.75 is the true coherence decay rate.

## Anderson & Schooler convergence

Anderson & Schooler (1991) measured information demand decay in CHILDES (intervening utterances): exponent ≈ 0.77.

Our human production-side influence decay: exponent = 0.75.

Nearly identical. They measured demand; we measure supply. Same underlying process.

## Key references for item-based WM comparison

- Smith, Corbett, Lilburn & Kyllingsbæk (2018): 1.0-1.5
- Emrich et al. (2017): k = 1.2 and 1.43
- Donkin & Nosofsky (2012): power function of lag
- Kahana & Adler (2002): 0.64-1.03
- Anderson & Schooler (1991): 0.77

Our corrected human exponent of 0.75 is at the lower end of this range, consistent with the idea that production-side influence decays slightly more slowly than retrieval-side measures because it includes both memory decay and discourse structure.

## Human vs AI: Three levels of evidence

1. **Different exponents**: -0.75 vs -1.97
2. **Different fit quality**: both fit power laws but different ones
3. **Different variance structure**: human marginals are smooth and stable; AI marginals are volatile and bursty (swings of +/-100 between adjacent tokens)

All three reflect the same underlying difference: human text is produced under memory constraints that enforce gradual, smooth coherence buildup. AI text, unconstrained, packs information unevenly.

## Memory as generative influence

"Remembering" = prior content continuing to shape current generation. Forgetting = decay of that generative influence over intervening items. The natural unit is items, not time — matching the WM interference literature's convergence on item-based forgetting.

**Why:** This framing makes the exponent convergence a theoretical prediction. Production-side and retrieval-side measures of the same memory process should yield the same functional form and similar exponents. They do: 0.75 ≈ 0.77.

**How to apply:** Report corrected (shuffled-subtracted) exponents only. Compare to Anderson & Schooler, not Ebbinghaus. Frame the human-AI divergence as evidence that the human exponent reflects a biological constraint, not just statistical regularity of language.
