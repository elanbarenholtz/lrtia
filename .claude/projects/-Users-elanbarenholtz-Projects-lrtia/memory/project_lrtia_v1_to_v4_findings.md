---
name: LRTIA v1-v7 findings progression
description: Comprehensive record of RAID analysis findings across all notebook versions, including methodological lessons, what survived controls, and final results.
type: project
---

## v1-v4: Methodological development (see detailed notes below)

Perplexity-based measures were contaminated by distributional calibration (v3 control). Token-level accuracy was flat after shuffled subtraction (v3). Expected embedding similarity showed small but real signal (v4/v4b). These led to the insight that token-level metrics miss long-range coherence — need sentence-level measurement.

## v5 (Sentence Contrast): First clean sentence-level signal
- Contrast score (log ppl_impostor/ppl_real) increases monotonically W4→W128 (0.32→1.34, 4x gain)
- Impostors from other docs in same genre — controls for topic
- Shuffled text shows 25% less context benefit than intact — the sequential coherence component
- Universal across 8 genres

## v6 (Sentence Influence Matrices): Revealing the structure
- Leave-one-sentence-out design produces sentence × sentence influence matrices
- Influence significantly positive out to 16+ sentences
- **Individual sentences do NOT decay smoothly** (monotonicity = 0.54, ~random)
- Smooth aggregate = average over spiky, content-driven individual curves
- **Long-range hotspots**: 3.5% of pairs at distance 15+ show substantial influence
- **Human vs AI**: AI has 2x hotspot density at distance 10+ (14.4% vs 7.9%)
- Load-bearing sentences: character introductions, ingredient lists, thesis statements

## v7 (Fine-Grained Token-by-Token): Power law discovery
- Token-by-token context reveal (1 to 100 tokens) on 266 documents
- **Marginal benefit follows power law: influence ~ distance^(-α)**
- Human: α = 1.21 [95% CI: 1.13-1.29], n=108
- AI: α = 1.43 [95% CI: 1.34-1.53], n=146
- **Human vs AI significantly different**: t=3.42, p=0.0007, d=0.45
- 50% of benefit in first 8 tokens, 90% by 62 tokens
- Step at ~15 tokens (sentence boundary): 77x jump in marginal gain

## Developmental results
- ECSC (scaffolded frog stories): age × distance interaction r=0.111, p=0.045 at W128
- ECSC influence matrices: no age effect on hotspot density (scaffolding masks individual differences)
- Peterson-McCabe (free narratives): same direction but non-significant (texts too short for W128)

## Theoretical framing (2026-04-12)
- **NOT Ebbinghaus** (wrong timescale, wrong units)
- **Item-based WM interference** literature is correct comparison (exponents 1.0-1.5)
- Memory = generative influence measured in intervening items
- Anderson & Schooler (1991) as bridge: demand-side (0.77) vs supply-side (1.21)
- See project_lrtia_theoretical_framing.md for full argument

## Key methodological lessons
- Always run shuffled + uniform random controls
- Perplexity conflates three signals; sentence contrast and influence matrices isolate coherence
- Leave-one-out influence is confounded by text length (shorter texts inflate values)
- Base models are better measuring instruments than instruct models
