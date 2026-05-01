---
name: Always run shuffled and uniform random controls
description: Hard-won lesson from RAID analysis — perplexity-based context ablation results are uninterpretable without proper controls. Always include shuffled tokens and uniform random tokens as baselines.
type: feedback
---

Never present perplexity-based memory curves or context-scaling results without shuffled-token and uniform-random controls. The perplexity metric conflates distributional calibration with genuine predictive structure, and this can produce dramatic-looking but artifactual results.

**Why:** In the RAID analysis, the "universal memory curve" across genres appeared to be a strong finding about human cognition. Controls revealed it was primarily an artifact of how LMs reduce perplexity by estimating input statistics — present even for random text.

**How to apply:** Every LRTIA notebook should include (1) shuffled tokens from real docs (preserves vocabulary, destroys order) and (2) uniform random tokens from full vocabulary (destroys everything). The real signal = condition minus shuffled baseline.
