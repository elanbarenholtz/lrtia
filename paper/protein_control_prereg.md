# Pre-registration: protein-sequence control for the contextual persistence law

**Status:** written and committed *before* running the protein probes. Interpretations
below are fixed in advance so the outcome is read against a committed rule, not chosen
after the fact. (Author: Fable, with E. Barenholtz.)

## Hypothesis under test
The contextual persistence law (P(d) ~ scale-free power law, exponent near unity) is a
signature of sequences generated **autoregressively against a running, logarithmically
compressed memory of the output so far**. The discriminating variable is not
biological-vs-artificial nor communicative-vs-structural, but whether each element is
produced by conditioning on a compressed representation of the preceding sequence during
generation. Language qualifies; autoregressive transformers qualify (predicted to
reproduce the law — the sufficiency result). DNA and protein do **not** qualify: written
by evolution over populations and deep time, not by online generation against a running
memory.

## Prediction
Protein P(d), under the identical protocol, comes out **null**: order-specific persistence
weak in magnitude, poorly fit by a power law, no clean heavy-tailed 1/d decay. Same
direction as the DNA result already confirmed, now on a sequence carrying *more* long-range
dependence than DNA (tertiary-contact coevolution) — a stronger, not weaker, test.

## Probe design (fixed in advance)
**ProGen2 (autoregressive decoder transformer), not ESM-2.** The CPF is defined by
left-to-right conditioning on prior context; the cross-domain comparison is only
interpretable if the *operation* is held identical while the sequence varies. Language
probes (Llama-3.1-8B, Mistral-7B) and the DNA probe (HyenaDNA) are all AR decoders. ESM-2
is a masked bidirectional encoder — it cannot execute the left-to-right CPF without a
left-context-only adaptation that is off-distribution and not the same operation; an
off-distribution null is the most attackable null (indistinguishable from "probe too
weak"). ProGen2 matches objective, architecture family, and protocol exactly, at the cost
of a fiddlier custom HF port (accepted: integration risk is front-loaded and verifiable;
a protocol deviation is a permanent inferential weakness).

- **>= 2 scales** from the ProGen2 ladder (151M, 764M, 2.7B, 6.4B); final choice recorded
  before analysis. Wide span is an asset for the capability-scaling direction.
- **Capability-scaling is the decisive check.** Report per-residue perplexity on held-out
  natural protein; define "stronger" as *lower per-residue loss*. Load-bearing result is
  the direction: if apparent persistence structure shrinks as loss falls, the null cannot
  be a capability ceiling.
- **Identical CPF protocol** to language and DNA: same target span (30), same log-spaced
  context lengths, same shuffled-residue baseline, same d >= 10 fit floor, same
  functional-form comparison.

## Probe-validation steps (required because the port is custom)
1. **Reproduce published ProGen2 per-residue perplexity** on held-out natural protein at
   each scale before the CPF. A faithful port is a precondition for every downstream number.
2. **Within-protein positive-control floor.** Confirm ordered beats shuffled at *short*
   range (must hold, given secondary-structure periodicity). If ordered-vs-shuffled shows
   nothing even short-range, the pipeline is broken, not the biology.

## Interpretation rule (committed before looking)
1. **Null** (weak, non-power-law, no heavy tail; deepening/stable as probe strengthens):
   prediction confirmed. Protein joins DNA as a second non-autoregressively-generated
   sequence lacking the law. Two nulls + capability-scaling make the positive case by
   elimination. Fold in as a named control.
2. **Law-like** (heavy-tailed, exponent near unity, clean fit rivaling language): thesis is
   wrong or incomplete. Do not absorb as a minor caveat. Either (a) invoke the
   autoregressive-sufficiency escape hatch (what about protein evolution mimics
   autoregression-against-memory — coevolutionary constraint as long-range conditioning),
   which weakens but does not destroy the account, or (b) treat as genuine disconfirmation
   and revise the central claim. The choice must itself be justified.
3. **Intermediate** (long-range structure but not clean 1/d — different exponent/shape, or
   law-like only over a sub-range): most likely given protein tertiary structure. Interpret
   against the exponent *value*, not mere presence of structure. The claim was never "only
   language has long-range dependence"; it is that language has scale-free, near-unity
   persistence. Structure that is long-range but not scale-free, or scale-free at a clearly
   different exponent, is consistent with the account — frame as protein having a different
   *kind* of order (structural, coevolutionary), not the log-uniform tail of language.

## Recorded caveats
- **Short-range periodicity, not noise.** The shuffled-residue baseline destroys
  secondary-structure periodicity (alpha-helix ~3.6-residue, beta-strand alternation).
  Expect short-range non-monotonicity in protein P(d), the protein analog of DNA
  codon/nucleosome periodicity — interpret honestly as the method detecting periodic
  structural order, a different object from language's scale-free tail.
- **Architecture/objective match is a strength.** ProGen2 matches the language probes on
  architecture family *and* objective (and the DNA probe on objective; HyenaDNA is AR but
  SSM/conv, not attention). So protein is the tightest match in the whole design: a null
  cannot be attributed to objective mismatch (as with a masked encoder) or to an
  architecture difference from the language probes.
- **Scope limit, range-matched by construction.** Max probed residue distance ~1000
  (capped by the 1024 context), narrower than where coevolutionary contacts live. State
  up front: untested beyond ~1000 residues. But the core comparison is range-matched
  across all three domains — language shows clean 1/d within 0–~1000; the claim is protein
  lacks the law over the same range where language plainly has it. The cap is a limit on
  generality, not a gap in the core comparison. Do not assert protein has "real long-range
  dependence" inside the probed range, since at ~1000 residues it does not reach the
  contact scale.
