"""Corrected-marginal computation with condition-specific shuffled baselines.

For each condition X and context length c:
  1. Get the disrupted full context for condition X
  2. The "ordered" prefix at length c = last c tokens of the disrupted context
  3. The "shuffled" baseline at length c = average PPL over n_shuffles random
     permutations of the SAME c-token multiset
  4. Marginal at distance d: m_d = ppl[c=d-1] - ppl[c=d]
  5. Corrected marginal: Delta_d = m_d_ordered - m_d_shuffled

Critical: at every (condition, c), the shuffled baseline shuffles the c tokens
that the ordered prefix at that c contains for THAT condition. Different
conditions reveal different prefixes at the same c.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from experiments.exp1_disruption.pipeline.probe import Probe


@dataclass
class MarginalResult:
    """Result of corrected-marginal computation for one target + condition."""
    distances: list[int]          # d = 1, 2, ..., C (or C+K for D5)
    ppl_ordered: list[float]      # PPL at each c for ordered prefix
    ppl_shuffled: list[float]     # PPL at each c for shuffled baseline (averaged)
    m_ordered: list[float]        # raw ordered marginals
    m_shuffled: list[float]       # raw shuffled marginals
    delta: list[float]            # corrected marginals (ordered - shuffled)


def compute_corrected_marginals(
    target_ids: list[int],
    disrupted_ctx: list[int],
    probe: Probe,
    n_shuffles: int = 50,
    seed: int = 42,
) -> MarginalResult:
    """Compute corrected marginals with condition-specific shuffled baselines.

    Args:
        target_ids: Token IDs of the target region.
        disrupted_ctx: Full disrupted context, ordered [most_distant, ..., closest].
            This is whatever the disruption function produced for this condition.
            For D0-D4: length C. For D5: length C+K.
        probe: Loaded Probe instance.
        n_shuffles: Number of random permutations to average for shuffled baseline.
        seed: Random seed for shuffled permutations (deterministic).

    Returns:
        MarginalResult with per-distance corrected marginals.
    """
    C = len(disrupted_ctx)
    rng = random.Random(seed)

    ppl_ordered = []
    ppl_shuffled = []

    # Compute PPL at each context length c = 0, 1, ..., C
    for c in range(C + 1):
        if c == 0:
            # No context — just target tokens
            chunk = list(target_ids)
            ppl = probe.compute_ppl(chunk, 0, len(chunk))
            ppl_ordered.append(ppl)
            ppl_shuffled.append(ppl)  # no context to shuffle
        else:
            # Ordered prefix: last c tokens of disrupted context
            ordered_prefix = disrupted_ctx[-c:]
            chunk = ordered_prefix + list(target_ids)
            ppl = probe.compute_ppl(chunk, len(ordered_prefix), len(chunk))
            ppl_ordered.append(ppl)

            # Shuffled baseline: shuffle the same c-token multiset n_shuffles times
            shuffled_ppls = []
            for _ in range(n_shuffles):
                shuffled_prefix = list(ordered_prefix)
                rng.shuffle(shuffled_prefix)
                chunk_s = shuffled_prefix + list(target_ids)
                ppl_s = probe.compute_ppl(chunk_s, len(shuffled_prefix), len(chunk_s))
                if not math.isinf(ppl_s):
                    shuffled_ppls.append(ppl_s)

            if shuffled_ppls:
                ppl_shuffled.append(np.mean(shuffled_ppls))
            else:
                ppl_shuffled.append(ppl_ordered[-1])  # fallback

    # Compute marginals: m_d = ppl[c=d-1] - ppl[c=d]
    # d=1 is the marginal from adding the first (closest) token
    distances = list(range(1, C + 1))
    m_ordered = []
    m_shuffled = []
    delta = []

    for d in distances:
        mo = ppl_ordered[d - 1] - ppl_ordered[d]  # drop in PPL from adding token d
        ms = ppl_shuffled[d - 1] - ppl_shuffled[d]
        m_ordered.append(mo)
        m_shuffled.append(ms)
        delta.append(mo - ms)

    return MarginalResult(
        distances=distances,
        ppl_ordered=ppl_ordered,
        ppl_shuffled=ppl_shuffled,
        m_ordered=m_ordered,
        m_shuffled=m_shuffled,
        delta=delta,
    )
