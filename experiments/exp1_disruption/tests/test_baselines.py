"""Unit tests for condition-specific shuffled baselines.

These tests verify the corrected-marginal computation without requiring
a GPU. They use a mock probe that returns deterministic perplexity values
based on token identity, allowing us to verify the shuffling and marginal
logic independently of the LLM.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.exp1_disruption.pipeline.marginals import (
    MarginalResult,
    compute_corrected_marginals,
)
from experiments.exp1_disruption.disruptions.core import d0_no_op


class MockProbe:
    """A mock probe that returns deterministic PPL based on token sum.

    PPL = base_ppl - sum(context_tokens) * scale
    This means ordered context (which preserves the multiset) always
    gives the same PPL as any permutation of the same multiset.
    For testing the shuffled baseline: since PPL depends only on the
    multiset (not order), m_shuffled should equal m_ordered and
    delta should be zero.
    """

    def __init__(self, base_ppl: float = 100.0, scale: float = 0.1):
        self.base_ppl = base_ppl
        self.scale = scale
        self.call_count = 0

    def compute_ppl(self, token_ids: list[int], target_start: int, target_end: int) -> float:
        self.call_count += 1
        context = token_ids[:target_start]
        # PPL decreases with more context (sum of token values)
        return self.base_ppl - sum(context) * self.scale


class OrderSensitiveProbe:
    """A mock probe where PPL depends on token ORDER, not just multiset.

    PPL = base - sum(ctx[i] * weight[i]) where weight decays with position.
    Tokens closer to target (later positions) contribute more when they
    have higher values. This makes the probe sensitive to order.
    """

    def __init__(self, base_ppl: float = 100.0):
        self.base_ppl = base_ppl

    def compute_ppl(self, token_ids: list[int], target_start: int, target_end: int) -> float:
        context = token_ids[:target_start]
        if not context:
            return self.base_ppl
        # Weight by position: later (closer to target) positions matter more
        weighted_sum = sum(
            tok * (1.0 / (len(context) - i))
            for i, tok in enumerate(context)
        )
        return self.base_ppl - weighted_sum * 0.01


class TestMarginalComputation:
    """Test the basic marginal computation mechanics."""

    def test_output_shape(self):
        probe = MockProbe()
        target = [500, 501, 502]
        ctx = list(range(10))  # 10-token context
        result = compute_corrected_marginals(target, ctx, probe, n_shuffles=5, seed=42)

        assert len(result.distances) == 10
        assert len(result.ppl_ordered) == 11  # c=0..10
        assert len(result.ppl_shuffled) == 11
        assert len(result.m_ordered) == 10
        assert len(result.m_shuffled) == 10
        assert len(result.delta) == 10
        assert result.distances == list(range(1, 11))

    def test_ppl_decreases_with_context(self):
        probe = MockProbe()
        target = [500]
        ctx = list(range(1, 11))  # tokens 1-10
        result = compute_corrected_marginals(target, ctx, probe, n_shuffles=5, seed=42)

        # PPL should decrease as context grows
        for i in range(len(result.ppl_ordered) - 1):
            assert result.ppl_ordered[i] >= result.ppl_ordered[i + 1]

    def test_marginals_are_positive(self):
        """With a monotonically decreasing PPL, marginals should be positive."""
        probe = MockProbe()
        target = [500]
        ctx = list(range(1, 11))
        result = compute_corrected_marginals(target, ctx, probe, n_shuffles=5, seed=42)

        for m in result.m_ordered:
            assert m >= 0

    def test_zero_context_ppl_matches(self):
        """At c=0, ordered and shuffled PPL should be identical (no context)."""
        probe = MockProbe()
        target = [500, 501]
        ctx = list(range(10))
        result = compute_corrected_marginals(target, ctx, probe, n_shuffles=5, seed=42)

        assert result.ppl_ordered[0] == result.ppl_shuffled[0]


class TestDeterminism:
    def test_same_seed_same_result(self):
        probe = MockProbe()
        target = [500, 501]
        ctx = list(range(20))

        r1 = compute_corrected_marginals(target, ctx, probe, n_shuffles=10, seed=42)
        r2 = compute_corrected_marginals(target, ctx, probe, n_shuffles=10, seed=42)

        assert r1.ppl_ordered == r2.ppl_ordered
        np.testing.assert_array_almost_equal(r1.ppl_shuffled, r2.ppl_shuffled)
        np.testing.assert_array_almost_equal(r1.delta, r2.delta)

    def test_different_seed_different_shuffled(self):
        probe = OrderSensitiveProbe()
        target = [500, 501]
        ctx = list(range(20))

        r1 = compute_corrected_marginals(target, ctx, probe, n_shuffles=10, seed=42)
        r2 = compute_corrected_marginals(target, ctx, probe, n_shuffles=10, seed=99)

        # Ordered should be identical
        assert r1.ppl_ordered == r2.ppl_ordered
        # Shuffled should differ (different random permutations)
        assert r1.ppl_shuffled != r2.ppl_shuffled


class TestD0EquivalenceToIntact:
    """D0 (no-op) must produce corrected marginals identical to intact."""

    def test_d0_matches_intact(self):
        probe = OrderSensitiveProbe()
        target = [500, 501, 502]
        ctx = list(range(100))

        # Intact
        intact_result = compute_corrected_marginals(
            target, ctx, probe, n_shuffles=10, seed=42
        )

        # D0: cut and rejoin (should be identical)
        d0_ctx = d0_no_op(ctx, M=50)
        d0_result = compute_corrected_marginals(
            target, d0_ctx, probe, n_shuffles=10, seed=42
        )

        # Must match exactly (same tokens, same order, same seed)
        assert intact_result.ppl_ordered == d0_result.ppl_ordered
        np.testing.assert_array_almost_equal(
            intact_result.ppl_shuffled, d0_result.ppl_shuffled
        )
        np.testing.assert_array_almost_equal(
            intact_result.delta, d0_result.delta
        )


class TestConditionSpecificBaseline:
    """Verify that different conditions produce different shuffled baselines."""

    def test_c1_depends_on_revealed_token(self):
        """At c=1, the shuffled baseline is a single token (no permutation
        possible). The PPL at c=1 should depend on which single token is
        revealed, which differs across conditions."""
        probe = OrderSensitiveProbe()
        target = [500]

        # Context where token values differ by position
        ctx = list(range(100, 200))  # tokens 100..199
        # ctx[-1] = 199 (closest to target, revealed at c=1 for intact)

        from experiments.exp1_disruption.disruptions.core import d3_swap_halves

        intact_result = compute_corrected_marginals(
            target, ctx, probe, n_shuffles=5, seed=42
        )
        d3_ctx = d3_swap_halves(ctx, M=50)
        d3_result = compute_corrected_marginals(
            target, d3_ctx, probe, n_shuffles=5, seed=42
        )

        # At c=1: intact reveals ctx[-1]=199, D3 reveals the swapped token
        # These should produce different PPLs
        assert intact_result.ppl_ordered[1] != d3_result.ppl_ordered[1]

    def test_shuffled_baseline_differs_across_conditions(self):
        """Different conditions at the same c should have different shuffled
        baselines because they shuffle different token multisets."""
        probe = OrderSensitiveProbe()
        target = [500]
        ctx = list(range(100, 200))

        from experiments.exp1_disruption.disruptions.core import d4_full_reverse

        intact_result = compute_corrected_marginals(
            target, ctx, probe, n_shuffles=20, seed=42
        )
        d4_ctx = d4_full_reverse(ctx)
        d4_result = compute_corrected_marginals(
            target, d4_ctx, probe, n_shuffles=20, seed=42
        )

        # At small c (e.g., c=5), intact and D4 reveal different 5-token sets
        # (intact: last 5 of ctx, D4: last 5 of reversed ctx = first 5 of ctx)
        # So their shuffled baselines should differ
        # At c=100 (full context), multisets are identical so shuffled should match
        assert intact_result.ppl_shuffled[5] != d4_result.ppl_shuffled[5]
        # At c=100 (all tokens), multisets are the same so shuffled baselines
        # should be close (not exact — different seeds produce different
        # permutation sequences, so with finite n_shuffles there's sampling noise)
        diff_at_100 = abs(intact_result.ppl_shuffled[100] - d4_result.ppl_shuffled[100])
        diff_at_5 = abs(intact_result.ppl_shuffled[5] - d4_result.ppl_shuffled[5])
        # The difference at c=100 (same multiset) should be much smaller than
        # the difference at c=5 (different multisets)
        assert diff_at_100 < diff_at_5
