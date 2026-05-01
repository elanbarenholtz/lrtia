"""Unit tests for D0–D4 permutation and D5 insertion disruptions."""

import pytest
from collections import Counter

from experiments.exp1_disruption.disruptions.core import (
    d0_no_op,
    d1_reverse_far,
    d2_reverse_near,
    d3_swap_halves,
    d4_full_reverse,
)
from experiments.exp1_disruption.disruptions.insertion import d5_insert_block


# Fixture: deterministic 100-token context
@pytest.fixture
def ctx100():
    return list(range(100))  # [0, 1, 2, ..., 99]
    # ctx[0]=0 is most distant (position 100)
    # ctx[99]=99 is closest to target (position 1)


@pytest.fixture
def M():
    return 50


class TestD0NoOp:
    def test_identical_to_input(self, ctx100, M):
        result = d0_no_op(ctx100, M)
        assert result == ctx100

    def test_length_preserved(self, ctx100, M):
        assert len(d0_no_op(ctx100, M)) == len(ctx100)

    def test_different_M_values(self, ctx100):
        for m in [10, 25, 50, 75, 90]:
            assert d0_no_op(ctx100, m) == ctx100


class TestD1ReverseFar:
    def test_near_half_unchanged(self, ctx100, M):
        result = d1_reverse_far(ctx100, M)
        # Last M tokens should be unchanged
        assert result[-M:] == ctx100[-M:]

    def test_far_half_reversed(self, ctx100, M):
        result = d1_reverse_far(ctx100, M)
        original_far = ctx100[:-M]
        assert result[:-M] == list(reversed(original_far))

    def test_multiset_preserved(self, ctx100, M):
        result = d1_reverse_far(ctx100, M)
        assert Counter(result) == Counter(ctx100)

    def test_length_preserved(self, ctx100, M):
        assert len(d1_reverse_far(ctx100, M)) == len(ctx100)


class TestD2ReverseNear:
    def test_far_half_unchanged(self, ctx100, M):
        result = d2_reverse_near(ctx100, M)
        original_far = ctx100[:-M]
        assert result[:len(original_far)] == original_far

    def test_near_half_reversed(self, ctx100, M):
        result = d2_reverse_near(ctx100, M)
        original_near = ctx100[-M:]
        assert result[-M:] == list(reversed(original_near))

    def test_multiset_preserved(self, ctx100, M):
        result = d2_reverse_near(ctx100, M)
        assert Counter(result) == Counter(ctx100)

    def test_length_preserved(self, ctx100, M):
        assert len(d2_reverse_near(ctx100, M)) == len(ctx100)


class TestD3SwapHalves:
    def test_first_M_tokens_are_original_near(self, ctx100, M):
        result = d3_swap_halves(ctx100, M)
        assert result[:M] == ctx100[-M:]

    def test_last_tokens_are_original_far(self, ctx100, M):
        result = d3_swap_halves(ctx100, M)
        assert result[M:] == ctx100[:-M]

    def test_multiset_preserved(self, ctx100, M):
        result = d3_swap_halves(ctx100, M)
        assert Counter(result) == Counter(ctx100)

    def test_length_preserved(self, ctx100, M):
        assert len(d3_swap_halves(ctx100, M)) == len(ctx100)

    def test_different_M_values(self, ctx100):
        for m in [25, 50, 75]:
            result = d3_swap_halves(ctx100, m)
            assert result[:m] == ctx100[-m:]
            assert result[m:] == ctx100[:-m]


class TestD4FullReverse:
    def test_produces_reversed_input(self, ctx100):
        result = d4_full_reverse(ctx100)
        assert result == list(reversed(ctx100))

    def test_multiset_preserved(self, ctx100):
        result = d4_full_reverse(ctx100)
        assert Counter(result) == Counter(ctx100)

    def test_length_preserved(self, ctx100):
        assert len(d4_full_reverse(ctx100)) == len(ctx100)

    def test_double_reverse_is_identity(self, ctx100):
        assert d4_full_reverse(d4_full_reverse(ctx100)) == ctx100


class TestD5InsertBlock:
    def test_output_length(self, ctx100, M):
        block = [1000, 1001, 1002]
        result = d5_insert_block(ctx100, block, M)
        assert len(result) == len(ctx100) + len(block)

    def test_block_inserted_at_cut(self, ctx100, M):
        block = [1000, 1001, 1002]
        result = d5_insert_block(ctx100, block, M)
        far_len = len(ctx100) - M
        # Far half at the start
        assert result[:far_len] == ctx100[:-M]
        # Block in the middle
        assert result[far_len:far_len + len(block)] == block
        # Near half at the end
        assert result[far_len + len(block):] == ctx100[-M:]

    def test_original_tokens_preserved(self, ctx100, M):
        block = [1000, 1001, 1002]
        result = d5_insert_block(ctx100, block, M)
        # All original tokens present
        result_without_block = result[:len(ctx100) - M] + result[len(ctx100) - M + len(block):]
        assert Counter(result_without_block) == Counter(ctx100)

    def test_block_tokens_added(self, ctx100, M):
        block = [1000, 1001, 1002]
        result = d5_insert_block(ctx100, block, M)
        result_counter = Counter(result)
        for tok in block:
            assert result_counter[tok] >= 1

    def test_k20_block(self, ctx100, M):
        block = list(range(1000, 1020))  # K=20
        result = d5_insert_block(ctx100, block, M)
        assert len(result) == 120  # C + K = 100 + 20

    def test_different_M_values(self, ctx100):
        block = [999]
        for m in [25, 50, 75]:
            result = d5_insert_block(ctx100, block, m)
            assert len(result) == 101
            assert result[len(ctx100) - m] == 999


class TestMultisetPreservation:
    """Cross-cutting: D0-D4 preserve multiset, D5 adds block tokens."""

    def test_all_permutation_disruptions_preserve_multiset(self, ctx100):
        for fn, kwargs in [
            (d0_no_op, {'M': 50}),
            (d1_reverse_far, {'M': 50}),
            (d2_reverse_near, {'M': 50}),
            (d3_swap_halves, {'M': 50}),
            (d4_full_reverse, {}),
        ]:
            result = fn(ctx100, **kwargs)
            assert Counter(result) == Counter(ctx100), f"{fn.__name__} altered multiset"

    def test_d5_adds_block_tokens(self, ctx100):
        block = [500, 501, 502]
        result = d5_insert_block(ctx100, block, M=50)
        expected = Counter(ctx100)
        expected.update(block)
        assert Counter(result) == expected
