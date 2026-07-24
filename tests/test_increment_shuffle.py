"""Tests for the Exp 2 increment-shuffle transform."""

import random

import pytest

from lrtia.intervention.increment_shuffle import (
    stable_seed,
    increment_shuffle_prefix,
    near_and_far_shuffle_prefix,
    random_token_increment_prefix,
    ladder_pairs,
    LadderPair,
)

LADDER = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


class TestStableSeed:
    def test_deterministic_across_calls(self):
        a = stable_seed("doc__chunk_001", "doc__chunk_001_pos50", 256, 7)
        b = stable_seed("doc__chunk_001", "doc__chunk_001_pos50", 256, 7)
        assert a == b

    def test_varies_with_inputs(self):
        base = stable_seed("doc", "tgt", 256, 0)
        assert stable_seed("doc", "tgt", 256, 1) != base
        assert stable_seed("doc", "tgt", 128, 0) != base
        assert stable_seed("doc2", "tgt", 256, 0) != base

    def test_non_negative_and_bounded(self):
        s = stable_seed("x", 1, 2, 3)
        assert 0 <= s < (1 << 63)


class TestIncrementShufflePrefix:
    def test_length_preserved(self):
        pfx = list(range(100, 116))  # c_cur = 16
        out = increment_shuffle_prefix(pfx, c_prev=8, c_cur=16, seed=1)
        assert len(out) == len(pfx)

    def test_near_band_untouched(self):
        # near band = last c_prev tokens (distances 1..c_prev)
        pfx = list(range(100, 116))  # 16 tokens
        c_prev, c_cur = 8, 16
        out = increment_shuffle_prefix(pfx, c_prev, c_cur, seed=3)
        assert out[c_cur - c_prev:] == pfx[c_cur - c_prev:]

    def test_far_band_is_permutation(self):
        pfx = list(range(100, 116))
        c_prev, c_cur = 8, 16
        out = increment_shuffle_prefix(pfx, c_prev, c_cur, seed=3)
        band_end = c_cur - c_prev
        assert sorted(out[:band_end]) == sorted(pfx[:band_end])

    def test_far_band_actually_reordered(self):
        pfx = list(range(100, 116))
        out = increment_shuffle_prefix(pfx, c_prev=8, c_cur=16, seed=3)
        assert out[:8] != pfx[:8]  # multi-token band must change order

    def test_single_token_band_is_identity(self):
        # (1, 2) pair: far band is a single token -> nothing to shuffle
        pfx = [11, 22]
        out = increment_shuffle_prefix(pfx, c_prev=1, c_cur=2, seed=5)
        assert out == pfx

    def test_deterministic(self):
        pfx = list(range(50))
        s = stable_seed("d", "t", 50, 2)
        a = increment_shuffle_prefix(pfx, 10, 50, s)
        b = increment_shuffle_prefix(pfx, 10, 50, s)
        assert a == b

    def test_does_not_mutate_input(self):
        pfx = list(range(20))
        snapshot = list(pfx)
        increment_shuffle_prefix(pfx, 5, 20, seed=1)
        assert pfx == snapshot

    def test_rejects_bad_lengths(self):
        with pytest.raises(ValueError):
            increment_shuffle_prefix([1, 2, 3], c_prev=1, c_cur=4, seed=1)
        with pytest.raises(ValueError):
            increment_shuffle_prefix([1, 2, 3, 4], c_prev=4, c_cur=4, seed=1)


class TestNullControls:
    def test_near_and_far_shuffle_is_full_permutation(self):
        pfx = list(range(30))
        out = near_and_far_shuffle_prefix(pfx, c_prev=8, c_cur=30, seed=2)
        assert sorted(out) == sorted(pfx)
        assert out != pfx

    def test_random_token_replaces_only_far_band(self):
        pfx = list(range(1000, 1016))
        c_prev, c_cur = 8, 16
        out = random_token_increment_prefix(
            pfx, c_prev, c_cur, seed=1, vocab_size=128256
        )
        assert out[c_cur - c_prev:] == pfx[c_cur - c_prev:]  # near intact
        # far band should now be within vocab range
        assert all(0 <= t < 128256 for t in out[: c_cur - c_prev])


class TestLadderPairs:
    def test_skips_zero_interval(self):
        pairs = ladder_pairs(LADDER)
        assert all(p.c_prev >= 1 for p in pairs)
        assert pairs[0] == LadderPair(index=2, c_prev=1, c_cur=2)
        assert pairs[-1].c_prev == 512 and pairs[-1].c_cur == 1024

    def test_geometry(self):
        p = LadderPair(index=6, c_prev=16, c_cur=32)
        assert p.width == 16
        assert abs(p.distance - (16 * 32) ** 0.5) < 1e-9

    def test_count(self):
        # ladder has 12 entries incl 0; 11 intervals; skip 0->1 => 10 pairs
        assert len(ladder_pairs(LADDER)) == 10
