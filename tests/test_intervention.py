"""Tests for intervention module."""

import pytest

from lrtia.intervention.span_selection import SpanSelector, SpanInfo
from lrtia.intervention.transforms import (
    MaskReplace,
    ShuffleSpan,
    DeleteSpan,
    apply_intervention,
    InterventionType,
)


class TestSpanInfo:
    def test_basic_properties(self):
        span = SpanInfo(start=10, end=20, distance=32, actual_distance=30, width=10)
        assert len(span) == 10
        assert span.center == 15.0


class TestSpanSelector:
    def test_basic_selection(self):
        selector = SpanSelector(
            distances=[32, 64],
            widths=[16],
            spans_per_distance=1,
            seed=42,
        )

        spans = selector.select_spans(
            target_position=100,
            context_length=200,
            doc_id="test_doc",
            window_id=0,
        )

        # Should get spans for each distance
        assert len(spans) >= 1

        for span in spans:
            # All spans should be before target
            assert span.end <= 100
            # Span width should be correct
            assert span.end - span.start <= 16

    def test_reproducibility(self):
        selector = SpanSelector(
            distances=[32, 64, 128],
            widths=[16],
            spans_per_distance=1,
            seed=42,
        )

        spans1 = selector.select_spans(100, 200, "doc1", 0)
        spans2 = selector.select_spans(100, 200, "doc1", 0)

        # Same inputs should give same outputs
        assert len(spans1) == len(spans2)
        for s1, s2 in zip(spans1, spans2):
            assert s1.start == s2.start
            assert s1.end == s2.end

    def test_different_seeds(self):
        selector1 = SpanSelector(distances=[64], widths=[16], seed=42)
        selector2 = SpanSelector(distances=[64], widths=[16], seed=123)

        spans1 = selector1.select_spans(100, 200, "doc1", 0)
        spans2 = selector2.select_spans(100, 200, "doc1", 0)

        # Different seeds may give different positions
        # (though this isn't guaranteed for all cases)
        # Just check both return valid spans
        assert len(spans1) > 0
        assert len(spans2) > 0


class TestMaskReplace:
    def test_mask_replace(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        span = SpanInfo(start=2, end=5, distance=32, actual_distance=32, width=3)

        intervention = MaskReplace(mask_token_id=0)
        result = intervention.apply(token_ids, span)

        # Check mask applied
        assert result.token_ids[2:5] == [0, 0, 0]
        # Check length preserved
        assert len(result.token_ids) == len(token_ids)
        # Check other tokens unchanged
        assert result.token_ids[:2] == [1, 2]
        assert result.token_ids[5:] == [6, 7, 8, 9, 10]
        # Check flags
        assert not result.length_changed
        assert result.original_tokens == [3, 4, 5]
        assert result.new_tokens == [0, 0, 0]


class TestShuffleSpan:
    def test_shuffle_span(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        span = SpanInfo(start=2, end=6, distance=32, actual_distance=32, width=4)

        intervention = ShuffleSpan()
        result = intervention.apply(token_ids, span, seed=42)

        # Check length preserved
        assert len(result.token_ids) == len(token_ids)
        assert not result.length_changed

        # Check original tokens are shuffled (same elements, different order)
        assert sorted(result.token_ids[2:6]) == sorted([3, 4, 5, 6])

        # Check other tokens unchanged
        assert result.token_ids[:2] == [1, 2]
        assert result.token_ids[6:] == [7, 8, 9, 10]

    def test_shuffle_reproducibility(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        span = SpanInfo(start=2, end=6, distance=32, actual_distance=32, width=4)

        intervention = ShuffleSpan()
        result1 = intervention.apply(token_ids, span, seed=42)
        result2 = intervention.apply(token_ids, span, seed=42)

        assert result1.token_ids == result2.token_ids


class TestDeleteSpan:
    def test_delete_span(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        span = SpanInfo(start=2, end=5, distance=32, actual_distance=32, width=3)

        intervention = DeleteSpan()
        result = intervention.apply(token_ids, span)

        # Check deletion
        assert len(result.token_ids) == 7  # 10 - 3
        assert result.token_ids == [1, 2, 6, 7, 8, 9, 10]
        assert result.length_changed
        assert result.original_tokens == [3, 4, 5]
        assert result.new_tokens == []


class TestApplyIntervention:
    def test_apply_mask(self):
        token_ids = [1, 2, 3, 4, 5]
        span = SpanInfo(start=1, end=3, distance=32, actual_distance=32, width=2)

        result = apply_intervention(token_ids, span, "mask", mask_token_id=0)
        assert result.intervention_type == InterventionType.MASK
        assert result.token_ids[1:3] == [0, 0]

    def test_apply_shuffle(self):
        token_ids = [1, 2, 3, 4, 5]
        span = SpanInfo(start=1, end=4, distance=32, actual_distance=32, width=3)

        result = apply_intervention(token_ids, span, "shuffle", seed=42)
        assert result.intervention_type == InterventionType.SHUFFLE
        assert len(result.token_ids) == 5

    def test_apply_delete(self):
        token_ids = [1, 2, 3, 4, 5]
        span = SpanInfo(start=1, end=3, distance=32, actual_distance=32, width=2)

        result = apply_intervention(token_ids, span, "delete")
        assert result.intervention_type == InterventionType.DELETE
        assert len(result.token_ids) == 3
