"""Span selection for causal interventions."""

import random
from dataclasses import dataclass


@dataclass
class SpanInfo:
    """Information about a selected span for intervention."""

    start: int  # Start index (inclusive)
    end: int  # End index (exclusive)
    distance: int  # Distance bucket
    actual_distance: int  # Actual distance from span center to target
    width: int  # Span width

    @property
    def center(self) -> float:
        """Center position of the span."""
        return (self.start + self.end) / 2

    def __len__(self) -> int:
        """Length of the span."""
        return self.end - self.start


class SpanSelector:
    """Selects spans at various distances from target positions."""

    def __init__(
        self,
        distances: list[int],
        widths: list[int],
        spans_per_distance: int = 1,
        seed: int = 42,
    ):
        """Initialize the span selector.

        Args:
            distances: List of distance buckets (tokens from span center to target).
            widths: List of span widths to use.
            spans_per_distance: Number of spans to sample per distance bucket.
            seed: Random seed for reproducibility.
        """
        self.distances = sorted(distances)
        self.widths = sorted(widths)
        self.spans_per_distance = spans_per_distance
        self.seed = seed

    def select_spans(
        self,
        target_position: int,
        context_length: int,
        doc_id: str = "",
        window_id: int = 0,
    ) -> list[SpanInfo]:
        """Select spans at various distances from the target position.

        Args:
            target_position: Position of the target token (or start of target region).
            context_length: Total length of the context (number of tokens).
            doc_id: Document ID for deterministic seeding.
            window_id: Window ID for deterministic seeding.

        Returns:
            List of SpanInfo objects for each selected span.
        """
        # Create deterministic RNG based on seed and identifiers
        combined_seed = hash((self.seed, doc_id, window_id, target_position)) % (2**32)
        rng = random.Random(combined_seed)

        spans = []

        for distance in self.distances:
            for width in self.widths:
                selected = self._select_spans_for_distance(
                    target_position=target_position,
                    context_length=context_length,
                    distance=distance,
                    width=width,
                    rng=rng,
                )
                spans.extend(selected)

        return spans

    def _select_spans_for_distance(
        self,
        target_position: int,
        context_length: int,
        distance: int,
        width: int,
        rng: random.Random,
    ) -> list[SpanInfo]:
        """Select spans for a specific distance and width.

        Args:
            target_position: Position of the target.
            context_length: Total context length.
            distance: Target distance from span center to target.
            width: Width of the span.
            rng: Random number generator.

        Returns:
            List of SpanInfo objects.
        """
        # Calculate the ideal span center position
        # Distance is measured from span center to target position
        ideal_center = target_position - distance

        # Calculate valid range for span start
        # Span must fit within [0, target_position)
        half_width = width // 2
        min_center = half_width
        max_center = target_position - half_width - 1

        if min_center > max_center:
            # Not enough room for a span at this distance
            return []

        # Constrain center to be close to ideal distance
        # Allow some tolerance for finding valid positions
        tolerance = max(width // 2, 16)  # At least half span width tolerance
        search_min = max(min_center, ideal_center - tolerance)
        search_max = min(max_center, ideal_center + tolerance)

        if search_min > search_max:
            # Cannot place a span at this distance
            return []

        spans = []
        selected_centers = set()

        for _ in range(self.spans_per_distance):
            # Find a valid center position
            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                center = rng.randint(search_min, search_max)
                if center not in selected_centers:
                    selected_centers.add(center)
                    break
                attempts += 1
            else:
                # Could not find a unique position
                if selected_centers:
                    # Reuse an existing center if we must
                    center = rng.choice(list(selected_centers))
                else:
                    continue

            # Calculate span boundaries
            start = center - half_width
            end = start + width

            # Ensure span is within bounds
            if start < 0:
                start = 0
                end = width
            if end > target_position:
                end = target_position
                start = max(0, end - width)

            actual_distance = target_position - (start + end) // 2

            span = SpanInfo(
                start=start,
                end=end,
                distance=distance,
                actual_distance=actual_distance,
                width=end - start,
            )
            spans.append(span)

        return spans

    def get_all_spans_for_target(
        self,
        target_start: int,
        target_end: int,
        context_length: int,
        doc_id: str = "",
        window_id: int = 0,
    ) -> list[SpanInfo]:
        """Get all intervention spans for a target region.

        Args:
            target_start: Start of target region.
            target_end: End of target region.
            context_length: Total context length.
            doc_id: Document ID for deterministic seeding.
            window_id: Window ID for deterministic seeding.

        Returns:
            List of SpanInfo objects.
        """
        # Use target_start as the reference point for distance calculation
        return self.select_spans(
            target_position=target_start,
            context_length=context_length,
            doc_id=doc_id,
            window_id=window_id,
        )
