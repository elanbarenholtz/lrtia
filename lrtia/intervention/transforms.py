"""Intervention transforms for causal analysis."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lrtia.intervention.span_selection import SpanInfo


class InterventionType(str, Enum):
    """Types of interventions."""

    MASK = "mask"
    SHUFFLE = "shuffle"
    DELETE = "delete"


@dataclass
class InterventionResult:
    """Result of applying an intervention."""

    token_ids: list[int]
    intervention_type: InterventionType
    span_info: SpanInfo
    length_changed: bool
    original_tokens: list[int]  # Original tokens in the span
    new_tokens: list[int]  # New tokens in the span (or empty for delete)

    @property
    def length_delta(self) -> int:
        """Change in length (negative for deletion)."""
        return len(self.new_tokens) - len(self.original_tokens)


class Intervention(ABC):
    """Abstract base class for interventions."""

    intervention_type: InterventionType

    @abstractmethod
    def apply(
        self,
        token_ids: list[int],
        span: SpanInfo,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InterventionResult:
        """Apply the intervention to the token sequence.

        Args:
            token_ids: Original token IDs.
            span: Span to intervene on.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments.

        Returns:
            InterventionResult with modified tokens.
        """
        pass


class MaskReplace(Intervention):
    """Replace tokens in span with a mask token (length-preserving)."""

    intervention_type = InterventionType.MASK

    def __init__(self, mask_token_id: int):
        """Initialize with the mask token ID.

        Args:
            mask_token_id: Token ID to use for masking.
        """
        self.mask_token_id = mask_token_id

    def apply(
        self,
        token_ids: list[int],
        span: SpanInfo,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InterventionResult:
        """Replace span tokens with mask token."""
        token_ids = list(token_ids)  # Copy
        original_tokens = token_ids[span.start : span.end]

        # Replace with mask tokens (same length)
        new_tokens = [self.mask_token_id] * len(original_tokens)
        token_ids[span.start : span.end] = new_tokens

        return InterventionResult(
            token_ids=token_ids,
            intervention_type=self.intervention_type,
            span_info=span,
            length_changed=False,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
        )


class ShuffleSpan(Intervention):
    """Shuffle tokens within the span (length-preserving)."""

    intervention_type = InterventionType.SHUFFLE

    def apply(
        self,
        token_ids: list[int],
        span: SpanInfo,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InterventionResult:
        """Shuffle tokens within the span."""
        token_ids = list(token_ids)  # Copy
        original_tokens = token_ids[span.start : span.end]

        # Create deterministic shuffle
        rng = random.Random(seed)
        new_tokens = list(original_tokens)
        rng.shuffle(new_tokens)

        # Ensure we actually shuffled (if possible)
        if len(new_tokens) > 1 and new_tokens == original_tokens:
            # Force at least one swap
            new_tokens[0], new_tokens[-1] = new_tokens[-1], new_tokens[0]

        token_ids[span.start : span.end] = new_tokens

        return InterventionResult(
            token_ids=token_ids,
            intervention_type=self.intervention_type,
            span_info=span,
            length_changed=False,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
        )


class DeleteSpan(Intervention):
    """Delete the span entirely (length-changing)."""

    intervention_type = InterventionType.DELETE

    def apply(
        self,
        token_ids: list[int],
        span: SpanInfo,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InterventionResult:
        """Delete the span from the token sequence."""
        token_ids = list(token_ids)  # Copy
        original_tokens = token_ids[span.start : span.end]

        # Delete the span
        del token_ids[span.start : span.end]

        return InterventionResult(
            token_ids=token_ids,
            intervention_type=self.intervention_type,
            span_info=span,
            length_changed=True,
            original_tokens=original_tokens,
            new_tokens=[],
        )


def get_intervention(
    intervention_type: str | InterventionType,
    mask_token_id: int | None = None,
) -> Intervention:
    """Get an intervention instance by type.

    Args:
        intervention_type: Type of intervention.
        mask_token_id: Token ID for masking (required for mask intervention).

    Returns:
        Intervention instance.
    """
    if isinstance(intervention_type, str):
        intervention_type = InterventionType(intervention_type)

    if intervention_type == InterventionType.MASK:
        if mask_token_id is None:
            raise ValueError("mask_token_id required for mask intervention")
        return MaskReplace(mask_token_id)
    elif intervention_type == InterventionType.SHUFFLE:
        return ShuffleSpan()
    elif intervention_type == InterventionType.DELETE:
        return DeleteSpan()
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


def apply_intervention(
    token_ids: list[int],
    span: SpanInfo,
    intervention_type: str | InterventionType,
    mask_token_id: int | None = None,
    seed: int | None = None,
) -> InterventionResult:
    """Apply an intervention to a token sequence.

    Args:
        token_ids: Original token IDs.
        span: Span to intervene on.
        intervention_type: Type of intervention.
        mask_token_id: Token ID for masking (required for mask intervention).
        seed: Random seed for reproducibility.

    Returns:
        InterventionResult with modified tokens.
    """
    intervention = get_intervention(intervention_type, mask_token_id)
    return intervention.apply(token_ids, span, seed=seed)
