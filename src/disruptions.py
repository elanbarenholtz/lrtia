"""Disruption functions for Experiment 1A.

Convention: context_tokens is stored in temporal order, oldest to newest:
    [t_C, t_{C-1}, ..., t_2, t_1]
where t_1 is immediately before the target.

When revealing context length c, the prefix is:
    revealed = context_tokens[-c:]

For disrupted conditions, the function returns a new temporal-order list
with the same convention (oldest to newest).

Given C = 100 and M = 50:
    far half  = positions M+1 to C (older tokens) = context_tokens[:C-M]
    near half = positions 1 to M (newer tokens)    = context_tokens[C-M:]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TargetExample:
    """A single target region with its context."""
    corpus_id: str
    document_id: str
    target_id: str
    language: str
    probe_tokenizer_name: str
    context_tokens: list[int]     # temporal order: oldest to newest
    target_tokens: list[int]
    target_position_fraction: float
    metadata: dict | None = None


def split_context(context_tokens: list[int], M: int) -> tuple[list[int], list[int]]:
    """Split context into far and near halves.

    Args:
        context_tokens: [t_C, ..., t_{M+1}, t_M, ..., t_1] oldest to newest
        M: cut point (number of near-half tokens)

    Returns:
        (far, near) where:
            far  = context_tokens[:C-M]  = [t_C, ..., t_{M+1}]
            near = context_tokens[C-M:]  = [t_M, ..., t_1]
    """
    C = len(context_tokens)
    far = context_tokens[:C - M]
    near = context_tokens[C - M:]
    return far, near


# === D0: No-op control ===

def d0_noop(context_tokens: list[int], M: int = 50) -> list[int]:
    """Cut at M and rejoin unchanged. Sanity check on the code path.

    Returns: far + near (identical to input).
    """
    far, near = split_context(context_tokens, M)
    return far + near


# === D1: Reverse far half ===

def d1_reverse_far(context_tokens: list[int], M: int = 50) -> list[int]:
    """Reverse the far half; keep near half intact.

    Desired temporal order:
        [t_{M+1}, t_{M+2}, ..., t_C, t_M, ..., t_1]

    Implementation: reversed(far) + near
    """
    far, near = split_context(context_tokens, M)
    return list(reversed(far)) + near


# === D2: Reverse near half ===

def d2_reverse_near(context_tokens: list[int], M: int = 50) -> list[int]:
    """Reverse the near half; keep far half intact.

    Desired temporal order:
        [t_C, ..., t_{M+1}, t_1, t_2, ..., t_M]

    Implementation: far + reversed(near)
    """
    far, near = split_context(context_tokens, M)
    return far + list(reversed(near))


# === D3: Swap halves ===

def d3_swap_halves(context_tokens: list[int], M: int = 50) -> list[int]:
    """Swap halves: near goes to far positions, far goes to near positions.

    Desired temporal order:
        [t_M, ..., t_1, t_C, ..., t_{M+1}]

    Implementation: near + far
    """
    far, near = split_context(context_tokens, M)
    return near + far


# === D4: Full reverse ===

def d4_full_reverse(context_tokens: list[int]) -> list[int]:
    """Reverse the entire context.

    Desired temporal order:
        [t_1, t_2, ..., t_C]

    Implementation: reversed(context_tokens)
    """
    return list(reversed(context_tokens))


# === D5: Foreign block insertion ===

def d5_insert_block(
    context_tokens: list[int],
    insert_tokens: list[int],
    M: int = 50,
) -> list[int]:
    """Insert a foreign block between near and far halves.

    Returns context of length C + K (not C).

    Desired temporal order:
        [t_C, ..., t_{M+1}, s_1, ..., s_K, t_M, ..., t_1]

    Implementation: far + insert_tokens + near

    For D5 conditions, formal analysis runs c = 0 through C + K.
    """
    far, near = split_context(context_tokens, M)
    return far + list(insert_tokens) + near


# === Condition registry ===

CONDITION_FUNCTIONS = {
    'intact': lambda ctx, **kw: list(ctx),
    'D0_noop': d0_noop,
    'D1_reverse_far': d1_reverse_far,
    'D2_reverse_near': d2_reverse_near,
    'D3_swap_halves': d3_swap_halves,
    'D4_full_reverse': lambda ctx, **kw: d4_full_reverse(ctx),
    # D5 variants handled separately (need insert_tokens)
}


def get_revealed_context(condition_context: list[int], c: int) -> list[int]:
    """Get the c-token prefix revealed under a condition.

    Args:
        condition_context: disrupted context in temporal order (oldest to newest)
        c: number of tokens to reveal (0 = empty)

    Returns:
        The last c tokens of condition_context, or empty list if c=0.
    """
    if c <= 0:
        return []
    return condition_context[-c:]
