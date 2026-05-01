"""D5a–d insertion disruptions.

These disruptions insert a foreign block of K tokens between the near and
far halves of the context. The resulting context length is C + K.

Unlike D0–D4, D5 variants do NOT preserve the original token multiset.
They are insertion controls, not permutation controls, and are evaluated
separately in the decision rules.
"""

from __future__ import annotations


def d5_insert_block(
    ctx: list[int], block: list[int], M: int = 50
) -> list[int]:
    """Insert a foreign block of K tokens between near and far halves.

    Args:
        ctx: Original context tokens [most_distant, ..., closest_to_target].
        block: K foreign tokens to insert.
        M: Cut point (number of near-half tokens).

    Returns:
        Disrupted context of length len(ctx) + len(block):
        [far_half] + [block] + [near_half]
    """
    near = ctx[-M:]
    far = ctx[:-M]
    return far + list(block) + near
