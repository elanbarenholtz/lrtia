"""D0–D4 permutation disruptions.

Each function takes a context token sequence (ordered from most distant to
closest to target: ctx[0] = most distant, ctx[-1] = closest) and returns
the disrupted sequence in the same convention.

The cut point M divides context into:
  - near half: ctx[-M:]  (positions 1..M, closest to target)
  - far half:  ctx[:-M]  (positions M+1..C, most distant)
"""

from __future__ import annotations


def d0_no_op(ctx: list[int], M: int = 50) -> list[int]:
    """Cut at M and rejoin unchanged. Sanity check on the code path.

    The disrupted context is byte-identical to the input, but has been
    routed through the same slicing logic as all other disruptions.
    """
    near = ctx[-M:]
    far = ctx[:-M]
    return far + near


def d1_reverse_far(ctx: list[int], M: int = 50) -> list[int]:
    """Reverse the far half (positions M+1..C); keep near half intact.

    Disrupted ordering (temporal sequence preceding target):
    [t_{M+1}, t_{M+2}, ..., t_C, t_M, ..., t_1, TARGET]

    When the pipeline grows context past length M, the tokens added beyond
    the cut are the original far-half tokens but in reversed temporal order.
    """
    near = ctx[-M:]
    far = ctx[:-M]
    return list(reversed(far)) + near


def d2_reverse_near(ctx: list[int], M: int = 50) -> list[int]:
    """Reverse the near half (positions 1..M); keep far half intact.

    Disrupted ordering:
    [t_C, t_{C-1}, ..., t_{M+1}, t_1, t_2, ..., t_M, TARGET]

    The originally-immediate-predecessor token now sits at distance M.
    Caveat: creates highly unnatural local syntax immediately before target.
    """
    near = ctx[-M:]
    far = ctx[:-M]
    return far + list(reversed(near))


def d3_swap_halves(ctx: list[int], M: int = 50) -> list[int]:
    """Swap halves: near goes far, far comes close.

    Disrupted ordering:
    [t_M, t_{M-1}, ..., t_1, t_C, t_{C-1}, ..., t_{M+1}, TARGET]

    The originally-closest token (strong predictor of target) now sits at
    distance M+1 rather than distance 1, generating a large upward jump
    in the marginal at d = M+1.
    """
    near = ctx[-M:]
    far = ctx[:-M]
    return near + far


def d4_full_reverse(ctx: list[int]) -> list[int]:
    """Reverse the entire context.

    Disrupted ordering:
    [t_1, t_2, ..., t_C, TARGET]

    The originally-closest token is now most distant, and vice versa.
    The simplest and most decisive test of forward-direction sensitivity.
    """
    return list(reversed(ctx))
