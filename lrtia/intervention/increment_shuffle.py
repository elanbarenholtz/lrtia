"""Increment-shuffle transform for Experiment 2 (span-level order at distance).

This adds the *increment-shuffle* condition used by Exp 2 alongside the main
CPF pipeline. It is deliberately kept in its own module so that the headline
CPF code paths (``transforms.ShuffleSpan``, which permutes the *entire* revealed
prefix to give the marginal baseline for ``P(d)``) are not touched.

Distance convention (matches the CPF runner)
--------------------------------------------
For a target starting at ``target_start``, the revealed prefix of length ``c``
is ``full_ids[target_start - c : target_start]``. Within that prefix:

    prefix[-1]  is the token at distance 1  (immediately before the target)
    prefix[0]   is the token at distance c  (furthest revealed token)

So "distance ``k``" (``1 <= k <= c``) is ``prefix[c - k]``, and the band of
distances ``a .. b`` (inclusive, ``1 <= a <= b <= c``) is the slice
``prefix[c - b : c - a + 1]``.

Increment-shuffle for an adjacent ladder pair ``(c_prev, c_cur)``
----------------------------------------------------------------
Condition B of Exp 2: tokens at distances ``1 .. c_prev`` are left intact and
in place; tokens at distances ``c_prev + 1 .. c_cur`` are permuted uniformly at
random *within that band only*, in place. The target is never permuted and no
special/boundary tokens are introduced.

In prefix-array terms, with a prefix of length ``c_cur``:

    far band  (distances c_prev+1 .. c_cur) -> prefix[0 : c_cur - c_prev]   (permuted)
    near band (distances 1 .. c_prev)       -> prefix[c_cur - c_prev :]     (intact)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts.

    ``hash()`` is process-randomized for str/bytes in CPython, so it cannot be
    used for reproducible seeds across runs or machines. This hashes the
    repr-joined parts with BLAKE2b and returns a stable non-negative int, so
    ``seed = stable_seed(document_id, target_id, c_cur, k)`` reproduces exactly,
    as required by the preregistration.
    """
    key = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def increment_shuffle_prefix(
    prefix: list[int],
    c_prev: int,
    c_cur: int,
    seed: int,
) -> list[int]:
    """Return a copy of ``prefix`` with the far band permuted in place.

    Args:
        prefix: Token ids of the revealed context, length ``c_cur``. Index 0 is
            the furthest token (distance ``c_cur``); index -1 is distance 1.
        c_prev: Nearer ladder length; distances ``1 .. c_prev`` stay intact.
        c_cur: Current ladder length; equals ``len(prefix)``.
        seed: Deterministic seed for the permutation (see :func:`stable_seed`).

    Returns:
        A new list of length ``c_cur``: the last ``c_prev`` tokens are untouched,
        the first ``c_cur - c_prev`` tokens are a uniform permutation of the
        original far-band tokens.

    Notes:
        When the far band has fewer than 2 tokens (e.g. the ``(1, 2)`` pair,
        whose band is a single token), no shuffle is possible and the prefix is
        returned unchanged. This is inherent to the log-spaced ladder and is
        handled the same way downstream (Q for such a band is ~0 by construction).
    """
    if len(prefix) != c_cur:
        raise ValueError(f"len(prefix)={len(prefix)} != c_cur={c_cur}")
    if not (0 <= c_prev < c_cur):
        raise ValueError(f"need 0 <= c_prev < c_cur, got c_prev={c_prev}, c_cur={c_cur}")

    band_end = c_cur - c_prev  # exclusive; far band is prefix[0:band_end]
    out = list(prefix)
    band = out[:band_end]
    if len(band) > 1:
        rng = random.Random(seed)
        rng.shuffle(band)
        # Guarantee a real permutation when one exists (avoid identity draws).
        if band == prefix[:band_end]:
            band[0], band[-1] = band[-1], band[0]
        out[:band_end] = band
    return out


def near_and_far_shuffle_prefix(
    prefix: list[int],
    c_prev: int,
    c_cur: int,
    seed: int,
) -> list[int]:
    """Null control: shuffle the WHOLE prefix (near + far).

    With the near band also destroyed this collapses toward the ``P(d)``-style
    marginal baseline; used for the H2 null in the preregistration.
    """
    if len(prefix) != c_cur:
        raise ValueError(f"len(prefix)={len(prefix)} != c_cur={c_cur}")
    out = list(prefix)
    if len(out) > 1:
        rng = random.Random(seed)
        rng.shuffle(out)
        if out == list(prefix):
            out[0], out[-1] = out[-1], out[0]
    return out


def random_token_increment_prefix(
    prefix: list[int],
    c_prev: int,
    c_cur: int,
    seed: int,
    vocab_size: int,
) -> list[int]:
    """Random-token control: replace the far band with uniform random token ids.

    Isolates the contribution of *ordered content* in the far band from the
    contribution of its mere presence: near band intact, far band content
    replaced (not just reordered).
    """
    if len(prefix) != c_cur:
        raise ValueError(f"len(prefix)={len(prefix)} != c_cur={c_cur}")
    band_end = c_cur - c_prev
    out = list(prefix)
    rng = random.Random(seed)
    for i in range(band_end):
        out[i] = rng.randrange(vocab_size)
    return out


@dataclass
class LadderPair:
    """An adjacent pair on the context ladder and its Q-band geometry."""

    index: int          # interval index i in the ladder (pair is ladder[i-1], ladder[i])
    c_prev: int
    c_cur: int

    @property
    def width(self) -> int:
        """Number of tokens in the far band = denominator of Q."""
        return self.c_cur - self.c_prev

    @property
    def distance(self) -> float:
        """Geometric midpoint d_i = sqrt(c_prev * c_cur)."""
        return (self.c_prev * self.c_cur) ** 0.5


def ladder_pairs(context_lengths: list[int], skip_zero: bool = True) -> list[LadderPair]:
    """Adjacent pairs of the context ladder used for Q(d).

    By default the ``0 -> 1`` interval is skipped (its band is undefined /
    distance 0), matching ``analysis/longrange_slope_fit.py``.
    """
    pairs: list[LadderPair] = []
    for i in range(1, len(context_lengths)):
        c_prev, c_cur = context_lengths[i - 1], context_lengths[i]
        if skip_zero and c_prev == 0:
            continue
        pairs.append(LadderPair(index=i, c_prev=c_prev, c_cur=c_cur))
    return pairs


# --- 2x2 factorial (near-order x far-order) for the conditioning/redundancy test -----
#
# The increment-shuffle Q(d) = (ppl_B - ppl_A)/width measures far-band-internal
# order value *conditioned on intact near context*. To test whether Q decays
# faster than the marginal P(d) because near context explains away distant order
# (a conditioning artifact) rather than because a different kind of order carries
# the law, we run the full 2x2:
#
#     A: near intact,  far intact   (= the intact prefix)
#     B: near intact,  far shuffled (= increment_shuffle_prefix)
#     C: near shuffled, far intact   (= shuffle_near_band_prefix)
#     D: near shuffled, far shuffled (= both_bands_shuffle_prefix)
#
# far-internal order value | intact near   = (B - A) / width   [current Q]
# far-internal order value | shuffled near = (D - C) / width   [Q_cond]
# (D - C) uses the SAME far permutation as (B - A), so the contrast isolates the
# effect of the near context being ordered vs not.

def shuffle_near_band_prefix(
    prefix: list[int],
    c_prev: int,
    c_cur: int,
    seed: int,
) -> list[int]:
    """Permute the NEAR band (distances 1..c_prev) in place; far band untouched."""
    if len(prefix) != c_cur:
        raise ValueError(f"len(prefix)={len(prefix)} != c_cur={c_cur}")
    band_end = c_cur - c_prev  # near band is prefix[band_end:]
    out = list(prefix)
    near = out[band_end:]
    if len(near) > 1:
        rng = random.Random(seed)
        rng.shuffle(near)
        if near == prefix[band_end:]:
            near[0], near[-1] = near[-1], near[0]
        out[band_end:] = near
    return out


def both_bands_shuffle_prefix(
    prefix: list[int],
    c_prev: int,
    c_cur: int,
    seed_far: int,
    seed_near: int,
) -> list[int]:
    """Permute the far band and the near band independently, in place.

    ``seed_far`` should match the seed used for the far-only (B) condition and
    ``seed_near`` the near-only (C) condition, so the 2x2 reuses identical
    permutations and the (D - C) vs (B - A) contrast is clean.
    """
    if len(prefix) != c_cur:
        raise ValueError(f"len(prefix)={len(prefix)} != c_cur={c_cur}")
    band_end = c_cur - c_prev
    out = list(prefix)
    far = out[:band_end]
    near = out[band_end:]
    if len(far) > 1:
        rf = random.Random(seed_far)
        rf.shuffle(far)
        if far == prefix[:band_end]:
            far[0], far[-1] = far[-1], far[0]
        out[:band_end] = far
    if len(near) > 1:
        rn = random.Random(seed_near)
        rn.shuffle(near)
        if near == prefix[band_end:]:
            near[0], near[-1] = near[-1], near[0]
        out[band_end:] = near
    return out
