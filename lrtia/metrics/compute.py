"""Metrics computation for intervention effects."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsResult:
    """Results from computing metrics for a single intervention."""

    delta_nll: float  # Change in negative log likelihood
    delta_nll_per_token: np.ndarray | None  # Per-token delta NLL
    kl_divergence: float  # KL divergence between distributions
    kl_divergence_per_token: np.ndarray | None  # Per-token KL
    topk_overlap: float  # Fraction of top-k tokens that overlap
    topk_overlap_per_token: np.ndarray | None  # Per-token top-k overlap
    rank_change: float  # Average rank change of ground truth token
    rank_change_per_token: np.ndarray | None  # Per-token rank change

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "delta_nll": self.delta_nll,
            "kl_divergence": self.kl_divergence,
            "topk_overlap": self.topk_overlap,
            "rank_change": self.rank_change,
        }


def compute_delta_nll(
    original_log_probs: np.ndarray,
    intervened_log_probs: np.ndarray,
    aggregate: bool = True,
) -> float | np.ndarray:
    """Compute change in negative log likelihood.

    ΔNLL = NLL_intervened - NLL_original
         = -log_prob_intervened - (-log_prob_original)
         = log_prob_original - log_prob_intervened

    Positive values indicate the intervention increased perplexity (made prediction harder).

    Args:
        original_log_probs: Log probabilities from original context.
        intervened_log_probs: Log probabilities from intervened context.
        aggregate: If True, return mean across positions.

    Returns:
        Delta NLL (positive = worse predictions after intervention).
    """
    # Ensure arrays
    original_log_probs = np.asarray(original_log_probs)
    intervened_log_probs = np.asarray(intervened_log_probs)

    # ΔNLL = -log(p_intervened) - (-log(p_original))
    #      = log(p_original) - log(p_intervened)
    delta_nll = original_log_probs - intervened_log_probs

    if aggregate:
        return float(np.mean(delta_nll))
    return delta_nll


def compute_kl_divergence(
    original_log_probs: np.ndarray,
    intervened_log_probs: np.ndarray,
    top_k: int | None = None,
    aggregate: bool = True,
) -> float | np.ndarray:
    """Compute KL divergence between original and intervened distributions.

    KL(P_original || P_intervened) = sum_x P_original(x) * log(P_original(x) / P_intervened(x))

    Args:
        original_log_probs: Log probabilities from original context [positions, vocab].
        intervened_log_probs: Log probabilities from intervened context [positions, vocab].
        top_k: If provided, approximate KL using only top-k tokens from original.
        aggregate: If True, return mean across positions.

    Returns:
        KL divergence (always non-negative, 0 means identical distributions).
    """
    original_log_probs = np.asarray(original_log_probs)
    intervened_log_probs = np.asarray(intervened_log_probs)

    # Handle 1D case (single position)
    if original_log_probs.ndim == 1:
        original_log_probs = original_log_probs[np.newaxis, :]
        intervened_log_probs = intervened_log_probs[np.newaxis, :]

    num_positions = original_log_probs.shape[0]
    kl_values = np.zeros(num_positions)

    for i in range(num_positions):
        orig_lp = original_log_probs[i]
        int_lp = intervened_log_probs[i]

        if top_k is not None:
            # Approximate KL using top-k tokens from original distribution
            top_indices = np.argsort(orig_lp)[-top_k:]
            orig_probs = np.exp(orig_lp[top_indices])
            # Renormalize
            orig_probs = orig_probs / orig_probs.sum()
            orig_lp_subset = np.log(orig_probs + 1e-10)
            int_lp_subset = int_lp[top_indices]
        else:
            orig_lp_subset = orig_lp
            int_lp_subset = int_lp

        # Convert to probabilities
        orig_probs = np.exp(orig_lp_subset)
        # Renormalize to ensure it sums to 1
        orig_probs = orig_probs / (orig_probs.sum() + 1e-10)

        # KL = sum_x p(x) * (log p(x) - log q(x))
        log_ratio = orig_lp_subset - int_lp_subset
        kl = np.sum(orig_probs * log_ratio)

        # KL should be non-negative; numerical issues can make it slightly negative
        kl_values[i] = max(0.0, kl)

    if aggregate:
        return float(np.mean(kl_values))
    return kl_values


def compute_topk_stability(
    original_log_probs: np.ndarray,
    intervened_log_probs: np.ndarray,
    k: int = 10,
    aggregate: bool = True,
) -> float | np.ndarray:
    """Compute top-k stability (fraction of top-k tokens that overlap).

    Args:
        original_log_probs: Log probabilities from original context [positions, vocab].
        intervened_log_probs: Log probabilities from intervened context [positions, vocab].
        k: Number of top tokens to compare.
        aggregate: If True, return mean across positions.

    Returns:
        Fraction of top-k tokens that overlap (0 to 1, higher = more stable).
    """
    original_log_probs = np.asarray(original_log_probs)
    intervened_log_probs = np.asarray(intervened_log_probs)

    # Handle 1D case
    if original_log_probs.ndim == 1:
        original_log_probs = original_log_probs[np.newaxis, :]
        intervened_log_probs = intervened_log_probs[np.newaxis, :]

    num_positions = original_log_probs.shape[0]
    overlap_values = np.zeros(num_positions)

    for i in range(num_positions):
        # Get top-k token indices
        orig_topk = set(np.argsort(original_log_probs[i])[-k:])
        int_topk = set(np.argsort(intervened_log_probs[i])[-k:])

        # Compute overlap
        overlap = len(orig_topk & int_topk) / k
        overlap_values[i] = overlap

    if aggregate:
        return float(np.mean(overlap_values))
    return overlap_values


def compute_rank_change(
    original_log_probs: np.ndarray,
    intervened_log_probs: np.ndarray,
    target_tokens: np.ndarray | list[int],
    aggregate: bool = True,
) -> float | np.ndarray:
    """Compute rank change of ground truth tokens.

    Args:
        original_log_probs: Log probabilities from original context [positions, vocab].
        intervened_log_probs: Log probabilities from intervened context [positions, vocab].
        target_tokens: Actual next tokens at each position.
        aggregate: If True, return mean across positions.

    Returns:
        Rank change (positive = rank got worse, negative = rank improved).
    """
    original_log_probs = np.asarray(original_log_probs)
    intervened_log_probs = np.asarray(intervened_log_probs)
    target_tokens = np.asarray(target_tokens)

    # Handle 1D case
    if original_log_probs.ndim == 1:
        original_log_probs = original_log_probs[np.newaxis, :]
        intervened_log_probs = intervened_log_probs[np.newaxis, :]

    num_positions = original_log_probs.shape[0]
    rank_changes = np.zeros(num_positions)

    for i in range(num_positions):
        token = target_tokens[i]

        # Compute ranks (0 = highest probability)
        orig_ranks = np.argsort(np.argsort(-original_log_probs[i]))
        int_ranks = np.argsort(np.argsort(-intervened_log_probs[i]))

        orig_rank = orig_ranks[token]
        int_rank = int_ranks[token]

        # Positive change = rank got worse (higher number)
        rank_changes[i] = int_rank - orig_rank

    if aggregate:
        return float(np.mean(rank_changes))
    return rank_changes


def compute_all_metrics(
    original_log_probs: np.ndarray,
    intervened_log_probs: np.ndarray,
    target_tokens: np.ndarray | list[int],
    top_k_for_kl: int = 1000,
    top_k_for_stability: int = 10,
    per_token: bool = False,
) -> MetricsResult:
    """Compute all metrics for an intervention.

    Args:
        original_log_probs: Log probabilities from original context.
            Shape [positions] for token log probs, or [positions, vocab] for distributions.
        intervened_log_probs: Log probabilities from intervened context.
        target_tokens: Actual next tokens at each position.
        top_k_for_kl: Number of top tokens for KL approximation.
        top_k_for_stability: K for top-k stability metric.
        per_token: If True, also return per-token metrics.

    Returns:
        MetricsResult with all computed metrics.
    """
    original_log_probs = np.asarray(original_log_probs)
    intervened_log_probs = np.asarray(intervened_log_probs)

    # Check if we have full distributions or just token log probs
    has_distributions = original_log_probs.ndim == 2

    # Compute delta NLL
    if has_distributions:
        # Extract token log probs from distributions
        target_tokens = np.asarray(target_tokens)
        positions = np.arange(len(target_tokens))
        orig_token_lp = original_log_probs[positions, target_tokens]
        int_token_lp = intervened_log_probs[positions, target_tokens]
    else:
        orig_token_lp = original_log_probs
        int_token_lp = intervened_log_probs

    delta_nll = compute_delta_nll(orig_token_lp, int_token_lp, aggregate=True)
    delta_nll_per_token = (
        compute_delta_nll(orig_token_lp, int_token_lp, aggregate=False) if per_token else None
    )

    # Compute distribution-based metrics if we have full distributions
    if has_distributions:
        kl = compute_kl_divergence(
            original_log_probs, intervened_log_probs, top_k=top_k_for_kl, aggregate=True
        )
        kl_per_token = (
            compute_kl_divergence(
                original_log_probs, intervened_log_probs, top_k=top_k_for_kl, aggregate=False
            )
            if per_token
            else None
        )

        topk = compute_topk_stability(
            original_log_probs, intervened_log_probs, k=top_k_for_stability, aggregate=True
        )
        topk_per_token = (
            compute_topk_stability(
                original_log_probs, intervened_log_probs, k=top_k_for_stability, aggregate=False
            )
            if per_token
            else None
        )

        rank = compute_rank_change(
            original_log_probs, intervened_log_probs, target_tokens, aggregate=True
        )
        rank_per_token = (
            compute_rank_change(
                original_log_probs, intervened_log_probs, target_tokens, aggregate=False
            )
            if per_token
            else None
        )
    else:
        # Cannot compute distribution-based metrics without full distributions
        kl = 0.0
        kl_per_token = None
        topk = 1.0  # Assume stable if we can't measure
        topk_per_token = None
        rank = 0.0
        rank_per_token = None

    return MetricsResult(
        delta_nll=delta_nll,
        delta_nll_per_token=delta_nll_per_token,
        kl_divergence=kl,
        kl_divergence_per_token=kl_per_token,
        topk_overlap=topk,
        topk_overlap_per_token=topk_per_token,
        rank_change=rank,
        rank_change_per_token=rank_per_token,
    )
