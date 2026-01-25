"""Tests for metrics computation module."""

import numpy as np
import pytest

from lrtia.metrics.compute import (
    compute_delta_nll,
    compute_kl_divergence,
    compute_topk_stability,
    compute_rank_change,
    compute_all_metrics,
)


class TestDeltaNLL:
    def test_basic(self):
        original = np.array([-1.0, -2.0, -3.0])
        intervened = np.array([-2.0, -3.0, -4.0])

        delta = compute_delta_nll(original, intervened)

        # Original has higher log prob, so delta should be positive
        # (intervention made predictions worse)
        assert delta == pytest.approx(1.0)

    def test_per_token(self):
        original = np.array([-1.0, -2.0])
        intervened = np.array([-2.0, -4.0])

        delta = compute_delta_nll(original, intervened, aggregate=False)

        assert len(delta) == 2
        assert delta[0] == pytest.approx(1.0)
        assert delta[1] == pytest.approx(2.0)

    def test_no_change(self):
        original = np.array([-1.0, -2.0])
        intervened = np.array([-1.0, -2.0])

        delta = compute_delta_nll(original, intervened)
        assert delta == pytest.approx(0.0)


class TestKLDivergence:
    def test_identical_distributions(self):
        log_probs = np.log(np.array([[0.5, 0.3, 0.2]]))

        kl = compute_kl_divergence(log_probs, log_probs)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        original = np.log(np.array([[0.8, 0.1, 0.1]]))
        intervened = np.log(np.array([[0.4, 0.3, 0.3]]))

        kl = compute_kl_divergence(original, intervened)
        assert kl > 0  # KL should be positive for different distributions

    def test_top_k_approximation(self):
        # Create distributions
        original = np.random.dirichlet(np.ones(100), size=1)
        original_log = np.log(original)
        intervened_log = np.log(np.random.dirichlet(np.ones(100), size=1))

        kl_full = compute_kl_divergence(original_log, intervened_log, top_k=None)
        kl_topk = compute_kl_divergence(original_log, intervened_log, top_k=10)

        # Both should be non-negative
        assert kl_full >= 0
        assert kl_topk >= 0


class TestTopKStability:
    def test_perfect_stability(self):
        log_probs = np.array([[0.0, -1.0, -2.0, -3.0, -4.0]])

        overlap = compute_topk_stability(log_probs, log_probs, k=3)
        assert overlap == pytest.approx(1.0)

    def test_no_overlap(self):
        original = np.array([[0.0, -1.0, -2.0, -10.0, -11.0]])
        intervened = np.array([[-10.0, -11.0, -12.0, 0.0, -1.0]])

        # Top 2 of original: indices 0, 1
        # Top 2 of intervened: indices 3, 4
        overlap = compute_topk_stability(original, intervened, k=2)
        assert overlap == pytest.approx(0.0)

    def test_partial_overlap(self):
        original = np.array([[0.0, -1.0, -2.0, -3.0]])
        intervened = np.array([[0.0, -2.0, -1.0, -3.0]])

        # Top 2 of original: indices 0, 1
        # Top 2 of intervened: indices 0, 2
        # Overlap: {0}
        overlap = compute_topk_stability(original, intervened, k=2)
        assert overlap == pytest.approx(0.5)


class TestRankChange:
    def test_no_change(self):
        log_probs = np.array([[0.0, -1.0, -2.0, -3.0]])
        target_tokens = np.array([0])  # Top token

        change = compute_rank_change(log_probs, log_probs, target_tokens)
        assert change == pytest.approx(0.0)

    def test_rank_worsens(self):
        original = np.array([[0.0, -1.0, -2.0, -3.0]])  # Token 0 is rank 0
        intervened = np.array([[-3.0, -2.0, -1.0, 0.0]])  # Token 0 is rank 3
        target_tokens = np.array([0])

        change = compute_rank_change(original, intervened, target_tokens)
        assert change == pytest.approx(3.0)  # Rank went from 0 to 3

    def test_rank_improves(self):
        original = np.array([[-3.0, -2.0, -1.0, 0.0]])  # Token 0 is rank 3
        intervened = np.array([[0.0, -1.0, -2.0, -3.0]])  # Token 0 is rank 0
        target_tokens = np.array([0])

        change = compute_rank_change(original, intervened, target_tokens)
        assert change == pytest.approx(-3.0)  # Rank improved by 3


class TestComputeAllMetrics:
    def test_with_distributions(self):
        # Create simple distributions
        np.random.seed(42)
        n_positions = 5
        vocab_size = 100

        original = np.random.dirichlet(np.ones(vocab_size), size=n_positions)
        intervened = np.random.dirichlet(np.ones(vocab_size), size=n_positions)

        original_log = np.log(original + 1e-10)
        intervened_log = np.log(intervened + 1e-10)

        # Random target tokens
        target_tokens = np.random.randint(0, vocab_size, size=n_positions)

        result = compute_all_metrics(
            original_log,
            intervened_log,
            target_tokens,
            per_token=True,
        )

        # Check all metrics are computed
        assert result.delta_nll is not None
        assert result.kl_divergence is not None
        assert result.topk_overlap is not None
        assert result.rank_change is not None

        # Check per-token metrics
        assert result.delta_nll_per_token is not None
        assert len(result.delta_nll_per_token) == n_positions

    def test_to_dict(self):
        np.random.seed(42)
        original = np.log(np.random.dirichlet(np.ones(10), size=3) + 1e-10)
        intervened = np.log(np.random.dirichlet(np.ones(10), size=3) + 1e-10)
        targets = np.array([0, 1, 2])

        result = compute_all_metrics(original, intervened, targets)
        d = result.to_dict()

        assert "delta_nll" in d
        assert "kl_divergence" in d
        assert "topk_overlap" in d
        assert "rank_change" in d
