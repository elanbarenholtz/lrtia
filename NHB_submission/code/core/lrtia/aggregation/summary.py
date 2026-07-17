"""Summary metrics for memory curves."""

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

from lrtia.aggregation.curves import MemoryCurve


@dataclass
class SummaryMetrics:
    """Summary metrics for a memory curve."""

    auc: float  # Area under the curve
    auc_normalized: float  # AUC normalized by distance range
    half_life: float | None  # Distance to 50% of baseline effect
    tail_mass: float  # Fraction of effect beyond threshold distance
    peak_effect: float  # Maximum effect size
    peak_distance: int  # Distance at peak effect
    decay_rate: float | None  # Exponential decay rate (if fit succeeds)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auc": self.auc,
            "auc_normalized": self.auc_normalized,
            "half_life": self.half_life,
            "tail_mass": self.tail_mass,
            "peak_effect": self.peak_effect,
            "peak_distance": self.peak_distance,
            "decay_rate": self.decay_rate,
        }


def compute_auc(
    curve: MemoryCurve,
    distance_range: tuple[int, int] | None = None,
    normalize: bool = False,
) -> float:
    """Compute area under the memory curve.

    Args:
        curve: Memory curve to analyze.
        distance_range: Optional (min, max) distance range. Default: full range.
        normalize: If True, normalize by distance range.

    Returns:
        Area under the curve.
    """
    if not curve.points:
        return 0.0

    distances = np.array(curve.distances)
    means = np.array(curve.means)

    if distance_range is not None:
        mask = (distances >= distance_range[0]) & (distances <= distance_range[1])
        distances = distances[mask]
        means = means[mask]

    if len(distances) < 2:
        return 0.0

    # Use trapezoidal integration
    auc = float(integrate.trapezoid(means, distances))

    if normalize:
        range_width = distances[-1] - distances[0]
        if range_width > 0:
            auc /= range_width

    return auc


def compute_half_life(
    curve: MemoryCurve,
    baseline_fraction: float = 0.5,
) -> float | None:
    """Compute the half-life (distance to reach fraction of baseline effect).

    Args:
        curve: Memory curve to analyze.
        baseline_fraction: Fraction of baseline effect to find (default: 0.5 for half-life).

    Returns:
        Distance at which effect drops to baseline_fraction, or None if not found.
    """
    if not curve.points or len(curve.points) < 2:
        return None

    distances = np.array(curve.distances)
    means = np.array(curve.means)

    # Get baseline effect (effect at shortest distance)
    baseline = means[0]
    if baseline <= 0:
        return None

    # Find target effect level
    target = baseline * baseline_fraction

    # Find where curve crosses target
    for i in range(len(means) - 1):
        if means[i] >= target >= means[i + 1]:
            # Linear interpolation
            if means[i] == means[i + 1]:
                return float(distances[i])
            t = (target - means[i]) / (means[i + 1] - means[i])
            half_life = distances[i] + t * (distances[i + 1] - distances[i])
            return float(half_life)

    # If effect never drops below target, return max distance
    if means[-1] > target:
        return None

    return float(distances[-1])


def compute_tail_mass(
    curve: MemoryCurve,
    threshold_distance: int | None = None,
) -> float:
    """Compute fraction of total effect that occurs beyond threshold.

    Args:
        curve: Memory curve to analyze.
        threshold_distance: Distance threshold. Default: median of distances.

    Returns:
        Fraction of effect (AUC) beyond threshold.
    """
    if not curve.points or len(curve.points) < 2:
        return 0.0

    distances = np.array(curve.distances)
    means = np.array(curve.means)

    if threshold_distance is None:
        threshold_distance = int(np.median(distances))

    total_auc = compute_auc(curve)
    if total_auc <= 0:
        return 0.0

    # Compute AUC beyond threshold
    mask = distances >= threshold_distance
    if mask.sum() < 2:
        return 0.0

    tail_distances = distances[mask]
    tail_means = means[mask]
    tail_auc = float(integrate.trapezoid(tail_means, tail_distances))

    return tail_auc / total_auc


@dataclass
class DecayFit:
    """Results from fitting a decay model."""

    model: str  # "exponential", "power_law", "linear"
    amplitude: float | None  # A parameter
    decay_param: float | None  # tau (exp), alpha (power), slope (linear)
    r_squared: float | None  # Goodness of fit
    half_life: float | None  # Distance to 50% of initial effect
    predicted: np.ndarray | None  # Predicted values at original distances

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "amplitude": self.amplitude,
            "decay_param": self.decay_param,
            "r_squared": self.r_squared,
            "half_life": self.half_life,
        }


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def fit_exponential_decay(
    curve: MemoryCurve,
) -> tuple[float | None, float | None]:
    """Fit an exponential decay model to the curve.

    Model: effect(d) = A * exp(-d / tau)

    Args:
        curve: Memory curve to fit.

    Returns:
        Tuple of (amplitude, decay_rate) or (None, None) if fit fails.
    """
    if not curve.points or len(curve.points) < 3:
        return None, None

    distances = np.array(curve.distances, dtype=float)
    means = np.array(curve.means)

    # Filter out non-positive values
    mask = means > 0
    if mask.sum() < 3:
        return None, None

    distances = distances[mask]
    means = means[mask]

    try:

        def exp_model(d, a, tau):
            return a * np.exp(-d / tau)

        # Initial guess
        a0 = means[0]
        tau0 = distances[-1] / 2

        popt, _ = optimize.curve_fit(
            exp_model,
            distances,
            means,
            p0=[a0, tau0],
            bounds=([0, 1], [np.inf, np.inf]),
            maxfev=1000,
        )

        amplitude, tau = popt
        decay_rate = 1.0 / tau  # rate = 1/tau

        return float(amplitude), float(decay_rate)

    except (RuntimeError, ValueError):
        return None, None


def fit_decay_models(
    curve: MemoryCurve,
) -> dict[str, DecayFit]:
    """Fit multiple decay models and return results with goodness of fit.

    Models:
    - Exponential: y = A * exp(-d / tau), half-life = tau * ln(2)
    - Power law: y = A * d^(-alpha), half-life = d0 * 2^(1/alpha)
    - Linear: y = A - slope * d (simple linear decay)

    Args:
        curve: Memory curve to fit.

    Returns:
        Dictionary mapping model name to DecayFit results.
    """
    results = {}

    if not curve.points or len(curve.points) < 3:
        return results

    distances = np.array(curve.distances, dtype=float)
    means = np.array(curve.means)

    # Filter out non-positive values for log-based fits
    mask = means > 0
    if mask.sum() < 3:
        return results

    d_pos = distances[mask]
    y_pos = means[mask]

    # 1. Exponential decay: y = A * exp(-d / tau)
    try:
        def exp_model(d, a, tau):
            return a * np.exp(-d / tau)

        popt, _ = optimize.curve_fit(
            exp_model, d_pos, y_pos,
            p0=[y_pos[0], d_pos[-1] / 2],
            bounds=([0, 1], [np.inf, np.inf]),
            maxfev=2000,
        )
        a, tau = popt
        y_pred = exp_model(distances, a, tau)
        r2 = _compute_r_squared(means, y_pred)
        half_life = tau * np.log(2)  # Distance where effect = A/2

        results["exponential"] = DecayFit(
            model="exponential",
            amplitude=float(a),
            decay_param=float(tau),  # tau (time constant)
            r_squared=r2,
            half_life=float(half_life),
            predicted=y_pred,
        )
    except (RuntimeError, ValueError):
        pass

    # 2. Power law: y = A * d^(-alpha)
    try:
        # Avoid d=0 issues
        d_safe = np.maximum(d_pos, 1)

        def power_model(d, a, alpha):
            return a * np.power(d, -alpha)

        popt, _ = optimize.curve_fit(
            power_model, d_safe, y_pos,
            p0=[y_pos[0] * d_safe[0], 0.5],
            bounds=([0, 0.01], [np.inf, 5]),
            maxfev=2000,
        )
        a, alpha = popt
        d_pred_safe = np.maximum(distances, 1)
        y_pred = power_model(d_pred_safe, a, alpha)
        r2 = _compute_r_squared(means, y_pred)
        # Half-life for power law: find d where y = y(d0)/2
        # A * d^(-alpha) = A * d0^(-alpha) / 2
        # d = d0 * 2^(1/alpha)
        d0 = d_safe[0]
        half_life = d0 * np.power(2, 1 / alpha) if alpha > 0 else None

        results["power_law"] = DecayFit(
            model="power_law",
            amplitude=float(a),
            decay_param=float(alpha),  # alpha exponent
            r_squared=r2,
            half_life=float(half_life) if half_life else None,
            predicted=y_pred,
        )
    except (RuntimeError, ValueError):
        pass

    # 3. Linear decay: y = A - slope * d
    try:
        def linear_model(d, a, slope):
            return np.maximum(a - slope * d, 0)

        popt, _ = optimize.curve_fit(
            linear_model, distances, means,
            p0=[means[0], (means[0] - means[-1]) / (distances[-1] - distances[0])],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=2000,
        )
        a, slope = popt
        y_pred = linear_model(distances, a, slope)
        r2 = _compute_r_squared(means, y_pred)
        half_life = a / (2 * slope) if slope > 0 else None

        results["linear"] = DecayFit(
            model="linear",
            amplitude=float(a),
            decay_param=float(slope),  # slope
            r_squared=r2,
            half_life=float(half_life) if half_life else None,
            predicted=y_pred,
        )
    except (RuntimeError, ValueError):
        pass

    return results


def get_best_decay_fit(curve: MemoryCurve) -> DecayFit | None:
    """Get the best-fitting decay model based on R-squared.

    Args:
        curve: Memory curve to fit.

    Returns:
        Best DecayFit or None if no fits succeeded.
    """
    fits = fit_decay_models(curve)
    if not fits:
        return None

    # Return model with highest R-squared
    best = max(fits.values(), key=lambda f: f.r_squared or 0)
    return best


def compute_summary_metrics(
    curve: MemoryCurve,
    tail_threshold: int | None = None,
) -> SummaryMetrics:
    """Compute all summary metrics for a memory curve.

    Args:
        curve: Memory curve to analyze.
        tail_threshold: Distance threshold for tail mass. Default: median.

    Returns:
        SummaryMetrics object.
    """
    if not curve.points:
        return SummaryMetrics(
            auc=0.0,
            auc_normalized=0.0,
            half_life=None,
            tail_mass=0.0,
            peak_effect=0.0,
            peak_distance=0,
            decay_rate=None,
        )

    # AUC
    auc = compute_auc(curve)
    auc_normalized = compute_auc(curve, normalize=True)

    # Half-life
    half_life = compute_half_life(curve)

    # Tail mass
    tail_mass = compute_tail_mass(curve, tail_threshold)

    # Peak effect
    means = np.array(curve.means)
    peak_idx = np.argmax(means)
    peak_effect = float(means[peak_idx])
    peak_distance = curve.distances[peak_idx]

    # Decay rate
    _, decay_rate = fit_exponential_decay(curve)

    return SummaryMetrics(
        auc=auc,
        auc_normalized=auc_normalized,
        half_life=half_life,
        tail_mass=tail_mass,
        peak_effect=peak_effect,
        peak_distance=peak_distance,
        decay_rate=decay_rate,
    )
