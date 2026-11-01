"""Difference-in-means effect estimates with Welch (unequal-variance) inference.

The estimand is the average treatment effect of assignment (intent to treat): the treated mean
minus the control mean. Inference uses Welch's t with Welch-Satterthwaite degrees of freedom, so
it stays valid when the arms have different variances (spend is far noisier when e-mailed).
The relative lift ``mean_t / mean_c - 1`` gets a delta-method interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EffectEstimate:
    """A two-arm effect estimate. ``ci_*`` bound the absolute effect at level ``1 - alpha``."""

    effect: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    df: float
    alpha: float
    n_treated: int
    n_control: int
    mean_treated: float
    mean_control: float
    relative_lift: float
    relative_lift_ci_low: float
    relative_lift_ci_high: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def as_float_array(values: ArrayLike, name: str) -> FloatArray:
    """Validate a 1-D finite numeric sample with at least two observations."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} needs at least two observations, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def welch_difference(
    treated: ArrayLike, control: ArrayLike, *, alpha: float = 0.05
) -> EffectEstimate:
    """Estimate ``mean(treated) - mean(control)`` with a Welch t interval and p-value."""
    _check_alpha(alpha)
    y_t = as_float_array(treated, "treated")
    y_c = as_float_array(control, "control")
    n_t, n_c = y_t.size, y_c.size
    mean_t, mean_c = float(y_t.mean()), float(y_c.mean())
    var_t = float(y_t.var(ddof=1)) / n_t
    var_c = float(y_c.var(ddof=1)) / n_c
    se = float(np.sqrt(var_t + var_c))
    if se == 0.0:
        raise ValueError("both samples are constant; the effect has zero standard error")
    df = (var_t + var_c) ** 2 / (var_t**2 / (n_t - 1) + var_c**2 / (n_c - 1))
    effect = mean_t - mean_c
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    p_value = float(2.0 * stats.t.sf(abs(effect) / se, df))

    lift, lift_low, lift_high = _relative_lift(mean_t, mean_c, var_t, var_c, alpha)
    return EffectEstimate(
        effect=effect,
        se=se,
        ci_low=effect - t_crit * se,
        ci_high=effect + t_crit * se,
        p_value=p_value,
        df=float(df),
        alpha=alpha,
        n_treated=n_t,
        n_control=n_c,
        mean_treated=mean_t,
        mean_control=mean_c,
        relative_lift=lift,
        relative_lift_ci_low=lift_low,
        relative_lift_ci_high=lift_high,
    )


def _relative_lift(
    mean_t: float, mean_c: float, var_mean_t: float, var_mean_c: float, alpha: float
) -> tuple[float, float, float]:
    """Delta-method interval for ``mean_t / mean_c - 1`` (independent arms); NaN if mean_c = 0."""
    if mean_c == 0.0:
        return float("nan"), float("nan"), float("nan")
    ratio = mean_t / mean_c
    se = float(np.sqrt(var_mean_t / mean_c**2 + ratio**2 * var_mean_c / mean_c**2))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return ratio - 1.0, ratio - 1.0 - z * se, ratio - 1.0 + z * se
