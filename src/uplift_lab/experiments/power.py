"""Power, sample size and minimum detectable effect for a two-sided two-sample test.

Normal approximation with known per-unit standard deviation ``sd`` (for a rate ``p`` use
``sqrt(p * (1 - p))``). The control arm has ``n`` units and the treated arm ``ratio * n``.

``variance_factor`` scales the outcome variance: CUPED with covariates explaining a share ``R^2``
of it leaves ``1 - R^2``, so the required sample size shrinks by the same factor.
"""

from __future__ import annotations

import math

from scipy import stats


def _check(alpha: float, sd: float, ratio: float, variance_factor: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if sd <= 0.0 or ratio <= 0.0:
        raise ValueError("sd and ratio must be positive")
    if not 0.0 < variance_factor <= 1.0:
        raise ValueError(f"variance_factor must be in (0, 1], got {variance_factor}")


def _se_unit(sd: float, ratio: float, variance_factor: float) -> float:
    """Standard error of the difference in means times sqrt(n_control)."""
    return sd * math.sqrt(variance_factor * (1.0 + 1.0 / ratio))


def power_at(
    n_control: float,
    mde: float,
    sd: float,
    *,
    alpha: float = 0.05,
    ratio: float = 1.0,
    variance_factor: float = 1.0,
) -> float:
    """Probability that a two-sided level-``alpha`` test rejects when the true effect is ``mde``."""
    _check(alpha, sd, ratio, variance_factor)
    if n_control <= 0:
        raise ValueError("n_control must be positive")
    shift = abs(mde) * math.sqrt(n_control) / _se_unit(sd, ratio, variance_factor)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift))


def required_n(
    mde: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_factor: float = 1.0,
) -> int:
    """Smallest control-arm size reaching ``power`` (closed form, far tail ignored)."""
    _check(alpha, sd, ratio, variance_factor)
    if mde == 0.0 or not 0.0 < power < 1.0:
        raise ValueError("mde must be non-zero and power in (0, 1)")
    z_sum = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    return math.ceil((z_sum * _se_unit(sd, ratio, variance_factor) / mde) ** 2)


def minimum_detectable_effect(
    n_control: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_factor: float = 1.0,
) -> float:
    """Smallest absolute effect detected with probability ``power`` at this sample size."""
    _check(alpha, sd, ratio, variance_factor)
    if n_control <= 0 or not 0.0 < power < 1.0:
        raise ValueError("n_control must be positive and power in (0, 1)")
    z_sum = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    return z_sum * _se_unit(sd, ratio, variance_factor) / math.sqrt(n_control)
