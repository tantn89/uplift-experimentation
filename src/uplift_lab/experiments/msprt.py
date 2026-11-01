"""Mixture sequential probability ratio test (mSPRT) for a difference in means.

Johari, Koomen, Pekelis and Walsh, "Always Valid Inference" (Operations Research, 2022).
After ``n`` units per arm the estimate ``d`` is approximately ``N(delta, V)`` with
``V = s_t^2 / n_t + s_c^2 / n_c``. Mixing the likelihood ratio for ``H0: delta = 0`` over a prior
``delta ~ N(0, tau^2)`` gives the closed form

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * d^2 / (2 V (V + tau^2)))

``Lambda`` is a non-negative martingale under H0, so by Ville's inequality the rule "stop and
reject the first time Lambda >= 1 / alpha" has type-I error at most ``alpha`` however often the
data are inspected. The always-valid p-value is the running minimum of ``1 / Lambda``.

Mixing variance ``tau^2``: validity holds for any ``tau > 0``; the choice only moves power. The
mixture concentrates power near ``|delta| ~ tau``, so ``tau`` is set before the experiment to the
effect it is designed to detect (the MDE). Too small a ``tau`` makes early stopping slow; too
large spends evidence on implausibly big effects. ``V`` uses plug-in sample variances, the usual
large-sample approximation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uplift_lab.experiments.estimators import FloatArray


def _check_tau(tau: float) -> float:
    if not tau > 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    return tau * tau


def msprt_log_lr(estimate: ArrayLike, variance: ArrayLike, tau: float) -> FloatArray:
    """Log mixture likelihood ratio, elementwise; NaN where the variance is not positive."""
    tau2 = _check_tau(tau)
    d = np.asarray(estimate, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lr = 0.5 * np.log(v / (v + tau2)) + tau2 * d**2 / (2.0 * v * (v + tau2))
    return np.where(v > 0.0, log_lr, np.nan)


def msprt_reject(
    estimate: ArrayLike, variance: ArrayLike, tau: float, *, alpha: float = 0.05
) -> NDArray[np.bool_]:
    """True where ``Lambda >= 1 / alpha`` (never where the variance is not positive)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    log_lr = msprt_log_lr(estimate, variance, tau)
    return np.asarray(np.nan_to_num(log_lr, nan=-np.inf) >= np.log(1.0 / alpha))


def always_valid_p_values(log_lr_path: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Running minimum of ``min(1, 1 / Lambda)`` along ``axis`` (the look index)."""
    log_lr = np.nan_to_num(np.asarray(log_lr_path, dtype=np.float64), nan=-np.inf)
    p_values = np.minimum(1.0, np.exp(-log_lr))
    return np.minimum.accumulate(p_values, axis=axis)


def always_valid_ci_halfwidth(
    variance: ArrayLike, tau: float, *, alpha: float = 0.05
) -> FloatArray:
    """Half-width of the confidence sequence ``d +/- h``, valid simultaneously over all looks."""
    tau2 = _check_tau(tau)
    v = np.asarray(variance, dtype=np.float64)
    if (v <= 0.0).any():
        raise ValueError("variance must be positive")
    squared = v * (v + tau2) / tau2 * (np.log((v + tau2) / v) + 2.0 * np.log(1.0 / alpha))
    return np.asarray(np.sqrt(squared), dtype=np.float64)
