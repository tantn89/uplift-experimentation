"""CUPED (Deng, Xu, Kohavi and Walker, WSDM 2013) and the equivalent regression adjustment.

CUPED replaces the outcome ``y`` with ``y - (X - mean(X)) @ theta`` where ``X`` holds covariates
measured before treatment and ``theta`` is the OLS coefficient of ``y`` on ``X`` in the pooled
sample. Because ``X`` is independent of assignment, the adjusted difference in means is still
unbiased for the effect (up to an O(1/n) term from estimating ``theta``), and its variance falls
by roughly the share of outcome variance the covariates explain. With one covariate ``theta`` is
``cov(y, x) / var(x)``.

Covariates must be pre-treatment: adjusting for anything the treatment can move biases the
estimate. The regression ``y ~ 1 + T + (X - mean X)`` with HC2 errors gives nearly the same
estimate (its slope is the within-arm one rather than the pooled one) and is reported as a check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from numpy.typing import ArrayLike

from uplift_lab.experiments.estimators import (
    EffectEstimate,
    FloatArray,
    as_float_array,
    welch_difference,
)


@dataclass(frozen=True)
class CupedResult:
    """Unadjusted and CUPED-adjusted estimates for the same comparison."""

    raw: EffectEstimate
    adjusted: EffectEstimate
    theta: tuple[float, ...]
    variance_reduction: float
    ci_width_shrink: float


@dataclass(frozen=True)
class RegressionAdjustment:
    """Treatment coefficient from ``y ~ 1 + T + centred X`` with HC2 standard errors."""

    effect: float
    se: float
    ci_low: float
    ci_high: float


def _as_covariates(values: ArrayLike, n_rows: int, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def cuped_theta(y: ArrayLike, covariates: ArrayLike) -> FloatArray:
    """OLS slopes of ``y`` on the covariates (with intercept): ``Cov(X)^-1 Cov(X, y)``."""
    outcome = as_float_array(y, "y")
    x = _as_covariates(covariates, outcome.size, "covariates")
    x_centred = x - x.mean(axis=0)
    y_centred = outcome - outcome.mean()
    if np.any(x_centred.std(axis=0) == 0.0):
        raise ValueError("a covariate is constant; it cannot explain outcome variance")
    theta, *_ = np.linalg.lstsq(x_centred, y_centred, rcond=None)
    return np.asarray(theta, dtype=np.float64)


def cuped_difference(
    y_treated: ArrayLike,
    y_control: ArrayLike,
    x_treated: ArrayLike,
    x_control: ArrayLike,
    *,
    alpha: float = 0.05,
) -> CupedResult:
    """Compare the arms before and after CUPED with a pooled ``theta``."""
    y_t = as_float_array(y_treated, "y_treated")
    y_c = as_float_array(y_control, "y_control")
    x_t = _as_covariates(x_treated, y_t.size, "x_treated")
    x_c = _as_covariates(x_control, y_c.size, "x_control")
    if x_t.shape[1] != x_c.shape[1]:
        raise ValueError("treated and control covariates have different column counts")

    x_all = np.vstack([x_t, x_c])
    theta = cuped_theta(np.concatenate([y_t, y_c]), x_all)
    x_mean = x_all.mean(axis=0)
    raw = welch_difference(y_t, y_c, alpha=alpha)
    adjusted = welch_difference(
        y_t - (x_t - x_mean) @ theta, y_c - (x_c - x_mean) @ theta, alpha=alpha
    )
    return CupedResult(
        raw=raw,
        adjusted=adjusted,
        theta=tuple(float(value) for value in theta),
        variance_reduction=1.0 - adjusted.se**2 / raw.se**2,
        ci_width_shrink=1.0 - adjusted.ci_width / raw.ci_width,
    )


def regression_adjusted_effect(
    y_treated: ArrayLike,
    y_control: ArrayLike,
    x_treated: ArrayLike,
    x_control: ArrayLike,
    *,
    alpha: float = 0.05,
) -> RegressionAdjustment:
    """ANCOVA estimate of the effect, the regression form of CUPED."""
    y_t = as_float_array(y_treated, "y_treated")
    y_c = as_float_array(y_control, "y_control")
    x_all = np.vstack(
        [
            _as_covariates(x_treated, y_t.size, "x_treated"),
            _as_covariates(x_control, y_c.size, "x_control"),
        ]
    )
    treatment = np.concatenate([np.ones(y_t.size), np.zeros(y_c.size)])
    design = np.column_stack([np.ones(treatment.size), treatment, x_all - x_all.mean(axis=0)])
    fit = sm.OLS(np.concatenate([y_t, y_c]), design).fit(cov_type="HC2")
    low, high = fit.conf_int(alpha=alpha)[1]
    return RegressionAdjustment(
        effect=float(fit.params[1]), se=float(fit.bse[1]), ci_low=float(low), ci_high=float(high)
    )
