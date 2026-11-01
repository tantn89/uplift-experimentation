"""CUPED: unbiasedness, variance reduction, theta = OLS slope, agreement with ANCOVA."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from uplift_lab.experiments.cuped import (
    cuped_difference,
    cuped_theta,
    regression_adjusted_effect,
)
from uplift_lab.experiments.estimators import FloatArray

TRUE_EFFECT = 0.25


def _experiment(
    rng: np.random.Generator, n: int, rho: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Unit-variance outcome with correlation ``rho`` to a pre-period covariate."""
    x = rng.normal(size=2 * n)
    noise = rng.normal(size=2 * n)
    y = 1.0 + rho * x + np.sqrt(1.0 - rho**2) * noise
    y[:n] += TRUE_EFFECT
    return y[:n], y[n:], x[:n], x[n:]


def test_unbiased_and_lower_variance_in_simulation() -> None:
    rng = np.random.default_rng(20261101)
    raw, adjusted = [], []
    for _ in range(1_000):
        result = cuped_difference(*_experiment(rng, 400, rho=0.7))
        raw.append(result.raw.effect)
        adjusted.append(result.adjusted.effect)
    # Sampling sd of each estimate is ~0.07 (raw) and ~0.05 (CUPED); the mean of 1,000 is ~0.002.
    assert np.mean(adjusted) == pytest.approx(TRUE_EFFECT, abs=0.006)
    assert np.mean(raw) == pytest.approx(TRUE_EFFECT, abs=0.008)
    # Theory: Var(CUPED) / Var(raw) = 1 - rho^2 = 0.51.
    assert np.var(adjusted) / np.var(raw) == pytest.approx(0.51, abs=0.05)


def test_reported_variance_reduction_tracks_correlation() -> None:
    rng = np.random.default_rng(3)
    correlated = cuped_difference(*_experiment(rng, 20_000, rho=0.6))
    assert correlated.variance_reduction == pytest.approx(0.36, abs=0.02)
    assert correlated.ci_width_shrink == pytest.approx(1 - np.sqrt(0.64), abs=0.02)
    unrelated = cuped_difference(*_experiment(rng, 20_000, rho=0.0))
    assert abs(unrelated.variance_reduction) < 0.005


def test_theta_matches_ols_slopes_with_several_covariates() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 3))
    y = 2.0 + x @ np.array([0.5, -1.0, 0.0]) + rng.normal(size=500)
    expected = sm.OLS(y, sm.add_constant(x)).fit().params[1:]
    np.testing.assert_allclose(cuped_theta(y, x), expected, rtol=1e-10)
    single = cuped_theta(y, x[:, 0])
    assert single[0] == pytest.approx(np.cov(y, x[:, 0])[0, 1] / np.var(x[:, 0], ddof=1))


def test_matches_regression_adjustment() -> None:
    rng = np.random.default_rng(5)
    arms = _experiment(rng, 5_000, rho=0.5)
    cuped = cuped_difference(*arms)
    ancova = regression_adjusted_effect(*arms)
    # Pooled versus within-arm slope differ by O(effect / n); both errors agree closely.
    assert ancova.effect == pytest.approx(cuped.adjusted.effect, abs=0.1 * cuped.adjusted.se)
    assert ancova.se == pytest.approx(cuped.adjusted.se, rel=0.02)
    assert ancova.ci_low < ancova.effect < ancova.ci_high


def test_rejects_constant_covariate() -> None:
    with pytest.raises(ValueError, match="constant"):
        cuped_theta([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="rows"):
        cuped_difference([1.0, 2.0], [1.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="column counts"):
        cuped_difference([1.0, 2.0], [1.0, 3.0], [[1.0, 2.0], [2.0, 1.0]], [1.0, 2.0])
