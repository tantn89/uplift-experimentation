"""Welch difference in means: agreement with scipy and CI coverage in a seeded simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from uplift_lab.experiments.estimators import welch_difference


def test_matches_scipy_welch_test() -> None:
    rng = np.random.default_rng(1)
    treated, control = rng.normal(1.2, 3.0, 400), rng.normal(1.0, 1.0, 250)
    result = welch_difference(treated, control)
    reference = stats.ttest_ind(treated, control, equal_var=False)
    interval = reference.confidence_interval(0.95)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert result.df == pytest.approx(reference.df, rel=1e-9)
    assert (result.ci_low, result.ci_high) == pytest.approx((interval.low, interval.high))


def test_ci_coverage_is_nominal_with_unequal_variances() -> None:
    rng = np.random.default_rng(20261101)
    true_effect, reps = 0.3, 4_000
    covered = 0
    for _ in range(reps):
        result = welch_difference(rng.normal(true_effect, 2.0, 60), rng.normal(0.0, 0.5, 30))
        covered += result.ci_low <= true_effect <= result.ci_high
    # Binomial sd of the coverage at 4,000 reps is about 0.0034; allow roughly 3 sd.
    assert abs(covered / reps - 0.95) < 0.011


def test_relative_lift_and_its_interval() -> None:
    rng = np.random.default_rng(2)
    result = welch_difference(rng.binomial(1, 0.12, 50_000), rng.binomial(1, 0.10, 50_000))
    assert result.relative_lift == pytest.approx(result.mean_treated / result.mean_control - 1)
    assert result.relative_lift_ci_low < result.relative_lift < result.relative_lift_ci_high
    assert result.relative_lift_ci_low < 0.2 < result.relative_lift_ci_high


def test_zero_control_mean_gives_undefined_lift() -> None:
    result = welch_difference([0.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert math.isnan(result.relative_lift)


@pytest.mark.parametrize(
    ("treated", "control", "message"),
    [
        ([1.0], [1.0, 2.0], "at least two"),
        ([1.0, float("nan")], [1.0, 2.0], "NaN"),
        ([[1.0, 2.0]], [1.0, 2.0], "one-dimensional"),
        ([1.0, 1.0], [2.0, 2.0], "zero standard error"),
    ],
)
def test_rejects_invalid_samples(
    treated: list[float] | list[list[float]], control: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        welch_difference(treated, control)


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        welch_difference([1.0, 2.0], [1.0, 3.0], alpha=1.5)
