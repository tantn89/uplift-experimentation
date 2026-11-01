"""mSPRT closed form and the seeded peeking simulation: type-I control versus naive peeking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

from uplift_lab.experiments.msprt import (
    always_valid_ci_halfwidth,
    always_valid_p_values,
    msprt_log_lr,
    msprt_reject,
)
from uplift_lab.experiments.peeking_simulation import PeekingConfig, simulate_peeking, summarise

ALPHA = 0.05
CONFIG = PeekingConfig(n_sims=4_000, looks=20, batch_per_arm=500, alpha=ALPHA)


@pytest.mark.parametrize(
    ("d", "v", "tau"), [(0.0, 1e-4, 0.01), (0.03, 2e-4, 0.02), (-0.5, 0.04, 1.0)]
)
def test_log_lr_equals_numerical_mixture(d: float, v: float, tau: float) -> None:
    def integrand(delta: float) -> float:
        ratio = stats.norm.pdf(d, delta, math.sqrt(v)) / stats.norm.pdf(d, 0.0, math.sqrt(v))
        return float(ratio * stats.norm.pdf(delta, 0.0, tau))

    numeric, _ = integrate.quad(integrand, -12 * tau, 12 * tau, points=[0.0, d])
    assert float(msprt_log_lr(d, v, tau)) == pytest.approx(math.log(numeric), rel=1e-6, abs=1e-9)


def test_confidence_sequence_boundary_is_the_rejection_threshold() -> None:
    v, tau = 3e-5, 0.01
    half = float(always_valid_ci_halfwidth(v, tau, alpha=ALPHA))
    assert float(msprt_log_lr(half, v, tau)) == pytest.approx(math.log(1 / ALPHA))
    assert bool(msprt_reject(half * 1.001, v, tau, alpha=ALPHA))
    assert not bool(msprt_reject(half * 0.999, v, tau, alpha=ALPHA))


def test_always_valid_p_values_are_monotone_and_bounded() -> None:
    p = always_valid_p_values([[-1.0, 0.5, 2.0, 1.0, np.nan]])
    np.testing.assert_allclose(
        p[0], [1.0, math.exp(-0.5), math.exp(-2.0), math.exp(-2.0), math.exp(-2.0)]
    )


def test_non_positive_variance_never_rejects() -> None:
    assert not msprt_reject(0.5, 0.0, 0.1).any()
    with pytest.raises(ValueError, match="tau"):
        msprt_log_lr(0.1, 0.01, 0.0)


def test_msprt_controls_type_one_error_while_naive_peeking_inflates_it() -> None:
    summary = summarise(simulate_peeking(CONFIG, 0.0))
    # 4,000 runs: binomial sd of a 5% rate is ~0.0034.
    assert summary["msprt"]["rejection_rate"] <= ALPHA + 0.01
    assert summary["fixed_horizon"]["rejection_rate"] == pytest.approx(ALPHA, abs=0.012)
    assert summary["naive_peeking"]["rejection_rate"] > 3 * ALPHA


def test_power_and_stopping_time_under_the_design_effect() -> None:
    effect = CONFIG.design_effect()
    outcome = simulate_peeking(CONFIG, effect)
    summary = summarise(outcome)
    assert summary["fixed_horizon"]["rejection_rate"] == pytest.approx(CONFIG.power, abs=0.03)
    assert 0.3 < summary["msprt"]["rejection_rate"] < summary["fixed_horizon"]["rejection_rate"]
    median = summary["msprt"]["median_stop_n_if_rejected"]
    assert CONFIG.batch_per_arm <= median <= CONFIG.horizon_per_arm
    assert outcome.cumulative_rejection("msprt")[-1] == pytest.approx(
        outcome.rejection_rate("msprt")
    )


def test_simulation_is_deterministic_for_a_seed() -> None:
    small = PeekingConfig(n_sims=200, looks=5, batch_per_arm=200)
    first, second = simulate_peeking(small, 0.01), simulate_peeking(small, 0.01)
    for rule, stops in first.first_rejection.items():
        np.testing.assert_array_equal(stops, second.first_rejection[rule])


@pytest.mark.parametrize(
    "kwargs", [{"n_sims": 0}, {"batch_per_arm": 1}, {"base_rate": 1.0}, {"alpha": 0.0}]
)
def test_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PeekingConfig(**kwargs)


def test_effect_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="base_rate"):
        simulate_peeking(PeekingConfig(n_sims=10), 0.95)
