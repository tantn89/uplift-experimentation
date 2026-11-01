"""Strong-covariate CUPED simulation and the price of peeking (mSPRT cap for target power)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from uplift_lab.experiments.cuped_simulation import CupedSimConfig, simulate_cuped_contrast
from uplift_lab.experiments.peeking_price import price_of_peeking
from uplift_lab.experiments.peeking_simulation import PeekingConfig

SMALL_CUPED = CupedSimConfig(n_per_arm=300, n_reps=600, rhos=(0.0, 0.8), seed=5)
SMALL_PEEKING = PeekingConfig(n_sims=3_000, looks=10, batch_per_arm=400, seed=11)


def test_cuped_variance_ratio_follows_one_minus_rho_squared() -> None:
    weak, strong = simulate_cuped_contrast(SMALL_CUPED)["by_rho"]
    # With 600 replicates a variance ratio has a relative sd of about 6%.
    assert weak["empirical_variance_ratio"] == pytest.approx(1.0, abs=0.06)
    assert strong["empirical_variance_ratio"] == pytest.approx(0.36, abs=0.05)
    assert strong["mean_reported_variance_reduction"] == pytest.approx(0.64, abs=0.02)
    assert weak["mean_reported_variance_reduction"] == pytest.approx(0.0, abs=0.01)


def test_cuped_stays_unbiased_with_nominal_coverage() -> None:
    for row in simulate_cuped_contrast(SMALL_CUPED)["by_rho"]:
        # Sampling sd of the raw estimate is ~0.08; the mean of 600 is within ~0.01.
        assert row["cuped_mean_effect"] == pytest.approx(SMALL_CUPED.effect, abs=0.012)
        assert row["cuped_ci_coverage"] == pytest.approx(0.95, abs=0.03)
        assert row["raw_ci_coverage"] == pytest.approx(0.95, abs=0.03)


def test_cuped_contrast_is_seeded_and_validated() -> None:
    assert simulate_cuped_contrast(SMALL_CUPED) == simulate_cuped_contrast(SMALL_CUPED)
    cases: list[dict[str, Any]] = [{"n_reps": 5}, {"rhos": ()}, {"rhos": (1.0,)}, {"n_per_arm": 1}]
    for kwargs in cases:
        with pytest.raises(ValueError):
            CupedSimConfig(**kwargs)


@pytest.fixture(scope="module")
def price() -> dict[str, Any]:
    return price_of_peeking(SMALL_PEEKING)


def test_msprt_needs_a_larger_cap_for_the_fixed_tests_power(price: dict[str, Any]) -> None:
    assert price["msprt_power_at_fixed_horizon"] < 0.8
    assert price["cap_ratio_vs_fixed"] > 1.0
    curve = price["power_by_cap"]
    power = np.asarray(curve["msprt_power"])
    assert np.all(np.diff(power) >= 0.0)  # a larger cap never loses a rejection
    cap_index = curve["n_per_arm"].index(price["msprt_cap_per_arm"])
    assert power[cap_index] >= 0.8 > power[cap_index - 1]
    # It stops early often enough that the average run is far below the cap.
    assert price["msprt_mean_n_per_arm_under_cap"] < price["msprt_cap_per_arm"]


def test_larger_cap_keeps_msprt_valid(price: dict[str, Any]) -> None:
    # 3,000 A/A runs: the se of a 0.05 rate is 0.004.
    assert price["msprt_type_one_error_under_cap"] <= SMALL_PEEKING.alpha + 0.01


def test_unreached_target_and_bad_arguments() -> None:
    unreached = price_of_peeking(SMALL_PEEKING, max_horizon_factor=1)
    assert unreached["msprt_cap_per_arm"] is None
    assert unreached["msprt_type_one_error_under_cap"] is None
    with pytest.raises(ValueError):
        price_of_peeking(SMALL_PEEKING, target_power=1.0)
    with pytest.raises(ValueError):
        price_of_peeking(SMALL_PEEKING, max_horizon_factor=0)
