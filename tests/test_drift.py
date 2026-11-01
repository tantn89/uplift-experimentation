"""PSI and KS drift statistics, and the seeded shift scenarios."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from synthetic_uplift import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.monitoring.drift import (
    EPSILON,
    DriftThresholds,
    ReferenceProfile,
    bin_shares,
    decision_rate_drift,
    drift_summary,
    psi,
    quantile_edges,
)
from uplift_lab.monitoring.shift_simulation import (
    SCENARIOS,
    ShiftConfig,
    all_scenarios,
    simulate_traffic,
)


def _normal_frame(n: int, seed: int, shift: dict[str, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    for name, delta in (shift or {}).items():
        frame[name] += delta
    return frame


def test_psi_known_values() -> None:
    # (0.25 - 0.5) ln(0.5) + (0.75 - 0.5) ln(1.5)
    expected = 0.25 * math.log(2.0) + 0.25 * math.log(1.5)
    assert psi([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)
    assert psi([0.25, 0.75], [0.5, 0.5]) == pytest.approx(expected)  # symmetric
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0
    # An empty current bin is floored, so the index stays finite and large.
    empty = psi([0.5, 0.5], [0.0, 1.0])
    assert math.isfinite(empty) and empty == pytest.approx(
        (EPSILON - 0.5) * math.log(EPSILON / 0.5) + 0.5 * math.log(2.0)
    )
    with pytest.raises(ValueError, match="same bins"):
        psi([0.5, 0.5], [1.0])


def test_quantile_bins_are_right_closed_and_merge_point_masses() -> None:
    uniform = np.arange(1, 1001, dtype=float)
    edges = quantile_edges(uniform, 10)
    assert edges.size == 9
    np.testing.assert_allclose(bin_shares(uniform, edges), np.full(10, 0.1), atol=0.002)
    assert bin_shares([edges[0]], edges)[0] == 1.0  # a value on an edge falls in the lower bin
    massed = np.concatenate([np.zeros(800), np.linspace(1, 2, 200)])
    assert quantile_edges(massed, 10).size < 9
    with pytest.raises(ValueError, match="finite"):
        quantile_edges([1.0, np.nan], 10)
    with pytest.raises(ValueError, match="n_bins"):
        quantile_edges(uniform, 1)


def test_identical_distributions_have_zero_psi() -> None:
    frame = _normal_frame(5_000, seed=1)
    profile = ReferenceProfile.fit(frame, FEATURE_NAMES)
    summary = drift_summary(profile, frame, DriftThresholds())
    assert summary["status"] == "ok" and summary["flagged_features"] == []
    assert all(r["psi"] == pytest.approx(0.0, abs=1e-12) for r in summary["features"].values())
    assert all(r["ks_statistic"] == 0.0 for r in summary["features"].values())
    assert summary["prediction"] is None


def test_no_false_alarm_on_a_same_distribution_sample() -> None:
    profile = ReferenceProfile.fit(_normal_frame(50_000, seed=2), FEATURE_NAMES)
    summary = drift_summary(profile, _normal_frame(5_000, seed=3), DriftThresholds())
    assert summary["status"] == "ok"
    assert summary["ks_rejected_features"] == []
    # E[PSI] ~ (bins - 1) * (1/n_ref + 1/n_cur) = 9 * 2.2e-4 = 0.002 for unshifted data.
    assert max(r["psi"] for r in summary["features"].values()) < 0.01


def test_injected_shift_is_flagged_on_the_right_feature() -> None:
    profile = ReferenceProfile.fit(_normal_frame(50_000, seed=4), FEATURE_NAMES)
    current = _normal_frame(5_000, seed=5, shift={"f3": 0.6, "f7": 0.35})
    summary = drift_summary(profile, current, DriftThresholds())
    assert summary["features"]["f3"]["status"] == "alert"
    assert summary["features"]["f7"]["status"] == "warn"
    assert summary["flagged_features"] == ["f3", "f7"]
    assert summary["status"] == "alert"
    assert {"f3", "f7"} <= set(summary["ks_rejected_features"])
    with pytest.raises(ValueError, match="missing"):
        drift_summary(profile, current.drop(columns="f2"), DriftThresholds())


def test_thresholds_validate_and_classify() -> None:
    t = DriftThresholds(psi_warn=0.1, psi_alert=0.25)
    assert [t.status(v) for v in (0.05, 0.1, 0.2, 0.25)] == ["ok", "warn", "warn", "alert"]
    for bad in ({"psi_warn": 0.3, "psi_alert": 0.2}, {"ks_alpha": 0.0}):
        with pytest.raises(ValueError):
            DriftThresholds(**bad)  # type: ignore[arg-type]


def test_decision_rate_bounds_classify_and_validate() -> None:
    t = DriftThresholds()
    ratios = (0.49, 0.5, 0.74, 0.75, 1.0, 1.33, 1.34, 1.5, 1.51)
    expected = ["alert", "warn", "warn", "ok", "ok", "ok", "warn", "warn", "alert"]
    assert [t.rate_status(r) for r in ratios] == expected
    narrow = DriftThresholds(rate_warn=(0.9, 1.1), rate_alert=(0.9, 1.1))  # no warn band
    assert [narrow.rate_status(r) for r in (0.89, 1.0, 1.11)] == ["alert", "ok", "alert"]
    for bad in (
        {"rate_warn": (1.1, 1.3)},  # the band must contain 1
        {"rate_warn": (0.4, 1.33)},  # warn wider than alert
        {"rate_alert": (0.0, 1.5)},
        {"rate_alert": (0.5, 1.2)},
    ):
        with pytest.raises(ValueError, match="decision-rate"):
            DriftThresholds(**bad)  # type: ignore[arg-type]


def test_decision_rate_catches_a_shift_that_psi_misses() -> None:
    """Scores that all move up a little: PSI on deciles stays low, the treat rate does not."""
    rng = np.random.default_rng(9)
    reference = pd.DataFrame({"f0": rng.normal(size=50_000), "uplift": rng.normal(size=50_000)})
    profile = ReferenceProfile.fit(reference, ["f0", "uplift"])
    threshold = float(np.quantile(reference["uplift"], 0.9))  # a 10% budget
    current = pd.DataFrame({"f0": rng.normal(size=20_000), "uplift": rng.normal(size=20_000)})
    current["uplift"] += 0.28
    summary = drift_summary(profile, current, DriftThresholds(), decision=(0.1, threshold))
    rate = summary["decision_rate"]
    assert summary["prediction"]["status"] == "ok"  # PSI below 0.1
    assert rate["treat_rate"] == pytest.approx(0.158, abs=0.01)  # P(Z > 1.2816 - 0.28)
    assert rate["ratio"] == pytest.approx(rate["treat_rate"] / 0.1)
    assert rate["status"] == "alert" and summary["status"] == "alert"
    unshifted = drift_summary(profile, reference, DriftThresholds(), decision=(0.1, threshold))
    assert unshifted["decision_rate"]["ratio"] == pytest.approx(1.0, abs=1e-3)
    assert unshifted["status"] == "ok"
    assert drift_summary(profile, current, DriftThresholds())["decision_rate"] is None
    with pytest.raises(ValueError, match="'uplift' column"):
        drift_summary(profile, current[["f0"]], DriftThresholds(), decision=(0.1, threshold))
    with pytest.raises(ValueError, match="budget"):
        decision_rate_drift(current["uplift"], 0.0, threshold, DriftThresholds())


def test_shift_scenarios_move_what_they_claim() -> None:
    frame = synthetic_uplift_frame(30_000, seed=6)
    config = ShiftConfig(covariate_features=("f2",))
    traffic = all_scenarios(frame, 20_000, seed=7, config=config)
    assert list(traffic) == list(SCENARIOS)
    base_visit = frame["visit"].mean()
    assert traffic["no_shift"]["treatment"].mean() == pytest.approx(0.85, abs=0.01)
    assert abs(traffic["no_shift"]["f2"].mean()) < 0.03
    assert traffic["covariate_shift"]["f2"].mean() > 0.4
    assert traffic["treatment_mix_shift"]["treatment"].mean() == pytest.approx(0.5, abs=0.015)
    assert traffic["outcome_prior_shift"]["visit"].mean() == pytest.approx(3 * base_visit, rel=0.05)
    again = simulate_traffic(frame, "covariate_shift", 100, seed=8, config=config)
    pd.testing.assert_frame_equal(
        again, simulate_traffic(frame, "covariate_shift", 100, seed=8, config=config)
    )
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate_traffic(frame, "concept_shift", 10, seed=1, config=config)
    with pytest.raises(ValueError):
        ShiftConfig(treated_share=1.0)
