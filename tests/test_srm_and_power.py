"""SRM chi-square test and power / sample-size formulas against statsmodels."""

from __future__ import annotations

import pytest
from statsmodels.stats.power import NormalIndPower

from uplift_lab.experiments.power import minimum_detectable_effect, power_at, required_n
from uplift_lab.experiments.srm import srm_test


def test_srm_passes_balanced_split() -> None:
    result = srm_test({"a": 21_307, "b": 21_387, "c": 21_306})
    assert not result.mismatch
    assert result.p_value > 0.5


def test_srm_flags_skewed_split() -> None:
    # A 1.5% imbalance on 100k units: chi2 = 22.2, far past the p < 0.001 alarm.
    result = srm_test({"control": 50_000, "treatment": 51_500})
    assert result.mismatch
    assert result.p_value < 1e-5
    # 1% is borderline (p = 0.0017) and stays below the alarm: the threshold is deliberately strict.
    assert not srm_test({"control": 50_000, "treatment": 51_000}).mismatch


def test_srm_uses_design_ratios() -> None:
    assert not srm_test({"treated": 85_100, "control": 14_900}, [0.85, 0.15]).mismatch
    assert srm_test({"treated": 50_000, "control": 50_000}, [0.85, 0.15]).mismatch


@pytest.mark.parametrize(
    ("counts", "ratios"),
    [
        ({"a": 10}, None),
        ({"a": 0, "b": 0}, None),
        ({"a": 5, "b": 5}, [1.0]),
        ({"a": 5, "b": 5}, [1, 0]),
    ],
)
def test_srm_rejects_invalid_input(counts: dict[str, int], ratios: list[float] | None) -> None:
    with pytest.raises(ValueError):
        srm_test(counts, ratios)


@pytest.mark.parametrize(
    ("n", "mde", "sd", "ratio"), [(1_000, 0.1, 1.0, 1.0), (21_306, 0.0085, 0.308, 2.0)]
)
def test_power_matches_statsmodels(n: int, mde: float, sd: float, ratio: float) -> None:
    expected = NormalIndPower().power(effect_size=mde / sd, nobs1=n, alpha=0.05, ratio=ratio)
    assert power_at(n, mde, sd, ratio=ratio) == pytest.approx(expected, rel=1e-10)


def test_required_n_matches_statsmodels_and_closed_form() -> None:
    # Visit-like rate: 10.6% control, 10% relative lift.
    sd, mde = (0.106 * 0.894) ** 0.5, 0.0106
    n = required_n(mde, sd)
    solved = NormalIndPower().solve_power(effect_size=mde / sd, alpha=0.05, power=0.8)
    assert n == pytest.approx(solved, abs=1.0)
    assert n == pytest.approx(2 * (1.959964 + 0.841621) ** 2 * sd**2 / mde**2, abs=1.0)
    assert power_at(n, mde, sd) == pytest.approx(0.8, abs=1e-3)


def test_mde_inverts_power_and_variance_factor_scales_n() -> None:
    mde = minimum_detectable_effect(10_000, 2.0, power=0.9)
    assert power_at(10_000, mde, 2.0) == pytest.approx(0.9, abs=1e-6)
    full, halved = required_n(0.05, 1.0), required_n(0.05, 1.0, variance_factor=0.5)
    assert halved == pytest.approx(full / 2, abs=1.0)


def test_power_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="variance_factor"):
        power_at(100, 0.1, 1.0, variance_factor=0.0)
    with pytest.raises(ValueError, match="mde"):
        required_n(0.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        minimum_detectable_effect(0, 1.0)
