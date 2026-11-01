"""Policy value, bootstrap intervals and the evaluation payload."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_uplift import synthetic_uplift_frame, true_uplift
from uplift_lab.uplift.bootstrap import bootstrap_intervals, bootstrap_weights
from uplift_lab.uplift.evaluation import RANDOM_SCORER, decile_table, evaluate_scores
from uplift_lab.uplift.metrics import Ranking, gains
from uplift_lab.uplift.policy import average_treatment_effect, policy_value

SCORES = np.arange(8, 0, -1, dtype=float)
T = np.array([1, 0, 1, 0, 1, 0, 1, 0])
Y = np.array([1, 0, 1, 0, 0, 0, 0, 1])


def test_policy_value_hand_computed() -> None:
    value = policy_value(gains(Ranking(SCORES), T, Y), 0.5)
    # The top four: treated 2/2 visit, control 0/2 -> +1 visit per contact, 0.5 per user.
    # Horvitz-Thompson with e = 0.5: sum(z) = 2 / 0.5 = 4 over 4 contacts -> also 1.
    assert value.targeted_share == 0.5
    assert value.model_per_1000_contacts == pytest.approx(1000.0)
    assert value.model_ipw_per_1000_contacts == pytest.approx(1000.0)
    assert value.model_per_1000_users == pytest.approx(500.0)
    # ATE = 2/4 - 1/4.
    assert value.treat_all_per_1000_contacts == pytest.approx(250.0)
    assert value.random_per_1000_users == pytest.approx(125.0)
    assert value.gain_over_random_per_1000_contacts == pytest.approx(750.0)


def test_treat_all_equals_difference_in_means() -> None:
    frame = synthetic_uplift_frame(20_000, seed=4)
    t, y = frame["treatment"].to_numpy(), frame["visit"].to_numpy()
    ate = y[t == 1].mean() - y[t == 0].mean()
    g = gains(Ranking(np.random.default_rng(0).random(len(frame))), t, y)
    value = policy_value(g, 1.0)
    assert average_treatment_effect(g) == pytest.approx(ate, rel=1e-12)
    assert value.model_per_1000_contacts == pytest.approx(1000 * ate, rel=1e-12)
    assert value.model_ipw_per_1000_contacts == pytest.approx(1000 * ate, rel=1e-12)
    assert value.gain_over_random_per_1000_contacts == pytest.approx(0.0, abs=1e-9)


def test_difference_estimator_is_robust_to_arm_imbalance_in_the_segment() -> None:
    # Everyone visits, so the true effect is zero, but the top half is 3 treated : 1 control
    # while the overall share is 0.5. Horvitz-Thompson reads (3/0.5 - 1/0.5) / 4 = +1 per contact.
    t = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    value = policy_value(gains(Ranking(SCORES), t, np.ones(8)), 0.5)
    assert value.model_per_1000_contacts == 0.0
    assert value.model_ipw_per_1000_contacts == pytest.approx(1000.0)


def test_policy_value_matches_true_effect_of_targeted_users() -> None:
    frame = synthetic_uplift_frame(400_000, seed=5)
    tau = true_uplift(frame)
    g = gains(Ranking(tau), frame["treatment"], frame["visit"])
    value = policy_value(g, 0.2)
    top = np.sort(tau)[::-1][: int(0.2 * tau.size)]
    # Standard errors here are about 4 per 1,000 contacts (0.8 per 1,000 users at a 20%
    # budget) for the difference estimator, a little more for Horvitz-Thompson.
    assert value.model_per_1000_users == pytest.approx(1000 * top.sum() / tau.size, abs=3.0)
    assert value.model_per_1000_contacts == pytest.approx(1000 * top.mean(), abs=15.0)
    assert value.model_ipw_per_1000_contacts == pytest.approx(1000 * top.mean(), abs=20.0)
    assert value.gain_over_random_per_1000_contacts > 50


def test_bootstrap_is_seeded_and_covers_the_mean() -> None:
    first = [w.copy() for w in bootstrap_weights(100, 3, seed=1)]
    second = list(bootstrap_weights(100, 3, seed=1))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.sum() == 100
    x = np.random.default_rng(2).normal(1.0, 1.0, 400)

    def mean(w: np.ndarray | None) -> dict[str, float]:
        return {"mean": float(np.average(x, weights=w))}

    interval = bootstrap_intervals(mean, x.size, n_boot=400, seed=3)["mean"]
    assert interval.estimate == pytest.approx(x.mean())
    # Percentile interval close to the normal one, 1.96 * sd / sqrt(n) ~ 0.098 half-width.
    assert interval.ci_low < x.mean() < interval.ci_high
    assert interval.ci_high - interval.ci_low == pytest.approx(0.196, rel=0.2)
    assert interval.excludes_zero
    with pytest.raises(ValueError, match="positive"):
        next(bootstrap_weights(0, 1, seed=0))


def test_evaluate_scores_payload() -> None:
    frame = synthetic_uplift_frame(20_000, seed=6, treated_share=0.5)
    tau = true_uplift(frame)
    scores = {"oracle": tau, RANDOM_SCORER: np.random.default_rng(1).random(tau.size)}
    payload = evaluate_scores(
        scores, frame["treatment"], frame["visit"], budgets=(0.1, 0.5), n_bootstrap=50, seed=1
    )
    assert payload["best_model"] == "oracle"
    assert payload["best_beats_random"] is True
    oracle = payload["models"]["oracle"]
    assert set(oracle) == {"qini_coefficient", "auuc", "auuc_gain", "uplift_at_k", "policy"}
    assert set(oracle["policy"]) == {"0.1", "0.5"}
    gain = oracle["policy"]["0.1"]["gain_over_random_per_1000_contacts"]
    assert gain["ci_low"] > 0
    ate = payload["ate"]
    assert ate["ci_low"] < ate["estimate"] < ate["ci_high"]
    assert len(payload["curves"]["oracle"]["qini"]) == 101
    assert payload["deciles"]["oracle"][0]["observed_uplift"] > ate["estimate"]
    with pytest.raises(ValueError, match="same length"):
        evaluate_scores({"a": tau[:-1]}, frame["treatment"], frame["visit"], n_bootstrap=2)


def test_decile_table_partitions_rows() -> None:
    rows = decile_table(SCORES, T, Y, n_bins=4)
    assert [r["decile"] for r in rows] == [1, 2, 3, 4]
    assert sum(r["n_treated"] + r["n_control"] for r in rows) == 8
    assert rows[0]["observed_uplift"] == pytest.approx(1.0)
    assert rows[0]["mean_predicted"] == pytest.approx(7.5)
