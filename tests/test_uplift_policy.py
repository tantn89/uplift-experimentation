"""Policy value, bootstrap intervals and the evaluation payload."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_uplift import synthetic_uplift_frame, true_uplift
from uplift_lab.uplift.bootstrap import (
    bootstrap_intervals,
    bootstrap_p_value,
    bootstrap_weights,
    holm_adjust,
)
from uplift_lab.uplift.champion import CHAMPION_RULE, select_champion
from uplift_lab.uplift.evaluation import RANDOM_SCORER, decile_table, evaluate_scores
from uplift_lab.uplift.metrics import Ranking, gains, qini_coefficient
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
    # Only the random scorer is a rival here, and it is excluded from the winner rule.
    assert set(payload["qini_diff_vs_best"]) == {RANDOM_SCORER}
    assert payload["qini_diff_vs_best"][RANDOM_SCORER]["ci_high"] < 0
    assert payload["best_is_clear_winner"] is False
    with pytest.raises(ValueError, match="same length"):
        evaluate_scores({"a": tau[:-1]}, frame["treatment"], frame["visit"], n_bootstrap=2)


def test_paired_qini_differences_separate_signal_from_noise() -> None:
    frame = synthetic_uplift_frame(20_000, seed=8, treated_share=0.5)
    tau = true_uplift(frame)
    noise = np.random.default_rng(2).normal(scale=1e-6, size=tau.size)
    scores = {"oracle": tau, "oracle_jitter": tau + noise, "weak": -tau}
    # Two Holm-corrected comparisons: 100 draws let the p-value reach 2 * 2/101 < 0.05.
    payload = evaluate_scores(scores, frame["treatment"], frame["visit"], n_bootstrap=100, seed=3)
    diffs = payload["qini_diff_vs_best"]
    best = payload["best_model"]
    assert best in {"oracle", "oracle_jitter"} and best not in diffs
    twin = ({"oracle", "oracle_jitter"} - {best}).pop()
    # A near-identical ranking: the paired interval is tight and covers zero.
    assert diffs[twin]["ci_low"] <= 0.0 <= diffs[twin]["ci_high"] + 1e-12
    assert diffs[twin]["ci_high"] - diffs[twin]["ci_low"] < 1e-3
    assert diffs["weak"]["ci_high"] < 0
    assert diffs["weak"]["p_holm"] <= 0.05 < diffs[twin]["p_holm"]
    assert payload["best_is_clear_winner"] is False
    assert payload["champion"]["tied_with_best"] == [twin]
    # With 30 draws the smallest attainable p-value is 2/31 > 0.05: no winner can be named.
    too_few = evaluate_scores(
        {"oracle": tau, "weak": -tau}, frame["treatment"], frame["visit"], n_bootstrap=30, seed=3
    )
    assert too_few["best_is_clear_winner"] is False
    clear = evaluate_scores(
        {"oracle": tau, "weak": -tau}, frame["treatment"], frame["visit"], n_bootstrap=60, seed=3
    )
    assert clear["best_is_clear_winner"] is True
    assert clear["champion"]["model"] == "oracle"
    assert clear["champion"]["rule"] == CHAMPION_RULE


def test_champion_rule_prefers_the_simplest_tied_learner() -> None:
    frame = synthetic_uplift_frame(20_000, seed=8, treated_share=0.5)
    tau = true_uplift(frame)
    jitter = np.random.default_rng(2).normal(scale=1e-6, size=tau.size)
    # The transformed outcome scores best by a hair; S-learner ties with it; T is much worse.
    scores = {"transformed_outcome": tau, "s_learner": tau + jitter, "t_learner": -tau}
    payload = evaluate_scores(scores, frame["treatment"], frame["visit"], n_bootstrap=100, seed=3)
    champion = payload["champion"]
    assert champion["best_by_qini"] in {"transformed_outcome", "s_learner"}
    assert champion["model"] == "s_learner"
    twin = ({"transformed_outcome", "s_learner"} - {champion["best_by_qini"]}).pop()
    assert champion["tied_with_best"] == [twin]  # T is significantly worse, so not tied
    assert champion["reason"].startswith("no clear winner")
    assert payload["best_is_clear_winner"] is False


def test_select_champion_rule() -> None:
    order = ["s_learner", "t_learner", "x_learner", "transformed_outcome"]
    tied = select_champion("transformed_outcome", order[:3], {n: 0.4 for n in order[:3]}, 0.05)
    assert tied["model"] == "s_learner" and tied["tied_with_best"] == order[:3]
    partial = select_champion(
        "x_learner",
        ["s_learner", "t_learner", "transformed_outcome"],
        {"s_learner": 0.001, "t_learner": 0.2, "transformed_outcome": 0.3},
        0.05,
    )
    assert partial["model"] == "t_learner"  # S is significantly worse; T is the simplest tie
    clear = select_champion("x_learner", ["s_learner"], {"s_learner": 0.01}, 0.05)
    assert clear["model"] == "x_learner" and "clear winner" in clear["reason"]
    missing = select_champion("x_learner", ["s_learner"], {"s_learner": float("nan")}, 0.05)
    assert missing["model"] == "s_learner"
    alone = select_champion("t_learner", [], {}, 0.05)
    assert alone["model"] == "t_learner"
    with pytest.raises(ValueError, match="alpha"):
        select_champion("t_learner", [], {}, 1.5)


def test_holm_and_bootstrap_p_values() -> None:
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )
    assert holm_adjust({"a": 0.5, "b": float("nan")})["a"] == pytest.approx(0.5)
    assert np.isnan(holm_adjust({"b": float("nan")})["b"])
    below = -np.linspace(1.0, 2.0, 99)
    assert bootstrap_p_value(below) == pytest.approx(2.0 / 100.0)
    assert bootstrap_p_value(np.concatenate([below, -below])) == pytest.approx(1.0)
    assert np.isnan(bootstrap_p_value(np.array([np.nan, np.nan])))


def test_replicates_that_lose_an_arm_are_dropped() -> None:
    # 1 control and 19 treated users: about a third of the resamples contain no control user.
    treatment = np.array([0] + [1] * 19)
    outcome = np.array([0, 1] * 10)
    scores = {"a": np.arange(20.0), "b": np.arange(20.0)[::-1]}
    payload = evaluate_scores(scores, treatment, outcome, budgets=(0.5,), n_bootstrap=50, seed=4)
    qini = payload["models"]["a"]["qini_coefficient"]
    assert np.isfinite([qini["estimate"], qini["ci_low"], qini["ci_high"]]).all()
    assert np.isfinite(payload["ate"]["ci_low"])
    weights = np.zeros(20)
    weights[1:] = 1.0  # a replicate without the control user
    g = gains(Ranking(scores["a"]), treatment, outcome, weights)
    assert not g.has_both_arms
    assert np.isnan(qini_coefficient(g)) and np.isnan(average_treatment_effect(g))
    assert np.isnan(policy_value(g, 0.5).model_ipw_per_1000_contacts)
    # The original sample is still validated: without any control user it is an error.
    with pytest.raises(ValueError, match="both treated and control"):
        evaluate_scores(scores, np.ones(20), outcome, n_bootstrap=5)


def test_decile_table_partitions_rows() -> None:
    rows = decile_table(SCORES, T, Y, n_bins=4)
    assert [r["decile"] for r in rows] == [1, 2, 3, 4]
    assert sum(r["n_treated"] + r["n_control"] for r in rows) == 8
    assert rows[0]["observed_uplift"] == pytest.approx(1.0)
    assert rows[0]["mean_predicted"] == pytest.approx(7.5)
