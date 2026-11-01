"""Qini, AUUC and uplift@k: hand-computed values, ties, weights and ranking quality ordering."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_uplift import synthetic_uplift_frame, true_uplift
from uplift_lab.uplift.metrics import (
    Ranking,
    auuc,
    auuc_gain,
    gains,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_curve,
)

# Four users ranked 1..4: a treated visitor, a control non-visitor, a treated and a control
# non-visitor. N = 4, so curves are counts / 4 at targeted fractions 0, 0.25, 0.5, 0.75, 1.
SCORES = np.array([4.0, 3.0, 2.0, 1.0])
T = np.array([1, 0, 1, 0])
Y = np.array([1, 0, 0, 0])


def test_hand_computed_curves_and_areas() -> None:
    g = gains(Ranking(SCORES), T, Y)
    fraction, qini = qini_curve(g)
    np.testing.assert_allclose(fraction, [0, 0.25, 0.5, 0.75, 1.0])
    # Q = (Y_t - Y_c * N_t / N_c) / N with Y_c = 0 throughout.
    np.testing.assert_allclose(qini, [0, 0.25, 0.25, 0.25, 0.25])
    _, uplift = uplift_curve(g)
    # U = (Y_t/N_t - Y_c/N_c) * k / N: k=1 -> 1 * 1/4 (empty control counts 0), k=2 -> 1 * 2/4,
    # k=3 -> 1/2 * 3/4, k=4 -> 1/2 * 1.
    np.testing.assert_allclose(uplift, [0, 0.25, 0.5, 0.375, 0.5])
    # Trapezoids of width 0.25: Qini area 0.21875 minus random 0.125.
    assert qini_coefficient(g) == pytest.approx(0.09375)
    assert auuc(g) == pytest.approx(0.34375)
    assert auuc_gain(g) == pytest.approx(0.34375 - 0.25)
    assert uplift_at_k(g, 0.5) == pytest.approx(1.0)
    assert uplift_at_k(g, 1.0) == pytest.approx(0.5)
    assert np.isnan(uplift_at_k(g, 0.25))  # no control user in the top one


def test_reversed_ranking_is_below_random() -> None:
    g = gains(Ranking(-SCORES), T, Y)
    assert qini_coefficient(g) < 0


def test_ties_do_not_depend_on_row_order() -> None:
    scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 2.0])
    t = np.array([1, 0, 1, 0, 1, 0])
    y = np.array([1, 0, 0, 1, 1, 0])
    reference = gains(Ranking(scores), t, y)
    for seed in range(5):
        p = np.random.default_rng(seed).permutation(scores.size)
        permuted = gains(Ranking(scores[p]), t[p], y[p])
        assert qini_coefficient(permuted) == pytest.approx(qini_coefficient(reference))
        assert auuc(permuted) == pytest.approx(auuc(reference))
    # Tie groups {2}, {1,1,1}, {0,0}: the curve is evaluated only at their ends.
    np.testing.assert_allclose(reference.fraction, [0, 1 / 6, 4 / 6, 1.0])


def test_weights_equal_duplicated_rows() -> None:
    rng = np.random.default_rng(4)
    scores, t, y = rng.random(50), rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    weights = rng.integers(0, 4, 50).astype(float)
    weighted = gains(Ranking(scores), t, y, weights)
    reps = weights.astype(int)
    duplicated = gains(Ranking(np.repeat(scores, reps)), np.repeat(t, reps), np.repeat(y, reps))
    assert qini_coefficient(weighted) == pytest.approx(qini_coefficient(duplicated))
    assert auuc(weighted) == pytest.approx(auuc(duplicated))
    assert uplift_at_k(weighted, 0.3) == pytest.approx(uplift_at_k(duplicated, 0.3))


def test_perfect_ranking_beats_noisy_beats_random() -> None:
    frame = synthetic_uplift_frame(60_000, seed=8, treated_share=0.5)
    tau = true_uplift(frame)
    t, y = frame["treatment"], frame["visit"]
    noisy = tau + np.random.default_rng(1).normal(0, 0.05, tau.size)
    random = np.random.default_rng(2).random(tau.size)
    perfect_q, noisy_q, random_q = (
        qini_coefficient(gains(Ranking(s), t, y)) for s in (tau, noisy, random)
    )
    assert perfect_q > noisy_q > random_q
    assert perfect_q > 0.005
    assert uplift_at_k(gains(Ranking(tau), t, y), 0.2) > 0.12


def test_random_ranking_is_zero_in_expectation() -> None:
    frame = synthetic_uplift_frame(5_000, seed=9)
    t, y = frame["treatment"], frame["visit"]
    rng = np.random.default_rng(20261101)
    values = np.array(
        [qini_coefficient(gains(Ranking(rng.random(len(frame))), t, y)) for _ in range(300)]
    )
    # Mean of 300 draws within 4 standard errors of zero, and far below the signal size.
    assert abs(values.mean()) < 4 * values.std(ddof=1) / np.sqrt(values.size)
    assert abs(values.mean()) < 0.001


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="NaN"):
        Ranking([1.0, np.nan])
    with pytest.raises(ValueError, match="at least two"):
        Ranking([1.0])
    with pytest.raises(ValueError, match="same length"):
        gains(Ranking(SCORES), T[:3], Y[:3])
    with pytest.raises(ValueError, match="both treated and control"):
        gains(Ranking(SCORES), np.ones(4), Y)
    with pytest.raises(ValueError, match="non-negative"):
        gains(Ranking(SCORES), T, Y, weights=[1, -1, 1, 1])
    with pytest.raises(ValueError, match="budget"):
        uplift_at_k(gains(Ranking(SCORES), T, Y), 0.0)
