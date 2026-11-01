"""Uplift evaluation on randomised data: Qini and uplift curves, Qini coefficient, AUUC, uplift@k.

Users are ranked by predicted uplift, highest first. For the top ``k`` users let ``N_t, N_c`` be
the treated and control counts, ``Y_t, Y_c`` their outcome sums and ``N`` the population size.
Curves are expressed per user of the whole population, against the targeted fraction ``k / N``:

- Qini curve ``Q = (Y_t - Y_c * N_t / N_c) / N``: incremental outcomes among the targeted
  treated users, the control arm rescaled to the treated arm's size (Radcliffe 2007);
- uplift curve ``U = (Y_t / N_t - Y_c / N_c) * k / N``: incremental outcomes if all ``k`` were
  treated, so ``U(1)`` is the average treatment effect.

The Qini coefficient is the area between the Qini curve and the straight line of random
targeting (zero in expectation for a random ranking); AUUC is the area under the uplift curve and
``auuc_gain`` its excess over random targeting, ``AUUC - U(1) / 2``. An arm that is empty among
the top ``k`` contributes zero there, as in scikit-uplift. Tied scores are evaluated only at the
end of each tie group (linear in between), so the result does not depend on row order.

Every statistic accepts optional non-negative ``weights`` (bootstrap multiplicities), which lets
a bootstrap reuse one sort per model. The unweighted sample must contain both arms. A weighted
replicate in which one arm has zero total weight is undefined: every statistic of it is NaN, and
the bootstrap drops it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.features import binary_array


class Ranking:
    """Users ordered by descending score (stable), with the end of every tie group."""

    def __init__(self, scores: ArrayLike) -> None:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("scores must be one-dimensional with at least two values")
        if not np.isfinite(values).all():
            raise ValueError("scores contain NaN or infinite values")
        self.size = values.size
        self.order: NDArray[np.intp] = np.argsort(-values, kind="stable")
        ranked = values[self.order]
        # Prefix-array positions after each tie group; position 0 is the empty selection.
        self.cuts: NDArray[np.intp] = np.concatenate(
            [[0], np.flatnonzero(np.diff(ranked)) + 1, [values.size]]
        )


@dataclass(frozen=True)
class Gains:
    """Prefix sums in ranked order; index ``i`` covers the first ``i`` ranked users.

    The curves are cached, so the Qini coefficient, AUUC and AUUC gain of one replicate share
    one pass over the tie-group ends.
    """

    weight: FloatArray
    n_t: FloatArray
    n_c: FloatArray
    y_t: FloatArray
    y_c: FloatArray
    cuts: NDArray[np.intp]

    @property
    def total(self) -> float:
        return float(self.weight[-1])

    @property
    def has_both_arms(self) -> bool:
        """Whether both arms carry weight; statistics of a one-arm replicate are NaN."""
        return bool(self.n_t[-1] > 0.0 and self.n_c[-1] > 0.0)

    @cached_property
    def fraction(self) -> FloatArray:
        return self.weight[self.cuts] / self.total

    @cached_property
    def qini(self) -> FloatArray:
        c = self.cuts
        return (self.y_t[c] - self.y_c[c] * _ratio(self.n_t[c], self.n_c[c])) / self.total

    @cached_property
    def uplift(self) -> FloatArray:
        c = self.cuts
        rate_gap = _ratio(self.y_t[c], self.n_t[c]) - _ratio(self.y_c[c], self.n_c[c])
        return rate_gap * self.fraction


def _prefix(values: FloatArray) -> FloatArray:
    return np.concatenate([[0.0], np.cumsum(values)])


class SortedSample:
    """Treatment and outcome of a sample, validated once and reordered once per ranking.

    A bootstrap evaluates the same rankings on many weight vectors; this keeps the validation
    and the reordering of treatment and outcome out of the per-replicate work.
    """

    def __init__(self, ranking: Ranking, treatment: ArrayLike, outcome: ArrayLike) -> None:
        t = binary_array(treatment, "treatment")
        y = np.asarray(outcome, dtype=np.float64)
        if not t.size == y.size == ranking.size:
            raise ValueError("scores, treatment and outcome must have the same length")
        if not np.isfinite(y).all():
            raise ValueError("outcome must be finite")
        if np.all(t == t[0]):
            raise ValueError("both treated and control units are needed")
        self.ranking = ranking
        self.t = t[ranking.order]
        self.y = y[ranking.order]
        self.c = 1.0 - self.t

    def gains(self, weights: ArrayLike | None = None) -> Gains:
        """Prefix sums per arm under ``weights`` (``None``: every row once)."""
        if weights is None:
            w = np.ones(self.ranking.size)
        else:
            raw = np.asarray(weights, dtype=np.float64)
            if raw.shape != (self.ranking.size,) or not (raw >= 0).all():
                raise ValueError("weights must be non-negative, one per row")
            w = raw[self.ranking.order]
        wt, wc = w * self.t, w * self.c
        return Gains(
            _prefix(w),
            _prefix(wt),
            _prefix(wc),
            _prefix(wt * self.y),
            _prefix(wc * self.y),
            self.ranking.cuts,
        )


def gains(
    ranking: Ranking, treatment: ArrayLike, outcome: ArrayLike, weights: ArrayLike | None = None
) -> Gains:
    """Prefix sums of counts and outcomes per arm, in the ranking's order.

    The unweighted sample must hold both arms; ``weights`` may still empty one (see module doc).
    """
    return SortedSample(ranking, treatment, outcome).gains(weights)


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def qini_curve(g: Gains) -> tuple[FloatArray, FloatArray]:
    """Targeted fraction and Qini value per population user at each tie-group end."""
    return g.fraction, g.qini


def uplift_curve(g: Gains) -> tuple[FloatArray, FloatArray]:
    """Targeted fraction and uplift-curve value per population user at each tie-group end."""
    return g.fraction, g.uplift


def _trapezoid(values: FloatArray, fraction: FloatArray) -> float:
    return float(np.sum(np.diff(fraction) * (values[1:] + values[:-1]) / 2.0))


def _area_over_random(fraction: FloatArray, values: FloatArray) -> float:
    return _trapezoid(values - fraction * values[-1], fraction)


def qini_coefficient(g: Gains) -> float:
    """Area between the Qini curve and random targeting; 0 in expectation for random scores."""
    if not g.has_both_arms:
        return float("nan")
    return _area_over_random(g.fraction, g.qini)


def auuc(g: Gains) -> float:
    """Area under the uplift curve (per population user, over the targeted fraction)."""
    if not g.has_both_arms:
        return float("nan")
    return _trapezoid(g.uplift, g.fraction)


def auuc_gain(g: Gains) -> float:
    """AUUC minus the area under random targeting, ``U(1) / 2``."""
    if not g.has_both_arms:
        return float("nan")
    return _area_over_random(g.fraction, g.uplift)


def top_count(g: Gains, budget: float) -> int:
    """Number of ranked users whose cumulative weight before them is below ``budget * N``."""
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    return int(np.searchsorted(g.weight[:-1], budget * g.total, side="left"))


def uplift_at_k(g: Gains, budget: float) -> float:
    """Treated minus control outcome rate among the top ``budget`` share; NaN if an arm is empty."""
    j = top_count(g, budget)
    if g.n_t[j] == 0 or g.n_c[j] == 0:
        return float("nan")
    return float(g.y_t[j] / g.n_t[j] - g.y_c[j] / g.n_c[j])
