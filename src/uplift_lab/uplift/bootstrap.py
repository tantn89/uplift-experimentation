"""Seeded non-parametric bootstrap expressed as resampling weights.

Each replicate draws ``n`` rows with replacement and turns the draw into per-row multiplicities.
Statistics that accept weights can then reuse precomputed sorts, and all statistics in one
replicate see the same resample, so differences between models are paired.

What the intervals cover: the rows being resampled (a test set) and nothing else. A caller that
scores fixed, already fitted models gets intervals for test-set sampling noise only; the
variability from refitting the models on another training sample is not in them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from uplift_lab.experiments.estimators import FloatArray


@dataclass(frozen=True)
class Interval:
    """Point estimate on the full sample with a percentile bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapDraws:
    """Full-sample estimates and the replicate values of every named statistic."""

    point: dict[str, float]
    draws: dict[str, FloatArray]

    def interval(self, name: str, alpha: float) -> Interval:
        return percentile_interval(self.point[name], self.draws[name], alpha)


def bootstrap_weights(n: int, n_boot: int, seed: int) -> Iterator[FloatArray]:
    """Yield ``n_boot`` multiplicity vectors of length ``n`` from one seeded generator."""
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be positive")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def bootstrap_draws(
    statistic: Callable[[FloatArray | None], Mapping[str, float]],
    n: int,
    *,
    n_boot: int,
    seed: int,
) -> BootstrapDraws:
    """``statistic(None)`` on the full sample and ``statistic(weights)`` on every replicate.

    The full sample is evaluated first, so a statistic that validates its input fails before
    any replicate runs. A replicate value may be NaN (an arm empty in that resample).
    """
    point = dict(statistic(None))
    draws = {name: np.empty(n_boot) for name in point}
    for b, weights in enumerate(bootstrap_weights(n, n_boot, seed)):
        replicate = statistic(weights)
        for name in point:
            draws[name][b] = replicate[name]
    return BootstrapDraws(point, draws)


def percentile_interval(estimate: float, draws: FloatArray, alpha: float) -> Interval:
    """Percentile interval of the finite ``draws``; NaN bounds if none is finite."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return Interval(float(estimate), float("nan"), float("nan"))
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(estimate), float(low), float(high))


def bootstrap_intervals(
    statistic: Callable[[FloatArray | None], Mapping[str, float]],
    n: int,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Interval]:
    """Percentile intervals for every named value ``statistic`` returns.

    ``statistic(None)`` is the full-sample estimate; ``statistic(weights)`` a replicate. NaN
    replicates (an arm empty in a resample) are dropped per statistic.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    result = bootstrap_draws(statistic, n, n_boot=n_boot, seed=seed)
    return {name: result.interval(name, alpha) for name in result.point}


def bootstrap_p_value(draws: FloatArray) -> float:
    """Two-sided bootstrap p-value for "the statistic is 0", from its replicate values.

    ``2 * min(P(d <= 0), P(d >= 0))`` over the finite replicates, each share counted as
    ``(k + 1) / (B + 1)`` so that a finite number of draws never reports exactly 0. NaN when no
    replicate is finite.
    """
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan")
    denominator = finite.size + 1.0
    below = (np.count_nonzero(finite <= 0.0) + 1.0) / denominator
    above = (np.count_nonzero(finite >= 0.0) + 1.0) / denominator
    return float(min(1.0, 2.0 * min(below, above)))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error control over the given tests).

    Sorted ascending, ``p_(i)`` becomes ``max_{j <= i} min(1, (m - j + 1) p_(j))``. A test is
    rejected at level ``alpha`` when its adjusted p-value is ``<= alpha``. NaN inputs stay NaN
    and do not count towards ``m``.
    """
    valid = sorted((p, name) for name, p in p_values.items() if np.isfinite(p))
    m = len(valid)
    adjusted = {name: float("nan") for name in p_values}
    running = 0.0
    for i, (p, name) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
