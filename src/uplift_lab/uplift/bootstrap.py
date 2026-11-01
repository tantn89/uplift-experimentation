"""Seeded non-parametric bootstrap expressed as resampling weights.

Each replicate draws ``n`` rows with replacement and turns the draw into per-row multiplicities.
Statistics that accept weights can then reuse precomputed sorts, and all statistics in one
replicate see the same resample, so differences between models are paired.
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


def bootstrap_weights(n: int, n_boot: int, seed: int) -> Iterator[FloatArray]:
    """Yield ``n_boot`` multiplicity vectors of length ``n`` from one seeded generator."""
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be positive")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


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
    point = dict(statistic(None))
    draws = {name: np.empty(n_boot) for name in point}
    for b, weights in enumerate(bootstrap_weights(n, n_boot, seed)):
        replicate = statistic(weights)
        for name in point:
            draws[name][b] = replicate[name]
    intervals: dict[str, Interval] = {}
    for name, estimate in point.items():
        finite = draws[name][np.isfinite(draws[name])]
        if finite.size == 0:
            low = high = float("nan")
        else:
            low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        intervals[name] = Interval(float(estimate), float(low), float(high))
    return intervals
