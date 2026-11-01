"""Sample-ratio-mismatch (SRM) check: are the arm sizes consistent with the planned split?

A chi-square goodness-of-fit test of the observed arm counts against the design ratios. A tiny
p-value means assignment or logging is broken and the effect estimates should not be trusted.
The conventional alarm threshold is p < 0.001, strict because the check runs on every experiment
and a false alarm halts analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

DEFAULT_SRM_THRESHOLD = 1e-3


@dataclass(frozen=True)
class SrmResult:
    statistic: float
    p_value: float
    threshold: float
    observed: dict[str, int]
    expected: dict[str, float]

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.threshold


def srm_test(
    counts: Mapping[str, int],
    ratios: Sequence[float] | None = None,
    *,
    threshold: float = DEFAULT_SRM_THRESHOLD,
) -> SrmResult:
    """Chi-square test of ``counts`` against ``ratios`` (equal split when omitted)."""
    if len(counts) < 2:
        raise ValueError("an SRM test needs at least two arms")
    observed = np.array([int(count) for count in counts.values()], dtype=np.float64)
    if (observed < 0).any() or observed.sum() == 0:
        raise ValueError("arm counts must be non-negative with a positive total")
    weights = np.ones(len(counts)) if ratios is None else np.asarray(ratios, dtype=np.float64)
    if weights.shape != observed.shape or (weights <= 0).any():
        raise ValueError("ratios must be positive and match the number of arms")
    expected = observed.sum() * weights / weights.sum()
    statistic, p_value = stats.chisquare(observed, expected)
    return SrmResult(
        statistic=float(statistic),
        p_value=float(p_value),
        threshold=threshold,
        observed={arm: int(count) for arm, count in counts.items()},
        expected=dict(zip(counts, (float(value) for value in expected), strict=True)),
    )
