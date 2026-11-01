"""Seeded CUPED contrast: how much variance a strong pre-period covariate removes.

On Hillstrom the pre-period covariates explain little of the two-week outcomes (visit variance
falls by 2.6%), so CUPED barely helps there. This simulation shows the other end of the scale.
Each replicate draws ``n_per_arm`` units per arm with a standardised pre-period metric ``x`` and
an outcome ``y = 1 + rho x + sqrt(1 - rho^2) e`` (plus ``effect`` in the treated arm), so the
covariate explains ``R^2 = rho^2`` of the outcome variance. Per ``rho`` the result holds:

- ``empirical_variance_ratio``: Var(CUPED estimate) / Var(raw estimate) across replicates; the
  theory is ``1 - rho^2``;
- ``mean_reported_variance_reduction``: the per-experiment estimate ``1 - se_adj^2 / se_raw^2``;
- ``*_mean_effect`` and ``*_ci_coverage`` for both estimators (CUPED stays unbiased and keeps its
  nominal coverage).

At the same power CUPED needs a share ``1 - variance ratio`` fewer units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from uplift_lab.experiments.cuped import cuped_difference

RHOS = (0.0, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class CupedSimConfig:
    n_per_arm: int = 2_000
    n_reps: int = 1_000
    effect: float = 0.05
    rhos: tuple[float, ...] = RHOS
    alpha: float = 0.05
    seed: int = 20261101

    def __post_init__(self) -> None:
        if self.n_per_arm < 10 or self.n_reps < 10:
            raise ValueError("n_per_arm and n_reps must be >= 10")
        if not self.rhos or not all(0.0 <= rho < 1.0 for rho in self.rhos):
            raise ValueError("rhos must be non-empty and in [0, 1)")


def _contrast(rng: np.random.Generator, config: CupedSimConfig, rho: float) -> dict[str, float]:
    n = config.n_per_arm
    raw, adjusted, reported, raw_cover, adjusted_cover = [], [], [], [], []
    for _ in range(config.n_reps):
        x = rng.normal(size=2 * n)
        y = 1.0 + rho * x + np.sqrt(1.0 - rho**2) * rng.normal(size=2 * n)
        y[:n] += config.effect
        result = cuped_difference(y[:n], y[n:], x[:n], x[n:], alpha=config.alpha)
        raw.append(result.raw.effect)
        adjusted.append(result.adjusted.effect)
        reported.append(result.variance_reduction)
        raw_cover.append(result.raw.ci_low <= config.effect <= result.raw.ci_high)
        adjusted_cover.append(result.adjusted.ci_low <= config.effect <= result.adjusted.ci_high)
    ratio = float(np.var(adjusted) / np.var(raw))
    return {
        "rho": rho,
        "r_squared": rho**2,
        "theoretical_variance_ratio": 1.0 - rho**2,
        "empirical_variance_ratio": ratio,
        "mean_reported_variance_reduction": float(np.mean(reported)),
        "raw_mean_effect": float(np.mean(raw)),
        "cuped_mean_effect": float(np.mean(adjusted)),
        "raw_ci_coverage": float(np.mean(raw_cover)),
        "cuped_ci_coverage": float(np.mean(adjusted_cover)),
    }


def simulate_cuped_contrast(config: CupedSimConfig | None = None) -> dict[str, Any]:
    """One seeded contrast per ``rho``, all from a single generator, in ``config.rhos`` order."""
    config = config or CupedSimConfig()
    rng = np.random.default_rng(config.seed)
    return {
        "config": asdict(config),
        "by_rho": [_contrast(rng, config, rho) for rho in config.rhos],
    }
