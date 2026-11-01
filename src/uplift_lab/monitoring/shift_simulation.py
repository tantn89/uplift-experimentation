"""Seeded traffic scenarios drawn from held-out data, with and without a distribution shift.

Each scenario resamples rows of the held-out set with replacement, with weights:

- ``no_shift``: uniform. The same distribution as the reference, so nothing should flag. It is
  a sanity check, not an independent test: it resamples the very held-out rows that set the
  decision threshold when the drift report fits its own model, so its decision rate is close
  to the budget by construction. (An exported model brings the thresholds of its own test
  set, which may overlap these rows as well.)
- ``covariate_shift``: an exponential tilt towards high values of the chosen features,
  ``w = exp(strength * sum_f u_f)`` with ``u_f`` the percentile rank of feature ``f``. The
  marginals of those features move. Correlated features move with them, and that is real
  drift too.
- ``treatment_mix_shift``: reweights the arms to ``treated_share``, as if the campaign changed
  its hold-out ratio. With randomised assignment the features should not move, so only a
  feature-dependent assignment shows up here.
- ``outcome_prior_shift``: reweights visitors to ``outcome_rate``, a label (prior) shift such
  as a seasonal surge. Visitors differ from non-visitors, so features and scores move even
  though the model is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift_lab.experiments.estimators import FloatArray

SCENARIOS = ("no_shift", "covariate_shift", "treatment_mix_shift", "outcome_prior_shift")


@dataclass(frozen=True)
class ShiftConfig:
    covariate_features: tuple[str, ...] = ("f0", "f6")
    strength: float = 2.0
    treated_share: float = 0.5
    outcome_rate_multiplier: float = 3.0
    outcome: str = "visit"

    def __post_init__(self) -> None:
        if not self.covariate_features or self.strength <= 0.0:
            raise ValueError("covariate shift needs features and a positive strength")
        if not 0.0 < self.treated_share < 1.0 or self.outcome_rate_multiplier <= 0.0:
            raise ValueError("treated_share must be in (0, 1) and the multiplier positive")

    def describe(self, scenario: str) -> str:
        return {
            "no_shift": "uniform resample of held-out traffic",
            "covariate_shift": (
                f"tilt towards high {', '.join(self.covariate_features)} "
                f"(strength {self.strength:g})"
            ),
            "treatment_mix_shift": f"arms reweighted to {self.treated_share:.0%} treated",
            "outcome_prior_shift": (
                f"{self.outcome} rate x{self.outcome_rate_multiplier:g} (label shift)"
            ),
        }[scenario]


def _reweight(flag: FloatArray, target: float) -> FloatArray:
    share = float(flag.mean())
    if not 0.0 < share < 1.0:
        raise ValueError("both groups are needed to reweight")
    return np.where(flag == 1.0, target / share, (1.0 - target) / (1.0 - share))


def scenario_weights(frame: pd.DataFrame, scenario: str, config: ShiftConfig) -> FloatArray:
    if scenario == "no_shift":
        return np.ones(len(frame))
    if scenario == "covariate_shift":
        ranks = sum(
            frame[name].rank(pct=True).to_numpy(dtype=np.float64)
            for name in config.covariate_features
        )
        return np.exp(config.strength * np.asarray(ranks, dtype=np.float64))
    if scenario == "treatment_mix_shift":
        flag = frame["treatment"].to_numpy(dtype=np.float64)
        return _reweight(flag, config.treated_share)
    if scenario == "outcome_prior_shift":
        flag = frame[config.outcome].to_numpy(dtype=np.float64)
        return _reweight(flag, min(0.5, float(flag.mean()) * config.outcome_rate_multiplier))
    raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")


def simulate_traffic(
    frame: pd.DataFrame, scenario: str, n: int, *, seed: int, config: ShiftConfig
) -> pd.DataFrame:
    """``n`` rows of ``frame`` drawn with replacement under ``scenario``'s weights."""
    if n < 1:
        raise ValueError("n must be positive")
    weights = scenario_weights(frame, scenario, config)
    rows = np.random.default_rng(seed).choice(len(frame), size=n, p=weights / weights.sum())
    return frame.iloc[rows].reset_index(drop=True)


def all_scenarios(
    frame: pd.DataFrame, n: int, *, seed: int, config: ShiftConfig, names: Sequence[str] = SCENARIOS
) -> dict[str, pd.DataFrame]:
    """Every scenario with its own seed offset, so adding one does not change the others."""
    return {
        name: simulate_traffic(frame, name, n, seed=seed + SCENARIOS.index(name), config=config)
        for name in names
    }
