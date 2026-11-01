"""Seeded simulation of peeking: a naive Welch t-test checked at every look versus mSPRT.

Each simulated experiment has two arms of Bernoulli conversions (``base_rate`` in control) and is
inspected after every batch of ``batch_per_arm`` new units per arm, up to ``looks`` looks. Batches
are drawn as binomial counts, so a run costs O(n_sims * looks) whatever the sample size.

Three rules are scored on the same data:
- ``naive_peeking``: reject at the first look where the fixed-horizon Welch test has p < alpha;
- ``msprt``: reject at the first look where the mixture likelihood ratio reaches 1 / alpha;
- ``fixed_horizon``: one Welch test at the final look only (the valid reference).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from uplift_lab.experiments.msprt import msprt_reject
from uplift_lab.experiments.power import minimum_detectable_effect

RULES = ("naive_peeking", "msprt", "fixed_horizon")


@dataclass(frozen=True)
class PeekingConfig:
    n_sims: int = 10_000
    looks: int = 20
    batch_per_arm: int = 500
    base_rate: float = 0.10
    alpha: float = 0.05
    power: float = 0.8
    seed: int = 20261101

    def __post_init__(self) -> None:
        if self.n_sims < 1 or self.looks < 1 or self.batch_per_arm < 2:
            raise ValueError("n_sims and looks must be >= 1 and batch_per_arm >= 2")
        if not 0.0 < self.base_rate < 1.0 or not 0.0 < self.alpha < 1.0:
            raise ValueError("base_rate and alpha must be in (0, 1)")

    @property
    def horizon_per_arm(self) -> int:
        return self.looks * self.batch_per_arm

    def design_effect(self) -> float:
        """Absolute lift the fixed-horizon test detects with ``power``; also the mSPRT ``tau``."""
        sd = float(np.sqrt(self.base_rate * (1.0 - self.base_rate)))
        return minimum_detectable_effect(
            self.horizon_per_arm, sd, alpha=self.alpha, power=self.power
        )

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "horizon_per_arm": self.horizon_per_arm}


@dataclass(frozen=True)
class PeekingOutcome:
    """First rejecting look per simulation for each rule (0-based; -1 if it never rejects)."""

    effect: float
    tau: float
    looks_n: NDArray[np.int64]
    first_rejection: dict[str, NDArray[np.int64]]

    def rejection_rate(self, rule: str) -> float:
        return float((self.first_rejection[rule] >= 0).mean())

    def cumulative_rejection(self, rule: str) -> NDArray[np.float64]:
        """Share of simulations that have rejected by each look."""
        first = self.first_rejection[rule]
        hits = np.bincount(first[first >= 0], minlength=self.looks_n.size)
        return np.cumsum(hits) / first.size

    def stopping_n(self, rule: str) -> NDArray[np.int64]:
        """Units per arm at stopping; runs that never reject stop at the horizon."""
        first = self.first_rejection[rule]
        return np.where(first >= 0, self.looks_n[np.maximum(first, 0)], self.looks_n[-1])


def _first_true(reject: NDArray[np.bool_]) -> NDArray[np.int64]:
    index = reject.argmax(axis=1).astype(np.int64)
    return np.where(reject.any(axis=1), index, -1)


def simulate_peeking(
    config: PeekingConfig, effect: float, *, tau: float | None = None
) -> PeekingOutcome:
    """Run ``config.n_sims`` experiments with true absolute lift ``effect``."""
    treated_rate = config.base_rate + effect
    if not 0.0 < treated_rate < 1.0:
        raise ValueError(f"base_rate + effect must be in (0, 1), got {treated_rate}")
    mixing_tau = config.design_effect() if tau is None else tau
    rng = np.random.default_rng(config.seed)
    shape = (config.n_sims, config.looks)
    successes_t = rng.binomial(config.batch_per_arm, treated_rate, size=shape).cumsum(axis=1)
    successes_c = rng.binomial(config.batch_per_arm, config.base_rate, size=shape).cumsum(axis=1)
    looks_n = config.batch_per_arm * np.arange(1, config.looks + 1, dtype=np.int64)

    p_t, p_c = successes_t / looks_n, successes_c / looks_n
    # Per-arm variance of the mean with the unbiased (ddof=1) Bernoulli sample variance.
    var_t = p_t * (1.0 - p_t) / (looks_n - 1)
    var_c = p_c * (1.0 - p_c) / (looks_n - 1)
    variance = var_t + var_c
    estimate = p_t - p_c

    with np.errstate(divide="ignore", invalid="ignore"):
        df = variance**2 / ((var_t**2 + var_c**2) / (looks_n - 1))
        p_welch = 2.0 * stats.t.sf(np.abs(estimate) / np.sqrt(variance), df)
    welch_reject = np.asarray(np.nan_to_num(p_welch, nan=1.0) < config.alpha, dtype=np.bool_)

    fixed = np.full(shape, False)
    fixed[:, -1] = welch_reject[:, -1]
    first = {
        "naive_peeking": _first_true(welch_reject),
        "msprt": _first_true(msprt_reject(estimate, variance, mixing_tau, alpha=config.alpha)),
        "fixed_horizon": _first_true(fixed),
    }
    return PeekingOutcome(effect=effect, tau=mixing_tau, looks_n=looks_n, first_rejection=first)


def summarise(outcome: PeekingOutcome) -> dict[str, dict[str, float]]:
    """Rejection rate and stopping-time summary (units per arm) for each rule."""
    summary: dict[str, dict[str, float]] = {}
    for rule in RULES:
        stopped = outcome.first_rejection[rule] >= 0
        n_stop = outcome.stopping_n(rule)
        rate = float(stopped.mean())
        p25, p50, p75 = (
            np.percentile(n_stop[stopped], [25, 50, 75]) if stopped.any() else [np.nan] * 3
        )
        summary[rule] = {
            "rejection_rate": rate,
            "rejection_rate_se": float(np.sqrt(rate * (1.0 - rate) / stopped.size)),
            "mean_n_per_arm": float(n_stop.mean()),
            "p25_stop_n_if_rejected": float(p25),
            "median_stop_n_if_rejected": float(p50),
            "p75_stop_n_if_rejected": float(p75),
        }
    return summary
