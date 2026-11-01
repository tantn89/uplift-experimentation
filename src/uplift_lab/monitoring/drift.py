"""In-house drift detection: PSI on reference quantile bins plus a two-sample KS test.

**PSI** (population stability index) compares the share of current rows in each bin with the
reference share: ``sum((c - r) * ln(c / r))``. Bins are the reference deciles (unique cut
points only, so a feature with a large point mass gets fewer bins). They are closed on the right:
``(-inf, e1], (e1, e2], ..., (em, inf)``. Empty shares are floored at ``EPSILON`` so the log stays
finite. Common reading: below 0.1 stable, 0.1-0.2 moderate shift (warn), 0.2 or more major
shift (alert). PSI hardly depends on sample size, which makes it the alerting statistic.

**KS** is reported as a second opinion. At serving volumes its p-value flags shifts far too
small to matter, so it never raises the status on its own. Its rejection uses a Bonferroni
level ``ks_alpha / m``, where ``m`` counts every tested column: the features plus the uplift
score when it is present (13 for Criteo). The summary records ``m`` and the level.

Prediction drift is the same pair of statistics on the model's uplift scores (column
``uplift``).

**Decision rate** is the operational check: the share of traffic at or above a named budget's
score threshold, divided by that budget. The threshold was set so that a ``budget`` share of
held-out users clears it, so the ratio is 1 without drift. A ratio outside ``rate_warn`` warns
and outside ``rate_alert`` alerts (defaults [0.75, 1.33] and [0.5, 1.5]): at 1.5 the campaign
would overspend its budget by half. This catches a label or prior shift that moves the scores
of many users a little, which PSI on coarse deciles of point-massed features can miss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from uplift_lab.experiments.estimators import FloatArray

EPSILON = 1e-4
STATUSES = ("ok", "warn", "alert")
PREDICTION = "uplift"


@dataclass(frozen=True)
class DriftThresholds:
    psi_warn: float = 0.1
    psi_alert: float = 0.2
    ks_alpha: float = 0.01
    rate_warn: tuple[float, float] = (0.75, 1.33)
    rate_alert: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self) -> None:
        if not 0.0 < self.psi_warn < self.psi_alert:
            raise ValueError("need 0 < psi_warn < psi_alert")
        if not 0.0 < self.ks_alpha < 1.0:
            raise ValueError("ks_alpha must be in (0, 1)")
        (alert_low, alert_high), (warn_low, warn_high) = self.rate_alert, self.rate_warn
        if not 0.0 < alert_low <= warn_low < 1.0 < warn_high <= alert_high:
            raise ValueError(
                "decision-rate bounds need 0 < alert low <= warn low < 1 < warn high <= alert high"
            )

    def status(self, psi: float) -> str:
        return "alert" if psi >= self.psi_alert else "warn" if psi >= self.psi_warn else "ok"

    def rate_status(self, ratio: float) -> str:
        """Status of a decision-rate ratio (treat rate / budget); the bounds themselves are in."""
        (alert_low, alert_high), (warn_low, warn_high) = self.rate_alert, self.rate_warn
        if not alert_low <= ratio <= alert_high:
            return "alert"
        return "ok" if warn_low <= ratio <= warn_high else "warn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty array of finite values")
    return array


def quantile_edges(reference: ArrayLike, n_bins: int = 10) -> FloatArray:
    """Unique interior cut points at reference quantiles ``1/n_bins, ..., (n_bins-1)/n_bins``."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    values = _finite(reference, "reference")
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))


def bin_shares(values: ArrayLike, edges: FloatArray) -> FloatArray:
    """Share of ``values`` in each right-closed bin defined by ``edges``."""
    array = _finite(values, "values")
    counts = np.bincount(np.searchsorted(edges, array, side="left"), minlength=edges.size + 1)
    return counts / array.size


def psi(reference_shares: ArrayLike, current_shares: ArrayLike) -> float:
    """Population stability index of two share vectors over the same bins."""
    r = np.maximum(np.asarray(reference_shares, dtype=np.float64), EPSILON)
    c = np.maximum(np.asarray(current_shares, dtype=np.float64), EPSILON)
    if r.shape != c.shape:
        raise ValueError("share vectors must have the same bins")
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class ReferenceProfile:
    """Bins, shares and raw values of the reference (training) data, per column."""

    edges: dict[str, FloatArray]
    shares: dict[str, FloatArray]
    values: dict[str, FloatArray]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str], n_bins: int = 10) -> ReferenceProfile:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"reference is missing columns: {missing}")
        values = {name: _finite(frame[name], name) for name in columns}
        edges = {name: quantile_edges(v, n_bins) for name, v in values.items()}
        shares = {name: bin_shares(values[name], edges[name]) for name in columns}
        return cls(edges, shares, values)

    @property
    def columns(self) -> list[str]:
        return list(self.edges)


def column_drift(
    profile: ReferenceProfile, name: str, current: ArrayLike, ks_level: float, t: DriftThresholds
) -> dict[str, Any]:
    values = _finite(current, name)
    value = psi(profile.shares[name], bin_shares(values, profile.edges[name]))
    ks = stats.ks_2samp(profile.values[name], values, method="asymp")
    return {
        "psi": value,
        "status": t.status(value),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "ks_reject": bool(ks.pvalue < ks_level),
        "n_bins": int(profile.edges[name].size + 1),
    }


def decision_rate_drift(
    scores: ArrayLike, budget: float, threshold: float, t: DriftThresholds
) -> dict[str, Any]:
    """Share of ``scores`` at or above a budget's ``threshold``, against that ``budget``."""
    if not 0.0 < budget <= 1.0 or not math.isfinite(threshold):
        raise ValueError("need a budget in (0, 1] and a finite threshold")
    rate = float(np.mean(_finite(scores, PREDICTION) >= threshold))
    ratio = rate / budget
    return {
        "budget": budget,
        "threshold": threshold,
        "treat_rate": rate,
        "ratio": ratio,
        "status": t.rate_status(ratio),
    }


def drift_summary(
    profile: ReferenceProfile,
    current: pd.DataFrame,
    thresholds: DriftThresholds,
    decision: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Per-column drift of ``current`` against ``profile``; prediction drift if ``uplift`` given.

    ``decision`` is ``(budget, threshold)`` of a named budget rule. It adds the decision-rate
    check on the ``uplift`` column, and its status counts towards the overall one.
    """
    columns = [c for c in profile.columns if c in current.columns]
    missing = sorted(set(profile.columns) - set(current.columns) - {PREDICTION})
    if missing or not columns:
        raise ValueError(f"current traffic is missing columns: {missing or profile.columns}")
    if decision is not None and PREDICTION not in current.columns:
        raise ValueError(f"the decision-rate check needs the {PREDICTION!r} column")
    level = thresholds.ks_alpha / len(columns)
    results = {c: column_drift(profile, c, current[c], level, thresholds) for c in columns}
    prediction = results.pop(PREDICTION, None)
    rate = (
        None
        if decision is None
        else decision_rate_drift(current[PREDICTION], *decision, thresholds)
    )
    statuses = [r["status"] for r in [*results.values(), prediction, rate] if r is not None]
    return {
        "n": len(current),
        "status": max(statuses, key=STATUSES.index),
        "flagged_features": [name for name, r in results.items() if r["status"] != "ok"],
        "ks_tests": len(columns),
        "ks_level": level,
        "ks_rejected_features": [name for name, r in results.items() if r["ks_reject"]],
        "features": results,
        "prediction": prediction,
        "decision_rate": rate,
    }


def worst_status(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    return max((s["status"] for s in summaries.values()), key=STATUSES.index, default="ok")
