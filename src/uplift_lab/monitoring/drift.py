"""In-house drift detection: PSI on reference quantile bins plus a two-sample KS test.

**PSI** (population stability index) compares the share of current rows in each bin with the
reference share: ``sum((c - r) * ln(c / r))``. Bins are the reference deciles (unique cut
points only, so a feature with a large point mass gets fewer bins). They are closed on the right:
``(-inf, e1], (e1, e2], ..., (em, inf)``. Empty shares are floored at ``EPSILON`` so the log stays
finite. Common reading: below 0.1 stable, 0.1-0.2 moderate shift (warn), 0.2 or more major
shift (alert). PSI hardly depends on sample size, which makes it the alerting statistic.

**KS** is reported as a second opinion. At serving volumes its p-value flags shifts far too
small to matter, so it never raises the status on its own. Its rejection uses a Bonferroni
level ``ks_alpha / number of columns``.

Prediction drift is the same pair of statistics on the model's uplift scores (column
``uplift``).
"""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        if not 0.0 < self.psi_warn < self.psi_alert:
            raise ValueError("need 0 < psi_warn < psi_alert")
        if not 0.0 < self.ks_alpha < 1.0:
            raise ValueError("ks_alpha must be in (0, 1)")

    def status(self, psi: float) -> str:
        return "alert" if psi >= self.psi_alert else "warn" if psi >= self.psi_warn else "ok"

    def to_dict(self) -> dict[str, float]:
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


def drift_summary(
    profile: ReferenceProfile, current: pd.DataFrame, thresholds: DriftThresholds
) -> dict[str, Any]:
    """Per-column drift of ``current`` against ``profile``; prediction drift if ``uplift`` given."""
    columns = [c for c in profile.columns if c in current.columns]
    missing = sorted(set(profile.columns) - set(current.columns) - {PREDICTION})
    if missing or not columns:
        raise ValueError(f"current traffic is missing columns: {missing or profile.columns}")
    level = thresholds.ks_alpha / len(columns)
    results = {c: column_drift(profile, c, current[c], level, thresholds) for c in columns}
    prediction = results.pop(PREDICTION, None)
    statuses = [r["status"] for r in results.values()]
    if prediction is not None:
        statuses.append(prediction["status"])
    return {
        "n": len(current),
        "status": max(statuses, key=STATUSES.index),
        "flagged_features": [name for name, r in results.items() if r["status"] != "ok"],
        "ks_rejected_features": [name for name, r in results.items() if r["ks_reject"]],
        "features": results,
        "prediction": prediction,
    }


def worst_status(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    return max((s["status"] for s in summaries.values()), key=STATUSES.index, default="ok")
