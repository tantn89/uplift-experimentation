"""Evaluate several uplift scorings of one randomised test set, with paired bootstrap CIs.

The payload holds, per model: Qini coefficient, AUUC, AUUC gain over random, uplift@k and the
policy value table per budget, each as ``{estimate, ci_low, ci_high}``; the average treatment
effect; Qini and uplift curves on a fixed grid for plotting; and observed uplift by decile of
predicted uplift. A model "beats random" when its Qini coefficient's interval excludes zero
from above.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.bootstrap import bootstrap_intervals
from uplift_lab.uplift.features import binary_array
from uplift_lab.uplift.metrics import (
    Gains,
    Ranking,
    auuc,
    auuc_gain,
    gains,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_curve,
)
from uplift_lab.uplift.policy import PolicyValue, average_treatment_effect, policy_value

DEFAULT_BUDGETS = (0.1, 0.2, 0.3, 0.5)
RANDOM_SCORER = "random"
CURVE_GRID = np.linspace(0.0, 1.0, 101)
_SEP = "|"
_POLICY_FIELDS = tuple(f.name for f in fields(PolicyValue) if f.name != "budget")


def _point_metrics(g: Gains, budgets: Sequence[float]) -> dict[str, float]:
    values = {"qini_coefficient": qini_coefficient(g), "auuc": auuc(g), "auuc_gain": auuc_gain(g)}
    for budget in budgets:
        values[f"uplift_at_k{_SEP}{budget:g}"] = uplift_at_k(g, budget)
        policy = policy_value(g, budget).to_dict()
        for name in _POLICY_FIELDS:
            values[f"policy{_SEP}{budget:g}{_SEP}{name}"] = policy[name]
    return values


def _nest(flat: Mapping[str, dict[str, float]]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        *parents, leaf = key.split(_SEP)
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return nested


def decile_table(
    scores: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, n_bins: int = 10
) -> list[dict[str, float]]:
    """Observed treated-minus-control rate and mean prediction per bin of predicted uplift."""
    s = np.asarray(scores, dtype=np.float64)
    t = binary_array(treatment, "treatment").astype(bool)
    y = np.asarray(outcome, dtype=np.float64)
    rows = []
    for rank, members in enumerate(np.array_split(Ranking(s).order, n_bins), start=1):
        treated, control = members[t[members]], members[~t[members]]
        observed = (
            float(y[treated].mean() - y[control].mean())
            if treated.size and control.size
            else float("nan")
        )
        rows.append(
            {
                "decile": rank,
                "n_treated": int(treated.size),
                "n_control": int(control.size),
                "mean_predicted": float(s[members].mean()),
                "observed_uplift": observed,
            }
        )
    return rows


def evaluate_scores(
    scores: Mapping[str, ArrayLike],
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    n_bootstrap: int = 200,
    seed: int = 20261101,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Metrics, paired bootstrap intervals, curves and deciles for every scoring in ``scores``."""
    if not scores:
        raise ValueError("at least one scoring is required")
    t = binary_array(treatment, "treatment")
    y = binary_array(outcome, "outcome")
    rankings = {name: Ranking(values) for name, values in scores.items()}
    if any(r.size != t.size for r in rankings.values()) or y.size != t.size:
        raise ValueError("every scoring, treatment and outcome must have the same length")

    def statistic(weights: FloatArray | None) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, ranking in rankings.items():
            g = gains(ranking, t, y, weights)
            values["ate"] = average_treatment_effect(g)
            for key, value in _point_metrics(g, budgets).items():
                values[f"{name}{_SEP}{key}"] = value
        return values

    intervals = bootstrap_intervals(statistic, t.size, n_boot=n_bootstrap, seed=seed, alpha=alpha)
    flat = {name: interval.to_dict() for name, interval in intervals.items()}
    payload: dict[str, Any] = {
        "n_test": int(t.size),
        "treated_share": float(t.mean()),
        "budgets": list(budgets),
        "bootstrap": {"n": n_bootstrap, "seed": seed, "alpha": alpha, "method": "percentile"},
        "ate": flat.pop("ate"),
        "models": _nest(flat),
        "curves": {},
        "deciles": {},
    }
    for name, ranking in rankings.items():
        g = gains(ranking, t, y)
        qini_x, qini_y = qini_curve(g)
        uplift_x, uplift_y = uplift_curve(g)
        payload["curves"][name] = {
            "fraction": CURVE_GRID.tolist(),
            "qini": np.interp(CURVE_GRID, qini_x, qini_y).tolist(),
            "uplift": np.interp(CURVE_GRID, uplift_x, uplift_y).tolist(),
        }
        payload["deciles"][name] = decile_table(scores[name], t, y)
    candidates = [name for name in scores if name != RANDOM_SCORER] or list(scores)
    best = max(candidates, key=lambda n: payload["models"][n]["qini_coefficient"]["estimate"])
    payload["best_model"] = best
    payload["best_beats_random"] = payload["models"][best]["qini_coefficient"]["ci_low"] > 0.0
    return payload
