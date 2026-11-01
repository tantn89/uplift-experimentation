"""The Hillstrom A/B analysis as plain data: SRM, Welch effects, CUPED and power.

CUPED covariates are all measured before the campaign, so the e-mail cannot have moved them:
- ``history``: dollars spent in the past year, the pre-period analogue of the spend outcome and
  the strongest single predictor of visits and purchases;
- ``recency`` (months since last purchase), ``mens`` / ``womens`` (bought that line in the past
  year), ``newbie`` (new customer in the past year) and ``channel`` (purchase channels used).
``zip_code`` is pre-treatment too but adds three noisy dummies for almost no variance explained,
so it is left out. Two sets are reported: ``history`` alone (classic one-covariate CUPED) and the
full ``pre_period`` set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from uplift_lab.data.hillstrom import ARMS, CONTROL_ARM
from uplift_lab.experiments.cuped import cuped_difference, cuped_theta, regression_adjusted_effect
from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.experiments.power import minimum_detectable_effect, power_at, required_n
from uplift_lab.experiments.srm import srm_test

OUTCOMES = ("visit", "conversion", "spend")
TREATED_ARMS = tuple(arm for arm in ARMS if arm != CONTROL_ARM)
COVARIATE_SETS: dict[str, tuple[str, ...]] = {
    "history": ("history",),
    "pre_period": ("history", "recency", "mens", "womens", "newbie", "channel"),
}
POWER_SET = "pre_period"


def covariate_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> FloatArray:
    """Numeric covariates as-is; categorical ones one-hot encoded with the first level dropped."""
    parts = [
        pd.get_dummies(frame[column], prefix=column, drop_first=True, dtype=float)
        if isinstance(frame[column].dtype, pd.CategoricalDtype)
        else frame[[column]].astype(float)
        for column in columns
    ]
    return pd.concat(parts, axis=1).to_numpy(dtype=np.float64)


def _residual_variance_share(y: FloatArray, x: FloatArray) -> float:
    """``1 - R^2`` of an OLS fit: the outcome variance CUPED would leave."""
    theta = cuped_theta(y, x)
    residual = y - (x - x.mean(axis=0)) @ theta
    return float(residual.var() / y.var())


def _effects(frame: pd.DataFrame, alpha: float) -> dict[str, Any]:
    control = (frame["segment"] == CONTROL_ARM).to_numpy()
    matrices = {name: covariate_matrix(frame, cols) for name, cols in COVARIATE_SETS.items()}
    effects: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        treated = (frame["segment"] == arm).to_numpy()
        effects[arm] = {}
        for outcome in OUTCOMES:
            y = frame[outcome].to_numpy(dtype=np.float64)
            entry: dict[str, Any] = {"cuped": {}, "regression_adjustment": {}}
            for name, x in matrices.items():
                result = cuped_difference(
                    y[treated], y[control], x[treated], x[control], alpha=alpha
                )
                entry["raw"] = result.raw.to_dict()
                entry["cuped"][name] = {
                    "adjusted": result.adjusted.to_dict(),
                    "theta": list(result.theta),
                    "variance_reduction": result.variance_reduction,
                    "ci_width_shrink": result.ci_width_shrink,
                }
                ols = regression_adjusted_effect(
                    y[treated], y[control], x[treated], x[control], alpha=alpha
                )
                entry["regression_adjustment"][name] = vars(ols)
            effects[arm][outcome] = entry
    return effects


def _power(frame: pd.DataFrame, alpha: float, power: float, relative_mde: float) -> dict[str, Any]:
    control = frame[frame["segment"] == CONTROL_ARM]
    x = covariate_matrix(control, COVARIATE_SETS[POWER_SET])
    n_control = len(control)
    n_grid = np.unique(np.geomspace(1_000, 2_000_000, 60).astype(int))
    report: dict[str, Any] = {}
    for outcome in OUTCOMES:
        y = control[outcome].to_numpy(dtype=np.float64)
        sd, mean = float(y.std(ddof=1)), float(y.mean())
        factor = _residual_variance_share(y, x)
        mde_abs = mean * relative_mde
        n_raw = required_n(mde_abs, sd, alpha=alpha, power=power)
        n_cuped = required_n(mde_abs, sd, alpha=alpha, power=power, variance_factor=factor)
        report[outcome] = {
            "control_mean": mean,
            "control_sd": sd,
            "variance_factor_cuped": factor,
            "mde_at_n": {
                "n_per_arm": n_control,
                "raw": minimum_detectable_effect(n_control, sd, alpha=alpha, power=power),
                "cuped": minimum_detectable_effect(
                    n_control, sd, alpha=alpha, power=power, variance_factor=factor
                ),
            },
            "required_n_per_arm": {
                "relative_mde": relative_mde,
                "absolute_mde": mde_abs,
                "raw": n_raw,
                "cuped": n_cuped,
                "reduction": 1.0 - n_cuped / n_raw,
            },
            "curve": {
                "n_per_arm": n_grid.tolist(),
                "power_raw": [power_at(n, mde_abs, sd, alpha=alpha) for n in n_grid],
                "power_cuped": [
                    power_at(n, mde_abs, sd, alpha=alpha, variance_factor=factor) for n in n_grid
                ],
            },
        }
    return report


def analyse_hillstrom(
    frame: pd.DataFrame, *, alpha: float = 0.05, power: float = 0.8, relative_mde: float = 0.10
) -> dict[str, Any]:
    """Every number the Hillstrom report shows, keyed for JSON."""
    counts = frame["segment"].value_counts().reindex(list(ARMS), fill_value=0)
    srm = srm_test({str(arm): int(count) for arm, count in counts.items()})
    return {
        "n_rows": len(frame),
        "alpha": alpha,
        "power_target": power,
        "covariate_sets": {name: list(cols) for name, cols in COVARIATE_SETS.items()},
        "srm": {**vars(srm), "mismatch": srm.mismatch},
        "effects": _effects(frame, alpha),
        "power": _power(frame, alpha, power, relative_mde),
    }
