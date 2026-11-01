"""Randomisation check: how well do the pre-treatment features predict treatment assignment?

In a clean RCT assignment is independent of the features, so a classifier trained to predict
treatment from them has a held-out ROC AUC of 0.5. An interval that excludes 0.5 means the
treated share varies with the features (for Criteo, pooled from several tests with slightly
different ratios). Meta-learners that model each arm separately are unaffected, but estimators
that weight by a single constant propensity (the transformed outcome, Horvitz-Thompson IPW)
are biased where the local treated share differs from the overall one.

The AUC is the Mann-Whitney statistic with ties counted as one half. Rows are sorted once and
grouped by tied score; each bootstrap replicate is then an O(n) weighted sum over tie groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.bootstrap import bootstrap_intervals
from uplift_lab.uplift.features import binary_array, feature_matrix
from uplift_lab.uplift.learners import LgbmParams


class TiedAuc:
    """Weighted ROC AUC of fixed scores and labels, for many weight vectors."""

    def __init__(self, scores: ArrayLike, labels: ArrayLike) -> None:
        s = np.asarray(scores, dtype=np.float64)
        self.labels = binary_array(labels, "labels")
        if s.shape != self.labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if np.unique(self.labels).size < 2:
            raise ValueError("both classes are needed for an AUC")
        unique, inverse = np.unique(s, return_inverse=True)  # groups in ascending score order
        self.groups: NDArray[np.intp] = inverse.astype(np.intp)
        self.n_groups = int(unique.size)

    def __call__(self, weights: FloatArray | None = None) -> float:
        w = np.ones(self.labels.size) if weights is None else weights
        positive = np.bincount(self.groups, w * self.labels, self.n_groups)
        negative = np.bincount(self.groups, w * (1.0 - self.labels), self.n_groups)
        below = np.cumsum(negative) - negative
        denominator = positive.sum() * negative.sum()
        if denominator <= 0.0:
            return float("nan")
        return float((positive * (below + 0.5 * negative)).sum() / denominator)


def propensity_check(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    params: LgbmParams,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit treatment ~ features on ``train``; held-out AUC on ``test`` with a bootstrap CI."""
    x_train, x_test = feature_matrix(train, features), feature_matrix(test, features)
    t_train = binary_array(train["treatment"], "treatment")
    t_test = binary_array(test["treatment"], "treatment")
    booster = params.train(x_train, t_train, "binary")
    auc = TiedAuc(np.asarray(booster.predict(x_test), dtype=np.float64), t_test)
    interval = bootstrap_intervals(
        lambda w: {"auc": auc(w)}, t_test.size, n_boot=n_bootstrap, seed=seed, alpha=alpha
    )["auc"]
    return {
        "auc": interval.to_dict(),
        "randomisation_rejected": interval.ci_low > 0.5 or interval.ci_high < 0.5,
        "treated_share_train": float(t_train.mean()),
        "treated_share_test": float(t_test.mean()),
        "model": "LightGBM binary classifier, same settings as the uplift learners",
    }
