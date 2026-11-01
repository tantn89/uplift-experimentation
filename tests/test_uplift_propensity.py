"""Randomisation check: tie-aware weighted AUC and the held-out treatment-prediction AUC."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from synthetic_uplift import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.uplift.learners import LgbmParams
from uplift_lab.uplift.pipeline import split_train_test
from uplift_lab.uplift.propensity import TiedAuc, propensity_check

PARAMS = LgbmParams(n_estimators=30, min_child_samples=50)


def test_tied_auc_matches_sklearn_with_ties_and_weights() -> None:
    rng = np.random.default_rng(4)
    scores = rng.integers(0, 6, 500).astype(float)  # heavy ties
    labels = (rng.random(500) < 0.3 + 0.08 * scores).astype(float)
    weights = rng.integers(0, 3, 500).astype(float)
    auc = TiedAuc(scores, labels)
    assert auc() == pytest.approx(roc_auc_score(labels, scores))
    assert auc(weights) == pytest.approx(roc_auc_score(labels, scores, sample_weight=weights))
    assert TiedAuc([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])() == 1.0
    assert TiedAuc([1.0, 1.0], [0, 1])() == 0.5
    with pytest.raises(ValueError, match="both classes"):
        TiedAuc([1.0, 2.0], [1, 1])


def test_randomised_assignment_is_not_predictable() -> None:
    frame = synthetic_uplift_frame(20_000, seed=41)
    train, test = split_train_test(frame, "visit", test_size=0.3, seed=1)
    check = propensity_check(train, test, FEATURE_NAMES, params=PARAMS, n_bootstrap=100, seed=2)
    auc = check["auc"]
    assert auc["ci_low"] < 0.5 < auc["ci_high"]
    assert check["randomisation_rejected"] is False
    assert check["treated_share_test"] == pytest.approx(0.85, abs=0.01)


def test_feature_dependent_assignment_is_detected() -> None:
    frame = synthetic_uplift_frame(20_000, seed=42)
    rng = np.random.default_rng(5)
    share = np.where(frame["f3"] > 0, 0.9, 0.8)  # two pooled tests with different ratios
    frame["treatment"] = (rng.random(len(frame)) < share).astype(np.int8)
    train, test = split_train_test(frame, "visit", test_size=0.3, seed=1)
    check = propensity_check(train, test, FEATURE_NAMES, params=PARAMS, n_bootstrap=100, seed=2)
    assert check["auc"]["ci_low"] > 0.52
    assert check["randomisation_rejected"] is True
