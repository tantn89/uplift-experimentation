"""Meta-learners: recovery of a known heterogeneous effect, determinism, interface and guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from synthetic_uplift import FEATURE_NAMES, synthetic_uplift_frame, true_uplift
from uplift_lab.uplift.features import LeakageError, check_features, feature_matrix
from uplift_lab.uplift.learners import LEARNERS, LgbmParams, UpliftLearner, make_learner

PARAMS = LgbmParams(n_estimators=100, min_child_samples=50, seed=11)
FEATURES = list(FEATURE_NAMES)


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return synthetic_uplift_frame(30_000, seed=1)


@pytest.fixture(scope="module")
def test_frame() -> pd.DataFrame:
    return synthetic_uplift_frame(5_000, seed=2)


@pytest.fixture(scope="module")
def fitted(train: pd.DataFrame) -> dict[str, UpliftLearner]:
    return {
        kind: make_learner(kind, PARAMS).fit(train[FEATURES], train["treatment"], train["visit"])
        for kind in LEARNERS
    }


@pytest.mark.parametrize("kind", sorted(LEARNERS))
def test_learner_ranks_true_effect(
    kind: str, fitted: dict[str, UpliftLearner], test_frame: pd.DataFrame
) -> None:
    predicted = fitted[kind].predict(test_frame)
    assert predicted.shape == (len(test_frame),)
    assert predicted.dtype == np.float64
    # Measured 0.69 (transformed outcome) to 0.92 (S-learner) at an 85% treated share.
    assert spearmanr(predicted, true_uplift(test_frame)).statistic > 0.6
    # The mean prediction lands near the true average effect.
    assert abs(predicted.mean() - true_uplift(test_frame).mean()) < 0.02


@pytest.mark.parametrize("kind", sorted(LEARNERS))
def test_learner_is_deterministic_for_a_seed(kind: str) -> None:
    frame = synthetic_uplift_frame(4_000, seed=3)
    params = LgbmParams(n_estimators=20, seed=5)

    def fit_predict(p: LgbmParams) -> np.ndarray:
        learner = make_learner(kind, p).fit(frame[FEATURES], frame["treatment"], frame["visit"])
        return learner.predict(frame)

    np.testing.assert_array_equal(fit_predict(params), fit_predict(params))


def test_x_learner_blends_with_training_propensity(fitted: dict[str, UpliftLearner]) -> None:
    assert fitted["x_learner"].propensity_ == pytest.approx(0.85, abs=0.01)
    assert set(fitted["x_learner"].boosters_) == {"effect_control", "effect_treated"}


def test_predict_selects_columns_by_name(
    fitted: dict[str, UpliftLearner], test_frame: pd.DataFrame
) -> None:
    learner = fitted["t_learner"]
    shuffled = test_frame[list(reversed(test_frame.columns))]
    np.testing.assert_array_equal(learner.predict(shuffled), learner.predict(test_frame))
    with pytest.raises(ValueError, match="missing feature columns"):
        learner.predict(test_frame.drop(columns="f3"))


def test_exposure_as_feature_is_rejected(train: pd.DataFrame) -> None:
    with pytest.raises(LeakageError, match="exposure is post-treatment"):
        make_learner("t_learner", PARAMS).fit(
            train[[*FEATURES, "exposure"]], train["treatment"], train["visit"]
        )
    for column in ("exposure", "treatment", "visit", "conversion"):
        with pytest.raises(LeakageError):
            check_features(["f0", column])
    with pytest.raises(LeakageError):
        feature_matrix(train, ["f0", "exposure"])


def test_feature_validation() -> None:
    frame = pd.DataFrame({"f0": [1.0, np.nan], "f1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least one"):
        check_features([])
    with pytest.raises(ValueError, match="duplicate"):
        check_features(["f0", "f0"])
    with pytest.raises(ValueError, match="NaN"):
        feature_matrix(frame, ["f0"])
    assert feature_matrix(frame, ["f1"]).shape == (2, 1)


def test_fit_and_predict_validation(train: pd.DataFrame) -> None:
    small = train.iloc[:200]
    with pytest.raises(RuntimeError, match="not fitted"):
        make_learner("s_learner").predict(small)
    with pytest.raises(ValueError, match="both treated and control"):
        make_learner("s_learner").fit(small[FEATURES], np.ones(200), small["visit"])
    with pytest.raises(ValueError, match="0/1"):
        make_learner("s_learner").fit(small[FEATURES], small["treatment"], small["f0"])
    with pytest.raises(ValueError, match="same length"):
        make_learner("s_learner").fit(small[FEATURES], small["treatment"][:-1], small["visit"])
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("r_learner")
    with pytest.raises(ValueError, match="learning_rate"):
        LgbmParams(learning_rate=0.0)
