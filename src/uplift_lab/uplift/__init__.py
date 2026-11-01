"""Uplift modelling: in-house meta-learners, Qini/AUUC, policy value and a serving artifact.

MLflow support lives in ``uplift_lab.uplift.tracking`` and is imported only when used, so the
core package works without the ``mlflow`` extra.
"""

from uplift_lab.uplift.bootstrap import Interval, bootstrap_intervals, bootstrap_weights
from uplift_lab.uplift.evaluation import decile_table, evaluate_scores
from uplift_lab.uplift.features import LeakageError, check_features, feature_matrix
from uplift_lab.uplift.learners import (
    LEARNERS,
    LgbmParams,
    SLearner,
    TLearner,
    TransformedOutcome,
    UpliftLearner,
    XLearner,
    make_learner,
)
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
from uplift_lab.uplift.pipeline import UpliftConfig, UpliftRun, run_uplift, split_train_test
from uplift_lab.uplift.policy import PolicyValue, average_treatment_effect, policy_value

__all__ = [
    "LEARNERS",
    "Gains",
    "Interval",
    "LeakageError",
    "LgbmParams",
    "PolicyValue",
    "Ranking",
    "SLearner",
    "TLearner",
    "TransformedOutcome",
    "UpliftConfig",
    "UpliftLearner",
    "UpliftRun",
    "XLearner",
    "auuc",
    "auuc_gain",
    "average_treatment_effect",
    "bootstrap_intervals",
    "bootstrap_weights",
    "check_features",
    "decile_table",
    "evaluate_scores",
    "feature_matrix",
    "gains",
    "make_learner",
    "policy_value",
    "qini_coefficient",
    "qini_curve",
    "run_uplift",
    "split_train_test",
    "uplift_at_k",
    "uplift_curve",
]
