"""Uplift modelling: in-house meta-learners, Qini/AUUC, policy value and a serving artifact.

Public names are re-exported lazily (``uplift_lab.lazy_exports``): loading a model with
``uplift_lab.uplift.artifacts`` needs only numpy, pandas and LightGBM, while the training
pipeline (scikit-learn, the ``analysis`` extra) is imported on first use. MLflow support lives
in ``uplift_lab.uplift.tracking`` and is imported only when used.
"""

from typing import TYPE_CHECKING

from uplift_lab.lazy_exports import lazy_exports

if TYPE_CHECKING:
    from uplift_lab.uplift.artifacts import ArtifactError, load_learner, read_manifest, save_learner
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
    "ArtifactError",
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
    "load_learner",
    "make_learner",
    "policy_value",
    "qini_coefficient",
    "qini_curve",
    "read_manifest",
    "run_uplift",
    "save_learner",
    "split_train_test",
    "uplift_at_k",
    "uplift_curve",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "artifacts": ("ArtifactError", "load_learner", "read_manifest", "save_learner"),
        "bootstrap": ("Interval", "bootstrap_intervals", "bootstrap_weights"),
        "evaluation": ("decile_table", "evaluate_scores"),
        "features": ("LeakageError", "check_features", "feature_matrix"),
        "learners": (
            "LEARNERS",
            "LgbmParams",
            "SLearner",
            "TLearner",
            "TransformedOutcome",
            "UpliftLearner",
            "XLearner",
            "make_learner",
        ),
        "metrics": (
            "Gains",
            "Ranking",
            "auuc",
            "auuc_gain",
            "gains",
            "qini_coefficient",
            "qini_curve",
            "uplift_at_k",
            "uplift_curve",
        ),
        "pipeline": ("UpliftConfig", "UpliftRun", "run_uplift", "split_train_test"),
        "policy": ("PolicyValue", "average_treatment_effect", "policy_value"),
    },
)
