"""Train and evaluate every learner on one seeded split of a randomised dataset.

The split is stratified by treatment x outcome so both sets keep the 85/15 assignment ratio and
the per-arm outcome rate. Learners see only pre-treatment features; scores on the test set feed
``evaluate_scores`` together with a seeded random scoring as the random-targeting reference.
The same split also runs the randomisation check in ``propensity.py`` (treatment predicted from
the features, held-out AUC with a bootstrap CI).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_lab.data.criteo import DEFAULT_SEED, FEATURES
from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.evaluation import DEFAULT_BUDGETS, RANDOM_SCORER, evaluate_scores
from uplift_lab.uplift.features import binary_array, check_features
from uplift_lab.uplift.learners import LEARNERS, LgbmParams, UpliftLearner, make_learner
from uplift_lab.uplift.propensity import propensity_check


@dataclass(frozen=True)
class UpliftConfig:
    outcome: str = "visit"
    test_size: float = 0.3
    seed: int = DEFAULT_SEED
    learners: tuple[str, ...] = tuple(LEARNERS)
    params: LgbmParams = field(default_factory=LgbmParams)
    n_bootstrap: int = 200
    budgets: tuple[float, ...] = DEFAULT_BUDGETS
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")
        if not self.learners or set(self.learners) - set(LEARNERS):
            raise ValueError(f"learners must be a non-empty subset of {sorted(LEARNERS)}")
        if not self.budgets or not all(0.0 < k <= 1.0 for k in self.budgets):
            raise ValueError("budgets must be non-empty and in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "learners": list(self.learners), "budgets": list(self.budgets)}


@dataclass
class UpliftRun:
    config: UpliftConfig
    features: tuple[str, ...]
    learners: dict[str, UpliftLearner]
    scores: dict[str, FloatArray]
    evaluation: dict[str, Any]
    n_train: int
    n_test: int
    timings: dict[str, float]
    # Test-set labels aligned with ``scores``, for callers that re-evaluate the same scorings.
    test_treatment: FloatArray
    test_outcome: FloatArray


def split_train_test(
    frame: pd.DataFrame, outcome: str, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded split stratified by treatment x outcome."""
    strata = frame["treatment"].astype(int) * 2 + frame[outcome].astype(int)
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=strata)
    return train, test


def budget_thresholds(scores: FloatArray, budgets: Sequence[float]) -> dict[str, float]:
    """Score cut-off that targets each budget share of a reference population."""
    return {f"{k:g}": float(np.quantile(scores, 1.0 - k)) for k in budgets}


def run_uplift(
    frame: pd.DataFrame, config: UpliftConfig, features: Sequence[str] = FEATURES
) -> UpliftRun:
    """Split, fit each configured learner on train, score and evaluate on test."""
    names = check_features(features)
    missing = {*names, "treatment", config.outcome} - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    binary_array(frame["treatment"], "treatment")
    binary_array(frame[config.outcome], config.outcome)
    train, test = split_train_test(
        frame, config.outcome, test_size=config.test_size, seed=config.seed
    )
    timings: dict[str, float] = {}
    learners: dict[str, UpliftLearner] = {}
    scores: dict[str, FloatArray] = {}
    for kind in config.learners:
        start = time.perf_counter()
        learner = make_learner(kind, config.params).fit(
            train[list(names)], train["treatment"], train[config.outcome]
        )
        timings[f"fit_{kind}_seconds"] = time.perf_counter() - start
        learners[kind] = learner
        scores[kind] = learner.predict(test)
    scores[RANDOM_SCORER] = np.random.default_rng(config.seed + 1).random(len(test))

    start = time.perf_counter()
    evaluation = evaluate_scores(
        scores,
        test["treatment"].to_numpy(),
        test[config.outcome].to_numpy(),
        budgets=config.budgets,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
        alpha=config.alpha,
    )
    timings["evaluate_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    evaluation["propensity_check"] = propensity_check(
        train,
        test,
        names,
        params=config.params,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
        alpha=config.alpha,
    )
    timings["propensity_check_seconds"] = time.perf_counter() - start
    return UpliftRun(
        config,
        names,
        learners,
        scores,
        evaluation,
        len(train),
        len(test),
        timings,
        binary_array(test["treatment"], "treatment"),
        binary_array(test[config.outcome], config.outcome),
    )
