"""Meta-learners for a binary outcome in a randomised experiment, on LightGBM boosters.

Every learner predicts the conditional average treatment effect of assignment
``tau(x) = P(y = 1 | x, treated) - P(y = 1 | x, control)`` (intent to treat) and shares one
interface: ``fit(X, treatment, outcome)`` then ``predict(X)``. A fitted learner is fully
described by its LightGBM boosters, the feature order and the treatment propensity, which is
what ``uplift_lab.uplift.artifacts`` saves for serving.

- S-learner: one classifier on ``[x, t]``; ``tau = mu(x, 1) - mu(x, 0)``.
- T-learner: one classifier per arm; ``tau = mu_1(x) - mu_0(x)``.
- X-learner (Kuenzel et al. 2019): impute each unit's effect with the other arm's model, regress
  it per arm, then blend ``tau = e * tau_0(x) + (1 - e) * tau_1(x)`` with the propensity ``e``.
- Transformed outcome (Athey and Imbens 2015): regress ``z = y (t / e - (1 - t) / (1 - e))``,
  whose conditional mean is ``tau(x)``. The propensity weighting matters here: Criteo treats
  85% of users, and the unweighted class transformation assumes 50%.

In an RCT the propensity is a constant, estimated as the training treatment share.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.features import binary_array, check_features, feature_matrix


@dataclass(frozen=True)
class LgbmParams:
    """LightGBM settings shared by every component model; deterministic for a given seed."""

    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 100
    seed: int = 20261101

    def __post_init__(self) -> None:
        if self.n_estimators < 1 or self.num_leaves < 2 or self.min_child_samples < 1:
            raise ValueError("n_estimators >= 1, num_leaves >= 2, min_child_samples >= 1")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def train(self, x: FloatArray, y: FloatArray, objective: str) -> Any:
        """Train one booster; ``deterministic`` + col-wise histograms make reruns identical."""
        params = {
            "objective": objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
        }
        return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=self.n_estimators)


def _predict(booster: Any, x: FloatArray) -> FloatArray:
    return np.asarray(booster.predict(x), dtype=np.float64)


class UpliftLearner(ABC):
    """Common fit/predict interface; subclasses define the components and how they combine."""

    kind: ClassVar[str]
    components: ClassVar[tuple[str, ...]]

    def __init__(self, params: LgbmParams | None = None) -> None:
        self.params = params or LgbmParams()
        self.boosters_: dict[str, Any] = {}
        self.features_: tuple[str, ...] = ()
        self.propensity_: float = float("nan")

    def fit(self, X: pd.DataFrame, treatment: ArrayLike, outcome: ArrayLike) -> Self:
        features = check_features(X.columns)
        x = feature_matrix(X, features)
        t = binary_array(treatment, "treatment")
        y = binary_array(outcome, "outcome")
        if not len(x) == len(t) == len(y):
            raise ValueError("X, treatment and outcome must have the same length")
        if np.unique(t).size < 2:
            raise ValueError("both treated and control units are needed")
        propensity = float(t.mean())
        self.boosters_ = self._fit(x, t, y, propensity)
        self.features_, self.propensity_ = features, propensity
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Predicted uplift per row; columns are selected by the fitted feature names."""
        if not self.boosters_:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._uplift(feature_matrix(X, self.features_))

    @classmethod
    def from_boosters(
        cls,
        boosters: Mapping[str, Any],
        features: tuple[str, ...],
        propensity: float,
        params: LgbmParams | None = None,
    ) -> Self:
        """Rebuild a fitted learner from saved boosters (see ``artifacts.load_learner``)."""
        if set(boosters) != set(cls.components):
            raise ValueError(f"{cls.kind} needs boosters {cls.components}, got {sorted(boosters)}")
        if not 0.0 < propensity < 1.0:
            raise ValueError(f"propensity must be in (0, 1), got {propensity}")
        learner = cls(params)
        learner.boosters_ = dict(boosters)
        learner.features_ = check_features(features)
        learner.propensity_ = propensity
        return learner

    @abstractmethod
    def _fit(self, x: FloatArray, t: FloatArray, y: FloatArray, e: float) -> dict[str, Any]: ...

    @abstractmethod
    def _uplift(self, x: FloatArray) -> FloatArray: ...


class SLearner(UpliftLearner):
    kind, components = "s_learner", ("outcome",)

    def _fit(self, x: FloatArray, t: FloatArray, y: FloatArray, e: float) -> dict[str, Any]:
        return {"outcome": self.params.train(np.column_stack([x, t]), y, "binary")}

    def _uplift(self, x: FloatArray) -> FloatArray:
        model = self.boosters_["outcome"]
        ones, zeros = np.ones((len(x), 1)), np.zeros((len(x), 1))
        return _predict(model, np.hstack([x, ones])) - _predict(model, np.hstack([x, zeros]))


class TLearner(UpliftLearner):
    kind, components = "t_learner", ("control", "treated")

    def _fit(self, x: FloatArray, t: FloatArray, y: FloatArray, e: float) -> dict[str, Any]:
        arm = t == 1.0
        return {
            "control": self.params.train(x[~arm], y[~arm], "binary"),
            "treated": self.params.train(x[arm], y[arm], "binary"),
        }

    def _uplift(self, x: FloatArray) -> FloatArray:
        return _predict(self.boosters_["treated"], x) - _predict(self.boosters_["control"], x)


class XLearner(UpliftLearner):
    kind, components = "x_learner", ("effect_control", "effect_treated")

    def _fit(self, x: FloatArray, t: FloatArray, y: FloatArray, e: float) -> dict[str, Any]:
        arm = t == 1.0
        mu0 = self.params.train(x[~arm], y[~arm], "binary")
        mu1 = self.params.train(x[arm], y[arm], "binary")
        imputed_treated = y[arm] - _predict(mu0, x[arm])
        imputed_control = _predict(mu1, x[~arm]) - y[~arm]
        return {
            "effect_control": self.params.train(x[~arm], imputed_control, "regression"),
            "effect_treated": self.params.train(x[arm], imputed_treated, "regression"),
        }

    def _uplift(self, x: FloatArray) -> FloatArray:
        e = self.propensity_
        tau0 = _predict(self.boosters_["effect_control"], x)
        tau1 = _predict(self.boosters_["effect_treated"], x)
        return e * tau0 + (1.0 - e) * tau1


class TransformedOutcome(UpliftLearner):
    kind, components = "transformed_outcome", ("transformed",)

    def _fit(self, x: FloatArray, t: FloatArray, y: FloatArray, e: float) -> dict[str, Any]:
        z = y * (t / e - (1.0 - t) / (1.0 - e))
        return {"transformed": self.params.train(x, z, "regression")}

    def _uplift(self, x: FloatArray) -> FloatArray:
        return _predict(self.boosters_["transformed"], x)


LEARNERS: dict[str, type[UpliftLearner]] = {
    cls.kind: cls for cls in (SLearner, TLearner, XLearner, TransformedOutcome)
}


def make_learner(kind: str, params: LgbmParams | None = None) -> UpliftLearner:
    """Instantiate a learner by its ``kind`` name."""
    if kind not in LEARNERS:
        raise ValueError(f"unknown learner {kind!r}; choose from {sorted(LEARNERS)}")
    return LEARNERS[kind](params)
