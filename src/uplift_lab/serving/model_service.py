"""The loaded model and its decision rules, independent of the web framework.

Two rules turn uplift scores into treat/skip decisions:

- ``threshold``: a named budget from the manifest (``metadata.budget_thresholds``). Its cut-off is
  the score quantile that targeted that share of the held-out test population, so a single
  customer gets the same decision as offline scoring: treat if ``uplift >= threshold``.
- ``batch_top_k``: treat the top ``budget_fraction`` of this batch by uplift. The count matches
  the offline evaluation, ``ceil(fraction * n)`` users in stable descending order. This suits
  a campaign list scored in one call. It is relative to the batch, so a batch of one always
  treats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from uplift_lab.data.criteo import FEATURES
from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.serving.schemas import Customer, ModelInfo
from uplift_lab.uplift.artifacts import ArtifactError, load_learner, read_manifest


class UnknownBudgetError(ValueError):
    """A named budget that the model manifest does not define."""


def top_share_mask(scores: FloatArray, fraction: float) -> NDArray[np.bool_]:
    """Boolean mask of the top ``fraction`` of ``scores``, same count rule as ``top_count``."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = int(np.searchsorted(np.arange(scores.size), fraction * scores.size, side="left"))
    mask = np.zeros(scores.size, dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def _thresholds(metadata: Any) -> dict[str, float]:
    raw = metadata.get("budget_thresholds", {}) if isinstance(metadata, dict) else {}
    if not isinstance(raw, dict) or not all(
        isinstance(v, int | float) and math.isfinite(v) for v in raw.values()
    ):
        raise ArtifactError("metadata.budget_thresholds must map budget names to finite numbers")
    return {str(name): float(value) for name, value in raw.items()}


class ModelService:
    """Load once at startup; score and decide on validated customer rows."""

    def __init__(self, model_dir: Path, source: str) -> None:
        self.manifest = read_manifest(model_dir)
        self.learner = load_learner(model_dir)
        if set(self.learner.features_) != set(FEATURES):
            expected = list(self.learner.features_)
            raise ArtifactError(f"the API accepts {list(FEATURES)}, the model expects {expected}")
        metadata = self.manifest.get("metadata", {})
        self.thresholds = _thresholds(metadata)
        self.info = ModelInfo(
            kind=self.learner.kind,
            features=list(self.learner.features_),
            schema_version=int(self.manifest["schema_version"]),
            uplift_lab_version=str(self.manifest.get("uplift_lab_version", "unknown")),
            git_sha=metadata.get("git_sha") if isinstance(metadata, dict) else None,
            source=source,
            budgets=self.thresholds,
        )

    @staticmethod
    def frame(customers: Sequence[Customer]) -> pd.DataFrame:
        return pd.DataFrame([c.features() for c in customers], columns=list(FEATURES))

    def score(self, frame: pd.DataFrame) -> FloatArray:
        return self.learner.predict(frame)

    def threshold(self, budget: str) -> float:
        if budget not in self.thresholds:
            raise UnknownBudgetError(
                f"unknown budget {budget!r}; this model defines {sorted(self.thresholds)}"
            )
        return self.thresholds[budget]
