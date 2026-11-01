"""A small exported model trained on synthetic data, and request payload builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_uplift import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.uplift.artifacts import save_learner
from uplift_lab.uplift.learners import LgbmParams, make_learner
from uplift_lab.uplift.pipeline import budget_thresholds

BUDGETS = (0.1, 0.3)


def export_model(directory: Path, kind: str = "t_learner") -> Path:
    """Fit a learner on synthetic data and save it with test-score budget thresholds."""
    frame = synthetic_uplift_frame(4_000, seed=51)
    learner = make_learner(kind, LgbmParams(n_estimators=20, min_child_samples=50)).fit(
        frame[list(FEATURE_NAMES)], frame["treatment"], frame["visit"]
    )
    reference = synthetic_uplift_frame(2_000, seed=52)
    thresholds = budget_thresholds(learner.predict(reference), BUDGETS)
    save_learner(learner, directory, metadata={"budget_thresholds": thresholds, "git_sha": "abc"})
    return directory


def customers(n: int, seed: int = 53) -> pd.DataFrame:
    return synthetic_uplift_frame(n, seed=seed)[list(FEATURE_NAMES)]


def payload(frame: pd.DataFrame, **extra: Any) -> dict[str, Any]:
    rows = [{"id": f"c{i}", **row} for i, row in enumerate(frame.to_dict(orient="records"))]
    return {"customers": rows, **extra}
