"""Feature validation at the model boundary, including the post-treatment leakage guard.

Criteo's ``exposure`` flag records whether a treated user actually saw an ad. It is measured
after assignment and is caused by it (control users are never exposed), so a model given it
would learn the treatment itself rather than who responds to it. The same holds for the
treatment flag and the outcomes. Learners therefore refuse any of these columns as features,
and every estimate stays intent-to-treat: the effect of being assigned to treatment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from uplift_lab.experiments.estimators import FloatArray

POST_TREATMENT_COLUMNS = frozenset({"exposure"})
NON_FEATURE_COLUMNS = frozenset({"treatment", "visit", "conversion"}) | POST_TREATMENT_COLUMNS


class LeakageError(ValueError):
    """A post-treatment, treatment or outcome column was offered as a model feature."""


def check_features(columns: Iterable[object]) -> tuple[str, ...]:
    """Validate feature names: non-empty, unique, and free of post-treatment or label columns."""
    names = tuple(str(column) for column in columns)
    if not names:
        raise ValueError("at least one feature is required")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names: {sorted(names)}")
    leaked = sorted(set(names) & NON_FEATURE_COLUMNS)
    if leaked:
        raise LeakageError(
            f"{leaked} cannot be features: exposure is post-treatment and treatment/outcome "
            "columns are labels; uplift models use pre-treatment features only"
        )
    return names


def feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> FloatArray:
    """Select ``features`` (in that order) from ``frame`` as a finite float64 matrix."""
    check_features(features)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    matrix = frame[list(features)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinite values")
    return matrix


def binary_array(values: object, name: str) -> FloatArray:
    """Validate a 1-D array of 0/1 values and return it as float64."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0 or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be a non-empty array of 0/1 values")
    return array
