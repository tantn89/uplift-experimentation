"""Synthetic randomised data with a known effect; see ``uplift_lab.data.synthetic``."""

from __future__ import annotations

from uplift_lab.data.synthetic import (
    FEATURE_NAMES,
    N_FEATURES,
    synthetic_uplift_frame,
    true_uplift,
)

__all__ = ["FEATURE_NAMES", "N_FEATURES", "synthetic_uplift_frame", "true_uplift"]
