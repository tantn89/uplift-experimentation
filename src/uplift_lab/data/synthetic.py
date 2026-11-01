"""Synthetic randomised data with a known heterogeneous effect, shaped like Criteo.

Used by the test suite and by report scripts that must run without the real dataset (CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_lab.experiments.estimators import FloatArray

N_FEATURES = 12
FEATURE_NAMES = tuple(f"f{i}" for i in range(N_FEATURES))


def true_uplift(frame: pd.DataFrame) -> FloatArray:
    """CATE on the visit probability: rises with f0, slightly negative for low f0."""
    f0 = frame["f0"].to_numpy(dtype=np.float64)
    return 0.20 / (1.0 + np.exp(-2.0 * f0)) - 0.03


def synthetic_uplift_frame(n: int, seed: int, treated_share: float = 0.85) -> pd.DataFrame:
    """f0..f11 normal; baseline visit rate driven by f1; effect by f0; exposure post-treatment."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    treatment = (rng.random(n) < treated_share).astype(np.int8)
    base = 0.04 + 0.10 / (1.0 + np.exp(-frame["f1"].to_numpy()))
    rate = np.clip(base + treatment * true_uplift(frame), 0.0, 1.0)
    visit = (rng.random(n) < rate).astype(np.int8)
    exposure = (treatment * (rng.random(n) < 0.3)).astype(np.int8)
    conversion = (visit * (rng.random(n) < 0.1)).astype(np.int8)
    return frame.assign(treatment=treatment, exposure=exposure, visit=visit, conversion=conversion)
