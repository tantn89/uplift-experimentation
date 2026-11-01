"""Shared report plumbing: strict JSON output and the optional-matplotlib message."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MISSING_MATPLOTLIB = 'Figures need matplotlib: python -m pip install -e ".[report]"'


def _clean(value: Any) -> Any:
    """Make a payload strict-JSON safe: numpy values to Python, NaN/inf to null."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as indented strict JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
