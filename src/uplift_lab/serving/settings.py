"""Serving configuration from environment variables, validated once at startup.

- ``UPLIFT_LAB_MODEL_DIR``: an exported model directory (``manifest.json`` + LightGBM text
  models). The Docker image bakes one in at ``/app/model``.
- ``UPLIFT_LAB_MODEL_URI``: optional MLflow model URI such as
  ``models:/uplift-criteo-visit@champion``. When set it takes precedence. It is resolved with the
  ``MLFLOW_TRACKING_URI`` / ``MLFLOW_REGISTRY_URI`` settings and needs ``mlflow-skinny`` (or
  ``mlflow``) installed.
- ``UPLIFT_LAB_TRAFFIC_LOG``: path of the sampled traffic log; unset means no logging.
- ``UPLIFT_LAB_TRAFFIC_SAMPLE_RATE`` (default 0.05), ``UPLIFT_LAB_TRAFFIC_MAX_ROWS`` (20 rows per
  record) and ``UPLIFT_LAB_TRAFFIC_MAX_BYTES`` (10 MB before rotation) bound that log.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PREFIX = "UPLIFT_LAB_"


@dataclass(frozen=True)
class ServingSettings:
    model_dir: Path | None = None
    model_uri: str | None = None
    traffic_log: Path | None = None
    traffic_sample_rate: float = 0.05
    traffic_max_rows: int = 20
    traffic_max_bytes: int = 10_000_000

    def __post_init__(self) -> None:
        if self.model_dir is None and not self.model_uri:
            raise ValueError(f"set {PREFIX}MODEL_DIR or {PREFIX}MODEL_URI")
        if not 0.0 <= self.traffic_sample_rate <= 1.0:
            raise ValueError("traffic sample rate must be in [0, 1]")
        if self.traffic_max_rows < 1 or self.traffic_max_bytes < 1:
            raise ValueError("traffic max rows and max bytes must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServingSettings:
        values = os.environ if env is None else env

        def get(name: str) -> str | None:
            return values.get(PREFIX + name) or None

        def number(name: str, default: float, kind: type[float] | type[int]) -> float:
            raw = get(name)
            try:
                return default if raw is None else kind(raw)
            except ValueError as error:
                raise ValueError(f"{PREFIX}{name} must be a number, got {raw!r}") from error

        model_dir, log = get("MODEL_DIR"), get("TRAFFIC_LOG")
        return cls(
            model_dir=Path(model_dir) if model_dir else None,
            model_uri=get("MODEL_URI"),
            traffic_log=Path(log) if log else None,
            traffic_sample_rate=float(number("TRAFFIC_SAMPLE_RATE", 0.05, float)),
            traffic_max_rows=int(number("TRAFFIC_MAX_ROWS", 20, int)),
            traffic_max_bytes=int(number("TRAFFIC_MAX_BYTES", 10_000_000, int)),
        )
