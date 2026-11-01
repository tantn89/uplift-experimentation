"""A sampled, size-bounded JSON Lines log of scoring traffic, read back by the drift report.

The serving app records a request with probability ``sample_rate``. A record holds the
timestamp, endpoint, batch size and at most ``max_rows`` randomly chosen rows of the batch
(features and predicted uplift, column-wise). The log stays within ``2 * max_bytes`` on disk:
when the next record would push the file past ``max_bytes`` it is renamed to ``<name>.1``
(replacing the previous one) and a new file is started. A failed write is logged and never fails the
request. Customer identifiers are not recorded.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from uplift_lab.experiments.estimators import FloatArray

LOGGER = logging.getLogger(__name__)
UPLIFT_COLUMN = "uplift"


class TrafficLog:
    """Thread-safe sampled writer; one instance per serving process."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: float,
        max_rows: int,
        max_bytes: int,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        if max_rows < 1 or max_bytes < 1:
            raise ValueError("max_rows and max_bytes must be positive")
        self.path, self.sample_rate = path, sample_rate
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def record(self, endpoint: str, frame: pd.DataFrame, uplift: FloatArray) -> bool:
        """Maybe append one record for a scored batch; returns whether one was written."""
        with self._lock:
            if self._rng.random() >= self.sample_rate:
                return False
            keep = np.arange(len(frame))
            if keep.size > self.max_rows:
                keep = np.sort(self._rng.choice(keep.size, self.max_rows, replace=False))
            rows = {str(c): frame[c].to_numpy()[keep].tolist() for c in frame.columns}
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "endpoint": endpoint,
                "n": len(frame),
                "rows": {**rows, UPLIFT_COLUMN: np.asarray(uplift)[keep].tolist()},
            }
            try:
                line = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
                self._rotate_if_full(len(line))
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("ab") as handle:
                    handle.write(line)
            except (OSError, ValueError) as error:
                LOGGER.warning("traffic log write to %s failed: %s", self.path, error)
                return False
            return True

    def _rotate_if_full(self, incoming: int) -> None:
        """Start a new file if appending ``incoming`` bytes would pass ``max_bytes``."""
        if self.path.is_file() and self.path.stat().st_size + incoming > self.max_bytes:
            self.path.replace(self.path.with_name(self.path.name + ".1"))


def read_traffic_log(path: Path, features: Sequence[str]) -> pd.DataFrame:
    """Rows of a traffic log as a frame of ``features`` plus ``uplift``; strict on format."""
    columns = [*features, UPLIFT_COLUMN]
    parts: list[pd.DataFrame] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows = json.loads(line)["rows"]
                part = pd.DataFrame({name: rows[name] for name in columns}, dtype=np.float64)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: malformed traffic record ({error})") from error
            parts.append(part)
    if not parts:
        raise ValueError(f"{path} holds no traffic records")
    return pd.concat(parts, ignore_index=True)
