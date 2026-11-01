"""Provenance recorded with every uplift run: git state and a content checksum of the data."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    """``(HEAD sha, working tree dirty)``; ``("unknown", False)`` outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, bool(status.strip())


def frame_sha256(frame: pd.DataFrame) -> str:
    """Content checksum of the exact rows and columns a run trained and tested on."""
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(",".join(map(str, frame.columns)).encode())
    digest.update(hashed.tobytes())
    return digest.hexdigest()
