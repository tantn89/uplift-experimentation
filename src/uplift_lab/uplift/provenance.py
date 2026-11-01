"""Provenance recorded with every uplift run: git state and a content checksum of the data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

BUILD_INFO = Path(__file__).resolve().parents[1] / "build_info.json"


def build_info(path: Path = BUILD_INFO) -> dict[str, Any] | None:
    """Git state recorded in the installed wheel by ``databricks/build_wheel.py``, if any.

    A job that runs from a wheel has no checkout to ask; this file is how a bundle deploy tells
    it whether the commit it was built from had uncommitted changes. ``None`` when absent or
    malformed.
    """
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(info, dict) or not isinstance(info.get("git_sha"), str):
        return None
    if not isinstance(info.get("git_dirty"), bool):
        return None
    return info


def job_git_state(git_sha: str | None, info: dict[str, Any] | None) -> dict[str, Any]:
    """``{sha, dirty, source}`` for a run: the checkout, or a passed SHA plus the build info.

    With ``git_sha`` (the bundle's ``${bundle.git.commit}``) the dirty flag comes from the
    wheel's build info, and only if that was built from the same commit; otherwise it is unknown
    (``None``).
    """
    if not git_sha:
        sha, dirty = git_state()
        return {"sha": sha, "dirty": dirty, "source": "checkout"}
    if info is not None and info["git_sha"] == git_sha:
        return {"sha": git_sha, "dirty": info["git_dirty"], "source": "wheel build info"}
    return {"sha": git_sha, "dirty": None, "source": "argument"}


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
