"""Bundle artifact build: a fresh wheel in ``dist/`` that records the git state it was built from.

``databricks bundle deploy`` runs this from the repository root (see ``databricks.yml``):

1. Deletes ``dist/``, so the job's ``dist/*.whl`` glob can only match the wheel built now.
2. Writes ``src/uplift_lab/build_info.json`` with the HEAD SHA and whether tracked files had
   uncommitted changes. The bundle passes ``${bundle.git.commit}`` to the job but has no dirty
   flag; the training job reads this file and records ``git.dirty`` in its report, the MLflow
   tags and the model manifest. The file is git-ignored and removed again after the build.
3. Builds the wheel with ``python -m pip wheel --no-deps``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD_INFO = ROOT / "src" / "uplift_lab" / "build_info.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if DIST.exists():
        shutil.rmtree(DIST)
    BUILD_INFO.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "git_dirty": dirty,
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(DIST), "."],
            cwd=ROOT,
            check=True,
        )
    finally:
        BUILD_INFO.unlink(missing_ok=True)
    wheels = sorted(DIST.glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected exactly one wheel in {DIST}, found {wheels}", file=sys.stderr)
        return 1
    print(f"built {wheels[0].name} at {sha[:12]} (dirty: {dirty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
