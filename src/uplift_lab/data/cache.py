"""Download-on-demand cache with SHA-256 verification.

Raw datasets are never committed to the repository. They are fetched once into a per-user
cache directory and verified against a pinned SHA-256 before use, so every result can be
traced to the exact bytes it was computed from.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs

CACHE_ENV_VAR = "UPLIFT_LAB_CACHE"
_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 60


class ChecksumError(RuntimeError):
    """A file's SHA-256 does not match the pinned value."""


@dataclass(frozen=True)
class RemoteFile:
    """A pinned remote file: where to get it, what to call it, and what its bytes hash to."""

    url: str
    filename: str
    sha256: str
    size_bytes: int


def cache_dir(override: Path | str | None = None) -> Path:
    """Resolve the cache directory: explicit override, then $UPLIFT_LAB_CACHE, then user cache."""
    if override is not None:
        path = Path(override)
    elif env := os.environ.get(CACHE_ENV_VAR):
        path = Path(env)
    else:
        path = Path(platformdirs.user_cache_dir("uplift-lab", appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    """Raise ChecksumError if the file's SHA-256 differs from the expected value."""
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ChecksumError(
            f"{path.name}: sha256 {actual} does not match pinned {expected_sha256.lower()}"
        )


def _download(url: str, dest: Path) -> None:
    """Stream a URL to dest through a temporary file, so a failed download leaves no partial."""
    if not url.startswith(("https://", "http://", "file://")):
        raise ValueError(f"unsupported URL scheme: {url}")
    tmp = dest.with_name(dest.name + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, _CHUNK)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(remote: RemoteFile, directory: Path | str | None = None) -> Path:
    """Return a verified local copy of a remote file, downloading it at most once.

    A cached copy that fails verification is treated as corrupt: it is removed and fetched
    again once. A fresh download that still fails verification raises ChecksumError and is
    deleted, so a bad file never stays in the cache.
    """
    dest = cache_dir(directory) / remote.filename
    if dest.exists():
        try:
            verify(dest, remote.sha256)
            return dest
        except ChecksumError:
            dest.unlink()
    _download(remote.url, dest)
    try:
        verify(dest, remote.sha256)
    except ChecksumError:
        dest.unlink(missing_ok=True)
        raise
    return dest
