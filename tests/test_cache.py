"""Checksum verification and download-once behaviour, using file:// URLs (no network)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from uplift_lab.data import cache
from uplift_lab.data.cache import ChecksumError, RemoteFile, cache_dir, fetch, verify

PAYLOAD = b"recency,history\n1,2.5\n"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def remote(tmp_path: Path) -> RemoteFile:
    source = tmp_path / "origin" / "file.csv"
    source.parent.mkdir()
    source.write_bytes(PAYLOAD)
    return RemoteFile(source.as_uri(), "file.csv", PAYLOAD_SHA, len(PAYLOAD))


def test_fetch_downloads_and_verifies(remote: RemoteFile, tmp_path: Path) -> None:
    path = fetch(remote, tmp_path / "cache")
    assert path.read_bytes() == PAYLOAD
    assert not list(path.parent.glob("*.part"))


def test_fetch_uses_valid_cache_without_downloading(
    remote: RemoteFile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch(remote, tmp_path / "cache")

    def fail(url: str, dest: Path) -> None:
        raise AssertionError("must not download when the cached copy verifies")

    monkeypatch.setattr(cache, "_download", fail)
    assert fetch(remote, tmp_path / "cache").read_bytes() == PAYLOAD


def test_fetch_replaces_corrupt_cached_copy(remote: RemoteFile, tmp_path: Path) -> None:
    target = cache_dir(tmp_path / "cache") / remote.filename
    target.write_bytes(b"truncated")
    assert fetch(remote, tmp_path / "cache").read_bytes() == PAYLOAD


def test_fetch_rejects_and_deletes_mismatched_download(remote: RemoteFile, tmp_path: Path) -> None:
    wrong = RemoteFile(remote.url, remote.filename, "0" * 64, remote.size_bytes)
    with pytest.raises(ChecksumError, match="does not match"):
        fetch(wrong, tmp_path / "cache")
    assert not (tmp_path / "cache" / remote.filename).exists()


def test_verify_accepts_uppercase_pin(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(PAYLOAD)
    verify(path, PAYLOAD_SHA.upper())


def test_unsupported_scheme_is_rejected(tmp_path: Path) -> None:
    bad = RemoteFile("ftp://example.org/f", "f", PAYLOAD_SHA, 1)
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        fetch(bad, tmp_path)


def test_cache_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cache.CACHE_ENV_VAR, str(tmp_path / "env"))
    assert cache_dir() == tmp_path / "env"
    assert cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert (tmp_path / "env").is_dir()
