"""Stratified subsample: exact size, preserved strata, determinism. Synthetic CSV, no network."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from uplift_lab.data.criteo import FEATURES, LABELS, build_subsample
from uplift_lab.data.sampling import stratified_indices

N_SOURCE = 20_000


def _synthetic_criteo(path: Path, seed: int = 7) -> pd.DataFrame:
    """Write a gzipped CSV with the Criteo schema and a rare outcome, mirroring the real file."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(size=N_SOURCE) for name in FEATURES})
    treatment = (rng.random(N_SOURCE) < 0.85).astype(int)
    exposure = treatment * (rng.random(N_SOURCE) < 0.04)
    visit = (rng.random(N_SOURCE) < 0.04 + 0.02 * exposure).astype(int)
    conversion = visit * (rng.random(N_SOURCE) < 0.06)
    frame = frame.assign(treatment=treatment, conversion=conversion, visit=visit, exposure=exposure)
    with gzip.open(path, "wt", newline="") as handle:
        frame.to_csv(handle, index=False)
    return frame


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    path = tmp_path / "criteo.csv.gz"
    return path, _synthetic_criteo(path)


def test_subsample_is_deterministic(source: tuple[Path, pd.DataFrame], tmp_path: Path) -> None:
    path, _ = source
    first = build_subsample(path, tmp_path / "a.parquet", n_rows=3_000, seed=20261101)
    second = build_subsample(path, tmp_path / "b.parquet", n_rows=3_000, seed=20261101)
    other = build_subsample(path, tmp_path / "c.parquet", n_rows=3_000, seed=1)
    a, b, c = (pq.read_table(p).to_pandas() for p in (first, second, other))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_subsample_size_schema_and_strata(
    source: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    path, full = source
    sample = pq.read_table(
        build_subsample(path, tmp_path / "s.parquet", n_rows=5_000, seed=3, expected_rows=N_SOURCE)
    ).to_pandas()
    assert len(sample) == 5_000
    assert list(sample.columns) == [*FEATURES, *LABELS]
    expected_counts = full.value_counts(list(LABELS)) * (5_000 / N_SOURCE)
    sample_counts = sample.value_counts(list(LABELS))
    # Largest-remainder allocation: every stratum within one row of its proportional quota.
    gap = (sample_counts.reindex(expected_counts.index, fill_value=0) - expected_counts).abs()
    assert gap.max() < 1.0


def test_subsample_rows_come_from_source(source: tuple[Path, pd.DataFrame], tmp_path: Path) -> None:
    path, full = source
    sample = pq.read_table(build_subsample(path, tmp_path / "s.parquet", n_rows=500, seed=9))
    merged = sample.to_pandas().merge(full, on=[*FEATURES, *LABELS], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_row_count_mismatch_is_rejected(source: tuple[Path, pd.DataFrame], tmp_path: Path) -> None:
    path, _ = source
    with pytest.raises(ValueError, match="expected 13979592 rows"):
        build_subsample(path, tmp_path / "s.parquet", n_rows=10, seed=1, expected_rows=13_979_592)
    assert not (tmp_path / "s.parquet").exists()


@pytest.mark.parametrize("n_rows", [0, -1, 100, 101])
def test_stratified_indices_rejects_bad_sizes(n_rows: int) -> None:
    frame = pd.DataFrame({"t": [0, 1] * 50})
    with pytest.raises(ValueError, match="n_rows"):
        stratified_indices(frame, ["t"], n_rows=n_rows, seed=0)


def test_stratified_indices_sorted_unique() -> None:
    frame = pd.DataFrame({"t": [0] * 70 + [1] * 30, "y": [0, 1] * 50})
    idx = stratified_indices(frame, ["t", "y"], n_rows=33, seed=5)
    assert len(idx) == 33
    assert np.all(np.diff(idx) > 0)
