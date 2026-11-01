"""Criteo Uplift Prediction dataset v2.1 and its seeded, stratified local subsample.

13,979,592 users from several incrementality tests (an RCT where a random part of the
population is held out from advertising), with twelve anonymised dense features f0-f11, a
treatment flag, an exposure flag and two outcomes (visit, conversion).

Licence: CC BY-NC-SA 4.0 (Criteo AI Lab). Neither the raw file nor any subsample derived from
it is committed here: a subsample is an adaptation, and redistributing it would carry the same
non-commercial, share-alike terms. Each user builds it locally from the verified source.

PyArrow (the ``analysis`` extra) is imported only when a subsample is built or read, so the
serving image can import ``FEATURES`` from here without it.

The subsample is built in two streaming passes so memory stays small: the first reads only
the four binary label columns to choose rows, the second streams the CSV and keeps those rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uplift_lab.data.cache import RemoteFile, cache_dir, fetch
from uplift_lab.data.sampling import stratified_indices

CRITEO = RemoteFile(
    url=(
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/"
        "2424920019e49d52d72c13ac1143ec5d53af276b/criteo-research-uplift-v2.1.csv.gz"
    ),
    filename="criteo-research-uplift-v2.1.csv.gz",
    sha256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    size_bytes=311_422_618,
)

EXPECTED_ROWS = 13_979_592
FEATURES = tuple(f"f{i}" for i in range(12))
LABELS = ("treatment", "exposure", "visit", "conversion")
DEFAULT_SUBSAMPLE_ROWS = 1_000_000
DEFAULT_SEED = 20261101


def _read_labels(source: Path) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    options = pacsv.ConvertOptions(
        include_columns=list(LABELS), column_types={name: pa.int8() for name in LABELS}
    )
    labels: pd.DataFrame = pacsv.read_csv(source, convert_options=options).to_pandas()
    for name in LABELS:
        if not labels[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be binary")
    return labels


def build_subsample(
    source: Path,
    dest: Path,
    *,
    n_rows: int,
    seed: int,
    expected_rows: int | None = None,
) -> Path:
    """Write a stratified subsample of a Criteo-format CSV (optionally gzipped) to Parquet.

    Strata are the joint values of treatment, exposure, visit and conversion, so the sample
    keeps the population's treatment ratio and its rare conversion rate per arm. The same
    source bytes, n_rows and seed always yield the same rows in the same (source) order.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    labels = _read_labels(source)
    if expected_rows is not None and len(labels) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(labels)}")
    keep = stratified_indices(labels, list(LABELS), n_rows=n_rows, seed=seed)

    column_types = {
        **{name: pa.float64() for name in FEATURES},
        **{name: pa.int8() for name in LABELS},
    }
    options = pacsv.ConvertOptions(include_columns=[*FEATURES, *LABELS], column_types=column_types)
    batches: list[pa.RecordBatch] = []
    offset = 0
    with pacsv.open_csv(source, convert_options=options) as reader:
        schema = reader.schema
        for batch in reader:
            lo, hi = np.searchsorted(keep, [offset, offset + batch.num_rows])
            if hi > lo:
                batches.append(batch.take(pa.array(keep[lo:hi] - offset)))
            offset += batch.num_rows
    if offset != len(labels):
        raise ValueError("source changed between passes")

    table = pa.Table.from_batches(batches, schema=schema)
    tmp = dest.with_name(dest.name + ".part")
    try:
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def subsample_path(n_rows: int, seed: int, cache: Path | str | None = None) -> Path:
    """Cache location of a subsample; the name encodes everything that determines its rows."""
    return cache_dir(cache) / f"criteo-uplift-v2.1-n{n_rows}-seed{seed}.parquet"


def load_criteo_subsample(
    n_rows: int = DEFAULT_SUBSAMPLE_ROWS,
    seed: int = DEFAULT_SEED,
    cache: Path | str | None = None,
) -> pd.DataFrame:
    """Load the seeded subsample, building it from the verified raw file on first use."""
    import pyarrow.parquet as pq

    dest = subsample_path(n_rows, seed, cache)
    if not dest.exists():
        source = fetch(CRITEO, cache)
        build_subsample(source, dest, n_rows=n_rows, seed=seed, expected_rows=EXPECTED_ROWS)
    frame: pd.DataFrame = pq.read_table(dest).to_pandas()
    return frame
