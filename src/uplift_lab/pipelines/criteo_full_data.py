"""Load the complete Criteo Uplift v2.1 file (13,979,592 rows) from an explicit, verified path.

The local reports build a seeded subsample from the cache; a full-data job instead points at a
copy of the same pinned file (for example on a Databricks Unity Catalog Volume). The bytes are
checked against the pinned SHA-256 before parsing, so a run on any platform is traceable to the
exact source the local runs use. Columns and types match the subsample: float64 features
``f0``-``f11`` and int8 labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from uplift_lab.data.cache import RemoteFile, verify
from uplift_lab.data.criteo import CRITEO, EXPECTED_ROWS, FEATURES, LABELS

_COLUMN_TYPES: dict[str, pa.DataType] = {
    **{name: pa.float64() for name in FEATURES},
    **{name: pa.int8() for name in LABELS},
}


@dataclass(frozen=True)
class SourceCheck:
    """What was verified about the source file before it was parsed."""

    path: str
    size_bytes: int
    sha256: str
    verified: bool


def check_source(
    path: Path, remote: RemoteFile = CRITEO, *, verify_checksum: bool = True
) -> SourceCheck:
    """Confirm ``path`` is the pinned file: size first (cheap), then the SHA-256.

    Raises ``FileNotFoundError`` for a missing file, ``ValueError`` for a size mismatch and
    ``ChecksumError`` for a content mismatch.
    """
    if not path.is_file():
        raise FileNotFoundError(f"source file not found: {path}")
    size = path.stat().st_size
    if size != remote.size_bytes:
        raise ValueError(f"{path.name}: {size} bytes, pinned file has {remote.size_bytes}")
    if verify_checksum:
        verify(path, remote.sha256)
    return SourceCheck(str(path), size, remote.sha256, verify_checksum)


def read_criteo_csv(path: Path, *, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Parse a Criteo-format CSV (optionally gzipped) into features plus validated binary labels."""
    options = pacsv.ConvertOptions(include_columns=[*FEATURES, *LABELS], column_types=_COLUMN_TYPES)
    table = pacsv.read_csv(path, convert_options=options)
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {table.num_rows}")
    frame: pd.DataFrame = table.to_pandas()
    del table
    for name in LABELS:
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be binary")
    return frame
