"""Hillstrom parsing and validation on a synthetic file; real download only under -m network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from uplift_lab.data.hillstrom import ARMS, load_hillstrom, read_hillstrom_csv

HEADER = (
    "recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,"
    "visit,conversion,spend"
)
ROWS = [
    "10,2) $100 - $200,142.44,1,0,Surburban,0,Phone,Womens E-Mail,0,0,0",
    "6,3) $200 - $350,329.08,1,1,Rural,1,Web,No E-Mail,0,0,0",
    "7,2) $100 - $200,180.65,0,1,Surburban,1,Web,Mens E-Mail,1,1,29.99",
]


def _write(path: Path, rows: list[str], header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_parses_schema_and_arm_order(tmp_path: Path) -> None:
    frame = read_hillstrom_csv(_write(tmp_path / "h.csv", ROWS), expected_rows=3)
    assert list(frame["segment"].cat.categories) == list(ARMS)
    assert frame["spend"].sum() == pytest.approx(29.99)
    assert str(frame["conversion"].dtype) == "int8"


def test_rejects_wrong_row_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 64000 rows"):
        read_hillstrom_csv(_write(tmp_path / "h.csv", ROWS))


def test_rejects_unknown_arm(tmp_path: Path) -> None:
    rows = [*ROWS, "1,1) $0 - $100,50.0,1,0,Urban,0,Web,SMS,0,0,0"]
    with pytest.raises(ValueError, match="unexpected segment"):
        read_hillstrom_csv(_write(tmp_path / "h.csv", rows), expected_rows=None)


def test_rejects_non_binary_outcome(tmp_path: Path) -> None:
    rows = [*ROWS, "1,1) $0 - $100,50.0,1,0,Urban,0,Web,No E-Mail,2,0,0"]
    with pytest.raises(ValueError, match="visit must be binary"):
        read_hillstrom_csv(_write(tmp_path / "h.csv", rows), expected_rows=None)


def test_rejects_missing_column(tmp_path: Path) -> None:
    header = HEADER.replace(",spend", "")
    rows = [row.rsplit(",", 1)[0] for row in ROWS]
    with pytest.raises(ValueError, match="missing columns"):
        read_hillstrom_csv(_write(tmp_path / "h.csv", rows, header), expected_rows=None)


@pytest.mark.network
def test_real_download_matches_published_shape() -> None:
    frame = load_hillstrom()
    assert len(frame) == 64_000
    counts = frame["segment"].value_counts()
    assert isinstance(counts, pd.Series)
    assert set(counts.index) == set(ARMS)
