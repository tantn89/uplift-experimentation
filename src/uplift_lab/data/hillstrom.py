"""Kevin Hillstrom's MineThatData E-Mail Analytics challenge dataset (2008).

64,000 customers who bought in the last twelve months, randomised into three arms for a
two-week e-mail campaign: Mens E-Mail, Womens E-Mail and No E-Mail (control). Outcomes are
visit, conversion and spend in the two weeks after the send; recency, history and the other
columns describe the customer before the campaign, so they are valid CUPED covariates.

The original host serves the file over plain HTTP only (its HTTPS certificate is invalid), so
integrity rests on the pinned SHA-256.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uplift_lab.data.cache import RemoteFile, fetch

HILLSTROM = RemoteFile(
    url=(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    filename="hillstrom-2008-03-20.csv",
    sha256="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
    size_bytes=3_964_977,
)

CONTROL_ARM = "No E-Mail"
ARMS = ("Mens E-Mail", "Womens E-Mail", CONTROL_ARM)
EXPECTED_ROWS = 64_000

_DTYPES: dict[str, str] = {
    "recency": "int16",
    "history_segment": "category",
    "history": "float64",
    "mens": "int8",
    "womens": "int8",
    "zip_code": "category",
    "newbie": "int8",
    "channel": "category",
    "segment": "category",
    "visit": "int8",
    "conversion": "int8",
    "spend": "float64",
}


def read_hillstrom_csv(path: Path, *, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Parse and validate a Hillstrom CSV: schema, arm labels, binary outcomes and row count."""
    frame = pd.read_csv(path, dtype=_DTYPES)
    missing = set(_DTYPES) - set(frame.columns)
    if missing:
        raise ValueError(f"Hillstrom file is missing columns: {sorted(missing)}")
    unknown_arms = set(frame["segment"].unique()) - set(ARMS)
    if unknown_arms:
        raise ValueError(f"unexpected segment values: {sorted(unknown_arms)}")
    for column in ("visit", "conversion"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary")
    if (frame["spend"] < 0).any():
        raise ValueError("spend must be non-negative")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(frame)}")
    frame["segment"] = frame["segment"].cat.set_categories(list(ARMS))
    return frame[list(_DTYPES)]


def load_hillstrom(cache: Path | str | None = None) -> pd.DataFrame:
    """Download (once), verify and load the Hillstrom dataset."""
    return read_hillstrom_csv(fetch(HILLSTROM, cache))
