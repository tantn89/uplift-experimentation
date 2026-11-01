"""Seeded stratified sampling with exact proportional allocation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def stratified_indices(
    frame: pd.DataFrame, strata: list[str], *, n_rows: int, seed: int
) -> npt.NDArray[np.int64]:
    """Choose exactly n_rows row positions, allocated to strata in proportion to their size.

    Allocation uses the largest-remainder method, so the sample size is exact and each
    stratum's share is within one row of proportional. Strata are visited in sorted key order
    and drawn with a single seeded generator, which makes the result a pure function of
    (frame contents, strata, n_rows, seed). Positions are returned sorted ascending.
    """
    total = len(frame)
    if not 0 < n_rows < total:
        raise ValueError(f"n_rows must be in (0, {total}), got {n_rows}")

    grouped = frame.groupby(strata, sort=True, observed=True)
    # size() is indexed by stratum key in sorted order, which fixes the draw order for the seed.
    counts = grouped.size()
    keys = list(counts.index)
    groups = grouped.indices
    sizes = counts.to_numpy(dtype=np.int64)
    quotas = sizes * n_rows / total
    alloc = np.floor(quotas).astype(np.int64)
    shortfall = n_rows - int(alloc.sum())
    # Stable sort on the negated remainder breaks ties by stratum order, keeping it seed-free.
    order = np.argsort(-(quotas - alloc), kind="stable")
    alloc[order[:shortfall]] += 1

    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(np.asarray(groups[key], dtype=np.int64), size=int(k), replace=False)
        for key, k in zip(keys, alloc, strict=True)
        if k > 0
    ]
    return np.sort(np.concatenate(chosen))
