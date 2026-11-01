"""Drift figure: PSI per feature and for the uplift score, one bar group per scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from uplift_lab.monitoring.drift import EPSILON, PREDICTION


def psi_figure(payload: dict[str, Any], path: Path) -> Path:
    """Grouped bars on a log scale with the warn and alert thresholds as horizontal lines."""
    scenarios = payload["scenarios"]
    columns = [*next(iter(scenarios.values()))["features"], PREDICTION]
    x = np.arange(len(columns))
    width = 0.8 / len(scenarios)
    figure = Figure(figsize=(11, 4.8), layout="constrained")
    axis = figure.subplots()
    for i, (name, summary) in enumerate(scenarios.items()):
        values = [summary["features"][c]["psi"] for c in columns[:-1]]
        values.append(summary["prediction"]["psi"] if summary["prediction"] else np.nan)
        axis.bar(
            x + (i - (len(scenarios) - 1) / 2) * width,
            np.maximum(values, EPSILON),
            width,
            label=f"{name} ({summary['status']})",
        )
    thresholds = payload["thresholds"]
    for value, label, color in (
        (thresholds["psi_warn"], "warn", "tab:orange"),
        (thresholds["psi_alert"], "alert", "tab:red"),
    ):
        axis.axhline(
            value, color=color, linestyle="--", linewidth=1, label=f"PSI {label} {value:g}"
        )
    axis.set_yscale("log")
    axis.set_ylim(EPSILON / 2, 2.0)
    axis.set_xticks(x, [*columns[:-1], "uplift score"])
    axis.set_ylabel(f"PSI vs training reference (log scale, floor {EPSILON:g})")
    axis.set_title("Feature and prediction drift on simulated traffic")
    axis.legend(fontsize="small", ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    figure.savefig(path, dpi=120)
    return path
