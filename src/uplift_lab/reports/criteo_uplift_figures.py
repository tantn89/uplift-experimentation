"""Criteo uplift figures from the report payload (object-oriented matplotlib, no pyplot state)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from uplift_lab.uplift.evaluation import RANDOM_SCORER


def qini_figure(evaluation: dict[str, Any], path: Path) -> Path:
    """Qini curves per model against the random-targeting diagonal."""
    figure = Figure(figsize=(7.5, 5), layout="constrained")
    axis = figure.subplots()
    for name, curve in evaluation["curves"].items():
        qini = evaluation["models"][name]["qini_coefficient"]
        style = ":" if name == RANDOM_SCORER else "-"
        label = (
            f"{name} (Qini {qini['estimate']:.5f} [{qini['ci_low']:.5f}, {qini['ci_high']:.5f}])"
        )
        axis.plot(curve["fraction"], curve["qini"], linestyle=style, label=label)
    end = next(iter(evaluation["curves"].values()))["qini"][-1]
    axis.plot([0.0, 1.0], [0.0, end], color="black", linewidth=0.8, label="random targeting")
    axis.set_xlabel("Share of users targeted (ranked by predicted uplift)")
    axis.set_ylabel("Incremental visits per user (Qini)")
    axis.set_title("Criteo visit uplift: Qini curves on the test set")
    axis.legend(fontsize="small")
    figure.savefig(path, dpi=120)
    return path


def deciles_figure(evaluation: dict[str, Any], path: Path) -> Path:
    """Observed treated-minus-control visit rate per decile of predicted uplift, per model."""
    names = list(evaluation["deciles"])
    columns = min(3, len(names))
    rows = -(-len(names) // columns)
    figure = Figure(figsize=(4.2 * columns, 3.4 * rows), layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows, columns, sharey=True)).ravel()
    ate = evaluation["ate"]["estimate"]
    for axis, name in zip(axes, names, strict=False):
        table = evaluation["deciles"][name]
        deciles = [row["decile"] for row in table]
        observed = [row["observed_uplift"] for row in table]
        predicted = [row["mean_predicted"] for row in table]
        axis.bar(deciles, observed, color="tab:blue", label="observed")
        if name != RANDOM_SCORER:  # random scores are draws in [0, 1], not predicted uplift
            axis.plot(deciles, predicted, "o-", color="tab:orange", label="mean predicted")
        axis.axhline(ate, color="black", linewidth=0.8, linestyle=":", label="ATE")
        axis.set_title(name)
        axis.set_xlabel("Decile of predicted uplift (1 = highest)")
        axis.set_xticks(deciles)
    for axis in axes[len(names) :]:
        axis.set_visible(False)
    axes[0].set_ylabel("Visit rate, treated - control")
    axes[0].legend(fontsize="small")
    figure.suptitle("Uplift by decile on the test set")
    figure.savefig(path, dpi=120)
    return path
