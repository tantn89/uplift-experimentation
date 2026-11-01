"""Peeking figures: cumulative rejection by look and the mSPRT stopping-time distribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from uplift_lab.experiments.peeking_simulation import RULES, PeekingOutcome

_STYLE = {
    "naive_peeking": ("tab:red", "naive t-test, every look"),
    "msprt": ("tab:blue", "mSPRT (always valid)"),
    "fixed_horizon": ("tab:grey", "t-test, final look only"),
}


def peeking_figure(
    null: PeekingOutcome, alternative: PeekingOutcome, alpha: float, path: Path
) -> Path:
    """Cumulative share of experiments that have rejected, A/A (left) and A/B (right)."""
    figure = Figure(figsize=(11, 4), layout="constrained")
    axes = figure.subplots(1, 2)
    panels = (
        (null, "A/A (no effect): false-positive rate"),
        (alternative, f"A/B (lift {alternative.effect:+.4f}): power"),
    )
    for axis, (outcome, title) in zip(axes, panels, strict=True):
        for rule in RULES:
            colour, label = _STYLE[rule]
            axis.step(
                outcome.looks_n,
                outcome.cumulative_rejection(rule),
                where="post",
                color=colour,
                label=label,
            )
        axis.set_xlabel("Units per arm at look")
        axis.set_ylabel("Share of experiments rejected")
        axis.set_title(title)
    axes[0].axhline(alpha, color="black", linewidth=0.8, linestyle=":", label=f"alpha = {alpha}")
    axes[0].legend(loc="upper left", fontsize="small")
    figure.savefig(path, dpi=120)
    return path


def stopping_figure(alternative: PeekingOutcome, path: Path) -> Path:
    """Histogram of units per arm at stopping under the A/B scenario, rejecting runs only."""
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.subplots()
    edges = np.concatenate([[0], alternative.looks_n]) + alternative.looks_n[0] / 2
    for rule in ("naive_peeking", "msprt"):
        stopped = alternative.first_rejection[rule] >= 0
        colour, label = _STYLE[rule]
        axis.hist(
            alternative.stopping_n(rule)[stopped], bins=edges, alpha=0.5, color=colour, label=label
        )
    axis.set_xlabel("Units per arm when the test stopped and rejected")
    axis.set_ylabel("Simulated experiments")
    axis.set_title("Stopping time under the A/B scenario")
    axis.legend(fontsize="small")
    figure.savefig(path, dpi=120)
    return path
