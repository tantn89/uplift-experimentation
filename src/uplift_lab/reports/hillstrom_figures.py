"""Hillstrom figures from the analysis payload (object-oriented matplotlib, no pyplot state)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from uplift_lab.reports.hillstrom_analysis import OUTCOMES, POWER_SET, TREATED_ARMS

_LABELS = {
    "visit": "Visit rate",
    "conversion": "Conversion rate",
    "spend": "Spend per customer ($)",
}


def effects_figure(analysis: dict[str, Any], path: Path) -> Path:
    """Forest plot: effect and CI per arm, unadjusted versus CUPED, one panel per outcome."""
    figure = Figure(figsize=(12, 3.8), layout="constrained")
    axes = figure.subplots(1, len(OUTCOMES))
    for axis, outcome in zip(axes, OUTCOMES, strict=True):
        rows: list[tuple[str, dict[str, float]]] = []
        for arm in TREATED_ARMS:
            entry = analysis["effects"][arm][outcome]
            short = arm.split()[0]
            rows.append((f"{short} raw", entry["raw"]))
            rows.append((f"{short} CUPED", entry["cuped"][POWER_SET]["adjusted"]))
        for position, (label, estimate) in enumerate(reversed(rows)):
            colour = "tab:blue" if "CUPED" in label else "tab:grey"
            axis.errorbar(
                estimate["effect"],
                position,
                xerr=[
                    [estimate["effect"] - estimate["ci_low"]],
                    [estimate["ci_high"] - estimate["effect"]],
                ],
                fmt="o",
                color=colour,
                capsize=4,
            )
        axis.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)])
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_title(_LABELS[outcome])
        axis.set_xlabel("Effect vs No E-Mail (95% CI)")
    figure.suptitle("Hillstrom e-mail campaign: difference in means, unadjusted vs CUPED")
    figure.savefig(path, dpi=120)
    return path


def power_figure(analysis: dict[str, Any], path: Path) -> Path:
    """Power against sample size per arm at the configured relative MDE, raw versus CUPED."""
    figure = Figure(figsize=(12, 3.8), layout="constrained")
    axes = figure.subplots(1, len(OUTCOMES))
    for axis, outcome in zip(axes, OUTCOMES, strict=True):
        power = analysis["power"][outcome]
        curve = power["curve"]
        axis.plot(curve["n_per_arm"], curve["power_raw"], color="tab:grey", label="unadjusted")
        axis.plot(curve["n_per_arm"], curve["power_cuped"], color="tab:blue", label="CUPED")
        axis.axhline(analysis["power_target"], color="black", linewidth=0.8, linestyle=":")
        axis.axvline(
            power["mde_at_n"]["n_per_arm"], color="tab:red", linewidth=0.8, label="Hillstrom n"
        )
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.0)
        relative = power["required_n_per_arm"]["relative_mde"]
        axis.set_title(f"{_LABELS[outcome]}: {relative:.0%} relative lift")
        axis.set_xlabel("Units per arm")
        axis.set_ylabel("Power")
        axis.legend(loc="lower right", fontsize="small")
    figure.suptitle("Power of a two-sided test at alpha = 0.05")
    figure.savefig(path, dpi=120)
    return path
