"""Hillstrom A/B report: ``python -m uplift_lab.reports.hillstrom --output-dir reports/hillstrom``.

Downloads the dataset once (checksummed), runs the analysis and writes ``hillstrom.json`` plus
two PNG figures. The analysis is deterministic; nothing in it is random.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_lab.data.hillstrom import load_hillstrom
from uplift_lab.reports.hillstrom_analysis import analyse_hillstrom
from uplift_lab.reports.output import MISSING_MATPLOTLIB, write_json


def write_hillstrom_report(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    alpha: float = 0.05,
    relative_mde: float = 0.10,
    figures: bool = True,
) -> dict[str, Any]:
    """Analyse ``frame`` and write the JSON and figures; returns the analysis payload."""
    analysis = analyse_hillstrom(frame, alpha=alpha, relative_mde=relative_mde)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "hillstrom.json", analysis)
    if figures:
        try:
            from uplift_lab.reports.hillstrom_figures import effects_figure, power_figure
        except ImportError as error:
            raise SystemExit(MISSING_MATPLOTLIB) from error
        effects_figure(analysis, output_dir / "hillstrom-effects.png")
        power_figure(analysis, output_dir / "hillstrom-power.png")
    return analysis


def _summary_lines(analysis: dict[str, Any]) -> list[str]:
    srm = analysis["srm"]
    lines = [f"SRM chi2={srm['statistic']:.3f} p={srm['p_value']:.4f} mismatch={srm['mismatch']}"]
    for arm, outcomes in analysis["effects"].items():
        for outcome, entry in outcomes.items():
            raw, cuped = entry["raw"], entry["cuped"]["pre_period"]
            adjusted = cuped["adjusted"]
            lines.append(
                f"{arm:14s} {outcome:10s} raw {raw['effect']:+.5f} "
                f"[{raw['ci_low']:+.5f}, {raw['ci_high']:+.5f}]  CUPED {adjusted['effect']:+.5f} "
                f"[{adjusted['ci_low']:+.5f}, {adjusted['ci_high']:+.5f}]  "
                f"var -{cuped['variance_reduction']:.2%}"
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hillstrom A/B analysis report.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hillstrom"))
    parser.add_argument("--cache", type=Path, default=None, help="dataset cache directory")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--relative-mde", type=float, default=0.10)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    analysis = write_hillstrom_report(
        load_hillstrom(args.cache),
        args.output_dir,
        alpha=args.alpha,
        relative_mde=args.relative_mde,
        figures=not args.no_figures,
    )
    print("\n".join(_summary_lines(analysis)))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
