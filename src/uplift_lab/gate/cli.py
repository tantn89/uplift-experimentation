"""``uplift-metric-gate``: fail CI when uplift quality drops against the committed baseline.

    uplift-metric-gate                      # train on seeded synthetic data, compare, exit 1 on
                                            # a regression, 2 if the baseline is not comparable
    uplift-metric-gate --update-baseline    # rewrite the committed baseline from this run

The head payload and a Markdown summary go to ``--output-dir`` (``reports/metric-gate``).
See ``uplift_lab.gate.metric_gate`` for the rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from uplift_lab.gate.metric_gate import QINI, BaselineMismatch, GateResult, compare, gate_metrics
from uplift_lab.reports.output import write_json

DEFAULT_BASELINE = Path("evals/metric-gate-baseline.json")


def _read_baseline(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BaselineMismatch(
            f"{path} does not exist; create it with uplift-metric-gate --update-baseline"
        ) from error
    except json.JSONDecodeError as error:
        raise BaselineMismatch(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise BaselineMismatch(f"{path} must hold a JSON object")
    return payload


def render_markdown(result: GateResult) -> str:
    """The verdict and one row per learner and metric, for a CI job summary."""
    lines = [
        f"## uplift metric gate: {result.verdict}",
        "",
        "Fails when a head estimate drops below the baseline's lower 95% bootstrap bound, or "
        "a learner's Qini interval no longer excludes 0.",
        "",
        "| Learner | Metric | Baseline [95% CI] | Head | Status |",
        "|---|---|---|---|---|",
    ]
    for f in result.findings:
        digits = 5 if f.metric == QINI else 1
        base = (
            f"{f.baseline['estimate']:.{digits}f} "
            f"[{f.baseline['ci_low']:.{digits}f}, {f.baseline['ci_high']:.{digits}f}]"
            if f.baseline
            else "-"
        )
        head = f"{f.head:.{digits}f}" if f.head is not None else "-"
        lines.append(f"| {f.learner} | {f.metric} | {base} | {head} | {f.status} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uplift Qini / policy-value regression gate.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metric-gate"))
    parser.add_argument(
        "--update-baseline", action="store_true", help="write this run as the new baseline"
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    head = gate_metrics()
    seconds = time.perf_counter() - start
    write_json(args.output_dir / "head.json", head)
    if args.update_baseline:
        write_json(args.baseline, head)
        print(f"wrote baseline {args.baseline} in {seconds:.1f} s")
        return 0
    try:
        result = compare(_read_baseline(args.baseline), head)
    except (BaselineMismatch, KeyError, TypeError) as error:
        print(f"metric gate cannot compare: {error}", file=sys.stderr)
        return 2
    summary = render_markdown(result)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"verdict {result.verdict} in {seconds:.1f} s")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
