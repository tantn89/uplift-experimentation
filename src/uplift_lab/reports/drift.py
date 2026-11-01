"""Drift report: ``uplift-drift-report [--synthetic N] [--model-dir DIR] [--traffic-log PATH]``.

Builds the payload with ``reports.drift_analysis.write_drift_report``: PSI/KS per feature and for
the uplift score, plus the decision rate at the 10% budget, for every seeded scenario. With
``--model-dir`` (e.g. ``reports/criteo-uplift/model``) the exported model is used; otherwise an
S-learner is fitted on the reference split. ``--traffic-log`` adds the sampled serving log as
an "observed" scenario. ``--synthetic N`` uses seeded synthetic data instead of Criteo (no
download; used in CI). Writes ``drift.json``, ``drift-psi.png``, ``drift-report.md`` and a
self-contained ``drift-report.html``.

``--expect SCENARIO=STATUS`` (repeatable) turns the report into a regression gate: the command
exits with 1 if a scenario's status differs, after writing all outputs. CI expects
``no_shift=ok`` and ``covariate_shift=alert`` on synthetic data.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from uplift_lab.data.criteo import DEFAULT_SEED, FEATURES, load_criteo_subsample
from uplift_lab.data.synthetic import synthetic_uplift_frame
from uplift_lab.monitoring.drift import STATUSES, DriftThresholds
from uplift_lab.monitoring.shift_simulation import SCENARIOS, ShiftConfig
from uplift_lab.reports.drift_analysis import DECISION_BUDGET, write_drift_report
from uplift_lab.uplift.learners import LgbmParams

__all__ = ["DECISION_BUDGET", "main", "write_drift_report"]

OBSERVED = "observed_traffic"


def _expectation(raw: str) -> tuple[str, str]:
    """Parse ``SCENARIO=STATUS`` for ``--expect``."""
    scenario, _, status = raw.partition("=")
    if scenario not in (*SCENARIOS, OBSERVED) or status not in STATUSES:
        raise argparse.ArgumentTypeError(
            f"expected SCENARIO=STATUS with SCENARIO in {[*SCENARIOS, OBSERVED]} and STATUS in "
            f"{list(STATUSES)}, got {raw!r}"
        )
    return scenario, status


def unmet_expectations(payload: dict[str, Any], expected: Sequence[tuple[str, str]]) -> list[str]:
    """One message per expected scenario status that the payload does not show."""
    problems = []
    for scenario, status in expected:
        summary = payload["scenarios"].get(scenario)
        if summary is None:
            problems.append(f"{scenario}: not in this report")
        elif summary["status"] != status:
            problems.append(f"{scenario}: status {summary['status']}, expected {status}")
    return problems


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature, prediction and decision-rate drift.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/drift"))
    parser.add_argument("--synthetic", type=int, default=None, metavar="N", help="synthetic rows")
    parser.add_argument("--cache", type=Path, default=None, help="dataset cache directory")
    parser.add_argument("--model-dir", type=Path, default=None, help="exported model directory")
    parser.add_argument("--traffic-log", type=Path, default=None, help="serving traffic JSONL")
    parser.add_argument("--traffic-rows", type=int, default=50_000)
    parser.add_argument(
        "--covariate-features", nargs="+", default=list(ShiftConfig().covariate_features)
    )
    parser.add_argument("--strength", type=float, default=ShiftConfig().strength)
    defaults = DriftThresholds()
    parser.add_argument("--psi-warn", type=float, default=defaults.psi_warn)
    parser.add_argument("--psi-alert", type=float, default=defaults.psi_alert)
    parser.add_argument("--ks-alpha", type=float, default=defaults.ks_alpha)
    for name, bounds in (("warn", defaults.rate_warn), ("alert", defaults.rate_alert)):
        parser.add_argument(
            f"--rate-{name}",
            type=float,
            nargs=2,
            default=list(bounds),
            metavar=("LOW", "HIGH"),
            help=f"{name} when treat rate / budget leaves [LOW, HIGH] (default {bounds})",
        )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=int, default=LgbmParams().n_estimators)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--expect",
        type=_expectation,
        action="append",
        default=[],
        metavar="SCENARIO=STATUS",
        help="exit 1 unless SCENARIO has STATUS (repeatable), e.g. no_shift=ok",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        shift = ShiftConfig(
            covariate_features=tuple(args.covariate_features), strength=args.strength
        )
        thresholds = DriftThresholds(
            args.psi_warn,
            args.psi_alert,
            args.ks_alpha,
            rate_warn=(args.rate_warn[0], args.rate_warn[1]),
            rate_alert=(args.rate_alert[0], args.rate_alert[1]),
        )
        params = LgbmParams(n_estimators=args.n_estimators, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.traffic_rows < 100
        or args.bins < 2
        or (args.synthetic is not None and args.synthetic < 1000)
    ):
        parser.error("--traffic-rows >= 100, --bins >= 2 and --synthetic >= 1000")
    unknown = sorted(set(shift.covariate_features) - set(FEATURES))
    if unknown:
        parser.error(f"unknown covariate features {unknown}")
    if args.synthetic is not None:
        frame = synthetic_uplift_frame(args.synthetic, seed=args.seed)
    else:
        frame = load_criteo_subsample(cache=args.cache)
    payload = write_drift_report(
        frame,
        args.output_dir,
        seed=args.seed,
        traffic_rows=args.traffic_rows,
        shift=shift,
        thresholds=thresholds,
        n_bins=args.bins,
        model_dir=args.model_dir,
        params=params,
        traffic_log=args.traffic_log,
        figures=not args.no_figures,
    )
    for name, summary in payload["scenarios"].items():
        prediction = summary["prediction"]["psi"] if summary["prediction"] else float("nan")
        rate = summary["decision_rate"]
        print(
            f"{name:20s} {summary['status']:5s} flagged {summary['flagged_features'] or '-'} "
            f"prediction PSI {prediction:.3f} treat rate {rate['treat_rate']:.3f} "
            f"({rate['ratio']:.2f}x budget, {rate['status']})"
        )
    print(
        f"overall {payload['overall_status']}; wrote {args.output_dir} "
        f"in {payload['seconds']:.1f} s"
    )
    problems = unmet_expectations(payload, args.expect)
    for problem in problems:
        print(f"drift expectation failed: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
