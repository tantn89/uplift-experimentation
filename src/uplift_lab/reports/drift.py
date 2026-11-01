"""Drift report: ``uplift-drift-report [--synthetic N] [--model-dir DIR] [--traffic-log PATH]``.

The reference is the training split of the same seeded 70/30 split the uplift report uses
(features plus the model's uplift scores). Simulated traffic is drawn from the held-out split
under each scenario in ``monitoring.shift_simulation`` and scored by the same model. With
``--model-dir`` (e.g. ``reports/criteo-uplift/model``) the exported model is used, and the JSON
records whether its training data checksum matches this frame. Otherwise an S-learner is fitted
on the training split. ``--traffic-log`` adds the sampled serving log as an "observed"
scenario. ``--synthetic N`` uses seeded synthetic data instead of Criteo (no download; used in
CI). Writes ``drift.json``, ``drift-psi.png``, ``drift-report.md`` and a self-contained
``drift-report.html``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uplift_lab.data.criteo import DEFAULT_SEED, FEATURES, load_criteo_subsample
from uplift_lab.data.synthetic import synthetic_uplift_frame
from uplift_lab.monitoring.drift import (
    PREDICTION,
    DriftThresholds,
    ReferenceProfile,
    drift_summary,
    worst_status,
)
from uplift_lab.monitoring.shift_simulation import ShiftConfig, all_scenarios
from uplift_lab.monitoring.traffic_log import read_traffic_log
from uplift_lab.reports.drift_render import write_html, write_markdown
from uplift_lab.reports.output import MISSING_MATPLOTLIB, write_json
from uplift_lab.uplift.artifacts import load_learner, read_manifest
from uplift_lab.uplift.learners import LgbmParams, UpliftLearner, make_learner
from uplift_lab.uplift.pipeline import budget_thresholds, split_train_test
from uplift_lab.uplift.provenance import frame_sha256

DECISION_BUDGET = "0.1"


def _model(
    model_dir: Path | None,
    train: pd.DataFrame,
    frame: pd.DataFrame,
    params: LgbmParams,
    outcome: str,
) -> tuple[UpliftLearner, dict[str, Any]]:
    if model_dir is None:
        learner = make_learner("s_learner", params).fit(
            train[list(FEATURES)], train["treatment"], train[outcome]
        )
        return learner, {"source": "fitted s_learner on the reference split", "thresholds": {}}
    metadata = read_manifest(model_dir).get("metadata", {})
    trained_sha = metadata.get("dataset", {}).get("frame_sha256")
    info = {
        "source": str(model_dir),
        "thresholds": metadata.get("budget_thresholds", {}),
        "trained_on_this_frame": trained_sha == frame_sha256(frame),
    }
    return load_learner(model_dir), info


def write_drift_report(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    traffic_rows: int = 50_000,
    shift: ShiftConfig | None = None,
    thresholds: DriftThresholds | None = None,
    n_bins: int = 10,
    model_dir: Path | None = None,
    params: LgbmParams | None = None,
    traffic_log: Path | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    """Profile the reference, score every scenario and write all outputs; returns the payload."""
    shift, thresholds = shift or ShiftConfig(), thresholds or DriftThresholds()
    start = time.perf_counter()
    train, test = split_train_test(frame, shift.outcome, test_size=0.3, seed=seed)
    learner, model = _model(model_dir, train, frame, params or LgbmParams(seed=seed), shift.outcome)
    reference = train[list(FEATURES)].assign(**{PREDICTION: learner.predict(train)})
    profile = ReferenceProfile.fit(reference, [*FEATURES, PREDICTION], n_bins)
    cut = model["thresholds"].get(DECISION_BUDGET)
    if cut is None:
        cut = budget_thresholds(learner.predict(test), [float(DECISION_BUDGET)])[DECISION_BUDGET]

    scenarios: dict[str, Any] = {}
    for name, traffic in all_scenarios(test, traffic_rows, seed=seed, config=shift).items():
        scored = traffic[list(FEATURES)].assign(**{PREDICTION: learner.predict(traffic)})
        summary = drift_summary(profile, scored, thresholds)
        summary["description"] = shift.describe(name)
        summary["treated_share"] = float(traffic["treatment"].mean())
        summary[f"{shift.outcome}_rate"] = float(traffic[shift.outcome].mean())
        summary["treat_rate_at_budget"] = float(np.mean(scored[PREDICTION] >= cut))
        scenarios[name] = summary
    if traffic_log is not None:
        observed = read_traffic_log(traffic_log, FEATURES)
        summary = drift_summary(profile, observed, thresholds)
        summary["description"] = f"sampled serving traffic from {traffic_log}"
        summary["treat_rate_at_budget"] = float(np.mean(observed[PREDICTION] >= cut))
        scenarios["observed_traffic"] = summary

    payload: dict[str, Any] = {
        "reference": {
            "n": len(train),
            "n_held_out": len(test),
            "frame_sha256": frame_sha256(frame),
        },
        "model": {key: v for key, v in model.items() if key != "thresholds"}
        | {"kind": learner.kind},
        "decision": {"budget": DECISION_BUDGET, "threshold": float(cut)},
        "thresholds": thresholds.to_dict(),
        "config": {"seed": seed, "traffic_rows": traffic_rows, "n_bins": n_bins, **vars(shift)},
        "overall_status": worst_status(scenarios),
        "scenarios": scenarios,
        "seconds": time.perf_counter() - start,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = None
    if figures:
        try:
            from uplift_lab.reports.drift_figures import psi_figure
        except ImportError as error:
            raise SystemExit(MISSING_MATPLOTLIB) from error
        figure = psi_figure(payload, output_dir / "drift-psi.png")
    write_json(output_dir / "drift.json", payload)
    write_markdown(payload, output_dir / "drift-report.md", figure)
    write_html(payload, output_dir / "drift-report.html", figure)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature and prediction drift report.")
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
    parser.add_argument("--psi-warn", type=float, default=DriftThresholds().psi_warn)
    parser.add_argument("--psi-alert", type=float, default=DriftThresholds().psi_alert)
    parser.add_argument("--ks-alpha", type=float, default=DriftThresholds().ks_alpha)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=int, default=LgbmParams().n_estimators)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        shift = ShiftConfig(
            covariate_features=tuple(args.covariate_features), strength=args.strength
        )
        thresholds = DriftThresholds(args.psi_warn, args.psi_alert, args.ks_alpha)
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
        print(
            f"{name:20s} {summary['status']:5s} flagged {summary['flagged_features'] or '-'} "
            f"prediction PSI {prediction:.3f}"
        )
    print(
        f"overall {payload['overall_status']}; wrote {args.output_dir} "
        f"in {payload['seconds']:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
