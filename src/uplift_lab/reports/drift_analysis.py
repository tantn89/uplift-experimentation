"""Build the drift payload and write its outputs; the CLI lives in ``reports.drift``.

The reference is the training split of the same seeded 70/30 split the uplift report uses
(features plus the model's uplift scores). Simulated traffic is drawn from the held-out split
under each scenario in ``monitoring.shift_simulation`` and scored by the same model. With a
``model_dir`` the exported model and its manifest budget thresholds are used, and the payload
records whether its training data checksum matches this frame. Otherwise an S-learner is fitted
on the training split and the threshold is the held-out score quantile. Every scenario also
gets the decision-rate check at budget ``DECISION_BUDGET``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_lab.data.criteo import DEFAULT_SEED, FEATURES
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
    decision = (float(DECISION_BUDGET), float(cut))

    scenarios: dict[str, Any] = {}
    for name, traffic in all_scenarios(test, traffic_rows, seed=seed, config=shift).items():
        scored = traffic[list(FEATURES)].assign(**{PREDICTION: learner.predict(traffic)})
        summary = drift_summary(profile, scored, thresholds, decision)
        summary["description"] = shift.describe(name)
        summary["treated_share"] = float(traffic["treatment"].mean())
        summary[f"{shift.outcome}_rate"] = float(traffic[shift.outcome].mean())
        scenarios[name] = summary
    if traffic_log is not None:
        observed = read_traffic_log(traffic_log, FEATURES)
        summary = drift_summary(profile, observed, thresholds, decision)
        summary["description"] = f"sampled serving traffic from {traffic_log}"
        scenarios["observed_traffic"] = summary

    payload: dict[str, Any] = {
        "reference": {
            "n": len(train),
            "n_held_out": len(test),
            "frame_sha256": frame_sha256(frame),
        },
        "model": {key: v for key, v in model.items() if key != "thresholds"}
        | {"kind": learner.kind},
        "decision": {"budget": DECISION_BUDGET, "threshold": decision[1]},
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
