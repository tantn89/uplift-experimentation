"""Drift report end to end on synthetic data: scenario flags, outputs, traffic log and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from serving_fixtures import customers, export_model
from synthetic_uplift import synthetic_uplift_frame
from uplift_lab.monitoring.traffic_log import TrafficLog
from uplift_lab.reports import drift
from uplift_lab.reports.drift import write_drift_report
from uplift_lab.uplift.learners import LgbmParams

PARAMS = LgbmParams(n_estimators=30, min_child_samples=50)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_uplift_frame(30_000, seed=61)


@pytest.fixture(scope="module")
def report(frame: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("drift")
    payload = write_drift_report(frame, out, traffic_rows=8_000, params=PARAMS, seed=3)
    return {"payload": payload, "dir": out}


def test_scenarios_flag_the_expected_drift(report: dict[str, Any]) -> None:
    scenarios = report["payload"]["scenarios"]
    assert scenarios["no_shift"]["status"] == "ok"
    assert scenarios["no_shift"]["ks_rejected_features"] == []
    covariate = scenarios["covariate_shift"]
    assert covariate["status"] == "alert"
    assert set(covariate["flagged_features"]) == {"f0", "f6"}  # synthetic features are independent
    assert covariate["prediction"]["status"] == "alert"  # the effect rises with f0
    unshifted = scenarios["no_shift"]["treat_rate_at_budget"]
    assert unshifted == pytest.approx(0.1, abs=0.015)
    assert covariate["treat_rate_at_budget"] > 1.5 * unshifted  # the budget would overspend
    mix = scenarios["treatment_mix_shift"]
    assert mix["flagged_features"] == [] and mix["treated_share"] == pytest.approx(0.5, abs=0.02)
    prior = scenarios["outcome_prior_shift"]
    # Visits depend only weakly on f1 here, so a ~3x label shift stays below the PSI warn level
    # while KS, far more sensitive, already rejects on it.
    assert prior["visit_rate"] > 2.5 * scenarios["no_shift"]["visit_rate"]
    assert prior["flagged_features"] == [] and "f1" in prior["ks_rejected_features"]
    assert report["payload"]["overall_status"] == "alert"


def test_outputs_are_written_and_self_contained(report: dict[str, Any]) -> None:
    out: Path = report["dir"]
    saved = json.loads((out / "drift.json").read_text(encoding="utf-8"))
    assert saved["thresholds"] == {"psi_warn": 0.1, "psi_alert": 0.2, "ks_alpha": 0.01}
    assert saved["model"]["kind"] == "s_learner"
    assert (out / "drift-psi.png").stat().st_size > 10_000
    markdown = (out / "drift-report.md").read_text(encoding="utf-8")
    assert "| covariate_shift | ALERT | f0, f6 |" in markdown
    page = (out / "drift-report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in page and "<script" not in page and "http" not in page


def test_exported_model_and_traffic_log(frame: pd.DataFrame, tmp_path: Path) -> None:
    model_dir = export_model(tmp_path / "model")
    log_path = tmp_path / "traffic.jsonl"
    log = TrafficLog(log_path, sample_rate=1.0, max_rows=200, max_bytes=10**7, seed=1)
    for seed in range(10):
        rows = customers(200, seed=seed)
        log.record("score", rows, np.zeros(len(rows)))
    payload = write_drift_report(
        frame,
        tmp_path / "out",
        traffic_rows=2_000,
        model_dir=model_dir,
        traffic_log=log_path,
        figures=False,
        seed=3,
    )
    assert payload["model"]["kind"] == "t_learner"
    assert payload["model"]["trained_on_this_frame"] is False
    assert payload["decision"]["threshold"] == pytest.approx(
        json.loads((model_dir / "manifest.json").read_text())["metadata"]["budget_thresholds"][
            "0.1"
        ]
    )
    observed = payload["scenarios"]["observed_traffic"]
    assert observed["n"] == 2_000
    assert observed["prediction"]["status"] == "alert"  # constant logged scores
    assert not (tmp_path / "out" / "drift-psi.png").exists()


def test_cli_runs_on_synthetic_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--synthetic",
        "6000",
        "--traffic-rows",
        "1000",
        "--n-estimators",
        "15",
        "--no-figures",
        "--output-dir",
        str(tmp_path),
    ]
    assert drift.main(args) == 0
    out = capsys.readouterr().out
    assert "covariate_shift" in out and "overall" in out
    assert (tmp_path / "drift-report.html").exists()
    for bad in (["--covariate-features", "f99"], ["--psi-warn", "0.5"], ["--traffic-rows", "5"]):
        with pytest.raises(SystemExit):
            drift.main([*args, *bad])
