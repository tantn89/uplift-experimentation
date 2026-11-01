"""MLflow configuration, metric flattening and a full log-and-register run in a temp store."""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from synthetic_uplift import synthetic_uplift_frame
from uplift_lab.uplift.artifacts import save_learner
from uplift_lab.uplift.learners import LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, run_uplift
from uplift_lab.uplift.provenance import build_info, frame_sha256, git_state, job_git_state
from uplift_lab.uplift.tracking import (
    CHAMPION_ALIAS,
    MAX_RUN_METRICS,
    PYFUNC_REQUIREMENTS,
    TrackingConfig,
    code_paths,
    evaluation_metrics,
    flatten_metrics,
    outcome_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_defaults_to_local_sqlite(tmp_path: Path) -> None:
    config = TrackingConfig.from_env(tmp_path / "mlruns", env={})
    assert config.tracking_uri.startswith("sqlite:///")
    assert config.tracking_uri.endswith("/mlruns/mlflow.db")
    assert config.artifact_root is not None and config.artifact_root.startswith("file:")
    assert (config.registry_uri, config.experiment) == (None, "uplift-criteo")


def test_config_reads_databricks_style_environment(tmp_path: Path) -> None:
    env = {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_REGISTRY_URI": "databricks-uc",
        "UPLIFT_LAB_MLFLOW_EXPERIMENT": "/Users/me/uplift-criteo",
        "UPLIFT_LAB_MODEL_NAME": "main.uplift.criteo_visit",
    }
    config = TrackingConfig.from_env(tmp_path / "unused", env=env)
    expected = TrackingConfig(
        "databricks", "databricks-uc", "/Users/me/uplift-criteo", "main.uplift.criteo_visit"
    )
    assert config == expected
    assert not (tmp_path / "unused").exists()


def test_flatten_metrics_and_provenance() -> None:
    nested = {"t": {"qini": {"estimate": 1.0, "ci_low": 0.5, "ci_high": 1.5}, "note": "x"}}
    assert flatten_metrics(nested) == {"t/qini": 1.0, "t/qini_ci_low": 0.5, "t/qini_ci_high": 1.5}
    frame = synthetic_uplift_frame(100, seed=1)
    assert frame_sha256(frame) == frame_sha256(frame.copy())
    assert frame_sha256(frame) != frame_sha256(frame.iloc[1:])
    sha, _ = git_state(Path(os.sep))
    assert sha == "unknown" or len(sha) == 40


def test_pyfunc_declares_the_package_code_and_its_dependencies() -> None:
    # uplift_lab is not on PyPI: the registered model carries its source as code_paths...
    (package,) = code_paths()
    assert Path(package).name == "uplift_lab"
    assert (Path(package) / "uplift" / "pyfunc_model.py").is_file()
    assert (Path(package) / "uplift" / "artifacts.py").is_file()
    # ...and pins every core dependency of the package (pyproject's [project] dependencies).
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {
        re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0]
        for spec in pyproject["project"]["dependencies"]
    }
    assert core <= set(PYFUNC_REQUIREMENTS)
    assert "mlflow" in PYFUNC_REQUIREMENTS


def test_full_job_metric_count_stays_under_the_logged_model_cap() -> None:
    # The full job's shape: four learners plus random, four budgets, two outcomes.
    frame = synthetic_uplift_frame(3_000, seed=5)
    params = LgbmParams(n_estimators=10)
    runs = {
        outcome: run_uplift(frame, UpliftConfig(outcome=outcome, params=params, n_bootstrap=5))
        for outcome in ("visit", "conversion")
    }
    timings = {f"stage_{i}_seconds": 1.0 for i in range(20)}
    payload = {
        "timings": timings,
        "outcomes": {name: {"evaluation": run.evaluation} for name, run in runs.items()},
    }
    visit = runs["visit"]
    total = {
        **evaluation_metrics(visit.evaluation),
        **visit.timings,
        **outcome_metrics(payload, "visit"),
    }
    assert "conversion/s_learner/qini_coefficient" in total
    assert "stage/stage_0_seconds" in total
    # Measured 632; the old full set (every policy column, logged in two steps) exceeded 1,000.
    assert len(total) < 0.7 * MAX_RUN_METRICS


def test_build_info_and_job_git_state(tmp_path: Path) -> None:
    info = tmp_path / "build_info.json"
    assert build_info(info) is None
    info.write_text(json.dumps({"git_sha": "abc", "git_dirty": True}), encoding="utf-8")
    recorded = build_info(info)
    assert recorded == {"git_sha": "abc", "git_dirty": True}
    assert job_git_state("abc", recorded) == {
        "sha": "abc",
        "dirty": True,
        "source": "wheel build info",
    }
    # A wheel built from another commit says nothing about this one.
    assert job_git_state("def", recorded)["dirty"] is None
    info.write_text(json.dumps({"git_sha": "abc", "git_dirty": "no"}), encoding="utf-8")
    assert build_info(info) is None
    info.write_text("{broken", encoding="utf-8")
    assert build_info(info) is None
    assert job_git_state(None, None)["source"] == "checkout"


def test_log_and_register_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    from uplift_lab.uplift.tracking import log_uplift_run

    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    frame = synthetic_uplift_frame(4_000, seed=31)
    config = UpliftConfig(
        learners=("t_learner", "s_learner"), params=LgbmParams(n_estimators=15), n_bootstrap=10
    )
    run = run_uplift(frame, config)
    best = run.evaluation["best_model"]
    model_dir = tmp_path / "model"
    save_learner(run.learners[best], model_dir)
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    tracking = TrackingConfig.from_env(tmp_path / "mlruns", env={})
    tracked = log_uplift_run(
        run, model_dir, config=tracking, tags={"git_sha": "abc"}, report_files=[report]
    )
    assert tracked.model_version == "1"
    client = mlflow.MlflowClient(tracking.tracking_uri)
    logged = client.get_run(tracked.run_id).data
    assert logged.tags["git_sha"] == "abc"
    assert logged.tags["best_model"] == best
    assert logged.tags["champion"] == run.evaluation["champion"]["model"]
    assert logged.tags["champion_rule"] == run.evaluation["champion"]["rule"]
    assert logged.params["n_test"] == str(run.n_test)
    assert f"{best}/qini_coefficient_ci_low" in logged.metrics
    assert client.get_model_version_by_alias(tracking.model_name, CHAMPION_ALIAS).version == 1
    uri = f"models:/{tracking.model_name}@{CHAMPION_ALIAS}"
    # The logged model is self-contained: the package source and pinned requirements travel
    # with it, so a clean environment needs only requirements.txt to load it.
    local = Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(tmp_path / "download")))
    assert (local / "code" / "uplift_lab" / "uplift" / "pyfunc_model.py").is_file()
    requirements = (local / "requirements.txt").read_text(encoding="utf-8")
    for name in PYFUNC_REQUIREMENTS:
        assert f"{name}==" in requirements
    model = mlflow.pyfunc.load_model(uri)
    predicted = model.predict(frame.iloc[:50])
    np.testing.assert_allclose(predicted["uplift"], run.learners[best].predict(frame.iloc[:50]))
