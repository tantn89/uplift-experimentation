"""MLflow configuration, metric flattening and a full log-and-register run in a temp store."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from synthetic_uplift import synthetic_uplift_frame
from uplift_lab.uplift.artifacts import save_learner
from uplift_lab.uplift.learners import LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, run_uplift
from uplift_lab.uplift.provenance import frame_sha256, git_state
from uplift_lab.uplift.tracking import CHAMPION_ALIAS, TrackingConfig, flatten_metrics


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
    assert logged.params["n_test"] == str(run.n_test)
    assert f"{best}/qini_coefficient_ci_low" in logged.metrics
    assert client.get_model_version_by_alias(tracking.model_name, CHAMPION_ALIAS).version == 1
    model = mlflow.pyfunc.load_model(f"models:/{tracking.model_name}@{CHAMPION_ALIAS}")
    predicted = model.predict(frame.iloc[:50])
    np.testing.assert_allclose(predicted["uplift"], run.learners[best].predict(frame.iloc[:50]))
