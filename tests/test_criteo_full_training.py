"""Full-data training job on a synthetic Criteo-format gzip: source checks, outputs, MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_uplift import synthetic_uplift_frame
from uplift_lab.data.cache import ChecksumError, RemoteFile, sha256_file
from uplift_lab.pipelines import criteo_full_training as job
from uplift_lab.pipelines.criteo_full_data import check_source, read_criteo_csv
from uplift_lab.uplift.artifacts import load_learner, read_manifest
from uplift_lab.uplift.tracking import MAX_RUN_METRICS

N_ROWS = 6_000
SMALL_ARGS = ["--n-estimators", "30", "--n-bootstrap", "20", "--learners", "s_learner", "t_learner"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_uplift_frame(N_ROWS, seed=41)


@pytest.fixture(scope="module")
def source(frame: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Criteo column order (features, treatment, conversion, visit, exposure), gzipped like v2.1.
    path = tmp_path_factory.mktemp("criteo") / "criteo-format.csv.gz"
    columns = [*(f"f{i}" for i in range(12)), "treatment", "conversion", "visit", "exposure"]
    frame[columns].to_csv(path, index=False, compression="gzip")
    return path


@pytest.fixture()
def pinned(source: Path, monkeypatch: pytest.MonkeyPatch) -> RemoteFile:
    remote = RemoteFile("file://unused", source.name, sha256_file(source), source.stat().st_size)
    monkeypatch.setattr(job, "CRITEO", remote)
    monkeypatch.setattr(job, "EXPECTED_ROWS", N_ROWS)
    return remote


def test_check_source_verifies_size_and_checksum(source: Path, tmp_path: Path) -> None:
    good = RemoteFile("file://unused", source.name, sha256_file(source), source.stat().st_size)
    checked = check_source(source, good)
    assert (checked.size_bytes, checked.sha256, checked.verified) == (
        good.size_bytes,
        good.sha256,
        True,
    )
    with pytest.raises(FileNotFoundError):
        check_source(tmp_path / "missing.csv.gz", good)
    with pytest.raises(ValueError, match="bytes"):
        check_source(source, RemoteFile(good.url, good.filename, good.sha256, good.size_bytes + 1))
    wrong = RemoteFile(good.url, good.filename, "0" * 64, good.size_bytes)
    with pytest.raises(ChecksumError):
        check_source(source, wrong)
    assert check_source(source, wrong, verify_checksum=False).verified is False


def test_read_criteo_csv_types_and_validation(
    source: Path, frame: pd.DataFrame, tmp_path: Path
) -> None:
    loaded = read_criteo_csv(source, expected_rows=N_ROWS)
    assert loaded["f0"].dtype == np.float64 and loaded["visit"].dtype == np.int8
    assert np.allclose(loaded["f3"], frame["f3"])
    assert (loaded["conversion"].to_numpy() == frame["conversion"].to_numpy()).all()
    with pytest.raises(ValueError, match="expected 10 rows"):
        read_criteo_csv(source, expected_rows=10)
    bad = tmp_path / "bad.csv"
    frame.head(5).assign(visit=[0, 1, 2, 0, 1]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="visit must be binary"):
        read_criteo_csv(bad, expected_rows=None)


def test_tracking_env_flags_override_environment() -> None:
    args = argparse.Namespace(
        tracking_uri="databricks", registry_uri=None, experiment="/Users/x/e", model_name=None
    )
    env = job.tracking_env(args, {"MLFLOW_REGISTRY_URI": "databricks-uc", "OTHER": "1"})
    assert env == {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_REGISTRY_URI": "databricks-uc",
        "UPLIFT_LAB_MLFLOW_EXPERIMENT": "/Users/x/e",
        "OTHER": "1",
    }


def test_job_trains_both_outcomes_and_publishes(
    source: Path, pinned: RemoteFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, published = tmp_path / "out", tmp_path / "volume" / "run-1"
    argv = ["--source", str(source), "--output-dir", str(out), "--publish-dir", str(published)]
    assert job.main([*argv, *SMALL_ARGS, "--git-sha", "abc123"]) == 0

    payload = json.loads((published / job.REPORT_NAME).read_text(encoding="utf-8"))
    assert set(payload["outcomes"]) == {"visit", "conversion"}
    assert payload["dataset"]["n_rows"] == N_ROWS
    assert payload["dataset"]["source"]["verified"] is True
    assert payload["git"] == {"sha": "abc123", "dirty": None, "source": "argument"}
    for stage in ("verify_source_seconds", "read_csv_seconds", "visit_seconds", "total_seconds"):
        assert payload["timings"][stage] >= 0.0
    visit = payload["outcomes"]["visit"]
    assert visit["config"]["outcome"] == "visit"
    assert visit["n_train"] + visit["n_test"] == N_ROWS
    assert set(visit["evaluation"]["models"]) == {"s_learner", "t_learner", "random"}
    assert payload["outcomes"]["conversion"]["config"]["outcome"] == "conversion"

    manifest = read_manifest(published / "model")
    assert manifest["kind"] == payload["model"]["kind"] == visit["evaluation"]["best_model"]
    assert manifest["metadata"]["outcome"] == "visit"
    assert manifest["metadata"]["git_sha"] == "abc123"
    assert "P(visit" in manifest["metadata"]["score"]
    learner = load_learner(published / "model")
    assert np.isfinite(learner.predict(read_criteo_csv(source, expected_rows=None))).all()
    assert "[conversion] ATE" in capsys.readouterr().out


def test_job_rejects_inconsistent_arguments(source: Path, pinned: RemoteFile) -> None:
    with pytest.raises(SystemExit):
        job.main(["--source", str(source), "--outcomes", "conversion"])
    with pytest.raises(SystemExit):
        job.main(["--source", str(source), "--sample-rows", "10"])


def test_job_rejects_a_source_that_is_not_the_pinned_file(source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bytes"):
        job.main(["--source", str(source), "--output-dir", str(tmp_path)])


def test_job_logs_and_registers_with_mlflow(
    source: Path, pinned: RemoteFile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "UPLIFT_LAB_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    root, out = tmp_path / "mlruns", tmp_path / "out"
    argv = ["--source", str(source), "--output-dir", str(out), "--sample-rows", "4000"]
    flags = ["--mlflow", "--mlflow-root", str(root), "--model-name", "criteo-full-test"]
    assert job.main([*argv, *SMALL_ARGS, *flags]) == 0

    payload = json.loads((out / job.REPORT_NAME).read_text(encoding="utf-8"))
    tracked = payload["mlflow"]
    assert tracked["model_name"] == "criteo-full-test"
    assert payload["dataset"]["n_rows"] == 4000
    run = mlflow.MlflowClient().get_run(tracked["run_id"])
    assert run.data.params["outcome"] == "visit"
    assert run.data.tags["platform"] == "local"
    assert "conversion/ate" in run.data.metrics
    assert "conversion/s_learner/qini_coefficient_ci_low" in run.data.metrics
    assert "stage/read_csv_seconds" in run.data.metrics
    # MLflow 3 links the run's metrics to the LoggedModel, where Databricks caps them at 1,000.
    # Everything is logged once, before log_model, so the linked set is exactly the run's.
    (logged,) = mlflow.search_logged_models(
        experiment_ids=[tracked["experiment_id"]], output_format="list"
    )
    linked = {metric.key for metric in logged.metrics or []}
    assert linked == set(run.data.metrics)
    assert len(linked) < MAX_RUN_METRICS
    model = mlflow.pyfunc.load_model("models:/criteo-full-test@champion")
    rows = read_criteo_csv(source, expected_rows=None).head(50)
    scored = model.predict(rows)
    assert np.allclose(scored["uplift"], load_learner(out / "model").predict(rows))
