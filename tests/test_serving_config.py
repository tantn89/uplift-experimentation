"""Serving settings, model resolution (local and MLflow registry) and the sampled traffic log."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving_fixtures import customers, export_model, payload
from synthetic_uplift import FEATURE_NAMES
from uplift_lab.monitoring.traffic_log import TrafficLog, read_traffic_log
from uplift_lab.serving.app import create_app
from uplift_lab.serving.model_source import (
    download_registered_model,
    find_exported_model,
    resolve_model_dir,
)
from uplift_lab.serving.settings import ServingSettings
from uplift_lab.uplift.artifacts import ArtifactError, load_learner


def test_settings_from_env() -> None:
    settings = ServingSettings.from_env({"UPLIFT_LAB_MODEL_DIR": "/app/model"})
    assert settings == ServingSettings(model_dir=Path("/app/model"))
    assert settings.traffic_log is None
    env = {
        "UPLIFT_LAB_MODEL_URI": "models:/m@champion",
        "UPLIFT_LAB_TRAFFIC_LOG": "/tmp/t.jsonl",
        "UPLIFT_LAB_TRAFFIC_SAMPLE_RATE": "1",
        "UPLIFT_LAB_TRAFFIC_MAX_ROWS": "5",
    }
    custom = ServingSettings.from_env(env)
    assert (custom.model_uri, custom.traffic_sample_rate, custom.traffic_max_rows) == (
        "models:/m@champion",
        1.0,
        5,
    )
    with pytest.raises(ValueError, match="MODEL_DIR or"):
        ServingSettings.from_env({})
    with pytest.raises(ValueError, match="must be a number"):
        ServingSettings.from_env({"UPLIFT_LAB_MODEL_DIR": "m", "UPLIFT_LAB_TRAFFIC_MAX_ROWS": "x"})
    with pytest.raises(ValueError, match="sample rate"):
        ServingSettings(model_dir=Path("m"), traffic_sample_rate=1.5)


def test_find_exported_model_inside_a_pyfunc_layout(tmp_path: Path) -> None:
    export_model(tmp_path / "pyfunc" / "artifacts" / "model")
    found = find_exported_model(tmp_path / "pyfunc")
    assert found == tmp_path / "pyfunc" / "artifacts" / "model"
    shutil.copytree(found, tmp_path / "pyfunc" / "artifacts" / "other")
    with pytest.raises(ArtifactError, match="found 2"):
        find_exported_model(tmp_path / "pyfunc")


def test_registry_uri_without_mlflow_explains_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow.artifacts", None)
    with pytest.raises(RuntimeError, match="mlflow-skinny"):
        download_registered_model("models:/m@champion", tmp_path)


def test_traffic_log_is_sampled_bounded_and_readable(tmp_path: Path) -> None:
    frame = customers(50)
    uplift = np.linspace(0.0, 1.0, 50)
    log = TrafficLog(tmp_path / "t.jsonl", sample_rate=1.0, max_rows=10, max_bytes=3_000, seed=1)
    assert all(log.record("score", frame, uplift) for _ in range(4))
    assert (tmp_path / "t.jsonl.1").exists()  # rotated once the file passed max_bytes
    sizes = [p.stat().st_size for p in tmp_path.iterdir()]
    assert len(sizes) == 2 and max(sizes) <= 3_000
    record = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["n"] == 50 and record["endpoint"] == "score"
    rows = read_traffic_log(tmp_path / "t.jsonl", FEATURE_NAMES)
    assert list(rows.columns) == [*FEATURE_NAMES, "uplift"]
    assert len(rows) % 10 == 0 and rows["uplift"].between(0, 1).all()
    silent = TrafficLog(tmp_path / "s.jsonl", sample_rate=0.0, max_rows=5, max_bytes=100)
    assert silent.record("score", frame, uplift) is False
    assert not (tmp_path / "s.jsonl").exists()
    blocked = TrafficLog(tmp_path, sample_rate=1.0, max_rows=5, max_bytes=10**6)  # a directory
    assert blocked.record("score", frame, uplift) is False
    (tmp_path / "bad.jsonl").write_text('{"rows": {"f0": [1]}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        read_traffic_log(tmp_path / "bad.jsonl", FEATURE_NAMES)
    with pytest.raises(ValueError, match="sample_rate"):
        TrafficLog(tmp_path / "x", sample_rate=2.0, max_rows=1, max_bytes=1)


def test_app_writes_the_traffic_log_when_enabled(tmp_path: Path) -> None:
    model_dir = export_model(tmp_path / "model")
    log = tmp_path / "logs" / "traffic.jsonl"
    settings = ServingSettings(model_dir=model_dir, traffic_log=log, traffic_sample_rate=1.0)
    with TestClient(create_app(settings)) as client:
        assert client.post("/score", json=payload(customers(5))).status_code == 200
        assert client.post("/decide", json=payload(customers(5), budget="0.1")).status_code == 200
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["endpoint"] for line in lines] == ["score", "decide"]
    assert "id" not in lines[0]["rows"]


def test_registered_champion_serves_like_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    from synthetic_uplift import synthetic_uplift_frame
    from uplift_lab.uplift.learners import LgbmParams
    from uplift_lab.uplift.pipeline import UpliftConfig, run_uplift
    from uplift_lab.uplift.tracking import TrackingConfig, log_uplift_run

    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    config = UpliftConfig(
        learners=("t_learner",), params=LgbmParams(n_estimators=10), n_bootstrap=5
    )
    run = run_uplift(synthetic_uplift_frame(3_000, seed=58), config)
    model_dir = export_model(tmp_path / "model")
    tracking = TrackingConfig.from_env(tmp_path / "mlruns", env={})
    tracked = log_uplift_run(run, model_dir, config=tracking, tags={})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking.tracking_uri)
    uri = f"models:/{tracked.model_name}@{tracked.alias}"
    resolved, source = resolve_model_dir(ServingSettings(model_uri=uri), tmp_path / "download")
    assert source == uri
    frame = customers(20)
    expected = load_learner(model_dir).predict(frame)
    np.testing.assert_array_equal(load_learner(resolved).predict(frame), expected)
    with TestClient(create_app(ServingSettings(model_uri=uri))) as client:
        assert client.get("/health").json()["model"]["source"] == uri
        scores = client.post("/score", json=payload(frame)).json()["scores"]
    np.testing.assert_allclose([s["uplift"] for s in scores], expected, rtol=0, atol=1e-12)
