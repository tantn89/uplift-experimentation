"""Latency benchmark logic against the in-process app, and its failure messages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from serving_fixtures import export_model
from uplift_lab.serving import benchmark
from uplift_lab.serving.app import create_app
from uplift_lab.serving.settings import ServingSettings


class InProcessEndpoint(benchmark.Endpoint):
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method: str, path: str, body: bytes | None = None) -> Any:
        response = self.client.request(
            method, path, content=body, headers={"content-type": "application/json"}
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.json()


def test_benchmark_summarises_each_endpoint_and_batch(tmp_path: Path) -> None:
    settings = ServingSettings(model_dir=export_model(tmp_path / "model"))
    with TestClient(create_app(settings)) as client:
        result = benchmark.run_benchmark(
            InProcessEndpoint(client), batch_sizes=[1, 5], n_requests=4, warmup=1, seed=1
        )
    assert result["decide_rule"] == {"budget": "0.1"}
    assert set(result["results"]) == {
        "score|batch_1",
        "decide|batch_1",
        "score|batch_5",
        "decide|batch_5",
    }
    for stats in result["results"].values():
        assert 0 < stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def test_cli_reports_unreachable_services(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="benchmark failed"):
        benchmark.main(["--url", "http://127.0.0.1:9", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="benchmark failed"):
        benchmark.main(["--url", "ftp://host", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        benchmark.main(["--requests", "0"])
    assert not list(tmp_path.iterdir())


def test_cli_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stats = {"p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.2, "max_ms": 3.0}
    fake = {"model": {}, "decide_rule": {}, "results": {"score|batch_1": stats}}
    monkeypatch.setattr(benchmark, "run_benchmark", lambda endpoint, **kwargs: fake)
    assert benchmark.main(["--label", "unit", "--output-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "unit.json").read_text(encoding="utf-8"))
    assert saved["label"] == "unit" and saved["n_requests"] == 200
    assert saved["results"]["score|batch_1"] == stats
