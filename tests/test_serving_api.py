"""Scoring API: happy paths, boundary validation (422), batch cap and offline-equal decisions."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving_fixtures import customers, export_model, payload
from uplift_lab.serving.app import create_app
from uplift_lab.serving.model_service import top_share_mask
from uplift_lab.serving.schemas import MAX_BATCH
from uplift_lab.serving.settings import ServingSettings
from uplift_lab.uplift.artifacts import ArtifactError, load_learner, read_manifest
from uplift_lab.uplift.metrics import Ranking, gains, top_count


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_model(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="module")
def client(model_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(ServingSettings(model_dir=model_dir))) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client: TestClient, model_dir: Path) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["kind"] == "t_learner"
    assert body["model"]["features"] == [f"f{i}" for i in range(12)]
    assert body["model"]["git_sha"] == "abc"
    assert set(body["model"]["budgets"]) == {"0.1", "0.3"}
    assert body["model"]["source"] == str(model_dir)


def test_score_matches_offline_predict(client: TestClient, model_dir: Path) -> None:
    frame = customers(25)
    response = client.post("/score", json=payload(frame))
    assert response.status_code == 200
    scores = response.json()["scores"]
    assert [s["id"] for s in scores] == [f"c{i}" for i in range(25)]
    offline = load_learner(model_dir).predict(frame)
    np.testing.assert_allclose([s["uplift"] for s in scores], offline, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"exposure": 1}, "post-treatment"),
        ({"treatment": 1}, "post-treatment"),
        ({"visit": 0}, "post-treatment"),
        ({"f12": 0.5}, "Extra inputs"),
        ({"f3": "high"}, "valid number"),
        ({"id": "x" * 65}, "at most 64"),
    ],
)
def test_bad_rows_are_rejected(client: TestClient, change: dict[str, object], message: str) -> None:
    body = payload(customers(2))
    body["customers"][1].update(change)
    response = client.post("/score", json=body)
    assert response.status_code == 422
    assert message in response.text


def test_missing_and_non_finite_features_are_rejected(client: TestClient) -> None:
    body = payload(customers(1))
    del body["customers"][0]["f7"]
    assert client.post("/score", json=body).status_code == 422
    raw = '{"customers": [{' + ", ".join(f'"f{i}": NaN' for i in range(12)) + "}]}"
    response = client.post("/score", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "finite number" in response.text and "input" not in response.json()["detail"][0]


def test_batch_size_is_capped(client: TestClient) -> None:
    assert client.post("/score", json={"customers": []}).status_code == 422
    frame = customers(MAX_BATCH + 1, seed=54)
    assert client.post("/score", json=payload(frame)).status_code == 422
    assert client.post("/score", json=payload(frame.iloc[:MAX_BATCH])).status_code == 200


def test_named_budget_decisions_match_offline_threshold(
    client: TestClient, model_dir: Path
) -> None:
    frame = customers(200, seed=55)
    threshold = read_manifest(model_dir)["metadata"]["budget_thresholds"]["0.1"]
    response = client.post("/decide", json=payload(frame, budget="0.1"))
    assert response.status_code == 200
    body = response.json()
    assert body["rule"] == {
        "mode": "threshold",
        "budget": "0.1",
        "budget_fraction": None,
        "threshold": threshold,
    }
    expected = load_learner(model_dir).predict(frame) >= threshold
    assert [d["treat"] for d in body["decisions"]] == expected.tolist()
    assert body["n_treated"] == int(expected.sum())
    # A single customer gets the same decision as inside the batch.
    one = client.post("/decide", json=payload(frame.iloc[:1], budget="0.1")).json()
    assert one["decisions"][0]["treat"] == bool(expected[0])


def test_batch_fraction_treats_the_top_share(client: TestClient, model_dir: Path) -> None:
    frame = customers(30, seed=56)
    body = client.post("/decide", json=payload(frame, budget_fraction=0.2)).json()
    assert body["rule"]["mode"] == "batch_top_k"
    scores = load_learner(model_dir).predict(frame)
    treated = np.array([d["treat"] for d in body["decisions"]])
    assert body["n_treated"] == treated.sum() == 6
    assert scores[treated].min() >= scores[~treated].max()


@pytest.mark.parametrize(
    "rule",
    [{}, {"budget": "0.1", "budget_fraction": 0.1}, {"budget_fraction": 0.0}, {"budget": "0.7"}],
)
def test_decide_needs_exactly_one_known_rule(client: TestClient, rule: dict[str, object]) -> None:
    response = client.post("/decide", json=payload(customers(3), **rule))
    assert response.status_code == 422


def test_top_share_mask_counts_like_the_offline_evaluation() -> None:
    rng = np.random.default_rng(57)
    for n in (1, 7, 10, 101):
        scores = rng.integers(0, 4, n).astype(float)  # ties
        treatment, outcome = np.tile([0, 1], n)[:n], np.zeros(n)
        for fraction in (0.1, 0.3, 0.5, 1.0):
            mask = top_share_mask(scores, fraction)
            if n >= 2:
                g = gains(Ranking(scores), treatment, outcome)
                assert mask.sum() == top_count(g, fraction)
            assert mask.sum() == int(np.ceil(round(fraction * n, 9)))
    with pytest.raises(ValueError, match="fraction"):
        top_share_mask(np.ones(3), 0.0)


def test_startup_fails_on_a_bad_model(tmp_path: Path, model_dir: Path) -> None:
    with pytest.raises(ArtifactError), TestClient(create_app(ServingSettings(model_dir=tmp_path))):
        pass
    narrow = tmp_path / "narrow"
    export_model(narrow)
    manifest = (narrow / "manifest.json").read_text(encoding="utf-8")
    (narrow / "manifest.json").write_text(manifest.replace('"f11"', '"g11"'), encoding="utf-8")
    with (
        pytest.raises(ArtifactError, match="the API accepts"),
        TestClient(create_app(ServingSettings(model_dir=narrow))),
    ):
        pass
