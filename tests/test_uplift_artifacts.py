"""Serving artifact contract: lossless round trip for every learner and strict validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_uplift import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.uplift.artifacts import (
    FORMAT,
    MANIFEST,
    ArtifactError,
    load_learner,
    read_manifest,
    save_learner,
)
from uplift_lab.uplift.learners import LEARNERS, LgbmParams, UpliftLearner, make_learner


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_uplift_frame(3_000, seed=12)


def _fit(kind: str, frame: pd.DataFrame) -> UpliftLearner:
    learner = make_learner(kind, LgbmParams(n_estimators=15, seed=3))
    return learner.fit(frame[list(FEATURE_NAMES)], frame["treatment"], frame["visit"])


@pytest.mark.parametrize("kind", sorted(LEARNERS))
def test_round_trip_reproduces_predictions(kind: str, frame: pd.DataFrame, tmp_path: Path) -> None:
    learner = _fit(kind, frame)
    save_learner(learner, tmp_path, metadata={"outcome": "visit", "budget_thresholds": {"0.1": 0}})
    loaded = load_learner(tmp_path)
    assert type(loaded) is type(learner)
    assert loaded.features_ == learner.features_
    assert loaded.propensity_ == learner.propensity_
    np.testing.assert_array_equal(loaded.predict(frame), learner.predict(frame))
    manifest = read_manifest(tmp_path)
    assert manifest["format"] == FORMAT
    assert manifest["kind"] == kind
    assert manifest["metadata"]["outcome"] == "visit"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [MANIFEST, *manifest["boosters"].values()]
    )


def _write_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_invalid_manifests_are_rejected(frame: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="no manifest"):
        load_learner(tmp_path)
    save_learner(_fit("t_learner", frame), tmp_path)
    cases: list[tuple[dict[str, object], str]] = [
        ({"schema_version": 2}, "schema_version"),
        ({"format": "other"}, "manifest"),
        ({"kind": "r_learner"}, "unknown learner kind"),
        ({"boosters": {"control": "../control.txt", "treated": "treated.txt"}}, "boosters"),
        ({"boosters": {"control": "control.txt"}}, "needs boosters"),
        ({"features": ["f0", "exposure"]}, "post-treatment"),
        ({"propensity": 1.0}, "propensity"),
    ]
    original = (tmp_path / MANIFEST).read_text(encoding="utf-8")
    for changes, message in cases:
        (tmp_path / MANIFEST).write_text(original, encoding="utf-8")
        _write_manifest(tmp_path, **changes)
        with pytest.raises(ArtifactError, match=message):
            load_learner(tmp_path)
    (tmp_path / MANIFEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_unfitted_learner_cannot_be_saved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unfitted"):
        save_learner(make_learner("s_learner"), tmp_path)
