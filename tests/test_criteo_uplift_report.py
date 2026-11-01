"""Criteo uplift report end to end on synthetic data; the real subsample only under -m network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_uplift import synthetic_uplift_frame
from uplift_lab.reports import criteo_uplift
from uplift_lab.reports.criteo_uplift import sample_frame, write_criteo_uplift_report
from uplift_lab.uplift.artifacts import load_learner, read_manifest
from uplift_lab.uplift.learners import LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, run_uplift, split_train_test

SMALL = UpliftConfig(params=LgbmParams(n_estimators=40, min_child_samples=50), n_bootstrap=30)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_uplift_frame(12_000, seed=21)


def test_split_is_seeded_and_stratified(frame: pd.DataFrame) -> None:
    train, test = split_train_test(frame, "visit", test_size=0.3, seed=1)
    again, _ = split_train_test(frame, "visit", test_size=0.3, seed=1)
    assert len(test) == 3_600
    assert train.index.equals(again.index)
    for column in ("treatment", "visit"):
        assert train[column].mean() == pytest.approx(test[column].mean(), abs=0.002)


def test_run_rejects_bad_frames(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        run_uplift(frame.drop(columns="visit"), SMALL)
    with pytest.raises(ValueError, match="post-treatment"):
        run_uplift(frame, SMALL, features=("f0", "exposure"))
    with pytest.raises(ValueError, match="learners"):
        UpliftConfig(learners=("r_learner",))


def test_report_writes_metrics_figures_and_model(frame: pd.DataFrame, tmp_path: Path) -> None:
    payload = write_criteo_uplift_report(frame, tmp_path, SMALL)
    saved = json.loads((tmp_path / "criteo-uplift.json").read_text(encoding="utf-8"))
    evaluation = saved["evaluation"]
    assert set(evaluation["models"]) == {*SMALL.learners, "random"}
    assert saved["model"]["kind"] == evaluation["best_model"] != "random"
    assert evaluation["best_beats_random"] is True
    assert saved["dataset"]["n_rows"] == len(frame)
    assert len(saved["dataset"]["frame_sha256"]) == 64
    assert saved["mlflow"] is None
    for name in ("criteo-uplift-qini.png", "criteo-uplift-deciles.png"):
        assert (tmp_path / name).stat().st_size > 10_000
    learner = load_learner(tmp_path / "model")
    manifest = read_manifest(tmp_path / "model")
    assert learner.kind == payload["model"]["kind"]
    assert set(manifest["metadata"]["budget_thresholds"]) == {"0.1", "0.2", "0.3", "0.5"}
    assert np.isfinite(learner.predict(frame)).all()


def test_cli_samples_rows(
    frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(criteo_uplift, "load_criteo_subsample", lambda cache: frame)
    args = [
        *("--output-dir", str(tmp_path), "--sample-rows", "6000", "--n-bootstrap", "10"),
        *("--n-estimators", "20", "--learners", "t_learner", "s_learner", "--no-figures"),
    ]
    assert criteo_uplift.main(args) == 0
    saved = json.loads((tmp_path / "criteo-uplift.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["n_rows"] == 6_000
    assert saved["config"]["learners"] == ["t_learner", "s_learner"]
    out = capsys.readouterr().out
    assert "Qini" in out and "beats random" in out
    assert not (tmp_path / "criteo-uplift-qini.png").exists()
    with pytest.raises(SystemExit):
        criteo_uplift.main(["--output-dir", str(tmp_path), "--test-size", "1.5"])


def test_sample_frame_is_stratified(frame: pd.DataFrame) -> None:
    sample = sample_frame(frame, 3_000, "visit", seed=2)
    assert len(sample) == 3_000
    assert sample["treatment"].mean() == pytest.approx(frame["treatment"].mean(), abs=0.001)
    assert sample_frame(frame, None, "visit", seed=2) is frame


@pytest.mark.network
def test_real_criteo_smoke(tmp_path: Path) -> None:
    from uplift_lab.data.criteo import load_criteo_subsample

    real = sample_frame(load_criteo_subsample(), 100_000, "visit", seed=20261101)
    payload = write_criteo_uplift_report(real, tmp_path, SMALL, figures=False)
    assert payload["evaluation"]["ate"]["estimate"] == pytest.approx(0.0103, abs=0.004)
