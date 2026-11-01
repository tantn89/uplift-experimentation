"""Report scripts end to end on synthetic data; the real Hillstrom run only under -m network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_lab.data.hillstrom import ARMS
from uplift_lab.experiments.peeking_simulation import PeekingConfig
from uplift_lab.reports import hillstrom, sequential
from uplift_lab.reports.hillstrom import write_hillstrom_report
from uplift_lab.reports.hillstrom_analysis import OUTCOMES, TREATED_ARMS
from uplift_lab.reports.output import write_json


def _synthetic_hillstrom(n: int = 6_000, seed: int = 7) -> pd.DataFrame:
    """Hillstrom-shaped frame where history drives visits and the Mens arm adds a lift."""
    rng = np.random.default_rng(seed)
    spend_level = rng.random(n)
    history = 30.0 + 970.0 * spend_level
    segment = pd.Categorical(rng.choice(ARMS, n), categories=list(ARMS))
    lift = np.where(segment == "Mens E-Mail", 0.08, np.where(segment == "Womens E-Mail", 0.04, 0.0))
    # Visit probability is linear in history, so history explains roughly 10% of its variance.
    visit = rng.random(n) < 0.05 + 0.5 * spend_level + lift
    conversion = visit & (rng.random(n) < 0.1)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history": history,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "newbie": rng.integers(0, 2, n),
            "channel": pd.Categorical(rng.choice(["Phone", "Web", "Multichannel"], n)),
            "segment": segment,
            "visit": visit.astype(int),
            "conversion": conversion.astype(int),
            "spend": np.where(conversion, rng.gamma(2.0, 50.0, n), 0.0),
        }
    )


def test_hillstrom_report_writes_metrics_and_figures(tmp_path: Path) -> None:
    analysis = write_hillstrom_report(_synthetic_hillstrom(), tmp_path)
    saved = json.loads((tmp_path / "hillstrom.json").read_text(encoding="utf-8"))
    assert saved["srm"]["mismatch"] is False
    assert set(saved["effects"]) == set(TREATED_ARMS)
    for outcome in OUTCOMES:
        entry = saved["effects"]["Mens E-Mail"][outcome]
        assert entry["raw"]["ci_low"] < entry["raw"]["effect"] < entry["raw"]["ci_high"]
        assert set(entry["cuped"]) == {"history", "pre_period"}
        power = saved["power"][outcome]
        assert power["required_n_per_arm"]["cuped"] <= power["required_n_per_arm"]["raw"]
    visit = analysis["effects"]["Mens E-Mail"]["visit"]
    assert visit["cuped"]["pre_period"]["variance_reduction"] > 0.05
    assert visit["raw"]["ci_low"] > 0
    assert (tmp_path / "hillstrom-effects.png").stat().st_size > 10_000
    assert (tmp_path / "hillstrom-power.png").stat().st_size > 10_000


def test_hillstrom_cli_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _synthetic_hillstrom(n=3_000)
    monkeypatch.setattr(hillstrom, "load_hillstrom", lambda cache: frame)
    assert hillstrom.main(["--output-dir", str(tmp_path), "--no-figures"]) == 0
    out = capsys.readouterr().out
    assert "SRM chi2=" in out
    assert "Mens E-Mail" in out
    assert (tmp_path / "hillstrom.json").is_file()
    assert not (tmp_path / "hillstrom-effects.png").exists()


def test_sequential_report_writes_metrics_and_figures(tmp_path: Path) -> None:
    config = PeekingConfig(n_sims=500, looks=10, batch_per_arm=300)
    payload = sequential.write_sequential_report(config, tmp_path)
    saved = json.loads((tmp_path / "sequential.json").read_text(encoding="utf-8"))
    assert saved["tau"] == pytest.approx(config.design_effect())
    assert saved["aa"]["msprt"]["rejection_rate"] == payload["aa"]["msprt"]["rejection_rate"]
    assert (tmp_path / "sequential-peeking.png").is_file()
    assert (tmp_path / "sequential-stopping.png").is_file()


def test_sequential_cli_applies_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--output-dir", str(tmp_path), "--n-sims", "300", "--looks", "4", "--no-figures"]
    assert sequential.main(args) == 0
    saved = json.loads((tmp_path / "sequential.json").read_text(encoding="utf-8"))
    assert saved["config"]["n_sims"] == 300
    assert saved["config"]["looks"] == 4
    assert "rejection rate" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        sequential.main(["--output-dir", str(tmp_path), "--n-sims", "0"])


def test_json_output_is_strict(tmp_path: Path) -> None:
    path = write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.int64(3), "c": np.ones(2)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": None, "b": 3, "c": [1.0, 1.0]}


@pytest.mark.network
def test_real_hillstrom_report(tmp_path: Path) -> None:
    from uplift_lab.data.hillstrom import load_hillstrom

    analysis = write_hillstrom_report(load_hillstrom(), tmp_path, figures=False)
    assert analysis["srm"]["mismatch"] is False
    mens_visit = analysis["effects"]["Mens E-Mail"]["visit"]["raw"]
    assert mens_visit["effect"] == pytest.approx(0.07659, abs=1e-5)
