"""Uplift metric gate: the rule, baseline compatibility, the CLI and the committed baseline."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from uplift_lab.data.synthetic import synthetic_uplift_frame
from uplift_lab.gate import cli
from uplift_lab.gate.metric_gate import (
    POLICY,
    QINI,
    BaselineMismatch,
    GateConfig,
    compare,
    gate_metrics,
)
from uplift_lab.uplift.learners import LgbmParams

SMALL = GateConfig(n_rows=40_000, n_bootstrap=50, params=LgbmParams(n_estimators=100))
COMMITTED = Path(__file__).resolve().parents[1] / "evals" / "metric-gate-baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return gate_metrics(SMALL)


def test_same_code_passes_against_its_own_baseline(baseline: dict[str, Any]) -> None:
    result = compare(baseline, gate_metrics(SMALL))
    assert result.passed and result.verdict == "PASS"
    assert {f.status for f in result.findings} == {"ok"}
    assert len(result.findings) == 4 * 2  # four learners x (Qini, policy value); random not gated


def test_a_model_without_the_effect_feature_fails(baseline: dict[str, Any]) -> None:
    # The synthetic effect is driven by f0 only; without it every learner ranks at random.
    blind = replace(SMALL, features=tuple(f"f{i}" for i in range(1, 12)))
    result = compare(baseline, gate_metrics(blind))
    assert not result.passed and result.verdict == "REGRESSION"
    statuses = {(f.metric, f.status) for f in result.findings}
    assert (QINI, "not_beating_random") in statuses
    assert (POLICY, "regression") in statuses
    assert "REGRESSION" in cli.render_markdown(result)


def _with(payload: dict[str, Any], learner: str, metric: str, **values: float) -> dict[str, Any]:
    changed = copy.deepcopy(payload)
    changed["metrics"][learner][metric].update(values)
    return changed


def test_rule_boundaries(baseline: dict[str, Any]) -> None:
    base = baseline["metrics"]["s_learner"][POLICY]
    below = _with(baseline, "s_learner", POLICY, estimate=base["ci_low"] - 1e-9)
    at_bound = _with(baseline, "s_learner", POLICY, estimate=base["ci_low"])
    above = _with(baseline, "s_learner", POLICY, estimate=base["ci_high"] + 1e-9)
    assert compare(baseline, below).verdict == "REGRESSION"
    assert compare(baseline, at_bound).verdict == "PASS"
    assert compare(baseline, above).verdict == "IMPROVEMENT"
    # A Qini interval that touches 0 fails even when the point estimate holds.
    touching = _with(baseline, "t_learner", QINI, ci_low=0.0)
    assert {f.status for f in compare(baseline, touching).findings if f.failed} == {
        "not_beating_random"
    }


def test_a_missing_learner_fails(baseline: dict[str, Any]) -> None:
    head = copy.deepcopy(baseline)
    del head["metrics"]["x_learner"]
    failed = [f for f in compare(baseline, head).findings if f.failed]
    assert {(f.learner, f.status) for f in failed} == {("x_learner", "missing")}


def test_random_scorer_is_context_not_gated(baseline: dict[str, Any]) -> None:
    head = _with(baseline, "random", QINI, estimate=-1.0, ci_low=-1.0)
    assert compare(baseline, head).passed


def test_incomparable_baselines_are_refused(baseline: dict[str, Any]) -> None:
    other_data = copy.deepcopy(baseline)
    other_data["fingerprint"]["n_rows"] = 1
    with pytest.raises(BaselineMismatch, match="n_rows"):
        compare(other_data, baseline)
    wrong_format = {**baseline, "format": "something-else"}
    with pytest.raises(BaselineMismatch, match="not a"):
        compare(wrong_format, baseline)
    with pytest.raises(ValueError, match="n_rows"):
        GateConfig(n_rows=100)


def test_cli_update_compare_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payloads = iter([gate_metrics(SMALL), gate_metrics(SMALL)])
    monkeypatch.setattr(cli, "gate_metrics", lambda: next(payloads))
    path, out = tmp_path / "baseline.json", tmp_path / "out"
    args = ["--baseline", str(path), "--output-dir", str(out)]
    assert cli.main([*args, "--update-baseline"]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"]["n_rows"] == SMALL.n_rows
    assert cli.main(args) == 0
    assert "PASS" in (out / "summary.md").read_text(encoding="utf-8")
    assert (out / "head.json").is_file()

    blind = gate_metrics(replace(SMALL, features=("f1", "f2", "f3")))
    monkeypatch.setattr(cli, "gate_metrics", lambda: blind)
    assert cli.main(args) == 1
    assert "verdict REGRESSION" in capsys.readouterr().out

    assert cli.main(["--baseline", str(tmp_path / "missing.json"), "--output-dir", str(out)]) == 2
    path.write_text("[1, 2]", encoding="utf-8")
    assert cli.main(args) == 2
    path.write_text("{not json", encoding="utf-8")
    assert cli.main(args) == 2
    assert "cannot compare" in capsys.readouterr().err


def test_committed_baseline_matches_the_default_gate() -> None:
    """Changing the gate's data or evaluation settings needs a baseline refresh in the same PR."""
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    config = GateConfig()
    frame = synthetic_uplift_frame(config.n_rows, seed=config.seed)
    assert committed["fingerprint"] == config.fingerprint(frame)
    for learner in ("s_learner", "t_learner", "x_learner", "transformed_outcome"):
        qini = committed["metrics"][learner][QINI]
        assert 0.0 < qini["ci_low"] <= qini["estimate"] <= qini["ci_high"]
