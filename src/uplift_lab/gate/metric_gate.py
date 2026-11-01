"""Uplift metric regression gate: seeded synthetic data, committed baseline, bootstrap-CI rule.

``gate_metrics`` trains every learner on a fixed seeded synthetic RCT
(``uplift_lab.data.synthetic``, no download) and records, per learner, the Qini coefficient and
the policy value at the 10% budget (incremental visits per 1,000 contacts), each with its
percentile bootstrap interval. ``compare`` checks a fresh run (head) against the committed
baseline:

- **regression**: the head point estimate is below the baseline's lower 95% bound. The data,
  split and bootstrap draws are fixed, so a code change is the only thing that moves the
  estimate; the rule fails it only when the drop is larger than the test set's own sampling
  noise (a drop the baseline's bootstrap distribution puts below its 2.5% quantile);
- **no longer beats random**: the head Qini interval no longer excludes 0 from above;
- **improvement**: the head estimate is above the baseline's upper bound (passes; refresh the
  baseline so later changes are judged against it).

The data and evaluation settings form the baseline's fingerprint. If they differ, the two runs
are not comparable and ``BaselineMismatch`` is raised instead of a verdict. Model settings
(features, LightGBM parameters) are recorded but not part of the fingerprint: they are what the
gate judges.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from uplift_lab.data.synthetic import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.uplift.evaluation import RANDOM_SCORER
from uplift_lab.uplift.learners import LEARNERS, LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, run_uplift
from uplift_lab.uplift.provenance import frame_sha256

FORMAT = "uplift-lab.metric-gate"
SCHEMA_VERSION = 1
GATE_BUDGET = 0.1
QINI = "qini_coefficient"
POLICY = f"policy_per_1000_contacts@{GATE_BUDGET:g}"
METRICS = (QINI, POLICY)


@dataclass(frozen=True)
class GateConfig:
    """Fixed data and evaluation settings (the fingerprint) plus the model under test.

    The defaults use the same LightGBM settings and bootstrap size as the real reports and run in
    a few seconds.
    """

    n_rows: int = 100_000
    seed: int = 20261101
    test_size: float = 0.3
    n_bootstrap: int = 200
    alpha: float = 0.05
    features: tuple[str, ...] = FEATURE_NAMES
    params: LgbmParams = field(default_factory=LgbmParams)

    def __post_init__(self) -> None:
        if self.n_rows < 2_000:
            raise ValueError("n_rows must be >= 2000 for a stable gate")

    def fingerprint(self, frame: pd.DataFrame) -> dict[str, Any]:
        """What must match between baseline and head for the comparison to mean anything.

        The frame is hashed after rounding to 9 decimals: numpy's normal sampler calls libm in
        its tail, which can differ in the last bit between platforms (seen on 1 value in 10^5
        between Windows and Linux), and that is not a change of data.
        """
        return {
            "generator": "uplift_lab.data.synthetic.synthetic_uplift_frame",
            "n_rows": self.n_rows,
            "seed": self.seed,
            "frame_sha256_rounded_9dp": frame_sha256(frame.round(9)),
            "outcome": "visit",
            "test_size": self.test_size,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "budget": GATE_BUDGET,
        }


class BaselineMismatch(ValueError):
    """The baseline was built on other data or evaluation settings; refresh it."""


@dataclass(frozen=True)
class Finding:
    learner: str
    metric: str
    status: str  # ok | regression | improvement | not_beating_random | missing
    head: float | None
    baseline: Mapping[str, float] | None

    @property
    def failed(self) -> bool:
        return self.status in {"regression", "not_beating_random", "missing"}


@dataclass(frozen=True)
class GateResult:
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not any(f.failed for f in self.findings)

    @property
    def verdict(self) -> str:
        if not self.passed:
            return "REGRESSION"
        return "IMPROVEMENT" if any(f.status == "improvement" for f in self.findings) else "PASS"


def gate_metrics(config: GateConfig | None = None) -> dict[str, Any]:
    """Train and evaluate every learner on the seeded synthetic RCT; the gate payload."""
    config = config or GateConfig()
    frame = synthetic_uplift_frame(config.n_rows, seed=config.seed)
    run = run_uplift(
        frame,
        UpliftConfig(
            outcome="visit",
            test_size=config.test_size,
            seed=config.seed,
            params=config.params,
            n_bootstrap=config.n_bootstrap,
            budgets=(GATE_BUDGET,),
            alpha=config.alpha,
        ),
        features=config.features,
    )
    models = run.evaluation["models"]
    metrics = {
        name: {
            QINI: models[name][QINI],
            POLICY: models[name]["policy"][f"{GATE_BUDGET:g}"]["model_per_1000_contacts"],
        }
        for name in (*LEARNERS, RANDOM_SCORER)
        if name in models
    }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": config.fingerprint(frame),
        "model": {"features": list(config.features), "params": config.params.to_dict()},
        "metrics": metrics,
    }


def _check_comparable(baseline: Mapping[str, Any], head: Mapping[str, Any]) -> None:
    if baseline.get("format") != FORMAT or baseline.get("schema_version") != SCHEMA_VERSION:
        raise BaselineMismatch(f"baseline is not a {FORMAT} v{SCHEMA_VERSION} file")
    if baseline.get("fingerprint") != head["fingerprint"]:
        diff = sorted(
            key
            for key in {*baseline.get("fingerprint", {}), *head["fingerprint"]}
            if baseline.get("fingerprint", {}).get(key) != head["fingerprint"].get(key)
        )
        raise BaselineMismatch(
            f"data or evaluation settings differ from the baseline ({', '.join(diff)}); "
            "refresh it with uplift-metric-gate --update-baseline"
        )


def _judge(metric: str, base: Mapping[str, float], head: Mapping[str, float]) -> str:
    if metric == QINI and not head["ci_low"] > 0.0:
        return "not_beating_random"
    if head["estimate"] < base["ci_low"]:
        return "regression"
    return "improvement" if head["estimate"] > base["ci_high"] else "ok"


def compare(baseline: Mapping[str, Any], head: Mapping[str, Any]) -> GateResult:
    """Judge every baseline learner and metric (the random scorer is context, not gated)."""
    _check_comparable(baseline, head)
    findings = []
    for learner, base_metrics in baseline["metrics"].items():
        if learner == RANDOM_SCORER:
            continue
        for metric in METRICS:
            base = base_metrics[metric]
            current = head["metrics"].get(learner, {}).get(metric)
            if current is None:
                findings.append(Finding(learner, metric, "missing", None, base))
                continue
            status = _judge(metric, base, current)
            findings.append(Finding(learner, metric, status, current["estimate"], base))
    return GateResult(tuple(findings))
