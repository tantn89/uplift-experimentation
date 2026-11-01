"""Uplift metric regression gate: seeded synthetic data, committed baseline, paired bootstrap rule.

``gate_metrics`` trains every learner on a fixed seeded synthetic RCT
(``uplift_lab.data.synthetic``, no download) and records, per learner, the Qini coefficient and
the policy value at the 10% budget (incremental visits per 1,000 contacts). Each comes with its
percentile bootstrap interval and the value on every bootstrap draw. The draws are seeded and the
test set is fixed by the fingerprint, so draw ``b`` of the baseline and draw ``b`` of a later run
resample exactly the same rows. ``compare`` uses that pairing:

- **regression**: the head-minus-baseline difference, bootstrapped on the same draws, has its
  95% interval entirely below 0, *and* the point drop exceeds ``MIN_RELATIVE_CHANGE`` of the
  baseline estimate. The paired interval measures the drop against the test set's sampling
  noise far more tightly than the baseline's own interval would. The floor keeps a change too
  small to matter (for example floating-point differences between platforms that move one
  split) from failing the build;
- **no longer beats random**: the head Qini interval no longer excludes 0 from above;
- **improvement**: the same two conditions in the other direction (passes; refresh the baseline
  so later changes are judged against it).

The fingerprint holds what must match for the pairing and the comparison to mean anything: the
data generator and seed, the frame hash, the evaluation settings and the versions of the
libraries that produce the numbers (LightGBM, numpy, pandas, scikit-learn for the split). If any
differs, ``BaselineMismatch`` is raised instead of a verdict. CI installs the pinned versions from
``evals/metric-gate-constraints.txt``. Model settings (features, LightGBM parameters) are recorded
but not fingerprinted: they are what the gate judges. With the fingerprint fixed, the estimates
move when the model code or settings change; an identical rerun reproduces them exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from uplift_lab.data.synthetic import FEATURE_NAMES, synthetic_uplift_frame
from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.uplift.bootstrap import Interval, bootstrap_draws, percentile_interval
from uplift_lab.uplift.evaluation import RANDOM_SCORER
from uplift_lab.uplift.learners import LEARNERS, LgbmParams
from uplift_lab.uplift.metrics import Ranking, SortedSample, qini_coefficient
from uplift_lab.uplift.pipeline import UpliftConfig, UpliftRun, run_uplift
from uplift_lab.uplift.policy import policy_value
from uplift_lab.uplift.provenance import frame_sha256

FORMAT = "uplift-lab.metric-gate"
SCHEMA_VERSION = 2
GATE_BUDGET = 0.1
QINI = "qini_coefficient"
POLICY = f"policy_per_1000_contacts@{GATE_BUDGET:g}"
METRICS = (QINI, POLICY)
#: Libraries whose versions change the numbers; pinned for CI in evals/metric-gate-constraints.txt.
VERSIONED_LIBRARIES = ("lightgbm", "numpy", "pandas", "scikit-learn")
#: A paired change smaller than this share of the baseline estimate is never a verdict.
MIN_RELATIVE_CHANGE = 0.02


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
            "versions": library_versions(),
        }


def library_versions() -> dict[str, str]:
    return {name: version(name) for name in VERSIONED_LIBRARIES}


class BaselineMismatch(ValueError):
    """The baseline was built on other data, settings or library versions; refresh it."""


@dataclass(frozen=True)
class Finding:
    learner: str
    metric: str
    status: str  # ok | regression | improvement | not_beating_random | missing
    head: float | None
    baseline: Mapping[str, float] | None
    change: Interval | None = None  # head minus baseline on the same bootstrap draws

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


def _gate_draws(run: UpliftRun, n_bootstrap: int, seed: int, alpha: float) -> dict[str, Any]:
    """Qini and 10%-budget policy value per scoring, with intervals and every draw's value.

    Uses the same seed and weights as the run's own evaluation, so the estimates and intervals
    equal the ones in ``run.evaluation``.
    """
    samples = {
        name: SortedSample(Ranking(scores), run.test_treatment, run.test_outcome)
        for name, scores in run.scores.items()
        if name in LEARNERS or name == RANDOM_SCORER
    }

    def statistic(weights: FloatArray | None) -> dict[str, float]:
        values = {}
        for name, sample in samples.items():
            g = sample.gains(weights)
            values[f"{name}|{QINI}"] = qini_coefficient(g)
            values[f"{name}|{POLICY}"] = policy_value(g, GATE_BUDGET).model_per_1000_contacts
        return values

    result = bootstrap_draws(statistic, run.n_test, n_boot=n_bootstrap, seed=seed)
    metrics: dict[str, Any] = {}
    for key, draws in result.draws.items():
        name, metric = key.split("|")
        entry: dict[str, Any] = result.interval(key, alpha).to_dict()
        entry["draws"] = draws.tolist()
        metrics.setdefault(name, {})[metric] = entry
    return metrics


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
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": config.fingerprint(frame),
        "model": {"features": list(config.features), "params": config.params.to_dict()},
        "metrics": _gate_draws(run, config.n_bootstrap, config.seed, config.alpha),
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
            f"data, evaluation settings or library versions differ from the baseline "
            f"({', '.join(diff)}); install evals/metric-gate-constraints.txt, or refresh the "
            "baseline with uplift-metric-gate --update-baseline"
        )


def paired_change(base: Mapping[str, Any], head: Mapping[str, Any], alpha: float) -> Interval:
    """Head minus baseline, with a percentile interval over the paired bootstrap draws."""
    base_draws = np.asarray(base["draws"], dtype=np.float64)
    head_draws = np.asarray(head["draws"], dtype=np.float64)
    if base_draws.shape != head_draws.shape or base_draws.ndim != 1:
        raise BaselineMismatch("baseline and head hold different numbers of bootstrap draws")
    estimate = float(head["estimate"]) - float(base["estimate"])
    return percentile_interval(estimate, head_draws - base_draws, alpha)


def _judge(metric: str, base: Mapping[str, Any], head: Mapping[str, Any], change: Interval) -> str:
    if metric == QINI and not head["ci_low"] > 0.0:
        return "not_beating_random"
    floor = MIN_RELATIVE_CHANGE * abs(float(base["estimate"]))
    if change.ci_high < 0.0 and change.estimate < -floor:
        return "regression"
    if change.ci_low > 0.0 and change.estimate > floor:
        return "improvement"
    return "ok"


def compare(baseline: Mapping[str, Any], head: Mapping[str, Any]) -> GateResult:
    """Judge every baseline learner and metric (the random scorer is context, not gated)."""
    _check_comparable(baseline, head)
    alpha = float(baseline["fingerprint"]["alpha"])
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
            change = paired_change(base, current, alpha)
            status = _judge(metric, base, current, change)
            findings.append(Finding(learner, metric, status, current["estimate"], base, change))
    return GateResult(tuple(findings))
