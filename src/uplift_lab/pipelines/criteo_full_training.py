"""Train and evaluate the uplift learners on the full Criteo data; register the best one.

``python -m uplift_lab.pipelines.criteo_full_training --source <criteo .csv.gz>`` runs the same
code path as the local report (``run_uplift``: seeded stratified 70/30 split, S/T/X-learners and
the transformed outcome, paired bootstrap evaluation, randomisation check) once per outcome, on
every row instead of the 1M subsample. The source is verified against the pinned SHA-256 first.

Outputs in ``--output-dir`` (optionally copied to ``--publish-dir``, e.g. a Unity Catalog
Volume): ``criteo-full.json`` with per-stage wall times and the evaluation of every outcome, and
``model/``, the best learner for ``--register-outcome`` in the pickle-free artifact format. With
``--mlflow`` the run is logged and that model registered under the alias ``champion``; the
other outcomes' metrics are logged on the same run with an ``<outcome>/`` prefix. The Databricks
job (``databricks/``) calls this module with ``--tracking-uri databricks --registry-uri
databricks-uc`` and a Unity Catalog model name.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_lab.data.criteo import CRITEO, DEFAULT_SEED, EXPECTED_ROWS, FEATURES
from uplift_lab.pipelines.criteo_full_data import check_source, read_criteo_csv
from uplift_lab.reports.criteo_uplift import sample_frame
from uplift_lab.reports.output import write_json
from uplift_lab.uplift.artifacts import save_learner
from uplift_lab.uplift.evaluation import DEFAULT_BUDGETS
from uplift_lab.uplift.learners import LEARNERS, LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, UpliftRun, budget_thresholds, run_uplift
from uplift_lab.uplift.provenance import frame_sha256, git_state

OUTCOMES = ("visit", "conversion")
REPORT_NAME = "criteo-full.json"


def score_note(outcome: str) -> str:
    """What a model's score means, for the artifact metadata."""
    return (
        f"P({outcome} | x, assigned treatment) - P({outcome} | x, assigned control); "
        "intent to treat"
    )


def tracking_env(args: argparse.Namespace, base: Mapping[str, str]) -> dict[str, str]:
    """Environment for ``TrackingConfig.from_env``: CLI flags override inherited variables."""
    env = dict(base)
    for name, value in (
        ("MLFLOW_TRACKING_URI", args.tracking_uri),
        ("MLFLOW_REGISTRY_URI", args.registry_uri),
        ("UPLIFT_LAB_MLFLOW_EXPERIMENT", args.experiment),
        ("UPLIFT_LAB_MODEL_NAME", args.model_name),
    ):
        if value:
            env[name] = value
    return env


def _outcome_payload(run: UpliftRun) -> dict[str, Any]:
    return {
        "config": run.config.to_dict(),
        "n_train": run.n_train,
        "n_test": run.n_test,
        "timings": run.timings,
        "evaluation": run.evaluation,
    }


def train_full(
    frame: pd.DataFrame,
    output_dir: Path,
    configs: Sequence[UpliftConfig],
    *,
    register_outcome: str,
    dataset: dict[str, Any],
    git: dict[str, Any],
    timings: dict[str, float],
) -> tuple[dict[str, Any], UpliftRun]:
    """Run every outcome's config on ``frame``; save the registered outcome's best model."""
    outcomes: dict[str, Any] = {}
    registered: UpliftRun | None = None
    for config in configs:
        start = time.perf_counter()
        run = run_uplift(frame, config, FEATURES)
        timings[f"{config.outcome}_seconds"] = time.perf_counter() - start
        outcomes[config.outcome] = _outcome_payload(run)
        if config.outcome == register_outcome:
            registered = run
    if registered is None:
        raise ValueError(f"register outcome {register_outcome!r} was not trained")

    evaluation = registered.evaluation
    best = evaluation["best_model"]
    thresholds = budget_thresholds(registered.scores[best], registered.config.budgets)
    model_dir = output_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    save_learner(
        registered.learners[best],
        model_dir,
        metadata={
            "outcome": register_outcome,
            "score": score_note(register_outcome),
            "budget_thresholds": thresholds,
            "test_qini_coefficient": evaluation["models"][best]["qini_coefficient"],
            "dataset": dataset,
            "n_train": registered.n_train,
            "git_sha": git["sha"],
        },
    )
    payload: dict[str, Any] = {
        "dataset": dataset,
        "features": list(FEATURES),
        "git": git,
        "platform": {
            "python": platform.python_version(),
            "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        },
        "timings": timings,
        "outcomes": outcomes,
        "model": {
            "outcome": register_outcome,
            "kind": best,
            "directory": str(model_dir),
            "budget_thresholds": thresholds,
        },
        "mlflow": None,
    }
    return payload, registered


def log_to_mlflow(
    payload: dict[str, Any],
    run: UpliftRun,
    output_dir: Path,
    env: Mapping[str, str],
    mlflow_root: Path,
) -> dict[str, Any]:
    """Log the registered outcome's run, register its model, add the other outcomes' metrics."""
    from uplift_lab.uplift.tracking import TrackingConfig, flatten_metrics, log_uplift_run

    dirty = payload["git"]["dirty"]
    tags = {"git_sha": str(payload["git"]["sha"]), "git_dirty": str(dirty).lower()}
    tags.update({f"dataset_{k}": str(v) for k, v in payload["dataset"].items() if k != "source"})
    tags["platform"] = "databricks" if payload["platform"]["databricks_runtime"] else "local"
    config = TrackingConfig.from_env(mlflow_root, env)
    tracked = log_uplift_run(
        run, output_dir / "model", config=config, tags=tags, report_files=[output_dir / REPORT_NAME]
    )
    import mlflow

    extra: dict[str, float] = {f"stage/{k}": v for k, v in payload["timings"].items()}
    for outcome, result in payload["outcomes"].items():
        if outcome != run.config.outcome:
            evaluation = result["evaluation"]
            extra.update(flatten_metrics({"ate": evaluation["ate"]}, f"{outcome}/"))
            extra.update(flatten_metrics(evaluation["models"], f"{outcome}/"))
    with mlflow.start_run(run_id=tracked.run_id):
        mlflow.log_metrics(extra)
    return vars(tracked)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-data Criteo uplift training job.")
    parser.add_argument("--source", type=Path, required=True, help="Criteo v2.1 .csv.gz path")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/criteo-full"))
    parser.add_argument("--publish-dir", type=Path, default=None, help="copy outputs here")
    parser.add_argument("--outcomes", nargs="+", choices=OUTCOMES, default=list(OUTCOMES))
    parser.add_argument("--register-outcome", choices=OUTCOMES, default="visit")
    parser.add_argument("--sample-rows", type=int, default=None, help="seeded smoke-run sample")
    parser.add_argument("--skip-checksum", action="store_true", help="size check only")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=LgbmParams().n_estimators)
    parser.add_argument("--learners", nargs="+", choices=sorted(LEARNERS), default=list(LEARNERS))
    parser.add_argument("--git-sha", default=None, help="commit trained (default: git HEAD)")
    parser.add_argument("--mlflow", action="store_true", help="log the run and register the model")
    parser.add_argument("--mlflow-root", type=Path, default=Path("mlruns"))
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--registry-uri", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--model-name", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.register_outcome not in args.outcomes:
        parser.error("--register-outcome must be one of --outcomes")
    if args.sample_rows is not None and args.sample_rows < 100:
        parser.error("--sample-rows must be at least 100")
    try:
        configs = [
            UpliftConfig(
                outcome=outcome,
                test_size=args.test_size,
                seed=args.seed,
                learners=tuple(args.learners),
                params=LgbmParams(n_estimators=args.n_estimators, seed=args.seed),
                n_bootstrap=args.n_bootstrap,
                budgets=DEFAULT_BUDGETS,
            )
            for outcome in args.outcomes
        ]
    except ValueError as error:
        parser.error(str(error))

    total = time.perf_counter()
    timings: dict[str, float] = {}
    start = time.perf_counter()
    source = check_source(args.source, CRITEO, verify_checksum=not args.skip_checksum)
    timings["verify_source_seconds"] = time.perf_counter() - start
    print(f"source {source.path}: {source.size_bytes} bytes, sha256 verified {source.verified}")

    start = time.perf_counter()
    frame = read_criteo_csv(args.source, expected_rows=EXPECTED_ROWS)
    timings["read_csv_seconds"] = time.perf_counter() - start
    frame = sample_frame(frame, args.sample_rows, args.register_outcome, args.seed)
    start = time.perf_counter()
    dataset = {
        "n_rows": len(frame),
        "frame_sha256": frame_sha256(frame),
        "source_sha256": CRITEO.sha256,
        "source": vars(source),
    }
    timings["frame_hash_seconds"] = time.perf_counter() - start
    print(f"loaded {len(frame):,} rows in {timings['read_csv_seconds']:.1f} s")

    sha, dirty = (args.git_sha, None) if args.git_sha else git_state()
    payload, run = train_full(
        frame,
        args.output_dir,
        configs,
        register_outcome=args.register_outcome,
        dataset=dataset,
        git={"sha": sha, "dirty": dirty},
        timings=timings,
    )
    del frame
    report = write_json(args.output_dir / REPORT_NAME, payload)
    if args.mlflow:
        start = time.perf_counter()
        payload["mlflow"] = log_to_mlflow(
            payload, run, args.output_dir, tracking_env(args, os.environ), args.mlflow_root
        )
        timings["mlflow_seconds"] = time.perf_counter() - start
    timings["total_seconds"] = time.perf_counter() - total
    write_json(report, payload)
    if args.publish_dir is not None:
        shutil.copytree(args.output_dir, args.publish_dir, dirs_exist_ok=True)

    for line in summary_lines(payload):
        print(line)
    return 0


def summary_lines(payload: Mapping[str, Any]) -> list[str]:
    """Human-readable headline numbers per outcome and the stage timings."""
    lines = [f"rows {payload['dataset']['n_rows']:,}, git {payload['git']['sha']}"]
    for outcome, result in payload["outcomes"].items():
        evaluation = result["evaluation"]
        ate = evaluation["ate"]
        lines.append(
            f"[{outcome}] ATE {ate['estimate']:+.5f} [{ate['ci_low']:+.5f}, {ate['ci_high']:+.5f}]"
            f"; best {evaluation['best_model']}, clear winner {evaluation['best_is_clear_winner']}"
        )
        for name, model in evaluation["models"].items():
            qini = model["qini_coefficient"]
            per_1k = model["policy"]["0.1"]["model_per_1000_contacts"]["estimate"]
            lines.append(
                f"  {name:20s} Qini {qini['estimate']:+.5f} [{qini['ci_low']:+.5f}, "
                f"{qini['ci_high']:+.5f}]  per 1k contacts @10% {per_1k:.1f}"
            )
    lines.append("timings: " + ", ".join(f"{k} {v:.1f}" for k, v in payload["timings"].items()))
    if payload["mlflow"]:
        lines.append(
            f"MLflow run {payload['mlflow']['run_id']} -> {payload['mlflow']['model_uri']}"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
