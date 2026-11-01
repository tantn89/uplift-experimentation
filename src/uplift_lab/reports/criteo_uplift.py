"""Criteo uplift report: ``python -m uplift_lab.reports.criteo_uplift [--mlflow]``.

Loads the seeded 1M-row Criteo subsample (built once from the checksummed source), fits the S-,
T-, X-learners and the transformed-outcome baseline on a stratified 70/30 split with ``visit``
as the outcome, and evaluates them on the test set with paired bootstrap intervals. Writes
``criteo-uplift.json``, two PNG figures and the best model's serving directory (``model/``);
``--mlflow`` also logs the run and registers that model. ``--sample-rows`` runs on a seeded
stratified sample for a quick smoke test.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_lab.data.criteo import CRITEO, DEFAULT_SEED, FEATURES, load_criteo_subsample
from uplift_lab.data.sampling import stratified_indices
from uplift_lab.reports.output import MISSING_MATPLOTLIB, write_json
from uplift_lab.uplift.artifacts import save_learner
from uplift_lab.uplift.evaluation import DEFAULT_BUDGETS
from uplift_lab.uplift.learners import LEARNERS, LgbmParams
from uplift_lab.uplift.pipeline import UpliftConfig, budget_thresholds, run_uplift
from uplift_lab.uplift.provenance import frame_sha256, git_state

SCORE_NOTE = "P(visit | x, assigned treatment) - P(visit | x, assigned control); intent to treat"


def sample_frame(frame: pd.DataFrame, n_rows: int | None, outcome: str, seed: int) -> pd.DataFrame:
    """Seeded sample stratified by treatment x outcome; the whole frame if ``n_rows`` covers it."""
    if n_rows is None or n_rows >= len(frame):
        return frame
    keep = stratified_indices(frame, ["treatment", outcome], n_rows=n_rows, seed=seed)
    return frame.iloc[keep].reset_index(drop=True)


def write_criteo_uplift_report(
    frame: pd.DataFrame,
    output_dir: Path,
    config: UpliftConfig,
    *,
    features: Sequence[str] = FEATURES,
    figures: bool = True,
    mlflow_root: Path | None = None,
) -> dict[str, Any]:
    """Run the uplift pipeline on ``frame`` and write every output; returns the JSON payload."""
    run = run_uplift(frame, config, features)
    evaluation = run.evaluation
    best = evaluation["best_model"]
    sha, dirty = git_state()
    dataset = {
        "n_rows": len(frame),
        "outcome": config.outcome,
        "frame_sha256": frame_sha256(frame),
        "source_sha256": CRITEO.sha256,
    }
    model_dir = output_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    thresholds = budget_thresholds(run.scores[best], config.budgets)
    save_learner(
        run.learners[best],
        model_dir,
        metadata={
            "outcome": config.outcome,
            "score": SCORE_NOTE,
            "budget_thresholds": thresholds,
            "test_qini_coefficient": evaluation["models"][best]["qini_coefficient"],
            "dataset": dataset,
            "n_train": run.n_train,
            "git_sha": sha,
        },
    )
    payload: dict[str, Any] = {
        "dataset": dataset,
        "config": config.to_dict(),
        "features": list(run.features),
        "n_train": run.n_train,
        "n_test": run.n_test,
        "timings": run.timings,
        "git": {"sha": sha, "dirty": dirty},
        "model": {"kind": best, "directory": str(model_dir), "budget_thresholds": thresholds},
        "evaluation": evaluation,
        "mlflow": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [write_json(output_dir / "criteo-uplift.json", payload)]
    if figures:
        try:
            from uplift_lab.reports.criteo_uplift_figures import deciles_figure, qini_figure
        except ImportError as error:
            raise SystemExit(MISSING_MATPLOTLIB) from error
        files.append(qini_figure(evaluation, output_dir / "criteo-uplift-qini.png"))
        files.append(deciles_figure(evaluation, output_dir / "criteo-uplift-deciles.png"))
    if mlflow_root is not None:
        from uplift_lab.uplift.tracking import TrackingConfig, log_uplift_run

        tags = {"git_sha": sha, "git_dirty": str(dirty).lower()}
        tags.update({f"dataset_{key}": str(value) for key, value in dataset.items()})
        tracking = TrackingConfig.from_env(mlflow_root)
        tracked = log_uplift_run(run, model_dir, config=tracking, tags=tags, report_files=files)
        payload["mlflow"] = vars(tracked)
        write_json(files[0], payload)
    return payload


def _summary_lines(payload: dict[str, Any]) -> list[str]:
    evaluation = payload["evaluation"]
    ate = evaluation["ate"]
    lines = [f"ATE visit {ate['estimate']:+.5f} [{ate['ci_low']:+.5f}, {ate['ci_high']:+.5f}]"]
    budget = f"{evaluation['budgets'][len(evaluation['budgets']) // 2]:g}"
    for name, model in evaluation["models"].items():
        qini, policy = model["qini_coefficient"], model["policy"][budget]
        lines.append(
            f"{name:20s} Qini {qini['estimate']:+.5f} [{qini['ci_low']:+.5f}, "
            f"{qini['ci_high']:+.5f}]  AUUC {model['auuc']['estimate']:.5f}  "
            f"uplift@{budget} {model['uplift_at_k'][budget]['estimate']:+.4f}  "
            f"per 1k contacts @{budget} {policy['model_per_1000_contacts']['estimate']:.1f} "
            f"(treat-all {policy['treat_all_per_1000_contacts']['estimate']:.1f})"
        )
    lines.append(
        f"best {payload['model']['kind']}, beats random: {evaluation['best_beats_random']}"
    )
    if payload["mlflow"]:
        tracked = payload["mlflow"]
        lines.append(f"MLflow run {tracked['run_id']} registered {tracked['model_uri']}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criteo uplift modelling report.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/criteo-uplift"))
    parser.add_argument("--cache", type=Path, default=None, help="dataset cache directory")
    parser.add_argument("--sample-rows", type=int, default=None, help="seeded stratified sample")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=LgbmParams().n_estimators)
    parser.add_argument("--learners", nargs="+", choices=sorted(LEARNERS), default=list(LEARNERS))
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--mlflow", action="store_true", help="log the run and register the model")
    parser.add_argument("--mlflow-root", type=Path, default=Path("mlruns"))
    args = parser.parse_args(argv)
    if args.sample_rows is not None and args.sample_rows < 100:
        parser.error("--sample-rows must be at least 100")
    try:
        config = UpliftConfig(
            test_size=args.test_size,
            seed=args.seed,
            learners=tuple(args.learners),
            params=LgbmParams(n_estimators=args.n_estimators, seed=args.seed),
            n_bootstrap=args.n_bootstrap,
            budgets=tuple(args.budgets) if args.budgets else DEFAULT_BUDGETS,
        )
    except ValueError as error:
        parser.error(str(error))

    start = time.perf_counter()
    full = load_criteo_subsample(cache=args.cache)
    frame = sample_frame(full, args.sample_rows, config.outcome, args.seed)
    payload = write_criteo_uplift_report(
        frame,
        args.output_dir,
        config,
        figures=not args.no_figures,
        mlflow_root=args.mlflow_root if args.mlflow else None,
    )
    print("\n".join(_summary_lines(payload)))
    print(f"wrote {args.output_dir} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
