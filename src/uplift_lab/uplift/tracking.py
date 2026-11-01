"""MLflow tracking and model registry for an uplift run (needs the optional ``mlflow`` extra).

Configuration comes from the environment, so the same code path runs locally and on Databricks:

- ``MLFLOW_TRACKING_URI``: default ``sqlite:///<root>/mlflow.db`` with ``root = ./mlruns``
  (git-ignored) and run artifacts as files in ``<root>/artifacts``; on Databricks ``databricks``.
  MLflow 3 keeps the plain ``./mlruns`` file store in maintenance mode and refuses it by
  default, hence a local sqlite file instead.
- ``MLFLOW_REGISTRY_URI``: default the tracking URI; ``databricks-uc`` for Unity Catalog.
- ``UPLIFT_LAB_MLFLOW_EXPERIMENT``: default ``uplift-criteo`` (Databricks wants a workspace
  path such as ``/Users/<you>/uplift-criteo``).
- ``UPLIFT_LAB_MODEL_NAME``: default ``uplift-criteo-visit`` (Unity Catalog wants
  ``catalog.schema.model``).

A run logs the configuration as params; metrics with CI bounds and timings; the git SHA, dataset
checksums, best model and champion (with its rule) as tags; the report files; and registers the
champion as a pyfunc model (``pyfunc_model.py`` wrapping the ``artifacts`` directory) under the
alias ``champion``.

The pyfunc imports ``uplift_lab``, which is not on PyPI. The package source is therefore logged
with the model (``code_paths``; MLflow puts it on ``sys.path`` before loading), and the pip
requirements pin its core runtime dependencies, so the registered model loads in a clean
environment that has only those requirements installed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_lab.uplift.artifacts import PYFUNC_ARTIFACT_KEY
from uplift_lab.uplift.pipeline import UpliftRun

MISSING_MLFLOW = 'MLflow tracking needs the mlflow extra: python -m pip install -e ".[mlflow]"'
CHAMPION_ALIAS = "champion"
PYFUNC_SCRIPT = Path(__file__).with_name("pyfunc_model.py")


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str
    registry_uri: str | None
    experiment: str
    model_name: str
    artifact_root: str | None = None

    @classmethod
    def from_env(
        cls, root: Path = Path("mlruns"), env: Mapping[str, str] | None = None
    ) -> TrackingConfig:
        """Resolve the configuration from ``env`` (default ``os.environ``) and a local root."""
        values = os.environ if env is None else env
        tracking_uri = values.get("MLFLOW_TRACKING_URI")
        artifact_root = None
        if not tracking_uri:
            local = root.resolve()
            local.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"sqlite:///{(local / 'mlflow.db').as_posix()}"
            artifact_root = (local / "artifacts").as_uri()
        return cls(
            tracking_uri=tracking_uri,
            registry_uri=values.get("MLFLOW_REGISTRY_URI") or None,
            experiment=values.get("UPLIFT_LAB_MLFLOW_EXPERIMENT") or "uplift-criteo",
            model_name=values.get("UPLIFT_LAB_MODEL_NAME") or "uplift-criteo-visit",
            artifact_root=artifact_root,
        )


@dataclass(frozen=True)
class TrackedRun:
    run_id: str
    experiment_id: str
    model_uri: str
    model_name: str
    model_version: str
    alias: str = CHAMPION_ALIAS


def flatten_metrics(node: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """``{estimate, ci_low, ci_high}`` leaves of a nested payload as MLflow metric names."""
    metrics: dict[str, float] = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and "estimate" in value:
            for part, suffix in (("estimate", ""), ("ci_low", "_ci_low"), ("ci_high", "_ci_high")):
                metrics[name + suffix] = float(value[part])
        elif isinstance(value, Mapping):
            metrics.update(flatten_metrics(value, f"{name}/"))
    return metrics


#: Databricks rejects more than 1,000 metrics linked to one MLflow 3 LoggedModel. ``log_model``
#: links every metric the run already has, and metrics logged afterwards are linked too, so the
#: run logs one curated set (well under the cap) in a single step before the model. The full
#: evaluation travels in the JSON report artifact.
MAX_RUN_METRICS = 1000
#: Policy-table columns logged as metrics; the rest (shares, treat-all, random per user) follow
#: from these and the ATE, and are in the JSON report.
LOGGED_POLICY_FIELDS = (
    "model_per_1000_contacts",
    "model_ipw_per_1000_contacts",
    "gain_over_random_per_1000_contacts",
)


def evaluation_metrics(evaluation: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """The curated metric set of one evaluation payload (about 300 for five scorings)."""
    models = {
        name: {
            **{k: v for k, v in model.items() if k != "policy"},
            "policy": {
                budget: {k: row[k] for k in LOGGED_POLICY_FIELDS}
                for budget, row in model["policy"].items()
            },
        }
        for name, model in evaluation["models"].items()
    }
    return {
        **flatten_metrics({"ate": evaluation["ate"]}, prefix),
        **flatten_metrics(models, prefix),
        **flatten_metrics({"qini_diff_vs_best": evaluation["qini_diff_vs_best"]}, prefix),
        **flatten_metrics({"propensity_auc": evaluation["propensity_check"]["auc"]}, prefix),
    }


def outcome_metrics(payload: Mapping[str, Any], registered_outcome: str) -> dict[str, float]:
    """Stage timings and every other outcome's curated metrics, for a multi-outcome job."""
    extra: dict[str, float] = {f"stage/{k}": float(v) for k, v in payload["timings"].items()}
    for outcome, result in payload["outcomes"].items():
        if outcome != registered_outcome:
            extra.update(evaluation_metrics(result["evaluation"], f"{outcome}/"))
    return extra


def log_uplift_run(
    run: UpliftRun,
    model_dir: Path,
    *,
    config: TrackingConfig,
    tags: Mapping[str, str],
    report_files: Sequence[Path] = (),
    extra_metrics: Mapping[str, float] | None = None,
) -> TrackedRun:
    """Log ``run`` to MLflow and register the model in ``model_dir`` as the champion.

    ``extra_metrics`` (e.g. other outcomes, stage timings) are logged with the run's own metrics,
    all before the model and nothing after it: MLflow 3 links the run's metrics to the
    LoggedModel, where Databricks caps them at ``MAX_RUN_METRICS``.
    """
    try:
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types import ColSpec, Schema
    except ImportError as error:
        raise RuntimeError(MISSING_MLFLOW) from error

    mlflow.set_tracking_uri(config.tracking_uri)
    if config.registry_uri:
        mlflow.set_registry_uri(config.registry_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(config.experiment, artifact_location=config.artifact_root)
    )
    evaluation = run.evaluation
    params = {
        **{k: v for k, v in run.config.to_dict().items() if k != "params"},
        **{f"lgbm_{k}": v for k, v in run.config.params.to_dict().items()},
        "features": ",".join(run.features),
        "n_train": run.n_train,
        "n_test": run.n_test,
    }
    metrics = {**evaluation_metrics(evaluation), **run.timings, **(extra_metrics or {})}
    if len(metrics) >= MAX_RUN_METRICS:
        raise ValueError(
            f"{len(metrics)} metrics would exceed the {MAX_RUN_METRICS}-metric cap; "
            "keep the detail in the JSON report"
        )
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", name) for name in run.features]),
        outputs=Schema([ColSpec("double", "uplift")]),
    )
    champion = evaluation["champion"]
    with mlflow.start_run(experiment_id=experiment_id) as active:
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                **tags,
                "best_model": evaluation["best_model"],
                "champion": champion["model"],
                "champion_rule": champion["rule"],
                # MLflow caps tag values at 8,000 characters; the reason is a sentence.
                "champion_reason": champion["reason"],
            }
        )
        mlflow.log_metrics(metrics)  # the only metric call: log_model links these to the model
        for path in report_files:
            mlflow.log_artifact(str(path), artifact_path="report")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(PYFUNC_SCRIPT),
            artifacts={PYFUNC_ARTIFACT_KEY: str(model_dir)},
            signature=signature,
            # A synthetic all-zero row: MLflow checks the model scores it, and no data is copied.
            input_example=pd.DataFrame(0.0, index=[0], columns=list(run.features)),
            registered_model_name=config.model_name,
            pip_requirements=pip_requirements(),
            code_paths=code_paths(),
        )
    version = str(info.registered_model_version)
    mlflow.MlflowClient().set_registered_model_alias(config.model_name, CHAMPION_ALIAS, version)
    return TrackedRun(
        run_id=active.info.run_id,
        experiment_id=str(experiment_id),
        model_uri=f"models:/{config.model_name}/{version}",
        model_name=config.model_name,
        model_version=version,
    )


#: What loading and scoring the pyfunc imports besides ``uplift_lab`` itself: the package's core
#: dependencies (``pyproject.toml``) plus MLflow.
PYFUNC_REQUIREMENTS = ("lightgbm", "numpy", "pandas", "platformdirs", "mlflow")


def pip_requirements() -> list[str]:
    """Pinned runtime requirements, so MLflow does not spend time inferring them."""
    from importlib.metadata import version

    return [f"{name}=={version(name)}" for name in PYFUNC_REQUIREMENTS]


def code_paths() -> list[str]:
    """The ``uplift_lab`` package directory, logged with the model as its source code."""
    import uplift_lab

    return [str(Path(uplift_lab.__file__).resolve().parent)]
