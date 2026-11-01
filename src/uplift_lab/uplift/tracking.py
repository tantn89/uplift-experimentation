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
checksums and best model as tags; the report files; and registers the best learner as a pyfunc
model (``pyfunc_model.py`` wrapping the ``artifacts`` directory) under the alias ``champion``.
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


def log_uplift_run(
    run: UpliftRun,
    model_dir: Path,
    *,
    config: TrackingConfig,
    tags: Mapping[str, str],
    report_files: Sequence[Path] = (),
) -> TrackedRun:
    """Log ``run`` to MLflow and register the model in ``model_dir`` as the champion."""
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
    metrics = {
        **flatten_metrics({"ate": evaluation["ate"]}),
        **flatten_metrics(evaluation["models"]),
        **flatten_metrics({"qini_diff_vs_best": evaluation["qini_diff_vs_best"]}),
        **flatten_metrics({"propensity_auc": evaluation["propensity_check"]["auc"]}),
        **run.timings,
    }
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", name) for name in run.features]),
        outputs=Schema([ColSpec("double", "uplift")]),
    )
    with mlflow.start_run(experiment_id=experiment_id) as active:
        mlflow.log_params(params)
        mlflow.set_tags({**tags, "best_model": evaluation["best_model"]})
        mlflow.log_metrics(metrics)
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
            pip_requirements=_pip_requirements(),
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


def _pip_requirements() -> list[str]:
    """Pinned runtime requirements, so MLflow does not spend time inferring them."""
    from importlib.metadata import version

    return [f"{name}=={version(name)}" for name in ("lightgbm", "numpy", "pandas", "mlflow")]
