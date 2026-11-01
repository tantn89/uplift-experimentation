"""Locate the model directory to serve: a local export, or an MLflow registry download.

A registered model is the pyfunc wrapper written by ``uplift.tracking``. It carries the same
pickle-free directory as an artifact under ``artifacts/<name>/``, which is found by its
``manifest.json``. The pyfunc loader itself is not used, so serving never imports the logged
``pyfunc_model.py`` and needs only ``mlflow.artifacts`` (present in ``mlflow-skinny``).
"""

from __future__ import annotations

from pathlib import Path

from uplift_lab.serving.settings import ServingSettings
from uplift_lab.uplift.artifacts import MANIFEST, ArtifactError, read_manifest

MISSING_MLFLOW = (
    "UPLIFT_LAB_MODEL_URI needs MLflow: pip install mlflow-skinny (or the mlflow extra), "
    "or bake the model directory into the image and set UPLIFT_LAB_MODEL_DIR instead"
)


def find_exported_model(pyfunc_dir: Path) -> Path:
    """The single uplift model directory inside a downloaded pyfunc model."""
    candidates = sorted(p.parent for p in (pyfunc_dir / "artifacts").glob(f"*/{MANIFEST}"))
    if len(candidates) != 1:
        raise ArtifactError(
            f"expected one uplift model under {pyfunc_dir / 'artifacts'}, found {len(candidates)}"
        )
    read_manifest(candidates[0])
    return candidates[0]


def download_registered_model(uri: str, destination: Path) -> Path:
    """Download ``uri`` (e.g. ``models:/name@champion``) and return its model directory."""
    try:
        from mlflow.artifacts import download_artifacts
    except ImportError as error:
        raise RuntimeError(MISSING_MLFLOW) from error
    destination.mkdir(parents=True, exist_ok=True)
    local = download_artifacts(artifact_uri=uri, dst_path=str(destination))
    return find_exported_model(Path(local))


def resolve_model_dir(settings: ServingSettings, download_root: Path) -> tuple[Path, str]:
    """``(model directory, human-readable source)``; the registry URI wins when both are set."""
    if settings.model_uri:
        return download_registered_model(settings.model_uri, download_root), settings.model_uri
    if settings.model_dir is None:  # unreachable: ServingSettings validates this
        raise ValueError("no model configured")
    read_manifest(settings.model_dir)
    return settings.model_dir, str(settings.model_dir)
