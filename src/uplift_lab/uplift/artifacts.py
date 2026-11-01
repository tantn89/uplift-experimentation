"""A small, pickle-free on-disk contract for a fitted uplift learner, used for serving.

Layout of a model directory (schema version 1)::

    manifest.json        format, schema_version, kind, features, propensity, boosters,
                         params, lightgbm_version, uplift_lab_version, metadata
    <component>.txt      one LightGBM text model per component named in "boosters"

``kind`` is one of ``s_learner``, ``t_learner``, ``x_learner``, ``transformed_outcome``;
``features`` is the exact column order the model expects; ``propensity`` is the training
treatment share. ``metadata`` is free-form JSON written by the training pipeline (outcome,
dataset checksum, score thresholds per budget, test metrics). Loading needs only numpy, pandas,
lightgbm and this package; MLflow is not required. The MLflow pyfunc model wraps this same
directory, so a registry download and a plain copy load identically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightgbm as lgb

from uplift_lab import __version__
from uplift_lab.uplift.learners import LEARNERS, LgbmParams, UpliftLearner

FORMAT = "uplift-lab.uplift-model"
SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
# Name under which the MLflow pyfunc model stores this directory (``context.artifacts[...]``).
PYFUNC_ARTIFACT_KEY = "uplift_model"


class ArtifactError(ValueError):
    """A model directory does not satisfy the artifact contract."""


def save_learner(
    learner: UpliftLearner, directory: Path, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Write ``learner`` as a manifest plus LightGBM text models; returns the manifest path."""
    if not learner.boosters_:
        raise ValueError("cannot save an unfitted learner")
    directory.mkdir(parents=True, exist_ok=True)
    files = {name: f"{name}.txt" for name in learner.boosters_}
    for name, filename in files.items():
        learner.boosters_[name].save_model(str(directory / filename))
    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "kind": learner.kind,
        "features": list(learner.features_),
        "propensity": learner.propensity_,
        "boosters": files,
        "params": learner.params.to_dict(),
        "lightgbm_version": lgb.__version__,
        "uplift_lab_version": __version__,
        "metadata": dict(metadata or {}),
    }
    path = directory / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read and validate ``manifest.json``; raises ArtifactError on any contract violation."""
    path = directory / MANIFEST
    if not path.is_file():
        raise ArtifactError(f"no {MANIFEST} in {directory}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ArtifactError(f"{path} is not an {FORMAT} manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {manifest.get('schema_version')!r}")
    if manifest.get("kind") not in LEARNERS:
        raise ArtifactError(f"unknown learner kind {manifest.get('kind')!r}")
    boosters = manifest.get("boosters")
    if not isinstance(boosters, dict) or not all(
        isinstance(f, str) and Path(f).name == f and (directory / f).is_file()
        for f in boosters.values()
    ):
        raise ArtifactError("boosters must map component names to model files in the directory")
    return manifest


def load_learner(directory: Path) -> UpliftLearner:
    """Rebuild the fitted learner described by a model directory."""
    manifest = read_manifest(directory)
    boosters = {
        name: lgb.Booster(model_file=str(directory / filename))
        for name, filename in manifest["boosters"].items()
    }
    try:
        return LEARNERS[manifest["kind"]].from_boosters(
            boosters,
            tuple(manifest["features"]),
            float(manifest["propensity"]),
            LgbmParams(**manifest["params"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"invalid manifest in {directory}: {error}") from error
