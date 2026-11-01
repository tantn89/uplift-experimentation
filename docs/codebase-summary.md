# Codebase summary

Navigation only. Behaviour is in the modules (each starts with a docstring that states its design);
reasons are in [system-architecture.md](system-architecture.md). Python 3.11+, package `uplift_lab`
under `src/`, built with hatchling. The version is in `src/uplift_lab/__init__.py`.

## Source map

| Path | Owns |
|---|---|
| `data/` | `cache.py` (fetch, SHA-256 verify, cache dir), `hillstrom.py`, `criteo.py` (pinned source, seeded stratified 1M subsample), `sampling.py`, `synthetic.py` (Criteo-shaped RCT with a known CATE, used by tests, the CI drift job and the metric gate) |
| `experiments/` | `estimators.py` (Welch), `cuped.py`, `srm.py`, `power.py`, `msprt.py`, `peeking_simulation.py`, `peeking_price.py`, `cuped_simulation.py` |
| `uplift/` | `learners.py` (S, T, X, transformed outcome), `metrics.py` (Qini, AUUC, uplift@k), `policy.py`, `bootstrap.py`, `evaluation.py`, `pipeline.py`, `propensity.py`, `features.py` (leakage guard), `artifacts.py`, `provenance.py`, `tracking.py` and `pyfunc_model.py` (MLflow) |
| `pipelines/` | `criteo_full_data.py` (verified full-file reader), `criteo_full_training.py` (full-data job, local or Databricks) |
| `serving/` | `app.py` (FastAPI), `schemas.py`, `model_service.py`, `model_source.py` (directory or MLflow URI), `settings.py` (env), `benchmark.py` |
| `monitoring/` | `drift.py` (PSI, KS, decision rate, thresholds), `shift_simulation.py` (seeded scenarios), `traffic_log.py` |
| `reports/` | CLIs `hillstrom.py`, `sequential.py`, `criteo_uplift.py`, `drift.py`, with their `*_analysis.py`, `*_figures.py` and `drift_render.py` helpers; `output.py` (strict JSON) |
| `gate/` | `metric_gate.py` (payload, fingerprint, rule), `cli.py` (`uplift-metric-gate`, Markdown summary) |
| `lazy_exports.py` | PEP 562 lazy re-exports, so the serving import path stays free of the analysis stack |

## Commands

| Command | Extra | Writes |
|---|---|---|
| `uplift-hillstrom-report` | `report` | `reports/hillstrom/` (JSON, 2 PNG) |
| `uplift-sequential-report [--large]` | `report` | `reports/sequential/` (JSON, 2 PNG) |
| `uplift-criteo-report [--sample-rows N] [--n-bootstrap B] [--mlflow]` | `report` (+ `mlflow`) | `reports/criteo-uplift/` (JSON, 2 PNG, `model/`) |
| `python -m uplift_lab.pipelines.criteo_full_training --source <csv.gz>` | `analysis` (+ `mlflow`) | full-data JSON and `model/` |
| `uplift-drift-report [--synthetic N] [--model-dir DIR] [--traffic-log PATH] [--expect S=STATUS]` | `report` | `reports/drift/` (JSON, PNG, Markdown, HTML) |
| `uplift-metric-gate [--update-baseline]` | `analysis` | `reports/metric-gate/` (`head.json`, `summary.md`); the baseline with `--update-baseline` |
| `uvicorn uplift_lab.serving.app:create_app --factory` | `serve` | none |
| `uplift-serve-benchmark --url URL [--token T]` | none (stdlib) | `reports/serving-benchmark/<label>.json` |

Extras: `analysis` (pyarrow, scipy, scikit-learn, statsmodels), `report` (`analysis` + matplotlib),
`mlflow` (`analysis` + MLflow 3), `serve` (FastAPI, uvicorn, pydantic), and `dev` (everything for
tests plus ruff, mypy and pytest). A plain `pip install uplift-lab` only loads and scores exported
models.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `UPLIFT_LAB_CACHE` | data loaders | Dataset cache directory (default: platformdirs user cache) |
| `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI` | `--mlflow`, full-data job | Default local sqlite `mlruns/mlflow.db`; `databricks` / `databricks-uc` on Databricks |
| `UPLIFT_LAB_MLFLOW_EXPERIMENT`, `UPLIFT_LAB_MODEL_NAME` | same | Experiment path and registered model name (`catalog.schema.model` on Unity Catalog) |
| `UPLIFT_LAB_MODEL_DIR`, `UPLIFT_LAB_MODEL_URI` | serving | Exported directory, or an MLflow URI such as `models:/uplift-criteo-visit@champion` |
| `UPLIFT_LAB_TRAFFIC_LOG`, `..._SAMPLE_RATE`, `..._MAX_ROWS`, `..._MAX_BYTES` | serving | Sampled traffic log; see [deployment-guide.md](deployment-guide.md) |

## Other top-level files

| Path | Purpose |
|---|---|
| `evals/metric-gate-baseline.json` | Committed metric-gate baseline (refresh with `uplift-metric-gate --update-baseline`) |
| `Dockerfile`, `.dockerignore` | Two-stage, non-root serving image with the model baked in; the ignore file is an allow-list |
| `databricks.yml`, `databricks/` | Asset bundle and job entry for the full-data run |
| `.github/workflows/ci.yml` | Jobs `lint-and-types`, `tests` (3.11, 3.13), `drift-report`, `metric-gate`, optional `mlflow-round-trip` |
| `tests/` | Unit and statistical tests on synthetic data; `-m network` runs the real-data tests |
