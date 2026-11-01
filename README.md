# uplift-experimentation

Work in progress. Two questions a growth team asks, answered on public randomised data:

1. **Did the campaign work, and how sure are we?** A/B analysis on the Hillstrom e-mail RCT:
   effect estimates with confidence intervals, CUPED, sample-ratio-mismatch checks, power
   analysis and sequential tests that stay valid under peeking.
2. **Whom should we target next time?** Uplift modelling on the Criteo Uplift dataset:
   meta-learners evaluated with Qini/AUUC and policy value under a budget.

## Data

Nothing is committed. Datasets download on first use into a per-user cache (override with
`UPLIFT_LAB_CACHE`) and are verified against pinned SHA-256 checksums.

| Dataset | Source | Licence |
|---|---|---|
| Hillstrom MineThatData E-Mail Analytics (64,000 rows) | [minethatdata.com](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) | Published by Kevin Hillstrom for the 2008 challenge; no formal licence |
| Criteo Uplift Prediction v2.1 (13,979,592 rows) | [huggingface.co/datasets/criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

Criteo data: Diemert, Betlei, Renaudin and Amini, *A Large Scale Benchmark for Uplift
Modeling*, AdKDD 2018. Use is non-commercial; this repository ships code only.

## Reports

Each report writes JSON metrics and PNG figures into `reports/` (git-ignored). Figures need the
`report` extra (`python -m pip install -e ".[report]"`).

```bash
uplift-hillstrom-report      # Welch effects, CUPED, SRM and power on Hillstrom -> reports/hillstrom
uplift-sequential-report     # seeded peeking simulation, naive t-test vs mSPRT -> reports/sequential
uplift-sequential-report --large   # 20,000 runs x 100 looks instead of 10,000 x 20
```

Both are also runnable as `python -m uplift_lab.reports.hillstrom` and
`python -m uplift_lab.reports.sequential`.

```bash
uplift-criteo-report               # S/T/X-learners + transformed outcome on the Criteo 1M subsample
                                   # (visit, 70/30 split, 200 bootstrap draws) -> reports/criteo-uplift
uplift-criteo-report --sample-rows 50000 --n-bootstrap 50   # seeded stratified smoke run
uplift-criteo-report --mlflow      # also log the run to MLflow and register the best model
```

The default Criteo run takes about 35 s on a laptop: 8 s to fit the four learners on
700,000 rows and 22 s for the paired bootstrap. The first run also builds the subsample
(about 15 s). Learners use the twelve pre-treatment features `f0`-`f11` only. `exposure`
is post-treatment and is rejected as a feature, so every effect is intent-to-treat. The
report writes `criteo-uplift.json` (Qini coefficient, AUUC, uplift@k and policy value per
1,000 contacts against treat-all and random, each with a bootstrap CI), Qini and
uplift-by-decile PNGs, and the best model's serving directory `model/`.

"Best" only means the highest point estimate. The report also gives a paired bootstrap CI
(same draws) on each learner's Qini coefficient minus the best one's; `best_is_clear_winner`
is true only if every one of those intervals excludes 0, and no winner is named otherwise.
A randomisation check fits treatment ~ features on the training split and reports the held-out
AUC with a bootstrap CI (`propensity_check`). 0.5 means assignment is unpredictable, as a
clean RCT should be.

### Model artifact

`model/` holds `manifest.json` and one LightGBM text model per component. The manifest
records the format, `schema_version` 1, learner `kind`, ordered `features`, training
`propensity`, the booster files, and metadata (outcome, dataset checksum, score thresholds per
budget). `uplift_lab.uplift.load_learner(path)` rebuilds the model without MLflow or pickle.
The registered MLflow pyfunc model wraps the same directory. It takes a DataFrame of features
and returns an `uplift` column.

### MLflow

Install the extra (`python -m pip install -e ".[mlflow]"`). Locally, runs go to
`mlruns/mlflow.db` (sqlite) with artifacts in `mlruns/artifacts/`, all git-ignored; MLflow 3
no longer accepts a plain `./mlruns` file store by default. The best model is registered as
`uplift-criteo-visit` with the alias `champion`. Set `MLFLOW_TRACKING_URI`,
`MLFLOW_REGISTRY_URI`, `UPLIFT_LAB_MLFLOW_EXPERIMENT` and `UPLIFT_LAB_MODEL_NAME` to point the
same code at another server, e.g. Databricks (`databricks`, `databricks-uc`, a workspace
experiment path, `catalog.schema.model`).

## Serving

`uplift_lab.serving` is a FastAPI app (`serve` extra) with `GET /health`, `POST /score` (uplift
per customer) and `POST /decide` (treat/skip under a named budget threshold from the manifest,
or for the top share of a batch). Rows must carry exactly `f0`-`f11`. `exposure`, treatment and
outcome fields are rejected with 422, and batches are capped at 1,000 rows. The model loads
once at startup from `UPLIFT_LAB_MODEL_DIR`, or from an MLflow URI in `UPLIFT_LAB_MODEL_URI`
such as `models:/uplift-criteo-visit@champion`. Sampled traffic can be logged to JSONL for
drift checks.

```bash
UPLIFT_LAB_MODEL_DIR=reports/criteo-uplift/model \
  uvicorn uplift_lab.serving.app:create_app --factory --port 8000
uplift-serve-benchmark --url http://127.0.0.1:8000     # p50/p95 for /score and /decide
docker build -t uplift-serve . && docker run --rm -p 8080:8080 uplift-serve
```

The image bakes the exported model directory in, runs as a non-root user and has no MLflow.
See [docs/deployment-guide.md](docs/deployment-guide.md) for configuration, the Docker design
and the Cloud Run commands (not deployed yet).

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests && mypy && pytest
```

## Licence

Code: MIT. Data: see the table above.
