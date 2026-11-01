# Deployment guide: scoring service

The scoring service is a FastAPI app (`uplift_lab.serving`, `serve` extra). It loads one
exported uplift model at startup and serves:

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | none | `status` plus model info (kind, features, git SHA, source, named budgets) |
| `POST /score` | `{"customers": [{"id"?, "f0": .., ..., "f11": ..}, ...]}` (1-1000 rows) | uplift per customer |
| `POST /decide` | the same plus `budget` (a named budget such as `"0.1"`) **or** `budget_fraction` | uplift plus treat/skip per customer, and the rule used |

Validation happens at the boundary, and every failure returns 422. Each row needs exactly
`f0`-`f11` as finite numbers. Unknown fields are rejected. `treatment`, `exposure`
(post-treatment), `visit` and `conversion` are rejected with a leakage message. Batches over
1000 rows are rejected. Error responses do not echo inputs.

Decision rules:

- `budget` uses the manifest threshold for that budget. These are score quantiles on the
  held-out test set. A customer is treated if `uplift >= threshold`, which is the same decision
  as offline `load_learner(dir).predict(df) >= threshold`, whatever the batch size.
- `budget_fraction` treats the top `ceil(fraction * n)` rows of this batch. Use it for one call
  per campaign list. It is relative to the batch, so a batch of one is always treated.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `UPLIFT_LAB_MODEL_DIR` | `/app/model` in the image | exported model directory (`manifest.json` + LightGBM text files) |
| `UPLIFT_LAB_MODEL_URI` | unset | MLflow URI such as `models:/uplift-criteo-visit@champion`; wins over the directory |
| `UPLIFT_LAB_TRAFFIC_LOG` | unset (off) | JSONL path for sampled traffic, read by `uplift-drift-report --traffic-log` |
| `UPLIFT_LAB_TRAFFIC_SAMPLE_RATE` | `0.05` | share of requests recorded |
| `UPLIFT_LAB_TRAFFIC_MAX_ROWS` | `20` | rows kept per recorded request (random subset; ids never logged) |
| `UPLIFT_LAB_TRAFFIC_MAX_BYTES` | `10000000` | rotation size; disk use stays below twice this |

A bad model directory fails startup, not the first request.

## Run locally

```bash
python -m pip install -e ".[serve,report]"   # report: the analysis stack for training
uplift-criteo-report                      # exports reports/criteo-uplift/model
UPLIFT_LAB_MODEL_DIR=reports/criteo-uplift/model \
  uvicorn uplift_lab.serving.app:create_app --factory --port 8000
uplift-serve-benchmark --url http://127.0.0.1:8000 --label local
```

The benchmark sends sequential keep-alive requests (200 measured after 20 warm-up) to
`/score` and `/decide` at batch sizes 1 and 100. It writes p50/p95/mean/max in milliseconds to
`reports/serving-benchmark/<label>.json`. It uses only the standard library, and `--token`
adds a bearer token for an authenticated Cloud Run service.

## Container

To measure the service rather than the network path, run the benchmark inside the container:
`docker exec <container> uplift-serve-benchmark --url http://127.0.0.1:8080 --output-dir /tmp/b`.
On Docker Desktop for Windows, POSTs from the host pay a fixed ~40 ms through the userspace
port forwarder (headers and body arrive as two segments and meet a delayed ACK). GETs do not.

```bash
docker build -t uplift-serve .
docker run --rm -p 8080:8080 uplift-serve
uplift-serve-benchmark --url http://127.0.0.1:8080 --label docker
```

The `Dockerfile` has two stages:

1. The build stage turns the package into a wheel.
2. The runtime stage installs `uplift_lab-*.whl[serve]` on `python:3.12-slim`, adds
   `libgomp1` (LightGBM needs the system OpenMP runtime) and runs as the non-root user
   `app` (uid 10001).

**What the image installs.** The package's core dependencies are only what loading and scoring
an exported model needs: numpy, pandas, LightGBM and platformdirs. The `serve` extra adds
FastAPI, uvicorn and pydantic. The analysis stack (pyarrow, scikit-learn, statsmodels; the
`analysis` extra) and matplotlib (`report`) are not installed. The serving import path never
imports them: the `uplift_lab.uplift` and `uplift_lab.experiments` packages re-export lazily,
and pyarrow and `scipy.stats` are imported inside the functions that use them. scipy itself
stays, because LightGBM requires it. `tests/test_serving_imports.py` serves `/health`, `/score`
and `/decide` with those modules blocked, so a new top-level import fails the tests.

Measured on Docker Desktop (Windows 11, WSL 2), `--cpus 1 --memory 512m`, same baked model (an
export from before the current v4 champion; not yet re-measured with it):

| | Before (all dependencies) | Now (core + `serve`) |
|---|---|---|
| Image size, uncompressed / compressed | 1.01 GB / 233 MB | **621 MB / 143 MB** |
| Dependency layer | 643 MB | 343 MB |
| Cold start: `docker run` to first 200 on `/health` (median of 5) | 2.12 s | **1.22 s** |
| `import uplift_lab.serving.app` inside the container | 1.53 s | 0.79 s |
| Memory after startup / after the benchmark | 155 / 160 MiB | **83 / 84 MiB** |
| In-container latency p50/p95, score batch 1 | 1.18 / 1.39 ms | 1.17 / 1.41 ms |
| In-container latency p50/p95, score batch 100 | 2.29 / 2.91 ms | 2.16 / 2.55 ms |

The cold-start medians were taken back to back with warm host caches. An earlier series of 5
starts of the old image, before its layers were in the page cache, had a median of 4.8 s, so a
cold host can roughly double these numbers. Scoring latency does not change: the removed packages were never on the
request path.

`.dockerignore` is an allow-list: `pyproject.toml`, `README.md`, `LICENSE`, `src/` and the
exported model only. Data, `mlruns/` and other reports never enter the build context.

**Model baking (the default).** The exported directory is copied into the image at
`/app/model`, owned by root and read-only for the service user. The image is then immutable
and self-contained: no registry credentials and no network at startup, and a cold start loads
the model in well under a second. A new model means a new image tag, and rolling back is
redeploying the previous tag. Use `--build-arg MODEL_DIR=<path inside the build context>` for
another export, such as a directory downloaded from the Databricks registry.

**Registry loading (optional).** MLflow is left out of the image on purpose, because the
full `mlflow` package is several hundred MB and is not needed to score. To resolve
`UPLIFT_LAB_MODEL_URI` at startup instead, build with
`--build-arg EXTRA_PACKAGES="mlflow-skinny==3.17.2"`. The service only calls
`mlflow.artifacts.download_artifacts` and then reads the pickle-free directory inside the
pyfunc model, so the logged Python code never runs. Two caveats: the container then needs
registry credentials (e.g. `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from Secret Manager), and a
local sqlite tracking store additionally needs `sqlalchemy`. Baking stays the recommended path
for Cloud Run.

## Cloud Run

These commands target project `filings-copilot` in `asia-southeast1`. They create billable
resources.

### Live deployment (2026-11-01, current champion)

The service runs the registered champion, Unity Catalog version 4. That is the full-data
S-learner from job run `369767185615309`, git SHA `4238f32`, chosen by the pre-declared champion
rule (see [databricks-run.md](databricks-run.md)).
- **Build:** `--build-arg MODEL_DIR=reports/databricks/369767185615309/model`, image tag
  `uplift-serve:0.1.0-s`, revision `uplift-serve-00002-97f`.
- **Check:** `/health` reports `s_learner`, package 0.1.0, SHA `4238f32`.
- **Access:** an anonymous call gets 403.

| Case | Client p50 / p95 | Server p50 / p95 |
|---|---|---|
| `/score` batch 1 | 74.5 / 83.5 ms | 9.0 / 16.1 ms (`/score`, both batch sizes) |
| `/decide` batch 1 | 73.2 / 79.5 ms | |
| `/score` batch 100 | 90.4 / 100.9 ms | |
| `/decide` batch 100 | 91.6 / 99.1 ms | |

Client latency is from Vietnam, 200 sequential requests per case. Server latency comes from
Cloud Run's request logs for 400 `/score` requests. The S-learner scores each row twice (treated
and control), which makes it slightly slower than the earlier single-model deployment below.

### Earlier deployment (2026-11-01, historical: previous model)

The commands below were run as written, except that the build baked the model trained on the
full Criteo data on Databricks:
`--build-arg MODEL_DIR=reports/databricks/286670365587179/model`. That is the transformed-outcome
learner (Unity Catalog version 2), git SHA `740f074`; see [databricks-run.md](databricks-run.md).

- **Service:** `uplift-serve`, revision `uplift-serve-00001-bbc`. Image `uplift-serve:0.1.0`.
- **Access:** an anonymous call gets 403; an identity token gets 200.
- **Latency:** the benchmark ran from Vietnam with 200 sequential requests per case.
  "Server" is Cloud Run's request latency, taken from the logs for 400 `/score` requests.

| Case | Client p50 / p95 | Server p50 / p95 |
|---|---|---|
| `/score` batch 1 | 72.2 / 78.0 ms | 6.9 / 9.6 ms (`/score`, both batch sizes) |
| `/decide` batch 1 | 71.6 / 76.1 ms | |
| `/score` batch 100 | 88.3 / 96.4 ms | |
| `/decide` batch 100 | 90.3 / 95.6 ms | |

Most of the client latency is the network round trip from Vietnam to Singapore.

- **Budget:** the project already has a monthly budget ("filings-copilot monthly", in the
  billing account's currency, about USD 5), so no second budget was added.

```bash
PROJECT=filings-copilot
REGION=asia-southeast1
REPO=uplift
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/uplift-serve:0.1.0

gcloud config set project $PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Image registry (one-off) and push.
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION \
  --description="uplift-lab scoring images"
gcloud auth configure-docker $REGION-docker.pkg.dev
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE

# Dedicated runtime identity without any project roles (the model is baked into the image).
gcloud iam service-accounts create uplift-serve --display-name="uplift-serve runtime"

gcloud run deploy uplift-serve \
  --image=$IMAGE \
  --region=$REGION \
  --service-account=uplift-serve@$PROJECT.iam.gserviceaccount.com \
  --no-allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --cpu=1 \
  --memory=512Mi \
  --concurrency=8 \
  --timeout=30s \
  --cpu-boost \
  --port=8080

# Call it (authenticated) and benchmark it.
URL=$(gcloud run services describe uplift-serve --region=$REGION --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" $URL/health
uplift-serve-benchmark --url $URL --token $TOKEN --label cloud-run --requests 100
```

Settings and why:

- **`--min-instances=0`**: scale to zero, so no traffic means no instance cost (the price is a
  cold start).
- **`--max-instances=1`**: a hard cap on spend and on abuse. Excess load queues and then gets
  429s instead of scaling out.
- **`--cpu=1 --memory=512Mi`**: scoring is single-threaded LightGBM (`OMP_NUM_THREADS=1`).
  The container used 84 MiB after a 1,600-request benchmark (`docker stats`), so 512 MiB
  leaves ample headroom. Raise it to 1 GiB if a larger model is baked in.
- **`--concurrency=8`**: requests are CPU-bound and run on the uvicorn thread pool. A batch of
  100 takes a few milliseconds, so 8 in flight keeps one vCPU busy without letting tail
  latency grow unbounded. The Cloud Run default of 80 would only queue inside the process.
- **CPU allocated only during requests** (the default, request-based billing). Nothing runs in
  the background.
- **`--no-allow-unauthenticated` (recommended).** This is a demo model on non-commercial data
  (Criteo, CC BY-NC-SA) with no product need for anonymous access. IAM authentication keeps
  bots and scanners from waking the instance and spending the budget. It also keeps the
  endpoint from becoming a free public scoring API built on the dataset. A public demo would
  instead need rate limiting in front (API Gateway or Cloud Armor), which costs more than the
  service itself. To share it, grant `roles/run.invoker` to specific accounts:
  `gcloud run services add-iam-policy-binding uplift-serve --region=$REGION --member=user:someone@example.com --role=roles/run.invoker`.

### Budget alert

`filings-copilot` may already have a budget. Check with `gcloud billing budgets list
--billing-account=$BILLING_ACCOUNT` before adding one.

```bash
gcloud services enable billingbudgets.googleapis.com
BILLING_ACCOUNT=$(gcloud billing projects describe $PROJECT --format='value(billingAccountName)' | sed 's|billingAccounts/||')
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')

gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT \
  --display-name="filings-copilot uplift-serve cap" \
  --budget-amount=5USD \
  --filter-projects=projects/$PROJECT_NUMBER \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend
```

A budget only alerts (e-mail to billing admins); it does not stop spending. The actual
spending limit is `--max-instances=1` plus authentication. To scope the budget to Cloud Run
alone, add `--filter-services=services/<Cloud Run service id>` after looking the ID up in the
Cloud Billing catalog.

### Cost envelope

Request-based billing charges vCPU-seconds and GiB-seconds only while requests are being
handled, plus requests. The monthly free tier (180,000 vCPU-s, 360,000 GiB-s, 2 million
requests at the time of writing) covers a demo: at about 5 ms per request, 2 million requests
use about 10,000 vCPU-s. The worst case under `--max-instances=1` is one instance busy all
month: about 2.6 million vCPU-s and 1.3 million GiB-s. That is tens of US dollars at list
prices. Check the current per-region prices; `asia-southeast1` is not a tier-1 region. Artifact
Registry storage for the image (143 MB compressed) fits in the 0.5 GB free tier.

### Roll back and remove

```bash
gcloud run services update-traffic uplift-serve --region=$REGION --to-revisions=<previous>=100
gcloud run services delete uplift-serve --region=$REGION
gcloud artifacts repositories delete $REPO --location=$REGION
```
