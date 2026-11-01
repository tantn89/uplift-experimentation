# Full Criteo run on Databricks Free Edition

The local report trains on a seeded 1M-row subsample. This run trains the same learners with the
same code on all 13,979,592 rows, as a serverless job on a Databricks Free Edition workspace, and
registers the best visit model in Unity Catalog.

- Job entry: [`databricks/train_criteo_full.py`](../databricks/train_criteo_full.py), a Python
  script task that calls `uplift_lab.pipelines.criteo_full_training.main`.
- Bundle: [`databricks.yml`](../databricks.yml) and
  [`databricks/resources/criteo_full_job.yml`](../databricks/resources/criteo_full_job.yml).
- The same module runs locally:
  `python -m uplift_lab.pipelines.criteo_full_training --source <criteo .csv.gz>`.

## What the job does

1. Checks the source on the Volume: the size, then the pinned SHA-256 (`2716e1bf...16dc`, the
   same Hugging Face revision as the local cache). A mismatch stops the job before parsing.
2. Parses all rows with pyarrow (float64 features, int8 labels) and checks the row count and
   that the labels are binary.
3. For `visit`, then `conversion`, runs `run_uplift`. It uses the same seeded stratified 70/30
   split as the local runs (seed 20261101) and fits the S-, T- and X-learners plus the
   transformed outcome on `f0`-`f11`. `exposure` is rejected as a feature. It then runs the
   paired bootstrap evaluation and the randomisation check.
4. Saves the best visit learner in the pickle-free artifact format (`manifest.json` + LightGBM
   text models). It logs the run to MLflow (tracking `databricks`), with the conversion metrics
   under `conversion/` and stage timings under `stage/`. It registers the model in Unity Catalog
   (`databricks-uc`) and sets the alias `champion`.
5. Copies `criteo-full.json` and `model/` to `/Volumes/<catalog>/<schema>/<volume>/runs/<run id>`.

The MLflow experiment is `/Users/<you>/uplift-criteo`. The bundle resolves `<you>` with
`${workspace.current_user.userName}` at deploy time and passes `${bundle.git.commit}` as the
trained commit, so no user name or SHA is committed.

## Reproduce

Needs the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) and the
project venv on `PATH`, because the bundle builds the wheel with `python -m pip wheel`.

```powershell
databricks auth login --host https://<workspace>.cloud.databricks.com --profile uplift

# One-off: schema and Volume (the bundle's artifact path lives in the Volume, so they come first).
databricks schemas create uplift_lab workspace -p uplift
databricks volumes create workspace uplift_lab raw MANAGED -p uplift

# One-off: the pinned source file from the local cache (uplift-criteo-report fetches it).
databricks fs mkdir dbfs:/Volumes/workspace/uplift_lab/raw/criteo -p uplift
databricks fs cp <cache>/criteo-research-uplift-v2.1.csv.gz `
  dbfs:/Volumes/workspace/uplift_lab/raw/criteo/criteo-research-uplift-v2.1.csv.gz -p uplift

# Build the wheel, upload it to the Volume and create or update the job, then run it.
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue   # the job installs dist/*.whl
databricks bundle deploy -p uplift
databricks bundle run criteo_full_uplift -p uplift

# Fetch the outputs and check that the model loads without MLflow or pickle.
databricks fs cp -r dbfs:/Volumes/workspace/uplift_lab/raw/runs/<run id> reports/databricks/<run id> -p uplift
python -c "from pathlib import Path; from uplift_lab.uplift import load_learner; print(load_learner(Path('reports/databricks/<run id>/model')).kind)"
```

`<cache>` is `$UPLIFT_LAB_CACHE`, or the platformdirs user cache (`%LOCALAPPDATA%\uplift-lab\Cache`
on Windows). Variables such as `n_bootstrap` can be overridden at deploy time, for example
`databricks bundle deploy -p uplift --var n_bootstrap=100`.

Clean up with `databricks bundle destroy -p uplift`, which removes the job and the bundle files.
Then delete the model, the Volume and the schema:

```powershell
databricks registered-models delete workspace.uplift_lab.uplift_criteo_visit -p uplift
databricks volumes delete workspace.uplift_lab.raw -p uplift
databricks schemas delete workspace.uplift_lab -p uplift
```

The experiment `/Users/<you>/uplift-criteo` is deleted from the workspace UI or with
`databricks experiments delete-experiment <id>`.

## Measured run (2026-11-01)

The run used job run `286670365587179` at commit `740f074`, on serverless environment version 4
(`client.4.10`, Python 3.12.3) with lightgbm 4.7.0 and mlflow 3.17.2. It processed all
13,979,592 rows: a stratified 70/30 split into 9,785,714 train and 4,193,878 test rows. Each
outcome used LightGBM with 200 trees, learning rate 0.05, 31 leaves and min_child_samples 100,
plus 200 paired bootstrap draws. Seed 20261101 throughout.

Reproducibility check: the frame SHA-256 (`b0ec020a...73be57`) is the same as in a local run of
the same module on the same file. Every one of the 4,346 numbers in the two evaluations agrees
to within a relative 2e-14, which is floating-point summation noise.

### Wall time per stage (seconds)

| Stage | Databricks serverless | Local laptop |
|---|---|---|
| Verify size + SHA-256 (311 MB on the Volume) | 1.9 | 0.3 |
| Parse CSV (gzip) | 8.7 | 4.0 |
| Frame checksum | 5.7 | 3.1 |
| visit: fit S / T / X / TO | 110 / 120 / 189 / 47 | 34 / 33 / 58 / 17 |
| visit: bootstrap evaluation (5 scorings x 200 draws) | 344 | 430 |
| visit: randomisation check | 95 | 42 |
| conversion: fit S / T / X / TO | 78 / 90 / 148 / 45 | 22 / 25 / 45 / 16 |
| conversion: bootstrap evaluation | 323 | 372 |
| conversion: randomisation check | 86 | 41 |
| MLflow logging + Unity Catalog registration | 67 | not run |
| **Script total** | **2,134 (35.6 min)** | **1,177 (19.6 min)** |
| Job run, including environment setup | 2,169 | |

The local laptop's peak working set was about 6.2 GB. The serverless job did not run out of
memory, and nothing had to be downsized: features stayed float64, with all four learners and
200 draws.

### Visit (ATE +0.01034 [+0.00980, +0.01088])

95% percentile bootstrap CIs. Qini and AUUC are per population user over the targeted
fraction, as in the local report.

| Model | Qini coefficient | AUUC | Uplift@10% | Uplift@30% |
|---|---|---|---|---|
| S-learner | +0.00314 [+0.00295, +0.00333] | 0.00882 [0.00836, 0.00927] | +0.0637 [0.0599, 0.0670] | +0.0287 [0.0271, 0.0304] |
| T-learner | +0.00321 [+0.00304, +0.00340] | 0.00891 [0.00848, 0.00939] | +0.0583 [0.0544, 0.0619] | +0.0298 [0.0282, 0.0315] |
| X-learner | +0.00313 [+0.00296, +0.00330] | 0.00882 [0.00839, 0.00928] | +0.0618 [0.0582, 0.0657] | +0.0287 [0.0273, 0.0304] |
| **Transformed outcome** | **+0.00325 [+0.00307, +0.00343]** | 0.00895 [0.00851, 0.00938] | +0.0625 [0.0590, 0.0666] | +0.0295 [0.0280, 0.0310] |
| Random scores | -0.00001 [-0.00014, +0.00013] | 0.00517 [0.00486, 0.00545] | +0.0109 | +0.0103 |

The paired Qini differences against the best (transformed outcome) are S -0.00012
[-0.00025, +0.00003], T -0.00004 [-0.00019, +0.00014] and X -0.00012 [-0.00023, +0.00004]. All
three cover 0, so no learner is a clear winner (`best_is_clear_winner` false). The registered
model is the highest point estimate.

Policy value: incremental visits per 1,000 contacts, from the within-segment difference in means.
Treat-all is 10.3 [9.8, 10.9]:

| Model | 10% budget | Gain vs random @10% | 30% budget | Gain vs random @30% |
|---|---|---|---|---|
| S-learner | 63.7 [59.9, 67.0] | +53.3 [50.1, 56.2] | 28.7 [27.1, 30.4] | +18.4 [17.2, 19.6] |
| T-learner | 58.3 [54.4, 61.9] | +47.9 [44.4, 51.4] | 29.8 [28.2, 31.5] | +19.5 [18.3, 20.7] |
| X-learner | 61.8 [58.2, 65.7] | +51.4 [48.1, 54.7] | 28.7 [27.3, 30.4] | +18.4 [17.4, 19.7] |
| Transformed outcome | 62.5 [59.0, 66.6] | +52.1 [49.1, 55.8] | 29.5 [28.0, 31.0] | +19.2 [18.0, 20.3] |
| Random scores | 10.9 [9.1, 12.3] | +0.5 [-1.0, 1.9] | 10.3 [9.2, 11.2] | -0.1 [-0.9, 0.7] |

Randomisation check: the held-out AUC of treatment ~ features is 0.5105 [0.5098, 0.5112]. At
4.2M test rows that is clearly above 0.5: assignment is slightly predictable, as on the
subsample (0.508). This is the reason the constant-propensity HT-IPW policy column (S @10%:
93.8) stays secondary.

### Conversion (ATE +0.00115 [+0.00104, +0.00127])

Too sparse on the 1M subsample, but measurable on the full data:

| Model | Qini coefficient | Per 1,000 contacts @10% | Gain vs random @10% | Per 1,000 contacts @30% |
|---|---|---|---|---|
| **S-learner** | **+0.00040 [+0.00035, +0.00045]** | 8.1 [7.0, 9.3] | +7.0 [6.0, 8.1] | 3.5 [3.2, 3.9] |
| T-learner | +0.00025 [+0.00020, +0.00031] | 6.8 [5.8, 7.8] | +5.6 [4.7, 6.6] | 2.9 [2.6, 3.3] |
| X-learner | +0.00034 [+0.00028, +0.00039] | 8.0 [6.9, 9.3] | +6.9 [5.9, 8.0] | 3.2 [2.9, 3.6] |
| Transformed outcome | +0.00036 [+0.00030, +0.00041] | 8.2 [7.1, 9.4] | +7.0 [6.0, 8.1] | 3.3 [3.0, 3.7] |
| Random scores | +0.00000 [-0.00003, +0.00003] | 1.3 [0.9, 1.6] | +0.1 [-0.3, 0.5] | 1.1 [0.8, 1.3] |

Treat-all: 1.15 [1.04, 1.27] incremental conversions per 1,000 contacts. The S-learner is a
clear winner on conversion: every paired difference interval lies below 0, with the transformed
outcome closest at -0.00004 [-0.00007, -0.00000]. The conversion model is evaluated and logged
(`conversion/` metrics on the MLflow run) but not registered.

### Compared with the local 1M subsample

| | 1M subsample (local) | Full data (Databricks) |
|---|---|---|
| Best visit learner (point estimate) | S-learner | Transformed outcome |
| Best visit Qini | +0.00335 [+0.00273, +0.00404] | +0.00325 [+0.00307, +0.00343] |
| S-learner visit Qini | +0.00335 | +0.00314 [+0.00295, +0.00333] |
| S-learner per 1,000 contacts @10% | 65.1 [51.1, 81.9] | 63.7 [59.9, 67.0] |
| Clear winner on visit | no | no |

The headline holds: top-10% targeting yields about six times the incremental visits per contact
of treat-all. The full data shrinks the intervals about 3-4x, but the learners stay within noise
of each other on visit, so the "winner" changed from S to the transformed outcome by a
non-significant margin.

### Registry

The model is `workspace.uplift_lab.uplift_criteo_visit` version 2, alias `champion`, from MLflow
run `8f6eb8549a03402abff84ed8bfb03f9a`. It is a transformed-outcome learner whose manifest records
the git SHA, dataset checksums and budget thresholds. Version 1 was a 50,000-row smoke
run that tested the environment and the Unity Catalog path before the full run. It was deleted
afterwards, together with its Volume outputs, so version 2 is the only version left.

## Limits hit

- No memory or quota limit was hit with the default settings.
- Serverless fits are 2.5-3.5x slower than on the laptop, but the bootstrap is a little faster.
  Each learner's fit is single-node LightGBM, not Spark.
- Free Edition has no DBFS library root and no classic clusters. That is why the wheel goes to
  the Volume (`workspace.artifact_path`) and the job uses a serverless environment.
- `mode: development` requires the artifact path to contain the user's short name, and prefixes
  the job name with `[dev <user>]`.

## Cost

Free. Free Edition has no billing; serverless compute is subject to a daily fair-use quota.
Nothing else was created: no classic cluster, no SQL warehouse use, no Model Serving endpoint
and no sharing.
