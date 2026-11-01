# Full Criteo run on Databricks Free Edition

The local report trains on a seeded 1M-row subsample. This run trains the same learners with the
same code on all 13,979,592 rows, as a serverless job on a Databricks Free Edition workspace, and
registers the visit champion in Unity Catalog.

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
   paired bootstrap evaluation (Holm-corrected paired Qini tests) and the randomisation check.
4. Picks the visit champion with the pre-declared rule in `uplift/champion.py`: the best learner
   by Qini if it is a clear winner, otherwise the first learner not significantly worse than it
   in the order S > T > X > transformed outcome. It saves that learner in the pickle-free
   artifact format (`manifest.json` + LightGBM text models), with the rule and the reason in the
   manifest. It logs the run to MLflow (tracking `databricks`) in one step before the model: a
   curated set of about 630 metrics, with the conversion metrics under `conversion/` and stage
   timings under `stage/`. MLflow 3 links the run's metrics to the LoggedModel, and Databricks
   caps those at 1,000; the full evaluation is in the `report/criteo-full.json` artifact. It
   registers the model in Unity Catalog (`databricks-uc`), with the package source as
   `code_paths`, and sets the alias `champion`.
5. Copies `criteo-full.json` and `model/` to `/Volumes/<catalog>/<schema>/<volume>/runs/<run id>`.

The MLflow experiment is `/Users/<you>/uplift-criteo`. The bundle resolves `<you>` with
`${workspace.current_user.userName}` at deploy time and passes `${bundle.git.commit}` as the
trained commit, so no user name or SHA is committed. The wheel is built by
[`databricks/build_wheel.py`](../databricks/build_wheel.py): it empties `dist/` first, so the
job's `dist/*.whl` glob cannot pick a stale wheel, and it records the commit and whether tracked
files had uncommitted changes. The job writes that `git.dirty` flag into its report, the MLflow
tags and the manifest. The training task has `max_retries: 0`, because the job is deterministic
and a retry would repeat a failure.

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

# Build the wheel (databricks/build_wheel.py empties dist/ first), upload it to the Volume and
# create or update the job, then run it. Deploy from a committed tree, or the run records dirty.
databricks bundle deploy -p uplift
databricks bundle run criteo_full_uplift -p uplift

# Fetch the outputs and check that the model loads without MLflow or pickle.
databricks fs cp -r dbfs:/Volumes/workspace/uplift_lab/raw/runs/<run id> reports/databricks/<run id> -p uplift
python -c "from pathlib import Path; from uplift_lab.uplift import load_learner; print(load_learner(Path('reports/databricks/<run id>/model')).kind)"
```

`<cache>` is `$UPLIFT_LAB_CACHE`, or the platformdirs user cache (`%LOCALAPPDATA%\uplift-lab\Cache`
on Windows). Variables such as `n_bootstrap` (default 1,000) can be overridden at deploy time,
for example `databricks bundle deploy -p uplift --var n_bootstrap=200`.

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

The published run is job run `369767185615309` at commit `4238f32` (clean tree, `git.dirty`
false from the wheel's build info), on serverless environment version 4 (`client.4.10`, Python
3.12.3) with lightgbm 4.7.0 and mlflow 3.17.2. It processed all 13,979,592 rows: a stratified
70/30 split into 9,785,714 train and 4,193,878 test rows. Each outcome used LightGBM with 200
trees, learning rate 0.05, 31 leaves and min_child_samples 100, plus 1,000 paired bootstrap draws
(up from 200 in the earlier run). Seed 20261101 throughout.

Reproducibility checks:
- The frame SHA-256 (`b0ec020a...73be57`) and every point estimate (175 per outcome) are exactly
  equal to the earlier 200-draw run `286670365587179` at commit `740f074`. Only the intervals
  changed, because they now use 1,000 draws.
- That earlier run was also compared with a local run of the same module on the same file: all
  4,346 numbers of the two evaluations agreed to within a relative 2e-14 (floating-point
  summation noise), and the local script took 1,177 s. The local run's outputs are not in the
  repository, so those two figures rest on that one-off comparison.

### Wall time per stage (seconds)

| Stage | Databricks serverless, 1,000 draws |
|---|---|
| Verify size + SHA-256 (311 MB on the Volume) | 1.6 |
| Parse CSV (gzip) | 8.8 |
| Frame checksum | 6.0 |
| visit: fit S / T / X / TO | 110 / 119 / 189 / 46 |
| visit: bootstrap evaluation (5 scorings x 1,000 draws) | 1,096 |
| visit: randomisation check (1,000 draws) | 165 |
| conversion: fit S / T / X / TO | 75 / 89 / 146 / 39 |
| conversion: bootstrap evaluation | 1,038 |
| conversion: randomisation check | 183 |
| MLflow logging + Unity Catalog registration | 79 |
| **Script total** | **3,756 (62.6 min)** |
| Job run, including environment setup | 3,798 |

The earlier 200-draw run took 2,134 s; its bootstrap evaluation took 344 s per outcome. Five
times the draws cost about 3.2x the evaluation time, because each replicate now validates and
reorders the arms once per ranking instead of once per draw. The serverless job did not run out
of memory, and nothing had to be downsized: features stayed float64, with all four learners.

A first attempt of this run (job run `453029091906461`) failed after 68 minutes, at MLflow
logging. MLflow 3 links a run's metrics to its LoggedModel, Databricks caps those at 1,000, and
that version of the job logged about 600 visit metrics, then the model, then another 600. It had
already registered version 3. Version 3 was deleted, the job now logs one curated set before the
model (commit `4238f32`), and the training task no longer retries.

### Visit (ATE +0.01034 [+0.00984, +0.01086])

95% percentile bootstrap CIs over the 4.2M test rows, with the fitted models held fixed: they
measure test-set sampling noise, not the variance of retraining. Qini and AUUC are per
population user over the targeted fraction, as in the local report.

| Model | Qini coefficient | AUUC | Uplift@10% | Uplift@30% |
|---|---|---|---|---|
| **S-learner (champion)** | +0.00314 [+0.00296, +0.00332] | 0.00882 [0.00840, 0.00925] | +0.0637 [0.0600, 0.0672] | +0.0287 [0.0271, 0.0303] |
| T-learner | +0.00321 [+0.00302, +0.00339] | 0.00891 [0.00849, 0.00930] | +0.0583 [0.0546, 0.0619] | +0.0298 [0.0283, 0.0313] |
| X-learner | +0.00313 [+0.00296, +0.00331] | 0.00882 [0.00839, 0.00925] | +0.0618 [0.0581, 0.0655] | +0.0287 [0.0272, 0.0303] |
| Transformed outcome (best by Qini) | +0.00325 [+0.00307, +0.00343] | 0.00895 [0.00854, 0.00936] | +0.0625 [0.0590, 0.0662] | +0.0295 [0.0281, 0.0310] |
| Random scores | -0.00001 [-0.00013, +0.00013] | 0.00517 [0.00487, 0.00546] | +0.0109 | +0.0103 |

The paired Qini differences against the best by Qini (transformed outcome), with two-sided
bootstrap p-values and their Holm adjustment across the three learners:

| Learner | Qini minus best [95% CI] | p | Holm p |
|---|---|---|---|
| S-learner | -0.00012 [-0.00026, +0.00003] | 0.118 | 0.354 |
| T-learner | -0.00004 [-0.00020, +0.00013] | 0.651 | 0.651 |
| X-learner | -0.00012 [-0.00026, +0.00003] | 0.122 | 0.354 |

None is significant, so there is no clear winner (`best_is_clear_winner` false), and all three
are tied with the best. The champion rule (`champion_rule`
`highest_qini_if_clear_winner_else_simplest`) then takes the first tied learner in the order
S > T > X > transformed outcome: the **S-learner**. It never uses the propensity, whereas the
transformed outcome weights every row by one constant propensity that the randomisation check
below shows is slightly wrong. The rule and the reason are in the manifest, the JSON report and
the MLflow tags.

Policy value: incremental visits per 1,000 contacts, from the within-segment difference in means
(less biased than constant-propensity IPW; not unbiased if the treated share varies inside a
segment). Treat-all is 10.3 [9.8, 10.9]:

| Model | 10% budget | Gain vs random @10% | 30% budget | Gain vs random @30% |
|---|---|---|---|---|
| **S-learner (champion)** | 63.7 [60.0, 67.2] | +53.3 [49.9, 56.6] | 28.7 [27.1, 30.3] | +18.4 [17.2, 19.5] |
| T-learner | 58.3 [54.6, 61.9] | +47.9 [44.6, 51.3] | 29.8 [28.3, 31.3] | +19.5 [18.4, 20.6] |
| X-learner | 61.8 [58.1, 65.5] | +51.4 [48.1, 54.7] | 28.7 [27.2, 30.3] | +18.4 [17.3, 19.6] |
| Transformed outcome | 62.5 [59.0, 66.2] | +52.1 [48.9, 55.5] | 29.5 [28.1, 31.0] | +19.2 [18.1, 20.3] |
| Random scores | 10.9 [9.2, 12.5] | +0.5 [-1.0, 2.1] | 10.3 [9.4, 11.2] | -0.1 [-0.9, 0.7] |

Randomisation check: the held-out AUC of treatment ~ features is 0.5105 [0.5098, 0.5113]. At
4.2M test rows that is clearly above 0.5: assignment is slightly predictable, as on the
subsample (0.509). This is why the constant-propensity HT-IPW policy column (S @10%: 93.8) stays
secondary.

### Conversion (ATE +0.00115 [+0.00103, +0.00127])

Too sparse on the 1M subsample, but measurable on the full data:

| Model | Qini coefficient | Per 1,000 contacts @10% | Gain vs random @10% | Per 1,000 contacts @30% |
|---|---|---|---|---|
| **S-learner** | **+0.00040 [+0.00035, +0.00045]** | 8.1 [6.9, 9.4] | +7.0 [5.9, 8.1] | 3.5 [3.2, 3.9] |
| T-learner | +0.00025 [+0.00020, +0.00030] | 6.8 [5.7, 7.8] | +5.6 [4.6, 6.6] | 2.9 [2.6, 3.3] |
| X-learner | +0.00034 [+0.00028, +0.00039] | 8.0 [6.9, 9.2] | +6.9 [5.8, 7.9] | 3.2 [2.8, 3.6] |
| Transformed outcome | +0.00036 [+0.00031, +0.00041] | 8.2 [7.0, 9.3] | +7.0 [6.0, 8.1] | 3.3 [2.9, 3.7] |
| Random scores | +0.00000 [-0.00003, +0.00003] | 1.3 [0.9, 1.6] | +0.1 [-0.2, 0.5] | 1.1 [0.8, 1.3] |

Treat-all: 1.15 [1.03, 1.27] incremental conversions per 1,000 contacts. The S-learner is a
clear winner on conversion, also after Holm correction:

| Learner | Qini minus S-learner [95% CI] | p | Holm p |
|---|---|---|---|
| T-learner | -0.00015 [-0.00020, -0.00010] | 0.002 | 0.006 |
| X-learner | -0.00006 [-0.00010, -0.00003] | 0.002 | 0.006 |
| Transformed outcome | -0.00004 [-0.00008, -0.00000] | 0.022 | 0.022 |

The transformed outcome is the narrowest margin: its upper bound sits at 0 to five decimals, so
that comparison is significant but not by much. With 1,000 draws the smallest attainable p-value
is 2/1,001. The conversion model is evaluated and logged (`conversion/` metrics on the MLflow
run) but not registered.

### Compared with the local 1M subsample

| | 1M subsample (local, 200 draws) | Full data (Databricks, 1,000 draws) |
|---|---|---|
| Best visit learner by Qini | S-learner | Transformed outcome |
| Best visit Qini | +0.00335 [+0.00273, +0.00404] | +0.00325 [+0.00307, +0.00343] |
| S-learner visit Qini | +0.00335 | +0.00314 [+0.00296, +0.00332] |
| S-learner per 1,000 contacts @10% | 65.1 [51.1, 81.9] | 63.7 [60.0, 67.2] |
| Clear winner on visit (Holm) | no | no |
| Champion under the rule | S-learner | S-learner |

The headline holds: top-10% targeting yields about six times the incremental visits per contact
of treat-all. The full data shrinks the intervals about 3-4x, but the learners stay within noise
of each other on visit. The best point estimate changed from S to the transformed outcome by a
non-significant margin; the pre-declared rule picks the S-learner on both.

### Registry

The champion is `workspace.uplift_lab.uplift_criteo_visit` **version 4**, alias `champion`, from
MLflow run `cc0f13b550284e1eb5014a9bbccd0eaa`. It is the S-learner; its manifest records the git
SHA and dirty flag, the dataset checksums, the budget thresholds and the champion rule and
reason. The pyfunc carries the `uplift_lab` source (`code_paths`) and pinned requirements. It was
loaded from Unity Catalog in a fresh virtual environment with only those requirements (plus
`pip-system-certs`, for a local TLS-inspecting proxy) and no `uplift_lab` installed: MLflow
imported the package from the model's `code/` directory, and the score of a sample row matched
`load_learner` on the downloaded model directory exactly.

Other versions: version 1 was a 50,000-row smoke run, deleted after the first full run. Version 2
is the earlier full run (transformed outcome, run `8f6eb8549a03402abff84ed8bfb03f9a`), kept but
no longer aliased. Version 3 was the incomplete output of the failed first attempt and was
deleted.

## Limits hit

- No memory or serverless quota limit was hit, with 1,000 draws for both outcomes (62.6 min).
- The MLflow limit of 1,000 metrics per LoggedModel was hit once (see above) and is now guarded
  by a check in `log_uplift_run` and a test that counts the job's metrics.
- Serverless fits are 2.5-3.5x slower than on a laptop (measured on the earlier run). Each
  learner's fit is single-node LightGBM, not Spark.
- Free Edition has no DBFS library root and no classic clusters. That is why the wheel goes to
  the Volume (`workspace.artifact_path`) and the job uses a serverless environment.
- `mode: development` requires the artifact path to contain the user's short name, and prefixes
  the job name with `[dev <user>]`.

## Cost

Free. Free Edition has no billing; serverless compute is subject to a daily fair-use quota.
Nothing else was created: no classic cluster, no SQL warehouse use, no Model Serving endpoint
and no sharing.
