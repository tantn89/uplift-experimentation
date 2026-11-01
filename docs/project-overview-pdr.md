# Project overview and requirements

`uplift-lab` (repository `uplift-experimentation`) is a Python package and a set of report scripts
for two growth-team questions, answered on public randomised data:

1. Did a campaign work, and how sure are we? (A/B analysis on the Hillstrom e-mail RCT)
2. Whom should we target next time? (uplift modelling on the Criteo Uplift RCT)

The model is trained on the full Criteo data on Databricks, served on Cloud Run and monitored for
drift. Shipped behaviour is described by the code and tests; this page records the requirements
and where each claim is proven.

## Problem

| Failure | What goes wrong in practice |
|---|---|
| Peeking | Checking a fixed-horizon t-test every day turns a 5% false-positive rate into about 25% |
| Overselling CUPED | Variance reduction is promised without checking that the pre-period metric predicts the outcome |
| Response models used as uplift models | Ranking by predicted conversion targets people who would convert anyway |
| A "best model" without uncertainty | Learners within noise of each other get ranked, and the ranking flips on new data |
| Post-treatment leakage | `exposure` is recorded after assignment; using it as a feature makes the model look better and the decisions worse |
| Silent model or traffic drift | A model change, or a label shift that leaves the features alone, quietly overspends the budget |

## Goals

- Every effect has a confidence interval, and every reported statistic comes from a seeded script
  or job. Operational measurements (latency, image size, cold start, memory) are labelled as
  such.
- The sequential test holds its stated type-I error under continuous peeking, and the cost of that
  guarantee is measured.
- Uplift models are judged by Qini and policy value under a budget, against "treat everyone" and
  random targeting, with bootstrap intervals and paired comparisons.
- The same code trains locally on a 1M subsample and on Databricks on all 13,979,592 rows.
- The served model cannot see post-treatment fields, scores in milliseconds and is not public.
- Drift and quality regressions fail CI.

## Non-goals

| Non-goal | Reason |
|---|---|
| An experimentation platform UI | The surface is report scripts, an API and JSON outputs |
| Multi-armed bandits | Out of scope for v0.1 (see [project-roadmap.md](project-roadmap.md)) |
| Real customer data | Public RCTs only; Criteo is non-commercial and never redistributed |
| Kubernetes | Cloud Run with one instance is enough for a demo endpoint |
| Public anonymous scoring | The endpoint is IAM-authenticated by design |

## Requirements

| # | Requirement | Acceptance criteria | Evidence |
|---|---|---|---|
| R1 | Valid two-arm inference | Welch CI coverage is nominal with unequal variances; matches scipy | `experiments/estimators.py`; `test_ci_coverage_is_nominal_with_unequal_variances`, `test_matches_scipy_welch_test` |
| R2 | CUPED is unbiased and reduces variance by R² | Mean over 1,000 seeded runs equals the true effect; Var ratio is 1 − ρ²; equals ANCOVA | `experiments/cuped.py`; `test_unbiased_and_lower_variance_in_simulation`, `test_matches_regression_adjustment`, `test_cuped_variance_ratio_follows_one_minus_rho_squared` |
| R3 | SRM and power | Imbalance flagged at p < 0.001; power and sample size match statsmodels | `experiments/srm.py`, `power.py`; `tests/test_srm_and_power.py` |
| R4 | Always-valid sequential test | mSPRT type-I ≤ alpha under the null in simulation, while naive peeking exceeds 3x alpha; the likelihood ratio equals numerical integration | `experiments/msprt.py`, `peeking_simulation.py`; `test_msprt_controls_type_one_error_while_naive_peeking_inflates_it`, `test_log_lr_equals_numerical_mixture` |
| R5 | The price of peeking is measured | The mSPRT cap for 80% power, its expected n under the alternative and the null, and its type-I error at that cap | `experiments/peeking_price.py`; `tests/test_cuped_and_peeking_contrasts.py` |
| R6 | Uplift learners beat random with a CI that excludes 0 | Qini intervals of S/T/X/transformed outcome exclude 0; random's covers it; one-arm resamples are dropped, not fatal | `uplift/evaluation.py`; `tests/test_uplift_metrics.py`, `tests/test_uplift_policy.py`; [databricks-run.md](databricks-run.md) |
| R7 | No winner without evidence; a pre-declared champion | `best_is_clear_winner` only if every paired Qini difference is significant after Holm correction; otherwise the champion is the simplest tied learner (S > T > X > transformed outcome) | `uplift/evaluation.py`, `uplift/champion.py`; `tests/test_uplift_policy.py` |
| R8 | No post-treatment leakage | `exposure`, `treatment`, `visit`, `conversion` rejected in fit, artifact load and the API | `uplift/features.py`, `serving/schemas.py`; `test_bad_rows_are_rejected`, `tests/test_uplift_learners.py` |
| R9 | Pickle-free, reproducible artifact | The manifest is validated (format, version, kind, file names, features); a reload predicts identically; the registered pyfunc carries the package source and its pinned requirements | `uplift/artifacts.py`, `uplift/tracking.py`; `tests/test_uplift_artifacts.py`, `tests/test_uplift_tracking.py` |
| R10 | API contract | `/score` matches offline predict; `/decide` matches offline thresholds; features are strict JSON numbers; batch cap 1,000; errors never echo inputs | `serving/`; `tests/test_serving_api.py` |
| R11 | Slim serving path | `/health`, `/score` and `/decide` work with scikit-learn, statsmodels, pyarrow, matplotlib and `scipy.stats` blocked | `tests/test_serving_imports.py` |
| R12 | Drift is detected | Seeded no-shift is ok and covariate shift alerts (CI gate); the decision rate flags a label shift that PSI misses (a warning at 1.46x with the current champion) | `monitoring/drift.py`; `tests/test_drift.py`, `.github/workflows/ci.yml` (`drift-report`) |
| R13 | Quality regressions fail CI | A paired head-minus-baseline interval below 0 with a drop over 2%, or a Qini interval covering 0, fails; an incomparable baseline (data, settings or library versions) exits 2 | `gate/`; `tests/test_metric_gate.py`, `.github/workflows/ci.yml` (`metric-gate`) |
| R14 | Same code locally and on Databricks; reruns reproduce | The current run's point estimates equal the earlier Databricks run exactly; that earlier run agreed with a local full-data run to a relative 2e-14 (a one-off local comparison whose outputs are not in the repository) | `pipelines/criteo_full_training.py`; [databricks-run.md](databricks-run.md) |
| R15 | Live, capped, authenticated endpoint | Anonymous call 403, token 200; `max-instances=1`; budget alert | [deployment-guide.md](deployment-guide.md) (live deployment section; serving the v4 champion) |

## Evidence

Where each README number comes from. Local report outputs go to `reports/` (git-ignored) and are
reproduced by the command shown. All statistical results are seeded or deterministic. The operational measurements (latency, image size, cold start, memory) and the library spike are one-off measurements and are labelled as such.

| README number | Command | Output field |
|---|---|---|
| Hillstrom effects, relative lift, CUPED variance reduction, SRM p 0.904 | `uplift-hillstrom-report` | `reports/hillstrom/hillstrom.json`: `effects.<arm>.<outcome>.raw`, `.cuped.pre_period.variance_reduction`, `srm.p_value` |
| 13,217 vs 12,873 users per arm for a 10% visit lift | `uplift-hillstrom-report` | `power.visit.required_n_per_arm` |
| Simulated CUPED variance ratios and coverage (ρ 0 to 0.9) | `uplift-hillstrom-report` | `cuped_simulated_contrast.by_rho[]` |
| Peeking: 0.248 / 0.0098 / 0.0499 false positives, power 0.874 / 0.465 / 0.787, median stop 3,000 / 6,500 | `uplift-sequential-report` | `reports/sequential/sequential.json`: `aa.<rule>.rejection_rate`, `ab.<rule>` |
| Naive false positives 0.377 with 100 looks | `uplift-sequential-report --large` | `aa.naive_peeking.rejection_rate` |
| Price of peeking: cap 18,000 (1.8x), mean n 11,190 (1.12x) under the alternative and 17,848 (1.78x) under the null, type-I 0.016 | `uplift-sequential-report` | `price_of_peeking` (`msprt_mean_n_per_arm_under_cap`, `msprt_mean_n_per_arm_under_null`) |
| Criteo full data: Qini, per 1,000 contacts, paired differences and Holm p-values, champion (S-learner, UC v4), conversion, AUC 0.5105, IPW 93.8, run time 3,756 s | Databricks job run `369767185615309` (`databricks bundle run criteo_full_uplift`) | `reports/databricks/369767185615309/criteo-full.json` (downloaded from the Volume): `outcomes.<outcome>.evaluation`, `.champion`, `timings`; summarised in [databricks-run.md](databricks-run.md) |
| Full-data point estimates equal to the earlier run | Compare `criteo-full.json` of runs `369767185615309` and `286670365587179` | `outcomes.<outcome>.evaluation.models` |
| Agreement with a local full-data run to a relative 2e-14; 1,177 s locally | One-off local run of `python -m uplift_lab.pipelines.criteo_full_training`, compared with the earlier Databricks run | Not in the repository: local outputs are git-ignored and were not kept; recorded in [databricks-run.md](databricks-run.md) |
| Criteo 1M subsample: S Qini +0.00335, 65.1 per 1,000 contacts, champion S-learner | `uplift-criteo-report` | `reports/criteo-uplift/criteo-uplift.json`: `evaluation.models.s_learner`, `evaluation.champion` |
| Drift table (statuses, PSI, KS, treat rate), scored by the champion | Download run `369767185615309` (see databricks-run.md), then `uplift-drift-report --model-dir reports/databricks/369767185615309/model` | `reports/drift/drift.json`: `scenarios.<name>` (`ks_tests` 13) |
| Latency (Cloud Run and in-container), image size, cold start, memory | `uplift-serve-benchmark`, `docker image ls`, `docker stats` | Measurements, not seeded, taken with the previous model: [deployment-guide.md](deployment-guide.md) |
| Metric-gate baseline Qini values; probes the paired rule now fails (learning rate 0.2, 63 leaves) | `uplift-metric-gate --update-baseline`; the probes pass `LgbmParams` to `gate_metrics` and `compare` with the baseline | [`evals/metric-gate-baseline.json`](../evals/metric-gate-baseline.json) |
| CUPED vs regression adjustment: within 1e-5 (visit, conversion) and 1e-4 (spend) | `uplift-hillstrom-report` | `effects.<arm>.<outcome>.cuped.pre_period.adjusted.effect` vs `.regression_adjustment.pre_period.effect` |
| Library spike: causalml, scikit-uplift, Evidently findings; parity to four decimals | One-off spike on 2026-11-01 (Python 3.12 and 3.14) | Recorded here and in [system-architecture.md](system-architecture.md); not a script in the repository |

The library spike and the local 2e-14 / 1,177 s comparison are the claims without a script output in the repository. The spike describes those libraries' releases at the
time of the spike and can change with new releases.
