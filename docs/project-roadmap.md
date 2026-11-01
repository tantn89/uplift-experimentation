# Roadmap

Direction, not a commitment. Items are unstarted unless marked. Shipped behaviour is described by
the code and by [project-overview-pdr.md](project-overview-pdr.md).

## v0.1 (current: `0.1.0`)

Done:

- Hillstrom A/B analysis: Welch effects, CUPED with a regression-adjustment check, SRM and power.
  A seeded strong-covariate CUPED contrast sits next to it.
- A peeking simulation (naive t-test, mSPRT, fixed horizon) and the price of peeking.
- In-house S/T/X learners and a transformed-outcome baseline. Evaluation covers Qini, AUUC, uplift@k
  and policy value, with paired bootstrap intervals, a no-winner rule and a randomisation check.
- A pickle-free model artifact, local MLflow tracking and a registry alias.
- A full-data Databricks serverless job (13,979,592 rows), registered in Unity Catalog. It matches
  the local run to a relative 2e-14.
- A FastAPI service with `/score` and `/decide`, and a slim non-root image. It is live on Cloud Run
  behind IAM, with one instance at most.
- Drift monitoring with PSI, KS and the decision rate; a CI drift gate.
- A CI metric gate against a committed baseline.

Open at v0.1:

- Both drift alert margins on Criteo are thin: f0 and f6 PSI just over 0.2, and a 1.53x decision
  rate against the 1.5x bound.
- The served visit model is the highest point estimate among learners that are within noise of
  each other.

## Next

| Item | Why | Notes |
|---|---|---|
| AIPW (doubly robust) policy value | Treatment is slightly predictable (AUC 0.51), so constant-propensity IPW is biased | Use the fitted propensity from `uplift/propensity.py`; compare with the within-segment estimate |
| Scheduled drift job on served traffic | The drift report runs on demand today | Read the sampled traffic log on a schedule; alert on the decision rate |
| Finer bins for point-mass features | Decile PSI missed the label shift on Criteo | Or a categorical PSI for heavy point masses |
| mSPRT tau sensitivity | Power depends on the mixing sd | Sweep tau in the sequential report |
| Expose scored segments to agents | Lets an agent answer "whom should we contact" over governed SQL | Through [warehouse-mcp](https://github.com/tantn89/warehouse-mcp) |
| Conversion model in the registry | Conversion has a clear winner (S-learner) on the full data, but only visit is registered | Register it under a second name |

## Explicitly out of scope

An experimentation platform UI, multi-armed bandits, real customer data and Kubernetes. See the
non-goals in [project-overview-pdr.md](project-overview-pdr.md).
