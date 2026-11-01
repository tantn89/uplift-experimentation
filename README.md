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

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests && mypy && pytest
```

## Licence

Code: MIT. Data: see the table above.
