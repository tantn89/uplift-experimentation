# Code standards

Conventions the repository actually follows. The enforcing configuration is in `pyproject.toml`
and `.github/workflows/ci.yml`; this page records the reasons.

## Tooling

| Check | Command | Configuration |
|---|---|---|
| Lint and import order | `ruff check src tests` | Line length 100, target py311, rules `E F I B UP SIM RUF` |
| Formatting | `ruff format --check src tests` | ruff defaults |
| Types | `mypy` | `strict = true` over `src` and `tests`; optional libraries without stubs are listed under `ignore_missing_imports` |
| Tests | `pytest` | `-m 'not network'` by default; `pytest -m network` downloads the real datasets |

CI runs lint, format and types on Python 3.11 and the tests on 3.11 and 3.13. Run all of them
before every commit.

## Statistics

- **Every number is reproducible.** Simulations and bootstraps take an explicit seed, and reports
  record it in their JSON. A new reported statistic needs a script that produces it and a test or
  seeded simulation that backs it.
- **Test statistical claims against known results**, not against snapshots: nominal coverage,
  unbiasedness over many seeded replicates, parity with scipy or statsmodels, closed forms. State
  the tolerance's reasoning in a comment (the standard error of the simulated quantity).
- **Intervals before rankings.** Comparisons between models use paired bootstrap differences. A
  winner is named only when the interval excludes 0.
- **Pre-treatment features only.** New features go through `uplift/features.py`. Post-treatment
  fields (`exposure`) and labels are rejected, never silently dropped.
- **Report honestly.** If a method does not help on the data (CUPED on Hillstrom), the report and
  the README say so, with the measured number.

## Code

- **Validate at the boundary.** Configs are frozen dataclasses with `__post_init__` checks. API
  input is pydantic with `extra="forbid"` and finite numbers only. Artifact manifests are validated
  on load. Error messages name the bad value's field, not the value, at the API.
- **Optional dependencies stay optional.** Heavy libraries (scikit-learn, statsmodels, pyarrow,
  matplotlib, MLflow) are imported lazily or inside functions. `tests/test_serving_imports.py` fails
  if serving starts importing them.
- **No pickle.** Models persist as LightGBM text plus a JSON manifest.
- **Strict JSON.** Reports write through `reports/output.py`, which turns NaN and infinity into
  `null` and numpy types into Python ones.
- **Files.** `snake_case` modules named for what they own, kept to about 200 lines. Split only at
  a real boundary (analysis, figures, rendering, CLI).
- **Comments explain why.** Module docstrings state the design and its trade-off. No ticket, plan
  or phase labels in code, tests or commit messages.

## Commits

Conventional commits (`feat`, `fix`, `test`, `docs`, `ci`, `build`, `refactor`), imperative
subject, optional scope (for example `feat(uplift): ...`). One concern per commit; the body
explains the why. A change to the metric-gate data or evaluation settings refreshes
`evals/metric-gate-baseline.json` in the same commit.

## Documentation

Docs own why and where; code owns what and how. Measured numbers live in one place each: the
README for headlines, with [project-overview-pdr.md](project-overview-pdr.md#evidence) pointing to
the command and output field that produce them.
