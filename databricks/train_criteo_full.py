"""Databricks job entry point (Python script task) for the full-data Criteo uplift training.

The job's serverless environment installs the ``uplift-lab`` wheel built from this repository,
so this file only forwards the task parameters to the same module a local run uses:
``python -m uplift_lab.pipelines.criteo_full_training``.
"""

import sys

from uplift_lab.pipelines.criteo_full_training import main

if __name__ == "__main__":
    code = main(sys.argv[1:])
    if code:
        raise SystemExit(code)
