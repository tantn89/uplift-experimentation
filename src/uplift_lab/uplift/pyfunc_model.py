"""MLflow pyfunc wrapper around an uplift model directory (an MLflow models-from-code script).

``tracking.log_uplift_run`` logs this file by path rather than pickling an object, so loading a
registered model runs reviewed source code instead of deserialising arbitrary bytes. The model
directory itself travels as the ``uplift_model`` artifact and is loaded with
``artifacts.load_learner``. Input: a DataFrame with the manifest's feature columns (extra
columns are ignored). Output: a DataFrame with one ``uplift`` column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.pyfunc.model import PythonModel

from uplift_lab.uplift.artifacts import PYFUNC_ARTIFACT_KEY, load_learner
from uplift_lab.uplift.learners import UpliftLearner


class UpliftPyfunc(PythonModel):
    _learner: UpliftLearner

    def load_context(self, context: Any) -> None:
        self._learner = load_learner(Path(context.artifacts[PYFUNC_ARTIFACT_KEY]))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        uplift = self._learner.predict(model_input)
        return pd.DataFrame({"uplift": uplift}, index=model_input.index)


mlflow.models.set_model(UpliftPyfunc())
