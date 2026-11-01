"""Request and response schemas: the API boundary validates every customer row.

A customer carries exactly the twelve pre-treatment Criteo features ``f0``-``f11`` as finite
numbers, plus an optional caller ``id`` that is echoed back. Unknown fields are rejected. The
treatment flag, ``exposure`` (post-treatment) and the outcomes are rejected with a leakage
message, because a score must not depend on what happened after assignment. Batches hold 1 to
``MAX_BATCH`` customers.

Features are strict: a JSON number (integer or float) is accepted, but a boolean or a numeric
string (``true``, ``"12.5"``) is rejected instead of coerced. A client that sends those has a
serialisation bug, and a coerced score would hide it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uplift_lab.data.criteo import FEATURES
from uplift_lab.uplift.features import NON_FEATURE_COLUMNS

MAX_BATCH = 1000
# Strict: int or float only (no bool, no numeric string); finiteness comes from the model config.
Feature = Annotated[float, Field(strict=True)]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str | None = Field(default=None, max_length=64)
    f0: Feature
    f1: Feature
    f2: Feature
    f3: Feature
    f4: Feature
    f5: Feature
    f6: Feature
    f7: Feature
    f8: Feature
    f9: Feature
    f10: Feature
    f11: Feature

    @model_validator(mode="before")
    @classmethod
    def _reject_leakage(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaked = sorted(set(data) & NON_FEATURE_COLUMNS)
            if leaked:
                raise ValueError(
                    f"{leaked} are not accepted: exposure is post-treatment and treatment/outcome "
                    "columns are labels; send pre-treatment features f0-f11 only"
                )
        return data

    def features(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in FEATURES)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class DecideRequest(ScoreRequest):
    """Exactly one rule: a named budget from the model manifest, or a share of this batch."""

    budget: str | None = Field(
        default=None, description="named budget, e.g. '0.1': treat if uplift >= its threshold"
    )
    budget_fraction: float | None = Field(
        default=None, gt=0.0, le=1.0, description="treat the top share of this batch by uplift"
    )

    @model_validator(mode="after")
    def _one_rule(self) -> Self:
        if (self.budget is None) == (self.budget_fraction is None):
            raise ValueError("give exactly one of budget (named) or budget_fraction")
        return self


class ModelInfo(BaseModel):
    kind: str
    features: list[str]
    schema_version: int
    uplift_lab_version: str
    git_sha: str | None
    source: str
    budgets: dict[str, float]


class Health(BaseModel):
    status: Literal["ok"]
    model: ModelInfo


class Score(BaseModel):
    id: str | None
    uplift: float


class ScoreResponse(BaseModel):
    model_kind: str
    scores: list[Score]


class Decision(Score):
    treat: bool


class DecisionRule(BaseModel):
    mode: Literal["threshold", "batch_top_k"]
    budget: str | None = None
    budget_fraction: float | None = None
    threshold: float | None = None


class DecideResponse(BaseModel):
    model_kind: str
    rule: DecisionRule
    n_treated: int
    decisions: list[Decision]
