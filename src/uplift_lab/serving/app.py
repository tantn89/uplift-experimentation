"""FastAPI service: ``GET /health``, ``POST /score`` and ``POST /decide``.

Run with ``uvicorn uplift_lab.serving.app:create_app --factory``. Settings come from the
environment (``settings.py``). The model is resolved and loaded once, in the lifespan hook, and a
bad model directory fails startup instead of the first request. Endpoints are plain ``def``,
so FastAPI runs the CPU-bound LightGBM scoring in its thread pool.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from uplift_lab import __version__
from uplift_lab.experiments.estimators import FloatArray
from uplift_lab.monitoring.traffic_log import TrafficLog
from uplift_lab.serving.model_service import ModelService, UnknownBudgetError, top_share_mask
from uplift_lab.serving.model_source import resolve_model_dir
from uplift_lab.serving.schemas import (
    DecideRequest,
    DecideResponse,
    Decision,
    DecisionRule,
    Health,
    Score,
    ScoreRequest,
    ScoreResponse,
)
from uplift_lab.serving.settings import ServingSettings


def _service(request: Request) -> ModelService:
    service = getattr(request.app.state, "service", None)
    if not isinstance(service, ModelService):
        raise HTTPException(status_code=503, detail="model not loaded")
    return service


def _score_and_log(
    request: Request, endpoint: str, service: ModelService, body: ScoreRequest
) -> FloatArray:
    """Score a validated batch and record it in the sampled traffic log, if enabled."""
    frame = service.frame(body.customers)
    scores = service.score(frame)
    traffic = getattr(request.app.state, "traffic", None)
    if isinstance(traffic, TrafficLog):
        traffic.record(endpoint, frame, scores)
    return scores


async def _validation_error(request: Request, error: Exception) -> JSONResponse:
    """422 without echoing inputs: no customer data in responses, and NaN inputs stay valid JSON."""
    details = error.errors() if isinstance(error, RequestValidationError) else []
    cleaned = [
        {
            "loc": list(item.get("loc", ())),
            "msg": str(item.get("msg", "")),
            "type": item.get("type"),
        }
        for item in details
    ]
    return JSONResponse(status_code=422, content={"detail": cleaned})


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """Build the app; ``settings`` default to the environment (read here, so errors are early)."""
    config = settings or ServingSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with tempfile.TemporaryDirectory(prefix="uplift-model-") as download_root:
            model_dir, source = resolve_model_dir(config, Path(download_root))
            app.state.service = ModelService(model_dir, source)
            app.state.traffic = (
                TrafficLog(
                    config.traffic_log,
                    sample_rate=config.traffic_sample_rate,
                    max_rows=config.traffic_max_rows,
                    max_bytes=config.traffic_max_bytes,
                )
                if config.traffic_log is not None
                else None
            )
            yield

    app = FastAPI(title="uplift-lab scoring", version=__version__, lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, _validation_error)

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model=_service(request).info)

    @app.post("/score")
    def score(body: ScoreRequest, request: Request) -> ScoreResponse:
        service = _service(request)
        uplift = _score_and_log(request, "score", service, body)
        return ScoreResponse(
            model_kind=service.info.kind,
            scores=[
                Score(id=c.id, uplift=float(u)) for c, u in zip(body.customers, uplift, strict=True)
            ],
        )

    @app.post("/decide")
    def decide(body: DecideRequest, request: Request) -> DecideResponse:
        service = _service(request)
        uplift = _score_and_log(request, "decide", service, body)
        if body.budget is not None:
            try:
                threshold = service.threshold(body.budget)
            except UnknownBudgetError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            treat = uplift >= threshold
            rule = DecisionRule(mode="threshold", budget=body.budget, threshold=threshold)
        else:
            fraction = float(body.budget_fraction or 0.0)  # set: the schema requires one rule
            treat = top_share_mask(uplift, fraction)
            rule = DecisionRule(mode="batch_top_k", budget_fraction=fraction)
        decisions = [
            Decision(id=c.id, uplift=float(u), treat=bool(t))
            for c, u, t in zip(body.customers, uplift, treat, strict=True)
        ]
        return DecideResponse(
            model_kind=service.info.kind,
            rule=rule,
            n_treated=int(np.count_nonzero(treat)),
            decisions=decisions,
        )

    return app
