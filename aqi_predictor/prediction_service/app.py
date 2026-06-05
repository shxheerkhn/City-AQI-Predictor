from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from aqi_predictor.configs.settings import load_config
from aqi_predictor.prediction_service.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastPointResponse,
    HealthResponse,
    MetricResponse,
    ModelInfoResponse,
    PredictRequest,
)
from aqi_predictor.prediction_service.service import AQIPredictionService
from aqi_predictor.monitoring.metrics import REGISTRY
from aqi_predictor.utils.logging import configure_logging, get_logger


configure_logging()
LOGGER = get_logger(__name__)
CONFIG = load_config()
SERVICE = AQIPredictionService(CONFIG)

_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_MAX_REQUESTS = 60
_RATE_LIMIT_WINDOW_SECONDS = 60.0

app = FastAPI(title="Pearls AQI Predictor API", version="0.1.0")


def _authorize(x_api_key: str | None = Header(default=None)) -> None:
    try:
        if CONFIG.api_access_token and x_api_key != CONFIG.api_access_token:
            raise HTTPException(status_code=401, detail="Invalid API key")
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Authorization failure")
        raise HTTPException(status_code=500, detail=f"Authorization failure: {exc}") from exc


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _RATE_LIMIT_BUCKETS[client_ip]
        while bucket and now - bucket[0] > _RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
            REGISTRY.record_request(request.url.path, 429, (time.perf_counter() - start) * 1000.0)
            REGISTRY.record_error("rate limit exceeded")
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        bucket.append(now)
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        REGISTRY.record_request(request.url.path, response.status_code, duration_ms)
        return response
    except Exception as exc:
        LOGGER.exception("Rate limit middleware failed")
        REGISTRY.record_error(str(exc))
        return JSONResponse(status_code=500, content={"detail": f"Rate limiting failure: {exc}"})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    LOGGER.exception("Unhandled API exception")
    REGISTRY.record_error(str(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(_: Any = Depends(_authorize)) -> HealthResponse:
    try:
        payload = SERVICE.health()
        return HealthResponse(status=payload["status"], timestamp=datetime.fromisoformat(payload["timestamp"]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", response_model=MetricResponse)
def metrics(_: Any = Depends(_authorize)) -> MetricResponse:
    try:
        payload = SERVICE.metrics()
        return MetricResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(_: Any = Depends(_authorize)) -> ModelInfoResponse:
    try:
        payload = SERVICE.model_info()
        return ModelInfoResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest, _: Any = Depends(_authorize)) -> dict[str, Any]:
    try:
        payload = SERVICE.predict_next(request.city_id)
        return {"prediction": payload}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest, _: Any = Depends(_authorize)) -> ForecastResponse:
    try:
        payload = SERVICE.forecast(request.city_id, request.forecast_hours)
        points = [ForecastPointResponse(**point) for point in payload["points"]]
        return ForecastResponse(
            city_id=payload["city_id"],
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            forecast_hours=payload["forecast_hours"],
            points=points,
            hazard_count=payload["hazard_count"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
