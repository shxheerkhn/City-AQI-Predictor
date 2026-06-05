from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    city_id: str = Field(..., min_length=1, max_length=128)
    forecast_hours: int = Field(default=1, ge=1, le=72)


class ForecastRequest(BaseModel):
    city_id: str = Field(..., min_length=1, max_length=128)
    forecast_hours: int = Field(default=72, ge=1, le=72)


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class MetricResponse(BaseModel):
    model_name: str
    model_version: str
    metrics: dict[str, float]
    city_scope: str
    is_champion: bool
    service_metrics: dict[str, Any]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_family: str
    model_version: str
    artifact_path: str
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    trained_at: str
    city_scope: str
    is_champion: bool


class ForecastPointResponse(BaseModel):
    city_id: str
    timestamp: datetime
    aqi: float
    lower_bound: float | None
    upper_bound: float | None
    hazard_flag: bool
    model_name: str


class ForecastResponse(BaseModel):
    city_id: str
    model_name: str
    model_version: str
    forecast_hours: int
    points: list[ForecastPointResponse]
    hazard_count: int
