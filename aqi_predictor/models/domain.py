from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class AQIMeasurement:
    city_id: str
    timestamp: datetime
    aqi: float
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float
    rainfall: float
    pm25: float
    pm10: float
    co: float
    no2: float
    so2: float
    o3: float
    source: str = "synthetic"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class ForecastPoint:
    city_id: str
    timestamp: datetime
    aqi: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    hazard_flag: bool = False
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(slots=True)
class ModelMetadata:
    model_name: str
    model_family: str
    model_version: str
    artifact_path: str
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    trained_at: datetime
    city_scope: str = "global"
    is_champion: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trained_at"] = self.trained_at.isoformat()
        return payload
