from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from aqi_predictor.configs.settings import AppConfig, CityProfile
from aqi_predictor.feature_pipeline.engine import build_future_context, build_supervised_frame, prepare_model_frame
from aqi_predictor.feature_store.factory import build_feature_store
from aqi_predictor.models.domain import ForecastPoint
from aqi_predictor.monitoring.metrics import REGISTRY
from aqi_predictor.training_pipeline.forecasting import recursive_forecast
from aqi_predictor.training_pipeline.modeling import ModelArtifact
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PredictionBundle:
    model_metadata: dict[str, Any]
    model_artifact: ModelArtifact


class AQIPredictionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = build_feature_store(config)
        self._bundle: PredictionBundle | None = None

    def _load_bundle(self) -> PredictionBundle:
        try:
            metadata = self.store.load_latest_champion_model()
            model_artifact = ModelArtifact.load(metadata["artifact_path"])
            bundle = PredictionBundle(model_metadata=metadata, model_artifact=model_artifact)
            self._bundle = bundle
            return bundle
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to load prediction bundle")
            raise RuntimeError(f"Failed to load prediction bundle: {exc}") from exc

    def get_bundle(self) -> PredictionBundle:
        try:
            if self._bundle is None:
                return self._load_bundle()
            return self._bundle
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to get prediction bundle")
            raise RuntimeError(f"Failed to get prediction bundle: {exc}") from exc

    def try_get_bundle(self) -> PredictionBundle | None:
        try:
            return self.get_bundle()
        except LookupError:
            return None
        except RuntimeError as exc:
            if "No trained model registered" in str(exc):
                return None
            raise

    def health(self) -> dict[str, Any]:
        try:
            bundle = self.get_bundle()
            return {
                "status": "ok",
                "model_name": bundle.model_metadata["model_name"],
                "model_version": bundle.model_metadata["model_version"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            LOGGER.exception("Health check failed")
            raise RuntimeError(f"Health check failed: {exc}") from exc

    def model_info(self) -> dict[str, Any]:
        try:
            bundle = self.get_bundle()
            return bundle.model_metadata
        except Exception as exc:
            LOGGER.exception("Model info retrieval failed")
            raise RuntimeError(f"Model info retrieval failed: {exc}") from exc

    def metrics(self) -> dict[str, Any]:
        try:
            metadata = self.model_info()
            return {
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "metrics": metadata["metrics"],
                "city_scope": metadata["city_scope"],
                "is_champion": metadata["is_champion"],
                "service_metrics": REGISTRY.snapshot(),
            }
        except Exception as exc:
            LOGGER.exception("Metrics retrieval failed")
            raise RuntimeError(f"Metrics retrieval failed: {exc}") from exc

    def _load_history(self, city_id: str) -> pd.DataFrame:
        try:
            history = self.store.load_raw_measurements(city_id=city_id)
            if history.empty:
                raise LookupError(f"No raw measurements found for city {city_id}")
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
            return history.sort_values("timestamp").reset_index(drop=True)
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to load history")
            raise RuntimeError(f"Failed to load history: {exc}") from exc

    def predict_next(self, city_id: str) -> dict[str, Any]:
        try:
            bundle = self.get_bundle()
            history = self._load_history(city_id)
            city = self._resolve_city(city_id)
            point = recursive_forecast(bundle.model_artifact, history, city, horizon_hours=1)[0]
            payload = {
                "city_id": point.city_id,
                "timestamp": point.timestamp.isoformat(),
                "aqi": point.aqi,
                "lower_bound": point.lower_bound,
                "upper_bound": point.upper_bound,
                "hazard_flag": point.hazard_flag,
                "model_name": point.model_name,
            }
            self.store.record_prediction(city_id, datetime.now(timezone.utc), {"city_id": city_id, "forecast_hours": 1}, payload, bundle.model_metadata["model_version"])
            REGISTRY.record_prediction()
            return payload
        except Exception as exc:
            LOGGER.exception("Next prediction failed")
            REGISTRY.record_error(str(exc))
            raise RuntimeError(f"Next prediction failed: {exc}") from exc

    def forecast(self, city_id: str, horizon_hours: int) -> dict[str, Any]:
        try:
            bundle = self.get_bundle()
            history = self._load_history(city_id)
            city = self._resolve_city(city_id)
            points = recursive_forecast(bundle.model_artifact, history, city, horizon_hours=horizon_hours)
            payload_points = [
                {
                    "city_id": point.city_id,
                    "timestamp": point.timestamp.isoformat(),
                    "aqi": point.aqi,
                    "lower_bound": point.lower_bound,
                    "upper_bound": point.upper_bound,
                    "hazard_flag": point.hazard_flag,
                    "model_name": point.model_name,
                }
                for point in points
            ]
            hazard_count = int(sum(1 for point in points if point.hazard_flag))
            self.store.record_prediction(
                city_id,
                datetime.now(timezone.utc),
                {"city_id": city_id, "forecast_hours": horizon_hours},
                {"points": payload_points, "hazard_count": hazard_count},
                bundle.model_metadata["model_version"],
            )
            REGISTRY.record_forecast()
            return {
                "city_id": city_id,
                "model_name": bundle.model_metadata["model_name"],
                "model_version": bundle.model_metadata["model_version"],
                "forecast_hours": horizon_hours,
                "points": payload_points,
                "hazard_count": hazard_count,
            }
        except Exception as exc:
            LOGGER.exception("Forecast failed")
            REGISTRY.record_error(str(exc))
            raise RuntimeError(f"Forecast failed: {exc}") from exc

    def available_cities(self) -> list[str]:
        try:
            configured_city_ids = [city.city_id for city in self.config.cities]
            if configured_city_ids:
                return configured_city_ids
            city_ids = self.store.list_city_ids()
            return city_ids
        except Exception as exc:
            LOGGER.exception("Failed to get available cities")
            raise RuntimeError(f"Failed to get available cities: {exc}") from exc

    def latest_history(self, city_id: str, window_hours: int = 168) -> pd.DataFrame:
        try:
            history = self._load_history(city_id)
            cutoff = history["timestamp"].max() - pd.Timedelta(hours=window_hours)
            return history[history["timestamp"] >= cutoff].reset_index(drop=True)
        except LookupError:
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.exception("Failed to get latest history")
            raise RuntimeError(f"Failed to get latest history: {exc}") from exc

    def try_latest_history(self, city_id: str, window_hours: int = 168) -> pd.DataFrame:
        try:
            return self.latest_history(city_id, window_hours=window_hours)
        except RuntimeError as exc:
            if "No raw measurements found" in str(exc):
                return pd.DataFrame()
            raise

    def _resolve_city(self, city_id: str) -> CityProfile:
        try:
            for city in self.config.cities:
                if city.city_id == city_id:
                    return city
            return CityProfile(city_id=city_id, country="unknown", latitude=0.0, longitude=0.0)
        except Exception as exc:
            LOGGER.exception("Failed to resolve city")
            raise RuntimeError(f"Failed to resolve city: {exc}") from exc
