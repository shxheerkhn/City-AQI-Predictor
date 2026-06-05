from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from aqi_predictor.configs.settings import AppConfig
from aqi_predictor.feature_store.base import FeatureStoreBackend
from aqi_predictor.models.domain import AQIMeasurement, ModelMetadata
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


class HopsworksFeatureStore(FeatureStoreBackend):
    def __init__(self, project_name: str, feature_group_name: str | None = None) -> None:
        self.project_name = project_name
        self.feature_group_name = feature_group_name or "aqi_feature_group"

    @classmethod
    def from_config(cls, config: AppConfig) -> "HopsworksFeatureStore":
        try:
            project_name = config.hopsworks_project_name or config.cloud_project_id or "aqi-project"
            return cls(project_name=project_name)
        except Exception as exc:
            LOGGER.exception("Failed to initialize Hopsworks feature store")
            raise RuntimeError(f"Failed to initialize Hopsworks feature store: {exc}") from exc

    def _unsupported(self, action: str) -> RuntimeError:
        return RuntimeError(
            f"Hopsworks feature store backend is selected, but the Hopsworks SDK is not wired in this workspace for {action}. "
            "Use the local backend in development or install/configure Hopsworks in production."
        )

    def save_raw_measurements(self, measurements: list[AQIMeasurement]) -> None:
        raise self._unsupported("save_raw_measurements")

    def save_feature_frame(self, frame: pd.DataFrame, feature_version: str) -> None:
        raise self._unsupported("save_feature_frame")

    def load_feature_frame(self, city_id: str | None = None, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        raise self._unsupported("load_feature_frame")

    def load_raw_measurements(self, city_id: str | None = None, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        raise self._unsupported("load_raw_measurements")

    def list_city_ids(self) -> list[str]:
        raise self._unsupported("list_city_ids")

    def register_model(self, metadata: ModelMetadata) -> None:
        raise self._unsupported("register_model")

    def load_latest_champion_model(self) -> dict[str, Any]:
        raise self._unsupported("load_latest_champion_model")

    def record_prediction(self, city_id: str, timestamp: datetime, request_payload: dict[str, Any], response_payload: dict[str, Any], model_version: str) -> None:
        raise self._unsupported("record_prediction")

    def get_model_history(self, city_scope: str | None = None, limit: int = 100) -> pd.DataFrame:
        raise self._unsupported("get_model_history")

    def get_model_comparison(self, city_scope: str | None = None) -> pd.DataFrame:
        raise self._unsupported("get_model_comparison")

    def get_model_by_version(self, model_version: str) -> dict[str, Any]:
        raise self._unsupported("get_model_by_version")

    def set_champion_model(self, model_version: str) -> None:
        raise self._unsupported("set_champion_model")

    def register_model_to_history(self, metadata: ModelMetadata, rank_position: int) -> None:
        raise self._unsupported("register_model_to_history")
