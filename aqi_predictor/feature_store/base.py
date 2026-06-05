from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from aqi_predictor.models.domain import AQIMeasurement, ModelMetadata


class FeatureStoreBackend(ABC):
    @abstractmethod
    def save_raw_measurements(self, measurements: list[AQIMeasurement]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_feature_frame(self, frame: pd.DataFrame, feature_version: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_feature_frame(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_raw_measurements(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def list_city_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def register_model(self, metadata: ModelMetadata) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_latest_champion_model(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def record_prediction(
        self,
        city_id: str,
        timestamp: datetime,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        model_version: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_model_history(self, city_scope: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Get history of all trained models with metrics."""
        raise NotImplementedError

    @abstractmethod
    def get_model_comparison(self, city_scope: str | None = None) -> pd.DataFrame:
        """Get model comparison table sorted by RMSE."""
        raise NotImplementedError

    @abstractmethod
    def get_model_by_version(self, model_version: str) -> dict[str, Any]:
        """Load a specific model version by version string."""
        raise NotImplementedError

    @abstractmethod
    def set_champion_model(self, model_version: str) -> None:
        """Mark a specific model version as champion (rollback)."""
        raise NotImplementedError

    @abstractmethod
    def register_model_to_history(self, metadata: ModelMetadata, rank_position: int) -> None:
        """Register a model to the history table with rank position."""
        raise NotImplementedError
