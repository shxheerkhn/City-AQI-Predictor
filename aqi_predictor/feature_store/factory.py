from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aqi_predictor.configs.settings import AppConfig
from aqi_predictor.feature_store.local_store import LocalFeatureStore
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class FeatureStoreConfig:
    backend: str
    database_path: Path
    project_name: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    bucket_name: str | None = None


def build_feature_store(config: AppConfig):
    selected = config.feature_store_backend.lower()
    if selected not in {"local", "vertex", "hopsworks"}:
        selected = "local"
    try:
        if selected == "local":
            return LocalFeatureStore(config.database_path)
        if selected == "vertex":
            from aqi_predictor.feature_store.vertex_store import VertexFeatureStore

            return VertexFeatureStore.from_config(config)
        if selected == "hopsworks":
            from aqi_predictor.feature_store.hopsworks_store import HopsworksFeatureStore

            return HopsworksFeatureStore.from_config(config)
        return LocalFeatureStore(config.database_path)
    except Exception as exc:
        LOGGER.exception("Failed to build feature store backend %s", selected)
        raise RuntimeError(f"Failed to build feature store backend {selected}: {exc}") from exc
