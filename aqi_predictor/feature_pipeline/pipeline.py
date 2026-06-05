from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from aqi_predictor.feature_pipeline.engine import FEATURE_VERSION, build_supervised_frame
from aqi_predictor.feature_store.local_store import LocalFeatureStore
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


def materialize_features(store: LocalFeatureStore, city_id: str | None = None) -> pd.DataFrame:
    try:
        raw_frame = store.load_raw_measurements(city_id=city_id)
        if raw_frame.empty:
            LOGGER.info("No raw measurements available for feature materialization")
            return raw_frame
        raw_frame["timestamp"] = pd.to_datetime(raw_frame["timestamp"], utc=True)
        supervised = build_supervised_frame(raw_frame)
        if supervised.empty:
            LOGGER.info("Supervised frame is empty after feature materialization")
            return supervised
        store.save_feature_frame(supervised, FEATURE_VERSION)
        LOGGER.info("Materialized %s feature rows", len(supervised))
        return supervised
    except Exception as exc:
        LOGGER.exception("Feature materialization failed")
        raise RuntimeError(f"Feature materialization failed: {exc}") from exc


def materialize_all_cities(store: LocalFeatureStore) -> pd.DataFrame:
    try:
        city_ids = store.list_city_ids()
        frames: list[pd.DataFrame] = []
        for city_id in city_ids:
            frame = materialize_features(store, city_id=city_id)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    except Exception as exc:
        LOGGER.exception("Materializing all cities failed")
        raise RuntimeError(f"Materializing all cities failed: {exc}") from exc
