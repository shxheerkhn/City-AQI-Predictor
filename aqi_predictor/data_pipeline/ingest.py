from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from aqi_predictor.configs.settings import AppConfig, CityProfile
from aqi_predictor.data_pipeline.sources import AQIDataSource, SyntheticAQIDataSource
from aqi_predictor.feature_store.local_store import LocalFeatureStore
from aqi_predictor.models.domain import AQIMeasurement
from aqi_predictor.utils.logging import get_logger
from aqi_predictor.utils.validation import validate_sensor_frame


LOGGER = get_logger(__name__)


RAW_COLUMNS = [
    "city_id",
    "timestamp",
    "aqi",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "rainfall",
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "source",
]


def select_data_source(config: AppConfig) -> AQIDataSource:
    provider = config.external_api_provider.lower()
    if provider == "openweather" and config.external_api_key:
        from aqi_predictor.data_pipeline.sources import OpenWeatherAQIDataSource

        return OpenWeatherAQIDataSource(api_key=config.external_api_key, base_url=config.external_api_url or "https://api.openweathermap.org/data/2.5")
    if provider == "aqicn" and config.external_api_key:
        from aqi_predictor.data_pipeline.sources import AQICNAQIDataSource

        return AQICNAQIDataSource(
            api_token=config.external_api_key,
            base_url=config.aqicn_api_url,
            station_id=config.aqicn_station_id,
        )
    if provider in {"openweather", "aqicn"}:
        LOGGER.warning("Configured provider %s is missing API credentials. Falling back to synthetic source.", provider)
    return SyntheticAQIDataSource()


def measurement_to_frame(measurement: AQIMeasurement) -> pd.DataFrame:
    try:
        return pd.DataFrame([measurement.to_dict()])[RAW_COLUMNS]
    except Exception as exc:
        LOGGER.exception("Failed to convert measurement to frame")
        raise RuntimeError(f"Failed to convert measurement to frame: {exc}") from exc


def ingest_city_range(
    store: LocalFeatureStore,
    source: AQIDataSource,
    city: CityProfile,
    start: datetime,
    end: datetime,
    frequency_hours: int = 1,
) -> pd.DataFrame:
    try:
        current = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        end_utc = end.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        records: list[pd.DataFrame] = []
        measurements: list[AQIMeasurement] = []
        while current <= end_utc:
            measurement = source.fetch_observation(city, current)
            validation_frame = measurement_to_frame(measurement)
            validation = validate_sensor_frame(validation_frame, RAW_COLUMNS)
            if not validation.is_valid:
                raise ValueError(f"Invalid measurement for {city.city_id} at {current.isoformat()}: {validation.issues}")
            records.append(validation_frame)
            measurements.append(measurement)
            current += timedelta(hours=frequency_hours)
        if measurements:
            store.save_raw_measurements(measurements)
        if not records:
            return pd.DataFrame(columns=RAW_COLUMNS)
        frame = pd.concat(records, ignore_index=True)
        LOGGER.info("Ingested %s records for city %s", len(frame), city.city_id)
        return frame
    except Exception as exc:
        LOGGER.exception("City ingestion failed")
        raise RuntimeError(f"City ingestion failed: {exc}") from exc


def backfill_cities(config: AppConfig, store: LocalFeatureStore, source: AQIDataSource | None = None) -> pd.DataFrame:
    try:
        provider = source or select_data_source(config)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=config.backfill_days)
        city_frames: list[pd.DataFrame] = []
        for city in config.cities:
            city_frame = ingest_city_range(store, provider, city, start, end, frequency_hours=config.ingestion_frequency_hours)
            if not city_frame.empty:
                city_frames.append(city_frame)
        if not city_frames:
            return pd.DataFrame(columns=RAW_COLUMNS)
        combined = pd.concat(city_frames, ignore_index=True)
        LOGGER.info("Backfill completed with %s rows", len(combined))
        return combined
    except Exception as exc:
        LOGGER.exception("Backfill failed")
        raise RuntimeError(f"Backfill failed: {exc}") from exc


def ingest_current_snapshot(config: AppConfig, store: LocalFeatureStore, source: AQIDataSource | None = None) -> pd.DataFrame:
    try:
        provider = source or select_data_source(config)
        now = datetime.now(timezone.utc)
        frames: list[pd.DataFrame] = []
        measurements: list[AQIMeasurement] = []
        for city in config.cities:
            measurement = provider.fetch_observation(city, now)
            validation_frame = measurement_to_frame(measurement)
            validation = validate_sensor_frame(validation_frame, RAW_COLUMNS)
            if not validation.is_valid:
                raise ValueError(f"Invalid snapshot for {city.city_id}: {validation.issues}")
            frames.append(validation_frame)
            measurements.append(measurement)
        if measurements:
            store.save_raw_measurements(measurements)
        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        LOGGER.info("Incremental ingestion completed with %s rows", len(combined))
        return combined
    except Exception as exc:
        LOGGER.exception("Incremental ingestion failed")
        raise RuntimeError(f"Incremental ingestion failed: {exc}") from exc
