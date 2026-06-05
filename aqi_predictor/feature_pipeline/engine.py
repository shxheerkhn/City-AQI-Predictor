from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aqi_predictor.configs.settings import CityProfile
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)
FEATURE_VERSION = "v1.0"


NUMERIC_FEATURES = [
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
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "day_of_year",
    "season",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "day_sin",
    "day_cos",
    "pm25_roll_mean_24",
    "aqi_roll_mean_24",
    "aqi_roll_std_24",
    "aqi_rate_24",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_lag_72",
]


@dataclass(slots=True)
class FutureContext:
    city_id: str
    timestamp: pd.Timestamp
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "city_id": self.city_id,
            "timestamp": self.timestamp,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "pressure": self.pressure,
            "rainfall": self.rainfall,
            "pm25": self.pm25,
            "pm10": self.pm10,
            "co": self.co,
            "no2": self.no2,
            "so2": self.so2,
            "o3": self.o3,
        }


def _season_from_month(month: int) -> int:
    if month in {12, 1, 2}:
        return 0
    if month in {3, 4, 5}:
        return 1
    if month in {6, 7, 8}:
        return 2
    return 3


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def build_supervised_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        if raw_frame.empty:
            return raw_frame.copy()
        frame = raw_frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values(["city_id", "timestamp"]).reset_index(drop=True)
        group = frame.groupby("city_id", group_keys=False)
        frame["hour_of_day"] = frame["timestamp"].dt.hour
        frame["day_of_week"] = frame["timestamp"].dt.dayofweek
        frame["day_of_month"] = frame["timestamp"].dt.day
        frame["month"] = frame["timestamp"].dt.month
        frame["quarter"] = frame["timestamp"].dt.quarter
        frame["day_of_year"] = frame["timestamp"].dt.dayofyear
        frame["season"] = frame["month"].apply(_season_from_month)
        frame["hour_sin"], frame["hour_cos"] = zip(*frame["hour_of_day"].map(lambda x: _cyclical(float(x), 24.0)))
        frame["dow_sin"], frame["dow_cos"] = zip(*frame["day_of_week"].map(lambda x: _cyclical(float(x), 7.0)))
        frame["day_sin"], frame["day_cos"] = zip(*frame["day_of_year"].map(lambda x: _cyclical(float(x), 365.25)))
        frame["aqi_lag_1"] = group["aqi"].shift(1)
        frame["aqi_lag_24"] = group["aqi"].shift(24)
        frame["aqi_lag_72"] = group["aqi"].shift(72)
        frame["pm25_roll_mean_24"] = group["pm25"].transform(lambda series: series.rolling(24, min_periods=6).mean())
        frame["aqi_roll_mean_24"] = group["aqi"].transform(lambda series: series.rolling(24, min_periods=6).mean())
        frame["aqi_roll_std_24"] = group["aqi"].transform(lambda series: series.rolling(24, min_periods=6).std(ddof=0))
        frame["aqi_rate_24"] = (frame["aqi"] - frame["aqi_lag_24"]) / 24.0
        frame["aqi_target"] = group["aqi"].shift(-1)
        frame = frame.dropna(subset=["aqi_target"]).reset_index(drop=True)
        frame["season"] = frame["season"].astype(int)
        return frame
    except Exception as exc:
        LOGGER.exception("Failed to build supervised frame")
        raise RuntimeError(f"Failed to build supervised frame: {exc}") from exc


def build_future_context(history_frame: pd.DataFrame, future_timestamp: pd.Timestamp, city: CityProfile) -> FutureContext:
    try:
        if history_frame.empty:
            raise ValueError("History frame is empty")
        hist = history_frame.sort_values("timestamp").tail(48).copy()
        last = hist.iloc[-1]
        recent = hist.tail(24)
        hour_cycle = math.sin(2.0 * math.pi * future_timestamp.hour / 24.0)
        day_cycle = math.sin(2.0 * math.pi * future_timestamp.dayofyear / 365.25)
        temp = float(np.clip(float(recent["temperature"].mean()) + 4.0 * hour_cycle + 2.0 * day_cycle, -10.0, 55.0))
        humidity = float(np.clip(float(recent["humidity"].mean()) - 5.0 * hour_cycle + 2.0 * day_cycle, 5.0, 100.0))
        wind_speed = float(np.clip(float(recent["wind_speed"].mean()) + 0.6 * math.cos(2.0 * math.pi * future_timestamp.hour / 24.0), 0.1, 15.0))
        pressure = float(np.clip(float(recent["pressure"].mean()) + 1.5 * math.cos(2.0 * math.pi * future_timestamp.dayofyear / 365.25), 970.0, 1040.0))
        rainfall = float(max(0.0, float(recent["rainfall"].mean()) + 0.2 * (1.0 - hour_cycle)))
        pm25 = float(max(1.0, float(last["pm25"]) * 0.92 + 0.08 * float(recent["pm25"].mean()) + 1.5 * (1.0 - wind_speed / 10.0)))
        pm10 = float(max(pm25 + 5.0, float(last["pm10"]) * 0.93 + 0.07 * float(recent["pm10"].mean()) + 3.0))
        co = float(max(0.1, float(last["co"]) * 0.96 + 0.04 * float(recent["co"].mean())))
        no2 = float(max(1.0, float(last["no2"]) * 0.95 + 0.05 * float(recent["no2"].mean())))
        so2 = float(max(1.0, float(last["so2"]) * 0.94 + 0.06 * float(recent["so2"].mean())))
        o3 = float(max(1.0, float(last["o3"]) * 0.93 + 0.07 * float(recent["o3"].mean()) + 0.2 * temp - 0.1 * humidity))
        return FutureContext(
            city_id=city.city_id,
            timestamp=future_timestamp,
            temperature=temp,
            humidity=humidity,
            wind_speed=wind_speed,
            pressure=pressure,
            rainfall=rainfall,
            pm25=pm25,
            pm10=pm10,
            co=co,
            no2=no2,
            so2=so2,
            o3=o3,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build future context")
        raise RuntimeError(f"Failed to build future context: {exc}") from exc


def assemble_feature_row(history_frame: pd.DataFrame, future_context: FutureContext) -> pd.DataFrame:
    try:
        hist = history_frame.sort_values("timestamp").copy()
        future_timestamp = pd.Timestamp(future_context.timestamp)
        last_aqi = float(hist.iloc[-1]["aqi"])
        aqi_lag_1 = last_aqi
        aqi_lag_24 = float(hist.iloc[-24]["aqi"]) if len(hist) >= 24 else last_aqi
        aqi_lag_72 = float(hist.iloc[-72]["aqi"]) if len(hist) >= 72 else aqi_lag_24
        history_window = hist.tail(24)
        hour_of_day = int(future_timestamp.hour)
        day_of_week = int(future_timestamp.dayofweek)
        day_of_month = int(future_timestamp.day)
        month = int(future_timestamp.month)
        quarter = int(future_timestamp.quarter)
        day_of_year = int(future_timestamp.dayofyear)
        season = _season_from_month(month)
        hour_sin, hour_cos = _cyclical(float(hour_of_day), 24.0)
        dow_sin, dow_cos = _cyclical(float(day_of_week), 7.0)
        day_sin, day_cos = _cyclical(float(day_of_year), 365.25)
        pm25_roll_mean_24 = float(history_window["pm25"].mean())
        aqi_roll_mean_24 = float(history_window["aqi"].mean())
        aqi_roll_std_24 = float(history_window["aqi"].std(ddof=0) if len(history_window) > 1 else 0.0)
        aqi_rate_24 = float((last_aqi - aqi_lag_24) / 24.0)
        row = {
            "city_id": future_context.city_id,
            "timestamp": future_timestamp,
            "temperature": future_context.temperature,
            "humidity": future_context.humidity,
            "wind_speed": future_context.wind_speed,
            "pressure": future_context.pressure,
            "rainfall": future_context.rainfall,
            "pm25": future_context.pm25,
            "pm10": future_context.pm10,
            "co": future_context.co,
            "no2": future_context.no2,
            "so2": future_context.so2,
            "o3": future_context.o3,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "quarter": quarter,
            "day_of_year": day_of_year,
            "season": season,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "pm25_roll_mean_24": pm25_roll_mean_24,
            "aqi_roll_mean_24": aqi_roll_mean_24,
            "aqi_roll_std_24": aqi_roll_std_24,
            "aqi_rate_24": aqi_rate_24,
            "aqi_lag_1": aqi_lag_1,
            "aqi_lag_24": aqi_lag_24,
            "aqi_lag_72": aqi_lag_72,
        }
        return pd.DataFrame([row])
    except Exception as exc:
        LOGGER.exception("Failed to assemble feature row")
        raise RuntimeError(f"Failed to assemble feature row: {exc}") from exc


def prepare_model_frame(supervised_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    try:
        frame = supervised_frame.copy()
        frame = pd.get_dummies(frame, columns=["city_id"], prefix="city", drop_first=False)
        frame = pd.get_dummies(frame, columns=["season"], prefix="season", drop_first=False)
        drop_columns = ["timestamp", "created_at", "aqi_target", "source"]
        feature_columns = [column for column in frame.columns if column not in drop_columns and column != "aqi"]
        X = frame[feature_columns].copy()
        y = frame["aqi_target"].astype(float).copy()
        X = X.ffill().bfill().fillna(0.0)
        return X, y, feature_columns
    except Exception as exc:
        LOGGER.exception("Failed to prepare model frame")
        raise RuntimeError(f"Failed to prepare model frame: {exc}") from exc


def align_columns(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    try:
        aligned = frame.copy()
        for column in expected_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0
        extra_columns = [column for column in aligned.columns if column not in expected_columns]
        if extra_columns:
            aligned = aligned.drop(columns=extra_columns)
        return aligned[expected_columns].fillna(0.0)
    except Exception as exc:
        LOGGER.exception("Failed to align model columns")
        raise RuntimeError(f"Failed to align model columns: {exc}") from exc
