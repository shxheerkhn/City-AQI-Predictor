from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from aqi_predictor.configs.settings import CityProfile
from aqi_predictor.feature_pipeline.engine import FutureContext, align_columns, assemble_feature_row, build_future_context, build_supervised_frame, prepare_model_frame
from aqi_predictor.models.domain import AQIMeasurement, ForecastPoint
from aqi_predictor.training_pipeline.modeling import ModelArtifact
from aqi_predictor.training_pipeline.modeling import ModelEvaluation
from aqi_predictor.training_pipeline.modeling import compute_evaluation
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RecursiveBacktestResult:
    overall: ModelEvaluation
    daily: dict[str, ModelEvaluation]
    frame: pd.DataFrame
    error_scale: float


def _row_to_measurement(row: dict[str, Any], source: str = "forecast") -> AQIMeasurement:
    return AQIMeasurement(
        city_id=str(row["city_id"]),
        timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
        aqi=float(row["aqi"]),
        temperature=float(row["temperature"]),
        humidity=float(row["humidity"]),
        wind_speed=float(row["wind_speed"]),
        pressure=float(row["pressure"]),
        rainfall=float(row["rainfall"]),
        pm25=float(row["pm25"]),
        pm10=float(row["pm10"]),
        co=float(row["co"]),
        no2=float(row["no2"]),
        so2=float(row["so2"]),
        o3=float(row["o3"]),
        source=source,
    )


def recursive_forecast(
    model_artifact: ModelArtifact,
    history_frame: pd.DataFrame,
    city: CityProfile,
    horizon_hours: int = 72,
) -> list[ForecastPoint]:
    try:
        if history_frame.empty:
            raise ValueError("History frame is empty")
        raw_history = history_frame.copy()
        raw_history["timestamp"] = pd.to_datetime(raw_history["timestamp"], utc=True)
        raw_history = raw_history.sort_values("timestamp").reset_index(drop=True)
        points: list[ForecastPoint] = []
        current_history = raw_history.copy()
        for step in range(1, horizon_hours + 1):
            future_timestamp = pd.Timestamp(current_history.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)
            future_context = build_future_context(current_history, future_timestamp, city)
            feature_row = assemble_feature_row(current_history, future_context)
            if model_artifact.model_family in {"lstm", "gru"}:
                historical_supervised = build_supervised_frame(current_history)
                encoded_history, _, _ = prepare_model_frame(historical_supervised)
                sequence = _build_sequence_input(encoded_history, model_artifact.feature_columns, model_artifact.sequence_window)
                prediction = float(model_artifact.predict_sequence(sequence)[0])
            else:
                feature_row["aqi_target"] = np.nan
                encoded_future, _, _ = prepare_model_frame(feature_row)
                aligned = align_columns(encoded_future, model_artifact.feature_columns).tail(1).fillna(0.0)
                prediction = float(model_artifact.predict(aligned)[0])
            prediction = float(np.clip(prediction, 1.0, 500.0))
            error_scale = float(model_artifact.hyperparameters.get("residual_p90", 15.0))
            if error_scale <= 0.0:
                error_scale = 15.0
            points.append(
                ForecastPoint(
                    city_id=city.city_id,
                    timestamp=future_timestamp.to_pydatetime(),
                    aqi=prediction,
                    lower_bound=max(1.0, prediction - error_scale),
                    upper_bound=min(500.0, prediction + error_scale),
                    hazard_flag=prediction >= 200.0,
                    model_name=model_artifact.model_name,
                )
            )
            forecast_row = _feature_row_to_raw(feature_row.iloc[0])
            forecast_row["aqi"] = prediction
            forecast_row["source"] = "forecast"
            current_history = pd.concat([current_history, pd.DataFrame([forecast_row])], ignore_index=True)
        return points
    except Exception as exc:
        LOGGER.exception("Recursive forecast failed")
        raise RuntimeError(f"Recursive forecast failed: {exc}") from exc


def recursive_backtest(
    model_artifact: ModelArtifact,
    history_frame: pd.DataFrame,
    actual_frame: pd.DataFrame,
    city: CityProfile,
    horizon_hours: int = 72,
) -> RecursiveBacktestResult:
    try:
        if history_frame.empty:
            raise ValueError("History frame is empty")
        if actual_frame.empty:
            raise ValueError("Actual frame is empty")
        ordered_actual = actual_frame.copy()
        ordered_actual["timestamp"] = pd.to_datetime(ordered_actual["timestamp"], utc=True)
        ordered_actual = ordered_actual.sort_values("timestamp").reset_index(drop=True)
        horizon = min(int(horizon_hours), len(ordered_actual))
        forecast_points = recursive_forecast(
            model_artifact=model_artifact,
            history_frame=history_frame,
            city=city,
            horizon_hours=horizon,
        )
        forecast_frame = pd.DataFrame([point.to_dict() for point in forecast_points])
        forecast_frame["timestamp"] = pd.to_datetime(forecast_frame["timestamp"], utc=True)
        actual_slice = ordered_actual.head(horizon).reset_index(drop=True)
        forecast_frame = forecast_frame.head(len(actual_slice)).reset_index(drop=True)
        frame = pd.DataFrame(
            {
                "timestamp": forecast_frame["timestamp"],
                "actual": actual_slice["aqi"].astype(float).to_numpy(),
                "predicted": forecast_frame["aqi"].astype(float).to_numpy(),
            }
        )
        frame["residual"] = frame["actual"] - frame["predicted"]
        frame["abs_residual"] = frame["residual"].abs()
        frame["day_number"] = (np.arange(len(frame)) // 24) + 1
        overall = _frame_evaluation(frame)
        daily: dict[str, ModelEvaluation] = {}
        for day_number, day_frame in frame.groupby("day_number", sort=True):
            if int(day_number) > 3 or day_frame.empty:
                continue
            daily[f"day_{int(day_number)}"] = _frame_evaluation(day_frame)
        error_scale = float(np.quantile(frame["abs_residual"].to_numpy(dtype=float), 0.9))
        return RecursiveBacktestResult(overall=overall, daily=daily, frame=frame, error_scale=error_scale)
    except Exception as exc:
        LOGGER.exception("Recursive backtest failed")
        raise RuntimeError(f"Recursive backtest failed: {exc}") from exc


def _frame_evaluation(frame: pd.DataFrame) -> ModelEvaluation:
    return compute_evaluation(frame["actual"].to_numpy(dtype=float), frame["predicted"].to_numpy(dtype=float))


def _feature_row_to_raw(row: pd.Series) -> dict[str, Any]:
    return {
        "city_id": row["city_id"],
        "timestamp": row["timestamp"],
        "aqi": float(row.get("aqi_lag_1", row.get("aqi", 0.0))),
        "temperature": float(row["temperature"]),
        "humidity": float(row["humidity"]),
        "wind_speed": float(row["wind_speed"]),
        "pressure": float(row["pressure"]),
        "rainfall": float(row["rainfall"]),
        "pm25": float(row["pm25"]),
        "pm10": float(row["pm10"]),
        "co": float(row["co"]),
        "no2": float(row["no2"]),
        "so2": float(row["so2"]),
        "o3": float(row["o3"]),
        "source": "forecast",
    }


def _build_sequence_input(frame: pd.DataFrame, feature_columns: list[str], window_size: int) -> np.ndarray:
    try:
        if frame.empty:
            raise ValueError("Encoded frame is empty")
        values = frame[feature_columns].astype(float).to_numpy()
        if len(values) < window_size:
            pad = np.repeat(values[:1], window_size - len(values), axis=0)
            values = np.vstack([pad, values])
        sequence = values[-window_size:]
        return sequence[np.newaxis, :, :]
    except Exception as exc:
        LOGGER.exception("Failed to build sequence input")
        raise RuntimeError(f"Failed to build sequence input: {exc}") from exc
