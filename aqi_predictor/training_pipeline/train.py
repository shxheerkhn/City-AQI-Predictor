from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqi_predictor.configs.settings import AppConfig, CityProfile, load_config
from aqi_predictor.data_pipeline.ingest import backfill_cities, select_data_source
from aqi_predictor.feature_pipeline.engine import build_supervised_frame, prepare_model_frame
from aqi_predictor.feature_pipeline.pipeline import materialize_all_cities
from aqi_predictor.feature_store.factory import build_feature_store
from aqi_predictor.models.domain import ModelMetadata
from aqi_predictor.training_pipeline.evaluate import build_evaluation_report, plot_prediction_vs_actual, plot_residuals
from aqi_predictor.training_pipeline.modeling import (
    ModelArtifact,
    ModelEvaluation,
    SequenceRegressor,
    compute_evaluation,
    build_sequence_dataset,
    model_factory,
)
from aqi_predictor.training_pipeline.forecasting import RecursiveBacktestResult, recursive_backtest
from aqi_predictor.reporting.generate import generate_final_report
from aqi_predictor.analysis.reporter import generate_eda_for_features
from aqi_predictor.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainingOutcome:
    best_model_name: str
    best_metrics: ModelEvaluation
    champion_artifact: Path
    evaluation_path: Path


def _time_split_frame(frame: pd.DataFrame, validation_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ordered = frame.sort_values(["timestamp", "city_id"]).reset_index(drop=True)
        split_index = max(1, int(len(ordered) * (1.0 - validation_ratio)))
        train = ordered.iloc[:split_index].reset_index(drop=True)
        valid = ordered.iloc[split_index:].reset_index(drop=True)
        if valid.empty:
            valid = ordered.tail(max(1, len(ordered) // 5)).reset_index(drop=True)
        return train, valid
    except Exception as exc:
        LOGGER.exception("Failed to split data")
        raise RuntimeError(f"Failed to split data: {exc}") from exc


def _select_feature_subset(frame: pd.DataFrame, feature_scope: str) -> list[str]:
    try:
        columns = [column for column in frame.columns if column not in {"timestamp", "created_at", "aqi_target", "aqi"}]
        if feature_scope == "all":
            return columns
        if feature_scope == "time":
            prefixes = ("hour_", "dow_", "day_", "month", "quarter", "season_", "city_")
            return [column for column in columns if column.startswith(prefixes)]
        if feature_scope == "lag":
            prefixes = ("aqi_lag_", "aqi_roll_", "pm25_roll_", "city_")
            return [column for column in columns if column.startswith(prefixes)]
        return columns
    except Exception as exc:
        LOGGER.exception("Failed to select feature subset")
        raise RuntimeError(f"Failed to select feature subset: {exc}") from exc


def _resolve_city(config: AppConfig, city_id: str) -> CityProfile | None:
    for city in config.cities:
        if city.city_id == city_id:
            return city
    return None


def _build_recursive_backtest_summary(
    artifact: ModelArtifact,
    raw_measurements: pd.DataFrame,
    valid_start: pd.Timestamp,
    config: AppConfig,
) -> RecursiveBacktestResult | None:
    try:
        if raw_measurements.empty:
            return None
        city_frames: list[pd.DataFrame] = []
        for city_id, city_frame in raw_measurements.groupby("city_id"):
            ordered = city_frame.sort_values("timestamp").reset_index(drop=True)
            history = ordered[pd.to_datetime(ordered["timestamp"], utc=True) < valid_start].reset_index(drop=True)
            actual = ordered[pd.to_datetime(ordered["timestamp"], utc=True) >= valid_start].reset_index(drop=True)
            horizon = min(config.forecast_horizon_hours, len(actual))
            if history.empty or actual.empty or horizon < 1:
                continue
            city = _resolve_city(config, str(city_id))
            if city is None:
                continue
            backtest = recursive_backtest(
                artifact,
                history_frame=history,
                actual_frame=actual.head(horizon).reset_index(drop=True),
                city=city,
                horizon_hours=horizon,
            )
            frame = backtest.frame.copy()
            frame["city_id"] = city_id
            city_frames.append(frame)
        if not city_frames:
            return None
        combined = pd.concat(city_frames, ignore_index=True)
        combined["day_number"] = combined["day_number"].astype(int)
        overall = compute_evaluation(combined["actual"].to_numpy(), combined["predicted"].to_numpy())
        daily: dict[str, ModelEvaluation] = {}
        for day_number, day_frame in combined.groupby("day_number", sort=True):
            if int(day_number) > 3 or day_frame.empty:
                continue
            daily[f"day_{int(day_number)}"] = compute_evaluation(day_frame["actual"].to_numpy(), day_frame["predicted"].to_numpy())
        error_scale = float(np.quantile(np.abs(combined["actual"].to_numpy() - combined["predicted"].to_numpy()), 0.9))
        return RecursiveBacktestResult(overall=overall, daily=daily, frame=combined, error_scale=error_scale)
    except Exception as exc:
        LOGGER.exception("Failed to build recursive backtest summary")
        raise RuntimeError(f"Failed to build recursive backtest summary: {exc}") from exc


def _train_single_model(
    model_entry: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[ModelArtifact, ModelEvaluation, pd.DataFrame, np.ndarray]:
    try:
        model_name = model_entry["name"]
        feature_scope = model_entry["feature_scope"]
        builder = model_entry["builder"]
        if model_entry["family"] in {"lstm", "gru"}:
            combined_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
            dummied_all = pd.get_dummies(combined_frame, columns=["city_id", "season"], drop_first=False)
            combined_columns = sorted(dummied_all.columns)
            dummied_all = dummied_all.reindex(columns=combined_columns, fill_value=0.0)
            seq_features = [
                column
                for column in dummied_all.columns
                if column not in {"timestamp", "created_at", "aqi_target", "aqi", "source", "feature_version", "city_id"}
            ]
            dummied_all = dummied_all.assign(city_id=combined_frame["city_id"].values)
            all_sequences, all_targets, all_timestamps = build_sequence_dataset(dummied_all, seq_features, window_size=24)
            if len(all_sequences) == 0:
                raise ValueError(f"Insufficient sequence data for {model_name}")
            valid_start = pd.Timestamp(valid_frame["timestamp"].min())
            train_mask = np.asarray([timestamp < valid_start for timestamp in all_timestamps], dtype=bool)
            valid_mask = ~train_mask
            train_sequences, train_targets = all_sequences[train_mask], all_targets[train_mask]
            valid_sequences, valid_targets = all_sequences[valid_mask], all_targets[valid_mask]
            if len(train_sequences) == 0 or len(valid_sequences) == 0:
                raise ValueError(f"Insufficient sequence data for {model_name}")
            model = builder()
            model.fit(train_sequences, train_targets)
            valid_predictions = model.predict(valid_sequences)
            evaluation = compute_evaluation(valid_targets, valid_predictions)
            artifact = ModelArtifact(
                model_name=model_name,
                model_family=model_entry["family"],
                feature_columns=seq_features,
                hyperparameters={"window_size": 24},
                model_object=model.model_object,
                sequence_window=24,
            )
            prediction_frame = pd.DataFrame({"actual": valid_targets, "predicted": valid_predictions})
            return artifact, evaluation, prediction_frame, valid_predictions
        encoded_train, y_train, _ = prepare_model_frame(train_frame)
        encoded_valid, y_valid, _ = prepare_model_frame(valid_frame)
        feature_columns = _select_feature_subset(encoded_train, feature_scope)
        if not feature_columns:
            feature_columns = [column for column in encoded_train.columns]
        train_X = encoded_train[feature_columns].copy()
        valid_X = encoded_valid.reindex(columns=feature_columns, fill_value=0.0).copy()
        model = builder()
        model.fit(train_X, y_train)
        valid_predictions = model.predict(valid_X)
        evaluation = compute_evaluation(y_valid, valid_predictions)
        artifact = ModelArtifact(
            model_name=model_name,
            model_family=model_entry["family"],
            feature_columns=feature_columns,
            hyperparameters={},
            model_object=model.estimator if hasattr(model, "estimator") else model,
        )
        prediction_frame = pd.DataFrame({"actual": y_valid, "predicted": valid_predictions})
        return artifact, evaluation, prediction_frame, valid_predictions
    except Exception as exc:
        LOGGER.exception("Failed to train %s", model_entry.get("name"))
        raise RuntimeError(f"Failed to train {model_entry.get('name')}: {exc}") from exc


def run_training_pipeline(config: AppConfig | None = None) -> TrainingOutcome:
    try:
        configure_logging()
        config = config or load_config()
        store = build_feature_store(config)
        if not store.list_city_ids():
            LOGGER.info("No data found. Running synthetic backfill before training.")
            source = select_data_source(config)
            backfill_cities(config, store, source)
        materialize_all_cities(store)
        raw_measurements = store.load_raw_measurements()
        if raw_measurements.empty:
            raise ValueError("No raw measurements available for training")
        supervised = build_supervised_frame(raw_measurements)
        if supervised.empty:
            raise ValueError("Supervised dataset is empty after feature engineering")
        
        try:
            generate_eda_for_features(supervised, config.report_dir)
        except Exception as exc:
            LOGGER.warning(f"EDA generation failed, continuing with training: {exc}")
        
        train_frame, valid_frame = _time_split_frame(supervised, validation_ratio=0.2)
        valid_start = pd.Timestamp(valid_frame["timestamp"].min())
        metrics: dict[str, ModelEvaluation] = {}
        artifacts: list[tuple[str, ModelArtifact, ModelEvaluation, pd.DataFrame, np.ndarray, RecursiveBacktestResult | None]] = []
        best_entry: tuple[str, ModelArtifact, ModelEvaluation, pd.DataFrame, np.ndarray, RecursiveBacktestResult | None] | None = None
        for entry in model_factory():
            try:
                artifact, evaluation, prediction_frame, predictions = _train_single_model(entry, train_frame, valid_frame, config.model_dir)
                backtest = _build_recursive_backtest_summary(artifact, raw_measurements, valid_start, config)
                if backtest is None:
                    backtest = RecursiveBacktestResult(overall=evaluation, daily={}, frame=prediction_frame.rename(columns={"actual": "actual", "predicted": "predicted"}), error_scale=float(max(evaluation.rmse, 15.0)))
                metrics[entry["name"]] = backtest.overall
                artifacts.append((entry["name"], artifact, evaluation, prediction_frame, predictions, backtest))
                if best_entry is None or backtest.overall.rmse < best_entry[2].rmse:
                    best_entry = (entry["name"], artifact, backtest.overall, prediction_frame, predictions, backtest)
                LOGGER.info("Model %s achieved recursive RMSE %.4f", entry["name"], backtest.overall.rmse)
            except Exception as exc:
                LOGGER.warning("Skipping model %s due to error: %s", entry["name"], exc)
        if best_entry is None:
            raise RuntimeError("No model trained successfully")
        best_name, best_artifact, best_metrics, best_prediction_frame, best_predictions, best_backtest = best_entry
        model_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        artifact_path = config.model_dir / f"{best_name}_{model_version}.joblib"
        if best_artifact.model_family in {"lstm", "gru"}:
            artifact_path = artifact_path.with_suffix(".keras")
        best_artifact.hyperparameters = {
            **best_artifact.hyperparameters,
            "residual_p90": float(best_backtest.error_scale),
            "backtest_horizon_hours": int(config.forecast_horizon_hours),
        }
        best_artifact.save(artifact_path)
        metadata = ModelMetadata(
            model_name=best_artifact.model_name,
            model_family=best_artifact.model_family,
            model_version=model_version,
            artifact_path=str(artifact_path),
            metrics=best_metrics.to_dict(),
            hyperparameters=best_artifact.hyperparameters,
            feature_columns=best_artifact.feature_columns,
            trained_at=datetime.now(timezone.utc),
            city_scope="global",
            is_champion=True,
        )
        store.register_model(metadata)
        
        # Register all models to history table with ranking
        model_version_ts = datetime.now(timezone.utc)
        ranked_artifacts = sorted(
            [(name, artifact, eval, pred_frame, preds, backtest) for name, artifact, eval, pred_frame, preds, backtest in artifacts],
            key=lambda x: x[2].rmse
        )
        for rank_position, (model_name, artifact, evaluation, pred_frame, predictions, backtest) in enumerate(ranked_artifacts, start=1):
            try:
                history_metadata = ModelMetadata(
                    model_name=artifact.model_name,
                    model_family=artifact.model_family,
                    model_version=f"{model_version}_{model_name.lower().replace(' ', '_')}",
                    artifact_path=str(config.model_dir / f"{model_name}_{model_version}.joblib"),
                    metrics=evaluation.to_dict(),
                    hyperparameters=artifact.hyperparameters,
                    feature_columns=artifact.feature_columns,
                    trained_at=model_version_ts,
                    city_scope="global",
                    is_champion=(rank_position == 1),
                )
                store.register_model_to_history(history_metadata, rank_position)
            except Exception as exc:
                LOGGER.warning(f"Failed to register {model_name} to history: {exc}")
        
        evaluation_payload = {
            model_name: {
                "recursive": evaluation.to_dict(),
                "daily_recursive": (
                    {day: day_eval.to_dict() for day, day_eval in backtest.daily.items()}
                    if backtest is not None
                    else {}
                ),
            }
            for model_name, artifact_obj, evaluation, pred_frame, pred_vals, backtest in artifacts
        }
        evaluation_path = build_evaluation_report(evaluation_payload, config.report_dir / f"evaluation_{model_version}.json")
        plot_prediction_vs_actual(
            best_backtest.frame["actual"].to_numpy(),
            best_backtest.frame["predicted"].to_numpy(),
            config.report_dir / f"pred_vs_actual_{model_version}.png",
        )
        plot_residuals(
            best_backtest.frame["actual"].to_numpy(),
            best_backtest.frame["predicted"].to_numpy(),
            config.report_dir / f"residuals_{model_version}.png",
        )
        generate_final_report(config, store)
        return TrainingOutcome(
            best_model_name=best_name,
            best_metrics=best_metrics,
            champion_artifact=artifact_path,
            evaluation_path=evaluation_path,
        )
    except Exception as exc:
        LOGGER.exception("Training pipeline failed")
        raise RuntimeError(f"Training pipeline failed: {exc}") from exc


def main() -> None:
    try:
        outcome = run_training_pipeline()
        LOGGER.info(
            "Training complete. Champion=%s RMSE=%.4f MAE=%.4f R2=%.4f",
            outcome.best_model_name,
            outcome.best_metrics.rmse,
            outcome.best_metrics.mae,
            outcome.best_metrics.r2,
        )
    except Exception as exc:
        LOGGER.exception("Training entrypoint failed")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
