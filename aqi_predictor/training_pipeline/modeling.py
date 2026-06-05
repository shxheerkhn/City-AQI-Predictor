from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from aqi_predictor.feature_pipeline.engine import align_columns
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover
    keras = None
    layers = None

try:
    from google.cloud import storage
except Exception:  # pragma: no cover
    storage = None


@dataclass(slots=True)
class ModelEvaluation:
    rmse: float
    mae: float
    r2: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2, "mape": self.mape}


@dataclass(slots=True)
class ModelArtifact:
    model_name: str
    model_family: str
    feature_columns: list[str]
    hyperparameters: dict[str, Any]
    model_object: Any
    scaler: StandardScaler | None = None
    sequence_window: int = 24

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            if self.model_family in {"lstm", "gru"}:
                raise ValueError("Sequence models require predict_sequence")
            aligned = align_columns(frame, self.feature_columns)
            model_input = aligned.astype(float)
            return np.asarray(self.model_object.predict(model_input), dtype=float)
        except Exception as exc:
            LOGGER.exception("Tabular prediction failed")
            raise RuntimeError(f"Tabular prediction failed: {exc}") from exc

    def predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        try:
            if self.model_family not in {"lstm", "gru"}:
                raise ValueError("Only sequence models accept sequence input")
            return np.asarray(self.model_object.predict(sequence, verbose=0), dtype=float).reshape(-1)
        except Exception as exc:
            LOGGER.exception("Sequence prediction failed")
            raise RuntimeError(f"Sequence prediction failed: {exc}") from exc

    def save(self, artifact_path: Path) -> None:
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if self.model_family in {"lstm", "gru"}:
                self.model_object.save(str(artifact_path), include_optimizer=False)
            else:
                joblib.dump(self.model_object, artifact_path)
            metadata_path = artifact_path.with_suffix(".meta.json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_name": self.model_name,
                        "model_family": self.model_family,
                        "feature_columns": self.feature_columns,
                        "hyperparameters": self.hyperparameters,
                        "sequence_window": self.sequence_window,
                        "scaler": self.scaler is not None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            LOGGER.exception("Failed to save model artifact")
            raise RuntimeError(f"Failed to save model artifact: {exc}") from exc

    @classmethod
    def load(cls, artifact_path: str | Path) -> "ModelArtifact":
        try:
            local_artifact_path = cls._resolve_local_artifact_path(artifact_path)
            metadata_path = local_artifact_path.with_suffix(".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_family = metadata["model_family"]
            if model_family in {"lstm", "gru"}:
                model_object = keras.models.load_model(str(local_artifact_path), compile=False)
            else:
                model_object = joblib.load(local_artifact_path)
            return cls(
                model_name=metadata["model_name"],
                model_family=model_family,
                feature_columns=list(metadata["feature_columns"]),
                hyperparameters=dict(metadata["hyperparameters"]),
                model_object=model_object,
                scaler=None,
                sequence_window=int(metadata.get("sequence_window", 24)),
            )
        except Exception as exc:
            LOGGER.exception("Failed to load model artifact")
            raise RuntimeError(f"Failed to load model artifact: {exc}") from exc

    @staticmethod
    def _resolve_local_artifact_path(artifact_path: str | Path) -> Path:
        try:
            path_str = str(artifact_path)
            if path_str.startswith("gs://"):
                return ModelArtifact._download_gcs_artifact(path_str)
            return Path(artifact_path)
        except Exception as exc:
            LOGGER.exception("Failed to resolve local artifact path")
            raise RuntimeError(f"Failed to resolve local artifact path: {exc}") from exc

    @staticmethod
    def _download_gcs_artifact(gs_uri: str) -> Path:
        try:
            if storage is None:
                raise RuntimeError("google-cloud-storage is not available for GCS artifact download")
            parsed = urlparse(gs_uri)
            bucket_name = parsed.netloc
            blob_name = parsed.path.lstrip("/")
            if not bucket_name or not blob_name:
                raise ValueError(f"Invalid GCS URI: {gs_uri}")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            temp_dir = Path(tempfile.mkdtemp(prefix="pearls-aqi-artifact-"))
            local_model_path = temp_dir / Path(blob_name).name
            bucket.blob(blob_name).download_to_filename(str(local_model_path))
            meta_blob_name = str(Path(blob_name).with_suffix(".meta.json")).replace("\\", "/")
            bucket.blob(meta_blob_name).download_to_filename(str(local_model_path.with_suffix(".meta.json")))
            return local_model_path
        except Exception as exc:
            LOGGER.exception("Failed to download GCS artifact")
            raise RuntimeError(f"Failed to download GCS artifact: {exc}") from exc


def compute_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> ModelEvaluation:
    try:
        y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
        mse = float(mean_squared_error(y_true_arr, y_pred_arr))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 else 0.0
        mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
        return ModelEvaluation(rmse=rmse, mae=mae, r2=r2, mape=mape)
    except Exception as exc:
        LOGGER.exception("Failed to compute evaluation")
        raise RuntimeError(f"Failed to compute evaluation: {exc}") from exc


def build_sequence_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    try:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        timestamps: list[pd.Timestamp] = []
        for city_id, city_frame in frame.groupby("city_id"):
            ordered = city_frame.sort_values("timestamp").reset_index(drop=True)
            values = ordered[feature_columns].astype(float).to_numpy()
            y_values = ordered["aqi_target"].astype(float).to_numpy()
            for end_index in range(window_size, len(ordered)):
                sequence = values[end_index - window_size : end_index]
                target = y_values[end_index]
                if np.isnan(sequence).any() or np.isnan(target):
                    continue
                sequences.append(sequence)
                targets.append(float(target))
                timestamps.append(pd.Timestamp(ordered.loc[end_index, "timestamp"]))
        if not sequences:
            return np.empty((0, window_size, len(feature_columns))), np.empty((0,)), []
        return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), timestamps
    except Exception as exc:
        LOGGER.exception("Failed to build sequence dataset")
        raise RuntimeError(f"Failed to build sequence dataset: {exc}") from exc


class MeanPredictor:
    model_name = "mean_predictor"
    model_family = "baseline"

    def __init__(self) -> None:
        self.mean_value = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanPredictor":
        try:
            self.mean_value = float(pd.Series(y).mean())
            return self
        except Exception as exc:
            LOGGER.exception("Mean predictor fit failed")
            raise RuntimeError(f"Mean predictor fit failed: {exc}") from exc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return np.full(shape=(len(X),), fill_value=self.mean_value, dtype=float)
        except Exception as exc:
            LOGGER.exception("Mean predictor prediction failed")
            raise RuntimeError(f"Mean predictor prediction failed: {exc}") from exc


class PersistencePredictor:
    model_name = "persistence_model"
    model_family = "baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistencePredictor":
        try:
            return self
        except Exception as exc:
            LOGGER.exception("Persistence fit failed")
            raise RuntimeError(f"Persistence fit failed: {exc}") from exc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            if "aqi_lag_1" in X.columns:
                return pd.to_numeric(X["aqi_lag_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return np.zeros(len(X), dtype=float)
        except Exception as exc:
            LOGGER.exception("Persistence prediction failed")
            raise RuntimeError(f"Persistence prediction failed: {exc}") from exc


class SklearnRegressor:
    def __init__(self, model_name: str, model_family: str, estimator: Any) -> None:
        self.model_name = model_name
        self.model_family = model_family
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnRegressor":
        try:
            self.estimator.fit(X, y)
            return self
        except Exception as exc:
            LOGGER.exception("Sklearn fit failed for %s", self.model_name)
            raise RuntimeError(f"Sklearn fit failed for {self.model_name}: {exc}") from exc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return np.asarray(self.estimator.predict(X), dtype=float)
        except Exception as exc:
            LOGGER.exception("Sklearn prediction failed for %s", self.model_name)
            raise RuntimeError(f"Sklearn prediction failed for {self.model_name}: {exc}") from exc


class SequenceRegressor:
    def __init__(self, model_name: str, model_family: str, sequence_window: int = 24, hidden_units: int = 32) -> None:
        if keras is None or layers is None:
            raise RuntimeError("TensorFlow is not installed")
        self.model_name = model_name
        self.model_family = model_family
        self.sequence_window = sequence_window
        self.hidden_units = hidden_units
        self.model_object = self._build_model()

    def _build_model(self) -> Any:
        try:
            input_shape = (self.sequence_window, None)
            inputs = keras.Input(shape=(self.sequence_window, 1))
            x = inputs
            if self.model_family == "gru":
                x = layers.GRU(self.hidden_units, return_sequences=False)(x)
            else:
                x = layers.LSTM(self.hidden_units, return_sequences=False)(x)
            x = layers.Dense(self.hidden_units // 2, activation="relu")(x)
            outputs = layers.Dense(1, activation="linear")(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            return model
        except Exception as exc:
            LOGGER.exception("Failed to build sequence model")
            raise RuntimeError(f"Failed to build sequence model: {exc}") from exc

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SequenceRegressor":
        try:
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 3 and X.shape[-1] > 1:
                X = X.mean(axis=-1, keepdims=True)
            y = np.asarray(y, dtype=np.float32)
            callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)]
            self.model_object.fit(X, y, epochs=8, batch_size=32, verbose=0, callbacks=callbacks)
            return self
        except Exception as exc:
            LOGGER.exception("Sequence fit failed for %s", self.model_name)
            raise RuntimeError(f"Sequence fit failed for {self.model_name}: {exc}") from exc

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 3 and X.shape[-1] > 1:
                X = X.mean(axis=-1, keepdims=True)
            return np.asarray(self.model_object.predict(X, verbose=0), dtype=float).reshape(-1)
        except Exception as exc:
            LOGGER.exception("Sequence prediction failed for %s", self.model_name)
            raise RuntimeError(f"Sequence prediction failed for {self.model_name}: {exc}") from exc


def model_factory() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = [
        {"name": "mean_predictor", "family": "baseline", "builder": lambda: MeanPredictor(), "feature_scope": "all"},
        {"name": "persistence_model", "family": "baseline", "builder": lambda: PersistencePredictor(), "feature_scope": "lag"},
        {"name": "ridge_regression", "family": "classical", "builder": lambda: SklearnRegressor("ridge_regression", "classical", Ridge(alpha=1.0, random_state=42)), "feature_scope": "all"},
        {"name": "random_forest", "family": "classical", "builder": lambda: SklearnRegressor("random_forest", "classical", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)), "feature_scope": "all"},
        {"name": "xgboost", "family": "classical", "builder": lambda: SklearnRegressor("xgboost", "classical", XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)), "feature_scope": "all", "requires": "xgboost"},
        {"name": "prophet_style", "family": "statistical", "builder": lambda: SklearnRegressor("prophet_style", "statistical", LinearRegression()), "feature_scope": "time"},
        {"name": "arima_style", "family": "statistical", "builder": lambda: SklearnRegressor("arima_style", "statistical", LinearRegression()), "feature_scope": "lag"},
    ]
    if keras is not None and layers is not None:
        models.append({"name": "lstm", "family": "lstm", "builder": lambda: SequenceRegressor("lstm", "lstm", sequence_window=24), "feature_scope": "sequence"})
        models.append({"name": "gru", "family": "gru", "builder": lambda: SequenceRegressor("gru", "gru", sequence_window=24), "feature_scope": "sequence"})
    return models
