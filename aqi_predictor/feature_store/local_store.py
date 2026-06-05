from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from aqi_predictor.feature_store.base import FeatureStoreBackend
from aqi_predictor.models.domain import AQIMeasurement, ModelMetadata
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


class LocalFeatureStore(FeatureStoreBackend):
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS raw_measurements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        city_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        city_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        feature_json TEXT NOT NULL,
                        target_aqi REAL NOT NULL,
                        feature_version TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_family TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        artifact_path TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        hyperparameters_json TEXT NOT NULL,
                        feature_columns_json TEXT NOT NULL,
                        trained_at TEXT NOT NULL,
                        city_scope TEXT NOT NULL,
                        is_champion INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        city_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_family TEXT NOT NULL,
                        model_version TEXT UNIQUE NOT NULL,
                        artifact_path TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        hyperparameters_json TEXT NOT NULL,
                        feature_columns_json TEXT NOT NULL,
                        trained_at TEXT NOT NULL,
                        city_scope TEXT NOT NULL,
                        is_champion INTEGER NOT NULL DEFAULT 0,
                        rank_position INTEGER,
                        registered_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_features_city_ts ON features(city_id, timestamp)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_raw_city_ts ON raw_measurements(city_id, timestamp)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_models_champion ON models(is_champion, trained_at)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_model_history_champion ON model_history(is_champion, trained_at)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_model_history_version ON model_history(model_version)")
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_measurements_city_ts_source ON raw_measurements(city_id, timestamp, source)"
                )
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_features_city_ts_version ON features(city_id, timestamp, feature_version)"
                )
        except Exception as exc:
            LOGGER.exception("Failed to initialize local feature store")
            raise RuntimeError(f"Failed to initialize local feature store: {exc}") from exc

    def save_raw_measurements(self, measurements: list[AQIMeasurement]) -> None:
        try:
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO raw_measurements(city_id, timestamp, source, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            measurement.city_id,
                            measurement.timestamp.isoformat(),
                            measurement.source,
                            json.dumps(measurement.to_dict(), ensure_ascii=True),
                            measurement.created_at.isoformat(),
                        )
                        for measurement in measurements
                    ],
                )
        except Exception as exc:
            LOGGER.exception("Failed to save raw measurements")
            raise RuntimeError(f"Failed to save raw measurements: {exc}") from exc

    def save_feature_frame(self, frame: pd.DataFrame, feature_version: str) -> None:
        try:
            required_columns = {"city_id", "timestamp", "aqi_target"}
            if not required_columns.issubset(frame.columns):
                raise ValueError(f"Feature frame missing columns: {sorted(required_columns - set(frame.columns))}")
            records: list[tuple[Any, ...]] = []
            for _, row in frame.iterrows():
                feature_payload = {
                    key: self._json_safe_value(value)
                    for key, value in row.items()
                    if key not in {"city_id", "timestamp", "aqi_target"}
                }
                records.append(
                    (
                        str(row["city_id"]),
                        pd.Timestamp(row["timestamp"]).isoformat(),
                        json.dumps(feature_payload, ensure_ascii=True),
                        float(row["aqi_target"]),
                        feature_version,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO features(city_id, timestamp, feature_json, target_aqi, feature_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
        except Exception as exc:
            LOGGER.exception("Failed to save feature frame")
            raise RuntimeError(f"Failed to save feature frame: {exc}") from exc

    def load_feature_frame(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            query = "SELECT city_id, timestamp, feature_json, target_aqi, feature_version, created_at FROM features WHERE 1=1"
            params: list[Any] = []
            if city_id is not None:
                query += " AND city_id = ?"
                params.append(city_id)
            if start is not None:
                query += " AND timestamp >= ?"
                params.append(start.isoformat())
            if end is not None:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())
            query += " ORDER BY city_id, timestamp"
            with self._connect() as connection:
                rows = connection.execute(query, params).fetchall()
            records: list[dict[str, Any]] = []
            for row in rows:
                feature_payload = json.loads(row["feature_json"])
                record = {
                    "city_id": row["city_id"],
                    "timestamp": pd.Timestamp(row["timestamp"]),
                    "aqi_target": float(row["target_aqi"]),
                    "feature_version": row["feature_version"],
                    "created_at": pd.Timestamp(row["created_at"]),
                }
                record.update(feature_payload)
                records.append(record)
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to load feature frame")
            raise RuntimeError(f"Failed to load feature frame: {exc}") from exc

    def load_latest_features(self, city_id: str) -> pd.DataFrame:
        try:
            frame = self.load_feature_frame(city_id=city_id)
            if frame.empty:
                return frame
            return frame.sort_values("timestamp").tail(1).reset_index(drop=True)
        except Exception as exc:
            LOGGER.exception("Failed to load latest features")
            raise RuntimeError(f"Failed to load latest features: {exc}") from exc

    def load_raw_measurements(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            query = "SELECT city_id, timestamp, source, payload_json, created_at FROM raw_measurements WHERE 1=1"
            params: list[Any] = []
            if city_id is not None:
                query += " AND city_id = ?"
                params.append(city_id)
            if start is not None:
                query += " AND timestamp >= ?"
                params.append(start.isoformat())
            if end is not None:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())
            query += " ORDER BY city_id, timestamp"
            with self._connect() as connection:
                rows = connection.execute(query, params).fetchall()
            records: list[dict[str, Any]] = []
            for row in rows:
                payload = json.loads(row["payload_json"])
                payload["timestamp"] = pd.Timestamp(payload["timestamp"])
                payload["created_at"] = pd.Timestamp(payload["created_at"])
                records.append(payload)
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to load raw measurements")
            raise RuntimeError(f"Failed to load raw measurements: {exc}") from exc

    def list_city_ids(self) -> list[str]:
        try:
            with self._connect() as connection:
                rows = connection.execute("SELECT DISTINCT city_id FROM features ORDER BY city_id").fetchall()
            if rows:
                return [str(row["city_id"]) for row in rows]
            with self._connect() as connection:
                raw_rows = connection.execute("SELECT DISTINCT city_id FROM raw_measurements ORDER BY city_id").fetchall()
            return [str(row["city_id"]) for row in raw_rows]
        except Exception as exc:
            LOGGER.exception("Failed to list city ids")
            raise RuntimeError(f"Failed to list city ids: {exc}") from exc

    def register_model(self, metadata: ModelMetadata) -> None:
        try:
            with self._connect() as connection:
                if metadata.is_champion:
                    connection.execute("UPDATE models SET is_champion = 0")
                connection.execute(
                    """
                    INSERT INTO models(
                        model_name, model_family, model_version, artifact_path, metrics_json,
                        hyperparameters_json, feature_columns_json, trained_at,
                        city_scope, is_champion
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.model_name,
                        metadata.model_family,
                        metadata.model_version,
                        metadata.artifact_path,
                        json.dumps(metadata.metrics, ensure_ascii=True),
                        json.dumps(metadata.hyperparameters, ensure_ascii=True),
                        json.dumps(metadata.feature_columns, ensure_ascii=True),
                        metadata.trained_at.isoformat(),
                        metadata.city_scope,
                        1 if metadata.is_champion else 0,
                    ),
                )
        except Exception as exc:
            LOGGER.exception("Failed to register model")
            raise RuntimeError(f"Failed to register model: {exc}") from exc

    def load_latest_champion_model(self) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT model_name, model_family, model_version, artifact_path, metrics_json,
                           hyperparameters_json, feature_columns_json, trained_at,
                           city_scope, is_champion
                    FROM models
                    WHERE is_champion = 1
                    ORDER BY trained_at DESC
                    LIMIT 1
                    """
                ).fetchone()
            if row is None:
                with self._connect() as connection:
                    row = connection.execute(
                        """
                        SELECT model_name, model_family, model_version, artifact_path, metrics_json,
                               hyperparameters_json, feature_columns_json, trained_at,
                               city_scope, is_champion
                        FROM models
                        ORDER BY trained_at DESC
                        LIMIT 1
                        """
                    ).fetchone()
            if row is None:
                raise LookupError("No trained model registered")
            return {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "model_version": row["model_version"],
                "artifact_path": row["artifact_path"],
                "metrics": json.loads(row["metrics_json"]),
                "hyperparameters": json.loads(row["hyperparameters_json"]),
                "feature_columns": json.loads(row["feature_columns_json"]),
                "trained_at": row["trained_at"],
                "city_scope": row["city_scope"],
                "is_champion": bool(row["is_champion"]),
            }
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to load champion model metadata")
            raise RuntimeError(f"Failed to load champion model metadata: {exc}") from exc

    def record_prediction(
        self,
        city_id: str,
        timestamp: datetime,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        model_version: str,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO predictions(city_id, timestamp, request_json, response_json, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        city_id,
                        timestamp.isoformat(),
                        json.dumps(request_payload, ensure_ascii=True),
                        json.dumps(response_payload, ensure_ascii=True),
                        model_version,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as exc:
            LOGGER.exception("Failed to record prediction")
            raise RuntimeError(f"Failed to record prediction: {exc}") from exc

    def get_model_history(self, city_scope: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Get history of all trained models with metrics, ordered by trained_at DESC."""
        try:
            query = "SELECT * FROM model_history WHERE 1=1"
            params: list[Any] = []
            if city_scope is not None:
                query += " AND city_scope = ?"
                params.append(city_scope)
            query += " ORDER BY trained_at DESC LIMIT ?"
            params.append(limit)
            with self._connect() as connection:
                rows = connection.execute(query, params).fetchall()
            records: list[dict[str, Any]] = []
            for row in rows:
                records.append(
                    {
                        "id": row["id"],
                        "model_name": row["model_name"],
                        "model_family": row["model_family"],
                        "model_version": row["model_version"],
                        "artifact_path": row["artifact_path"],
                        "metrics": json.loads(row["metrics_json"]),
                        "hyperparameters": json.loads(row["hyperparameters_json"]),
                        "feature_columns": json.loads(row["feature_columns_json"]),
                        "trained_at": row["trained_at"],
                        "city_scope": row["city_scope"],
                        "is_champion": bool(row["is_champion"]),
                        "rank_position": row["rank_position"],
                        "registered_at": row["registered_at"],
                    }
                )
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to get model history")
            raise RuntimeError(f"Failed to get model history: {exc}") from exc

    def get_model_comparison(self, city_scope: str | None = None) -> pd.DataFrame:
        """Get model comparison table with metrics, sorted by RMSE (ascending)."""
        try:
            history_df = self.get_model_history(city_scope=city_scope, limit=1000)
            if history_df.empty:
                return pd.DataFrame()
            comparison_rows: list[dict[str, Any]] = []
            for _, row in history_df.iterrows():
                metrics = row["metrics"]
                comparison_rows.append(
                    {
                        "rank": row["rank_position"],
                        "model_name": row["model_name"],
                        "model_family": row["model_family"],
                        "model_version": row["model_version"],
                        "trained_at": row["trained_at"],
                        "is_champion": row["is_champion"],
                        "rmse": float(metrics.get("rmse", 0.0)),
                        "mae": float(metrics.get("mae", 0.0)),
                        "r2": float(metrics.get("r2", 0.0)),
                        "mape": float(metrics.get("mape", 0.0)),
                    }
                )
            comparison_df = pd.DataFrame(comparison_rows)
            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values("rmse")
            return comparison_df
        except Exception as exc:
            LOGGER.exception("Failed to get model comparison")
            raise RuntimeError(f"Failed to get model comparison: {exc}") from exc

    def get_model_by_version(self, model_version: str) -> dict[str, Any]:
        """Load a specific model version by version string."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM model_history WHERE model_version = ?",
                    (model_version,),
                ).fetchone()
            if row is None:
                raise LookupError(f"Model version {model_version} not found in history")
            return {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "model_version": row["model_version"],
                "artifact_path": row["artifact_path"],
                "metrics": json.loads(row["metrics_json"]),
                "hyperparameters": json.loads(row["hyperparameters_json"]),
                "feature_columns": json.loads(row["feature_columns_json"]),
                "trained_at": row["trained_at"],
                "city_scope": row["city_scope"],
                "is_champion": bool(row["is_champion"]),
                "rank_position": row["rank_position"],
            }
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception(f"Failed to get model by version {model_version}")
            raise RuntimeError(f"Failed to get model by version: {exc}") from exc

    def set_champion_model(self, model_version: str) -> None:
        """Mark a specific model version as champion (rollback)."""
        try:
            with self._connect() as connection:
                connection.execute("UPDATE model_history SET is_champion = 0")
                result = connection.execute(
                    "UPDATE model_history SET is_champion = 1 WHERE model_version = ?",
                    (model_version,),
                )
                if result.rowcount == 0:
                    raise LookupError(f"Model version {model_version} not found")
                connection.execute(
                    """
                    UPDATE models SET is_champion = 0
                    """
                )
                model_data = self.get_model_by_version(model_version)
                connection.execute(
                    """
                    INSERT INTO models(
                        model_name, model_family, model_version, artifact_path, metrics_json,
                        hyperparameters_json, feature_columns_json, trained_at,
                        city_scope, is_champion
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_data["model_name"],
                        model_data["model_family"],
                        model_data["model_version"],
                        model_data["artifact_path"],
                        json.dumps(model_data["metrics"]),
                        json.dumps(model_data["hyperparameters"]),
                        json.dumps(model_data["feature_columns"]),
                        model_data["trained_at"],
                        model_data["city_scope"],
                        1,
                    ),
                )
                LOGGER.info(f"Set model {model_version} as champion")
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception(f"Failed to set champion model {model_version}")
            raise RuntimeError(f"Failed to set champion model: {exc}") from exc

    def register_model_to_history(self, metadata: ModelMetadata, rank_position: int) -> None:
        """Register a model to the history table with rank position."""
        try:
            registered_at = datetime.now(timezone.utc).isoformat()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO model_history(
                        model_name, model_family, model_version, artifact_path, metrics_json,
                        hyperparameters_json, feature_columns_json, trained_at,
                        city_scope, is_champion, rank_position, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.model_name,
                        metadata.model_family,
                        metadata.model_version,
                        metadata.artifact_path,
                        json.dumps(metadata.metrics, ensure_ascii=True),
                        json.dumps(metadata.hyperparameters, ensure_ascii=True),
                        json.dumps(metadata.feature_columns, ensure_ascii=True),
                        metadata.trained_at.isoformat(),
                        metadata.city_scope,
                        1 if metadata.is_champion else 0,
                        rank_position,
                        registered_at,
                    ),
                )
                LOGGER.info(f"Registered model {metadata.model_name} to history at rank {rank_position}")
        except Exception as exc:
            LOGGER.exception("Failed to register model to history")
            raise RuntimeError(f"Failed to register model to history: {exc}") from exc

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value).isoformat()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        if pd.isna(value):
            return None
        return value
