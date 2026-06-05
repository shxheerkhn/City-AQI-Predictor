from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from aqi_predictor.configs.settings import AppConfig
from aqi_predictor.feature_store.base import FeatureStoreBackend
from aqi_predictor.models.domain import AQIMeasurement, ModelMetadata
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)

try:
    from google.cloud import aiplatform
    from google.cloud import bigquery
    from google.cloud import storage
except Exception as exc:  # pragma: no cover
    aiplatform = None
    bigquery = None
    storage = None
    LOGGER.warning("Google Cloud SDK imports are unavailable: %s", exc)


@dataclass(slots=True)
class VertexResources:
    project_id: str
    location: str
    dataset_id: str
    bucket_name: str
    raw_table_id: str
    features_table_id: str
    models_table_id: str
    predictions_table_id: str


class VertexFeatureStore(FeatureStoreBackend):
    def __init__(self, resources: VertexResources) -> None:
        self.resources = resources
        self._bq_client = None
        self._storage_client = None
        self._initialized = False

    @classmethod
    def from_config(cls, config: AppConfig) -> "VertexFeatureStore":
        try:
            project_id = config.cloud_project_id or config.vertex_project_name
            if not project_id:
                raise ValueError("AQI_CLOUD_PROJECT_ID or AQI_VERTEX_PROJECT_NAME is required for Vertex backend")
            bucket_name = config.vertex_staging_bucket or f"{project_id}-pearls-aqi-artifacts"
            resources = VertexResources(
                project_id=project_id,
                location=config.vertex_location,
                dataset_id=config.bigquery_dataset,
                bucket_name=bucket_name,
                raw_table_id=f"{project_id}.{config.bigquery_dataset}.raw_measurements",
                features_table_id=f"{project_id}.{config.bigquery_dataset}.features",
                models_table_id=f"{project_id}.{config.bigquery_dataset}.models",
                predictions_table_id=f"{project_id}.{config.bigquery_dataset}.predictions",
            )
            store = cls(resources)
            store._initialize()
            return store
        except Exception as exc:
            LOGGER.exception("Failed to initialize Vertex feature store")
            raise RuntimeError(f"Failed to initialize Vertex feature store: {exc}") from exc

    @property
    def bq_client(self):
        if self._bq_client is None:
            if bigquery is None:
                raise RuntimeError("google-cloud-bigquery is not available")
            self._bq_client = bigquery.Client(project=self.resources.project_id)
        return self._bq_client

    @property
    def storage_client(self):
        if self._storage_client is None:
            if storage is None:
                raise RuntimeError("google-cloud-storage is not available")
            self._storage_client = storage.Client(project=self.resources.project_id)
        return self._storage_client

    def _initialize(self) -> None:
        if self._initialized:
            return
        try:
            self._ensure_dataset()
            self._ensure_tables()
            self._ensure_bucket()
            self._initialized = True
        except Exception as exc:
            LOGGER.exception("Vertex feature store initialization failed")
            raise RuntimeError(f"Vertex feature store initialization failed: {exc}") from exc

    def _ensure_dataset(self) -> None:
        try:
            dataset_ref = bigquery.Dataset(f"{self.resources.project_id}.{self.resources.dataset_id}")
            dataset_ref.location = self.resources.location
            self.bq_client.create_dataset(dataset_ref, exists_ok=True)
        except Exception as exc:
            LOGGER.exception("Failed to ensure BigQuery dataset")
            raise RuntimeError(f"Failed to ensure BigQuery dataset: {exc}") from exc

    def _ensure_tables(self) -> None:
        try:
            schema_map = {
                self.resources.raw_table_id: [
                    bigquery.SchemaField("city_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("payload_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                ],
                self.resources.features_table_id: [
                    bigquery.SchemaField("city_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("feature_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("target_aqi", "FLOAT", mode="REQUIRED"),
                    bigquery.SchemaField("feature_version", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                ],
                self.resources.models_table_id: [
                    bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("model_family", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("artifact_path", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("metrics_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("hyperparameters_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("feature_columns_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("trained_at", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("city_scope", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("is_champion", "BOOL", mode="REQUIRED"),
                ],
                self.resources.predictions_table_id: [
                    bigquery.SchemaField("city_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("request_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("response_json", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                ],
            }
            for table_id, schema in schema_map.items():
                table = bigquery.Table(table_id, schema=schema)
                self.bq_client.create_table(table, exists_ok=True)
        except Exception as exc:
            LOGGER.exception("Failed to ensure BigQuery tables")
            raise RuntimeError(f"Failed to ensure BigQuery tables: {exc}") from exc

    def _ensure_bucket(self) -> None:
        try:
            bucket = self.storage_client.bucket(self.resources.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(bucket, project=self.resources.project_id, location=self.resources.location)
            bucket.storage_class = "STANDARD"
        except Exception as exc:
            LOGGER.exception("Failed to ensure GCS bucket")
            raise RuntimeError(f"Failed to ensure GCS bucket: {exc}") from exc

    def save_raw_measurements(self, measurements: list[AQIMeasurement]) -> None:
        try:
            rows = [
                {
                    "city_id": measurement.city_id,
                    "timestamp": measurement.timestamp.isoformat(),
                    "source": measurement.source,
                    "payload_json": json.dumps(measurement.to_dict(), ensure_ascii=True),
                    "created_at": measurement.created_at.isoformat(),
                }
                for measurement in measurements
            ]
            self._insert_rows(self.resources.raw_table_id, rows)
        except Exception as exc:
            LOGGER.exception("Failed to save raw measurements to Vertex")
            raise RuntimeError(f"Failed to save raw measurements to Vertex: {exc}") from exc

    def save_feature_frame(self, frame: pd.DataFrame, feature_version: str) -> None:
        try:
            required_columns = {"city_id", "timestamp", "aqi_target"}
            if not required_columns.issubset(frame.columns):
                raise ValueError(f"Feature frame missing columns: {sorted(required_columns - set(frame.columns))}")
            rows: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                feature_payload = {
                    key: self._json_safe_value(value)
                    for key, value in row.items()
                    if key not in {"city_id", "timestamp", "aqi_target"}
                }
                rows.append(
                    {
                        "city_id": str(row["city_id"]),
                        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                        "feature_json": json.dumps(feature_payload, ensure_ascii=True),
                        "target_aqi": float(row["aqi_target"]),
                        "feature_version": feature_version,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            self._insert_rows(self.resources.features_table_id, rows)
        except Exception as exc:
            LOGGER.exception("Failed to save feature frame to Vertex")
            raise RuntimeError(f"Failed to save feature frame to Vertex: {exc}") from exc

    def load_feature_frame(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            query = f"SELECT city_id, timestamp, feature_json, target_aqi, feature_version, created_at FROM `{self.resources.features_table_id}` WHERE 1=1"
            params = []
            if city_id is not None:
                query += " AND city_id = @city_id"
                params.append(bigquery.ScalarQueryParameter("city_id", "STRING", city_id))
            if start is not None:
                query += " AND timestamp >= @start"
                params.append(bigquery.ScalarQueryParameter("start", "TIMESTAMP", start.isoformat()))
            if end is not None:
                query += " AND timestamp <= @end"
                params.append(bigquery.ScalarQueryParameter("end", "TIMESTAMP", end.isoformat()))
            query += " ORDER BY city_id, timestamp"
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            rows = self.bq_client.query(query, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
            return self._expand_feature_rows(rows)
        except Exception as exc:
            LOGGER.exception("Failed to load feature frame from Vertex")
            raise RuntimeError(f"Failed to load feature frame from Vertex: {exc}") from exc

    def load_raw_measurements(
        self,
        city_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            query = f"SELECT city_id, timestamp, source, payload_json, created_at FROM `{self.resources.raw_table_id}` WHERE 1=1"
            params = []
            if city_id is not None:
                query += " AND city_id = @city_id"
                params.append(bigquery.ScalarQueryParameter("city_id", "STRING", city_id))
            if start is not None:
                query += " AND timestamp >= @start"
                params.append(bigquery.ScalarQueryParameter("start", "TIMESTAMP", start.isoformat()))
            if end is not None:
                query += " AND timestamp <= @end"
                params.append(bigquery.ScalarQueryParameter("end", "TIMESTAMP", end.isoformat()))
            query += " ORDER BY city_id, timestamp"
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            rows = self.bq_client.query(query, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
            records: list[dict[str, Any]] = []
            for _, row in rows.iterrows():
                payload = json.loads(row["payload_json"])
                payload["timestamp"] = pd.Timestamp(payload["timestamp"])
                payload["created_at"] = pd.Timestamp(payload["created_at"])
                records.append(payload)
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to load raw measurements from Vertex")
            raise RuntimeError(f"Failed to load raw measurements from Vertex: {exc}") from exc

    def list_city_ids(self) -> list[str]:
        try:
            query = f"""
                SELECT DISTINCT city_id
                FROM `{self.resources.features_table_id}`
                ORDER BY city_id
            """
            rows = self.bq_client.query(query).result().to_dataframe(create_bqstorage_client=False)
            if not rows.empty:
                return rows["city_id"].astype(str).tolist()
            query = f"""
                SELECT DISTINCT city_id
                FROM `{self.resources.raw_table_id}`
                ORDER BY city_id
            """
            rows = self.bq_client.query(query).result().to_dataframe(create_bqstorage_client=False)
            return rows["city_id"].astype(str).tolist()
        except Exception as exc:
            LOGGER.exception("Failed to list city ids from Vertex")
            raise RuntimeError(f"Failed to list city ids from Vertex: {exc}") from exc

    def register_model(self, metadata: ModelMetadata) -> None:
        try:
            artifact_uri = self._upload_model_artifacts(Path(metadata.artifact_path), metadata)
            if aiplatform is not None:
                self._register_vertex_model(artifact_uri, metadata)
            row = {
                "model_name": metadata.model_name,
                "model_family": metadata.model_family,
                "model_version": metadata.model_version,
                "artifact_path": artifact_uri,
                "metrics_json": json.dumps(metadata.metrics, ensure_ascii=True),
                "hyperparameters_json": json.dumps(metadata.hyperparameters, ensure_ascii=True),
                "feature_columns_json": json.dumps(metadata.feature_columns, ensure_ascii=True),
                "trained_at": metadata.trained_at.isoformat(),
                "city_scope": metadata.city_scope,
                "is_champion": bool(metadata.is_champion),
            }
            self._insert_rows(self.resources.models_table_id, [row])
        except Exception as exc:
            LOGGER.exception("Failed to register model in Vertex")
            raise RuntimeError(f"Failed to register model in Vertex: {exc}") from exc

    def load_latest_champion_model(self) -> dict[str, Any]:
        try:
            query = f"""
                SELECT model_name, model_family, model_version, artifact_path, metrics_json,
                       hyperparameters_json, feature_columns_json, trained_at, city_scope, is_champion
                FROM `{self.resources.models_table_id}`
                WHERE is_champion = TRUE
                ORDER BY trained_at DESC
                LIMIT 1
            """
            rows = self.bq_client.query(query).result().to_dataframe(create_bqstorage_client=False)
            if rows.empty:
                query = f"""
                    SELECT model_name, model_family, model_version, artifact_path, metrics_json,
                           hyperparameters_json, feature_columns_json, trained_at, city_scope, is_champion
                    FROM `{self.resources.models_table_id}`
                    ORDER BY trained_at DESC
                    LIMIT 1
                """
                rows = self.bq_client.query(query).result().to_dataframe(create_bqstorage_client=False)
            if rows.empty:
                raise LookupError("No trained model registered")
            row = rows.iloc[0].to_dict()
            return {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "model_version": row["model_version"],
                "artifact_path": row["artifact_path"],
                "metrics": json.loads(row["metrics_json"]),
                "hyperparameters": json.loads(row["hyperparameters_json"]),
                "feature_columns": json.loads(row["feature_columns_json"]),
                "trained_at": pd.Timestamp(row["trained_at"]).isoformat(),
                "city_scope": row["city_scope"],
                "is_champion": bool(row["is_champion"]),
            }
        except Exception as exc:
            LOGGER.exception("Failed to load latest champion model from Vertex")
            raise RuntimeError(f"Failed to load latest champion model from Vertex: {exc}") from exc

    def record_prediction(
        self,
        city_id: str,
        timestamp: datetime,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        model_version: str,
    ) -> None:
        try:
            row = {
                "city_id": city_id,
                "timestamp": timestamp.isoformat(),
                "request_json": json.dumps(request_payload, ensure_ascii=True),
                "response_json": json.dumps(response_payload, ensure_ascii=True),
                "model_version": model_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._insert_rows(self.resources.predictions_table_id, [row])
        except Exception as exc:
            LOGGER.exception("Failed to record prediction in Vertex")
            raise RuntimeError(f"Failed to record prediction in Vertex: {exc}") from exc

    def _insert_rows(self, table_id: str, rows: list[dict[str, Any]]) -> None:
        try:
            errors = self.bq_client.insert_rows_json(table_id, rows)
            if errors:
                raise RuntimeError(f"BigQuery insert errors: {errors}")
        except Exception as exc:
            LOGGER.exception("BigQuery insert failed for %s", table_id)
            raise RuntimeError(f"BigQuery insert failed for {table_id}: {exc}") from exc

    def _expand_feature_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        try:
            records: list[dict[str, Any]] = []
            for _, row in rows.iterrows():
                payload = json.loads(row["feature_json"])
                payload["timestamp"] = pd.Timestamp(row["timestamp"])
                payload["created_at"] = pd.Timestamp(row["created_at"])
                payload["city_id"] = row["city_id"]
                payload["aqi_target"] = float(row["target_aqi"])
                payload["feature_version"] = row["feature_version"]
                records.append(payload)
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to expand feature rows")
            raise RuntimeError(f"Failed to expand feature rows: {exc}") from exc

    def _upload_model_artifacts(self, artifact_path: Path, metadata: ModelMetadata) -> str:
        try:
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
            bucket = self.storage_client.bucket(self.resources.bucket_name)
            prefix = f"models/{metadata.model_name}/{metadata.model_version}"
            artifact_blob = bucket.blob(f"{prefix}/{artifact_path.name}")
            artifact_blob.upload_from_filename(str(artifact_path))
            meta_path = artifact_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta_blob = bucket.blob(f"{prefix}/{meta_path.name}")
                meta_blob.upload_from_filename(str(meta_path))
            return f"gs://{self.resources.bucket_name}/{prefix}/{artifact_path.name}"
        except Exception as exc:
            LOGGER.exception("Failed to upload model artifacts to GCS")
            raise RuntimeError(f"Failed to upload model artifacts to GCS: {exc}") from exc

    def _register_vertex_model(self, artifact_uri: str, metadata: ModelMetadata) -> None:
        try:
            aiplatform.init(
                project=self.resources.project_id,
                location=self.resources.location,
                staging_bucket=f"gs://{self.resources.bucket_name}",
            )
            serving_image = (
                "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
                if metadata.model_family in {"lstm", "gru"}
                else "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
            )
            aiplatform.Model.upload(
                display_name=f"{metadata.model_name}-{metadata.model_version}",
                artifact_uri=artifact_uri.rsplit("/", 1)[0],
                serving_container_image_uri=serving_image,
                description=f"Pearls AQI model {metadata.model_name} ({metadata.model_version})",
                labels={
                    "model_name": metadata.model_name,
                    "model_family": metadata.model_family,
                    "city_scope": metadata.city_scope,
                },
            )
        except Exception as exc:
            LOGGER.warning("Vertex model registry upload failed, continuing with BigQuery registry metadata: %s", exc)

    def get_model_history(self, city_scope: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Get history of all trained models with metrics from BigQuery."""
        try:
            query = f"SELECT * FROM `{self.resources.models_table_id}` WHERE 1=1"
            if city_scope is not None:
                query += f" AND city_scope = '{city_scope}'"
            query += " ORDER BY trained_at DESC"
            query += f" LIMIT {limit}"
            rows = self.bq_client.query(query).result().to_dataframe(create_bqstorage_client=False)
            if rows.empty:
                return pd.DataFrame()
            records: list[dict[str, Any]] = []
            for _, row in rows.iterrows():
                records.append(
                    {
                        "model_name": row["model_name"],
                        "model_family": row["model_family"],
                        "model_version": row["model_version"],
                        "artifact_path": row["artifact_path"],
                        "metrics": json.loads(row["metrics_json"]),
                        "trained_at": pd.Timestamp(row["trained_at"]).isoformat(),
                        "city_scope": row["city_scope"],
                        "is_champion": bool(row["is_champion"]),
                    }
                )
            return pd.DataFrame(records)
        except Exception as exc:
            LOGGER.exception("Failed to get model history from Vertex")
            raise RuntimeError(f"Failed to get model history from Vertex: {exc}") from exc

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
            LOGGER.exception("Failed to get model comparison from Vertex")
            raise RuntimeError(f"Failed to get model comparison from Vertex: {exc}") from exc

    def get_model_by_version(self, model_version: str) -> dict[str, Any]:
        """Load a specific model version by version string."""
        try:
            query = f"""
                SELECT model_name, model_family, model_version, artifact_path, metrics_json,
                       hyperparameters_json, feature_columns_json, trained_at, city_scope, is_champion
                FROM `{self.resources.models_table_id}`
                WHERE model_version = @version
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("version", "STRING", model_version)]
            )
            rows = self.bq_client.query(query, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
            if rows.empty:
                raise LookupError(f"Model version {model_version} not found")
            row = rows.iloc[0].to_dict()
            return {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "model_version": row["model_version"],
                "artifact_path": row["artifact_path"],
                "metrics": json.loads(row["metrics_json"]),
                "hyperparameters": json.loads(row["hyperparameters_json"]),
                "feature_columns": json.loads(row["feature_columns_json"]),
                "trained_at": pd.Timestamp(row["trained_at"]).isoformat(),
                "city_scope": row["city_scope"],
                "is_champion": bool(row["is_champion"]),
            }
        except LookupError:
            raise
        except Exception as exc:
            LOGGER.exception(f"Failed to get model by version {model_version}")
            raise RuntimeError(f"Failed to get model by version: {exc}") from exc

    def set_champion_model(self, model_version: str) -> None:
        """Mark a specific model version as champion (rollback)."""
        try:
            query = f"UPDATE `{self.resources.models_table_id}` SET is_champion = FALSE WHERE is_champion = TRUE"
            self.bq_client.query(query).result()
            query = f"UPDATE `{self.resources.models_table_id}` SET is_champion = TRUE WHERE model_version = @version"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("version", "STRING", model_version)]
            )
            self.bq_client.query(query, job_config=job_config).result()
            LOGGER.info(f"Set model {model_version} as champion in Vertex")
        except Exception as exc:
            LOGGER.exception(f"Failed to set champion model {model_version}")
            raise RuntimeError(f"Failed to set champion model: {exc}") from exc

    def register_model_to_history(self, metadata: ModelMetadata, rank_position: int) -> None:
        """Register a model to the history table with rank position (delegates to register_model for Vertex)."""
        self.register_model(metadata)

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
