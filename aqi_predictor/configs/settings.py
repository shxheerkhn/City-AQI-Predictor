from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CityProfile:
    city_id: str
    country: str
    latitude: float
    longitude: float
    timezone_offset_hours: int = 0
    aqi_bias: float = 0.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CityProfile":
        return cls(
            city_id=str(payload["city_id"]),
            country=str(payload.get("country", "unknown")),
            latitude=float(payload.get("latitude", 0.0)),
            longitude=float(payload.get("longitude", 0.0)),
            timezone_offset_hours=int(payload.get("timezone_offset_hours", 0)),
            aqi_bias=float(payload.get("aqi_bias", 0.0)),
        )


def _load_env_file(env_path: Path | None = None) -> None:
    try:
        candidate_paths = [
            env_path,
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for candidate in candidate_paths:
            if candidate is None or not candidate.exists():
                continue
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = _normalize_env_value(value)
            break
    except Exception:
        # Environment loading is best-effort; callers still have explicit env vars.
        return


def _default_city_profiles() -> list["CityProfile"]:
    return [
        CityProfile(
            city_id="karachi",
            country="Pakistan",
            latitude=24.8607,
            longitude=67.0011,
            timezone_offset_hours=5,
            aqi_bias=12.0,
        )
    ]


def _normalize_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1].strip()
    return cleaned


@dataclass(slots=True)
class AppConfig:
    app_name: str = "pearls-aqi-predictor"
    environment: str = "local"
    data_dir: Path = field(default_factory=lambda: Path("artifacts"))
    database_path: Path = field(default_factory=lambda: Path("artifacts") / "aqi_store.sqlite3")
    model_dir: Path = field(default_factory=lambda: Path("artifacts") / "models")
    report_dir: Path = field(default_factory=lambda: Path("artifacts") / "reports")
    raw_dir: Path = field(default_factory=lambda: Path("artifacts") / "raw")
    processed_dir: Path = field(default_factory=lambda: Path("artifacts") / "processed")
    api_base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    api_access_token: str | None = None
    external_api_provider: str = "synthetic"
    external_api_key: str | None = None
    external_api_url: str | None = None
    aqicn_api_url: str = "http://api.waqi.info"
    aqicn_station_id: str | None = None
    feature_store_backend: str = "local"
    cloud_project_id: str | None = None
    vertex_location: str = "us-central1"
    vertex_project_name: str | None = None
    vertex_staging_bucket: str | None = None
    hopsworks_project_name: str | None = None
    hopsworks_api_key: str | None = None
    hopsworks_api_secret: str | None = None
    bigquery_dataset: str = "aqi_feature_store"
    forecast_horizon_hours: int = 72
    ingestion_frequency_hours: int = 1
    backfill_days: int = 60
    rmse_threshold: float = 50.0
    hazard_aqi_threshold: float = 200.0
    auto_bootstrap_demo: bool = False
    cities: list[CityProfile] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AppConfig":
        try:
            _load_env_file()
            data_dir = Path(os.getenv("AQI_DATA_DIR", "artifacts"))
            city_payload = os.getenv("AQI_CITIES_JSON", "").strip()
            if city_payload:
                parsed_cities = [CityProfile.from_mapping(item) for item in json.loads(city_payload)]
                cities = parsed_cities or _default_city_profiles()
            else:
                cities = _default_city_profiles()
            config = cls(
                app_name=os.getenv("AQI_APP_NAME", "pearls-aqi-predictor"),
                environment=os.getenv("APP_ENV", "local"),
                data_dir=data_dir,
                database_path=Path(os.getenv("AQI_DATABASE_PATH", str(data_dir / "aqi_store.sqlite3"))),
                model_dir=Path(os.getenv("AQI_MODEL_DIR", str(data_dir / "models"))),
                report_dir=Path(os.getenv("AQI_REPORT_DIR", str(data_dir / "reports"))),
                raw_dir=Path(os.getenv("AQI_RAW_DIR", str(data_dir / "raw"))),
                processed_dir=Path(os.getenv("AQI_PROCESSED_DIR", str(data_dir / "processed"))),
                api_base_url=os.getenv("AQI_API_BASE_URL", "http://127.0.0.1:8000"),
                api_key=os.getenv("API_ACCESS_TOKEN"),
                api_access_token=os.getenv("API_ACCESS_TOKEN"),
                external_api_provider=os.getenv("AQI_SOURCE_PROVIDER", "synthetic"),
                external_api_key=os.getenv("AQI_EXTERNAL_API_KEY"),
                external_api_url=os.getenv("AQI_EXTERNAL_API_URL"),
                aqicn_api_url=os.getenv("AQI_AQICN_API_URL", "http://api.waqi.info"),
                aqicn_station_id=os.getenv("AQI_AQICN_STATION_ID"),
                feature_store_backend=os.getenv("AQI_FEATURE_STORE_BACKEND", "local"),
                cloud_project_id=os.getenv("AQI_CLOUD_PROJECT_ID"),
                vertex_location=os.getenv("AQI_VERTEX_LOCATION", "us-central1"),
                vertex_project_name=os.getenv("AQI_VERTEX_PROJECT_NAME"),
                vertex_staging_bucket=os.getenv("AQI_VERTEX_STAGING_BUCKET"),
                hopsworks_project_name=os.getenv("AQI_HOPSWORKS_PROJECT_NAME"),
                hopsworks_api_key=os.getenv("AQI_HOPSWORKS_API_KEY"),
                hopsworks_api_secret=os.getenv("AQI_HOPSWORKS_API_SECRET"),
                bigquery_dataset=os.getenv("AQI_BIGQUERY_DATASET", "aqi_feature_store"),
                forecast_horizon_hours=int(os.getenv("AQI_FORECAST_HOURS", "72")),
                ingestion_frequency_hours=int(os.getenv("AQI_INGESTION_HOURS", "1")),
                backfill_days=int(os.getenv("AQI_BACKFILL_DAYS", "60")),
                rmse_threshold=float(os.getenv("AQI_RMSE_THRESHOLD", "50.0")),
                hazard_aqi_threshold=float(os.getenv("AQI_HAZARD_THRESHOLD", "200.0")),
                auto_bootstrap_demo=os.getenv("AQI_AUTO_BOOTSTRAP_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"},
                cities=cities,
            )
            config.ensure_directories()
            return config
        except Exception as exc:
            raise RuntimeError(f"Failed to load application settings: {exc}") from exc

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.model_dir,
            self.report_dir,
            self.raw_dir,
            self.processed_dir,
            self.database_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    return AppConfig.from_env()
