from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from aqi_predictor.configs.settings import AppConfig, CityProfile
from aqi_predictor.data_pipeline.ingest import backfill_cities, ingest_current_snapshot, select_data_source
from aqi_predictor.data_pipeline.sources import AQICNAQIDataSource
from aqi_predictor.feature_pipeline.engine import build_supervised_frame
from aqi_predictor.feature_pipeline.pipeline import materialize_all_cities
from aqi_predictor.feature_store.local_store import LocalFeatureStore
from aqi_predictor.monitoring.metrics import REGISTRY


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.config = AppConfig(
            data_dir=base,
            database_path=base / "store.sqlite3",
            model_dir=base / "models",
            report_dir=base / "reports",
            raw_dir=base / "raw",
            processed_dir=base / "processed",
            backfill_days=3,
            cities=[CityProfile("city_test", "testland", 24.0, 67.0, 5, 10.0)],
        )
        self.config.ensure_directories()
        self.store = LocalFeatureStore(self.config.database_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_backfill_and_materialize(self) -> None:
        source = select_data_source(self.config)
        frame = backfill_cities(self.config, self.store, source)
        self.assertFalse(frame.empty)
        features = materialize_all_cities(self.store)
        self.assertFalse(features.empty)
        raw = self.store.load_raw_measurements()
        supervised = build_supervised_frame(raw)
        self.assertIn("aqi_target", supervised.columns)
        self.assertGreater(len(supervised), 0)

    def test_incremental_ingest_appends_rows(self) -> None:
        source = select_data_source(self.config)
        first = ingest_current_snapshot(self.config, self.store, source)
        second = ingest_current_snapshot(self.config, self.store, source)
        self.assertFalse(first.empty)
        self.assertFalse(second.empty)
        raw = self.store.load_raw_measurements()
        self.assertGreaterEqual(len(raw), 1)

    def test_aqicn_source_parses_feed_payload(self) -> None:
        payload = {
            "status": "ok",
            "data": {
                "aqi": 87,
                "iaqi": {
                    "pm25": {"v": 42},
                    "pm10": {"v": 58},
                    "co": {"v": 0.8},
                    "no2": {"v": 12},
                    "so2": {"v": 4},
                    "o3": {"v": 19},
                    "t": {"v": 31},
                    "h": {"v": 64},
                    "w": {"v": 3.2},
                },
            },
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        with patch("requests.get", return_value=response):
            source = AQICNAQIDataSource(api_token="token", base_url="http://api.waqi.info")
            observation = source.fetch_observation(self.config.cities[0], datetime.now(timezone.utc))
        self.assertEqual(observation.aqi, 87.0)
        self.assertEqual(observation.pm25, 42.0)
        self.assertEqual(observation.source, "aqicn")

    def test_metrics_registry_tracks_requests(self) -> None:
        REGISTRY.record_request("/forecast", 200, 15.0)
        snapshot = REGISTRY.snapshot()
        self.assertGreaterEqual(snapshot["total_requests"], 1)
        self.assertIn("/forecast", snapshot["by_route"])


if __name__ == "__main__":
    unittest.main()
