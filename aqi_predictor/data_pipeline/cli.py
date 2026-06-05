from __future__ import annotations

from aqi_predictor.configs.settings import load_config
from aqi_predictor.data_pipeline.ingest import ingest_current_snapshot
from aqi_predictor.feature_pipeline.pipeline import materialize_all_cities
from aqi_predictor.feature_store.factory import build_feature_store
from aqi_predictor.utils.logging import configure_logging, get_logger


configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    try:
        config = load_config()
        store = build_feature_store(config)
        ingest_current_snapshot(config, store)
        materialize_all_cities(store)
        LOGGER.info("Hourly feature pipeline completed")
    except Exception as exc:
        LOGGER.exception("Hourly feature pipeline failed")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
