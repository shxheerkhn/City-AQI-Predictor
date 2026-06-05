from __future__ import annotations

from aqi_predictor.configs.settings import load_config
from aqi_predictor.data_pipeline.ingest import backfill_cities, ingest_current_snapshot, select_data_source
from aqi_predictor.feature_pipeline.pipeline import materialize_all_cities
from aqi_predictor.feature_store.factory import build_feature_store
from aqi_predictor.utils.logging import configure_logging, get_logger


configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    try:
        config = load_config()
        store = build_feature_store(config)
        source = select_data_source(config)
        if store.load_raw_measurements().empty:
            backfill_cities(config, store, source)
        else:
            ingest_current_snapshot(config, store, source)
        materialize_all_cities(store)
        LOGGER.info("Feature pipeline completed")
    except Exception as exc:
        LOGGER.exception("Feature pipeline failed")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
