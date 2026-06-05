from __future__ import annotations

from aqi_predictor.configs.settings import load_config
from aqi_predictor.feature_store.factory import build_feature_store
from aqi_predictor.reporting.generate import generate_final_report
from aqi_predictor.utils.logging import configure_logging, get_logger


configure_logging()
LOGGER = get_logger(__name__)


def main() -> None:
    try:
        config = load_config()
        store = build_feature_store(config)
        artifacts = generate_final_report(config, store)
        LOGGER.info("Final report generated at %s and %s", artifacts.markdown_path, artifacts.json_path)
    except Exception as exc:
        LOGGER.exception("Final report generation failed")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
