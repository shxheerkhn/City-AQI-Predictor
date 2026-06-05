from __future__ import annotations

from pathlib import Path

import pandas as pd

from aqi_predictor.analysis.eda import generate_eda_report
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


def generate_eda_for_features(feature_frame: pd.DataFrame, report_dir: Path) -> Path:
    """Generate EDA report for feature frame."""
    try:
        LOGGER.info("Generating EDA report for feature frame...")
        report_path = generate_eda_report(feature_frame, report_dir / "eda")
        LOGGER.info(f"EDA report generated at {report_path}")
        return report_path
    except Exception as exc:
        LOGGER.warning(f"Failed to generate EDA report: {exc}")
        raise RuntimeError(f"Failed to generate EDA report: {exc}") from exc
