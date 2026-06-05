from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aqi_predictor.configs.settings import AppConfig
from aqi_predictor.feature_store.base import FeatureStoreBackend
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ReportArtifacts:
    markdown_path: Path
    json_path: Path


def _latest_evaluation_file(report_dir: Path) -> Path:
    candidates = sorted(report_dir.glob("evaluation_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation report found in {report_dir}")
    return candidates[-1]


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.exception("Failed to read JSON report")
        raise RuntimeError(f"Failed to read JSON report {path}: {exc}") from exc


def generate_final_report(config: AppConfig, store: FeatureStoreBackend) -> ReportArtifacts:
    try:
        config.report_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = _latest_evaluation_file(config.report_dir)
        evaluation = _safe_read_json(evaluation_path)
        model_metadata = store.load_latest_champion_model()
        report_timestamp = datetime.now(timezone.utc).isoformat()
        best_model_name = model_metadata["model_name"]
        best_metrics = model_metadata["metrics"]
        champion_eval = evaluation.get(best_model_name, {}) if isinstance(evaluation, dict) else {}
        champion_recursive = champion_eval.get("recursive", best_metrics) if isinstance(champion_eval, dict) else best_metrics
        daily_recursive = champion_eval.get("daily_recursive", {}) if isinstance(champion_eval, dict) else {}

        comparison_rows: list[dict[str, float | str]] = []
        for model_name, metric_values in evaluation.items():
            model_metrics = metric_values.get("recursive", metric_values) if isinstance(metric_values, dict) else metric_values
            comparison_rows.append(
                {
                    "model_name": model_name,
                    "rmse": float(model_metrics.get("rmse", 0.0)),
                    "mae": float(model_metrics.get("mae", 0.0)),
                    "r2": float(model_metrics.get("r2", 0.0)),
                    "mape": float(model_metrics.get("mape", 0.0)),
                }
            )
        comparison_rows = sorted(comparison_rows, key=lambda row: float(row["rmse"]))

        comparison_table = [
            "| Rank | Model | RMSE | MAE | R2 | MAPE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for index, row in enumerate(comparison_rows, start=1):
            comparison_table.append(
                f"| {index} | {row['model_name']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} | {row['mape']:.4f} |"
            )

        horizon_lines = []
        for day_name in ("day_1", "day_2", "day_3"):
            if day_name in daily_recursive:
                metrics = daily_recursive[day_name]
                horizon_lines.append(
                    f"- {day_name.upper()}: RMSE {float(metrics.get('rmse', 0.0)):.4f}, "
                    f"MAE {float(metrics.get('mae', 0.0)):.4f}, "
                    f"R2 {float(metrics.get('r2', 0.0)):.4f}, "
                    f"MAPE {float(metrics.get('mape', 0.0)):.4f}"
                )
        horizon_section = "\n".join(horizon_lines) if horizon_lines else "- No day-wise backtest summary available."

        markdown = f"""# Pearls AQI Predictor Final Report

Generated at: {report_timestamp}

## Champion Model

- Name: {best_model_name}
- Version: {model_metadata['model_version']}
- Family: {model_metadata['model_family']}
- Scope: {model_metadata['city_scope']}
- Artifact: {model_metadata['artifact_path']}

## Champion Metrics

- RMSE: {float(champion_recursive.get('rmse', 0.0)):.4f}
- MAE: {float(champion_recursive.get('mae', 0.0)):.4f}
- R2: {float(champion_recursive.get('r2', 0.0)):.4f}
- MAPE: {float(champion_recursive.get('mape', 0.0)):.4f}

## Horizon Checks

{horizon_section}

## Model Comparison

{chr(10).join(comparison_table)}

## Evaluation Assets

- Evaluation JSON: `{evaluation_path.name}`
- Prediction vs Actual plot: `pred_vs_actual_*.png`
- Residual plot: `residuals_*.png`

## Operational Notes

- Raw data is stored immutably in the feature store backend.
- Features are versioned and materialized before training.
- The champion model is registered with metadata and used by the prediction service.
- Vertex AI is the primary production backend, with local SQLite used for development.
"""
        markdown_path = config.report_dir / "final_report.md"
        json_path = config.report_dir / "final_report.json"
        markdown_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "generated_at": report_timestamp,
                    "evaluation_file": evaluation_path.name,
                    "champion": model_metadata,
                    "comparison": comparison_rows,
                    "champion_recursive": champion_recursive,
                    "daily_recursive": daily_recursive,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ReportArtifacts(markdown_path=markdown_path, json_path=json_path)
    except Exception as exc:
        LOGGER.exception("Failed to generate final report")
        raise RuntimeError(f"Failed to generate final report: {exc}") from exc
