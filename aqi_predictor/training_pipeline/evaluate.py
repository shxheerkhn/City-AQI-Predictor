from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from aqi_predictor.training_pipeline.modeling import ModelEvaluation
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    try:
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(residuals, bins=30, color="#0f766e", alpha=0.85)
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Count")
        axes[1].scatter(y_pred, residuals, s=10, alpha=0.6, color="#7c3aed")
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_title("Residuals vs Predictions")
        axes[1].set_xlabel("Predicted AQI")
        axes[1].set_ylabel("Residual")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception as exc:
        LOGGER.exception("Failed to plot residuals")
        raise RuntimeError(f"Failed to plot residuals: {exc}") from exc


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    try:
        series = pd.DataFrame({"actual": np.asarray(y_true, dtype=float), "predicted": np.asarray(y_pred, dtype=float)})
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(series["actual"], series["predicted"], alpha=0.6, color="#2563eb")
        max_value = float(max(series["actual"].max(), series["predicted"].max()))
        min_value = float(min(series["actual"].min(), series["predicted"].min()))
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
        ax.set_title("Prediction vs Actual")
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    except Exception as exc:
        LOGGER.exception("Failed to plot prediction vs actual")
        raise RuntimeError(f"Failed to plot prediction vs actual: {exc}") from exc


def build_evaluation_report(metrics: dict[str, Any], output_path: Path) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return output_path
    except Exception as exc:
        LOGGER.exception("Failed to build evaluation report")
        raise RuntimeError(f"Failed to build evaluation report: {exc}") from exc
