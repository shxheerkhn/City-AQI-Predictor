from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from aqi_predictor.feature_pipeline.engine import align_columns
from aqi_predictor.training_pipeline.modeling import ModelArtifact
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


@dataclass(slots=True)
class ExplanationBundle:
    global_importance: pd.DataFrame
    local_summary: pd.DataFrame
    direction_summary: str


def global_feature_importance(model_artifact: ModelArtifact, frame: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    try:
        if frame.empty:
            return pd.DataFrame(columns=["feature", "importance"])
        if model_artifact.model_family in {"lstm", "gru"}:
            return pd.DataFrame(columns=["feature", "importance"])
        sample = frame.tail(sample_size).copy()
        sample = align_columns(sample, model_artifact.feature_columns)
        if shap is not None and model_artifact.model_family not in {"lstm", "gru"}:
            explainer = shap.Explainer(model_artifact.model_object, sample)
            values = explainer(sample)
            importance = np.abs(values.values).mean(axis=0)
            return pd.DataFrame({"feature": model_artifact.feature_columns, "importance": importance}).sort_values("importance", ascending=False)
        if hasattr(model_artifact.model_object, "feature_importances_"):
            importance = np.asarray(model_artifact.model_object.feature_importances_, dtype=float)
            return pd.DataFrame({"feature": model_artifact.feature_columns, "importance": importance}).sort_values("importance", ascending=False)
        if hasattr(model_artifact.model_object, "coef_"):
            importance = np.abs(np.asarray(model_artifact.model_object.coef_, dtype=float)).reshape(-1)
            return pd.DataFrame({"feature": model_artifact.feature_columns, "importance": importance}).sort_values("importance", ascending=False)
        result = permutation_importance(model_artifact.model_object, sample, np.repeat(frame["aqi_target"].tail(len(sample)).to_numpy(), 1), n_repeats=5, random_state=42)
        return pd.DataFrame({"feature": model_artifact.feature_columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)
    except Exception as exc:
        LOGGER.exception("Failed to compute global feature importance")
        raise RuntimeError(f"Failed to compute global feature importance: {exc}") from exc


def local_prediction_explanation(model_artifact: ModelArtifact, frame: pd.DataFrame, feature_row: pd.DataFrame) -> pd.DataFrame:
    try:
        if feature_row.empty:
            return pd.DataFrame(columns=["feature", "impact"])
        if model_artifact.model_family in {"lstm", "gru"}:
            return pd.DataFrame(columns=["feature", "impact"])
        aligned = align_columns(feature_row, model_artifact.feature_columns)
        baseline = align_columns(frame.tail(200), model_artifact.feature_columns)
        baseline_means = baseline.mean(numeric_only=True)
        row = aligned.iloc[0]
        if shap is not None and model_artifact.model_family not in {"lstm", "gru"}:
            explainer = shap.Explainer(model_artifact.model_object, baseline)
            values = explainer(aligned)
            impacts = values.values.reshape(-1)
            return pd.DataFrame({"feature": model_artifact.feature_columns, "impact": impacts}).sort_values("impact", key=lambda s: s.abs(), ascending=False)
        if hasattr(model_artifact.model_object, "coef_"):
            coefficients = np.asarray(model_artifact.model_object.coef_, dtype=float).reshape(-1)
            impacts = (row.to_numpy(dtype=float) - baseline_means.to_numpy(dtype=float)) * coefficients
            return pd.DataFrame({"feature": model_artifact.feature_columns, "impact": impacts}).sort_values("impact", key=lambda s: s.abs(), ascending=False)
        impacts = (row.to_numpy(dtype=float) - baseline_means.to_numpy(dtype=float)) * 0.05
        return pd.DataFrame({"feature": model_artifact.feature_columns, "impact": impacts}).sort_values("impact", key=lambda s: s.abs(), ascending=False)
    except Exception as exc:
        LOGGER.exception("Failed to compute local explanation")
        raise RuntimeError(f"Failed to compute local explanation: {exc}") from exc


def summarize_direction(explanation: pd.DataFrame, threshold: float = 0.0) -> str:
    try:
        if explanation.empty:
            return "No explanation available."
        positive = explanation[explanation["impact"] > threshold].head(3)
        negative = explanation[explanation["impact"] < -threshold].head(3)
        parts: list[str] = []
        if not positive.empty:
            parts.append(
                "Increased by "
                + ", ".join(f"{row.feature} (+{row.impact:.2f})" for row in positive.itertuples(index=False))
            )
        if not negative.empty:
            parts.append(
                "Decreased by "
                + ", ".join(f"{row.feature} ({row.impact:.2f})" for row in negative.itertuples(index=False))
            )
        return " | ".join(parts) if parts else "Prediction impact appears balanced."
    except Exception as exc:
        LOGGER.exception("Failed to summarize direction")
        raise RuntimeError(f"Failed to summarize direction: {exc}") from exc


def explain_prediction(model_artifact: ModelArtifact, frame: pd.DataFrame, feature_row: pd.DataFrame) -> ExplanationBundle:
    try:
        global_imp = global_feature_importance(model_artifact, frame)
        local_imp = local_prediction_explanation(model_artifact, frame, feature_row)
        direction = summarize_direction(local_imp)
        return ExplanationBundle(global_importance=global_imp, local_summary=local_imp, direction_summary=direction)
    except Exception as exc:
        LOGGER.exception("Failed to build explanation bundle")
        raise RuntimeError(f"Failed to build explanation bundle: {exc}") from exc
