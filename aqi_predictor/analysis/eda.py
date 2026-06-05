"""Comprehensive Exploratory Data Analysis (EDA) module for AQI data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EDAReport:
    """Container for EDA analysis results."""

    data_quality: dict[str, Any]
    univariate: dict[str, Any]
    bivariate: dict[str, Any]
    time_series: dict[str, Any]
    feature_engineering: dict[str, Any]


class EDAnalyzer:
    """Comprehensive EDA analyzer for AQI prediction data."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """Initialize EDA analyzer with a dataframe."""
        self.frame = frame
        self.numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = frame.select_dtypes(include=["object"]).columns.tolist()

    def analyze(self) -> EDAReport:
        """Run comprehensive EDA analysis."""
        try:
            return EDAReport(
                data_quality=self._analyze_data_quality(),
                univariate=self._analyze_univariate(),
                bivariate=self._analyze_bivariate(),
                time_series=self._analyze_time_series(),
                feature_engineering=self._analyze_feature_engineering(),
            )
        except Exception as exc:
            LOGGER.exception("Failed to run EDA analysis")
            raise RuntimeError(f"Failed to run EDA analysis: {exc}") from exc

    def _analyze_data_quality(self) -> dict[str, Any]:
        """Analyze data quality: missing values, duplicates, data types."""
        try:
            missing_info = {
                col: {
                    "count": int(self.frame[col].isna().sum()),
                    "percentage": float((self.frame[col].isna().sum() / len(self.frame)) * 100),
                }
                for col in self.frame.columns
            }

            duplicates = {
                "count": int(self.frame.duplicated().sum()),
                "percentage": float((self.frame.duplicated().sum() / len(self.frame)) * 100),
            }

            return {
                "total_rows": int(len(self.frame)),
                "total_columns": int(len(self.frame.columns)),
                "memory_usage_mb": float(self.frame.memory_usage(deep=True).sum() / 1024 ** 2),
                "missing_values": missing_info,
                "duplicates": duplicates,
                "data_types": {str(col): str(dtype) for col, dtype in self.frame.dtypes.items()},
            }
        except Exception as exc:
            LOGGER.warning(f"Failed to analyze data quality: {exc}")
            return {}

    def _analyze_univariate(self) -> dict[str, Any]:
        """Analyze distribution of individual features."""
        try:
            univariate = {}
            for col in self.numeric_cols:
                col_data = self.frame[col].dropna()
                if len(col_data) > 0:
                    univariate[col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "q25": float(col_data.quantile(0.25)),
                        "q75": float(col_data.quantile(0.75)),
                        "max": float(col_data.max()),
                        "skewness": float(stats.skew(col_data)),
                        "kurtosis": float(stats.kurtosis(col_data)),
                        "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    }

            for col in self.categorical_cols:
                univariate[col] = {
                    "unique_count": int(self.frame[col].nunique()),
                    "value_counts": self.frame[col].value_counts().head(10).to_dict(),
                }

            return univariate
        except Exception as exc:
            LOGGER.warning(f"Failed to analyze univariate distributions: {exc}")
            return {}

    def _analyze_bivariate(self) -> dict[str, Any]:
        """Analyze relationships between features."""
        try:
            bivariate = {}

            if len(self.numeric_cols) > 1:
                corr_matrix = self.frame[self.numeric_cols].corr()
                bivariate["correlation_matrix"] = corr_matrix.to_dict()

                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append(
                                {
                                    "col1": str(corr_matrix.columns[i]),
                                    "col2": str(corr_matrix.columns[j]),
                                    "correlation": float(corr_val),
                                }
                            )
                bivariate["high_correlations"] = high_corr_pairs

            if "aqi_target" in self.frame.columns:
                target = self.frame["aqi_target"].dropna()
                for col in self.numeric_cols:
                    if col != "aqi_target":
                        col_data = self.frame[col].dropna()
                        if len(col_data) > 0 and len(target) > 0:
                            aligned_target = target[col_data.index]
                            corr = col_data.corr(aligned_target)
                            bivariate[f"target_correlation_{col}"] = float(corr)

            return bivariate
        except Exception as exc:
            LOGGER.warning(f"Failed to analyze bivariate relationships: {exc}")
            return {}

    def _analyze_time_series(self) -> dict[str, Any]:
        """Analyze time series patterns."""
        try:
            ts_analysis = {}

            if "timestamp" in self.frame.columns:
                frame_sorted = self.frame.sort_values("timestamp")
                frame_sorted["date"] = pd.to_datetime(frame_sorted["timestamp"]).dt.date
                frame_sorted["hour"] = pd.to_datetime(frame_sorted["timestamp"]).dt.hour
                frame_sorted["day_of_week"] = pd.to_datetime(frame_sorted["timestamp"]).dt.day_name()
                frame_sorted["month"] = pd.to_datetime(frame_sorted["timestamp"]).dt.month

                if "aqi_target" in frame_sorted.columns:
                    ts_analysis["hourly_mean_aqi"] = frame_sorted.groupby("hour")["aqi_target"].mean().to_dict()
                    ts_analysis["hourly_std_aqi"] = frame_sorted.groupby("hour")["aqi_target"].std().to_dict()
                    ts_analysis["dow_mean_aqi"] = frame_sorted.groupby("day_of_week")["aqi_target"].mean().to_dict()
                    ts_analysis["monthly_mean_aqi"] = frame_sorted.groupby("month")["aqi_target"].mean().to_dict()

                ts_analysis["date_range"] = {
                    "start": str(frame_sorted["timestamp"].min()),
                    "end": str(frame_sorted["timestamp"].max()),
                    "days_covered": len(frame_sorted["date"].unique()),
                }

            return ts_analysis
        except Exception as exc:
            LOGGER.warning(f"Failed to analyze time series patterns: {exc}")
            return {}

    def _analyze_feature_engineering(self) -> dict[str, Any]:
        """Analyze engineered features."""
        try:
            fe_analysis = {
                "total_features": len(self.numeric_cols),
                "feature_names": self.numeric_cols,
                "zero_variance_features": [],
                "low_variance_features": [],
            }

            for col in self.numeric_cols:
                variance = self.frame[col].var()
                if variance == 0:
                    fe_analysis["zero_variance_features"].append(col)
                elif variance < 0.01:
                    fe_analysis["low_variance_features"].append(col)

            return fe_analysis
        except Exception as exc:
            LOGGER.warning(f"Failed to analyze feature engineering: {exc}")
            return {}

    def plot_distributions(self) -> dict[str, Any]:
        """Generate distribution plots for numeric features."""
        plots = {}
        try:
            for col in self.numeric_cols[:10]:
                if col in {"city_id", "feature_version"}:
                    continue
                fig = px.histogram(
                    self.frame,
                    x=col,
                    nbins=30,
                    title=f"Distribution of {col}",
                    labels={col: col},
                )
                fig.update_layout(height=400, width=600)
                plots[f"distribution_{col}"] = fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot distributions: {exc}")
        return plots

    def plot_correlation_heatmap(self) -> Any:
        """Generate correlation heatmap."""
        try:
            if len(self.numeric_cols) > 1:
                corr_matrix = self.frame[self.numeric_cols].corr()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text:.2f}",
                    )
                )
                fig.update_layout(title="Feature Correlation Heatmap", height=600, width=800)
                return fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot correlation heatmap: {exc}")
        return None

    def plot_target_relationships(self) -> dict[str, Any]:
        """Generate plots showing relationships with target variable."""
        plots = {}
        try:
            if "aqi_target" not in self.frame.columns:
                return plots

            for col in self.numeric_cols[:8]:
                if col in {"aqi_target", "city_id"}:
                    continue
                fig = px.scatter(
                    self.frame,
                    x=col,
                    y="aqi_target",
                    title=f"AQI vs {col}",
                    trendline="ols",
                    labels={col: col, "aqi_target": "AQI Target"},
                )
                fig.update_layout(height=400, width=600)
                plots[f"target_vs_{col}"] = fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot target relationships: {exc}")
        return plots

    def plot_time_series(self) -> dict[str, Any]:
        """Generate time series plots."""
        plots = {}
        try:
            if "timestamp" in self.frame.columns and "aqi_target" in self.frame.columns:
                frame_sorted = self.frame.sort_values("timestamp").tail(500)
                fig = px.line(
                    frame_sorted,
                    x="timestamp",
                    y="aqi_target",
                    title="AQI over Time (Last 500 records)",
                    labels={"timestamp": "Time", "aqi_target": "AQI"},
                )
                fig.update_layout(height=400, width=900)
                plots["aqi_timeseries"] = fig

            if "timestamp" in self.frame.columns and "pm25" in self.frame.columns:
                frame_sorted = self.frame.sort_values("timestamp").tail(500)
                fig = px.line(
                    frame_sorted,
                    x="timestamp",
                    y="pm25",
                    title="PM2.5 over Time (Last 500 records)",
                    labels={"timestamp": "Time", "pm25": "PM2.5 (µg/m³)"},
                )
                fig.update_layout(height=400, width=900)
                plots["pm25_timeseries"] = fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot time series: {exc}")
        return plots

    def plot_hourly_patterns(self) -> Any:
        """Generate hourly pattern plot."""
        try:
            if "timestamp" not in self.frame.columns or "aqi_target" not in self.frame.columns:
                return None

            frame_sorted = self.frame.copy()
            frame_sorted["hour"] = pd.to_datetime(frame_sorted["timestamp"]).dt.hour
            hourly_mean = frame_sorted.groupby("hour")["aqi_target"].mean()

            fig = px.bar(
                x=hourly_mean.index,
                y=hourly_mean.values,
                title="Average AQI by Hour of Day",
                labels={"x": "Hour of Day", "y": "Average AQI"},
            )
            fig.update_layout(height=400, width=700)
            return fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot hourly patterns: {exc}")
        return None

    def plot_outliers(self) -> dict[str, Any]:
        """Generate outlier detection plots using IQR method."""
        plots = {}
        try:
            for col in self.numeric_cols[:5]:
                Q1 = self.frame[col].quantile(0.25)
                Q3 = self.frame[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                fig = go.Figure()
                fig.add_trace(
                    go.Box(y=self.frame[col], name=col, boxmean="sd")
                )
                fig.update_layout(
                    title=f"Outlier Detection: {col} (IQR method)",
                    height=400,
                    width=600,
                )
                plots[f"outliers_{col}"] = fig
        except Exception as exc:
            LOGGER.warning(f"Failed to plot outliers: {exc}")
        return plots


def generate_eda_report(frame: pd.DataFrame, report_dir: Path) -> Path:
    """Generate comprehensive EDA report."""
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        analyzer = EDAnalyzer(frame)
        analysis = analyzer.analyze()

        report_json = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "data_quality": analysis.data_quality,
            "univariate": analysis.univariate,
            "bivariate": analysis.bivariate,
            "time_series": analysis.time_series,
            "feature_engineering": analysis.feature_engineering,
        }

        report_path = report_dir / "eda_report.json"
        report_path.write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")
        LOGGER.info(f"EDA report saved to {report_path}")

        return report_path
    except Exception as exc:
        LOGGER.exception("Failed to generate EDA report")
        raise RuntimeError(f"Failed to generate EDA report: {exc}") from exc
