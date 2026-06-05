from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ValidationIssue:
    column: str
    issue_type: str
    detail: str


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    issues: list[ValidationIssue]


def ensure_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    try:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    except Exception as exc:
        LOGGER.exception("Column validation failed")
        raise


def validate_numeric_range(frame: pd.DataFrame, columns: Iterable[str]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    try:
        for column in columns:
            if column not in frame.columns:
                issues.append(ValidationIssue(column, "missing_column", "Column absent from frame"))
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.isna().all():
                issues.append(ValidationIssue(column, "non_numeric", "Column could not be coerced to numeric"))
                continue
            finite_values = series[np.isfinite(series)]
            if finite_values.empty:
                issues.append(ValidationIssue(column, "empty_numeric", "No finite numeric values found"))
                continue
            mean = float(finite_values.mean())
            std = float(finite_values.std(ddof=0))
            if math.isnan(std) or std == 0.0:
                continue
            outlier_mask = (finite_values - mean).abs() > 4.0 * std
            if outlier_mask.any():
                issues.append(
                    ValidationIssue(column, "outliers", f"{int(outlier_mask.sum())} observations exceed 4 standard deviations")
                )
        return ValidationResult(is_valid=len(issues) == 0, issues=issues)
    except Exception as exc:
        LOGGER.exception("Numeric range validation failed")
        raise RuntimeError(f"Numeric range validation failed: {exc}") from exc


def validate_sensor_frame(frame: pd.DataFrame, required: Iterable[str]) -> ValidationResult:
    try:
        ensure_columns(frame, required)
        issues: list[ValidationIssue] = []
        if frame.empty:
            issues.append(ValidationIssue("frame", "empty", "No rows supplied"))
        duplicate_count = int(frame.duplicated(subset=[c for c in ["city_id", "timestamp"] if c in frame.columns]).sum())
        if duplicate_count:
            issues.append(ValidationIssue("city_id/timestamp", "duplicates", f"{duplicate_count} duplicate rows found"))
        missing_ratio = frame.isna().mean()
        for column, ratio in missing_ratio.items():
            if ratio > 0:
                issues.append(ValidationIssue(column, "missing_values", f"{ratio:.2%} missing"))
        numeric_result = validate_numeric_range(frame, [c for c in required if c not in {"city_id", "timestamp", "source"}])
        issues.extend(numeric_result.issues)
        return ValidationResult(is_valid=len(issues) == 0, issues=issues)
    except Exception as exc:
        LOGGER.exception("Sensor frame validation failed")
        raise RuntimeError(f"Sensor frame validation failed: {exc}") from exc
