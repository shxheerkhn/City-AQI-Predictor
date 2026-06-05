# Pearls AQI Predictor Architecture

## Core Decisions

- FastAPI serves predictions and health/metrics endpoints.
- Streamlit serves the analytics dashboard.
- SQLite-backed local store is used for development and tests.
- The code is structured so Hopsworks or Vertex AI can replace the local store adapter without rewriting the pipeline.

## Data Flow

1. Hourly ingestion fetches weather and pollutant observations.
2. Raw observations are validated and stored immutably.
3. Feature materialization creates lagged, rolling, and temporal features.
4. Training consumes raw history to build supervised datasets.
5. Models are trained, compared, evaluated, and registered.
6. The champion model is loaded by the serving layer for one-step and 72-hour forecasts.

## Operational Notes

- The workflow is idempotent at the raw and feature store layers.
- Feature and model artifacts are versioned in the local registry by timestamp and model version.
- SHAP is used when available; otherwise a transparent approximation is used for explainability.
