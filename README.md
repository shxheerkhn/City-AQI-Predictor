# Pearls AQI Predictor

Pearls AQI Predictor is an end-to-end air quality forecasting platform for hourly AQI ingestion, feature engineering, model training, model registry/history, explainability, reporting, and dashboarding.

## What It Does

- Ingests weather and pollutant data from synthetic, AQICN, or OpenWeather sources
- Validates raw observations and stores them in a feature store backend
- Builds supervised training data with lag, rolling, and time-based features
- Trains and compares multiple models, including baseline, classical, statistical-style, XGBoost, LSTM, and GRU variants
- Generates forecast reports, residual plots, prediction-vs-actual plots, and EDA outputs
- Serves one-step and multi-hour forecasts through a FastAPI service
- Displays forecast, history, explainability, operations, reports, and EDA views in Streamlit

## Repository Layout

- `aqi_predictor/data_pipeline` - data source adapters and ingestion logic
- `aqi_predictor/feature_pipeline` - feature engineering and materialization
- `aqi_predictor/feature_store` - local SQLite, Vertex, and Hopsworks store adapters
- `aqi_predictor/training_pipeline` - model training, evaluation, and forecasting utilities
- `aqi_predictor/prediction_service` - FastAPI prediction service and schemas
- `aqi_predictor/dashboard` - Streamlit dashboard
- `aqi_predictor/analysis` - EDA generation and reporting helpers
- `aqi_predictor/monitoring` - metrics and explainability utilities
- `aqi_predictor/reporting` - final report generation
- `tests` - unit tests for the pipeline
- `docs` - architecture, runbook, and deployment notes
- `infrastructure` - Cloud Run manifests

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m aqi_predictor.feature_pipeline.cli
python -m aqi_predictor.training_pipeline.train
python -m aqi_predictor.api.main
streamlit run aqi_predictor/dashboard/app.py
```

## Local Workflow

1. Run the feature pipeline to seed or refresh raw data.
2. Run training to materialize features, compare models, register the champion, and generate reports.
3. Start the API to serve `/health`, `/metrics`, `/model-info`, `/predict`, and `/forecast`.
4. Start the dashboard to inspect live forecasts, history, operations, explainability, and reports.

## Configuration

Configuration is loaded from environment variables and `.env` files when present.

Common variables:

- `AQI_FEATURE_STORE_BACKEND` - `local`, `vertex`, or `hopsworks`
- `AQI_SOURCE_PROVIDER` - `synthetic`, `aqicn`, or `openweather`
- `AQI_EXTERNAL_API_KEY` - provider token
- `AQI_API_BASE_URL` - API base URL used by the dashboard
- `API_ACCESS_TOKEN` - optional API key for protected endpoints
- `AQI_DATABASE_PATH` - SQLite path for local development
- `AQI_REPORT_DIR` - report output directory
- `AQI_MODEL_DIR` - trained model artifact directory
- `AQI_FORECAST_HOURS` - forecast horizon, default `72`
- `AQI_BACKFILL_DAYS` - historical backfill window
- `AQI_AUTO_BOOTSTRAP_DEMO` - auto-train on dashboard startup when enabled

See [`docs/production_setup.md`](docs/production_setup.md) and [`docs/runbook.md`](docs/runbook.md) for cloud deployment guidance.

## Training Pipeline

The training pipeline:

- loads and validates raw measurements
- materializes supervised feature frames
- generates EDA artifacts
- splits data into train and validation sets
- trains multiple candidate models
- evaluates RMSE, MAE, R2, and MAPE
- selects the best model as champion
- saves the champion artifact and metadata
- records model history and comparison data
- generates residual and prediction plots
- writes final report files

Run it with:

```bash
python -m aqi_predictor.training_pipeline.train
```

## Feature Pipeline

The feature pipeline:

- ingests hourly observations for configured cities
- backfills historical data when the store is empty
- validates sensor frames before persisting
- creates engineered features such as lag values, rolling statistics, and cyclical time features

Run it with:

```bash
python -m aqi_predictor.feature_pipeline.cli
```

## API Service

Run the API with:

```bash
python -m aqi_predictor.api.main
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /model-info`
- `POST /predict`
- `POST /forecast`

## Dashboard

Run the dashboard with:

```bash
streamlit run aqi_predictor/dashboard/app.py
```

The dashboard includes:

- forecast explorer
- historical trends
- explainability and SHAP-style analysis
- operational metrics
- model reports and comparison tables
- EDA summaries and visualizations

## Reports and Artifacts

Training produces artifacts under `artifacts/` by default:

- `artifacts/models/` for saved models
- `artifacts/reports/` for evaluation JSON, plots, final report, and EDA outputs
- `artifacts/aqi_store.sqlite3` for the local development store

Generate the final summary report separately with:

```bash
python -m aqi_predictor.reporting.cli
```

## Testing

Run the test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Deployment

The repository includes Cloud Run manifests and production notes for a cloud-based setup.

- API manifest: `infrastructure/cloudrun-api.yaml`
- Dashboard manifest: `infrastructure/cloudrun-dashboard.yaml`
- Setup guide: `docs/production_setup.md`
- Runbook: `docs/runbook.md`

## Optional Dependencies

Some features are optional and need extra packages beyond the core local install:

- `shap` for richer explainability
- `google-cloud-bigquery`, `google-cloud-storage`, and `google-cloud-aiplatform` for Vertex-backed storage and registry flows
- `scipy` for the EDA module

## Notes

- The default local backend is SQLite so the project can run without cloud credentials.
- The production target is a cloud-backed feature store and artifact workflow.
- The system is designed to keep raw measurements, engineered features, and trained models versioned and queryable.
