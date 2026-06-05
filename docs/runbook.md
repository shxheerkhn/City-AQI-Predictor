# Pearls AQI Predictor Runbook

## Local Demo

Use this path for local validation without cloud credentials.

```bash
python -m pip install -r requirements.txt
python -m aqi_predictor.feature_pipeline.cli
python -m aqi_predictor.training_pipeline.train
python -m aqi_predictor.reporting.cli
python -m aqi_predictor.api.main
streamlit run aqi_predictor/dashboard/app.py
```

Expected behavior:
- The feature pipeline seeds or refreshes the store.
- Training registers a champion model and writes evaluation assets.
- Reporting produces `final_report.md` and `final_report.json`.
- The API serves health, metrics, model-info, predict, and forecast endpoints.
- The dashboard shows forecast, hazards, performance, operations, and explainability.

## CI Run

The GitHub Actions workflow performs:
- Unit tests
- Hourly feature pipeline
- Daily training pipeline
- Final report generation
- Artifact upload for the report directory

If the workflow fails:
- Check provider credentials first.
- Inspect feature validation logs.
- Check training metrics and artifact upload steps.

## Vertex Production Run

Set the environment:

```bash
export AQI_FEATURE_STORE_BACKEND=vertex
export AQI_CLOUD_PROJECT_ID=<project-id>
export AQI_VERTEX_LOCATION=us-central1
export AQI_VERTEX_STAGING_BUCKET=<bucket-name>
export AQI_BIGQUERY_DATASET=aqi_feature_store
export AQI_SOURCE_PROVIDER=aqicn
export AQI_EXTERNAL_API_KEY=<provider-token>
export API_ACCESS_TOKEN=<service-token>
```

Provision order:
1. Create the BigQuery dataset and GCS bucket.
2. Grant IAM roles to the service account.
3. Deploy the API.
4. Run hourly ingestion.
5. Run daily training.
6. Generate the final report.
7. Deploy the dashboard.

## Recovery

If a deployment regresses:
- Repoint to the previous champion metadata entry.
- Re-run the training pipeline after the data issue is fixed.
- Compare the latest and previous report artifacts before promoting again.
