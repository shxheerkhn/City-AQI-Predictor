# Production Setup

## Recommended Backend

- Feature store: Vertex AI-friendly BigQuery tables
- Artifact store: Google Cloud Storage
- Model registry: Vertex AI Model Registry best-effort upload plus BigQuery metadata registry
- Serving: FastAPI on Cloud Run
- Dashboard: Streamlit on Cloud Run

## Required Environment Variables

- `AQI_FEATURE_STORE_BACKEND=vertex`
- `AQI_CLOUD_PROJECT_ID=<gcp-project-id>`
- `AQI_VERTEX_LOCATION=us-central1`
- `AQI_VERTEX_STAGING_BUCKET=<gcs-bucket-name>`
- `AQI_BIGQUERY_DATASET=aqi_feature_store`
- `AQI_EXTERNAL_API_KEY=<aqicn-or-openweather-token>`
- `AQI_SOURCE_PROVIDER=aqicn` or `openweather`
- `API_ACCESS_TOKEN=<service-token>`

## Deployment Order

1. Create the GCS bucket and BigQuery dataset.
2. Grant the service account permissions for BigQuery, GCS, and Vertex AI.
3. Deploy the FastAPI service.
4. Run the hourly data pipeline.
5. Run the daily training pipeline.
6. Deploy the dashboard once a champion model exists.

## Operational Caveats

- AQICN requires a valid token and has usage restrictions, so production data usage must respect their terms.
- Vertex model upload can fail if a serving container image is not appropriate for the saved artifact family; the code logs and continues with BigQuery registry metadata in that case.
