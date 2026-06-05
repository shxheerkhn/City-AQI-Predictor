# Vertex AI Deployment Checklist

## 1. Project and Billing

- Create or select the GCP project.
- Enable billing.
- Enable APIs:
  - Vertex AI API
  - BigQuery API
  - Cloud Storage API
  - Cloud Run API
  - Cloud Build API
  - Secret Manager API

## 2. Service Account

Create a dedicated service account for the AQI platform and grant the following roles:

- `roles/aiplatform.admin` for model registry and Vertex resources
- `roles/bigquery.admin` or a tighter dataset-scoped BigQuery role for feature/model tables
- `roles/storage.admin` or bucket-scoped object admin for staging artifacts
- `roles/run.admin` for deployment
- `roles/secretmanager.secretAccessor` for API tokens
- `roles/logging.logWriter`
- `roles/monitoring.metricWriter`

## 3. Storage

- Create a GCS bucket for staging artifacts.
- Create a BigQuery dataset for feature, model, and prediction tables.
- Confirm the service account has access to both resources.

## 4. Environment Variables

Set the runtime configuration:

- `AQI_FEATURE_STORE_BACKEND=vertex`
- `AQI_CLOUD_PROJECT_ID=<project-id>`
- `AQI_VERTEX_LOCATION=us-central1`
- `AQI_VERTEX_STAGING_BUCKET=<bucket-name>`
- `AQI_BIGQUERY_DATASET=aqi_feature_store`
- `AQI_SOURCE_PROVIDER=aqicn` or `openweather`
- `AQI_EXTERNAL_API_KEY=<provider-token>`
- `API_ACCESS_TOKEN=<service-token>`

## 5. Secrets

- Store API tokens in Secret Manager.
- Inject them into Cloud Run at deploy time.
- Do not hardcode secrets in manifests or application code.

## 6. Deployment Order

1. Deploy the API service.
2. Run hourly ingestion once to seed raw data.
3. Run feature materialization.
4. Run training.
5. Confirm a champion model exists in the registry tables.
6. Deploy the dashboard.

## 7. Validation Gates

- Data validation passes.
- Model RMSE is below threshold.
- Champion model metadata exists.
- API `/health` returns `ok`.
- Forecast endpoint returns a 3-day forecast.

## 8. Rollback Strategy

- Keep the last champion model metadata in the registry.
- Repoint the API to the prior artifact if a new deployment regresses.
- Use the BigQuery model registry metadata as the source of truth for rollback selection.
