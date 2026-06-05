# Infrastructure Notes

This repository is designed for a serverless deployment target.

Recommended production mapping:

- FastAPI -> Cloud Run
- Streamlit -> Cloud Run or managed Streamlit hosting
- Hourly ingestion -> Cloud Scheduler + Cloud Run job
- Daily training -> Cloud Scheduler + Cloud Run job
- Feature store -> Hopsworks or Vertex AI Feature Store
- Model registry -> Vertex AI Model Registry or MLflow-compatible registry
- Secrets -> Secret Manager

The current workspace ships with a local SQLite-backed development implementation so the project remains runnable without cloud credentials.
