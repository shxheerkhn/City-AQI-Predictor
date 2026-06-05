# Pearls AQI Predictor Final Report

Generated at: 2026-06-03T14:56:22.194378+00:00

## Champion Model

- Name: prophet_style
- Version: 20260603145503
- Family: statistical
- Scope: global
- Artifact: artifacts\models\prophet_style_20260603145503.joblib

## Champion Metrics

- RMSE: 15.6399
- MAE: 11.1942
- R2: 0.3802
- MAPE: 0.0511

## Horizon Checks

- No day-wise backtest summary available.

## Model Comparison

| Rank | Model | RMSE | MAE | R2 | MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | prophet_style | 15.6399 | 11.1942 | 0.3802 | 0.0511 |
| 2 | ridge_regression | 16.1724 | 11.2605 | 0.3373 | 0.0513 |
| 3 | random_forest | 16.3334 | 11.6847 | 0.3240 | 0.0537 |
| 4 | xgboost | 16.6231 | 12.1561 | 0.2998 | 0.0557 |
| 5 | arima_style | 17.1448 | 12.4653 | 0.2552 | 0.0570 |
| 6 | persistence_model | 19.7439 | 15.4913 | 0.0123 | 0.0688 |
| 7 | mean_predictor | 21.5707 | 16.8704 | -0.1790 | 0.0775 |
| 8 | lstm | 79.6779 | 77.4851 | -17.4209 | 0.3315 |
| 9 | gru | 180.4619 | 179.5044 | -93.4943 | 0.7766 |

## Evaluation Assets

- Evaluation JSON: `evaluation_20260603145503.json`
- Prediction vs Actual plot: `pred_vs_actual_*.png`
- Residual plot: `residuals_*.png`

## Operational Notes

- Raw data is stored immutably in the feature store backend.
- Features are versioned and materialized before training.
- The champion model is registered with metadata and used by the prediction service.
- Vertex AI is the primary production backend, with local SQLite used for development.
