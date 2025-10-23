# UKPLC Segmentation Report

- **Recipe**: discretised
- **Algorithm**: kmeans
- **Detected clusters**: 3
- **Silhouette**: 0.25695969896247023
- **Calinski–Harabasz**: 36768.06650936203
- **Davies–Bouldin**: 1.9188140200427968

## Notes
- Input assumed exported from `UKPLC_CLTV_CSP_MV`/`UKPLC_CLTV_DCSP_MV`, containing performance features
  (`ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`, `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`).
- Clustering uses only performance features; explanatory columns describe clusters but are never used as inputs.
- SHAP explanations are provided by training a simple RandomForest to predict cluster labels and summarising mean absolute SHAP values.
