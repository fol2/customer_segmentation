# UKPLC Segmentation Report

- **Recipe**: continuous
- **Algorithm**: kmeans
- **Detected clusters**: 9
- **Silhouette**: 0.38511101704552936
- **Calinski–Harabasz**: 88482.72574609135
- **Davies–Bouldin**: 1.0286140590620856

## Notes
- Input assumed exported from `UKPLC_CLTV_CSP_MV`/`UKPLC_CLTV_DCSP_MV`, containing performance features
  (`ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`, `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`).
- Clustering uses only performance features; explanatory columns describe clusters but are never used as inputs.
- SHAP explanations are provided by training a simple RandomForest to predict cluster labels and summarising mean absolute SHAP values.
