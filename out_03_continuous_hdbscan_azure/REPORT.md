# UKPLC Segmentation Report

- **Recipe**: continuous
- **Algorithm**: auto
- **Detected clusters**: 120
- **Silhouette**: 0.47574212736379407
- **Calinski–Harabasz**: 528340.8680500438
- **Davies–Bouldin**: 0.9455496226134487

## Notes
- Input assumed exported from `UKPLC_CLTV_CSP_MV`/`UKPLC_CLTV_DCSP_MV`, containing performance features
  (`ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`, `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`).
- Clustering uses only performance features; explanatory columns describe clusters but are never used as inputs.
- SHAP explanations are provided by training a simple RandomForest to predict cluster labels and summarising mean absolute SHAP values.
