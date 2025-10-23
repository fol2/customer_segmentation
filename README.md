
# UKPLC Segmentation (Clustering on Performance Features)

This package implements a **scientific alternative** to score‑cut segmentation: cluster customers
directly on **performance features** (the six CSP factor inputs: *Actual CLTV*, *Current Year FAP*,
*Future Lifetime Value*, *Actual Lifetime Duration*, *Cross‑sell diversity (last year)* and *Claims‑over‑plan hitcount*).

Two **recipes**:

1. **continuous** – impute → optional `log1p` → scaling (`robust`/`standard`/`none`) → clustering (**HDBSCAN** if available, else **KMeans** with auto k).
2. **discretised** – case‑study style *quantile binning* (configurable `--n-bins`) with optional `log1p` pre‑transform → one‑hot → **KMeans**.

> Explanatory metrics（e.g. ELV, GWP, loss ratios, existing `CUSTOMER_SEGMENT`）are **not** used for clustering；只用於 **事後解釋**。

## Quick start (with `uv`)

```bash
uv venv -p 3.11
uv pip install -e .

# CSP（Mid-Market）示例：包含全部渠道、包含 inspection（你要求保留）
ukplc-seg --input /data/CSP_export.csv --outdir ./out_csp --recipe discretised --algorithm auto --n-bins 7

# DCSP（Digital）示例：同一段代碼，只換輸入檔
ukplc-seg --input /data/DCSP_export.csv --outdir ./out_dcsp --recipe continuous --algorithm auto

# 需要測參數（recipes/algorithms/k-select/scaler 等），用 sweep 工具：
ukplc-seg-sweep --input /data/CSP_export.csv --outdir ./sweep --recipes continuous,discretised --algorithms auto,kmeans,hdbscan
```

### Useful switches

- `--filter COL=VAL` 可重覆（例如 `--filter CUSTOMER_PORTFOLIO=MID` 或 `--filter ACTIVE_CUSTOMER=Y`）
- `--exclude-inspection-only` 會剔除 `INSP_FLAG='Y'` 的 inspection-only 客戶（**預設不剔除**，符合你現時需要）
- `--k-select composite|silhouette` 控制自動選 k 的評分準則
- `--scaler robust|standard|none` 以及 `--log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE` 幫手處理 heavy tails
- `--id-col` 指定輸出對應的識別欄位名（預設 `CUSTOMER_ID`）
- `--rows-limit` 用少量樣本快速 dry-run

**Outputs**（在 `out_*`）:
- `cluster_assignments.csv`、`cluster_profiles.csv`、`cluster_vs_existing_segment.csv`（如有）
- `internal_metrics.json`、`explain.json`（含 SHAP 摘要如可用）、`cluster_sizes.png`、`REPORT.md`

## Data assumptions

輸入係 `UKPLC_CLTV_CSP_MV` / `UKPLC_CLTV_DCSP_MV` 匯出（兩者**結構一致**，僅濾唔同 channel/portfolio），並包含：  
**features**：`ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`, `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`；  
**explanatory**：`EXPECTED_LIFETIME_VALUE`, `GWP/GEP/GEC/CLM`, `WEIGHTED_PLAN_LOSS_RATIO`, `ACTUAL_LOSS_RATIO`, `CUSTOMER_SEGMENT`, `CUSTOMER_PORTFOLIO`, `TOTAL_SCORE` 等。

## Licence
MIT
