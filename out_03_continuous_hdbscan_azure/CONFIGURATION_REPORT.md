# UKPLC CSP Segmentation - Configuration & Results Report

**Generated**: 2025-10-24
**Run ID**: `out_csp_discretised` (Azure ML)
**Note**: Folder name says "discretised" but this run used **continuous recipe** with HDBSCAN

---

## Executive Summary

This segmentation run processed **165,751 Mid-Market customers** using a **continuous feature recipe** with **HDBSCAN clustering** on Azure ML. The algorithm identified **120 distinct customer clusters** with excellent separation metrics, though 67% of customers were classified as outliers (cluster -1).

---

## Configuration Used

### Recipe & Algorithm
- **Recipe**: `continuous` (NOT discretised despite folder name)
  - Imputation: Median strategy
  - Log transform: Applied to ACTUAL_CLTV and FUTURE_LIFETIME_VALUE (with clipping to handle negatives)
  - Scaling: StandardScaler (mean/std-based)

- **Algorithm**: `auto` → **HDBSCAN selected**
  - min_cluster_size: 100
  - min_samples: None (auto)

- **Random State**: 42 (reproducible)

### Performance Features (Clustering Inputs)
1. **ACTUAL_CLTV** - Historical customer profitability (log1p transformed)
2. **CURRENT_YEAR_FAP** - Current year Full Annual Premium
3. **FUTURE_LIFETIME_VALUE** - Forward-looking value projection (log1p transformed)
4. **ACTUAL_LIFETIME_DURATION** - Customer tenure
5. **NUM_CROSS_SOLD_LY** - Cross-sell diversity (distinct products)
6. **CLM_OVER_PROFIT_HITCOUNT** - Claims-over-plan frequency

### Explanatory Columns (Profiling Only)
CUSTOMER_SEGMENT, CUSTOMER_PORTFOLIO, ACTIVE_CUSTOMER, CURRENT_GWP, LAST_YEAR_GWP, GEP_FIN_FULL, GEC_FIN_FULL, CLM_INC_FULL, WEIGHTED_PLAN_LOSS_RATIO, ACTUAL_LOSS_RATIO, EXPECTED_LIFETIME_VALUE, TOTAL_SCORE, INSP_FLAG

---

## Results

### Clustering Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Clusters Detected** | 120 | High granularity |
| **Silhouette Score** | 0.476 | **Good** - Clusters are well-separated |
| **Calinski-Harabasz** | 528,341 | **Excellent** - Very high between/within cluster variance ratio |
| **Davies-Bouldin** | 0.946 | **Excellent** - Low intra-cluster similarity (closer to 0 is better) |

**Overall Assessment**: Metrics indicate excellent cluster quality for the 33% of customers in dense clusters.

### Cluster Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| **Dense Clusters (0-119)** | 54,456 | 32.9% |
| **Noise/Outliers (cluster -1)** | 111,295 | 67.1% |
| **Total Customers** | 165,751 | 100% |

**Note**: HDBSCAN assigns customers who don't fit dense patterns to cluster -1 (noise). This is expected behavior for heterogeneous datasets.

### Top Clusters by Size

| Cluster | Size | % Share | Median CLTV | Mean CLTV | Product Index | Primary Segment |
|---------|------|---------|-------------|-----------|---------------|-----------------|
| -1 (noise) | 111,295 | 67.1% | £4,377 | £16,268 | 1.36 | F |
| 7 | 11,502 | 6.9% | £1,155 | £2,522 | 0.00 | F (Inactive) |
| 43 | 3,590 | 2.2% | £202 | £1,417 | 1.01 | E |
| 6 | 2,245 | 1.4% | -£2,456 | -£3,689 | 0.00 | F (Inactive, Loss-making) |
| 94 | 1,526 | 0.9% | £2,347 | £2,364 | 0.00 | F (Inactive) |

---

## Key Insights

### 1. High Noise Proportion (67%)
- **67% of customers classified as outliers** suggests high heterogeneity in the Mid-Market portfolio
- Cluster -1 still shows meaningful statistics (median CLTV £4,377)
- **Recommendation**: Consider trying KMeans for more balanced cluster sizes

### 2. Cluster Quality
- **Silhouette 0.48** indicates strong separation for customers in dense clusters
- **Low Davies-Bouldin (0.95)** confirms minimal overlap between clusters
- Clusters represent genuinely distinct customer behaviors

### 3. Comparison with Local Run
- Local run (out_csp): 118 clusters, Silhouette 0.470
- Azure run (this): 120 clusters, Silhouette 0.476
- **Very similar results** - validates reproducibility despite different hardware

### 4. Inactive Customer Patterns
- Several large clusters (7, 6, 94) contain primarily inactive customers (ACTIVE_CUSTOMER='N')
- Cluster 6 shows negative CLTV (-£3,689 mean) - loss-making customers

### 5. Cross-Sell Insights
- Product Index varies significantly:
  - Cluster -1: 1.36 (above-average cross-sell)
  - Cluster 43: 1.01 (average cross-sell)
  - Many clusters: 0.00 (single-product customers)

---

## Environment

### Execution Platform
- **Platform**: Azure ML Compute Instance (Linux)
- **Python**: 3.11
- **Environment**: conda environment with conda-forge channel
- **Runtime**: ~2-3 minutes (much faster than Windows local)

### Reproducibility Command
```bash
ukplc-seg \
  --input ./data/CSP_export.csv \
  --outdir ./out_csp_discretised \
  --config CONFIG_DCSP.yaml
```

**Note**: CONFIG_DCSP.yaml contains `recipe: continuous` (not discretised as folder name suggests)

---

## Output Files

| File | Description |
|------|-------------|
| `cluster_assignments.csv` | All 165,751 customers with cluster labels |
| `cluster_profiles.csv` | Statistics per cluster (medians, means, product_index) |
| `cluster_vs_existing_segment.csv` | Cross-tab: new clusters vs existing A/A+/B/C/D/E/F segments |
| `internal_metrics.json` | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| `explain.json` | SHAP feature importances (RandomForest surrogate) |
| `cluster_sizes.png` | Cluster size bar chart |
| `REPORT.md` | Basic summary report |

---

## Recommendations

### Immediate Actions
1. **Rename folder** to `out_csp_continuous_azure` to avoid confusion
2. **Try true discretised recipe** with KMeans for comparison
3. **Investigate Cluster -1** - Understand characteristics of the 67% outlier population

### Business Application
- **High-value targeting**: Focus on clusters with high CLTV and low loss ratios
- **Cross-sell opportunities**: Target clusters with low product_index but high CLTV
- **Churn risk**: Investigate inactive clusters (7, 6, 94) for re-engagement

---

**End of Report**
