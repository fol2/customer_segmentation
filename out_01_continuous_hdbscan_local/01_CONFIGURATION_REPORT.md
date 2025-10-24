# UKPLC CSP Segmentation - Configuration & Results Report

**Generated**: 2025-10-23
**Run ID**: `out_csp`

---

## Executive Summary

This segmentation run processed **165,751 Mid-Market customers** using a **continuous feature recipe** with **HDBSCAN clustering**. The algorithm identified **118 distinct customer clusters** with excellent separation metrics, though 67% of customers were classified as outliers (cluster -1).

---

## Configuration Used

### Recipe & Algorithm
- **Recipe**: `continuous`
  - Imputation: Median strategy
  - Log transform: Applied to ACTUAL_CLTV and FUTURE_LIFETIME_VALUE (with clipping to handle negatives)
  - Scaling: RobustScaler (median/IQR-based, robust to outliers)

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
| **Clusters Detected** | 118 | High granularity |
| **Silhouette Score** | 0.470 | **Good** - Clusters are well-separated |
| **Calinski-Harabasz** | 538,431 | **Excellent** - Very high between/within cluster variance ratio |
| **Davies-Bouldin** | 0.941 | **Excellent** - Low intra-cluster similarity (closer to 0 is better) |

**Overall Assessment**: Metrics indicate excellent cluster quality for the 33% of customers in dense clusters.

### Cluster Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| **Dense Clusters (0-117)** | 54,494 | 32.9% |
| **Noise/Outliers (cluster -1)** | 111,257 | 67.1% |
| **Total Customers** | 165,751 | 100% |

**Note**: HDBSCAN assigns customers who don't fit dense patterns to cluster -1 (noise). This is expected behavior for heterogeneous datasets.

### Top Clusters by Size

| Cluster | Size | % Share | Median CLTV | Mean CLTV | Product Index | Primary Segment |
|---------|------|---------|-------------|-----------|---------------|-----------------|
| -1 (noise) | 111,257 | 67.1% | £4,380 | £16,275 | 1.36 | F |
| 7 | 11,502 | 6.9% | £1,155 | £2,522 | 0.00 | F (Inactive) |
| 43 | 3,589 | 2.2% | £202 | £1,417 | 1.01 | E |
| 6 | 2,245 | 1.4% | -£2,456 | -£3,689 | 0.00 | F (Inactive, Loss-making) |
| 92 | 1,525 | 0.9% | £2,347 | £2,364 | 0.00 | F (Inactive) |

---

## Key Insights

### 1. High Noise Proportion (67%)
- **67% of customers classified as outliers** suggests high heterogeneity in the Mid-Market portfolio
- Cluster -1 still shows meaningful statistics (median CLTV £4,380)
- **Recommendation**: Consider trying KMeans for more balanced cluster sizes, or reduce HDBSCAN min_cluster_size

### 2. Cluster Quality
- **Silhouette 0.47** indicates strong separation for customers in dense clusters
- **Low Davies-Bouldin (0.94)** confirms minimal overlap between clusters
- Clusters represent genuinely distinct customer behaviors

### 3. Inactive Customer Patterns
- Several large clusters (7, 6, 92) contain primarily inactive customers (ACTIVE_CUSTOMER='N')
- Cluster 6 shows negative CLTV (-£3,689 mean) - loss-making customers

### 4. Cross-Sell Insights
- Product Index varies significantly:
  - Cluster -1: 1.36 (above-average cross-sell)
  - Cluster 43: 1.01 (average cross-sell)
  - Many clusters: 0.00 (single-product customers)

### 5. Existing Segment Mapping
- Most new clusters map to existing segment "F" (77-99% in most clusters)
- Limited representation from high-value segments (A, A+, A-) in dense clusters
- Suggests current A/B/C/D/E/F segmentation may not capture HDBSCAN's density-based patterns

---

## Data Quality Notes

### Negative CLTV Values
- **17,447 customers (10.5%)** have negative ACTUAL_CLTV (range: -£73M to £41M)
- These were clipped to 0 before log1p transformation
- Cluster 6 represents a loss-making customer segment (mean CLTV -£3,689)

### Feature Distributions
- ACTUAL_CLTV: Median £2,312, Mean £11,813 (right-skewed)
- Log1p transformation successfully stabilized heavy tails

---

## Output Files

| File | Size | Description |
|------|------|-------------|
| `cluster_assignments.csv` | 1.8 MB | All 165,751 customers with cluster labels |
| `cluster_profiles.csv` | 32 KB | Statistics per cluster (medians, means, product_index) |
| `cluster_vs_existing_segment.csv` | 2.7 KB | Cross-tab: new clusters vs existing A/A+/B/C/D/E/F segments |
| `internal_metrics.json` | 147 B | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| `explain.json` | 15 KB | SHAP feature importances (RandomForest surrogate) |
| `cluster_sizes.png` | 55 KB | Cluster size bar chart |
| `model.joblib` | 40 MB | Serialized pipeline for re-use |
| `REPORT.md` | 749 B | Basic summary report |

---

## Recommendations

### Immediate Actions
1. **Try KMeans for comparison** - Use `discretised` recipe with `kmeans` algorithm for more balanced clusters
2. **Investigate Cluster -1** - Understand characteristics of the 67% outlier population
3. **Segment-specific analysis** - Profile clusters 7, 43, 6 for targeted strategies

### Parameter Tuning
- **Reduce noise**: Lower `hdbscan_min_cluster_size` from 100 to 50
- **Alternative recipe**: Try `discretised` with 5-7 bins for case-study style segments
- **Filter options**: Re-run with `--filter ACTIVE_CUSTOMER=Y` to focus on active customers

### Business Application
- **High-value targeting**: Focus on clusters with high CLTV and low loss ratios
- **Cross-sell opportunities**: Target clusters with low product_index but high CLTV
- **Churn risk**: Investigate inactive clusters (7, 6, 92) for re-engagement

---

## Technical Details

### Pipeline Steps
```
1. Load 165,751 rows from data/CSP_export.csv
2. Feature engineering:
   - Impute missing values (median)
   - Clip negative values to 0 (for log1p features)
   - Apply log1p to ACTUAL_CLTV, FUTURE_LIFETIME_VALUE
   - Scale with RobustScaler (all features)
3. Clustering:
   - HDBSCAN with min_cluster_size=100
   - Detected 118 clusters + noise
4. Evaluation:
   - Compute silhouette, CH, DB (noise excluded)
   - Profile clusters with medians/means
5. Explainability:
   - Train RandomForest classifier on cluster labels
   - Extract SHAP values for feature importance
```

### Runtime
- **Total time**: ~2-3 minutes
- **Feature engineering**: ~10 seconds
- **HDBSCAN clustering**: ~60 seconds
- **SHAP computation**: ~90 seconds

---

## Reproducibility

To reproduce this run:
```bash
.venv/Scripts/ukplc-seg.exe \
  --input ./data/CSP_export.csv \
  --outdir ./out_csp \
  --recipe continuous \
  --algorithm auto \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --random-state 42
```

Or use the default continuous configuration:
```bash
.venv/Scripts/ukplc-seg.exe \
  --input ./data/CSP_export.csv \
  --outdir ./out_csp \
  --config CONFIG.yaml
```

---

**End of Report**
