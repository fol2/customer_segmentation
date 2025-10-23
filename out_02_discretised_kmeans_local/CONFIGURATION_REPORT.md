# UKPLC CSP Segmentation - Configuration & Results Report

**Generated**: 2025-10-24
**Run ID**: `out_csp_discretised_kmeans` (Local Windows)

---

## Executive Summary

This segmentation run processed **165,751 Mid-Market customers** using a **discretised feature recipe** with **KMeans clustering**. The algorithm tested k=3 to k=12 and selected **k=3** as optimal based on composite scoring, creating three broad customer segments with balanced sizes.

---

## Configuration Used

### Recipe & Algorithm
- **Recipe**: `discretised` (quantile binning approach)
  - Imputation: Median strategy
  - Log transform: Applied to ACTUAL_CLTV and FUTURE_LIFETIME_VALUE (with clipping to handle negatives)
  - Quantile binning: 7 bins per feature
  - One-hot encoding: Converts ordinal bins to binary features
  - Scaling: RobustScaler (median/IQR-based)

- **Algorithm**: `kmeans`
  - K-selection: Grid search from k=3 to k=12
  - Scoring method: Composite (silhouette + CH/10000 - DB)
  - Selected k: **3** (optimal trade-off between quality and simplicity)

- **Random State**: 42 (reproducible)

### Performance Features (Clustering Inputs)
1. **ACTUAL_CLTV** - Historical customer profitability (log1p transformed)
2. **CURRENT_YEAR_FAP** - Current year Full Annual Premium
3. **FUTURE_LIFETIME_VALUE** - Forward-looking value projection (log1p transformed)
4. **ACTUAL_LIFETIME_DURATION** - Customer tenure
5. **NUM_CROSS_SOLD_LY** - Cross-sell diversity (distinct products)
6. **CLM_OVER_PROFIT_HITCOUNT** - Claims-over-plan frequency

**Feature Engineering**: Each feature discretised into 7 quantile bins, then one-hot encoded → **26 binary features** for clustering.

### Explanatory Columns (Profiling Only)
CUSTOMER_SEGMENT, CUSTOMER_PORTFOLIO, ACTIVE_CUSTOMER, CURRENT_GWP, LAST_YEAR_GWP, GEP_FIN_FULL, GEC_FIN_FULL, CLM_INC_FULL, WEIGHTED_PLAN_LOSS_RATIO, ACTUAL_LOSS_RATIO, EXPECTED_LIFETIME_VALUE, TOTAL_SCORE

---

## Results

### Clustering Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Clusters Detected** | 3 | Simple, interpretable segmentation |
| **Silhouette Score** | 0.257 | **Fair** - Moderate cluster separation |
| **Calinski-Harabasz** | 36,768 | **Good** - Decent between/within cluster variance ratio |
| **Davies-Bouldin** | 1.919 | **Fair** - Some cluster overlap (lower is better) |

**Overall Assessment**: Trade-off between granularity and interpretability. Lower quality metrics than HDBSCAN but much simpler segmentation with balanced cluster sizes.

### Cluster Distribution

| Cluster | Count | Percentage | Interpretation |
|---------|-------|------------|----------------|
| **0** | 98,506 | 59.4% | Largest segment - Mid-value customers |
| **1** | 39,726 | 24.0% | Mid-size segment - Active cross-sellers |
| **2** | 27,519 | 16.6% | Smallest segment - High-value customers |
| **Total** | 165,751 | 100% | All customers assigned (no noise) |

**Note**: KMeans assigns every customer to a cluster (no outliers/noise unlike HDBSCAN).

### Cluster Profiles

#### Cluster 0 - "Mid-Value Inactive" (59.4%, 98,506 customers)
| Feature | Median | Mean | Notes |
|---------|--------|------|-------|
| **ACTUAL_CLTV** | £1,810 | £2,715 | Mid-range value |
| **CURRENT_YEAR_FAP** | £6,023 | £6,624 | Active premiums |
| **LIFETIME_DURATION** | 3.0 | 4.6 | Moderate tenure |
| **NUM_CROSS_SOLD_LY** | 0.0 | 0.08 | Low cross-sell |
| **CLM_OVER_PROFIT** | 0.0 | 0.23 | Occasional claims issues |
| **Product Index** | - | 0.17 | Low diversification |
| **Primary Segment** | F | MID | Majority F-rated |
| **Active Status** | N | - | **Mostly inactive** |

**Business Interpretation**: Inactive customers with mid-range value and low cross-sell potential. Focus on reactivation strategies.

#### Cluster 1 - "Active Cross-Sellers" (24.0%, 39,726 customers)
| Feature | Median | Mean | Notes |
|---------|--------|------|-------|
| **ACTUAL_CLTV** | £1,005 | £2,598 | Slightly lower value |
| **CURRENT_YEAR_FAP** | £1,155 | £1,982 | Lower premiums |
| **LIFETIME_DURATION** | 4.5 | 6.3 | **Higher tenure** |
| **NUM_CROSS_SOLD_LY** | 1.0 | 0.89 | **Strong cross-sell** |
| **CLM_OVER_PROFIT** | 0.0 | 0.09 | Low claims issues |
| **Product Index** | - | 1.86 | **High diversification** |
| **Primary Segment** | E | MID | E-rated customers |
| **Active Status** | Y | - | **Active customers** |

**Business Interpretation**: Active customers with strong cross-sell behavior. Prime targets for upsell and retention programs.

#### Cluster 2 - "High-Value Premium" (16.6%, 27,519 customers)
| Feature | Median | Mean | Notes |
|---------|--------|------|-------|
| **ACTUAL_CLTV** | £25,400 | £55,849 | **Very high value** |
| **CURRENT_YEAR_FAP** | £18,855 | £61,039 | **High premiums** |
| **LIFETIME_DURATION** | 6.2 | 7.6 | **Long tenure** |
| **NUM_CROSS_SOLD_LY** | 1.0 | 1.31 | Good cross-sell |
| **CLM_OVER_PROFIT** | 0.0 | 0.58 | Higher claims frequency |
| **Product Index** | - | 2.74 | **Very high diversification** |
| **Primary Segment** | D | MID | D-rated customers |
| **Active Status** | Y | - | **Active customers** |

**Business Interpretation**: High-value, long-tenure customers with excellent diversification. VIP treatment and retention focus.

---

## Comparison with HDBSCAN Approach

| Metric | Discretised KMeans (This) | Continuous HDBSCAN (out_csp) |
|--------|---------------------------|------------------------------|
| **Clusters** | 3 | 118 |
| **Silhouette** | 0.257 | 0.470 |
| **Calinski-Harabasz** | 36,768 | 538,431 |
| **Davies-Bouldin** | 1.919 | 0.941 |
| **Noise/Outliers** | 0% | 67% |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Granularity** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Trade-off Analysis**:
- **KMeans**: Simple, balanced segments with clear business interpretation. Lower quality metrics but 100% customer assignment.
- **HDBSCAN**: High-quality dense clusters but 67% noise. Better for finding niche segments, worse for broad strategies.

---

## Key Insights

### 1. Simple Segmentation Structure
- **3 clusters** provide actionable segmentation: Inactive Mid-Value, Active Cross-Sellers, High-Value Premium
- All customers assigned (no noise) - simpler operationalization
- Clear differentiation: 59% inactive, 24% active cross-sellers, 17% high-value

### 2. Cross-Sell Patterns
- Cluster 0: Product Index 0.17 (low diversification)
- Cluster 1: Product Index 1.86 (high diversification)
- Cluster 2: Product Index 2.74 (very high diversification)
- **Strong correlation between cross-sell and cluster assignment**

### 3. Value Distribution
- Cluster 2 captures the high-value tail (mean CLTV £55k vs £2.7k and £2.6k)
- 16.6% of customers contribute disproportionate value
- Median CLTV ranges from £1k to £25k across clusters

### 4. Activity Status
- Cluster 0: Mostly inactive (59% of portfolio) - reactivation opportunity
- Clusters 1 & 2: Active customers (41% of portfolio) - retention focus

### 5. Why k=3 Was Selected
- Composite scoring tested k=3 to k=12
- k=3 provided best balance of silhouette score, CH index, and DB index
- Higher k values didn't significantly improve metrics
- Business interpretability favors simpler segmentation

---

## Environment

### Execution Platform
- **Platform**: Windows 10/11 Local
- **Python**: 3.11.13
- **Environment**: uv venv
- **Runtime**: ~30 minutes (slower than Azure due to Windows overhead)

### Performance Notes
- Quantile binning created 26 binary features from 6 numeric features
- K-means grid search tested 10 different k values (3-12)
- Feature engineering (discretisation) was computationally intensive on Windows
- Progress logs visible (9-step pipeline)

### Reproducibility Command
```bash
.venv/Scripts/ukplc-seg.exe \
  --input ./data/CSP_export.csv \
  --outdir ./out_csp_discretised_kmeans \
  --recipe discretised \
  --algorithm kmeans \
  --n-bins 7 \
  --k-min 3 \
  --k-max 12 \
  --k-select composite \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --random-state 42
```

---

## Output Files

| File | Description |
|------|-------------|
| `cluster_assignments.csv` | All 165,751 customers with cluster labels (0, 1, 2) |
| `cluster_profiles.csv` | Statistics per cluster (medians, means, product_index) |
| `cluster_vs_existing_segment.csv` | Cross-tab: new clusters vs existing A/A+/B/C/D/E/F segments |
| `internal_metrics.json` | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| `explain.json` | SHAP feature importances (RandomForest surrogate) |
| `cluster_sizes.png` | Cluster size bar chart |
| `model.joblib` | Serialized pipeline for re-use (655KB) |
| `REPORT.md` | Basic summary report |

---

## Recommendations

### Immediate Actions
1. **Operationalize 3-segment strategy**: Map clusters to business actions (reactivate, upsell, retain)
2. **Profile Cluster 0** (59%): Understand why customers are inactive and design reactivation campaigns
3. **Protect Cluster 2** (17%): VIP retention programs for high-value segment

### Business Application

**Cluster 0 - Reactivation Strategy**:
- 98,506 customers (59%)
- Target: Reactivate inactive customers with mid-range value
- Tactics: Win-back campaigns, retention offers, re-engagement surveys

**Cluster 1 - Cross-Sell Strategy**:
- 39,726 customers (24%)
- Target: Active customers with strong cross-sell potential
- Tactics: Product bundling, cross-sell incentives, relationship management

**Cluster 2 - VIP Retention Strategy**:
- 27,519 customers (17%)
- Target: High-value customers with long tenure
- Tactics: Dedicated account management, premium services, loyalty programs

### Parameter Tuning
- **Alternative k values**: Try k=5 or k=7 for more granular segments
- **Fewer bins**: Reduce n-bins from 7 to 5 to simplify feature space
- **Alternative scoring**: Try silhouette-only k-selection for purer cluster quality

---

**End of Report**
