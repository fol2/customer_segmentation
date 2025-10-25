# Scatter Plot & Distance Analysis - Summary

**Generated:** October 24, 2025
**Run Directory:** out_04_continuous_kmeans_main/
**Total Customers:** 165,751
**Clusters:** 9 (K=9 via silhouette selection)

---

## Generated Outputs

### Visualization Charts (5 PNG files)

1. **distance_01_pca_scatter.png** (497 KB)
   - 2D PCA scatter plot with cluster centers overlaid
   - Shows customer distribution in principal component space
   - PCA explains 75.8% of variance

2. **distance_02_tsne_scatter_marginal.png** (1.6 MB)
   - t-SNE scatter plot with marginal customers highlighted
   - Non-linear dimensionality reduction for visual cluster separation
   - Red points = marginal customers (30.8%)

3. **distance_03_distance_distributions.png** (174 KB)
   - 4-panel histogram showing:
     - Distance to assigned cluster center
     - Distance to nearest other cluster
     - Confidence ratio distribution
     - Silhouette score distribution

4. **distance_04_silhouette_plot.png** (104 KB)
   - Silhouette plot by cluster
   - Shows cluster cohesion and separation
   - Wider bars = better cluster quality

5. **distance_05_feature_pair_scatter.png** (941 KB)
   - 6-panel scatter matrix of key performance features
   - Shows natural cluster separation in original feature space
   - Pairs: ACTUAL_CLTV, CURRENT_YEAR_FAP, FUTURE_LIFETIME_VALUE

### Power BI Datasets (4 CSV files)

1. **customer_scatter_data.csv** (36 MB, 165,751 rows)
   - **Main dataset** with full customer-level data
   - Coordinates: pca_x, pca_y, tsne_x, tsne_y
   - Distance metrics: distance_to_center, distance_to_nearest_other, confidence_ratio, silhouette_score
   - Identifiers: CUSTOMER_ID, CLIENT_NAME, cluster
   - Flags: is_marginal, nearest_other_cluster
   - Performance features: All 6 CSP inputs
   - Segments: CUSTOMER_SEGMENT, CUSTOMER_PORTFOLIO, ACTIVE_CUSTOMER, CURRENT_GWP, TOTAL_SCORE

2. **cluster_centers.csv** (1.9 KB, 9 rows)
   - Cluster center coordinates in both PCA and t-SNE space
   - Mean distance and confidence statistics per cluster
   - Use for overlay on scatter plots

3. **cluster_metadata.csv** (1.2 KB, 9 rows)
   - Cluster summary statistics for slicers
   - Customer counts, CLTV statistics, silhouette scores
   - Cluster labels: "Low-Value Base", "Anomalous", "VIP Elite", etc.
   - Marginal customer counts and percentages per cluster

4. **marginal_customers.csv** (12 MB, 51,014 rows)
   - Subset of customer_scatter_data filtered to is_marginal = True
   - Customers sitting on cluster boundaries
   - Candidates for model fine-tuning or manual review

---

## Key Statistics

### Overall Clustering Quality
- **Mean Silhouette Score:** 0.386 (Fair quality)
- **Mean Distance to Center:** 1.15
- **Mean Distance to Nearest Other:** 2.48
- **Mean Confidence Ratio:** 2.81 (customers ~2.8x farther from nearest other cluster than their own)

### Marginal Customer Identification
- **Total Marginal Customers:** 51,014 (30.8% of dataset)
- **Marginal Definition:**
  - Silhouette score < 0.2 OR
  - Confidence ratio < 1.5
- **Interpretation:** ~31% of customers sit on cluster boundaries and may benefit from review

### Dimensionality Reduction
- **PCA Variance Explained:** 75.8% (2 components capture most variation)
- **t-SNE Perplexity:** 30 (default, optimized for 165k samples)

---

## Cluster Breakdown (Top 3 by size)

| Cluster | Label | Count | % | Mean CLTV | Mean Silhouette | Marginal % |
|---------|-------|-------|---|-----------|-----------------|------------|
| C0 | Low-Value Base | 71,742 | 43.3% | £10,175 | 0.486 | 19.9% |
| C1 | Anomalous | 802 | 0.5% | £6,888 | 0.747 | 4.0% |
| C2 | Unprofitable | ~15,000 | 9% | - | - | - |

*(Full breakdown in cluster_metadata.csv)*

---

## Use Cases

### 1. Identify Specific Customer Location
**Example:** Find "FIRMDALE HOTELS PLC" in scatter plot
- **Cluster:** C3 (VIP Elite)
- **PCA Position:** (9.01, 3.57)
- **t-SNE Position:** (122.57, 21.69)
- **Distance to Center:** 4.30 (moderate)
- **Confidence Ratio:** 1.58 (borderline - closest to C8)
- **Silhouette:** 0.287 (fair fit)

### 2. Review Marginal Customers
- 51,014 customers flagged as marginal
- Filter by high ACTUAL_CLTV to prioritize
- Use marginal_customers.csv for batch export

### 3. Compare Cluster Distances
- Use cluster_centers.csv to identify tight vs loose clusters
- C1 (Anomalous): Very tight, high confidence
- C0 (Low-Value Base): Looser, more diverse

### 4. Monitor Model Quality
- Track silhouette scores over time
- Identify clusters with low mean silhouette (<0.3)
- Flag customers with confidence ratio < 1.5

---

## Power BI Setup

**Quick Start:**
1. Import all 4 CSV files into Power BI
2. Create relationships:
   - customer_scatter_data[cluster] → cluster_metadata[cluster]
   - customer_scatter_data[cluster] → cluster_centers[cluster]
3. Create scatter chart:
   - X: pca_x or tsne_x
   - Y: pca_y or tsne_y
   - Legend: cluster or cluster_display
   - Tooltip: CLIENT_NAME, distance_to_center, confidence_ratio
4. Add slicers: cluster_display, CUSTOMER_SEGMENT, is_marginal
5. Add cluster centers overlay using cluster_centers table

**Full Guide:** See POWER_BI_SCATTER_GUIDE.md

---

## Distance Interpretation

### Distance to Center
- **0-1:** Core customer, very typical of cluster
- **1-2:** Normal customer, fits cluster well
- **2-3:** Peripheral customer, less typical
- **3+:** Outlier, may be misclassified

### Confidence Ratio
- **>3.0:** High confidence, clearly belongs to cluster
- **2.0-3.0:** Moderate confidence, reasonably placed
- **1.5-2.0:** Low confidence, near boundary (marginal)
- **<1.5:** Very low confidence, almost equidistant to multiple clusters

### Silhouette Score
- **0.7-1.0:** Excellent fit, cluster is well-separated
- **0.5-0.7:** Good fit, cluster is distinct
- **0.3-0.5:** Fair fit, some overlap with other clusters
- **0.0-0.3:** Poor fit, customer may belong elsewhere
- **<0.0:** Likely misclassified

---

## Next Steps

1. **Review High-Value Marginals**
   - Filter marginal_customers.csv for ACTUAL_CLTV > £500,000
   - Examine which clusters they're between (nearest_other_cluster)
   - Consider business implications of cluster boundaries

2. **Validate Cluster Labels**
   - Use scatter plots to visually confirm cluster separation
   - Compare to existing CUSTOMER_SEGMENT assignments
   - Update cluster labels if needed (Low-Value Base, VIP Elite, etc.)

3. **Model Fine-Tuning**
   - If >30% marginal seems high, consider:
     - Adjusting k (try k=7 or k=11)
     - Different distance metrics
     - Re-weighting features
   - Re-run with adjusted thresholds for marginal flag

4. **Business Integration**
   - Use scatter plots in stakeholder presentations
   - Set up Power BI dashboard for ongoing monitoring
   - Define business rules for segment transitions

---

## Files Location

All outputs in: `out_04_continuous_kmeans_main/`

**Charts:**
- distance_01_pca_scatter.png
- distance_02_tsne_scatter_marginal.png
- distance_03_distance_distributions.png
- distance_04_silhouette_plot.png
- distance_05_feature_pair_scatter.png

**Datasets:**
- customer_scatter_data.csv
- cluster_centers.csv
- cluster_metadata.csv
- marginal_customers.csv

**Documentation:**
- POWER_BI_SCATTER_GUIDE.md (detailed usage guide)
- DISTANCE_ANALYSIS_SUMMARY.md (this file)

---

**Analysis Complete!** All scatter plots and distance datasets are ready for Power BI integration.
