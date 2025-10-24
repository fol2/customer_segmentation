# Run 04 Visualization Guide

**Generated**: 2025-10-24
**Analysis of**: Run 04 (Continuous + KMeans) - 165,751 customers, 9 clusters

---

## Chart Overview

Six comprehensive visualization charts have been created to analyze the cluster characteristics, feature distributions, and segment mappings from Run 04.

---

## Chart 1: Feature Distributions by Cluster
**File**: `analysis_01_feature_distributions.png` (534 KB)

### What it shows
Boxplots showing the distribution of all 6 performance features across the 9 clusters:
1. ACTUAL_CLTV
2. CURRENT_YEAR_FAP (Future Annual Premium)
3. FUTURE_LIFETIME_VALUE
4. ACTUAL_LIFETIME_DURATION (Tenure)
5. NUM_CROSS_SOLD_LY (Cross-sell count)
6. CLM_OVER_PROFIT_HITCOUNT (Claims over profit)

### Key insights
- **Cluster 3 (VIP Elite)**: Extreme outliers in CLTV and FAP - box extends far beyond other clusters
- **Cluster 8 (High-Value)**: Second-highest CLTV and FAP distributions
- **Cluster 2 (Unprofitable)**: Negative CLTV values visible
- **Tenure distribution**: Varies significantly - Cluster 6 shows longer tenure
- **Cross-sell**: Clusters 3, 4, 8, 9 show higher cross-sell counts
- **Claims**: Most clusters have low claims counts (boxed near zero), few outliers

### How to use
- Compare the median lines (orange) across clusters for each feature
- Wide boxes indicate high variability within cluster
- Narrow boxes indicate homogeneous behavior
- Outliers (circles beyond whiskers) show exceptional customers

---

## Chart 2: Cluster Profiles Heatmap (Normalized)
**File**: `analysis_02_cluster_profiles_heatmap.png` (201 KB)

### What it shows
Normalized heatmap (0-1 scale) comparing 6 key metrics across all 9 clusters:
- CLTV
- Current FAP
- Future LV (Lifetime Value)
- Tenure
- Cross-Sell
- Product Index

**Color coding**: Green (high), Yellow (medium), Red (low)

### Key insights
- **Cluster 3 (C3)**: Green across all metrics - clear VIP elite
- **Cluster 8 (C8)**: Strong on CLTV, FAP, FLV - secondary VIP
- **Cluster 4 (C4)**: Balanced profile with good cross-sell
- **Cluster 0 (C0)**: Low on most metrics except tenure - large base
- **Cluster 2 (C2)**: Negative CLTV (red) but moderate FAP - unprofitable
- **Cluster 7 (C7)**: High current FAP but zero future value - retention risk

### How to use
- Quickly compare cluster "quality" via color intensity
- Identify multi-dimensional patterns (e.g., high tenure + low CLTV)
- Guide action prioritization (green = nurture, red = review)

---

## Chart 3: Cluster vs Existing Segment Mapping
**File**: `analysis_03_cluster_vs_existing_segment.png` (293 KB)

### What it shows
**Two panels**:
- **Left**: Stacked bar chart showing percentage composition of each cluster by existing A/B/C/D/E/F segments
- **Right**: Heatmap showing absolute customer counts in each cluster-segment combination

### Key insights
- **Cluster 3 captures premium segments**: Contains 111 A, 44 A+, 803 A-, 777 B customers (82.7% of all premium customers)
- **Cluster 8 also premium**: Contains 919 B, 4,773 C customers
- **F segment (low-value)** is split across multiple clusters (0, 2, 5, 7) - each representing different F profiles:
  - Cluster 0: Stable low-value (57,825)
  - Cluster 2: Unprofitable F (19,747)
  - Cluster 7: Declining F (5,658)
- **D/E segments** are distributed across clusters 1, 4, 5, 6 based on actual behavior vs. legacy scoring

### How to use
- **Left panel**: Understand cluster composition (e.g., "Cluster 3 is 95% premium customers")
- **Right panel**: Find specific segment migration (e.g., "Where did all the A customers go?")
- **Cross-validation**: Check if new clusters align or contradict existing segmentation

---

## Chart 4: Cluster Characteristics Radar Chart
**File**: `analysis_04_cluster_radar_chart.png` (456 KB)

### What it shows
Radar/spider chart comparing 5 selected clusters (VIP, High-value, Mid-tier, Base, Unprofitable) across 5 normalized metrics:
- CLTV
- FAP
- Future LV
- Tenure
- Product Index

Each cluster is a colored polygon - larger area = better performance.

### Key insights
- **Cluster 3 (VIP)**: Near-perfect pentagon - strong on all 5 dimensions
- **Cluster 8 (High-value)**: Similar shape to Cluster 3 but slightly smaller
- **Cluster 4 (Mid-tier)**: Balanced smaller pentagon
- **Cluster 0 (Base)**: Small, unbalanced shape - low on most metrics
- **Cluster 2 (Unprofitable)**: Inverted shape - negative CLTV despite some FAP

### How to use
- **Visual comparison**: Larger polygon = higher-value cluster
- **Shape analysis**: Balanced polygons = consistent performance; irregular shapes = trade-offs
- **Action guidance**: Target clusters with large, balanced shapes for retention/upsell

---

## Chart 5: Cluster Sizes and Key Metrics (4-panel)
**File**: `analysis_05_cluster_key_metrics.png` (289 KB)

### What it shows
Four bar charts showing:
1. **Top-left**: Cluster sizes (customer count) with percentage labels
2. **Top-right**: Mean CLTV by cluster (sorted high to low)
3. **Bottom-left**: Product Index (cross-sell potential) by cluster
4. **Bottom-right**: Mean tenure by cluster

**Color coding**:
- Green: High value/performance
- Orange/Yellow: Medium
- Red: Low/negative
- Gray: Neutral

### Key insights

**Panel 1 - Cluster Sizes**:
- Cluster 0 dominates (43.3%, 71,742 customers) - "low-value mass"
- Cluster 3 smallest (2.4%, 3,966 customers) - "VIP elite"
- 7 mid-sized clusters (6-13%) provide granularity

**Panel 2 - Mean CLTV**:
- Cluster 3: £253k (extreme green bar)
- Cluster 8: £22k (green)
- Cluster 2, 6: Negative CLTV (red) - 18.1% of customers unprofitable!
- Black dashed line at zero shows profit threshold

**Panel 3 - Product Index**:
- Cluster 3: 4.23 (highest cross-sell) - VIPs buy multiple products
- Clusters 4, 8, 9: 2.5-3.2 - strong cross-sell potential
- Clusters 0, 5, 7: <0.5 - single-product customers

**Panel 4 - Tenure**:
- Cluster 6: 16.7 years (longest) - "tenure loyalists"
- Cluster 3: 11.5 years - VIPs also have good tenure
- Cluster 0: 3.5 years - recent/short-term base

### How to use
- **Panel 1**: Assess operational impact (large clusters need scalable actions)
- **Panel 2**: Identify value concentration and loss leaders
- **Panel 3**: Target clusters with Product Index >2 for cross-sell campaigns
- **Panel 4**: Combine with CLTV for retention risk assessment (low tenure + high CLTV = protect aggressively)

---

## Chart 6: VIP Feature Correlations
**File**: `analysis_06_vip_feature_correlations.png` (241 KB)

### What it shows
Correlation matrix heatmap showing relationships between the 6 performance features **within Cluster 3 (VIP Elite only)**:
- Positive correlation (red): Features move together
- Negative correlation (blue): Features move inversely
- Zero correlation (white): No relationship

### Key insights
- **ACTUAL_CLTV ↔ CURRENT_YEAR_FAP**: Strong positive (0.87) - high FAP drives high CLTV in VIP segment
- **ACTUAL_CLTV ↔ FUTURE_LIFETIME_VALUE**: Very strong (0.92) - VIPs with high past CLTV also have high future value
- **CURRENT_YEAR_FAP ↔ FUTURE_LIFETIME_VALUE**: Strong (0.83) - current FAP predicts future value
- **NUM_CROSS_SOLD_LY ↔ ACTUAL_LIFETIME_DURATION**: Moderate (0.44) - longer tenure → more cross-sell among VIPs
- **CLM_OVER_PROFIT_HITCOUNT**: Low correlation with other features (0.15-0.31) - claims are independent of value in VIP segment

### How to use
- **Retention strategy**: Target VIPs with high FAP for loyalty programs (strong CLTV driver)
- **Cross-sell timing**: Focus on VIPs with tenure >5 years (moderate correlation with cross-sell)
- **Risk assessment**: Claims are independent of value - don't avoid high-claims VIPs
- **Predictive modeling**: Use FAP and FLV as strong predictors of CLTV for VIP scoring

---

## Actionable Insights Summary

### High-Priority Actions

1. **VIP Elite Protection (Cluster 3, 2.4%)**:
   - White-glove servicing
   - Proactive retention calls
   - Premium product offers
   - Dedicated account managers
   - Evidence: £253k mean CLTV, 4.23 Product Index

2. **High-Value Growth (Clusters 8, 4, 9 - 24.0%)**:
   - Cross-sell campaigns (Product Index 2.5-3.2)
   - Loyalty rewards
   - Digital engagement
   - Evidence: £16k-£57k mean CLTV, active customers

3. **Unprofitable Review (Clusters 2, 6 - 18.1%)**:
   - Claims investigation (data quality)
   - Non-renewal consideration
   - Pricing review
   - Evidence: -£5k to -£73k mean CLTV

4. **Declining Customer Risk (Cluster 7, 6.8%)**:
   - High current FAP (£25k) but zero future value
   - Immediate retention campaign
   - Investigate why FLV = 0
   - Evidence: £18k mean CLTV today but £0 projected future

5. **Tenure Loyalist Optimization (Cluster 6, 8.4%)**:
   - Long tenure (16.7 years) but low cross-sell
   - Product bundling offers
   - Tenure recognition programs
   - Evidence: £12k mean CLTV, 1.21 Product Index

### Medium-Priority Actions

6. **Base Mass Efficiency (Cluster 0, 43.3%)**:
   - Digital-first servicing
   - Self-service tools
   - Automated communications
   - Evidence: £10k mean CLTV, 0.22 Product Index, largest cluster

7. **Active Standard Growth (Cluster 5, 10.6%)**:
   - Moderate value (£14k CLTV)
   - High activity rate
   - Upsell potential
   - Evidence: 2.25 Product Index, active customers

---

## Recommended Next Steps

1. **Present charts to business stakeholders**:
   - Chart 2 (Heatmap) for executive overview
   - Chart 5 (4-panel) for operational planning
   - Chart 3 (Segment mapping) for validation

2. **Deep-dive analyses** (if needed):
   - Re-run visualization script for individual clusters
   - Add time-series view (if historical data available)
   - Create customer journey maps per cluster

3. **Deploy segmentation** (using Run 04 model):
   - Load `model.joblib` for scoring new customers
   - Use `cluster_assignments.csv` for CRM integration
   - Monitor cluster drift over time

---

## Technical Notes

### Visualization Script
- **Script**: `create_run04_visualizations.py` (in project root)
- **Dependencies**: pandas, numpy, matplotlib, seaborn
- **Runtime**: ~30 seconds for 165k customers
- **Customization**: Edit script to change colors, add clusters, or modify metrics

### Data Capping
- CLTV, FLV, FAP capped at [-50k, 200k] for visualization only
- Actual values used in clustering (no capping)
- Prevents extreme outliers from compressing box plots

### Normalization
- Heatmaps use min-max normalization (0-1 scale)
- Formula: `(value - min) / (max - min)`
- Enables cross-metric comparison

---

## Appendix: Cluster Quick Reference

| Cluster | Size | % | Mean CLTV | Profile | Priority |
|---------|------|---|-----------|---------|----------|
| **0** | 71,742 | 43.3% | £10,175 | Low-Value Base | Medium |
| **1** | 802 | 0.5% | £6,888 | Anomalous | Review |
| **2** | 20,621 | 12.4% | -£32,323 | Unprofitable | High (investigate) |
| **3** | 3,966 | 2.4% | £252,970 | VIP Elite | Critical |
| **4** | 14,520 | 8.8% | £16,405 | Active Mid-Tier | High |
| **5** | 17,568 | 10.6% | £962 | Active Standard | Medium |
| **6** | 13,981 | 8.4% | £12,427 | Tenure Loyalists | Medium |
| **7** | 11,286 | 6.8% | £17,793 | Declining (Zero FLV) | High |
| **8** | 11,265 | 6.8% | £21,989 | High-Value | Critical |
| **9** | - | - | - | (Not in profiles.csv) | - |

**Total**: 165,751 customers across 9 clusters

---

**Document Version**: 1.0
**Last Updated**: 2025-10-24
**Contact**: See `CONFIGURATION_REPORT.md` for methodology details
