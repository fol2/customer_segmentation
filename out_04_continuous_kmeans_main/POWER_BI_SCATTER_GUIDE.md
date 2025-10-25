# Power BI Scatter Plot & Distance Analysis Guide

This guide explains how to use the generated CSV datasets to create interactive scatter plots in Power BI for cluster distance analysis and customer identification.

---

## Generated Datasets

### 1. customer_scatter_data.csv (165,751 rows, 36 MB)
**Main dataset with customer-level scatter plot coordinates and distance metrics**

**Key Columns:**
- `CUSTOMER_ID`, `CLIENT_NAME` - Customer identifiers
- `cluster` - Assigned cluster (0-8)
- **PCA Coordinates:** `pca_x`, `pca_y` - 2D PCA projection (75.8% variance)
- **t-SNE Coordinates:** `tsne_x`, `tsne_y` - Non-linear 2D projection
- **Distance Metrics:**
  - `dist_assigned` - Euclidean distance to assigned cluster center
  - `dist_nearest_other` - Distance to nearest alternative cluster
  - `nearest_other_cluster` - ID of nearest alternative cluster
  - `rnd_assigned` - Radius-Normalized Distance to assigned cluster (lower = better, typically 0.5-1.5)
  - `rnd_nearest_other` - RND to nearest other cluster
  - `margin_r` - Safety margin (rnd_nearest_other - rnd_assigned). >0 = safe, <0 = boundary
  - `mahalanobis_first` - Diagonal Mahalanobis distance to assigned cluster
  - `mahalanobis_second` - Mahalanobis distance to nearest other cluster
  - `gap_euclid` - Euclidean distance gap (top-2)
  - `gap_mahal` - Mahalanobis distance gap (top-2)
  - `confidence_prob` - [0,1] probability from Soft K-means (recommended threshold: 0.6-0.7)
  - `alt_label_from_prob` - Alternative cluster label from soft probabilities
  - `silhouette_score` - Quality metric (-1 to 1, higher = better fit)
  - `is_marginal` - Boolean flag for borderline customers (32.7% of dataset)
- **Performance Features:** ACTUAL_CLTV, CURRENT_YEAR_FAP, FUTURE_LIFETIME_VALUE, etc.
- **Segments:** CUSTOMER_SEGMENT, CUSTOMER_PORTFOLIO, ACTIVE_CUSTOMER

### 2. cluster_centers.csv (9 rows, 1.9 KB)
**Cluster center coordinates for overlay on scatter plots**

**Columns:**
- `cluster` - Cluster ID (0-8)
- `pca_center_x`, `pca_center_y` - PCA space coordinates
- `tsne_center_x`, `tsne_center_y` - t-SNE space coordinates
- `mean_distance_to_center` - Average distance for customers in cluster
- `mean_confidence_prob` - Average confidence probability
- `mean_silhouette` - Average silhouette score

### 3. cluster_metadata.csv (9 rows, 1.2 KB)
**Cluster summary statistics for slicers and filters**

**Columns:**
- `cluster` - Cluster ID
- `customer_count` - Number of customers
- `mean_cltv`, `median_cltv`, `std_cltv` - CLTV statistics
- `mean_silhouette` - Average cluster quality
- `cluster_label` - Short descriptive name (e.g., "Low-Value Base", "VIP Elite")
- `cluster_display` - Full display name (e.g., "C0 - Low-Value Base")
- `marginal_count`, `marginal_pct` - Number and % of borderline customers

### 4. marginal_customers.csv (54,220 rows, 13 MB)
**Subset of borderline customers for model fine-tuning analysis**

Same columns as customer_scatter_data.csv, but filtered to `is_marginal = True` only.

---

## Power BI Setup

### Step 1: Import Data

1. Open Power BI Desktop
2. **Get Data > Text/CSV**
3. Import all 4 CSV files:
   - customer_scatter_data.csv
   - cluster_centers.csv
   - cluster_metadata.csv
   - marginal_customers.csv (optional, already in customer_scatter_data)

### Step 2: Create Relationships

In the **Model** view, create these relationships:

```
customer_scatter_data[cluster] --> cluster_metadata[cluster] (Many-to-One)
customer_scatter_data[cluster] --> cluster_centers[cluster] (Many-to-One)
```

---

## Visualization Examples

### 1. PCA Scatter Plot with Cluster Centers

**Visual Type:** Scatter Chart

**Setup:**
- **X-Axis:** customer_scatter_data[pca_x]
- **Y-Axis:** customer_scatter_data[pca_y]
- **Legend:** customer_scatter_data[cluster] or cluster_metadata[cluster_display]
- **Tooltips:** CLIENT_NAME, CUSTOMER_SEGMENT, dist_assigned, silhouette_score, confidence_prob, ACTUAL_CLTV
- **Size:** (Optional) ACTUAL_CLTV or CURRENT_GWP

**Add Cluster Centers Overlay:**
1. Duplicate the scatter chart
2. Add a second scatter layer using cluster_centers table:
   - X: cluster_centers[pca_center_x]
   - Y: cluster_centers[pca_center_y]
   - Change marker to **X** or **Diamond** with larger size
   - Use distinct color (e.g., black)

**Result:** Shows customer distribution in 2D PCA space with cluster centroids overlaid

---

### 2. t-SNE Scatter Plot with Marginal Customers Highlighted

**Visual Type:** Scatter Chart

**Setup:**
- **X-Axis:** customer_scatter_data[tsne_x]
- **Y-Axis:** customer_scatter_data[tsne_y]
- **Legend:** customer_scatter_data[is_marginal]
- **Color Saturation:** dist_assigned (darker = farther from center)
- **Tooltips:** CLIENT_NAME, cluster_display, confidence_prob, dist_assigned, dist_nearest_other, margin_r

**Conditional Formatting:**
- Marginal customers (is_marginal = True): Red/Orange
- Core customers (is_marginal = False): Blue/Green

**Result:** Identifies borderline customers sitting between clusters

---

### 3. Interactive Cluster Explorer

**Page Layout:**

**Top Section - Slicers:**
1. **Cluster Slicer:** cluster_metadata[cluster_display] (dropdown or tile)
2. **Customer Segment Slicer:** customer_scatter_data[CUSTOMER_SEGMENT]
3. **Portfolio Slicer:** customer_scatter_data[CUSTOMER_PORTFOLIO]
4. **Marginal Filter:** customer_scatter_data[is_marginal] (checkbox)

**Middle Section - Scatter Plot:**
- Main scatter (PCA or t-SNE) showing filtered customers
- Cluster centers overlay

**Bottom Section - Detail Cards:**
1. **Table Visual:** Top 10 customers by ACTUAL_CLTV in selected area
   - Columns: CLIENT_NAME, CUSTOMER_ID, cluster_display, ACTUAL_CLTV, dist_assigned, confidence_prob, margin_r
2. **Card Visuals:**
   - Count of selected customers
   - Average silhouette score
   - % Marginal customers

**Interaction:**
- Click cluster in slicer → scatter plot filters to that cluster
- Click customer in scatter → detail table highlights that row
- Search by CLIENT_NAME → scatter zooms to that customer

---

### 4. Distance Distribution Analysis

**Visual Type:** Histogram

**Setup:**
- **X-Axis (Bins):** customer_scatter_data[dist_assigned]
- **Y-Axis:** Count of customers
- **Legend:** cluster_metadata[cluster_display]

**Add Reference Lines:**
- Mean distance: Calculate `AVERAGE(customer_scatter_data[dist_assigned])`
- Median distance: Calculate `MEDIAN(customer_scatter_data[dist_assigned])`

**Result:** Shows how tightly customers are grouped around cluster centers

---

### 5. Confidence Probability Analysis

**Visual Type:** Scatter Chart

**Setup:**
- **X-Axis:** customer_scatter_data[dist_assigned]
- **Y-Axis:** customer_scatter_data[confidence_prob]
- **Legend:** customer_scatter_data[cluster]
- **Size:** ACTUAL_CLTV
- **Tooltips:** CLIENT_NAME, cluster_display, dist_nearest_other, margin_r, silhouette_score

**Add Quadrant Lines:**
- Vertical line at mean dist_assigned
- Horizontal line at confidence_prob = 0.6 or 0.7 (recommended marginal threshold)

**Quadrants:**
- **Top-Left:** High confidence, close to center (ideal)
- **Top-Right:** High confidence, far from center (distinct outliers)
- **Bottom-Left:** Low confidence, close to center (boundary)
- **Bottom-Right:** Low confidence, far from center (misclassified?)

**Additional Analysis:**
- Color by `margin_r` to show safety margin (red = margin_r < 0, green = margin_r > 0)
- Filter to customers with `confidence_prob < 0.7` to focus on marginal assignments

**Result:** Identifies customers that may need cluster reassignment

---

## Key Metrics to Create

### DAX Measures

```dax
// Total Customers
Total Customers = COUNTROWS(customer_scatter_data)

// Marginal Customer %
Marginal % =
DIVIDE(
    COUNTROWS(FILTER(customer_scatter_data, customer_scatter_data[is_marginal] = TRUE)),
    COUNTROWS(customer_scatter_data)
) * 100

// Average Distance to Center
Avg Distance = AVERAGE(customer_scatter_data[dist_assigned])

// Average Confidence Probability
Avg Confidence = AVERAGE(customer_scatter_data[confidence_prob])

// Average Safety Margin
Avg Margin = AVERAGE(customer_scatter_data[margin_r])

// Average Silhouette Score
Avg Silhouette = AVERAGE(customer_scatter_data[silhouette_score])

// Cluster Quality Rating
Cluster Quality =
SWITCH(TRUE(),
    [Avg Silhouette] >= 0.5, "Excellent",
    [Avg Silhouette] >= 0.3, "Good",
    [Avg Silhouette] >= 0.1, "Fair",
    "Poor"
)

// Selected Customer Details (for tooltip)
Selected Customer =
IF(
    HASONEVALUE(customer_scatter_data[CLIENT_NAME]),
    "Customer: " & SELECTEDVALUE(customer_scatter_data[CLIENT_NAME]) &
    UNICHAR(10) & "Cluster: " & SELECTEDVALUE(customer_scatter_data[cluster]) &
    UNICHAR(10) & "Distance: " & FORMAT(SELECTEDVALUE(customer_scatter_data[dist_assigned]), "0.00") &
    UNICHAR(10) & "Confidence: " & FORMAT(SELECTEDVALUE(customer_scatter_data[confidence_prob]), "0.00") &
    UNICHAR(10) & "Margin: " & FORMAT(SELECTEDVALUE(customer_scatter_data[margin_r]), "0.00"),
    "Select a customer"
)
```

---

## Use Cases

### Use Case 1: Find Specific Company in Scatter Plot

**Goal:** Identify where "FIRMDALE HOTELS PLC" sits in the cluster space

**Steps:**
1. Add a **Search** slicer for CLIENT_NAME
2. Type "FIRMDALE" in search box
3. Scatter plot highlights the customer's position
4. Tooltip shows:
   - Cluster: C3 (VIP Elite)
   - Distance to center: 4.30
   - Confidence probability: 0.65 (moderate)
   - Safety margin: 0.12 (positive, safe)
   - Silhouette: 0.287 (fair fit)

**Insight:** Customer is in the VIP cluster with moderate confidence (0.65), but has a positive safety margin (0.12), indicating a stable assignment despite being moderately far from center.

---

### Use Case 2: Identify Marginal Customers for Review

**Goal:** Find customers sitting on cluster boundaries for potential reclassification

**Steps:**
1. Set `is_marginal = True` in slicer
2. View scatter plot colored by `nearest_other_cluster`
3. Filter to high-value customers: `ACTUAL_CLTV > 500000`
4. Export list to CSV for manual review

**Insight:** 54,220 customers (32.7%) are marginal. High-value marginal customers may benefit from tailored strategies. Marginal detection now uses: confidence_prob < p20[cluster] OR silhouette < 0.10 OR margin_r < 0.

---

### Use Case 3: Compare Distance Across Clusters

**Goal:** Understand which clusters are tightly vs loosely grouped

**Steps:**
1. Create bar chart: X = cluster_display, Y = mean_distance_to_center (from cluster_centers)
2. Add error bars using std deviation
3. Sort by mean distance descending

**Expected Result:**
- **Tightest clusters:** C1 (Anomalous), C3 (VIP Elite) - homogeneous groups
- **Loosest clusters:** C0 (Low-Value Base) - diverse, large segment

---

### Use Case 4: Drill-Down from Cluster to Customer

**Goal:** Explore customers within a specific cluster

**Steps:**
1. Click cluster in slicer (e.g., "C3 - VIP Elite")
2. Scatter plot filters to show only C3 customers
3. Table visual shows all C3 customers sorted by dist_assigned
4. Identify customers farthest from center (potential outliers)
5. Check margin_r values to distinguish true outliers (margin_r > 0) from boundary cases (margin_r < 0)

**Insight:** Customers far from cluster center may have unique characteristics worth investigating.

---

## Tips for Analysis

1. **Use t-SNE for visual exploration** - Better cluster separation than PCA
2. **Use PCA for geometric distances** - Preserves actual distances and variance
3. **Filter marginal customers separately** - Analyze core vs boundary customers differently
4. **Cross-reference with existing segments** - Compare cluster assignments to CUSTOMER_SEGMENT
5. **Monitor confidence_prob** - Values < 0.6-0.7 indicate ambiguous assignments (recommended threshold)
6. **Check margin_r for boundary detection** - Negative values indicate customer sits on cluster boundary
7. **Use rnd_assigned for normalized comparisons** - Accounts for cluster size when comparing distances
8. **Size by CLTV** - Identify high-value customers in scatter plots
9. **Highlight nearest_other_cluster** - Shows which clusters are adjacent/overlapping

---

## Troubleshooting

**Q: Scatter plot is too crowded**
- A: Apply filters (portfolio, segment, marginal flag) or sample data using DAX

**Q: Can't find a specific customer**
- A: Use search slicer on CLIENT_NAME or CUSTOMER_ID

**Q: Cluster centers not showing**
- A: Check relationship is correct and cluster_centers layer uses X marker with size 100+

**Q: Colors don't match between visuals**
- A: Set conditional formatting manually to match cluster IDs to consistent colors

**Q: Performance is slow with 165k rows**
- A: Enable "Sample data" in scatter plot settings or filter to subset

---

## Summary Statistics

**Dataset Overview:**
- Total customers: 165,751
- Clusters: 9 (C0-C8)
- Marginal customers: 54,220 (32.7%)
- Mean silhouette score: 0.386 (Fair clustering quality)
- PCA variance explained: 75.8%
- Mean confidence probability: 0.917
- Median confidence probability: 0.972

**Cluster Breakdown (see cluster_metadata.csv):**
- C0 (Low-Value Base): 71,742 customers (43.3%)
- C1 (Anomalous): 802 customers (0.5%)
- C2 (Unprofitable): ~15k customers
- C3 (VIP Elite): ~8k customers
- ...and 5 more clusters

---

## Next Steps

1. **Validate Marginal Customers:** Review high-value marginal customers for potential reclassification
2. **Refine Model:** Consider adjusting feature engineering or k value based on distance distributions
3. **Business Rules:** Use scatter plots to define business rules for segment transitions
4. **Monitor Over Time:** Track customer movement in scatter space across model versions

---

For questions or issues, refer to the main project documentation in CLAUDE.md.
