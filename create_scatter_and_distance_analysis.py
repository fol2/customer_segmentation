#!/usr/bin/env python3
"""
Generate scatter plots with distance analysis and Power BI datasets.

This script creates:
1. Multiple scatter plot visualizations showing cluster separation and distances
2. Customer-level data with coordinates, distances, and confidence metrics
3. Marginal/borderline customer identification
4. Power BI-ready datasets for interactive exploration

Outputs:
- 5 scatter plot visualization charts
- customer_scatter_data.csv (for Power BI - customer level)
- cluster_centers.csv (for Power BI - cluster centers)
- cluster_metadata.csv (for Power BI - slicers)
- marginal_customers.csv (borderline customers requiring review)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples, pairwise_distances
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("./data")
RUN_DIR = Path("./out_04b_mbk_whiten")
OUTPUT_DIR = RUN_DIR

print("="*80)
print("SCATTER PLOT & DISTANCE ANALYSIS")
print("="*80)

# ============================================================================
# 1. Load Data and Model
# ============================================================================
print("\n[1/7] Loading data and model...")

df_original = pd.read_csv(DATA_DIR / "CSP_export.csv")
df_clusters = pd.read_csv(RUN_DIR / "cluster_assignments.csv")
df = df_original.merge(df_clusters, on="CUSTOMER_ID", how="inner")

# Load the trained model
model_artifacts = joblib.load(RUN_DIR / "model.joblib")
pipeline = model_artifacts['pipeline']
kmeans_model = model_artifacts['cluster_model']
cluster_centers = kmeans_model.cluster_centers_

print(f"  Loaded {len(df):,} customers")
print(f"  Clusters: {sorted(df['cluster'].unique())}")
print(f"  K-means centers shape: {cluster_centers.shape}")

# Performance features
perf_features = [
    'ACTUAL_CLTV',
    'CURRENT_YEAR_FAP',
    'FUTURE_LIFETIME_VALUE',
    'ACTUAL_LIFETIME_DURATION',
    'NUM_CROSS_SOLD_LY',
    'CLM_OVER_PROFIT_HITCOUNT'
]

# Extract feature matrix
X_raw = df[perf_features].values
labels = df['cluster'].values

# ============================================================================
# 2. Transform Data (same as model training)
# ============================================================================
print("\n[2/7] Transforming features...")

# Apply the trained pipeline transformation
X_transformed = pipeline.transform(df[perf_features])

print(f"  Original features: {X_raw.shape[1]}")
print(f"  Transformed features: {X_transformed.shape[1]}")

# ============================================================================
# 3. Calculate Distances and Confidence Metrics (Vectorised, Calibrated)
# ============================================================================
print("\n[3/7] Calculating distances and confidence metrics...")

from numpy.linalg import norm

# Convert to float32 for memory efficiency
X = X_transformed.astype("float32")
C = kmeans_model.cluster_centers_.astype("float32")
n, k = X.shape[0], C.shape[0]

print(f"  Computing {n:,} × {k} distance matrix (vectorised)...")

# (n,k) squared Euclidean distances in transformed space
# Using broadcasting for speed
diff = X[:, None, :] - C[None, :, :]
d2 = (diff ** 2).sum(axis=2)                       # (n, k)
first_idx = labels
first_d2 = d2[np.arange(n), first_idx]

# nearest-other cluster indices & distances
# argpartition: O(k) selection for 2 smallest
order2 = np.argpartition(d2, 2, axis=1)[:, :2]     # indices of 2 smallest per row (unsorted)
# ensure first is the assigned cluster; pick the other as second
second_idx = np.where(order2[:, 0] == first_idx, order2[:, 1], order2[:, 0])
second_d2 = d2[np.arange(n), second_idx]

print("  (1) Computing radius-normalised distances & margin...")
# --- (1) radius-normalised distances & margin ---
# robust radius per cluster: median of sqrt distance of in-cluster points
radii = np.zeros(k, dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    if np.any(idx):
        radii[c] = np.median(np.sqrt(d2[idx, c]))
    else:
        radii[c] = np.median(np.sqrt(first_d2))  # fallback
eps = 1e-6

d1_r = np.sqrt(first_d2) / (radii[first_idx] + eps)
d2_r = np.sqrt(second_d2) / (radii[second_idx] + eps)
margin_r = d2_r - d1_r  # >0 is better (inside cluster core relatively)

print("  (2) Computing diagonal Mahalanobis distances...")
# --- (2) diagonal Mahalanobis distances ---
# per-cluster diagonal variance (regularised)
variances = np.zeros((k, X.shape[1]), dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    if np.any(idx):
        variances[c] = X[idx].var(axis=0, ddof=1) + 1e-6
    else:
        variances[c] = X.var(axis=0, ddof=1) + 1e-6

d2_mahal = ((diff ** 2) / variances[None, :, :]).sum(axis=2)  # (n,k)
d2M_first = d2_mahal[np.arange(n), first_idx]
d2M_second = d2_mahal[np.arange(n), second_idx]

print("  (3) Computing Soft K-means probabilities (calibrated)...")
# --- (3) Soft K-means probabilities (calibrated) ---
tau = 1.0  # temperature
logits = - d2_mahal / (2.0 * tau)
# subtract row max for numerical stability (log-sum-exp trick)
logits = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(logits)
probs = probs / probs.sum(axis=1, keepdims=True)
confidence = probs.max(axis=1).astype("float32")
alt_label = probs.argmax(axis=1).astype(int)       # soft-argmax (usually=labels)

print("  (4) Computing geometric gap (top-2)...")
# --- (4) geometric gap (top-2) ---
gap_euclid = np.sqrt(second_d2) - np.sqrt(first_d2)
gap_mahal  = np.sqrt(d2M_second) - np.sqrt(d2M_first)

# Calculate silhouette scores
print("  Computing silhouette scores...")
silhouette_scores = silhouette_samples(X_transformed, labels)

print("  Applying cluster-adaptive marginal detection rule...")
# === New marginal rule (cluster-adaptive) ===
# per-cluster 20th percentile of confidence + global silhouette threshold
conf_p20 = np.zeros(k, dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    conf_p20[c] = np.percentile(confidence[idx], 20) if np.any(idx) else 0.5

is_marginal = (confidence < conf_p20[first_idx]) | (silhouette_scores < 0.10) | (margin_r < 0)

# Store old distance metrics for comparison
distances_to_center = np.sqrt(first_d2)
distances_to_nearest_other = np.sqrt(second_d2)
nearest_other_cluster = second_idx

print(f"\n  Distance Metrics:")
print(f"    Mean Euclidean distance to assigned: {distances_to_center.mean():.2f}")
print(f"    Mean Euclidean distance to nearest other: {distances_to_nearest_other.mean():.2f}")
print(f"    Mean radius-normalised distance (RND): {d1_r.mean():.2f}")
print(f"    Mean margin (RND): {margin_r.mean():.2f}")
print(f"\n  Confidence Metrics:")
print(f"    Mean confidence (soft K-means): {confidence.mean():.3f}")
print(f"    Median confidence: {np.median(confidence):.3f}")
print(f"    Mean Mahalanobis distance to assigned: {np.sqrt(d2M_first).mean():.2f}")
print(f"    Mean gap (Euclidean): {gap_euclid.mean():.2f}")
print(f"    Mean gap (Mahalanobis): {gap_mahal.mean():.2f}")
print(f"\n  Quality Metrics:")
print(f"    Mean silhouette score: {silhouette_scores.mean():.3f}")
print(f"    Marginal customers (new rule): {is_marginal.sum():,} ({is_marginal.sum()/len(df)*100:.1f}%)")
print(f"    Criteria: confidence < p20[cluster] OR silhouette < 0.10 OR margin_r < 0")

# ============================================================================
# 4. Dimensionality Reduction for Visualization
# ============================================================================
print("\n[4/7] Performing dimensionality reduction...")

# PCA (faster, linear)
print("  PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_transformed)
print(f"    Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# t-SNE (slower, non-linear, better for visualization)
print("  t-SNE (this may take 1-2 minutes)...")
# Use perplexity based on dataset size
perplexity = min(30, len(df) // 4)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
X_tsne = tsne.fit_transform(X_transformed)
print("    Done!")

# Calculate cluster centers in reduced space
centers_pca = np.zeros((len(cluster_centers), 2))
centers_tsne = np.zeros((len(cluster_centers), 2))
for c in range(len(cluster_centers)):
    mask = labels == c
    centers_pca[c] = X_pca[mask].mean(axis=0)
    centers_tsne[c] = X_tsne[mask].mean(axis=0)

# ============================================================================
# 5. Create Visualizations
# ============================================================================
print("\n[5/7] Creating visualizations...")

# -------------------------
# Chart 1: PCA Scatter Plot with Cluster Centers
# -------------------------
print("  Chart 1: PCA scatter plot...")
fig, ax = plt.subplots(figsize=(14, 10))

# Plot all points
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=labels, cmap='tab10',
                    s=20, alpha=0.6, edgecolors='none')

# Plot cluster centers
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
          c='red', marker='X', s=300, edgecolors='black', linewidths=2,
          label='Cluster Centers', zorder=100)

# Annotate centers with cluster numbers
for c in range(len(cluster_centers)):
    ax.annotate(f'C{c}', (centers_pca[c, 0], centers_pca[c, 1]),
               fontsize=12, fontweight='bold', color='white',
               ha='center', va='center', zorder=101)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster', fontsize=12)

ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('Cluster Visualization (PCA Projection)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_01_pca_scatter.png", dpi=150, bbox_inches='tight')
print("    Saved: distance_01_pca_scatter.png")
plt.close()

# -------------------------
# Chart 2: t-SNE Scatter Plot with Marginal Customers Highlighted
# -------------------------
print("  Chart 2: t-SNE scatter plot with marginal customers...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Left: All customers colored by cluster
scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=labels, cmap='tab10',
                      s=20, alpha=0.6, edgecolors='none')
ax1.scatter(centers_tsne[:, 0], centers_tsne[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Cluster Centers', zorder=100)
for c in range(len(cluster_centers)):
    ax1.annotate(f'C{c}', (centers_tsne[c, 0], centers_tsne[c, 1]),
                fontsize=12, fontweight='bold', color='white',
                ha='center', va='center', zorder=101)
ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax1.set_title('t-SNE Visualization (All Customers)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Right: Marginal customers highlighted
# Plot non-marginal in gray
ax2.scatter(X_tsne[~is_marginal, 0], X_tsne[~is_marginal, 1],
           c='lightgray', s=20, alpha=0.3, edgecolors='none', label='Confident')
# Plot marginal in red
ax2.scatter(X_tsne[is_marginal, 0], X_tsne[is_marginal, 1],
           c=labels[is_marginal], cmap='tab10',
           s=30, alpha=0.8, edgecolors='red', linewidths=0.5, label='Marginal')
ax2.scatter(centers_tsne[:, 0], centers_tsne[:, 1],
           c='blue', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Cluster Centers', zorder=100)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax2.set_title(f'Marginal Customers ({is_marginal.sum():,} / {len(df):,})', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_02_tsne_scatter_marginal.png", dpi=150, bbox_inches='tight')
print("    Saved: distance_02_tsne_scatter_marginal.png")
plt.close()

# -------------------------
# Chart 3: Distance Distributions
# -------------------------
print("  Chart 3: Distance distributions...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Distance to assigned cluster center
ax1.hist(distances_to_center, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(distances_to_center.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances_to_center.mean():.2f}')
ax1.axvline(np.median(distances_to_center), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(distances_to_center):.2f}')
ax1.set_xlabel('Distance to Assigned Cluster Center', fontsize=11)
ax1.set_ylabel('Customer Count', fontsize=11)
ax1.set_title('Distribution of Distance to Assigned Cluster', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Distance to nearest other cluster
ax2.hist(distances_to_nearest_other, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(distances_to_nearest_other.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances_to_nearest_other.mean():.2f}')
ax2.axvline(np.median(distances_to_nearest_other), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(distances_to_nearest_other):.2f}')
ax2.set_xlabel('Distance to Nearest Other Cluster', fontsize=11)
ax2.set_ylabel('Customer Count', fontsize=11)
ax2.set_title('Distribution of Distance to Nearest Other Cluster', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Confidence probability (soft K-means)
ax3.hist(confidence, bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.axvline(confidence.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidence.mean():.3f}')
ax3.axvline(np.median(confidence), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(confidence):.3f}')
ax3.axvline(0.6, color='purple', linestyle='--', linewidth=2, label='Suggested Threshold: 0.6')
ax3.set_xlabel('Confidence Probability (Soft K-means)', fontsize=11)
ax3.set_ylabel('Customer Count', fontsize=11)
ax3.set_title('Confidence Probability Distribution [0,1]', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Silhouette scores
ax4.hist(silhouette_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(silhouette_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {silhouette_scores.mean():.3f}')
ax4.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='Marginal Threshold: 0.10')
ax4.set_xlabel('Silhouette Score', fontsize=11)
ax4.set_ylabel('Customer Count', fontsize=11)
ax4.set_title('Silhouette Score Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_03_distance_distributions.png", dpi=150, bbox_inches='tight')
print("    Saved: distance_03_distance_distributions.png")
plt.close()

# -------------------------
# Chart 4: Silhouette Plot by Cluster
# -------------------------
print("  Chart 4: Silhouette plot by cluster...")
fig, ax = plt.subplots(figsize=(12, 10))

y_lower = 10
cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_centers)))

for c in sorted(np.unique(labels)):
    # Get silhouette scores for this cluster
    cluster_silhouette_scores = silhouette_scores[labels == c]
    cluster_silhouette_scores.sort()

    size_cluster = cluster_silhouette_scores.shape[0]
    y_upper = y_lower + size_cluster

    color = cluster_colors[c]
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_scores,
                     facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster, f'C{c}', fontsize=11, fontweight='bold')

    y_lower = y_upper + 10

# Add vertical line for average silhouette score
ax.axvline(x=silhouette_scores.mean(), color="red", linestyle="--", linewidth=2,
          label=f'Average: {silhouette_scores.mean():.3f}')
ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.set_title("Silhouette Plot by Cluster", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_04_silhouette_plot.png", dpi=150, bbox_inches='tight')
print("    Saved: distance_04_silhouette_plot.png")
plt.close()

# -------------------------
# Chart 5: Feature Pair Scatter Plots (2x2 grid)
# -------------------------
print("  Chart 5: Feature pair scatter plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Key Feature Pair Scatter Plots', fontsize=16, fontweight='bold', y=0.995)

# Cap extreme values for visualization
df_plot = df.copy()
for col in perf_features:
    cap = df_plot[col].quantile(0.99)
    df_plot[col] = df_plot[col].clip(upper=cap)

# Feature pairs to plot
pairs = [
    ('ACTUAL_CLTV', 'CURRENT_YEAR_FAP'),
    ('ACTUAL_CLTV', 'FUTURE_LIFETIME_VALUE'),
    ('ACTUAL_LIFETIME_DURATION', 'NUM_CROSS_SOLD_LY'),
    ('CURRENT_YEAR_FAP', 'FUTURE_LIFETIME_VALUE')
]

for idx, (feat1, feat2) in enumerate(pairs):
    ax = axes[idx // 2, idx % 2]

    # Plot all points
    scatter = ax.scatter(df_plot[feat1], df_plot[feat2],
                        c=labels, cmap='tab10',
                        s=15, alpha=0.5, edgecolors='none')

    # Highlight marginal customers
    ax.scatter(df_plot[is_marginal][feat1], df_plot[is_marginal][feat2],
              facecolors='none', edgecolors='red', s=30, linewidths=1.5,
              alpha=0.6, label='Marginal')

    ax.set_xlabel(feat1.replace('_', ' '), fontsize=10)
    ax.set_ylabel(feat2.replace('_', ' '), fontsize=10)
    ax.set_title(f'{feat1.replace("_", " ")} vs {feat2.replace("_", " ")}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_05_feature_pair_scatter.png", dpi=150, bbox_inches='tight')
print("    Saved: distance_05_feature_pair_scatter.png")
plt.close()

# ============================================================================
# 6. Create Power BI Datasets
# ============================================================================
print("\n[6/7] Creating Power BI datasets...")

# Dataset 1: Customer-level scatter data (with calibrated confidence metrics)
print("  Dataset 1: customer_scatter_data.csv...")
customer_scatter = pd.DataFrame({
    'CUSTOMER_ID': df['CUSTOMER_ID'],
    'CLIENT_NAME': df['CLIENT_NAME'],
    'cluster': labels,
    'pca_x': X_pca[:, 0],
    'pca_y': X_pca[:, 1],
    'tsne_x': X_tsne[:, 0],
    'tsne_y': X_tsne[:, 1],
    # Original Euclidean distances
    'dist_assigned': distances_to_center.astype("float32"),
    'dist_nearest_other': distances_to_nearest_other.astype("float32"),
    'nearest_other_cluster': nearest_other_cluster,
    'silhouette_score': silhouette_scores.astype("float32"),
    # New calibrated metrics
    'rnd_assigned': d1_r,                 # radius-normalised distance (lower = better)
    'rnd_nearest_other': d2_r,
    'margin_r': margin_r,                 # >0 indicates safe boundary
    'mahalanobis_first': np.sqrt(d2M_first),
    'mahalanobis_second': np.sqrt(d2M_second),
    'gap_euclid': gap_euclid,
    'gap_mahal': gap_mahal,
    'confidence_prob': confidence,        # [0,1]; recommend 0.6-0.7 as threshold
    'alt_label_from_prob': alt_label,
    'is_marginal': is_marginal.astype(bool),
    # Explanatory columns
    'CUSTOMER_SEGMENT': df['CUSTOMER_SEGMENT'],
    'CUSTOMER_PORTFOLIO': df['CUSTOMER_PORTFOLIO'],
    'ACTIVE_CUSTOMER': df['ACTIVE_CUSTOMER'],
    'ACTUAL_CLTV': df['ACTUAL_CLTV'],
    'CURRENT_YEAR_FAP': df['CURRENT_YEAR_FAP'],
    'FUTURE_LIFETIME_VALUE': df['FUTURE_LIFETIME_VALUE'],
    'ACTUAL_LIFETIME_DURATION': df['ACTUAL_LIFETIME_DURATION'],
    'NUM_CROSS_SOLD_LY': df['NUM_CROSS_SOLD_LY'],
    'CLM_OVER_PROFIT_HITCOUNT': df['CLM_OVER_PROFIT_HITCOUNT'],
    'CURRENT_GWP': df['CURRENT_GWP'],
    'TOTAL_SCORE': df['TOTAL_SCORE']
})
customer_scatter.to_csv(OUTPUT_DIR / "customer_scatter_data.csv", index=False)
print(f"    Saved: customer_scatter_data.csv ({len(customer_scatter):,} rows)")
print(f"      New columns: rnd_assigned, margin_r, mahalanobis_*, gap_*, confidence_prob")

# Dataset 2: Cluster centers
print("  Dataset 2: cluster_centers.csv...")
cluster_summary = df.groupby('cluster').agg({
    'CUSTOMER_ID': 'count',
    'ACTUAL_CLTV': 'mean',
    'CURRENT_YEAR_FAP': 'mean',
    'FUTURE_LIFETIME_VALUE': 'mean',
    'ACTUAL_LIFETIME_DURATION': 'mean',
    'NUM_CROSS_SOLD_LY': 'mean',
    'CLM_OVER_PROFIT_HITCOUNT': 'mean'
}).reset_index()
cluster_summary.columns = ['cluster', 'customer_count', 'mean_cltv', 'mean_fap',
                           'mean_future_lv', 'mean_tenure', 'mean_cross_sell', 'mean_claims']

cluster_centers_df = pd.DataFrame({
    'cluster': range(len(cluster_centers)),
    'center_pca_x': centers_pca[:, 0],
    'center_pca_y': centers_pca[:, 1],
    'center_tsne_x': centers_tsne[:, 0],
    'center_tsne_y': centers_tsne[:, 1]
})
cluster_centers_df = cluster_centers_df.merge(cluster_summary, on='cluster', how='left')
cluster_centers_df.to_csv(OUTPUT_DIR / "cluster_centers.csv", index=False)
print(f"    Saved: cluster_centers.csv ({len(cluster_centers_df)} rows)")

# Dataset 3: Cluster metadata (for slicers, with calibrated confidence metrics)
print("  Dataset 3: cluster_metadata.csv...")
cluster_metadata = df.groupby('cluster').agg({
    'CUSTOMER_ID': 'count',
    'ACTUAL_CLTV': ['mean', 'median', 'std']
}).reset_index()
cluster_metadata.columns = ['cluster', 'customer_count', 'mean_cltv', 'median_cltv', 'std_cltv']

# Calculate cluster-level metrics (silhouette, confidence, RND, marginal)
cluster_stats = []
for c in sorted(df['cluster'].unique()):
    mask = labels == c
    cluster_stats.append({
        'cluster': c,
        'mean_silhouette': float(silhouette_scores[mask].mean()),
        'p20_confidence': float(np.percentile(confidence[mask], 20)) if mask.any() else None,
        'median_rnd': float(np.median(d1_r[mask])) if mask.any() else None,
        'marginal_count': int(is_marginal[mask].sum()),
        'marginal_pct': float((is_marginal[mask].mean() * 100.0) if mask.any() else 0.0)
    })
stats_df = pd.DataFrame(cluster_stats)
cluster_metadata = cluster_metadata.merge(stats_df, on='cluster', how='left')

# Add cluster labels
cluster_labels = {
    0: 'Low-Value Base',
    1: 'Anomalous',
    2: 'Unprofitable',
    3: 'VIP Elite',
    4: 'Active Mid-Tier',
    5: 'Active Standard',
    6: 'Tenure Loyalists',
    7: 'Declining (Zero FLV)',
    8: 'High-Value'
}
cluster_metadata['cluster_label'] = cluster_metadata['cluster'].map(cluster_labels)
cluster_metadata['cluster_display'] = cluster_metadata.apply(
    lambda row: f"C{row['cluster']} - {row['cluster_label']}", axis=1
)

cluster_metadata.to_csv(OUTPUT_DIR / "cluster_metadata.csv", index=False)
print(f"    Saved: cluster_metadata.csv ({len(cluster_metadata)} rows)")
print(f"      New columns: p20_confidence, median_rnd, marginal_pct")

# Dataset 4: Marginal customers list
print("  Dataset 4: marginal_customers.csv...")
marginal_df = customer_scatter[customer_scatter['is_marginal']].copy()
marginal_df = marginal_df.sort_values('silhouette_score')
marginal_df['rank'] = range(1, len(marginal_df) + 1)
marginal_df.to_csv(OUTPUT_DIR / "marginal_customers.csv", index=False)
print(f"    Saved: marginal_customers.csv ({len(marginal_df):,} rows)")

# ============================================================================
# 7. Summary Report
# ============================================================================
print("\n[7/7] Generating summary report...")

summary_stats = {
    'total_customers': len(df),
    'num_clusters': len(cluster_centers),
    'mean_distance_to_center': distances_to_center.mean(),
    'median_distance_to_center': np.median(distances_to_center),
    'mean_distance_to_nearest_other': distances_to_nearest_other.mean(),
    'mean_rnd': d1_r.mean(),
    'mean_margin_r': margin_r.mean(),
    'mean_confidence': confidence.mean(),
    'median_confidence': np.median(confidence),
    'mean_silhouette': silhouette_scores.mean(),
    'marginal_customers': is_marginal.sum(),
    'marginal_pct': (is_marginal.sum() / len(df) * 100),
    'pca_variance_explained': pca.explained_variance_ratio_.sum() * 100
}

print("\n" + "="*80)
print("DISTANCE ANALYSIS SUMMARY (Calibrated Metrics)")
print("="*80)
print(f"Total customers: {summary_stats['total_customers']:,}")
print(f"Number of clusters: {summary_stats['num_clusters']}")
print(f"\nEuclidean Distance Metrics:")
print(f"  Mean distance to assigned cluster: {summary_stats['mean_distance_to_center']:.2f}")
print(f"  Median distance to assigned cluster: {summary_stats['median_distance_to_center']:.2f}")
print(f"  Mean distance to nearest other cluster: {summary_stats['mean_distance_to_nearest_other']:.2f}")
print(f"\nCalibrated Confidence Metrics:")
print(f"  Mean radius-normalised distance (RND): {summary_stats['mean_rnd']:.2f}")
print(f"  Mean margin (RND): {summary_stats['mean_margin_r']:.2f} (>0 = safe boundary)")
print(f"  Mean confidence (soft K-means): {summary_stats['mean_confidence']:.3f}")
print(f"  Median confidence: {summary_stats['median_confidence']:.3f}")
print(f"\nQuality Metrics:")
print(f"  Mean silhouette score: {summary_stats['mean_silhouette']:.3f}")
print(f"  PCA variance explained: {summary_stats['pca_variance_explained']:.1f}%")
print(f"\nMarginal Customers (Cluster-Adaptive Rule):")
print(f"  Count: {summary_stats['marginal_customers']:,}")
print(f"  Percentage: {summary_stats['marginal_pct']:.1f}%")
print(f"  Criteria: confidence < p20[cluster] OR silhouette < 0.10 OR margin_r < 0")

print("\nPer-Cluster Breakdown:")
print("-" * 80)
print(f"{'Cluster':<10} {'Size':<10} {'Marginal':<12} {'Marginal %':<12} {'Mean Sil':<12}")
print("-" * 80)
for _, row in cluster_metadata.iterrows():
    print(f"C{row['cluster']:<9} {row['customer_count']:<10,} {row['marginal_count']:<12,} "
          f"{row['marginal_pct']:<12.1f} {row['mean_silhouette']:<12.3f}")

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)
print("\nVisualization Charts:")
print("  1. distance_01_pca_scatter.png")
print("  2. distance_02_tsne_scatter_marginal.png")
print("  3. distance_03_distance_distributions.png")
print("  4. distance_04_silhouette_plot.png")
print("  5. distance_05_feature_pair_scatter.png")
print("\nPower BI Datasets:")
print("  1. customer_scatter_data.csv (customer-level data)")
print("  2. cluster_centers.csv (cluster center coordinates)")
print("  3. cluster_metadata.csv (cluster statistics for slicers)")
print("  4. marginal_customers.csv (borderline customers list)")
print("="*80)
print("\n✓ Distance analysis complete!")
print(f"✓ Output directory: {OUTPUT_DIR}")
print("="*80)
