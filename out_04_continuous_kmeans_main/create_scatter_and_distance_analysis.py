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
RUN_DIR = Path("./out_04_continuous_kmeans_main")
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

print(f"  Loaded {len(df):,} customers")
print(f"  Clusters: {sorted(df['cluster'].unique())}")

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
# 3. Calculate Distances and Confidence Metrics
# ============================================================================
print("\n[3/7] Calculating distances and confidence metrics...")

# Get cluster centers from trained model
cluster_centers = kmeans_model.cluster_centers_

# Calculate distance to assigned cluster center
distances_to_center = np.zeros(len(df))
for i in range(len(df)):
    assigned_cluster = labels[i]
    center = cluster_centers[assigned_cluster]
    distances_to_center[i] = np.linalg.norm(X_transformed[i] - center)

# Calculate distance to nearest other cluster
distances_to_nearest_other = np.zeros(len(df))
nearest_other_cluster = np.zeros(len(df), dtype=int)
for i in range(len(df)):
    assigned_cluster = labels[i]
    min_dist = np.inf
    nearest_cluster = -1
    for c in range(len(cluster_centers)):
        if c != assigned_cluster:
            dist = np.linalg.norm(X_transformed[i] - cluster_centers[c])
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = c
    distances_to_nearest_other[i] = min_dist
    nearest_other_cluster[i] = nearest_cluster

# Calculate confidence ratio (distance to nearest other / distance to assigned)
confidence_ratio = distances_to_nearest_other / (distances_to_center + 1e-10)

# Calculate silhouette scores
silhouette_scores = silhouette_samples(X_transformed, labels)

# Identify marginal customers (low confidence)
# Criteria: low silhouette score OR small confidence ratio
marginal_threshold_silhouette = 0.2
marginal_threshold_ratio = 1.5
is_marginal = (silhouette_scores < marginal_threshold_silhouette) | (confidence_ratio < marginal_threshold_ratio)

print(f"  Mean distance to center: {distances_to_center.mean():.2f}")
print(f"  Mean distance to nearest other: {distances_to_nearest_other.mean():.2f}")
print(f"  Mean confidence ratio: {confidence_ratio.mean():.2f}")
print(f"  Mean silhouette score: {silhouette_scores.mean():.3f}")
print(f"  Marginal customers: {is_marginal.sum():,} ({is_marginal.sum()/len(df)*100:.1f}%)")

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

# Panel 3: Confidence ratio
ax3.hist(confidence_ratio[confidence_ratio < 10], bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.axvline(confidence_ratio.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidence_ratio.mean():.2f}')
ax3.axvline(marginal_threshold_ratio, color='purple', linestyle='--', linewidth=2, label=f'Marginal Threshold: {marginal_threshold_ratio}')
ax3.set_xlabel('Confidence Ratio (Nearest Other / Assigned)', fontsize=11)
ax3.set_ylabel('Customer Count', fontsize=11)
ax3.set_title('Confidence Ratio Distribution (capped at 10 for visibility)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Silhouette scores
ax4.hist(silhouette_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(silhouette_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {silhouette_scores.mean():.3f}')
ax4.axvline(marginal_threshold_silhouette, color='orange', linestyle='--', linewidth=2, label=f'Marginal Threshold: {marginal_threshold_silhouette}')
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

# Dataset 1: Customer-level scatter data
print("  Dataset 1: customer_scatter_data.csv...")
customer_scatter = pd.DataFrame({
    'CUSTOMER_ID': df['CUSTOMER_ID'],
    'CLIENT_NAME': df['CLIENT_NAME'],
    'cluster': labels,
    'pca_x': X_pca[:, 0],
    'pca_y': X_pca[:, 1],
    'tsne_x': X_tsne[:, 0],
    'tsne_y': X_tsne[:, 1],
    'distance_to_center': distances_to_center,
    'distance_to_nearest_other': distances_to_nearest_other,
    'nearest_other_cluster': nearest_other_cluster,
    'confidence_ratio': confidence_ratio,
    'silhouette_score': silhouette_scores,
    'is_marginal': is_marginal,
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

# Dataset 3: Cluster metadata (for slicers)
print("  Dataset 3: cluster_metadata.csv...")
cluster_metadata = df.groupby('cluster').agg({
    'CUSTOMER_ID': 'count',
    'ACTUAL_CLTV': ['mean', 'median', 'std']
}).reset_index()
cluster_metadata.columns = ['cluster', 'customer_count', 'mean_cltv', 'median_cltv', 'std_cltv']

# Calculate mean silhouette score per cluster separately
silhouette_by_cluster = []
for c in sorted(df['cluster'].unique()):
    mask = labels == c
    mean_sil = silhouette_scores[mask].mean()
    silhouette_by_cluster.append({'cluster': c, 'mean_silhouette': mean_sil})
silhouette_df = pd.DataFrame(silhouette_by_cluster)
cluster_metadata = cluster_metadata.merge(silhouette_df, on='cluster', how='left')

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

# Add marginal customer count per cluster
marginal_counts = df[is_marginal].groupby('cluster').size().reset_index(name='marginal_count')
cluster_metadata = cluster_metadata.merge(marginal_counts, on='cluster', how='left')
cluster_metadata['marginal_count'] = cluster_metadata['marginal_count'].fillna(0).astype(int)
cluster_metadata['marginal_pct'] = (cluster_metadata['marginal_count'] / cluster_metadata['customer_count'] * 100).round(1)

cluster_metadata.to_csv(OUTPUT_DIR / "cluster_metadata.csv", index=False)
print(f"    Saved: cluster_metadata.csv ({len(cluster_metadata)} rows)")

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
    'mean_confidence_ratio': confidence_ratio.mean(),
    'mean_silhouette': silhouette_scores.mean(),
    'marginal_customers': is_marginal.sum(),
    'marginal_pct': (is_marginal.sum() / len(df) * 100),
    'pca_variance_explained': pca.explained_variance_ratio_.sum() * 100
}

print("\n" + "="*80)
print("DISTANCE ANALYSIS SUMMARY")
print("="*80)
print(f"Total customers: {summary_stats['total_customers']:,}")
print(f"Number of clusters: {summary_stats['num_clusters']}")
print(f"\nDistance Metrics:")
print(f"  Mean distance to assigned cluster: {summary_stats['mean_distance_to_center']:.2f}")
print(f"  Median distance to assigned cluster: {summary_stats['median_distance_to_center']:.2f}")
print(f"  Mean distance to nearest other cluster: {summary_stats['mean_distance_to_nearest_other']:.2f}")
print(f"  Mean confidence ratio: {summary_stats['mean_confidence_ratio']:.2f}")
print(f"\nQuality Metrics:")
print(f"  Mean silhouette score: {summary_stats['mean_silhouette']:.3f}")
print(f"  PCA variance explained: {summary_stats['pca_variance_explained']:.1f}%")
print(f"\nMarginal Customers:")
print(f"  Count: {summary_stats['marginal_customers']:,}")
print(f"  Percentage: {summary_stats['marginal_pct']:.1f}%")
print(f"  Criteria: Silhouette < {marginal_threshold_silhouette} OR Confidence ratio < {marginal_threshold_ratio}")

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
