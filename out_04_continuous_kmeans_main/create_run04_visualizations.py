"""
Create comprehensive visualizations for Run 04 analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = Path("./data/CSP_export.csv")
OUT_DIR = Path("./out_04_continuous_kmeans_main")
ASSIGNMENTS_PATH = OUT_DIR / "cluster_assignments.csv"
PROFILES_PATH = OUT_DIR / "cluster_profiles.csv"
CROSSTAB_PATH = OUT_DIR / "cluster_vs_existing_segment.csv"

# Load data
print("Loading data...")
df_main = pd.read_csv(DATA_PATH)
df_assignments = pd.read_csv(ASSIGNMENTS_PATH)
df_profiles = pd.read_csv(PROFILES_PATH)
df_crosstab = pd.read_csv(CROSSTAB_PATH)

# Merge assignments with main data
df = df_main.merge(df_assignments, on='CUSTOMER_ID', how='inner')

# Performance features
PERF_FEATURES = [
    'ACTUAL_CLTV',
    'CURRENT_YEAR_FAP',
    'FUTURE_LIFETIME_VALUE',
    'ACTUAL_LIFETIME_DURATION',
    'NUM_CROSS_SOLD_LY',
    'CLM_OVER_PROFIT_HITCOUNT'
]

# Cluster names for better readability
CLUSTER_NAMES = {
    0: "Low-Value Base (43.3%)",
    2: "Unprofitable (12.4%)",
    5: "Active Standard (10.6%)",
    4: "Active Mid-Tier (8.8%)",
    6: "Tenure Loyalists (8.4%)",
    7: "Zero Future Value (6.8%)",
    8: "High-Value (6.8%)",
    3: "VIP Elite (2.4%)",
    1: "Anomalous (0.5%)"
}

print(f"Total customers: {len(df):,}")
print(f"Clusters: {df['cluster'].nunique()}")

# ============================================================================
# Figure 1: Feature Distributions by Cluster (6 subplots)
# ============================================================================
print("\nCreating Figure 1: Feature Distributions by Cluster...")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(PERF_FEATURES):
    ax = axes[idx]

    # Create violin plot
    data_to_plot = []
    labels_to_plot = []

    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster][feature].dropna()
        # Cap extreme values for visualization
        if feature in ['ACTUAL_CLTV', 'FUTURE_LIFETIME_VALUE', 'CURRENT_YEAR_FAP']:
            cluster_data = cluster_data.clip(lower=-50000, upper=200000)
        data_to_plot.append(cluster_data)
        labels_to_plot.append(f"C{cluster}")

    # Create boxplot (violin plots can be too dense)
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_01_feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_01_feature_distributions.png'}")
plt.close()

# ============================================================================
# Figure 2: Cluster Profiles Heatmap (Normalized)
# ============================================================================
print("\nCreating Figure 2: Cluster Profiles Heatmap...")

# Select key metrics for heatmap
heatmap_cols = [
    'mean_ACTUAL_CLTV',
    'mean_CURRENT_YEAR_FAP',
    'mean_FUTURE_LIFETIME_VALUE',
    'mean_ACTUAL_LIFETIME_DURATION',
    'mean_NUM_CROSS_SOLD_LY',
    'product_index'
]

# Create heatmap data
heatmap_data = df_profiles[['cluster'] + heatmap_cols].set_index('cluster')

# Normalize each column to 0-1 for comparison
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

# Rename columns for display
heatmap_normalized.columns = [
    'CLTV',
    'Current FAP',
    'Future LV',
    'Tenure',
    'Cross-Sell',
    'Product Index'
]

# Add cluster names
heatmap_normalized.index = [f"C{i}" for i in heatmap_normalized.index]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, cbar_kws={'label': 'Normalized Value (0-1)'}, ax=ax)
ax.set_title('Cluster Profiles Heatmap (Normalized Metrics)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_02_cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_02_cluster_profiles_heatmap.png'}")
plt.close()

# ============================================================================
# Figure 3: Cluster vs Existing Segment Mapping (Stacked Bar)
# ============================================================================
print("\nCreating Figure 3: Cluster vs Existing Segment Mapping...")

# Prepare crosstab data
crosstab_data = df_crosstab.set_index('cluster')
crosstab_pct = crosstab_data.div(crosstab_data.sum(axis=1), axis=0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Stacked bar chart (percentage)
crosstab_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='tab10')
ax1.set_title('Cluster Composition by Existing Segment (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Percentage', fontsize=12)
ax1.legend(title='Existing Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xticklabels([f"C{i}" for i in crosstab_pct.index], rotation=0)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Heatmap (absolute counts)
sns.heatmap(crosstab_data, annot=True, fmt=',.0f', cmap='Blues',
            linewidths=0.5, ax=ax2, cbar_kws={'label': 'Customer Count'})
ax2.set_title('Cluster vs Existing Segment (Absolute Counts)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Existing Segment', fontsize=12)
ax2.set_ylabel('Cluster', fontsize=12)
ax2.set_yticklabels([f"C{i}" for i in crosstab_data.index], rotation=0)

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_03_cluster_vs_existing_segment.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_03_cluster_vs_existing_segment.png'}")
plt.close()

# ============================================================================
# Figure 4: Cluster Characteristics Radar Chart
# ============================================================================
print("\nCreating Figure 4: Cluster Characteristics Radar Chart...")

# Select VIP and key clusters for radar
selected_clusters = [3, 8, 4, 0, 2]  # VIP, High-value, Mid-tier, Base, Unprofitable

# Metrics for radar
radar_metrics = [
    'mean_ACTUAL_CLTV',
    'mean_CURRENT_YEAR_FAP',
    'mean_FUTURE_LIFETIME_VALUE',
    'mean_ACTUAL_LIFETIME_DURATION',
    'product_index'
]

radar_labels = ['CLTV', 'FAP', 'Future LV', 'Tenure', 'Product Index']

# Prepare data
radar_data = df_profiles[df_profiles['cluster'].isin(selected_clusters)][['cluster'] + radar_metrics].set_index('cluster')

# Normalize each metric to 0-1
radar_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

# Number of variables
num_vars = len(radar_metrics)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for cluster_id in selected_clusters:
    values = radar_normalized.loc[cluster_id].tolist()
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, 'o-', linewidth=2, label=f"Cluster {cluster_id}")
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Cluster Characteristics Comparison (Normalized)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_04_cluster_radar_chart.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_04_cluster_radar_chart.png'}")
plt.close()

# ============================================================================
# Figure 5: Cluster Sizes and Key Metrics
# ============================================================================
print("\nCreating Figure 5: Cluster Sizes and Key Metrics...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cluster sizes (already exists but enhance)
cluster_sizes = df_profiles.sort_values('n_customers', ascending=False)
colors_size = ['#e74c3c' if i in [2, 1] else '#3498db' if i in [3, 8] else '#95a5a6'
               for i in cluster_sizes['cluster']]

ax1.bar(range(len(cluster_sizes)), cluster_sizes['n_customers'], color=colors_size)
ax1.set_xticks(range(len(cluster_sizes)))
ax1.set_xticklabels([f"C{i}" for i in cluster_sizes['cluster']])
ax1.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Customer Count')
ax1.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (idx, row) in enumerate(cluster_sizes.iterrows()):
    pct = row['share'] * 100
    ax1.text(i, row['n_customers'] + 1000, f"{pct:.1f}%", ha='center', fontsize=9)

# 2. Mean CLTV by cluster
cltv_data = df_profiles.sort_values('mean_ACTUAL_CLTV', ascending=False)
colors_cltv = ['#27ae60' if x > 50000 else '#f39c12' if x > 10000 else '#e74c3c' if x < 0 else '#95a5a6'
               for x in cltv_data['mean_ACTUAL_CLTV']]

ax2.bar(range(len(cltv_data)), cltv_data['mean_ACTUAL_CLTV'], color=colors_cltv)
ax2.set_xticks(range(len(cltv_data)))
ax2.set_xticklabels([f"C{i}" for i in cltv_data['cluster']])
ax2.set_title('Mean CLTV by Cluster', fontsize=14, fontweight='bold')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Mean CLTV (£)')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Product Index (Cross-sell potential)
pi_data = df_profiles.sort_values('product_index', ascending=False)
colors_pi = ['#27ae60' if x > 2 else '#f39c12' if x > 1 else '#95a5a6'
             for x in pi_data['product_index']]

ax3.bar(range(len(pi_data)), pi_data['product_index'], color=colors_pi)
ax3.set_xticks(range(len(pi_data)))
ax3.set_xticklabels([f"C{i}" for i in pi_data['cluster']])
ax3.set_title('Product Index (Cross-Sell Potential)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Product Index')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Tenure distribution
tenure_data = df_profiles.sort_values('mean_ACTUAL_LIFETIME_DURATION', ascending=False)
colors_tenure = ['#9b59b6' if x > 10 else '#3498db' if x > 5 else '#95a5a6'
                 for x in tenure_data['mean_ACTUAL_LIFETIME_DURATION']]

ax4.bar(range(len(tenure_data)), tenure_data['mean_ACTUAL_LIFETIME_DURATION'], color=colors_tenure)
ax4.set_xticks(range(len(tenure_data)))
ax4.set_xticklabels([f"C{i}" for i in tenure_data['cluster']])
ax4.set_title('Mean Tenure by Cluster', fontsize=14, fontweight='bold')
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Mean Tenure (years)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_05_cluster_key_metrics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_05_cluster_key_metrics.png'}")
plt.close()

# ============================================================================
# Figure 6: Feature Correlations within VIP Cluster
# ============================================================================
print("\nCreating Figure 6: Feature Correlations (VIP Cluster)...")

# Focus on VIP cluster (3)
df_vip = df[df['cluster'] == 3][PERF_FEATURES]

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_vip.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlations within VIP Elite Cluster (C3)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_06_vip_feature_correlations.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_DIR / 'analysis_06_vip_feature_correlations.png'}")
plt.close()

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
print("\nGenerated files:")
print("  1. analysis_01_feature_distributions.png - Boxplots of 6 features across clusters")
print("  2. analysis_02_cluster_profiles_heatmap.png - Normalized metrics heatmap")
print("  3. analysis_03_cluster_vs_existing_segment.png - Mapping to A/B/C/D/E/F segments")
print("  4. analysis_04_cluster_radar_chart.png - Radar chart comparing key clusters")
print("  5. analysis_05_cluster_key_metrics.png - Size, CLTV, Product Index, Tenure")
print("  6. analysis_06_vip_feature_correlations.png - VIP cluster feature correlations")
print("="*70)
