#!/usr/bin/env python3
"""
Generate comparative visualizations between original CUSTOMER_SEGMENT and Run 04 clusters.

This script creates 7 additional charts analyzing the relationship between:
- Original A/A+/A-/B/C/D/E/F customer segments
- New Run 04 KMeans clusters (0-8)

Charts generated:
1. Performance Comparison: Original Segments vs New Clusters
2. Detailed Migration Matrix with Percentages
3. Scatter Plot: CLTV vs FAP (colored by segment, shaped by cluster)
4. Segment Homogeneity Analysis (variance comparison)
5. Value Migration Analysis (misclassifications)
6. Feature Distribution Comparison (top segments vs top clusters)
7. Alluvial Flow Diagram (segment to cluster migration)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("./data")
RUN_DIR = Path("./out_04_continuous_kmeans_main")
OUTPUT_DIR = RUN_DIR

# Load data
print("Loading data...")
df_original = pd.read_csv(DATA_DIR / "CSP_export.csv")
df_clusters = pd.read_csv(RUN_DIR / "cluster_assignments.csv")

# Merge
df = df_original.merge(df_clusters, on="CUSTOMER_ID", how="inner")
print(f"Loaded {len(df):,} customers with both segment and cluster labels")

# Performance features
perf_features = [
    'ACTUAL_CLTV',
    'CURRENT_YEAR_FAP',
    'FUTURE_LIFETIME_VALUE',
    'ACTUAL_LIFETIME_DURATION',
    'NUM_CROSS_SOLD_LY',
    'CLM_OVER_PROFIT_HITCOUNT'
]

# Segment order
segment_order = ['A+', 'A', 'A-', 'B', 'C', 'D', 'E', 'F']

# ============================================================================
# Chart 1: Performance Comparison - Original Segments vs New Clusters
# ============================================================================
print("\nGenerating Chart 1: Performance Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Performance Comparison: Original Segments vs New Clusters',
             fontsize=16, fontweight='bold', y=0.995)

metrics = [
    ('ACTUAL_CLTV', 'Mean CLTV (£)'),
    ('CURRENT_YEAR_FAP', 'Mean FAP (£)'),
    ('FUTURE_LIFETIME_VALUE', 'Mean Future LV (£)'),
    ('ACTUAL_LIFETIME_DURATION', 'Mean Tenure (years)'),
    ('NUM_CROSS_SOLD_LY', 'Mean Cross-Sell Count'),
    ('CLM_OVER_PROFIT_HITCOUNT', 'Mean Claims Count')
]

for idx, (metric, label) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Calculate means
    segment_means = df[df['CUSTOMER_SEGMENT'].isin(segment_order)].groupby('CUSTOMER_SEGMENT')[metric].mean()
    segment_means = segment_means.reindex(segment_order)

    cluster_means = df.groupby('cluster')[metric].mean().sort_values(ascending=False)

    # Plot
    x_seg = np.arange(len(segment_means))
    x_clus = np.arange(len(cluster_means)) + len(segment_means) + 1

    bars1 = ax.bar(x_seg, segment_means.values, width=0.6, color='steelblue', alpha=0.7, label='Original Segments')
    bars2 = ax.bar(x_clus, cluster_means.values, width=0.6, color='coral', alpha=0.7, label='New Clusters')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}' if abs(height) > 1000 else f'{height:.1f}',
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=7)

    # Formatting
    ax.set_ylabel(label, fontsize=10)
    ax.set_xticks(list(x_seg) + list(x_clus))
    ax.set_xticklabels(list(segment_means.index) + [f'C{c}' for c in cluster_means.index], fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

    # Add separator line
    ax.axvline(len(segment_means) + 0.5, color='gray', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_01_performance_segments_vs_clusters.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_01_performance_segments_vs_clusters.png")
plt.close()

# ============================================================================
# Chart 2: Detailed Migration Matrix with Percentages
# ============================================================================
print("Generating Chart 2: Detailed Migration Matrix...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Customer Migration: Original Segments → New Clusters',
             fontsize=16, fontweight='bold')

# Cross-tabulation
migration = pd.crosstab(df['CUSTOMER_SEGMENT'], df['cluster'], normalize='index') * 100
migration = migration.reindex(segment_order)

# Panel 1: Percentage heatmap
sns.heatmap(migration, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1,
           cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, linecolor='gray')
ax1.set_title('Percentage of Each Segment Migrating to Each Cluster', fontsize=12, fontweight='bold')
ax1.set_xlabel('New Cluster', fontsize=11)
ax1.set_ylabel('Original Segment', fontsize=11)

# Panel 2: Absolute counts heatmap
migration_counts = pd.crosstab(df['CUSTOMER_SEGMENT'], df['cluster'])
migration_counts = migration_counts.reindex(segment_order)

sns.heatmap(migration_counts, annot=True, fmt='d', cmap='Blues', ax=ax2,
           cbar_kws={'label': 'Customer Count'}, linewidths=0.5, linecolor='gray')
ax2.set_title('Absolute Customer Counts (Segment → Cluster)', fontsize=12, fontweight='bold')
ax2.set_xlabel('New Cluster', fontsize=11)
ax2.set_ylabel('Original Segment', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_02_migration_matrix_detailed.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_02_migration_matrix_detailed.png")
plt.close()

# ============================================================================
# Chart 3: Scatter Plot - CLTV vs FAP (colored by segment, shaped by cluster)
# ============================================================================
print("Generating Chart 3: Scatter Plot CLTV vs FAP...")

# Sample for visualization (too many points)
np.random.seed(42)
sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Cap extreme values for visualization
cltv_cap = df_sample['ACTUAL_CLTV'].quantile(0.99)
fap_cap = df_sample['CURRENT_YEAR_FAP'].quantile(0.99)
df_sample_plot = df_sample.copy()
df_sample_plot['ACTUAL_CLTV_capped'] = df_sample_plot['ACTUAL_CLTV'].clip(upper=cltv_cap)
df_sample_plot['CURRENT_YEAR_FAP_capped'] = df_sample_plot['CURRENT_YEAR_FAP'].clip(upper=fap_cap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Customer Distribution: CLTV vs FAP (Sample: {sample_size:,} customers)',
             fontsize=16, fontweight='bold')

# Panel 1: Colored by original segment
segment_colors = {'A+': 'darkgreen', 'A': 'green', 'A-': 'lightgreen',
                  'B': 'yellow', 'C': 'orange', 'D': 'coral', 'E': 'red', 'F': 'darkred'}
for seg in segment_order:
    seg_data = df_sample_plot[df_sample_plot['CUSTOMER_SEGMENT'] == seg]
    if len(seg_data) > 0:
        ax1.scatter(seg_data['CURRENT_YEAR_FAP_capped'], seg_data['ACTUAL_CLTV_capped'],
                   c=segment_colors.get(seg, 'gray'), label=f'Segment {seg}',
                   alpha=0.5, s=20, edgecolors='none')

ax1.set_xlabel('Current Year FAP (£)', fontsize=11)
ax1.set_ylabel('Actual CLTV (£)', fontsize=11)
ax1.set_title('Colored by Original Segment', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Panel 2: Colored by new cluster
cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for clus in sorted(df_sample_plot['cluster'].unique()):
    clus_data = df_sample_plot[df_sample_plot['cluster'] == clus]
    if len(clus_data) > 0:
        ax2.scatter(clus_data['CURRENT_YEAR_FAP_capped'], clus_data['ACTUAL_CLTV_capped'],
                   c=[cluster_colors[clus]], label=f'Cluster {clus}',
                   alpha=0.5, s=20, edgecolors='none')

ax2.set_xlabel('Current Year FAP (£)', fontsize=11)
ax2.set_ylabel('Actual CLTV (£)', fontsize=11)
ax2.set_title('Colored by New Cluster', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_03_scatter_cltv_vs_fap.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_03_scatter_cltv_vs_fap.png")
plt.close()

# ============================================================================
# Chart 4: Segment Homogeneity Analysis (variance comparison)
# ============================================================================
print("Generating Chart 4: Segment Homogeneity Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Homogeneity Analysis: Within-Group Variance (Lower = More Consistent)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, metric in enumerate(perf_features):
    ax = axes[idx // 3, idx % 3]

    # Calculate coefficient of variation (CV) for each segment/cluster
    segment_cv = df[df['CUSTOMER_SEGMENT'].isin(segment_order)].groupby('CUSTOMER_SEGMENT')[metric].apply(
        lambda x: x.std() / abs(x.mean()) if x.mean() != 0 else 0
    )
    segment_cv = segment_cv.reindex(segment_order)

    cluster_cv = df.groupby('cluster')[metric].apply(
        lambda x: x.std() / abs(x.mean()) if x.mean() != 0 else 0
    )

    # Plot
    x_seg = np.arange(len(segment_cv))
    x_clus = np.arange(len(cluster_cv)) + len(segment_cv) + 1

    ax.bar(x_seg, segment_cv.values, width=0.6, color='steelblue', alpha=0.7, label='Original Segments')
    ax.bar(x_clus, cluster_cv.values, width=0.6, color='coral', alpha=0.7, label='New Clusters')

    # Formatting
    ax.set_ylabel(f'CV ({metric.replace("_", " ")})', fontsize=9)
    ax.set_xticks(list(x_seg) + list(x_clus))
    ax.set_xticklabels(list(segment_cv.index) + [f'C{c}' for c in cluster_cv.index], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)
    ax.axvline(len(segment_cv) + 0.5, color='gray', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_04_homogeneity_analysis.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_04_homogeneity_analysis.png")
plt.close()

# ============================================================================
# Chart 5: Value Migration Analysis (misclassifications)
# ============================================================================
print("Generating Chart 5: Value Migration Analysis...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Value Migration Analysis: Segment vs Cluster Alignment',
             fontsize=16, fontweight='bold')

# Define segment tiers
premium_segments = ['A+', 'A', 'A-', 'B']
low_segments = ['D', 'E', 'F']

# Define cluster tiers (based on CLTV)
cluster_cltv = df.groupby('cluster')['ACTUAL_CLTV'].mean().sort_values(ascending=False)
high_clusters = cluster_cltv.head(3).index.tolist()  # Top 3 clusters
low_clusters = cluster_cltv.tail(3).index.tolist()   # Bottom 3 clusters

# 1. High-value customers in low segments (hidden gems)
df['is_premium_seg'] = df['CUSTOMER_SEGMENT'].isin(premium_segments)
df['is_high_cluster'] = df['cluster'].isin(high_clusters)
df['is_low_seg'] = df['CUSTOMER_SEGMENT'].isin(low_segments)
df['is_low_cluster'] = df['cluster'].isin(low_clusters)

# Panel 1: Premium customers by segment
premium_by_seg = df[df['is_high_cluster']].groupby('CUSTOMER_SEGMENT').size()
premium_by_seg = premium_by_seg.reindex(segment_order, fill_value=0)
colors1 = ['green' if seg in premium_segments else 'red' for seg in segment_order]
bars1 = ax1.bar(segment_order, premium_by_seg.values, color=colors1, alpha=0.7)
ax1.set_title('High-Value Customers (Top 3 Clusters) by Original Segment', fontsize=11, fontweight='bold')
ax1.set_ylabel('Customer Count', fontsize=10)
ax1.set_xlabel('Original Segment', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
# Add percentage labels
for i, (seg, bar) in enumerate(zip(segment_order, bars1)):
    height = bar.get_height()
    total = len(df[df['CUSTOMER_SEGMENT'] == seg])
    pct = (height / total * 100) if total > 0 else 0
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=8)

# Panel 2: Low-value customers by segment
low_by_seg = df[df['is_low_cluster']].groupby('CUSTOMER_SEGMENT').size()
low_by_seg = low_by_seg.reindex(segment_order, fill_value=0)
colors2 = ['red' if seg in premium_segments else 'orange' for seg in segment_order]
bars2 = ax2.bar(segment_order, low_by_seg.values, color=colors2, alpha=0.7)
ax2.set_title('Low-Value Customers (Bottom 3 Clusters) by Original Segment', fontsize=11, fontweight='bold')
ax2.set_ylabel('Customer Count', fontsize=10)
ax2.set_xlabel('Original Segment', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for i, (seg, bar) in enumerate(zip(segment_order, bars2)):
    height = bar.get_height()
    total = len(df[df['CUSTOMER_SEGMENT'] == seg])
    pct = (height / total * 100) if total > 0 else 0
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=8)

# Panel 3: Segment distribution in high clusters
high_cluster_segs = df[df['cluster'].isin(high_clusters)].groupby('cluster')['CUSTOMER_SEGMENT'].value_counts(normalize=True).unstack(fill_value=0) * 100
high_cluster_segs = high_cluster_segs.reindex(columns=segment_order, fill_value=0)
high_cluster_segs.plot(kind='bar', stacked=True, ax=ax3, colormap='RdYlGn_r', alpha=0.8)
ax3.set_title('Segment Composition in High-Value Clusters', fontsize=11, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=10)
ax3.set_xlabel('Cluster', fontsize=10)
ax3.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.set_xticklabels([f'C{c}' for c in high_cluster_segs.index], rotation=0)

# Panel 4: Segment distribution in low clusters
low_cluster_segs = df[df['cluster'].isin(low_clusters)].groupby('cluster')['CUSTOMER_SEGMENT'].value_counts(normalize=True).unstack(fill_value=0) * 100
low_cluster_segs = low_cluster_segs.reindex(columns=segment_order, fill_value=0)
low_cluster_segs.plot(kind='bar', stacked=True, ax=ax4, colormap='RdYlGn_r', alpha=0.8)
ax4.set_title('Segment Composition in Low-Value Clusters', fontsize=11, fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontsize=10)
ax4.set_xlabel('Cluster', fontsize=10)
ax4.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.set_xticklabels([f'C{c}' for c in low_cluster_segs.index], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_05_value_migration_analysis.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_05_value_migration_analysis.png")
plt.close()

# ============================================================================
# Chart 6: Feature Distribution Comparison (top segments vs top clusters)
# ============================================================================
print("Generating Chart 6: Feature Distribution Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distributions: Premium Segments vs High-Value Clusters',
             fontsize=16, fontweight='bold', y=0.995)

# Focus on premium segments and high clusters
df_premium_seg = df[df['CUSTOMER_SEGMENT'].isin(['A+', 'A', 'A-', 'B'])]
df_high_clus = df[df['cluster'].isin(high_clusters)]

for idx, metric in enumerate(perf_features):
    ax = axes[idx // 3, idx % 3]

    # Cap extreme values
    cap_val = df[metric].quantile(0.95)
    df_plot = df.copy()
    df_plot[metric] = df_plot[metric].clip(upper=cap_val)

    # Violin plots
    parts1 = ax.violinplot([df_plot[df_plot['CUSTOMER_SEGMENT'] == seg][metric].dropna()
                            for seg in ['A+', 'A', 'A-', 'B']],
                           positions=[0, 1, 2, 3], widths=0.6, showmeans=True, showmedians=True)
    for pc in parts1['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.5)

    parts2 = ax.violinplot([df_plot[df_plot['cluster'] == c][metric].dropna()
                            for c in high_clusters],
                           positions=[5, 6, 7], widths=0.6, showmeans=True, showmedians=True)
    for pc in parts2['bodies']:
        pc.set_facecolor('coral')
        pc.set_alpha(0.5)

    # Formatting
    ax.set_ylabel(metric.replace('_', ' '), fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 5, 6, 7])
    ax.set_xticklabels(['A+', 'A', 'A-', 'B'] + [f'C{c}' for c in high_clusters], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axvline(4, color='gray', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_06_feature_distributions_premium.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_06_feature_distributions_premium.png")
plt.close()

# ============================================================================
# Chart 7: Alluvial-style Flow Diagram (simplified)
# ============================================================================
print("Generating Chart 7: Alluvial Flow Diagram...")

fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle('Customer Flow: Original Segments → New Clusters (Alluvial)',
             fontsize=16, fontweight='bold')

# Get migration flows
flow_data = df.groupby(['CUSTOMER_SEGMENT', 'cluster']).size().reset_index(name='count')
flow_data = flow_data[flow_data['CUSTOMER_SEGMENT'].isin(segment_order)]

# Calculate positions
seg_totals = df[df['CUSTOMER_SEGMENT'].isin(segment_order)].groupby('CUSTOMER_SEGMENT').size()
seg_totals = seg_totals.reindex(segment_order)
cluster_totals = df.groupby('cluster').size()

# Normalize for visualization
max_total = max(seg_totals.max(), cluster_totals.max())

# Left side: segments
y_seg = {}
y_pos = 0
for seg in segment_order:
    height = seg_totals[seg] / max_total * 8
    y_seg[seg] = (y_pos, y_pos + height)
    y_pos += height + 0.2

# Right side: clusters
y_clus = {}
y_pos = 0
for clus in sorted(cluster_totals.index):
    height = cluster_totals[clus] / max_total * 8
    y_clus[clus] = (y_pos, y_pos + height)
    y_pos += height + 0.2

# Draw segments
for seg in segment_order:
    y1, y2 = y_seg[seg]
    rect = plt.Rectangle((0, y1), 0.8, y2-y1, facecolor=segment_colors.get(seg, 'gray'),
                         edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(rect)
    ax.text(-0.2, (y1+y2)/2, seg, ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(0.4, (y1+y2)/2, f'{seg_totals[seg]:,}', ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')

# Draw clusters
for clus in sorted(cluster_totals.index):
    y1, y2 = y_clus[clus]
    rect = plt.Rectangle((3.2, y1), 0.8, y2-y1, facecolor=cluster_colors[clus],
                         edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(rect)
    ax.text(4.2, (y1+y2)/2, f'C{clus}', ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(3.6, (y1+y2)/2, f'{cluster_totals[clus]:,}', ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')

# Draw flows (only major flows > 500 customers)
for _, row in flow_data[flow_data['count'] > 500].iterrows():
    seg = row['CUSTOMER_SEGMENT']
    clus = row['cluster']
    count = row['count']

    # Calculate flow height proportional to count
    seg_height = (y_seg[seg][1] - y_seg[seg][0]) * (count / seg_totals[seg])
    clus_height = (y_clus[clus][1] - y_clus[clus][0]) * (count / cluster_totals[clus])

    # Get current positions (track partial fills)
    if not hasattr(flow_data, '_seg_pos'):
        flow_data._seg_pos = {s: y_seg[s][0] for s in segment_order}
        flow_data._clus_pos = {c: y_clus[c][0] for c in cluster_totals.index}

    seg_y1 = flow_data._seg_pos[seg]
    seg_y2 = seg_y1 + seg_height
    clus_y1 = flow_data._clus_pos[clus]
    clus_y2 = clus_y1 + clus_height

    # Draw curved flow
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    verts = [
        (0.8, seg_y1),
        (1.5, seg_y1),
        (2.5, clus_y1),
        (3.2, clus_y1),
        (3.2, clus_y2),
        (2.5, clus_y2),
        (1.5, seg_y2),
        (0.8, seg_y2),
        (0.8, seg_y1)
    ]
    codes = [Path.MOVETO] + [Path.CURVE4]*3 + [Path.LINETO] + [Path.CURVE4]*3 + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=segment_colors.get(seg, 'gray'),
                               alpha=0.15, edgecolor='none')
    ax.add_patch(patch)

    # Update positions
    flow_data._seg_pos[seg] = seg_y2
    flow_data._clus_pos[clus] = clus_y2

# Formatting
ax.set_xlim(-1, 5)
ax.set_ylim(-0.5, max(y_pos, max([y[1] for y in y_seg.values()])) + 0.5)
ax.axis('off')
ax.text(0.4, -0.3, 'Original Segments', ha='center', fontsize=13, fontweight='bold')
ax.text(3.6, -0.3, 'New Clusters', ha='center', fontsize=13, fontweight='bold')
ax.text(2, -0.8, 'Note: Only flows >500 customers shown for clarity', ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_07_alluvial_flow_diagram.png", dpi=150, bbox_inches='tight')
print(f"  Saved: comparison_07_alluvial_flow_diagram.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Alignment analysis
print("\n1. PREMIUM SEGMENT CAPTURE RATE")
premium_in_high = len(df[(df['CUSTOMER_SEGMENT'].isin(premium_segments)) & (df['cluster'].isin(high_clusters))])
total_premium = len(df[df['CUSTOMER_SEGMENT'].isin(premium_segments)])
print(f"   Premium customers (A+/A/A-/B) in high-value clusters: {premium_in_high:,} / {total_premium:,} ({premium_in_high/total_premium*100:.1f}%)")

print("\n2. LOW SEGMENT SEPARATION")
low_in_low = len(df[(df['CUSTOMER_SEGMENT'].isin(low_segments)) & (df['cluster'].isin(low_clusters))])
total_low = len(df[df['CUSTOMER_SEGMENT'].isin(low_segments)])
print(f"   Low-value customers (D/E/F) in low-value clusters: {low_in_low:,} / {total_low:,} ({low_in_low/total_low*100:.1f}%)")

print("\n3. MISCLASSIFICATIONS (Opportunities)")
low_seg_high_clus = len(df[(df['CUSTOMER_SEGMENT'].isin(low_segments)) & (df['cluster'].isin(high_clusters))])
print(f"   Low segments → High clusters (hidden gems): {low_seg_high_clus:,}")
high_seg_low_clus = len(df[(df['CUSTOMER_SEGMENT'].isin(premium_segments)) & (df['cluster'].isin(low_clusters))])
print(f"   Premium segments → Low clusters (at-risk): {high_seg_low_clus:,}")

print("\n4. CLUSTER CLTV RANKING")
print("   Top 3 high-value clusters:", [f"C{c}" for c in high_clusters])
print("   Top 3 low-value clusters:", [f"C{c}" for c in low_clusters])

print("\n" + "="*80)
print(f"✓ All 7 comparative visualization charts generated successfully!")
print(f"✓ Output directory: {OUTPUT_DIR}")
print("="*80)
