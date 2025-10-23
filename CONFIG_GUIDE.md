# Configuration Files Quick Reference

This directory contains three configuration files for UKPLC Segmentation:

## Files

### 1. `CONFIG.yaml` (Main Configuration)
**Purpose**: Comprehensive configuration with all options documented
**Best for**: Understanding all available parameters, creating custom configs
**Default recipe**: `continuous`

```bash
ukplc-seg --input /data/your_export.csv --outdir ./out --config CONFIG.yaml
```

### 2. `CONFIG_CSP.yaml` (Mid-Market Optimized)
**Purpose**: Pre-configured for CSP (Mid-Market) segmentation
**Recipe**: `discretised` with 7 quantile bins
**Algorithm**: `auto` (tries HDBSCAN → KMeans)
**K-selection**: `composite`
**Scaler**: `robust`

**Typical use case**:
```bash
ukplc-seg --input /data/CSP_export.csv --outdir ./out_csp --config CONFIG_CSP.yaml
```

**With filters**:
```bash
ukplc-seg --input /data/CSP_export.csv --outdir ./out_csp --config CONFIG_CSP.yaml \
  --filter CUSTOMER_PORTFOLIO=MID \
  --filter ACTIVE_CUSTOMER=Y
```

**Quick dry-run**:
```bash
ukplc-seg --input /data/CSP_export.csv --outdir ./test_csp --config CONFIG_CSP.yaml \
  --rows-limit 5000
```

### 3. `CONFIG_DCSP.yaml` (Digital Optimized)
**Purpose**: Pre-configured for DCSP (Digital) segmentation
**Recipe**: `continuous` with standard scaling
**Algorithm**: `auto` (prefers HDBSCAN)
**K-selection**: `silhouette`
**Scaler**: `standard`

**Typical use case**:
```bash
ukplc-seg --input /data/DCSP_export.parquet --outdir ./out_dcsp --config CONFIG_DCSP.yaml
```

**With filters**:
```bash
ukplc-seg --input /data/DCSP_export.parquet --outdir ./out_dcsp --config CONFIG_DCSP.yaml \
  --filter CUSTOMER_PORTFOLIO=DIGITAL \
  --filter ACTIVE_CUSTOMER=Y
```

---

## Key Configuration Parameters

### Recipe Selection
| Recipe | Description | Best For |
|--------|-------------|----------|
| `continuous` | Impute → log1p → scaling → clustering | HDBSCAN, density patterns, continuous metrics |
| `discretised` | Quantile binning → one-hot → KMeans | Case-study style, robust to outliers |

### Algorithm Options
| Algorithm | Description | When to Use |
|-----------|-------------|-------------|
| `auto` | Try HDBSCAN first, fallback to KMeans | Most versatile (recommended) |
| `kmeans` | Force KMeans with auto k-selection | When you want spherical clusters |
| `hdbscan` | Force HDBSCAN | When you want density-based clusters |
| `agglomerative` | Hierarchical clustering | When you want dendrogram structure |

### Scaler Types
| Scaler | Description | Best For |
|--------|-------------|----------|
| `robust` | Median/IQR-based (RobustScaler) | Heavy tails, outliers (recommended for CLTV/FLV) |
| `standard` | Mean/Std-based (StandardScaler) | Normally distributed features |
| `none` | No scaling | Already normalized or using discretised recipe |

### K-Selection Criteria
| Criterion | Formula | Best For |
|-----------|---------|----------|
| `composite` | silhouette + (CH/10000) - DB | Balanced trade-off |
| `silhouette` | Pure silhouette score | Maximum cluster separation |

---

## CLI Overrides

**All CLI arguments override YAML values.** This allows you to keep a base config and experiment:

```bash
# Use CSP config but switch to continuous recipe
ukplc-seg --input data.csv --outdir ./out --config CONFIG_CSP.yaml --recipe continuous

# Use DCSP config but force KMeans with 8 clusters
ukplc-seg --input data.parquet --outdir ./out --config CONFIG_DCSP.yaml \
  --algorithm kmeans --k-min 8 --k-max 8

# Override scaler and disable SHAP
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml \
  --scaler standard --disable-shap
```

---

## Common Usage Patterns

### Pattern 1: Quick Exploration (Dry-Run)
```bash
# Test with 5000 rows, no SHAP
ukplc-seg --input /data/CSP_export.csv --outdir ./test \
  --config CONFIG_CSP.yaml --rows-limit 5000 --disable-shap
```

### Pattern 2: Full CSP Run (Production)
```bash
# Full dataset, all metrics, include inspection
ukplc-seg --input /data/CSP_export.csv --outdir ./out_csp_prod \
  --config CONFIG_CSP.yaml
```

### Pattern 3: Active MID Portfolio Only
```bash
# Filter to active Mid-Market customers
ukplc-seg --input /data/CSP_export.csv --outdir ./out_mid_active \
  --config CONFIG_CSP.yaml \
  --filter CUSTOMER_PORTFOLIO=MID \
  --filter ACTIVE_CUSTOMER=Y
```

### Pattern 4: Exclude Inspection-Only
```bash
# Remove inspection-only customers
ukplc-seg --input /data/CSP_export.csv --outdir ./out_no_insp \
  --config CONFIG_CSP.yaml --exclude-inspection-only
```

### Pattern 5: Parameter Sweep (No Config)
```bash
# Sweep ignores config files - use CLI only
ukplc-seg-sweep --input /data/CSP_export.csv --outdir ./sweep \
  --recipes continuous,discretised \
  --algorithms auto,kmeans,hdbscan \
  --k-min 3 --k-max 12 --n-bins 7 \
  --scaler robust --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

---

## Log Transform (Heavy Tails)

Both CSP and DCSP configs apply `log1p` to CLTV and FLV by default:

```yaml
log1p:
  - ACTUAL_CLTV
  - FUTURE_LIFETIME_VALUE
```

**Why?** These features are heavily right-skewed. Log transform stabilizes distances.

To disable on CLI:
```bash
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml --log1p ""
```

To add more features:
```bash
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE,CURRENT_YEAR_FAP
```

---

## HDBSCAN Tuning

If HDBSCAN produces too many noise points or too few clusters:

**Too many noise (-1)**:
```bash
# Decrease min cluster size
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml \
  --hdbscan-min-cluster-size 50
```

**Too few clusters**:
```bash
# Increase min samples or switch to KMeans
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml \
  --algorithm kmeans
```

---

## Reproducibility

All configs set `random_state: 42` for reproducibility. To change:

```bash
ukplc-seg --input data.csv --outdir ./out --config CONFIG.yaml --random-state 123
```

---

## Performance Tips

1. **Quick tests**: Use `--rows-limit 5000 --disable-shap`
2. **Large datasets**: Use Parquet input, increase `hdbscan_min_cluster_size`
3. **Slow SHAP**: Add `--disable-shap` (keeps RF feature importances)
4. **Memory issues**: Filter data before export or use row limits

---

## Output Files

All runs produce (in `--outdir`):

| File | Description |
|------|-------------|
| `cluster_assignments.csv` | Customer ID + cluster label |
| `cluster_profiles.csv` | Cluster statistics, medians, product_index |
| `cluster_vs_existing_segment.csv` | Cross-tab with CUSTOMER_SEGMENT |
| `internal_metrics.json` | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| `explain.json` | RF + SHAP feature importances |
| `cluster_sizes.png` | Bar chart of cluster sizes |
| `REPORT.md` | Run summary |
| `model.joblib` | Serialized pipeline |

---

## Troubleshooting

**Problem**: Missing columns error
**Solution**: Ensure your export contains all 6 performance features

**Problem**: HDBSCAN gives 1 cluster
**Solution**: Use `--algorithm kmeans` or reduce `--hdbscan-min-cluster-size`

**Problem**: SHAP too slow
**Solution**: Add `--disable-shap`

**Problem**: Out of memory
**Solution**: Use `--rows-limit` or filter data before export

**Problem**: Poor silhouette score
**Solution**: Try different recipe/scaler, adjust k range, apply log1p

---

## See Also

- `README.md` - Quick start guide
- `UKPLC_Segmentation_Operations_Guide.md` - Full operational documentation
- `CLAUDE.md` - Development notes for Claude Code
