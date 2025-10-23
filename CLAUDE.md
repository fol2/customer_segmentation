# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UKPLC Segmentation** is a customer segmentation package for Allianz UK PLC Commercial insurance. It clusters customers on **performance features** (the six CSP factor inputs) rather than using traditional score-cut segmentation. The system supports two data sources with identical structure: `UKPLC_CLTV_CSP_MV` (Mid-Market) and `UKPLC_CLTV_DCSP_MV` (Digital).

## Key Commands

### Installation
```bash
uv venv -p 3.11
uv pip install -e .
```

### Running Segmentation
```bash
# Main segmentation with discretised recipe (quantile binning)
ukplc-seg --input /data/CSP_export.csv --outdir ./out_csp --recipe discretised --algorithm auto --n-bins 7

# Main segmentation with continuous recipe (scaling-based)
ukplc-seg --input /data/DCSP_export.csv --outdir ./out_dcsp --recipe continuous --algorithm auto

# Parameter sweep across multiple recipes/algorithms
ukplc-seg-sweep --input /data/CSP_export.csv --outdir ./sweep --recipes continuous,discretised --algorithms auto,kmeans,hdbscan
```

### Common CLI Options
- `--filter COL=VAL` - Repeatable filter (e.g., `--filter CUSTOMER_PORTFOLIO=MID`)
- `--exclude-inspection-only` - Remove rows with `INSP_FLAG='Y'` (default: include)
- `--k-select composite|silhouette` - Auto k selection criterion
- `--scaler robust|standard|none` - Scaling method for continuous recipe
- `--log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE` - Apply log1p transform to handle heavy tails
- `--rows-limit N` - Use subset for quick dry-run
- `--disable-shap` - Skip SHAP explanations to speed up runs

## Architecture

### Data Flow Pipeline

1. **Data Loading** (`io.py`): Load CSV/Parquet from UKPLC_CLTV views
2. **Feature Engineering** (`features.py`): Transform via one of two recipes
3. **Clustering** (`cluster.py`): Fit clustering algorithm with auto k-selection
4. **Evaluation** (`evaluate.py`): Compute internal metrics and cluster profiles
5. **Explanation** (`explain.py`): Generate SHAP-based explanations using RandomForest surrogate

### Two-Recipe System

**Continuous recipe** (`features.py:72-105`):
- Imputation (median) → optional log1p → scaling (RobustScaler/StandardScaler/none)
- Used with HDBSCAN or KMeans
- Better for density-based clustering

**Discretised recipe** (`features.py:107-138`):
- Imputation → optional log1p → quantile binning (configurable n_bins, default 7) → one-hot encoding
- Always used with KMeans
- Mimics case-study style discrete segments

### Performance Features vs Explanatory Columns

**Critical distinction**: The system uses two sets of columns:

1. **Performance features** (used for clustering, defined in `run.py:21-28`):
   - `ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`
   - `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`

2. **Explanatory columns** (used ONLY for post-hoc profiling, defined in `run.py:30-43`):
   - `CUSTOMER_SEGMENT`, `CUSTOMER_PORTFOLIO`, `ACTIVE_CUSTOMER`
   - `CURRENT_GWP`, `LAST_YEAR_GWP`, `GEP_FIN_FULL`, `GEC_FIN_FULL`, `CLM_INC_FULL`
   - `WEIGHTED_PLAN_LOSS_RATIO`, `ACTUAL_LOSS_RATIO`, `EXPECTED_LIFETIME_VALUE`, `TOTAL_SCORE`

Never use explanatory columns as clustering inputs. They describe clusters after formation.

### Clustering Algorithm Selection (`cluster.py:54-86`)

The `algorithm='auto'` logic:
1. Try HDBSCAN first (if available, with configurable `min_cluster_size` and `min_samples`)
2. If HDBSCAN produces ≤1 valid cluster or all noise, fall back to KMeans with auto k
3. Manual options: `kmeans`, `hdbscan`, `agglomerative`

**K selection** (`cluster.py:28-52`):
- Grid search over `[k_min, k_max]` (default 3-12)
- Two scoring modes:
  - `silhouette`: Pure silhouette score
  - `composite`: `silhouette + (calinski_harabasz/10000) - davies_bouldin`

### Output Artefacts (`run.py:152-190`)

All saved to `--outdir`:
- `cluster_assignments.csv` - Customer IDs with cluster labels
- `cluster_profiles.csv` - Aggregate statistics per cluster
- `cluster_vs_existing_segment.csv` - Cross-tab with existing `CUSTOMER_SEGMENT` (if present)
- `internal_metrics.json` - Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- `explain.json` - SHAP summary (unless `--disable-shap`)
- `model.joblib` - Serialized pipeline and cluster model
- `cluster_sizes.png` - Bar chart of cluster sizes
- `REPORT.md` - Markdown summary

### Configuration System

Configuration is merged from three sources (CLI takes precedence):
1. YAML file (via `--config`, see `CONFIG.sample.yaml`)
2. CLI arguments
3. Defaults in code

The `config.py` module handles YAML loading; `run.py:72-77` implements merge logic.

## Development Notes

### Testing Configuration Changes

When modifying feature engineering or clustering logic:
```bash
# Quick dry-run with 1000 rows
ukplc-seg --input data.csv --outdir test_out --rows-limit 1000 --recipe continuous --algorithm kmeans

# Compare multiple approaches
ukplc-seg-sweep --input data.csv --outdir sweep_out --recipes continuous,discretised --algorithms kmeans,hdbscan --rows-limit 5000
```

### Filter Mechanics (`run.py:86-105`)

The `--filter` implementation handles three value types:
1. Y/N flags (case-insensitive match)
2. Numeric values (parsed as float, matched with `pd.to_numeric`)
3. String literals (exact match on string-cast column)

### Inspection-Only Customers

By default, customers with `INSP_FLAG='Y'` are **included**. Use `--exclude-inspection-only` to remove them. This matches current business requirements to analyze inspection-only customers as a distinct segment.

### SHAP Explanation Approach (`explain.py`)

The system trains a RandomForest classifier to predict cluster labels, then extracts SHAP values to explain which features drive cluster membership. This is a surrogate model approach since clustering models (especially HDBSCAN) don't have native feature importances. Always uses continuous recipe internally for SHAP computation.

### Dependencies

- **Required**: pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.2, matplotlib 3.8.4, joblib, pyyaml
- **Recommended**: hdbscan 0.8.38, shap 0.46.0 (both require Python ≥3.8)
- **Optional**: kmodes 0.12.2 (mixed feature type handling, not used in current recipes)

## Data Assumptions

Input files exported from `UKPLC_CLTV_CSP_MV` or `UKPLC_CLTV_DCSP_MV` must contain:
- All six performance features (clustering inputs)
- Customer identifier column (default `CUSTOMER_ID`, configurable via `--id-col`)
- Optional explanatory columns for profiling

Missing performance features will cause runtime errors. Missing explanatory columns are silently dropped from profiling.
