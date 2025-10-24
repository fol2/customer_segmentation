# Next Segmentation Runs Plan (out_04 to out_08)

**Date**: 2025-10-24
**Status**: Code improvements complete, ready to execute

---

## Code Improvements Summary

All requested improvements from `comments_01.md` have been implemented:

### 1. MiniBatchKMeans Support
- **File**: `cluster.py`
- **Features**: New `mbkmeans` algorithm, configurable batch_size (8192) and max_iter (100)
- **Benefit**: 5-10x faster on 165k+ datasets

### 2. TruncatedSVD for Discretised Recipe
- **File**: `features.py`
- **Features**: Reduces dimensions from ~42 to 24, uses sparse OneHotEncoder
- **Benefit**: Solves the "discretised too slow" problem

### 3. Sampling for Metrics
- **Files**: `evaluate.py`, `cluster.py`
- **Features**: Silhouette sampled at 15k (eval) and 30k (k-selection)
- **Benefit**: Massive speed improvement for evaluation

### 4. Float32 Casting & New CLI Parameters
- **File**: `run.py`
- **Features**: Memory reduction via float32, all new parameters exposed in CLI
- **Benefit**: 50% memory reduction

---

## Recommended Runs

### HIGH PRIORITY

#### Run 04: Continuous + KMeans (Main Approach)
**Purpose**: Replace HDBSCAN's 67% noise with balanced clusters

```bash
ukplc-seg \
  --input ./data/CSP_export.csv \
  --outdir ./out_04_continuous_kmeans_main \
  --recipe continuous \
  --algorithm kmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE,CURRENT_YEAR_FAP \
  --cast-float32 --random-state 42
```

**Expected**: 6-10 clusters, Silhouette ~0.35-0.45, 100% coverage (no noise)

---

#### Run 05: Continuous + MiniBatchKMeans (Speed Comparison)
**Purpose**: Test speed improvement with different scaler

```bash
ukplc-seg \
  --input ./data/CSP_export.csv \
  --outdir ./out_05_continuous_mbkmeans_fast \
  --recipe continuous \
  --algorithm mbkmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 --mbk-batch-size 8192 --mbk-max-iter 100 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler standard \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42
```

**Expected**: Same cluster count as Run 04, but 3-5x faster

---

#### Run 06: Discretised SVD + MiniBatchKMeans (Speed Fix)
**Purpose**: Solve "discretised too slow" problem with SVD

```bash
ukplc-seg \
  --input ./data/CSP_export.csv \
  --outdir ./out_06_discretised_svd_mbkmeans \
  --recipe discretised \
  --n-bins 5 --svd-components 24 \
  --algorithm mbkmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 --mbk-batch-size 8192 --mbk-max-iter 100 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --cast-float32 --random-state 42
```

**Expected**: 5-10 minutes runtime (vs 30+ min for old discretised), similar clusters to Run 04/05

---

### MEDIUM PRIORITY

#### Run 07: DCSP (Digital Channel)
**Purpose**: Separate segmentation for Digital customers

**Note**: Only run if DCSP data is available

```bash
ukplc-seg \
  --input ./data/DCSP_export.csv \
  --outdir ./out_07_dcsp_continuous_kmeans \
  --recipe continuous \
  --algorithm kmeans \
  --k-min 5 --k-max 9 --k-select silhouette \
  --kmeans-n-init 20 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42
```

**Expected**: Fewer clusters than CSP (Digital behavior more homogeneous)

---

### OPTIONAL

#### Run 08: HDBSCAN Noise Reduction (Proof of Concept)
**Purpose**: Test if lowering min_cluster_size reduces 67% noise

**Warning**: Run on sample first (60k rows) to verify

```bash
ukplc-seg \
  --input ./data/CSP_export.csv \
  --outdir ./out_08_hdbscan_low_noise_test \
  --recipe continuous \
  --algorithm hdbscan \
  --hdbscan-min-cluster-size 50 --hdbscan-min-samples 10 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42 \
  --rows-limit 60000
```

**Expected**: Less noise than 67%, but still may be too granular

---

## Execution Order

**Recommended sequence**:

1. **Start with Run 04** (continuous KMeans) - establishes baseline
2. **Run 05** (MiniBatchKMeans) - validates speed improvement
3. **Run 06** (Discretised SVD) - validates discretised speed fix
4. **Run 07** (DCSP) - if data available
5. **Run 08** (HDBSCAN) - only if time permits and interested in density-based approach

---

## What to Monitor

After each run, check:

1. **Runtime**: Run 05/06 should be significantly faster than previous discretised runs
2. **Cluster count**: Should be 6-10 (more actionable than 118 or 3)
3. **Silhouette**: Target 0.35-0.45 (balanced between separation and coverage)
4. **Cluster sizes**: Check `cluster_profiles.csv` for balance (avoid 90% in one cluster)
5. **Cross-tab**: Review `cluster_vs_existing_segment.csv` for A/B/C/D/E/F mapping

---

## Files to Return for Analysis

After runs complete, provide:

1. `internal_metrics.json` - All metrics
2. `cluster_profiles.csv` - Size distribution
3. `cluster_vs_existing_segment.csv` - Segment mapping
4. Runtime logs (to verify speed improvements)

---

## Expected Improvements vs Runs 01-03

| Metric | Runs 01/03 (HDBSCAN) | Run 02 (Discretised) | Runs 04-06 (New) |
|--------|----------------------|----------------------|------------------|
| **Clusters** | 118-120 | 3 | 6-10 |
| **Silhouette** | 0.47 | 0.26 | 0.35-0.45 |
| **Noise** | 67% | 0% | 0% |
| **Runtime (165k)** | ~3 min | ~30 min | ~5-7 min |
| **Interpretability** | Low (too granular) | High (too broad) | Medium-High |

---

## Questions for User

1. **Priority**: Should we run all 4 main runs (04-07) or focus on 04-06 first?
2. **DCSP**: Do you have DCSP data ready for Run 07?
3. **Platform**: Azure ML or local for these runs? (Azure recommended for speed)
4. **Parallel**: Can we run 04, 05, 06 in parallel on Azure?

---

## Next Steps

1. User confirms which runs to execute
2. Execute runs on Azure ML (faster) or local
3. Collect outputs and create comparison report
4. Fine-tune parameters based on results
