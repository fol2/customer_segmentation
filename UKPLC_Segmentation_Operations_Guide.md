# UKPLC Segmentation — **Operations User Guide** (v2)

> **Scope**: This guide describes how to **set up, operate, configure, and interpret** the `ukplc-segmentation` package to produce **customer clustering (segmentation)** from **performance features**. It covers **CSP (Mid‑Market)** and **DCSP (Digital)** runs using the same codebase and parameterised CLI/YAML configuration.

---

## 0. At a Glance

- **One codebase** for both **CSP** (`UKPLC_CLTV_CSP_MV`) and **DCSP** (`UKPLC_CLTV_DCSP_MV`) exports — identical schema, run separately with different input files.  
- **Two feature recipes**:  
  1) `continuous` – impute → optional `log1p` → scaling (robust/standard/none) → clustering.  
  2) `discretised` – quantile binning + one‑hot (aligned to the case‑study approach).  
- **Algorithms**: `auto` (HDBSCAN if available, fallback to KMeans with **auto‑k**), or force `kmeans` / `hdbscan` / `agglomerative`.
- **Parameterised**: K range & selection rule, scaler type, `log1p`, bin count, HDBSCAN settings, filters, ID column, row limit, YAML config.
- **Outputs**: cluster assignments, cluster profiles (incl. *product_index*), internal metrics (silhouette/CH/DB), explainability (RF + SHAP), cross‑tab vs. existing segments, quick visual.

> **Data source**: Exports from CSP/DCSP MV already contain the six **performance features** and **explanatory factual columns** used here (see §2), including `INSP_FLAG`, `CUSTOMER_PORTFOLIO`, `CUSTOMER_SEGMENT`.  

---

## 1. Prerequisites & Installation

### 1.1 Platform Requirements
- **Python**: 3.11
- **Package manager**: [`uv`](https://github.com/astral-sh/uv)
- **OS**: Linux or Windows (WSL2 recommended for Windows)
- **Memory**: ≥ 8 GB RAM recommended for >250k rows

### 1.2 Install with `uv`

```bash
uv venv -p 3.11
uv pip install -e /path/to/unzip/ukplc_segmentation_v2
```

> The package uses `pyproject.toml` (no `requirements.txt`). Optional extra `kmodes` (group `mixed`) is available if you later need K‑Prototypes/K‑Modes.

---

## 2. Data Inputs

### 2.1 Source Views & Extracts
Run on CSV/Parquet exports from the **CSP (Mid‑Market)** or **DCSP (Digital)** materialised views. Both share the same schema; portfolio hard‑cut ensures customers belong to exactly one portfolio. CSP v5.3 also surfaces `INSP_FLAG` and supports inspection handling.  
*References*: CSP v5.3 SQL and documentation. fileciteturn0file0turn0file1

### 2.2 Required **Performance Features** (used for clustering)
- `ACTUAL_CLTV`
- `CURRENT_YEAR_FAP`
- `FUTURE_LIFETIME_VALUE`
- `ACTUAL_LIFETIME_DURATION`
- `NUM_CROSS_SOLD_LY`
- `CLM_OVER_PROFIT_HITCOUNT`

> These six columns are already computed in CSP/DCSP outputs; definitions derive from **policy**, **FLV**, **customer**, and **CLTV** marts (see §2.4 for lineage). fileciteturn0file1

### 2.3 Recommended **Explanatory** columns (not used for clustering)
For post‑hoc interpretation and reporting:
`CUSTOMER_SEGMENT`, `CUSTOMER_PORTFOLIO`, `ACTIVE_CUSTOMER`, `CURRENT_GWP`, `LAST_YEAR_GWP`,
`GEP_FIN_FULL`, `GEC_FIN_FULL`, `CLM_INC_FULL`, `WEIGHTED_PLAN_LOSS_RATIO`, `ACTUAL_LOSS_RATIO`,
`EXPECTED_LIFETIME_VALUE`, `TOTAL_SCORE`, `INSP_FLAG`. fileciteturn0file1

### 2.4 Lineage & Metric Definitions
- **Policy‑level** CLTV/financials: `UKPLC_CLTV_POLICY_MV` (NEP, incurred claims, commission, dates, etc.). fileciteturn0file2  
- **Future Lifetime Value (FLV)** & LOB‑aware/WPE/WPLR/FLD logic: `UKPLC_CLTV_FLV_MV`. fileciteturn0file3  
- **Customer‑level** aggregations (ELV = Actual + FLV; durations; financials): `UKPLC_CLTV_CUSTOMER_MV`. fileciteturn0file4  
- **Actual CLTV** (LOB‑aware with WPE proxy) canonical view: `UKPLC_CLTV_CLTV_MV`. fileciteturn0file5  
- **Broker retention / FLD** (cap at 20 years): `UKPLC_CLTV_FLD_MV`. fileciteturn0file6

> CSP v5.3 integrates these metrics and provides dynamic segmentation fields, inspection handling, and cross‑sell configuration (e.g., `XS_DIM_COL`, `XS_USE_LAST_YEAR`). fileciteturn0file0

---

## 3. Running the Tool

### 3.1 Single Run (CLI)

```bash
ukplc-seg   --input /data/CSP_export.csv   --outdir ./out_csp   --recipe discretised --n-bins 7   --algorithm auto --k-select composite   --scaler robust --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

```bash
ukplc-seg   --input /data/DCSP_export.parquet   --outdir ./out_dcsp   --recipe continuous   --algorithm auto --k-min 4 --k-max 12 --k-select silhouette   --scaler standard --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

**Filtering** (repeatable): `--filter COL=VAL` (e.g., `--filter CUSTOMER_PORTFOLIO=MID`, `--filter ACTIVE_CUSTOMER=Y`).  
**Inspection handling**: By default **included**. To exclude inspection‑only customers: add `--exclude-inspection-only` (uses `INSP_FLAG`). fileciteturn0file0

### 3.2 Parameter Sweep (Compare Settings)

```bash
ukplc-seg-sweep   --input /data/CSP_export.csv   --outdir ./sweep_csp   --recipes continuous,discretised   --algorithms auto,kmeans,hdbscan   --k-min 3 --k-max 12 --k-select composite   --n-bins 7 --scaler robust --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

Outputs `experiments_summary.csv` with silhouette/CH/DB for quick selection.

### 3.3 YAML‑Driven Runs

```bash
ukplc-seg   --input /data/CSP_export.csv   --outdir ./out_csp_yaml   --config /path/CONFIG.yaml
```

See §6 for YAML schema. CLI flags override YAML entries.

---

## 4. Feature Recipes

### 4.1 `continuous`
- **Imputation**: median
- **Optional**: `log1p` per feature (e.g., heavy‑tailed CLTV/FLV)
- **Scaler**: `robust` (default) | `standard` | `none`
- **Use when**: features are continuous and you want distance/centroid or density‑based structure.

### 4.2 `discretised`
- **Quantile binning** (`n_bins`, default 7) → **One‑Hot** per feature
- Optional `log1p` then binning
- **Use when**: you want the case‑study style “bin‑then‑cluster” behaviour and robust handling of skew.

---

## 5. Clustering Algorithms & Auto‑k

- **`algorithm=auto`**: Try **HDBSCAN** (if installed); if single‑cluster or all‑noise, fallback to **KMeans** with **auto k**.  
- **Auto‑k**: search `k ∈ [k_min, k_max]` using either `k_select=composite` (silhouette + CH − DB) or `k_select=silhouette`.  
- **Force‑select**: `kmeans` or `agglomerative`; HDBSCAN parameters tunable via `--hdbscan-min-cluster-size`, `--hdbscan-min-samples`.

> Cross‑sell exposure (`NUM_CROSS_SOLD_LY`) and Current‑Year FAP are included by CSP even under inspection policy exclusion rules; other metrics exclude inspection when configured (CSP v5.3). This keeps performance features faithful to business logic. fileciteturn0file0

---

## 6. Configuration Reference

### 6.1 CLI Parameters

| Parameter | Type/Default | Purpose |
|---|---|---|
| `--input` | path | CSV/Parquet export from CSP/DCSP MV |
| `--outdir` | path | Output directory (created if absent) |
| `--config` | path | Optional YAML; CLI overrides YAML |
| `--recipe` | `continuous`/`discretised` (default `continuous`) | Feature engineering recipe |
| `--algorithm` | `auto`/`kmeans`/`hdbscan`/`agglomerative` (default `auto`) | Clustering algorithm |
| `--k-min/--k-max` | int (3/12) | k search bounds for KMeans/Agglomerative |
| `--k-select` | `composite` (default) / `silhouette` | Criterion to pick k |
| `--n-bins` | int (7) | Quantile bins for `discretised` |
| `--scaler` | `robust` (default) / `standard` / `none` | Scaling for `continuous` |
| `--log1p` | CSV of cols | Columns to `log1p` before scaling/binning |
| `--hdbscan-min-cluster-size` | int (100) | HDBSCAN min cluster size |
| `--hdbscan-min-samples` | int/None | HDBSCAN min samples |
| `--id-col` | `CUSTOMER_ID` | ID for assignments output |
| `--filter COL=VAL` | repeatable | Row filter (string/number/Y/N) |
| `--exclude-inspection-only` | flag | Exclude `INSP_FLAG='Y'` customers |
| `--rows-limit` | int | Quick dry‑run with head(n) |
| `--disable-shap` | flag | Skip SHAP computation |

### 6.2 YAML Schema (`CONFIG.yaml`)

```yaml
recipe: continuous   # or discretised
algorithm: auto
k_min: 3
k_max: 12
k_select: composite
n_bins: 7
random_state: 42
scaler: robust
log1p: [ACTUAL_CLTV, FUTURE_LIFETIME_VALUE]

# Explicit feature lists are optional; default to the 6 performance features.
features:
  - ACTUAL_CLTV
  - CURRENT_YEAR_FAP
  - FUTURE_LIFETIME_VALUE
  - ACTUAL_LIFETIME_DURATION
  - NUM_CROSS_SOLD_LY
  - CLM_OVER_PROFIT_HITCOUNT

# Explanatory columns are not used for clustering; they inform profiles.
explanatory:
  - CUSTOMER_SEGMENT
  - CUSTOMER_PORTFOLIO
  - ACTIVE_CUSTOMER
  - CURRENT_GWP
  - LAST_YEAR_GWP
  - GEP_FIN_FULL
  - GEC_FIN_FULL
  - CLM_INC_FULL
  - WEIGHTED_PLAN_LOSS_RATIO
  - ACTUAL_LOSS_RATIO
  - EXPECTED_LIFETIME_VALUE
  - TOTAL_SCORE
```

---

## 7. Outputs & Interpretation

| File | Description |
|---|---|
| `cluster_assignments.csv` | One row per customer with `cluster` and `id-col` |
| `cluster_profiles.csv` | Per‑cluster **size/share**, medians/means of the **six performance features** and means/modes of **explanatory** factual metrics; includes **`product_index`** = avg(`NUM_CROSS_SOLD_LY`) ÷ portfolio avg (case‑study “Product Index”). |
| `cluster_vs_existing_segment.csv` | Cross‑tab of new clusters vs `CUSTOMER_SEGMENT` (if present) — **diagnostic only** (no mapping). |
| `internal_metrics.json` | `n_clusters`, `silhouette`, `calinski_harabasz`, `davies_bouldin` (noise excluded for metrics when using HDBSCAN). |
| `explain.json` | RandomForest classification report + feature importances; SHAP summary (if available) for top features. |
| `cluster_sizes.png` | Quick bar chart of cluster sizes. |
| `REPORT.md` | Run summary with settings & metrics. |

**Reading the profiles**  
- Prioritise clusters with **high ELV/CLTV/GWP** and **lower Actual LR vs WPLR**; consider **cross‑sell diversity** via `product_index`.  
- Remember CSP’s inspection rules: **FAP** and **cross‑sell** may include inspection exposure while **CLTV/FLV/ELV** typically exclude inspection from calculations when configured. fileciteturn0file0

---

## 8. Recommended Operating Patterns

1) **CSP & DCSP**: Run twice (two input files) and archive outputs separately. Use the same YAML with minor overrides. fileciteturn0file1  
2) **Sweep → Pick**: Use `ukplc-seg-sweep` to shortlist recipe/algorithm combos; select by silhouette (and CH/DB).  
3) **Heavy Tails**: Add `--log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE` (often stabilises distances).  
4) **Explainability First**: Check `explain.json` to learn which features drive separation; use this when socialising results.  
5) **Stability (advanced)**: Repeat runs with different seeds; compare cluster counts and silhouette. (Bootstrap ARI/AMI can be added later.)

---

## 9. Troubleshooting

- **“Missing required columns”**: Confirm the six performance features exist in the export (see §2.2). fileciteturn0file1  
- **HDBSCAN yields 1 cluster / many `-1` noise**: Either increase sample size, reduce `--hdbscan-min-cluster-size`, or switch to `kmeans` with `--k-select silhouette`.  
- **SHAP slow / fails**: Use `--disable-shap` (still keeps RF importances).  
- **Memory**: Prefer Parquet; apply `--rows-limit` for smoke tests; drop unneeded columns before export.  
- **Inspection exclusion**: If business needs exclusion at runtime, add `--exclude-inspection-only` (uses MV’s `INSP_FLAG`). fileciteturn0file0

---

## 10. Performance, Scale & Hygiene

- **I/O**: Use **Parquet** for speed and types.  
- **Scaling**: For 0.5–1M rows, start with `continuous` + `kmeans` auto‑k; consider stratified samples for parameter tuning.  
- **Reproducibility**: Fix `--random-state`; store `internal_metrics.json`, `explain.json`, `REPORT.md`.  
- **Data hygiene**: Pre‑clean obvious outliers; ensure numeric dtypes; verify that **CSP/DCSP exports** have the expected inspection and portfolio flags. fileciteturn0file1

---

## 11. Governance & Audit

- **Data domain**: UK corporate customers only; **no personal data**.  
- **Lineage**: CSP/DCSP leverage policy/customer/FLV/CLTV/FLD marts as described in §2.4. Keep the export SQL & run logs with each segmentation run. fileciteturn0file0turn0file1turn0file2turn0file3turn0file4turn0file5turn0file6  
- **Inspection policies**: Be explicit in documentation whether inspection‑only customers were included or excluded in the run (see §3.1). fileciteturn0file0

---

## 12. Examples (Copy‑Paste)

### 12.1 CSP (Mid‑Market), discretised recipe

```bash
ukplc-seg   --input /data/CSP_export.csv   --outdir ./out_csp_disc   --recipe discretised --n-bins 7   --algorithm auto --k-select composite   --scaler robust --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

### 12.2 DCSP (Digital), continuous recipe + HDBSCAN

```bash
ukplc-seg   --input /data/DCSP_export.parquet   --outdir ./out_dcsp_cont   --recipe continuous --algorithm auto   --k-min 4 --k-max 12 --k-select silhouette   --scaler standard --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE
```

### 12.3 Exclude inspection‑only customers

```bash
ukplc-seg   --input /data/CSP_export.csv   --outdir ./out_csp_noinsp   --recipe continuous   --algorithm kmeans --k-min 5 --k-max 10   --exclude-inspection-only
```

### 12.4 Filter active MID portfolio only

```bash
ukplc-seg   --input /data/CSP_export.csv   --outdir ./out_csp_active_mid   --filter CUSTOMER_PORTFOLIO=MID   --filter ACTIVE_CUSTOMER=Y
```

### 12.5 YAML‑driven run

```bash
ukplc-seg --input /data/CSP_export.csv --outdir ./out_yaml --config /path/CONFIG.yaml
```

---

## 13. Appendix — Column Dictionary (selected)

| Column | Meaning / Origin |
|---|---|
| `ACTUAL_CLTV` | Historical profitability; LOB‑aware CLTV (policy‑level logic in CLTV MV, aggregated for CSP). fileciteturn0file5 |
| `FUTURE_LIFETIME_VALUE` | Forward value using WPLR/WPE and (lapse‑aware) FLD; LOB‑aware; see FLV MV. fileciteturn0file3 |
| `EXPECTED_LIFETIME_VALUE` | `ACTUAL_CLTV + FUTURE_LIFETIME_VALUE`; customer‑level also in CUSTOMER MV. fileciteturn0file4 |
| `CURRENT_YEAR_FAP` | Current year Full Annual Premium from policy aggregations. fileciteturn0file2 |
| `CLM_OVER_PROFIT_HITCOUNT` | Count of years where Actual LR > Plan LR (weighted plan loss ratios). fileciteturn0file1 |
| `NUM_CROSS_SOLD_LY` | Distinct count across configurable cross‑sell dimension (e.g., prefix/revenue group), year switchable. fileciteturn0file0 |
| `WEIGHTED_PLAN_LOSS_RATIO` | FAP‑weighted plan loss ratio. fileciteturn0file2 |
| `INSP_FLAG` | Inspection‑only customer flag derived from policy mix; surfaced by CSP. fileciteturn0file0 |
| `CUSTOMER_PORTFOLIO` | MID / DIGITAL (customer hard‑cut). fileciteturn0file1 |

---

## 14. Versioning & Change Log (tool)

- **v2 (this guide)**: Added parameter sweep, YAML, inspection inclusion by default, richer outputs.  
- **v1**: Initial release (continuous/discretised recipes; auto HDBSCAN→KMeans; profiles/metrics/explain).

---

## 15. Contact & Support

- **Runbook ownership**: DataLab – Allianz UK (Commercial).  
- For enhancements (stability metrics, AML/MLflow integration, automatic naming), raise a work item in the backlog.

---

**End of document.**
