
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from .io import ensure_dir, load_dataset, save_csv, save_json, save_fig
from .config import load_yaml, merge
from .features import FeatureConfig, build_feature_pipeline
from .cluster import ClusterConfig, fit_cluster_model
from .evaluate import compute_internal_metrics, cluster_profile_table
from .explain import ExplainConfig, fit_explainer

import matplotlib.pyplot as plt

DEFAULT_FEATURES = [
    "ACTUAL_CLTV",
    "CURRENT_YEAR_FAP",
    "FUTURE_LIFETIME_VALUE",
    "ACTUAL_LIFETIME_DURATION",
    "NUM_CROSS_SOLD_LY",
    "CLM_OVER_PROFIT_HITCOUNT",
]

DEFAULT_EXPLANATORY = [
    "CUSTOMER_SEGMENT",
    "CUSTOMER_PORTFOLIO",
    "ACTIVE_CUSTOMER",
    "CURRENT_GWP",
    "LAST_YEAR_GWP",
    "GEP_FIN_FULL",
    "GEC_FIN_FULL",
    "CLM_INC_FULL",
    "WEIGHTED_PLAN_LOSS_RATIO",
    "ACTUAL_LOSS_RATIO",
    "EXPECTED_LIFETIME_VALUE",
    "TOTAL_SCORE",
]

def parse_args():
    p = argparse.ArgumentParser(description="UKPLC segmentation via clustering on performance features.")
    p.add_argument("--input", required=True, help="Path to CSV/Parquet exported from UKPLC_CLTV_CSP_MV (or DCSP).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--config", help="Optional YAML config to override CLI parameters.")
    p.add_argument("--recipe", choices=["continuous", "discretised"], default="continuous",
                   help="Feature engineering recipe.")
    p.add_argument("--algorithm", choices=["auto", "kmeans", "mbkmeans", "hdbscan", "agglomerative"], default="auto",
                   help="Clustering algorithm.")
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--k-select", choices=["composite", "silhouette"], default="composite")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=7, help="Number of quantile bins for discretised recipe.")
    p.add_argument("--scaler", choices=["robust", "standard", "none"], default="robust")
    p.add_argument("--log1p", help="Comma-separated feature names to log1p before scaling/binning.")
    p.add_argument("--id-col", default="CUSTOMER_ID", help="Identifier column name for assignments.")
    p.add_argument("--rows-limit", type=int, help="Optional row limit for quick tests (head).")
    p.add_argument("--filter", action="append", help="Row filter in the form COL=VALUE. Repeatable.")
    p.add_argument("--exclude-inspection-only", action="store_true", help="Drop rows with INSP_FLAG='Y'. Default: include.")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    p.add_argument("--hdbscan-min-samples", type=int, help="HDBSCAN min_samples; default None.")
    p.add_argument("--disable-shap", action="store_true", help="Skip SHAP to speed up runs.")
    p.add_argument("--cast-float32", action="store_true", help="Cast numeric columns to float32 to reduce memory")
    p.add_argument("--mbk-batch-size", type=int, default=8192, help="MiniBatchKMeans batch size")
    p.add_argument("--mbk-max-iter", type=int, default=100, help="MiniBatchKMeans max iterations")
    p.add_argument("--svd-components", type=int, help="TruncatedSVD components for discretised recipe (e.g., 24)")
    p.add_argument("--silhouette-sample-size", type=int, default=15000, help="Sample size for silhouette computation")
    p.add_argument("--kselect-sample-size", type=int, default=30000, help="Sample size for k-selection")
    return p.parse_args()

def main():
    args = parse_args()
    # Load YAML config if provided and merge (CLI takes precedence)
    cfg_dict = {}
    if args.config:
        cfg_dict = load_yaml(args.config)
    yaml_features = cfg_dict.get("features")
    yaml_explanatory = cfg_dict.get("explanatory")

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.input)

    # Optional float32 casting to reduce memory
    if args.cast_float32:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        print(f"      Cast {len(numeric_cols)} numeric columns to float32")

    # Optional row limit for quick tests
    if args.rows_limit:
        df = df.head(args.rows_limit).copy()

    # Optional filters: --filter "COL=VAL" (repeatable)
    if args.filter:
        for f in args.filter:
            if "=" not in f:
                continue
            col, val = f.split("=", 1)
            col = col.strip()
            val = val.strip()
            if col in df.columns:
                if val.upper() in {"Y", "N"}:
                    df = df[df[col].astype(str).str.upper() == val.upper()]
                else:
                    try:
                        num = float(val)
                        df = df[(pd.to_numeric(df[col], errors="coerce") == num)]
                    except Exception:
                        df = df[df[col].astype(str) == val]
    # Optional exclude inspection-only
    if args.exclude_inspection_only and ("INSP_FLAG" in df.columns):
        df = df[df["INSP_FLAG"].astype(str).str.upper() != "Y"].copy()
        print(f"      After filters: {len(df):,} rows remaining")
    else:
        print(f"      After filters: {len(df):,} rows")

    # Features from YAML override if present
    features_list = yaml_features if yaml_features else [c for c in DEFAULT_FEATURES if c in df.columns]
    explanatory_list = yaml_explanatory if yaml_explanatory else [c for c in DEFAULT_EXPLANATORY if c in df.columns]

    # Log1p columns
    log_cols = []
    if args.log1p:
        log_cols = [c.strip() for c in args.log1p.split(",") if c.strip()]

    print(f"[4/9] Building feature pipeline (recipe={args.recipe})...")
    fcfg = FeatureConfig(
        numeric_features=[c for c in features_list if c in df.columns],
        explanatory_columns=[c for c in explanatory_list if c in df.columns],
        recipe=args.recipe,
        n_bins=args.n_bins,
        random_state=args.random_state,
        scaler=args.scaler,
        log1p_columns=log_cols,
        svd_components=args.svd_components,
    )
    pipe, feature_names = build_feature_pipeline(fcfg)
    print(f"[5/9] Transforming features ({len(fcfg.numeric_features)} features)...")
    X = pipe.fit_transform(df)
    print(f"      Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")

    # Fit clustering
    print(f"[6/9] Fitting clustering model (algorithm={args.algorithm})...")
    if args.algorithm in ['kmeans', 'auto'] and args.k_min != args.k_max:
        print(f"      Testing k from {args.k_min} to {args.k_max}...")
    ccfg = ClusterConfig(
        algorithm=args.algorithm,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        k_select=args.k_select,
        mbk_batch_size=args.mbk_batch_size,
        mbk_max_iter=args.mbk_max_iter,
        k_select_sample_size=args.kselect_sample_size,
    )
    model, labels = fit_cluster_model(X, ccfg)
    print(f"      Clustering complete! Found {len(set(labels))} clusters")

    # Metrics
    print(f"[7/9] Computing metrics...")
    metrics = compute_internal_metrics(X, labels, sample_size=args.silhouette_sample_size)
    metrics_json = {
        "n_clusters": metrics.n_clusters,
        "silhouette": metrics.silhouette,
        "calinski_harabasz": metrics.calinski_harabasz,
        "davies_bouldin": metrics.davies_bouldin,
    }

    # Profiles
    prof = cluster_profile_table(df, labels, fcfg.explanatory_columns, fcfg.numeric_features)

    # Save artefacts
    print(f"[8/9] Saving outputs to {outdir}...")
    id_col = args.id_col if args.id_col in df.columns else "CUSTOMER_ID"
    assignments = pd.DataFrame({id_col: df.get(id_col, pd.Series(range(len(df)))),
                                "cluster": labels})
    save_csv(assignments, outdir / "cluster_assignments.csv")
    save_csv(prof, outdir / "cluster_profiles.csv")
    save_json(metrics_json, outdir / "internal_metrics.json")
    print(f"      Saving model.joblib (this may take a moment)...")
    dump({"pipeline": pipe, "cluster_model": model, "feature_names": feature_names},
         outdir / "model.joblib")
    print(f"      Model saved successfully")

    # Quick visual: cluster sizes
    fig, ax = plt.subplots()
    prof.plot.bar(x="cluster", y="n_customers", ax=ax, legend=False)
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    save_fig(fig, outdir / "cluster_sizes.png")

    # Explainability
    print(f"[9/9] Generating explanations...")
    explain = {}
    if not args.disable_shap:
        print(f"      Computing SHAP values (this may take 1-2 minutes)...")
        try:
            # build_feature_pipeline already imported at top
            fcfg_explain = FeatureConfig(numeric_features=fcfg.numeric_features, explanatory_columns=fcfg.explanatory_columns, recipe="continuous",
                                         scaler=args.scaler, log1p_columns=log_cols)
            pipe_explain, feat_names_explain = build_feature_pipeline(fcfg_explain)
            X_explain = pipe_explain.fit_transform(df)
            e = fit_explainer(X_explain, labels, feat_names_explain, ExplainConfig(random_state=args.random_state))
            explain = e
        except Exception as ex:
            explain = {"error": str(ex)}
    save_json(explain, outdir / "explain.json")

    # Cross-tab with existing CUSTOMER_SEGMENT (if available)
    if "CUSTOMER_SEGMENT" in df.columns:
        ctab = (assignments.assign(CUSTOMER_SEGMENT=df["CUSTOMER_SEGMENT"])
                .pivot_table(index="cluster", columns="CUSTOMER_SEGMENT", values=id_col, aggfunc="count", fill_value=0))
        ctab.reset_index().to_csv(outdir / "cluster_vs_existing_segment.csv", index=False)

    # Minimal Markdown report
    report = f"""# UKPLC Segmentation Report

- **Recipe**: {args.recipe}
- **Algorithm**: {args.algorithm}
- **Detected clusters**: {metrics_json['n_clusters']}
- **Silhouette**: {metrics_json['silhouette']}
- **Calinski–Harabasz**: {metrics_json['calinski_harabasz']}
- **Davies–Bouldin**: {metrics_json['davies_bouldin']}

## Notes
- Input assumed exported from `UKPLC_CLTV_CSP_MV`/`UKPLC_CLTV_DCSP_MV`, containing performance features
  (`ACTUAL_CLTV`, `CURRENT_YEAR_FAP`, `FUTURE_LIFETIME_VALUE`, `ACTUAL_LIFETIME_DURATION`, `NUM_CROSS_SOLD_LY`, `CLM_OVER_PROFIT_HITCOUNT`).
- Clustering uses only performance features; explanatory columns describe clusters but are never used as inputs.
- SHAP explanations are provided by training a simple RandomForest to predict cluster labels and summarising mean absolute SHAP values.
"""
    with open(outdir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("Done. Outputs written to:", outdir)

if __name__ == "__main__":
    main()
