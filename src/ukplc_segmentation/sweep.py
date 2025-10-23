
from __future__ import annotations
import argparse, itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .io import ensure_dir, load_dataset, save_csv
from .features import FeatureConfig, build_feature_pipeline
from .cluster import ClusterConfig, fit_cluster_model
from .evaluate import compute_internal_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Grid sweep for UKPLC segmentation (compare settings).")
    p.add_argument("--input", required=True, help="CSV/Parquet dataset")
    p.add_argument("--outdir", required=True, help="Output directory for sweep results")
    p.add_argument("--recipes", default="continuous,discretised", help="Comma-separated list")
    p.add_argument("--algorithms", default="auto,kmeans,hdbscan", help="Comma-separated list")
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--k-select", choices=["composite","silhouette"], default="composite")
    p.add_argument("--n-bins", type=int, default=7)
    p.add_argument("--scaler", choices=["robust","standard","none"], default="robust")
    p.add_argument("--log1p", help="Comma-separated feature names to log1p")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    p.add_argument("--hdbscan-min-samples", type=int)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--rows-limit", type=int)
    p.add_argument("--features", help="Comma-separated feature names (override defaults)")
    return p.parse_args()

DEFAULT_FEATURES = [
    "ACTUAL_CLTV",
    "CURRENT_YEAR_FAP",
    "FUTURE_LIFETIME_VALUE",
    "ACTUAL_LIFETIME_DURATION",
    "NUM_CROSS_SOLD_LY",
    "CLM_OVER_PROFIT_HITCOUNT",
]

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.input)
    if args.rows_limit:
        df = df.head(args.rows_limit).copy()

    features_list = DEFAULT_FEATURES
    if args.features:
        features_list = [c.strip() for c in args.features.split(",") if c.strip()]
    features_list = [c for c in features_list if c in df.columns]

    log_cols = [c.strip() for c in (args.log1p or "").split(",") if c.strip()]
    log_cols = [c for c in log_cols if c in df.columns]

    recipes = [s.strip() for s in args.recipes.split(",") if s.strip()]
    algorithms = [s.strip() for s in args.algorithms.split(",") if s.strip()]

    rows = []
    run_id = 0
    for recipe, algorithm in itertools.product(recipes, algorithms):
        fcfg = FeatureConfig(numeric_features=features_list, explanatory_columns=[], recipe=recipe,
                             n_bins=args.n_bins, random_state=args.random_state, scaler=args.scaler,
                             log1p_columns=log_cols)
        pipe, feat_names = build_feature_pipeline(fcfg)
        X = pipe.fit_transform(df)

        ccfg = ClusterConfig(algorithm=algorithm, k_min=args.k_min, k_max=args.k_max, random_state=args.random_state,
                             hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, hdbscan_min_samples=args.hdbscan_min_samples,
                             k_select=args.k_select)
        model, labels = fit_cluster_model(X, ccfg)
        metrics = compute_internal_metrics(X, labels)
        rows.append({
            "run_id": run_id,
            "recipe": recipe,
            "algorithm": algorithm,
            "n_clusters": metrics.n_clusters,
            "silhouette": metrics.silhouette,
            "calinski_harabasz": metrics.calinski_harabasz,
            "davies_bouldin": metrics.davies_bouldin,
        })
        run_id += 1

    df_out = pd.DataFrame(rows).sort_values(["silhouette","calinski_harabasz"], ascending=[False, False])
    save_csv(df_out, outdir / "experiments_summary.csv")
    print("Wrote", outdir / "experiments_summary.csv")

if __name__ == "__main__":
    main()
