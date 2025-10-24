
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

@dataclass
class EvaluationResult:
    n_clusters: int
    silhouette: Optional[float]
    calinski_harabasz: Optional[float]
    davies_bouldin: Optional[float]

def silhouette_sampled(X: np.ndarray, labels: np.ndarray, sample_size: int = 15000, random_state: int = 42) -> float:
    """
    Compute silhouette score on a sample of rows to speed up evaluation.

    Args:
        X: Feature matrix
        labels: Cluster labels
        sample_size: Maximum number of samples to use (default 15000)
        random_state: Random seed for reproducibility

    Returns:
        Silhouette score computed on sampled data
    """
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        return silhouette_score(X[idx], labels[idx])
    return silhouette_score(X, labels)

def compute_internal_metrics(X: np.ndarray, labels: np.ndarray, sample_size: Optional[int] = None) -> EvaluationResult:
    """
    Compute internal clustering metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels
        sample_size: If specified, sample this many rows for silhouette computation (default: None, use all rows)

    Returns:
        EvaluationResult with silhouette, Calinski-Harabasz, and Davies-Bouldin scores
    """
    unique = set(labels)
    # Handle HDBSCAN noise label -1 by ignoring it for metrics if it dominates
    mask = labels != -1 if (-1 in unique and len(unique) > 1) else np.ones_like(labels, dtype=bool)
    X_eval = X[mask]
    y_eval = labels[mask]
    if len(set(y_eval)) < 2:
        return EvaluationResult(n_clusters=len(set(labels)), silhouette=None, calinski_harabasz=None, davies_bouldin=None)

    # Use sampled silhouette if sample_size is specified, otherwise compute on full data
    if sample_size is not None:
        sil = silhouette_sampled(X_eval, y_eval, sample_size=sample_size)
    else:
        sil = silhouette_score(X_eval, y_eval)

    ch = calinski_harabasz_score(X_eval, y_eval)
    db = davies_bouldin_score(X_eval, y_eval)
    return EvaluationResult(n_clusters=len(set(labels)) - (1 if -1 in unique else 0),
                            silhouette=sil, calinski_harabasz=ch, davies_bouldin=db)

def cluster_profile_table(df: pd.DataFrame,
                          labels: np.ndarray,
                          explanatory_columns: List[str],
                          numeric_features: List[str]) -> pd.DataFrame:
    """
    Create a tidy profile table with:
      - cluster size
      - median/mean of numeric features used for clustering
      - mean of explanatory factual metrics (not used in clustering)
      - product index analogue: avg NUM_CROSS_SOLD_LY vs portfolio avg
    """
    out = df.copy()
    out["__cluster__"] = labels
    stats_cols = [c for c in explanatory_columns if c in out.columns]
    feat_cols = [c for c in numeric_features if c in out.columns]
    group = out.groupby("__cluster__", dropna=False)

    def safe_mean(x):
        return float(np.nanmean(pd.to_numeric(x, errors="coerce"))) if len(x) else np.nan
    def safe_median(x):
        return float(np.nanmedian(pd.to_numeric(x, errors="coerce"))) if len(x) else np.nan

    records = []
    portfolio_avg_xs = safe_mean(out.get("NUM_CROSS_SOLD_LY", pd.Series(dtype=float)))

    for cl, g in group:
        rec = {
            "cluster": int(cl),
            "n_customers": int(len(g)),
            "share": float(len(g) / len(out)),
        }
        for f in feat_cols:
            rec[f"median_{f}"] = safe_median(g[f])
            rec[f"mean_{f}"] = safe_mean(g[f])
        for s in stats_cols:
            if s in {"CUSTOMER_SEGMENT", "CUSTOMER_PORTFOLIO", "ACTIVE_CUSTOMER"}:
                top = g[s].mode(dropna=True)
                rec[f"mode_{s}"] = (None if top.empty else str(top.iloc[0]))
            else:
                rec[f"mean_{s}"] = safe_mean(g[s])
        if "NUM_CROSS_SOLD_LY" in g.columns and portfolio_avg_xs not in (None, 0, np.nan):
            rec["product_index"] = safe_mean(g["NUM_CROSS_SOLD_LY"]) / portfolio_avg_xs
        records.append(rec)
    prof = pd.DataFrame.from_records(records).sort_values(["share", "cluster"], ascending=[False, True])
    return prof
