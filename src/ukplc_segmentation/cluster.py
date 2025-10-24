
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Try to import HDBSCAN if available
try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

@dataclass
class ClusterConfig:
    algorithm: str = "auto"  # "auto", "kmeans", "mbkmeans", "hdbscan", "agglomerative"
    k_min: int = 3
    k_max: int = 12
    random_state: int = 42
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: Optional[int] = None
    k_select: str = "composite"  # "composite" or "silhouette"
    mbk_batch_size: int = 8192
    mbk_max_iter: int = 100
    k_select_sample_size: Optional[int] = 30000  # Sample size for k-selection (None = use all data)

def silhouette_sampled(X: np.ndarray, labels: np.ndarray, sample_size: int = 30000, random_state: int = 42) -> float:
    """
    Compute silhouette score on a sample of rows to speed up k-selection.

    Args:
        X: Feature matrix
        labels: Cluster labels
        sample_size: Maximum number of samples to use
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

def select_k(X: np.ndarray, cfg: ClusterConfig) -> Tuple[int, Dict[int, float]]:
    """
    Grid search k in [k_min, k_max] using a composite or silhouette score.
    Optionally samples data for faster silhouette computation during k-selection.

    Args:
        X: Feature matrix
        cfg: Cluster configuration with k_select_sample_size parameter

    Returns:
        Tuple of (best_k, results_dict)
    """
    results: Dict[int, float] = {}
    best_k = cfg.k_min
    best_score = -np.inf

    # Sample data for k-selection if requested
    X_kselect = X
    if cfg.k_select_sample_size is not None and X.shape[0] > cfg.k_select_sample_size:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(X.shape[0], size=cfg.k_select_sample_size, replace=False)
        X_kselect = X[idx]

    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=cfg.random_state)
        labels = km.fit_predict(X_kselect)
        if len(set(labels)) < 2:
            score = -np.inf
        else:
            # Silhouette computation is now on sampled data
            sil = silhouette_score(X_kselect, labels)
            if cfg.k_select == "silhouette":
                score = sil
            else:
                ch = calinski_harabasz_score(X_kselect, labels)
                db = davies_bouldin_score(X_kselect, labels)
                score = sil + (ch / 10000.0) - db  # simple composite
        results[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, results

def fit_cluster_model(X: np.ndarray, cfg: ClusterConfig):
    """
    Fit clustering per config. If algorithm='auto', try HDBSCAN first (if available), and fall back to KMeans with k selection.
    Returns fitted model and labels.
    """
    if cfg.algorithm == "hdbscan" or (cfg.algorithm == "auto" and _HDBSCAN_AVAILABLE):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size,
                                    min_samples=cfg.hdbscan_min_samples,
                                    prediction_data=True)
        labels = clusterer.fit_predict(X)
        # If HDBSCAN collapses to a single cluster or all noise, fall back to KMeans
        valid = [l for l in labels if l != -1]
        if len(set(labels)) <= 1 or len(set(valid)) < 2:
            k, _ = select_k(X, cfg)
            km = KMeans(n_clusters=k, n_init="auto", random_state=cfg.random_state)
            labels = km.fit_predict(X)
            return km, labels
        return clusterer, labels

    elif cfg.algorithm in ("kmeans", "auto"):
        k, _ = select_k(X, cfg)
        km = KMeans(n_clusters=k, n_init="auto", random_state=cfg.random_state)
        labels = km.fit_predict(X)
        return km, labels

    elif cfg.algorithm == "mbkmeans":
        k, _ = select_k(X, cfg)
        mbk = MiniBatchKMeans(n_clusters=k,
                              batch_size=cfg.mbk_batch_size,
                              max_iter=cfg.mbk_max_iter,
                              n_init=20,
                              random_state=cfg.random_state)
        labels = mbk.fit_predict(X)
        return mbk, labels

    elif cfg.algorithm == "agglomerative":
        k, _ = select_k(X, cfg)
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        return agg, labels

    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
