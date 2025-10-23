
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Try to import HDBSCAN if available
try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

@dataclass
class ClusterConfig:
    algorithm: str = "auto"  # "auto", "kmeans", "hdbscan", "agglomerative"
    k_min: int = 3
    k_max: int = 12
    random_state: int = 42
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: Optional[int] = None
    k_select: str = "composite"  # "composite" or "silhouette"

def select_k(X: np.ndarray, cfg: ClusterConfig) -> Tuple[int, Dict[int, float]]:
    """
    Grid search k in [k_min, k_max] using a composite or silhouette score.
    """
    results: Dict[int, float] = {}
    best_k = cfg.k_min
    best_score = -np.inf
    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=cfg.random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            score = -np.inf
        else:
            sil = silhouette_score(X, labels)
            if cfg.k_select == "silhouette":
                score = sil
            else:
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
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

    elif cfg.algorithm == "agglomerative":
        k, _ = select_k(X, cfg)
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        return agg, labels

    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
