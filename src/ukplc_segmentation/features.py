
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, KBinsDiscretizer, FunctionTransformer
from sklearn.decomposition import TruncatedSVD, PCA

# Picklable transformation functions (cannot use lambdas)
def clip_to_non_negative(X):
    """Clip values to be non-negative (for log1p preprocessing)."""
    return np.maximum(X, 0)

def log1p_transform(X):
    """Apply log1p transformation."""
    return np.log1p(X)

# Default feature names (aligned to UKPLC_CLTV_CSP_MV output columns)
DEFAULT_NUMERIC_FEATURES = [
    "ACTUAL_CLTV",
    "CURRENT_YEAR_FAP",
    "FUTURE_LIFETIME_VALUE",
    "ACTUAL_LIFETIME_DURATION",
    "NUM_CROSS_SOLD_LY",
    "CLM_OVER_PROFIT_HITCOUNT",
]

# Optional columns used only for profiling/explanation (not for clustering)
DEFAULT_EXPLANATORY_COLUMNS = [
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

@dataclass
class FeatureConfig:
    numeric_features: List[str] = None
    explanatory_columns: List[str] = None
    recipe: str = "continuous"  # "continuous" or "discretised"
    n_bins: int = 7  # for discretised recipe
    random_state: int = 42
    scaler: str = "robust"  # "robust", "standard", "none"
    log1p_columns: List[str] | None = None
    svd_components: int | None = None  # None means no SVD, used in discretised recipe
    pca_whiten: bool = False  # PCA whitening for continuous recipe (orthogonalize feature space)
    pca_variance: float = 0.95  # Variance to retain when whitening (0-1 range)

    def __post_init__(self):
        if self.numeric_features is None:
            self.numeric_features = list(DEFAULT_NUMERIC_FEATURES)
        if self.explanatory_columns is None:
            self.explanatory_columns = list(DEFAULT_EXPLANATORY_COLUMNS)
        if self.log1p_columns is None:
            self.log1p_columns = []

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):  # pragma: no cover
        return self
    def transform(self, X):  # pragma: no cover
        return X

def _pick_scaler(name: str):
    if name == "standard":
        return StandardScaler(with_mean=True, with_std=True)
    if name == "none":
        return IdentityTransformer()
    return RobustScaler(with_centering=True, with_scaling=True)

def build_continuous_pipeline(cfg: FeatureConfig) -> Tuple[Pipeline, List[str]]:
    """
    Impute + optional log1p + scaling + optional PCA whitening for numeric features.

    PCA whitening orthogonalizes the feature space and equalizes variances, which can improve
    cluster margin calculations and distance-based metrics, especially for elongated clusters.
    """
    num_features = cfg.numeric_features
    log_cols = [c for c in (cfg.log1p_columns or []) if c in num_features]
    other_cols = [c for c in num_features if c not in log_cols]

    scaler = _pick_scaler(cfg.scaler)

    transformers = []
    if other_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", scaler),
            ]),
            other_cols,
        ))
    if log_cols:
        transformers.append((
            "log_num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clip", FunctionTransformer(clip_to_non_negative, validate=False)),
                ("log1p", FunctionTransformer(log1p_transform, validate=False)),
                ("scale", scaler),
            ]),
            log_cols,
        ))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    # Build pipeline steps
    steps = [("preprocess", pre)]

    # Add PCA whitening step if enabled
    if cfg.pca_whiten:
        # n_components can be float (variance to retain) or int (exact components)
        # Using float value (0-1) retains that fraction of variance
        steps.append((
            "pca_whiten",
            PCA(
                n_components=cfg.pca_variance,
                whiten=True,
                random_state=cfg.random_state
            )
        ))

    pipe = Pipeline(steps=steps)
    return pipe, num_features

def build_discretised_pipeline(cfg: FeatureConfig) -> Tuple[Pipeline, List[str]]:
    """
    Case-study style: quantile discretisation per feature to ordinal bins (e.g., 7 bins),
    optional log1p prior to binning, then one-hot encode the ordinal categories.
    Optionally applies TruncatedSVD for dimensionality reduction after one-hot encoding.
    """
    num_features = cfg.numeric_features
    log_cols = [c for c in (cfg.log1p_columns or []) if c in num_features]
    other_cols = [c for c in num_features if c not in log_cols]
    disc = KBinsDiscretizer(n_bins=cfg.n_bins, encode="ordinal", strategy="quantile")

    # Use sparse_output=True when SVD is enabled for memory efficiency
    sparse_out = cfg.svd_components is not None

    transformers = []
    if other_cols:
        transformers.append((
            "num_disc",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("disc", disc),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_out)),
            ]), other_cols))
    if log_cols:
        transformers.append((
            "log_disc",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clip", FunctionTransformer(clip_to_non_negative, validate=False)),
                ("log1p", FunctionTransformer(log1p_transform, validate=False)),
                ("disc", disc),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_out)),
            ]), log_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    # Build pipeline steps
    steps = [("preprocess", pre)]

    # Add SVD step if configured
    if cfg.svd_components:
        steps.append(("svd", TruncatedSVD(n_components=cfg.svd_components, random_state=cfg.random_state)))

    pipe = Pipeline(steps=steps)
    return pipe, num_features

def build_feature_pipeline(cfg: FeatureConfig) -> Tuple[Pipeline, List[str]]:
    if cfg.recipe == "continuous":
        return build_continuous_pipeline(cfg)
    elif cfg.recipe == "discretised":
        return build_discretised_pipeline(cfg)
    else:
        raise ValueError(f"Unknown recipe: {cfg.recipe}")
