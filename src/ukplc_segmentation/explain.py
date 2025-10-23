
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# SHAP is optional
try:
    import shap
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False

@dataclass
class ExplainConfig:
    random_state: int = 42
    test_size: float = 0.25
    max_samples: int = 2000  # sample for speed
    shap_summary_max_display: int = 12

def fit_explainer(X: np.ndarray, y: np.ndarray, feature_names: List[str], cfg: ExplainConfig) -> Dict[str, Any]:
    """
    Train a simple RandomForest to predict cluster labels and return SHAP values (if available)
    plus classification report (macro metrics are helpful to see if clusters are separable).
    """
    mask = y != -1
    X = X[mask]
    y = y[mask]
    if X.shape[0] > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(X.shape[0], size=cfg.max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, random_state=cfg.random_state, class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    result: Dict[str, Any] = {"classification_report": report, "feature_importances": clf.feature_importances_.tolist(),
              "feature_names": feature_names}

    if _SHAP_OK:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_test, check_additivity=False)
            if isinstance(shap_vals, list):
                shap_abs = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
            else:
                shap_abs = np.mean(np.abs(shap_vals), axis=0)
            order = np.argsort(-shap_abs)
            top = min(cfg.shap_summary_max_display, len(feature_names))
            shap_summary = [{"feature": feature_names[i], "mean_abs_shap": float(shap_abs[i])} for i in order[:top]]
            result["shap_summary"] = shap_summary
        except Exception as e:
            result["shap_error"] = str(e)
    return result
