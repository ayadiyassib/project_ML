"""
Comprehensive ML evaluation metrics suite for classification, regression, and clustering.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Optional[Any] = None,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate exhaustive metrics for classification tasks."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)
    is_binary = len(classes) == 2

    # Basic scores
    acc = float(accuracy_score(y_t, y_p))
    prec_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "precision": round(prec_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall": round(rec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "f1": round(f1_weighted, 4),
        "f1_score": round(f1_weighted, 4),  # alias
        "f1_macro": round(f1_macro, 4),
        "is_binary": is_binary,
        "n_classes": len(classes),
    }

    # ROC-AUC calculation
    if y_proba is not None:
        try:
            proba_arr = np.asarray(y_proba)
            if is_binary:
                # Binary: second column is positive class
                score_col = proba_arr[:, 1] if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2 else proba_arr.ravel()
                auc = float(roc_auc_score(y_t, score_col))
                metrics["roc_auc"] = round(auc, 4)
            else:
                # Multiclass: One-vs-Rest ROC-AUC
                auc = float(roc_auc_score(y_t, proba_arr, multi_class="ovr", average="weighted"))
                metrics["roc_auc"] = round(auc, 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # Confusion Matrix
    try:
        cm = confusion_matrix(y_t, y_p)
        labels = [str(c) for c in (class_labels if class_labels is not None else classes)]
        metrics["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": labels,
        }
    except Exception:
        metrics["confusion_matrix"] = None

    # Per-class classification report
    try:
        report_dict = classification_report(y_t, y_p, output_dict=True, zero_division=0)
        per_class: List[Dict[str, Any]] = []
        for key, val in report_dict.items():
            if key not in ("accuracy", "macro avg", "weighted avg") and isinstance(val, dict):
                per_class.append({
                    "class": str(key),
                    "precision": round(float(val.get("precision", 0.0)), 4),
                    "recall": round(float(val.get("recall", 0.0)), 4),
                    "f1": round(float(val.get("f1-score", 0.0)), 4),
                    "support": int(val.get("support", 0)),
                })
        metrics["per_class_report"] = per_class
    except Exception:
        metrics["per_class_report"] = []

    return metrics


def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    n_features: int = 1,
) -> Dict[str, Any]:
    """Calculate exhaustive metrics for regression tasks."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    n_samples = len(y_t)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    mx_err = float(max_error(y_t, y_p))

    # Adjusted R2
    if n_samples > n_features + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        adj_r2 = r2

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "r2_score": round(r2, 4),  # alias
        "r2_adjusted": round(adj_r2, 4),
        "max_error": round(mx_err, 4),
        "n_samples": n_samples,
        "n_features": n_features,
    }


def compute_clustering_metrics(
    X: Any,
    labels: Any,
    inertia: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate evaluation metrics for unsupervised clustering (K-Means)."""
    X_arr = np.asarray(X)
    lbls = np.asarray(labels)
    unique_clusters = np.unique(lbls)
    n_clusters = len(unique_clusters)

    metrics: Dict[str, Any] = {
        "n_clusters": int(n_clusters),
        "inertia": round(float(inertia), 4) if inertia is not None else 0.0,
    }

    if n_clusters > 1 and len(X_arr) > n_clusters:
        try:
            sil = float(silhouette_score(X_arr, lbls))
            metrics["silhouette"] = round(sil, 4)
            metrics["silhouette_score"] = round(sil, 4)
        except Exception:
            metrics["silhouette"] = 0.0

        try:
            db = float(davies_bouldin_score(X_arr, lbls))
            metrics["davies_bouldin"] = round(db, 4)
        except Exception:
            metrics["davies_bouldin"] = 0.0

        try:
            ch = float(calinski_harabasz_score(X_arr, lbls))
            metrics["calinski_harabasz"] = round(ch, 2)
        except Exception:
            metrics["calinski_harabasz"] = 0.0
    else:
        metrics["silhouette"] = 0.0
        metrics["davies_bouldin"] = 0.0
        metrics["calinski_harabasz"] = 0.0

    return metrics
