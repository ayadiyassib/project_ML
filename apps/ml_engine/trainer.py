"""
Model Trainer and Cross-Validation Engine for ML Studio.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from apps.ml_engine.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
)


@dataclass
class TrainingResult:
    """Holds the complete output of a model training and evaluation run."""

    algo_key: str
    algo_name: str
    task_type: str
    params: Dict[str, Any]
    training_time_ms: float
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    cv_mean: Optional[float]
    cv_std: Optional[float]
    cv_scores: List[float] = field(default_factory=list)
    y_pred: List[Any] = field(default_factory=list)
    y_true: List[Any] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    summary_score: float = 0.0  # Main score used to rank on leaderboard (Accuracy, R2, or Silhouette)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo_key": self.algo_key,
            "algo_name": self.algo_name,
            "task_type": self.task_type,
            "params": self.params,
            "training_time_ms": round(self.training_time_ms, 2),
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "cv_scores": self.cv_scores,
            "y_pred": self.y_pred,
            "y_true": self.y_true,
            "diagnostics": self.diagnostics,
            "summary_score": round(self.summary_score, 4),
        }


class ModelTrainer:
    """Orchestrates model fitting, cross-validation, and performance evaluation."""

    def __init__(self, model: BaseEstimator, task_type: str, cv_folds: int = 5):
        self.model = model
        self.task_type = task_type
        self.cv_folds = max(2, min(10, cv_folds))

    def train_and_evaluate(
        self,
        X_train: Any,
        y_train: Optional[Any],
        X_test: Optional[Any],
        y_test: Optional[Any],
        feature_names: List[str],
        class_labels: Optional[List[str]] = None,
        diagnostics_extractor=None,
    ) -> TrainingResult:
        """Execute full training, cross-validation, and metrics computation."""
        X_tr = np.asarray(X_train)
        y_tr = np.asarray(y_train) if y_train is not None else None
        X_te = np.asarray(X_test) if X_test is not None else None
        y_te = np.asarray(y_test) if y_test is not None else None

        start_time = time.perf_counter()

        # -------------------------------------------------------------------
        # 1. Unsupervised Clustering (K-Means)
        # -------------------------------------------------------------------
        if self.task_type == "clustering":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_tr)

            duration_ms = (time.perf_counter() - start_time) * 1000
            labels = self.model.labels_
            inertia = getattr(self.model, "inertia_", None)
            metrics = compute_clustering_metrics(X_tr, labels, inertia=inertia)

            diag = {}
            if diagnostics_extractor:
                diag = diagnostics_extractor(self.model, feature_names, self.task_type)

            summary_score = metrics.get("silhouette", 0.0)

            return TrainingResult(
                algo_key="kmeans",
                algo_name="K-Means",
                task_type="clustering",
                params=getattr(self.model, "get_params", lambda: {})(),
                training_time_ms=duration_ms,
                train_metrics=metrics,
                test_metrics=metrics,
                cv_mean=None,
                cv_std=None,
                cv_scores=[],
                y_pred=labels.tolist() if hasattr(labels, "tolist") else list(labels),
                y_true=[],
                diagnostics=diag,
                summary_score=summary_score,
            )

        # -------------------------------------------------------------------
        # 2. Supervised Learning: Fit model
        # -------------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_tr, y_tr)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Predictions on Train & Test
        y_train_pred = self.model.predict(X_tr)
        y_test_pred = self.model.predict(X_te) if X_te is not None else y_train_pred

        # Probabilities if available
        train_proba = None
        test_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                train_proba = self.model.predict_proba(X_tr)
                if X_te is not None:
                    test_proba = self.model.predict_proba(X_te)
            except Exception:
                pass

        # -------------------------------------------------------------------
        # 3. Cross-Validation (K-Fold)
        # -------------------------------------------------------------------
        cv_scores: List[float] = []
        cv_mean: Optional[float] = None
        cv_std: Optional[float] = None

        try:
            n_samples = len(X_tr)
            effective_folds = min(self.cv_folds, n_samples)
            if self.task_type == "classification":
                # Ensure each class has enough members for stratified folds
                _, class_counts = np.unique(y_tr, return_counts=True)
                min_class = class_counts.min() if len(class_counts) > 0 else 1
                if min_class >= 2:
                    effective_folds = max(2, min(effective_folds, min_class))
                    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=max(2, min(effective_folds, 5)), shuffle=True, random_state=42)
                scoring = "accuracy"
            else:
                effective_folds = max(2, min(effective_folds, 5))
                cv = KFold(n_splits=effective_folds, shuffle=True, random_state=42)
                scoring = "r2"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(self.model, X_tr, y_tr, cv=cv, scoring=scoring)
                cv_scores = [round(float(s), 4) for s in scores]
                cv_mean = round(float(np.mean(scores)), 4)
                cv_std = round(float(np.std(scores)), 4)
        except Exception:
            cv_scores = []
            cv_mean = None
            cv_std = None

        # -------------------------------------------------------------------
        # 4. Metrics Computation
        # -------------------------------------------------------------------
        if self.task_type == "classification":
            train_metrics = compute_classification_metrics(y_tr, y_train_pred, train_proba, class_labels)
            test_target = y_te if y_te is not None else y_tr
            test_metrics = compute_classification_metrics(test_target, y_test_pred, test_proba, class_labels)
            summary_score = test_metrics.get("accuracy", 0.0)
        else:
            n_feat = X_tr.shape[1] if X_tr.ndim > 1 else 1
            train_metrics = compute_regression_metrics(y_tr, y_train_pred, n_features=n_feat)
            test_target = y_te if y_te is not None else y_tr
            test_metrics = compute_regression_metrics(test_target, y_test_pred, n_features=n_feat)
            summary_score = test_metrics.get("r2", 0.0)

        # -------------------------------------------------------------------
        # 5. Diagnostics extraction
        # -------------------------------------------------------------------
        diag = {}
        if diagnostics_extractor:
            diag = diagnostics_extractor(self.model, feature_names, self.task_type)

        algo_name = getattr(self.model, "__class__", type(self.model)).__name__

        return TrainingResult(
            algo_key="",  # Will be enriched by caller
            algo_name=algo_name,
            task_type=self.task_type,
            params=getattr(self.model, "get_params", lambda: {})(),
            training_time_ms=duration_ms,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_scores=cv_scores,
            y_pred=y_test_pred.tolist() if hasattr(y_test_pred, "tolist") else list(y_test_pred),
            y_true=test_target.tolist() if hasattr(test_target, "tolist") else list(test_target),
            diagnostics=diag,
            summary_score=summary_score,
        )
