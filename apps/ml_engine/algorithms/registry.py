"""
Algorithm Registry: Central lookup and factory for all 11 ML models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from sklearn.base import BaseEstimator

from apps.ml_engine.algorithms.base import BaseAlgorithm, AlgorithmMeta
from apps.ml_engine.algorithms.regression import (
    SimpleLinearRegressionAlgorithm,
    MultipleLinearRegressionAlgorithm,
    PolynomialRegressionAlgorithm,
)
from apps.ml_engine.algorithms.classification import (
    LogisticRegressionAlgorithm,
    DecisionTreeAlgorithm,
    RandomForestAlgorithm,
    SupportVectorMachineAlgorithm,
    NaiveBayesAlgorithm,
    KNearestNeighborsAlgorithm,
)
from apps.ml_engine.algorithms.clustering import KMeansAlgorithm
from apps.ml_engine.algorithms.neural_network import NeuralNetworkAlgorithm


class ModelRegistry:
    """Registry coordinating all 11 Machine Learning algorithms."""

    _ALGORITHMS: Dict[str, BaseAlgorithm] = {
        # Regression family
        "simple_linear_regression": SimpleLinearRegressionAlgorithm(),
        "multiple_linear_regression": MultipleLinearRegressionAlgorithm(),
        "polynomial_regression": PolynomialRegressionAlgorithm(),
        # Classification & Dual family
        "logistic_regression": LogisticRegressionAlgorithm(),
        "decision_tree": DecisionTreeAlgorithm(),
        "random_forest": RandomForestAlgorithm(),
        "svm": SupportVectorMachineAlgorithm(),
        "naive_bayes": NaiveBayesAlgorithm(),
        "knn": KNearestNeighborsAlgorithm(),
        # Clustering family
        "kmeans": KMeansAlgorithm(),
        # Neural Network family
        "neural_network": NeuralNetworkAlgorithm(),
    }

    @classmethod
    def get(cls, key: str) -> BaseAlgorithm:
        """Fetch algorithm handler by its unique key."""
        if key not in cls._ALGORITHMS:
            raise ValueError(f"Algorithme inconnu : '{key}'. Choix valides : {list(cls._ALGORITHMS.keys())}")
        return cls._ALGORITHMS[key]

    @classmethod
    def all_algorithms(cls) -> Dict[str, BaseAlgorithm]:
        """Return full map of all registered algorithms."""
        return dict(cls._ALGORITHMS)

    @classmethod
    def for_task(cls, task_type: str) -> List[AlgorithmMeta]:
        """
        Return list of algorithms compatible with the given task type.
        task_type: 'classification', 'regression', or 'clustering'.
        """
        compatible: List[AlgorithmMeta] = []
        for algo in cls._ALGORITHMS.values():
            if algo.category == "both" and task_type in ("classification", "regression"):
                compatible.append(algo.get_meta(task_type))
            elif algo.category == task_type:
                compatible.append(algo.get_meta(task_type))
        return compatible

    @classmethod
    def build_model(cls, key: str, task_type: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        """Instantiate the estimator with sanitized hyperparameters."""
        algo = cls.get(key)
        # Verify compatibility
        is_compat = (algo.category == "both" and task_type in ("classification", "regression")) or (algo.category == task_type)
        if not is_compat:
            raise ValueError(
                f"L'algorithme '{algo.name}' ({algo.category}) n'est pas compatible avec la tâche '{task_type}'."
            )
        return algo.build_model(task_type, params or {})

    @classmethod
    def get_default_key(cls, task_type: str) -> str:
        """Suggest a default algorithm for a given task."""
        if task_type == "regression":
            return "multiple_linear_regression"
        elif task_type == "clustering":
            return "kmeans"
        return "random_forest"
