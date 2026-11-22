"""
Base interfaces and schema definitions for ML algorithms in ML Studio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator


@dataclass
class HyperparamField:
    """Definition of an adjustable hyperparameter for the user interface."""

    name: str
    label: str
    field_type: str  # 'int', 'float', 'str', 'select', 'bool'
    default: Any
    options: List[tuple[str, str]] = field(default_factory=list)  # (value, label) for selects
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    help_text: str = ""

    def clean_value(self, raw_value: Any) -> Any:
        """Coerce raw input (often str from request.POST) to the expected type."""
        if raw_value is None or raw_value == "":
            return self.default

        if self.field_type == "int":
            try:
                val = int(raw_value)
                if self.min_val is not None:
                    val = max(self.min_val, val)
                if self.max_val is not None:
                    val = min(self.max_val, val)
                return val
            except (ValueError, TypeError):
                return self.default

        elif self.field_type == "float":
            try:
                val = float(raw_value)
                if self.min_val is not None:
                    val = max(self.min_val, val)
                if self.max_val is not None:
                    val = min(self.max_val, val)
                return val
            except (ValueError, TypeError):
                return self.default

        elif self.field_type == "bool":
            if isinstance(raw_value, bool):
                return raw_value
            return str(raw_value).lower() in ("true", "1", "yes", "on")

        elif self.field_type == "select":
            valid_keys = [opt[0] for opt in self.options]
            return str(raw_value) if str(raw_value) in valid_keys else self.default

        return str(raw_value)


@dataclass
class AlgorithmMeta:
    """Metadata describing an algorithm for display and UI selection."""

    key: str
    name: str
    category: str  # 'regression', 'classification', 'clustering'
    description: str
    formula_latex: str
    hyperparams: List[HyperparamField] = field(default_factory=list)


class BaseAlgorithm(ABC):
    """Abstract base class for all 11 ML Studio algorithm wrappers."""

    key: str = ""
    name: str = ""
    category: str = ""  # 'regression', 'classification', 'clustering', or 'both'
    description: str = ""
    formula_latex: str = ""

    @abstractmethod
    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        """Return the list of tunable hyperparameters for this algorithm."""
        pass

    @abstractmethod
    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        """Instantiate and return the Scikit-Learn estimator or Pipeline."""
        pass

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """Extract algorithm-specific diagnostics (weights, importances, loss curve, centroids)."""
        return {}

    def get_meta(self, task_type: str) -> AlgorithmMeta:
        """Generate serializable metadata for the frontend."""
        return AlgorithmMeta(
            key=self.key,
            name=self.name,
            category=self.category,
            description=self.description,
            formula_latex=self.formula_latex,
            hyperparams=self.get_hyperparams(task_type),
        )
