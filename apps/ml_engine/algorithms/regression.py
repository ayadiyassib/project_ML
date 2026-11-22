"""
Regression algorithms implementation:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Linear Regression
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from apps.ml_engine.algorithms.base import BaseAlgorithm, HyperparamField


class SingleFeatureLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for Simple Linear Regression that restricts learning to 1 feature.
    """

    def __init__(self, feature_idx: int = 0, fit_intercept: bool = True):
        self.feature_idx = feature_idx
        self.fit_intercept = fit_intercept
        self.regressor_ = LinearRegression(fit_intercept=fit_intercept)
        self.selected_feature_name_: Optional[str] = None

    def _select_feature(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            return X_arr.reshape(-1, 1)
        idx = min(max(0, self.feature_idx), X_arr.shape[1] - 1)
        return X_arr[:, [idx]]

    def fit(self, X, y):
        X_single = self._select_feature(X)
        self.regressor_.fit(X_single, y)
        self.coef_ = self.regressor_.coef_
        self.intercept_ = self.regressor_.intercept_
        return self

    def predict(self, X):
        X_single = self._select_feature(X)
        return self.regressor_.predict(X_single)


class SimpleLinearRegressionAlgorithm(BaseAlgorithm):
    """Algorithme 1 : Régression Linéaire Simple (1 variable explicative)."""

    key = "simple_linear_regression"
    name = "Régression Linéaire Simple"
    category = "regression"
    description = "Modélise la relation linéaire entre une seule variable prédictive continue et la cible : y = β₀ + β₁·x."
    formula_latex = r"\hat{y} = \beta_0 + \beta_1 \cdot x"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="feature_idx",
                label="Indice de la variable (x)",
                field_type="int",
                default=0,
                min_val=0,
                max_val=100,
                help_text="Indice (0-indexé) de la variable explicative à utiliser.",
            ),
            HyperparamField(
                name="fit_intercept",
                label="Calculer l'ordonnée à l'origine (β₀)",
                field_type="bool",
                default=True,
                help_text="Si activé, calcule l'ordonnée à l'origine β₀.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        feature_idx = field_dict["feature_idx"].clean_value(params.get("feature_idx"))
        fit_intercept = field_dict["fit_intercept"].clean_value(params.get("fit_intercept"))
        return SingleFeatureLinearRegressor(feature_idx=feature_idx, fit_intercept=fit_intercept)

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        if isinstance(model, SingleFeatureLinearRegressor):
            idx = min(max(0, model.feature_idx), len(feature_names) - 1) if feature_names else 0
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
            slope = float(model.coef_[0]) if hasattr(model, "coef_") and len(model.coef_) > 0 else 0.0
            return {
                "type": "simple_linear",
                "feature_used": fname,
                "intercept": round(intercept, 4),
                "slope": round(slope, 4),
                "equation": f"y = {round(intercept, 4)} + {round(slope, 4)} · {fname}",
                "coefficients": [{"feature": fname, "coefficient": round(slope, 4)}],
            }
        return {}


class MultipleLinearRegressionAlgorithm(BaseAlgorithm):
    """Algorithme 2 : Régression Linéaire Multiple (p variables explicatives)."""

    key = "multiple_linear_regression"
    name = "Régression Linéaire Multiple"
    category = "regression"
    description = "Modélise la cible comme combinaison linéaire de toutes les variables explicatives : y = β₀ + Σ βⱼ·xⱼ."
    formula_latex = r"\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="fit_intercept",
                label="Calculer l'ordonnée à l'origine (β₀)",
                field_type="bool",
                default=True,
                help_text="Ajuste un terme constant β₀ pour le modèle.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        fit_intercept = field_dict["fit_intercept"].clean_value(params.get("fit_intercept"))
        return LinearRegression(fit_intercept=fit_intercept)

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        if hasattr(model, "coef_"):
            intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
            coefs = []
            for i, c in enumerate(model.coef_):
                name = feature_names[i] if i < len(feature_names) else f"x_{i+1}"
                coefs.append({"feature": name, "coefficient": round(float(c), 4)})
            
            # Sort by absolute coefficient size
            coefs_sorted = sorted(coefs, key=lambda x: abs(x["coefficient"]), reverse=True)
            return {
                "type": "multiple_linear",
                "intercept": round(intercept, 4),
                "coefficients": coefs_sorted,
                "n_features": len(coefs),
            }
        return {}


class PolynomialRegressionAlgorithm(BaseAlgorithm):
    """Algorithme 3 : Régression Linéaire Polynomiale (degré 2 à 5)."""

    key = "polynomial_regression"
    name = "Régression Polynomiale"
    category = "regression"
    description = "Étend les variables à l'ordre polynomial d ∈ [2, 5] pour capturer les non-linéarités complexes."
    formula_latex = r"\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j + \sum \beta_{jk} x_j x_k + \dots"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="degree",
                label="Degré du polynôme (d)",
                field_type="int",
                default=2,
                min_val=2,
                max_val=5,
                step=1,
                help_text="Degré de développement polynomial (2 à 5).",
            ),
            HyperparamField(
                name="fit_intercept",
                label="Calculer l'ordonnée à l'origine",
                field_type="bool",
                default=True,
                help_text="Ajuste le biais constant.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        degree = field_dict["degree"].clean_value(params.get("degree"))
        fit_intercept = field_dict["fit_intercept"].clean_value(params.get("fit_intercept"))

        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression(fit_intercept=fit_intercept)),
        ])

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        if isinstance(model, Pipeline) and "linear" in model.named_steps:
            linear_step = model.named_steps["linear"]
            poly_step = model.named_steps["poly"]
            intercept = float(linear_step.intercept_) if hasattr(linear_step, "intercept_") else 0.0
            
            try:
                poly_feature_names = poly_step.get_feature_names_out(feature_names)
            except Exception:
                poly_feature_names = [f"poly_{i}" for i in range(len(linear_step.coef_))]
                
            coefs = []
            for name, c in zip(poly_feature_names, linear_step.coef_):
                coefs.append({"feature": str(name), "coefficient": round(float(c), 4)})
                
            coefs_sorted = sorted(coefs, key=lambda x: abs(x["coefficient"]), reverse=True)[:20]
            return {
                "type": "polynomial_linear",
                "degree": poly_step.degree,
                "n_poly_features": len(linear_step.coef_),
                "intercept": round(intercept, 4),
                "top_coefficients": coefs_sorted,
            }
        return {}
