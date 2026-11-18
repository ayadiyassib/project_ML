"""
Supervised learning algorithms (Classification & dual Classification/Regression):
4. Logistic Regression
5. Decision Trees (Classifier & Regressor)
6. Random Forest (Classifier & Regressor)
7. Support Vector Machine (SVC & SVR)
8. Naive Bayes (GaussianNB & MultinomialNB)
9. K-Nearest Neighbors (Classifier & Regressor)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from apps.ml_engine.algorithms.base import BaseAlgorithm, HyperparamField


class LogisticRegressionAlgorithm(BaseAlgorithm):
    """Algorithme 4 : Régression Logistique (binaire et multi-classe)."""

    key = "logistic_regression"
    name = "Régression Logistique"
    category = "classification"
    description = "Modélise la probabilité d'appartenance aux classes via la fonction sigmoïde/softmax avec régularisation L1 ou L2."
    formula_latex = r"P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{X}\beta)}}"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="C",
                label="Inverse de régularisation (C)",
                field_type="float",
                default=1.0,
                min_val=0.001,
                max_val=100.0,
                step=0.1,
                help_text="Valeurs faibles = forte régularisation (évite le surapprentissage).",
            ),
            HyperparamField(
                name="penalty",
                label="Type de pénalité",
                field_type="select",
                default="l2",
                options=[("l2", "L2 (Ridge - Réduit les poids)"), ("l1", "L1 (Lasso - Sélection parcimonieuse)")],
                help_text="L1 force certains coefficients à zéro, L2 les réduit globalement.",
            ),
            HyperparamField(
                name="max_iter",
                label="Nombre maximal d'itérations",
                field_type="int",
                default=500,
                min_val=50,
                max_val=5000,
                step=50,
                help_text="Nombre de passes d'optimisation.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        c_val = field_dict["C"].clean_value(params.get("C"))
        penalty = field_dict["penalty"].clean_value(params.get("penalty"))
        max_iter = field_dict["max_iter"].clean_value(params.get("max_iter"))
        solver = "saga" if penalty == "l1" else "lbfgs"

        return LogisticRegression(
            C=float(c_val),
            penalty=penalty,
            solver=solver,
            max_iter=int(max_iter),
            random_state=42,
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        if hasattr(model, "coef_"):
            coef = model.coef_
            classes = [str(c) for c in getattr(model, "classes_", [])]
            coef_list = []
            if coef.ndim == 1 or coef.shape[0] == 1:
                flat_coef = coef.ravel()
                for i, c in enumerate(flat_coef):
                    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    coef_list.append({"feature": name, "coefficient": round(float(c), 4)})
            else:
                # Multiclass: average absolute influence
                mean_abs = np.mean(np.abs(coef), axis=0)
                for i, c in enumerate(mean_abs):
                    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    coef_list.append({"feature": name, "coefficient": round(float(c), 4)})

            coef_sorted = sorted(coef_list, key=lambda x: abs(x["coefficient"]), reverse=True)
            return {
                "type": "logistic_regression",
                "classes": classes,
                "n_classes": len(classes),
                "coefficients": coef_sorted,
            }
        return {}


class DecisionTreeAlgorithm(BaseAlgorithm):
    """Algorithme 5 : Arbres de Décision (Classifier & Regressor)."""

    key = "decision_tree"
    name = "Arbre de Décision"
    category = "both"
    description = "Partitionne l'espace des descripteurs en segments homogènes basés sur des seuils de décision (Gini, Entropie ou MSE)."
    formula_latex = r"Gini = 1 - \sum p_i^2 \quad\text{ou}\quad MSE = \frac{1}{N}\sum(y_i - \bar{y})^2"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        criterion_options = (
            [("gini", "Gini (Impureté)"), ("entropy", "Entropie (Gain d'information)"), ("log_loss", "Log-Loss")]
            if task_type == "classification"
            else [("squared_error", "Erreur quadratique (MSE)"), ("absolute_error", "Erreur absolue (MAE)")]
        )
        default_crit = "gini" if task_type == "classification" else "squared_error"

        return [
            HyperparamField(
                name="criterion",
                label="Critère de division (Split)",
                field_type="select",
                default=default_crit,
                options=criterion_options,
                help_text="Métrique guidant le découpage des nœuds.",
            ),
            HyperparamField(
                name="max_depth",
                label="Profondeur maximale",
                field_type="int",
                default=5,
                min_val=1,
                max_val=30,
                step=1,
                help_text="Limite la hauteur de l'arbre pour limiter le surapprentissage (None = sans limite).",
            ),
            HyperparamField(
                name="min_samples_split",
                label="Échantillons min. pour division",
                field_type="int",
                default=2,
                min_val=2,
                max_val=50,
                help_text="Nombre minimal d'individus requis pour scinder un nœud.",
            ),
            HyperparamField(
                name="min_samples_leaf",
                label="Échantillons min. par feuille",
                field_type="int",
                default=1,
                min_val=1,
                max_val=50,
                help_text="Nombre minimal d'individus dans une feuille finale.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        criterion = field_dict["criterion"].clean_value(params.get("criterion"))
        max_depth = field_dict["max_depth"].clean_value(params.get("max_depth"))
        min_split = field_dict["min_samples_split"].clean_value(params.get("min_samples_split"))
        min_leaf = field_dict["min_samples_leaf"].clean_value(params.get("min_samples_leaf"))

        # Convert max_depth=0 or very large to None if needed
        depth_val = max_depth if max_depth and max_depth > 0 else None

        if task_type == "classification":
            return DecisionTreeClassifier(
                criterion=criterion,
                max_depth=depth_val,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                random_state=42,
            )
        return DecisionTreeRegressor(
            criterion=criterion,
            max_depth=depth_val,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "decision_tree",
            "n_nodes": int(getattr(model.tree_, "node_count", 0)) if hasattr(model, "tree_") else 0,
            "max_depth": int(getattr(model, "max_depth", 0) or 0),
        }
        if hasattr(model, "feature_importances_"):
            importances = []
            for i, imp in enumerate(model.feature_importances_):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                importances.append({"feature": name, "importance": round(float(imp), 4)})
            result["feature_importances"] = sorted(importances, key=lambda x: x["importance"], reverse=True)
        return result


class RandomForestAlgorithm(BaseAlgorithm):
    """Algorithme 6 : Forêts Aléatoires (Classifier & Regressor)."""

    key = "random_forest"
    name = "Forêt Aléatoire (Random Forest)"
    category = "both"
    description = "Ensemble d'arbres de décision décorrélés entraînés par bagging et sélection aléatoire de variables pour une grande robustesse."
    formula_latex = r"\hat{y} = \frac{1}{B}\sum_{b=1}^B T_b(X) \quad\text{ou}\quad \text{Vote Majoritaire}"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="n_estimators",
                label="Nombre d'arbres (n_estimators)",
                field_type="int",
                default=100,
                min_val=10,
                max_val=500,
                step=10,
                help_text="Nombre d'arbres agrégés dans la forêt.",
            ),
            HyperparamField(
                name="max_depth",
                label="Profondeur maximale",
                field_type="int",
                default=10,
                min_val=1,
                max_val=30,
                step=1,
                help_text="Profondeur maximale de chaque arbre individuel.",
            ),
            HyperparamField(
                name="max_features",
                label="Variables par division (max_features)",
                field_type="select",
                default="sqrt",
                options=[("sqrt", "Racine carrée (sqrt)"), ("log2", "Logarithme base 2 (log2)"), ("all", "Toutes les variables")],
                help_text="Nombre de variables testées à chaque nœud.",
            ),
            HyperparamField(
                name="min_samples_leaf",
                label="Échantillons min. par feuille",
                field_type="int",
                default=1,
                min_val=1,
                max_val=20,
                help_text="Taille minimale d'une feuille terminale.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        n_estimators = field_dict["n_estimators"].clean_value(params.get("n_estimators"))
        max_depth = field_dict["max_depth"].clean_value(params.get("max_depth"))
        max_feat_raw = field_dict["max_features"].clean_value(params.get("max_features"))
        min_leaf = field_dict["min_samples_leaf"].clean_value(params.get("min_samples_leaf"))

        max_features = None if max_feat_raw == "all" else max_feat_raw
        depth_val = max_depth if max_depth and max_depth > 0 else None

        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=depth_val,
                max_features=max_features,
                min_samples_leaf=int(min_leaf),
                random_state=42,
            )
        return RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=depth_val,
            max_features=max_features,
            min_samples_leaf=int(min_leaf),
            random_state=42,
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "random_forest",
            "n_estimators": getattr(model, "n_estimators", 100),
        }
        if hasattr(model, "feature_importances_"):
            importances = []
            for i, imp in enumerate(model.feature_importances_):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                importances.append({"feature": name, "importance": round(float(imp), 4)})
            result["feature_importances"] = sorted(importances, key=lambda x: x["importance"], reverse=True)
        return result


class SupportVectorMachineAlgorithm(BaseAlgorithm):
    """Algorithme 7 : Support Vector Machine (SVC & SVR)."""

    key = "svm"
    name = "Support Vector Machine (SVM / SVR)"
    category = "both"
    description = "Recherche l'hyperplan optimal maximisant la marge géométrique entre classes ou prédisant une zone d'insensibilité ε."
    formula_latex = r"\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum \xi_i \quad\text{avec noyau } K(\mathbf{x}_i, \mathbf{x}_j)"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="kernel",
                label="Type de Noyau (Kernel)",
                field_type="select",
                default="rbf",
                options=[("rbf", "RBF (Gaussien - polyvalent)"), ("linear", "Linéaire"), ("poly", "Polynomial")],
                help_text="Projette les données dans un espace de plus grande dimension.",
            ),
            HyperparamField(
                name="C",
                label="Pénalité d'erreur (C)",
                field_type="float",
                default=1.0,
                min_val=0.01,
                max_val=100.0,
                step=0.1,
                help_text="Équilibre entre largeur de marge et tolérance aux erreurs.",
            ),
            HyperparamField(
                name="gamma",
                label="Coefficient Gamma",
                field_type="select",
                default="scale",
                options=[("scale", "scale (1 / (n_features * Var))"), ("auto", "auto (1 / n_features)")],
                help_text="Rayon d'influence d'un vecteur support pour noyaux RBF/Poly.",
            ),
            HyperparamField(
                name="degree",
                label="Degré polynomial (si kernel=poly)",
                field_type="int",
                default=3,
                min_val=2,
                max_val=5,
                step=1,
                help_text="Degré utilisé uniquement si le noyau Polynomial est sélectionné.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        kernel = field_dict["kernel"].clean_value(params.get("kernel"))
        c_val = field_dict["C"].clean_value(params.get("C"))
        gamma = field_dict["gamma"].clean_value(params.get("gamma"))
        degree = field_dict["degree"].clean_value(params.get("degree"))

        if task_type == "classification":
            return SVC(
                kernel=kernel,
                C=float(c_val),
                gamma=gamma,
                degree=int(degree),
                probability=True,  # Crucial for ROC-AUC & probas
                random_state=42,
            )
        return SVR(
            kernel=kernel,
            C=float(c_val),
            gamma=gamma,
            degree=int(degree),
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "svm",
            "kernel": getattr(model, "kernel", "rbf"),
            "C": getattr(model, "C", 1.0),
        }
        if hasattr(model, "n_support_"):
            result["n_support_vectors"] = [int(n) for n in model.n_support_]
            result["total_support_vectors"] = int(sum(model.n_support_))
        return result


class NaiveBayesAlgorithm(BaseAlgorithm):
    """Algorithme 8 : Naïf Bayésien (GaussianNB & MultinomialNB)."""

    key = "naive_bayes"
    name = "Naïf Bayésien (Naive Bayes)"
    category = "classification"
    description = "Classifieur probabiliste fondé sur le théorème de Bayes avec hypothèse forte d'indépendance conditionnelle des variables."
    formula_latex = r"P(y|X) = \frac{P(y) \prod_{j=1}^p P(x_j|y)}{P(X)}"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="variant",
                label="Variante Naive Bayes",
                field_type="select",
                default="gaussian",
                options=[("gaussian", "GaussianNB (Variables continues)"), ("multinomial", "MultinomialNB (Fréquences / comptage)")],
                help_text="GaussianNB suppose des distributions normales conditionnelles.",
            ),
            HyperparamField(
                name="var_smoothing",
                label="Lissage de variance (var_smoothing)",
                field_type="float",
                default=1e-9,
                min_val=1e-12,
                max_val=1e-2,
                help_text="Portion de la plus grande variance ajoutée pour la stabilité numérique.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        variant = field_dict["variant"].clean_value(params.get("variant"))
        smoothing = field_dict["var_smoothing"].clean_value(params.get("var_smoothing"))

        if variant == "multinomial":
            return MultinomialNB()
        return GaussianNB(var_smoothing=float(smoothing))

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "naive_bayes",
            "classes": [str(c) for c in getattr(model, "classes_", [])],
        }
        if hasattr(model, "class_prior_"):
            result["class_priors"] = [round(float(p), 4) for p in model.class_prior_]
        return result


class KNearestNeighborsAlgorithm(BaseAlgorithm):
    """Algorithme 9 : Plus Proche Voisin (KNN - K-Nearest Neighbors)."""

    key = "knn"
    name = "K-Plus Proches Voisins (KNN)"
    category = "both"
    description = "Méthode non paramétrique basée sur la distance géométrique aux k plus proches voisins dans l'espace des descripteurs."
    formula_latex = r"\hat{y} = \text{Mode}(y_{i \in N_k}) \quad\text{ou}\quad \frac{1}{k}\sum_{i \in N_k} y_i"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="n_neighbors",
                label="Nombre de voisins (k)",
                field_type="int",
                default=5,
                min_val=1,
                max_val=25,
                step=1,
                help_text="Nombre de voisins examinés pour voter ou moyenner.",
            ),
            HyperparamField(
                name="weights",
                label="Pondération des voisins (weights)",
                field_type="select",
                default="uniform",
                options=[("uniform", "Uniforme (tous égaux)"), ("distance", "Distance inverse (voisins proches avantagés)")],
                help_text="Pondération appliquée aux votes des voisins.",
            ),
            HyperparamField(
                name="metric",
                label="Métrique de distance",
                field_type="select",
                default="euclidean",
                options=[("euclidean", "Euclidienne (L2)"), ("manhattan", "Manhattan (L1)"), ("minkowski", "Minkowski (p=3)")],
                help_text="Fonction de distance pour mesurer la proximité.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        n_neighbors = field_dict["n_neighbors"].clean_value(params.get("n_neighbors"))
        weights = field_dict["weights"].clean_value(params.get("weights"))
        metric = field_dict["metric"].clean_value(params.get("metric"))
        p = 3 if metric == "minkowski" else (1 if metric == "manhattan" else 2)

        if task_type == "classification":
            return KNeighborsClassifier(
                n_neighbors=int(n_neighbors),
                weights=weights,
                metric="minkowski",
                p=p,
            )
        return KNeighborsRegressor(
            n_neighbors=int(n_neighbors),
            weights=weights,
            metric="minkowski",
            p=p,
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        return {
            "type": "knn",
            "n_neighbors": getattr(model, "n_neighbors", 5),
            "weights": getattr(model, "weights", "uniform"),
        }
