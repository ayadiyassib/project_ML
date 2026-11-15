"""
Service module for Module 8 (Documentation & Pédagogie).
Handles Markdown rendering, math formula preservation, and content catalogs.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import markdown
from django.conf import settings

CONTENT_DIR = Path(settings.BASE_DIR) / "apps" / "docs_tutorials" / "content"
ALGORITHMS_DIR = CONTENT_DIR / "algorithms"
TUTORIALS_DIR = CONTENT_DIR / "tutorials"

# Catalog of the 11 Machine Learning Algorithms
ALGORITHMS_CATALOG: List[Dict[str, Any]] = [
    {
        "slug": "linear-regression",
        "name": "Régression Linéaire (Simple & Multiple)",
        "family": "regression",
        "category_label": "Régression",
        "desc": "Modélisation des relations linéaires continues via les moindres carrés ordinaires (OLS).",
        "equation": "y = \\beta_0 + \\sum_{j=1}^{p} \\beta_j X_j + \\varepsilon",
        "file": "01_linear_regression.md",
        "difficulty": "Fondamental",
        "color": "indigo",
    },
    {
        "slug": "polynomial-regression",
        "name": "Régression Polynomiale",
        "family": "regression",
        "category_label": "Régression",
        "desc": "Extension non-linéaire par transformation des variables explicatives jusqu'au degré d.",
        "equation": "y = \\beta_0 + \\beta_1 X + \\beta_2 X^2 + \\dots + \\beta_d X^d",
        "file": "02_polynomial_regression.md",
        "difficulty": "Intermédiaire",
        "color": "indigo",
    },
    {
        "slug": "logistic-regression",
        "name": "Régression Logistique (Binaire & Multiclasse)",
        "family": "classification",
        "category_label": "Classification",
        "desc": "Estimation de probabilités a posteriori et frontières de décision linéaires via fonction sigmoïde.",
        "equation": "P(Y=1|X) = \\frac{1}{1 + e^{-(\\beta^T X)}}",
        "file": "03_logistic_regression.md",
        "difficulty": "Fondamental",
        "color": "purple",
    },
    {
        "slug": "decision-trees",
        "name": "Arbres de Décision (Classifier & Regressor)",
        "family": "trees",
        "category_label": "Arbres",
        "desc": "Partitionnement récursif de l'espace selon les critères d'impureté de Gini ou d'entropie de Shannon.",
        "equation": "I_G(p) = 1 - \\sum_{k=1}^{K} p_k^2",
        "file": "04_decision_trees.md",
        "difficulty": "Fondamental",
        "color": "emerald",
    },
    {
        "slug": "naive-bayes",
        "name": "Classifieur Naïf Bayésien (Gaussian & Multinomial)",
        "family": "classification",
        "category_label": "Bayésien",
        "desc": "Théorème de Bayes combiné à l'hypothèse simplificatrice d'indépendance conditionnelle.",
        "equation": "P(Y|X) \\propto P(Y) \\prod_{j=1}^{p} P(X_j|Y)",
        "file": "05_naive_bayes.md",
        "difficulty": "Fondamental",
        "color": "purple",
    },
    {
        "slug": "svm",
        "name": "Machines à Vecteurs de Support (SVC & SVR)",
        "family": "kernel",
        "category_label": "Marges & Noyaux",
        "desc": "Séparation par hyperplan à marge maximale et projection via noyaux (Linéaire, RBF, Poly).",
        "equation": "\\min_{w, b} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^{n} \\xi_i",
        "file": "06_svm.md",
        "difficulty": "Avancé",
        "color": "cyan",
    },
    {
        "slug": "random-forest",
        "name": "Forêts Aléatoires (Random Forest Classifier & Regressor)",
        "family": "ensemble",
        "category_label": "Ensemble",
        "desc": "Ensemble bagging d'arbres décisionnels décorrelés avec estimation d'erreur Out-Of-Bag (OOB).",
        "equation": "\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} T_b(x)",
        "file": "07_random_forest.md",
        "difficulty": "Intermédiaire",
        "color": "emerald",
    },
    {
        "slug": "knn",
        "name": "K-Plus Proches Voisins (KNN Classifier & Regressor)",
        "family": "neighborhood",
        "category_label": "Voisinage",
        "desc": "Algorithme non paramétrique basé sur les métriques de distance (Euclidienne, Manhattan).",
        "equation": "d(u, v) = \\sqrt{\\sum_{i=1}^{p} (u_i - v_i)^2}",
        "file": "08_knn.md",
        "difficulty": "Fondamental",
        "color": "blue",
    },
    {
        "slug": "kmeans",
        "name": "K-Means Clustering",
        "family": "clustering",
        "category_label": "Clustering Non-Supervisé",
        "desc": "Optimisation itérative par centroïdes, méthode du coude (Elbow) et coefficient de Silhouette.",
        "equation": "J = \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|^2",
        "file": "09_kmeans.md",
        "difficulty": "Intermédiaire",
        "color": "amber",
    },
    {
        "slug": "neural-network-mlp",
        "name": "Perceptron Multicouche (ANN / MLP)",
        "family": "deep-learning",
        "category_label": "Réseaux de Neurones",
        "desc": "Réseau dense avec couches cachées, fonctions d'activation non linéaires et rétropropagation Adam.",
        "equation": "a^{[l]} = g\\left(W^{[l]} a^{[l-1]} + b^{[l]}\\right)",
        "file": "10_neural_networks_mlp.md",
        "difficulty": "Avancé",
        "color": "rose",
    },
]

# Catalog of the 3 Guided Tutorials
TUTORIALS_CATALOG: List[Dict[str, Any]] = [
    {
        "slug": "tutorial-california-housing",
        "title": "Tutoriel 1 : Prédire le prix des logements (Régression)",
        "dataset_name": "california_housing.csv",
        "category": "Régression Continue",
        "models_used": ["Régression Linéaire", "Régression Polynomiale", "Random Forest Regressor"],
        "duration": "15 min",
        "desc": "Parcours complet : de l'ingestion du dataset California Housing jusqu'à l'évaluation du RMSE et du R².",
        "file": "tutorial_1_regression_housing.md",
    },
    {
        "slug": "tutorial-titanic-iris",
        "title": "Tutoriel 2 : Prédire la survie & classification d'espèces (Classification)",
        "dataset_name": "titanic.csv & iris.csv",
        "category": "Classification",
        "models_used": ["Régression Logistique", "SVM", "Random Forest Classifier"],
        "duration": "20 min",
        "desc": "Imputation des âges manquants sur Titanic, encodage des ports d'embarquement, et comparaison F1-score / ROC AUC.",
        "file": "tutorial_2_classification_titanic.md",
    },
    {
        "slug": "tutorial-customer-segmentation",
        "title": "Tutoriel 3 : Segmentation client non-supervisée (Clustering)",
        "dataset_name": "wine.csv",
        "category": "Clustering Non-Supervisé",
        "models_used": ["K-Means", "PCA 2D"],
        "duration": "15 min",
        "desc": "Mise à l'échelle obligatoire des features, identification du nombre de clusters optimal via la courbe du coude.",
        "file": "tutorial_3_clustering_kmeans.md",
    },
]


def get_all_algorithms() -> List[Dict[str, Any]]:
    """Returns the full catalog of 11 ML algorithms."""
    return ALGORITHMS_CATALOG


def get_algorithm_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Returns algorithm metadata and rendered HTML content."""
    for algo in ALGORITHMS_CATALOG:
        if algo["slug"] == slug:
            algo_data = algo.copy()
            file_path = ALGORITHMS_DIR / algo["file"]
            if file_path.exists():
                raw_text = file_path.read_text(encoding="utf-8")
                algo_data["html_content"] = render_markdown_content(raw_text)
            else:
                algo_data["html_content"] = "<p class='text-slate-400'>Fiche en cours de rédaction.</p>"
            return algo_data
    return None


def get_all_tutorials() -> List[Dict[str, Any]]:
    """Returns the full catalog of guided tutorials."""
    return TUTORIALS_CATALOG


def get_tutorial_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Returns tutorial metadata and rendered HTML content."""
    for tut in TUTORIALS_CATALOG:
        if tut["slug"] == slug:
            tut_data = tut.copy()
            file_path = TUTORIALS_DIR / tut["file"]
            if file_path.exists():
                raw_text = file_path.read_text(encoding="utf-8")
                tut_data["html_content"] = render_markdown_content(raw_text)
            else:
                tut_data["html_content"] = "<p class='text-slate-400'>Tutoriel en cours de rédaction.</p>"
            return tut_data
    return None


def render_markdown_content(text: str) -> str:
    """
    Renders Markdown to HTML preserving KaTeX LaTeX math blocks ($...$ and $$...$$)
    and adding code highlighting and tables.
    """
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
        ]
    )
    return md.convert(text)

