"""
Core Views for ML Studio Dashboard and Workspace Management.
"""

from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_POST
from apps.core.context_processors import DEFAULT_PROJECT_STATE


def dashboard_view(request):
    """
    Renders the main Studio overview dashboard.
    """
    state = request.session.get("project_state", DEFAULT_PROJECT_STATE)
    
    # Overview metrics
    context = {
        "title": "Vue d'ensemble du Studio",
        "state": state,
        "pipeline_steps": [
            {
                "num": 1,
                "name": "Ingestion de Données",
                "desc": "Import CSV, XLSX, JSON ou datasets de référence (California Housing, Iris, Titanic, Wine).",
                "url": "datasets:index",
                "done": bool(state.get("dataset_name")),
                "icon": "database",
            },
            {
                "num": 2,
                "name": "Exploration EDA",
                "desc": "Profiling statistique complet, détection des types, valeurs manquantes et graphiques Plotly interactifs.",
                "url": "eda:index",
                "done": bool(state.get("dataset_name")),
                "icon": "chart",
            },
            {
                "num": 3,
                "name": "Prétraitement & Features",
                "desc": "Imputation, One-Hot/Label Encoding, Standard/MinMax Scalers sans fuite d'information (data leakage).",
                "url": "preprocessing:index",
                "done": state.get("preprocessing_done", False),
                "icon": "adjustments",
            },
            {
                "num": 4,
                "name": "Modélisation (11 Algorithmes)",
                "desc": "Configuration des hyperparamètres, Cross-Validation K-Fold et calcul des métriques (Accuracy, RMSE, Silhouette, etc.).",
                "url": "ml_engine:index",
                "done": state.get("model_trained", False),
                "icon": "cpu",
            },
            {
                "num": 5,
                "name": "Diagnostics & Visualisations",
                "desc": "Matrices de confusion, courbes ROC, résidus, projections PCA 2D et feature importances.",
                "url": "model_viz:index",
                "done": state.get("model_trained", False),
                "icon": "presentation",
            },
            {
                "num": 6,
                "name": "Export & Inférence en direct",
                "desc": "Sauvegarde du pipeline complet en .joblib, bac à sable de prédiction en temps réel et rapport PDF exécutif.",
                "url": "export_inference:index",
                "done": state.get("model_trained", False),
                "icon": "download",
            },
        ],
    }
    return render(request, "core/dashboard.html", context)


@require_POST
def reset_session_view(request):
    """
    Resets current user project workspace.
    """
    request.session["project_state"] = DEFAULT_PROJECT_STATE.copy()
    request.session.modified = True
    messages.success(request, "L'espace de travail a été réinitialisé avec succès.")
    return redirect("core:dashboard")

