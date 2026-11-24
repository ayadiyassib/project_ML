"""
Core Views for ML Studio Dashboard, Landing Page, and Workspace Management.
"""

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.db.models import Sum
from django.views.decorators.http import require_POST
from apps.accounts.decorators import studio_login_required
from apps.core.context_processors import DEFAULT_PROJECT_STATE
from apps.datasets.models import Dataset, human_size
from apps.ml_engine.models import TrainedModel


def root_view(request):
    """
    Public landing page for anonymous visitors,
    or redirect to dashboard for logged-in users.
    """
    if request.user.is_authenticated:
        return redirect("core:dashboard")
    return render(request, "core/landing.html")


@studio_login_required
def dashboard_view(request):
    """
    Renders the main Studio overview dashboard.
    Enriched with platform-wide KPIs when viewed by an administrator.
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

    # 🛡️ Injection des statistiques globales pour l'Administrateur
    if request.user.is_staff or request.user.is_superuser:
        total_storage = Dataset.objects.aggregate(total=Sum("size_bytes"))["total"] or 0
        context["admin_stats"] = {
            "total_users": User.objects.count(),
            "active_users": User.objects.filter(is_active=True).count(),
            "total_datasets": Dataset.objects.count(),
            "total_models": TrainedModel.objects.count(),
            "storage_used_bytes": total_storage,
            "storage_used_human": human_size(total_storage),
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

