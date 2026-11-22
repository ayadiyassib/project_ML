"""
Views for the ML Engine module (Step 4 of the ML Studio pipeline).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from apps.datasets.state import get_active_dataset, update_state
from apps.preprocessing.services import load_prepared
from apps.ml_engine.algorithms.registry import ModelRegistry
from apps.ml_engine.services import (
    MLEngineError,
    get_available_algorithms,
    train_model_for_dataset,
    load_latest_result,
    load_leaderboard,
    clear_models,
)


def _build_context(request: HttpRequest, dataset, selected_key: Optional[str] = None) -> Dict[str, Any]:
    """Helper to assemble full rendering context for the ML training interface."""
    prepared = load_prepared(dataset) if dataset else None
    task_type = prepared.task_type if prepared else None

    available_algos = get_available_algorithms(task_type) if task_type else []
    
    # Determine which algorithm is currently active in the form
    valid_keys = [a.key for a in available_algos]
    if selected_key not in valid_keys:
        selected_key = ModelRegistry.get_default_key(task_type) if task_type else None

    selected_algo = None
    hyperparams = []
    if selected_key and selected_key in ModelRegistry.all_algorithms():
        selected_algo = ModelRegistry.get(selected_key)
        hyperparams = selected_algo.get_hyperparams(task_type or "classification")

    latest_result = load_latest_result(dataset) if dataset else None
    leaderboard = load_leaderboard(dataset) if dataset else []

    return {
        "title": "Entraînement des Modèles ML",
        "dataset": dataset,
        "prepared": prepared,
        "task_type": task_type,
        "available_algorithms": available_algos,
        "selected_algo_key": selected_key,
        "selected_algo": selected_algo,
        "hyperparams": hyperparams,
        "latest_result": latest_result,
        "leaderboard": leaderboard,
        "cv_folds_options": [3, 5, 10],
    }


def index_view(request: HttpRequest) -> HttpResponse:
    """Main view of the ML Engine studio."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return render(
            request,
            "ml_engine/index.html",
            {
                "title": "Entraînement des Modèles ML",
                "dataset": None,
                "prepared": None,
            },
        )

    prepared = load_prepared(dataset)
    if prepared is None:
        messages.warning(request, "Veuillez d'abord configurer et exécuter le prétraitement des données (Étape 3).")

    selected_key = request.GET.get("algo")
    context = _build_context(request, dataset, selected_key)
    return render(request, "ml_engine/index.html", context)


@require_GET
def params_view(request: HttpRequest) -> HttpResponse:
    """HTMX endpoint to return the dynamic hyperparameter controls for a selected algorithm."""
    dataset = get_active_dataset(request)
    algo_key = request.GET.get("algo")

    prepared = load_prepared(dataset) if dataset else None
    task_type = prepared.task_type if prepared else "classification"

    try:
        algo = ModelRegistry.get(algo_key)
        hyperparams = algo.get_hyperparams(task_type)
    except Exception:
        return HttpResponse('<div class="text-xs text-rose-400 p-2">Algorithme introuvable.</div>')

    return render(
        request,
        "ml_engine/_params_form.html",
        {
            "algo": algo,
            "hyperparams": hyperparams,
            "task_type": task_type,
        },
    )


@require_POST
def train_view(request: HttpRequest) -> HttpResponse:
    """Execute model training, update session state, and return updated components."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Aucun jeu de données actif.")
        return redirect("datasets:index")

    prepared = load_prepared(dataset)
    if prepared is None:
        messages.error(request, "Veuillez d'abord exécuter le prétraitement à l'Étape 3.")
        return redirect("preprocessing:index")

    algo_key = request.POST.get("algo_key")
    cv_folds = int(request.POST.get("cv_folds", 5))

    try:
        algo = ModelRegistry.get(algo_key)
        fields = algo.get_hyperparams(prepared.task_type)
        params: Dict[str, Any] = {}
        for f in fields:
            raw_val = request.POST.get(f.name)
            params[f.name] = f.clean_value(raw_val)

        # Train model & record metrics
        result = train_model_for_dataset(
            dataset=dataset,
            algo_key=algo_key,
            params=params,
            cv_folds=cv_folds,
        )

        # Update global session workspace state
        update_state(
            request,
            model_trained=True,
            selected_model_name=result.algo_name,
            best_score=result.summary_score,
            active_step=4,
        )

        messages.success(
            request,
            f"Modèle '{result.algo_name}' entraîné avec succès en {round(result.training_time_ms, 1)} ms !"
        )

    except (MLEngineError, ValueError, Exception) as exc:
        messages.error(request, f"Erreur lors de l'entraînement : {str(exc)}")
        if request.headers.get("HX-Request") == "true":
            context = _build_context(request, dataset, algo_key)
            return render(request, "ml_engine/index.html", context)
        return redirect("ml_engine:index")

    # If HTMX request, render the updated page or partials
    if request.headers.get("HX-Request") == "true":
        context = _build_context(request, dataset, algo_key)
        return render(request, "ml_engine/index.html", context)

    return redirect("ml_engine:index")


@require_GET
def leaderboard_view(request: HttpRequest) -> HttpResponse:
    """HTMX endpoint returning the refreshed leaderboard component."""
    dataset = get_active_dataset(request)
    board = load_leaderboard(dataset) if dataset else []
    return render(
        request,
        "ml_engine/_leaderboard.html",
        {"leaderboard": board},
    )


@require_POST
def reset_view(request: HttpRequest) -> HttpResponse:
    """Clear all trained models and leaderboard for the active dataset."""
    dataset = get_active_dataset(request)
    if dataset:
        clear_models(dataset)
        update_state(
            request,
            model_trained=False,
            selected_model_name=None,
            best_score=None,
            active_step=4,
        )
        messages.success(request, "Historique des modèles et leaderboard réinitialisés.")

    return redirect("ml_engine:index")
