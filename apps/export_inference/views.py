"""
Views for Model Export, Predictions Export, and Live Inference (Step 6 of ML Studio).
"""

from __future__ import annotations

import os
from typing import Any, Dict
from django.contrib import messages
from django.http import FileResponse, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from apps.datasets.state import get_active_dataset
from apps.preprocessing.services import load_prepared
from apps.ml_engine.services import load_latest_model, load_latest_result, load_leaderboard
from apps.export_inference.exporter import (
    ExportError,
    export_model_joblib,
    generate_predictions_csv,
    generate_python_snippet,
)
from apps.export_inference.predictor import get_feature_input_schema, predict_from_inputs


def _build_context(request: HttpRequest, dataset) -> Dict[str, Any]:
    """Assemble context for the export and inference hub."""
    prepared = load_prepared(dataset) if dataset else None
    latest_result = load_latest_result(dataset) if dataset else None
    leaderboard = load_leaderboard(dataset) if dataset else []
    model = load_latest_model(dataset) if dataset else None
    inference_schema = get_feature_input_schema(dataset) if dataset else []

    # Sample row for Python code snippet
    sample_row = {}
    if prepared and prepared.raw_features and prepared.df_clean is not None and not prepared.df_clean.empty:
        first_row = prepared.df_clean[prepared.raw_features].iloc[0].to_dict()
        # Ensure values are JSON serializable
        sample_row = {
            k: (float(v) if isinstance(v, (int, float)) else str(v))
            for k, v in first_row.items()
        }

    python_snippet = generate_python_snippet(dataset, sample_row) if dataset else ""

    return {
        "title": "Export de Pipeline & Inférence",
        "dataset": dataset,
        "prepared": prepared,
        "latest_result": latest_result,
        "leaderboard": leaderboard,
        "has_model": model is not None,
        "python_snippet": python_snippet,
        "sample_row": sample_row,
        "inference_schema": inference_schema,
    }


def index_view(request: HttpRequest) -> HttpResponse:
    """Main hub for exports, playground, and audit report."""
    dataset = get_active_dataset(request)
    context = _build_context(request, dataset)
    return render(request, "export_inference/index.html", context)


@require_GET
def download_model_view(request: HttpRequest) -> HttpResponse:
    """Download the production-ready Scikit-Learn Pipeline (.joblib)."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Aucun jeu de données actif.")
        return redirect("datasets:index")

    try:
        file_path = export_model_joblib(dataset)
        response = FileResponse(open(file_path, "rb"), as_attachment=True, filename=file_path.name)
        return response
    except (ExportError, Exception) as exc:
        messages.error(request, f"Échec de l'export du modèle : {exc}")
        return redirect("export_inference:index")


@require_GET
def download_predictions_view(request: HttpRequest) -> HttpResponse:
    """Download test set predictions (CSV)."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Aucun jeu de données actif.")
        return redirect("datasets:index")

    try:
        csv_path = generate_predictions_csv(dataset)
        response = FileResponse(open(csv_path, "rb"), as_attachment=True, filename=csv_path.name)
        return response
    except (ExportError, Exception) as exc:
        messages.error(request, f"Échec de l'export des prédictions : {exc}")
        return redirect("export_inference:index")


@require_POST
def predict_view(request: HttpRequest) -> HttpResponse:
    """Execute live single-sample prediction and return HTMX partial card."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return HttpResponse('<div class="p-4 text-xs text-rose-400 bg-rose-950/40 rounded-xl border border-rose-900">Aucun jeu de données actif.</div>')

    try:
        result = predict_from_inputs(dataset, request.POST)
        return render(
            request,
            "export_inference/_prediction_result.html",
            {"prediction_result": result},
        )
    except Exception as exc:
        return HttpResponse(f'<div class="p-4 text-xs text-rose-400 bg-rose-950/40 rounded-xl border border-rose-900">Erreur lors de la prédiction : {exc}</div>')
