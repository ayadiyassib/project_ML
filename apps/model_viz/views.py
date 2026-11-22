"""
Views for the diagnostics module (step 5 of the pipeline).

The page renders every fast diagnostic of the last trained model; the three
expensive ones (decision boundary, permutation importance, elbow curve) are
HTMX partials loaded when their card scrolls into view.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET

from apps.datasets.services import DatasetError
from apps.datasets.state import clear_active_dataset, current_state, get_active_dataset, update_state
from apps.model_viz import services
from apps.model_viz.services import DiagnosticsError
from apps.preprocessing.services import load_prepared

logger = logging.getLogger(__name__)
PAGE_TITLE = "Diagnostics & Visualisations"


def _context(request: HttpRequest):
    """Return ``(dataset, prepared, model, warning)``; any of them may be None."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return None, None, None, None
    prepared = load_prepared(dataset)
    if prepared is None:
        return dataset, None, None, None
    model, warning = services.load_trained_model(dataset, prepared)
    return dataset, prepared, model, warning


@require_GET
def index_view(request: HttpRequest) -> HttpResponse:
    try:
        dataset, prepared, model, warning = _context(request)
    except DatasetError as exc:
        messages.error(request, str(exc))
        clear_active_dataset(request)
        return redirect("datasets:index")

    context = {
        "title": PAGE_TITLE,
        "dataset": dataset,
        "prepared": prepared.to_summary() if prepared else None,
        "model": model,
        "model_warning": warning,
        "diagnostics": None,
        "error": None,
    }
    if model is not None and prepared is not None:
        try:
            context["diagnostics"] = services.build_diagnostics(model, prepared)
            # Same convention as the other modules: active_step = furthest step reached.
            if (current_state(request).get("active_step") or 1) < 5:
                update_state(request, active_step=5)
        except Exception as exc:  # noqa: BLE001 - never 500 the page on a diagnostics failure
            logger.exception("Diagnostics failed")
            context["error"] = f"Impossible de calculer les diagnostics : {exc}"
    return render(request, "model_viz/index.html", context)


def _lazy(request: HttpRequest, builder) -> HttpResponse:
    dataset, prepared, model, warning = _context(request)
    if prepared is None:
        return render(request, "eda/_error.html", {"error": "Aucun prétraitement disponible."})
    try:
        chart = builder(prepared, model)
    except DiagnosticsError as exc:
        return render(request, "eda/_error.html", {"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lazy diagnostic failed")
        return render(request, "eda/_error.html", {"error": f"Calcul impossible : {exc}"})
    return render(request, "eda/_chart_card.html", {"chart": chart})


def _require_model(model):
    if model is None:
        raise DiagnosticsError("Aucun modèle entraîné à jour : entraînez un modèle à l'étape 4.")
    return model


@require_GET
def boundary_view(request: HttpRequest) -> HttpResponse:
    return _lazy(request, lambda prepared, model: services.decision_boundary(_require_model(model), prepared))


@require_GET
def permutation_view(request: HttpRequest) -> HttpResponse:
    return _lazy(request, lambda prepared, model: services.permutation_chart(_require_model(model), prepared))


@require_GET
def elbow_view(request: HttpRequest) -> HttpResponse:
    def build(prepared, model):
        chosen = getattr(services.final_estimator(model.estimator), "n_clusters", None) if model else None
        chart, _data = services.elbow_chart(prepared, chosen)
        return chart

    return _lazy(request, build)
