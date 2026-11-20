"""
Views for the EDA module (step 2 of the pipeline).

The page is server-rendered with the dataset overview; every other tab is an
HTMX partial (``hx-get`` on load and on change) so choosing another column or
correlation method only re-renders one card.
"""

from __future__ import annotations

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET

from apps.datasets.services import DatasetError, load_dataframe
from apps.datasets.state import clear_active_dataset, current_state, get_active_dataset
from apps.eda import services
from apps.eda.services import EDAError

PAGE_TITLE = "Analyse Exploratoire des Données (EDA)"


def _load(request: HttpRequest):
    """Return ``(dataset, frame, target)`` or ``None`` when no dataset is active."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return None
    frame = load_dataframe(dataset)
    target = services.resolve_target(dataset, current_state(request), frame)
    return dataset, frame, target


def _partial_error(request: HttpRequest, message: str) -> HttpResponse:
    return render(request, "eda/_error.html", {"error": message})


def _flag(request: HttpRequest, name: str, default: bool) -> bool:
    value = request.GET.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "on", "yes")


@require_GET
def index_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
    except DatasetError as exc:
        messages.error(request, str(exc))
        clear_active_dataset(request)
        return redirect("datasets:index")

    if loaded is None:
        return render(request, "eda/index.html", {"title": PAGE_TITLE, "dataset": None})

    dataset, frame, target = loaded
    overview = services.overview(dataset, frame, target)
    defaults = services.default_scatter_axes(dataset, frame, target)
    columns = [str(c) for c in frame.columns]
    # Open on the most informative continuous feature (the target is one click away).
    features = [c for c in defaults["numeric"] if c != target]
    default_column = defaults["x"] if defaults["x"] in features else (features[0] if features else columns[0])

    context = {
        "title": PAGE_TITLE,
        "dataset": dataset,
        "target": target,
        "columns": columns,
        "kinds": {r["name"]: r["kind"] for r in overview["records"]},
        "default_column": default_column,
        "numeric_columns": defaults["numeric"],
        "color_columns": services.color_columns(dataset, frame),
        "defaults": defaults,
        "max_pairplot": services.MAX_PAIRPLOT_COLUMNS,
        **overview,
    }
    return render(request, "eda/index.html", context)


@require_GET
def univariate_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
        if loaded is None:
            return _partial_error(request, "Aucun jeu de données actif.")
        dataset, frame, target = loaded
        result = services.univariate(dataset, frame, request.GET.get("column", ""), target)
    except (EDAError, DatasetError) as exc:
        return _partial_error(request, str(exc))
    return render(request, "eda/_univariate.html", result)


@require_GET
def correlation_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
        if loaded is None:
            return _partial_error(request, "Aucun jeu de données actif.")
        dataset, frame, _target = loaded
        result = services.correlation(
            dataset, frame, request.GET.get("method", "pearson"), _flag(request, "mask", True)
        )
    except (EDAError, DatasetError) as exc:
        return _partial_error(request, str(exc))
    return render(request, "eda/_correlation.html", result)


@require_GET
def scatter_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
        if loaded is None:
            return _partial_error(request, "Aucun jeu de données actif.")
        _dataset, frame, _target = loaded
        chart = services.scatter(
            frame, request.GET.get("x", ""), request.GET.get("y", ""),
            request.GET.get("color") or None, _flag(request, "trend", False),
        )
    except (EDAError, DatasetError, ValueError) as exc:
        return _partial_error(request, str(exc))
    return render(request, "eda/_chart_card.html", {"chart": chart})


@require_GET
def scatter3d_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
        if loaded is None:
            return _partial_error(request, "Aucun jeu de données actif.")
        _dataset, frame, _target = loaded
        chart = services.scatter_3d(
            frame, request.GET.get("x", ""), request.GET.get("y", ""), request.GET.get("z", ""),
            request.GET.get("color") or None,
        )
    except (EDAError, DatasetError, ValueError) as exc:
        return _partial_error(request, str(exc))
    return render(request, "eda/_chart_card.html", {"chart": chart})


@require_GET
def pairplot_view(request: HttpRequest) -> HttpResponse:
    try:
        loaded = _load(request)
        if loaded is None:
            return _partial_error(request, "Aucun jeu de données actif.")
        _dataset, frame, _target = loaded
        chart = services.pairplot(frame, request.GET.getlist("columns"), request.GET.get("color") or None)
    except (EDAError, DatasetError, ValueError) as exc:
        return _partial_error(request, str(exc))
    return render(request, "eda/_chart_card.html", {"chart": chart})
