"""
Views for the ingestion module (step 1 of the pipeline).

The page is server-rendered; only the preview table is paginated through HTMX
so the main document is never reloaded when browsing rows.
"""

from __future__ import annotations

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.accounts.decorators import studio_login_required
from apps.datasets import services
from apps.datasets.models import Dataset
from apps.datasets.state import clear_active_dataset, get_active_dataset, set_active_dataset

DEFAULT_PAGE_SIZE = services.DEFAULT_PREVIEW_PAGE_SIZE


def _page_size(request: HttpRequest) -> int:
    try:
        size = int(request.GET.get("page_size", DEFAULT_PAGE_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_PAGE_SIZE
    return size if size in (10, 25, 50, 100) else DEFAULT_PAGE_SIZE


def index_view(request: HttpRequest) -> HttpResponse:
    """Upload zone, sample catalog, profile and preview of the active dataset."""
    dataset = get_active_dataset(request)
    profile = None
    preview = None

    if dataset is not None:
        try:
            frame = services.load_dataframe(dataset)
            profile = services.profile_dataframe(frame)
            preview = services.preview_page(frame, page=1, page_size=_page_size(request))
        except services.DatasetError as exc:
            messages.error(request, f"Impossible de relire « {dataset.name} » : {exc}")
            clear_active_dataset(request)
            dataset = None

    user = getattr(request, "user", None)
    if user and user.is_authenticated and (user.is_staff or user.is_superuser):
        recent = Dataset.objects.all().select_related("user")
    elif user and user.is_authenticated:
        recent = Dataset.objects.filter(
            Q(user=user) | Q(source=Dataset.Source.SAMPLE) | Q(user__isnull=True)
        )
    else:
        recent = Dataset.objects.filter(
            Q(source=Dataset.Source.SAMPLE) | Q(user__isnull=True)
        )

    if dataset is not None:
        recent = recent.exclude(pk=dataset.pk)

    context = {
        "title": "Ingestion de Données",
        "dataset": dataset,
        "profile": profile,
        "preview": preview,
        "samples": services.available_samples(),
        "recent_datasets": recent[:6],
        "max_upload_human": services.human_size(services.MAX_UPLOAD_BYTES),
        "accepted_extensions": ", ".join(services.SUPPORTED_EXTENSIONS),
    }
    return render(request, "datasets/index.html", context)


def preview_view(request: HttpRequest) -> HttpResponse:
    """HTMX partial: one page of rows for the active dataset."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return render(
            request,
            "datasets/_preview.html",
            {"preview": None, "error": "Aucun jeu de données actif."},
        )

    try:
        frame = services.load_dataframe(dataset)
        preview = services.preview_page(frame, page=_page_int(request), page_size=_page_size(request))
    except services.DatasetError as exc:
        return render(request, "datasets/_preview.html", {"preview": None, "error": str(exc)})

    return render(request, "datasets/_preview.html", {"preview": preview, "dataset": dataset})


def _page_int(request: HttpRequest) -> int:
    try:
        return max(1, int(request.GET.get("page", 1)))
    except (TypeError, ValueError):
        return 1


@require_POST
def upload_view(request: HttpRequest) -> HttpResponse:
    """Handle a CSV / TSV / JSON / Excel upload."""
    user = getattr(request, "user", None)
    try:
        dataset = services.create_dataset_from_upload(
            request.FILES.get("file"),
            user=user if (user and user.is_authenticated) else None,
        )
    except services.DatasetError as exc:
        messages.error(request, str(exc))
        return redirect("datasets:index")

    set_active_dataset(request, dataset)
    messages.success(
        request,
        f"« {dataset.name} » importé : {dataset.n_rows:,} lignes × {dataset.n_cols} colonnes.",
    )
    return redirect("datasets:index")


@require_POST
def sample_view(request: HttpRequest) -> HttpResponse:
    """Load one of the bundled demonstration datasets."""
    slug = (request.POST.get("slug") or "").strip()
    try:
        dataset = services.create_dataset_from_sample(slug)
    except services.DatasetError as exc:
        messages.error(request, str(exc))
        return redirect("datasets:index")

    set_active_dataset(request, dataset)
    messages.success(request, f"Jeu de démonstration « {dataset.name} » chargé.")
    return redirect("datasets:index")


@require_POST
def url_view(request: HttpRequest) -> HttpResponse:
    """Import a remote CSV / JSON / Excel file from an HTTP URL."""
    url = (request.POST.get("url") or "").strip()
    if not url:
        messages.error(request, "Veuillez saisir une URL.")
        return redirect("datasets:index")

    user = getattr(request, "user", None)
    try:
        dataset = services.create_dataset_from_url(
            url,
            user=user if (user and user.is_authenticated) else None,
        )
    except services.DatasetError as exc:
        messages.error(request, str(exc))
        return redirect("datasets:index")

    set_active_dataset(request, dataset)
    messages.success(request, f"« {dataset.name} » importé depuis l'URL distante.")
    return redirect("datasets:index")


@require_POST
def activate_view(request: HttpRequest, pk) -> HttpResponse:
    """Re-open a previously imported dataset."""
    user = getattr(request, "user", None)
    if user and user.is_authenticated and (user.is_staff or user.is_superuser):
        dataset = get_object_or_404(Dataset, pk=pk)
    elif user and user.is_authenticated:
        dataset = get_object_or_404(
            Dataset.objects.filter(
                Q(user=user) | Q(source=Dataset.Source.SAMPLE) | Q(user__isnull=True)
            ),
            pk=pk,
        )
    else:
        dataset = get_object_or_404(
            Dataset.objects.filter(
                Q(source=Dataset.Source.SAMPLE) | Q(user__isnull=True)
            ),
            pk=pk,
        )
    set_active_dataset(request, dataset)
    messages.success(request, f"« {dataset.name} » est maintenant le jeu de données actif.")
    return redirect("datasets:index")


@require_POST
def delete_view(request: HttpRequest) -> HttpResponse:
    """Delete an imported dataset (and its file) from the workspace."""
    dataset = get_object_or_404(Dataset, pk=request.POST.get("dataset_id"))
    user = getattr(request, "user", None)

    # Only owner or admin can delete
    is_staff = bool(user and user.is_authenticated and (user.is_staff or user.is_superuser))
    is_owner = bool(user and user.is_authenticated and dataset.user_id == user.pk)
    if not (is_staff or is_owner or dataset.user is None):
        messages.error(request, "Vous n'avez pas l'autorisation de supprimer ce jeu de données.")
        return redirect("datasets:index")

    name = dataset.name
    is_active = get_active_dataset(request) == dataset

    dataset.delete()
    if is_active:
        clear_active_dataset(request)

    messages.success(request, f"« {name} » supprimé du studio.")
    return redirect("datasets:index")
