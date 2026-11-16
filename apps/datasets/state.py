"""
Session helpers for the active dataset.

The workspace state lives in ``request.session['project_state']`` (a contract
defined by the core app and consumed by the sidebar, the topbar and the
dashboard).  These helpers are the only place that should mutate it, so the
other modules can rely on a consistent shape.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from apps.core.context_processors import DEFAULT_PROJECT_STATE
from apps.datasets.models import Dataset

SESSION_KEY = "project_state"


def current_state(request: HttpRequest) -> dict[str, Any]:
    """Return the workspace state, creating it from defaults when missing."""
    state = request.session.get(SESSION_KEY)
    if not isinstance(state, dict):
        state = dict(DEFAULT_PROJECT_STATE)
        request.session[SESSION_KEY] = state
        request.session.modified = True
    return state


def update_state(request: HttpRequest, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into the workspace state and persist the session."""
    state = current_state(request)
    state.update(fields)
    request.session[SESSION_KEY] = state
    request.session.modified = True
    return state


def set_active_dataset(request: HttpRequest, dataset: Dataset) -> dict[str, Any]:
    """Make ``dataset`` the active project and reset every downstream artefact."""
    return update_state(
        request,
        dataset_id=str(dataset.id),
        dataset_name=dataset.name,
        total_rows=dataset.n_rows,
        total_cols=dataset.n_cols,
        target_column=None,
        task_type=None,
        preprocessing_done=False,
        model_trained=False,
        selected_model_name=None,
        best_score=None,
        active_step=2,
    )


def clear_active_dataset(request: HttpRequest) -> dict[str, Any]:
    """Detach the active dataset and invalidate the pipeline state."""
    return update_state(
        request,
        dataset_id=None,
        dataset_name=None,
        total_rows=0,
        total_cols=0,
        target_column=None,
        task_type=None,
        preprocessing_done=False,
        model_trained=False,
        selected_model_name=None,
        best_score=None,
        active_step=1,
    )


def get_active_dataset(request: HttpRequest) -> Dataset | None:
    """Return the active ``Dataset`` row, or ``None`` if it no longer exists."""
    dataset_id = current_state(request).get("dataset_id")
    if not dataset_id:
        return None
    try:
        return Dataset.objects.get(pk=dataset_id)
    except (Dataset.DoesNotExist, ValueError, TypeError):
        clear_active_dataset(request)
        return None
