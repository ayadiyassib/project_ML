"""
Global context processor for ML Studio.
Injects persistent workspace and pipeline state into every template.
"""

from typing import Dict, Any

DEFAULT_PROJECT_STATE: Dict[str, Any] = {
    "project_id": None,
    "dataset_name": None,
    "dataset_path": None,
    "total_rows": 0,
    "total_cols": 0,
    "target_column": None,
    "task_type": None,  # 'classification' | 'regression' | 'clustering'
    "preprocessing_done": False,
    "model_trained": False,
    "selected_model_name": None,
    "best_score": None,
    "active_step": 1,  # 1: Dataset -> 2: EDA -> 3: Preprocessing -> 4: ML -> 5: Viz -> 6: Export
}


def project_context(request) -> Dict[str, Any]:
    """
    Context processor that ensures 'project', 'current_step',
    and user RBAC context variables are available in every template.
    """
    if "project_state" not in request.session:
        request.session["project_state"] = DEFAULT_PROJECT_STATE.copy()

    state = request.session.get("project_state", DEFAULT_PROJECT_STATE)

    user = getattr(request, "user", None)
    if user and user.is_authenticated:
        profile = getattr(user, "profile", None)
        initials = profile.initials if profile else (user.username[:2]).upper()
        avatar_color = profile.avatar_color if profile else "#B8323C"
        is_admin = bool(user.is_staff or user.is_superuser)
        role_label = "Administrateur" if is_admin else "Membre Studio"
    else:
        initials = "GU"
        avatar_color = "#756e66"
        is_admin = False
        role_label = "Visiteur"

    return {
        "project": state,
        "current_step": state.get("active_step", 1),
        "user_initials": initials,
        "avatar_color": avatar_color,
        "is_admin": is_admin,
        "user_role_label": role_label,
    }

