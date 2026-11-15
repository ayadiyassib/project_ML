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
    Context processor that ensures 'project' and 'current_step'
    are always available in any template.
    """
    if "project_state" not in request.session:
        request.session["project_state"] = DEFAULT_PROJECT_STATE.copy()

    state = request.session.get("project_state", DEFAULT_PROJECT_STATE)
    return {
        "project": state,
        "current_step": state.get("active_step", 1),
    }

