"""
Orchestration and persistence service layer for the ML Engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
from django.conf import settings
from sklearn.base import BaseEstimator

from apps.datasets.models import Dataset
from apps.preprocessing.services import load_prepared, project_dir
from apps.ml_engine.algorithms.registry import ModelRegistry
from apps.ml_engine.algorithms.base import AlgorithmMeta
from apps.ml_engine.trainer import ModelTrainer, TrainingResult


class MLEngineError(Exception):
    """Custom exception for ML Engine operational errors."""
    pass


def models_dir(dataset: Dataset) -> Path:
    """Return the directory path where trained models and metrics are saved."""
    path = project_dir(dataset) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_artifacts(dataset: Dataset) -> Dict[str, Path]:
    """Dictionary of file paths for model persistence."""
    root = models_dir(dataset)
    return {
        "dir": root,
        "latest_model": root / "latest_model.joblib",
        "latest_result": root / "latest_result.json",
        "leaderboard": root / "leaderboard.json",
    }


def algo_model_path(dataset: Dataset, algo_key: str) -> Path:
    """One fitted estimator per leaderboard entry (used by the diagnostics comparison)."""
    return models_dir(dataset) / f"model_{algo_key}.joblib"


def get_available_algorithms(task_type: str) -> List[AlgorithmMeta]:
    """Return algorithms compatible with current dataset task."""
    return ModelRegistry.for_task(task_type)


def load_latest_model(dataset: Dataset) -> Optional[BaseEstimator]:
    """Load the latest trained model estimator from disk."""
    path = model_artifacts(dataset)["latest_model"]
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_latest_result(dataset: Dataset) -> Optional[Dict[str, Any]]:
    """Load the latest model training evaluation dictionary."""
    path = model_artifacts(dataset)["latest_result"]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_leaderboard(dataset: Dataset) -> List[Dict[str, Any]]:
    """Load the model comparison leaderboard sorted by best score."""
    path = model_artifacts(dataset)["leaderboard"]
    if not path.exists():
        return []
    try:
        board = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(board, list):
            return []
        # Sort descending by summary_score
        return sorted(board, key=lambda x: x.get("summary_score", 0.0), reverse=True)
    except Exception:
        return []


def save_leaderboard(dataset: Dataset, leaderboard: List[Dict[str, Any]]) -> None:
    """Persist the leaderboard JSON."""
    path = model_artifacts(dataset)["leaderboard"]
    path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")


def record_in_leaderboard(dataset: Dataset, result: TrainingResult) -> List[Dict[str, Any]]:
    """Insert or update a training run in the leaderboard."""
    board = load_leaderboard(dataset)
    
    # Check if an entry for this algo already exists; update it or append
    entry = {
        "algo_key": result.algo_key,
        "algo_name": result.algo_name,
        "task_type": result.task_type,
        "summary_score": round(result.summary_score, 4),
        "cv_mean": result.cv_mean,
        "cv_std": result.cv_std,
        "training_time_ms": round(result.training_time_ms, 2),
        "params": result.params,
        "test_metrics": result.test_metrics,
        "is_best": False,
    }

    # Replace if exists, else append
    existing_idx = next((i for i, b in enumerate(board) if b.get("algo_key") == result.algo_key), None)
    if existing_idx is not None:
        board[existing_idx] = entry
    else:
        board.append(entry)

    # Sort descending
    board = sorted(board, key=lambda x: x.get("summary_score", 0.0), reverse=True)

    # Flag champion
    if board:
        for b in board:
            b["is_best"] = False
        board[0]["is_best"] = True

    save_leaderboard(dataset, board)
    return board


def train_model_for_dataset(
    dataset: Dataset,
    algo_key: str,
    params: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
) -> TrainingResult:
    """
    Complete workflow:
    1. Loads PreparedData
    2. Builds model from ModelRegistry
    3. Trains and computes metrics
    4. Saves latest_model.joblib and latest_result.json
    5. Updates leaderboard
    """
    prepared = load_prepared(dataset)
    if prepared is None:
        raise MLEngineError("Les données n'ont pas encore été prétraitées. Effectuez d'abord l'étape 3.")

    algo_wrapper = ModelRegistry.get(algo_key)
    model = ModelRegistry.build_model(algo_key, prepared.task_type, params)

    trainer = ModelTrainer(model=model, task_type=prepared.task_type, cv_folds=cv_folds)

    result = trainer.train_and_evaluate(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        feature_names=prepared.feature_names,
        class_labels=prepared.class_labels,
        diagnostics_extractor=algo_wrapper.extract_diagnostics,
    )

    # Enrich metadata
    result.algo_key = algo_key
    result.algo_name = algo_wrapper.name
    result.params = {f.name: f.clean_value(params.get(f.name) if params else None) for f in algo_wrapper.get_hyperparams(prepared.task_type)}

    # Persist artifacts
    paths = model_artifacts(dataset)
    joblib.dump(model, paths["latest_model"])
    joblib.dump(model, algo_model_path(dataset, algo_key))  # kept for model comparison
    paths["latest_result"].write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Update Leaderboard
    record_in_leaderboard(dataset, result)

    return result


def clear_models(dataset: Dataset) -> None:
    """Remove trained model artifacts for a dataset."""
    root = models_dir(dataset)
    for f in root.glob("*"):
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass
