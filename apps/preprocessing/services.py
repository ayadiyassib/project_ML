"""
Artifact and orchestration layer for preprocessing.

Responsibilities
----------------
* Turn a ``Dataset`` + ``PreprocessingConfig`` into a ``PreparedData`` bundle.
* Persist the artefacts under ``media/projects/<dataset_id>/`` so that the ML
  engine and the inference module can reload them without recomputing anything.
* Provide the UI with sensible default feature selection and per-column task
  suggestions.
* Export the cleaned dataframe as CSV.

The contract exposed to the other modules is ``load_prepared(dataset)`` and
``load_pipeline(dataset)`` — both are safe to call at any time and return
``None`` when preprocessing has not run yet.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from django.conf import settings

from apps.datasets.models import Dataset
from apps.datasets.services import load_dataframe
from apps.preprocessing.config import PreprocessingConfig, PreprocessingError
from apps.preprocessing.pipeline import DataPreprocessor, PreparedData, infer_task_type

MAX_ONEHOT_CARDINALITY = 30


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def project_dir(dataset: Dataset) -> Path:
    return Path(settings.MEDIA_ROOT) / "projects" / str(dataset.id)


def artifact_paths(dataset: Dataset) -> dict[str, Path]:
    root = project_dir(dataset)
    return {
        "dir": root,
        "pipeline": root / "pipeline.joblib",
        "prepared": root / "prepared.joblib",
        "config": root / "config.json",
    }


def prepare(dataset: Dataset, config: PreprocessingConfig) -> PreparedData:
    """Run the full preprocessing and persist the artefacts."""
    frame = load_dataframe(dataset)
    preprocessor = DataPreprocessor(frame, target_col=config.target)
    prepared = preprocessor.build_and_fit_transform(config)
    save_prepared(dataset, prepared)
    return prepared


def save_prepared(dataset: Dataset, prepared: PreparedData) -> None:
    paths = artifact_paths(dataset)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared.pipeline, paths["pipeline"])
    joblib.dump(prepared, paths["prepared"])
    paths["config"].write_text(
        json.dumps(prepared.config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_prepared(dataset: Dataset) -> PreparedData | None:
    """Return the saved preprocessing bundle, or ``None`` if it does not exist."""
    path = artifact_paths(dataset)["prepared"]
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except (OSError, ValueError, EOFError, ModuleNotFoundError):
        return None


def load_pipeline(dataset: Dataset):
    """Return the fitted ``ColumnTransformer`` used for inference."""
    path = artifact_paths(dataset)["pipeline"]
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except (OSError, ValueError, EOFError, ModuleNotFoundError):
        return None


def load_config(dataset: Dataset) -> PreprocessingConfig | None:
    path = artifact_paths(dataset)["config"]
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return PreprocessingConfig.from_dict(payload)


def clear_artifacts(dataset: Dataset) -> None:
    """Delete every artefact of the project (preprocessing + trained models)."""
    root = project_dir(dataset)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# UI defaults
# ---------------------------------------------------------------------------


def default_config(dataset: Dataset, frame: pd.DataFrame) -> PreprocessingConfig:
    """Pre-fill the form from the dataset profile and its declared task hint."""
    target = dataset.target_hint
    if target not in frame.columns:
        target = None

    if dataset.task_hint == "clustering":
        task_type = "clustering"
    elif dataset.task_hint in ("classification", "regression"):
        task_type = dataset.task_hint
    else:
        task_type = infer_task_type(frame[target]) if target is not None else "classification"

    if target is None:
        task_type = "clustering"

    selected, _ = suggested_feature_selection(dataset, target)

    return PreprocessingConfig(
        target=target,
        features=selected,
        task_type=task_type,
        missing_threshold=1.0,
    )


def suggested_feature_selection(
    dataset: Dataset, target: str | None
) -> tuple[list[str], dict[str, str]]:
    """Return the recommended feature list plus the reason each column is excluded."""
    selected: list[str] = []
    reasons: dict[str, str] = {}

    for column in dataset.columns_meta or []:
        name = column["name"]
        if name == target:
            reasons[name] = "Variable cible"
            continue
        if column.get("is_constant"):
            reasons[name] = "Colonne constante"
            continue
        if column.get("is_id_like"):
            reasons[name] = "Identifiant probable"
            continue
        if column.get("kind") in ("categorical", "text") and column.get("unique", 0) > MAX_ONEHOT_CARDINALITY:
            reasons[name] = f"Trop de modalités ({column['unique']}) pour un encodage one-hot"
            continue
        selected.append(name)

    return selected, reasons


def feature_options(
    dataset: Dataset,
    config: PreprocessingConfig,
    reasons: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the checkbox list rendered in the form."""
    chosen = set(config.features or [])
    options: list[dict[str, Any]] = []

    for column in dataset.columns_meta or []:
        name = column["name"]
        options.append(
            {
                "name": name,
                "kind": column.get("kind", "categorical"),
                "dtype": column.get("dtype", ""),
                "missing_pct": column.get("missing_pct", 0),
                "unique": column.get("unique", 0),
                "is_target": name == config.target,
                "selected": name in chosen and name != config.target,
                "note": reasons.get(name, ""),
            }
        )
    return options


def task_map(frame: pd.DataFrame) -> dict[str, str]:
    """Map every column to the task it would imply as a target."""
    mapping: dict[str, str] = {"": "clustering"}
    for column in frame.columns:
        mapping[str(column)] = infer_task_type(frame[column])
    return mapping


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def clean_frame(dataset: Dataset) -> pd.DataFrame:
    """Dataframe exported as "cleaned data" (structural cleaning only)."""
    prepared = load_prepared(dataset)
    if prepared is not None and prepared.df_clean is not None:
        return prepared.df_clean.copy()
    return load_dataframe(dataset)


def export_clean_csv(dataset: Dataset) -> tuple[str, bytes]:
    """Return ``(filename, payload)`` for the cleaned CSV download."""
    frame = clean_frame(dataset)
    if frame is None or frame.empty:
        raise PreprocessingError("Aucune donnée à exporter.")

    stem = Path(dataset.original_filename or dataset.name).stem or "dataset"
    payload = frame.to_csv(index=False).encode("utf-8-sig")
    return f"{stem}_clean.csv", payload
