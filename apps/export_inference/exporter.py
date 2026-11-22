"""
Standalone pipeline packaging and artifact export service (.joblib, predictions CSV, Python snippet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.utils.text import slugify
from sklearn.pipeline import Pipeline

from apps.datasets.models import Dataset
from apps.preprocessing.services import load_pipeline, load_prepared, project_dir
from apps.ml_engine.services import load_latest_model, load_latest_result


class ExportError(Exception):
    """Custom exception raised during model or prediction export."""
    pass


def exports_dir(dataset: Dataset) -> Path:
    """Return directory for exported deliverables."""
    path = project_dir(dataset) / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_standalone_pipeline(dataset: Dataset) -> Pipeline:
    """
    Combine the fitted ColumnTransformer from Preprocessing with the trained ML Model.
    Produces a single production-ready Scikit-Learn Pipeline:
        Final Pipeline = ColumnTransformer + Model
    """
    preprocessor = load_pipeline(dataset)
    if preprocessor is None:
        raise ExportError("Le pipeline de prétraitement est introuvable. Veuillez d'abord prétraiter vos données.")

    model = load_latest_model(dataset)
    if model is None:
        raise ExportError("Aucun modèle entraîné n'a été trouvé. Veuillez d'abord entraîner un modèle à l'étape 4.")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def export_model_joblib(dataset: Dataset) -> Path:
    """Serialize the standalone pipeline to disk and return its path."""
    pipeline = get_standalone_pipeline(dataset)
    slug = slugify(dataset.name) or "model"
    path = exports_dir(dataset) / f"{slug}_pipeline.joblib"
    joblib.dump(pipeline, path, compress=3)
    return path


def generate_predictions_csv(dataset: Dataset) -> Path:
    """
    Compile test set predictions into a detailed CSV file:
    Features + Actual Values + Predicted Values + Residuals / Probabilities.
    """
    prepared = load_prepared(dataset)
    if prepared is None:
        raise ExportError("Données préparées introuvables.")

    model = load_latest_model(dataset)
    if model is None:
        raise ExportError("Modèle entraîné introuvable.")

    # In clustering, we use X_train
    if prepared.task_type == "clustering":
        X_mat = prepared.X_train
        y_pred = getattr(model, "labels_", model.predict(X_mat))
        
        # Build DataFrame with transformed features
        df_out = pd.DataFrame(X_mat, columns=prepared.feature_names)
        df_out["cluster_assigned"] = y_pred
        
        csv_path = exports_dir(dataset) / "clustering_predictions.csv"
        df_out.to_csv(csv_path, index=False)
        return csv_path

    # Supervised learning (Regression or Classification)
    if prepared.X_test is None or prepared.y_test is None:
        raise ExportError("Le jeu de test est vide ou non défini.")

    X_test_mat = prepared.X_test
    y_test_arr = np.asarray(prepared.y_test)
    y_pred_arr = model.predict(X_test_mat)

    # Reconstruct columns
    df_out = pd.DataFrame(X_test_mat, columns=prepared.feature_names)
    df_out["actual_value"] = y_test_arr
    df_out["predicted_value"] = y_pred_arr

    if prepared.task_type == "regression":
        df_out["residual"] = np.round(y_test_arr - y_pred_arr, 4)
        df_out["abs_error"] = np.round(np.abs(y_test_arr - y_pred_arr), 4)
    elif prepared.task_type == "classification":
        df_out["is_correct"] = (y_test_arr == y_pred_arr)
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test_mat)
                classes = getattr(model, "classes_", [f"class_{i}" for i in range(probas.shape[1])])
                for idx, cls_name in enumerate(classes):
                    df_out[f"proba_{cls_name}"] = np.round(probas[:, idx], 4)
                df_out["confidence"] = np.round(np.max(probas, axis=1) * 100, 2)
            except Exception:
                pass

    csv_path = exports_dir(dataset) / "predictions.csv"
    df_out.to_csv(csv_path, index=False)
    return csv_path


def generate_python_snippet(dataset: Dataset, sample_dict: Optional[Dict[str, Any]] = None) -> str:
    """Generate ready-to-copy Python script demonstrating how to consume the exported .joblib pipeline."""
    sample_data_repr = json.dumps([sample_dict] if sample_dict else [{"feature_example": 42.0}], indent=4)
    slug = slugify(dataset.name) or "model"
    file_name = f"{slug}_pipeline.joblib"

    return f'''import joblib
import pandas as pd

# 1. Charger le pipeline autonome Scikit-Learn
pipeline = joblib.load("{file_name}")

# 2. Préparer de nouvelles données brutes (non transformées)
new_data = pd.DataFrame({sample_data_repr})

# 3. Prédire directement : le pipeline exécute imputation, encodage et prédiction
predictions = pipeline.predict(new_data)
print("Prédictions :", predictions)

# Optionnel (Classification) : Obtenir les probabilités
if hasattr(pipeline, "predict_proba"):
    probabilities = pipeline.predict_proba(new_data)
    print("Probabilités de classes :", probabilities)
'''
