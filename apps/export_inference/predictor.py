"""
Interactive inference engine for real-time predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from apps.datasets.models import Dataset
from apps.preprocessing.services import load_prepared
from apps.export_inference.exporter import ExportError, get_standalone_pipeline


def get_feature_input_schema(dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Generate dynamic form field specifications for each original feature
    in the dataset, including min/max for numbers and category choices.
    """
    prepared = load_prepared(dataset)
    if prepared is None:
        return []

    frame = prepared.df_clean
    schema: List[Dict[str, Any]] = []

    for feat in prepared.raw_features:
        is_num = feat in prepared.numeric_features
        field_info: Dict[str, Any] = {
            "name": feat,
            "label": feat.replace("_", " ").title(),
            "type": "numeric" if is_num else "categorical",
            "default": None,
            "min": None,
            "max": None,
            "step": None,
            "choices": [],
        }

        if frame is not None and feat in frame.columns:
            series = frame[feat].dropna()
            if is_num and not series.empty:
                min_v = float(series.min())
                max_v = float(series.max())
                median_v = float(series.median())
                is_int = pd.api.types.is_integer_dtype(series)

                field_info["min"] = round(min_v, 2) if not is_int else int(min_v)
                field_info["max"] = round(max_v, 2) if not is_int else int(max_v)
                field_info["default"] = round(median_v, 2) if not is_int else int(median_v)
                field_info["step"] = 1 if is_int else 0.01
            elif not is_num and not series.empty:
                unique_vals = [str(v) for v in series.unique()][:30]
                mode_v = str(series.mode().iloc[0]) if not series.mode().empty else unique_vals[0]
                field_info["choices"] = unique_vals
                field_info["default"] = mode_v

        schema.append(field_info)

    return schema


def predict_from_inputs(dataset: Dataset, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute instant prediction on a single user-submitted record using
    the bundled ColumnTransformer + Model standalone pipeline.
    """
    prepared = load_prepared(dataset)
    if prepared is None:
        raise ExportError("Données préparées introuvables.")

    pipeline = get_standalone_pipeline(dataset)
    schema = get_feature_input_schema(dataset)

    # Clean and cast input dictionary
    cleaned: Dict[str, Any] = {}
    for f in schema:
        name = f["name"]
        val = raw_inputs.get(name, f["default"])
        if f["type"] == "numeric":
            try:
                cleaned[name] = float(val) if val is not None and val != "" else (f["default"] or 0.0)
            except (ValueError, TypeError):
                cleaned[name] = f["default"] or 0.0
        else:
            cleaned[name] = str(val) if val is not None else (f["default"] or "")

    input_df = pd.DataFrame([cleaned])

    # Transform through pipeline and predict
    prediction_raw = pipeline.predict(input_df)[0]
    task = prepared.task_type

    result: Dict[str, Any] = {
        "task_type": task,
        "raw_prediction": prediction_raw,
        "prediction": str(prediction_raw),
        "inputs": cleaned,
        "probabilities": {},
        "confidence": None,
    }

    # Formatting and probabilities
    if task == "classification":
        model_step = pipeline.named_steps.get("model")
        if model_step and hasattr(model_step, "predict_proba"):
            try:
                preprocessor_step = pipeline.named_steps["preprocessor"]
                X_trans = preprocessor_step.transform(input_df)
                probas = model_step.predict_proba(X_trans)[0]
                classes = getattr(model_step, "classes_", [f"Class {i}" for i in range(len(probas))])

                proba_dict = {
                    str(c): round(float(p) * 100, 2)
                    for c, p in zip(classes, probas)
                }
                result["probabilities"] = proba_dict
                result["confidence"] = max(proba_dict.values())
            except Exception:
                pass

    elif task == "regression":
        pred_num = float(prediction_raw)
        result["prediction"] = f"{pred_num:,.4f}".replace(",", " ")

    elif task == "clustering":
        result["prediction"] = f"Cluster #{prediction_raw}"

    return result
