"""
Diagnostics engine: turns a trained model + the preprocessing artefacts into
the charts of step 5.

Contract with the ML engine (M4)
--------------------------------
The ML engine persists the last trained estimator and its evaluation under
``media/projects/<dataset_id>/models/`` (see ``apps.ml_engine.services``)::

    latest_model.joblib   fitted scikit-learn estimator (or Pipeline)
    latest_result.json    TrainingResult.to_dict(): algo_name, test_metrics…

Nothing here retrains the user's model: predictions are recomputed from
``prepared.X_test`` with the saved estimator, so the charts always describe
exactly the model the user trained.  The only fits performed here are *for
visualisation*: K-Means for k = 2…10 (elbow curve) and a clone of the model
on a 2D PCA projection (decision boundary) — both are labelled as such.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from apps.datasets.models import Dataset
from apps.eda.theme import Chart, make_chart, series_color
from apps.model_viz.diagnostic_plots import ModelDiagnosticPlots as Plots
from apps.preprocessing.pipeline import PreparedData
from apps.preprocessing.services import artifact_paths, project_dir

ELBOW_K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE = 2000
BOUNDARY_TRAIN_SAMPLE = 2000
BOUNDARY_GRID = 120
PERMUTATION_SAMPLE = 600
TOP_FEATURES = 15
MAX_ROC_CLASSES = 8


class DiagnosticsError(Exception):
    """User-facing error for diagnostics that cannot be produced."""


# ---------------------------------------------------------------------------
# Loading the trained model
# ---------------------------------------------------------------------------


def model_paths(dataset: Dataset) -> dict[str, Path]:
    """Mirror of ``apps.ml_engine.services.model_artifacts`` (read-only, no mkdir)."""
    root = project_dir(dataset) / "models"
    return {
        "dir": root,
        "model": root / "latest_model.joblib",
        "result": root / "latest_result.json",
    }


@dataclass
class TrainedModel:
    estimator: Any
    name: str
    key: str
    task_type: str
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def test_metrics(self) -> dict[str, Any]:
        return self.result.get("test_metrics") or {}


def load_trained_model(dataset: Dataset, prepared: PreparedData | None) -> tuple[TrainedModel | None, str | None]:
    """Return ``(model, warning)``.

    ``model`` is ``None`` when nothing was trained, or when the saved model is
    older than the current preprocessing (the matrices changed after training,
    so its predictions would be meaningless).
    """
    paths = model_paths(dataset)
    if not paths["model"].exists():
        return None, None

    prepared_path = artifact_paths(dataset)["prepared"]
    if prepared_path.exists() and paths["model"].stat().st_mtime < prepared_path.stat().st_mtime:
        return None, ("Le prétraitement a été relancé après le dernier entraînement : "
                      "ré-entraînez le modèle pour mettre à jour les diagnostics.")
    try:
        estimator = joblib.load(paths["model"])
    except Exception:  # noqa: BLE001 - corrupt or incompatible pickle
        return None, "Le modèle sauvegardé est illisible : ré-entraînez-le."

    result: dict[str, Any] = {}
    if paths["result"].exists():
        try:
            result = json.loads(paths["result"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}

    task_type = result.get("task_type") or (prepared.task_type if prepared else "classification")
    model = TrainedModel(
        estimator=estimator,
        name=result.get("algo_name") or type(estimator).__name__,
        key=result.get("algo_key") or "",
        task_type=task_type,
        result=result,
    )
    if prepared is not None:
        issue = compatibility_issue(model, prepared)
        if issue:
            return None, issue
    return model, None


def compatibility_issue(model: TrainedModel, prepared: PreparedData) -> str | None:
    if model.task_type != prepared.task_type:
        return (f"Le modèle a été entraîné pour « {model.task_type} » mais le prétraitement actuel "
                f"cible « {prepared.task_type} ». Ré-entraînez le modèle.")
    expected = getattr(model.estimator, "n_features_in_", None)
    if expected is not None and expected != prepared.X_train.shape[1]:
        return (f"Le modèle attend {expected} variables, le prétraitement en produit "
                f"{prepared.X_train.shape[1]}. Ré-entraînez le modèle.")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def final_estimator(estimator: Any) -> Any:
    """Last step of a Pipeline, the estimator itself otherwise."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def class_order(estimator: Any, y_true: np.ndarray, y_pred: np.ndarray) -> list[Any]:
    classes = getattr(final_estimator(estimator), "classes_", None)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    return list(classes)


def class_scores(estimator: Any, X: np.ndarray) -> np.ndarray | None:
    """Probabilities when available, decision function otherwise."""
    for method in ("predict_proba", "decision_function"):
        if hasattr(estimator, method):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return np.asarray(getattr(estimator, method)(X))
            except Exception:  # noqa: BLE001 - e.g. SVC(probability=False).predict_proba
                continue
    return None


def _feature_label(prepared: PreparedData, index: int) -> str:
    names = prepared.feature_names
    name = names[index] if index < len(names) else f"x{index}"
    scaler = (prepared.config or {}).get("scaler", "none")
    return f"{name} (normalisé)" if scaler and scaler != "none" else name


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def _linear_basis(estimator: Any):
    """Return ``(transform, linear_model)`` for models linear in their parameters."""
    from sklearn.linear_model import LinearRegression

    if hasattr(estimator, "regressor_") and hasattr(estimator, "feature_idx"):
        return (lambda x: x), estimator.regressor_
    if isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], LinearRegression):
        head = Pipeline(estimator.steps[:-1]) if len(estimator.steps) > 1 else None
        return (head.transform if head is not None else (lambda x: x)), estimator.steps[-1][1]
    if isinstance(estimator, LinearRegression):
        return (lambda x: x), estimator
    return None


def fit_curve_data(estimator: Any, prepared: PreparedData) -> dict[str, Any] | None:
    """Fitted line/curve of a single-feature regression, with its 95 % CI.

    Applies to simple linear regression (M4's ``SingleFeatureLinearRegressor``)
    and to any regressor trained on a single feature (polynomial included).
    """
    X_train, X_test = _dense(prepared.X_train), _dense(prepared.X_test)
    n_features = X_train.shape[1]
    feature_idx = getattr(estimator, "feature_idx", None)
    if feature_idx is not None:
        feature_idx = min(max(0, int(feature_idx)), n_features - 1)
    elif n_features == 1:
        feature_idx = 0
    else:
        return None

    def expand(column: np.ndarray, template: np.ndarray) -> np.ndarray:
        full = np.zeros((len(column), template.shape[1]))
        full[:, feature_idx] = column
        return full

    x_train = X_train[:, feature_idx]
    x_test = X_test[:, feature_idx]
    grid = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 200)
    fitted = np.asarray(estimator.predict(expand(grid, X_train)), dtype=float)

    lower = upper = None
    basis = _linear_basis(estimator)
    if basis is not None:
        transform, linear = basis
        add_bias = getattr(linear, "fit_intercept", True)

        def design(column: np.ndarray) -> np.ndarray:
            phi = np.asarray(transform(column.reshape(-1, 1)), dtype=float)
            return np.column_stack([np.ones(len(column)), phi]) if add_bias else phi

        phi_train = design(x_train)
        y_train = np.asarray(prepared.y_train, dtype=float)
        residuals = y_train - np.asarray(estimator.predict(X_train), dtype=float)
        dof = max(len(y_train) - phi_train.shape[1], 1)
        s2 = float(residuals @ residuals) / dof
        xtx_inv = np.linalg.pinv(phi_train.T @ phi_train)
        phi_grid = design(grid)
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", phi_grid, xtx_inv, phi_grid) * s2, 0, None))
        t_value = scipy_stats.t.ppf(0.975, dof)
        lower, upper = fitted - t_value * se, fitted + t_value * se

    return {
        "x": x_test,
        "y": np.asarray(prepared.y_test, dtype=float),
        "grid": grid,
        "fitted": fitted,
        "lower": lower,
        "upper": upper,
        "x_label": _feature_label(prepared, feature_idx),
    }


def regression_diagnostics(model: TrainedModel, prepared: PreparedData) -> dict[str, Any]:
    X_test = _dense(prepared.X_test)
    y_true = np.asarray(prepared.y_test, dtype=float)
    y_pred = np.asarray(model.estimator.predict(X_test), dtype=float)
    residuals = y_true - y_pred
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None
    target = prepared.target or "y"
    charts: list[Chart] = []

    curve = fit_curve_data(model.estimator, prepared)
    if curve is not None:
        charts.append(make_chart(
            Plots.fit_curve(curve["x"], curve["y"], curve["grid"], curve["fitted"], curve["lower"],
                            curve["upper"], curve["x_label"], target),
            "fit", "Courbe d'ajustement",
            subtitle="Points du jeu de test, modèle ŷ et intervalle de confiance à 95 % de la moyenne"
            if curve["lower"] is not None else "Points du jeu de test et prédiction du modèle",
            height=400, wide=True,
        ))

    charts.append(make_chart(
        Plots.predicted_vs_actual(y_true, y_pred, r2), "pred-actual", "Valeurs prédites vs réelles",
        subtitle="Plus les points collent à la diagonale y = x, meilleur est le modèle", height=400,
    ))
    charts.append(make_chart(
        Plots.residuals_vs_predicted(y_true, y_pred), "residuals", "Résidus vs valeurs prédites",
        subtitle="Homoscédasticité : nuage sans forme ni entonnoir autour de 0", height=400,
    ))

    normality = None
    if len(residuals) >= 20:
        sample = residuals if len(residuals) <= 5000 else np.random.default_rng(0).choice(residuals, 5000, False)
        _stat, p_value = scipy_stats.normaltest(sample)
        normality = float(p_value)
    charts.append(make_chart(
        Plots.residual_histogram(residuals), "residual-hist", "Distribution des résidus",
        subtitle=(f"Test de D'Agostino : p = {normality:.3g} — "
                  + ("compatible avec une loi normale" if normality >= 0.05 else "écart significatif à la normalité"))
        if normality is not None else "Hypothèse de normalité des erreurs", height=340,
    ))
    charts.append(make_chart(
        Plots.qq_plot(residuals), "qq", "Q-Q plot des résidus",
        subtitle="Points alignés sur la diagonale ⇒ résidus gaussiens", height=340,
    ))

    importance = attribute_importance(model, prepared)
    if importance is not None:
        charts.append(importance)

    return {
        "charts": charts,
        "summary": {
            "r2": r2,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "residual_mean": float(residuals.mean()),
            "n_test": int(len(y_true)),
        },
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def roc_pr_data(y_true: np.ndarray, scores: np.ndarray, classes: list[Any]) -> dict[str, Any]:
    """ROC and PR curves (binary, or one-vs-rest per class + micro-average)."""
    labels = [str(c) for c in classes]
    if len(classes) == 2:
        positive = classes[1]
        score = scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores.ravel()
        y_bin = (y_true == positive).astype(int)
        if y_bin.min() == y_bin.max():
            return {"roc": [], "pr": [], "baseline": None}
        fpr, tpr, _ = roc_curve(y_bin, score)
        precision, recall, _ = precision_recall_curve(y_bin, score)
        name = f"Classe positive « {labels[1]} »"
        return {
            "roc": [{"name": name, "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}],
            "pr": [{"name": name, "precision": precision, "recall": recall,
                    "ap": float(average_precision_score(y_bin, score))}],
            "baseline": float(y_bin.mean()),
        }

    if scores.ndim != 2 or scores.shape[1] != len(classes):
        return {"roc": [], "pr": [], "baseline": None}
    y_bin = label_binarize(y_true, classes=classes)
    roc, pr = [], []
    if len(classes) <= MAX_ROC_CLASSES:
        for index, label in enumerate(labels):
            if y_bin[:, index].min() == y_bin[:, index].max():
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, index], scores[:, index])
            precision, recall, _ = precision_recall_curve(y_bin[:, index], scores[:, index])
            roc.append({"name": label, "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)),
                        "color": None, "slot": index})
            pr.append({"name": label, "precision": precision, "recall": recall,
                       "ap": float(average_precision_score(y_bin[:, index], scores[:, index])), "slot": index})
    present = y_bin.sum(axis=0) > 0
    if present.any():
        fpr, tpr, _ = roc_curve(y_bin[:, present].ravel(), scores[:, present].ravel())
        roc.append({"name": "Micro-moyenne", "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)),
                    "color": "#F3EDE2", "dash": "dot"})
    # Keep each class on its own categorical slot even if a class is absent from the test split.
    for curve in roc + pr:
        if "slot" in curve:
            curve["color"] = series_color(curve["slot"])
    return {"roc": roc, "pr": pr, "baseline": None}


def classification_diagnostics(model: TrainedModel, prepared: PreparedData) -> dict[str, Any]:
    X_test = _dense(prepared.X_test)
    y_true = np.asarray(prepared.y_test)
    y_pred = np.asarray(model.estimator.predict(X_test))
    classes = class_order(model.estimator, y_true, y_pred)
    labels = [str(c) for c in classes]
    charts: list[Chart] = []

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    charts.append(make_chart(
        Plots.confusion_matrix(matrix, labels), "confusion", "Matrice de confusion",
        subtitle=f"Effectifs et % de chaque classe réelle · exactitude {accuracy * 100:.1f} %",
        height=max(340, 60 * len(labels) + 160),
    ))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    charts.append(make_chart(
        Plots.class_metrics(labels, precision, recall, f1, support), "class-metrics",
        "Précision, rappel et F1 par classe", subtitle="Repère les classes que le modèle confond",
        height=max(340, 60 * len(labels) + 160),
    ))

    scores = class_scores(model.estimator, X_test)
    curves = roc_pr_data(y_true, scores, classes) if scores is not None else {"roc": [], "pr": []}
    score_kind = "probabilités" if hasattr(model.estimator, "predict_proba") else "fonction de décision"
    if curves["roc"]:
        charts.append(make_chart(
            Plots.roc_curves(curves["roc"]), "roc", "Courbe ROC",
            subtitle=("Binaire" if len(classes) == 2 else "Un-contre-tous par classe") + f" · scores : {score_kind}",
            height=420,
        ))
    if curves["pr"]:
        charts.append(make_chart(
            Plots.pr_curves(curves["pr"], curves.get("baseline")), "pr", "Courbe Précision-Rappel",
            subtitle="Plus informative que la ROC quand les classes sont déséquilibrées", height=420,
        ))

    importance = attribute_importance(model, prepared)
    if importance is not None:
        charts.append(importance)

    return {
        "charts": charts,
        "summary": {
            "accuracy": accuracy,
            "n_test": int(len(y_true)),
            "n_errors": int((y_true != y_pred).sum()),
            "n_classes": len(labels),
            "roc_auc": next((c["auc"] for c in curves["roc"] if c["name"] == "Micro-moyenne"),
                            curves["roc"][0]["auc"] if curves["roc"] else None),
        },
        "has_scores": scores is not None,
    }


def decision_boundary(model: TrainedModel, prepared: PreparedData) -> Chart:
    """Re-fit a clone of the model on 2 PCA components and draw its class regions."""
    X_train, X_test = _dense(prepared.X_train), _dense(prepared.X_test)
    y_train, y_test = np.asarray(prepared.y_train), np.asarray(prepared.y_test)
    if X_train.shape[1] < 2:
        raise DiagnosticsError("Au moins deux variables sont nécessaires pour une frontière de décision.")

    if X_train.shape[1] == 2:
        P_train, P_test = X_train, X_test
        axis_labels = (_feature_label(prepared, 0), _feature_label(prepared, 1))
        explained_text = "Espace d'origine (2 variables)"
    else:
        pca = PCA(n_components=2, random_state=0).fit(X_train)
        P_train, P_test = pca.transform(X_train), pca.transform(X_test)
        ratio = pca.explained_variance_ratio_
        axis_labels = (f"PC1 ({ratio[0] * 100:.1f} % de variance)", f"PC2 ({ratio[1] * 100:.1f} % de variance)")
        explained_text = f"PCA : {ratio.sum() * 100:.1f} % de la variance conservée"

    if len(P_train) > BOUNDARY_TRAIN_SAMPLE:
        idx = np.random.default_rng(0).choice(len(P_train), BOUNDARY_TRAIN_SAMPLE, replace=False)
        P_fit, y_fit = P_train[idx], y_train[idx]
    else:
        P_fit, y_fit = P_train, y_train

    estimator_2d = clone(model.estimator)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator_2d.fit(P_fit, y_fit)
    except Exception as exc:  # noqa: BLE001 - e.g. MultinomialNB on negative PCA scores
        raise DiagnosticsError(f"Ce modèle ne peut pas être ré-entraîné sur la projection 2D : {exc}") from exc

    pad_x = (P_test[:, 0].max() - P_test[:, 0].min()) * 0.08 or 1.0
    pad_y = (P_test[:, 1].max() - P_test[:, 1].min()) * 0.08 or 1.0
    both = np.vstack([P_train, P_test])
    xs = np.linspace(both[:, 0].min() - pad_x, both[:, 0].max() + pad_x, BOUNDARY_GRID)
    ys = np.linspace(both[:, 1].min() - pad_y, both[:, 1].max() + pad_y, BOUNDARY_GRID)
    grid_x, grid_y = np.meshgrid(xs, ys)
    zone_labels = estimator_2d.predict(np.column_stack([grid_x.ravel(), grid_y.ravel()]))

    classes = class_order(model.estimator, y_test, y_test)
    index_of = {c: i for i, c in enumerate(classes)}
    zone = np.array([index_of.get(v, -1) for v in zone_labels], dtype=float).reshape(grid_x.shape)
    point_classes = np.array([index_of.get(v, -1) for v in y_test])
    misclassified = np.asarray(model.estimator.predict(X_test)) != y_test
    accuracy_2d = float((estimator_2d.predict(P_test) == y_test).mean())

    return make_chart(
        Plots.decision_boundary(xs, ys, zone, P_test, point_classes, misclassified,
                                [str(c) for c in classes], axis_labels),
        "boundary", "Frontière de décision (projection 2D)",
        subtitle=f"{explained_text} · clone du modèle ré-entraîné en 2D (exactitude 2D : {accuracy_2d * 100:.1f} %)",
        height=480, wide=True,
        note="Approximation pédagogique : le vrai modèle décide dans l'espace complet ; "
             "les points cerclés sont les erreurs du vrai modèle.",
    )


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


def _top(names: list[str], values: np.ndarray, errors: np.ndarray | None = None, by_abs: bool = True):
    order = np.argsort(-np.abs(values) if by_abs else -values)[:TOP_FEATURES]
    return ([names[i] for i in order], values[order], errors[order] if errors is not None else None)


def importance_data(model: TrainedModel, prepared: PreparedData) -> dict[str, Any] | None:
    """Importance read from the fitted model (trees, forests, linear models).

    Shared by the interactive chart and the static PDF figure.
    """
    estimator = model.estimator
    final = final_estimator(estimator)
    names = list(prepared.feature_names)

    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
        if len(values) != len(names):
            return None
        top_names, top_values, _ = _top(names, values, by_abs=False)
        return {
            "names": top_names, "values": top_values, "signed": False,
            "axis": "Importance (réduction moyenne d'impureté)", "title": "Importance des variables",
            "subtitle": f"Importance de Gini (MDI) · {min(TOP_FEATURES, len(names))} premières variables",
        }

    linear = getattr(estimator, "regressor_", None) or final
    coef = getattr(linear, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=float)
    if isinstance(estimator, Pipeline) and len(estimator.steps) > 1:
        try:
            names = [str(n) for n in Pipeline(estimator.steps[:-1]).get_feature_names_out(names)]
        except Exception:  # noqa: BLE001
            names = [f"terme_{i}" for i in range(coef.shape[-1])]
    if hasattr(estimator, "feature_idx"):
        return None  # one coefficient: the fit curve already shows it.
    if coef.shape[-1] != len(names):
        return None

    if coef.ndim == 2 and coef.shape[0] > 1:
        top_names, top_values, _ = _top(names, np.abs(coef).mean(axis=0), by_abs=False)
        return {
            "names": top_names, "values": top_values, "signed": False,
            "axis": "|coefficient| moyen sur les classes", "title": "Poids des variables",
            "subtitle": "Moyenne des coefficients absolus (variables normalisées, modèle multi-classe)",
        }
    top_names, top_values, _ = _top(names, coef.ravel())
    return {
        "names": top_names, "values": top_values, "signed": True,
        "axis": "Coefficient", "title": "Coefficients du modèle",
        "subtitle": "Triés par |β| · rouge = effet positif, bleu = effet négatif",
    }


def attribute_importance(model: TrainedModel, prepared: PreparedData) -> Chart | None:
    data = importance_data(model, prepared)
    if data is None:
        return None
    return make_chart(
        Plots.feature_importance(data["names"], data["values"], data["axis"], signed=data["signed"]),
        "importance", data["title"], subtitle=data["subtitle"],
        height=max(260, 26 * len(data["names"]) + 90),
    )


def has_attribute_importance(model: TrainedModel) -> bool:
    final = final_estimator(model.estimator)
    if hasattr(model.estimator, "feature_idx"):
        return True
    return hasattr(final, "feature_importances_") or hasattr(final, "coef_")


def permutation_chart(model: TrainedModel, prepared: PreparedData) -> Chart:
    """Model-agnostic importance (SVM RBF, KNN, Naive Bayes, MLP…)."""
    X_test = _dense(prepared.X_test)
    y_test = np.asarray(prepared.y_test)
    if len(X_test) > PERMUTATION_SAMPLE:
        idx = np.random.default_rng(0).choice(len(X_test), PERMUTATION_SAMPLE, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(model.estimator, X_test, y_test, n_repeats=5, random_state=0)
    names = list(prepared.feature_names)
    top_names, top_values, top_errors = _top(names, result.importances_mean, result.importances_std, by_abs=False)
    metric = "exactitude" if prepared.task_type == "classification" else "R²"
    return make_chart(
        Plots.feature_importance(top_names, top_values, f"Baisse de {metric} quand la variable est permutée",
                                 errors=top_errors),
        "permutation", "Importance par permutation",
        subtitle="Méthode agnostique au modèle · 5 permutations sur le jeu de test (± écart-type)",
        height=max(260, 26 * len(top_names) + 90),
    )


def loss_chart(model: TrainedModel) -> Chart | None:
    final = final_estimator(model.estimator)
    loss = getattr(final, "loss_curve_", None)
    if loss is None or not len(loss):
        return None
    validation = getattr(final, "validation_scores_", None)
    n_iter = getattr(final, "n_iter_", len(loss))
    max_iter = getattr(final, "max_iter", None)
    converged = max_iter is None or n_iter < max_iter
    return make_chart(
        Plots.loss_curve(list(loss), list(validation) if validation is not None else None),
        "loss", "Convergence du réseau de neurones",
        subtitle=f"{n_iter} époque(s) · perte finale {loss[-1]:.4g} · "
                 + ("convergé" if converged else "max_iter atteint : augmentez le nombre d'époques"),
        height=420 if validation is not None else 340, wide=True,
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def elbow_point(ks: list[int], inertia: list[float]) -> int | None:
    """Kneedle-style elbow: max distance to the chord between the first and last points."""
    if len(ks) < 3:
        return None
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)
    x_n = (x - x.min()) / (np.ptp(x) or 1)
    y_n = (y - y.min()) / (np.ptp(y) or 1)
    distances = np.abs((y_n[-1] - y_n[0]) * x_n - (x_n[-1] - x_n[0]) * y_n + x_n[-1] * y_n[0] - y_n[-1] * x_n[0])
    return int(x[int(np.argmax(distances))])


def elbow_data(X: np.ndarray) -> dict[str, Any]:
    ks = [k for k in ELBOW_K_RANGE if k < len(X)]
    inertia, silhouette = [], []
    for k in ks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        inertia.append(float(km.inertia_))
        if len(np.unique(km.labels_)) > 1:
            sample = min(SILHOUETTE_SAMPLE, len(X))
            silhouette.append(float(silhouette_score(X, km.labels_, sample_size=sample, random_state=0)))
        else:
            silhouette.append(None)
    valid = [(k, s) for k, s in zip(ks, silhouette) if s is not None]
    return {
        "ks": ks,
        "inertia": inertia,
        "silhouette": silhouette,
        "elbow_k": elbow_point(ks, inertia),
        "best_silhouette_k": max(valid, key=lambda item: item[1])[0] if valid else None,
    }


def elbow_chart(prepared: PreparedData, chosen_k: int | None = None) -> tuple[Chart, dict[str, Any]]:
    X = _dense(prepared.X_train)
    if len(X) < 4:
        raise DiagnosticsError("Trop peu d'observations pour la méthode du coude.")
    data = elbow_data(X)
    chart = make_chart(
        Plots.elbow(data["ks"], data["inertia"], data["silhouette"], data["elbow_k"],
                    data["best_silhouette_k"], chosen_k),
        "elbow", "Méthode du coude & score de silhouette",
        subtitle=f"K-Means pour k = {data['ks'][0]}…{data['ks'][-1]} · coude suggéré : k = {data['elbow_k']} · "
                 f"meilleure silhouette : k = {data['best_silhouette_k']}",
        height=460, wide=True,
    )
    return chart, data


def clustering_diagnostics(model: TrainedModel, prepared: PreparedData) -> dict[str, Any]:
    X = _dense(prepared.X_train)
    estimator = final_estimator(model.estimator)
    labels = getattr(estimator, "labels_", None)
    if labels is None or len(labels) != len(X):
        labels = model.estimator.predict(X)
    labels = np.asarray(labels)
    charts: list[Chart] = []
    summary: dict[str, Any] = {"n_clusters": int(len(np.unique(labels))), "n_samples": int(len(X))}

    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=0).fit(X)
        ratio = pca.explained_variance_ratio_
        centers = getattr(estimator, "cluster_centers_", None)
        if centers is None:
            centers = np.vstack([X[labels == c].mean(axis=0) for c in sorted(np.unique(labels))])
        charts.append(make_chart(
            Plots.clusters_2d(pca.transform(X), labels, pca.transform(centers),
                              (f"PC1 ({ratio[0] * 100:.1f} %)", f"PC2 ({ratio[1] * 100:.1f} %)")),
            "clusters", "Clusters projetés sur les 2 premiers axes PCA",
            subtitle=f"{ratio.sum() * 100:.1f} % de la variance représentée · ✕ = centroïdes finaux",
            height=460,
        ))

    if summary["n_clusters"] > 1:
        if len(X) > SILHOUETTE_SAMPLE:
            idx = np.random.default_rng(0).choice(len(X), SILHOUETTE_SAMPLE, replace=False)
            X_s, labels_s = X[idx], labels[idx]
        else:
            X_s, labels_s = X, labels
        values = silhouette_samples(X_s, labels_s)
        average = float(values.mean())
        summary["silhouette"] = average
        summary["negative_share"] = float((values < 0).mean())
        charts.append(make_chart(
            Plots.silhouette_plot(values, labels_s, average), "silhouette", "Silhouette par observation",
            subtitle=f"Silhouette moyenne {average:.3f} · {summary['negative_share'] * 100:.1f} % "
                     "d'observations probablement mal affectées (s < 0)",
            height=460,
        ))
    return {"charts": charts, "summary": summary}


# ---------------------------------------------------------------------------
# Model comparison (every algorithm of the ML engine leaderboard)
# ---------------------------------------------------------------------------

# Fixed order = fixed colour per algorithm, whatever subset was trained.
COMPARISON_ORDER: dict[str, list[str]] = {
    "classification": ["logistic_regression", "decision_tree", "random_forest", "svm",
                       "naive_bayes", "knn", "neural_network"],
    "regression": ["simple_linear_regression", "multiple_linear_regression", "polynomial_regression",
                   "decision_tree", "random_forest", "svm", "knn", "neural_network"],
}
LINE_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot", "solid", "dash")


def _leaderboard(dataset: Dataset) -> list[dict[str, Any]]:
    path = model_paths(dataset)["dir"] / "leaderboard.json"
    if not path.exists():
        return []
    try:
        board = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return board if isinstance(board, list) else []


def _algo_model_file(dataset: Dataset, algo_key: str) -> Path:
    """Mirror of ``apps.ml_engine.services.algo_model_path`` (read-only, no mkdir)."""
    return model_paths(dataset)["dir"] / f"model_{algo_key}.joblib"


def _fresh(path: Path, dataset: Dataset) -> bool:
    prepared_path = artifact_paths(dataset)["prepared"]
    return path.exists() and (not prepared_path.exists() or path.stat().st_mtime >= prepared_path.stat().st_mtime)


def comparison_candidates(dataset: Dataset, prepared: PreparedData) -> int:
    """Number of leaderboard models that can be compared (cheap: no unpickling)."""
    order = COMPARISON_ORDER.get(prepared.task_type, [])
    keys = {entry.get("algo_key") for entry in _leaderboard(dataset) if entry.get("task_type") == prepared.task_type}
    return sum(1 for key in keys if key in order and _fresh(_algo_model_file(dataset, key), dataset))


def comparable_models(dataset: Dataset, prepared: PreparedData) -> list[tuple[int, TrainedModel]]:
    """``(colour slot, model)`` for every up-to-date leaderboard model of the current task."""
    order = COMPARISON_ORDER.get(prepared.task_type, [])
    models: list[tuple[int, TrainedModel]] = []
    for entry in _leaderboard(dataset):
        key = entry.get("algo_key")
        if entry.get("task_type") != prepared.task_type or key not in order:
            continue
        path = _algo_model_file(dataset, key)
        if not _fresh(path, dataset):
            continue
        try:
            estimator = joblib.load(path)
        except Exception:  # noqa: BLE001
            continue
        model = TrainedModel(estimator, entry.get("algo_name") or key, key, prepared.task_type, entry)
        if compatibility_issue(model, prepared) is None:
            models.append((order.index(key), model))
    return sorted(models, key=lambda item: item[0])


def comparison_curves(models: list[tuple[int, TrainedModel]], prepared: PreparedData) -> dict[str, Any]:
    """ROC / PR curve of each classifier on the same test split.

    Binary: curve of the positive class. Multi-class: micro-average, so that
    each model is summarised by a single curve.
    """
    X_test = _dense(prepared.X_test)
    y_test = np.asarray(prepared.y_test)
    roc, pr = [], []
    classes = None
    for slot, model in models:
        scores = class_scores(model.estimator, X_test)
        if scores is None:
            continue
        classes = class_order(model.estimator, y_test, y_test)
        style = {"color": series_color(slot), "dash": LINE_DASHES[slot % len(LINE_DASHES)], "slot": slot}
        if len(classes) == 2:
            score = scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores.ravel()
            y_bin, s = (y_test == classes[1]).astype(int), score
        else:
            if scores.ndim != 2 or scores.shape[1] != len(classes):
                continue
            y_bin, s = label_binarize(y_test, classes=classes).ravel(), scores.ravel()
        if y_bin.min() == y_bin.max():
            continue
        fpr, tpr, _ = roc_curve(y_bin, s)
        precision, recall, _ = precision_recall_curve(y_bin, s)
        roc.append({"name": model.name, "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)), **style})
        pr.append({"name": model.name, "precision": precision, "recall": recall,
                   "ap": float(average_precision_score(y_bin, s)), **style})
    baseline = None
    if classes is not None:
        baseline = float(np.mean(y_test == classes[1])) if len(classes) == 2 else 1 / len(classes)
    return {"roc": roc, "pr": pr, "baseline": baseline, "binary": classes is not None and len(classes) == 2}


def regression_errors(models: list[tuple[int, TrainedModel]], prepared: PreparedData) -> list[dict[str, Any]]:
    """Absolute test errors of each regressor, best (lowest median) first."""
    X_test = _dense(prepared.X_test)
    y_test = np.asarray(prepared.y_test, dtype=float)
    rows = []
    for slot, model in models:
        predicted = np.asarray(model.estimator.predict(X_test), dtype=float)
        errors = np.abs(y_test - predicted)
        rows.append({"name": model.name, "errors": errors, "color": series_color(slot),
                     "median": float(np.median(errors)), "r2": float(r2_score(y_test, predicted))})
    return sorted(rows, key=lambda row: row["median"])


def comparison_charts(dataset: Dataset, prepared: PreparedData) -> list[Chart]:
    models = comparable_models(dataset, prepared)
    if len(models) < 2:
        raise DiagnosticsError("Entraînez au moins deux algorithmes à l'étape 4 pour les comparer.")
    names = ", ".join(model.name for _slot, model in models)
    if prepared.task_type == "classification":
        curves = comparison_curves(models, prepared)
        kind = "classe positive" if curves["binary"] else "micro-moyenne sur les classes"
        return [
            make_chart(Plots.roc_curves(curves["roc"]), "compare-roc", "Comparaison des courbes ROC",
                       subtitle=f"{len(curves['roc'])} modèles sur le même jeu de test · {kind}", height=440),
            make_chart(Plots.pr_curves(curves["pr"], curves["baseline"]), "compare-pr",
                       "Comparaison des courbes Précision-Rappel",
                       subtitle=f"Average Precision de chaque modèle · {kind}", height=440),
        ]
    rows = regression_errors(models, prepared)
    return [make_chart(
        Plots.error_boxplot([f"{r['name']} · R² = {r['r2']:.3f}" for r in rows], [r["errors"] for r in rows],
                            [r["color"] for r in rows]),
        "compare-errors", "Comparaison des erreurs absolues |y − ŷ|",
        subtitle=f"Jeu de test · du modèle le plus précis (médiane la plus basse) au moins précis · {names}",
        height=max(300, 56 * len(rows) + 120), wide=True,
    )]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_diagnostics(model: TrainedModel, prepared: PreparedData) -> dict[str, Any]:
    """Every fast diagnostic for the model's task (heavy ones are lazy-loaded)."""
    if model.task_type == "regression":
        result = regression_diagnostics(model, prepared)
    elif model.task_type == "classification":
        result = classification_diagnostics(model, prepared)
    else:
        result = clustering_diagnostics(model, prepared)

    loss = loss_chart(model)
    if loss is not None:
        result["charts"].append(loss)
    result["needs_permutation"] = model.task_type != "clustering" and not has_attribute_importance(model)
    result["can_draw_boundary"] = model.task_type == "classification" and prepared.X_train.shape[1] >= 2
    return result
