"""
Views for the preprocessing module (step 3 of the pipeline).

The page is a single server-rendered form; submitting it runs the pipeline,
persists the artefacts and re-renders the page with the result summary.  No
client-side state is required, so the form also works without JavaScript.
"""

from __future__ import annotations

from django.contrib import messages
from django.http import FileResponse, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from apps.datasets.services import DatasetError, load_dataframe, profile_dataframe
from apps.datasets.state import (
    clear_active_dataset,
    get_active_dataset,
    update_state,
)
from apps.preprocessing import services
from apps.preprocessing.config import (
    MISSING_THRESHOLDS,
    TEST_SIZES,
    PreprocessingConfig,
    PreprocessingError,
)
from apps.preprocessing.pipeline import infer_task_type

NUMERIC_OPTIONS = [
    ("mean", "Moyenne", "Remplace par la moyenne de la colonne (calculée sur le train)."),
    ("median", "Médiane", "Plus robuste que la moyenne en présence d'outliers."),
    ("most_frequent", "Mode", "Valeur la plus fréquente."),
    ("constant", "Constante", "Remplace par 0 — utile pour signaler l'absence."),
    ("knn", "KNN Imputer", "Bonus : imputation par les k plus proches voisins."),
    ("drop_rows", "Supprimer les lignes", "Aucune imputation : les lignes incomplètes sont retirées."),
]

CATEGORICAL_OPTIONS = [
    ("most_frequent", "Mode", "Remplace par la modalité la plus fréquente."),
    ("constant", "Constante", "Remplace par la valeur texte saisie ci-dessous."),
    ("drop_rows", "Supprimer les lignes", "Retire les lignes contenant une modalité manquante."),
]

SCALER_OPTIONS = [
    ("standard", "StandardScaler", "Z-score : moyenne 0, écart-type 1. Recommandé pour SVM, KNN, MLP."),
    ("minmax", "MinMaxScaler", "Ramène dans [0, 1]. Adapté aux distributions non gaussiennes."),
    ("robust", "RobustScaler", "Basé sur médiane et IQR : résistant aux outliers."),
    ("none", "Aucun", "Les valeurs restent dans leur échelle d'origine."),
]

ENCODER_OPTIONS = [
    ("onehot", "One-Hot Encoding", "Une colonne binaire par modalité (pas d'ordre implicite)."),
    ("ordinal", "Ordinal Encoding", "Une seule colonne d'entiers — pour les catégories ordonnées."),
]

OUTLIER_OPTIONS = [
    ("none", "Conserver", "Aucun traitement des valeurs aberrantes."),
    ("clip", "Écrêter (IQR)", "Ramène les valeurs aux bornes Q1−1.5·IQR et Q3+1.5·IQR."),
    ("drop", "Supprimer les lignes", "Retire du train et du test les lignes hors bornes IQR."),
]

TASK_OPTIONS = [
    ("classification", "Classification", "Cible catégorielle (classes à prédire)."),
    ("regression", "Régression", "Cible numérique continue."),
    ("clustering", "Clustering", "Aucune cible : segmentation non supervisée (K-Means)."),
]


def _context_for(dataset, frame) -> dict:
    config = services.load_config(dataset) or services.default_config(dataset, frame)
    valid = set(frame.columns)
    if config.features:
        config.features = [feature for feature in config.features if feature in valid]
    if config.target is not None and config.target not in valid:
        config.target = None
    if config.target is None:
        config.task_type = "clustering"
    elif config.task_type not in ("classification", "regression"):
        config.task_type = infer_task_type(frame[config.target])

    _selected, reasons = services.suggested_feature_selection(dataset, config.target)
    prepared = services.load_prepared(dataset)
    options = services.feature_options(dataset, config, reasons)

    return {
        "dataset": dataset,
        "profile": profile_dataframe(frame),
        "config": config,
        "options": options,
        "prepared": prepared.to_summary() if prepared else None,
        "task_map": services.task_map(frame),
        "numeric_options": NUMERIC_OPTIONS,
        "categorical_options": CATEGORICAL_OPTIONS,
        "scaler_options": SCALER_OPTIONS,
        "encoder_options": ENCODER_OPTIONS,
        "outlier_options": OUTLIER_OPTIONS,
        "task_options": TASK_OPTIONS,
        "test_size_options": [(str(size), f"{int((1 - size) * 100)} / {int(size * 100)}") for size in TEST_SIZES],
        "threshold_options": [
            (str(value), "Garder toutes les colonnes" if value == 1.0 else f"Supprimer > {int(value * 100)}% manquant")
            for value in MISSING_THRESHOLDS
        ],
        "selected_count": sum(1 for option in options if option["selected"]),
    }


def index_view(request: HttpRequest) -> HttpResponse:
    """Configuration form + summary of the last preprocessing run."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return render(
            request,
            "preprocessing/index.html",
            {"title": "Prétraitement & Feature Engineering", "dataset": None},
        )

    try:
        frame = load_dataframe(dataset)
    except DatasetError as exc:
        messages.error(request, str(exc))
        clear_active_dataset(request)
        return redirect("datasets:index")

    context = {"title": "Prétraitement & Feature Engineering"}
    context.update(_context_for(dataset, frame))
    return render(request, "preprocessing/index.html", context)


@require_POST
def configure_view(request: HttpRequest) -> HttpResponse:
    """Validate the form, fit the pipeline, persist it and update the session."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Chargez d'abord un jeu de données.")
        return redirect("datasets:index")

    try:
        frame = load_dataframe(dataset)
    except DatasetError as exc:
        messages.error(request, str(exc))
        clear_active_dataset(request)
        return redirect("datasets:index")

    try:
        config = PreprocessingConfig.from_form(request.POST, list(frame.columns))
        prepared = services.prepare(dataset, config)
    except (PreprocessingError, DatasetError) as exc:
        messages.error(request, str(exc))
        return redirect("preprocessing:index")

    update_state(
        request,
        target_column=prepared.target,
        task_type=prepared.task_type,
        preprocessing_done=True,
        model_trained=False,
        selected_model_name=None,
        best_score=None,
        active_step=3,
    )
    messages.success(
        request,
        f"Pipeline ajusté : {prepared.train_rows} lignes d'entraînement, "
        f"{prepared.test_rows} de test, {prepared.n_features_out} variables en sortie.",
    )
    return redirect("preprocessing:index")


@require_POST
def reset_view(request: HttpRequest) -> HttpResponse:
    """Delete the artefacts and mark preprocessing as not done."""
    dataset = get_active_dataset(request)
    if dataset is None:
        return redirect("datasets:index")

    services.clear_artifacts(dataset)
    update_state(
        request,
        preprocessing_done=False,
        model_trained=False,
        selected_model_name=None,
        best_score=None,
        active_step=3,
    )
    messages.success(request, "Les artefacts de prétraitement ont été supprimés.")
    return redirect("preprocessing:index")


def export_csv_view(request: HttpRequest) -> HttpResponse:
    """Download the cleaned dataframe as CSV."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Chargez d'abord un jeu de données.")
        return redirect("datasets:index")

    try:
        filename, payload = services.export_clean_csv(dataset)
    except (PreprocessingError, DatasetError) as exc:
        messages.error(request, str(exc))
        return redirect("preprocessing:index")

    response = HttpResponse(payload, content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def download_pipeline_view(request: HttpRequest) -> HttpResponse:
    """Download the serialised scikit-learn preprocessing pipeline."""
    dataset = get_active_dataset(request)
    if dataset is None:
        messages.error(request, "Chargez d'abord un jeu de données.")
        return redirect("datasets:index")

    path = services.artifact_paths(dataset)["pipeline"]
    if not path.exists():
        messages.error(request, "Aucun pipeline entraîné : appliquez d'abord un prétraitement.")
        return redirect("preprocessing:index")

    return FileResponse(
        path.open("rb"),
        as_attachment=True,
        filename=f"preprocessing_{dataset.id}.joblib",
        content_type="application/octet-stream",
    )
