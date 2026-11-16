"""
Leak-free preprocessing engine.

The single rule this module enforces is: **nothing is fitted on the test set**.
Imputation, encoding, scaling and IQR outlier bounds are all computed on
``X_train`` only.  This is why imputation lives inside the ``ColumnTransformer``
instead of being applied to the whole dataframe beforehand — fitting a mean on
the full dataset would leak test information into the training folds and
inflate the scores.

``DataPreprocessor.build_and_fit_transform`` is the entry point used by
``apps.preprocessing.services``; it returns a ``PreparedData`` bundle that the
ML engine consumes directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from apps.preprocessing.config import PreprocessingConfig, PreprocessingError

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@dataclass
class PreparedData:
    """Result of a preprocessing run: matrices, fitted pipeline and metadata."""

    X_train: Any
    X_test: Any | None
    y_train: Any | None
    y_test: Any | None
    pipeline: ColumnTransformer
    feature_names: list[str]
    raw_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    task_type: str
    target: str | None
    class_labels: list[str]
    class_distribution: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    df_clean: pd.DataFrame | None = None
    train_rows: int = 0
    test_rows: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    dropped_rows: int = 0

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names)

    def to_summary(self) -> dict[str, Any]:
        """JSON-safe summary used by templates and the session."""
        return {
            "task_type": self.task_type,
            "target": self.target,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "raw_feature_count": len(self.raw_features),
            "feature_count": self.n_features_out,
            "numeric_count": len(self.numeric_features),
            "categorical_count": len(self.categorical_features),
            "class_labels": self.class_labels,
            "class_distribution": self.class_distribution,
            "warnings": self.warnings,
            "dropped_columns": self.dropped_columns,
            "dropped_rows": self.dropped_rows,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_task_type(series: pd.Series | None) -> str:
    """Infer the learning task from the target column.

    A numeric target with few distinct values (Iris/Wine labels are integers) is
    a classification problem, not a regression one — hence the cardinality rule
    instead of a dtype-only check.
    """
    if series is None:
        return "clustering"
    if pd.api.types.is_bool_dtype(series):
        return "classification"
    if pd.api.types.is_numeric_dtype(series):
        distinct = int(series.nunique(dropna=True))
        return "classification" if 0 < distinct <= 10 else "regression"
    return "classification"


def _make_scaler(name: str):
    factory = _SCALERS.get(name)
    return factory() if factory else None


def build_transformer(
    config: PreprocessingConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    n_train: int,
) -> ColumnTransformer:
    """Assemble the fitted-on-train ``ColumnTransformer``."""
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        steps: list[tuple[str, Any]] = []
        if config.numeric_strategy == "knn":
            scaler = _make_scaler(config.scaler)
            if scaler is not None:
                steps.append(("scaler", scaler))
            neighbours = max(1, min(5, max(n_train - 1, 1)))
            steps.append(("imputer", KNNImputer(n_neighbors=neighbours)))
        else:
            strategy = (
                config.numeric_strategy
                if config.numeric_strategy in ("mean", "median", "most_frequent", "constant")
                else "mean"
            )
            steps.append(("imputer", SimpleImputer(strategy=strategy, fill_value=0)))
            scaler = _make_scaler(config.scaler)
            if scaler is not None:
                steps.append(("scaler", scaler))
        transformers.append(("num", Pipeline(steps), numeric_features))

    if categorical_features:
        cat_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                    if config.categorical_strategy == "most_frequent"
                    else "constant",
                    fill_value=config.constant_value,
                ),
            )
        ]
        if config.encoder == "ordinal":
            cat_steps.append(
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            )
        else:
            cat_steps.append(
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        min_frequency=None,
                    ),
                )
            )
        transformers.append(("cat", Pipeline(cat_steps), categorical_features))

    if not transformers:
        raise PreprocessingError(
            "Aucune variable exploitable : sélectionnez au moins une colonne numérique ou catégorielle."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def iqr_bounds(
    frame: pd.DataFrame, columns: list[str], factor: float = 1.5
) -> dict[str, tuple[float, float]]:
    """Compute IQR fences on the training split only."""
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        spread = q3 - q1
        if not np.isfinite(spread) or spread <= 0:
            continue
        bounds[column] = (float(q1 - factor * spread), float(q3 + factor * spread))
    return bounds


def clip_frame(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    clipped = frame.copy()
    for column, (lower, upper) in bounds.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower, upper)
    return clipped


def outlier_mask(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.Series:
    """Keep rows that fall inside every IQR fence (missing values are kept)."""
    mask = pd.Series(True, index=frame.index)
    for column, (lower, upper) in bounds.items():
        if column not in frame.columns:
            continue
        values = frame[column]
        mask &= values.between(lower, upper) | values.isna()
    return mask


def _class_summary(y: pd.Series) -> tuple[list[str], list[dict[str, Any]]]:
    counts = y.value_counts(dropna=True)
    total = int(counts.sum()) or 1
    labels = [str(label) for label in counts.index]
    distribution = [
        {"label": str(label), "count": int(count), "pct": round(int(count) / total * 100, 2)}
        for label, count in counts.items()
    ]
    return labels, distribution


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """Fit imputation + encoding + scaling and split the dataset.

    Parameters
    ----------
    df:
        Raw dataframe as loaded from the dataset file.
    target_col:
        Target column name, or ``None`` for unsupervised (clustering) usage.
    """

    def __init__(self, df: pd.DataFrame, target_col: str | None = None) -> None:
        self.raw_df = df.copy()
        self.df = df.copy()
        self.target_col = target_col
        self.pipeline: ColumnTransformer | None = None
        self.feature_names_out: list[str] = []

    # -- main entry point ---------------------------------------------------
    def build_and_fit_transform(self, config: PreprocessingConfig) -> PreparedData:
        frame = self.df.copy()
        notes: list[str] = []
        dropped_columns: list[str] = []
        dropped_rows = 0

        if config.drop_duplicates:
            before = len(frame)
            frame = frame.drop_duplicates().reset_index(drop=True)
            removed = before - len(frame)
            if removed:
                notes.append(f"{removed} ligne(s) dupliquée(s) supprimée(s).")

        target = config.target
        if target is not None and target not in frame.columns:
            raise PreprocessingError(f"La colonne cible « {target} » est introuvable.")

        # 1. Drop unusable columns: fully empty ones, plus those above the threshold.
        ratios = frame.isna().mean()
        dropped_columns = [
            column
            for column in frame.columns
            if column != target
            and (ratios[column] >= 1.0 or ratios[column] > config.missing_threshold)
        ]
        if dropped_columns:
            frame = frame.drop(columns=dropped_columns)
            notes.append(
                f"{len(dropped_columns)} colonne(s) supprimée(s) : vides ou au-delà du seuil de "
                f"{int(config.missing_threshold * 100)}% de valeurs manquantes."
            )

        # 2. Feature selection (explicit from UI, otherwise everything left).
        features = [f for f in (config.features or list(frame.columns)) if f in frame.columns]
        features = [f for f in features if f != target]
        if not features:
            raise PreprocessingError(
                "Aucune variable explicative valide après nettoyage : ajustez la sélection."
            )

        X = frame[features].copy()
        y = frame[target].copy() if target is not None else None

        # 3. Datetimes cannot be one-hot encoded directly: normalise to strings.
        datetime_columns = [
            column for column in X.columns if pd.api.types.is_datetime64_any_dtype(X[column])
        ]
        for column in datetime_columns:
            X[column] = X[column].astype(str)
        if datetime_columns:
            notes.append(
                f"{len(datetime_columns)} colonne(s) date convertie(s) en texte : "
                f"{', '.join(datetime_columns)}."
            )

        # 4. Rows without a target cannot be learned from.
        if y is not None:
            mask = y.notna()
            if not mask.all():
                removed = int((~mask).sum())
                X = X[mask]
                y = y[mask]
                notes.append(f"{removed} ligne(s) sans valeur cible supprimée(s).")
            if X.empty:
                raise PreprocessingError(
                    "Plus aucune ligne après suppression des cibles manquantes."
                )

        # 5. Explicit "drop incomplete rows" strategy.
        if "drop_rows" in (config.numeric_strategy, config.categorical_strategy):
            before = len(X)
            complete = X.notna().all(axis=1)
            X = X[complete]
            if y is not None:
                y = y[complete]
            dropped_rows = before - len(X)
            if dropped_rows:
                notes.append(f"{dropped_rows} ligne(s) incomplète(s) supprimée(s).")
            if X.empty:
                raise PreprocessingError(
                    "Toutes les lignes ont été supprimées : choisissez une stratégie d'imputation."
                )

        # 6. Train/test split (stratified when it is meaningful).
        stratify_vector = None
        if y is not None and config.task_type == "classification" and config.stratify:
            counts = y.value_counts()
            enough_rows = int(len(y) * config.test_size) >= int(y.nunique())
            if len(counts) > 1 and counts.min() >= 2 and enough_rows:
                stratify_vector = y
            else:
                notes.append(
                    "Stratification ignorée : classes trop petites pour le découpage demandé."
                )

        if y is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=stratify_vector,
                )
            except ValueError as exc:
                if stratify_vector is None:
                    raise PreprocessingError(f"Découpage train/test impossible : {exc}") from exc
                notes.append(f"Stratification ignorée ({exc}), découpage aléatoire appliqué.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                )
        else:
            X_train, X_test, y_train, y_test = X, None, None, None

        # 7. Feature families (computed before transformation).
        numeric_features = [
            column for column in X_train.columns if pd.api.types.is_numeric_dtype(X_train[column])
        ]
        categorical_features = [
            column for column in X_train.columns if column not in numeric_features
        ]

        # 8. IQR outliers: bounds from train, applied to both splits.
        if config.outlier_method != "none" and numeric_features:
            bounds = iqr_bounds(X_train, numeric_features, config.outlier_factor)
            if bounds:
                if config.outlier_method == "clip":
                    X_train = clip_frame(X_train, bounds)
                    if X_test is not None:
                        X_test = clip_frame(X_test, bounds)
                    notes.append(
                        f"Outliers écrêtés sur {len(bounds)} colonne(s) (bornes IQR × "
                        f"{config.outlier_factor} calculées sur le train)."
                    )
                else:
                    keep_train = outlier_mask(X_train, bounds)
                    removed_train = int((~keep_train).sum())
                    X_train = X_train[keep_train]
                    y_train = y_train[keep_train] if y_train is not None else None
                    removed_test = 0
                    if X_test is not None:
                        keep_test = outlier_mask(X_test, bounds)
                        removed_test = int((~keep_test).sum())
                        X_test = X_test[keep_test]
                        y_test = y_test[keep_test] if y_test is not None else None
                    notes.append(
                        f"{removed_train} ligne(s) aberrante(s) retirée(s) du train, "
                        f"{removed_test} du test (bornes IQR × {config.outlier_factor})."
                    )
                    if len(X_train) < 10:
                        raise PreprocessingError(
                            "Le filtre IQR supprime trop de lignes : augmentez le facteur "
                            "ou choisissez l'écrêtage."
                        )

        if len(X_train) == 0:
            raise PreprocessingError("Le jeu d'entraînement est vide après nettoyage.")

        # 9. Fit on train, transform both — the leak-free part.
        transformer = build_transformer(
            config, numeric_features, categorical_features, n_train=len(X_train)
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                X_train_matrix = transformer.fit_transform(X_train)
                X_test_matrix = (
                    transformer.transform(X_test) if X_test is not None else None
                )
        except PreprocessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing message
            raise PreprocessingError(f"Échec de l'application du pipeline : {exc}") from exc

        self.pipeline = transformer
        self.feature_names_out = [str(name) for name in transformer.get_feature_names_out()]

        labels, distribution = (
            _class_summary(y) if y is not None and config.task_type == "classification" else ([], [])
        )

        return PreparedData(
            X_train=X_train_matrix,
            X_test=X_test_matrix,
            y_train=y_train,
            y_test=y_test,
            pipeline=transformer,
            feature_names=self.feature_names_out,
            raw_features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            task_type=config.task_type,
            target=target,
            class_labels=labels,
            class_distribution=distribution,
            warnings=notes,
            config=config.to_dict(),
            df_clean=frame.loc[X.index].copy(),
            train_rows=int(X_train_matrix.shape[0]),
            test_rows=int(X_test_matrix.shape[0]) if X_test_matrix is not None else 0,
            dropped_columns=dropped_columns,
            dropped_rows=dropped_rows,
        )

    # -- persistence --------------------------------------------------------
    def save_pipeline(self, filepath: str) -> None:
        """Serialise the fitted ``ColumnTransformer`` for the inference module."""
        import joblib

        if self.pipeline is None:
            raise PreprocessingError("Aucun pipeline entraîné à sauvegarder.")
        joblib.dump(self.pipeline, filepath)
