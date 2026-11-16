"""
Typed configuration for the preprocessing pipeline.

The config is the only thing stored between two runs (as ``config.json`` next to
the joblib artefacts), so it must stay JSON-serialisable and stable.  It is
parsed from raw POST data by ``from_form`` which validates every choice and
raises ``PreprocessingError`` with a user-facing French message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

NUMERIC_STRATEGIES = ("mean", "median", "most_frequent", "constant", "knn", "drop_rows")
CATEGORICAL_STRATEGIES = ("most_frequent", "constant", "drop_rows")
SCALERS = ("standard", "minmax", "robust", "none")
ENCODERS = ("onehot", "ordinal")
OUTLIER_METHODS = ("none", "clip", "drop")
TASK_TYPES = ("classification", "regression", "clustering")
TEST_SIZES = (0.15, 0.2, 0.25, 0.3)
MISSING_THRESHOLDS = (1.0, 0.8, 0.6, 0.4, 0.2)


class PreprocessingError(Exception):
    """Raised when the configuration or the data cannot be processed."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes", "oui"}


def _as_choice(value: Any, choices: tuple[str, ...], default: str, label: str) -> str:
    candidate = str(value).strip().lower() if value not in (None, "") else default
    if candidate not in choices:
        raise PreprocessingError(
            f"Valeur invalide pour {label} : « {value} ». Choix possibles : {', '.join(choices)}."
        )
    return candidate


def _as_float(value: Any, default: float, label: str, minimum: float, maximum: float) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"{label} doit être un nombre.") from exc
    if not minimum <= number <= maximum:
        raise PreprocessingError(f"{label} doit être compris entre {minimum} et {maximum}.")
    return number


@dataclass(slots=True)
class PreprocessingConfig:
    """Everything the user can tune before fitting the pipeline."""

    target: str | None = None
    features: list[str] | None = None
    task_type: str = "classification"

    numeric_strategy: str = "mean"
    categorical_strategy: str = "most_frequent"
    constant_value: str = "Inconnu"
    missing_threshold: float = 1.0
    drop_duplicates: bool = True

    outlier_method: str = "none"
    outlier_factor: float = 1.5

    scaler: str = "standard"
    encoder: str = "onehot"

    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PreprocessingConfig":
        if not data:
            return cls()
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_form(cls, form: Mapping[str, Any], columns: list[str]) -> "PreprocessingConfig":
        """Build a validated config from ``request.POST`` and the dataset columns."""
        get_list = getattr(form, "getlist", None)

        target = (form.get("target") or "").strip() or None
        if target is not None and target not in columns:
            raise PreprocessingError(f"La colonne cible « {target} » n'existe pas dans le dataset.")

        if get_list is not None:
            features = [str(item) for item in get_list("features") if str(item).strip()]
        else:
            raw_features = form.get("features") or []
            features = [str(item) for item in (raw_features if isinstance(raw_features, list) else [raw_features]) if str(item).strip()]

        features = [feature for feature in features if feature in columns]
        if target is not None:
            features = [feature for feature in features if feature != target]
        if not features:
            raise PreprocessingError("Sélectionnez au moins une variable explicative.")

        task_type = (
            "clustering"
            if target is None
            else _as_choice(form.get("task_type"), TASK_TYPES[:2], "classification", "le type de tâche")
        )

        test_size = _as_float(form.get("test_size"), 0.2, "La proportion de test", 0.05, 0.5)
        threshold = _as_float(
            form.get("missing_threshold"), 1.0, "Le seuil de valeurs manquantes", 0.0, 1.0
        )

        config = cls(
            target=target,
            features=features,
            task_type=task_type,
            numeric_strategy=_as_choice(
                form.get("numeric_strategy"), NUMERIC_STRATEGIES, "mean", "la stratégie numérique"
            ),
            categorical_strategy=_as_choice(
                form.get("categorical_strategy"),
                CATEGORICAL_STRATEGIES,
                "most_frequent",
                "la stratégie catégorielle",
            ),
            constant_value=(form.get("constant_value") or "Inconnu").strip() or "Inconnu",
            missing_threshold=threshold,
            drop_duplicates=_as_bool(form.get("drop_duplicates", False)),
            outlier_method=_as_choice(
                form.get("outlier_method"), OUTLIER_METHODS, "none", "le traitement des outliers"
            ),
            outlier_factor=_as_float(form.get("outlier_factor"), 1.5, "Le facteur IQR", 0.5, 5.0),
            scaler=_as_choice(form.get("scaler"), SCALERS, "standard", "le scaler"),
            encoder=_as_choice(form.get("encoder"), ENCODERS, "onehot", "l'encodeur"),
            test_size=test_size,
            stratify=_as_bool(form.get("stratify", False)),
            random_state=42,
        )
        return config
