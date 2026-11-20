"""
Descriptive statistics engine for the EDA module (tasks 1.1 and 2.1).

Pure pandas / NumPy / SciPy: no Django import, so every function is unit-tested
directly.  The EDA runs on the *raw* dataset (before the train/test split):
nothing computed here is ever fed back into a model, so it cannot leak test
information — it only informs the user's preprocessing choices.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from apps.datasets.services import classify_column

NORMALITY_MIN_ROWS = 20
NORMALITY_MAX_ROWS = 5000
STRONG_CORRELATION = 0.8
HIGH_SKEW = 1.0
RARE_CATEGORY_PCT = 1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value: Any, digits: int = 4) -> float | None:
    """Round a scalar to a JSON-friendly float, ``None`` when NaN/inf."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def completeness_level(pct: float) -> str:
    """Badge level for a completeness percentage: good, ok, warning, critical."""
    if pct >= 100:
        return "good"
    if pct >= 95:
        return "ok"
    if pct >= 70:
        return "warning"
    return "critical"


def skew_label(skew: float | None) -> str:
    if skew is None:
        return "—"
    magnitude = abs(skew)
    if magnitude < 0.5:
        return "Symétrique"
    side = "à droite" if skew > 0 else "à gauche"
    if magnitude < HIGH_SKEW:
        return f"Légèrement étalée {side}"
    return f"Fortement étalée {side}"


def kurtosis_label(kurtosis: float | None) -> str:
    """Interpret pandas' *excess* kurtosis (0 for a normal distribution)."""
    if kurtosis is None:
        return "—"
    if kurtosis > 1:
        return "Leptokurtique (queues lourdes)"
    if kurtosis < -1:
        return "Platykurtique (aplatie)"
    return "Mésokurtique (proche normale)"


def numeric_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Numeric (non-boolean) columns, minus ``exclude`` (typically ID-like columns)."""
    excluded = exclude or set()
    return [
        str(column)
        for column in frame.columns
        if str(column) not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def normality_test(values: pd.Series) -> dict[str, Any] | None:
    """D'Agostino-Pearson K² test on a sample of the column.

    Returns ``None`` when the column is too short or constant.  Very large
    samples are subsampled: with tens of thousands of rows every real-world
    variable is "significantly" non-normal, which tells the user nothing.
    """
    clean = values.dropna().astype(float)
    if len(clean) < NORMALITY_MIN_ROWS or clean.nunique() < 3:
        return None
    if len(clean) > NORMALITY_MAX_ROWS:
        clean = clean.sample(NORMALITY_MAX_ROWS, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            statistic, p_value = scipy_stats.normaltest(clean.to_numpy())
        except ValueError:
            return None
    p_value = _finite(p_value, 6)
    if p_value is None:
        return None
    return {
        "test": "D'Agostino-Pearson K²",
        "statistic": _finite(statistic, 3),
        "p_value": p_value,
        "is_normal": p_value >= 0.05,
    }


# ---------------------------------------------------------------------------
# Per-column statistics
# ---------------------------------------------------------------------------


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    """Central tendency, dispersion, shape and IQR outliers of a numeric column."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"empty": True}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((clean < lower) | (clean > upper)).sum())
    skew = _finite(clean.skew()) if len(clean) > 2 else None
    kurtosis = _finite(clean.kurt()) if len(clean) > 3 else None
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0

    return {
        "empty": False,
        "mean": _finite(mean),
        "median": _finite(clean.median()),
        "std": _finite(std),
        "variance": _finite(clean.var(ddof=1)) if len(clean) > 1 else 0.0,
        "min": _finite(clean.min()),
        "max": _finite(clean.max()),
        "range": _finite(clean.max() - clean.min()),
        "q1": _finite(q1),
        "q3": _finite(q3),
        "iqr": _finite(iqr),
        "lower_fence": _finite(lower),
        "upper_fence": _finite(upper),
        "outliers": outliers,
        "outliers_pct": round(outliers / len(clean) * 100, 2),
        "skewness": skew,
        "skew_label": skew_label(skew),
        "kurtosis": kurtosis,
        "kurtosis_label": kurtosis_label(kurtosis),
        "cv": _finite(std / abs(mean)) if mean else None,
        "zeros": int((clean == 0).sum()),
        "normality": normality_test(clean),
    }


def categorical_stats(series: pd.Series) -> dict[str, Any]:
    """Mode, frequencies and rarity of a categorical / boolean / text column."""
    clean = series.dropna()
    if clean.empty:
        return {"empty": True}
    counts = clean.astype(str).value_counts()
    total = int(counts.sum())
    probabilities = counts.to_numpy() / total
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    rare = int((counts / total * 100 < RARE_CATEGORY_PCT).sum())

    return {
        "empty": False,
        "mode": str(counts.index[0]),
        "mode_count": int(counts.iloc[0]),
        "mode_pct": round(int(counts.iloc[0]) / total * 100, 2),
        "entropy": round(entropy, 3),
        "balance": round(entropy / max_entropy, 3) if len(counts) > 1 else 0.0,
        "rare_categories": rare,
        "avg_length": round(float(clean.astype(str).str.len().mean()), 1),
        "top_values": [
            {"value": str(value), "count": int(count), "pct": round(int(count) / total * 100, 2)}
            for value, count in counts.head(5).items()
        ],
    }


def column_stats(series: pd.Series, kind: str | None = None) -> dict[str, Any]:
    """Unified statistics record for one column."""
    kind = kind or classify_column(series)
    rows = len(series)
    non_null = int(series.notna().sum())
    missing = rows - non_null
    unique = int(series.nunique(dropna=True))
    completeness = round(non_null / rows * 100, 2) if rows else 0.0

    record: dict[str, Any] = {
        "name": str(series.name),
        "kind": kind,
        "dtype": str(series.dtype),
        "count": non_null,
        "missing": missing,
        "missing_pct": round(missing / rows * 100, 2) if rows else 0.0,
        "completeness": completeness,
        "completeness_level": completeness_level(completeness),
        "unique": unique,
        "cardinality_ratio": round(unique / non_null, 4) if non_null else 0.0,
        "is_discrete": kind == "numeric" and 0 < unique <= 15,
    }

    if kind == "numeric":
        record["numeric"] = numeric_stats(series)
    elif kind == "datetime":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        record["datetime"] = {
            "min": str(parsed.min()) if parsed.notna().any() else None,
            "max": str(parsed.max()) if parsed.notna().any() else None,
        }
    elif kind in ("categorical", "boolean", "text"):
        record["categorical"] = categorical_stats(series)
    return record


def describe_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Statistics for every column of ``frame``."""
    return [column_stats(frame[column]) for column in frame.columns]


def summary_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten numeric records into rows for the "describe()"-style table."""
    rows = []
    for record in records:
        numeric = record.get("numeric")
        if not numeric or numeric.get("empty"):
            continue
        skew = numeric.get("skewness")
        rows.append({
            "name": record["name"],
            **numeric,
            "unique": record["unique"],
            "cardinality_ratio": record["cardinality_ratio"],
            "completeness": record["completeness"],
            "completeness_level": record["completeness_level"],
            "high_skew": skew is not None and abs(skew) >= HIGH_SKEW,
        })
    return rows


# ---------------------------------------------------------------------------
# Correlations (task 2.1)
# ---------------------------------------------------------------------------


def correlation_matrix(
    frame: pd.DataFrame, method: str = "pearson", columns: list[str] | None = None
) -> pd.DataFrame:
    """Pearson (linear) or Spearman (monotonic) correlation of numeric columns."""
    if method not in ("pearson", "spearman"):
        raise ValueError("La méthode doit être 'pearson' ou 'spearman'.")
    cols = columns if columns is not None else numeric_columns(frame)
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method)


def top_correlations(matrix: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    """Strongest pairs (upper triangle, by absolute value)."""
    if matrix.empty:
        return []
    values = matrix.to_numpy()
    names = list(matrix.columns)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = values[i, j]
            if np.isfinite(value):
                pairs.append((names[i], names[j], float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return [
        {
            "a": a,
            "b": b,
            "value": round(value, 3),
            "abs": round(abs(value), 3),
            "strength": correlation_strength(value),
            "sign": "positive" if value >= 0 else "negative",
        }
        for a, b, value in pairs[:limit]
    ]


def correlation_strength(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= STRONG_CORRELATION:
        return "Très forte"
    if magnitude >= 0.6:
        return "Forte"
    if magnitude >= 0.4:
        return "Modérée"
    if magnitude >= 0.2:
        return "Faible"
    return "Négligeable"


def target_correlations(
    frame: pd.DataFrame, target: str, method: str = "pearson", exclude: set[str] | None = None
) -> list[dict[str, Any]]:
    """Correlation of every numeric feature with a numeric target."""
    if target not in frame.columns or not pd.api.types.is_numeric_dtype(frame[target]):
        return []
    columns = [c for c in numeric_columns(frame, exclude) if c != target]
    result = []
    for column in columns:
        pair = frame[[column, target]].dropna()
        if len(pair) < 3 or pair[column].nunique() < 2 or pair[target].nunique() < 2:
            continue
        value = pair[column].corr(pair[target], method=method)
        if np.isfinite(value):
            result.append({"feature": column, "value": round(float(value), 3)})
    result.sort(key=lambda item: abs(item["value"]), reverse=True)
    return result


# ---------------------------------------------------------------------------
# Target & storytelling
# ---------------------------------------------------------------------------


def target_summary(frame: pd.DataFrame, target: str | None) -> dict[str, Any] | None:
    """Class balance for a categorical target, distribution for a numeric one."""
    if not target or target not in frame.columns:
        return None
    series = frame[target].dropna()
    if series.empty:
        return None
    unique = int(series.nunique())
    is_classification = (
        not pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or unique <= 10
    )
    summary: dict[str, Any] = {"name": target, "unique": unique}
    if is_classification:
        counts = series.astype(str).value_counts()
        total = int(counts.sum())
        ratio = float(counts.iloc[0] / counts.iloc[-1]) if counts.iloc[-1] else None
        summary.update(
            task="classification",
            classes=[
                {"label": str(label), "count": int(count), "pct": round(int(count) / total * 100, 2)}
                for label, count in counts.items()
            ],
            imbalance_ratio=round(ratio, 2) if ratio else None,
            is_imbalanced=bool(ratio and ratio >= 3),
        )
    else:
        summary.update(task="regression", stats=numeric_stats(series))
    return summary


def build_insights(
    records: list[dict[str, Any]],
    top_pairs: list[dict[str, Any]],
    target: dict[str, Any] | None,
    duplicates: int = 0,
) -> list[dict[str, str]]:
    """Plain-language findings ("data storytelling") shown above the charts."""
    insights: list[dict[str, str]] = []

    for record in records:
        if record["missing_pct"] >= 40:
            insights.append({
                "level": "critical",
                "title": f"« {record['name']} » est incomplète",
                "text": f"{record['missing_pct']}% de valeurs manquantes : envisagez de la supprimer.",
            })
        elif record["missing_pct"] > 0:
            insights.append({
                "level": "warning",
                "title": f"Valeurs manquantes dans « {record['name']} »",
                "text": f"{record['missing']} cellule(s) vide(s) ({record['missing_pct']}%) à imputer.",
            })

    for record in records:
        numeric = record.get("numeric") or {}
        skew = numeric.get("skewness")
        if skew is not None and abs(skew) >= HIGH_SKEW and not record["is_discrete"]:
            insights.append({
                "level": "info",
                "title": f"Distribution asymétrique : « {record['name']} »",
                "text": f"Skewness = {skew}. Une transformation log ou un RobustScaler peut aider.",
            })
        if numeric.get("outliers_pct", 0) >= 5:
            insights.append({
                "level": "warning",
                "title": f"Outliers dans « {record['name']} »",
                "text": f"{numeric['outliers']} valeur(s) hors des bornes IQR ({numeric['outliers_pct']}%).",
            })
        if record["unique"] <= 1 and record["count"]:
            insights.append({
                "level": "warning",
                "title": f"« {record['name']} » est constante",
                "text": "Aucune information pour un modèle : à exclure.",
            })

    target_name = target["name"] if target else None
    for pair in top_pairs:
        # A feature strongly tied to the target is good news, not collinearity.
        if target_name in (pair["a"], pair["b"]):
            continue
        if pair["abs"] >= STRONG_CORRELATION:
            insights.append({
                "level": "info",
                "title": f"Colinéarité : {pair['a']} ↔ {pair['b']}",
                "text": f"r = {pair['value']}. Les deux variables portent presque la même information.",
            })

    if target and target.get("is_imbalanced"):
        insights.append({
            "level": "warning",
            "title": f"Classes déséquilibrées sur « {target['name']} »",
            "text": f"Ratio majoritaire/minoritaire = {target['imbalance_ratio']} : "
                    "privilégiez F1, rappel et la courbe Précision-Rappel.",
        })

    if duplicates:
        insights.append({
            "level": "info",
            "title": "Lignes dupliquées",
            "text": f"{duplicates} doublon(s) détecté(s) : supprimables au prétraitement.",
        })

    order = {"critical": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda item: order.get(item["level"], 3))
    return insights
