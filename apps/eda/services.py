"""
Orchestration layer of the EDA module.

Views stay thin: they resolve the active dataset and call the functions below,
which combine ``stats_engine`` (numbers) and ``plot_factory`` (figures) into
template-ready dictionaries of :class:`apps.eda.theme.Chart` objects.

Statistics are cached per dataset (files are immutable once imported, see
``apps.datasets.cache``) so switching tabs or columns never recomputes the
whole profile.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from django.core.cache import cache

from apps.datasets.models import Dataset
from apps.datasets.services import profile_dataframe
from apps.eda import stats_engine
from apps.eda.plot_factory import EDAPlotFactory, is_categorical_like
from apps.eda.theme import Chart, make_chart

CACHE_TIMEOUT = 60 * 30
MAX_PAIRPLOT_COLUMNS = 6
LOW_CARDINALITY = 10
DEFAULT_PAIRPLOT_COLUMNS = 4


class EDAError(Exception):
    """User-facing error raised for invalid column choices."""


# ---------------------------------------------------------------------------
# Dataset context
# ---------------------------------------------------------------------------


def _cache_key(dataset: Dataset, name: str) -> str:
    return f"eda:{dataset.id}:{dataset.size_bytes}:{dataset.n_rows}:{name}"


def id_like_columns(dataset: Dataset) -> set[str]:
    return {c["name"] for c in (dataset.columns_meta or []) if c.get("is_id_like")}


def resolve_target(dataset: Dataset, state: dict[str, Any], frame: pd.DataFrame) -> str | None:
    """Target chosen at preprocessing time, else the ingestion suggestion."""
    for candidate in (state.get("target_column"), dataset.target_hint):
        if candidate and candidate in frame.columns:
            return candidate
    return None


def analysis_columns(dataset: Dataset, frame: pd.DataFrame) -> list[str]:
    """Numeric columns worth plotting (identifiers excluded)."""
    return stats_engine.numeric_columns(frame, exclude=id_like_columns(dataset))


def color_columns(dataset: Dataset, frame: pd.DataFrame) -> list[str]:
    """Columns usable as a colour channel: low-cardinality ones and numerics."""
    excluded = id_like_columns(dataset)
    result = []
    for column in frame.columns:
        name = str(column)
        if name in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 30:
            result.append(name)
    return result


def get_records(dataset: Dataset, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column statistics, cached per dataset."""
    key = _cache_key(dataset, "records")
    records = cache.get(key)
    if records is None:
        records = stats_engine.describe_frame(frame)
        cache.set(key, records, CACHE_TIMEOUT)
    return records


def get_correlation(dataset: Dataset, frame: pd.DataFrame, method: str) -> pd.DataFrame:
    key = _cache_key(dataset, f"corr:{method}")
    matrix = cache.get(key)
    if matrix is None:
        matrix = stats_engine.correlation_matrix(frame, method, analysis_columns(dataset, frame))
        cache.set(key, matrix, CACHE_TIMEOUT)
    return matrix


def record_for(records: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    return next((r for r in records if r["name"] == column), None)


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------


def overview(dataset: Dataset, frame: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Metric cards, statistics table, insights and dataset-level charts."""
    records = get_records(dataset, frame)
    profile = profile_dataframe(frame)
    matrix = get_correlation(dataset, frame, "pearson")
    pairs = stats_engine.top_correlations(matrix)
    target_info = stats_engine.target_summary(frame, target)

    charts: dict[str, Chart] = {}
    if any(r["missing"] for r in records):
        incomplete = sum(1 for r in records if r["missing"])
        charts["missing"] = make_chart(
            EDAPlotFactory.missing_values(records), "missing", "Valeurs manquantes par colonne",
            subtitle=f"{incomplete} colonne(s) incomplète(s) sur {len(records)}",
            height=max(200, 34 * incomplete + 80),
        )

    if target_info and target_info["task"] == "classification":
        charts["target"] = make_chart(
            EDAPlotFactory.class_balance(target_info), "target", f"Équilibre des classes — {target}",
            subtitle=(f"Ratio majoritaire / minoritaire : {target_info['imbalance_ratio']}"
                      if target_info.get("imbalance_ratio") else ""),
            height=300,
        )
    elif target_info:
        stats = target_info["stats"]
        charts["target"] = make_chart(
            EDAPlotFactory.histogram_kde(frame[target], stats.get("mean"), stats.get("median")),
            "target", f"Distribution de la cible — {target}",
            subtitle=f"Asymétrie : {stats.get('skew_label', '—')}", height=300,
        )

    if target_info and target_info["task"] == "regression":
        items = stats_engine.target_correlations(frame, target, exclude=id_like_columns(dataset))
        if items:
            charts["target_corr"] = make_chart(
                EDAPlotFactory.target_correlation_bar(items[:15], target), "target-corr",
                f"Corrélation des variables avec {target}", subtitle="Pearson, triée par |r|",
                height=max(220, 30 * min(len(items), 15) + 80),
            )

    kinds = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1

    return {
        "profile": profile,
        "records": records,
        "summary_rows": stats_engine.summary_table(records),
        "categorical_rows": [r for r in records if r.get("categorical") and not r["categorical"].get("empty")],
        "kinds": kinds,
        "target_info": target_info,
        "insights": stats_engine.build_insights(records, pairs, target_info, profile["duplicates"]),
        "top_pairs": pairs[:5],
        "charts": charts,
    }


# ---------------------------------------------------------------------------
# Univariate tab
# ---------------------------------------------------------------------------


def univariate(
    dataset: Dataset, frame: pd.DataFrame, column: str, target: str | None
) -> dict[str, Any]:
    """Distribution charts for one column (numeric or categorical)."""
    if column not in frame.columns:
        raise EDAError(f"Colonne inconnue : « {column} ».")
    records = get_records(dataset, frame)
    record = record_for(records, column) or stats_engine.column_stats(frame[column])
    series = frame[column]
    charts: list[Chart] = []

    group = None
    if target and target != column and target in frame.columns and is_categorical_like(frame[target]):
        group = frame[target]

    numeric_ok = record["kind"] == "numeric" and record.get("numeric") and not record["numeric"].get("empty")
    # 0/1 flags, class labels, Pclass…: a KDE or a violin of 3 values says nothing.
    few_values = numeric_ok and record["unique"] <= LOW_CARDINALITY

    if numeric_ok and not few_values:
        stats = record["numeric"]
        charts.append(make_chart(
            EDAPlotFactory.histogram_kde(series, stats["mean"], stats["median"]), f"hist-{column}",
            "Histogramme & densité (KDE)",
            subtitle=f"{record['count']} valeurs · {stats['skew_label']}", height=340, wide=True,
        ))
        charts.append(make_chart(
            EDAPlotFactory.boxplot(series), f"box-{column}", "Boîte à moustaches",
            subtitle=f"Q1 {stats['q1']} · médiane {stats['median']} · Q3 {stats['q3']} · "
                     f"{stats['outliers']} outlier(s)",
            height=220,
        ))
        charts.append(make_chart(
            EDAPlotFactory.violin(series), f"violin-{column}", "Graphique en violon",
            subtitle="Forme de la distribution (multimodalité)", height=220,
        ))
        if group is not None:
            n_groups = min(group.nunique(dropna=True), 10)
            charts.append(make_chart(
                EDAPlotFactory.violin(series, group), f"violin-by-{column}",
                f"Distribution par classe de « {target} »",
                subtitle="Une variable discriminante sépare nettement les violons",
                height=max(240, 70 * n_groups + 60), wide=True,
            ))
        if record["is_discrete"]:
            charts.append(make_chart(
                EDAPlotFactory.frequency_bar(series), f"freq-{column}", "Fréquences des valeurs",
                subtitle="Variable numérique discrète", height=max(200, 28 * record["unique"] + 70),
            ))
    elif few_values or record["kind"] in ("categorical", "boolean", "text"):
        shown = min(record["unique"] + (1 if record["missing"] else 0), 20)
        charts.append(make_chart(
            EDAPlotFactory.frequency_bar(series), f"freq-{column}", "Fréquences absolues & relatives",
            subtitle=f"{record['unique']} modalité(s)"
            + (" · variable numérique à peu de valeurs, traitée comme catégorielle" if few_values else ""),
            height=max(220, 28 * shown + 70),
        ))
        charts.append(make_chart(
            EDAPlotFactory.donut(series), f"donut-{column}", "Répartition (donut)",
            subtitle="Les plus petites modalités sont regroupées", height=300,
        ))
    return {"column": column, "record": record, "charts": charts, "target": target if group is not None else None}


# ---------------------------------------------------------------------------
# Correlation tab
# ---------------------------------------------------------------------------


def correlation(dataset: Dataset, frame: pd.DataFrame, method: str = "pearson", mask: bool = True) -> dict[str, Any]:
    method = method if method in ("pearson", "spearman") else "pearson"
    matrix = get_correlation(dataset, frame, method)
    if matrix.empty:
        return {"method": method, "mask": mask, "chart": None, "pairs": []}
    size = len(matrix.columns)
    chart = make_chart(
        EDAPlotFactory.correlation_heatmap(matrix, mask_upper=mask), f"corr-{method}",
        f"Matrice de corrélation — {'Pearson' if method == 'pearson' else 'Spearman'}",
        subtitle=("Relations linéaires" if method == "pearson" else "Relations monotones (rangs)")
        + f" · {size} variables numériques",
        height=min(760, max(500, 38 * size + 140)),
        margin=dict(l=110, r=20, t=24, b=110),
    )
    return {"method": method, "mask": mask, "chart": chart, "pairs": stats_engine.top_correlations(matrix, 10)}


# ---------------------------------------------------------------------------
# Scatter tab
# ---------------------------------------------------------------------------


def _check_columns(frame: pd.DataFrame, *columns: str | None) -> None:
    for column in columns:
        if column and column not in frame.columns:
            raise EDAError(f"Colonne inconnue : « {column} ».")


def default_scatter_axes(dataset: Dataset, frame: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Pre-select axes, colour by target, 4 pairplot columns.

    Continuous variables come first: the strongest pair on Titanic is
    Pclass ↔ Fare, but a 3-valued axis makes a poor first scatter plot.
    """
    numeric = analysis_columns(dataset, frame)
    features = [c for c in numeric if c != target] or numeric
    continuous = [c for c in features if frame[c].nunique(dropna=True) > LOW_CARDINALITY]
    ranked_features = continuous + [c for c in features if c not in continuous]
    pairs = stats_engine.top_correlations(get_correlation(dataset, frame, "pearson"), 50)
    pair = next((p for p in pairs if p["a"] in continuous and p["b"] in continuous), None)
    x = pair["a"] if pair else (ranked_features[0] if ranked_features else None)
    y = pair["b"] if pair else (ranked_features[1] if len(ranked_features) > 1 else x)
    z = next((c for c in ranked_features if c not in (x, y)), y)

    if target and target in numeric and not is_categorical_like(frame[target]):
        ranked = stats_engine.target_correlations(frame, target, exclude=id_like_columns(dataset))
        pairplot = [item["feature"] for item in ranked[: DEFAULT_PAIRPLOT_COLUMNS - 1]] + [target]
    else:
        pairplot = ranked_features[:DEFAULT_PAIRPLOT_COLUMNS]
    return {"x": x, "y": y, "z": z, "color": target, "pairplot": pairplot, "numeric": numeric}


def scatter(frame: pd.DataFrame, x: str, y: str, color: str | None, trend: bool) -> Chart:
    _check_columns(frame, x, y, color)
    note = "Échantillon aléatoire de 3 000 points affiché." if len(frame) > 3000 else ""
    return make_chart(
        EDAPlotFactory.scatter_2d(frame, x, y, color or None, trendline=trend), f"scatter-{x}-{y}",
        f"{y} en fonction de {x}", subtitle=f"Couleur : {color}" if color else "", height=440, note=note,
    )


def scatter_3d(frame: pd.DataFrame, x: str, y: str, z: str, color: str | None) -> Chart:
    _check_columns(frame, x, y, z, color)
    note = "Échantillon aléatoire de 2 000 points affiché." if len(frame) > 2000 else ""
    return make_chart(
        EDAPlotFactory.scatter_3d(frame, x, y, z, color or None), "scatter3d",
        f"Vue 3D — {x} × {y} × {z}", subtitle="Glissez pour faire pivoter, molette pour zoomer",
        height=520, note=note,
    )


def pairplot(frame: pd.DataFrame, columns: list[str], color: str | None) -> Chart:
    columns = [c for c in dict.fromkeys(columns) if c][:MAX_PAIRPLOT_COLUMNS]
    _check_columns(frame, *columns, color)
    if len(columns) < 2:
        raise EDAError("Sélectionnez au moins deux variables numériques.")
    note = "Échantillon aléatoire de 1 200 points affiché." if len(frame) > 1200 else ""
    return make_chart(
        EDAPlotFactory.scatter_matrix(frame, columns, color or None), "pairplot",
        "Matrice de dispersion (pair plot)",
        subtitle=f"{len(columns)} variables · triangle inférieur · sélection au lasso liée",
        height=max(420, 150 * len(columns)), note=note,
        margin=dict(l=70, r=20, t=30, b=60),
    )
