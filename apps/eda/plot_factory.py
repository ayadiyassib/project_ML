"""
Plotly figure factory for the EDA module.

Every method returns an *unthemed* ``plotly.graph_objects.Figure`` so it can be
asserted on in tests; :func:`apps.eda.theme.make_chart` applies the studio
theme and serialises it for the template (``Plotly.newPlot`` on the client).

Large datasets are handled by sampling the *points* of scatter-like charts
(never the statistics): 20 000 markers in a pair plot would freeze the browser
and add nothing visually.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as scipy_stats

from apps.eda import theme

KDE_MAX_POINTS = 5000
SCATTER_MAX_POINTS = 3000
SCATTER_3D_MAX_POINTS = 2000
SPLOM_MAX_POINTS = 1200
MAX_HIST_BINS = 60
MAX_GROUPS = 10
TOP_CATEGORIES = 20
DONUT_SLICES = 6
OTHER_LABEL = "Autres"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_frame(frame: pd.DataFrame, limit: int, seed: int = 0) -> pd.DataFrame:
    """Random subset of rows when ``frame`` is larger than ``limit``."""
    if len(frame) <= limit:
        return frame
    return frame.sample(limit, random_state=seed)


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def is_categorical_like(series: pd.Series, max_unique: int = MAX_GROUPS) -> bool:
    """True when a column should be coloured by category rather than by a scale."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return True
    return 0 < series.nunique(dropna=True) <= max_unique


def category_groups(series: pd.Series, max_groups: int = MAX_GROUPS) -> tuple[pd.Series, list[str]]:
    """Map a column to string labels, folding the tail into « Autres ».

    The returned order is a *natural sort* of the labels, not their frequency,
    so a class keeps the same colour whatever the sample or filter (colour
    follows the entity, never its rank).
    """
    labels = series.astype("object").where(series.notna(), "Manquant").astype(str)
    counts = labels.value_counts()
    if len(counts) > max_groups:
        keep = set(counts.index[: max_groups - 1])
        labels = labels.where(labels.isin(keep), OTHER_LABEL)
    ordered = sorted((v for v in labels.unique() if v != OTHER_LABEL), key=_natural_key)
    if OTHER_LABEL in set(labels):
        ordered.append(OTHER_LABEL)
    return labels, ordered


def _groups_on_full(full: pd.Series, index: pd.Index) -> tuple[pd.Series, list[str]]:
    """Class order computed on the *whole* column, labels returned for ``index``.

    Sampling or dropping rows must never shift a class to another colour slot.
    """
    labels, order = category_groups(full)
    return labels.loc[index], order


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.3g}"
    return f"{value:,.4g}".replace(",", " ")


def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """Gaussian kernel density estimate evaluated on ``grid`` (Scott bandwidth)."""
    clean = values[np.isfinite(values)]
    if len(clean) < 3 or np.ptp(clean) == 0:
        return None
    if len(clean) > KDE_MAX_POINTS:
        rng = np.random.default_rng(0)
        clean = rng.choice(clean, KDE_MAX_POINTS, replace=False)
    try:
        return scipy_stats.gaussian_kde(clean)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class EDAPlotFactory:
    """Builds every exploratory chart of the studio."""

    # -- univariate: numeric ------------------------------------------------
    @staticmethod
    def histogram_kde(series: pd.Series, mean: float | None = None, median: float | None = None) -> go.Figure:
        """Count histogram + KDE rescaled to counts, with mean/median markers."""
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        fig = go.Figure()
        if values.size == 0:
            return fig

        if np.ptp(values) == 0:
            edges = np.array([values[0] - 0.5, values[0] + 0.5])
        else:
            edges = np.histogram_bin_edges(values, bins="auto")
            if len(edges) - 1 > MAX_HIST_BINS:
                edges = np.histogram_bin_edges(values, bins=MAX_HIST_BINS)
        counts, edges = np.histogram(values, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        total = counts.sum() or 1
        custom = np.column_stack([edges[:-1], edges[1:], counts / total * 100])

        fig.add_trace(go.Bar(
            x=centers,
            y=counts,
            width=widths,
            name="Effectif",
            marker=dict(color=theme.rgba(theme.PRIMARY, 0.55), line=dict(color=theme.SURFACE, width=1)),
            customdata=custom,
            hovertemplate="[%{customdata[0]:.4g} ; %{customdata[1]:.4g}[<br>"
                          "<b>%{y}</b> obs. (%{customdata[2]:.1f} %)<extra></extra>",
        ))

        grid = np.linspace(edges[0], edges[-1], 240)
        density = kde_curve(values, grid)
        if density is not None:
            scale = values.size * float(np.mean(widths))
            fig.add_trace(go.Scatter(
                x=grid,
                y=density * scale,
                mode="lines",
                name="Densité (KDE)",
                line=dict(color=theme.PRIMARY, width=2),
                hovertemplate="x = %{x:.4g}<br>densité estimée<extra>KDE</extra>",
            ))

        for value, label, dash in ((mean, "Moyenne", "dash"), (median, "Médiane", "dot")):
            if value is None:
                continue
            fig.add_vline(
                x=value,
                line=dict(color=theme.INK_SECONDARY, width=1, dash=dash),
                annotation=dict(
                    text=f"{label} {_fmt(value)}",
                    font=dict(size=10, color=theme.INK_SECONDARY),
                    yanchor="top",
                    bgcolor=theme.rgba(theme.SURFACE, 0.85),
                ),
                # Inside the plot area, on opposite sides of their line, clear of the legend.
                annotation_position="top right" if label == "Moyenne" else "top left",
            )
        fig.update_layout(
            xaxis_title=str(series.name),
            yaxis_title="Effectif",
            bargap=0,
            showlegend=True,
            legend=dict(x=1, xanchor="right"),
        )
        return fig

    @staticmethod
    def boxplot(series: pd.Series, groups: pd.Series | None = None) -> go.Figure:
        """Horizontal box plot (quartiles, median, mean ± σ, outlier points)."""
        values = pd.to_numeric(series, errors="coerce")
        fig = go.Figure()
        if groups is None:
            fig.add_trace(go.Box(
                x=values.dropna(),
                name=str(series.name),
                boxpoints="outliers",
                boxmean="sd",
                marker=dict(color=theme.PRIMARY, size=5, opacity=0.7),
                line=dict(color=theme.PRIMARY, width=1.5),
                fillcolor=theme.rgba(theme.PRIMARY, 0.18),
                hoverinfo="x",
            ))
            fig.update_layout(showlegend=False, xaxis_title=str(series.name))
            fig.update_yaxes(showticklabels=False)
            return fig

        labels, order = category_groups(groups)
        for index, label in enumerate(order):
            subset = values[(labels == label) & values.notna()]
            color = theme.series_color(index)
            fig.add_trace(go.Box(
                x=subset,
                name=label,
                boxpoints="outliers",
                boxmean=True,
                marker=dict(color=color, size=4, opacity=0.7),
                line=dict(color=color, width=1.5),
                fillcolor=theme.rgba(color, 0.18),
            ))
        fig.update_layout(xaxis_title=str(series.name), yaxis_title=str(groups.name), showlegend=False)
        return fig

    @staticmethod
    def violin(series: pd.Series, groups: pd.Series | None = None) -> go.Figure:
        """Violin plot (density shape + inner box), optionally split by class."""
        values = pd.to_numeric(series, errors="coerce")
        fig = go.Figure()
        if groups is None:
            fig.add_trace(go.Violin(
                x=values.dropna(),
                name=str(series.name),
                orientation="h",
                box_visible=True,
                meanline_visible=True,
                points=False,
                line=dict(color=theme.PRIMARY, width=1.5),
                fillcolor=theme.rgba(theme.PRIMARY, 0.22),
                hoveron="violins+kde",
            ))
            fig.update_layout(showlegend=False, xaxis_title=str(series.name))
            fig.update_yaxes(showticklabels=False)
            return fig

        labels, order = category_groups(groups)
        for index, label in enumerate(order):
            subset = values[(labels == label) & values.notna()]
            if subset.empty:
                continue
            color = theme.series_color(index)
            fig.add_trace(go.Violin(
                x=subset,
                name=label,
                orientation="h",
                box_visible=True,
                meanline_visible=True,
                points=False,
                line=dict(color=color, width=1.5),
                fillcolor=theme.rgba(color, 0.22),
            ))
        fig.update_layout(
            xaxis_title=str(series.name), yaxis_title=str(groups.name), showlegend=False, violingap=0.15
        )
        return fig

    # -- univariate: categorical --------------------------------------------
    @staticmethod
    def frequency_bar(series: pd.Series, top: int = TOP_CATEGORIES) -> go.Figure:
        """Horizontal bars: absolute counts, relative % in labels and hover."""
        labels = series.astype("object").where(series.notna(), "Manquant").astype(str)
        counts = labels.value_counts()
        if len(counts) > top:
            head = counts.iloc[: top - 1]
            counts = pd.concat([head, pd.Series({OTHER_LABEL: int(counts.iloc[top - 1:].sum())})])
        total = int(counts.sum()) or 1
        pct = counts / total * 100
        names = [theme.short_label(name, 32) for name in counts.index]

        fig = go.Figure(go.Bar(
            x=counts.values,
            y=names,
            orientation="h",
            marker=dict(color=theme.PRIMARY, line=dict(color=theme.SURFACE, width=2)),
            text=[f"{p:.1f} %" for p in pct.values],
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=10),
            cliponaxis=False,
            customdata=np.column_stack([pct.values, list(counts.index)]),
            hovertemplate="<b>%{customdata[1]}</b><br>%{x} obs. (%{customdata[0]:.2f} %)<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Effectif", showlegend=False)
        fig.update_yaxes(autorange="reversed", showline=False, ticks="", type="category")
        return fig

    @staticmethod
    def donut(series: pd.Series, max_slices: int = DONUT_SLICES) -> go.Figure:
        """Donut of the share of each modality (tail folded into « Autres »)."""
        labels = series.astype("object").where(series.notna(), "Manquant").astype(str)
        counts = labels.value_counts()
        if len(counts) > max_slices:
            head = counts.iloc[: max_slices - 1]
            counts = pd.concat([head, pd.Series({OTHER_LABEL: int(counts.iloc[max_slices - 1:].sum())})])
        colors = [
            theme.OVERFLOW_COLOR if name == OTHER_LABEL else theme.series_color(index)
            for index, name in enumerate(counts.index)
        ]
        fig = go.Figure(go.Pie(
            labels=[theme.short_label(name, 24) for name in counts.index],
            values=counts.values,
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color=theme.SURFACE, width=2)),
            textinfo="percent",
            textfont=dict(size=10, color="#ffffff"),
            hovertemplate="<b>%{label}</b><br>%{value} obs. (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle", xanchor="left"),
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text=f"<b>{len(series.dropna().unique())}</b><br>modalités",
                showarrow=False,
                font=dict(size=12, color=theme.INK),
                x=0.5, y=0.5, xref="paper", yref="paper",
            )],
        )
        return fig

    # -- dataset-level ------------------------------------------------------
    @staticmethod
    def missing_values(records: list[dict[str, Any]]) -> go.Figure:
        """% of missing cells for each incomplete column."""
        rows = sorted((r for r in records if r["missing"] > 0), key=lambda r: r["missing_pct"], reverse=True)
        fig = go.Figure(go.Bar(
            x=[r["missing_pct"] for r in rows],
            y=[theme.short_label(r["name"], 26) for r in rows],
            orientation="h",
            marker=dict(color=theme.PRIMARY, line=dict(color=theme.SURFACE, width=2)),
            text=[f"{r['missing_pct']:.1f} %" for r in rows],
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=10),
            cliponaxis=False,
            customdata=[[r["missing"], r["name"]] for r in rows],
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]} cellule(s) vide(s) "
                          "(%{x:.2f} %)<extra></extra>",
        ))
        fig.update_layout(xaxis_title="% de valeurs manquantes", showlegend=False)
        fig.update_xaxes(range=[0, max([r["missing_pct"] for r in rows] + [10]) * 1.18])
        fig.update_yaxes(autorange="reversed", showline=False, ticks="", type="category")
        return fig

    @staticmethod
    def class_balance(target: dict[str, Any]) -> go.Figure:
        """Bar per class, coloured by the class's fixed categorical slot."""
        classes = sorted(target["classes"], key=lambda c: _natural_key(c["label"]))
        fig = go.Figure(go.Bar(
            x=[c["label"] for c in classes],
            y=[c["count"] for c in classes],
            marker=dict(
                color=[theme.series_color(i) for i in range(len(classes))],
                line=dict(color=theme.SURFACE, width=2),
            ),
            text=[f"{c['pct']:.1f} %" for c in classes],
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=10),
            cliponaxis=False,
            hovertemplate="Classe <b>%{x}</b><br>%{y} obs.<extra></extra>",
        ))
        fig.update_layout(xaxis_title=target["name"], yaxis_title="Effectif", showlegend=False)
        fig.update_xaxes(type="category")
        return fig

    # -- bivariate ----------------------------------------------------------
    @staticmethod
    def correlation_heatmap(matrix: pd.DataFrame, mask_upper: bool = True, annotate: bool | None = None) -> go.Figure:
        """Diverging heatmap centred on 0, upper triangle optionally hidden."""
        values = matrix.to_numpy(dtype=float).copy()
        names = [str(c) for c in matrix.columns]
        if mask_upper:
            values[np.triu_indices_from(values, k=1)] = np.nan
        annotate = len(names) <= 15 if annotate is None else annotate
        text = np.where(np.isnan(values), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(values)))

        fig = go.Figure(go.Heatmap(
            z=values,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=theme.DIVERGING,
            colorbar=theme.style_colorbar("r"),
            text=text if annotate else None,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=10),
            hoverongaps=False,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticks="")
        fig.update_xaxes(showgrid=False, showline=False, ticks="", tickangle=-35)
        return fig

    @staticmethod
    def target_correlation_bar(items: list[dict[str, Any]], target: str) -> go.Figure:
        """Correlation of each feature with the target; sign → diverging pole."""
        positive, negative = theme.DIVERGING[-1][1], theme.DIVERGING[0][1]
        fig = go.Figure(go.Bar(
            x=[item["value"] for item in items],
            y=[theme.short_label(item["feature"], 26) for item in items],
            orientation="h",
            marker=dict(
                color=[positive if item["value"] >= 0 else negative for item in items],
                line=dict(color=theme.SURFACE, width=2),
            ),
            text=[f"{item['value']:+.2f}" for item in items],
            textposition="outside",
            textfont=dict(color=theme.INK_SECONDARY, size=10),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=f"Corrélation avec {target}", showlegend=False)
        fig.update_xaxes(range=[-1.15, 1.15], zeroline=True)
        fig.update_yaxes(autorange="reversed", showline=False, ticks="", type="category")
        return fig

    @staticmethod
    def scatter_2d(
        frame: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        trendline: bool = False,
    ) -> go.Figure:
        """Interactive scatter, coloured by category (with symbols) or by a scale."""
        data = frame[[c for c in dict.fromkeys([x, y, color]) if c]].copy()
        data[x] = pd.to_numeric(data[x], errors="coerce")
        data[y] = pd.to_numeric(data[y], errors="coerce")
        data = data.dropna(subset=[x, y])
        full = data
        data = sample_frame(data, SCATTER_MAX_POINTS)
        fig = go.Figure()
        hover = f"{x} = %{{x:.4g}}<br>{y} = %{{y:.4g}}"

        if color and is_categorical_like(data[color]):
            labels, order = _groups_on_full(frame[color], data.index)
            for index, label in enumerate(order):
                subset = data[labels == label]
                if subset.empty:
                    continue
                fig.add_trace(go.Scattergl(
                    x=subset[x], y=subset[y], mode="markers", name=label,
                    marker=dict(
                        color=theme.series_color(index), symbol=theme.series_symbol(index), size=8,
                        opacity=0.8, line=dict(color=theme.SURFACE, width=1),
                    ),
                    hovertemplate=f"{hover}<br>{color} = {label}<extra></extra>",
                ))
        elif color:
            fig.add_trace(go.Scattergl(
                x=data[x], y=data[y], mode="markers", name=color,
                marker=dict(
                    color=pd.to_numeric(data[color], errors="coerce"), colorscale=theme.SEQUENTIAL_BLUE,
                    size=8, opacity=0.85, line=dict(color=theme.SURFACE, width=1),
                    colorbar=theme.style_colorbar(color),
                ),
                customdata=data[color],
                hovertemplate=f"{hover}<br>{color} = %{{customdata:.4g}}<extra></extra>",
                showlegend=False,
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=data[x], y=data[y], mode="markers", name="Observations",
                marker=dict(color=theme.PRIMARY, size=8, opacity=0.75, line=dict(color=theme.SURFACE, width=1)),
                hovertemplate=f"{hover}<extra></extra>",
                showlegend=False,
            ))

        if trendline and len(full) >= 3 and full[x].nunique() > 1:
            slope, intercept = np.polyfit(full[x], full[y], 1)
            r = float(np.corrcoef(full[x], full[y])[0, 1])
            xs = np.linspace(full[x].min(), full[x].max(), 50)
            fig.add_trace(go.Scatter(
                x=xs, y=slope * xs + intercept, mode="lines",
                name=f"Tendance MCO (r = {r:.2f}, R² = {r * r:.2f})",
                line=dict(color=theme.INK, width=2, dash="dash"),
                hovertemplate=f"ŷ = {slope:.4g}·x + {intercept:.4g}<extra></extra>",
            ))
        fig.update_layout(xaxis_title=x, yaxis_title=y, showlegend=True)
        return fig

    @staticmethod
    def scatter_3d(frame: pd.DataFrame, x: str, y: str, z: str, color: str | None = None) -> go.Figure:
        """Rotatable 3D scatter (``Scatter3d``)."""
        data = frame[[c for c in dict.fromkeys([x, y, z, color]) if c]].copy()
        for axis in (x, y, z):
            data[axis] = pd.to_numeric(data[axis], errors="coerce")
        data = sample_frame(data.dropna(subset=[x, y, z]), SCATTER_3D_MAX_POINTS)
        fig = go.Figure()
        hover = f"{x} = %{{x:.4g}}<br>{y} = %{{y:.4g}}<br>{z} = %{{z:.4g}}"

        if color and is_categorical_like(data[color]):
            labels, order = _groups_on_full(frame[color], data.index)
            for index, label in enumerate(order):
                subset = data[labels == label]
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=subset[x], y=subset[y], z=subset[z], mode="markers", name=label,
                    marker=dict(color=theme.series_color(index), size=4, opacity=0.85,
                                symbol=("circle", "diamond", "square", "cross", "x")[index % 5]),
                    hovertemplate=f"{hover}<br>{color} = {label}<extra></extra>",
                ))
        else:
            marker: dict[str, Any] = dict(size=4, opacity=0.85, color=theme.PRIMARY)
            if color:
                marker.update(color=pd.to_numeric(data[color], errors="coerce"),
                              colorscale=theme.SEQUENTIAL_BLUE, colorbar=theme.style_colorbar(color))
            fig.add_trace(go.Scatter3d(
                x=data[x], y=data[y], z=data[z], mode="markers", marker=marker,
                hovertemplate=f"{hover}<extra></extra>", showlegend=False,
            ))

        axis = dict(
            backgroundcolor="rgba(0,0,0,0)", gridcolor=theme.GRID, zerolinecolor=theme.BASELINE,
            linecolor=theme.BASELINE, tickfont=dict(color=theme.INK_MUTED, size=9),
            title=dict(font=dict(color=theme.INK_SECONDARY, size=11)),
        )
        fig.update_layout(scene=dict(
            xaxis=dict(axis, title=dict(axis["title"], text=x)),
            yaxis=dict(axis, title=dict(axis["title"], text=y)),
            zaxis=dict(axis, title=dict(axis["title"], text=z)),
            bgcolor="rgba(0,0,0,0)",
        ), margin=dict(l=0, r=0, t=24, b=0))
        return fig

    @staticmethod
    def scatter_matrix(frame: pd.DataFrame, columns: list[str], color: str | None = None) -> go.Figure:
        """Pair plot (``Splom``): lower triangle only, coloured by class."""
        keep = list(dict.fromkeys(columns + ([color] if color else [])))
        data = frame[keep].copy()
        for column in columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = sample_frame(data.dropna(subset=columns), SPLOM_MAX_POINTS)
        dims_for = lambda subset: [dict(label=theme.short_label(c, 18), values=subset[c]) for c in columns]  # noqa: E731
        fig = go.Figure()
        common = dict(diagonal_visible=False, showupperhalf=False)

        if color and is_categorical_like(data[color]):
            labels, order = _groups_on_full(frame[color], data.index)
            for index, label in enumerate(order):
                subset = data[labels == label]
                if subset.empty:
                    continue
                fig.add_trace(go.Splom(
                    dimensions=dims_for(subset), name=label,
                    marker=dict(color=theme.series_color(index), symbol=theme.series_symbol(index),
                                size=5, opacity=0.75, line=dict(color=theme.SURFACE, width=0.5)),
                    **common,
                ))
        else:
            marker: dict[str, Any] = dict(color=theme.PRIMARY, size=5, opacity=0.7,
                                          line=dict(color=theme.SURFACE, width=0.5))
            if color:
                marker.update(color=pd.to_numeric(data[color], errors="coerce"),
                              colorscale=theme.SEQUENTIAL_BLUE, colorbar=theme.style_colorbar(color))
            fig.add_trace(go.Splom(dimensions=dims_for(data), marker=marker, showlegend=False, **common))

        # Declare every implicit splom axis so the theme reaches all of them.
        axes = {}
        for index in range(1, len(columns) + 1):
            suffix = "" if index == 1 else str(index)
            axes[f"xaxis{suffix}"] = dict(showgrid=True)
            axes[f"yaxis{suffix}"] = dict(showgrid=True)
        fig.update_layout(dragmode="select", **axes)
        return fig
