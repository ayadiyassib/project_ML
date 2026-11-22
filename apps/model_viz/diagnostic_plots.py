"""
Diagnostic figures for trained models (feature 6 of the specification).

``ModelDiagnosticPlots`` only *draws*: every method receives arrays that were
already computed by ``apps.model_viz.services`` (or by any caller, e.g. the
ML engine) and returns an unthemed ``plotly.graph_objects.Figure``.  Theming
and JSON serialisation happen in :func:`apps.eda.theme.make_chart`.

Design rules (shared with the EDA module):
* class *i* always gets categorical slot *i* (and marker symbol *i*);
* no dual-axis chart — two measures on different scales become two stacked
  panels sharing the x axis (elbow + silhouette, loss + validation score);
* reference lines (y = x, chance level, zero residual) are dashed neutral ink.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

from apps.eda import theme

MAX_POINTS = 3000


def _sample_idx(n: int, limit: int = MAX_POINTS, seed: int = 0) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def _discrete_colorscale(n_classes: int, alpha: float) -> list[list[Any]]:
    """Step colorscale mapping integer class index *i* to categorical slot *i*."""
    if n_classes <= 1:
        color = theme.rgba(theme.series_color(0), alpha)
        return [[0.0, color], [1.0, color]]
    scale = []
    for index in range(n_classes):
        color = theme.rgba(theme.series_color(index), alpha)
        scale.append([index / n_classes, color])
        scale.append([(index + 1) / n_classes, color])
    return scale


class ModelDiagnosticPlots:
    """Figure builders for regression, classification, clustering and MLP models."""

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    @staticmethod
    def fit_curve(
        x: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        fitted: np.ndarray,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
        x_label: str,
        y_label: str,
        band_label: str = "IC 95 % de la moyenne",
    ) -> go.Figure:
        """Observed points + fitted line/curve + optional confidence band."""
        idx = _sample_idx(len(x))
        fig = go.Figure()
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself", fillcolor=theme.rgba(theme.ACCENT, 0.18),
                line=dict(width=0), name=band_label, hoverinfo="skip",
            ))
        fig.add_trace(go.Scattergl(
            x=x[idx], y=y[idx], mode="markers", name="Observations (test)",
            marker=dict(color=theme.PRIMARY, size=7, opacity=0.6, line=dict(color=theme.SURFACE, width=1)),
            hovertemplate=f"{x_label} = %{{x:.4g}}<br>{y_label} = %{{y:.4g}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=grid, y=fitted, mode="lines", name="Modèle ŷ",
            line=dict(color=theme.ACCENT, width=2.5),
            hovertemplate=f"{x_label} = %{{x:.4g}}<br>ŷ = %{{y:.4g}}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        return fig

    @staticmethod
    def predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, r2: float | None = None) -> go.Figure:
        """ŷ against y with the ideal ``y = x`` diagonal."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        idx = _sample_idx(len(y_true))
        low = float(min(y_true.min(), y_pred.min()))
        high = float(max(y_true.max(), y_pred.max()))
        pad = (high - low) * 0.04 or 1.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[low - pad, high + pad], y=[low - pad, high + pad], mode="lines", name="Idéal (y = x)",
            line=dict(color=theme.REFERENCE_LINE, width=1.5, dash="dash"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=y_true[idx], y=y_pred[idx], mode="markers",
            name="Prédictions" + (f" (R² = {r2:.3f})" if r2 is not None else ""),
            marker=dict(color=theme.PRIMARY, size=7, opacity=0.6, line=dict(color=theme.SURFACE, width=1)),
            customdata=(y_true - y_pred)[idx],
            hovertemplate="Réel = %{x:.4g}<br>Prédit = %{y:.4g}<br>Erreur = %{customdata:.4g}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Valeur réelle (y)", yaxis_title="Valeur prédite (ŷ)")
        fig.update_xaxes(range=[low - pad, high + pad])
        fig.update_yaxes(range=[low - pad, high + pad], scaleanchor="x", scaleratio=1)
        return fig

    @staticmethod
    def residuals_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Residuals e = y − ŷ against ŷ, with 0 and ±1.96σ guides (homoscedasticity)."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        idx = _sample_idx(len(residuals))
        fig = go.Figure(go.Scattergl(
            x=y_pred[idx], y=residuals[idx], mode="markers", name="Résidus",
            marker=dict(color=theme.PRIMARY, size=7, opacity=0.6, line=dict(color=theme.SURFACE, width=1)),
            hovertemplate="ŷ = %{x:.4g}<br>e = %{y:.4g}<extra></extra>", showlegend=False,
        ))
        fig.add_hline(y=0, line=dict(color=theme.REFERENCE_LINE, width=1.5, dash="dash"))
        if sigma:
            for bound in (1.96 * sigma, -1.96 * sigma):
                fig.add_hline(y=bound, line=dict(color=theme.BASELINE, width=1, dash="dot"))
            fig.add_annotation(
                x=1, xref="paper", y=1.96 * sigma, text="+1,96σ", showarrow=False, xanchor="right",
                yanchor="bottom", font=dict(size=9, color=theme.INK_MUTED),
            )
        fig.update_layout(xaxis_title="Valeur prédite (ŷ)", yaxis_title="Résidu (y − ŷ)")
        return fig

    @staticmethod
    def residual_histogram(residuals: np.ndarray) -> go.Figure:
        """Residual counts with the fitted normal density rescaled to counts."""
        residuals = np.asarray(residuals, dtype=float)
        edges = np.histogram_bin_edges(residuals, bins="auto")
        if len(edges) > 61:
            edges = np.histogram_bin_edges(residuals, bins=60)
        counts, edges = np.histogram(residuals, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        fig = go.Figure(go.Bar(
            x=centers, y=counts, width=np.diff(edges), name="Résidus",
            marker=dict(color=theme.rgba(theme.PRIMARY, 0.55), line=dict(color=theme.SURFACE, width=1)),
            hovertemplate="e ≈ %{x:.4g}<br>%{y} obs.<extra></extra>",
        ))
        mu, sigma = float(residuals.mean()), float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
        if sigma > 0:
            grid = np.linspace(edges[0], edges[-1], 200)
            fig.add_trace(go.Scatter(
                x=grid, y=scipy_stats.norm.pdf(grid, mu, sigma) * len(residuals) * width, mode="lines",
                name=f"Loi normale N({mu:.2g} ; {sigma:.2g}²)", line=dict(color=theme.ACCENT, width=2),
                hoverinfo="skip",
            ))
        fig.update_layout(xaxis_title="Résidu (y − ŷ)", yaxis_title="Effectif", bargap=0)
        return fig

    @staticmethod
    def qq_plot(residuals: np.ndarray) -> go.Figure:
        """Normal Q-Q plot of the residuals."""
        (theoretical, ordered), (slope, intercept, _r) = scipy_stats.probplot(np.asarray(residuals, dtype=float))
        idx = _sample_idx(len(ordered))
        fig = go.Figure()
        line_x = np.array([theoretical.min(), theoretical.max()])
        fig.add_trace(go.Scatter(
            x=line_x, y=slope * line_x + intercept, mode="lines", name="Normalité parfaite",
            line=dict(color=theme.REFERENCE_LINE, width=1.5, dash="dash"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=theoretical[idx], y=ordered[idx], mode="markers", name="Résidus ordonnés",
            marker=dict(color=theme.PRIMARY, size=6, opacity=0.7),
            hovertemplate="Quantile théorique = %{x:.3f}<br>Résidu = %{y:.4g}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Quantiles théoriques N(0, 1)", yaxis_title="Quantiles observés")
        return fig

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    @staticmethod
    def confusion_matrix(matrix: np.ndarray, labels: Sequence[str]) -> go.Figure:
        """Counts annotated with the row-normalised % (recall of each true class)."""
        matrix = np.asarray(matrix, dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) * 100
        text = [
            [f"<b>{int(matrix[i, j])}</b><br>{normalised[i, j]:.1f} %" for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
        labels = [theme.short_label(label, 18) for label in labels]
        fig = go.Figure(go.Heatmap(
            z=normalised, x=labels, y=labels, text=text, texttemplate="%{text}",
            textfont=dict(size=12 if len(labels) <= 6 else 9),
            customdata=matrix,
            zmin=0, zmax=100, colorscale=theme.SEQUENTIAL_BLUE,
            colorbar=theme.style_colorbar("% de la classe réelle"),
            xgap=2, ygap=2,
            hovertemplate="Réel <b>%{y}</b> → prédit <b>%{x}</b><br>%{customdata:.0f} obs. "
                          "(%{z:.1f} % de la classe)<extra></extra>",
        ))
        fig.update_xaxes(title="Classe prédite", side="bottom", type="category", showgrid=False, showline=False, ticks="")
        fig.update_yaxes(title="Classe réelle", autorange="reversed", type="category", showgrid=False,
                         showline=False, ticks="")
        return fig

    @staticmethod
    def roc_curves(curves: list[dict[str, Any]]) -> go.Figure:
        """ROC curves: ``curves`` = [{"name", "fpr", "tpr", "auc"}]. One curve → shaded area."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Hasard (AUC = 0,5)",
            line=dict(color=theme.REFERENCE_LINE, width=1.5, dash="dash"), hoverinfo="skip",
        ))
        single = len(curves) == 1
        for index, curve in enumerate(curves):
            color = curve.get("color") or theme.series_color(index)
            fig.add_trace(go.Scatter(
                x=curve["fpr"], y=curve["tpr"], mode="lines",
                name=f"{curve['name']} (AUC = {curve['auc']:.3f})",
                line=dict(color=color, width=2.5 if single else 2, dash=curve.get("dash", "solid")),
                fill="tozeroy" if single else None,
                fillcolor=theme.rgba(color, 0.16) if single else None,
                hovertemplate="TFP = %{x:.3f}<br>TVP = %{y:.3f}<extra>" + str(curve["name"]) + "</extra>",
            ))
        fig.update_layout(xaxis_title="Taux de faux positifs (1 − spécificité)",
                          yaxis_title="Taux de vrais positifs (rappel)")
        fig.update_xaxes(range=[-0.01, 1.01], constrain="domain")
        fig.update_yaxes(range=[-0.01, 1.03], scaleanchor="x", scaleratio=1)
        return fig

    @staticmethod
    def pr_curves(curves: list[dict[str, Any]], baseline: float | None = None) -> go.Figure:
        """Precision-Recall curves: ``curves`` = [{"name", "precision", "recall", "ap"}]."""
        fig = go.Figure()
        if baseline is not None:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[baseline, baseline], mode="lines",
                name=f"Classifieur aléatoire (prévalence {baseline:.2f})",
                line=dict(color=theme.REFERENCE_LINE, width=1.5, dash="dash"), hoverinfo="skip",
            ))
        single = len(curves) == 1
        for index, curve in enumerate(curves):
            color = curve.get("color") or theme.series_color(index)
            fig.add_trace(go.Scatter(
                x=curve["recall"], y=curve["precision"], mode="lines", line_shape="hv",
                name=f"{curve['name']} (AP = {curve['ap']:.3f})",
                line=dict(color=color, width=2.5 if single else 2),
                fill="tozeroy" if single else None,
                fillcolor=theme.rgba(color, 0.16) if single else None,
                hovertemplate="Rappel = %{x:.3f}<br>Précision = %{y:.3f}<extra>" + str(curve["name"]) + "</extra>",
            ))
        fig.update_layout(xaxis_title="Rappel", yaxis_title="Précision")
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.03])
        return fig

    @staticmethod
    def error_boxplot(names: Sequence[str], errors: Sequence[np.ndarray], colors: Sequence[str]) -> go.Figure:
        """Absolute-error distribution of several regressors, best model on top."""
        fig = go.Figure()
        for name, values, color in zip(names, errors, colors):
            values = np.asarray(values, dtype=float)
            fig.add_trace(go.Box(
                x=values[_sample_idx(len(values))], name=theme.short_label(name, 46), orientation="h",
                boxpoints="outliers", boxmean=True,
                marker=dict(color=color, size=3, opacity=0.6),
                line=dict(color=color, width=1.5), fillcolor=theme.rgba(color, 0.22),
                hovertemplate="%{x:.4g}<extra>" + str(name) + "</extra>",
            ))
        fig.update_layout(xaxis_title="Erreur absolue |y − ŷ|", showlegend=False)
        fig.update_yaxes(autorange="reversed", showline=False, ticks="")
        return fig

    @staticmethod
    def class_metrics(labels: Sequence[str], precision: Sequence[float], recall: Sequence[float],
                      f1: Sequence[float], support: Sequence[int]) -> go.Figure:
        """Grouped bars of precision / recall / F1 for each class."""
        fig = go.Figure()
        labels = [str(label) for label in labels]
        for index, (name, values) in enumerate((("Précision", precision), ("Rappel", recall), ("F1-score", f1))):
            fig.add_trace(go.Bar(
                x=labels, y=values, name=name,
                marker=dict(color=theme.series_color(index), line=dict(color=theme.SURFACE, width=2)),
                customdata=np.asarray(support),
                hovertemplate="Classe <b>%{x}</b><br>" + name + " = %{y:.3f}<br>support = %{customdata}<extra></extra>",
            ))
        fig.update_layout(barmode="group", bargroupgap=0.08, yaxis_title="Score", xaxis_title="Classe")
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 1.05])
        return fig

    @staticmethod
    def decision_boundary(
        xx: np.ndarray,
        yy: np.ndarray,
        zone: np.ndarray,
        points: np.ndarray,
        point_classes: np.ndarray,
        misclassified: np.ndarray,
        class_names: Sequence[str],
        axis_labels: tuple[str, str],
    ) -> go.Figure:
        """Predicted class regions over a 2D grid + test points coloured by true class."""
        n_classes = len(class_names)
        fig = go.Figure(go.Heatmap(
            x=xx, y=yy, z=zone, zmin=-0.5, zmax=n_classes - 0.5,
            colorscale=_discrete_colorscale(n_classes, 0.26), showscale=False, hoverinfo="skip",
            zsmooth=False,
        ))
        for index, name in enumerate(class_names):
            mask = point_classes == index
            if not mask.any():
                continue
            fig.add_trace(go.Scattergl(
                x=points[mask, 0], y=points[mask, 1], mode="markers", name=str(name),
                marker=dict(color=theme.series_color(index), symbol=theme.series_symbol(index), size=8,
                            line=dict(color=theme.SURFACE, width=1)),
                hovertemplate=f"Classe réelle : {name}<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>",
            ))
        if misclassified.any():
            fig.add_trace(go.Scattergl(
                x=points[misclassified, 0], y=points[misclassified, 1], mode="markers",
                name="Mal classés par le modèle",
                marker=dict(symbol="circle-open", size=14, color=theme.INK, line=dict(width=1.5)),
                hoverinfo="skip",
            ))
        fig.update_layout(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1])
        fig.update_xaxes(range=[float(xx.min()), float(xx.max())], showgrid=False)
        fig.update_yaxes(range=[float(yy.min()), float(yy.max())], showgrid=False)
        return fig

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------
    @staticmethod
    def feature_importance(
        names: Sequence[str], values: Sequence[float], axis_title: str,
        errors: Sequence[float] | None = None, signed: bool = False,
    ) -> go.Figure:
        """Horizontal bars, most important on top. ``signed`` uses the diverging poles."""
        values = np.asarray(values, dtype=float)
        if signed:
            colors = [theme.DIVERGING[-1][1] if v >= 0 else theme.DIVERGING[0][1] for v in values]
        else:
            colors = theme.PRIMARY
        fig = go.Figure(go.Bar(
            x=values, y=[theme.short_label(n, 34) for n in names], orientation="h",
            marker=dict(color=colors, line=dict(color=theme.SURFACE, width=2)),
            error_x=dict(type="data", array=list(errors), color=theme.INK_MUTED, thickness=1, width=3)
            if errors is not None else None,
            customdata=list(names),
            hovertemplate="<b>%{customdata}</b><br>%{x:.4g}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=axis_title, showlegend=False)
        fig.update_yaxes(autorange="reversed", showline=False, ticks="")
        if signed:
            fig.update_xaxes(zeroline=True, zerolinecolor=theme.INK_MUTED)
        return fig

    @staticmethod
    def loss_curve(loss: Sequence[float], validation: Sequence[float] | None = None) -> go.Figure:
        """MLP training loss per epoch; validation score in its own panel if available."""
        epochs = np.arange(1, len(loss) + 1)
        best = int(np.argmin(loss)) if len(loss) else 0
        if validation is not None and len(validation):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=[0.6, 0.4])
            fig.add_trace(_loss_trace(epochs, loss), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(validation) + 1), y=validation, mode="lines", name="Score de validation",
                line=dict(color=theme.series_color(2), width=2),
                hovertemplate="Époque %{x}<br>score = %{y:.4f}<extra></extra>",
            ), row=2, col=1)
            fig.update_yaxes(title_text="Perte (loss)", row=1, col=1)
            fig.update_yaxes(title_text="Score val.", row=2, col=1)
            fig.update_xaxes(title_text="Époque", row=2, col=1)
        else:
            fig = go.Figure(_loss_trace(epochs, loss))
            fig.update_layout(xaxis_title="Époque", yaxis_title="Perte (loss)")
        if len(loss):
            fig.add_trace(go.Scatter(
                x=[best + 1], y=[loss[best]], mode="markers", name=f"Perte minimale ({loss[best]:.4g})",
                marker=dict(color=theme.ACCENT, size=10, line=dict(color=theme.SURFACE, width=2)),
                hovertemplate="Époque %{x}<br>loss min = %{y:.5f}<extra></extra>",
            ))
        return fig

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    @staticmethod
    def elbow(ks: Sequence[int], inertia: Sequence[float], silhouette: Sequence[float | None],
              elbow_k: int | None, best_silhouette_k: int | None, chosen_k: int | None = None) -> go.Figure:
        """WCSS (top) and mean silhouette (bottom) against k — two panels, one x axis."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.55, 0.45])
        fig.add_trace(go.Scatter(
            x=list(ks), y=list(inertia), mode="lines+markers", name="Inertie intra-cluster (WCSS)",
            line=dict(color=theme.PRIMARY, width=2), marker=dict(size=8, line=dict(color=theme.SURFACE, width=2)),
            hovertemplate="k = %{x}<br>WCSS = %{y:.4g}<extra></extra>",
        ), row=1, col=1)
        sil = [np.nan if value is None else value for value in silhouette]
        fig.add_trace(go.Bar(
            x=list(ks), y=sil, name="Silhouette moyenne",
            marker=dict(color=[theme.ACCENT if k == best_silhouette_k else theme.rgba(theme.ACCENT, 0.45)
                               for k in ks], line=dict(color=theme.SURFACE, width=2)),
            hovertemplate="k = %{x}<br>silhouette = %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        if elbow_k in ks:
            y_value = inertia[list(ks).index(elbow_k)]
            fig.add_annotation(x=elbow_k, y=y_value, text=f"coude k = {elbow_k}", showarrow=True, arrowhead=0,
                               ax=40, ay=-34, font=dict(size=10, color=theme.INK), arrowcolor=theme.INK_MUTED,
                               row=1, col=1)
        if chosen_k is not None:
            fig.add_vline(x=chosen_k, line=dict(color=theme.INK_MUTED, width=1, dash="dot"))
            fig.add_annotation(x=chosen_k, y=1, yref="paper", text=f"modèle : k = {chosen_k}", showarrow=False,
                               yanchor="bottom", font=dict(size=9, color=theme.INK_MUTED))
        fig.update_yaxes(title_text="WCSS", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette", row=2, col=1)
        fig.update_xaxes(title_text="Nombre de clusters k", dtick=1, row=2, col=1)
        fig.update_layout(showlegend=False)
        return fig

    @staticmethod
    def silhouette_plot(values: np.ndarray, labels: np.ndarray, average: float) -> go.Figure:
        """Per-sample silhouette coefficients grouped and sorted by cluster ("knife" plot)."""
        fig = go.Figure()
        y_lower = 0
        clusters = sorted(np.unique(labels))
        for index, cluster in enumerate(clusters):
            cluster_values = np.sort(values[labels == cluster])[::-1]
            positions = np.arange(y_lower, y_lower + len(cluster_values))
            color = theme.series_color(index)
            fig.add_trace(go.Scatter(
                x=cluster_values, y=positions, mode="lines", fill="tozerox",
                line=dict(color=color, width=0.5), fillcolor=theme.rgba(color, 0.7),
                name=f"Cluster {cluster} (n = {len(cluster_values)})",
                hovertemplate=f"Cluster {cluster}<br>s = %{{x:.3f}}<extra></extra>",
            ))
            y_lower += len(cluster_values) + max(4, len(values) // 60)
        fig.add_vline(x=average, line=dict(color=theme.INK, width=1.5, dash="dash"),
                      annotation=dict(text=f"moyenne {average:.3f}", font=dict(size=10, color=theme.INK)),
                      annotation_position="top right")
        fig.update_layout(xaxis_title="Coefficient de silhouette s(i)", yaxis_title="Observations (par cluster)")
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(range=[min(-0.1, float(values.min()) - 0.02), 1.0])
        return fig

    @staticmethod
    def clusters_2d(points: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                    axis_labels: tuple[str, str]) -> go.Figure:
        """Clusters on the first two principal components, with projected centroids."""
        fig = go.Figure()
        idx = _sample_idx(len(points))
        points, labels = points[idx], labels[idx]
        for index, cluster in enumerate(sorted(np.unique(labels))):
            mask = labels == cluster
            fig.add_trace(go.Scattergl(
                x=points[mask, 0], y=points[mask, 1], mode="markers", name=f"Cluster {cluster}",
                marker=dict(color=theme.series_color(index), symbol=theme.series_symbol(index), size=7,
                            opacity=0.8, line=dict(color=theme.SURFACE, width=1)),
                hovertemplate=f"Cluster {cluster}<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode="markers+text", name="Centroïdes",
            text=[f"C{i}" for i in range(len(centers))], textposition="top center",
            textfont=dict(color=theme.INK, size=11),
            marker=dict(symbol="x", size=16, color=theme.INK, line=dict(color=theme.SURFACE, width=2)),
            hovertemplate="Centroïde %{text}<br>%{x:.3f}, %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1])
        return fig


def _loss_trace(epochs: np.ndarray, loss: Sequence[float]) -> go.Scatter:
    return go.Scatter(
        x=epochs, y=list(loss), mode="lines", name="Perte d'entraînement",
        line=dict(color=theme.PRIMARY, width=2),
        hovertemplate="Époque %{x}<br>loss = %{y:.5f}<extra></extra>",
    )
