"""
Static (PNG) diagnostic figures for the PDF report of the export module.

The interactive diagnostics are Plotly charts rendered by JavaScript, which
``xhtml2pdf`` cannot execute.  This module redraws the key diagnostics with
matplotlib (pure Python, no browser needed) and returns them as base64 data
URIs that the report template embeds with ``<img src="…">``.

The computations are the ones of ``apps.model_viz.services``: the PDF and the
web page always show the same numbers.  Light theme for print, same
categorical order as the web charts (class *i* keeps slot *i*).
"""

from __future__ import annotations

import base64
import io
import logging
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: never opens a window, thread-safe enough for Django

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from apps.datasets.models import Dataset  # noqa: E402
from apps.model_viz import services  # noqa: E402
from apps.preprocessing.services import load_prepared  # noqa: E402

logger = logging.getLogger(__name__)

# Light-surface steps of the same categorical hues as the web theme.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OVERFLOW = "#8a847c"
INK = "#1d1b19"
MUTED = "#6b6660"
GRID = "#e1e0d9"
DISPLAY_WIDTH = 500  # points in the A4 report (page width minus margins)
MAX_POINTS = 2500


def color(index: int) -> str:
    return PALETTE[index] if 0 <= index < len(PALETTE) else OVERFLOW


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b3aa")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _encode(fig, title: str, caption: str) -> dict[str, Any]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    width_in, height_in = fig.get_size_inches()
    plt.close(fig)
    return {
        "title": title,
        "caption": caption,
        "uri": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "width": DISPLAY_WIDTH,
        "height": int(DISPLAY_WIDTH * height_in / width_in),
    }


def _sample(n: int) -> np.ndarray:
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, MAX_POINTS, replace=False))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def confusion_figure(model, prepared) -> dict[str, Any]:
    X_test = services._dense(prepared.X_test)
    y_true = np.asarray(prepared.y_test)
    y_pred = np.asarray(model.estimator.predict(X_test))
    classes = services.class_order(model.estimator, y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    rows = matrix.sum(axis=1, keepdims=True)
    pct = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0) * 100

    size = min(6.2, 2.4 + 0.55 * len(classes))
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labels = [str(c) for c in classes]
    ax.set_xticks(range(len(labels)), labels, fontsize=8, color=MUTED, rotation=30 if len(labels) > 4 else 0)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=MUTED)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]}\n{pct[i, j]:.1f} %", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 55 else INK)
    ax.set_xlabel("Classe prédite", color=INK, fontsize=9)
    ax.set_ylabel("Classe réelle", color=INK, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.outline.set_visible(False)
    accuracy = np.trace(matrix) / max(matrix.sum(), 1) * 100
    return _encode(fig, "Matrice de confusion",
                   f"Effectifs et pourcentage de chaque classe réelle sur le jeu de test — exactitude {accuracy:.1f} %.")


def _curves_figure(curves: list[dict[str, Any]], x_key: str, y_key: str, score_key: str, score_label: str,
                   baseline: float | None, diagonal: bool, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _style(ax)
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Hasard (0,5)")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color=MUTED, linewidth=1, label=f"Aléatoire ({baseline:.2f})")
    dashes = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-.", "longdash": (0, (8, 3)),
              "longdashdot": (0, (8, 3, 2, 3))}
    for index, curve in enumerate(curves):
        style = dashes.get(curve.get("dash", "solid"), "-")
        hue = curve.get("pdf_color") or color(curve.get("slot", index))
        drawstyle = "steps-post" if x_key == "recall" else "default"
        x, y = curve[x_key], curve[y_key]
        if x_key == "recall":
            x, y = x[::-1], y[::-1]
        ax.plot(x, y, linestyle=style, color=hue, linewidth=2, drawstyle=drawstyle,
                label=f"{curve['name']} ({score_label} = {curve[score_key]:.3f})")
        if len(curves) == 1:
            ax.fill_between(x, y, step="post" if x_key == "recall" else None, color=hue, alpha=0.12)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right" if diagonal else "lower left")
    return fig


def roc_figure(model, prepared) -> dict[str, Any] | None:
    X_test = services._dense(prepared.X_test)
    y_true = np.asarray(prepared.y_test)
    scores = services.class_scores(model.estimator, X_test)
    if scores is None:
        return None
    classes = services.class_order(model.estimator, y_true, y_true)
    data = services.roc_pr_data(y_true, scores, classes)
    if not data["roc"]:
        return None
    for curve in data["roc"]:
        if curve.get("name") == "Micro-moyenne":
            curve["pdf_color"], curve["dash"] = INK, "dot"
    fig = _curves_figure(data["roc"], "fpr", "tpr", "auc", "AUC", None, True,
                         "Taux de faux positifs", "Taux de vrais positifs (rappel)")
    kind = "binaire" if len(classes) == 2 else "un-contre-tous par classe et micro-moyenne"
    return _encode(fig, "Courbe ROC", f"Courbe ROC {kind}, calculée sur le jeu de test.")


def regression_figures(model, prepared) -> list[dict[str, Any]]:
    X_test = services._dense(prepared.X_test)
    y_true = np.asarray(prepared.y_test, dtype=float)
    y_pred = np.asarray(model.estimator.predict(X_test), dtype=float)
    residuals = y_true - y_pred
    idx = _sample(len(y_true))
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1.0
    r2 = 1 - ss_res / ss_tot

    low, high = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    _style(ax)
    ax.plot([low, high], [low, high], linestyle="--", color=MUTED, linewidth=1, label="Idéal (y = x)")
    ax.scatter(y_true[idx], y_pred[idx], s=9, color=PALETTE[0], alpha=0.45, edgecolors="none",
               label=f"Prédictions (R² = {r2:.3f})")
    ax.set_xlabel("Valeur réelle (y)")
    ax.set_ylabel("Valeur prédite (ŷ)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    predicted_vs_actual = _encode(fig, "Valeurs prédites vs réelles",
                                  "Plus les points sont proches de la diagonale y = x, meilleur est le modèle.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    _style(ax)
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    ax.axhline(0, linestyle="--", color=MUTED, linewidth=1)
    for bound in (1.96 * sigma, -1.96 * sigma):
        ax.axhline(bound, linestyle=":", color="#b9b3aa", linewidth=1)
    ax.scatter(y_pred[idx], residuals[idx], s=9, color=PALETTE[0], alpha=0.45, edgecolors="none")
    ax.set_xlabel("Valeur prédite (ŷ)")
    ax.set_ylabel("Résidu (y − ŷ)")
    residual_plot = _encode(fig, "Résidus vs valeurs prédites",
                            "Repères 0 et ±1,96 écart-type : un nuage sans forme ni entonnoir indique "
                            "des erreurs homogènes (homoscédasticité).")
    return [predicted_vs_actual, residual_plot]


def importance_figure(model, prepared) -> dict[str, Any] | None:
    data = services.importance_data(model, prepared)
    if data is None:
        return None
    names, values = list(data["names"])[::-1], np.asarray(data["values"])[::-1]
    fig, ax = plt.subplots(figsize=(6.6, max(2.4, 0.3 * len(names) + 0.9)))
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    colors = [("#e34948" if v >= 0 else "#2a78d6") for v in values] if data["signed"] else PALETTE[0]
    ax.barh(range(len(names)), values, color=colors, height=0.68)
    ax.set_yticks(range(len(names)), [n if len(n) <= 30 else n[:29] + "…" for n in names], fontsize=7.5)
    if data["signed"]:
        ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(data["axis"])
    return _encode(fig, data["title"], data["subtitle"] + ".")


def clustering_figures(model, prepared) -> list[dict[str, Any]]:
    X = services._dense(prepared.X_train)
    estimator = services.final_estimator(model.estimator)
    labels = getattr(estimator, "labels_", None)
    if labels is None or len(labels) != len(X):
        labels = model.estimator.predict(X)
    labels = np.asarray(labels)
    figures = []

    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=0).fit(X)
        points = pca.transform(X)
        centers = getattr(estimator, "cluster_centers_", None)
        if centers is None:
            centers = np.vstack([X[labels == c].mean(axis=0) for c in sorted(np.unique(labels))])
        projected = pca.transform(centers)
        ratio = pca.explained_variance_ratio_
        markers = "os^Dv<>Ppx"
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        _style(ax)
        idx = _sample(len(points))
        for index, cluster in enumerate(sorted(np.unique(labels))):
            mask = labels[idx] == cluster
            ax.scatter(points[idx][mask, 0], points[idx][mask, 1], s=16, color=color(index),
                       marker=markers[index % len(markers)], alpha=0.75, edgecolors="none", label=f"Cluster {cluster}")
        ax.scatter(projected[:, 0], projected[:, 1], s=120, marker="X", color=INK, edgecolors="white",
                   linewidths=1.2, label="Centroïdes")
        ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f} %)")
        ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f} %)")
        ax.legend(fontsize=7, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        figures.append(_encode(fig, "Clusters sur les deux premiers axes PCA",
                               f"{ratio.sum() * 100:.1f} % de la variance représentée ; les croix noires sont les centroïdes finaux."))

    if len(X) >= 4:
        data = services.elbow_data(X)
        chosen = getattr(estimator, "n_clusters", None)
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for ax in (top, bottom):
            _style(ax)
        top.plot(data["ks"], data["inertia"], marker="o", color=PALETTE[0], linewidth=2)
        top.set_ylabel("Inertie (WCSS)")
        sil = [np.nan if s is None else s for s in data["silhouette"]]
        bottom.bar(data["ks"], sil, color=[PALETTE[1] if k == data["best_silhouette_k"] else "#f3c3ad" for k in data["ks"]])
        bottom.set_ylabel("Silhouette")
        bottom.set_xlabel("Nombre de clusters k")
        bottom.set_xticks(data["ks"])
        for ax in (top, bottom):
            if chosen is not None:
                ax.axvline(chosen, linestyle=":", color=MUTED, linewidth=1)
        if data["elbow_k"] in data["ks"]:
            k = data["elbow_k"]
            top.annotate(f"coude k = {k}", (k, data["inertia"][data["ks"].index(k)]), xytext=(12, 12),
                         textcoords="offset points", fontsize=8, color=INK)
        fig.tight_layout()
        figures.append(_encode(fig, "Méthode du coude et score de silhouette",
                               f"Coude suggéré : k = {data['elbow_k']} ; meilleure silhouette : "
                               f"k = {data['best_silhouette_k']}" + (f" ; modèle entraîné : k = {chosen}." if chosen else ".")))
    return figures


def comparison_figure(dataset: Dataset, prepared) -> dict[str, Any] | None:
    models = services.comparable_models(dataset, prepared)
    if len(models) < 2:
        return None
    if prepared.task_type == "classification":
        curves = services.comparison_curves(models, prepared)
        fig = _curves_figure(curves["roc"], "fpr", "tpr", "auc", "AUC", None, True,
                             "Taux de faux positifs", "Taux de vrais positifs (rappel)")
        kind = "classe positive" if curves["binary"] else "micro-moyenne sur les classes"
        return _encode(fig, "Comparaison des modèles — courbes ROC",
                       f"{len(curves['roc'])} algorithmes du classement sur le même jeu de test ({kind}).")
    rows = services.regression_errors(models, prepared)
    slot_of = {model.name: slot for slot, model in models}
    fig, ax = plt.subplots(figsize=(6.6, max(2.6, 0.55 * len(rows) + 1.0)))
    _style(ax)
    ax.grid(False, axis="y")
    parts = ax.boxplot([r["errors"] for r in rows][::-1], vert=False, patch_artist=True, showfliers=False,
                       widths=0.6, medianprops=dict(color=INK))
    for patch, row in zip(parts["boxes"], rows[::-1]):
        patch.set_facecolor(color(slot_of[row["name"]]))
        patch.set_alpha(0.55)
    ax.set_yticks(range(1, len(rows) + 1), [f"{r['name']} (R² {r['r2']:.3f})" for r in rows][::-1], fontsize=7.5)
    ax.set_xlabel("Erreur absolue |y − ŷ|")
    return _encode(fig, "Comparaison des modèles — erreurs absolues",
                   "Distribution des erreurs sur le jeu de test, du modèle le plus précis (en haut) au moins précis.")


# ---------------------------------------------------------------------------
# Entry point used by apps.export_inference.pdf_generator
# ---------------------------------------------------------------------------


def report_images(dataset: Dataset) -> list[dict[str, Any]]:
    """Diagnostic figures of the latest model; never raises (the PDF must always build)."""
    try:
        prepared = load_prepared(dataset)
        if prepared is None:
            return []
        model, _warning = services.load_trained_model(dataset, prepared)
        if model is None:
            return []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model.task_type == "classification":
                figures = [confusion_figure(model, prepared), roc_figure(model, prepared),
                           importance_figure(model, prepared)]
            elif model.task_type == "regression":
                figures = regression_figures(model, prepared) + [importance_figure(model, prepared)]
            else:
                figures = clustering_figures(model, prepared)
            figures.append(comparison_figure(dataset, prepared))
        return [figure for figure in figures if figure is not None]
    except Exception:  # noqa: BLE001 - a chart must never break the report download
        logger.exception("PDF diagnostic figures failed")
        return []
