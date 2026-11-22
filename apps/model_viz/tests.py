"""Tests for the diagnostics module: figures, computations, M4 artefact contract and views."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from apps.datasets import cache
from apps.datasets.models import Dataset
from apps.model_viz import services
from apps.model_viz.diagnostic_plots import ModelDiagnosticPlots as Plots
from apps.preprocessing.config import PreprocessingConfig
from apps.preprocessing.pipeline import DataPreprocessor
from apps.preprocessing.services import load_prepared, save_prepared


def prepared_from(frame: pd.DataFrame, target: str | None, task: str, scaler: str = "standard"):
    config = PreprocessingConfig(target=target, task_type=task, scaler=scaler)
    return DataPreprocessor(frame, target).build_and_fit_transform(config)


def regression_frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "y": 3 * x + 2 + rng.normal(0, 1, rows)})


def classification_frame(rows: int = 240, classes: int = 3) -> pd.DataFrame:
    X, y = make_blobs(n_samples=rows, centers=classes, n_features=4, random_state=0)
    frame = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    frame["label"] = [f"c{v}" for v in y]
    return frame


class SingleFeature:
    """Minimal stand-in for M4's SingleFeatureLinearRegressor (feature_idx + regressor_)."""

    def __init__(self, feature_idx: int = 0) -> None:
        self.feature_idx = feature_idx
        self.regressor_ = LinearRegression()

    def fit(self, X, y):
        self.regressor_.fit(np.asarray(X)[:, [self.feature_idx]], y)
        return self

    def predict(self, X):
        return self.regressor_.predict(np.asarray(X)[:, [self.feature_idx]])


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------


class RegressionDiagnosticsTests(SimpleTestCase):
    def setUp(self) -> None:
        self.prepared = prepared_from(regression_frame(), "y", "regression", scaler="none")
        self.model = LinearRegression().fit(self.prepared.X_train, self.prepared.y_train)
        self.trained = services.TrainedModel(self.model, "OLS", "multiple_linear_regression", "regression")

    def test_fit_curve_confidence_band_matches_the_textbook_formula(self) -> None:
        curve = services.fit_curve_data(self.model, self.prepared)
        x = np.asarray(self.prepared.X_train)[:, 0]
        y = np.asarray(self.prepared.y_train, dtype=float)
        n = len(x)
        residuals = y - self.model.predict(self.prepared.X_train)
        s = np.sqrt(residuals @ residuals / (n - 2))
        sxx = ((x - x.mean()) ** 2).sum()
        from scipy import stats

        t = stats.t.ppf(0.975, n - 2)
        x0 = curve["grid"][17]
        expected = t * s * np.sqrt(1 / n + (x0 - x.mean()) ** 2 / sxx)
        self.assertAlmostEqual(curve["upper"][17] - curve["fitted"][17], expected, places=6)
        self.assertTrue(np.all(curve["lower"] <= curve["fitted"]))

    def test_fit_curve_for_m4_single_feature_regressor_on_multi_feature_data(self) -> None:
        frame = regression_frame()
        frame["noise"] = np.random.default_rng(1).normal(size=len(frame))
        prepared = prepared_from(frame, "y", "regression")
        model = SingleFeature(feature_idx=0).fit(prepared.X_train, prepared.y_train)
        curve = services.fit_curve_data(model, prepared)
        self.assertIsNotNone(curve)
        self.assertIsNotNone(curve["upper"])
        self.assertIn("(normalisé)", curve["x_label"])

    def test_no_fit_curve_for_multiple_regression(self) -> None:
        frame = regression_frame()
        frame["z"] = np.random.default_rng(1).normal(size=len(frame))
        prepared = prepared_from(frame, "y", "regression")
        model = LinearRegression().fit(prepared.X_train, prepared.y_train)
        self.assertIsNone(services.fit_curve_data(model, prepared))

    def test_regression_charts_and_summary(self) -> None:
        result = services.build_diagnostics(self.trained, self.prepared)
        titles = [chart.title for chart in result["charts"]]
        self.assertEqual(titles[:5], [
            "Courbe d'ajustement", "Valeurs prédites vs réelles", "Résidus vs valeurs prédites",
            "Distribution des résidus", "Q-Q plot des résidus",
        ])
        self.assertIn("Coefficients du modèle", titles)
        self.assertGreater(result["summary"]["r2"], 0.9)
        self.assertFalse(result["needs_permutation"])
        for chart in result["charts"]:
            json.loads(str(chart.payload))


class ClassificationDiagnosticsTests(SimpleTestCase):
    def test_binary_roc_auc_matches_sklearn(self) -> None:
        frame = classification_frame(classes=2)
        prepared = prepared_from(frame, "label", "classification")
        model = LogisticRegression(max_iter=500).fit(prepared.X_train, prepared.y_train)
        y_test = np.asarray(prepared.y_test)
        scores = model.predict_proba(prepared.X_test)
        data = services.roc_pr_data(y_test, scores, list(model.classes_))
        expected = roc_auc_score(y_test == model.classes_[1], scores[:, 1])
        self.assertAlmostEqual(data["roc"][0]["auc"], expected, places=6)
        self.assertAlmostEqual(data["baseline"], float(np.mean(y_test == model.classes_[1])))

    def test_multiclass_curves_one_vs_rest_plus_micro_average(self) -> None:
        prepared = prepared_from(classification_frame(classes=3), "label", "classification")
        model = RandomForestClassifier(20, random_state=0).fit(prepared.X_train, prepared.y_train)
        data = services.roc_pr_data(np.asarray(prepared.y_test), model.predict_proba(prepared.X_test),
                                    list(model.classes_))
        self.assertEqual([c["name"] for c in data["roc"]], ["c0", "c1", "c2", "Micro-moyenne"])
        self.assertEqual(len(data["pr"]), 3)

    def test_decision_function_is_used_without_probabilities(self) -> None:
        prepared = prepared_from(classification_frame(classes=2), "label", "classification")
        model = SVC().fit(prepared.X_train, prepared.y_train)
        trained = services.TrainedModel(model, "SVM", "svm", "classification")
        result = services.build_diagnostics(trained, prepared)
        self.assertTrue(result["has_scores"])
        self.assertIn("Courbe ROC", [c.title for c in result["charts"]])
        self.assertTrue(result["needs_permutation"])
        self.assertTrue(result["can_draw_boundary"])

    def test_confusion_matrix_counts_every_test_row(self) -> None:
        prepared = prepared_from(classification_frame(), "label", "classification")
        model = LogisticRegression(max_iter=500).fit(prepared.X_train, prepared.y_train)
        trained = services.TrainedModel(model, "LR", "logistic_regression", "classification")
        result = services.build_diagnostics(trained, prepared)
        self.assertEqual(result["charts"][0].title, "Matrice de confusion")
        self.assertEqual(result["summary"]["n_test"], prepared.test_rows)
        self.assertEqual(result["summary"]["n_classes"], 3)
        self.assertIn("Poids des variables", [c.title for c in result["charts"]])

    def test_decision_boundary_on_pca(self) -> None:
        prepared = prepared_from(classification_frame(), "label", "classification")
        model = LogisticRegression(max_iter=500).fit(prepared.X_train, prepared.y_train)
        chart = services.decision_boundary(services.TrainedModel(model, "LR", "", "classification"), prepared)
        payload = json.loads(str(chart.payload))
        self.assertEqual(payload["data"][0]["type"], "heatmap")
        self.assertIn("PCA", chart.subtitle)

    def test_boundary_failure_is_a_user_facing_error(self) -> None:
        frame = classification_frame()
        prepared = prepared_from(frame, "label", "classification", scaler="minmax")
        model = MultinomialNB().fit(prepared.X_train, prepared.y_train)  # PCA scores are negative
        with self.assertRaises(services.DiagnosticsError):
            services.decision_boundary(services.TrainedModel(model, "NB", "", "classification"), prepared)

    def test_permutation_importance_and_loss_curve_for_mlp(self) -> None:
        prepared = prepared_from(classification_frame(), "label", "classification")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # short training on purpose
            model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=60, random_state=0).fit(
                prepared.X_train, prepared.y_train)
        trained = services.TrainedModel(model, "MLP", "neural_network", "classification")
        result = services.build_diagnostics(trained, prepared)
        self.assertIn("Convergence du réseau de neurones", [c.title for c in result["charts"]])
        self.assertTrue(result["needs_permutation"])
        chart = services.permutation_chart(trained, prepared)
        self.assertEqual(len(json.loads(str(chart.payload))["data"][0]["y"]), 4)


class ClusteringDiagnosticsTests(SimpleTestCase):
    def test_elbow_point_finds_the_true_number_of_blobs(self) -> None:
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=3)
        data = services.elbow_data(X)
        self.assertEqual(data["ks"], list(range(2, 11)))
        self.assertEqual(data["elbow_k"], 4)
        self.assertEqual(data["best_silhouette_k"], 4)
        self.assertEqual(data["inertia"], sorted(data["inertia"], reverse=True))

    def test_clustering_charts(self) -> None:
        X, _ = make_blobs(n_samples=150, centers=3, n_features=3, cluster_std=0.5, random_state=3)
        prepared = prepared_from(pd.DataFrame(X, columns=["a", "b", "c"]), None, "clustering")
        model = KMeans(3, random_state=42, n_init="auto").fit(prepared.X_train)
        result = services.build_diagnostics(services.TrainedModel(model, "K-Means", "kmeans", "clustering"), prepared)
        self.assertEqual([c.title for c in result["charts"]],
                         ["Clusters projetés sur les 2 premiers axes PCA", "Silhouette par observation"])
        self.assertGreater(result["summary"]["silhouette"], 0.5)
        chart, _data = services.elbow_chart(prepared, chosen_k=3)
        self.assertIn("coude suggéré : k = 3", chart.subtitle)


class FigureTests(SimpleTestCase):
    def test_confusion_annotations_show_counts_and_row_percentages(self) -> None:
        fig = Plots.confusion_matrix(np.array([[8, 2], [1, 9]]), ["a", "b"])
        self.assertEqual(fig.data[0].text[0][0], "<b>8</b><br>80.0 %")
        self.assertAlmostEqual(fig.data[0].z[1][1], 90.0)

    def test_elbow_uses_two_panels_not_two_y_axes(self) -> None:
        fig = Plots.elbow([2, 3, 4], [30.0, 12.0, 10.0], [0.5, 0.7, 0.6], 3, 3)
        self.assertEqual(fig.data[0].yaxis, "y")
        self.assertEqual(fig.data[1].yaxis, "y2")
        self.assertIsNone(fig.layout.yaxis2.overlaying)

    def test_signed_importance_uses_the_diverging_poles(self) -> None:
        fig = Plots.feature_importance(["a", "b"], [0.5, -0.2], "β", signed=True)
        self.assertEqual(len(set(fig.data[0].marker.color)), 2)


# ---------------------------------------------------------------------------
# Artefact contract with the ML engine (M4)
# ---------------------------------------------------------------------------


class ModelVizIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-viz-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self) -> None:
        cache.invalidate()

    def preprocess_iris(self) -> Dataset:
        self.client.post(reverse("datasets:sample"), {"slug": "iris"})
        self.client.post(reverse("preprocessing:apply"), {
            "target": "target", "task_type": "classification",
            "features": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "numeric_strategy": "mean", "categorical_strategy": "most_frequent", "constant_value": "Inconnu",
            "missing_threshold": "1.0", "outlier_method": "none", "outlier_factor": "1.5",
            "encoder": "onehot", "scaler": "standard", "test_size": "0.2", "stratify": "on",
        })
        return Dataset.objects.get()

    def save_model(self, dataset: Dataset, estimator, name: str = "Forêt Aléatoire", task: str = "classification"):
        """Write artefacts exactly where apps.ml_engine.services puts them."""
        paths = services.model_paths(dataset)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, paths["model"])
        paths["result"].write_text(json.dumps({
            "algo_key": "random_forest", "algo_name": name, "task_type": task,
            "test_metrics": {"accuracy": 0.97},
        }), encoding="utf-8")

    def test_page_without_dataset(self) -> None:
        response = self.client.get(reverse("model_viz:index"))
        self.assertContains(response, "Aucun jeu de données actif")

    def test_page_requires_preprocessing(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "iris"})
        self.assertContains(self.client.get(reverse("model_viz:index")), "Prétraitement requis")

    def test_page_without_model_links_to_training(self) -> None:
        self.preprocess_iris()
        response = self.client.get(reverse("model_viz:index"))
        self.assertContains(response, "Aucun modèle entraîné")
        self.assertContains(response, reverse("ml_engine:index"))

    def test_trained_model_is_loaded_and_diagnosed(self) -> None:
        dataset = self.preprocess_iris()
        prepared = load_prepared(dataset)
        model = RandomForestClassifier(30, random_state=0).fit(prepared.X_train, np.asarray(prepared.y_train))
        self.save_model(dataset, model)
        response = self.client.get(reverse("model_viz:index"))
        self.assertEqual(response.status_code, 200)
        for text in ("Forêt Aléatoire", "Matrice de confusion", "Courbe ROC", "Courbe Précision-Rappel",
                     "Importance des variables", reverse("model_viz:boundary")):
            self.assertContains(response, text)
        self.assertNotContains(response, reverse("model_viz:permutation"))

        boundary = self.client.get(reverse("model_viz:boundary"))
        self.assertContains(boundary, "Frontière de décision (projection 2D)")

    def test_stale_model_is_ignored_after_new_preprocessing(self) -> None:
        dataset = self.preprocess_iris()
        prepared = load_prepared(dataset)
        model = LogisticRegression(max_iter=300).fit(prepared.X_train, np.asarray(prepared.y_train))
        self.save_model(dataset, model)
        old = time.time() - 60
        os.utime(services.model_paths(dataset)["model"], (old, old))
        save_prepared(dataset, prepared)  # preprocessing re-run after training
        response = self.client.get(reverse("model_viz:index"))
        self.assertContains(response, "Le prétraitement a été relancé après le dernier entraînement")

    def test_feature_count_mismatch_is_reported(self) -> None:
        dataset = self.preprocess_iris()
        prepared = load_prepared(dataset)
        model = LogisticRegression(max_iter=300).fit(prepared.X_train[:, :2], np.asarray(prepared.y_train))
        self.save_model(dataset, model)
        response = self.client.get(reverse("model_viz:index"))
        self.assertContains(response, "Le modèle attend 2 variables")

    def test_lazy_endpoints_without_model(self) -> None:
        self.preprocess_iris()
        self.assertContains(self.client.get(reverse("model_viz:boundary")), "Aucun modèle entraîné à jour")
        self.assertContains(self.client.get(reverse("model_viz:elbow")), "Méthode du coude")
