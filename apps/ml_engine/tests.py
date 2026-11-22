"""
Comprehensive tests for ML Studio engine, algorithms, and views.
"""

from __future__ import annotations

import shutil
import tempfile
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from sklearn.datasets import make_classification, make_regression, make_blobs

from apps.datasets.models import Dataset
from apps.datasets.state import set_active_dataset, SESSION_KEY
from apps.preprocessing.config import PreprocessingConfig
from apps.preprocessing import services as prep_services
from apps.ml_engine.algorithms.registry import ModelRegistry
from apps.ml_engine.trainer import ModelTrainer
from apps.ml_engine.services import (
    load_latest_result,
    load_leaderboard,
    train_model_for_dataset,
)


class ModelRegistryTests(TestCase):
    def test_all_11_algorithms_registered(self):
        """Verify that all 11 algorithms required by the project are registered."""
        all_algos = ModelRegistry.all_algorithms()
        expected_keys = [
            "simple_linear_regression",
            "multiple_linear_regression",
            "polynomial_regression",
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "svm",
            "naive_bayes",
            "knn",
            "kmeans",
            "neural_network",
        ]
        for key in expected_keys:
            self.assertIn(key, all_algos, f"Algorithme manquant : {key}")
        self.assertEqual(len(all_algos), 11)

    def test_for_task_filtering(self):
        """Check filtering of algorithms by task type."""
        reg_algos = [a.key for a in ModelRegistry.for_task("regression")]
        self.assertIn("simple_linear_regression", reg_algos)
        self.assertIn("multiple_linear_regression", reg_algos)
        self.assertIn("polynomial_regression", reg_algos)
        self.assertIn("random_forest", reg_algos)
        self.assertIn("decision_tree", reg_algos)
        self.assertIn("svm", reg_algos)
        self.assertIn("knn", reg_algos)
        self.assertIn("neural_network", reg_algos)
        self.assertNotIn("logistic_regression", reg_algos)
        self.assertNotIn("kmeans", reg_algos)

        clf_algos = [a.key for a in ModelRegistry.for_task("classification")]
        self.assertIn("logistic_regression", clf_algos)
        self.assertIn("random_forest", clf_algos)
        self.assertIn("naive_bayes", clf_algos)
        self.assertIn("neural_network", clf_algos)
        self.assertNotIn("simple_linear_regression", clf_algos)
        self.assertNotIn("kmeans", clf_algos)

        clust_algos = [a.key for a in ModelRegistry.for_task("clustering")]
        self.assertEqual(len(clust_algos), 1)
        self.assertEqual(clust_algos[0], "kmeans")


class AlgorithmTrainingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Classification synthetic dataset
        X_c, y_c = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        cls.X_c_tr, cls.X_c_te = X_c[:80], X_c[80:]
        cls.y_c_tr, cls.y_c_te = y_c[:80], y_c[80:]
        cls.c_feat_names = [f"feat_{i}" for i in range(5)]

        # Regression synthetic dataset
        X_r, y_r = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        cls.X_r_tr, cls.X_r_te = X_r[:80], X_r[80:]
        cls.y_r_tr, cls.y_r_te = y_r[:80], y_r[80:]
        cls.r_feat_names = [f"reg_{i}" for i in range(5)]

        # Clustering dataset
        X_k, _ = make_blobs(n_samples=80, n_features=4, centers=3, random_state=42)
        cls.X_k = X_k
        cls.k_feat_names = [f"k_{i}" for i in range(4)]

    def test_regression_family(self):
        """Test Simple, Multiple, and Polynomial Linear Regression."""
        for key in ["simple_linear_regression", "multiple_linear_regression", "polynomial_regression"]:
            model = ModelRegistry.build_model(key, "regression", {})
            trainer = ModelTrainer(model, "regression", cv_folds=3)
            result = trainer.train_and_evaluate(
                self.X_r_tr, self.y_r_tr, self.X_r_te, self.y_r_te, self.r_feat_names
            )
            self.assertIn("r2", result.test_metrics)
            self.assertIn("mae", result.test_metrics)
            self.assertIn("rmse", result.test_metrics)
            self.assertIsNotNone(result.cv_mean)

    def test_classification_family(self):
        """Test Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM, KNN, MLP."""
        clf_keys = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "svm",
            "naive_bayes",
            "knn",
            "neural_network",
        ]
        for key in clf_keys:
            model = ModelRegistry.build_model(key, "classification", {})
            trainer = ModelTrainer(model, "classification", cv_folds=3)
            result = trainer.train_and_evaluate(
                self.X_c_tr, self.y_c_tr, self.X_c_te, self.y_c_te, self.c_feat_names
            )
            self.assertIn("accuracy", result.test_metrics, f"Failed on {key}")
            self.assertIn("f1", result.test_metrics, f"Failed on {key}")
            self.assertGreaterEqual(result.test_metrics["accuracy"], 0.0)

    def test_clustering_kmeans(self):
        """Test K-Means clustering algorithm."""
        model = ModelRegistry.build_model("kmeans", "clustering", {"n_clusters": 3})
        trainer = ModelTrainer(model, "clustering")
        result = trainer.train_and_evaluate(
            self.X_k, None, None, None, self.k_feat_names
        )
        self.assertIn("silhouette", result.test_metrics)
        self.assertIn("inertia", result.test_metrics)
        self.assertEqual(len(result.y_pred), len(self.X_k))


class MLEngineViewsTests(TestCase):
    """End-to-end integration tests for ML Engine views, persistence, and leaderboard."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-mle-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        # Create a small classification dataset
        csv_data = (
            "f1,f2,f3,target\n"
            "1.0,2.0,3.0,0\n"
            "1.2,2.1,3.1,0\n"
            "0.9,1.8,2.9,0\n"
            "1.1,2.2,3.2,0\n"
            "5.0,6.0,7.0,1\n"
            "5.2,6.1,7.2,1\n"
            "4.9,5.9,6.8,1\n"
            "5.1,6.2,7.1,1\n"
        )
        upload = SimpleUploadedFile("test_clf.csv", csv_data.encode("utf-8"), content_type="text/csv")
        self.dataset = Dataset.objects.create(name="Test Classification", file=upload, n_rows=8, n_cols=4)

        # Preprocess it
        config = PreprocessingConfig(
            target="target",
            features=["f1", "f2", "f3"],
            task_type="classification",
            test_size=0.25,
            random_state=42,
        )
        prep_services.prepare(self.dataset, config)

    def test_index_view_with_preprocessed_dataset(self):
        """Index view shows available algorithms for classification."""
        session = self.client.session
        session[SESSION_KEY] = {
            "dataset_id": str(self.dataset.id),
            "dataset_name": self.dataset.name,
            "preprocessing_done": True,
            "task_type": "classification",
        }
        session.save()

        response = self.client.get(reverse("ml_engine:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Moteur d'Apprentissage Automatique")
        self.assertContains(response, "Forêt Aléatoire")
        self.assertContains(response, "Régression Logistique")

    def test_params_view_htmx(self):
        """Params view returns partial HTML for SVM."""
        response = self.client.get(reverse("ml_engine:params") + "?algo=svm")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Type de Noyau (Kernel)")
        self.assertContains(response, "Pénalité d&#x27;erreur (C)")
        self.assertContains(response, 'value="1.0"')

    def test_training_workflow_and_leaderboard(self):
        """Train random_forest, verify leaderboard, train logistic_regression, verify rank."""
        session = self.client.session
        session[SESSION_KEY] = {
            "dataset_id": str(self.dataset.id),
            "dataset_name": self.dataset.name,
            "preprocessing_done": True,
            "task_type": "classification",
        }
        session.save()

        # 1. Train Random Forest
        res_post = self.client.post(reverse("ml_engine:train"), {
            "algo_key": "random_forest",
            "n_estimators": 50,
            "max_depth": 5,
            "cv_folds": 2,
        })
        self.assertEqual(res_post.status_code, 302)

        # Check session
        updated_session = self.client.session[SESSION_KEY]
        self.assertTrue(updated_session["model_trained"])
        self.assertEqual(updated_session["active_step"], 4)

        # Check latest result & leaderboard
        latest = load_latest_result(self.dataset)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["algo_key"], "random_forest")

        board = load_leaderboard(self.dataset)
        self.assertEqual(len(board), 1)
        self.assertTrue(board[0]["is_best"])

        # 2. Train Logistic Regression
        self.client.post(reverse("ml_engine:train"), {
            "algo_key": "logistic_regression",
            "C": 1.0,
            "penalty": "l2",
            "cv_folds": 2,
        })

        board = load_leaderboard(self.dataset)
        self.assertEqual(len(board), 2)
        # Exactly one champion
        champions = [b for b in board if b["is_best"]]
        self.assertEqual(len(champions), 1)

        # 3. Reset models
        res_reset = self.client.post(reverse("ml_engine:reset"))
        self.assertEqual(res_reset.status_code, 302)
        board_cleared = load_leaderboard(self.dataset)
        self.assertEqual(len(board_cleared), 0)
