"""
Tests for model export, standalone pipeline serialization, and predictions CSV.
"""

from __future__ import annotations

import shutil
import tempfile
import joblib
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from sklearn.pipeline import Pipeline

from apps.datasets.models import Dataset
from apps.datasets.state import SESSION_KEY
from apps.preprocessing.config import PreprocessingConfig
from apps.preprocessing import services as prep_services
from apps.ml_engine.services import train_model_for_dataset
from apps.export_inference.exporter import (
    get_standalone_pipeline,
    export_model_joblib,
    generate_predictions_csv,
    generate_python_snippet,
)


class ExporterTests(TestCase):
    """Test suite for pipeline packaging, joblib export, and CSV predictions."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-export-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        # Create dataset
        csv_data = (
            "age,income,category,target\n"
            "25,50000,A,0\n"
            "30,60000,B,0\n"
            "22,45000,A,0\n"
            "28,52000,B,0\n"
            "55,120000,A,1\n"
            "60,150000,B,1\n"
            "58,130000,A,1\n"
            "62,160000,B,1\n"
        )
        upload = SimpleUploadedFile("customers.csv", csv_data.encode("utf-8"), content_type="text/csv")
        self.dataset = Dataset.objects.create(name="Customer Churn", file=upload, n_rows=8, n_cols=4)

        # Preprocess dataset
        config = PreprocessingConfig(
            target="target",
            features=["age", "income", "category"],
            task_type="classification",
            test_size=0.25,
            random_state=42,
        )
        prep_services.prepare(self.dataset, config)

        # Train a model (Logistic Regression)
        train_model_for_dataset(
            dataset=self.dataset,
            algo_key="logistic_regression",
            params={"C": 1.0, "penalty": "l2"},
            cv_folds=2,
        )

    def test_standalone_pipeline_creation_and_raw_prediction(self):
        """Verify that bundled pipeline can predict on raw unseen DataFrames directly."""
        pipeline = get_standalone_pipeline(self.dataset)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn("preprocessor", pipeline.named_steps)
        self.assertIn("model", pipeline.named_steps)

        # Predict on raw data with unencoded categories
        raw_input = pd.DataFrame([{
            "age": 45,
            "income": 80000,
            "category": "A",
        }])
        pred = pipeline.predict(raw_input)
        self.assertEqual(len(pred), 1)

    def test_joblib_file_export_and_reload(self):
        """Export to .joblib, reload with joblib.load and predict."""
        exported_path = export_model_joblib(self.dataset)
        self.assertTrue(exported_path.exists())

        # Reload independently
        reloaded = joblib.load(exported_path)
        self.assertIsInstance(reloaded, Pipeline)

        raw_sample = pd.DataFrame([{"age": 29, "income": 55000, "category": "B"}])
        pred = reloaded.predict(raw_sample)
        self.assertIn(pred[0], [0, 1])

    def test_generate_predictions_csv(self):
        """Generate test set predictions CSV and verify its content."""
        csv_path = generate_predictions_csv(self.dataset)
        self.assertTrue(csv_path.exists())

        df_preds = pd.read_csv(csv_path)
        self.assertIn("actual_value", df_preds.columns)
        self.assertIn("predicted_value", df_preds.columns)
        self.assertIn("is_correct", df_preds.columns)
        self.assertGreater(len(df_preds), 0)

    def test_download_views(self):
        """Test HTTP download endpoints for model and CSV."""
        session = self.client.session
        session[SESSION_KEY] = {"dataset_id": str(self.dataset.id)}
        session.save()

        # Download model
        res_model = self.client.get(reverse("export_inference:download_model"))
        self.assertEqual(res_model.status_code, 200)
        self.assertIn("attachment", res_model.get("Content-Disposition", ""))

        # Download CSV
        res_csv = self.client.get(reverse("export_inference:download_predictions"))
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("attachment", res_csv.get("Content-Disposition", ""))

    def test_inference_schema_and_prediction(self):
        """Test dynamic input schema generation and live prediction computation."""
        from apps.export_inference.predictor import get_feature_input_schema, predict_from_inputs

        schema = get_feature_input_schema(self.dataset)
        names = [f["name"] for f in schema]
        self.assertIn("age", names)
        self.assertIn("income", names)
        self.assertIn("category", names)

        # Single prediction
        res = predict_from_inputs(self.dataset, {
            "age": 35,
            "income": 70000,
            "category": "A",
        })
        self.assertIn("prediction", res)
        self.assertEqual(res["task_type"], "classification")
        self.assertIsNotNone(res["confidence"])

    def test_predict_view_endpoint(self):
        """Test HTMX predict endpoint."""
        session = self.client.session
        session[SESSION_KEY] = {"dataset_id": str(self.dataset.id)}
        session.save()

        res = self.client.post(reverse("export_inference:predict"), {
            "age": 28,
            "income": 50000,
            "category": "B",
        })
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Résultat de la Prédiction")
        self.assertContains(res, "Indice de Confiance")

    def test_pdf_report_generation(self):
        """Verify that xhtml2pdf compiles a valid PDF starting with %PDF."""
        from apps.export_inference.pdf_generator import generate_pdf_report_bytes

        pdf_bytes = generate_pdf_report_bytes(self.dataset)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_download_pdf_view(self):
        """Test PDF download endpoint returns 200 and application/pdf."""
        session = self.client.session
        session[SESSION_KEY] = {"dataset_id": str(self.dataset.id)}
        session.save()

        res = self.client.get(reverse("export_inference:download_pdf"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "application/pdf")
        self.assertIn("attachment", res.get("Content-Disposition", ""))

    def test_export_inference_index_view_rendering(self):
        """Test full index view rendering with all export cards and playground."""
        session = self.client.session
        session[SESSION_KEY] = {"dataset_id": str(self.dataset.id)}
        session.save()

        res = self.client.get(reverse("export_inference:index"))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Pipeline Scikit-Learn Autonome")
        self.assertContains(res, "Prédictions du Jeu de Test")
        self.assertContains(res, "Bac à Sable")
        self.assertContains(res, "Consommation du Pipeline en Python")
