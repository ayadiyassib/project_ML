"""Tests for the leak-free preprocessing engine, configuration and artefacts.

The most important test here is ``test_scaler_statistics_come_from_the_training_split``:
it recomputes the split independently and asserts that the fitted scaler
statistics equal the *training* mean, not the full-data mean.  That is the
regression test protecting us from the data-leakage bug described in the spec.
"""

from __future__ import annotations

import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from sklearn.model_selection import train_test_split

from apps.datasets import cache
from apps.datasets import services as dataset_services
from apps.datasets.models import Dataset
from apps.datasets.state import SESSION_KEY
from apps.preprocessing import services
from apps.preprocessing.config import PreprocessingConfig, PreprocessingError
from apps.preprocessing.pipeline import (
    DataPreprocessor,
    clip_frame,
    infer_task_type,
    iqr_bounds,
    outlier_mask,
)


class PreprocessingTestCase(TestCase):
    """Shared fixtures: a temporary MEDIA_ROOT and small CSV helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-prep-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self) -> None:
        cache.invalidate()

    # -- helpers ---------------------------------------------------------
    def upload_csv(self, name: str, frame: pd.DataFrame) -> Dataset:
        payload = frame.to_csv(index=False).encode("utf-8")
        return dataset_services.create_dataset_from_upload(
            SimpleUploadedFile(name, payload)
        )

    def numeric_dataset(self, rows: int = 240) -> Dataset:
        """Deterministic numeric dataset used by the leakage tests."""
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {
                "x1": rng.normal(10, 3, rows),
                "x2": rng.normal(-4, 1.5, rows),
            }
        )
        frame["y"] = 0.5 * frame["x1"] - 0.25 * frame["x2"] + rng.normal(0, 1, rows)
        return self.upload_csv("numerique.csv", frame)

    def classification_dataset(self, rows: int = 120) -> Dataset:
        """Numeric features with a missing value and a categorical label."""
        frame = pd.DataFrame(
            {
                "age": [float(20 + (index % 40)) for index in range(rows)],
                "income": [float(30 + (index * 7) % 90) for index in range(rows)],
                "segment": ["alpha" if index % 2 else "beta" for index in range(rows)],
                "label": ["oui" if index % 3 else "non" for index in range(rows)],
            }
        )
        frame.loc[::7, "age"] = np.nan
        frame.loc[::9, "segment"] = None
        return self.upload_csv("classification.csv", frame)

    def titanic(self) -> Dataset:
        return dataset_services.create_dataset_from_sample("titanic")


class ConfigTests(PreprocessingTestCase):
    def test_absent_checkboxes_default_to_false(self) -> None:
        config = PreprocessingConfig.from_form(
            {
                "target": "label",
                "features": ["age"],
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
            },
            ["age", "label"],
        )
        self.assertFalse(config.drop_duplicates)
        self.assertFalse(config.stratify)

    def test_present_checkboxes_are_enabled(self) -> None:
        config = PreprocessingConfig.from_form(
            {
                "target": "label",
                "features": ["age"],
                "drop_duplicates": "on",
                "stratify": "on",
            },
            ["age", "label"],
        )
        self.assertTrue(config.drop_duplicates)
        self.assertTrue(config.stratify)

    def test_defaults_are_sane(self) -> None:
        config = PreprocessingConfig.from_form(
            {"target": "label", "features": ["age"]}, ["age", "label"]
        )
        self.assertEqual(config.numeric_strategy, "mean")
        self.assertEqual(config.categorical_strategy, "most_frequent")
        self.assertEqual(config.constant_value, "Inconnu")
        self.assertEqual(config.scaler, "standard")
        self.assertEqual(config.encoder, "onehot")
        self.assertEqual(config.outlier_method, "none")
        self.assertEqual(config.outlier_factor, 1.5)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.random_state, 42)

    def test_multiple_features_are_read_from_getlist(self) -> None:
        class MultiDict(dict):
            def getlist(self, key):
                value = self.get(key)
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [] if value is None else [value]

        config = PreprocessingConfig.from_form(
            MultiDict(
                {
                    "target": "label",
                    "features": ["age", "income"],
                    "encoder": "onehot",
                    "scaler": "standard",
                    "test_size": "0.2",
                }
            ),
            ["age", "income", "label"],
        )
        self.assertEqual(config.features, ["age", "income"])

    def test_target_is_removed_from_the_feature_list(self) -> None:
        config = PreprocessingConfig.from_form(
            {"target": "label", "features": ["age", "label"]}, ["age", "label"]
        )
        self.assertEqual(config.features, ["age"])

    def test_no_feature_selected_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            PreprocessingConfig.from_form({"target": "label", "features": []}, ["age", "label"])

    def test_unknown_target_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            PreprocessingConfig.from_form(
                {"target": "inconnue", "features": ["age"]}, ["age", "label"]
            )

    def test_invalid_choices_raise(self) -> None:
        for field, value in (
            ("scaler", "magique"),
            ("encoder", "base64"),
            ("numeric_strategy", "moyenne"),
            ("outlier_method", "supprimer"),
        ):
            payload = {"target": "label", "features": ["age"], field: value}
            with self.assertRaises(PreprocessingError, msg=field):
                PreprocessingConfig.from_form(payload, ["age", "label"])

    def test_out_of_range_numbers_raise(self) -> None:
        for field, value in (("test_size", "0.9"), ("outlier_factor", "9"), ("missing_threshold", "-1")):
            payload = {"target": "label", "features": ["age"], field: value}
            with self.assertRaises(PreprocessingError, msg=field):
                PreprocessingConfig.from_form(payload, ["age", "label"])

    def test_missing_target_forces_clustering(self) -> None:
        config = PreprocessingConfig.from_form(
            {"target": "", "features": ["age"], "task_type": "regression"}, ["age"]
        )
        self.assertIsNone(config.target)
        self.assertEqual(config.task_type, "clustering")

    def test_dict_roundtrip(self) -> None:
        config = PreprocessingConfig(
            target="label",
            features=["age", "income"],
            task_type="classification",
            numeric_strategy="knn",
            outlier_method="clip",
            scaler="robust",
            encoder="ordinal",
            test_size=0.25,
        )
        restored = PreprocessingConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertEqual(PreprocessingConfig.from_dict(None), PreprocessingConfig())
        self.assertEqual(
            PreprocessingConfig.from_dict({"inconnu": 1}).features, PreprocessingConfig().features
        )


class TaskInferenceTests(PreprocessingTestCase):
    def test_inference_rules(self) -> None:
        self.assertEqual(infer_task_type(None), "clustering")
        self.assertEqual(infer_task_type(pd.Series([True, False, True])), "classification")
        self.assertEqual(infer_task_type(pd.Series([0, 1, 2, 1, 0])), "classification")
        self.assertEqual(
            infer_task_type(pd.Series(np.linspace(0.1, 0.9, 40))), "regression"
        )
        self.assertEqual(infer_task_type(pd.Series(["a", "b", "a"])), "classification")

    def test_default_config_uses_the_dataset_task_hint(self) -> None:
        iris = dataset_services.create_dataset_from_sample("iris")
        config = services.default_config(iris, dataset_services.load_dataframe(iris))
        self.assertEqual(config.target, "target")
        self.assertEqual(config.task_type, "classification")
        self.assertIn("sepal length (cm)", config.features)

        mall = dataset_services.create_dataset_from_sample("mall_customers")
        config = services.default_config(mall, dataset_services.load_dataframe(mall))
        self.assertIsNone(config.target)
        self.assertEqual(config.task_type, "clustering")
        self.assertNotIn("CustomerID", config.features)


class FeatureSelectionTests(PreprocessingTestCase):
    def test_id_like_and_constant_columns_are_excluded(self) -> None:
        dataset = self.classification_dataset()
        selected, reasons = services.suggested_feature_selection(dataset, "label")
        self.assertNotIn("label", selected)
        self.assertIn("age", selected)
        self.assertEqual(reasons["label"], "Variable cible")

    def test_high_cardinality_categories_are_excluded(self) -> None:
        # 40 distinct modalités over 100 rows: not ID-like, but too wide for one-hot.
        frame = pd.DataFrame(
            {
                "libelle": [f"cat-{index % 40}" for index in range(100)],
                "label": ["a"] * 100,
            }
        )
        dataset = self.upload_csv("cardinalite.csv", frame)
        selected, reasons = services.suggested_feature_selection(dataset, "label")
        self.assertNotIn("libelle", selected)
        self.assertIn("modalités", reasons["libelle"])

    def test_feature_options_mark_target_and_notes(self) -> None:
        dataset = self.classification_dataset()
        config = PreprocessingConfig(target="label", features=["age"])
        _, reasons = services.suggested_feature_selection(dataset, "label")
        options = {
            option["name"]: option
            for option in services.feature_options(dataset, config, reasons)
        }
        self.assertTrue(options["label"]["is_target"])
        self.assertFalse(options["label"]["selected"])
        self.assertTrue(options["age"]["selected"])
        self.assertFalse(options["segment"]["selected"])


class EngineTests(PreprocessingTestCase):
    def test_shapes_and_feature_expansion(self) -> None:
        dataset = self.classification_dataset()
        frame = dataset_services.load_dataframe(dataset)
        config = PreprocessingConfig(
            target="label",
            features=["age", "income", "segment"],
            task_type="classification",
            drop_duplicates=True,
            test_size=0.25,
        )
        prepared = services.prepare(dataset, config)

        self.assertEqual(prepared.train_rows + prepared.test_rows, len(frame))
        self.assertAlmostEqual(
            prepared.train_rows / (prepared.train_rows + prepared.test_rows), 0.75, delta=0.02
        )
        # one-hot adds a column per modalité: age + income + 2 segment levels.
        self.assertEqual(prepared.n_features_out, 4)
        self.assertEqual(prepared.numeric_features, ["age", "income"])
        self.assertEqual(prepared.categorical_features, ["segment"])
        self.assertEqual(set(prepared.class_labels), {"oui", "non"})
        self.assertFalse(np.isnan(prepared.X_train).any())

    def test_clustering_does_not_split(self) -> None:
        dataset = self.classification_dataset()
        config = PreprocessingConfig(
            target=None, features=["age", "income"], task_type="clustering"
        )
        prepared = services.prepare(dataset, config)
        self.assertIsNone(prepared.X_test)
        self.assertIsNone(prepared.y_train)
        self.assertIsNone(prepared.y_test)
        self.assertEqual(prepared.test_rows, 0)
        self.assertEqual(prepared.class_labels, [])

    def test_stratification_keeps_class_balance(self) -> None:
        dataset = self.classification_dataset(rows=300)
        config = PreprocessingConfig(
            target="label",
            features=["age", "income", "segment"],
            task_type="classification",
            test_size=0.25,
            stratify=True,
        )
        prepared = services.prepare(dataset, config)
        train_ratio = float((prepared.y_train == "oui").mean())
        test_ratio = float((prepared.y_test == "oui").mean())
        self.assertLess(abs(train_ratio - test_ratio), 0.05)

    def test_no_feature_left_raises(self) -> None:
        dataset = self.classification_dataset()
        preprocessor = DataPreprocessor(dataset_services.load_dataframe(dataset), target_col="label")
        with self.assertRaises(PreprocessingError):
            preprocessor.build_and_fit_transform(
                PreprocessingConfig(target="label", features=["colonne_absente"])
            )

    def test_ordinal_encoding_keeps_raw_feature_count(self) -> None:
        dataset = self.classification_dataset()
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="label",
                features=["age", "income", "segment"],
                task_type="classification",
                encoder="ordinal",
            ),
        )
        self.assertEqual(prepared.n_features_out, 3)
        self.assertEqual(prepared.feature_names, ["age", "income", "segment"])

    def test_unknown_categories_are_handled_at_inference_time(self) -> None:
        dataset = self.titanic()
        config = PreprocessingConfig(
            target="Survived",
            features=["Pclass", "Sex", "Age", "Fare", "Embarked"],
            task_type="classification",
            encoder="onehot",
            scaler="standard",
        )
        prepared = services.prepare(dataset, config)
        pipeline = services.load_pipeline(dataset)
        self.assertIsNotNone(pipeline)

        new_row = pd.DataFrame(
            [{"Pclass": 3, "Sex": "Inconnu", "Age": 30.0, "Fare": 12.5, "Embarked": "Z"}]
        )
        transformed = pipeline.transform(new_row)
        self.assertEqual(transformed.shape, (1, prepared.n_features_out))
        self.assertTrue(np.isfinite(transformed).all())

    def test_missing_values_are_imputed(self) -> None:
        dataset = self.classification_dataset()
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="label",
                features=["age", "income", "segment"],
                task_type="classification",
                numeric_strategy="median",
                categorical_strategy="constant",
                constant_value="Inconnu",
            ),
        )
        self.assertFalse(np.isnan(prepared.X_train).any())
        self.assertFalse(np.isnan(prepared.X_test).any())

    def test_drop_rows_strategy_removes_incomplete_rows(self) -> None:
        dataset = self.classification_dataset(rows=60)
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="label",
                features=["age", "income", "segment"],
                task_type="classification",
                numeric_strategy="drop_rows",
                categorical_strategy="drop_rows",
                scaler="none",
            ),
        )
        self.assertGreater(prepared.dropped_rows, 0)
        self.assertFalse(np.isnan(prepared.X_train).any())

    def test_missing_threshold_drops_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "complete": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "trouee": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        dataset = self.upload_csv("seuil.csv", frame)
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="label",
                features=["complete", "trouee"],
                task_type="classification",
                missing_threshold=0.6,
            ),
        )
        self.assertEqual(prepared.dropped_columns, ["trouee"])
        self.assertEqual(prepared.raw_features, ["complete"])
        self.assertTrue(any("seuil" in note or "supprimée" in note for note in prepared.warnings))

    def test_knn_imputation_produces_finite_matrices(self) -> None:
        dataset = self.classification_dataset(rows=80)
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="label",
                features=["age", "income"],
                task_type="classification",
                numeric_strategy="knn",
            ),
        )
        self.assertTrue(np.isfinite(prepared.X_train).all())
        self.assertTrue(np.isfinite(prepared.X_test).all())

    # -- leakage regression test ----------------------------------------
    def test_scaler_statistics_come_from_the_training_split(self) -> None:
        dataset = self.numeric_dataset()
        frame = dataset_services.load_dataframe(dataset)
        features = ["x1", "x2"]

        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="y",
                features=features,
                task_type="regression",
                scaler="standard",
                test_size=0.25,
                random_state=42,
            ),
        )

        expected_train, _, _, _ = train_test_split(
            frame[features], frame["y"], test_size=0.25, random_state=42
        )
        scaler = prepared.pipeline.named_transformers_["num"].named_steps["scaler"]

        np.testing.assert_allclose(
            scaler.mean_, expected_train.mean().to_numpy(), rtol=1e-9, atol=1e-9
        )
        # ... and *not* the full-data mean, which is what a leaky pipeline would fit.
        self.assertGreater(
            float(np.abs(scaler.mean_ - frame[features].mean().to_numpy()).max()), 1e-6
        )

    def test_two_test_sizes_produce_different_scaler_statistics(self) -> None:
        dataset = self.numeric_dataset()
        base = dict(
            target="y",
            features=["x1", "x2"],
            task_type="regression",
            scaler="standard",
        )
        small = services.prepare(dataset, PreprocessingConfig(**base, test_size=0.2))
        large = services.prepare(dataset, PreprocessingConfig(**base, test_size=0.35))

        mean_small = small.pipeline.named_transformers_["num"].named_steps["scaler"].mean_
        mean_large = large.pipeline.named_transformers_["num"].named_steps["scaler"].mean_
        self.assertGreater(float(np.abs(mean_small - mean_large).max()), 1e-9)


class OutlierTests(PreprocessingTestCase):
    def test_iqr_bounds_skip_constant_columns(self) -> None:
        frame = pd.DataFrame(
            {"varie": [1.0, 2.0, 3.0, 4.0, 100.0], "constante": [5.0] * 5, "vide": [np.nan] * 5}
        )
        bounds = iqr_bounds(frame, ["varie", "constante", "vide"], factor=1.5)
        self.assertIn("varie", bounds)
        self.assertNotIn("constante", bounds)
        self.assertNotIn("vide", bounds)
        lower, upper = bounds["varie"]
        self.assertLess(lower, 1.0)
        self.assertLess(upper, 100.0)

    def test_clip_and_mask_helpers(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 10.0]})
        bounds = {"a": (0.0, 5.0)}
        clipped = clip_frame(frame, bounds)
        self.assertEqual(clipped["a"].max(), 5.0)
        mask = outlier_mask(frame, bounds)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_clip_limits_extreme_values_in_both_splits(self) -> None:
        frame = pd.DataFrame(
            {
                "x": [float(value) for value in range(1, 101)] + [100000.0],
                "y": [float(value % 5) for value in range(1, 102)],
            }
        )
        dataset = self.upload_csv("outliers.csv", frame)
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="y",
                features=["x"],
                task_type="classification",
                outlier_method="clip",
                outlier_factor=1.5,
                scaler="none",
            ),
        )
        self.assertLess(float(np.max(prepared.X_train)), 9999)
        self.assertLess(float(np.max(prepared.X_test)), 9999)

    def test_drop_removes_extreme_rows_from_both_splits(self) -> None:
        frame = pd.DataFrame(
            {
                "x": [float(value) for value in range(1, 101)] + [100000.0],
                "y": [float(value % 2) for value in range(1, 102)],
            }
        )
        dataset = self.upload_csv("outliers_drop.csv", frame)
        prepared = services.prepare(
            dataset,
            PreprocessingConfig(
                target="y",
                features=["x"],
                task_type="classification",
                outlier_method="drop",
                outlier_factor=1.5,
                scaler="none",
            ),
        )
        self.assertLess(float(np.max(prepared.X_train)), 9999)
        self.assertLess(float(np.max(prepared.X_test)), 9999)
        self.assertTrue(any("aberrante" in note for note in prepared.warnings))


class ArtifactTests(PreprocessingTestCase):
    def test_prepare_persists_every_artifact(self) -> None:
        dataset = self.classification_dataset()
        config = PreprocessingConfig(
            target="label", features=["age", "income", "segment"], task_type="classification"
        )
        prepared = services.prepare(dataset, config)

        paths = services.artifact_paths(dataset)
        self.assertTrue(paths["dir"].is_dir())
        for key in ("pipeline", "prepared", "config"):
            self.assertTrue(paths[key].exists(), key)

        payload = json.loads(paths["config"].read_text(encoding="utf-8"))
        self.assertEqual(payload["target"], "label")
        self.assertEqual(payload["features"], ["age", "income", "segment"])

        reloaded = services.load_prepared(dataset)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.X_train.shape, prepared.X_train.shape)
        self.assertEqual(reloaded.feature_names, prepared.feature_names)

        self.assertEqual(services.load_config(dataset), config)

        import joblib

        pipeline = joblib.load(paths["pipeline"])
        self.assertEqual(pipeline.transform(prepared.df_clean[config.features]).shape[1], prepared.n_features_out)

    def test_missing_artifacts_return_none(self) -> None:
        dataset = self.classification_dataset()
        self.assertIsNone(services.load_prepared(dataset))
        self.assertIsNone(services.load_pipeline(dataset))
        self.assertIsNone(services.load_config(dataset))

    def test_clear_artifacts_removes_the_project_directory(self) -> None:
        dataset = self.classification_dataset()
        services.prepare(
            dataset,
            PreprocessingConfig(target="label", features=["age"], task_type="classification"),
        )
        root = services.project_dir(dataset)
        self.assertTrue(root.exists())
        services.clear_artifacts(dataset)
        self.assertFalse(root.exists())
        self.assertIsNone(services.load_prepared(dataset))

    def test_deleting_a_dataset_removes_its_artifacts(self) -> None:
        dataset = self.classification_dataset()
        services.prepare(
            dataset,
            PreprocessingConfig(target="label", features=["age"], task_type="classification"),
        )
        root = services.project_dir(dataset)
        self.assertTrue(root.exists())

        dataset.delete()
        self.assertFalse(root.exists())

    def test_export_clean_csv_is_utf8_with_bom(self) -> None:
        dataset = self.classification_dataset()
        services.prepare(
            dataset,
            PreprocessingConfig(target="label", features=["age"], task_type="classification"),
        )
        filename, payload = services.export_clean_csv(dataset)
        self.assertEqual(filename, "classification_clean.csv")
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertIn("age", payload.decode("utf-8-sig").splitlines()[0])

    def test_export_without_pipeline_falls_back_to_raw_data(self) -> None:
        dataset = self.classification_dataset()
        filename, payload = services.export_clean_csv(dataset)
        self.assertEqual(filename, "classification_clean.csv")
        self.assertGreater(len(payload), 0)


class ViewTests(PreprocessingTestCase):
    def test_page_without_dataset_shows_empty_state(self) -> None:
        response = self.client.get(reverse("preprocessing:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Aucun jeu de données actif")

    def test_page_prefills_the_form_from_the_dataset(self) -> None:
        self.titanic()
        self.client.post(reverse("datasets:sample"), {"slug": "titanic"})
        response = self.client.get(reverse("preprocessing:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="features"')
        self.assertContains(response, 'value="Survived" selected')

    def test_apply_persists_artifacts_and_updates_the_session(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "titanic"})
        response = self.client.post(
            reverse("preprocessing:apply"),
            {
                "target": "Survived",
                "task_type": "classification",
                "features": ["Pclass", "Sex", "Age", "Fare", "Embarked"],
                "numeric_strategy": "median",
                "categorical_strategy": "most_frequent",
                "constant_value": "Inconnu",
                "missing_threshold": "1.0",
                "drop_duplicates": "on",
                "outlier_method": "none",
                "outlier_factor": "1.5",
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
                "stratify": "on",
            },
        )
        self.assertRedirects(response, reverse("preprocessing:index"))

        dataset = Dataset.objects.get()
        self.assertTrue(services.artifact_paths(dataset)["pipeline"].exists())

        state = self.client.session[SESSION_KEY]
        self.assertEqual(state["target_column"], "Survived")
        self.assertEqual(state["task_type"], "classification")
        self.assertTrue(state["preprocessing_done"])
        self.assertEqual(state["active_step"], 3)

        page = self.client.get(reverse("preprocessing:index"))
        self.assertContains(page, "Dernier pipeline")

    def test_apply_without_features_shows_a_toast_and_keeps_the_page(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "titanic"})
        response = self.client.post(
            reverse("preprocessing:apply"),
            {
                "target": "Survived",
                "task_type": "classification",
                "features": [],
                "numeric_strategy": "mean",
                "categorical_strategy": "most_frequent",
                "constant_value": "Inconnu",
                "missing_threshold": "1.0",
                "outlier_method": "none",
                "outlier_factor": "1.5",
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sélectionnez au moins une variable explicative")
        self.assertFalse(self.client.session[SESSION_KEY]["preprocessing_done"])

    def test_export_and_pipeline_downloads(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "titanic"})
        self.client.post(
            reverse("preprocessing:apply"),
            {
                "target": "Survived",
                "task_type": "classification",
                "features": ["Pclass", "Sex", "Age"],
                "numeric_strategy": "median",
                "categorical_strategy": "most_frequent",
                "constant_value": "Inconnu",
                "missing_threshold": "1.0",
                "outlier_method": "none",
                "outlier_factor": "1.5",
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
            },
        )

        export = self.client.get(reverse("preprocessing:export"))
        self.assertEqual(export.status_code, 200)
        self.assertIn("attachment", export["Content-Disposition"])
        self.assertTrue(export.content.startswith(b"\xef\xbb\xbf"))

        download = self.client.get(reverse("preprocessing:pipeline"))
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download["Content-Disposition"])
        payload = b"".join(download.streaming_content)
        self.assertGreater(len(payload), 0)

    def test_reset_removes_artifacts_and_state(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "titanic"})
        self.client.post(
            reverse("preprocessing:apply"),
            {
                "target": "Survived",
                "task_type": "classification",
                "features": ["Pclass", "Age"],
                "numeric_strategy": "median",
                "categorical_strategy": "most_frequent",
                "constant_value": "Inconnu",
                "missing_threshold": "1.0",
                "outlier_method": "none",
                "outlier_factor": "1.5",
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
            },
        )
        dataset = Dataset.objects.get()
        paths = services.artifact_paths(dataset)
        self.assertTrue(paths["prepared"].exists())

        response = self.client.post(reverse("preprocessing:reset"))
        self.assertRedirects(response, reverse("preprocessing:index"))
        self.assertFalse(paths["prepared"].exists())
        self.assertFalse(self.client.session[SESSION_KEY]["preprocessing_done"])

    def test_clustering_flow_through_the_ui(self) -> None:
        self.client.post(reverse("datasets:sample"), {"slug": "mall_customers"})
        response = self.client.get(reverse("preprocessing:index"))
        self.assertContains(response, 'value="clustering" selected')

        apply = self.client.post(
            reverse("preprocessing:apply"),
            {
                "target": "",
                "task_type": "classification",
                "features": ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
                "numeric_strategy": "mean",
                "categorical_strategy": "most_frequent",
                "constant_value": "Inconnu",
                "missing_threshold": "1.0",
                "outlier_method": "none",
                "outlier_factor": "1.5",
                "encoder": "onehot",
                "scaler": "standard",
                "test_size": "0.2",
            },
        )
        self.assertRedirects(apply, reverse("preprocessing:index"))
        self.assertEqual(self.client.session[SESSION_KEY]["task_type"], "clustering")

        page = self.client.get(reverse("preprocessing:index"))
        self.assertContains(page, "Dernier pipeline")
        self.assertContains(page, "Clustering")
