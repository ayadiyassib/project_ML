"""Tests for the ingestion and profiling services."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase, override_settings

from apps.datasets import services
from apps.datasets.models import Dataset


class DatasetTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-datasets-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def make_file(self, name: str, content: bytes) -> Path:
        path = Path(self._tmpdir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    # -- readers ---------------------------------------------------------
    def test_reads_semicolon_separated_cp1252_file(self) -> None:
        path = self.make_file(
            "ventes.csv",
            "ville;montant;remise\nMontréal;12,5;0\nParis;8;1\n".encode("utf-8"),
        )
        frame = services.read_tabular(path)
        self.assertEqual(list(frame.columns), ["ville", "montant", "remise"])
        self.assertEqual(frame.shape, (2, 3))
        self.assertEqual(frame.loc[0, "ville"], "Montréal")

    def test_reads_latin1_encoded_file(self) -> None:
        path = self.make_file(
            "latin.csv",
            "nom;age\nJosé;42\nRenée;31\n".encode("cp1252"),
        )
        frame = services.read_tabular(path)
        self.assertEqual(frame.loc[0, "nom"], "José")
        self.assertEqual(frame.loc[1, "nom"], "Renée")

    def test_reads_tab_separated_file(self) -> None:
        path = self.make_file("data.tsv", b"a\tb\n1\t2\n")
        frame = services.read_tabular(path)
        self.assertEqual(list(frame.columns), ["a", "b"])

    def test_reads_json_records_and_wrapped_payload(self) -> None:
        path = self.make_file(
            "records.json",
            b'{"data": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}',
        )
        frame = services.read_tabular(path)
        self.assertEqual(frame.shape, (2, 2))
        self.assertEqual(frame.loc[1, "b"], "y")

    def test_reads_excel_file(self) -> None:
        path = Path(self._tmpdir) / "classeur.xlsx"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(path, index=False)
        frame = services.read_tabular(path)
        self.assertEqual(frame.shape, (2, 2))

    def test_rejects_unsupported_extension(self) -> None:
        path = self.make_file("notes.txt2", b"hello")
        with self.assertRaises(services.DatasetError):
            services.read_tabular(path)

    def test_rejects_empty_file(self) -> None:
        path = self.make_file("vide.csv", b"")
        with self.assertRaises(services.DatasetError):
            services.read_tabular(path)

    # -- import ----------------------------------------------------------
    def test_sample_import_profiles_the_dataset(self) -> None:
        dataset = services.create_dataset_from_sample("titanic")
        self.assertEqual(dataset.n_rows, 891)
        self.assertEqual(dataset.n_cols, 12)
        self.assertEqual(dataset.task_hint, "classification")
        self.assertTrue(Path(dataset.file.path).exists())
        self.assertEqual(dataset.target_hint, "Survived")

        names = [column["name"] for column in dataset.columns_meta]
        self.assertIn("Cabin", names)
        cabin = next(c for c in dataset.columns_meta if c["name"] == "Cabin")
        self.assertGreater(cabin["missing_pct"], 50)
        self.assertTrue(cabin["is_id_like"] is False)

        passenger_id = next(c for c in dataset.columns_meta if c["name"] == "PassengerId")
        self.assertTrue(passenger_id["is_id_like"])

    def test_sample_import_is_reused(self) -> None:
        first = services.create_dataset_from_sample("iris")
        second = services.create_dataset_from_sample("iris")
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(Dataset.objects.count(), 1)

    def test_clustering_sample_has_no_suggested_target(self) -> None:
        dataset = services.create_dataset_from_sample("mall_customers")
        self.assertEqual(dataset.task_hint, "clustering")
        self.assertIsNone(dataset.target_hint)

    @patch.object(services, "MAX_UPLOAD_BYTES", 32)
    def test_upload_size_limit_is_enforced(self) -> None:
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("big.csv", b"a,b\n" + b"1,2\n" * 20)
        with self.assertRaises(services.DatasetError):
            services.create_dataset_from_upload(upload)

    def test_upload_imports_and_returns_metadata(self) -> None:
        from django.core.files.uploadedfile import SimpleUploadedFile

        csv_bytes = "x,y,classe\n1,2,a\n3,4,b\n5,6,a\n".encode()
        upload = SimpleUploadedFile("mon_fichier.csv", csv_bytes)
        dataset = services.create_dataset_from_upload(upload)
        self.assertEqual(dataset.original_filename, "mon_fichier.csv")
        self.assertEqual((dataset.n_rows, dataset.n_cols), (3, 3))
        self.assertEqual(dataset.source, Dataset.Source.UPLOAD)

    def test_deleting_a_dataset_removes_its_file(self) -> None:
        dataset = services.create_dataset_from_sample("wine")
        path = Path(dataset.file.path)
        self.assertTrue(path.exists())
        dataset.delete()
        self.assertFalse(path.exists())

    # -- profiling & preview --------------------------------------------
    def test_profile_detects_kinds_missing_and_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [10, 20, None, 40],
                "sexe": ["h", "f", "h", "f"],
                "note": ["bonjour le monde"] * 4,
                "constante": [1, 1, 1, 1],
            }
        )
        profile = services.profile_dataframe(frame)
        kinds = {column["name"]: column["kind"] for column in profile["columns"]}
        self.assertEqual(kinds["age"], "numeric")
        self.assertEqual(kinds["sexe"], "categorical")
        self.assertEqual(profile["missing_cells"], 1)
        self.assertEqual(profile["duplicates"], 0)
        self.assertIn("constante", profile["constant_columns"])

    def test_preview_paginates_and_formats_cells(self) -> None:
        frame = pd.DataFrame({"a": range(1, 31), "b": [None] * 30})
        page = services.preview_page(frame, page=2, page_size=10)
        self.assertEqual(page["page"], 2)
        self.assertEqual(page["total_pages"], 3)
        self.assertEqual(page["start"], 11)
        self.assertEqual(page["end"], 20)
        self.assertEqual(page["rows"][0][0], "11")
        self.assertTrue(page["has_prev"])
        self.assertTrue(page["has_next"])

        clamped = services.preview_page(frame, page=99, page_size=10)
        self.assertEqual(clamped["page"], 3)
        self.assertFalse(clamped["has_next"])

    def test_excel_and_json_import_roundtrip(self) -> None:
        from django.core.files.uploadedfile import SimpleUploadedFile

        payload = b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]'
        dataset = services.create_dataset_from_upload(SimpleUploadedFile("data.json", payload))
        self.assertEqual((dataset.n_rows, dataset.n_cols), (2, 2))
