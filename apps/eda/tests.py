"""Tests for the EDA module: statistics engine, Plotly factory, theme and views."""

from __future__ import annotations

import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from django.core.cache import cache as django_cache
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from apps.datasets import cache
from apps.eda import stats_engine, theme
from apps.eda.plot_factory import (
    OTHER_LABEL,
    SCATTER_MAX_POINTS,
    EDAPlotFactory,
    category_groups,
    is_categorical_like,
)


# ---------------------------------------------------------------------------
# Statistics engine
# ---------------------------------------------------------------------------


class NumericStatsTests(SimpleTestCase):
    def setUp(self) -> None:
        self.values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], name="x", dtype=float)

    def test_central_tendency_and_dispersion_match_pandas(self) -> None:
        stats = stats_engine.numeric_stats(self.values)
        self.assertAlmostEqual(stats["mean"], self.values.mean(), places=4)
        self.assertAlmostEqual(stats["median"], self.values.median(), places=4)
        self.assertAlmostEqual(stats["std"], self.values.std(ddof=1), places=4)
        self.assertAlmostEqual(stats["variance"], self.values.var(ddof=1), places=3)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)

    def test_quartiles_iqr_and_outliers(self) -> None:
        stats = stats_engine.numeric_stats(self.values)
        q1, q3 = self.values.quantile(0.25), self.values.quantile(0.75)
        self.assertAlmostEqual(stats["q1"], q1, places=4)
        self.assertAlmostEqual(stats["q3"], q3, places=4)
        self.assertAlmostEqual(stats["iqr"], q3 - q1, places=4)
        self.assertEqual(stats["outliers"], 1)  # only the 100
        self.assertEqual(stats["outliers_pct"], 10.0)

    def test_shape_statistics_are_flagged(self) -> None:
        stats = stats_engine.numeric_stats(self.values)
        self.assertGreater(stats["skewness"], 1)
        self.assertEqual(stats["skew_label"], "Fortement étalée à droite")
        self.assertIn("Leptokurtique", stats["kurtosis_label"])

    def test_missing_values_are_ignored(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        self.assertEqual(stats_engine.numeric_stats(series)["mean"], 2.0)

    def test_normality_needs_enough_rows(self) -> None:
        self.assertIsNone(stats_engine.normality_test(pd.Series(range(10), dtype=float)))
        gaussian = pd.Series(np.random.default_rng(0).normal(size=500))
        result = stats_engine.normality_test(gaussian)
        self.assertTrue(result["is_normal"])
        skewed = pd.Series(np.random.default_rng(0).exponential(size=500))
        self.assertFalse(stats_engine.normality_test(skewed)["is_normal"])

    def test_labels(self) -> None:
        self.assertEqual(stats_engine.skew_label(0.1), "Symétrique")
        self.assertEqual(stats_engine.skew_label(-0.7), "Légèrement étalée à gauche")
        self.assertEqual(stats_engine.kurtosis_label(-2), "Platykurtique (aplatie)")
        self.assertEqual(stats_engine.kurtosis_label(None), "—")


class ColumnStatsTests(SimpleTestCase):
    def test_categorical_mode_entropy_and_rarity(self) -> None:
        series = pd.Series(["a"] * 150 + ["b"] * 49 + ["c"])
        stats = stats_engine.categorical_stats(series)
        self.assertEqual(stats["mode"], "a")
        self.assertEqual(stats["mode_pct"], 75.0)
        self.assertEqual(stats["rare_categories"], 1)  # "c" = 0.5 %
        self.assertGreater(stats["balance"], 0)
        self.assertLess(stats["balance"], 1)

    def test_balanced_categories_have_entropy_one(self) -> None:
        stats = stats_engine.categorical_stats(pd.Series(["x", "y"] * 10))
        self.assertEqual(stats["balance"], 1.0)

    def test_column_record_completeness_and_cardinality(self) -> None:
        series = pd.Series([1.0, 2.0, 2.0, np.nan], name="v")
        record = stats_engine.column_stats(series)
        self.assertEqual(record["kind"], "numeric")
        self.assertEqual(record["missing"], 1)
        self.assertEqual(record["completeness"], 75.0)
        self.assertEqual(record["completeness_level"], "warning")
        self.assertEqual(record["unique"], 2)
        self.assertAlmostEqual(record["cardinality_ratio"], 2 / 3, places=3)
        self.assertTrue(record["is_discrete"])

    def test_completeness_levels(self) -> None:
        self.assertEqual(stats_engine.completeness_level(100), "good")
        self.assertEqual(stats_engine.completeness_level(97), "ok")
        self.assertEqual(stats_engine.completeness_level(80), "warning")
        self.assertEqual(stats_engine.completeness_level(10), "critical")

    def test_summary_table_only_lists_numeric_columns(self) -> None:
        frame = pd.DataFrame({"n": [1.0, 2.0, 30.0, 4.0], "c": ["a", "b", "a", "b"]})
        rows = stats_engine.summary_table(stats_engine.describe_frame(frame))
        self.assertEqual([row["name"] for row in rows], ["n"])


class CorrelationTests(SimpleTestCase):
    def setUp(self) -> None:
        x = np.linspace(1, 10, 50)
        self.frame = pd.DataFrame({"x": x, "cube": x ** 3, "neg": -x, "const": 1.0, "label": ["a"] * 50})

    def test_spearman_captures_monotonic_relations(self) -> None:
        pearson = stats_engine.correlation_matrix(self.frame, "pearson")
        spearman = stats_engine.correlation_matrix(self.frame, "spearman")
        self.assertLess(pearson.loc["x", "cube"], 0.99)
        self.assertAlmostEqual(spearman.loc["x", "cube"], 1.0)

    def test_constant_and_non_numeric_columns_are_dropped(self) -> None:
        matrix = stats_engine.correlation_matrix(self.frame)
        self.assertEqual(list(matrix.columns), ["x", "cube", "neg"])

    def test_invalid_method(self) -> None:
        with self.assertRaises(ValueError):
            stats_engine.correlation_matrix(self.frame, "kendall-ish")

    def test_top_pairs_are_unique_and_sorted_by_magnitude(self) -> None:
        pairs = stats_engine.top_correlations(stats_engine.correlation_matrix(self.frame))
        self.assertEqual(len(pairs), 3)
        self.assertEqual((pairs[0]["a"], pairs[0]["b"], pairs[0]["value"]), ("x", "neg", -1.0))
        self.assertEqual(pairs[0]["sign"], "negative")
        magnitudes = [pair["abs"] for pair in pairs]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_target_correlations(self) -> None:
        items = stats_engine.target_correlations(self.frame, "x")
        self.assertEqual({item["feature"] for item in items}, {"cube", "neg"})


class TargetAndInsightTests(SimpleTestCase):
    def test_classification_target_detects_imbalance(self) -> None:
        frame = pd.DataFrame({"y": ["ok"] * 90 + ["fraude"] * 10})
        summary = stats_engine.target_summary(frame, "y")
        self.assertEqual(summary["task"], "classification")
        self.assertEqual(summary["imbalance_ratio"], 9.0)
        self.assertTrue(summary["is_imbalanced"])

    def test_numeric_target_with_many_values_is_regression(self) -> None:
        frame = pd.DataFrame({"price": np.arange(100, dtype=float)})
        self.assertEqual(stats_engine.target_summary(frame, "price")["task"], "regression")
        self.assertIsNone(stats_engine.target_summary(frame, None))

    def test_insights_cover_missing_constant_collinearity_and_imbalance(self) -> None:
        x = np.arange(40, dtype=float)
        frame = pd.DataFrame({
            "a": x, "b": x * 2 + 1, "const": 3.0,
            "holes": [np.nan] * 20 + list(range(20)),
            "y": ["p"] * 35 + ["n"] * 5,
        })
        records = stats_engine.describe_frame(frame)
        pairs = stats_engine.top_correlations(stats_engine.correlation_matrix(frame))
        target = stats_engine.target_summary(frame, "y")
        titles = " | ".join(i["title"] for i in stats_engine.build_insights(records, pairs, target, duplicates=2))
        self.assertIn("« holes » est incomplète", titles)
        self.assertIn("« const » est constante", titles)
        self.assertIn("Colinéarité : a ↔ b", titles)
        self.assertIn("Classes déséquilibrées", titles)
        self.assertIn("Lignes dupliquées", titles)

    def test_feature_target_pairs_are_not_reported_as_collinearity(self) -> None:
        x = np.arange(30, dtype=float)
        frame = pd.DataFrame({"x": x, "target": x * 3})
        records = stats_engine.describe_frame(frame)
        pairs = stats_engine.top_correlations(stats_engine.correlation_matrix(frame))
        target = {"name": "target", "is_imbalanced": False}
        insights = stats_engine.build_insights(records, pairs, target)
        self.assertFalse(any("Colinéarité" in i["title"] for i in insights))

    def test_critical_insights_come_first(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 50.0] * 5, "b": [np.nan] * 15 + [1.0] * 5})
        insights = stats_engine.build_insights(stats_engine.describe_frame(frame), [], None)
        self.assertEqual(insights[0]["level"], "critical")


# ---------------------------------------------------------------------------
# Plot factory & theme
# ---------------------------------------------------------------------------


class PlotFactoryTests(SimpleTestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.frame = pd.DataFrame({
            "a": rng.normal(size=300),
            "b": rng.normal(size=300),
            "c": rng.normal(size=300),
            "cls": rng.choice(["setosa", "virginica", "versicolor"], 300),
        })

    def test_histogram_counts_every_value_and_adds_kde(self) -> None:
        fig = EDAPlotFactory.histogram_kde(self.frame["a"], mean=0.0, median=0.0)
        bars, kde = fig.data[0], fig.data[1]
        self.assertEqual(int(np.sum(bars.y)), 300)
        self.assertEqual(kde.name, "Densité (KDE)")
        self.assertEqual(len(fig.layout.shapes), 2)  # mean + median lines

    def test_histogram_of_constant_column_does_not_crash(self) -> None:
        fig = EDAPlotFactory.histogram_kde(pd.Series([5.0] * 10, name="k"))
        self.assertEqual(len(fig.data), 1)

    def test_category_groups_are_naturally_sorted_and_folded(self) -> None:
        labels, order = category_groups(pd.Series(["c10", "c2", "c1", "c2"]))
        self.assertEqual(order, ["c1", "c2", "c10"])
        many = pd.Series([f"v{i}" for i in range(30)] + ["v0"] * 5)
        labels, order = category_groups(many, max_groups=5)
        self.assertEqual(len(order), 5)
        self.assertEqual(order[-1], OTHER_LABEL)

    def test_colour_follows_the_class_not_the_sample(self) -> None:
        colours = {t.name: t.marker.color for t in EDAPlotFactory.scatter_2d(self.frame, "a", "b", "cls").data}
        self.assertEqual(colours["setosa"], theme.CATEGORICAL[0])
        self.assertEqual(colours["versicolor"], theme.CATEGORICAL[1])
        self.assertEqual(colours["virginica"], theme.CATEGORICAL[2])

        # A rare class whose only rows have a missing x must not shift the others' slots.
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"x": rng.normal(size=8000), "y": rng.normal(size=8000),
                              "cls": rng.choice(["b", "c"], 8000)})
        frame.loc[:2, "cls"] = "a"
        frame.loc[:2, "x"] = np.nan
        sampled = {t.name: t.marker.color for t in EDAPlotFactory.scatter_2d(frame, "x", "y", "cls").data}
        self.assertNotIn("a", sampled)
        self.assertEqual(sampled["b"], theme.CATEGORICAL[1])
        self.assertEqual(sampled["c"], theme.CATEGORICAL[2])

    def test_scatter_samples_large_frames_and_adds_trendline(self) -> None:
        big = pd.DataFrame({"x": np.arange(10_000.0), "y": np.arange(10_000.0) * 2})
        fig = EDAPlotFactory.scatter_2d(big, "x", "y", trendline=True)
        self.assertEqual(len(fig.data[0].x), SCATTER_MAX_POINTS)
        self.assertIn("r = 1.00", fig.data[1].name)

    def test_numeric_colour_uses_a_sequential_scale(self) -> None:
        fig = EDAPlotFactory.scatter_2d(self.frame, "a", "b", "c")
        self.assertEqual(len(fig.data), 1)
        self.assertIsNotNone(fig.data[0].marker.colorscale)
        self.assertFalse(is_categorical_like(self.frame["c"]))

    def test_heatmap_masks_the_upper_triangle(self) -> None:
        matrix = stats_engine.correlation_matrix(self.frame)
        z = np.array(EDAPlotFactory.correlation_heatmap(matrix, mask_upper=True).data[0].z, dtype=float)
        self.assertTrue(np.isnan(z[0, 1]))
        self.assertFalse(np.isnan(z[1, 0]))
        z_full = np.array(EDAPlotFactory.correlation_heatmap(matrix, mask_upper=False).data[0].z, dtype=float)
        self.assertFalse(np.isnan(z_full).any())

    def test_frequency_bar_folds_the_tail(self) -> None:
        series = pd.Series([f"cat{i}" for i in range(50)])
        fig = EDAPlotFactory.frequency_bar(series, top=10)
        self.assertEqual(len(fig.data[0].y), 10)
        self.assertEqual(fig.data[0].y[-1], OTHER_LABEL)
        self.assertEqual(int(np.sum(fig.data[0].x)), 50)

    def test_violin_and_box_split_by_class(self) -> None:
        self.assertEqual(len(EDAPlotFactory.violin(self.frame["a"], self.frame["cls"]).data), 3)
        self.assertEqual(len(EDAPlotFactory.boxplot(self.frame["a"], self.frame["cls"]).data), 3)

    def test_scatter_3d_and_pairplot(self) -> None:
        fig3d = EDAPlotFactory.scatter_3d(self.frame, "a", "b", "c", "cls")
        self.assertEqual(len(fig3d.data), 3)
        splom = EDAPlotFactory.scatter_matrix(self.frame, ["a", "b", "c"], "cls")
        self.assertEqual(len(splom.data), 3)
        self.assertEqual(len(splom.data[0].dimensions), 3)

    def test_donut_has_at_most_n_slices(self) -> None:
        fig = EDAPlotFactory.donut(pd.Series([f"m{i % 12}" for i in range(120)]), max_slices=6)
        self.assertEqual(len(fig.data[0].labels), 6)


class ThemeTests(SimpleTestCase):
    def test_payload_is_valid_json_and_cannot_close_the_script_tag(self) -> None:
        frame = pd.DataFrame({"</script><b>x": [1.0, 2.0, 3.0]})
        chart = theme.make_chart(EDAPlotFactory.boxplot(frame["</script><b>x"]), "k", "Titre")
        self.assertNotIn("</script>", str(chart.payload))
        payload = json.loads(str(chart.payload))
        self.assertEqual(payload["layout"]["paper_bgcolor"], "rgba(0,0,0,0)")
        self.assertTrue(chart.id.startswith("fig-k-"))

    def test_series_colours_never_cycle(self) -> None:
        self.assertEqual(theme.series_color(0), "#3987e5")
        self.assertEqual(theme.series_color(8), theme.OVERFLOW_COLOR)
        self.assertEqual(theme.series_color(25), theme.OVERFLOW_COLOR)

    def test_rgba(self) -> None:
        self.assertEqual(theme.rgba("#ff0080", 0.5), "rgba(255,0,128,0.5)")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class EDAViewTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp(prefix="mlstudio-eda-")
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._override.disable()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self) -> None:
        cache.invalidate()
        django_cache.clear()

    def load(self, slug: str = "titanic") -> None:
        self.client.post(reverse("datasets:sample"), {"slug": slug})

    def test_page_without_dataset_shows_empty_state(self) -> None:
        response = self.client.get(reverse("eda:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Aucun jeu de données actif")

    def test_overview_renders_stats_insights_and_charts(self) -> None:
        self.load("titanic")
        response = self.client.get(reverse("eda:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Statistiques descriptives")
        self.assertContains(response, "Équilibre des classes — Survived")
        self.assertContains(response, "Valeurs manquantes par colonne")
        self.assertContains(response, "« Cabin » est incomplète")
        self.assertContains(response, 'data-plotly="fig-')
        self.assertContains(response, "js/ml_charts.js")

    def test_univariate_numeric_and_categorical(self) -> None:
        self.load("titanic")
        numeric = self.client.get(reverse("eda:univariate"), {"column": "Age"})
        self.assertContains(numeric, "Histogramme &amp; densité (KDE)")
        self.assertContains(numeric, "Graphique en violon")
        self.assertContains(numeric, "Distribution par classe de « Survived »")
        categorical = self.client.get(reverse("eda:univariate"), {"column": "Embarked"})
        self.assertContains(categorical, "Fréquences absolues &amp; relatives")
        self.assertContains(categorical, "Répartition (donut)")

    def test_low_cardinality_numeric_is_shown_as_categories(self) -> None:
        self.load("titanic")
        response = self.client.get(reverse("eda:univariate"), {"column": "Pclass"})
        self.assertContains(response, "Fréquences absolues &amp; relatives")
        self.assertContains(response, "traitée comme catégorielle")
        self.assertNotContains(response, "Histogramme &amp; densité (KDE)")

    def test_unknown_column_returns_an_error_partial(self) -> None:
        self.load("iris")
        response = self.client.get(reverse("eda:univariate"), {"column": "nope"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Colonne inconnue")

    def test_correlation_methods_and_id_exclusion(self) -> None:
        self.load("titanic")
        pearson = self.client.get(reverse("eda:correlation"), {"method": "pearson", "mask": "1"})
        self.assertContains(pearson, "Matrice de corrélation — Pearson")
        self.assertNotContains(pearson, "PassengerId")  # identifier excluded
        spearman = self.client.get(reverse("eda:correlation"), {"method": "spearman", "mask": ["0"]})
        self.assertContains(spearman, "Spearman")

    def test_scatter_endpoints(self) -> None:
        self.load("iris")
        x, y = "petal length (cm)", "petal width (cm)"
        self.assertContains(self.client.get(reverse("eda:scatter"), {"x": x, "y": y, "color": "target", "trend": "1"}),
                            "data-plotly")
        self.assertContains(self.client.get(reverse("eda:scatter3d"),
                                            {"x": x, "y": y, "z": "sepal length (cm)", "color": "target"}),
                            "Vue 3D")
        response = self.client.get(reverse("eda:pairplot"), {"columns": [x, y, "sepal width (cm)"], "color": "target"})
        self.assertContains(response, "Matrice de dispersion")

    def test_pairplot_needs_two_columns(self) -> None:
        self.load("iris")
        response = self.client.get(reverse("eda:pairplot"), {"columns": ["sepal width (cm)"]})
        self.assertContains(response, "au moins deux variables")

    def test_partials_without_dataset(self) -> None:
        response = self.client.get(reverse("eda:correlation"))
        self.assertContains(response, "Aucun jeu de données actif")
