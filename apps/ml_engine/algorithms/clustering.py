"""
Unsupervised clustering algorithm implementation:
10. K-Means Clustering
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from apps.ml_engine.algorithms.base import BaseAlgorithm, HyperparamField


class KMeansAlgorithm(BaseAlgorithm):
    """Algorithme 10 : K-Means (Clustering Non Supervisé)."""

    key = "kmeans"
    name = "K-Means (Clustering Non Supervisé)"
    category = "clustering"
    description = "Partitionne un jeu de données sans étiquette en k grappes homogènes en minimisant l'inertie intra-cluster (somme des carrés des distances)."
    formula_latex = r"\min \sum_{j=1}^k \sum_{\mathbf{x}_i \in S_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2"

    def get_hyperparams(self, task_type: str) -> List[HyperparamField]:
        return [
            HyperparamField(
                name="n_clusters",
                label="Nombre de grappes (k)",
                field_type="int",
                default=3,
                min_val=2,
                max_val=10,
                step=1,
                help_text="Nombre de groupes à former dans les données.",
            ),
            HyperparamField(
                name="init",
                label="Méthode d'initialisation",
                field_type="select",
                default="k-means++",
                options=[("k-means++", "k-means++ (Dispersion initiale optimale)"), ("random", "Aléatoire (random)")],
                help_text="Stratégie pour choisir les centroïdes de départ.",
            ),
            HyperparamField(
                name="max_iter",
                label="Nombre maximal d'itérations",
                field_type="int",
                default=300,
                min_val=50,
                max_val=1000,
                step=50,
                help_text="Nombre limite de recalculs des centroïdes.",
            ),
        ]

    def build_model(self, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        field_dict = {f.name: f for f in self.get_hyperparams(task_type)}
        n_clusters = field_dict["n_clusters"].clean_value(params.get("n_clusters"))
        init_method = field_dict["init"].clean_value(params.get("init"))
        max_iter = field_dict["max_iter"].clean_value(params.get("max_iter"))

        return KMeans(
            n_clusters=int(n_clusters),
            init=init_method,
            max_iter=int(max_iter),
            random_state=42,
            n_init="auto",
        )

    def extract_diagnostics(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "clustering"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "kmeans",
            "n_clusters": getattr(model, "n_clusters", 3),
            "inertia": round(float(getattr(model, "inertia_", 0.0)), 4),
            "n_iter": int(getattr(model, "n_iter_", 0)),
        }
        
        # Centroids
        if hasattr(model, "cluster_centers_"):
            centroids = []
            for c_idx, center in enumerate(model.cluster_centers_):
                center_dict = {"cluster": f"Cluster {c_idx}"}
                for f_idx, val in enumerate(center):
                    fname = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"
                    center_dict[fname] = round(float(val), 4)
                centroids.append(center_dict)
            result["centroids"] = centroids

        # Cluster distribution counts
        if hasattr(model, "labels_"):
            unique, counts = np.unique(model.labels_, return_counts=True)
            result["cluster_distribution"] = [
                {"cluster": f"Cluster {int(u)}", "count": int(c), "ratio": round(float(c / len(model.labels_)) * 100, 1)}
                for u, c in zip(unique, counts)
            ]

        return result
