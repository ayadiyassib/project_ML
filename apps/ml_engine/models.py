"""
Models for the ML Engine app: TrainedModel persistence.
"""

from __future__ import annotations

import uuid
from django.conf import settings
from django.db import models
from apps.datasets.models import Dataset


class TrainedModel(models.Model):
    """
    Persistent record of a machine learning model trained by a user.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="trained_models",
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="models",
    )
    name = models.CharField(max_length=200, help_text="Ex: Random Forest Optimisé")
    algorithm_key = models.CharField(max_length=50)  # 'rf', 'svc', 'logreg', etc.
    task_type = models.CharField(max_length=30)      # 'classification', 'regression', 'clustering'

    hyperparameters = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Score test, validation croisée, RMSE, F1, Silhouette...",
    )
    joblib_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Chemin d'accès au fichier .joblib sérialisé sur le disque",
    )
    is_favorite = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "modèle entraîné"
        verbose_name_plural = "modèles entraînés"

    def __str__(self) -> str:
        return f"{self.name} ({self.algorithm_key}) — {self.user.username}"

