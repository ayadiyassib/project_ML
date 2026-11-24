"""
Persistence layer for imported datasets.

A ``Dataset`` row is the single source of truth for "which file is the current
project".  The session only stores ``dataset_id`` (see ``apps.datasets.state``),
which keeps the workspace recoverable after a session reset and gives the other
modules (EDA, preprocessing, ML) a stable primary key to hang their artifacts on.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver


class Dataset(models.Model):
    """A tabular file imported by the user (upload, URL or sample dataset)."""

    class Source(models.TextChoices):
        UPLOAD = "upload", "Upload utilisateur"
        SAMPLE = "sample", "Jeu de démonstration"
        URL = "url", "URL distante"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="datasets",
        null=True,
        blank=True,
        help_text="Propriétaire du dataset. Si vide, il s'agit d'un jeu de démonstration partagé.",
    )
    name = models.CharField("nom affiché", max_length=200)
    original_filename = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="datasets/%Y/%m/")
    source = models.CharField(max_length=20, choices=Source.choices, default=Source.UPLOAD)

    size_bytes = models.PositiveBigIntegerField(default=0)
    n_rows = models.PositiveIntegerField(default=0)
    n_cols = models.PositiveIntegerField(default=0)
    task_hint = models.CharField(
        max_length=20,
        blank=True,
        default="",
        help_text="Tâche cible du jeu de démonstration : classification, regression ou clustering.",
    )
    columns_meta = models.JSONField(
        default=list,
        blank=True,
        help_text="Métadonnées par colonne : nom, dtype pandas et catégorie fonctionnelle.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "jeu de données"
        verbose_name_plural = "jeux de données"

    def __str__(self) -> str:
        return f"{self.name} ({self.n_rows} × {self.n_cols})"

    @property
    def memory_bytes(self) -> int:
        """Approximate in-memory footprint derived from the column metadata."""
        total = 0
        for column in self.columns_meta or []:
            kind = column.get("kind")
            if kind == "numeric":
                total += self.n_rows * 8
            elif kind == "boolean":
                total += self.n_rows * 1
            else:
                total += self.n_rows * 32
        return total

    @property
    def memory_human(self) -> str:
        return human_size(self.memory_bytes)

    @property
    def target_hint(self) -> str | None:
        """Best guess of the target column, used to pre-fill the preprocessing form."""
        if self.task_hint == "clustering":
            return None
        for column in self.columns_meta or []:
            if column.get("suggested_target"):
                return column["name"]
        return None


@receiver(post_delete, sender=Dataset)
def _delete_dataset_file(sender, instance: Dataset, **kwargs) -> None:
    """Remove the physical file when the row is deleted (sample copies included)."""
    if not instance.file:
        return
    try:
        Path(instance.file.path).unlink(missing_ok=True)
    except (ValueError, OSError):
        # Storage backends without a local path, or file already gone.
        pass


def human_size(num_bytes: int | float) -> str:
    """Format a byte count for display, e.g. ``1.8 Mo``."""
    size = float(num_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} Go"
