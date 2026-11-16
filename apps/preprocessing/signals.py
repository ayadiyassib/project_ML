"""
Signal handlers for preprocessing artefacts.

Deleting a ``Dataset`` must also remove its joblib artefacts and any trained
model stored alongside them, otherwise ``media/projects/`` grows forever with
orphaned files.
"""

from __future__ import annotations

from django.db.models.signals import post_delete
from django.dispatch import receiver

from apps.datasets.models import Dataset


@receiver(post_delete, sender=Dataset)
def _cleanup_project_artifacts(sender, instance: Dataset, **kwargs) -> None:
    # Imported lazily: services imports models, which would be circular at load time.
    from apps.preprocessing.services import clear_artifacts

    clear_artifacts(instance)
