"""
Models for the accounts app: UserProfile & UserActivityLog.
"""

from __future__ import annotations

from typing import Optional
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    Extended user profile linked 1-to-1 with Django User.
    Stores role, avatar color, organization and preferences.
    """

    class RoleChoices(models.TextChoices):
        ADMIN = "ADMIN", "Administrateur Plateforme"
        USER = "USER", "Data Scientist / Membre"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    role = models.CharField(
        max_length=20,
        choices=RoleChoices.choices,
        default=RoleChoices.USER,
    )
    avatar_color = models.CharField(max_length=7, default="#B8323C")  # Default Ume red
    organization = models.CharField(max_length=150, blank=True, default="Laboratoire ML")
    preferences = models.JSONField(
        default=dict,
        blank=True,
        help_text="Préférences utilisateur : notifications, graphiques par défaut, etc.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "profil utilisateur"
        verbose_name_plural = "profils utilisateurs"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Profil de {self.user.username} ({self.get_role_display()})"

    @property
    def initials(self) -> str:
        """Generate initials for avatar (e.g. John Doe -> JD, or username -> first 2 chars)."""
        first = self.user.first_name[:1] if self.user.first_name else ""
        last = self.user.last_name[:1] if self.user.last_name else ""
        if not first and not last:
            return self.user.username[:2].upper()
        return f"{first}{last}".upper()


class UserActivityLog(models.Model):
    """
    Audit log for user activities on the platform.
    """

    class ActionChoices(models.TextChoices):
        LOGIN = "LOGIN", "Connexion réussie"
        LOGOUT = "LOGOUT", "Déconnexion"
        UPLOAD_DATASET = "UPLOAD_DATASET", "Import de dataset"
        TRAIN_MODEL = "TRAIN_MODEL", "Entraînement de modèle"
        EXPORT_REPORT = "EXPORT_REPORT", "Génération rapport PDF"
        DELETE_DATASET = "DELETE_DATASET", "Suppression de dataset"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="activity_logs",
    )
    action = models.CharField(max_length=30, choices=ActionChoices.choices)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    details = models.CharField(max_length=255, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "journal d'activité"
        verbose_name_plural = "journaux d'activité"

    def __str__(self) -> str:
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.user.username} - {self.get_action_display()}"


@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance: User, created: bool, **kwargs) -> None:
    """Ensure every User automatically has an associated UserProfile."""
    if created:
        role = UserProfile.RoleChoices.ADMIN if (instance.is_staff or instance.is_superuser) else UserProfile.RoleChoices.USER
        UserProfile.objects.get_or_create(user=instance, defaults={"role": role})
    else:
        if hasattr(instance, "profile"):
            # Sync role if staff status changed
            if (instance.is_staff or instance.is_superuser) and instance.profile.role != UserProfile.RoleChoices.ADMIN:
                instance.profile.role = UserProfile.RoleChoices.ADMIN
                instance.profile.save(update_fields=["role"])
            elif not (instance.is_staff or instance.is_superuser) and instance.profile.role == UserProfile.RoleChoices.ADMIN:
                instance.profile.role = UserProfile.RoleChoices.USER
                instance.profile.save(update_fields=["role"])
            else:
                instance.profile.save()

