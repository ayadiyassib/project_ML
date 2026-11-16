from django.apps import AppConfig


class PreprocessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.preprocessing'

    def ready(self) -> None:
        # Register the post_delete cleanup of joblib artefacts.
        from apps.preprocessing import signals  # noqa: F401
