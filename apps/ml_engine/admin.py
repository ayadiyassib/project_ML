from django.contrib import admin
from apps.ml_engine.models import TrainedModel


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ("name", "algorithm_key", "task_type", "user", "dataset", "is_favorite", "created_at")
    list_filter = ("algorithm_key", "task_type", "is_favorite", "created_at")
    search_fields = ("name", "user__username", "algorithm_key")

