"""Django admin registration for imported datasets."""

from django.contrib import admin

from apps.datasets.models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "n_rows", "n_cols", "memory_human", "created_at")
    list_filter = ("source", "created_at")
    search_fields = ("name", "original_filename")
    readonly_fields = (
        "id",
        "size_bytes",
        "n_rows",
        "n_cols",
        "columns_meta",
        "created_at",
    )
    ordering = ("-created_at",)
