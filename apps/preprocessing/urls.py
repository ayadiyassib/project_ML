from django.urls import path

from apps.preprocessing import views

app_name = "preprocessing"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("apply/", views.configure_view, name="apply"),
    path("reset/", views.reset_view, name="reset"),
    path("export/", views.export_csv_view, name="export"),
    path("pipeline/", views.download_pipeline_view, name="pipeline"),
]
