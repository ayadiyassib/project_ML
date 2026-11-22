"""
URL configuration for export_inference app.
"""

from django.urls import path
from apps.export_inference import views

app_name = "export_inference"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("download/model/", views.download_model_view, name="download_model"),
    path("download/predictions/", views.download_predictions_view, name="download_predictions"),
    path("download/pdf/", views.download_pdf_view, name="download_pdf"),
    path("predict/", views.predict_view, name="predict"),
]
